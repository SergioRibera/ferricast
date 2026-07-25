use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpStream, UdpSocket};
use zbus::Connection;
use zbus::zvariant::{OwnedObjectPath, OwnedValue, Value};

use ferricast_core::{CastSession, Device, EncodedFrame, FerricastError, Result, StreamConfig};

use crate::dbus::{
    ActiveConnectionProxy, Ip4ConfigProxy, IwdP2pPeerProxy,
    NM_ACTIVE_CONNECTION_STATE_ACTIVATED, NM_ACTIVE_CONNECTION_STATE_DEACTIVATED,
    NM_ACTIVE_CONNECTION_STATE_DEACTIVATING, NetworkManagerProxy,
};

// ── constants ─────────────────────────────────────────────────────────────────

/// WFD default RTSP port (Wi-Fi Display spec §6.5).
const WFD_DEFAULT_PORT: u16 = 7236;
/// RTP payload type for MPEG-TS (RFC 2250).
const RTP_PT_MP2T: u8 = 33;
/// 90 kHz MPEG-TS / RTP clock.
const CLOCK_90K: u64 = 90_000;
/// MPEG-TS packet size.
const TS_PACKET: usize = 188;
/// TS packets per RTP packet — keeps payload ≤ 1316 bytes.
const TS_PER_RTP: usize = 7;
/// How long to wait for the P2P connection to reach ACTIVATED.
///
/// wpa_supplicant re-scans all P2P channels after P2P_CONNECT to locate
/// the peer (up to ~30 s on busy spectrum), then GO negotiation, WPS PBC
/// exchange, and DHCP each add 5-15 s.  120 s gives the full chain room
/// to complete even on congested 2.4 GHz.
const P2P_CONNECT_TIMEOUT: Duration = Duration::from_secs(120);

const PID_PAT: u16 = 0x0000;
const PID_PMT: u16 = 0x1000;
const PID_VIDEO: u16 = 0x0100;
const STREAM_TYPE_H264: u8 = 0x1B;
const PES_STREAM_ID_VIDEO: u8 = 0xE0;

// ── active connection discriminator ──────────────────────────────────────────

enum ActiveConnection {
    Nm(OwnedObjectPath),
    Iwd(OwnedObjectPath),
}

// ── session state ─────────────────────────────────────────────────────────────

/// Miracast (Wi-Fi Display) streaming session.
///
/// Lifecycle: `Default` → `connect()` → `setup_stream()` → `send_frame()` loop → `stop()`.
pub struct MiracastSession {
    dbus: Option<Connection>,
    active_connection: Option<ActiveConnection>,
    rtsp: Option<BufReader<TcpStream>>,
    rtp: Option<UdpSocket>,
    sink_rtp_addr: Option<SocketAddr>,
    session_id: Option<String>,
    presentation_url: Option<String>,
    /// RTSP CSeq counter for source-initiated requests.
    src_cseq: u32,
    /// RTP sequence number.
    rtp_seq: u16,
    /// RTP timestamp (90 kHz).
    rtp_ts: u32,
    /// RTP SSRC (random, fixed per session).
    ssrc: u32,
    /// MPEG-TS continuity counters.
    ts: TsState,
    alive: bool,
}

#[derive(Default)]
struct TsState {
    cc_pat: u8,
    cc_pmt: u8,
    cc_video: u8,
    psi_sent: bool,
}

impl Default for MiracastSession {
    fn default() -> Self {
        Self {
            dbus: None,
            active_connection: None,
            rtsp: None,
            rtp: None,
            sink_rtp_addr: None,
            session_id: None,
            presentation_url: None,
            src_cseq: 1,
            rtp_seq: 0,
            rtp_ts: 0,
            ssrc: rand::random(),
            ts: TsState::default(),
            alive: false,
        }
    }
}

impl CastSession for MiracastSession {
    async fn connect(&mut self, device: &Device) -> Result<()> {
        match device.metadata.get("backend").map(String::as_str) {
            Some("iwd") => self.connect_iwd(device).await,
            _ => self.connect_nm(device).await,
        }
    }

    /// Runs the WFD RTSP M1–M7 handshake and binds the RTP sender socket.
    async fn setup_stream(&mut self, _config: &StreamConfig) -> Result<()> {
        let rtsp = self.rtsp.as_mut().ok_or(FerricastError::NoActiveSession)?;

        // M1: we send OPTIONS, sink acknowledges.
        send_rtsp(
            rtsp,
            &format!(
                "OPTIONS * RTSP/1.0\r\nCSeq: {}\r\nRequire: org.wfa.wfd1.0\r\n\r\n",
                self.src_cseq
            ),
        )
        .await?;
        let m1_resp = recv_rtsp(rtsp).await?;
        ensure_200(&m1_resp, self.src_cseq)?;
        self.src_cseq += 1;

        // M2: sink sends OPTIONS to us; we reply.
        let m2_req = recv_rtsp(rtsp).await?;
        let sink_cseq = header_value(&m2_req.headers, "cseq")
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(100);
        send_rtsp(
            rtsp,
            &format!(
                "RTSP/1.0 200 OK\r\nCSeq: {sink_cseq}\r\nPublic: org.wfa.wfd1.0, GET_PARAMETER, SET_PARAMETER, SETUP, PLAY, PAUSE, TEARDOWN\r\n\r\n"
            ),
        )
        .await?;

        // M3: sink sends GET_PARAMETER querying our capabilities.
        let m3_req = recv_rtsp(rtsp).await?;
        let m3_cseq = header_value(&m3_req.headers, "cseq")
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(sink_cseq + 1);

        // Bind our RTP socket now so we can advertise the port.
        let rtp_sock = UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|e| FerricastError::Connection(format!("Cannot bind RTP socket: {e}")))?;
        let our_rtp_port = rtp_sock
            .local_addr()
            .map_err(|e| FerricastError::Connection(format!("RTP local_addr: {e}")))?
            .port();

        let caps_body = wfd_capabilities(our_rtp_port);
        send_rtsp(
            rtsp,
            &format!(
                "RTSP/1.0 200 OK\r\nCSeq: {m3_cseq}\r\nContent-Type: text/parameters\r\nContent-Length: {}\r\n\r\n{caps_body}",
                caps_body.len()
            ),
        )
        .await?;

        // M4: we send SET_PARAMETER with chosen parameters.
        let our_ip = rtp_sock
            .local_addr()
            .map(|a| a.ip())
            .unwrap_or(IpAddr::V4(Ipv4Addr::UNSPECIFIED));

        let presentation_url =
            format!("rtsp://{our_ip}/wfd1.0/streamid=0");

        let m4_body = wfd_set_parameter(&presentation_url, our_rtp_port);
        send_rtsp(
            rtsp,
            &format!(
                "SET_PARAMETER rtsp://localhost/wfd1.0 RTSP/1.0\r\nCSeq: {}\r\nContent-Type: text/parameters\r\nContent-Length: {}\r\n\r\n{m4_body}",
                self.src_cseq,
                m4_body.len()
            ),
        )
        .await?;
        let m4_resp = recv_rtsp(rtsp).await?;
        ensure_200(&m4_resp, self.src_cseq)?;
        self.src_cseq += 1;

        // M5: trigger SETUP from the sink.
        let trigger_body = "wfd_trigger_method: SETUP\r\n";
        send_rtsp(
            rtsp,
            &format!(
                "SET_PARAMETER rtsp://localhost/wfd1.0 RTSP/1.0\r\nCSeq: {}\r\nContent-Type: text/parameters\r\nContent-Length: {}\r\n\r\n{trigger_body}",
                self.src_cseq,
                trigger_body.len()
            ),
        )
        .await?;
        let m5_resp = recv_rtsp(rtsp).await?;
        ensure_200(&m5_resp, self.src_cseq)?;
        self.src_cseq += 1;

        // M6: sink sends SETUP; we respond with transport details.
        let m6_req = recv_rtsp(rtsp).await?;
        let m6_cseq = header_value(&m6_req.headers, "cseq")
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(m3_cseq + 2);

        // Parse sink's client_port from Transport header.
        let sink_rtp_port = parse_client_port(
            header_value(&m6_req.headers, "transport").as_deref().unwrap_or(""),
        )
        .unwrap_or(1028);

        // The sink's IP is the remote TCP address.
        let sink_tcp_addr = rtsp
            .get_ref()
            .peer_addr()
            .map_err(|e| FerricastError::Connection(format!("Cannot get sink TCP addr: {e}")))?;
        let sink_rtp_addr = SocketAddr::new(sink_tcp_addr.ip(), sink_rtp_port);

        let session_id = format!("{:016x}", rand::random::<u64>());
        send_rtsp(
            rtsp,
            &format!(
                "RTSP/1.0 200 OK\r\nCSeq: {m6_cseq}\r\nSession: {session_id};timeout=60\r\nTransport: RTP/AVP/UDP;unicast;client_port={sink_rtp_port};server_port={our_rtp_port}\r\n\r\n"
            ),
        )
        .await?;

        // M7: sink sends PLAY; we respond and streaming begins.
        let m7_req = recv_rtsp(rtsp).await?;
        let m7_cseq = header_value(&m7_req.headers, "cseq")
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(m6_cseq + 1);
        send_rtsp(
            rtsp,
            &format!(
                "RTSP/1.0 200 OK\r\nCSeq: {m7_cseq}\r\nSession: {session_id}\r\n\r\n"
            ),
        )
        .await?;

        tracing::info!(
            %sink_rtp_addr,
            our_rtp_port,
            session_id,
            "WFD RTSP handshake complete — streaming"
        );

        self.rtp = Some(rtp_sock);
        self.sink_rtp_addr = Some(sink_rtp_addr);
        self.session_id = Some(session_id);
        self.presentation_url = Some(presentation_url);
        Ok(())
    }

    /// Packetizes one video frame into MPEG-TS and sends it over RTP/UDP.
    async fn send_frame(&mut self, frame: &EncodedFrame) -> Result<()> {
        let sock = self
            .rtp
            .as_ref()
            .ok_or(FerricastError::NoActiveSession)?;
        let sink = self.sink_rtp_addr.ok_or(FerricastError::NoActiveSession)?;

        // Advance RTP timestamp from the frame's microsecond PTS.
        let rtp_ts = (frame.timestamp_us * CLOCK_90K / 1_000_000) as u32;
        self.rtp_ts = rtp_ts;

        let mut ts_packets: Vec<u8> = Vec::new();

        // Emit PAT + PMT once at stream start (or on every keyframe for robustness).
        if !self.ts.psi_sent || frame.is_keyframe {
            ts_packets.extend_from_slice(&build_pat(&mut self.ts.cc_pat));
            ts_packets.extend_from_slice(&build_pmt(&mut self.ts.cc_pmt));
            self.ts.psi_sent = true;
        }

        // Wrap the H.264 Annex-B data in a PES packet, then TS-packetize it.
        let pes = build_pes(&frame.data, frame.pts_dts);
        packetize_pes_into_ts(&pes, PID_VIDEO, &mut self.ts.cc_video, &mut ts_packets);

        // Send TS packets grouped into RTP packets (≤ 7 per packet).
        for chunk in ts_packets.chunks(TS_PER_RTP * TS_PACKET) {
            let pkt = build_rtp_packet(
                self.rtp_seq,
                rtp_ts,
                self.ssrc,
                chunk,
            );
            sock.send_to(&pkt, sink).await.map_err(|e| {
                FerricastError::Streaming(format!("RTP send failed: {e}"))
            })?;
            self.rtp_seq = self.rtp_seq.wrapping_add(1);
        }

        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        if !self.alive {
            return Ok(());
        }
        self.alive = false;

        // Send RTSP TEARDOWN if we have a session.
        if let (Some(rtsp), Some(session_id)) =
            (self.rtsp.as_mut(), self.session_id.as_deref())
        {
            let url = self
                .presentation_url
                .as_deref()
                .unwrap_or("rtsp://localhost/wfd1.0/streamid=0");
            let _ = send_rtsp(
                rtsp,
                &format!(
                    "TEARDOWN {url} RTSP/1.0\r\nCSeq: {}\r\nSession: {session_id}\r\n\r\n",
                    self.src_cseq
                ),
            )
            .await;
        }

        self.rtsp = None;
        self.rtp = None;

        if let Some(conn) = self.dbus.take() {
            match self.active_connection.take() {
                Some(ActiveConnection::Nm(path)) => {
                    if let Ok(nm) = NetworkManagerProxy::new(&conn).await {
                        if let Err(e) = nm.deactivate_connection(path).await {
                            tracing::warn!("NM.DeactivateConnection failed: {e}");
                        }
                    }
                }
                Some(ActiveConnection::Iwd(peer_path)) => {
                    match IwdP2pPeerProxy::new(&conn, peer_path).await {
                        Ok(peer) => {
                            if let Err(e) = peer.disconnect().await {
                                tracing::warn!("iwd P2P disconnect failed: {e}");
                            }
                        }
                        Err(e) => tracing::warn!("iwd peer proxy for disconnect: {e}"),
                    }
                }
                None => {}
            }
        }

        Ok(())
    }

    fn is_alive(&self) -> bool {
        self.alive
    }
}

// ── backend connect helpers ───────────────────────────────────────────────────

impl MiracastSession {
    async fn connect_nm(&mut self, device: &Device) -> Result<()> {
        let peer_dbus_path = device
            .metadata
            .get("path")
            .ok_or_else(|| FerricastError::Connection("Missing 'path' in device metadata".into()))
            .and_then(|s| {
                OwnedObjectPath::try_from(s.as_str())
                    .map_err(|e| FerricastError::Connection(format!("Invalid peer D-Bus path: {e}")))
            })?;

        let p2p_device_path = device
            .metadata
            .get("device_path")
            .ok_or_else(|| {
                FerricastError::Connection("Missing 'device_path' in device metadata".into())
            })
            .and_then(|s| {
                OwnedObjectPath::try_from(s.as_str())
                    .map_err(|e| FerricastError::Connection(format!("Invalid device D-Bus path: {e}")))
            })?;

        let hw_address = device
            .metadata
            .get("hw_address")
            .ok_or_else(|| {
                FerricastError::Connection("Missing 'hw_address' in device metadata".into())
            })?
            .clone();


        let conn = Connection::system().await.map_err(|e| {
            FerricastError::Connection(format!("D-Bus system bus unavailable: {e}"))
        })?;

        let nm = NetworkManagerProxy::new(&conn).await.map_err(|e| {
            FerricastError::Connection(format!("Cannot reach NetworkManager: {e}"))
        })?;

        let connection_dict = build_p2p_connection_dict(&hw_address)?;

        // "volatile" — NM discards this profile as soon as it deactivates,
        // so failed P2P attempts don't accumulate stale profiles in NM's
        // settings storage across reconnect attempts.
        let mut activate_opts: HashMap<String, Value> = HashMap::new();
        activate_opts.insert(
            "persist".into(),
            Value::from("volatile"),
        );

        activate_opts.insert(
            "bind-activation".into(),
            zvariant::Value::Str(zvariant::Str::from("none")),
        );

        let (_, active_path, _) = nm
            .add_and_activate_connection2(
                connection_dict,
                p2p_device_path,
                peer_dbus_path,
                activate_opts,
            )
            .await
            .map_err(|e| {
                FerricastError::Connection(format!("NM.AddAndActivateConnection2 failed: {e}"))
            })?;



        tracing::info!(%active_path, "P2P connection activating (NM)");

        let sink_ip = wait_for_ip(&conn, &active_path).await?;
        tracing::info!(%sink_ip, "P2P connection activated");

        let wfd_port = if device.port == 0 { WFD_DEFAULT_PORT } else { device.port };
        let sink_addr = SocketAddr::new(IpAddr::V4(sink_ip), wfd_port);
        tracing::info!(%sink_addr, "Connecting RTSP to Miracast sink (NM)");

        let tcp = tokio::time::timeout(
            Duration::from_secs(10),
            TcpStream::connect(sink_addr),
        )
        .await
        .map_err(|_| FerricastError::Connection("RTSP TCP connect timed out".into()))?
        .map_err(|e| FerricastError::Connection(format!("RTSP TCP connect failed: {e}")))?;

        self.dbus = Some(conn);
        self.active_connection = Some(ActiveConnection::Nm(active_path));
        self.rtsp = Some(BufReader::new(tcp));
        self.alive = true;
        Ok(())
    }

    async fn connect_iwd(&mut self, device: &Device) -> Result<()> {
        let peer_path = device
            .metadata
            .get("path")
            .ok_or_else(|| FerricastError::Connection("Missing 'path' in device metadata".into()))
            .and_then(|s| {
                OwnedObjectPath::try_from(s.as_str())
                    .map_err(|e| FerricastError::Connection(format!("Invalid peer D-Bus path: {e}")))
            })?;

        let conn = Connection::system().await.map_err(|e| {
            FerricastError::Connection(format!("D-Bus system bus unavailable: {e}"))
        })?;

        let peer = IwdP2pPeerProxy::new(&conn, peer_path.clone())
            .await
            .map_err(|e| FerricastError::Connection(format!("iwd P2P peer proxy: {e}")))?;

        tracing::info!(%peer_path, "Connecting iwd P2P peer (group formation + DHCP)");

        // iwd's connect() blocks until P2P group formation and DHCP are complete.
        tokio::time::timeout(Duration::from_secs(45), peer.connect())
            .await
            .map_err(|_| FerricastError::Connection("iwd P2P connect timed out after 45s".into()))?
            .map_err(|e| FerricastError::Connection(format!("iwd P2P connect: {e}")))?;

        tracing::info!("iwd P2P group formed — resolving peer IP from ARP cache");

        let sink_ip = find_p2p_peer_ip().await?;
        tracing::info!(%sink_ip, "P2P peer IP resolved");

        let wfd_port = if device.port == 0 { WFD_DEFAULT_PORT } else { device.port };
        let sink_addr = SocketAddr::new(IpAddr::V4(sink_ip), wfd_port);
        tracing::info!(%sink_addr, "Connecting RTSP to Miracast sink (iwd)");

        let tcp = tokio::time::timeout(
            Duration::from_secs(10),
            TcpStream::connect(sink_addr),
        )
        .await
        .map_err(|_| FerricastError::Connection("RTSP TCP connect timed out".into()))?
        .map_err(|e| FerricastError::Connection(format!("RTSP TCP connect failed: {e}")))?;

        self.dbus = Some(conn);
        self.active_connection = Some(ActiveConnection::Iwd(peer_path));
        self.rtsp = Some(BufReader::new(tcp));
        self.alive = true;
        Ok(())
    }
}

// ── NM helpers ────────────────────────────────────────────────────────────────

/// Builds a `a{sa{sv}}` NM connection dict for a Wi-Fi P2P connection.
fn build_p2p_connection_dict(
    peer_hw_address: &str,
) -> Result<HashMap<String, HashMap<String, OwnedValue>>> {
    let mut connection_settings: HashMap<String, OwnedValue> = HashMap::new();
    connection_settings.insert(
        "type".into(),
        OwnedValue::try_from(Value::from("wifi-p2p")).map_err(|e| {
            FerricastError::Connection(format!("zvariant: {e}"))
        })?,
    );
    connection_settings.insert(
        "id".into(),
        OwnedValue::try_from(Value::from("Ferricast Miracast")).map_err(|e| {
            FerricastError::Connection(format!("zvariant: {e}"))
        })?,
    );
    
    connection_settings.insert(
        "autoconnect".to_string(),
        OwnedValue::try_from(Value::Bool(false))
            .map_err(|e| {FerricastError::Connection(format!("zvariant: {e}"))})?
    );


    let mut p2p_settings: HashMap<String, OwnedValue> = HashMap::new();
    p2p_settings.insert(
        "peer".into(),
        OwnedValue::try_from(Value::from(peer_hw_address)).map_err(|e| {
            FerricastError::Connection(format!("zvariant: {e}"))
        })?,
    );
    let wfd_ies = wfd_source_ies();

    p2p_settings.insert(
        "wfd-ies".into(),
        OwnedValue::try_from(Value::from(wfd_ies)).map_err(|e| {
            FerricastError::Connection(format!("zvariant: {e}"))
        })?,
    );

    let mut ipv4_props: HashMap<String, OwnedValue> = HashMap::new();
    ipv4_props.insert("method".into(), OwnedValue::try_from(zvariant::Value::Str(zvariant::Str::from("auto"))).map_err(|e| FerricastError::Connection(format!("zvariant {e}")))?);

    ipv4_props.insert("never-default".into(), OwnedValue::try_from(zvariant::Value::Bool(true)).map_err(|e| FerricastError::Connection(format!("zvariant {e}")))?);

    
    let mut ipv6_props: HashMap<String, OwnedValue> = HashMap::new();
    ipv6_props.insert("method".into(), OwnedValue::try_from(zvariant::Value::Str(zvariant::Str::from("auto"))).map_err(|e| FerricastError::Connection(format!("zvariant {e}")))?);

    ipv6_props.insert("never-default".into(), OwnedValue::try_from(zvariant::Value::Bool(true)).map_err(|e| FerricastError::Connection(format!("zvariant {e}")))?);

        ipv6_props.insert("may-fail".into(), OwnedValue::try_from(zvariant::Value::Bool(true)).map_err(|e| FerricastError::Connection(format!("zvariant {e}")))?);

 

    let mut dict = HashMap::new();
    dict.insert("connection".into(), connection_settings);
    dict.insert("wifi-p2p".into(), p2p_settings);
    dict.insert("ipv4".into(), ipv4_props);
    dict.insert("ipv6".into(), ipv6_props);

    Ok(dict)
}

/// Polls the NM active connection state until ACTIVATED, then returns the
/// sink's IP address (the P2P group owner's address, our gateway).
async fn wait_for_ip(conn: &Connection, active_path: &OwnedObjectPath) -> Result<Ipv4Addr> {
    let active = ActiveConnectionProxy::new(conn, active_path.clone())
        .await
        .map_err(|e| {
            FerricastError::Connection(format!("Cannot create ActiveConnection proxy: {e}"))
        })?;

    tokio::time::timeout(P2P_CONNECT_TIMEOUT, async {
        let mut interval = tokio::time::interval(Duration::from_millis(500));
        let mut last_state = u32::MAX;
        loop {
            interval.tick().await;

            match active.state().await {
                Ok(NM_ACTIVE_CONNECTION_STATE_ACTIVATED) => {
                    tracing::info!("P2P connection ACTIVATED");
                    return extract_gateway(conn, &active).await;
                }
                Ok(NM_ACTIVE_CONNECTION_STATE_DEACTIVATED) => {
                    return Err(FerricastError::Connection(
                        "P2P connection deactivated before IP assignment".into(),
                    ));
                }
                // NM is tearing the connection down — GO negotiation or WPS
                // failed; wpa_supplicant will finish deactivating momentarily.
                Ok(NM_ACTIVE_CONNECTION_STATE_DEACTIVATING) => {
                    return Err(FerricastError::Connection(
                        "P2P connection deactivating — GO negotiation or WPS failed".into(),
                    ));
                }
                // NM removed the ActiveConnection object before we polled.
                Err(e) => {
                    return Err(FerricastError::Connection(format!(
                        "P2P active-connection proxy gone (NM cleaned up?): {e}"
                    )));
                }
                Ok(state) => {
                    if state != last_state {
                        last_state = state;
                        let state_text = match state {
                            1 => "NM_ACTIVE_CONNECTION_STATE_ACTIVATING",
                            2 => "NM_ACTIVE_CONNECTION_STATE_ACTIVATED",
                            3 => "NM_ACTIVE_CONNECTION_STATE_DEACTIVATING",
                            4 => "NM_ACTIVE_CONNECTION_STATE_DEACTIVATED",
                            _ => "NM_ACTIVE_CONNECTION_STATE_UNKNOWN",
                        };

                        tracing::info!(state_text, "P2P NM connection state changed");
                    }
                }
            }
        }
    })
    .await
    .map_err(|_| {
        FerricastError::Connection(format!(
            "P2P connection timed out after {}s",
            P2P_CONNECT_TIMEOUT.as_secs()
        ))
    })?
}

async fn extract_gateway(
    conn: &Connection,
    active: &ActiveConnectionProxy<'_>,
) -> Result<Ipv4Addr> {
    let ip4_path = active.ip4_config().await.map_err(|e| {
        FerricastError::Connection(format!("ActiveConnection.Ip4Config: {e}"))
    })?;

    let ip4 = Ip4ConfigProxy::new(conn, ip4_path).await.map_err(|e| {
        FerricastError::Connection(format!("Cannot create Ip4Config proxy: {e}"))
    })?;

    let gateway = ip4.gateway().await.map_err(|e| {
        FerricastError::Connection(format!("IP4Config.Gateway: {e}"))
    })?;

    if !gateway.is_empty() {
        return gateway.parse::<Ipv4Addr>().map_err(|e| {
            FerricastError::Connection(format!("Cannot parse gateway IP '{gateway}': {e}"))
        });
    }

    // Gateway is empty → our machine is the P2P Group Owner.
    // Sink's IP is in the ARP cache on the p2p-* interface.
    tracing::debug!("NM IP4Config.Gateway empty — we are GO; falling back to ARP lookup");
    find_p2p_peer_ip().await
}

// ── WFD source IEs ───────────────────────────────────────────────────────────

/// WFD Device Information subelement (ID=0, length=6) for a Miracast source:
///   bits[1:0]  = 0b00  (WFD Source)
///   bits[5:4]  = 0b01  (Session Available)
///   → device_info = 0x0010
///
/// 0x0090 (used previously) additionally sets bit 7 (Tunneled TDLS Support)
/// which is not required and causes some TVs to reject the connection.
pub(crate) fn wfd_source_ies() -> Vec<u8> {
    vec![
        0x00,       // Subelement ID: WFD Device Information
        0x00, 0x06, // Length: 6 bytes
        0x00, 0x10, // Device Info: Source, session available
        0x1C, 0x44, // RTSP port: 7236 (big-endian)
        0x00, 0x32, // Max throughput: 50 Mbps
    ]
}

// ── iwd helpers ──────────────────────────────────────────────────────────────

/// Polls `/proc/net/arp` until a complete ARP entry appears on a `p2p-*` interface.
///
/// iwd populates the ARP cache after DHCP completes; this may take a few
/// hundred milliseconds after `IwdP2pPeer.connect()` returns.
async fn find_p2p_peer_ip() -> Result<Ipv4Addr> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let content = tokio::fs::read_to_string("/proc/net/arp")
            .await
            .map_err(|e| FerricastError::Connection(format!("Cannot read /proc/net/arp: {e}")))?;

        // /proc/net/arp columns: IP address, HW type, Flags, HW address, Mask, Device
        // Flags 0x0 = incomplete; 0x2 = complete; 0x6 = published+complete.
        for line in content.lines().skip(1) {
            let mut cols = line.split_whitespace();
            let ip_str = cols.next().unwrap_or("");
            let _ = cols.next(); // HW type
            let flags = cols.next().unwrap_or("0x0");
            let _ = cols.next(); // HW address
            let _ = cols.next(); // Mask
            let dev = cols.next().unwrap_or("");

            if dev.starts_with("p2p-") && flags != "0x0" {
                if let Ok(ip) = ip_str.parse::<Ipv4Addr>() {
                    return Ok(ip);
                }
            }
        }

        if tokio::time::Instant::now() >= deadline {
            return Err(FerricastError::Connection(
                "Timed out waiting for P2P peer IP in ARP cache (p2p-* interface)".into(),
            ));
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

// ── RTSP helpers ─────────────────────────────────────────────────────────────

struct RtspMessage {
    start_line: String,
    headers: HashMap<String, String>,
    /// Body bytes, populated when Content-Length is present.
    /// Used for M3 GET_PARAMETER and other body-carrying messages.
    #[allow(dead_code)]
    body: String,
}

async fn send_rtsp(stream: &mut BufReader<TcpStream>, msg: &str) -> Result<()> {
    tracing::trace!(msg = msg.trim_end(), "RTSP →");
    stream
        .get_mut()
        .write_all(msg.as_bytes())
        .await
        .map_err(|e| FerricastError::Protocol(format!("RTSP write: {e}")))
}

async fn recv_rtsp(stream: &mut BufReader<TcpStream>) -> Result<RtspMessage> {
    let mut header_lines: Vec<String> = Vec::new();

    loop {
        let mut line = String::new();
        stream
            .read_line(&mut line)
            .await
            .map_err(|e| FerricastError::Protocol(format!("RTSP read: {e}")))?;
        let trimmed = line.trim_end_matches(['\r', '\n']).to_string();
        if trimmed.is_empty() {
            break;
        }
        header_lines.push(trimmed);
    }

    let start_line = header_lines
        .first()
        .cloned()
        .unwrap_or_default();

    let mut headers: HashMap<String, String> = HashMap::new();
    for line in header_lines.iter().skip(1) {
        if let Some((k, v)) = line.split_once(": ") {
            headers.insert(k.to_lowercase(), v.to_string());
        }
    }

    let body = if let Some(len) = headers
        .get("content-length")
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        let mut buf = vec![0u8; len];
        stream
            .read_exact(&mut buf)
            .await
            .map_err(|e| FerricastError::Protocol(format!("RTSP body read: {e}")))?;
        String::from_utf8_lossy(&buf).into_owned()
    } else {
        String::new()
    };

    tracing::trace!(start_line, "RTSP ←");
    Ok(RtspMessage { start_line, headers, body })
}

fn ensure_200(msg: &RtspMessage, cseq: u32) -> Result<()> {
    if msg.start_line.contains("200") {
        Ok(())
    } else {
        Err(FerricastError::Protocol(format!(
            "Expected 200 OK for CSeq {cseq}, got: {}",
            msg.start_line
        )))
    }
}

fn header_value<'m>(
    headers: &'m HashMap<String, String>,
    key: &str,
) -> Option<&'m str> {
    headers.get(key).map(String::as_str)
}

/// Parses the first `client_port` value from a RTSP Transport header.
///
/// Example: `RTP/AVP/UDP;unicast;client_port=1028-1029` → `1028`
fn parse_client_port(transport: &str) -> Option<u16> {
    transport
        .split(';')
        .find(|part| part.trim_start().starts_with("client_port="))
        .and_then(|part| part.split('=').nth(1))
        .and_then(|ports| ports.split('-').next())
        .and_then(|p| p.parse().ok())
}

// ── WFD capability strings ────────────────────────────────────────────────────

/// Builds the WFD GET_PARAMETER response body advertising our capabilities.
///
/// Video: H.264 CBP Level 3.2, 1920×1080p@30 (CEA mode 7 = bit 7 = 0x80).
/// Audio: LPCM stereo 44.1/48 kHz.
/// RTP:   UDP, port `rtp_port`.
fn wfd_capabilities(rtp_port: u16) -> String {
    // wfd_video_formats field layout (WFD spec §Table 27):
    // native | preferred_display_mode_supported |
    // <profile> <level> <CEA-bitmask> <VESA-bitmask> <HH-bitmask>
    // <latency> <min_slice_size> <slice_enc_params> <frame_rate_control>
    // [max_hres max_vres]
    //
    // Profile byte: bit0=CBP, bit1=CHP
    // Level byte: bit0=3.1, bit1=3.2, bit2=4, bit3=4.1, bit4=4.2
    // CEA bit 7 = 1920×1080p@30
    format!(
        "wfd_video_formats: 00 00 01 02 00000080 00000000 00000000 00 0000 0000 00 none none\r\n\
         wfd_audio_codecs: LPCM 00000003 00\r\n\
         wfd_client_rtp_ports: RTP/AVP/UDP;unicast {rtp_port} 0 mode=play\r\n\
         wfd_uibc_capability: none\r\n\
         wfd_standby_resume_capability: none\r\n"
    )
}

/// Builds the WFD SET_PARAMETER (M4) body selecting parameters to use.
fn wfd_set_parameter(presentation_url: &str, rtp_port: u16) -> String {
    format!(
        "wfd_video_formats: 00 00 01 02 00000080 00000000 00000000 00 0000 0000 00 none none\r\n\
         wfd_audio_codecs: LPCM 00000003 00\r\n\
         wfd_presentation_URL: {presentation_url} none\r\n\
         wfd_client_rtp_ports: RTP/AVP/UDP;unicast {rtp_port} 0 mode=play\r\n"
    )
}

// ── MPEG-TS inline packetizer ─────────────────────────────────────────────────

/// Builds a 188-byte PAT (Program Association Table) TS packet.
fn build_pat(cc: &mut u8) -> [u8; TS_PACKET] {
    let mut pkt = [0xFFu8; TS_PACKET];
    // TS header
    pkt[0] = 0x47;
    pkt[1] = 0x40 | (PID_PAT >> 8) as u8; // PUSI=1
    pkt[2] = PID_PAT as u8;
    pkt[3] = 0x10 | (*cc & 0x0F); // no AFC, payload only
    *cc = cc.wrapping_add(1) & 0x0F;

    // Pointer field
    pkt[4] = 0x00;

    // PAT section
    let pat = &mut pkt[5..];
    pat[0] = 0x00; // table_id = PAT
    pat[1] = 0xB0; // section_syntax_indicator=1, length high nibble=0
    pat[2] = 0x0D; // section_length = 13
    pat[3] = 0x00; // transport_stream_id high
    pat[4] = 0x01; // transport_stream_id low
    pat[5] = 0xC1; // version=0, current=1
    pat[6] = 0x00; // section_number
    pat[7] = 0x00; // last_section_number
    // Program 1 → PMT PID 0x1000
    pat[8] = 0x00;
    pat[9] = 0x01;
    pat[10] = 0xE0 | (PID_PMT >> 8) as u8;
    pat[11] = PID_PMT as u8;
    // CRC32 over bytes [5..5+12]
    let crc = crc32_mpeg(&pkt[5..5 + 12]);
    pkt[17] = (crc >> 24) as u8;
    pkt[18] = (crc >> 16) as u8;
    pkt[19] = (crc >> 8) as u8;
    pkt[20] = crc as u8;

    pkt
}

/// Builds a 188-byte PMT (Program Map Table) TS packet for one H.264 video stream.
fn build_pmt(cc: &mut u8) -> [u8; TS_PACKET] {
    let mut pkt = [0xFFu8; TS_PACKET];
    pkt[0] = 0x47;
    pkt[1] = 0x40 | (PID_PMT >> 8) as u8;
    pkt[2] = PID_PMT as u8;
    pkt[3] = 0x10 | (*cc & 0x0F);
    *cc = cc.wrapping_add(1) & 0x0F;

    pkt[4] = 0x00; // pointer field

    let pmt = &mut pkt[5..];
    pmt[0] = 0x02; // table_id = PMT
    pmt[1] = 0xB0;
    pmt[2] = 0x12; // section_length = 18
    pmt[3] = 0x00;
    pmt[4] = 0x01; // program_number
    pmt[5] = 0xC1; // version=0, current=1
    pmt[6] = 0x00;
    pmt[7] = 0x00;
    // PCR PID = video PID
    pmt[8] = 0xE0 | (PID_VIDEO >> 8) as u8;
    pmt[9] = PID_VIDEO as u8;
    // program_info_length = 0
    pmt[10] = 0xF0;
    pmt[11] = 0x00;
    // Elementary stream: H.264
    pmt[12] = STREAM_TYPE_H264;
    pmt[13] = 0xE0 | (PID_VIDEO >> 8) as u8;
    pmt[14] = PID_VIDEO as u8;
    // ES_info_length = 0
    pmt[15] = 0xF0;
    pmt[16] = 0x00;
    // CRC32
    let crc = crc32_mpeg(&pkt[5..5 + 17]);
    pkt[22] = (crc >> 24) as u8;
    pkt[23] = (crc >> 16) as u8;
    pkt[24] = (crc >> 8) as u8;
    pkt[25] = crc as u8;

    pkt
}

/// Builds a PES packet wrapping `payload` (H.264 Annex-B NALUs).
fn build_pes(payload: &[u8], (pts_us, dts_us): (u64, u64)) -> Vec<u8> {
    let pts_90k = pts_us * 90_000 / 1_000_000;
    let dts_90k = dts_us * 90_000 / 1_000_000;
    let pts_dts_differ = pts_90k != dts_90k;

    // PES optional header: flags + pts [+ dts]
    let pts_dts_flags: u8 = if pts_dts_differ { 0xC0 } else { 0x80 };
    let header_data_len: u8 = if pts_dts_differ { 10 } else { 5 };

    let mut pes: Vec<u8> = Vec::with_capacity(9 + header_data_len as usize + payload.len());
    // start code prefix
    pes.extend_from_slice(&[0x00, 0x00, 0x01]);
    pes.push(PES_STREAM_ID_VIDEO);
    // PES packet length = 0 (unbounded for video)
    pes.push(0x00);
    pes.push(0x00);
    // PES header flags
    pes.push(0x80); // marker bits
    pes.push(pts_dts_flags);
    pes.push(header_data_len);

    fn encode_pts(ts: u64, prefix: u8) -> [u8; 5] {
        [
            prefix | (((ts >> 30) & 0x07) as u8) << 1 | 0x01,
            ((ts >> 22) & 0xFF) as u8,
            (((ts >> 15) & 0x7F) as u8) << 1 | 0x01,
            ((ts >> 7) & 0xFF) as u8,
            ((ts & 0x7F) as u8) << 1 | 0x01,
        ]
    }

    if pts_dts_differ {
        pes.extend_from_slice(&encode_pts(pts_90k, 0x31));
        pes.extend_from_slice(&encode_pts(dts_90k, 0x11));
    } else {
        pes.extend_from_slice(&encode_pts(pts_90k, 0x21));
    }

    pes.extend_from_slice(payload);
    pes
}

/// Fragments a PES packet into 188-byte TS packets on `pid`.
fn packetize_pes_into_ts(pes: &[u8], pid: u16, cc: &mut u8, out: &mut Vec<u8>) {
    let pid_hi = 0xE0 | ((pid >> 8) as u8); // no error, payload start, priority
    let pid_lo = pid as u8;

    let mut first = true;
    let mut remaining = pes;

    while !remaining.is_empty() {
        let mut pkt = [0xFFu8; TS_PACKET];
        pkt[0] = 0x47;
        pkt[1] = if first { pid_hi | 0x40 } else { pid_hi & !0x40 }; // PUSI
        pkt[2] = pid_lo;
        pkt[3] = 0x10 | (*cc & 0x0F); // payload only
        *cc = cc.wrapping_add(1) & 0x0F;

        let payload_space = TS_PACKET - 4;
        let chunk_len = remaining.len().min(payload_space);
        pkt[4..4 + chunk_len].copy_from_slice(&remaining[..chunk_len]);
        // Stuffing bytes (0xFF) already filled from the array initialiser.

        out.extend_from_slice(&pkt);
        remaining = &remaining[chunk_len..];
        first = false;
    }
}

// ── RTP ───────────────────────────────────────────────────────────────────────

fn build_rtp_packet(seq: u16, ts: u32, ssrc: u32, payload: &[u8]) -> Vec<u8> {
    let mut pkt = Vec::with_capacity(12 + payload.len());
    pkt.push(0x80); // V=2, P=0, X=0, CC=0
    pkt.push(RTP_PT_MP2T); // M=0, PT=33
    pkt.push((seq >> 8) as u8);
    pkt.push(seq as u8);
    pkt.push((ts >> 24) as u8);
    pkt.push((ts >> 16) as u8);
    pkt.push((ts >> 8) as u8);
    pkt.push(ts as u8);
    pkt.push((ssrc >> 24) as u8);
    pkt.push((ssrc >> 16) as u8);
    pkt.push((ssrc >> 8) as u8);
    pkt.push(ssrc as u8);
    pkt.extend_from_slice(payload);
    pkt
}

// ── CRC32/MPEG (ISO 13818-1) ──────────────────────────────────────────────────

fn crc32_mpeg(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        for bit in (0..8).rev() {
            let b = (byte >> bit) & 1;
            let msb = (crc >> 31) as u8;
            crc <<= 1;
            if msb ^ b != 0 {
                crc ^= 0x0400_4C11;
            }
        }
    }
    crc
}
