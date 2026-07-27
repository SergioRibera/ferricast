use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use zbus::Connection;
use zbus::zvariant::{OwnedObjectPath, OwnedValue, Value};

use ferricast_core::{AudioFrame, CastSession, Device, EncodedFrame, FerricastError, Result, StreamConfig};

use crate::dbus::{
    ActiveConnectionProxy, DeviceProxy, IwdP2pPeerProxy, IwdSimpleConfigurationProxy,
    NM_ACTIVE_CONNECTION_STATE_ACTIVATED, NM_ACTIVE_CONNECTION_STATE_DEACTIVATED,
    NM_ACTIVE_CONNECTION_STATE_DEACTIVATING, NM_DEVICE_TYPE_WIFI_P2P, NetworkManagerProxy,
    P2pPeerProxy, WifiP2pProxy,
};

// ── constants ─────────────────────────────────────────────────────────────────

/// WFD default RTSP port (Wi-Fi Display spec §6.5).
/// The WFD Source listens on this port; the Sink connects to it.
pub(crate) const WFD_DEFAULT_PORT: u16 = 7236;
/// RTP payload type for MPEG-TS (RFC 2250).
const RTP_PT_MP2T: u8 = 33;
/// 90 kHz MPEG-TS / RTP clock.
const CLOCK_90K: u64 = 90_000;
/// MPEG-TS packet size.
const TS_PACKET: usize = 188;
/// TS packets per RTP packet — keeps payload ≤ 1316 bytes.
const TS_PER_RTP: usize = 7;
/// How long to wait for the P2P connection to reach ACTIVATED.
const P2P_CONNECT_TIMEOUT: Duration = Duration::from_secs(120);
/// How long to wait for the sink to connect to our RTSP server after P2P.
const RTSP_ACCEPT_TIMEOUT: Duration = Duration::from_secs(30);

const PID_PAT: u16 = 0x0000;
const PID_PMT: u16 = 0x1000;
const PID_VIDEO: u16 = 0x0100;
const PID_AUDIO: u16 = 0x0101;
const STREAM_TYPE_H264: u8 = 0x1B;
const STREAM_TYPE_AAC: u8 = 0x0F; // ISO 13818-7 (MPEG-2 AAC / ADTS)
const PES_STREAM_ID_VIDEO: u8 = 0xE0;
const PES_STREAM_ID_AUDIO: u8 = 0xC0;

// ── RTSP event (sink → source during streaming) ───────────────────────────────

enum RtspEvent {
    /// Sink sent TEARDOWN — session is ending.
    Teardown,
}

// ── active connection discriminator ──────────────────────────────────────────

enum ActiveConnection {
    Nm(OwnedObjectPath),
    Iwd(OwnedObjectPath),
}

// ── session state ─────────────────────────────────────────────────────────────

/// Miracast (Wi-Fi Display) streaming session.
///
/// Lifecycle: `Default` → `connect()` → `setup_stream()` → `send_frame()` loop → `stop()`.
///
/// The WFD Source (us) listens on TCP port 7236.  After P2P group formation
/// and DHCP, the WFD Sink (TV) connects to port 7236 and the M1–M7 RTSP
/// handshake runs over that accepted connection.
pub struct MiracastSession {
    dbus: Option<Connection>,
    active_connection: Option<ActiveConnection>,
    /// TCP listener on the WFD RTSP port.  Created in `connect()`, consumed
    /// (accepted) in `setup_stream()`.  Used for P2P mode.
    listener: Option<TcpListener>,
    /// Pre-connected outbound TCP stream for infrastructure Miracast (tcp_scan
    /// backend).  When set, `setup_stream()` uses this instead of accepting
    /// from `listener`.
    direct_conn: Option<TcpStream>,
    /// Write half of the RTSP TCP connection, shared with the reader task so it
    /// can respond to sink-initiated PAUSE/PLAY requests.
    rtsp_writer: Option<Arc<Mutex<OwnedWriteHalf>>>,
    /// Incoming events from the sink (IDR requests, TEARDOWN) read by the
    /// background reader task.
    rtsp_events: Option<mpsc::Receiver<RtspEvent>>,
    /// Background task that owns the read half and pumps sink messages.
    rtsp_task: Option<JoinHandle<()>>,
    rtp: Option<UdpSocket>,
    sink_rtp_addr: Option<SocketAddr>,
    session_id: Option<String>,
    presentation_url: Option<String>,
    /// RTSP CSeq counter for source-initiated requests.
    src_cseq: u32,
    /// RTP sequence number.
    rtp_seq: u16,
    /// RTP SSRC (random, fixed per session).
    ssrc: u32,
    /// MPEG-TS continuity counters.
    ts: TsState,
    /// Timestamp of last keep-alive GET_PARAMETER sent to the sink.
    last_keepalive: Option<Instant>,
    /// Set to `true` by the RTSP reader task when the sink sends
    /// `SET_PARAMETER wfd_idr_request`.  Cleared by `poll_keyframe_request`.
    keyframe_requested: Arc<AtomicBool>,
    alive: bool,
}

#[derive(Default)]
struct TsState {
    cc_pat: u8,
    cc_pmt: u8,
    cc_video: u8,
    cc_audio: u8,
    psi_sent: bool,
}

impl Default for MiracastSession {
    fn default() -> Self {
        Self {
            dbus: None,
            active_connection: None,
            listener: None,
            direct_conn: None,
            rtsp_writer: None,
            rtsp_events: None,
            rtsp_task: None,
            rtp: None,
            sink_rtp_addr: None,
            session_id: None,
            presentation_url: None,
            src_cseq: 1,
            rtp_seq: 0,
            ssrc: rand::random(),
            ts: TsState::default(),
            last_keepalive: None,
            keyframe_requested: Arc::new(AtomicBool::new(false)),
            alive: false,
        }
    }
}

impl CastSession for MiracastSession {
    async fn connect(&mut self, device: &Device) -> Result<()> {
        match device.metadata.get("backend").map(String::as_str) {
            Some("iwd") => self.connect_iwd(device).await,
            Some("nm_wifi") => self.connect_nm_wifi(device).await,
            Some("tcp_scan") => self.connect_tcp_scan(device).await,
            _ => self.connect_nm(device).await,
        }
    }

    /// Runs the WFD RTSP M1–M7 handshake over the accepted sink connection
    /// and binds the RTP sender socket.
    ///
    /// The WFD Source (us) listens on port 7236.  The Sink (TV) connects to
    /// us after P2P and DHCP complete.  We then initiate M1 (OPTIONS) and
    /// drive the handshake through M7 (PLAY).
    async fn setup_stream(&mut self, _config: &StreamConfig) -> Result<()> {
        // Bind the RTP socket before accepting so we can advertise the port.
        let rtp_sock = UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|e| FerricastError::Connection(format!("Cannot bind RTP socket: {e}")))?;
        let our_rtp_port = rtp_sock
            .local_addr()
            .map_err(|e| FerricastError::Connection(format!("RTP local_addr: {e}")))?
            .port();

        // Establish the RTSP TCP connection.
        //
        // P2P mode: sink dials our listener after DHCP completes.
        // Infrastructure mode (tcp_scan): we already connected outbound in
        // connect_tcp_scan(); retrieve that stream directly.
        let (tcp, peer_addr, our_p2p_ip) = if let Some(stream) = self.direct_conn.take() {
            let peer = stream
                .peer_addr()
                .map_err(|e| FerricastError::Connection(format!("TCP peer_addr: {e}")))?;
            let local_ip = stream
                .local_addr()
                .map_err(|e| FerricastError::Connection(format!("TCP local_addr: {e}")))?
                .ip();
            tracing::info!(%peer, %local_ip, "Infrastructure Miracast — using pre-connected stream");
            (stream, peer, local_ip)
        } else {
            let listener = self.listener.take().ok_or(FerricastError::NoActiveSession)?;
            let (tcp, addr) = tokio::time::timeout(RTSP_ACCEPT_TIMEOUT, listener.accept())
                .await
                .map_err(|_| {
                    FerricastError::Connection(format!(
                        "Sink did not connect within {}s — check firewall on port {WFD_DEFAULT_PORT}",
                        RTSP_ACCEPT_TIMEOUT.as_secs()
                    ))
                })?
                .map_err(|e| FerricastError::Connection(format!("RTSP accept failed: {e}")))?;
            let p2p_ip = tcp
                .local_addr()
                .map_err(|e| FerricastError::Connection(format!("RTSP socket local_addr: {e}")))?
                .ip();
            tracing::info!(%addr, %p2p_ip, "Sink connected — starting WFD M1–M7 handshake");
            (tcp, addr, p2p_ip)
        };

        let mut rtsp = BufReader::new(tcp);

        // M1: Source → Sink: OPTIONS (announce WFD requirement)
        send_rtsp(
            rtsp.get_mut(),
            &format!(
                "OPTIONS * RTSP/1.0\r\nCSeq: {}\r\nRequire: org.wfa.wfd1.0\r\n\r\n",
                self.src_cseq
            ),
        )
        .await?;
        let m1_resp = recv_rtsp(&mut rtsp).await?;
        ensure_200(&m1_resp, self.src_cseq)?;
        self.src_cseq += 1;

        // M2 / alt-M3: read the sink's next message.
        //
        // Per spec the sink sends OPTIONS (M2).  Some sinks deviate and send
        // GET_PARAMETER first asking for source capabilities (alt-M3 flow).
        // Handle both.
        let next = recv_rtsp(&mut rtsp).await?;
        let next_cseq = header_value(&next.headers, "cseq")
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(100);

        // Returns (sink_primary_rtp, sink_secondary_rtp) for use in M4.
        let (sink_primary_rtp, sink_secondary_rtp) =
            if next.start_line.starts_with("GET_PARAMETER") {
                // Alt-M3: sink is querying source capabilities first.
                let caps = wfd_capabilities();
                send_rtsp(
                    rtsp.get_mut(),
                    &format!(
                        "RTSP/1.0 200 OK\r\nCSeq: {next_cseq}\r\nContent-Type: text/parameters\r\nContent-Length: {}\r\n\r\n{caps}",
                        caps.len()
                    ),
                )
                .await?;
                // Skip normal M3; sink's port will be confirmed in M6.
                (16384, 16385)
            } else {
                // Normal M2: sink sends OPTIONS.
                send_rtsp(
                    rtsp.get_mut(),
                    &format!(
                        "RTSP/1.0 200 OK\r\nCSeq: {next_cseq}\r\nPublic: org.wfa.wfd1.0, GET_PARAMETER, SET_PARAMETER, SETUP, PLAY, PAUSE, TEARDOWN\r\n\r\n"
                    ),
                )
                .await?;

                // M3: Source → Sink: GET_PARAMETER (query sink WFD capabilities)
                // Body is bare parameter names — no colon/value — per WFD spec §6.8.
                let m3_body = "wfd_client_rtp_ports\r\n\
                               wfd_audio_codecs\r\n\
                               wfd_video_formats\r\n\
                               wfd_display_edid\r\n\
                               wfd_idr_request_capability\r\n\
                               microsoft_cursor\r\n";
                send_rtsp(
                    rtsp.get_mut(),
                    &format!(
                        "GET_PARAMETER rtsp://localhost/wfd1.0 RTSP/1.0\r\nCSeq: {}\r\nContent-Type: text/parameters\r\nContent-Length: {}\r\n\r\n{m3_body}",
                        self.src_cseq,
                        m3_body.len()
                    ),
                )
                .await?;
                let m3_resp = recv_rtsp(&mut rtsp).await?;
                self.src_cseq += 1;
                parse_sink_rtp_ports(&m3_resp.body)
            };

        // M4: Source → Sink: SET_PARAMETER (announce chosen streaming params)
        let presentation_url = format!(
            "rtsp://{}:{WFD_DEFAULT_PORT}/wfd1.0/streamid=0",
            our_p2p_ip
        );
        let m4_body = wfd_set_parameter(&presentation_url, sink_primary_rtp, sink_secondary_rtp);
        send_rtsp(
            rtsp.get_mut(),
            &format!(
                "SET_PARAMETER rtsp://localhost/wfd1.0 RTSP/1.0\r\nCSeq: {}\r\nContent-Type: text/parameters\r\nContent-Length: {}\r\n\r\n{m4_body}",
                self.src_cseq,
                m4_body.len()
            ),
        )
        .await?;
        let m4_resp = recv_rtsp(&mut rtsp).await?;
        ensure_200(&m4_resp, self.src_cseq)?;
        self.src_cseq += 1;

        // M5: Source → Sink: SET_PARAMETER wfd_trigger_method: SETUP
        let trigger_body = "wfd_trigger_method: SETUP\r\n";
        send_rtsp(
            rtsp.get_mut(),
            &format!(
                "SET_PARAMETER rtsp://localhost/wfd1.0 RTSP/1.0\r\nCSeq: {}\r\nContent-Type: text/parameters\r\nContent-Length: {}\r\n\r\n{trigger_body}",
                self.src_cseq,
                trigger_body.len()
            ),
        )
        .await?;
        let m5_resp = recv_rtsp(&mut rtsp).await?;
        ensure_200(&m5_resp, self.src_cseq)?;
        self.src_cseq += 1;

        // M6: Sink → Source: SETUP (negotiate RTP transport)
        let m6_req = recv_rtsp(&mut rtsp).await?;
        let m6_cseq = header_value(&m6_req.headers, "cseq")
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(200);

        // The sink's Transport header tells us which UDP port to send RTP to.
        // Fall back to the port negotiated in M3/M4 if the header is absent.
        let sink_rtp_port = parse_client_port(
            header_value(&m6_req.headers, "transport").unwrap_or(""),
        )
        .unwrap_or(sink_primary_rtp);
        let sink_rtp_addr = SocketAddr::new(peer_addr.ip(), sink_rtp_port);

        let session_id = format!("{:016x}", rand::random::<u64>());
        send_rtsp(
            rtsp.get_mut(),
            &format!(
                "RTSP/1.0 200 OK\r\nCSeq: {m6_cseq}\r\nSession: {session_id};timeout=60\r\nTransport: RTP/AVP/UDP;unicast;client_port={sink_rtp_port};server_port={our_rtp_port}\r\n\r\n"
            ),
        )
        .await?;

        // M7: Sink → Source: PLAY (streaming begins)
        let m7_req = recv_rtsp(&mut rtsp).await?;
        let m7_cseq = header_value(&m7_req.headers, "cseq")
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(m6_cseq + 1);
        send_rtsp(
            rtsp.get_mut(),
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

        // Split the TCP connection: write half stays here for keep-alive / TEARDOWN;
        // read half goes to a background task that pumps incoming sink messages.
        let (read_half, write_half) = rtsp.into_inner().into_split();
        let writer = Arc::new(Mutex::new(write_half));
        let (event_tx, event_rx) = mpsc::channel::<RtspEvent>(8);
        let task = tokio::spawn(rtsp_reader_task(
            BufReader::new(read_half),
            Arc::clone(&writer),
            session_id.clone(),
            event_tx,
            Arc::clone(&self.keyframe_requested),
        ));

        self.rtp = Some(rtp_sock);
        self.sink_rtp_addr = Some(sink_rtp_addr);
        self.session_id = Some(session_id);
        self.presentation_url = Some(presentation_url);
        self.last_keepalive = Some(Instant::now());
        self.rtsp_writer = Some(writer);
        self.rtsp_events = Some(event_rx);
        self.rtsp_task = Some(task);
        Ok(())
    }

    /// Packetizes one video frame into MPEG-TS and sends it over RTP/UDP.
    async fn send_frame(&mut self, frame: &EncodedFrame) -> Result<()> {
        // Drain sink events from the background RTSP reader.
        // IDR requests are handled via `keyframe_requested` AtomicBool (polled
        // by the manager via `poll_keyframe_request`).
        if let Some(events) = self.rtsp_events.as_mut() {
            while let Ok(event) = events.try_recv() {
                match event {
                    RtspEvent::Teardown => {
                        tracing::info!("Sink sent TEARDOWN — stopping session");
                        self.alive = false;
                    }
                }
            }
        }

        // Keep-alive: send GET_PARAMETER every 25 s (GNOME interval).
        // The background reader task will consume the sink's 200 OK response.
        if self
            .last_keepalive
            .map(|t| t.elapsed() >= Duration::from_secs(25))
            .unwrap_or(false)
        {
            if let (Some(writer), Some(session_id)) =
                (self.rtsp_writer.as_ref(), self.session_id.as_deref())
            {
                let mut w = writer.lock().await;
                let _ = send_rtsp(
                    &mut *w,
                    &format!(
                        "GET_PARAMETER rtsp://localhost/wfd1.0 RTSP/1.0\r\nCSeq: {}\r\nSession: {session_id}\r\n\r\n",
                        self.src_cseq
                    ),
                )
                .await;
                self.src_cseq += 1;
            }
            self.last_keepalive = Some(Instant::now());
        }

        let sock = self.rtp.as_ref().ok_or(FerricastError::NoActiveSession)?;
        let sink = self.sink_rtp_addr.ok_or(FerricastError::NoActiveSession)?;

        let rtp_ts = (frame.timestamp_us * CLOCK_90K / 1_000_000) as u32;

        let mut ts_packets: Vec<u8> = Vec::new();

        // Emit PAT + PMT once at stream start (and on every keyframe for robustness).
        if !self.ts.psi_sent || frame.is_keyframe {
            ts_packets.extend_from_slice(&build_pat(&mut self.ts.cc_pat));
            ts_packets.extend_from_slice(&build_pmt(&mut self.ts.cc_pmt));
            self.ts.psi_sent = true;
        }

        // Wrap H.264 Annex-B data in a PES packet, then TS-packetize it.
        // PCR is embedded in the first TS packet of every video frame so the
        // sink can recover the 90 kHz clock and maintain A/V sync.
        let pes = build_pes(&frame.data, frame.pts_dts);
        let pcr_90k = (frame.pts_dts.0 * CLOCK_90K / 1_000_000) & 0x1_FFFF_FFFF;
        packetize_pes_into_ts(&pes, PID_VIDEO, &mut self.ts.cc_video, Some(pcr_90k), &mut ts_packets);

        // Send TS packets grouped into RTP packets (≤ 7 per packet).
        for chunk in ts_packets.chunks(TS_PER_RTP * TS_PACKET) {
            let pkt = build_rtp_packet(self.rtp_seq, rtp_ts, self.ssrc, chunk);
            sock.send_to(&pkt, sink).await.map_err(|e| {
                FerricastError::Streaming(format!("RTP send failed: {e}"))
            })?;
            self.rtp_seq = self.rtp_seq.wrapping_add(1);
        }

        Ok(())
    }

    /// Packetizes one AAC audio frame (ADTS-framed) into MPEG-TS and sends it over RTP/UDP.
    async fn send_audio_frame(&mut self, frame: &AudioFrame) -> Result<()> {
        let sock = self.rtp.as_ref().ok_or(FerricastError::NoActiveSession)?;
        let sink = self.sink_rtp_addr.ok_or(FerricastError::NoActiveSession)?;

        let rtp_ts = (frame.timestamp_us * CLOCK_90K / 1_000_000) as u32;

        let mut ts_packets: Vec<u8> = Vec::new();
        let pes = build_pes_audio(&frame.data, frame.timestamp_us);
        packetize_pes_into_ts(&pes, PID_AUDIO, &mut self.ts.cc_audio, None, &mut ts_packets);

        for chunk in ts_packets.chunks(TS_PER_RTP * TS_PACKET) {
            let pkt = build_rtp_packet(self.rtp_seq, rtp_ts, self.ssrc, chunk);
            sock.send_to(&pkt, sink).await.map_err(|e| {
                FerricastError::Streaming(format!("RTP audio send failed: {e}"))
            })?;
            self.rtp_seq = self.rtp_seq.wrapping_add(1);
        }

        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        // Send TEARDOWN only if we were streaming; always clean up resources
        // so that sink-initiated TEARDOWN (which sets alive=false) doesn't
        // skip task abort and resource release.
        let was_alive = self.alive;
        self.alive = false;

        if was_alive {
            if let (Some(writer), Some(session_id)) =
                (self.rtsp_writer.as_ref(), self.session_id.as_deref())
            {
                let url = self
                    .presentation_url
                    .as_deref()
                    .unwrap_or("rtsp://localhost/wfd1.0/streamid=0");
                let mut w = writer.lock().await;
                let _ = send_rtsp(
                    &mut *w,
                    &format!(
                        "TEARDOWN {url} RTSP/1.0\r\nCSeq: {}\r\nSession: {session_id}\r\n\r\n",
                        self.src_cseq
                    ),
                )
                .await;
            }
        }

        // Abort the background reader task and release all RTSP state.
        if let Some(task) = self.rtsp_task.take() {
            task.abort();
        }
        self.rtsp_writer = None;
        self.rtsp_events = None;
        self.rtp = None;
        self.listener = None;
        self.direct_conn = None;

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

    fn poll_keyframe_request(&mut self) -> bool {
        self.keyframe_requested.swap(false, Ordering::Relaxed)
    }
}

// ── backend connect helpers ───────────────────────────────────────────────────

impl MiracastSession {
    async fn connect_nm(&mut self, device: &Device) -> Result<()> {
        let peer_dbus_path = device
            .metadata
            .get("path")
            .ok_or_else(|| {
                FerricastError::Connection("Missing 'path' in device metadata".into())
            })
            .and_then(|s| {
                OwnedObjectPath::try_from(s.as_str()).map_err(|e| {
                    FerricastError::Connection(format!("Invalid peer D-Bus path: {e}"))
                })
            })?;

        let p2p_device_path = device
            .metadata
            .get("device_path")
            .ok_or_else(|| {
                FerricastError::Connection("Missing 'device_path' in device metadata".into())
            })
            .and_then(|s| {
                OwnedObjectPath::try_from(s.as_str()).map_err(|e| {
                    FerricastError::Connection(format!("Invalid device D-Bus path: {e}"))
                })
            })?;

        let hw_address = device
            .metadata
            .get("hw_address")
            .ok_or_else(|| {
                FerricastError::Connection("Missing 'hw_address' in device metadata".into())
            })?
            .clone();

        // Bind the RTSP listener BEFORE starting P2P so the sink can connect
        // immediately after DHCP without racing against our bind.
        let wfd_port = if device.port == 0 { WFD_DEFAULT_PORT } else { device.port };
        let listener = TcpListener::bind(format!("0.0.0.0:{wfd_port}"))
            .await
            .map_err(|e| {
                FerricastError::Connection(format!(
                    "Cannot bind RTSP port {wfd_port} (another process may own it): {e}"
                ))
            })?;
        tracing::info!(port = wfd_port, "RTSP listener bound — waiting for sink");

        let conn = Connection::system().await.map_err(|e| {
            FerricastError::Connection(format!("D-Bus system bus unavailable: {e}"))
        })?;
        let nm = NetworkManagerProxy::new(&conn).await.map_err(|e| {
            FerricastError::Connection(format!("Cannot reach NetworkManager: {e}"))
        })?;

        let connection_dict = build_p2p_connection_dict(&hw_address)?;

        let mut activate_opts: HashMap<String, Value> = HashMap::new();
        activate_opts.insert("persist".into(), Value::from("volatile"));
        // dbus-client: NM drops the connection when our D-Bus client exits,
        // preventing stale P2P groups if ferricast crashes or is killed.
        activate_opts.insert(
            "bind-activation".into(),
            Value::Str(zvariant::Str::from("dbus-client")),
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
        wait_for_activation(&conn, &active_path).await?;
        tracing::info!("P2P DHCP complete — sink should connect to port {wfd_port} shortly");

        self.dbus = Some(conn);
        self.active_connection = Some(ActiveConnection::Nm(active_path));
        self.listener = Some(listener);
        self.alive = true;
        Ok(())
    }

    async fn connect_iwd(&mut self, device: &Device) -> Result<()> {
        let peer_path = device
            .metadata
            .get("path")
            .ok_or_else(|| {
                FerricastError::Connection("Missing 'path' in device metadata".into())
            })
            .and_then(|s| {
                OwnedObjectPath::try_from(s.as_str()).map_err(|e| {
                    FerricastError::Connection(format!("Invalid peer D-Bus path: {e}"))
                })
            })?;

        // Bind the RTSP listener BEFORE P2P so we don't miss the sink's connect.
        let wfd_port = if device.port == 0 { WFD_DEFAULT_PORT } else { device.port };
        let listener = TcpListener::bind(format!("0.0.0.0:{wfd_port}"))
            .await
            .map_err(|e| {
                FerricastError::Connection(format!(
                    "Cannot bind RTSP port {wfd_port}: {e}"
                ))
            })?;
        tracing::info!(port = wfd_port, "RTSP listener bound — waiting for sink");

        let conn = Connection::system().await.map_err(|e| {
            FerricastError::Connection(format!("D-Bus system bus unavailable: {e}"))
        })?;
        // WFD sinks require WPS-PBC group formation via SimpleConfiguration.PushButton(),
        // not p2p.Peer.Connect() (which returns NotFound for WFD peers under iwd).
        let wsc = IwdSimpleConfigurationProxy::new(&conn, peer_path.clone())
            .await
            .map_err(|e| FerricastError::Connection(format!("iwd SimpleConfiguration proxy: {e}")))?;

        tracing::info!(%peer_path, "WPS-PBC connecting to iwd WFD sink (group formation + DHCP)");

        // PushButton blocks until P2P group formation and DHCP complete.
        tokio::time::timeout(Duration::from_secs(120), wsc.push_button())
            .await
            .map_err(|_| {
                FerricastError::Connection("iwd WPS-PBC timed out after 120s".into())
            })?
            .map_err(|e| FerricastError::Connection(format!("iwd WPS-PBC: {e}")))?;

        tracing::info!("iwd P2P group formed — sink should connect to port {wfd_port} shortly");

        self.dbus = Some(conn);
        self.active_connection = Some(ActiveConnection::Iwd(peer_path));
        self.listener = Some(listener);
        self.alive = true;
        Ok(())
    }

    async fn connect_nm_wifi(&mut self, device: &Device) -> Result<()> {
        // DIRECT- networks are P2P Groups, not regular APs.  Infrastructure
        // WPS-PBC does not work — the sink immediately deactivates before DHCP.
        // Instead: look up the peer's BSSID in NM's P2P peer list and connect
        // via the proper P2P path (same as connect_nm).
        let bssid = device
            .metadata
            .get("bssid")
            .ok_or_else(|| FerricastError::Connection("Missing 'bssid' in nm_wifi metadata".into()))?
            .clone();
        let ssid = device
            .metadata
            .get("ssid")
            .cloned()
            .unwrap_or_default();

        let conn = Connection::system().await.map_err(|e| {
            FerricastError::Connection(format!("D-Bus system bus unavailable: {e}"))
        })?;
        let nm = NetworkManagerProxy::new(&conn).await.map_err(|e| {
            FerricastError::Connection(format!("Cannot reach NetworkManager: {e}"))
        })?;
        let devices = nm.get_devices().await.map_err(|e| {
            FerricastError::Connection(format!("NM GetDevices: {e}"))
        })?;

        // Collect P2P device paths once; we'll re-probe peers each iteration.
        let mut p2p_device_paths: Vec<OwnedObjectPath> = Vec::new();
        for dev_path in &devices {
            let dev = DeviceProxy::new(&conn, dev_path.clone()).await.map_err(|e| {
                FerricastError::Connection(format!("NM Device proxy: {e}"))
            })?;
            if dev.device_type().await.unwrap_or(0) == NM_DEVICE_TYPE_WIFI_P2P {
                p2p_device_paths.push(dev_path.clone());
            }
        }

        // The NM Wi-Fi scan may have seen the AP before P2P find reported the
        // peer.  Kick a fresh P2P find and retry up to ~10 s.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            for dev_path in &p2p_device_paths {
                let p2p = WifiP2pProxy::new(&conn, dev_path.clone()).await.map_err(|e| {
                    FerricastError::Connection(format!("WifiP2P proxy: {e}"))
                })?;
                for peer_path in p2p.peers().await.unwrap_or_default() {
                    let pp = P2pPeerProxy::new(&conn, peer_path.clone()).await.map_err(|e| {
                        FerricastError::Connection(format!("P2pPeer proxy: {e}"))
                    })?;
                    let hw = pp.hw_address().await.unwrap_or_default();
                    if !hw.eq_ignore_ascii_case(&bssid) {
                        continue;
                    }
                    tracing::info!(bssid, %peer_path, ssid, "DIRECT- AP matched to NM P2P peer");
                    let mut p2p_device = device.clone();
                    p2p_device.metadata.insert("path".into(), peer_path.to_string());
                    p2p_device.metadata.insert("device_path".into(), dev_path.to_string());
                    p2p_device.metadata.insert("hw_address".into(), hw);
                    p2p_device.metadata.insert("backend".into(), "nm".into());
                    return self.connect_nm(&p2p_device).await;
                }
                // Trigger a fresh P2P find so wpa_supplicant re-scans.
                let _ = p2p.start_find(HashMap::new()).await;
            }

            if tokio::time::Instant::now() >= deadline {
                break;
            }
            tracing::debug!(bssid, "DIRECT- peer not in P2P list yet — waiting for P2P scan");
            tokio::time::sleep(Duration::from_secs(2)).await;
        }

        Err(FerricastError::Connection(format!(
            "DIRECT- network '{ssid}' (BSSID {bssid}) did not appear as a P2P peer \
             within 10 s. The GO may be on a non-social channel (not 1/6/11); \
             wpa_supplicant P2P find may not reach it."
        )))
    }

    async fn connect_tcp_scan(&mut self, device: &Device) -> Result<()> {
        let addr = SocketAddr::new(device.addr, WFD_DEFAULT_PORT);
        tracing::info!(%addr, "Connecting to Miracast sink (infrastructure LAN mode)");
        let stream = tokio::time::timeout(
            Duration::from_secs(10),
            TcpStream::connect(addr),
        )
        .await
        .map_err(|_| {
            FerricastError::Connection(format!("TCP connect to {addr} timed out after 10s"))
        })?
        .map_err(|e| FerricastError::Connection(format!("TCP connect to {addr}: {e}")))?;
        self.direct_conn = Some(stream);
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
        OwnedValue::try_from(Value::from("wifi-p2p"))
            .map_err(|e| FerricastError::Connection(format!("zvariant: {e}")))?,
    );
    connection_settings.insert(
        "id".into(),
        OwnedValue::try_from(Value::from("Ferricast Miracast"))
            .map_err(|e| FerricastError::Connection(format!("zvariant: {e}")))?,
    );
    connection_settings.insert(
        "autoconnect".into(),
        OwnedValue::try_from(Value::Bool(false))
            .map_err(|e| FerricastError::Connection(format!("zvariant: {e}")))?,
    );

    let mut p2p_settings: HashMap<String, OwnedValue> = HashMap::new();
    p2p_settings.insert(
        "peer".into(),
        OwnedValue::try_from(Value::from(peer_hw_address))
            .map_err(|e| FerricastError::Connection(format!("zvariant: {e}")))?,
    );
    p2p_settings.insert(
        "wfd-ies".into(),
        OwnedValue::try_from(Value::from(wfd_source_ies()))
            .map_err(|e| FerricastError::Connection(format!("zvariant: {e}")))?,
    );

    let mut ipv4_props: HashMap<String, OwnedValue> = HashMap::new();
    ipv4_props.insert(
        "method".into(),
        OwnedValue::try_from(zvariant::Value::Str(zvariant::Str::from("auto")))
            .map_err(|e| FerricastError::Connection(format!("zvariant: {e}")))?,
    );
    ipv4_props.insert(
        "never-default".into(),
        OwnedValue::try_from(zvariant::Value::Bool(true))
            .map_err(|e| FerricastError::Connection(format!("zvariant: {e}")))?,
    );

    let mut ipv6_props: HashMap<String, OwnedValue> = HashMap::new();
    ipv6_props.insert(
        "method".into(),
        OwnedValue::try_from(zvariant::Value::Str(zvariant::Str::from("auto")))
            .map_err(|e| FerricastError::Connection(format!("zvariant: {e}")))?,
    );
    ipv6_props.insert(
        "never-default".into(),
        OwnedValue::try_from(zvariant::Value::Bool(true))
            .map_err(|e| FerricastError::Connection(format!("zvariant: {e}")))?,
    );
    ipv6_props.insert(
        "may-fail".into(),
        OwnedValue::try_from(zvariant::Value::Bool(true))
            .map_err(|e| FerricastError::Connection(format!("zvariant: {e}")))?,
    );

    let mut dict = HashMap::new();
    dict.insert("connection".into(), connection_settings);
    dict.insert("wifi-p2p".into(), p2p_settings);
    dict.insert("ipv4".into(), ipv4_props);
    dict.insert("ipv6".into(), ipv6_props);
    Ok(dict)
}

/// Polls the NM active connection state until ACTIVATED.
async fn wait_for_activation(
    conn: &Connection,
    active_path: &OwnedObjectPath,
) -> Result<()> {
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
                    return Ok(());
                }
                Ok(NM_ACTIVE_CONNECTION_STATE_DEACTIVATED) => {
                    return Err(FerricastError::Connection(
                        "P2P connection deactivated before DHCP assignment".into(),
                    ));
                }
                Ok(NM_ACTIVE_CONNECTION_STATE_DEACTIVATING) => {
                    return Err(FerricastError::Connection(
                        "P2P connection deactivating — GO negotiation or WPS failed".into(),
                    ));
                }
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

// ── WFD source IEs ───────────────────────────────────────────────────────────

/// WFD Device Information subelement (ID=0, length=6) for a Miracast source:
///   bits[1:0]  = 0b00  (WFD Source)
///   bits[5:4]  = 0b01  (Session Available)
///   → device_info = 0x0010
///
/// 0x0090 additionally sets bit 7 (WFD Content Protection / HDCP) which
/// is optional and causes some TVs to reject the connection.
pub(crate) fn wfd_source_ies() -> Vec<u8> {
    vec![
        0x00,       // Subelement ID: WFD Device Information
        0x00, 0x06, // Length: 6 bytes
        0x00, 0x10, // Device Info: Source, session available (0x0010)
        0x1C, 0x44, // RTSP port: 7236 (big-endian)
        0x00, 0xC8, // Max throughput: 200 Mbps
    ]
}

// ── RTSP helpers ─────────────────────────────────────────────────────────────

struct RtspMessage {
    start_line: String,
    headers: HashMap<String, String>,
    body: String,
}

async fn send_rtsp<W>(writer: &mut W, msg: &str) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    tracing::trace!(msg = msg.trim_end(), "RTSP →");
    writer
        .write_all(msg.as_bytes())
        .await
        .map_err(|e| FerricastError::Protocol(format!("RTSP write: {e}")))
}

async fn recv_rtsp<R>(stream: &mut R) -> Result<RtspMessage>
where
    R: AsyncBufRead + AsyncRead + Unpin,
{
    let mut header_lines: Vec<String> = Vec::new();

    loop {
        let mut line = String::new();
        let n = stream
            .read_line(&mut line)
            .await
            .map_err(|e| FerricastError::Protocol(format!("RTSP read: {e}")))?;
        if n == 0 {
            return Err(FerricastError::Protocol("RTSP connection closed".into()));
        }
        let trimmed = line.trim_end_matches(['\r', '\n']).to_string();
        if trimmed.is_empty() {
            break;
        }
        header_lines.push(trimmed);
    }

    let start_line = header_lines.first().cloned().unwrap_or_default();

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

/// Background task: owns the read half of the RTSP TCP connection.
///
/// - 200 OK to keep-alive GET_PARAMETER → discarded.
/// - SET_PARAMETER with `wfd_idr_request` body → sets `keyframe_requested`.
/// - PAUSE / PLAY (M8/M9) → responds 200 OK via the shared writer.
/// - TEARDOWN → sends `RtspEvent::Teardown` and exits.
async fn rtsp_reader_task(
    mut reader: BufReader<OwnedReadHalf>,
    writer: Arc<Mutex<OwnedWriteHalf>>,
    session_id: String,
    tx: mpsc::Sender<RtspEvent>,
    keyframe_requested: Arc<AtomicBool>,
) {
    loop {
        match recv_rtsp(&mut reader).await {
            Ok(msg) => {
                if msg.start_line.starts_with("RTSP/1.0") {
                    // Response to our keep-alive GET_PARAMETER — discard.
                    tracing::trace!(status = msg.start_line, "keep-alive ACK");
                } else if msg.start_line.starts_with("TEARDOWN") {
                    let _ = tx.send(RtspEvent::Teardown).await;
                    break;
                } else {
                    // Any request from the sink requires a 200 OK.
                    let cseq = header_value(&msg.headers, "cseq")
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(0);

                    if msg.start_line.starts_with("SET_PARAMETER")
                        && msg.body.contains("wfd_idr_request")
                    {
                        tracing::debug!("Sink requested IDR keyframe");
                        keyframe_requested.store(true, Ordering::Relaxed);
                    }

                    let response = format!(
                        "RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nSession: {session_id}\r\n\r\n"
                    );
                    let mut w = writer.lock().await;
                    let _ = send_rtsp(&mut *w, &response).await;
                }
            }
            Err(e) => {
                tracing::debug!("RTSP reader closed: {e}");
                break;
            }
        }
    }
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

fn header_value<'m>(headers: &'m HashMap<String, String>, key: &str) -> Option<&'m str> {
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
/// Used in the alt-M3 flow when the sink asks for source capabilities first.
/// Video: H.264 CBP Level 3.2, 1920×1080p@30 (CEA mode 7 = bit 7 = 0x80).
/// Audio: LPCM stereo 44.1/48 kHz.
///
/// `wfd_client_rtp_ports` is intentionally omitted — that field describes the
/// sink's preferred receive port, not a source capability.
fn wfd_capabilities() -> String {
    // profile=01 CBP, level=08 (4.1), CEA: bits5+7 = 720p@30 + 1080p@30,
    // HH: bit6 = 1080p@60 (for handhelds). Level 4.1 supports 1080p@60.
    "wfd_video_formats: 00 00 01 08 000000A0 00000000 00000040 00 0000 0000 00 none none\r\n\
     wfd_audio_codecs: LPCM 00000003 00, AAC 00000001 00\r\n\
     wfd_uibc_capability: none\r\n\
     wfd_standby_resume_capability: none\r\n"
        .to_string()
}

/// Builds the WFD SET_PARAMETER (M4) body selecting parameters to use.
///
/// `sink_primary_rtp` / `sink_secondary_rtp` are the sink's preferred RTP receive
/// ports (advertised by the sink in the M3 GET_PARAMETER response).
fn wfd_set_parameter(
    presentation_url: &str,
    sink_primary_rtp: u16,
    sink_secondary_rtp: u16,
) -> String {
    format!(
        "wfd_video_formats: 00 00 01 08 000000A0 00000000 00000040 00 0000 0000 00 none none\r\n\
         wfd_audio_codecs: AAC 00000001 00\r\n\
         wfd_presentation_URL: {presentation_url} none\r\n\
         wfd_client_rtp_ports: RTP/AVP/UDP;unicast {sink_primary_rtp} {sink_secondary_rtp} mode=play\r\n"
    )
}

/// Parses `wfd_client_rtp_ports` from the sink's M3 GET_PARAMETER response body.
///
/// Expected format: `wfd_client_rtp_ports: RTP/AVP/UDP;unicast 16384 16385 mode=play`
///
/// Defaults (from GNOME Network Displays wfd-params.c):
///   primary  = 16384 if missing or 0
///   secondary = primary + 1 if 0
fn parse_sink_rtp_ports(body: &str) -> (u16, u16) {
    for line in body.lines() {
        if let Some(val) = line.strip_prefix("wfd_client_rtp_ports:") {
            if let Some(rest) = val.trim().strip_prefix("RTP/AVP/UDP;unicast ") {
                let mut parts = rest.split_whitespace();
                let primary: u16 = parts
                    .next()
                    .and_then(|p| p.parse().ok())
                    .unwrap_or(0);
                let secondary: u16 = parts
                    .next()
                    .and_then(|p| p.parse().ok())
                    .unwrap_or(0);
                let primary = if primary == 0 { 16384 } else { primary };
                let secondary = if secondary == 0 { primary + 1 } else { secondary };
                return (primary, secondary);
            }
        }
    }
    (16384, 16385)
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

    pkt[4] = 0x00; // pointer field

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

/// Builds a 188-byte PMT (Program Map Table) TS packet for H.264 video + AAC audio.
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
    // section_length = 9 (fixed header) + 5 (video ES) + 5 (audio ES) + 4 (CRC) = 23 = 0x17
    pmt[2] = 0x17;
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
    // Elementary stream: H.264 video
    pmt[12] = STREAM_TYPE_H264;
    pmt[13] = 0xE0 | (PID_VIDEO >> 8) as u8;
    pmt[14] = PID_VIDEO as u8;
    pmt[15] = 0xF0; // ES_info_length = 0
    pmt[16] = 0x00;
    // Elementary stream: AAC audio (ADTS, ISO 13818-7)
    pmt[17] = STREAM_TYPE_AAC;
    pmt[18] = 0xE0 | (PID_AUDIO >> 8) as u8;
    pmt[19] = PID_AUDIO as u8;
    pmt[20] = 0xF0; // ES_info_length = 0
    pmt[21] = 0x00;
    // CRC32 over table_id through last ES entry (22 bytes)
    let crc = crc32_mpeg(&pkt[5..5 + 22]);
    pkt[27] = (crc >> 24) as u8;
    pkt[28] = (crc >> 16) as u8;
    pkt[29] = (crc >> 8) as u8;
    pkt[30] = crc as u8;

    pkt
}

/// Builds a PES packet wrapping `payload` (H.264 Annex-B NALUs).
fn build_pes(payload: &[u8], (pts_us, dts_us): (u64, u64)) -> Vec<u8> {
    let pts_90k = pts_us * 90_000 / 1_000_000;
    let dts_90k = dts_us * 90_000 / 1_000_000;
    let pts_dts_differ = pts_90k != dts_90k;

    let pts_dts_flags: u8 = if pts_dts_differ { 0xC0 } else { 0x80 };
    let header_data_len: u8 = if pts_dts_differ { 10 } else { 5 };

    let mut pes: Vec<u8> =
        Vec::with_capacity(9 + header_data_len as usize + payload.len());
    pes.extend_from_slice(&[0x00, 0x00, 0x01]);
    pes.push(PES_STREAM_ID_VIDEO);
    pes.push(0x00);
    pes.push(0x00);
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

/// Builds a PES packet wrapping ADTS-framed AAC audio data.
///
/// Audio has no B-frames, so DTS == PTS and only PTS is encoded.
fn build_pes_audio(data: &[u8], pts_us: u64) -> Vec<u8> {
    let pts_90k = pts_us * 90_000 / 1_000_000;

    // Fixed PES header: start_code(3) + stream_id(1) + packet_length(2) +
    //                   flags(2) + header_data_length(1) + PTS(5) = 14 bytes.
    let packet_length = (3 + 1 + 5 + data.len()) as u16; // bytes after the 6-byte fixed prefix
    let mut pes = Vec::with_capacity(14 + data.len());
    pes.extend_from_slice(&[0x00, 0x00, 0x01]);
    pes.push(PES_STREAM_ID_AUDIO);
    pes.push((packet_length >> 8) as u8);
    pes.push(packet_length as u8);
    pes.push(0x80); // marker=10, no scrambling, no priority, no alignment, no copyright, no original
    pes.push(0x80); // PTS_DTS_flags=10 (PTS only), no ESCR, no ES_rate
    pes.push(0x05); // header_data_length = 5 (one PTS field)

    fn encode_pts(ts: u64, prefix: u8) -> [u8; 5] {
        [
            prefix | (((ts >> 30) & 0x07) as u8) << 1 | 0x01,
            ((ts >> 22) & 0xFF) as u8,
            (((ts >> 15) & 0x7F) as u8) << 1 | 0x01,
            ((ts >> 7) & 0xFF) as u8,
            ((ts & 0x7F) as u8) << 1 | 0x01,
        ]
    }

    pes.extend_from_slice(&encode_pts(pts_90k, 0x21));
    pes.extend_from_slice(data);
    pes
}

/// Fragments a PES packet into 188-byte TS packets on `pid`.
///
/// When `pcr_90k` is `Some`, the first TS packet carries a PCR in its
/// adaptation field.  PCR lets the sink's clock recover the source timeline
/// and is required for proper A/V sync.
fn packetize_pes_into_ts(
    pes: &[u8],
    pid: u16,
    cc: &mut u8,
    pcr_90k: Option<u64>,
    out: &mut Vec<u8>,
) {
    let pid_hi = (pid >> 8) as u8; // bits [12:8] of PID (no reserved/flag bits here)
    let pid_lo = pid as u8;

    let mut first = true;
    let mut remaining = pes;

    while !remaining.is_empty() {
        let mut pkt = [0xFFu8; TS_PACKET];
        pkt[0] = 0x47;
        pkt[2] = pid_lo;

        let payload_offset = if first && pcr_90k.is_some() {
            // Adaptation field (8 bytes) + payload.
            // byte 3: adaptation_field_control=0b11 (adapt + payload)
            pkt[1] = 0x40 | pid_hi; // PUSI=1
            pkt[3] = 0x30 | (*cc & 0x0F);
            *cc = cc.wrapping_add(1) & 0x0F;

            let base = pcr_90k.unwrap_or(0) & 0x1_FFFF_FFFF; // 33-bit PCR_base
            pkt[4] = 7; // adaptation_field_length = 7 (flags + 6 PCR bytes)
            pkt[5] = 0x10; // PCR_flag=1, all others 0
            pkt[6] = (base >> 25) as u8;
            pkt[7] = (base >> 17) as u8;
            pkt[8] = (base >> 9) as u8;
            pkt[9] = (base >> 1) as u8;
            // byte 10: PCR_base[0] | reserved(111111) | PCR_ext[8]=0
            pkt[10] = ((base & 1) as u8) << 7 | 0x7E;
            pkt[11] = 0x00; // PCR_ext[7:0] = 0
            12 // payload starts after 4-byte header + 8-byte adaptation field
        } else {
            pkt[1] = if first { 0x40 | pid_hi } else { pid_hi };
            pkt[3] = 0x10 | (*cc & 0x0F); // payload only
            *cc = cc.wrapping_add(1) & 0x0F;
            4
        };

        let payload_space = TS_PACKET - payload_offset;
        let chunk_len = remaining.len().min(payload_space);
        pkt[payload_offset..payload_offset + chunk_len]
            .copy_from_slice(&remaining[..chunk_len]);

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
