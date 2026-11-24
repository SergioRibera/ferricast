//! Miracast sink: advertises as an MS-MICE display, runs the sink-side
//! RTSP M1-M7 state machine, and feeds the received MPEG2-TS/RTP into a
//! [`PushedStream`] so the receiver pipeline decodes + renders it the
//! same way AirPlay mirroring does.

use std::collections::HashMap;
use std::net::SocketAddr;

use bytes::Bytes;
use tokio::net::{TcpListener, UdpSocket};
use tokio::sync::Mutex;
use uuid::Uuid;

use ferricast_core::{
    AdvertiseInfo, Advertiser, AudioCodec, AudioFrame, AudioStreamInfo, Codec, ControlSession,
    DeviceCapabilities, EncodedFrame, FerricastError, MdnsAdvertiser, MediaCommand, MediaInfo,
    MediaPacket, MediaPuller, PlaybackState, PushSender, PushedStream, ReceiverProtocol,
    RemoteSender, Result, VideoStreamInfo,
};

use crate::parameters::{
    AudioFormat, AudioFormats, ClientRtpPorts, ContentProtection, PresentationUrl, TriggerMethod,
    VideoFormat,
};
use crate::rtp::{RTP_PACKET_SIZE, RtpMp2tPacket};
use crate::rtsp::{
    Headers, Message, Method, Request, Response, RtspFramed, WFD_REQUIRE_TAG, options_response,
};
use crate::transport::mice::{MICE_SERVICE_TYPE, WFD_RTSP_PORT};
use crate::ts::{AUDIO_PID, MpegTsDemuxer, VIDEO_PID};

/// Push channel capacity: ~1 s of 1080p30 video at 5 Mbps is a few
/// hundred packets. 1024 keeps the decoder on real hardware from
/// dropping during a GC pause.
const PUSH_CAPACITY: usize = 1024;

pub struct MiracastReceiver {
    pub friendly_name: String,
    pub device_id: String,
    pub bind_port: u16,
    #[cfg(target_os = "linux")]
    pub pairing: crate::transport::linux_p2p::PairingMode,
}

impl MiracastReceiver {
    #[must_use]
    pub fn new(friendly_name: impl Into<String>, device_id: impl Into<String>) -> Self {
        Self {
            friendly_name: friendly_name.into(),
            device_id: device_id.into(),
            bind_port: WFD_RTSP_PORT,
            #[cfg(target_os = "linux")]
            pairing: crate::transport::linux_p2p::PairingMode::PushButton,
        }
    }

    /// Override the RTSP port — defaults to the WFD-standard 7236.
    /// Primarily for tests on loopback.
    #[must_use]
    pub const fn with_bind_port(mut self, port: u16) -> Self {
        self.bind_port = port;
        self
    }

    /// Override the Wi-Fi Direct WPS pairing mode (Linux only).
    /// Default is Push-Button (no PIN). Set to
    /// [`PairingMode::DisplayPin`] when you want the GUI to display a
    /// fixed PIN the sender types.
    #[cfg(target_os = "linux")]
    #[must_use]
    pub fn with_pairing(mut self, mode: crate::transport::linux_p2p::PairingMode) -> Self {
        self.pairing = mode;
        self
    }
}

impl Clone for MiracastReceiver {
    fn clone(&self) -> Self {
        Self {
            friendly_name: self.friendly_name.clone(),
            device_id: self.device_id.clone(),
            bind_port: self.bind_port,
            #[cfg(target_os = "linux")]
            pairing: self.pairing.clone(),
        }
    }
}

impl Default for MiracastReceiver {
    fn default() -> Self {
        Self::new("Ferricast", Uuid::new_v4().to_string())
    }
}

impl ReceiverProtocol for MiracastReceiver {
    const PROTOCOL: &'static str = "miracast";
    const SUPPORTED_CODECS: &'static [Codec] = &[Codec::H264];

    type Advertiser = MiracastReceiverAdvertiser;
    type Control = MiracastReceiverControl;
    type Puller = MiracastPuller;

    fn create_advertiser(&self) -> Self::Advertiser {
        #[cfg(target_os = "linux")]
        {
            MiracastReceiverAdvertiser {
                daemon: None,
                fullname: None,
                device_id: None,
                p2p_task: None,
                pairing: self.pairing.clone(),
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            MiracastReceiverAdvertiser::default()
        }
    }

    fn create_control(&self) -> Result<Self::Control> {
        Ok(MiracastReceiverControl::new(self.bind_port))
    }

    fn create_puller(&self) -> Result<Self::Puller> {
        Ok(MiracastPuller)
    }

    fn advertise_info(&self) -> AdvertiseInfo {
        let mut txt = HashMap::new();
        txt.insert("mice".to_string(), "1".to_string());
        txt.insert("rtsp_port".to_string(), WFD_RTSP_PORT.to_string());
        txt.insert("txtvers".to_string(), "1".to_string());
        txt.insert("fn".to_string(), self.friendly_name.clone());
        AdvertiseInfo {
            friendly_name: self.friendly_name.clone(),
            port: crate::transport::mice::MICE_CONTROL_PORT,
            capabilities: DeviceCapabilities {
                supported_codecs: vec![Codec::H264],
                max_width: Some(1920),
                max_height: Some(1080),
                max_fps: Some(30),
                ..DeviceCapabilities::default()
            },
            device_id: self.device_id.clone(),
            icon: Bytes::new(),
            txt,
        }
    }
}

// ---------------------- Advertiser ----------------------

#[derive(Default)]
pub struct MiracastReceiverAdvertiser {
    daemon: Option<mdns_sd::ServiceDaemon>,
    fullname: Option<String>,
    device_id: Option<String>,
    #[cfg(target_os = "linux")]
    p2p_task: Option<tokio::task::JoinHandle<()>>,
    #[cfg(target_os = "linux")]
    pairing: crate::transport::linux_p2p::PairingMode,
}

impl Advertiser for MiracastReceiverAdvertiser {
    const PROTOCOL: &'static str = "miracast";

    async fn start(&mut self, info: AdvertiseInfo) -> Result<()> {
        let daemon = mdns_sd::ServiceDaemon::new()
            .map_err(|e| FerricastError::Receiver(format!("mdns-sd: {e}")))?;
        let host = local_hostname();
        // Carry the stable per-device id in the TXT so the discovery
        // loop can short-circuit when it encounters our own record.
        let mut txt = info.txt.clone();
        txt.insert("id".to_string(), info.device_id.clone());
        let service = mdns_sd::ServiceInfo::new(
            MICE_SERVICE_TYPE,
            &info.friendly_name,
            &format!("{host}.local."),
            "",
            info.port,
            txt,
        )
        .map_err(|e| FerricastError::Receiver(format!("mdns service: {e}")))?
        .enable_addr_auto();
        let fullname = service.get_fullname().to_string();
        daemon
            .register(service)
            .map_err(|e| FerricastError::Receiver(format!("mdns register: {e}")))?;
        crate::self_filter::register(&info.device_id);
        self.daemon = Some(daemon);
        self.fullname = Some(fullname);
        self.device_id = Some(info.device_id.clone());

        #[cfg(target_os = "linux")]
        {
            match start_linux_p2p_sink(info.friendly_name.clone(), self.pairing.clone()).await {
                Ok(handle) => self.p2p_task = Some(handle),
                Err(e) => tracing::warn!(
                    error = %e,
                    "miracast sink: Wi-Fi Direct advertise disabled (will still accept MICE)"
                ),
            }
        }
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        if let (Some(daemon), Some(fullname)) = (self.daemon.take(), self.fullname.take()) {
            let _ = daemon.unregister(&fullname);
        }
        if let Some(id) = self.device_id.take() {
            crate::self_filter::unregister(&id);
        }
        #[cfg(target_os = "linux")]
        if let Some(task) = self.p2p_task.take() {
            task.abort();
        }
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.daemon.is_some()
    }
}

#[cfg(target_os = "linux")]
async fn start_linux_p2p_sink(
    device_name: String,
    pairing: crate::transport::linux_p2p::PairingMode,
) -> Result<tokio::task::JoinHandle<()>> {
    use crate::ie::{DeviceInformation, WfdIe};
    use crate::transport::linux_p2p::LinuxP2pClient;

    let wfd_ie = WfdIe::new(DeviceInformation::primary_sink());
    let client = LinuxP2pClient::connect(&wfd_ie)
        .await
        .map_err(|e| FerricastError::Receiver(format!("wpa_supplicant: {e}")))?;
    let mut groups = client
        .listen_as_sink(&device_name, &pairing)
        .await
        .map_err(|e| FerricastError::Receiver(format!("Listen: {e}")))?;

    let handle = tokio::spawn(async move {
        while let Some(group) = groups.recv().await {
            tracing::info!(
                iface = %group.interface_path,
                role = ?group.role,
                "miracast sink: Wi-Fi Direct group formed"
            );
            match client.resolve_group_ifname(&group.interface_path).await {
                Ok(kernel_ifname) => {
                    tracing::info!(iface = %kernel_ifname, "miracast sink: running DHCP on group iface");
                    match crate::transport::dhcp::acquire_on(&kernel_ifname).await {
                        Ok(lease) => tracing::info!(
                            addr = %lease.addr,
                            mask = %lease.mask,
                            router = lease.router.map(|r| r.to_string()).unwrap_or_else(|| "-".into()),
                            "miracast sink: DHCP lease installed; RTSP on 7236 now reachable"
                        ),
                        Err(e) => tracing::error!(error = %e, "miracast sink: DHCP failed — phone cannot reach RTSP"),
                    }
                }
                Err(e) => tracing::warn!(error = %e, "miracast sink: cannot resolve group ifname"),
            }
        }
        drop(client); // keep WFDIEs live until stream ends
    });
    Ok(handle)
}

impl MdnsAdvertiser for MiracastReceiverAdvertiser {
    const SERVICE_TYPE: &'static str = MICE_SERVICE_TYPE;
}

fn local_hostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .unwrap_or_else(|| "ferricast".to_string())
}

// ---------------------- Control session ----------------------

/// Sink-side RTSP state machine + RTP pump. One instance per listener;
/// `accept` drives the full M1-M7 handshake, binds the UDP socket,
/// spawns the pump that fans RTP into a [`PushedStream`], and returns
/// the [`RemoteSender`]. `next_command` then hands the manager a
/// single [`MediaCommand::Push`] with that stream.
pub struct MiracastReceiverControl {
    bind_port: u16,
    listener: Option<TcpListener>,
    /// The `PushedStream` built once PLAY succeeds — `next_command`
    /// drains it exactly once.
    pending_stream: Mutex<Option<PushedStream>>,
    /// RTP pump handle; aborted in `close` to tear down the receiver.
    pump: Mutex<Option<tokio::task::JoinHandle<()>>>,
    alive: std::sync::atomic::AtomicBool,
}

impl MiracastReceiverControl {
    #[must_use]
    pub fn new(bind_port: u16) -> Self {
        Self {
            bind_port,
            listener: None,
            pending_stream: Mutex::new(None),
            pump: Mutex::new(None),
            alive: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Address the sink's RTSP TCP listener is bound to. Only valid
    /// after the first `accept` call (which lazily binds).
    pub async fn local_addr(&mut self) -> Result<SocketAddr> {
        self.ensure_listener().await?;
        self.listener
            .as_ref()
            .expect("ensured")
            .local_addr()
            .map_err(FerricastError::Io)
    }

    async fn ensure_listener(&mut self) -> Result<()> {
        if self.listener.is_some() {
            return Ok(());
        }
        let bind = format!("0.0.0.0:{}", self.bind_port);
        let listener = TcpListener::bind(&bind)
            .await
            .map_err(|e| FerricastError::Receiver(format!("bind {bind}: {e}")))?;
        tracing::info!(addr = %listener.local_addr().ok()
            .map(|a| a.to_string()).unwrap_or_default(),
            "miracast sink listening");
        self.listener = Some(listener);
        Ok(())
    }
}

impl ControlSession for MiracastReceiverControl {
    async fn accept(&mut self) -> Result<RemoteSender> {
        self.ensure_listener().await?;
        let listener = self
            .listener
            .as_ref()
            .ok_or_else(|| FerricastError::Receiver("no listener bound".into()))?;
        let (tcp, peer_addr) = listener.accept().await.map_err(FerricastError::Io)?;
        tcp.set_nodelay(true).ok();
        tracing::info!(peer = %peer_addr, "miracast sink: connection");

        let framed = RtspFramed::new(tcp);
        let (tx, stream, pump, remote) = drive_sink_m1_m7(framed, peer_addr).await?;

        *self.pending_stream.lock().await = Some(stream);
        *self.pump.lock().await = Some(pump);
        self.alive.store(true, std::sync::atomic::Ordering::Release);
        // The RTP → push bridge is now live; drop the producer handle
        // here so the Eos propagates once the pump task closes it.
        drop(tx);
        Ok(remote)
    }

    async fn next_command(&mut self) -> Result<MediaCommand> {
        if let Some(stream) = self.pending_stream.lock().await.take() {
            return Ok(MediaCommand::Push(stream));
        }
        // Block until the pump ends; after that the session is done.
        let ended = {
            let pump = self.pump.lock().await;
            pump.is_none()
        };
        if ended {
            return Err(FerricastError::RemoteStopped("miracast teardown".into()));
        }
        // Park the task — the manager will drop us once the push
        // stream it was handed emits Eos. A plain sleep is fine
        // because we don't accept mid-session commands from the
        // source (Miracast's sender→sink RTSP after PLAY carries at
        // most keepalives + wfd_idr_request, handled by the pump).
        std::future::pending::<()>().await;
        Err(FerricastError::RemoteStopped(
            "miracast session ended".into(),
        ))
    }

    async fn report_state(&mut self, _state: PlaybackState) -> Result<()> {
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        if let Some(handle) = self.pump.lock().await.take() {
            handle.abort();
        }
        self.pending_stream.lock().await.take();
        self.alive
            .store(false, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    fn is_alive(&self) -> bool {
        self.alive.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// Sink-side M1-M7. Returns the RTP push sender (kept open by the
/// pump task), the built `PushedStream`, the pump's `JoinHandle`, and
/// the `RemoteSender` summary.
async fn drive_sink_m1_m7(
    mut framed: RtspFramed,
    peer_addr: SocketAddr,
) -> Result<(
    PushSender,
    PushedStream,
    tokio::task::JoinHandle<()>,
    RemoteSender,
)> {
    // --- M1: Source → Sink OPTIONS, Sink responds.
    let m1 = read_request(&mut framed).await?;
    expect_method(&m1, Method::Options)?;
    let cseq = m1.headers.cseq().unwrap_or(1);
    framed
        .send_response(
            options_response(&["org.wfa.wfd1.0", "GET_PARAMETER", "SET_PARAMETER"])
                .with_header("cseq", cseq.to_string()),
        )
        .await
        .map_err(rtsp_err)?;

    // --- M2: Sink → Source OPTIONS (reverse), wait 200.
    let cseq2 = framed
        .send_request(
            Request::new(Method::Options, "*")
                .with_header("require", WFD_REQUIRE_TAG)
                .with_header("user-agent", "FerriCast-Sink/0.1"),
        )
        .await
        .map_err(rtsp_err)?;
    expect_response(&mut framed, cseq2, 200, "M2").await?;

    // --- M3: Source → Sink GET_PARAMETER probe. Respond with caps.
    let m3 = read_request(&mut framed).await?;
    expect_method(&m3, Method::GetParameter)?;
    let video = VideoFormat::baseline_720p30();
    let audio = AudioFormats(vec![AudioFormat::aac_mandatory()]);
    let rtp_ports = ClientRtpPorts::primary(19000);
    let body = format!(
        "wfd_audio_codecs: {audio}\r\n\
         wfd_video_formats: {video}\r\n\
         wfd_3d_video_formats: none\r\n\
         wfd_content_protection: {cp}\r\n\
         wfd_display_edid: none\r\n\
         wfd_coupled_sink: none\r\n\
         wfd_client_rtp_ports: {rtp_ports}\r\n\
         wfd_uibc_capability: none\r\n\
         wfd_standby_resume_capability: none\r\n\
         wfd_connector_type: 05\r\n\
         wfd_I2C: none\r\n",
        cp = ContentProtection::None,
    );
    framed
        .send_response(
            Response::ok()
                .with_header("cseq", m3.headers.cseq().unwrap_or(2).to_string())
                .with_body_params(body),
        )
        .await
        .map_err(rtsp_err)?;

    // --- M4: Source → Sink SET_PARAMETER with negotiated subset.
    let m4 = read_request(&mut framed).await?;
    expect_method(&m4, Method::SetParameter)?;
    let m4_body = std::str::from_utf8(&m4.body)
        .map_err(|_| FerricastError::Rtsp("non-UTF8 M4 body".into()))?;
    let m4_params = crate::parameters::decode_body(m4_body);
    let presentation_url: PresentationUrl = m4_params
        .iter()
        .find(|p| p.name.eq_ignore_ascii_case("wfd_presentation_URL"))
        .and_then(|p| p.value.as_deref())
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| FerricastError::Rtsp("M4 missing wfd_presentation_URL".into()))?;
    let client_ports: ClientRtpPorts = m4_params
        .iter()
        .find(|p| p.name.eq_ignore_ascii_case("wfd_client_rtp_ports"))
        .and_then(|p| p.value.as_deref())
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| ClientRtpPorts::primary(19000));
    framed
        .send_response(
            Response::ok().with_header("cseq", m4.headers.cseq().unwrap_or(3).to_string()),
        )
        .await
        .map_err(rtsp_err)?;

    // --- M5: Source → Sink SET_PARAMETER wfd_trigger_method:SETUP.
    let m5 = read_request(&mut framed).await?;
    expect_method(&m5, Method::SetParameter)?;
    let m5_body = std::str::from_utf8(&m5.body)
        .map_err(|_| FerricastError::Rtsp("non-UTF8 M5 body".into()))?;
    let trigger: TriggerMethod = crate::parameters::decode_body(m5_body)
        .into_iter()
        .find(|p| p.name.eq_ignore_ascii_case("wfd_trigger_method"))
        .and_then(|p| p.value)
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| FerricastError::Rtsp("M5 missing wfd_trigger_method".into()))?;
    if !matches!(trigger, TriggerMethod::Setup) {
        return Err(FerricastError::Rtsp(format!(
            "unexpected M5 trigger {trigger}"
        )));
    }
    framed
        .send_response(
            Response::ok().with_header("cseq", m5.headers.cseq().unwrap_or(4).to_string()),
        )
        .await
        .map_err(rtsp_err)?;

    // --- Bind UDP sink socket (prefer the client_port we advertised in
    // M3/M4; fall back to an ephemeral one on EADDRINUSE so parallel
    // tests on loopback don't collide).
    let rtp_socket = match UdpSocket::bind(format!("0.0.0.0:{}", client_ports.rtp_port0)).await {
        Ok(s) => s,
        Err(_) => UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(FerricastError::Io)?,
    };
    let rtp_local = rtp_socket.local_addr().map_err(FerricastError::Io)?;

    // --- M6: Sink → Source SETUP. We expect 200 with server_port + session token.
    let cseq6 = framed
        .send_request(
            Request::new(Method::Setup, presentation_url.url0.clone()).with_header(
                "transport",
                format!("RTP/AVP/UDP;unicast;client_port={}", rtp_local.port()),
            ),
        )
        .await
        .map_err(rtsp_err)?;
    let m6_resp = expect_response(&mut framed, cseq6, 200, "M6").await?;
    let session_token = m6_resp
        .headers
        .session()
        .map(str::to_string)
        .ok_or_else(|| FerricastError::Rtsp("M6 response missing Session".into()))?;

    // --- M7: Sink → Source PLAY. Expect 200.
    let cseq7 = framed
        .send_request(
            Request::new(Method::Play, presentation_url.url0.clone())
                .with_header("session", session_token.clone()),
        )
        .await
        .map_err(rtsp_err)?;
    expect_response(&mut framed, cseq7, 200, "M7").await?;

    // --- Push bridge + RTP pump.
    let info = MediaInfo {
        video: Some(VideoStreamInfo {
            hdr: None,
            codec: Codec::H264,
            width: 1280,
            height: 720,
            fps: 30,
            colour: None,
        }),
        audio: Some(AudioStreamInfo {
            codec: AudioCodec::Aac,
            sample_rate: 48_000,
            channels: 2,
            source: None,
        }),
        duration_us: None,
        is_live: true,
    };
    let (tx, stream) = PushedStream::channel(info, PUSH_CAPACITY);
    let tx_pump = tx.clone();
    let pump = tokio::spawn(async move {
        rtp_pump(rtp_socket, tx_pump).await;
    });
    // Also spawn a lightweight RTSP watcher for keepalives + teardown.
    let tx_watch = tx.clone();
    tokio::spawn(async move {
        rtsp_watch(framed, tx_watch).await;
    });

    Ok((
        tx,
        stream,
        pump,
        RemoteSender {
            id: Uuid::new_v4(),
            addr: peer_addr.ip(),
            name: None,
        },
    ))
}

async fn read_request(framed: &mut RtspFramed) -> Result<Request> {
    match framed.read_message().await.map_err(rtsp_err)? {
        Message::Request(req) => Ok(req),
        Message::Response(_) => Err(FerricastError::Rtsp("expected request".into())),
    }
}

async fn expect_response(
    framed: &mut RtspFramed,
    _expected_cseq: u32,
    status: u16,
    label: &str,
) -> Result<Response> {
    match framed.read_message().await.map_err(rtsp_err)? {
        Message::Response(resp) if resp.status == status => Ok(resp),
        Message::Response(resp) => Err(FerricastError::Rtsp(format!(
            "{label}: unexpected {} {}",
            resp.status, resp.reason
        ))),
        Message::Request(_) => Err(FerricastError::Rtsp(format!("{label}: expected response"))),
    }
}

fn expect_method(req: &Request, method: Method) -> Result<()> {
    if req.method == method {
        Ok(())
    } else {
        Err(FerricastError::Rtsp(format!(
            "expected {}, got {}",
            method.as_str(),
            req.raw_method
        )))
    }
}

fn rtsp_err(err: crate::rtsp::RtspError) -> FerricastError {
    FerricastError::Rtsp(err.to_string())
}

/// RTP pump: drains the UDP socket, demuxes MPEG2-TS, forwards
/// EncodedFrame / AudioFrame into the push stream. Exits on socket
/// error, task abort, or when every receiver dropped.
async fn rtp_pump(socket: UdpSocket, tx: PushSender) {
    let mut demux = MpegTsDemuxer::new();
    let mut buf = vec![0u8; RTP_PACKET_SIZE.max(2048)];
    loop {
        let (n, _src) = match socket.recv_from(&mut buf).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "miracast rtp pump: recv");
                break;
            }
        };
        let rtp = match RtpMp2tPacket::parse(&buf[..n]) {
            Ok(p) => p,
            Err(e) => {
                tracing::debug!(error = %e, "miracast rtp pump: bad packet");
                continue;
            }
        };
        for ts_packet in rtp.ts_packets() {
            for unit in demux.push_packet(ts_packet) {
                let packet = match pes_to_media_packet(&unit) {
                    Some(p) => p,
                    None => continue,
                };
                if tx.try_push(packet).is_err() {
                    tracing::debug!("miracast rtp pump: consumer dropped");
                    return;
                }
            }
        }
    }
    let _ = tx.try_push(MediaPacket::Eos);
}

fn pes_to_media_packet(unit: &crate::ts::PesUnit) -> Option<MediaPacket> {
    // 90 kHz PTS → µs at the receiver side.
    let timestamp_us = unit.pts_90khz.map_or(0, |pts| pts * 1_000_000 / 90_000);
    match unit.pid {
        VIDEO_PID => {
            let is_keyframe = detect_h264_idr(&unit.payload);
            Some(MediaPacket::Video(EncodedFrame {
                codec: Codec::H264,
                data: unit.payload.clone(),
                timestamp_us,
                is_keyframe,
                duration_us: None,
                pts_dts: (timestamp_us, timestamp_us),
            }))
        }
        AUDIO_PID => Some(MediaPacket::Audio(AudioFrame {
            codec: AudioCodec::Aac,
            data: unit.payload.clone(),
            timestamp_us,
            sample_rate: 48_000,
            channels: 2,
        })),
        _ => None,
    }
}

/// Scan an H.264 Annex-B access unit for an IDR (NAL type 5).
fn detect_h264_idr(au: &[u8]) -> bool {
    let mut i = 0;
    while i + 4 < au.len() {
        let has_long = au[i] == 0 && au[i + 1] == 0 && au[i + 2] == 0 && au[i + 3] == 1;
        let has_short = au[i] == 0 && au[i + 1] == 0 && au[i + 2] == 1;
        if has_long || has_short {
            let hdr_pos = if has_long { i + 4 } else { i + 3 };
            if hdr_pos < au.len() && (au[hdr_pos] & 0x1F) == 5 {
                return true;
            }
            i = hdr_pos + 1;
        } else {
            i += 1;
        }
    }
    false
}

/// Watch for post-PLAY RTSP traffic: respond to keepalives
/// (`GET_PARAMETER` empty body), ignore `wfd_idr_request` (we're a
/// sink, we don't encode), handle `TEARDOWN` by closing the push.
async fn rtsp_watch(mut framed: RtspFramed, tx: PushSender) {
    loop {
        let msg = match framed.read_message().await {
            Ok(m) => m,
            Err(e) => {
                tracing::debug!(error = %e, "miracast rtsp watch: read ended");
                let _ = tx.try_push(MediaPacket::Eos);
                return;
            }
        };
        match msg {
            Message::Request(req) => {
                let cseq = req.headers.cseq().unwrap_or(0).to_string();
                let resp = match req.method {
                    Method::Teardown => {
                        let _ = framed
                            .send_response(Response::ok().with_header("cseq", cseq))
                            .await;
                        let _ = tx.try_push(MediaPacket::Eos);
                        return;
                    }
                    Method::GetParameter | Method::SetParameter | Method::Options => {
                        Response::ok().with_header("cseq", cseq)
                    }
                    _ => Response::not_implemented().with_header("cseq", cseq),
                };
                if framed.send_response(resp).await.is_err() {
                    return;
                }
            }
            Message::Response(_) => {
                // Stray response — ignore.
            }
        }
    }
}

// ---------------------- Puller stub ----------------------

/// Miracast doesn't pull: the sink gets media pushed as a
/// [`PushedStream`] embedded in a [`MediaCommand::Push`]. The trait
/// still needs a type — this one errors on `open` so the manager never
/// accidentally kicks off a pull on it.
#[derive(Default, Clone)]
pub struct MiracastPuller;

impl MediaPuller for MiracastPuller {
    async fn open(&mut self, _spec: ferricast_core::PullSpec) -> Result<MediaInfo> {
        Err(FerricastError::Pull(
            "miracast is push-only: use the PushedStream from MediaCommand::Push".into(),
        ))
    }

    async fn next(&mut self) -> Result<MediaPacket> {
        Err(FerricastError::Pull("miracast puller is not used".into()))
    }

    async fn seek(&mut self, _position_us: u64) -> Result<()> {
        Err(FerricastError::Pull("miracast is not seekable".into()))
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

#[allow(dead_code, reason = "convenient parse helper exported for tests")]
pub(crate) fn parse_rtsp_headers_for_tests(input: &str) -> Headers {
    let mut h = Headers::new();
    for line in input.lines() {
        if let Some((k, v)) = line.split_once(':') {
            h.insert(k.trim(), v.trim());
        }
    }
    h
}
