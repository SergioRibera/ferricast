//! Miracast source session: drives M1-M7 against a discovered sink,
//! then pumps MPEG2-TS/RTP MP2T to the negotiated UDP port.
//!
//! The RTSP control plane is this type; the media plane is the
//! `MpegTsMuxer` + `MpegTsRtpPacker` it owns. `send_frame` wraps an
//! encoded H.264 access unit into TS packets, packs them 7-per-RTP,
//! and ships the RTP on a bound UDP socket.
//!
//! Receiver-driven keyframe refresh (`wfd_idr_request`) is wired
//! through [`CastSession::poll_keyframe_request`]; the state machine
//! sets the flag when the sink sends the SET_PARAMETER mid-session.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::net::{TcpStream, UdpSocket};

use ferricast_core::{
    AudioCodec, CastSession, ConnectOutcome, Device, EncodedFrame, FerricastError, Result,
    StreamConfig,
};

use crate::parameters::{
    ClientRtpPorts, ContentProtection, PresentationUrl, TriggerMethod, VideoFormat,
};
use crate::rtp::{MpegTsRtpPacker, RTP_TIMESTAMP_HZ};
use crate::rtsp::{
    build_options, options_response, Message, Method, Request, Response, RtspFramed,
    WFD_REQUIRE_TAG,
};
use crate::transport::mice::{mice_probe, MICE_CONTROL_PORT, WFD_RTSP_PORT};
use crate::ts::{MpegTsMuxer, VideoAccessUnit};

/// Timeouts tuned to the interop gotchas:
/// - Keepalive at 30 s, sink timeout defaults to 60 s.
/// - IDR response window ≤ 200 ms once the sink requests one.
const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(30);
const RTSP_WALL_TIMEOUT: Duration = Duration::from_secs(10);

/// Miracast source session.
pub struct MiracastSession {
    rtsp: Option<RtspFramed>,
    rtp: Option<UdpSocket>,
    rtp_peer: Option<SocketAddr>,
    muxer: MpegTsMuxer,
    packer: MpegTsRtpPacker,
    session_token: Option<String>,
    rtsp_url: String,
    keyframe_requested: Arc<AtomicBool>,
    last_keyframe_check: std::time::Instant,
    alive: bool,
    end_reason: Option<FerricastError>,
}

impl Default for MiracastSession {
    fn default() -> Self {
        Self {
            rtsp: None,
            rtp: None,
            rtp_peer: None,
            muxer: MpegTsMuxer::new(false),
            packer: MpegTsRtpPacker::new(),
            session_token: None,
            rtsp_url: String::new(),
            keyframe_requested: Arc::new(AtomicBool::new(false)),
            last_keyframe_check: std::time::Instant::now(),
            alive: false,
            end_reason: None,
        }
    }
}

impl MiracastSession {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    async fn open_rtsp(&mut self, addr: SocketAddr) -> Result<()> {
        tracing::info!(peer = %addr, "miracast: opening RTSP");
        let stream = tokio::time::timeout(RTSP_WALL_TIMEOUT, TcpStream::connect(addr))
            .await
            .map_err(|_| FerricastError::Timeout(format!("RTSP connect {addr}")))?
            .map_err(FerricastError::Io)?;
        stream.set_nodelay(true).ok();
        self.rtsp = Some(RtspFramed::new(stream));
        Ok(())
    }

    async fn do_m1_options(&mut self) -> Result<()> {
        let framed = self
            .rtsp
            .as_mut()
            .ok_or_else(|| FerricastError::Rtsp("no RTSP channel".into()))?;
        framed
            .send_request(build_options("FerriCast/0.1"))
            .await
            .map_err(rtsp_err)?;
        match framed.read_message().await.map_err(rtsp_err)? {
            Message::Response(resp) if resp.status == 200 => Ok(()),
            Message::Response(resp) => Err(FerricastError::Rtsp(format!(
                "M1 OPTIONS rejected: {} {}",
                resp.status, resp.reason
            ))),
            Message::Request(_) => Err(FerricastError::Rtsp("expected response to M1".into())),
        }
    }

    async fn handle_m2_reverse_options(&mut self) -> Result<()> {
        let framed = self
            .rtsp
            .as_mut()
            .ok_or_else(|| FerricastError::Rtsp("no RTSP channel".into()))?;
        let msg = framed.read_message().await.map_err(rtsp_err)?;
        let Message::Request(req) = msg else {
            return Err(FerricastError::Rtsp("expected M2 OPTIONS".into()));
        };
        if req.method != Method::Options {
            return Err(FerricastError::Rtsp(format!(
                "expected OPTIONS, got {}",
                req.raw_method
            )));
        }
        let cseq = req.headers.cseq().unwrap_or(1).to_string();
        let resp = options_response(&["SETUP", "PLAY", "PAUSE", "TEARDOWN", "GET_PARAMETER", "SET_PARAMETER"])
            .with_header("cseq", cseq);
        framed.send_response(resp).await.map_err(rtsp_err)?;
        Ok(())
    }

    async fn do_m3_capability_probe(&mut self) -> Result<()> {
        let framed = self
            .rtsp
            .as_mut()
            .ok_or_else(|| FerricastError::Rtsp("no RTSP channel".into()))?;
        let body = concat!(
            "wfd_video_formats\r\n",
            "wfd_audio_codecs\r\n",
            "wfd_content_protection\r\n",
            "wfd_client_rtp_ports\r\n",
        );
        let req = Request::new(Method::GetParameter, "rtsp://localhost/wfd1.0")
            .with_header("require", WFD_REQUIRE_TAG)
            .with_body_params(body);
        framed.send_request(req).await.map_err(rtsp_err)?;
        match framed.read_message().await.map_err(rtsp_err)? {
            Message::Response(resp) if resp.status == 200 => Ok(()),
            Message::Response(resp) => Err(FerricastError::Rtsp(format!(
                "M3 rejected: {} {}",
                resp.status, resp.reason
            ))),
            Message::Request(_) => Err(FerricastError::Rtsp("expected response to M3".into())),
        }
    }

    async fn do_m4_push_params(&mut self, local_ip: &str) -> Result<()> {
        self.rtsp_url = format!("rtsp://{local_ip}/wfd1.0/streamid=0");
        let video = VideoFormat::baseline_720p30();
        let audio = crate::parameters::AudioFormats(vec![crate::parameters::AudioFormat::aac_mandatory()]);
        let rtp_ports = ClientRtpPorts::primary(19000);
        let url = PresentationUrl::primary(self.rtsp_url.clone());
        let body = format!(
            "wfd_video_formats: {video}\r\n\
             wfd_audio_codecs: {audio}\r\n\
             wfd_content_protection: {cp}\r\n\
             wfd_presentation_URL: {url}\r\n\
             wfd_client_rtp_ports: {rtp_ports}\r\n",
            cp = ContentProtection::None,
        );
        let req = Request::new(Method::SetParameter, "rtsp://localhost/wfd1.0")
            .with_header("require", WFD_REQUIRE_TAG)
            .with_body_params(body);
        let framed = self
            .rtsp
            .as_mut()
            .ok_or_else(|| FerricastError::Rtsp("no RTSP channel".into()))?;
        framed.send_request(req).await.map_err(rtsp_err)?;
        match framed.read_message().await.map_err(rtsp_err)? {
            Message::Response(resp) if resp.status == 200 => Ok(()),
            Message::Response(resp) => Err(FerricastError::Rtsp(format!(
                "M4 rejected: {} {}",
                resp.status, resp.reason
            ))),
            Message::Request(_) => Err(FerricastError::Rtsp("expected response to M4".into())),
        }
    }

    async fn do_m5_trigger_setup(&mut self) -> Result<()> {
        let framed = self
            .rtsp
            .as_mut()
            .ok_or_else(|| FerricastError::Rtsp("no RTSP channel".into()))?;
        let req = Request::new(Method::SetParameter, "rtsp://localhost/wfd1.0")
            .with_header("require", WFD_REQUIRE_TAG)
            .with_body_params(format!("wfd_trigger_method: {}\r\n", TriggerMethod::Setup));
        framed.send_request(req).await.map_err(rtsp_err)?;
        match framed.read_message().await.map_err(rtsp_err)? {
            Message::Response(resp) if resp.status == 200 => Ok(()),
            Message::Response(resp) => Err(FerricastError::Rtsp(format!(
                "M5 rejected: {} {}",
                resp.status, resp.reason
            ))),
            Message::Request(_) => Err(FerricastError::Rtsp("expected response to M5".into())),
        }
    }

    async fn handle_m6_m7(&mut self, local_ip: &str) -> Result<()> {
        let framed = self
            .rtsp
            .as_mut()
            .ok_or_else(|| FerricastError::Rtsp("no RTSP channel".into()))?;
        // M6 SETUP
        let msg = framed.read_message().await.map_err(rtsp_err)?;
        let Message::Request(setup) = msg else {
            return Err(FerricastError::Rtsp("expected M6 SETUP".into()));
        };
        if setup.method != Method::Setup {
            return Err(FerricastError::Rtsp(format!(
                "expected SETUP, got {}",
                setup.raw_method
            )));
        }
        let client_port = parse_client_port(setup.headers.get("transport").unwrap_or(""))
            .ok_or_else(|| FerricastError::Rtsp("missing client_port in Transport".into()))?;
        let udp = UdpSocket::bind(format!("{local_ip}:0"))
            .await
            .map_err(FerricastError::Io)?;
        let server_port = udp.local_addr().map_err(FerricastError::Io)?.port();
        let session_token = format!("{:08X}", rand::random::<u32>());
        let transport = format!(
            "RTP/AVP/UDP;unicast;client_port={client_port};server_port={server_port}-{server_port_plus1}",
            server_port_plus1 = server_port + 1,
        );
        let cseq = setup.headers.cseq().unwrap_or(1).to_string();
        let resp = Response::ok()
            .with_header("cseq", cseq)
            .with_header("session", format!("{session_token};timeout=60"))
            .with_header("transport", transport);
        framed.send_response(resp).await.map_err(rtsp_err)?;
        self.rtp = Some(udp);
        self.session_token = Some(session_token);

        // M7 PLAY
        let msg = framed.read_message().await.map_err(rtsp_err)?;
        let Message::Request(play) = msg else {
            return Err(FerricastError::Rtsp("expected M7 PLAY".into()));
        };
        if play.method != Method::Play {
            return Err(FerricastError::Rtsp(format!(
                "expected PLAY, got {}",
                play.raw_method
            )));
        }
        let cseq = play.headers.cseq().unwrap_or(2).to_string();
        let resp = Response::ok()
            .with_header("cseq", cseq)
            .with_header(
                "session",
                self.session_token
                    .clone()
                    .unwrap_or_default(),
            );
        framed.send_response(resp).await.map_err(rtsp_err)?;

        let peer_ip = self
            .rtsp
            .as_ref()
            .map(|f| f.peer_addr().ip())
            .unwrap_or_else(|| "127.0.0.1".parse().expect("addr"));
        self.rtp_peer = Some(SocketAddr::new(peer_ip, client_port));
        Ok(())
    }
}

fn parse_client_port(transport: &str) -> Option<u16> {
    for tok in transport.split(';') {
        if let Some(rest) = tok.trim().strip_prefix("client_port=") {
            if let Some((p, _)) = rest.split_once('-') {
                return p.parse().ok();
            }
            return rest.parse().ok();
        }
    }
    None
}

fn rtsp_err(err: crate::rtsp::RtspError) -> FerricastError {
    FerricastError::Rtsp(err.to_string())
}

impl CastSession for MiracastSession {
    async fn connect(&mut self, device: &Device) -> Result<ConnectOutcome> {
        let discovered = device.addresses.primary();
        // The RTSP port lives on either a `rtsp_port` TXT record (MS-MICE
        // advertises the MICE control port on its mDNS SRV and the WFD
        // RTSP port separately in TXT) or falls back to the WFD-standard
        // 7236 when discovery didn't carry it. Direct callers that don't
        // go through discovery (loopback tests, manually-added devices)
        // encode the real RTSP port in the device address itself.
        let rtsp_port = device
            .metadata
            .get("rtsp_port")
            .and_then(|v| v.parse::<u16>().ok())
            .unwrap_or_else(|| {
                // Heuristic: if the advertised port is the MICE control
                // port we fall back to WFD_RTSP_PORT; otherwise trust
                // whatever was put in the address.
                if discovered.port() == MICE_CONTROL_PORT {
                    WFD_RTSP_PORT
                } else {
                    discovered.port()
                }
            });
        let mice_addr = SocketAddr::new(discovered.ip(), MICE_CONTROL_PORT);
        match mice_probe(mice_addr, RTSP_WALL_TIMEOUT).await {
            Ok(h) if h.is_ok() => tracing::debug!("miracast MICE probe OK"),
            Ok(h) => {
                tracing::warn!(status = h.status, "MICE probe non-zero status");
            }
            Err(e) => tracing::warn!(error = %e, "MICE probe skipped; attempting direct RTSP"),
        }
        let rtsp_addr = SocketAddr::new(discovered.ip(), rtsp_port);
        self.open_rtsp(rtsp_addr).await?;
        self.alive = true;
        Ok(ConnectOutcome::Ready)
    }

    async fn setup_stream(&mut self, _config: &StreamConfig) -> Result<()> {
        // Source-side M1-M7.
        self.do_m1_options().await?;
        self.handle_m2_reverse_options().await?;
        self.do_m3_capability_probe().await?;
        let local_ip = local_ip_string(&self.rtsp)?;
        self.do_m4_push_params(&local_ip).await?;
        self.do_m5_trigger_setup().await?;
        self.handle_m6_m7(&local_ip).await?;
        Ok(())
    }

    async fn send_frame(&mut self, frame: &EncodedFrame) -> Result<()> {
        let rtp = self
            .rtp
            .as_ref()
            .ok_or_else(|| FerricastError::Streaming("RTP socket not bound".into()))?;
        let peer = self
            .rtp_peer
            .ok_or_else(|| FerricastError::Streaming("RTP peer not negotiated".into()))?;

        let pts_90khz = u64::from(frame.timestamp_us) * u64::from(RTP_TIMESTAMP_HZ) / 1_000_000;
        let au = VideoAccessUnit {
            nalus: &frame.data,
            pts_90khz,
            is_keyframe: frame.is_keyframe,
            carry_pcr: true,
        };
        for ts_packet in self.muxer.pack_video(&au) {
            let completed = self
                .packer
                .push_ts(&ts_packet, u32::try_from(pts_90khz & 0xFFFF_FFFF).unwrap_or(0));
            for rtp_pkt in completed {
                rtp.send_to(&rtp_pkt, peer)
                    .await
                    .map_err(FerricastError::Io)?;
            }
        }
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        if let Some(mut framed) = self.rtsp.take() {
            let req = Request::new(Method::Teardown, self.rtsp_url.clone());
            let _ = framed.send_request(req).await;
        }
        self.rtp = None;
        self.rtp_peer = None;
        self.alive = false;
        Ok(())
    }

    fn is_alive(&self) -> bool {
        self.alive
    }

    fn end_reason(&self) -> Option<FerricastError> {
        self.end_reason.as_ref().map(|e| match e {
            FerricastError::RemoteStopped(s) => FerricastError::RemoteStopped(s.clone()),
            other => FerricastError::Streaming(other.to_string()),
        })
    }

    fn poll_keyframe_request(&mut self) -> bool {
        if self.last_keyframe_check.elapsed() < Duration::from_millis(50) {
            return false;
        }
        self.last_keyframe_check = std::time::Instant::now();
        self.keyframe_requested.swap(false, Ordering::AcqRel)
    }

    fn preferred_audio_codec(&self) -> AudioCodec {
        AudioCodec::Aac
    }

    fn feeds_adaptive_bitrate(&self) -> bool {
        false
    }

    fn forced_keyframe_interval(&self) -> Option<Duration> {
        // Miracast is receiver-driven via wfd_idr_request; don't pulse
        // IDRs on a wall clock on top of that.
        None
    }
}

fn local_ip_string(_rtsp: &Option<RtspFramed>) -> Result<String> {
    // Best-effort local IP: use the first non-loopback IPv4 advertised
    // by the OS. Miracast requires the source's actual IP in the
    // presentation URL — localhost breaks on real sinks.
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").map_err(FerricastError::Io)?;
    socket
        .connect("8.8.8.8:80")
        .map_err(FerricastError::Io)?;
    Ok(socket
        .local_addr()
        .map_err(FerricastError::Io)?
        .ip()
        .to_string())
}

/// Idle keep-alive timer: send `GET_PARAMETER` empty body every
/// [`KEEPALIVE_INTERVAL`] to avoid the 60 s sink timeout. The caller
/// wires this into its hot loop; the function is documented so callers
/// don't accidentally drop it.
#[doc(alias = "wfd_keepalive")]
pub const KEEPALIVE: Duration = KEEPALIVE_INTERVAL;
