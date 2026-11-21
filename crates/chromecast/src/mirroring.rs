//! Cast Streaming (mirroring) session — low-latency RTP/UDP alternative to HLS.
//!
//! Protocol flow on top of the CASTv2 TLS control channel:
//! 1. [`ChromecastMirrorSession::connect`] — TLS handshake, LAUNCH the
//!    Mirroring Receiver app (`0F5096E8`), latch its `transportId`/`sessionId`.
//! 2. [`ChromecastMirrorSession::setup_stream`] — send an OFFER (Opus audio +
//!    H.264 video) on `urn:x-cast:com.google.cast.webrtc` and wait for the
//!    ANSWER carrying the receiver's UDP port and per-stream SSRCs.
//! 3. [`ChromecastMirrorSession::send_frame`] /
//!    [`ChromecastMirrorSession::send_audio_frame`] hand frames to the UDP
//!    transport task, which encrypts, packetizes, paces, retransmits and
//!    exchanges RTCP with the receiver.
//! 4. [`ChromecastMirrorSession::stop`] — STOP the app, close the channel.
//!
//! Wire formats follow openscreen's `cast/streaming` implementation.

mod offer;
mod rtcp;
mod rtp;
mod sender;
mod transport;

use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, PoisonError};
use std::time::{Duration, Instant};

use bytes::BytesMut;
use ferricast_core::{
    AudioCodec, AudioFrame, CastSession, Codec, ConnectOutcome, Device, EncodedFrame,
    FerricastError, Result, StreamConfig, bind_udp,
};
use rand::{Rng, RngCore};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::castv2::{
    CastMessage, CastV2Error, DEFAULT_RECEIVER_ID, DEFAULT_SENDER_ID, MIRRORING_APP_ID,
    ReceiverStatusPayload, close_message, connect_message, get_status_message, launch_message,
    namespace, ping_message, pong_message, stop_app_message,
};
use crate::wire::{self, SharedWriter};

use self::offer::{AnswerProblem, AudioOfferParams, VideoOfferParams, build_offer, parse_answer};
use self::rtp::{FrameCrypto, RtpPayloadType};
use self::sender::{StreamKind, StreamParams, StreamSender};
use self::transport::{MediaFrame, Transport, TransportShared};

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
/// No message at all (PING/PONG included) for this long means the receiver
/// is gone.
const CONTROL_INACTIVITY_TIMEOUT: Duration = Duration::from_secs(15);
const LAUNCH_TIMEOUT: Duration = Duration::from_secs(15);
const ANSWER_TIMEOUT: Duration = Duration::from_secs(10);
/// Minimum spacing of GET_STATUS requests while waiting for the app.
const STATUS_NUDGE_INTERVAL: Duration = Duration::from_millis(500);
/// After the first frame is sent, the receiver must acknowledge a frame
/// within this time or mirroring is considered not to work.
const NO_PROGRESS_TIMEOUT: Duration = Duration::from_secs(5);
/// Receivers send RTCP every 500 ms; this much silence means it is gone.
const RTCP_SILENCE_TIMEOUT: Duration = Duration::from_secs(5);
/// Frames buffered between the encoder loop and the transport task.
const FRAME_QUEUE: usize = 256;
const DEFAULT_AUDIO_SAMPLE_RATE: u32 = 48_000;

/// Why a mirror session can no longer stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MirrorFailure {
    /// The receiver never acknowledged a frame: Cast Streaming does not work
    /// with this device or network (firewall, unsupported firmware).
    NoProgress,
    /// The receiver stopped sending RTCP after streaming had worked.
    ReceiverGone,
    /// The CASTv2 control channel closed or went silent.
    ControlLost(String),
    /// The mirroring app was stopped or replaced on the receiver.
    RemoteStopped(String),
}

/// Aborts the task when dropped, so early returns never leak tasks.
struct AbortOnDrop(JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// How the control channel ended, as seen by the read loop.
#[derive(Debug, Clone)]
enum ControlEnd {
    Lost(String),
    RemoteStopped(String),
}

#[derive(Debug, Default)]
struct ControlShared {
    alive: AtomicBool,
    end: StdMutex<Option<ControlEnd>>,
}

impl ControlShared {
    fn finish(&self, end: ControlEnd) {
        self.end
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_or_insert(end);
        self.alive.store(false, Ordering::Relaxed);
    }

    fn end(&self) -> Option<ControlEnd> {
        self.end
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

struct ConnectedState {
    writer: SharedWriter,
    transport_id: String,
    session_id: String,
    peer_ip: IpAddr,
    request_id: Arc<AtomicI64>,
    shared: Arc<ControlShared>,
    answers: mpsc::Receiver<String>,
    _read_task: AbortOnDrop,
    _heartbeat_task: AbortOnDrop,
}

struct MirrorStream {
    frames: mpsc::Sender<MediaFrame>,
    shared: Arc<TransportShared>,
    audio_enabled: bool,
    feeds_adaptive: bool,
    _task: AbortOnDrop,
}

#[derive(Default)]
pub struct ChromecastMirrorSession {
    control: Option<ConnectedState>,
    stream: Option<MirrorStream>,
    offer_seq: u32,
}

fn build_err(what: &str) -> impl FnOnce(CastV2Error) -> FerricastError + '_ {
    move |e| FerricastError::Protocol(format!("build {what}: {e}"))
}

impl ChromecastMirrorSession {
    /// Why streaming stopped working, if it did.
    #[must_use]
    pub fn failure(&self) -> Option<MirrorFailure> {
        let control = self.control.as_ref()?;
        if let Some(end) = control.shared.end() {
            return Some(match end {
                ControlEnd::Lost(why) => MirrorFailure::ControlLost(why),
                ControlEnd::RemoteStopped(why) => MirrorFailure::RemoteStopped(why),
            });
        }
        let stats = self.stream.as_ref()?.shared.stats();
        let now = Instant::now();
        let first_sent = stats.first_video_sent_at.or(stats.first_audio_sent_at)?;
        let progressed =
            stats.has_progress(StreamKind::Video) || stats.has_progress(StreamKind::Audio);
        if !progressed {
            return (now.saturating_duration_since(first_sent) > NO_PROGRESS_TIMEOUT)
                .then_some(MirrorFailure::NoProgress);
        }
        stats
            .last_rtcp_at
            .is_some_and(|at| now.saturating_duration_since(at) > RTCP_SILENCE_TIMEOUT)
            .then_some(MirrorFailure::ReceiverGone)
    }

    fn push_frame(&self, frame: MediaFrame) -> Result<()> {
        if let Some(MirrorFailure::RemoteStopped(why)) = self.failure() {
            return Err(FerricastError::RemoteStopped(why));
        }
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| FerricastError::Streaming("mirror stream not initialised".into()))?;
        match stream.frames.try_send(frame) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(frame)) => {
                if frame.kind == StreamKind::Video {
                    stream.shared.mark_upstream_drop();
                }
                debug!(kind = ?frame.kind, "mirror: transport queue full, frame dropped");
                Ok(())
            }
            Err(mpsc::error::TrySendError::Closed(_)) => Err(FerricastError::Streaming(
                "cast streaming transport stopped".into(),
            )),
        }
    }
}

/// Random SSRC in the range openscreen uses for the stream type; audio gets
/// the lower, higher-priority range.
fn random_ssrc(kind: StreamKind) -> u32 {
    let mut rng = rand::thread_rng();
    match kind {
        StreamKind::Audio => rng.gen_range(1..=50_000),
        StreamKind::Video => rng.gen_range(50_001..=100_000),
    }
}

/// AES key / IV mask material from the operating system CSPRNG.
fn random_key() -> [u8; 16] {
    let mut key = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut key);
    key
}

impl CastSession for ChromecastMirrorSession {
    async fn connect(&mut self, device: &Device) -> Result<ConnectOutcome> {
        if device.protocol != "chromecast" {
            return Err(FerricastError::Protocol(format!(
                "ChromecastMirrorSession cannot connect to {:?}",
                device.protocol
            )));
        }
        if self.control.is_some() {
            return Err(FerricastError::SessionAlreadyActive(device.name.clone()));
        }

        info!(name = %device.name, addr = %device.addr, "connecting (mirror) to chromecast");
        let (reader, writer, _local_ip) = wire::connect(device.addr, device.port).await?;

        let shared = Arc::new(ControlShared::default());
        shared.alive.store(true, Ordering::Relaxed);
        let request_id = Arc::new(AtomicI64::new(1));
        let (answer_tx, answers) = mpsc::channel::<String>(4);
        let (launch_tx, launch_rx) = oneshot::channel();
        // Dropped (and aborted) on every early return below.
        let read_task = AbortOnDrop(tokio::spawn(
            ControlReader {
                writer: writer.clone(),
                shared: shared.clone(),
                request_id: request_id.clone(),
                answers: answer_tx,
                launch: Some(launch_tx),
                app_session_id: None,
                app_transport_id: None,
                last_status_nudge: None,
            }
            .run(reader),
        ));

        let connect = connect_message(DEFAULT_SENDER_ID, DEFAULT_RECEIVER_ID)
            .map_err(build_err("CONNECT"))?;
        wire::send(&writer, &connect).await?;
        wire::send(&writer, &ping_message().map_err(build_err("PING"))?).await?;
        let launch_id = request_id.fetch_add(1, Ordering::Relaxed);
        let launch = launch_message(launch_id, MIRRORING_APP_ID).map_err(build_err("LAUNCH"))?;
        wire::send(&writer, &launch).await?;
        let status_id = request_id.fetch_add(1, Ordering::Relaxed);
        let status = get_status_message(status_id).map_err(build_err("GET_STATUS"))?;
        wire::send(&writer, &status).await?;

        let (transport_id, session_id) = match tokio::time::timeout(LAUNCH_TIMEOUT, launch_rx).await
        {
            Ok(Ok(Ok(ids))) => ids,
            Ok(Ok(Err(e))) => return Err(e),
            Ok(Err(_)) => {
                return Err(FerricastError::Connection(
                    "control channel closed before the mirroring app launched".into(),
                ));
            }
            Err(_) => {
                return Err(FerricastError::Timeout(format!(
                    "waiting for the mirroring app to launch ({}s)",
                    LAUNCH_TIMEOUT.as_secs()
                )));
            }
        };
        debug!(transport_id, session_id, "mirroring app launched");

        let app_connect =
            connect_message(DEFAULT_SENDER_ID, &transport_id).map_err(build_err("app CONNECT"))?;
        wire::send(&writer, &app_connect).await?;

        let heartbeat_task =
            AbortOnDrop(tokio::spawn(heartbeat_loop(writer.clone(), shared.clone())));
        self.control = Some(ConnectedState {
            writer,
            transport_id,
            session_id,
            peer_ip: device.addr,
            request_id,
            shared,
            answers,
            _read_task: read_task,
            _heartbeat_task: heartbeat_task,
        });
        Ok(ConnectOutcome::Ready)
    }

    async fn setup_stream(&mut self, config: &StreamConfig) -> Result<()> {
        if config.codec != Codec::H264 {
            return Err(FerricastError::UnsupportedCodec {
                codec: config.codec,
                protocol: "chromecast-mirror",
            });
        }
        self.offer_seq = self.offer_seq.wrapping_add(1);
        let seq = self.offer_seq;
        let control = self
            .control
            .as_mut()
            .ok_or(FerricastError::NoActiveSession)?;

        let audio = match &config.audio {
            Some(a) if a.codec == AudioCodec::Opus => Some(AudioOfferParams {
                ssrc: random_ssrc(StreamKind::Audio),
                aes_key: random_key(),
                aes_iv_mask: random_key(),
                sample_rate: if a.sample_rate == 0 {
                    DEFAULT_AUDIO_SAMPLE_RATE
                } else {
                    a.sample_rate
                },
                channels: a.channels.max(1),
                bit_rate: a.bitrate_kbps.saturating_mul(1_000),
            }),
            Some(a) => {
                warn!(codec = ?a.codec, "mirror: audio must be Opus; offering video only");
                None
            }
            None => None,
        };
        let video = VideoOfferParams {
            ssrc: random_ssrc(StreamKind::Video),
            aes_key: random_key(),
            aes_iv_mask: random_key(),
            width: config.width,
            height: config.height,
            fps: config.fps,
            max_bit_rate: config.bitrate_kbps.saturating_mul(1_000),
        };
        let envelope = build_offer(seq, audio.as_ref(), &video);
        let offered: Vec<u32> = envelope
            .offer
            .supported_streams
            .iter()
            .map(offer::StreamOffer::index)
            .collect();
        let audio_index = audio.as_ref().map(|_| 0u32);
        let video_index = u32::from(audio.is_some());

        let offer_msg = CastMessage::new_json(
            DEFAULT_SENDER_ID,
            &control.transport_id,
            namespace::WEBRTC,
            &envelope,
        )
        .map_err(build_err("OFFER"))?;
        if let Ok(json) = serde_json::to_string(&envelope) {
            debug!(offer_json = %json, "Cast Streaming OFFER");
        }
        // Discard ANSWERs to earlier OFFERs.
        while control.answers.try_recv().is_ok() {}
        wire::send(&control.writer, &offer_msg).await?;
        info!(transport_id = %control.transport_id, seq, "sent Cast Streaming OFFER");

        let deadline = tokio::time::Instant::now() + ANSWER_TIMEOUT;
        let answer = loop {
            let payload = match tokio::time::timeout_at(deadline, control.answers.recv()).await {
                Ok(Some(p)) => p,
                Ok(None) => {
                    return Err(FerricastError::Connection(
                        "control channel closed before the Cast Streaming ANSWER".into(),
                    ));
                }
                Err(_) => {
                    return Err(FerricastError::Timeout(format!(
                        "waiting for Cast Streaming ANSWER ({}s)",
                        ANSWER_TIMEOUT.as_secs()
                    )));
                }
            };
            match parse_answer(&payload, seq, &offered) {
                Ok(a) => break a,
                Err(AnswerProblem::SequenceMismatch { got, .. }) => {
                    debug!(?got, seq, "mirror: ignoring ANSWER to another OFFER");
                }
                Err(problem) => {
                    return Err(FerricastError::Protocol(format!(
                        "Cast Streaming {problem}"
                    )));
                }
            }
        };
        info!(udp_port = answer.udp_port, streams = ?answer.streams, "received Cast Streaming ANSWER");

        let Some(video_receiver_ssrc) = answer.receiver_ssrc_for(video_index) else {
            return Err(FerricastError::Protocol(
                "Cast Streaming ANSWER did not accept the video stream".into(),
            ));
        };
        let peer_udp = SocketAddr::new(control.peer_ip, answer.udp_port);
        let socket = bind_udp(control.peer_ip, config.port_range)
            .await
            .map_err(|e| FerricastError::Connection(format!("bind cast streaming UDP: {e}")))?;
        socket
            .connect(peer_udp)
            .await
            .map_err(|e| FerricastError::Connection(format!("connect cast streaming UDP: {e}")))?;

        let target_delay = Duration::from_millis(u64::from(offer::TARGET_DELAY_MS));
        let mut senders = vec![StreamSender::new(StreamParams {
            kind: StreamKind::Video,
            payload_type: RtpPayloadType::H264,
            sender_ssrc: video.ssrc,
            receiver_ssrc: video_receiver_ssrc,
            rtp_timebase: offer::VIDEO_TIMEBASE,
            target_playout_delay: target_delay,
            crypto: FrameCrypto::new(video.aes_key, video.aes_iv_mask),
            first_sequence: rand::thread_rng().r#gen(),
        })];
        let audio_receiver_ssrc = audio_index.and_then(|i| answer.receiver_ssrc_for(i));
        match (audio.as_ref(), audio_receiver_ssrc) {
            (Some(a), Some(receiver_ssrc)) => senders.push(StreamSender::new(StreamParams {
                kind: StreamKind::Audio,
                payload_type: RtpPayloadType::Opus,
                sender_ssrc: a.ssrc,
                receiver_ssrc,
                rtp_timebase: a.sample_rate,
                target_playout_delay: target_delay,
                crypto: FrameCrypto::new(a.aes_key, a.aes_iv_mask),
                first_sequence: rand::thread_rng().r#gen(),
            })),
            (Some(_), None) => {
                warn!("mirror: receiver did not accept the audio stream; sending video only");
            }
            (None, _) => {}
        }
        let audio_enabled = senders.len() == 2;

        let shared = Arc::new(TransportShared::default());
        let (frames_tx, frames_rx) = mpsc::channel(FRAME_QUEUE);
        let transport = Transport::new(
            Arc::new(socket),
            senders,
            shared.clone(),
            config.adaptive.clone(),
        );
        let task = AbortOnDrop(tokio::spawn(transport.run(frames_rx)));
        self.stream = Some(MirrorStream {
            frames: frames_tx,
            shared,
            audio_enabled,
            feeds_adaptive: config.adaptive.is_some(),
            _task: task,
        });
        info!(%peer_udp, video_ssrc = video.ssrc, audio_enabled, "Cast Streaming transport started");
        Ok(())
    }

    async fn send_frame(&mut self, frame: &EncodedFrame) -> Result<()> {
        if frame.codec != Codec::H264 {
            return Err(FerricastError::UnsupportedCodec {
                codec: frame.codec,
                protocol: "chromecast-mirror",
            });
        }
        self.push_frame(MediaFrame {
            kind: StreamKind::Video,
            data: frame.data.clone(),
            is_key_frame: frame.is_keyframe,
            capture_us: frame.timestamp_us,
        })
    }

    async fn send_audio_frame(&mut self, frame: &AudioFrame) -> Result<()> {
        if frame.codec != AudioCodec::Opus {
            return Err(FerricastError::Encoder(format!(
                "chromecast mirror: expected Opus audio, got {:?}",
                frame.codec
            )));
        }
        if !self.stream.as_ref().is_some_and(|s| s.audio_enabled) {
            return Ok(());
        }
        // Every Opus frame decodes on its own.
        self.push_frame(MediaFrame {
            kind: StreamKind::Audio,
            data: frame.data.clone(),
            is_key_frame: true,
            capture_us: frame.timestamp_us,
        })
    }

    async fn stop(&mut self) -> Result<()> {
        self.stream = None;
        let Some(control) = self.control.take() else {
            return Ok(());
        };
        let stop_id = control.request_id.fetch_add(1, Ordering::Relaxed);
        if let Ok(msg) = stop_app_message(stop_id, &control.session_id)
            && let Err(e) = wire::send(&control.writer, &msg).await
        {
            warn!(%e, "mirror STOP failed");
        }
        for destination in [control.transport_id.as_str(), DEFAULT_RECEIVER_ID] {
            if let Ok(close) = close_message(DEFAULT_SENDER_ID, destination) {
                let _ = wire::send(&control.writer, &close).await;
            }
        }
        control.shared.alive.store(false, Ordering::Relaxed);
        info!("chromecast mirror session stopped");
        Ok(())
    }

    fn is_alive(&self) -> bool {
        self.control
            .as_ref()
            .is_some_and(|c| c.shared.alive.load(Ordering::Relaxed))
            && self.failure().is_none()
    }

    fn poll_keyframe_request(&mut self) -> bool {
        self.stream
            .as_ref()
            .is_some_and(|s| s.shared.key_frame_requested.swap(false, Ordering::Relaxed))
    }

    fn preferred_audio_codec(&self) -> AudioCodec {
        AudioCodec::Opus
    }

    fn feeds_adaptive_bitrate(&self) -> bool {
        self.stream.as_ref().is_some_and(|s| s.feeds_adaptive)
    }

    fn end_reason(&self) -> Option<FerricastError> {
        self.failure().map(|failure| match failure {
            MirrorFailure::RemoteStopped(why) => FerricastError::RemoteStopped(why),
            other => FerricastError::Streaming(format!("{other:?}")),
        })
    }

    /// RTCP picture-loss requests drive keyframes; periodic IDRs only
    /// waste bandwidth on this path.
    fn forced_keyframe_interval(&self) -> Option<std::time::Duration> {
        None
    }
}

/// Reads the CASTv2 control channel of a mirroring session.
struct ControlReader {
    writer: SharedWriter,
    shared: Arc<ControlShared>,
    request_id: Arc<AtomicI64>,
    answers: mpsc::Sender<String>,
    launch: Option<oneshot::Sender<Result<(String, String)>>>,
    /// Session id of our app once launched.
    app_session_id: Option<String>,
    /// Transport id of our app once launched.
    app_transport_id: Option<String>,
    last_status_nudge: Option<Instant>,
}

impl ControlReader {
    async fn run<R>(mut self, mut reader: R)
    where
        R: tokio::io::AsyncRead + Unpin,
    {
        let mut buf = BytesMut::with_capacity(8 * 1024);
        let end = loop {
            let received = tokio::time::timeout(
                CONTROL_INACTIVITY_TIMEOUT,
                wire::recv(&mut reader, &mut buf),
            )
            .await;
            let msg = match received {
                Err(_) => {
                    break ControlEnd::Lost(format!(
                        "no message from the receiver for {}s",
                        CONTROL_INACTIVITY_TIMEOUT.as_secs()
                    ));
                }
                Ok(Ok(Some(msg))) => msg,
                Ok(Ok(None)) => break ControlEnd::Lost("receiver closed the connection".into()),
                Ok(Err(e)) => break ControlEnd::Lost(format!("control channel error: {e}")),
            };
            if let Some(end) = self.handle(&msg).await {
                break end;
            }
        };
        warn!(?end, "mirror control channel ended");
        if let Some(tx) = self.launch.take() {
            let (ControlEnd::Lost(why) | ControlEnd::RemoteStopped(why)) = &end;
            let _ = tx.send(Err(FerricastError::Connection(format!(
                "mirroring app launch failed: {why}"
            ))));
        }
        self.shared.finish(end);
    }

    async fn handle(&mut self, msg: &CastMessage) -> Option<ControlEnd> {
        let msg_type = msg.message_type();
        match (msg.namespace.as_str(), msg_type.as_deref()) {
            (namespace::HEARTBEAT, Some("PING")) => {
                if let Ok(pong) = pong_message()
                    && wire::send(&self.writer, &pong).await.is_err()
                {
                    return Some(ControlEnd::Lost("failed to answer PING".into()));
                }
                None
            }
            // The app closing its own virtual connection is how the receiver ends
            // a session on purpose (user stopped casting, app exited).
            (namespace::CONNECTION, Some("CLOSE"))
                if self.app_transport_id.as_deref() == Some(msg.source_id.as_str()) =>
            {
                Some(ControlEnd::RemoteStopped(
                    "mirroring app closed the connection".into(),
                ))
            }
            (namespace::CONNECTION, Some("CLOSE")) => Some(ControlEnd::Lost(format!(
                "receiver sent CLOSE from {}",
                msg.source_id
            ))),
            (namespace::RECEIVER, Some("LAUNCH_ERROR" | "INVALID_REQUEST")) => {
                let detail = msg.payload_utf8.clone().unwrap_or_default();
                if let Some(tx) = self.launch.take() {
                    let _ = tx.send(Err(FerricastError::Protocol(format!(
                        "mirroring app LAUNCH refused: {detail}"
                    ))));
                } else {
                    warn!(%detail, "mirror: receiver reported an error");
                }
                None
            }
            (namespace::RECEIVER, Some("RECEIVER_STATUS")) => self.on_receiver_status(msg).await,
            (namespace::WEBRTC, Some("ANSWER")) => {
                if let Some(payload) = msg.payload_utf8.clone() {
                    debug!(%payload, "mirror: ANSWER payload");
                    let _ = self.answers.try_send(payload);
                }
                None
            }
            (namespace::WEBRTC, ty) => {
                debug!(?ty, payload = ?msg.payload_utf8, "mirror: webrtc message ignored");
                None
            }
            // PONGs and anything else only count as activity.
            _ => None,
        }
    }

    async fn on_receiver_status(&mut self, msg: &CastMessage) -> Option<ControlEnd> {
        let payload: ReceiverStatusPayload = msg.parse_payload().ok()?;
        // Statuses without an application list (e.g. volume-only) say
        // nothing about our app.
        let applications = payload.status?.applications;
        match classify_status(self.app_session_id.as_deref(), &applications) {
            StatusVerdict::Running => return None,
            StatusVerdict::Stopped(why) => return Some(ControlEnd::RemoteStopped(why)),
            StatusVerdict::Launched {
                transport_id,
                session_id,
            } => {
                info!(%transport_id, "mirroring app launched");
                self.app_session_id = Some(session_id.clone());
                self.app_transport_id = Some(transport_id.clone());
                if let Some(tx) = self.launch.take() {
                    let _ = tx.send(Ok((transport_id, session_id)));
                }
                return None;
            }
            StatusVerdict::NotReady => {}
        }
        debug!(payload = ?msg.payload_utf8, "RECEIVER_STATUS: mirroring app not ready");
        let now = Instant::now();
        if self
            .last_status_nudge
            .is_none_or(|at| now.saturating_duration_since(at) >= STATUS_NUDGE_INTERVAL)
        {
            self.last_status_nudge = Some(now);
            let id = self.request_id.fetch_add(1, Ordering::Relaxed);
            if let Ok(gs) = get_status_message(id) {
                let _ = wire::send(&self.writer, &gs).await;
            }
        }
        None
    }
}

/// What a RECEIVER_STATUS application list means for our mirroring app.
#[derive(Debug, PartialEq, Eq)]
enum StatusVerdict {
    Running,
    Stopped(String),
    Launched {
        transport_id: String,
        session_id: String,
    },
    NotReady,
}

fn classify_status(
    app_session_id: Option<&str>,
    applications: &[crate::castv2::ApplicationInfo],
) -> StatusVerdict {
    if let Some(session_id) = app_session_id {
        if applications.iter().any(|a| a.session_id == session_id) {
            return StatusVerdict::Running;
        }
        let replaced_by = applications.iter().find(|a| !a.display_name.is_empty());
        return StatusVerdict::Stopped(replaced_by.map_or_else(
            || "mirroring app stopped on the receiver".to_owned(),
            |a| format!("mirroring replaced by {}", a.display_name),
        ));
    }
    applications
        .iter()
        .find(|a| {
            a.app_id.eq_ignore_ascii_case(MIRRORING_APP_ID)
                && !a.transport_id.is_empty()
                && !a.session_id.is_empty()
        })
        .map_or(StatusVerdict::NotReady, |a| StatusVerdict::Launched {
            transport_id: a.transport_id.clone(),
            session_id: a.session_id.clone(),
        })
}

async fn heartbeat_loop(writer: SharedWriter, shared: Arc<ControlShared>) {
    let mut tick = tokio::time::interval(HEARTBEAT_INTERVAL);
    tick.tick().await;
    while shared.alive.load(Ordering::Relaxed) {
        tick.tick().await;
        let Ok(ping) = ping_message() else { break };
        if wire::send(&writer, &ping).await.is_err() {
            shared.finish(ControlEnd::Lost("failed to send PING".into()));
            break;
        }
    }
}

#[cfg(test)]
mod tests;
