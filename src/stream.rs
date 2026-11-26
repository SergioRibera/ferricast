//! Sender pipeline: screen capture → encoder → protocol session.
//!
//! [`StreamRequest::prepare`] brings a stream up (capture, output
//! sizing against the device limits, encoder, session handshake) and
//! [`StreamTask::run`] drives it: paces frames, feeds the audio
//! pipeline, follows the adaptive bitrate and rebuilds the session
//! when the receiver drops it.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use uuid::Uuid;

use ferricast_capture::NativeAudioCapture;
use ferricast_core::{
    AdaptiveBitrateState, AudioCapture, AudioCaptureConfig, AudioCodec, AudioEncoder,
    AudioEncoderConfig, AudioFrame, AudioSource, AudioStreamConfig, CaptureConfig, CaptureSource,
    CapturedFrame, CastSession, ConnectOutcome, Device, EncodedFrame, EncoderConfig,
    FerricastError, GpuId, GpuPreference, PairingChallenge, PairingResponse, Result, ScreenCapture,
    StreamConfig, VideoEncoder,
};
use ferricast_encoder::aac::AacEncoder;
#[cfg(feature = "fdk-aac")]
use ferricast_encoder::aac::{ELD_SAMPLE_RATE, FdkEldEncoder};
#[cfg(feature = "opus-rs")]
use ferricast_encoder::opus::OpusEncoder;
use ferricast_encoder::scale::FrameSize;

use crate::manager::ManagerEvent;

type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// How long to wait for capture to deliver its first frame (the
/// portal dialog and PipeWire format negotiation happen before it).
const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(30);
/// How long the reachability probe of the device's control port may
/// take before the device counts as unreachable.
const REACHABILITY_TIMEOUT: Duration = Duration::from_secs(3);
/// How long the probe may hold back capture start (and with it the
/// portal dialog) to report a fast failure first.
const REACHABILITY_GRACE: Duration = Duration::from_millis(250);
/// How long the user has to answer a pairing prompt.
const PAIRING_TIMEOUT: Duration = Duration::from_mins(2);
/// Password prompts per connection before a receiver that keeps
/// rejecting the answer is given up on.
const MAX_PASSWORD_PROMPTS: u32 = 3;
/// Reconnect attempts in a row before the stream is given up.
const MAX_CONSECUTIVE_FAILURES: u32 = 5;
/// A session that stayed up this long resets the failure counter: one
/// transient after ten healthy minutes is not five drops in 20 s.
const HEALTHY_RESET_AFTER: Duration = Duration::from_mins(1);
/// Audio older than this is dropped instead of sent; it can only
/// arrive after the matching video and would play out of sync.
const MAX_AUDIO_AGE_US: u64 = 300_000;
/// Encode failures in a row (each followed by an encoder reconfigure)
/// before the stream is given up.
const MAX_CONSECUTIVE_ENCODE_ERRORS: u32 = 3;
/// Encoded audio frames buffered between the audio task and the
/// sender loop.
const AUDIO_QUEUE_FRAMES: usize = 64;
/// Frames read off the capture queue at most when a session becomes
/// ready; captures queue a few at most.
const MAX_STALE_FRAMES: usize = 16;

/// Object-safe view of a [`CastSession`]; the native trait returns
/// `impl Future`, which can't be boxed as `dyn`.
pub(crate) trait ErasedSession: Send + Sync {
    fn connect<'a>(&'a mut self, device: &'a Device) -> BoxFut<'a, Result<ConnectOutcome>>;
    fn submit_pairing(&mut self, response: PairingResponse) -> BoxFut<'_, Result<()>>;
    fn setup_stream<'a>(&'a mut self, config: &'a StreamConfig) -> BoxFut<'a, Result<()>>;
    fn send_frame<'a>(&'a mut self, frame: &'a EncodedFrame) -> BoxFut<'a, Result<()>>;
    fn send_audio_frame<'a>(&'a mut self, frame: &'a AudioFrame) -> BoxFut<'a, Result<()>>;
    fn stop(&mut self) -> BoxFut<'_, Result<()>>;
    fn is_alive(&self) -> bool;
    fn end_reason(&self) -> Option<FerricastError>;
    fn poll_keyframe_request(&mut self) -> bool;
    fn preferred_audio_codec(&self) -> AudioCodec;
    fn feeds_adaptive_bitrate(&self) -> bool;
    fn forced_keyframe_interval(&self) -> Option<Duration>;
}

impl<T: CastSession> ErasedSession for T {
    fn connect<'a>(&'a mut self, device: &'a Device) -> BoxFut<'a, Result<ConnectOutcome>> {
        Box::pin(CastSession::connect(self, device))
    }
    fn submit_pairing(&mut self, response: PairingResponse) -> BoxFut<'_, Result<()>> {
        Box::pin(CastSession::submit_pairing(self, response))
    }
    fn setup_stream<'a>(&'a mut self, config: &'a StreamConfig) -> BoxFut<'a, Result<()>> {
        Box::pin(CastSession::setup_stream(self, config))
    }
    fn send_frame<'a>(&'a mut self, frame: &'a EncodedFrame) -> BoxFut<'a, Result<()>> {
        Box::pin(CastSession::send_frame(self, frame))
    }
    fn send_audio_frame<'a>(&'a mut self, frame: &'a AudioFrame) -> BoxFut<'a, Result<()>> {
        Box::pin(CastSession::send_audio_frame(self, frame))
    }
    fn stop(&mut self) -> BoxFut<'_, Result<()>> {
        Box::pin(CastSession::stop(self))
    }
    fn is_alive(&self) -> bool {
        CastSession::is_alive(self)
    }
    fn end_reason(&self) -> Option<FerricastError> {
        CastSession::end_reason(self)
    }
    fn poll_keyframe_request(&mut self) -> bool {
        CastSession::poll_keyframe_request(self)
    }
    fn preferred_audio_codec(&self) -> AudioCodec {
        CastSession::preferred_audio_codec(self)
    }
    fn feeds_adaptive_bitrate(&self) -> bool {
        CastSession::feeds_adaptive_bitrate(self)
    }
    fn forced_keyframe_interval(&self) -> Option<Duration> {
        CastSession::forced_keyframe_interval(self)
    }
}

/// Builds a fresh session for a device; kept by the stream so it can
/// reconnect without tearing capture and encoder down.
pub(crate) type SessionFactory =
    Arc<dyn Fn(&Device) -> Result<Box<dyn ErasedSession>> + Send + Sync>;

/// Receiving end of a stream's stop signal. Fires on an explicit
/// signal and when the sender is dropped (the stream was removed
/// from the registry).
pub(crate) struct CancelRx(mpsc::Receiver<()>);

impl CancelRx {
    pub(crate) fn new(rx: mpsc::Receiver<()>) -> Self {
        Self(rx)
    }

    /// Resolves once the stream should stop. Cancel-safe.
    pub(crate) async fn cancelled(&mut self) {
        let _ = self.0.recv().await;
    }
}

/// How a stream ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StreamEnd {
    /// Stopped from this side.
    Cancelled,
    /// The receiver or its user ended the session on purpose.
    RemoteStopped(String),
    /// Unrecoverable failure.
    Failed(String),
}

impl StreamEnd {
    /// Publish the terminal events for a stream that ended this way.
    /// `StreamStopped` always comes last so apps need only one
    /// terminal state.
    pub(crate) async fn report(self, event_tx: &mpsc::Sender<ManagerEvent>, device_id: Uuid) {
        let reason = match self {
            StreamEnd::Cancelled => None,
            StreamEnd::RemoteStopped(reason) => Some(reason),
            StreamEnd::Failed(message) => {
                let _ = event_tx
                    .send(ManagerEvent::StreamError { device_id, message })
                    .await;
                None
            }
        };
        let _ = event_tx
            .send(ManagerEvent::StreamStopped { device_id, reason })
            .await;
    }
}

/// Everything needed to bring a stream up.
pub(crate) struct StreamRequest<C, E> {
    pub device: Device,
    pub source: CaptureSource,
    pub capture: C,
    pub encoder: E,
    pub config: StreamConfig,
    /// Which GPU the encoder runs on.
    pub gpu: GpuPreference,
    pub create_session: SessionFactory,
    pub event_tx: mpsc::Sender<ManagerEvent>,
}

/// Values negotiated during bring-up.
struct Negotiated {
    first_frame: CapturedFrame,
    encoder_config: EncoderConfig,
    session_config: StreamConfig,
    adaptive: Arc<AdaptiveBitrateState>,
    session: Box<dyn ErasedSession>,
}

impl<C, E> StreamRequest<C, E>
where
    C: ScreenCapture + 'static,
    E: VideoEncoder + 'static,
{
    /// Start capture, size and configure the encoder against the
    /// device limits, then connect and set up the session.
    ///
    /// # Errors
    ///
    /// Whatever capture, encoder configuration or the session
    /// handshake fail with; capture is stopped again on failure.
    pub(crate) async fn prepare(self) -> Result<StreamTask<C, E>> {
        let Self {
            device,
            source,
            mut capture,
            mut encoder,
            config,
            gpu,
            create_session,
            event_tx,
        } = self;
        let caps = &device.capabilities;
        // Apply the fps ceiling before start so the capture worker's
        // throttle anchors to the right rate from the first buffer.
        // Capture prefers DMA-BUF layouts the GPU encoder that will be
        // selected takes without a copy, and hands it rotated frames only
        // when that encoder turns them. The capture GPU is not known yet, so
        // the prediction uses the preference alone. The first query brings
        // up the GPU (EGL + CUDA, a VA-API VPP), so it runs off the async
        // workers; the answers are cached for the process.
        let predicted = EncoderConfig {
            codec: config.codec,
            gpu: gpu.clone(),
            ..EncoderConfig::default()
        };
        let (consumer_modifiers, consumer_orients) = tokio::task::spawn_blocking(move || {
            (
                ferricast_encoder::dmabuf_modifiers(&predicted),
                ferricast_encoder::orients_gpu_frames(&predicted),
            )
        })
        .await
        .unwrap_or_default();
        let capture_config = CaptureConfig {
            fps: caps.max_fps.map_or(config.fps, |cap| config.fps.min(cap)),
            width: caps.max_width,
            height: caps.max_height,
            show_cursor: true,
            consumer_modifiers,
            consumer_orients,
        };
        start_capture_if_reachable(&device, &mut capture, source, capture_config).await?;

        let audio = config.audio.clone();
        let negotiated = match negotiate(
            &device,
            &mut capture,
            &mut encoder,
            config,
            &gpu,
            &create_session,
            &event_tx,
        )
        .await
        {
            Ok(n) => n,
            Err(e) => {
                let _ = capture.stop().await;
                return Err(e);
            }
        };

        let fps = negotiated.encoder_config.fps.max(1);
        Ok(StreamTask {
            capture,
            encoder,
            applied_kbps: negotiated.encoder_config.bitrate_kbps,
            encoder_config: negotiated.encoder_config,
            session_config: negotiated.session_config,
            adaptive: negotiated.adaptive,
            create_session,
            audio: AudioStage::new(audio, device.id, event_tx.clone()),
            frames: FramePacer::new(negotiated.first_frame, fps),
            source_size: None,
            reconnects: ReconnectBudget::default(),
            initial_session: Some(negotiated.session),
            device,
            event_tx,
        })
    }
}

/// Start capture while probing the device's control port, so an
/// offline or firewalled receiver fails the start in seconds instead
/// of after the user picked a source and the handshake timed out.
///
/// A refusal or missing route usually answers within a few
/// milliseconds: the probe gets [`REACHABILITY_GRACE`] to report one
/// before the portal opens. A silent drop only shows up as the
/// [`REACHABILITY_TIMEOUT`], which runs alongside the portal and stops
/// capture again when it fires. Devices without a known control port
/// (port 0) are not probed. Every address is tried, in order.
async fn start_capture_if_reachable<C: ScreenCapture>(
    device: &Device,
    capture: &mut C,
    source: CaptureSource,
    capture_config: CaptureConfig,
) -> Result<()> {
    if device.addresses.primary().port() == 0 {
        return capture.start(source, capture_config).await;
    }
    let probe = ferricast_core::probe_tcp(device.addresses.as_slice(), REACHABILITY_TIMEOUT);
    tokio::pin!(probe);
    let early = tokio::time::timeout(REACHABILITY_GRACE, &mut probe).await;
    let probe_result = match early {
        Ok(result) => {
            result?;
            capture.start(source, capture_config).await?;
            return Ok(());
        }
        Err(_still_probing) => {
            let (started, probed) = tokio::join!(capture.start(source, capture_config), probe);
            started?;
            probed
        }
    };
    if let Err(e) = probe_result {
        tracing::warn!(device = %device.name, error = %e, "receiver unreachable; stopping capture");
        let _ = capture.stop().await;
        return Err(e);
    }
    Ok(())
}

/// The GPU that rendered `frame`, for GPU frames that name it.
fn source_gpu(frame: &CapturedFrame) -> Option<GpuId> {
    match frame {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        CapturedFrame::Gpu(gpu) => gpu.producer.map(|producer| GpuId::Dev(producer.into())),
        _ => None,
    }
}

async fn negotiate<C: ScreenCapture, E: VideoEncoder>(
    device: &Device,
    capture: &mut C,
    encoder: &mut E,
    config: StreamConfig,
    gpu: &GpuPreference,
    create_session: &SessionFactory,
    event_tx: &mpsc::Sender<ManagerEvent>,
) -> Result<Negotiated> {
    // Capture backends only emit frames once the format is agreed, so
    // the first frame is the barrier after which size, pixel format
    // and framerate are known.
    tracing::info!(device = %device.name, "awaiting first frame from capture");
    let first_frame = tokio::time::timeout(FIRST_FRAME_TIMEOUT, capture.next_frame())
        .await
        .map_err(|_| {
            FerricastError::Timeout(format!(
                "capture delivered no frame within {}s",
                FIRST_FRAME_TIMEOUT.as_secs()
            ))
        })??;

    let caps = &device.capabilities;
    let source_size = match (first_frame.width(), first_frame.height()) {
        (0, _) | (_, 0) => {
            let (w, h) = capture.get_screen_size();
            FrameSize::new(
                u32::try_from(w).unwrap_or(u32::MAX),
                u32::try_from(h).unwrap_or(u32::MAX),
            )
        }
        (w, h) => FrameSize::new(w, h),
    };
    // Fixed for the stream's lifetime: later source size changes are
    // letterboxed into it by the encoder.
    let output = source_size.fit_within(caps.max_width, caps.max_height);

    let negotiated_fps = capture.get_framerate();
    let source_fps = if negotiated_fps > 0 {
        negotiated_fps
    } else {
        config.fps
    };
    let fps = caps.max_fps.map_or(source_fps, |cap| source_fps.min(cap));
    let bitrate_kbps = caps
        .max_bitrate_kbps
        .map_or(config.bitrate_kbps, |cap| config.bitrate_kbps.min(cap));

    let encoder_config = EncoderConfig {
        pixel_format: capture.get_pixel_format(),
        width: output.width,
        height: output.height,
        fps,
        bitrate_kbps,
        max_h264_profile: caps.max_h264_profile,
        max_h265_profile: caps.max_h265_profile,
        // Matches the HLS segment target so natural IDRs line up
        // with segment boundaries.
        keyframe_interval_secs: 2.0,
        gpu: gpu.clone(),
        source_gpu: source_gpu(&first_frame),
        ..EncoderConfig::default()
    };
    tracing::info!(
        device = %device.name,
        model = %device.model.as_deref().unwrap_or(""),
        src_width = source_size.width,
        src_height = source_size.height,
        out_width = output.width,
        out_height = output.height,
        negotiated_fps,
        fps,
        bitrate_kbps,
        profile = ?encoder_config.max_h264_profile,
        pixel_format = ?encoder_config.pixel_format,
        "encoder configured against device capabilities"
    );
    encoder.configure(&encoder_config)?;

    let adaptive = AdaptiveBitrateState::new(bitrate_kbps);
    // The session gets the real encoded size and the clamped rate:
    // its segmenter paces PTS by fps and the mirror OFFER derives the
    // H.264 level from the size.
    let mut session_config = StreamConfig {
        width: output.width,
        height: output.height,
        fps,
        bitrate_kbps,
        adaptive: Some(adaptive.clone()),
        ..config
    };
    let session = open_session(create_session, device, &mut session_config, event_tx).await?;
    Ok(Negotiated {
        first_frame,
        encoder_config,
        session_config,
        adaptive,
        session,
    })
}

/// Create, connect (pairing included) and set up a session. The
/// audio codec in `config` follows the new session's preference.
async fn open_session(
    create_session: &SessionFactory,
    device: &Device,
    config: &mut StreamConfig,
    event_tx: &mpsc::Sender<ManagerEvent>,
) -> Result<Box<dyn ErasedSession>> {
    let mut session = create_session(device)?;
    match session.connect(device).await? {
        ConnectOutcome::Ready => {}
        ConnectOutcome::PairingRequired(challenge) => {
            let response = ask_pairing(event_tx, device.id, challenge).await?;
            session.submit_pairing(response).await?;
        }
    }
    if let Some(audio) = config.audio.as_mut() {
        audio.codec = session.preferred_audio_codec();
    }
    if let Err(e) = setup_stream(session.as_mut(), device, config, event_tx).await {
        // Connected but not streaming: release the receiver so it
        // doesn't sit on the launched app's splash screen.
        let _ = session.stop().await;
        return Err(e);
    }
    Ok(session)
}

/// Set up the stream, asking the user for the receiver's password and
/// retrying on the same session when the receiver wants one, so the first
/// connection to a password-protected receiver needs no second attempt.
async fn setup_stream(
    session: &mut dyn ErasedSession,
    device: &Device,
    config: &StreamConfig,
    event_tx: &mpsc::Sender<ManagerEvent>,
) -> Result<()> {
    let mut asked = 0;
    loop {
        match session.setup_stream(config).await {
            Err(FerricastError::CredentialRequired { rejected }) => {
                if asked == MAX_PASSWORD_PROMPTS {
                    return Err(FerricastError::Connection(format!(
                        "{} rejected the password {MAX_PASSWORD_PROMPTS} times",
                        device.name
                    )));
                }
                asked += 1;
                let challenge = PairingChallenge::Credential { rejected };
                let response = ask_pairing(event_tx, device.id, challenge).await?;
                session.submit_pairing(response).await?;
            }
            result => return result,
        }
    }
}

async fn ask_pairing(
    event_tx: &mpsc::Sender<ManagerEvent>,
    device_id: Uuid,
    challenge: PairingChallenge,
) -> Result<PairingResponse> {
    let (respond_tx, respond_rx) = oneshot::channel::<PairingResponse>();
    event_tx
        .send(ManagerEvent::PairingRequired {
            device_id,
            challenge,
            respond: Arc::new(tokio::sync::Mutex::new(Some(respond_tx))),
        })
        .await
        .map_err(|_| FerricastError::Protocol("no listener for the pairing prompt".into()))?;
    match tokio::time::timeout(PAIRING_TIMEOUT, respond_rx).await {
        Ok(Ok(response)) => Ok(response),
        Ok(Err(_)) => Err(FerricastError::Protocol(
            "pairing prompt closed without an answer".into(),
        )),
        Err(_) => Err(FerricastError::Timeout(format!(
            "no pairing answer within {}s",
            PAIRING_TIMEOUT.as_secs()
        ))),
    }
}

/// Reconnect bookkeeping.
#[derive(Debug, Default)]
struct ReconnectBudget {
    failures: u32,
    last_success: Option<Instant>,
}

impl ReconnectBudget {
    /// Count one more attempt; `None` once the budget is spent.
    fn next_attempt(&mut self) -> Option<u32> {
        if self
            .last_success
            .is_some_and(|t| t.elapsed() > HEALTHY_RESET_AFTER)
        {
            self.failures = 0;
        }
        self.failures += 1;
        (self.failures <= MAX_CONSECUTIVE_FAILURES).then_some(self.failures)
    }

    fn succeeded(&mut self) {
        self.last_success = Some(Instant::now());
    }

    /// 1 s, 2 s, 4 s, 8 s, 16 s.
    fn backoff(attempt: u32) -> Duration {
        Duration::from_secs(1u64 << attempt.saturating_sub(1).min(4))
    }
}

/// Keeps the encoder fed at a steady rate. Wayland compositors only
/// emit a frame when the screen changes; segment-based receivers
/// starve without a steady cadence, so the last frame is repeated
/// (timestamp advanced) when capture is idle for a frame period.
struct FramePacer {
    seed: Option<CapturedFrame>,
    last: Option<CapturedFrame>,
    period: Duration,
    period_us: u64,
}

impl FramePacer {
    fn new(first_frame: CapturedFrame, fps: u32) -> Self {
        let fps = u64::from(fps.max(1));
        Self {
            seed: Some(first_frame),
            last: None,
            period: Duration::from_micros(1_000_000 / fps),
            period_us: 1_000_000 / fps,
        }
    }

    async fn next<C: ScreenCapture>(&mut self, capture: &mut C) -> Result<CapturedFrame> {
        if let Some(frame) = self.seed.take() {
            self.last = Some(frame.clone());
            return Ok(frame);
        }
        match tokio::time::timeout(self.period, capture.next_frame()).await {
            Ok(Ok(frame)) => {
                self.last = Some(frame.clone());
                Ok(frame)
            }
            Ok(Err(e)) => Err(e),
            Err(_) => match self.last.as_mut() {
                Some(stored) => {
                    bump_timestamp(stored, self.period_us);
                    Ok(stored.clone())
                }
                None => capture.next_frame().await,
            },
        }
    }

    /// Keep `frame` as the one to repeat, without encoding it.
    fn remember(&mut self, frame: CapturedFrame) {
        self.seed = None;
        self.last = Some(frame);
    }

    /// Move the repeat frame to the current wall clock (capture
    /// timestamps are UNIX-epoch µs), so a pause does not resume with
    /// stale timestamps.
    fn retime(&mut self) {
        let now = now_us();
        for frame in self.seed.iter_mut().chain(self.last.iter_mut()) {
            frame.set_timestamp_us(now);
        }
    }
}

fn bump_timestamp(frame: &mut CapturedFrame, delta_us: u64) {
    frame.set_timestamp_us(frame.timestamp_us().saturating_add(delta_us));
}

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_micros()).unwrap_or(u64::MAX))
}

/// A running audio capture + encoder task.
struct AudioPipeline {
    codec: AudioCodec,
    task: JoinHandle<()>,
    rx: Option<mpsc::Receiver<AudioFrame>>,
}

/// Audio side of a stream. The codec follows the current session, so
/// the pipeline is rebuilt when a reconnect lands on a session that
/// wants another codec (Cast mirroring: Opus, HLS: AAC).
struct AudioStage {
    config: Option<AudioStreamConfig>,
    running: Option<AudioPipeline>,
    device_id: Uuid,
    event_tx: mpsc::Sender<ManagerEvent>,
}

impl AudioStage {
    fn new(
        config: Option<AudioStreamConfig>,
        device_id: Uuid,
        event_tx: mpsc::Sender<ManagerEvent>,
    ) -> Self {
        Self {
            config,
            running: None,
            device_id,
            event_tx,
        }
    }

    /// Make sure a pipeline producing `codec` is running (no-op when
    /// audio is off). Restarts it on a codec change or after it died.
    async fn sync_codec(&mut self, codec: AudioCodec) {
        let Some(base) = self.config.clone() else {
            return;
        };
        if let Some(running) = &self.running
            && running.codec == codec
            && running.rx.is_some()
        {
            return;
        }
        self.stop();
        if !audio_encoder_available(codec) {
            self.warn(format!(
                "no {codec:?} audio encoder in this build; streaming without audio"
            ))
            .await;
            return;
        }
        let config = AudioStreamConfig { codec, ..base };
        let (tx, rx) = mpsc::channel::<AudioFrame>(AUDIO_QUEUE_FRAMES);
        let event_tx = self.event_tx.clone();
        let device_id = self.device_id;
        let task = tokio::spawn(async move {
            match run_audio_pipeline(config, tx).await {
                Ok(()) => tracing::info!("audio pipeline exited"),
                Err(e) => {
                    tracing::error!(%e, "audio pipeline failed");
                    let _ = event_tx
                        .send(ManagerEvent::StreamWarning {
                            device_id,
                            message: format!("audio stopped: {e}"),
                        })
                        .await;
                }
            }
        });
        tracing::info!(?codec, "audio pipeline started");
        self.running = Some(AudioPipeline {
            codec,
            task,
            rx: Some(rx),
        });
    }

    async fn warn(&self, message: String) {
        tracing::warn!(device_id = %self.device_id, %message, "stream warning");
        let _ = self
            .event_tx
            .send(ManagerEvent::StreamWarning {
                device_id: self.device_id,
                message,
            })
            .await;
    }

    /// Next encoded frame; pending forever when no pipeline runs.
    /// Cancel-safe.
    async fn recv(&mut self) -> Option<AudioFrame> {
        match self.running.as_mut().and_then(|p| p.rx.as_mut()) {
            Some(rx) => rx.recv().await,
            None => std::future::pending().await,
        }
    }

    fn try_recv(&mut self) -> Option<AudioFrame> {
        let rx = self.running.as_mut()?.rx.as_mut()?;
        match rx.try_recv() {
            Ok(frame) => Some(frame),
            Err(mpsc::error::TryRecvError::Empty) => None,
            Err(mpsc::error::TryRecvError::Disconnected) => {
                self.mark_closed();
                None
            }
        }
    }

    /// The pipeline's channel closed (the task exited).
    fn mark_closed(&mut self) {
        if let Some(p) = self.running.as_mut() {
            p.rx = None;
        }
    }

    /// Drop everything queued, e.g. audio captured while the receiver
    /// was reconnecting.
    fn discard_queued(&mut self) -> usize {
        let mut dropped = 0;
        while self.try_recv().is_some() {
            dropped += 1;
        }
        dropped
    }

    fn stop(&mut self) {
        if let Some(p) = self.running.take() {
            p.task.abort();
        }
    }
}

fn audio_encoder_available(codec: AudioCodec) -> bool {
    match codec {
        AudioCodec::Aac => true,
        AudioCodec::Opus => cfg!(feature = "opus-rs"),
        AudioCodec::AacEld => cfg!(feature = "fdk-aac"),
        AudioCodec::Pcm | AudioCodec::Alac => false,
    }
}

/// Why one session's frame loop returned.
enum PumpExit {
    /// The stream is over.
    End(StreamEnd),
    /// The session dropped; reconnect.
    Lost(String),
}

/// A running stream. Owns capture, encoder and the current session.
pub(crate) struct StreamTask<C, E> {
    device: Device,
    capture: C,
    encoder: E,
    encoder_config: EncoderConfig,
    applied_kbps: u32,
    session_config: StreamConfig,
    adaptive: Arc<AdaptiveBitrateState>,
    create_session: SessionFactory,
    event_tx: mpsc::Sender<ManagerEvent>,
    audio: AudioStage,
    frames: FramePacer,
    source_size: Option<FrameSize>,
    reconnects: ReconnectBudget,
    initial_session: Option<Box<dyn ErasedSession>>,
}

impl<C, E> StreamTask<C, E>
where
    C: ScreenCapture + 'static,
    E: VideoEncoder + 'static,
{
    /// Drive the stream until it is cancelled, the receiver stops it
    /// or it fails for good. Capture and audio are torn down before
    /// returning.
    pub(crate) async fn run(mut self, mut cancel: CancelRx) -> StreamEnd {
        let _ = self
            .event_tx
            .send(ManagerEvent::StreamStarted {
                device_id: self.device.id,
                device_name: self.device.name.clone(),
            })
            .await;
        let end = self.supervise(&mut cancel).await;
        tracing::info!(device = %self.device.name, ?end, "stream ended");
        let _ = self.capture.stop().await;
        self.audio.stop();
        end
    }

    async fn supervise(&mut self, cancel: &mut CancelRx) -> StreamEnd {
        let mut next = self.initial_session.take();
        let mut lost_reason = String::new();
        loop {
            let mut session = match next.take() {
                Some(s) => s,
                None => match self.reconnect(cancel, &lost_reason).await {
                    Ok(s) => s,
                    Err(end) => return end,
                },
            };
            self.on_session_ready(session.as_ref()).await;
            match self.pump(&mut session, cancel).await {
                PumpExit::End(end) => return end,
                PumpExit::Lost(reason) => {
                    tracing::warn!(device = %self.device.name, %reason, "receiver session lost");
                    let feeds_adaptive = session.feeds_adaptive_bitrate();
                    if let Err(FerricastError::RemoteStopped(why)) = session.stop().await {
                        return StreamEnd::RemoteStopped(why);
                    }
                    if feeds_adaptive {
                        // The old rate didn't keep the receiver up;
                        // restart low, the session raises it again.
                        self.adaptive.drop_to_floor();
                    }
                    lost_reason = reason;
                }
            }
        }
    }

    /// Align audio codec, bitrate and keyframes with a freshly set-up
    /// session.
    async fn on_session_ready(&mut self, session: &dyn ErasedSession) {
        self.audio.sync_codec(session.preferred_audio_codec()).await;
        let stale = self.audio.discard_queued();
        if stale > 0 {
            tracing::debug!(stale, "dropped audio queued before the session was ready");
        }
        self.apply_bitrate(session.feeds_adaptive_bitrate());
        self.discard_queued_video();
        self.encoder.request_keyframe();
    }

    /// Keeps only the newest frame captured while the session was being
    /// set up (pairing can take minutes), stamped now: sent with their
    /// capture times, those frames would start the receiver's timeline
    /// in the past.
    fn discard_queued_video(&mut self) {
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        let mut stale = 0_usize;
        while stale < MAX_STALE_FRAMES {
            // Polled once and dropped, as in `idle`: reading a frame is
            // cancellation-safe.
            let std::task::Poll::Ready(Ok(frame)) =
                std::pin::pin!(self.capture.next_frame()).poll(&mut cx)
            else {
                break;
            };
            self.frames.remember(frame);
            stale += 1;
        }
        self.frames.retime();
        if stale > 0 {
            tracing::debug!(stale, "dropped video captured before the session was ready");
        }
    }

    /// Rebuild the session after it dropped, with backoff. Capture and
    /// audio keep draining meanwhile so nothing stale is sent after.
    async fn reconnect(
        &mut self,
        cancel: &mut CancelRx,
        lost_reason: &str,
    ) -> std::result::Result<Box<dyn ErasedSession>, StreamEnd> {
        let mut reason = lost_reason.to_owned();
        loop {
            let Some(attempt) = self.reconnects.next_attempt() else {
                return Err(StreamEnd::Failed(format!(
                    "receiver connection lost ({reason}); gave up after \
                     {MAX_CONSECUTIVE_FAILURES} reconnect attempts"
                )));
            };
            let backoff = ReconnectBudget::backoff(attempt);
            tracing::warn!(device = %self.device.name, attempt, ?backoff, %reason, "reconnecting");
            let _ = self
                .event_tx
                .send(ManagerEvent::StreamReconnecting {
                    device_id: self.device.id,
                    attempt,
                    reason: reason.clone(),
                })
                .await;
            if !self.idle(cancel, backoff).await {
                return Err(StreamEnd::Cancelled);
            }

            let opened = tokio::select! {
                biased;
                () = cancel.cancelled() => return Err(StreamEnd::Cancelled),
                r = open_session(
                    &self.create_session,
                    &self.device,
                    &mut self.session_config,
                    &self.event_tx,
                ) => r,
            };
            match opened {
                Ok(session) => {
                    tracing::info!(device = %self.device.name, attempt, "session re-established");
                    self.reconnects.succeeded();
                    self.frames.retime();
                    return Ok(session);
                }
                Err(FerricastError::RemoteStopped(why)) => {
                    return Err(StreamEnd::RemoteStopped(why));
                }
                Err(e) => reason = e.to_string(),
            }
        }
    }

    /// Wait `period` while draining capture and audio. `false` when
    /// cancelled.
    async fn idle(&mut self, cancel: &mut CancelRx, period: Duration) -> bool {
        let deadline = tokio::time::sleep(period);
        tokio::pin!(deadline);
        let mut capture_ok = true;
        loop {
            tokio::select! {
                biased;
                () = cancel.cancelled() => return false,
                () = &mut deadline => return true,
                frame = self.audio.recv() => {
                    if frame.is_none() {
                        self.audio.mark_closed();
                    }
                }
                frame = self.capture.next_frame(), if capture_ok => match frame {
                    Ok(frame) => self.frames.remember(frame),
                    // Surfaces again (and ends the stream) once the
                    // frame loop resumes.
                    Err(_) => capture_ok = false,
                },
            }
        }
    }

    /// Frame loop for one session.
    async fn pump(
        &mut self,
        session: &mut Box<dyn ErasedSession>,
        cancel: &mut CancelRx,
    ) -> PumpExit {
        let keyframe_every = session.forced_keyframe_interval();
        let mut next_keyframe_at = Instant::now();
        let mut encode_errors = 0u32;
        loop {
            tokio::select! {
                // Audio first so slow software encodes can't starve
                // the audio queue.
                biased;
                () = cancel.cancelled() => {
                    let _ = session.stop().await;
                    return PumpExit::End(StreamEnd::Cancelled);
                }
                frame = self.audio.recv() => match frame {
                    Some(frame) => {
                        if let Some(exit) = self.send_audio(session, frame).await {
                            return exit;
                        }
                    }
                    None => self.audio.mark_closed(),
                },
                frame = self.frames.next(&mut self.capture) => {
                    let raw = match frame {
                        Ok(raw) => raw,
                        Err(e) => {
                            let _ = session.stop().await;
                            return PumpExit::End(StreamEnd::Failed(format!("capture: {e}")));
                        }
                    };
                    self.note_source_size(&raw);
                    if let Some(every) = keyframe_every {
                        let now = Instant::now();
                        if now >= next_keyframe_at {
                            self.encoder.request_keyframe();
                            while next_keyframe_at <= now {
                                next_keyframe_at += every;
                            }
                        }
                    }
                    self.apply_bitrate(session.feeds_adaptive_bitrate());

                    let encoded = match self.encoder.encode(raw) {
                        Ok(encoded) => {
                            encode_errors = 0;
                            encoded
                        }
                        Err(e) => {
                            encode_errors += 1;
                            tracing::warn!(%e, encode_errors, "encode failed");
                            if encode_errors >= MAX_CONSECUTIVE_ENCODE_ERRORS {
                                let _ = session.stop().await;
                                return PumpExit::End(StreamEnd::Failed(format!("encoder: {e}")));
                            }
                            self.recover_encoder();
                            continue;
                        }
                    };
                    match session.send_frame(&encoded).await {
                        Ok(()) => {}
                        Err(FerricastError::RemoteStopped(why)) => {
                            let _ = session.stop().await;
                            return PumpExit::End(StreamEnd::RemoteStopped(why));
                        }
                        Err(e) => return PumpExit::Lost(format!("send_frame: {e}")),
                    }
                    // Receiver feedback (RTCP PLI): it lost the
                    // picture and needs an IDR to resync.
                    if session.poll_keyframe_request() {
                        self.encoder.request_keyframe();
                    }
                    // The receiver app can die while the local side
                    // keeps accepting frames (HLS serves from here).
                    if !session.is_alive() {
                        return match session.end_reason() {
                            Some(FerricastError::RemoteStopped(why)) => {
                                let _ = session.stop().await;
                                PumpExit::End(StreamEnd::RemoteStopped(why))
                            }
                            Some(e) => PumpExit::Lost(e.to_string()),
                            None => PumpExit::Lost("receiver session ended".into()),
                        };
                    }
                }
            }
        }
    }

    /// Send one audio frame plus whatever queued behind it; old frames
    /// are dropped.
    async fn send_audio(
        &mut self,
        session: &mut Box<dyn ErasedSession>,
        first: AudioFrame,
    ) -> Option<PumpExit> {
        let mut next = Some(first);
        while let Some(frame) = next {
            if now_us().saturating_sub(frame.timestamp_us) > MAX_AUDIO_AGE_US {
                tracing::trace!(ts = frame.timestamp_us, "dropping stale audio frame");
            } else {
                match session.send_audio_frame(&frame).await {
                    Ok(()) => {}
                    Err(FerricastError::RemoteStopped(why)) => {
                        let _ = session.stop().await;
                        return Some(PumpExit::End(StreamEnd::RemoteStopped(why)));
                    }
                    // Video is the primary content; audio errors don't
                    // end the stream.
                    Err(e) => tracing::warn!(%e, "send_audio_frame failed"),
                }
            }
            next = self.audio.try_recv();
        }
        None
    }

    /// Push the bitrate the current session should run at into the
    /// encoder. Sessions that don't report link quality run at the
    /// negotiated ceiling: nothing would ever raise a lowered target.
    fn apply_bitrate(&mut self, feeds_adaptive: bool) {
        let want = if feeds_adaptive {
            self.adaptive.target_kbps()
        } else {
            self.adaptive.ceiling_kbps
        };
        if want == self.applied_kbps {
            return;
        }
        match self.encoder.set_bitrate_kbps(want) {
            Ok(()) => tracing::info!(
                from_kbps = self.applied_kbps,
                to_kbps = want,
                "encoder bitrate updated"
            ),
            Err(e) => tracing::warn!(%e, want, "set_bitrate_kbps failed"),
        }
        self.applied_kbps = want;
    }

    /// A new source size (window resize, mode change, scaling toggle)
    /// is letterboxed into the fixed output by the encoder; start the
    /// new picture on an IDR.
    fn note_source_size(&mut self, frame: &CapturedFrame) {
        let size = FrameSize::new(frame.width(), frame.height());
        if let Some(previous) = self.source_size
            && previous != size
        {
            tracing::info!(
                from_width = previous.width,
                from_height = previous.height,
                to_width = size.width,
                to_height = size.height,
                "capture source size changed"
            );
            self.encoder.request_keyframe();
        }
        self.source_size = Some(size);
    }

    /// Reopen the encoder session after an encode failure, at the
    /// bitrate currently in effect.
    fn recover_encoder(&mut self) {
        self.encoder_config.bitrate_kbps = self.applied_kbps;
        // A new session even though the configuration did not change: the
        // old one is in a bad state (a lost GPU, a stuck hardware encoder).
        if let Err(e) = self.encoder.reset(&self.encoder_config) {
            tracing::warn!(%e, "encoder reset failed");
        }
        self.encoder.request_keyframe();
    }
}

/// Capture system audio and encode it with `config.codec`.
async fn run_audio_pipeline(config: AudioStreamConfig, tx: mpsc::Sender<AudioFrame>) -> Result<()> {
    match config.codec {
        #[cfg(feature = "opus-rs")]
        AudioCodec::Opus => run_audio_pipeline_with(OpusEncoder::new(), config, tx).await,
        AudioCodec::Aac => run_audio_pipeline_with(AacEncoder::new(), config, tx).await,
        // AirPlay decodes AAC-ELD at 44.1 kHz only; asking PipeWire for that
        // rate lets its resampler do the conversion.
        #[cfg(feature = "fdk-aac")]
        AudioCodec::AacEld => {
            let config = AudioStreamConfig {
                sample_rate: ELD_SAMPLE_RATE,
                ..config
            };
            run_audio_pipeline_with(FdkEldEncoder::new(), config, tx).await
        }
        other => Err(FerricastError::Encoder(format!(
            "no {other:?} audio encoder in this build"
        ))),
    }
}

/// System audio capture → encoder, until the receiving side goes
/// away. The encoder is configured on the first PCM chunk with the
/// format PipeWire actually negotiated.
async fn run_audio_pipeline_with<A: AudioEncoder>(
    mut encoder: A,
    config: AudioStreamConfig,
    tx: mpsc::Sender<AudioFrame>,
) -> Result<()> {
    let mut capture = NativeAudioCapture::new();
    let cap_cfg = AudioCaptureConfig {
        sample_rate: config.sample_rate,
        channels: config.channels,
    };
    capture
        .start(AudioSource::DefaultMonitor, cap_cfg, config.mute.clone())
        .await?;
    let result = pump_audio(&mut encoder, &mut capture, &config, &tx).await;
    let _ = capture.stop().await;
    result
}

async fn pump_audio<A: AudioEncoder>(
    encoder: &mut A,
    capture: &mut NativeAudioCapture,
    config: &AudioStreamConfig,
    tx: &mpsc::Sender<AudioFrame>,
) -> Result<()> {
    let mut configured = false;
    loop {
        let pcm = capture.next_frame().await?;
        if !configured {
            encoder.configure(&AudioEncoderConfig {
                codec: config.codec,
                sample_rate: pcm.sample_rate,
                channels: pcm.channels,
                bitrate_kbps: config.bitrate_kbps,
                backend: ferricast_core::BackendSelection::default(),
            })?;
            tracing::info!(
                sample_rate = pcm.sample_rate,
                channels = pcm.channels,
                bitrate_kbps = config.bitrate_kbps,
                codec = ?config.codec,
                "audio encoder configured"
            );
            configured = true;
        }
        if let Err(e) = encoder.encode(&pcm) {
            tracing::warn!(%e, "audio encode failed; dropping chunk");
            continue;
        }
        for frame in encoder.take_output() {
            if tx.send(frame).await.is_err() {
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use ferricast_core::DeviceCapabilities;

    use super::*;
    use crate::test_support::{
        EncoderLog, FakeCapture, FakeEncoder, SessionScript, device, session_factory,
    };

    struct Run {
        end: StreamEnd,
        events: Vec<ManagerEvent>,
        encoder: Arc<Mutex<EncoderLog>>,
        sessions: Arc<Mutex<crate::test_support::SessionLog>>,
    }

    async fn run_stream(
        caps: DeviceCapabilities,
        sizes: &[(u32, u32)],
        scripts: Vec<SessionScript>,
        config: StreamConfig,
    ) -> Run {
        let (event_tx, mut event_rx) = mpsc::channel(1024);
        let (factory, sessions) = session_factory(scripts);
        let encoder = FakeEncoder::default();
        let encoder_log = encoder.0.clone();
        let request = StreamRequest {
            device: device(caps),
            source: CaptureSource::FullScreen { monitor: None },
            capture: FakeCapture::new(sizes),
            encoder,
            config,
            gpu: GpuPreference::Auto,
            create_session: factory,
            event_tx,
        };
        let task = request.prepare().await.expect("prepare");
        let (_cancel_tx, cancel_rx) = mpsc::channel(1);
        let end = tokio::time::timeout(Duration::from_mins(10), task.run(CancelRx::new(cancel_rx)))
            .await
            .expect("stream ends");
        let mut events = Vec::new();
        while let Ok(e) = event_rx.try_recv() {
            events.push(e);
        }
        Run {
            end,
            events,
            encoder: encoder_log,
            sessions,
        }
    }

    /// Run `prepare` against a receiver whose control port is `port`
    /// on loopback.
    async fn prepare_with_port(port: u16) -> Result<StreamTask<FakeCapture, FakeEncoder>> {
        let (event_tx, _event_rx) = mpsc::channel(1024);
        let (factory, _sessions) = session_factory(vec![stop_after(1)]);
        let mut device = device(DeviceCapabilities::default());
        device.addresses = std::net::SocketAddr::from(([127, 0, 0, 1], port)).into();
        StreamRequest {
            device,
            source: CaptureSource::FullScreen { monitor: None },
            capture: FakeCapture::new(&[(64, 36)]),
            encoder: FakeEncoder::default(),
            config: StreamConfig::default(),
            gpu: GpuPreference::Auto,
            create_session: factory,
            event_tx,
        }
        .prepare()
        .await
    }

    #[tokio::test]
    async fn the_first_frame_is_stamped_when_the_session_is_ready() {
        // The fake capture stamps its frames in 2023: what was captured
        // before the session was set up must not go out with that time.
        let before = now_us();
        let run = run_stream(
            DeviceCapabilities::default(),
            &[(64, 36)],
            vec![stop_after(1)],
            StreamConfig::default(),
        )
        .await;
        let first = run.sessions.lock().unwrap().first_frame_us.unwrap();
        assert!(
            first >= before,
            "first frame stamped {first}, start {before}"
        );
    }

    #[tokio::test]
    async fn unreachable_device_fails_the_start_quickly() {
        // Bind then drop to get a loopback port nothing listens on.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let started = std::time::Instant::now();
        let Err(err) = prepare_with_port(port).await else {
            panic!("prepare succeeded against a closed port");
        };
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(
            err.to_string()
                .contains(&format!("device unreachable: 127.0.0.1:{port}")),
            "{err}"
        );
    }

    #[tokio::test]
    async fn reachable_device_starts_normally() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(prepare_with_port(port).await.is_ok());
    }

    fn stop_after(n: usize) -> SessionScript {
        SessionScript {
            remote_stop_after: Some(n),
            ..SessionScript::default()
        }
    }

    #[tokio::test(start_paused = true)]
    async fn output_size_is_fitted_to_device_caps_and_passed_to_the_session() {
        let caps = DeviceCapabilities {
            max_width: Some(32),
            max_height: Some(32),
            ..DeviceCapabilities::default()
        };
        let run = run_stream(
            caps,
            &[(64, 36)],
            vec![stop_after(3)],
            StreamConfig::default(),
        )
        .await;
        let enc = run.encoder.lock().unwrap();
        let cfg = &enc.configured[0];
        assert_eq!((cfg.width, cfg.height), (32, 18));
        let sessions = run.sessions.lock().unwrap();
        assert_eq!(
            (sessions.setups[0].width, sessions.setups[0].height),
            (32, 18)
        );
        assert_eq!(sessions.setups[0].fps, 30);
    }

    #[tokio::test(start_paused = true)]
    async fn source_resize_requests_an_idr_without_reconfiguring() {
        let sizes = [(64, 36), (64, 36), (64, 36), (48, 48), (48, 48)];
        let run = run_stream(
            DeviceCapabilities::default(),
            &sizes,
            vec![stop_after(5)],
            StreamConfig::default(),
        )
        .await;
        let enc = run.encoder.lock().unwrap();
        assert_eq!(enc.configured.len(), 1);
        assert!(enc.frame_sizes.contains(&(48, 48)));
        // One IDR when the session came up, one for the new size.
        assert_eq!(enc.keyframe_requests, 2);
    }

    #[tokio::test(start_paused = true)]
    async fn lost_session_reconnects_then_remote_stop_ends_without_error() {
        let scripts = vec![
            SessionScript {
                dies_after: Some(3),
                ..SessionScript::default()
            },
            stop_after(4),
        ];
        let run = run_stream(
            DeviceCapabilities::default(),
            &[(64, 36)],
            scripts,
            StreamConfig::default(),
        )
        .await;
        assert_eq!(
            run.end,
            StreamEnd::RemoteStopped("stopped on the TV".into())
        );
        let sessions = run.sessions.lock().unwrap();
        assert_eq!(sessions.created, 2);
        assert_eq!(sessions.stops, 2);
        assert!(
            run.events
                .iter()
                .any(|e| matches!(e, ManagerEvent::StreamReconnecting { attempt: 1, .. }))
        );
        assert!(
            !run.events
                .iter()
                .any(|e| matches!(e, ManagerEvent::StreamError { .. }))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn session_ended_by_the_receiver_is_not_reconnected() {
        let scripts = vec![SessionScript {
            dies_after: Some(3),
            dies_remotely: true,
            ..SessionScript::default()
        }];
        let run = run_stream(
            DeviceCapabilities::default(),
            &[(64, 36)],
            scripts,
            StreamConfig::default(),
        )
        .await;
        assert_eq!(
            run.end,
            StreamEnd::RemoteStopped("another app took over".into())
        );
        assert_eq!(run.sessions.lock().unwrap().created, 1);
        assert!(
            !run.events
                .iter()
                .any(|e| matches!(e, ManagerEvent::StreamReconnecting { .. }))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn reconnect_gives_up_after_the_budget() {
        let scripts = vec![
            SessionScript {
                dies_after: Some(1),
                ..SessionScript::default()
            },
            SessionScript {
                connect_fails: true,
                ..SessionScript::default()
            },
        ];
        let run = run_stream(
            DeviceCapabilities::default(),
            &[(64, 36)],
            scripts,
            StreamConfig::default(),
        )
        .await;
        assert!(matches!(run.end, StreamEnd::Failed(_)));
        let attempts = run
            .events
            .iter()
            .filter(|e| matches!(e, ManagerEvent::StreamReconnecting { .. }))
            .count();
        assert_eq!(attempts, MAX_CONSECUTIVE_FAILURES as usize);
    }

    #[tokio::test(start_paused = true)]
    async fn adaptive_sessions_restart_at_the_floor() {
        let config = StreamConfig {
            bitrate_kbps: 4000,
            ..StreamConfig::default()
        };
        let scripts = vec![
            SessionScript {
                dies_after: Some(2),
                feeds_adaptive: true,
                ..SessionScript::default()
            },
            SessionScript {
                remote_stop_after: Some(2),
                feeds_adaptive: true,
                ..SessionScript::default()
            },
        ];
        let run = run_stream(DeviceCapabilities::default(), &[(64, 36)], scripts, config).await;
        let enc = run.encoder.lock().unwrap();
        assert_eq!(enc.bitrates, vec![AdaptiveBitrateState::DEFAULT_FLOOR_KBPS]);
    }

    #[tokio::test(start_paused = true)]
    async fn non_adaptive_sessions_keep_the_negotiated_bitrate() {
        let config = StreamConfig {
            bitrate_kbps: 4000,
            ..StreamConfig::default()
        };
        let scripts = vec![
            SessionScript {
                dies_after: Some(2),
                feeds_adaptive: true,
                ..SessionScript::default()
            },
            // Mirror-like: nothing would raise a floored target.
            SessionScript {
                remote_stop_after: Some(2),
                feeds_adaptive: false,
                ..SessionScript::default()
            },
        ];
        let run = run_stream(DeviceCapabilities::default(), &[(64, 36)], scripts, config).await;
        let enc = run.encoder.lock().unwrap();
        assert!(
            enc.bitrates.is_empty(),
            "bitrate changed: {:?}",
            enc.bitrates
        );
    }

    #[tokio::test(start_paused = true)]
    async fn forced_keyframes_follow_the_session_cadence() {
        let every = SessionScript {
            remote_stop_after: Some(60),
            forced_keyframes: Some(Duration::from_millis(100)),
            ..SessionScript::default()
        };
        let never = SessionScript {
            forced_keyframes: None,
            ..every.clone()
        };
        let with = run_stream(
            DeviceCapabilities::default(),
            &[(64, 36)],
            vec![every],
            StreamConfig::default(),
        )
        .await;
        let without = run_stream(
            DeviceCapabilities::default(),
            &[(64, 36)],
            vec![never],
            StreamConfig::default(),
        )
        .await;
        let with = with.encoder.lock().unwrap().keyframe_requests;
        let without = without.encoder.lock().unwrap().keyframe_requests;
        assert_eq!(without, 1);
        assert!(with > 2, "expected periodic IDRs, got {with}");
    }

    #[tokio::test(start_paused = true)]
    async fn audio_codec_follows_each_session() {
        let config = StreamConfig {
            audio: Some(AudioStreamConfig::default()),
            ..StreamConfig::default()
        };
        let scripts = vec![
            SessionScript {
                dies_after: Some(2),
                audio_codec: AudioCodec::Opus,
                ..SessionScript::default()
            },
            SessionScript {
                remote_stop_after: Some(2),
                audio_codec: AudioCodec::Aac,
                ..SessionScript::default()
            },
        ];
        let run = run_stream(DeviceCapabilities::default(), &[(64, 36)], scripts, config).await;
        let sessions = run.sessions.lock().unwrap();
        let codecs: Vec<_> = sessions
            .setups
            .iter()
            .map(|c| c.audio.as_ref().map(|a| a.codec))
            .collect();
        assert_eq!(codecs, vec![Some(AudioCodec::Opus), Some(AudioCodec::Aac)]);
    }

    #[tokio::test]
    async fn audio_stage_restarts_only_on_codec_change() {
        let (event_tx, _event_rx) = mpsc::channel(16);
        let mut stage =
            AudioStage::new(Some(AudioStreamConfig::default()), Uuid::new_v4(), event_tx);
        stage.sync_codec(AudioCodec::Aac).await;
        let first = stage.running.as_ref().map(|p| p.task.id());
        stage.sync_codec(AudioCodec::Aac).await;
        assert_eq!(stage.running.as_ref().map(|p| p.task.id()), first);
        stage.sync_codec(AudioCodec::Opus).await;
        let after = stage.running.as_ref().map(|p| (p.codec, p.task.id()));
        if cfg!(feature = "opus-rs") {
            assert_eq!(after.map(|(c, _)| c), Some(AudioCodec::Opus));
            assert_ne!(after.map(|(_, id)| id), first);
        } else {
            assert!(after.is_none());
        }
        stage.stop();
    }

    #[tokio::test]
    async fn audio_stage_warns_when_no_encoder_exists() {
        let (event_tx, mut event_rx) = mpsc::channel(16);
        let mut stage =
            AudioStage::new(Some(AudioStreamConfig::default()), Uuid::new_v4(), event_tx);
        stage.sync_codec(AudioCodec::Alac).await;
        assert!(stage.running.is_none());
        assert!(matches!(
            event_rx.try_recv(),
            Ok(ManagerEvent::StreamWarning { .. })
        ));
    }

    #[test]
    fn reconnect_budget_backs_off_and_runs_out() {
        let mut budget = ReconnectBudget::default();
        let attempts: Vec<_> = std::iter::from_fn(|| budget.next_attempt()).collect();
        assert_eq!(attempts, vec![1, 2, 3, 4, 5]);
        assert_eq!(ReconnectBudget::backoff(1), Duration::from_secs(1));
        assert_eq!(ReconnectBudget::backoff(5), Duration::from_secs(16));
        assert_eq!(ReconnectBudget::backoff(9), Duration::from_secs(16));
    }

    /// Open a session against `script`, answering password prompts with
    /// `answers` in order. Returns the outcome, the `rejected` flag of
    /// every prompt and the session log.
    async fn open_with_answers(
        script: SessionScript,
        answers: &[&str],
    ) -> (Result<()>, Vec<bool>, crate::test_support::SessionLog) {
        let (event_tx, mut event_rx) = mpsc::channel(16);
        let (factory, sessions) = session_factory(vec![script]);
        let answers: Vec<String> = answers.iter().map(|a| (*a).to_owned()).collect();
        let answerer = tokio::spawn(async move {
            let mut prompts = Vec::new();
            while let Some(event) = event_rx.recv().await {
                let ManagerEvent::PairingRequired {
                    challenge, respond, ..
                } = event
                else {
                    continue;
                };
                let PairingChallenge::Credential { rejected } = challenge else {
                    panic!("unexpected challenge {challenge:?}");
                };
                let answer = answers
                    .get(prompts.len())
                    .cloned()
                    .map_or(PairingResponse::Cancelled, PairingResponse::Password);
                prompts.push(rejected);
                if let Some(tx) = respond.lock().await.take() {
                    let _ = tx.send(answer);
                }
            }
            prompts
        });
        let mut config = StreamConfig::default();
        let result = open_session(
            &factory,
            &device(DeviceCapabilities::default()),
            &mut config,
            &event_tx,
        )
        .await
        .map(drop);
        drop(event_tx);
        let prompts = answerer.await.expect("answerer");
        let log = std::mem::take(&mut *sessions.lock().unwrap());
        (result, prompts, log)
    }

    fn password_protected() -> SessionScript {
        SessionScript {
            password: Some("secret"),
            ..SessionScript::default()
        }
    }

    #[tokio::test]
    async fn password_prompt_retries_setup_on_the_same_session() {
        let (result, prompts, log) = open_with_answers(password_protected(), &["secret"]).await;
        result.expect("stream set up");
        assert_eq!(prompts, [false]);
        assert_eq!(log.created, 1);
        assert_eq!(log.setups.len(), 1);
        assert_eq!(log.passwords, ["secret"]);
    }

    #[tokio::test]
    async fn wrong_password_is_asked_again() {
        let (result, prompts, log) =
            open_with_answers(password_protected(), &["wrong", "secret"]).await;
        result.expect("stream set up");
        assert_eq!(prompts, [false, true]);
        assert_eq!(log.created, 1);
        assert_eq!(log.passwords, ["wrong", "secret"]);
    }

    #[tokio::test]
    async fn three_wrong_passwords_give_up() {
        let (result, prompts, log) =
            open_with_answers(password_protected(), &["a", "b", "c", "secret"]).await;
        let err = result.expect_err("gives up");
        assert!(
            err.to_string().contains("rejected the password 3 times"),
            "{err}"
        );
        assert_eq!(prompts, [false, true, true]);
        assert_eq!(log.stops, 1);
        assert!(log.setups.is_empty());
    }

    #[tokio::test]
    async fn cancelled_password_prompt_releases_the_receiver() {
        let (result, prompts, log) = open_with_answers(password_protected(), &[]).await;
        assert!(result.is_err());
        assert_eq!(prompts, [false]);
        assert_eq!(log.stops, 1);
    }
}
