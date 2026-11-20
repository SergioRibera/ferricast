use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, RwLock, mpsc};
use tokio::task::JoinHandle;
use uuid::Uuid;

use ferricast_capture::PipeWireAudioCapture;
use ferricast_core::{
    AudioCapture, AudioCaptureConfig, AudioCodec, AudioDecoder, AudioEncoder, AudioEncoderConfig,
    AudioFrame, AudioSource, CaptureConfig, CaptureSource, CapturedFrame, CastSession,
    ConnectOutcome, Device, Discovery, DiscoveryEvent, EncodedFrame, EncoderConfig, FerricastError,
    FrameSink, MediaInfo, PairingChallenge, PairingResponse, PlaybackState, ProtocolHandler,
    ReceiverProtocol, RemoteSender, Result, ScreenCapture, StreamConfig, VideoDecoder,
    VideoEncoder,
};
use ferricast_encoder::aac::AacEncoder;

use crate::receiver::ReceiverHost;
#[cfg(feature = "opus-rs")]
use ferricast_encoder::opus::OpusEncoder;

type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

trait ErasedDiscovery: Send + Sync {
    fn start(&mut self, tx: mpsc::Sender<DiscoveryEvent>) -> BoxFut<'_, Result<()>>;
    fn stop(&mut self) -> BoxFut<'_, Result<()>>;
}

impl<T: Discovery> ErasedDiscovery for T {
    fn start(&mut self, tx: mpsc::Sender<DiscoveryEvent>) -> BoxFut<'_, Result<()>> {
        Box::pin(Discovery::start(self, tx))
    }
    fn stop(&mut self) -> BoxFut<'_, Result<()>> {
        Box::pin(Discovery::stop(self))
    }
}

trait ErasedSession: Send + Sync {
    fn connect<'a>(&'a mut self, device: &'a Device) -> BoxFut<'a, Result<ConnectOutcome>>;
    fn submit_pairing(&mut self, response: PairingResponse) -> BoxFut<'_, Result<()>>;
    fn setup_stream<'a>(&'a mut self, config: &'a StreamConfig) -> BoxFut<'a, Result<()>>;
    fn send_frame<'a>(&'a mut self, frame: &'a EncodedFrame) -> BoxFut<'a, Result<()>>;
    fn send_audio_frame<'a>(&'a mut self, frame: &'a AudioFrame) -> BoxFut<'a, Result<()>>;
    fn stop(&mut self) -> BoxFut<'_, Result<()>>;
    fn is_alive(&self) -> bool;
    fn poll_keyframe_request(&mut self) -> bool;
    fn preferred_audio_codec(&self) -> AudioCodec;
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
    fn poll_keyframe_request(&mut self) -> bool {
        CastSession::poll_keyframe_request(self)
    }
    fn preferred_audio_codec(&self) -> AudioCodec {
        CastSession::preferred_audio_codec(self)
    }
}

struct RegisteredProtocol {
    protocol: &'static str,
    create_discovery: Box<dyn Fn() -> Box<dyn ErasedDiscovery> + Send + Sync>,
    /// Wrapped in `Arc` instead of `Box` so the streaming task can
    /// hold its own clone — needed for auto-reconnect, which builds
    /// a fresh session on `is_alive()` going false (chromecast 301,
    /// receiver-app crash) without tearing down capture/encoder.
    /// Takes `&Device` so the factory can pick the right session type
    /// per-device (e.g. Cast Streaming vs. HLS on Chromecast).
    create_session: Arc<dyn Fn(&Device) -> Result<Box<dyn ErasedSession>> + Send + Sync>,
}

struct ActiveStream {
    device: Device,
    cancel_tx: mpsc::Sender<()>,
    /// Handle to the spawned streaming task. Kept so `stop_stream`
    /// can `await` until the task has actually torn down its
    /// capture/encoder/session — without this, `shutdown()` can
    /// return while PipeWire/CASTv2 cleanup is still racing the
    /// process exit and the OS reaps fds + threads abruptly.
    task: JoinHandle<()>,
}

#[derive(Debug, Clone)]
pub enum ManagerEvent {
    DeviceFound(Device),
    DeviceLost(Uuid),
    StreamStarted {
        device_id: Uuid,
        device_name: String,
    },
    StreamStopped {
        device_id: Uuid,
    },
    /// Fatal stream error — the stream task has terminated and won't
    /// recover on its own. Apps typically surface this to the user
    /// (toast, retry button) since at this point the manager has
    /// already exhausted its own auto-reconnect budget.
    StreamError {
        device_id: Uuid,
        message: String,
    },
    /// Non-fatal: the receiver session went away mid-stream (e.g.
    /// Chromecast `detailedErrorCode=301`, app exit), and the
    /// manager is attempting to recreate it. Capture/encoder keep
    /// running; the receiver will go dark for a few seconds.
    /// Apps can use this to show a "reconnecting…" indicator
    /// without tearing down their own state.
    StreamReconnecting {
        device_id: Uuid,
        attempt: u32,
        reason: String,
    },
    DiscoveryError {
        protocol: &'static str,
        message: String,
    },

    /// The protocol requires user input before streaming can begin
    /// (e.g. AirPlay PIN shown on the TV screen). The UI should show
    /// the appropriate widget and send the response by calling
    /// `respond.lock().await.take()` and sending through the inner
    /// sender. `PairingResponse::Cancelled` aborts the stream attempt.
    ///
    /// `respond` is wrapped in `Arc<Mutex<Option>>` so the event stays
    /// `Clone` (needed for the in-process forward channel), while still
    /// transferring the oneshot sender to exactly one consumer.
    PairingRequired {
        device_id: Uuid,
        challenge: PairingChallenge,
        respond: std::sync::Arc<
            tokio::sync::Mutex<Option<tokio::sync::oneshot::Sender<PairingResponse>>>,
        >,
    },

    // ── receiver-side events ──────────────────────────────────────
    /// A remote sender connected on a registered receiver protocol's
    /// control channel. Fired before any LOAD command arrives — at
    /// this point we know who connected but not what they want to
    /// play. UI can open a "Connecting…" placeholder.
    ReceiverIncoming {
        receiver_id: Uuid,
        protocol: &'static str,
        remote: RemoteSender,
    },
    /// The remote sender's first LOAD has been processed and the
    /// puller has resolved [`MediaInfo`]. UI uses this to decide
    /// between a video window and an audio-only card view
    /// (`info.video.is_none()` ⇒ audio only).
    ReceiverStarted {
        receiver_id: Uuid,
        remote: RemoteSender,
        info: MediaInfo,
    },
    /// Playback state changed (PLAY/PAUSE/STOP from remote, EOS, …).
    /// Mirrors what the manager reports back to the sender through
    /// [`ControlSession::report_state`] so the GUI stays in sync
    /// without polling.
    ReceiverStateChanged {
        receiver_id: Uuid,
        state: PlaybackState,
    },
    /// Remote disconnected cleanly or the puller hit EOS.
    ReceiverStopped {
        receiver_id: Uuid,
    },
    /// Fatal receiver error — pipeline task terminated. Distinct
    /// from `ReceiverStopped` so the UI can surface a toast / retry
    /// affordance.
    ReceiverError {
        receiver_id: Uuid,
        message: String,
    },
}

pub struct StreamManager {
    protocols: Vec<RegisteredProtocol>,
    devices: Arc<RwLock<HashMap<Uuid, Device>>>,
    active_streams: Arc<Mutex<HashMap<Uuid, ActiveStream>>>,
    discovery_handles: Vec<Box<dyn ErasedDiscovery>>,
    event_tx: mpsc::Sender<ManagerEvent>,
    event_rx: Option<mpsc::Receiver<ManagerEvent>>,
    running: bool,
    // ── receiver-side state ──────────────────────────────────────
    receiver: ReceiverHost,

    /// Port range applied to every new stream session. `None` = OS
    /// picks an ephemeral port (default). Set via
    /// `StreamManagerBuilder::with_port_range` so firewall rules can
    /// target a fixed range regardless of protocol.
    port_range: Option<(u16, u16)>,
}

impl Default for StreamManager {
    fn default() -> Self {
        let (event_tx, event_rx) = mpsc::channel(256);
        Self {
            protocols: Vec::new(),
            devices: Arc::new(RwLock::new(HashMap::new())),
            active_streams: Arc::new(Mutex::new(HashMap::new())),
            discovery_handles: Vec::new(),
            event_tx,
            event_rx: Some(event_rx),
            running: false,
            receiver: ReceiverHost::default(),
            port_range: None,
        }
    }
}

impl StreamManager {
    pub fn register<H>(&mut self)
    where
        H: ProtocolHandler + Clone + Default + 'static,
    {
        let h_sess = H::default();
        let h_disc = h_sess.clone();

        self.protocols.push(RegisteredProtocol {
            protocol: H::PROTOCOL,
            create_discovery: Box::new(move || {
                Box::new(h_disc.create_discovery()) as Box<dyn ErasedDiscovery>
            }),
            create_session: Arc::new(move |device: &Device| {
                let session = h_sess.create_session_for_device(device)?;
                Ok(Box::new(session) as Box<dyn ErasedSession>)
            }),
        });
    }

    pub fn take_event_rx(&mut self) -> Option<mpsc::Receiver<ManagerEvent>> {
        self.event_rx.take()
    }

    pub fn registered_protocols(&self) -> Vec<&'static str> {
        self.protocols.iter().map(|p| p.protocol).collect()
    }

    /// Register a receiver-side protocol (advertiser + control +
    /// puller). Mirror of [`Self::register`] for senders.
    pub fn register_receiver<P>(&mut self)
    where
        P: ReceiverProtocol + Clone + Default + 'static,
    {
        self.receiver.register_receiver::<P>();
    }

    /// Register a video decoder. Pipeline picks one per session by
    /// `D::CODEC`. Last registration for a given codec wins —
    /// register the preferred backend last (the builder's `with_*`
    /// helpers do this in NVDEC → VA-API → sw order, same priority
    /// the encoder uses).
    pub fn register_video_decoder<D>(&mut self)
    where
        D: VideoDecoder + Default + 'static,
    {
        self.receiver.register_video_decoder::<D>();
    }

    pub fn register_audio_decoder<D>(&mut self)
    where
        D: AudioDecoder + Default + 'static,
    {
        self.receiver.register_audio_decoder::<D>();
    }

    /// Install the callback the pipeline uses to ask the host
    /// application for a [`FrameSink`] whenever a new receiver
    /// session resolves its `MediaInfo`. The GUI typically opens a
    /// new window here (video) or a transport-control card (audio
    /// only) and returns the sink the window owns. Required before
    /// [`Self::start_receivers`] — without it the pipeline rejects
    /// inbound sessions because it has nowhere to put decoded
    /// frames.
    pub fn set_sink_factory<F, Fut>(&mut self, factory: F)
    where
        F: Fn(&RemoteSender, &MediaInfo) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Box<dyn FrameSink>>> + Send + 'static,
    {
        self.receiver.set_sink_factory(factory);
    }

    pub fn registered_receivers(&self) -> Vec<&'static str> {
        self.receiver.registered_receivers()
    }

    pub async fn start_discovery(&mut self) -> Result<()> {
        if self.running {
            return Ok(());
        }

        let (disc_tx, mut disc_rx) = mpsc::channel::<DiscoveryEvent>(256);

        for proto in &self.protocols {
            let mut discovery = (proto.create_discovery)();
            let tx = disc_tx.clone();
            match discovery.start(tx).await {
                Ok(()) => {
                    self.discovery_handles.push(discovery);
                }
                Err(e) => {
                    tracing::warn!(
                        protocol = proto.protocol,
                        %e,
                        "Failed to start discovery, skipping"
                    );
                    let _ = self
                        .event_tx
                        .send(ManagerEvent::DiscoveryError {
                            protocol: proto.protocol,
                            message: e.to_string(),
                        })
                        .await;
                }
            }
        }

        let devices = self.devices.clone();
        let event_tx = self.event_tx.clone();

        tokio::spawn(async move {
            while let Some(event) = disc_rx.recv().await {
                match event {
                    DiscoveryEvent::DeviceFound(device) => {
                        let id = device.id;
                        tracing::info!(name = %device.name, protocol = device.protocol, "Device found");
                        devices.write().await.insert(id, device.clone());
                        let _ = event_tx.send(ManagerEvent::DeviceFound(device)).await;
                    }
                    DiscoveryEvent::DeviceLost(id) => {
                        tracing::info!(?id, "Device lost");
                        devices.write().await.remove(&id);
                        let _ = event_tx.send(ManagerEvent::DeviceLost(id)).await;
                    }
                    DiscoveryEvent::Error { protocol, message } => {
                        tracing::warn!(protocol, %message, "Discovery error");
                        let _ = event_tx
                            .send(ManagerEvent::DiscoveryError { protocol, message })
                            .await;
                    }
                }
            }
        });

        self.running = true;
        Ok(())
    }

    pub async fn stop_discovery(&mut self) -> Result<()> {
        for discovery in &mut self.discovery_handles {
            discovery.stop().await?;
        }
        self.discovery_handles.clear();
        self.running = false;
        Ok(())
    }

    /// Start every registered receiver protocol's advertiser and
    /// per-protocol accept supervisor. Requires
    /// [`Self::set_sink_factory`] to have been called.
    pub async fn start_receivers(&mut self) -> Result<()> {
        self.receiver.start_receivers(&self.event_tx).await
    }

    pub async fn stop_receivers(&mut self) -> Result<()> {
        self.receiver.stop_receivers().await
    }

    pub async fn devices(&self) -> Vec<Device> {
        self.devices.read().await.values().cloned().collect()
    }

    pub async fn start_stream(
        &self,
        device_id: Uuid,
        source: CaptureSource,
        mut capture: impl ScreenCapture + 'static,
        mut encoder: impl VideoEncoder + 'static,
        mut config: StreamConfig,
    ) -> Result<()> {
        if config.port_range.is_none() {
            config.port_range = self.port_range;
        }
        let device = {
            let devices = self.devices.read().await;
            devices
                .get(&device_id)
                .cloned()
                .ok_or_else(|| FerricastError::DeviceNotFound(device_id.to_string()))?
        };

        let proto = self
            .protocols
            .iter()
            .find(|p| p.protocol == device.protocol)
            .ok_or_else(|| {
                FerricastError::Protocol(format!("No handler for {}", device.protocol))
            })?;

        // Apply the device's fps ceiling BEFORE start so the
        // PipeWire worker's internal throttle anchors to the right
        // rate from buffer 0 — otherwise the worker burns CPU on
        // Vulkan readbacks at the compositor's refresh (60 Hz
        // typical) for frames the encoder will never consume.
        let device_fps_cap = device.capabilities.max_fps;
        let initial_fps = match device_fps_cap {
            Some(cap) => config.fps.min(cap),
            None => config.fps,
        };

        // Start capture first so the portal picker shows before connecting.
        let capture_config = CaptureConfig {
            fps: initial_fps,
            width: Some(config.width),
            height: Some(config.height),
            show_cursor: true,
        };

        capture.start(source, capture_config).await?;

        // Block on the first frame: PipeWire's format negotiation and
        // buffer setup happen on its main-loop thread asynchronously,
        // so `capture.start().await` returning doesn't mean the
        // negotiated format is available yet. Pulling a frame is the
        // simplest barrier — backends only emit frames once the
        // format is acked, so by the time this resolves
        // `get_screen_size` / `get_framerate` are populated.
        //
        // Without this the encoder gets configured with (0, 0) and
        // x264 dies with "invalid width x height (0x0)" while
        // PipeWire tears its own stream down with "no more input
        // formats" — same root cause.
        tracing::info!(device_id = %device_id, "awaiting first frame from capture (30s ceiling)");
        let first_frame =
            match tokio::time::timeout(Duration::from_secs(30), capture.next_frame()).await {
                Ok(Ok(f)) => {
                    tracing::info!(
                        device_id = %device_id,
                        "first frame received from capture"
                    );
                    f
                }
                Ok(Err(e)) => {
                    tracing::error!(
                        device_id = %device_id,
                        %e,
                        "capture returned error before first frame"
                    );
                    return Err(e);
                }
                Err(_) => {
                    tracing::error!(
                        device_id = %device_id,
                        "capture.next_frame() did not return within 30s — \
                         the capture worker isn't delivering buffers \
                         (compositor idle, or worker stalled)"
                    );
                    return Err(FerricastError::Timeout(
                        "capture.next_frame() exceeded 30s".into(),
                    ));
                }
            };
        tracing::info!(device_id = %device_id, "Capture negotiated, connecting to device");

        let (width, height) = capture.get_screen_size();
        let pixel_format = capture.get_pixel_format();
        // Prefer the framerate the source actually delivers; fall
        // back to the user's hint if the backend doesn't report one
        // (X11 polling or a backend that hasn't negotiated yet).
        let negotiated_fps = capture.get_framerate();
        let mut effective_fps = if negotiated_fps > 0 {
            negotiated_fps
        } else {
            config.fps
        };
        let mut effective_bitrate_kbps = config.bitrate_kbps;
        let mut effective_h264_profile: Option<ferricast_core::H264Profile> = None;

        // Apply receiver-side hardware limits before we configure
        // the encoder. `DeviceCapabilities` is populated by the
        // protocol's discovery code (chromecast/discovery.rs reads
        // the `md` mDNS field and maps it to a known device class)
        // so this caps for the right reason on the right device:
        // 1st-gen Chromecast → 1080p@30 Main, Ultra → 4K@30 High,
        // Google TV → 1080p@60 High, etc.
        let caps = &device.capabilities;
        if let Some(max_fps) = caps.max_fps {
            if effective_fps > max_fps {
                tracing::warn!(
                    device = %device.name,
                    requested_fps = effective_fps,
                    max_fps,
                    "device cap: lowering encode fps to receiver's decoder ceiling"
                );
                effective_fps = max_fps;
            }
        }
        if let Some(max_bitrate) = caps.max_bitrate_kbps {
            if effective_bitrate_kbps > max_bitrate {
                tracing::warn!(
                    device = %device.name,
                    requested_kbps = effective_bitrate_kbps,
                    max_kbps = max_bitrate,
                    "device cap: lowering bitrate to receiver's decoder ceiling"
                );
                effective_bitrate_kbps = max_bitrate;
            }
        }
        if let Some(prof) = caps.max_h264_profile {
            effective_h264_profile = Some(prof);
        }

        tracing::info!(
            device = %device.name,
            model = %device.model.as_deref().unwrap_or(""),
            width,
            height,
            negotiated_fps,
            effective_fps,
            effective_bitrate_kbps,
            ?effective_h264_profile,
            requires_audio = caps.requires_audio,
            ?pixel_format,
            "encoder configured against device capabilities"
        );

        encoder.configure(&EncoderConfig {
            pixel_format,
            width: width as u32,
            height: height as u32,
            fps: effective_fps,
            bitrate_kbps: effective_bitrate_kbps,
            max_h264_profile: effective_h264_profile,
            // Natural GOP = 2 s, matching HlsConfig::segment_target_secs.
            // Keeps encoder's internal IDR cadence aligned with the
            // segmenter's wall-clock requests so we don't pay for an
            // extra IDR mid-segment on 60 fps devices.
            keyframe_interval_secs: 2.0,
            ..Default::default()
        })?;

        // Adaptive bitrate state — shared between the receiver's HLS
        // server (records per-segment delivery pressure) and this
        // stream task (reads the recommended target on the hot path).
        // The HLS endpoint inside `session` will pick it up from
        // `StreamConfig::adaptive` in `setup_stream`. Initial target
        // is the already-clamped `effective_bitrate_kbps` so the
        // controller's ceiling matches what the manager negotiated.
        let adaptive = ferricast_core::AdaptiveBitrateState::new(effective_bitrate_kbps);
        tracing::info!(
            initial_kbps = effective_bitrate_kbps,
            ceiling_kbps = adaptive.ceiling_kbps,
            floor_kbps = adaptive.floor_kbps,
            "adaptive: controller initialised; bandwidth probe will fire after first {} segments",
            ferricast_hls::PROBE_SAMPLES_FOR_DOC,
        );
        // Hand the receiver session the *clamped* fps + bitrate
        // (device-cap-aware) rather than the raw values from the UI.
        // Without this, the HLS segmenter inside the chromecast
        // session sizes its synthetic PTS counter to the UI's 60 fps
        // even when we're really only producing 30, and audio
        // (whose PTS tracks wall clock through the audio capture
        // path) drifts ahead of video on the receiver.
        let mut session = (proto.create_session)(&device)?;
        match session.connect(&device).await? {
            ConnectOutcome::Ready => {}
            ConnectOutcome::PairingRequired(challenge) => {
                let (respond_tx, respond_rx) = tokio::sync::oneshot::channel::<PairingResponse>();
                let respond = std::sync::Arc::new(tokio::sync::Mutex::new(Some(respond_tx)));
                let _ = self
                    .event_tx
                    .send(ManagerEvent::PairingRequired {
                        device_id,
                        challenge,
                        respond,
                    })
                    .await;
                let response = respond_rx.await.map_err(|_| {
                    FerricastError::Protocol("pairing response channel dropped".into())
                })?;
                session.submit_pairing(response).await?;
            }
        }

        // Set the codec in session_config *before* setup_stream so that
        // protocol-level validation (e.g. mirror requires Opus) sees the
        // correct value. After setup_stream the session may have fallen
        // back from Mirror to Hls, so we re-read preferred_audio_codec()
        // below when spawning the encoder task.
        let mut session_config = StreamConfig {
            adaptive: Some(adaptive.clone()),
            fps: effective_fps,
            bitrate_kbps: effective_bitrate_kbps,
            ..config.clone()
        };
        if let Some(ref mut audio) = session_config.audio {
            audio.codec = session.preferred_audio_codec();
        }

        session.setup_stream(&session_config).await?;

        // Audio pipeline. Spawned *after* `session.connect()` finishes
        // so PipeWire-captured PCM doesn't pile up while the chromecast
        // handshake is running (LAUNCH ack + virtual-connect can take
        // 3-6 s). Without this delay the queue chain
        // `pipewire → AAC encoder → tx (64) → audio_tx (64)` fills
        // entirely during the wait, then the segmenter drains stale
        // frames whose `timestamp_us` is from seconds ago — the
        // muxer's silent-AAC watermark advances past those PTSes
        // immediately and the real audio is rejected for the rest
        // of the stream's life.
        //
        // When `config.audio` is `None` (caller didn't opt in), we
        // skip the whole thing — the chromecast HLS pipeline falls
        // back to the silent-AAC injection for receivers that
        // require it.
        let (audio_task, audio_frame_rx, audio_mute) = match config.audio.clone() {
            Some(mut audio_cfg) => {
                audio_cfg.codec = session.preferred_audio_codec();
                let (tx, rx) = mpsc::channel::<AudioFrame>(64);
                let mute = audio_cfg.mute.clone();
                let mute_for_task = mute.clone();
                let task = tokio::spawn(async move {
                    if let Err(e) = run_audio_pipeline(audio_cfg, mute_for_task, tx).await {
                        tracing::error!(%e, "audio pipeline exited with error");
                    } else {
                        tracing::info!("audio pipeline exited cleanly");
                    }
                });
                (Some(task), Some(rx), Some(mute))
            }
            None => (None, None, None),
        };
        let _ = audio_mute; // surfaced separately when UI mute API lands

        // Factory the supervisor uses to rebuild a session on
        // receiver-side disconnect. Cloned `Arc` — cheap and `Send`.
        let create_session = proto.create_session.clone();

        let (cancel_tx, mut cancel_rx) = mpsc::channel::<()>(1);

        let active_streams = self.active_streams.clone();
        let event_tx = self.event_tx.clone();
        let device_name = device.name.clone();
        let did = device.id;
        // Task captures the device by move (the reconnect supervisor
        // hands `&device_clone` to `s.connect`), and `ActiveStream`
        // below also needs the original — clone once before the spawn.
        let device_clone = device.clone();
        let mut audio_frame_rx = audio_frame_rx;
        let audio_task_handle = audio_task;

        let task = tokio::spawn(async move {
            let _ = event_tx
                .send(ManagerEvent::StreamStarted {
                    device_id: did,
                    device_name,
                })
                .await;

            // The frame we already pulled to barrier on negotiation
            // is the first frame of this loop — dropping it would
            // waste an entire keyframe interval on backends that
            // emit IDR on the very first frame.
            let mut seed = Some(first_frame);

            // Pacing for event-driven capture sources. Wayland
            // compositors only emit a new dmabuf when the screen
            // actually changes — meaning a stream-paused user
            // (e.g. waiting for the chromecast to start displaying)
            // can go many seconds with no new frame. Receivers
            // that need a steady cadence (HLS chromecast, DASH,
            // anything muxed into MPEG-TS) starve under that:
            // their segmenter expects 1 keyframe / second and
            // can't close a segment without two of them.
            //
            // Pace at `effective_fps`: if `capture.next_frame()`
            // doesn't deliver inside one frame period, re-emit
            // the most recent real frame with its timestamp
            // bumped forward. The encoder treats it as a fresh
            // input and produces another encoded frame; the
            // downstream session sees a steady stream regardless
            // of what's happening on screen.
            let frame_period = Duration::from_secs_f64(1.0 / (effective_fps.max(1) as f64));
            let frame_period_us = (1_000_000.0 / (effective_fps.max(1) as f64)) as u64;
            let mut last_frame: Option<CapturedFrame> = None;

            // Wall-clock keyframe pacing — see longer comment below.
            // Pinned to 2 s to match HlsConfig::segment_target_secs;
            // the segmenter already forces an IDR per segment via
            // request_keyframe, but this guarantees one even if the
            // capture stalls between segmenter ticks.
            let keyframe_interval = Duration::from_secs(2);
            let mut next_keyframe_at = Instant::now();

            // Mirror of `adaptive.target_kbps()` we've actually
            // pushed into the encoder. The HLS server may mutate
            // the atomic between any two frames; we only need to
            // call `encoder.set_bitrate_kbps` when it differs from
            // what's currently configured, otherwise every frame
            // would round-trip into NVENC's reconfigure call.
            let mut last_applied_kbps: u32 = effective_bitrate_kbps;

            // Auto-reconnect supervisor state. The receiver can die
            // mid-stream (Chromecast detailedErrorCode=301, Default
            // Media Receiver app exit, transient TCP RST burst on
            // a flaky 2.4 GHz link) without it being a permanent
            // failure of capture or encoder. The pattern here is
            // session-scoped retry: only the receiver session is
            // recreated, capture and encoder keep their internal
            // state. After `MAX_CONSECUTIVE_FAILURES` quick deaths
            // in a row we surface a fatal `StreamError` so the app
            // can fall back to whatever its policy is.
            const MAX_CONSECUTIVE_FAILURES: u32 = 5;
            // Reset the failure counter when the stream has been
            // healthy for at least this long — a stream that ran
            // fine for 10 min then hit one transient is not the
            // same case as five 301s in 20 s.
            const HEALTHY_RESET_AFTER: Duration = Duration::from_secs(60);

            let mut consecutive_failures: u32 = 0;
            let mut last_failure_at: Option<Instant> = None;
            let mut active_session: Option<Box<dyn ErasedSession>> = Some(session);
            let mut fatal_error: Option<String> = None;
            let mut cancelled = false;

            // Outer (supervisor) loop. Acquires a session, runs the
            // inner frame loop until that session dies or the user
            // cancels, then decides whether to retry or surface.
            'supervisor: loop {
                let mut session: Box<dyn ErasedSession> = match active_session.take() {
                    Some(s) => s,
                    None => {
                        // Reconnect path.
                        // Reset counter on long-healthy streaks.
                        if let Some(t) = last_failure_at {
                            if t.elapsed() > HEALTHY_RESET_AFTER {
                                consecutive_failures = 0;
                            }
                        }
                        consecutive_failures += 1;
                        if consecutive_failures > MAX_CONSECUTIVE_FAILURES {
                            fatal_error = Some(format!(
                                "receiver-side disconnect; gave up after {} consecutive reconnect attempts",
                                MAX_CONSECUTIVE_FAILURES
                            ));
                            break 'supervisor;
                        }
                        // Force the adaptive controller to the floor
                        // before reconnecting. The previous bitrate
                        // didn't keep the receiver alive; starting
                        // low gives us the best chance of getting
                        // through the soft window after the receiver
                        // app relaunches, when its buffer is still
                        // empty and any further pressure would
                        // trigger 301 again.
                        let floored = adaptive.drop_to_floor();
                        // Apply immediately so the next encode is
                        // already at the lower rate (no waiting for
                        // the inner loop's first iteration to notice).
                        if let Err(e) = encoder.set_bitrate_kbps(floored) {
                            tracing::warn!(%e, "reconnect: encoder set_bitrate_kbps failed");
                        }
                        last_applied_kbps = floored;
                        // Request an IDR so the new HLS sink's
                        // bootstrap finds SPS/PPS on the very first
                        // frame instead of waiting for the encoder's
                        // natural GOP boundary.
                        encoder.request_keyframe();

                        // Backoff: 1 s, 2 s, 4 s, 8 s, 16 s.
                        let backoff = Duration::from_secs(
                            1u64 << (consecutive_failures.saturating_sub(1).min(4) as u64),
                        );
                        let reason = format!(
                            "reconnecting (attempt {}/{}, backing off {:?})",
                            consecutive_failures, MAX_CONSECUTIVE_FAILURES, backoff
                        );
                        tracing::warn!(
                            ?did,
                            attempt = consecutive_failures,
                            ?backoff,
                            "reconnect: retrying session"
                        );
                        let _ = event_tx
                            .send(ManagerEvent::StreamReconnecting {
                                device_id: did,
                                attempt: consecutive_failures,
                                reason: reason.clone(),
                            })
                            .await;
                        tokio::select! {
                            _ = cancel_rx.recv() => {
                                cancelled = true;
                                break 'supervisor;
                            }
                            _ = tokio::time::sleep(backoff) => {}
                        }

                        let mut s = match (create_session)(&device_clone) {
                            Ok(s) => s,
                            Err(e) => {
                                tracing::warn!(%e, "reconnect: create_session failed");
                                continue 'supervisor;
                            }
                        };
                        match s.connect(&device_clone).await {
                            Err(e) => {
                                tracing::warn!(%e, "reconnect: session.connect failed");
                                continue 'supervisor;
                            }
                            Ok(ConnectOutcome::Ready) => {}
                            Ok(ConnectOutcome::PairingRequired(challenge)) => {
                                // On reconnect we don't re-ask for pairing —
                                // the device was already paired this session.
                                // Surface as a fatal error so the UI can prompt
                                // the user to start a fresh stream instead.
                                let msg = format!(
                                    "reconnect: device requested pairing again ({challenge:?}); \
                                     giving up"
                                );
                                tracing::warn!("{}", msg);
                                fatal_error = Some(msg);
                                break 'supervisor;
                            }
                        }
                        if let Err(e) = s.setup_stream(&session_config).await {
                            tracing::warn!(%e, "reconnect: session.setup_stream failed");
                            let _ = s.stop().await;
                            continue 'supervisor;
                        }
                        tracing::info!(
                            ?did,
                            attempt = consecutive_failures,
                            "reconnect: session re-established"
                        );
                        last_failure_at = Some(Instant::now());
                        s
                    }
                };

                // Inner frame loop. Same hot path as before; on
                // session death we drop out and let the supervisor
                // recreate.
                'inner: loop {
                    tokio::select! {
                        // Audio polled before frame arm so slow
                        // x264 encodes can't starve the AAC queue.
                        biased;
                        _ = cancel_rx.recv() => {
                            tracing::info!(?did, "Stream cancelled");
                            cancelled = true;
                            let _ = session.stop().await;
                            break 'supervisor;
                        }
                        audio = recv_audio_frame(&mut audio_frame_rx), if audio_frame_rx.is_some() => {
                            match audio {
                                Some(af) => {
                                    if let Err(e) = session.send_audio_frame(&af).await {
                                        // Audio errors do NOT kill
                                        // the stream — video is the
                                        // primary content. Log and
                                        // keep going.
                                        tracing::warn!(%e, "send_audio_frame failed");
                                    }
                                    // Drain remaining queued frames
                                    // in this wake-up.
                                    if let Some(rx) = audio_frame_rx.as_mut() {
                                        loop {
                                            match rx.try_recv() {
                                                Ok(extra) => {
                                                    if let Err(e) =
                                                        session.send_audio_frame(&extra).await
                                                    {
                                                        tracing::warn!(
                                                            %e,
                                                            "send_audio_frame (drain) failed"
                                                        );
                                                    }
                                                }
                                                Err(mpsc::error::TryRecvError::Empty) => break,
                                                Err(mpsc::error::TryRecvError::Disconnected) => {
                                                    audio_frame_rx = None;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                                None => {
                                    // Audio pipeline exited. For Cast
                                    // Streaming this is a problem: the
                                    // Chromecast will never get audio
                                    // → stays black. Log at WARN so
                                    // the user can see it in field logs.
                                    tracing::warn!(
                                        ?did,
                                        "audio pipeline exited — \
                                         Cast Streaming receiver may stay black \
                                         (no audio keyframe will arrive)"
                                    );
                                    audio_frame_rx = None;
                                }
                            }
                        }
                        frame_result = async {
                            if let Some(f) = seed.take() {
                                last_frame = Some(f.clone());
                                return Ok(f);
                            }
                            match tokio::time::timeout(frame_period, capture.next_frame()).await {
                                Ok(Ok(f)) => {
                                    last_frame = Some(f.clone());
                                    Ok(f)
                                }
                                Ok(Err(e)) => Err(e),
                                Err(_) => match last_frame.as_mut() {
                                    Some(stored) => {
                                        bump_timestamp(stored, frame_period_us);
                                        Ok(stored.clone())
                                    }
                                    None => capture.next_frame().await,
                                },
                            }
                        } => {
                            match frame_result {
                                Ok(raw_frame) => {
                                    let now = Instant::now();
                                    if now >= next_keyframe_at {
                                        encoder.request_keyframe();
                                        while next_keyframe_at <= now {
                                            next_keyframe_at += keyframe_interval;
                                        }
                                    }
                                    let want_kbps = adaptive.target_kbps();
                                    if want_kbps != last_applied_kbps {
                                        match encoder.set_bitrate_kbps(want_kbps) {
                                            Ok(()) => {
                                                tracing::info!(
                                                    from_kbps = last_applied_kbps,
                                                    to_kbps = want_kbps,
                                                    "adaptive: encoder bitrate updated"
                                                );
                                                last_applied_kbps = want_kbps;
                                            }
                                            Err(e) => {
                                                tracing::warn!(%e, want_kbps, "adaptive: set_bitrate_kbps failed");
                                                last_applied_kbps = want_kbps;
                                            }
                                        }
                                    }

                                    match encoder.encode(raw_frame) {
                                        Ok(encoded) => {
                                            if let Err(e) = session.send_frame(&encoded).await {
                                                tracing::error!(%e, "Failed to send frame");
                                                fatal_error = Some(e.to_string());
                                                let _ = session.stop().await;
                                                break 'supervisor;
                                            }
                                            // PLI from receiver (e.g. Cast
                                            // Streaming RTCP feedback on
                                            // packet loss): request an IDR
                                            // so the receiver can resync.
                                            if session.poll_keyframe_request() {
                                                tracing::debug!(?did, "PLI received — requesting keyframe");
                                                encoder.request_keyframe();
                                            }
                                            // Receiver-side liveness: when
                                            // the chromecast app dies it
                                            // sends CLOSE / ERROR 301; the
                                            // session marks itself not
                                            // alive but `send_frame` keeps
                                            // accepting (HLS sink is local
                                            // and stays up). Catching it
                                            // here pivots us back to the
                                            // supervisor for a clean
                                            // reconnect instead of feeding
                                            // a dead pipe.
                                            if !session.is_alive() {
                                                tracing::warn!(?did, "session reports not alive — entering reconnect supervisor");
                                                let _ = session.stop().await;
                                                break 'inner;
                                            }
                                        }
                                        Err(e) => {
                                            tracing::error!(%e, "Encoding error");
                                            fatal_error = Some(e.to_string());
                                            let _ = session.stop().await;
                                            break 'supervisor;
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::error!(%e, "Capture error");
                                    fatal_error = Some(e.to_string());
                                    let _ = session.stop().await;
                                    break 'supervisor;
                                }
                            }
                        }
                    }
                }
            }

            let _ = capture.stop().await;
            // Tear the audio task down alongside the video pipeline
            // so the PipeWire audio thread and AAC encoder don't
            // outlive the stream. Aborting is safe — the task
            // doesn't hold any externally-visible resources besides
            // its own PipeWire loop, which is cleaned up by the
            // capture's Drop.
            if let Some(h) = &audio_task_handle {
                h.abort();
            }
            active_streams.lock().await.remove(&did);
            // Surface a final error if the supervisor gave up.
            // `StreamStopped` is always emitted afterwards so the
            // app's "playing → stopped" state machine doesn't need
            // to special-case error paths — same terminal event,
            // optional error explainer just before it.
            if let Some(msg) = fatal_error {
                let _ = event_tx
                    .send(ManagerEvent::StreamError {
                        device_id: did,
                        message: msg,
                    })
                    .await;
            }
            let _ = cancelled; // bookkeeping; not all paths read it
            let _ = event_tx
                .send(ManagerEvent::StreamStopped { device_id: did })
                .await;
        });

        self.active_streams.lock().await.insert(
            device_id,
            ActiveStream {
                device,
                cancel_tx,
                task,
            },
        );

        Ok(())
    }

    pub async fn stop_stream(&self, device_id: Uuid) -> Result<()> {
        let stream = self.active_streams.lock().await.remove(&device_id);
        if let Some(stream) = stream {
            tracing::info!(name = %stream.device.name, "Stopping stream");
            // Signal the loop to break, then wait for the task to
            // actually finish — capture.stop() / session.stop() run
            // *inside* the task and we want them done before we
            // return so the caller (e.g. ctrl-c handler) can rely on
            // "stop_stream returned" meaning "no more PipeWire
            // worker thread / TLS connection / HLS listener alive".
            let _ = stream.cancel_tx.send(()).await;
            // 5s ceiling: if cleanup wedges (PipeWire main loop
            // refusing to quit, TLS write blocked behind a dead
            // peer) we'd rather log + move on than hang forever.
            match tokio::time::timeout(std::time::Duration::from_secs(5), stream.task).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::warn!(?e, "stream task panicked during shutdown"),
                Err(_) => {
                    tracing::warn!(name = %stream.device.name, "stream task did not finish in 5s, abandoning")
                }
            }
            Ok(())
        } else {
            Err(FerricastError::NoActiveSession)
        }
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        let ids: Vec<Uuid> = self.active_streams.lock().await.keys().copied().collect();
        for id in ids {
            let _ = self.stop_stream(id).await;
        }
        let _ = self.stop_receivers().await;
        self.stop_discovery().await
    }
}

/// Forward `frame`'s timestamp by `delta_us` microseconds. Used by
/// the manager loop's pacing fallback: when the upstream capture
/// stalls (Wayland compositor not emitting frames because nothing
/// on screen has changed) we re-emit the most recent real frame
/// with a synthetic monotonic timestamp so the downstream encoder
/// + segmenter keeps producing output at the configured fps.
fn bump_timestamp(frame: &mut CapturedFrame, delta_us: u64) {
    match frame {
        CapturedFrame::Cpu(r) => r.timestamp_us = r.timestamp_us.saturating_add(delta_us),
        CapturedFrame::Gpu(g) => g.timestamp_us = g.timestamp_us.saturating_add(delta_us),
    }
}

impl StreamManager {
    /// Start a builder for ergonomic construction.
    ///
    /// ```no_run
    /// use ferricast::StreamManager;
    /// let (manager, events) = StreamManager::builder()
    ///     .with_chromecast()
    ///     .build_with_events();
    /// # drop((manager, events));
    /// ```
    pub fn builder() -> StreamManagerBuilder {
        StreamManagerBuilder::default()
    }
}

/// Fluent builder for [`StreamManager`].
///
/// Each `with_*` helper registers a protocol handler. The generic
/// [`Self::register`] method takes any [`ProtocolHandler`] for
/// protocols that aren't exposed as a first-class convenience yet
/// (or that live outside the workspace).
#[derive(Default)]
pub struct StreamManagerBuilder {
    manager: StreamManager,
}

impl StreamManagerBuilder {
    /// Register an arbitrary protocol handler.
    pub fn register<H>(mut self) -> Self
    where
        H: ProtocolHandler + Clone + Default + 'static,
    {
        self.manager.register::<H>();
        self
    }

    /// Register the Chromecast (CASTv2) handler.
    #[cfg(feature = "chromecast")]
    pub fn with_chromecast(self) -> Self {
        self.register::<ferricast_chromecast::ChromecastHandler>()
    }

    #[cfg(feature = "airplay")]
    pub fn with_airplay(self) -> Self {
        self.register::<ferricast_airplay::AirPlayHandler>()
    }

    /// Register the Chromecast *receiver* — advertise this process
    /// as a Cast target and accept LOAD / PLAY / PAUSE / SEEK from
    /// senders.
    #[cfg(feature = "chromecast")]
    pub fn with_chromecast_receiver(self) -> Self {
        self.register_receiver::<ferricast_chromecast::ChromecastReceiver>()
    }

    /// Register the bundled H.264 video decoder facade
    /// ([`ferricast_decoder::H264Decoder`]).
    pub fn with_h264_decoder(self) -> Self {
        self.register_video_decoder::<ferricast_decoder::H264Decoder>()
    }

    /// Register the bundled AAC audio decoder
    /// ([`ferricast_decoder::AacDecoder`]).
    #[cfg(feature = "aac")]
    pub fn with_aac_decoder(self) -> Self {
        self.register_audio_decoder::<ferricast_decoder::AacDecoder>()
    }

    /// Register a receiver-side protocol handler.
    pub fn register_receiver<P>(mut self) -> Self
    where
        P: ReceiverProtocol + Clone + Default + 'static,
    {
        self.manager.register_receiver::<P>();
        self
    }

    /// Register a video decoder factory. See
    /// [`StreamManager::register_video_decoder`] for ordering.
    pub fn register_video_decoder<D>(mut self) -> Self
    where
        D: VideoDecoder + Default + 'static,
    {
        self.manager.register_video_decoder::<D>();
        self
    }

    pub fn register_audio_decoder<D>(mut self) -> Self
    where
        D: AudioDecoder + Default + 'static,
    {
        self.manager.register_audio_decoder::<D>();
        self
    }

    /// Set the sink factory the receiver pipeline asks for a
    /// destination whenever a new transmission arrives. The GUI
    /// typically opens a new window inside this callback.
    pub fn set_sink_factory<F, Fut>(mut self, factory: F) -> Self
    where
        F: Fn(&RemoteSender, &MediaInfo) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Box<dyn FrameSink>>> + Send + 'static,
    {
        self.manager.set_sink_factory(factory);
        self
    }

    /// Restrict every protocol's server sockets to ports in
    /// `start..=end`. Applied to all future `start_stream` calls
    /// (callers that explicitly set `StreamConfig::port_range` take
    /// precedence). Useful for firewall-managed hosts: open exactly
    /// this range and nothing else reaches the device.
    pub fn with_port_range(mut self, start: u16, end: u16) -> Self {
        self.manager.port_range = Some((start, end));
        self
    }

    /// Finalize the builder and return the manager. The event
    /// receiver stays inside the manager; call
    /// [`StreamManager::take_event_rx`] when you need it.
    pub fn build(self) -> StreamManager {
        self.manager
    }

    /// Finalize and split out the event receiver in one step — the
    /// common shape for apps that wrap the manager in `Arc<Mutex<_>>`
    /// and need the rx accessible from a separate task.
    pub fn build_with_events(
        mut self,
    ) -> (StreamManager, tokio::sync::mpsc::Receiver<ManagerEvent>) {
        let rx = self
            .manager
            .take_event_rx()
            .expect("freshly built manager always has its event_rx");
        (self.manager, rx)
    }
}

/// Helper for the supervisor's `tokio::select!` arm reading from
/// the optional audio frame channel. `select!` can't dispatch on an
/// `Option<&mut Receiver<_>>` directly, so we resolve it here:
/// returns `None` when no receiver is wired (the arm's `if` guard
/// disables it anyway, but we keep the future well-formed).
async fn recv_audio_frame(chan: &mut Option<mpsc::Receiver<AudioFrame>>) -> Option<AudioFrame> {
    match chan {
        Some(rx) => rx.recv().await,
        None => std::future::pending().await,
    }
}

/// Dispatch to the right encoder backend based on `audio_cfg.codec`.
async fn run_audio_pipeline(
    audio_cfg: ferricast_core::AudioStreamConfig,
    mute: ferricast_core::AudioMuteHandle,
    tx: mpsc::Sender<AudioFrame>,
) -> Result<()> {
    match audio_cfg.codec {
        #[cfg(feature = "opus-rs")]
        AudioCodec::Opus => run_audio_pipeline_with(OpusEncoder::new(), audio_cfg, mute, tx).await,
        _ => run_audio_pipeline_with(AacEncoder::new(), audio_cfg, mute, tx).await,
    }
}

/// Run the PipeWire audio capture → encoder pipeline forever.
/// Lives in its own task so the blocking PipeWire main-loop thread
/// doesn't interfere with the supervisor's video hot path.
///
/// The encoder is configured lazily on the first PCM chunk using
/// whatever rate / channels PipeWire actually negotiated, so we
/// never drive the codec at the wrong sample-rate index.
async fn run_audio_pipeline_with<E: AudioEncoder>(
    mut encoder: E,
    audio_cfg: ferricast_core::AudioStreamConfig,
    mute: ferricast_core::AudioMuteHandle,
    tx: mpsc::Sender<AudioFrame>,
) -> Result<()> {
    let mut capture = PipeWireAudioCapture::new();
    let cap_cfg = AudioCaptureConfig {
        sample_rate: audio_cfg.sample_rate,
        channels: audio_cfg.channels,
    };
    capture
        .start(AudioSource::DefaultMonitor, cap_cfg, mute)
        .await?;
    tracing::info!(
        requested_sample_rate = audio_cfg.sample_rate,
        requested_channels = audio_cfg.channels,
        bitrate_kbps = audio_cfg.bitrate_kbps,
        codec = ?audio_cfg.codec,
        "audio capture started; encoder will be configured on first PCM chunk"
    );

    let mut encoder_configured = false;

    loop {
        let pcm = match capture.next_frame().await {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!(%e, "audio capture next_frame failed; stopping pipeline");
                break;
            }
        };
        if !encoder_configured {
            encoder.configure(&AudioEncoderConfig {
                codec: audio_cfg.codec,
                sample_rate: pcm.sample_rate,
                channels: pcm.channels,
                bitrate_kbps: audio_cfg.bitrate_kbps,
            })?;
            tracing::info!(
                sample_rate = pcm.sample_rate,
                channels = pcm.channels,
                bitrate_kbps = audio_cfg.bitrate_kbps,
                codec = ?audio_cfg.codec,
                "audio encoder configured against PipeWire's negotiated format"
            );
            encoder_configured = true;
        }
        if let Err(e) = encoder.encode(&pcm) {
            tracing::warn!(%e, "audio encode failed; dropping chunk");
            continue;
        }
        for frame in encoder.take_output() {
            if tx.send(frame).await.is_err() {
                tracing::info!("audio supervisor dropped receiver; exiting pipeline");
                let _ = capture.stop().await;
                return Ok(());
            }
        }
    }

    let _ = capture.stop().await;
    Ok(())
}
