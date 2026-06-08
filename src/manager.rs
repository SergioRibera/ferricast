use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, RwLock, mpsc};
use tokio::task::JoinHandle;
use uuid::Uuid;

use ferricast_core::{
    AdvertiseInfo, Advertiser, AudioCodec, AudioDecoder, AudioDecoderConfig, AudioFrame,
    CaptureConfig, CaptureSource, CapturedFrame, CastSession, Codec, ControlSession, DecodedAudio,
    DecoderConfig, Device, Discovery, DiscoveryEvent, EncodedFrame, EncoderConfig, FerricastError,
    FrameSink, MediaCommand, MediaInfo, MediaPacket, MediaPuller, PixelFormat, PlaybackState,
    ProtocolHandler, PullSpec, ReceiverProtocol, RemoteSender, Result, ScreenCapture, StreamConfig,
    VideoDecoder, VideoEncoder,
};

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
    fn connect<'a>(&'a mut self, device: &'a Device) -> BoxFut<'a, Result<()>>;
    fn setup_stream<'a>(&'a mut self, config: &'a StreamConfig) -> BoxFut<'a, Result<()>>;
    fn send_frame<'a>(&'a mut self, frame: &'a EncodedFrame) -> BoxFut<'a, Result<()>>;
    fn stop(&mut self) -> BoxFut<'_, Result<()>>;
    fn is_alive(&self) -> bool;
}

impl<T: CastSession> ErasedSession for T {
    fn connect<'a>(&'a mut self, device: &'a Device) -> BoxFut<'a, Result<()>> {
        Box::pin(CastSession::connect(self, device))
    }
    fn setup_stream<'a>(&'a mut self, config: &'a StreamConfig) -> BoxFut<'a, Result<()>> {
        Box::pin(CastSession::setup_stream(self, config))
    }
    fn send_frame<'a>(&'a mut self, frame: &'a EncodedFrame) -> BoxFut<'a, Result<()>> {
        Box::pin(CastSession::send_frame(self, frame))
    }
    fn stop(&mut self) -> BoxFut<'_, Result<()>> {
        Box::pin(CastSession::stop(self))
    }
    fn is_alive(&self) -> bool {
        CastSession::is_alive(self)
    }
}

// ──────────────────────────────────────────────────────────────────
// Receiver-side erasure. Mirrors the sender-side erasure above:
// every receiver trait that the manager talks to via `dyn`-style
// dispatch goes through an `ErasedX` shim because the native trait
// uses RPITIT (`impl Future + Send`) which isn't object-safe.

trait ErasedAdvertiser: Send + Sync {
    fn start(&mut self, info: AdvertiseInfo) -> BoxFut<'_, Result<()>>;
    fn stop(&mut self) -> BoxFut<'_, Result<()>>;
}

impl<T: Advertiser> ErasedAdvertiser for T {
    fn start(&mut self, info: AdvertiseInfo) -> BoxFut<'_, Result<()>> {
        Box::pin(Advertiser::start(self, info))
    }
    fn stop(&mut self) -> BoxFut<'_, Result<()>> {
        Box::pin(Advertiser::stop(self))
    }
}

trait ErasedControlSession: Send + Sync {
    fn accept(&mut self) -> BoxFut<'_, Result<RemoteSender>>;
    fn next_command(&mut self) -> BoxFut<'_, Result<MediaCommand>>;
    fn report_state(&mut self, state: PlaybackState) -> BoxFut<'_, Result<()>>;
    fn close(&mut self) -> BoxFut<'_, Result<()>>;
}

impl<T: ControlSession> ErasedControlSession for T {
    fn accept(&mut self) -> BoxFut<'_, Result<RemoteSender>> {
        Box::pin(ControlSession::accept(self))
    }
    fn next_command(&mut self) -> BoxFut<'_, Result<MediaCommand>> {
        Box::pin(ControlSession::next_command(self))
    }
    fn report_state(&mut self, state: PlaybackState) -> BoxFut<'_, Result<()>> {
        Box::pin(ControlSession::report_state(self, state))
    }
    fn close(&mut self) -> BoxFut<'_, Result<()>> {
        Box::pin(ControlSession::close(self))
    }
}

trait ErasedPuller: Send {
    fn open(&mut self, spec: PullSpec) -> BoxFut<'_, Result<MediaInfo>>;
    fn next(&mut self) -> BoxFut<'_, Result<MediaPacket>>;
    fn seek(&mut self, position_us: u64) -> BoxFut<'_, Result<()>>;
    fn close(&mut self) -> BoxFut<'_, Result<()>>;
}

impl<T: MediaPuller> ErasedPuller for T {
    fn open(&mut self, spec: PullSpec) -> BoxFut<'_, Result<MediaInfo>> {
        Box::pin(MediaPuller::open(self, spec))
    }
    fn next(&mut self) -> BoxFut<'_, Result<MediaPacket>> {
        Box::pin(MediaPuller::next(self))
    }
    fn seek(&mut self, position_us: u64) -> BoxFut<'_, Result<()>> {
        Box::pin(MediaPuller::seek(self, position_us))
    }
    fn close(&mut self) -> BoxFut<'_, Result<()>> {
        Box::pin(MediaPuller::close(self))
    }
}

// Decoders and sinks aren't async (or are async but don't need
// lifetime tricks), so plain `dyn` works. `Box<dyn ErasedVideoDecoder>`
// is what the pipeline holds; the wrapper just exists so `decode`'s
// concrete type doesn't leak out of the registry.

trait ErasedVideoDecoder: Send {
    fn configure(&mut self, config: &DecoderConfig) -> Result<()>;
    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>>;
}

impl<T: VideoDecoder> ErasedVideoDecoder for T {
    fn configure(&mut self, config: &DecoderConfig) -> Result<()> {
        VideoDecoder::configure(self, config)
    }
    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        VideoDecoder::decode(self, frame)
    }
}

trait ErasedAudioDecoder: Send {
    fn configure(&mut self, config: &AudioDecoderConfig) -> Result<()>;
    fn decode(&mut self, frame: AudioFrame) -> Result<Option<DecodedAudio>>;
}

impl<T: AudioDecoder> ErasedAudioDecoder for T {
    fn configure(&mut self, config: &AudioDecoderConfig) -> Result<()> {
        AudioDecoder::configure(self, config)
    }
    fn decode(&mut self, frame: AudioFrame) -> Result<Option<DecodedAudio>> {
        AudioDecoder::decode(self, frame)
    }
}

// `FrameSink` is already object-safe (uses `async_trait`) so the
// pipeline holds `Box<dyn FrameSink>` directly — no `ErasedSink`
// wrapper needed.

// Factory used by the receiver pipeline to ask the application
// (typically the GUI) for somewhere to send decoded frames. Called
// once per accepted session, after the puller has reported
// `MediaInfo` — that way the app can open an audio-only card view
// vs. a video window based on whether `info.video.is_some()`.
type SinkFactory = Arc<
    dyn Fn(&RemoteSender, &MediaInfo) -> BoxFut<'static, Result<Box<dyn FrameSink>>> + Send + Sync,
>;

struct RegisteredReceiver {
    protocol: &'static str,
    create_advertiser: Box<dyn Fn() -> Box<dyn ErasedAdvertiser> + Send + Sync>,
    create_control: Arc<dyn Fn() -> Result<Box<dyn ErasedControlSession>> + Send + Sync>,
    create_puller: Arc<dyn Fn() -> Result<Box<dyn ErasedPuller>> + Send + Sync>,
    advertise_info: AdvertiseInfo,
}

struct ActiveReceiver {
    protocol: &'static str,
    cancel_tx: mpsc::Sender<()>,
    task: JoinHandle<()>,
    advertiser: Box<dyn ErasedAdvertiser>,
}

struct RegisteredProtocol {
    protocol: &'static str,
    create_discovery: Box<dyn Fn() -> Box<dyn ErasedDiscovery> + Send + Sync>,
    /// Wrapped in `Arc` instead of `Box` so the streaming task can
    /// hold its own clone — needed for auto-reconnect, which builds
    /// a fresh session on `is_alive()` going false (chromecast 301,
    /// receiver-app crash) without tearing down capture/encoder.
    create_session: Arc<dyn Fn() -> Result<Box<dyn ErasedSession>> + Send + Sync>,
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
    ReceiverStopped { receiver_id: Uuid },
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
    receivers: Vec<RegisteredReceiver>,
    /// Codec → factory map. Pipeline picks a video decoder based on
    /// what the puller's `MediaInfo` declares; multiple decoders
    /// MAY register for the same codec (e.g. VA-API + NVDEC + sw)
    /// and the last one wins — register the preferred backend last,
    /// or use the builder's `with_*_decoder` helpers which append
    /// in priority order.
    video_decoders: HashMap<Codec, Arc<dyn Fn() -> Box<dyn ErasedVideoDecoder> + Send + Sync>>,
    audio_decoders: HashMap<AudioCodec, Arc<dyn Fn() -> Box<dyn ErasedAudioDecoder> + Send + Sync>>,
    sink_factory: Option<SinkFactory>,
    active_receivers: Arc<Mutex<HashMap<Uuid, ActiveReceiver>>>,
    receivers_running: bool,
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
            receivers: Vec::new(),
            video_decoders: HashMap::new(),
            audio_decoders: HashMap::new(),
            sink_factory: None,
            active_receivers: Arc::new(Mutex::new(HashMap::new())),
            receivers_running: false,
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
            create_session: Arc::new(move || {
                let session = h_sess.create_session()?;
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
        let p_adv = P::default();
        let p_ctrl = p_adv.clone();
        let p_pull = p_adv.clone();
        let info = p_adv.advertise_info();

        self.receivers.push(RegisteredReceiver {
            protocol: P::PROTOCOL,
            create_advertiser: Box::new(move || {
                Box::new(p_adv.create_advertiser()) as Box<dyn ErasedAdvertiser>
            }),
            create_control: Arc::new(move || {
                let c = p_ctrl.create_control()?;
                Ok(Box::new(c) as Box<dyn ErasedControlSession>)
            }),
            create_puller: Arc::new(move || {
                let p = p_pull.create_puller()?;
                Ok(Box::new(p) as Box<dyn ErasedPuller>)
            }),
            advertise_info: info,
        });
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
        self.video_decoders.insert(
            D::CODEC,
            Arc::new(|| Box::new(D::default()) as Box<dyn ErasedVideoDecoder>),
        );
    }

    pub fn register_audio_decoder<D>(&mut self)
    where
        D: AudioDecoder + Default + 'static,
    {
        self.audio_decoders.insert(
            D::CODEC,
            Arc::new(|| Box::new(D::default()) as Box<dyn ErasedAudioDecoder>),
        );
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
        self.sink_factory = Some(Arc::new(move |remote, info| {
            let fut = factory(remote, info);
            Box::pin(async move { fut.await })
        }));
    }

    pub fn registered_receivers(&self) -> Vec<&'static str> {
        self.receivers.iter().map(|r| r.protocol).collect()
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
        if self.receivers_running {
            return Ok(());
        }
        let sink_factory = self.sink_factory.clone().ok_or_else(|| {
            FerricastError::Receiver(
                "start_receivers called before set_sink_factory — \
                 the pipeline has nowhere to send decoded frames"
                    .into(),
            )
        })?;

        for proto in &self.receivers {
            let mut advertiser = (proto.create_advertiser)();
            if let Err(e) = advertiser.start(proto.advertise_info.clone()).await {
                tracing::warn!(
                    protocol = proto.protocol,
                    %e,
                    "receiver advertiser failed to start; skipping"
                );
                continue;
            }
            tracing::info!(
                protocol = proto.protocol,
                friendly_name = %proto.advertise_info.friendly_name,
                port = proto.advertise_info.port,
                "receiver advertised"
            );

            // Per-protocol accept supervisor. One control session at
            // a time — when it dies (remote closed, fatal error) we
            // build a fresh one and call `accept()` again. Mirrors
            // the supervisor pattern on the sender side.
            let receiver_id = Uuid::new_v4();
            let create_control = proto.create_control.clone();
            let create_puller = proto.create_puller.clone();
            let event_tx = self.event_tx.clone();
            let active = self.active_receivers.clone();
            let video_decoders = self.video_decoders.clone();
            let audio_decoders = self.audio_decoders.clone();
            let sink_factory = sink_factory.clone();
            let protocol = proto.protocol;
            let (cancel_tx, mut cancel_rx) = mpsc::channel::<()>(1);

            let task = tokio::spawn(async move {
                loop {
                    let mut control = match (create_control)() {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::error!(protocol, %e, "create_control failed; exiting supervisor");
                            let _ = event_tx
                                .send(ManagerEvent::ReceiverError {
                                    receiver_id,
                                    message: format!("create_control: {e}"),
                                })
                                .await;
                            return;
                        }
                    };

                    let remote = tokio::select! {
                        _ = cancel_rx.recv() => return,
                        r = control.accept() => match r {
                            Ok(r) => r,
                            Err(e) => {
                                tracing::warn!(protocol, %e, "control.accept failed; retrying");
                                continue;
                            }
                        }
                    };
                    tracing::info!(
                        protocol,
                        sender_id = %remote.id,
                        sender_addr = %remote.addr,
                        "receiver: remote sender connected"
                    );
                    let _ = event_tx
                        .send(ManagerEvent::ReceiverIncoming {
                            receiver_id,
                            protocol,
                            remote: remote.clone(),
                        })
                        .await;

                    let session_result = run_receiver_session(
                        receiver_id,
                        protocol,
                        remote,
                        &mut control,
                        create_puller.clone(),
                        &video_decoders,
                        &audio_decoders,
                        sink_factory.clone(),
                        event_tx.clone(),
                        &mut cancel_rx,
                    )
                    .await;

                    if let Err(e) = session_result {
                        tracing::warn!(protocol, %e, "receiver session ended with error");
                        let _ = event_tx
                            .send(ManagerEvent::ReceiverError {
                                receiver_id,
                                message: e.to_string(),
                            })
                            .await;
                    }
                    let _ = control.close().await;
                    let _ = event_tx
                        .send(ManagerEvent::ReceiverStopped { receiver_id })
                        .await;
                }
            });

            active.lock().await.insert(
                receiver_id,
                ActiveReceiver {
                    protocol,
                    cancel_tx,
                    task,
                    advertiser,
                },
            );
        }

        self.receivers_running = true;
        Ok(())
    }

    pub async fn stop_receivers(&mut self) -> Result<()> {
        let active = {
            let mut guard = self.active_receivers.lock().await;
            std::mem::take(&mut *guard)
        };
        for (_, mut r) in active {
            let _ = r.cancel_tx.send(()).await;
            let _ = r.advertiser.stop().await;
            match tokio::time::timeout(Duration::from_secs(5), r.task).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::warn!(?e, "receiver task panicked"),
                Err(_) => tracing::warn!(protocol = r.protocol, "receiver task did not finish in 5s"),
            }
        }
        self.receivers_running = false;
        Ok(())
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
        config: StreamConfig,
    ) -> Result<()> {
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
        let session_config = StreamConfig {
            adaptive: Some(adaptive.clone()),
            ..config.clone()
        };

        let mut session = (proto.create_session)()?;
        session.connect(&device).await?;
        session.setup_stream(&session_config).await?;

        // Factory the supervisor uses to rebuild a session on
        // receiver-side disconnect. Cloned `Arc` — cheap and `Send`.
        let create_session = proto.create_session.clone();

        let (cancel_tx, mut cancel_rx) = mpsc::channel::<()>(1);

        let active_streams = self.active_streams.clone();
        let event_tx = self.event_tx.clone();
        let device_name = device.name.clone();
        let did = device.id;

        let device_clone = device.clone();

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
            let keyframe_interval = Duration::from_secs(4);
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

                        let mut s = match (create_session)() {
                            Ok(s) => s,
                            Err(e) => {
                                tracing::warn!(%e, "reconnect: create_session failed");
                                continue 'supervisor;
                            }
                        };
                        if let Err(e) = s.connect(&device_clone).await {
                            tracing::warn!(%e, "reconnect: session.connect failed");
                            continue 'supervisor;
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
                        _ = cancel_rx.recv() => {
                            tracing::info!(?did, "Stream cancelled");
                            cancelled = true;
                            let _ = session.stop().await;
                            break 'supervisor;
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

// ──────────────────────────────────────────────────────────────────
// Receiver pipeline helpers.
//
// `run_receiver_session` is the per-control-connection state machine:
// it reads commands off the control channel, on LOAD it builds a
// fresh puller + decoders + sink and hands them to `run_pump`, and
// it forwards PLAY/PAUSE/STOP/SEEK to the pump via `PumpCmd`.

enum PumpCmd {
    Play,
    Pause,
    Stop,
    Seek(u64),
}

#[allow(clippy::too_many_arguments)]
async fn run_receiver_session(
    receiver_id: Uuid,
    protocol: &'static str,
    remote: RemoteSender,
    control: &mut Box<dyn ErasedControlSession>,
    create_puller: Arc<dyn Fn() -> Result<Box<dyn ErasedPuller>> + Send + Sync>,
    video_decoders: &HashMap<Codec, Arc<dyn Fn() -> Box<dyn ErasedVideoDecoder> + Send + Sync>>,
    audio_decoders: &HashMap<AudioCodec, Arc<dyn Fn() -> Box<dyn ErasedAudioDecoder> + Send + Sync>>,
    sink_factory: SinkFactory,
    event_tx: mpsc::Sender<ManagerEvent>,
    cancel_rx: &mut mpsc::Receiver<()>,
) -> Result<()> {
    let mut pump: Option<JoinHandle<()>> = None;
    let mut pump_tx: Option<mpsc::Sender<PumpCmd>> = None;

    loop {
        tokio::select! {
            _ = cancel_rx.recv() => {
                if let Some(tx) = pump_tx.take() { let _ = tx.send(PumpCmd::Stop).await; }
                if let Some(h) = pump.take() { let _ = h.await; }
                return Ok(());
            }
            cmd = control.next_command() => {
                let cmd = cmd?;
                match cmd {
                    MediaCommand::Load { url, autoplay, .. } => {
                        // Tear down any in-flight pump from a prior LOAD.
                        if let Some(tx) = pump_tx.take() { let _ = tx.send(PumpCmd::Stop).await; }
                        if let Some(h) = pump.take() { let _ = h.await; }

                        let mut puller = (create_puller)()?;
                        let info = puller
                            .open(PullSpec { url: url.clone(), headers: HashMap::new() })
                            .await?;
                        tracing::info!(protocol, %url, ?info, "receiver: puller opened");

                        let _ = event_tx
                            .send(ManagerEvent::ReceiverStarted {
                                receiver_id,
                                remote: remote.clone(),
                                info: info.clone(),
                            })
                            .await;

                        let sink = (sink_factory)(&remote, &info).await?;

                        let video_dec = match &info.video {
                            Some(v) => {
                                let f = video_decoders.get(&v.codec).ok_or_else(|| {
                                    FerricastError::Decode(format!(
                                        "no video decoder registered for codec {:?}",
                                        v.codec
                                    ))
                                })?;
                                let mut d = (f)();
                                d.configure(&DecoderConfig {
                                    codec: v.codec,
                                    width: v.width,
                                    height: v.height,
                                    // NV12 is what every GPU decoder
                                    // emits natively (VA-API VPP, NVDEC
                                    // surface format); CPU fallbacks
                                    // can convert if needed.
                                    pixel_format: PixelFormat::Nv12,
                                })?;
                                Some(d)
                            }
                            None => None,
                        };
                        let audio_dec = match &info.audio {
                            Some(a) => {
                                let f = audio_decoders.get(&a.codec).ok_or_else(|| {
                                    FerricastError::Decode(format!(
                                        "no audio decoder registered for codec {:?}",
                                        a.codec
                                    ))
                                })?;
                                let mut d = (f)();
                                d.configure(&AudioDecoderConfig {
                                    codec: a.codec,
                                    sample_rate: a.sample_rate,
                                    channels: a.channels,
                                })?;
                                Some(d)
                            }
                            None => None,
                        };

                        let (ptx, prx) = mpsc::channel(8);
                        let ev = event_tx.clone();
                        let h = tokio::spawn(run_pump(
                            receiver_id,
                            puller,
                            video_dec,
                            audio_dec,
                            sink,
                            prx,
                            ev,
                            autoplay,
                        ));
                        pump = Some(h);
                        pump_tx = Some(ptx);

                        let state = if autoplay {
                            PlaybackState::Playing
                        } else {
                            PlaybackState::Paused
                        };
                        let _ = control.report_state(state.clone()).await;
                        let _ = event_tx
                            .send(ManagerEvent::ReceiverStateChanged {
                                receiver_id,
                                state,
                            })
                            .await;
                    }
                    MediaCommand::Play => {
                        if let Some(tx) = pump_tx.as_ref() {
                            let _ = tx.send(PumpCmd::Play).await;
                        }
                        let _ = control.report_state(PlaybackState::Playing).await;
                        let _ = event_tx
                            .send(ManagerEvent::ReceiverStateChanged {
                                receiver_id,
                                state: PlaybackState::Playing,
                            })
                            .await;
                    }
                    MediaCommand::Pause => {
                        if let Some(tx) = pump_tx.as_ref() {
                            let _ = tx.send(PumpCmd::Pause).await;
                        }
                        let _ = control.report_state(PlaybackState::Paused).await;
                        let _ = event_tx
                            .send(ManagerEvent::ReceiverStateChanged {
                                receiver_id,
                                state: PlaybackState::Paused,
                            })
                            .await;
                    }
                    MediaCommand::Stop => {
                        if let Some(tx) = pump_tx.take() {
                            let _ = tx.send(PumpCmd::Stop).await;
                        }
                        if let Some(h) = pump.take() {
                            let _ = h.await;
                        }
                        let _ = control.report_state(PlaybackState::Idle).await;
                        let _ = event_tx
                            .send(ManagerEvent::ReceiverStateChanged {
                                receiver_id,
                                state: PlaybackState::Idle,
                            })
                            .await;
                    }
                    MediaCommand::Seek { position_us } => {
                        if let Some(tx) = pump_tx.as_ref() {
                            let _ = tx.send(PumpCmd::Seek(position_us)).await;
                        }
                    }
                    MediaCommand::GetStatus => {
                        // Receiver protocol crates handle their own
                        // status reporting cadence; the manager just
                        // acknowledges the command was received and
                        // moves on.
                    }
                    other => {
                        // Queue ops, volume, track selection, app
                        // launch — wire these through as the protocol
                        // crates need them. Logged for diagnostic
                        // visibility; not an error.
                        tracing::debug!(
                            protocol,
                            ?other,
                            "receiver: command accepted but not yet routed"
                        );
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_pump(
    receiver_id: Uuid,
    mut puller: Box<dyn ErasedPuller>,
    mut video_dec: Option<Box<dyn ErasedVideoDecoder>>,
    mut audio_dec: Option<Box<dyn ErasedAudioDecoder>>,
    mut sink: Box<dyn FrameSink>,
    mut cmd_rx: mpsc::Receiver<PumpCmd>,
    event_tx: mpsc::Sender<ManagerEvent>,
    autoplay: bool,
) {
    let mut playing = autoplay;
    loop {
        tokio::select! {
            cmd = cmd_rx.recv() => match cmd {
                Some(PumpCmd::Play) => playing = true,
                Some(PumpCmd::Pause) => playing = false,
                Some(PumpCmd::Seek(pos)) => {
                    if let Err(e) = puller.seek(pos).await {
                        tracing::warn!(%e, "pump: seek failed");
                    }
                }
                Some(PumpCmd::Stop) | None => break,
            },
            packet = puller.next(), if playing => match packet {
                Ok(MediaPacket::Video(f)) => {
                    let Some(d) = video_dec.as_mut() else { continue };
                    match d.decode(f) {
                        Ok(Some(frame)) => {
                            if let Err(e) = sink.push_video(frame).await {
                                tracing::warn!(%e, "pump: sink.push_video failed");
                            }
                        }
                        Ok(None) => {}
                        Err(e) => tracing::warn!(%e, "pump: video decode failed"),
                    }
                }
                Ok(MediaPacket::Audio(a)) => {
                    let Some(d) = audio_dec.as_mut() else { continue };
                    match d.decode(a) {
                        Ok(Some(pcm)) => {
                            if let Err(e) = sink.push_audio(pcm).await {
                                tracing::warn!(%e, "pump: sink.push_audio failed");
                            }
                        }
                        Ok(None) => {}
                        Err(e) => tracing::warn!(%e, "pump: audio decode failed"),
                    }
                }
                Ok(MediaPacket::Eos) => {
                    let _ = event_tx
                        .send(ManagerEvent::ReceiverStateChanged {
                            receiver_id,
                            state: PlaybackState::Ended,
                        })
                        .await;
                    break;
                }
                Err(e) => {
                    tracing::error!(%e, "pump: puller error");
                    let _ = event_tx
                        .send(ManagerEvent::ReceiverError {
                            receiver_id,
                            message: e.to_string(),
                        })
                        .await;
                    break;
                }
            }
        }
    }
    let _ = puller.close().await;
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

    /// Finalize the builder and return the manager. The event
    /// receiver stays inside the manager; call
    /// [`StreamManager::take_event_rx`] when you need it.
    pub fn build(self) -> StreamManager {
        self.manager
    }

    /// Finalize and split out the event receiver in one step — the
    /// common shape for apps that wrap the manager in `Arc<Mutex<_>>`
    /// and need the rx accessible from a separate task.
    pub fn build_with_events(mut self) -> (StreamManager, tokio::sync::mpsc::Receiver<ManagerEvent>)
    {
        let rx = self
            .manager
            .take_event_rx()
            .expect("freshly built manager always has its event_rx");
        (self.manager, rx)
    }
}
