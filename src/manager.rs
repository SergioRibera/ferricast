use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Mutex, RwLock, mpsc};
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::stream::{CancelRx, ErasedSession, SessionFactory, StreamEnd, StreamRequest};
use ferricast_core::{
    AdvertiseInfo, Advertiser, AudioCodec, AudioDecoder, AudioDecoderConfig, AudioFrame,
    CaptureSource, CapturedFrame, Codec, ControlSession, DecodedAudio, DecoderConfig, Device,
    Discovery, DiscoveryEvent, EncodedFrame, FerricastError, FrameSink, MediaCommand, MediaInfo,
    MediaPacket, MediaPuller, PairingChallenge, PairingResponse, PixelFormat, PlaybackState,
    ProtocolHandler, PullSpec, ReceiverProtocol, RemoteSender, Result, ScreenCapture, StreamConfig,
    VideoDecoder, VideoEncoder,
};

type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// How long `stop_stream` waits for a stream task to clean up before
/// aborting it.
const STOP_TIMEOUT: Duration = Duration::from_secs(5);

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
    /// Takes `&Device` so the factory can pick the right session type
    /// per-device (e.g. Cast Streaming vs. HLS on Chromecast).
    create_session: SessionFactory,
}

/// One stream, from the moment its start is accepted until its task
/// has cleaned up.
struct ActiveStream {
    device: Device,
    /// Distinguishes this stream from a later one on the same device,
    /// so a finishing task only ever removes its own entry.
    generation: u64,
    /// Dropping it (or sending on it) stops the stream, during
    /// start-up as well as while running.
    cancel_tx: mpsc::Sender<()>,
    /// `None` while the stream is still starting.
    task: Option<JoinHandle<()>>,
}

/// Streams by device. Behind a `std` mutex: every critical section is
/// short and never awaits.
#[derive(Default)]
struct StreamRegistry {
    next_generation: u64,
    streams: HashMap<Uuid, ActiveStream>,
}

impl StreamRegistry {
    /// Claim `device` for a new stream.
    fn reserve(&mut self, device: &Device) -> Result<(u64, CancelRx)> {
        if self.streams.contains_key(&device.id) {
            return Err(FerricastError::SessionAlreadyActive(device.name.clone()));
        }
        self.next_generation += 1;
        let generation = self.next_generation;
        let (cancel_tx, cancel_rx) = mpsc::channel(1);
        self.streams.insert(
            device.id,
            ActiveStream {
                device: device.clone(),
                generation,
                cancel_tx,
                task: None,
            },
        );
        Ok((generation, CancelRx::new(cancel_rx)))
    }

    fn is_current(&self, id: Uuid, generation: u64) -> bool {
        self.streams
            .get(&id)
            .is_some_and(|s| s.generation == generation)
    }

    /// Attach the running task to its reservation; `false` when the
    /// reservation is gone (stopped while starting).
    fn attach(&mut self, id: Uuid, generation: u64, task: JoinHandle<()>) -> bool {
        match self.streams.get_mut(&id) {
            Some(s) if s.generation == generation => {
                s.task = Some(task);
                true
            }
            _ => false,
        }
    }

    /// Remove the entry only if it still belongs to `generation`.
    fn release(&mut self, id: Uuid, generation: u64) -> bool {
        if self.is_current(id, generation) {
            self.streams.remove(&id);
            true
        } else {
            false
        }
    }

    fn take(&mut self, id: Uuid) -> Option<ActiveStream> {
        self.streams.remove(&id)
    }

    fn devices(&self) -> Vec<Device> {
        self.streams.values().map(|s| s.device.clone()).collect()
    }

    fn ids(&self) -> Vec<Uuid> {
        self.streams.keys().copied().collect()
    }
}

fn lock_registry(
    registry: &std::sync::Mutex<StreamRegistry>,
) -> std::sync::MutexGuard<'_, StreamRegistry> {
    // The registry holds plain bookkeeping that stays consistent
    // even if a holder panicked.
    registry
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Releases a stream reservation if start-up does not get as far as
/// handing it to a running task (error, cancellation, or the
/// `start_stream` future being dropped).
struct Reservation {
    registry: Arc<std::sync::Mutex<StreamRegistry>>,
    id: Uuid,
    generation: u64,
    armed: bool,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if self.armed {
            lock_registry(&self.registry).release(self.id, self.generation);
        }
    }
}

#[derive(Default)]
struct DiscoveryState {
    handles: Vec<Box<dyn ErasedDiscovery>>,
    running: bool,
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
        /// Why the receiver ended it, when it did (another app took
        /// over, the user pressed "stop casting" on the TV). `None`
        /// when stopped from this side or after a `StreamError`.
        reason: Option<String>,
    },
    /// A stream start was accepted; capture (and possibly the portal
    /// dialog), the device handshake and pairing are in progress.
    /// Ends with `StreamStarted` or `StreamStopped`.
    StreamStarting {
        device_id: Uuid,
    },
    /// Non-fatal problem on a running stream (e.g. audio could not be
    /// captured or encoded; video continues).
    StreamWarning {
        device_id: Uuid,
        message: String,
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
    streams: Arc<std::sync::Mutex<StreamRegistry>>,
    discovery: Mutex<DiscoveryState>,
    event_tx: mpsc::Sender<ManagerEvent>,
    event_rx: Option<mpsc::Receiver<ManagerEvent>>,

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
            streams: Arc::default(),
            discovery: Mutex::default(),
            event_tx,
            event_rx: Some(event_rx),
            receivers: Vec::new(),
            video_decoders: HashMap::new(),
            audio_decoders: HashMap::new(),
            sink_factory: None,
            active_receivers: Arc::new(Mutex::new(HashMap::new())),
            receivers_running: false,
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
            Box::pin(fut)
        }));
    }

    pub fn registered_receivers(&self) -> Vec<&'static str> {
        self.receivers.iter().map(|r| r.protocol).collect()
    }

    /// Start discovery on every registered protocol. Idempotent.
    ///
    /// # Errors
    ///
    /// Never fails today: a protocol whose discovery can't start is
    /// skipped and reported as [`ManagerEvent::DiscoveryError`].
    pub async fn start_discovery(&self) -> Result<()> {
        let mut state = self.discovery.lock().await;
        if state.running {
            return Ok(());
        }

        let (disc_tx, mut disc_rx) = mpsc::channel::<DiscoveryEvent>(256);

        for proto in &self.protocols {
            let mut discovery = (proto.create_discovery)();
            match discovery.start(disc_tx.clone()).await {
                Ok(()) => state.handles.push(discovery),
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

        state.running = true;
        Ok(())
    }

    /// Stop every protocol's discovery and forget the known devices
    /// (each one is reported as [`ManagerEvent::DeviceLost`]).
    ///
    /// # Errors
    ///
    /// The first discovery stop error; the others are still stopped
    /// and logged.
    pub async fn stop_discovery(&self) -> Result<()> {
        let mut state = self.discovery.lock().await;
        let mut first_error = None;
        for mut discovery in state.handles.drain(..) {
            if let Err(e) = discovery.stop().await {
                tracing::warn!(%e, "discovery stop failed");
                first_error.get_or_insert(e);
            }
        }
        state.running = false;
        drop(state);

        let lost: Vec<Uuid> = self
            .devices
            .write()
            .await
            .drain()
            .map(|(id, _)| id)
            .collect();
        for id in lost {
            let _ = self.event_tx.send(ManagerEvent::DeviceLost(id)).await;
        }
        first_error.map_or(Ok(()), Err)
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
                Err(_) => {
                    tracing::warn!(protocol = r.protocol, "receiver task did not finish in 5s")
                }
            }
        }
        self.receivers_running = false;
        Ok(())
    }

    pub async fn devices(&self) -> Vec<Device> {
        self.devices.read().await.values().cloned().collect()
    }

    /// Devices with a stream that is starting or running.
    pub fn active_streams(&self) -> Vec<Device> {
        lock_registry(&self.streams).devices()
    }

    /// Start streaming `source` to `device_id`.
    ///
    /// Resolves once the stream is up (capture running, encoder sized
    /// to the device, receiver session set up); the stream then runs
    /// on its own task until [`Self::stop_stream`], a receiver-side
    /// stop, or an unrecoverable error. Progress and the outcome are
    /// also published as `StreamStarting` → `StreamStarted` /
    /// `StreamError` → `StreamStopped` events. A `stop_stream` while
    /// starting aborts the start.
    ///
    /// # Errors
    ///
    /// [`FerricastError::SessionAlreadyActive`] when the device already
    /// has a stream, [`FerricastError::DeviceNotFound`] for an unknown
    /// device, otherwise whatever capture, encoder or the receiver
    /// handshake failed with.
    pub async fn start_stream(
        &self,
        device_id: Uuid,
        source: CaptureSource,
        capture: impl ScreenCapture + 'static,
        encoder: impl VideoEncoder + 'static,
        mut config: StreamConfig,
    ) -> Result<()> {
        if config.port_range.is_none() {
            config.port_range = self.port_range;
        }
        let device = self
            .devices
            .read()
            .await
            .get(&device_id)
            .cloned()
            .ok_or_else(|| FerricastError::DeviceNotFound(device_id.to_string()))?;
        let proto = self
            .protocols
            .iter()
            .find(|p| p.protocol == device.protocol)
            .ok_or_else(|| {
                FerricastError::Protocol(format!("No handler for {}", device.protocol))
            })?;

        let (generation, mut cancel) = lock_registry(&self.streams).reserve(&device)?;
        let mut reservation = Reservation {
            registry: self.streams.clone(),
            id: device_id,
            generation,
            armed: true,
        };
        let _ = self
            .event_tx
            .send(ManagerEvent::StreamStarting { device_id })
            .await;

        let request = StreamRequest {
            device,
            source,
            capture,
            encoder,
            config,
            create_session: proto.create_session.clone(),
            event_tx: self.event_tx.clone(),
        };
        let prepared = tokio::select! {
            biased;
            () = cancel.cancelled() => None,
            r = request.prepare() => Some(r),
        };
        let task = match prepared {
            Some(Ok(task)) => task,
            Some(Err(e)) => {
                drop(reservation);
                StreamEnd::Failed(e.to_string())
                    .report(&self.event_tx, device_id)
                    .await;
                return Err(e);
            }
            None => {
                drop(reservation);
                StreamEnd::Cancelled.report(&self.event_tx, device_id).await;
                return Err(FerricastError::Streaming(
                    "stream stopped while starting".into(),
                ));
            }
        };

        // Spawn and attach under the registry lock so the task can't
        // finish and release its entry before it is attached.
        let attached = {
            let mut registry = lock_registry(&self.streams);
            registry.is_current(device_id, generation) && {
                let streams = self.streams.clone();
                let event_tx = self.event_tx.clone();
                let handle = tokio::spawn(async move {
                    let end = task.run(cancel).await;
                    lock_registry(&streams).release(device_id, generation);
                    end.report(&event_tx, device_id).await;
                });
                registry.attach(device_id, generation, handle)
            }
        };
        if !attached {
            drop(reservation);
            StreamEnd::Cancelled.report(&self.event_tx, device_id).await;
            return Err(FerricastError::Streaming(
                "stream stopped while starting".into(),
            ));
        }
        reservation.armed = false;
        Ok(())
    }

    /// Stop the stream on `device_id` (also aborts one that is still
    /// starting) and wait for its cleanup, at most 5 s.
    ///
    /// # Errors
    ///
    /// [`FerricastError::NoActiveSession`] when the device has no
    /// stream.
    pub async fn stop_stream(&self, device_id: Uuid) -> Result<()> {
        let stream = lock_registry(&self.streams)
            .take(device_id)
            .ok_or(FerricastError::NoActiveSession)?;
        tracing::info!(name = %stream.device.name, "Stopping stream");
        // Dropping the sender is the stop signal.
        drop(stream.cancel_tx);
        let Some(task) = stream.task else {
            // Still starting: `start_stream` sees the signal, cleans up
            // and reports the outcome itself.
            return Ok(());
        };
        let abort = task.abort_handle();
        match tokio::time::timeout(STOP_TIMEOUT, task).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::warn!(?e, "stream task panicked during shutdown"),
            Err(_) => {
                tracing::warn!(name = %stream.device.name, "stream task did not finish in time; aborting");
                abort.abort();
                let _ = self
                    .event_tx
                    .send(ManagerEvent::StreamStopped {
                        device_id,
                        reason: None,
                    })
                    .await;
            }
        }
        Ok(())
    }

    /// Stop every stream.
    pub async fn stop_all_streams(&self) {
        let ids = lock_registry(&self.streams).ids();
        for id in ids {
            let _ = self.stop_stream(id).await;
        }
    }

    /// Stop streams, receivers and discovery.
    ///
    /// # Errors
    ///
    /// The discovery stop error, if any.
    pub async fn shutdown(&mut self) -> Result<()> {
        self.stop_all_streams().await;
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
    audio_decoders: &HashMap<
        AudioCodec,
        Arc<dyn Fn() -> Box<dyn ErasedAudioDecoder> + Send + Sync>,
    >,
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
                                    pixel_format: PixelFormat::Bgra,
                                    low_latency: true,
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

#[cfg(test)]
mod tests {
    use ferricast_core::DeviceCapabilities;

    use super::*;
    use crate::test_support::{FakeCapture, FakeEncoder, SessionScript, device, session_factory};

    struct NoDiscovery;

    impl ErasedDiscovery for NoDiscovery {
        fn start(&mut self, _tx: mpsc::Sender<DiscoveryEvent>) -> BoxFut<'_, Result<()>> {
            Box::pin(async { Ok(()) })
        }
        fn stop(&mut self) -> BoxFut<'_, Result<()>> {
            Box::pin(async { Ok(()) })
        }
    }

    async fn manager_with(scripts: Vec<SessionScript>) -> (StreamManager, Device) {
        let mut manager = StreamManager::default();
        let (factory, _log) = session_factory(scripts);
        manager.protocols.push(RegisteredProtocol {
            protocol: "fake",
            create_discovery: Box::new(|| Box::new(NoDiscovery) as Box<dyn ErasedDiscovery>),
            create_session: factory,
        });
        let device = device(DeviceCapabilities::default());
        manager
            .devices
            .write()
            .await
            .insert(device.id, device.clone());
        (manager, device)
    }

    fn source() -> CaptureSource {
        CaptureSource::FullScreen { monitor: None }
    }

    fn drain(rx: &mut mpsc::Receiver<ManagerEvent>) -> Vec<ManagerEvent> {
        std::iter::from_fn(|| rx.try_recv().ok()).collect()
    }

    #[test]
    fn registry_rejects_a_second_reservation() {
        let mut reg = StreamRegistry::default();
        let d = device(DeviceCapabilities::default());
        assert!(reg.reserve(&d).is_ok());
        assert!(matches!(
            reg.reserve(&d),
            Err(FerricastError::SessionAlreadyActive(_))
        ));
    }

    #[test]
    fn registry_release_only_removes_its_own_generation() {
        let mut reg = StreamRegistry::default();
        let d = device(DeviceCapabilities::default());
        let (old, _rx) = reg.reserve(&d).unwrap();
        assert!(reg.take(d.id).is_some());
        let (new, _rx2) = reg.reserve(&d).unwrap();
        assert_ne!(old, new);
        // The old task finishing must not remove the new stream.
        assert!(!reg.release(d.id, old));
        assert!(reg.is_current(d.id, new));
        assert!(reg.release(d.id, new));
        assert!(reg.devices().is_empty());
    }

    #[tokio::test]
    async fn registry_refuses_to_attach_after_stop() {
        let mut reg = StreamRegistry::default();
        let d = device(DeviceCapabilities::default());
        let (generation, _rx) = reg.reserve(&d).unwrap();
        reg.take(d.id);
        assert!(!reg.attach(d.id, generation, tokio::spawn(async {})));
    }

    #[test]
    fn dropped_reservation_frees_the_device() {
        let registry = Arc::new(std::sync::Mutex::new(StreamRegistry::default()));
        let d = device(DeviceCapabilities::default());
        let (generation, _rx) = lock_registry(&registry).reserve(&d).unwrap();
        drop(Reservation {
            registry: registry.clone(),
            id: d.id,
            generation,
            armed: true,
        });
        assert!(lock_registry(&registry).devices().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn duplicate_start_is_rejected_and_stop_cleans_up() {
        let (mut manager, device) = manager_with(vec![SessionScript::default()]).await;
        let mut events = manager.take_event_rx().unwrap();
        let start = || {
            manager.start_stream(
                device.id,
                source(),
                FakeCapture::new(&[(64, 36)]),
                FakeEncoder::default(),
                StreamConfig::default(),
            )
        };
        start().await.unwrap();
        assert_eq!(manager.active_streams().len(), 1);
        assert!(matches!(
            start().await,
            Err(FerricastError::SessionAlreadyActive(_))
        ));

        manager.stop_stream(device.id).await.unwrap();
        assert!(manager.active_streams().is_empty());
        let events = drain(&mut events);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ManagerEvent::StreamStopped { reason: None, .. }))
        );

        // The device is free again.
        start().await.unwrap();
        manager.stop_all_streams().await;
        assert!(manager.active_streams().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn stop_while_starting_aborts_the_start() {
        let (manager, device) = manager_with(vec![SessionScript::default()]).await;
        let manager = Arc::new(manager);
        let starting = {
            let manager = manager.clone();
            tokio::spawn(async move {
                manager
                    .start_stream(
                        device.id,
                        source(),
                        FakeCapture::new(&[(64, 36)])
                            .with_first_frame_delay(Duration::from_secs(20)),
                        FakeEncoder::default(),
                        StreamConfig::default(),
                    )
                    .await
            })
        };
        tokio::time::sleep(Duration::from_secs(1)).await;
        assert_eq!(manager.active_streams().len(), 1);
        manager.stop_stream(device.id).await.unwrap();
        assert!(starting.await.unwrap().is_err());
        assert!(manager.active_streams().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn remote_stop_frees_the_device_and_reports_the_reason() {
        let (mut manager, device) = manager_with(vec![SessionScript {
            remote_stop_after: Some(3),
            ..SessionScript::default()
        }])
        .await;
        let mut events = manager.take_event_rx().unwrap();
        manager
            .start_stream(
                device.id,
                source(),
                FakeCapture::new(&[(64, 36)]),
                FakeEncoder::default(),
                StreamConfig::default(),
            )
            .await
            .unwrap();
        let reason = loop {
            match events.recv().await {
                Some(ManagerEvent::StreamStopped { reason, .. }) => break reason,
                Some(_) => {}
                None => panic!("event channel closed"),
            }
        };
        assert_eq!(reason.as_deref(), Some("stopped on the TV"));
        assert!(manager.active_streams().is_empty());
        assert!(matches!(
            manager.stop_stream(device.id).await,
            Err(FerricastError::NoActiveSession)
        ));
    }
}
