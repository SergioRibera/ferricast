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
    AudioDecoder, CaptureSource, Device, Discovery, DiscoveryEvent, FerricastError, FrameSink,
    MediaInfo, PairingChallenge, PairingResponse, PlaybackState, ProtocolHandler, ReceiverProtocol,
    RemoteSender, Result, ScreenCapture, StreamConfig, VideoDecoder, VideoEncoder,
};

use crate::receiver::ReceiverHost;

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
    /// The sender connected as `receiver_id` asks to pair with a PIN
    /// (AirPlay `pair-pin-start`): show `Some(pin)` to the user, who types
    /// it on the sender. `None` withdraws a PIN shown before (pairing
    /// finished or was abandoned); the session ending withdraws it too.
    ReceiverPin {
        receiver_id: Uuid,
        pin: Option<String>,
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
    /// [`ferricast_core::ControlSession::report_state`] so the GUI stays in sync
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
            streams: Arc::default(),
            discovery: Mutex::default(),
            event_tx,
            event_rx: Some(event_rx),
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

    /// Register a configured receiver-side protocol instance (custom
    /// port, friendly name, …).
    pub fn register_receiver_instance<P>(&mut self, protocol: P)
    where
        P: ReceiverProtocol + 'static,
    {
        self.receiver.register_receiver_instance(protocol);
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
        self.receiver.start_receivers(&self.event_tx).await
    }

    pub async fn stop_receivers(&mut self) -> Result<()> {
        self.receiver.stop_receivers().await
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

    /// Register the AirPlay *receiver* — advertise this process as an
    /// AirPlay target and play the screen and audio Apple devices (and
    /// ferricast's AirPlay sender) mirror to it. Also registers the audio
    /// decoders AirPlay needs (AAC-ELD with `fdk-aac`, ALAC). Build the
    /// receiver with [`ferricast_airplay::AirPlayReceiver::new`], which
    /// loads its persisted identity.
    #[cfg(feature = "airplay")]
    pub fn with_airplay_receiver(self, receiver: ferricast_airplay::AirPlayReceiver) -> Self {
        let builder = self.with_receiver(receiver);
        #[cfg(feature = "fdk-aac")]
        let builder = builder.register_audio_decoder::<ferricast_decoder::AacEldDecoder>();
        builder.register_audio_decoder::<ferricast_decoder::AlacDecoder>()
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

    /// Register a configured receiver-side protocol instance (custom
    /// port, friendly name, …).
    pub fn with_receiver<P>(mut self, protocol: P) -> Self
    where
        P: ReceiverProtocol + 'static,
    {
        self.manager.register_receiver_instance(protocol);
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
