//! Receiver side of the manager: advertise this process as a cast
//! target, accept control connections from remote senders, and run
//! the pull → decode → sink pipeline for each LOAD.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinHandle;
use uuid::Uuid;

use ferricast_core::{
    AdvertiseInfo, Advertiser, AudioCodec, AudioDecoder, AudioDecoderConfig, AudioFrame,
    CapturedFrame, Codec, ControlSession, DecodedAudio, DecoderConfig, EncodedFrame,
    FerricastError, FrameSink, MediaCommand, MediaInfo, MediaPacket, MediaPuller, PixelFormat,
    PlaybackState, PullSpec, ReceiverProtocol, RemoteSender, Result, VideoDecoder,
};

use crate::manager::ManagerEvent;

type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

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

/// Receiver-side state of the [`crate::StreamManager`]: registered
/// receiver protocols, decoder factories, the sink factory and the
/// running accept supervisors.
pub(crate) struct ReceiverHost {
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

impl Default for ReceiverHost {
    fn default() -> Self {
        Self {
            receivers: Vec::new(),
            video_decoders: HashMap::new(),
            audio_decoders: HashMap::new(),
            sink_factory: None,
            active_receivers: Arc::new(Mutex::new(HashMap::new())),
            receivers_running: false,
        }
    }
}

impl ReceiverHost {
    /// Register a receiver-side protocol (advertiser + control +
    /// puller). Mirror of [`Self::register`] for senders.
    pub(crate) fn register_receiver<P>(&mut self)
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
    pub(crate) fn register_video_decoder<D>(&mut self)
    where
        D: VideoDecoder + Default + 'static,
    {
        self.video_decoders.insert(
            D::CODEC,
            Arc::new(|| Box::new(D::default()) as Box<dyn ErasedVideoDecoder>),
        );
    }

    pub(crate) fn register_audio_decoder<D>(&mut self)
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
    pub(crate) fn set_sink_factory<F, Fut>(&mut self, factory: F)
    where
        F: Fn(&RemoteSender, &MediaInfo) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Box<dyn FrameSink>>> + Send + 'static,
    {
        self.sink_factory = Some(Arc::new(move |remote, info| {
            let fut = factory(remote, info);
            Box::pin(async move { fut.await })
        }));
    }

    pub(crate) fn registered_receivers(&self) -> Vec<&'static str> {
        self.receivers.iter().map(|r| r.protocol).collect()
    }

    /// Start every registered receiver protocol's advertiser and
    /// per-protocol accept supervisor. Requires
    /// [`Self::set_sink_factory`] to have been called.
    pub(crate) async fn start_receivers(
        &mut self,
        event_tx: &mpsc::Sender<ManagerEvent>,
    ) -> Result<()> {
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
            let event_tx = event_tx.clone();
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

    pub(crate) async fn stop_receivers(&mut self) -> Result<()> {
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
                Ok(MediaPacket::Discontinuity) => {}
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
