//! Receiver side of the manager: advertise this process as a cast
//! target, accept control connections from remote senders, and run
//! the pull → decode → sink pipeline for each LOAD.
//!
//! Layout:
//! - this file — type erasure, the registry ([`ReceiverHost`]) and
//!   the per-protocol accept supervisor;
//! - [`session`] — one task per accepted control connection;
//! - [`pump`] — one task per loaded media: pull, decode, pace, sink.

mod pump;
mod session;

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tokio::sync::{Mutex, mpsc, watch};
use tokio::task::{JoinHandle, JoinSet};
use uuid::Uuid;

use ferricast_core::{
    AdvertiseInfo, Advertiser, AudioCodec, AudioDecoder, AudioDecoderConfig, AudioFrame,
    CapturedFrame, Codec, ControlSession, DecodedAudio, DecoderConfig, EncodedFrame, FrameSink,
    MediaCommand, MediaInfo, MediaPacket, MediaPuller, PlaybackStatus, PullSpec, ReceiverProtocol,
    RemoteSender, Result, VideoDecoder,
};

use crate::manager::ManagerEvent;

type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

// ──────────────────────────────────────────────────────────────────
// Receiver-side erasure: the native traits use RPITIT
// (`impl Future + Send`), which isn't object-safe, so the manager
// talks to them through these `ErasedX` shims.

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
    fn report_status(&mut self, status: PlaybackStatus) -> BoxFut<'_, Result<()>>;
    fn close(&mut self) -> BoxFut<'_, Result<()>>;
}

impl<T: ControlSession> ErasedControlSession for T {
    fn accept(&mut self) -> BoxFut<'_, Result<RemoteSender>> {
        Box::pin(ControlSession::accept(self))
    }
    fn next_command(&mut self) -> BoxFut<'_, Result<MediaCommand>> {
        Box::pin(ControlSession::next_command(self))
    }
    fn report_status(&mut self, status: PlaybackStatus) -> BoxFut<'_, Result<()>> {
        Box::pin(ControlSession::report_status(self, status))
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

trait ErasedVideoDecoder: Send {
    fn configure(&mut self, config: &DecoderConfig) -> Result<()>;
    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>>;
    fn flush(&mut self) -> Result<Vec<CapturedFrame>>;
}

impl<T: VideoDecoder> ErasedVideoDecoder for T {
    fn configure(&mut self, config: &DecoderConfig) -> Result<()> {
        VideoDecoder::configure(self, config)
    }
    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        VideoDecoder::decode(self, frame)
    }
    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        VideoDecoder::flush(self)
    }
}

trait ErasedAudioDecoder: Send {
    fn configure(&mut self, config: &AudioDecoderConfig) -> Result<()>;
    fn decode(&mut self, frame: AudioFrame) -> Result<Option<DecodedAudio>>;
    fn flush(&mut self) -> Result<Vec<DecodedAudio>>;
}

impl<T: AudioDecoder> ErasedAudioDecoder for T {
    fn configure(&mut self, config: &AudioDecoderConfig) -> Result<()> {
        AudioDecoder::configure(self, config)
    }
    fn decode(&mut self, frame: AudioFrame) -> Result<Option<DecodedAudio>> {
        AudioDecoder::decode(self, frame)
    }
    fn flush(&mut self) -> Result<Vec<DecodedAudio>> {
        AudioDecoder::flush(self)
    }
}

// Factory used by the receiver pipeline to ask the application
// (typically the GUI) for somewhere to send decoded frames. Called
// once per LOAD, after the puller has reported `MediaInfo` and the
// decoders are resolved — that way the app can open an audio-only
// card view vs. a video window based on `info.video.is_some()`, and
// nothing opens for media that cannot play.
type SinkFactory = Arc<
    dyn Fn(&RemoteSender, &MediaInfo) -> BoxFut<'static, Result<Box<dyn FrameSink>>> + Send + Sync,
>;
type ControlFactory = Arc<dyn Fn() -> Result<Box<dyn ErasedControlSession>> + Send + Sync>;
type PullerFactory = Arc<dyn Fn() -> Result<Box<dyn ErasedPuller>> + Send + Sync>;
type VideoDecoderFactory = Arc<dyn Fn() -> Box<dyn ErasedVideoDecoder> + Send + Sync>;
type AudioDecoderFactory = Arc<dyn Fn() -> Box<dyn ErasedAudioDecoder> + Send + Sync>;

struct RegisteredReceiver {
    protocol: &'static str,
    create_advertiser: Box<dyn Fn() -> Box<dyn ErasedAdvertiser> + Send + Sync>,
    create_control: ControlFactory,
    create_puller: PullerFactory,
    advertise_info: AdvertiseInfo,
}

struct ActiveReceiver {
    protocol: &'static str,
    cancel_tx: mpsc::Sender<()>,
    task: JoinHandle<()>,
    advertiser: Box<dyn ErasedAdvertiser>,
}

/// How long `stop_receivers` waits for a supervisor (and the
/// sessions it winds down) before aborting it.
const SUPERVISOR_STOP_TIMEOUT: Duration = Duration::from_secs(5);
/// How long a supervisor gives its sessions to stop cleanly.
const SESSION_STOP_TIMEOUT: Duration = Duration::from_secs(3);
const ACCEPT_BACKOFF_START: Duration = Duration::from_millis(100);
const ACCEPT_BACKOFF_MAX: Duration = Duration::from_secs(5);

/// Receiver-side state of the [`crate::StreamManager`]: registered
/// receiver protocols, decoder factories, the sink factory and the
/// running accept supervisors.
#[derive(Default)]
pub(crate) struct ReceiverHost {
    receivers: Vec<RegisteredReceiver>,
    /// Codec → factory map. Pipeline picks a video decoder based on
    /// what the puller's `MediaInfo` declares; multiple decoders
    /// MAY register for the same codec (e.g. VA-API + NVDEC + sw)
    /// and the last one wins — register the preferred backend last,
    /// or use the builder's `with_*_decoder` helpers which append
    /// in priority order.
    video_decoders: HashMap<Codec, VideoDecoderFactory>,
    audio_decoders: HashMap<AudioCodec, AudioDecoderFactory>,
    sink_factory: Option<SinkFactory>,
    active_receivers: Arc<Mutex<HashMap<Uuid, ActiveReceiver>>>,
    receivers_running: bool,
    /// Sessions alive across every protocol.
    sessions: Arc<AtomicUsize>,
}

/// Everything a session needs to run LOADs, shared by all sessions
/// of one receiver protocol.
struct PipelineCtx {
    protocol: &'static str,
    create_puller: PullerFactory,
    video_decoders: HashMap<Codec, VideoDecoderFactory>,
    audio_decoders: HashMap<AudioCodec, AudioDecoderFactory>,
    sink_factory: SinkFactory,
    event_tx: mpsc::Sender<ManagerEvent>,
    /// Which session currently owns playback (Cast semantics: the
    /// latest LOAD wins, the previous one is interrupted).
    playback: session::PlaybackOwner,
    /// Sessions alive across every protocol.
    sessions: Arc<AtomicUsize>,
}

impl ReceiverHost {
    /// Register a receiver-side protocol built with `P::default()`.
    pub(crate) fn register_receiver<P>(&mut self)
    where
        P: ReceiverProtocol + Default + 'static,
    {
        self.register_receiver_instance(P::default());
    }

    /// Register a configured receiver-side protocol instance
    /// (advertiser + control + puller).
    pub(crate) fn register_receiver_instance<P>(&mut self, protocol: P)
    where
        P: ReceiverProtocol + 'static,
    {
        let p = Arc::new(protocol);
        let info = p.advertise_info();
        let (p_adv, p_ctrl, p_pull) = (p.clone(), p.clone(), p);
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
    /// `D::CODEC`. Last registration for a given codec wins.
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
    /// application for a [`FrameSink`] whenever a LOAD resolves its
    /// `MediaInfo`. Required before [`Self::start_receivers`].
    pub(crate) fn set_sink_factory<F, Fut>(&mut self, factory: F)
    where
        F: Fn(&RemoteSender, &MediaInfo) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Box<dyn FrameSink>>> + Send + 'static,
    {
        self.sink_factory = Some(Arc::new(move |remote, info| {
            let fut = factory(remote, info);
            Box::pin(fut)
        }));
    }

    pub(crate) fn session_count(&self) -> usize {
        self.sessions.load(Ordering::Relaxed)
    }

    pub(crate) fn registered_receivers(&self) -> Vec<&'static str> {
        self.receivers.iter().map(|r| r.protocol).collect()
    }

    /// Bind, advertise and start the accept supervisor of every
    /// registered receiver protocol. A protocol whose control port
    /// cannot be bound is reported with [`ManagerEvent::ReceiverError`]
    /// and not advertised.
    pub(crate) async fn start_receivers(
        &mut self,
        event_tx: &mpsc::Sender<ManagerEvent>,
    ) -> Result<()> {
        if self.receivers_running {
            return Ok(());
        }
        let sink_factory = self.sink_factory.clone().ok_or_else(|| {
            ferricast_core::FerricastError::Receiver(
                "start_receivers called before set_sink_factory — \
                 the pipeline has nowhere to send decoded frames"
                    .into(),
            )
        })?;

        for proto in &self.receivers {
            let receiver_id = Uuid::new_v4();
            let protocol = proto.protocol;

            // Bind before advertising, so a taken port never ends up
            // advertised with nothing behind it.
            let first_control = match (proto.create_control)() {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!(protocol, %e, "receiver control could not start; not advertising");
                    let _ = event_tx
                        .send(ManagerEvent::ReceiverError {
                            receiver_id,
                            message: e.to_string(),
                        })
                        .await;
                    continue;
                }
            };

            let mut advertiser = (proto.create_advertiser)();
            if let Err(e) = advertiser.start(proto.advertise_info.clone()).await {
                tracing::warn!(protocol, %e, "receiver advertiser failed to start; skipping");
                continue;
            }
            tracing::info!(
                protocol,
                friendly_name = %proto.advertise_info.friendly_name,
                port = proto.advertise_info.port,
                "receiver advertised"
            );

            let ctx = Arc::new(PipelineCtx {
                protocol,
                create_puller: proto.create_puller.clone(),
                video_decoders: self.video_decoders.clone(),
                audio_decoders: self.audio_decoders.clone(),
                sink_factory: sink_factory.clone(),
                event_tx: event_tx.clone(),
                playback: session::PlaybackOwner::default(),
                sessions: self.sessions.clone(),
            });
            let (cancel_tx, cancel_rx) = mpsc::channel::<()>(1);
            let supervisor = Supervisor {
                ctx,
                create_control: proto.create_control.clone(),
                cancel_rx,
            };
            let task = tokio::spawn(supervisor.run(first_control));

            self.active_receivers.lock().await.insert(
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
        let active = std::mem::take(&mut *self.active_receivers.lock().await);
        for (_, mut r) in active {
            let _ = r.cancel_tx.send(()).await;
            let _ = r.advertiser.stop().await;
            match tokio::time::timeout(SUPERVISOR_STOP_TIMEOUT, &mut r.task).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::warn!(?e, "receiver task panicked"),
                Err(_) => {
                    tracing::warn!(
                        protocol = r.protocol,
                        "receiver task did not stop in time; aborting"
                    );
                    r.task.abort();
                }
            }
        }
        self.receivers_running = false;
        Ok(())
    }
}

/// Accept loop of one receiver protocol: every accepted control
/// connection gets its own session task, so several senders can be
/// connected at once.
struct Supervisor {
    ctx: Arc<PipelineCtx>,
    create_control: ControlFactory,
    cancel_rx: mpsc::Receiver<()>,
}

impl Supervisor {
    async fn run(mut self, first_control: Box<dyn ErasedControlSession>) {
        let protocol = self.ctx.protocol;
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let mut sessions = JoinSet::new();
        let mut control = Some(first_control);
        let mut backoff = ACCEPT_BACKOFF_START;

        loop {
            let Some(ctrl) = control.as_mut() else {
                // Waiting to retry creating a control.
                tokio::select! {
                    _ = self.cancel_rx.recv() => break,
                    () = tokio::time::sleep(backoff) => {}
                }
                match (self.create_control)() {
                    Ok(c) => control = Some(c),
                    Err(e) => {
                        tracing::warn!(protocol, %e, "create_control failed; retrying");
                        backoff = (backoff * 2).min(ACCEPT_BACKOFF_MAX);
                    }
                }
                continue;
            };
            tokio::select! {
                _ = self.cancel_rx.recv() => break,
                Some(joined) = sessions.join_next() => {
                    if let Err(e) = joined {
                        tracing::warn!(protocol, ?e, "receiver session task failed");
                    }
                }
                accepted = ctrl.accept() => match accepted {
                    Ok(remote) => {
                        backoff = ACCEPT_BACKOFF_START;
                        let accepted_control = control.take().expect("invariant: matched Some above");
                        let session = session::Session::new(
                            self.ctx.clone(),
                            accepted_control,
                            remote,
                            shutdown_rx.clone(),
                        );
                        sessions.spawn(session.run());
                        match (self.create_control)() {
                            Ok(c) => control = Some(c),
                            Err(e) => {
                                tracing::warn!(protocol, %e, "create_control failed; retrying");
                            }
                        }
                    }
                    Err(e) => {
                        // A failed handshake or a listener error; back
                        // off so a persistent failure cannot spin.
                        tracing::warn!(protocol, %e, retry_in = ?backoff, "control.accept failed");
                        tokio::select! {
                            _ = self.cancel_rx.recv() => break,
                            () = tokio::time::sleep(backoff) => {}
                        }
                        backoff = (backoff * 2).min(ACCEPT_BACKOFF_MAX);
                    }
                },
            }
        }

        // Ask sessions to stop their pumps and close; abort laggards.
        let _ = shutdown_tx.send(true);
        let drained = tokio::time::timeout(SESSION_STOP_TIMEOUT, async {
            while sessions.join_next().await.is_some() {}
        })
        .await;
        if drained.is_err() {
            tracing::warn!(protocol, "receiver sessions did not stop in time; aborting");
            sessions.shutdown().await;
        }
    }
}
