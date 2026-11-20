//! One control connection: reads the sender's commands, loads media
//! (in the background, so the connection stays responsive), runs the
//! pump and reports playback state back to the sender.
//!
//! Every exit path stops the pump, and a LOAD that fails is reported
//! to the sender (`LOAD_FAILED`) without dropping the connection.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use uuid::Uuid;

use ferricast_core::{
    AudioDecoderConfig, DecoderConfig, FerricastError, FrameSink, MediaCommand, MediaInfo,
    PixelFormat, PlaybackState, PlaybackStatus, PullSpec, RemoteSender, Result,
};
use ferricast_demuxer::ProgressivePuller;

use super::pump::{self, PumpCmd, PumpEnd, PumpParts};
use super::{
    ErasedAudioDecoder, ErasedControlSession, ErasedPuller, ErasedVideoDecoder, PipelineCtx,
};
use crate::manager::ManagerEvent;

/// Upper bound for resolving a LOAD (playlist + first segment +
/// decoder setup + sink). The puller's own retries end well before.
const LOAD_TIMEOUT: Duration = Duration::from_secs(60);

/// Messages one session sends another.
enum Notice {
    /// Another session loaded media: stop playing.
    Preempted,
}

/// Tracks which session owns playback. Cast semantics: the latest
/// LOAD wins and the session it replaced is told to stop.
#[derive(Default)]
pub(super) struct PlaybackOwner {
    current: Mutex<Option<(u64, mpsc::Sender<Notice>)>>,
}

static NEXT_SESSION_SEQ: AtomicU64 = AtomicU64::new(1);

impl PlaybackOwner {
    fn claim(&self, seq: u64, notices: &mpsc::Sender<Notice>) {
        let previous = self
            .current
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .replace((seq, notices.clone()));
        if let Some((prev_seq, prev_tx)) = previous
            && prev_seq != seq
        {
            let _ = prev_tx.try_send(Notice::Preempted);
        }
    }

    fn release(&self, seq: u64) {
        let mut current = self.current.lock().unwrap_or_else(PoisonError::into_inner);
        if current.as_ref().is_some_and(|(owner, _)| *owner == seq) {
            *current = None;
        }
    }
}

/// What kind of media a LOAD points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MediaKind {
    Hls,
    Dash,
    SmoothStreaming,
    /// A single progressive file (MP4, WebM, MKV, MP3, …).
    Progressive,
    /// Nothing to go by; HLS is tried.
    Unknown,
}

impl MediaKind {
    fn classify(url: &str, content_type: Option<&str>) -> Self {
        let ct = content_type.unwrap_or("").to_ascii_lowercase();
        let path = url
            .split(['?', '#'])
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        let ext = path
            .rsplit('/')
            .next()
            .and_then(|f| f.rsplit_once('.'))
            .map(|(_, e)| e);
        if ct.contains("mpegurl") || matches!(ext, Some("m3u8" | "m3u")) {
            Self::Hls
        } else if ct.contains("dash+xml") || ext == Some("mpd") {
            Self::Dash
        } else if ct.contains("ms-sstr") || path.contains(".ism/") || ext == Some("ism") {
            Self::SmoothStreaming
        } else if ct.starts_with("video/")
            || ct.starts_with("audio/")
            || matches!(
                ext,
                Some(
                    "mp4" | "m4v" | "m4a" | "mov" | "mkv" | "webm" | "mp3" | "ogg" | "flac" | "wav"
                )
            )
        {
            Self::Progressive
        } else {
            Self::Unknown
        }
    }
}

/// A LOAD as the sender asked for it.
struct LoadRequest {
    url: String,
    content_type: Option<String>,
    start_time_us: Option<u64>,
}

/// A LOAD fully resolved and ready to play.
struct Prepared {
    puller: Box<dyn ErasedPuller>,
    info: MediaInfo,
    /// Media position playback starts from.
    start_us: u64,
    video_dec: Option<Box<dyn ErasedVideoDecoder>>,
    audio_dec: Option<Box<dyn ErasedAudioDecoder>>,
    sink: Box<dyn FrameSink>,
}

/// A running pump. Dropping it aborts the task, so no exit path can
/// leave a pump decoding into a stale sink.
struct PumpHandle {
    cmd_tx: mpsc::Sender<PumpCmd>,
    task: JoinHandle<PumpEnd>,
}

impl Drop for PumpHandle {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// A background LOAD; aborted on drop.
struct LoadingHandle(JoinHandle<Result<Prepared>>);

impl Drop for LoadingHandle {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[derive(Debug, Clone, Copy)]
struct Volume {
    level: f32,
    muted: bool,
}

impl Volume {
    fn gain(self) -> f32 {
        if self.muted { 0.0 } else { self.level }
    }
}

/// How the command loop ended.
enum End {
    /// Sender disconnected, stopped the app, or the receiver shut down.
    Normal,
    Failed(FerricastError),
}

pub(super) struct Session {
    ctx: Arc<PipelineCtx>,
    control: Box<dyn ErasedControlSession>,
    remote: RemoteSender,
    receiver_id: Uuid,
    seq: u64,
    shutdown: watch::Receiver<bool>,
    notice_tx: mpsc::Sender<Notice>,
    notice_rx: mpsc::Receiver<Notice>,
    pump: Option<PumpHandle>,
    loading: Option<LoadingHandle>,
    /// Whether playback should run once the pending LOAD resolves
    /// (LOAD `autoplay`, then PLAY / PAUSE received while loading).
    play_when_loaded: bool,
    /// A SEEK received while the LOAD was still resolving.
    seek_when_loaded: Option<u64>,
    volume: Volume,
    /// The loaded media, kept for status reports after the pump ends.
    media: Option<LoadedMedia>,
}

/// What status reports say about the loaded media.
struct LoadedMedia {
    duration_us: Option<u64>,
    is_live: bool,
    /// Written by the pump.
    position: Arc<AtomicU64>,
}

impl Session {
    pub(super) fn new(
        ctx: Arc<PipelineCtx>,
        control: Box<dyn ErasedControlSession>,
        remote: RemoteSender,
        shutdown: watch::Receiver<bool>,
    ) -> Self {
        let (notice_tx, notice_rx) = mpsc::channel(4);
        Self {
            ctx,
            control,
            remote,
            receiver_id: Uuid::new_v4(),
            seq: NEXT_SESSION_SEQ.fetch_add(1, Ordering::Relaxed),
            shutdown,
            notice_tx,
            notice_rx,
            pump: None,
            loading: None,
            play_when_loaded: true,
            seek_when_loaded: None,
            volume: Volume {
                level: 1.0,
                muted: false,
            },
            media: None,
        }
    }

    pub(super) async fn run(mut self) {
        let protocol = self.ctx.protocol;
        tracing::info!(
            protocol,
            receiver_id = %self.receiver_id,
            sender_addr = %self.remote.addr,
            "receiver: remote sender connected"
        );
        self.event(ManagerEvent::ReceiverIncoming {
            receiver_id: self.receiver_id,
            protocol,
            remote: self.remote.clone(),
        })
        .await;

        let end = self.command_loop().await;

        // Whatever ended the loop, the pump and any pending load go.
        self.loading = None;
        self.stop_pump().await;
        let _ = self.control.close().await;
        if let End::Failed(e) = &end {
            tracing::warn!(protocol, %e, "receiver session ended with error");
            self.event(ManagerEvent::ReceiverError {
                receiver_id: self.receiver_id,
                message: e.to_string(),
            })
            .await;
        }
        self.event(ManagerEvent::ReceiverStopped {
            receiver_id: self.receiver_id,
        })
        .await;
    }

    async fn event(&self, event: ManagerEvent) {
        let _ = self.ctx.event_tx.send(event).await;
    }

    async fn report(&mut self, state: PlaybackState) {
        let status = match (&self.media, &state) {
            (Some(media), _) if !matches!(state, PlaybackState::Error(_)) => PlaybackStatus {
                state: state.clone(),
                position_us: Some(media.position.load(Ordering::Relaxed)),
                duration_us: media.duration_us,
                is_live: media.is_live,
            },
            _ => PlaybackStatus::from(state.clone()),
        };
        if let Err(e) = self.control.report_status(status).await {
            tracing::debug!(%e, "receiver: report_status failed");
        }
        self.event(ManagerEvent::ReceiverStateChanged {
            receiver_id: self.receiver_id,
            state,
        })
        .await;
    }

    async fn command_loop(&mut self) -> End {
        loop {
            tokio::select! {
                changed = self.shutdown.changed() => {
                    if changed.is_err() || *self.shutdown.borrow() {
                        return End::Normal;
                    }
                }
                Some(Notice::Preempted) = self.notice_rx.recv() => {
                    tracing::info!(receiver_id = %self.receiver_id, "receiver: playback taken over by another sender");
                    let had_media = self.loading.take().is_some() | self.stop_pump().await;
                    if had_media {
                        self.report(PlaybackState::Idle).await;
                    }
                }
                loaded = join_loading(&mut self.loading) => {
                    self.loading = None;
                    self.on_loaded(loaded).await;
                }
                ended = join_pump(&mut self.pump) => {
                    self.pump = None;
                    self.ctx.playback.release(self.seq);
                    self.on_pump_end(ended).await;
                }
                cmd = self.control.next_command() => match cmd {
                    Ok(cmd) => {
                        if let Some(end) = self.on_command(cmd).await {
                            return end;
                        }
                    }
                    Err(FerricastError::RemoteStopped(reason)) => {
                        tracing::info!(%reason, "receiver: sender disconnected");
                        return End::Normal;
                    }
                    Err(e) => return End::Failed(e),
                },
            }
        }
    }

    /// Handles one command; `Some` ends the session.
    async fn on_command(&mut self, cmd: MediaCommand) -> Option<End> {
        match cmd {
            MediaCommand::Load {
                url,
                content_type,
                start_time_us,
                autoplay,
                metadata: _,
            } => {
                self.loading = None;
                self.stop_pump().await;
                self.media = None;
                self.play_when_loaded = autoplay;
                self.seek_when_loaded = None;
                self.ctx.playback.claim(self.seq, &self.notice_tx);
                let request = LoadRequest {
                    url,
                    content_type,
                    start_time_us,
                };
                let ctx = self.ctx.clone();
                let remote = self.remote.clone();
                self.loading = Some(LoadingHandle(tokio::spawn(async move {
                    tokio::time::timeout(LOAD_TIMEOUT, load(&ctx, &remote, request))
                        .await
                        .unwrap_or_else(|_| {
                            Err(FerricastError::Timeout(
                                "loading the media took too long".into(),
                            ))
                        })
                })));
            }
            MediaCommand::Play => {
                self.play_when_loaded = true;
                if self.send_pump(PumpCmd::Play).await {
                    self.report(PlaybackState::Playing).await;
                }
            }
            MediaCommand::Pause => {
                self.play_when_loaded = false;
                if self.send_pump(PumpCmd::Pause).await {
                    self.report(PlaybackState::Paused).await;
                }
            }
            MediaCommand::Seek { position_us } => {
                if !self.send_pump(PumpCmd::Seek(position_us)).await && self.loading.is_some() {
                    self.seek_when_loaded = Some(position_us);
                }
            }
            MediaCommand::Stop => {
                let had_media = self.loading.take().is_some() | self.stop_pump().await;
                if had_media {
                    self.report(PlaybackState::Idle).await;
                }
            }
            MediaCommand::StopApp => {
                let had_media = self.loading.take().is_some() | self.stop_pump().await;
                if had_media {
                    self.report(PlaybackState::Idle).await;
                }
            }
            MediaCommand::SetVolume(level) | MediaCommand::SetSystemVolume(level) => {
                self.volume.level = level.clamp(0.0, 1.0);
                self.send_pump(PumpCmd::Volume(self.volume.gain())).await;
            }
            MediaCommand::SetMute(muted) | MediaCommand::SetSystemMute(muted) => {
                self.volume.muted = muted;
                self.send_pump(PumpCmd::Volume(self.volume.gain())).await;
            }
            MediaCommand::GetStatus | MediaCommand::LaunchApp { .. } => {}
            other @ (MediaCommand::QueueLoad { .. }
            | MediaCommand::QueueInsert { .. }
            | MediaCommand::QueueUpdate { .. }
            | MediaCommand::QueueRemove { .. }
            | MediaCommand::QueueReorder { .. }
            | MediaCommand::QueueNext
            | MediaCommand::QueuePrev
            | MediaCommand::QueueJump { .. }
            | MediaCommand::QueueSetRepeat(_)
            | MediaCommand::EditTracks(_)
            | MediaCommand::SetPlaybackRate(_)
            | MediaCommand::Preload { .. }) => {
                tracing::debug!(?other, "receiver: command not supported; ignored");
            }
        }
        None
    }

    async fn on_loaded(&mut self, loaded: Result<Prepared>) {
        match loaded {
            Ok(mut prepared) => {
                if let Some(position_us) = self.seek_when_loaded.take() {
                    match prepared.puller.seek(position_us).await {
                        Ok(()) => prepared.start_us = position_us,
                        Err(e) => {
                            tracing::warn!(%e, "receiver: seek requested while loading failed");
                        }
                    }
                }
                let info = prepared.info.clone();
                tracing::info!(
                    protocol = self.ctx.protocol,
                    ?info,
                    "receiver: media loaded"
                );
                self.event(ManagerEvent::ReceiverStarted {
                    receiver_id: self.receiver_id,
                    remote: self.remote.clone(),
                    info,
                })
                .await;
                let position = Arc::new(AtomicU64::new(prepared.start_us));
                self.media = Some(LoadedMedia {
                    duration_us: prepared.info.duration_us,
                    is_live: prepared.info.is_live,
                    position: position.clone(),
                });
                let (cmd_tx, cmd_rx) = mpsc::channel(16);
                let parts = PumpParts {
                    puller: prepared.puller,
                    video_dec: prepared.video_dec,
                    audio_dec: prepared.audio_dec,
                    sink: prepared.sink,
                    playing: self.play_when_loaded,
                    gain: self.volume.gain(),
                    start_us: prepared.start_us,
                    position,
                };
                self.pump = Some(PumpHandle {
                    cmd_tx,
                    task: tokio::spawn(pump::run(parts, cmd_rx)),
                });
                let state = if self.play_when_loaded {
                    PlaybackState::Playing
                } else {
                    PlaybackState::Paused
                };
                self.report(state).await;
            }
            Err(e) => {
                tracing::warn!(%e, "receiver: LOAD failed");
                self.media = None;
                self.ctx.playback.release(self.seq);
                self.report(PlaybackState::Error(e.to_string())).await;
            }
        }
    }

    async fn on_pump_end(&mut self, ended: PumpEnd) {
        match ended {
            PumpEnd::Finished => self.report(PlaybackState::Ended).await,
            PumpEnd::Stopped => {}
            PumpEnd::Failed(message) => {
                tracing::warn!(%message, "receiver: playback failed");
                self.report(PlaybackState::Error(message)).await;
            }
        }
    }

    /// Sends a command to the pump; `false` when there is none.
    async fn send_pump(&self, cmd: PumpCmd) -> bool {
        match &self.pump {
            Some(pump) => pump.cmd_tx.send(cmd).await.is_ok(),
            None => false,
        }
    }

    /// Stops the pump (cleanly, then by abort) and gives up playback
    /// ownership. Returns whether a pump was running.
    async fn stop_pump(&mut self) -> bool {
        let Some(mut pump) = self.pump.take() else {
            return false;
        };
        let _ = pump.cmd_tx.send(PumpCmd::Stop).await;
        if tokio::time::timeout(Duration::from_secs(2), &mut pump.task)
            .await
            .is_err()
        {
            tracing::warn!("receiver: pump did not stop in time; aborting");
        }
        drop(pump);
        self.ctx.playback.release(self.seq);
        true
    }
}

async fn join_loading(loading: &mut Option<LoadingHandle>) -> Result<Prepared> {
    match loading {
        Some(handle) => match (&mut handle.0).await {
            Ok(result) => result,
            Err(e) => Err(FerricastError::Receiver(format!("load task failed: {e}"))),
        },
        None => std::future::pending().await,
    }
}

async fn join_pump(pump: &mut Option<PumpHandle>) -> PumpEnd {
    match pump {
        Some(handle) => (&mut handle.task)
            .await
            .unwrap_or_else(|e| PumpEnd::Failed(format!("pump task failed: {e}"))),
        None => std::future::pending().await,
    }
}

fn progressive() -> Box<dyn ErasedPuller> {
    Box::new(ProgressivePuller::default())
}

/// Opens `url` with `puller`.
async fn open(
    mut puller: Box<dyn ErasedPuller>,
    url: &str,
) -> Result<(Box<dyn ErasedPuller>, MediaInfo)> {
    let info = puller
        .open(PullSpec {
            url: url.to_string(),
            headers: HashMap::new(),
        })
        .await?;
    Ok((puller, info))
}

/// Resolves a LOAD: opens the puller, picks decoders, and only then
/// asks the application for a sink, so nothing opens for media that
/// cannot play.
async fn load(ctx: &PipelineCtx, remote: &RemoteSender, request: LoadRequest) -> Result<Prepared> {
    let kind = MediaKind::classify(&request.url, request.content_type.as_deref());
    tracing::info!(url = %request.url, content_type = ?request.content_type, ?kind, "receiver: loading");
    let (mut puller, info) = match kind {
        MediaKind::Hls => open((ctx.create_puller)()?, &request.url).await?,
        MediaKind::Progressive => open(progressive(), &request.url).await?,
        // No type and no telling extension: HLS first, then a plain file.
        MediaKind::Unknown => match open((ctx.create_puller)()?, &request.url).await {
            Ok(opened) => opened,
            Err(hls) => match open(progressive(), &request.url).await {
                Ok(opened) => opened,
                Err(file) => {
                    return Err(FerricastError::Pull(format!(
                        "neither an HLS playlist ({hls}) nor a playable file ({file})"
                    )));
                }
            },
        },
        MediaKind::Dash => {
            return Err(FerricastError::Pull(
                "MPEG-DASH streams are not supported".into(),
            ));
        }
        MediaKind::SmoothStreaming => {
            return Err(FerricastError::Pull(
                "Smooth Streaming is not supported".into(),
            ));
        }
    };
    let mut start_us = 0;
    if let Some(start) = request.start_time_us.filter(|&s| s > 0 && !info.is_live) {
        puller.seek(start).await?;
        start_us = start;
    }

    let video_dec = info
        .video
        .as_ref()
        .map(|v| {
            let factory = ctx.video_decoders.get(&v.codec).ok_or_else(|| {
                FerricastError::Decode(format!("no video decoder for {:?}", v.codec))
            })?;
            let mut d = factory();
            d.configure(&DecoderConfig {
                codec: v.codec,
                width: v.width,
                height: v.height,
                pixel_format: PixelFormat::Bgra,
                low_latency: true,
            })?;
            Ok::<_, FerricastError>(d)
        })
        .transpose()?;
    let audio_dec = match &info.audio {
        Some(a) => match ctx.audio_decoders.get(&a.codec) {
            Some(factory) => {
                let mut d = factory();
                d.configure(&AudioDecoderConfig {
                    codec: a.codec,
                    sample_rate: a.sample_rate,
                    channels: a.channels,
                })?;
                Some(d)
            }
            None if video_dec.is_some() => {
                tracing::warn!(codec = ?a.codec, "no audio decoder; playing video only");
                None
            }
            None => {
                return Err(FerricastError::Decode(format!(
                    "no audio decoder for {:?}",
                    a.codec
                )));
            }
        },
        None => None,
    };
    if video_dec.is_none() && audio_dec.is_none() {
        return Err(FerricastError::Pull(
            "the media has no playable stream".into(),
        ));
    }

    let sink = (ctx.sink_factory)(remote, &info).await?;
    Ok(Prepared {
        puller,
        info,
        start_us,
        video_dec,
        audio_dec,
        sink,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_kinds_are_classified_by_type_then_extension() {
        let c = MediaKind::classify;
        assert_eq!(c("http://h/a.m3u8?x=1", None), MediaKind::Hls);
        assert_eq!(
            c("http://h/a", Some("application/vnd.apple.mpegurl")),
            MediaKind::Hls
        );
        assert_eq!(
            c("http://h/a", Some("application/x-mpegURL")),
            MediaKind::Hls
        );
        assert_eq!(c("http://h/manifest.mpd", None), MediaKind::Dash);
        assert_eq!(
            c("http://h/v.ism/manifest", None),
            MediaKind::SmoothStreaming
        );
        assert_eq!(c("http://h/movie.mp4", None), MediaKind::Progressive);
        assert_eq!(
            c("http://h/stream", Some("video/webm")),
            MediaKind::Progressive
        );
        assert_eq!(c("http://h/stream", None), MediaKind::Unknown);
    }
}
