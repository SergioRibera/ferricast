//! One sender's video playback, shared by every connection that carries
//! the sender's `X-Apple-Session-ID`: HTTP clients may spread `/play`,
//! `/rate` and the `/playback-info` polls over several connections.
//!
//! A task owns the playback and its pipeline session; connections talk
//! to it through a [`VideoHandle`]. When the last handle goes (every
//! connection of the session closed) the playback stops.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Arc, Mutex, PoisonError};

use ferricast_core::{MediaCommand, PlaybackState, PlaybackStatus, RemoteSender};
use plist::Dictionary;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::reverse::Event;
use super::{EventState, PlayRequest, VideoPlayback};
use crate::receiver::server::{Accepted, PipelineEvent, Promoted, Shared, pipeline_event};

/// Requests queued for one video session, at most.
const REQUEST_QUEUE: usize = 16;

/// What a connection asks the video session.
#[derive(Debug)]
pub(in crate::receiver) enum VideoRequest {
    /// Load and play a URL; answers why it cannot, if so.
    Play(PlayRequest, oneshot::Sender<Result<(), &'static str>>),
    /// `/playback-info`; `None` when nothing plays.
    Info(oneshot::Sender<Option<Dictionary>>),
    /// `GET /scrub`; `None` when nothing plays.
    Position(oneshot::Sender<Option<String>>),
    Seek(u64),
    Rate(bool),
    Stop,
    /// `/action` `playlistRemove`: stop if this `uuid` is playing.
    Remove(String),
}

/// A connection's way to the video session of its sender. Cheap to clone.
#[derive(Clone)]
pub(in crate::receiver) struct VideoHandle {
    key: Arc<str>,
    tx: mpsc::Sender<VideoRequest>,
}

impl VideoHandle {
    pub fn key(&self) -> &str {
        &self.key
    }

    pub async fn send(&self, request: VideoRequest) {
        if self.tx.send(request).await.is_err() {
            debug!(key = %self.key, "AirPlay video session gone; request dropped");
        }
    }

    /// Sends a request that answers, and waits for the answer.
    pub async fn ask<T>(&self, make: impl FnOnce(oneshot::Sender<T>) -> VideoRequest) -> Option<T> {
        let (tx, rx) = oneshot::channel();
        self.send(make(tx)).await;
        rx.await.ok()
    }
}

/// The video sessions alive, by session key. Holds only weak senders, so
/// a session ends with its connections.
#[derive(Default)]
pub(in crate::receiver) struct VideoHub {
    sessions: Mutex<HashMap<Arc<str>, mpsc::WeakSender<VideoRequest>>>,
}

impl VideoHub {
    /// The session of `key`, started when there is none.
    pub fn handle(
        &self,
        shared: &Arc<Shared>,
        ready: &mpsc::Sender<Accepted>,
        key: &str,
        peer: IpAddr,
    ) -> VideoHandle {
        let mut sessions = self.sessions.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(tx) = sessions.get(key).and_then(mpsc::WeakSender::upgrade) {
            return VideoHandle {
                key: Arc::from(key),
                tx,
            };
        }
        let key: Arc<str> = Arc::from(key);
        let (tx, rx) = mpsc::channel(REQUEST_QUEUE);
        sessions.retain(|_, weak| weak.strong_count() > 0);
        sessions.insert(key.clone(), tx.downgrade());
        let session = VideoSession {
            shared: shared.clone(),
            ready: ready.clone(),
            key: key.clone(),
            id: shared.next_connection_id(),
            peer,
            promoted: None,
            playback: None,
        };
        tokio::spawn(session.run(rx));
        VideoHandle { key, tx }
    }
}

struct VideoSession {
    shared: Arc<Shared>,
    ready: mpsc::Sender<Accepted>,
    key: Arc<str>,
    /// Id under which the session claims the receiver's media.
    id: u64,
    peer: IpAddr,
    promoted: Option<Promoted>,
    playback: Option<VideoPlayback>,
}

impl VideoSession {
    async fn run(mut self, mut requests: mpsc::Receiver<VideoRequest>) {
        debug!(key = %self.key, "AirPlay video session started");
        loop {
            tokio::select! {
                request = requests.recv() => match request {
                    Some(request) => self.on_request(request).await,
                    None => break,
                },
                event = pipeline_event(&mut self.promoted) => match event {
                    PipelineEvent::Status(status) => self.on_status(&status).await,
                    PipelineEvent::Closed => {
                        debug!(key = %self.key, "pipeline closed the AirPlay video session");
                        self.promoted = None;
                        self.stop().await;
                    }
                },
                () = preempted(&mut self.playback) => {
                    info!(key = %self.key, "AirPlay video replaced by another session");
                    self.stop().await;
                }
            }
        }
        self.stop().await;
        debug!(key = %self.key, "AirPlay video session ended");
    }

    async fn on_status(&mut self, status: &PlaybackStatus) {
        let Some(playback) = self.playback.as_mut() else {
            return;
        };
        if let PlaybackState::Error(message) = &status.state {
            warn!(url = %playback.request.url, %message, "AirPlay video failed");
        }
        let change = playback.on_status(status);
        if let Some(position_us) = change.seek_us {
            self.command(MediaCommand::Seek { position_us }).await;
        }
        if let Some(state) = change.event {
            self.event(state);
        }
    }

    async fn on_request(&mut self, request: VideoRequest) {
        match request {
            VideoRequest::Play(play, reply) => {
                let _ = reply.send(self.play(play).await);
            }
            VideoRequest::Info(reply) => {
                let info = self
                    .playback
                    .as_ref()
                    .filter(|p| p.event_state() != EventState::Stopped)
                    .map(VideoPlayback::playback_info);
                let _ = reply.send(info);
            }
            VideoRequest::Position(reply) => {
                let _ = reply.send(self.playback.as_ref().map(VideoPlayback::scrub_text));
            }
            VideoRequest::Seek(position_us) => {
                if let Some(playback) = self.playback.as_mut() {
                    playback.seek(position_us);
                    self.command(MediaCommand::Seek { position_us }).await;
                }
            }
            VideoRequest::Rate(playing) => {
                let Some(playback) = self.playback.as_mut() else {
                    return;
                };
                let event = playback.set_playing(playing);
                self.command(if playing {
                    MediaCommand::Play
                } else {
                    MediaCommand::Pause
                })
                .await;
                if let Some(state) = event {
                    self.event(state);
                }
            }
            VideoRequest::Stop => self.stop().await,
            VideoRequest::Remove(uuid) => {
                let current = self
                    .playback
                    .as_ref()
                    .and_then(|p| p.request.playback_id.as_deref());
                if current == Some(uuid.as_str()) {
                    self.stop().await;
                }
            }
        }
    }

    async fn play(&mut self, play: PlayRequest) -> Result<(), &'static str> {
        info!(
            key = %self.key,
            url = %play.url,
            start = ?play.start,
            client = ?play.client,
            "AirPlay video: play"
        );
        if let Some(reason) = play.unsupported_reason() {
            warn!(url = %play.url, reason, "AirPlay video refused");
            return Err(reason);
        }
        let preempted = self.shared.claim_media(self.id);
        let playback = VideoPlayback::new(play, preempted);
        let load = MediaCommand::Load {
            url: playback.request.url.clone(),
            content_type: None,
            start_time_us: playback.load_start_us(),
            autoplay: true,
            metadata: HashMap::new(),
        };
        let name = playback.request.client.clone();
        self.playback = Some(playback);
        if self.promoted.is_none() {
            let remote = RemoteSender {
                id: Uuid::new_v4(),
                addr: self.peer,
                name,
            };
            self.promoted = Some(Promoted::open(&self.ready, remote).await);
        }
        self.command(load).await;
        self.event(EventState::Loading);
        Ok(())
    }

    /// Stops the playback and ends the pipeline session, which closes the
    /// player.
    async fn stop(&mut self) {
        let playback = self.playback.take();
        if let Some(promoted) = self.promoted.take() {
            promoted.command(MediaCommand::Stop).await;
        }
        if let Some(playback) = playback {
            info!(key = %self.key, "AirPlay video stopped");
            self.shared.release_media(self.id);
            self.shared.reverse.send(
                &self.key,
                Event {
                    state: EventState::Stopped,
                    playback_id: playback.request.playback_id,
                },
            );
        }
    }

    async fn command(&self, command: MediaCommand) {
        if let Some(promoted) = &self.promoted {
            promoted.command(command).await;
        }
    }

    fn event(&self, state: EventState) {
        let playback_id = self
            .playback
            .as_ref()
            .and_then(|p| p.request.playback_id.clone());
        self.shared
            .reverse
            .send(&self.key, Event { state, playback_id });
    }
}

/// Resolves when another session takes the receiver over.
async fn preempted(playback: &mut Option<VideoPlayback>) {
    match playback {
        Some(p) => {
            if p.preempted.recv().await.is_none() {
                std::future::pending::<()>().await;
            }
        }
        None => std::future::pending().await,
    }
}
