//! The RTSP listener and the [`ControlSession`] the receiver pipeline
//! drives.
//!
//! Every TCP connection gets its own [`Connection`] task straight away,
//! so probes (`GET /info` while the sender's AirPlay menu is open) are
//! answered without involving the pipeline. A connection becomes a
//! pipeline session (is handed out by [`AirPlayReceiverControl::accept`])
//! once the sender asks for a PIN or sets up a session (pairing alone
//! does not count: senders pair on every connection).

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use ferricast_core::{
    ControlSession, FerricastError, MediaCommand, PlaybackState, PlaybackStatus, RemoteSender,
    Result,
};
use tokio::net::{TcpListener, TcpSocket};
use tokio::sync::{mpsc, oneshot};
use tokio::task::{JoinHandle, JoinSet};
use tracing::{debug, info, warn};

use super::Settings;
use super::connection::Connection;
use super::video::{ReverseHub, VideoHub};

/// Connections waiting to be accepted by the pipeline, at most.
const READY_QUEUE: usize = 8;
/// Control connections open at once, at most.
const MAX_CONNECTIONS: usize = 32;
/// Control connections open at once from one address, at most.
const MAX_CONNECTIONS_PER_PEER: usize = 8;
const ACCEPT_BACKOFF_START: Duration = Duration::from_millis(50);
const ACCEPT_BACKOFF_MAX: Duration = Duration::from_secs(1);
/// Status reports queued for a connection; older ones are stale anyway.
const STATUS_QUEUE: usize = 32;

/// A connection handed to the pipeline.
pub(super) struct Accepted {
    pub remote: RemoteSender,
    pub commands: mpsc::Receiver<MediaCommand>,
    /// Where the pipeline reports playback status (AirPlay video
    /// answers `/playback-info` and sends events from it).
    pub status: mpsc::Sender<PlaybackStatus>,
    /// Dropped (or fired) to make the connection close.
    pub close: oneshot::Sender<()>,
}

/// The connection side of a session handed to the pipeline.
pub(super) struct Promoted {
    commands: mpsc::Sender<MediaCommand>,
    close: oneshot::Receiver<()>,
    status: mpsc::Receiver<PlaybackStatus>,
}

impl Promoted {
    /// Hands a new session from `remote` to the pipeline.
    pub async fn open(ready: &mpsc::Sender<Accepted>, remote: RemoteSender) -> Self {
        let (commands_tx, commands_rx) = mpsc::channel(32);
        let (close_tx, close_rx) = oneshot::channel();
        let (status_tx, status_rx) = mpsc::channel(STATUS_QUEUE);
        let accepted = Accepted {
            remote,
            commands: commands_rx,
            status: status_tx,
            close: close_tx,
        };
        if ready.send(accepted).await.is_err() {
            warn!("AirPlay receiver stopped; the session goes nowhere");
        }
        Self {
            commands: commands_tx,
            close: close_rx,
            status: status_rx,
        }
    }

    pub async fn command(&self, cmd: MediaCommand) {
        if self.commands.send(cmd).await.is_err() {
            debug!("pipeline session gone; command dropped");
        }
    }
}

/// Resolves when the pipeline closes a promoted session.
pub(super) async fn closed(promoted: &mut Option<Promoted>) {
    match promoted {
        Some(p) => {
            let _ = (&mut p.close).await;
        }
        None => std::future::pending().await,
    }
}

/// What the pipeline did with a promoted session.
pub(super) enum PipelineEvent {
    Status(PlaybackStatus),
    /// The pipeline closed the session.
    Closed,
}

/// The next [`PipelineEvent`] of a promoted session.
pub(super) async fn pipeline_event(promoted: &mut Option<Promoted>) -> PipelineEvent {
    let Some(p) = promoted else {
        return std::future::pending().await;
    };
    tokio::select! {
        Some(status) = p.status.recv() => PipelineEvent::Status(status),
        _ = &mut p.close => PipelineEvent::Closed,
    }
}

/// State shared by every connection of the receiver.
pub(super) struct Shared {
    pub settings: Arc<Settings>,
    /// The connection whose media streams are set up; a new one takes
    /// over (and the old one is closed), as only one sender can mirror at
    /// a time and the media ports are fixed.
    active: Mutex<Option<(u64, mpsc::Sender<()>)>>,
    /// Ed25519 keys (hex) of senders that paired with the PIN while this
    /// receiver ran; the credential store keeps them across restarts.
    paired: Mutex<std::collections::HashSet<String>>,
    /// Volume in the sender's dB scale (-144 mute, -30 to 0).
    volume_db: Mutex<f32>,
    next_id: AtomicU64,
    /// Reverse (event) connections of video senders.
    pub reverse: ReverseHub,
    /// Video playbacks, by sender session.
    pub videos: VideoHub,
}

impl Shared {
    pub fn next_connection_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Make connection `id` the one with media, closing the previous one.
    /// The returned receiver fires when another connection takes over.
    pub fn claim_media(&self, id: u64) -> mpsc::Receiver<()> {
        let (tx, rx) = mpsc::channel(1);
        let previous = self
            .active
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .replace((id, tx));
        if let Some((prev, prev_tx)) = previous
            && prev != id
        {
            info!(
                previous = prev,
                new = id,
                "a new AirPlay session replaces the current one"
            );
            let _ = prev_tx.try_send(());
        }
        rx
    }

    pub fn release_media(&self, id: u64) {
        let mut active = self.active.lock().unwrap_or_else(PoisonError::into_inner);
        if active.as_ref().is_some_and(|(owner, _)| *owner == id) {
            *active = None;
        }
    }

    /// Remember a sender paired with the PIN.
    pub fn add_paired(&self, public_key_hex: String) {
        self.paired
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(public_key_hex);
    }

    /// Whether a sender with this key paired with the PIN.
    pub fn is_paired(&self, public_key_hex: &str) -> bool {
        self.paired
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .contains(public_key_hex)
            || self
                .settings
                .store
                .paired_senders()
                .iter()
                .any(|s| s.public_key == public_key_hex)
    }

    pub fn volume_db(&self) -> f32 {
        *self
            .volume_db
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    pub fn set_volume_db(&self, db: f32) {
        *self
            .volume_db
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = db;
    }
}

/// The bound control port, its accept loop and the queue of connections
/// ready for the pipeline. Stops listening when dropped.
pub(super) struct Server {
    port: u16,
    ready: tokio::sync::Mutex<mpsc::Receiver<Accepted>>,
    acceptor: JoinHandle<()>,
}

impl Server {
    pub fn bind(settings: Arc<Settings>) -> Result<Arc<Self>> {
        let addr = SocketAddr::from(([0, 0, 0, 0], settings.config.rtsp_port));
        let bind = || -> std::io::Result<TcpListener> {
            let socket = TcpSocket::new_v4()?;
            // A restart must not fail on connections in TIME_WAIT.
            socket.set_reuseaddr(true)?;
            socket.bind(addr)?;
            socket.listen(64)
        };
        let listener =
            bind().map_err(|e| FerricastError::Receiver(format!("bind tcp {addr}: {e}")))?;
        let port = listener
            .local_addr()
            .map(|a| a.port())
            .unwrap_or(addr.port());
        info!(port, "AirPlay receiver listening");
        let shared = Arc::new(Shared {
            settings,
            active: Mutex::new(None),
            paired: Mutex::new(std::collections::HashSet::new()),
            volume_db: Mutex::new(0.0),
            next_id: AtomicU64::new(1),
            reverse: ReverseHub::default(),
            videos: VideoHub::default(),
        });
        let (ready_tx, ready_rx) = mpsc::channel(READY_QUEUE);
        let acceptor = tokio::spawn(accept_loop(listener, shared, ready_tx));
        Ok(Arc::new(Self {
            port,
            ready: tokio::sync::Mutex::new(ready_rx),
            acceptor,
        }))
    }

    pub fn port(&self) -> u16 {
        self.port
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.acceptor.abort();
    }
}

/// One AirPlay sender as seen by the receiver pipeline.
pub struct AirPlayReceiverControl {
    server: Arc<Server>,
    conn: Option<Accepted>,
    alive: bool,
}

impl AirPlayReceiverControl {
    pub(super) fn new(server: Arc<Server>) -> Self {
        Self {
            server,
            conn: None,
            alive: false,
        }
    }
}

impl ControlSession for AirPlayReceiverControl {
    async fn accept(&mut self) -> Result<RemoteSender> {
        let accepted = self
            .server
            .ready
            .lock()
            .await
            .recv()
            .await
            .ok_or_else(|| FerricastError::Receiver("AirPlay listener stopped".into()))?;
        let remote = accepted.remote.clone();
        self.conn = Some(accepted);
        self.alive = true;
        Ok(remote)
    }

    async fn next_command(&mut self) -> Result<MediaCommand> {
        let Some(conn) = self.conn.as_mut() else {
            return Err(FerricastError::Receiver(
                "no AirPlay sender accepted".into(),
            ));
        };
        match conn.commands.recv().await {
            Some(cmd) => Ok(cmd),
            None => {
                self.alive = false;
                Err(FerricastError::RemoteStopped(
                    "AirPlay sender disconnected".into(),
                ))
            }
        }
    }

    async fn report_state(&mut self, state: PlaybackState) -> Result<()> {
        self.report_status(PlaybackStatus::from(state)).await
    }

    /// Mirroring ignores it; a video session follows it to answer
    /// `/playback-info` and send `/event`s. Never waits on the session: a
    /// report finding the queue full is dropped.
    async fn report_status(&mut self, status: PlaybackStatus) -> Result<()> {
        if let Some(conn) = &self.conn {
            let _ = conn.status.try_send(status);
        }
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        if let Some(conn) = self.conn.take() {
            let _ = conn.close.send(());
        }
        self.alive = false;
        Ok(())
    }

    fn is_alive(&self) -> bool {
        self.alive
    }
}

/// Accepts control connections and runs each one in its own task. The
/// tasks belong to the loop, so they all end when the receiver stops.
/// One connection failing never slows accepting the next; only the
/// listener failing (out of fds, …) backs off.
async fn accept_loop(listener: TcpListener, shared: Arc<Shared>, ready: mpsc::Sender<Accepted>) {
    let mut connections: JoinSet<IpAddr> = JoinSet::new();
    let mut per_peer: HashMap<IpAddr, usize> = HashMap::new();
    let mut backoff = ACCEPT_BACKOFF_START;
    loop {
        tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, peer)) => {
                    backoff = ACCEPT_BACKOFF_START;
                    let open = per_peer.entry(peer.ip()).or_default();
                    if connections.len() >= MAX_CONNECTIONS || *open >= MAX_CONNECTIONS_PER_PEER {
                        debug!(%peer, "too many AirPlay connections; dropping");
                        if *open == 0 {
                            per_peer.remove(&peer.ip());
                        }
                        continue;
                    }
                    *open += 1;
                    debug!(%peer, "AirPlay control connection");
                    let _ = stream.set_nodelay(true);
                    let conn = Connection::new(shared.clone(), peer, ready.clone());
                    connections.spawn(async move {
                        conn.run(stream).await;
                        peer.ip()
                    });
                }
                Err(e) => {
                    warn!(error = %e, retry_in = ?backoff, "AirPlay accept failed");
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(ACCEPT_BACKOFF_MAX);
                }
            },
            Some(joined) = connections.join_next() => {
                let Ok(ip) = joined else { continue };
                if let Some(open) = per_peer.get_mut(&ip) {
                    *open = open.saturating_sub(1);
                    if *open == 0 {
                        per_peer.remove(&ip);
                    }
                }
            }
        }
    }
}
