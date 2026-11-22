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
    ControlSession, FerricastError, MediaCommand, PlaybackState, RemoteSender, Result,
};
use tokio::net::{TcpListener, TcpSocket};
use tokio::sync::{mpsc, oneshot};
use tokio::task::{JoinHandle, JoinSet};
use tracing::{debug, info, warn};

use super::Settings;
use super::connection::Connection;

/// Connections waiting to be accepted by the pipeline, at most.
const READY_QUEUE: usize = 8;
/// Control connections open at once, at most.
const MAX_CONNECTIONS: usize = 32;
/// Control connections open at once from one address, at most.
const MAX_CONNECTIONS_PER_PEER: usize = 8;
const ACCEPT_BACKOFF_START: Duration = Duration::from_millis(50);
const ACCEPT_BACKOFF_MAX: Duration = Duration::from_secs(1);

/// A connection handed to the pipeline.
pub(super) struct Accepted {
    pub remote: RemoteSender,
    pub commands: mpsc::Receiver<MediaCommand>,
    /// Dropped (or fired) to make the connection close.
    pub close: oneshot::Sender<()>,
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

    async fn report_state(&mut self, _state: PlaybackState) -> Result<()> {
        // Mirroring has no status channel back to the sender.
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
