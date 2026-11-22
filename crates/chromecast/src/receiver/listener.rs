//! The shared TLS listener of the Chromecast receiver.
//!
//! Accepting a TCP connection is all the accept loop does. Each
//! connection then gets its own task for the TLS handshake and the
//! device-auth exchange, and is handed to the pipeline only once the
//! sender sends something past device authentication. That keeps the
//! traffic Android devices generate away from the pipeline and the UI:
//!
//! - reachability probes that open TCP and close before TLS;
//! - Cast SDK senders that send the device-auth challenge, get the
//!   auth error back and drop the connection (several times a minute,
//!   while a casting app is open).
//!
//! A probe that misbehaves can never stall the listener: its handshake
//! has a timeout, failures of one connection never back off the
//! accept loop, and the number of connections waiting for their first
//! message is capped, overall and per peer address.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use bytes::BytesMut;
use tokio::io::{ReadHalf, split};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, mpsc};
use tokio::task::{JoinHandle, JoinSet};

use ferricast_core::{FerricastError, Result};

use crate::castv2::CastMessage;

use super::Shared;
use super::control::{
    DEVICE_AUTH_NAMESPACE, ServerStream, ServerWriter, answer_device_auth, server_recv,
};

/// How long a TLS handshake may take before the connection is dropped.
const TLS_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);
/// How long a connection may take, from TCP connect, to send its first
/// message past device authentication.
const FIRST_MESSAGE_TIMEOUT: Duration = Duration::from_secs(30);
/// Connections in handshake or device auth, at most.
const MAX_PENDING: usize = 64;
/// Connections in handshake or device auth from one address, at most.
const MAX_PENDING_PER_PEER: usize = 8;
/// Senders ready for the pipeline, queued at most.
const READY_QUEUE: usize = 8;
/// How often the connections that ended before casting are summarised.
const SUMMARY_INTERVAL: Duration = Duration::from_secs(60);
const ACCEPT_BACKOFF_START: Duration = Duration::from_millis(50);
const ACCEPT_BACKOFF_MAX: Duration = Duration::from_secs(1);

/// A sender past TLS and device authentication, ready for a control.
pub(crate) struct Admitted {
    pub(crate) peer: SocketAddr,
    pub(crate) reader: ReadHalf<ServerStream>,
    pub(crate) writer: ServerWriter,
    /// Bytes read past `first`.
    pub(crate) buf: BytesMut,
    /// The first message past device authentication.
    pub(crate) first: CastMessage,
}

/// How a connection's admission ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// Handed to the pipeline.
    Admitted,
    /// Closed before or during the TLS handshake.
    Probe,
    /// Closed after the device-auth exchange, without casting.
    AuthOnly,
    /// Handshake or first message took too long.
    TimedOut,
    /// TLS or framing error.
    Failed,
    /// Turned away: too many connections waiting.
    Rejected,
    /// The receiver stopped while the connection waited.
    Stopped,
}

/// Counts of connections that ended before casting, for the summary.
#[derive(Debug, Default)]
struct Tally(HashMap<&'static str, u64>);

impl Tally {
    fn record(&mut self, outcome: Outcome) {
        let key = match outcome {
            Outcome::Admitted | Outcome::Stopped => return,
            Outcome::Probe => "closed_before_tls",
            Outcome::AuthOnly => "closed_after_device_auth",
            Outcome::TimedOut => "timed_out",
            Outcome::Failed => "failed",
            Outcome::Rejected => "rejected",
        };
        *self.0.entry(key).or_default() += 1;
    }

    fn flush(&mut self) {
        if self.0.is_empty() {
            return;
        }
        let mut counts: Vec<_> = self.0.drain().collect();
        counts.sort_unstable();
        tracing::info!(
            ?counts,
            "Chromecast receiver: connections that ended before casting in the last minute"
        );
    }
}

/// The bound port, its accept loop and the queue of admitted senders.
/// Stops listening, and drops every connection not yet admitted, when
/// dropped.
pub(crate) struct Listener {
    local_addr: SocketAddr,
    ready: Mutex<mpsc::Receiver<Admitted>>,
    acceptor: JoinHandle<()>,
}

impl Listener {
    pub(crate) fn start(listener: TcpListener, shared: Arc<Shared>) -> Result<Self> {
        let local_addr = listener.local_addr().map_err(FerricastError::Io)?;
        let (ready_tx, ready_rx) = mpsc::channel(READY_QUEUE);
        let acceptor = tokio::spawn(accept_loop(listener, shared, ready_tx));
        Ok(Self {
            local_addr,
            ready: Mutex::new(ready_rx),
            acceptor,
        })
    }

    pub(crate) fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// The next admitted sender.
    pub(crate) async fn next(&self) -> Result<Admitted> {
        self.ready
            .lock()
            .await
            .recv()
            .await
            .ok_or_else(|| FerricastError::Receiver("Chromecast listener stopped".into()))
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        self.acceptor.abort();
    }
}

async fn accept_loop(listener: TcpListener, shared: Arc<Shared>, ready: mpsc::Sender<Admitted>) {
    let mut pending: JoinSet<(IpAddr, Outcome)> = JoinSet::new();
    let mut per_peer: HashMap<IpAddr, usize> = HashMap::new();
    let mut tally = Tally::default();
    let mut summary = tokio::time::interval(SUMMARY_INTERVAL);
    summary.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut backoff = ACCEPT_BACKOFF_START;
    loop {
        tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((tcp, peer)) => {
                    backoff = ACCEPT_BACKOFF_START;
                    let waiting = per_peer.entry(peer.ip()).or_default();
                    if pending.len() >= MAX_PENDING || *waiting >= MAX_PENDING_PER_PEER {
                        tracing::debug!(%peer, "Chromecast receiver: too many connections waiting; dropping");
                        tally.record(Outcome::Rejected);
                        if *waiting == 0 {
                            per_peer.remove(&peer.ip());
                        }
                        continue;
                    }
                    *waiting += 1;
                    tracing::debug!(%peer, "Chromecast receiver: TCP connect");
                    pending.spawn(admit(tcp, peer, shared.clone(), ready.clone()));
                }
                Err(e) => {
                    // The listener itself is failing (out of fds, …):
                    // back off so it cannot spin.
                    tracing::warn!(%e, retry_in = ?backoff, "Chromecast receiver: accept failed");
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(ACCEPT_BACKOFF_MAX);
                }
            },
            Some(joined) = pending.join_next() => {
                let Ok((ip, outcome)) = joined else { continue };
                if let Some(waiting) = per_peer.get_mut(&ip) {
                    *waiting = waiting.saturating_sub(1);
                    if *waiting == 0 {
                        per_peer.remove(&ip);
                    }
                }
                tally.record(outcome);
            }
            _ = summary.tick() => tally.flush(),
        }
    }
}

/// TLS handshake and device auth for one connection, then hand-off.
async fn admit(
    tcp: TcpStream,
    peer: SocketAddr,
    shared: Arc<Shared>,
    ready: mpsc::Sender<Admitted>,
) -> (IpAddr, Outcome) {
    let outcome =
        match tokio::time::timeout(FIRST_MESSAGE_TIMEOUT, handshake(tcp, peer, &shared)).await {
            Ok(Ok(admitted)) => {
                tracing::info!(%peer, "Chromecast receiver: sender connected");
                if ready.send(admitted).await.is_ok() {
                    Outcome::Admitted
                } else {
                    Outcome::Stopped
                }
            }
            Ok(Err(outcome)) => outcome,
            Err(_) => {
                tracing::debug!(%peer, "Chromecast receiver: no message past device auth in time");
                Outcome::TimedOut
            }
        };
    (peer.ip(), outcome)
}

async fn handshake(
    tcp: TcpStream,
    peer: SocketAddr,
    shared: &Shared,
) -> std::result::Result<Admitted, Outcome> {
    let acceptor = tokio_rustls::TlsAcceptor::from(shared.tls_config.clone());
    let tls = match tokio::time::timeout(TLS_HANDSHAKE_TIMEOUT, acceptor.accept(tcp)).await {
        Ok(Ok(tls)) => tls,
        Ok(Err(e)) => {
            // An EOF here is a reachability probe, not a failure.
            let outcome = if e.kind() == std::io::ErrorKind::UnexpectedEof {
                Outcome::Probe
            } else {
                Outcome::Failed
            };
            tracing::debug!(%peer, %e, "Chromecast receiver: TLS handshake ended");
            return Err(outcome);
        }
        Err(_) => {
            tracing::debug!(%peer, "Chromecast receiver: TLS handshake timed out");
            return Err(Outcome::TimedOut);
        }
    };
    tracing::debug!(%peer, "Chromecast receiver: TLS handshake complete");

    let (mut reader, write_half) = split(tls);
    let writer: ServerWriter = Arc::new(Mutex::new(write_half));
    let mut buf = BytesMut::with_capacity(8 * 1024);
    let mut authed = false;
    loop {
        let msg = match server_recv(&mut reader, &mut buf).await {
            Ok(Some(msg)) => msg,
            Ok(None) => {
                tracing::debug!(%peer, authed, "Chromecast receiver: closed before casting");
                return Err(if authed {
                    Outcome::AuthOnly
                } else {
                    Outcome::Probe
                });
            }
            Err(e) => {
                tracing::debug!(%peer, %e, "Chromecast receiver: connection failed before casting");
                return Err(Outcome::Failed);
            }
        };
        if msg.namespace != DEVICE_AUTH_NAMESPACE {
            return Ok(Admitted {
                peer,
                reader,
                writer,
                buf,
                first: msg,
            });
        }
        authed = true;
        if let Err(e) = answer_device_auth(&writer, &msg).await {
            tracing::debug!(%peer, %e, "Chromecast receiver: device-auth reply failed");
            return Err(Outcome::AuthOnly);
        }
    }
}
