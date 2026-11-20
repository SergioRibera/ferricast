//! Live HLS server backed by a continuous capture/encode pipeline.
//!
//! Architecture
//! ============
//!
//! Capture, encode and HTTP serving are decoupled. A single
//! background task ([`segmenter_loop`]) owns the capture + encoder
//! pair, runs as fast as the GPU/encoder allow, and writes finished
//! MPEG-TS segments into a bounded ring buffer ([`SegmentRing`]).
//! Segments end on the next encoder keyframe after the configured
//! target duration elapses, so each one is independently decodable
//! per RFC 8216 §3.
//!
//! HTTP requests are accepted from a separate task. Every connection
//! takes a short read lock on the ring, copies the playlist or
//! segment bytes it needs, and drops the lock before writing to the
//! client. Multiple players can pull from the same stream
//! concurrently.
//!
//! Steady-state latency is `target_duration × (keep_segments - 3)`:
//! the player needs ~3 segments buffered before it starts decoding.
//! With 2-second target / 6 retained segments that's ~6 s glass-to-glass.
//!
//! No third-party HTTP/parsing dependencies — the request reader is a
//! tiny Tokio-aware HTTP/1.x line splitter implemented inline.

mod http;
mod ring;
mod segmenter;
mod stats;
pub mod tls;

#[cfg(feature = "pull")]
pub mod puller;

#[cfg(feature = "pull")]
pub use puller::HlsPuller;

use std::future::Future;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex as StdMutex, PoisonError};
use std::time::Duration;

use tokio::net::{TcpListener, TcpStream, ToSocketAddrs};
use tokio::sync::{RwLock, Semaphore, mpsc};
use tokio::task::{JoinHandle, JoinSet};
use tokio_rustls::TlsAcceptor;
use tracing::{error, info, trace, warn};

use ferricast_core::{
    AudioFrame, EncodedFrame, FerricastError, Result, ScreenCapture, VideoEncoder,
};

pub use ring::SegmentRing;
pub use tls::build_self_signed_server_config;

/// Number of initial segment GETs the HLS server averages before
/// feeding the adaptive controller a one-shot bandwidth probe. Re-
/// exported here so external binaries can mention it in their own
/// startup logs without duplicating the constant.
pub const PROBE_SAMPLES_FOR_DOC: usize = stats::SessionStats::PROBE_SAMPLES;

/// Tunables that govern segment cadence and buffering.
///
/// Sensible defaults for desktop screen casting (2 s segments, 12
/// retained, target duration 4 s on the wire) are exposed via
/// [`HlsConfig::default`]. The playlist target duration is
/// deliberately larger than the segment target so the unavoidable
/// keyframe-delay tail never violates RFC 8216 §4.3.3.1.
///
/// The ring is sized generously — players (ffplay, hls.js, native
/// AVPlayer) typically start playback 3 segments behind the live
/// edge, and the small drift between segment wall-clock production
/// (~2.0-2.1 s, includes encode overhead) and player PTS-paced
/// consumption (exactly 2.000 s with the synthetic 60 fps PTS
/// counter) cumulatively pushes them toward the eviction line.
/// 12 retained segments × 2 s = 24 s window, leaving ~18 s of
/// margin past the player's initial offset; that's enough to
/// absorb several minutes of slow drift without any segment ever
/// expiring under the player's feet.
// Previously `Copy`. The `adaptive` field forced removing it
// because `Arc` isn't `Copy`. Call sites now clone explicitly;
// zero cost, the existing builders all own their config and
// don't depend on implicit copies.
#[derive(Debug, Clone)]
pub struct HlsConfig {
    /// Wallclock target a single segment tries to fit into. Real
    /// segments end at the next keyframe after this elapses.
    pub segment_target_secs: f32,
    /// Value advertised in `#EXT-X-TARGETDURATION`. Must be ≥
    /// `ceil(segment_target_secs)` plus headroom.
    pub playlist_target_duration: u8,
    /// Number of segments retained in the live ring. Must be ≥ 3 so
    /// players can prebuffer. Sized to give the player ~18 s of
    /// margin past the typical 3-segment live-edge offset — see
    /// the type-level docs for why under-sizing this caused ffplay
    /// to log "expired from playlists" + "Packet corrupt" every
    /// few segments after a couple minutes of streaming.
    pub keep_segments: usize,
    /// Frames-per-second target the segmenter paces to. Must agree
    /// with the encoder's configured fps. Used to:
    /// 1. Synthesise duplicate frames when the upstream capture
    ///    stalls (PipeWire on idle GNOME desktops can pause for
    ///    hundreds of ms — without this the segmenter would block
    ///    inside `next_frame().await` and the HLS playlist would
    ///    stop advancing).
    /// 2. Anchor segment boundaries to wall clock by requesting a
    ///    forced IDR once `segment_target_secs` has elapsed.
    pub target_fps: u32,
    /// Whether the underlying MPEG-TS muxer should advertise an
    /// audio elementary stream and inject silent AAC frames inline
    /// with video. Required for older Chromecasts whose firmware
    /// rejects HLS streams with only a video PID — they respond
    /// with `LOAD_FAILED, idleReason=ERROR`. The chromecast HLS
    /// pipeline turns this on when the target device's
    /// `DeviceCapabilities::requires_audio` is true. Leave off for
    /// receivers that accept video-only HLS (saves ~6 KB/s).
    pub inject_silent_audio: bool,

    /// Optional adaptive-bitrate controller shared with the encoder
    /// loop. When provided, the HTTP segment handler records each
    /// delivery's `budget_used_pct` into the controller; the encoder
    /// loop polls its `target_kbps` and live-reconfigures when the
    /// recommended bitrate moves. `None` disables the feedback loop
    /// entirely (legacy path / non-cast HLS where the consumer
    /// doesn't have a live encoder it can downshift).
    pub adaptive: Option<Arc<ferricast_core::AdaptiveBitrateState>>,

    /// HTTPS opt-in. When `Some(config)`, the accept loop wraps
    /// each connection with `tokio_rustls::TlsAcceptor` before
    /// dispatching to the HTTP handler. The URL the sink emits in
    /// [`HlsFrameSink::scheme`] flips to `https` in this case.
    ///
    /// Cast Application Framework 1.56+ rejects plain `http://` URLs
    /// on some firmwares even on the LAN. The chromecast handler
    /// turns this on with a per-session self-signed cert (see
    /// [`build_self_signed_server_config`]); the puller side
    /// (`HlsPuller`) accepts invalid certs unconditionally so
    /// cast-to-self still works end-to-end.
    pub tls: Option<Arc<rustls::ServerConfig>>,

    /// Low-Latency HLS opt-in. When `Some(seconds)`, the segmenter
    /// flushes a "part" every `secs` of accumulated wall time
    /// inside each segment, the ring publishes them immediately so
    /// the HTTP handler can serve them at `/part-N.M.ts`, and the
    /// playlist gets `#EXT-X-PART-INF` / `#EXT-X-SERVER-CONTROL`
    /// / per-segment `#EXT-X-PART` tags. Clients that understand
    /// LL-HLS get sub-second fetch granularity; clients that don't
    /// see the same classic `#EXTINF`/segment list and keep
    /// working unchanged. `None` (default) leaves the segmenter in
    /// classic mode: one body per segment, no parts.
    ///
    /// Recommended values: 0.2–0.5 s. Smaller means more overhead
    /// and a longer playlist; larger reduces the latency benefit.
    pub part_target_secs: Option<f32>,
}

impl Default for HlsConfig {
    fn default() -> Self {
        Self {
            // 4 s segments halve the IDR rate vs 2 s. On 1st-gen
            // Chromecast over 2.4 GHz Wi-Fi, one IDR at 1080p is
            // ~250 KB; at 2 s segments that overhead alone is
            // ~0.5 Mbps on top of the average bitrate, which was
            // enough to push sustained throughput past what the
            // receiver's Wi-Fi could pull and trigger
            // detailedErrorCode=301 after a few minutes. Trade-off
            // is +2 s of buffering latency at startup.
            segment_target_secs: 4.0,
            playlist_target_duration: 8,
            keep_segments: 12,
            target_fps: 60,
            inject_silent_audio: false,
            adaptive: None,
            tls: None,
            part_target_secs: None,
        }
    }
}

impl HlsConfig {
    /// Reject configurations no server can honour.
    fn validate(&self) -> Result<()> {
        if self.keep_segments < 3 {
            return Err(FerricastError::Hls(format!(
                "keep_segments={} too small (minimum 3)",
                self.keep_segments
            )));
        }
        if f32::from(self.playlist_target_duration) < self.segment_target_secs {
            return Err(FerricastError::Hls(format!(
                "playlist_target_duration={}s < segment_target_secs={}s",
                self.playlist_target_duration, self.segment_target_secs
            )));
        }
        Ok(())
    }

    /// The fixed `#EXT-X-TARGETDURATION` the playlist advertises:
    /// [`Self::playlist_target_duration`], raised to at least
    /// `ceil(segment_target_secs)`. Computed once per server and never
    /// changed afterwards (RFC 8216 §6.2.1).
    #[must_use]
    pub fn target_duration(&self) -> u8 {
        let ceil = if self.segment_target_secs.is_finite() && self.segment_target_secs > 0.0 {
            // Saturating float→int cast: a nonsensical huge target
            // pins to u8::MAX instead of wrapping.
            self.segment_target_secs.ceil() as u8
        } else {
            1
        };
        self.playlist_target_duration.max(ceil).max(1)
    }
}

/// Live HLS server. [`Self::start`] spawns the capture/encode loop in
/// the background and returns once the first segment is ready;
/// [`Self::run`] then accepts connections forever.
pub struct HlsServer {
    listener: TcpListener,
    connections: Arc<Connections>,
    segmenter: JoinHandle<()>,
}

impl HlsServer {
    /// Bind, start segmenting, wait until the first segment lands,
    /// and return. The server is ready to serve as soon as this
    /// resolves — players that connect immediately will see a
    /// non-empty playlist on their first request.
    ///
    /// # Errors
    ///
    /// Invalid configuration, bind failure, or no segment produced
    /// before the startup deadline.
    pub async fn start<S, E, A>(addr: A, capture: S, encoder: E, config: HlsConfig) -> Result<Self>
    where
        S: ScreenCapture + Send + 'static,
        E: VideoEncoder + Send + 'static,
        A: ToSocketAddrs,
    {
        config.validate()?;

        let listener = TcpListener::bind(addr).await?;
        let local = listener
            .local_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|_| "?".into());

        let ring = Arc::new(RwLock::new(SegmentRing::new(
            config.keep_segments,
            config.target_duration(),
        )));
        let state = Arc::new(http::ServerState::new(
            ring.clone(),
            config.adaptive.clone(),
        )?);
        let writer = ring.clone();
        let cfg = config.clone();
        let segmenter = tokio::spawn(async move {
            if let Err(e) = segmenter::run(capture, encoder, writer, cfg).await {
                error!(error = %e, "HLS segmenter loop exited");
            }
        });

        if !ring::wait_for_first_segment(&ring).await {
            segmenter.abort();
            return Err(FerricastError::Hls(
                "HLS segmenter did not produce any output within 30s; encoder likely failed".into(),
            ));
        }

        info!(
            listen = %local,
            segment_target_s = config.segment_target_secs,
            playlist_target_s = config.playlist_target_duration,
            keep = config.keep_segments,
            "HLS server ready"
        );

        Ok(Self {
            listener,
            connections: Arc::new(Connections::new(state, config.tls.clone())),
            segmenter,
        })
    }

    /// Bound socket address. Useful when [`Self::start`] is given
    /// `0.0.0.0:0` and the caller needs to discover the assigned port.
    ///
    /// # Errors
    ///
    /// The socket's address cannot be queried.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.listener.local_addr().map_err(FerricastError::from)
    }

    /// Path of the media playlist, `/<token>/index.m3u8`. Requests
    /// outside the per-server random token prefix get 404.
    #[must_use]
    pub fn playlist_path(&self) -> String {
        self.connections.state.token.playlist_path()
    }

    /// Only let `ip` (plus loopback and other allowed peers) fetch from
    /// this server; other peers get 403. Until the first call every
    /// peer may fetch. Applies to connections already open, too.
    pub fn allow_peer(&self, ip: IpAddr) {
        self.connections.state.allow_peer(ip);
    }

    /// Accept a single connection and dispatch it to a per-connection
    /// task. Useful when the caller wants to drive accept manually
    /// (e.g. integrate with a select! loop).
    ///
    /// # Errors
    ///
    /// The `accept` call failed; the caller decides whether to retry.
    pub async fn accept_one(&self) -> Result<()> {
        let (socket, peer) = self.listener.accept().await?;
        self.connections.dispatch(socket, peer);
        Ok(())
    }

    /// Accept loop. Runs until the returned future is dropped.
    /// Transient accept failures (fd exhaustion, aborted handshakes)
    /// are logged and retried after a short pause; they never stop the
    /// server.
    ///
    /// # Errors
    ///
    /// Never returns an error today; the `Result` is kept for API
    /// stability.
    pub async fn run(self) -> Result<()> {
        self.connections.accept_loop(&self.listener).await;
        Ok(())
    }
}

impl Drop for HlsServer {
    fn drop(&mut self) {
        // The segmenter owns the capture + encoder; it must not
        // outlive the server. Handler tasks would otherwise keep
        // serving a ring nobody updates.
        self.segmenter.abort();
        self.connections.shutdown();
    }
}

/// Upper bound on simultaneously served connections per server. A
/// Chromecast uses one or two; the cap only matters when something
/// floods the port.
const MAX_CONNECTIONS: usize = 64;

/// Pause after a failed `accept()` before trying again. Errors like
/// `EMFILE` repeat immediately while the condition lasts, so retrying
/// in a tight loop would spin a core.
const ACCEPT_RETRY_DELAY: Duration = Duration::from_millis(100);

/// A client that opens a TCP connection but never completes the TLS
/// handshake is dropped after this long.
const TLS_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// TCP keepalive: first probe after this much idle time…
const TCP_KEEPALIVE_IDLE: Duration = Duration::from_secs(10);
/// …then one probe every interval…
const TCP_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(5);
/// …and the connection is dropped after this many unanswered probes.
const TCP_KEEPALIVE_RETRIES: u32 = 3;

/// The connections one server accepts: TLS wrapping, the concurrency
/// cap, and the handler tasks (aborted together on shutdown).
struct Connections {
    state: Arc<http::ServerState>,
    tls: Option<TlsAcceptor>,
    permits: Arc<Semaphore>,
    handlers: StdMutex<JoinSet<()>>,
}

impl Connections {
    fn new(state: Arc<http::ServerState>, tls: Option<Arc<rustls::ServerConfig>>) -> Self {
        Self {
            state,
            tls: tls.map(TlsAcceptor::from),
            permits: Arc::new(Semaphore::new(MAX_CONNECTIONS)),
            handlers: StdMutex::new(JoinSet::new()),
        }
    }

    fn handlers(&self) -> std::sync::MutexGuard<'_, JoinSet<()>> {
        // A panic while holding the lock leaves the JoinSet intact.
        self.handlers.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Accept forever, retrying after transient errors.
    async fn accept_loop(&self, listener: &TcpListener) {
        loop {
            match listener.accept().await {
                Ok((socket, peer)) => self.dispatch(socket, peer),
                Err(e) => {
                    warn!(error = %e, "HLS accept failed; retrying");
                    tokio::time::sleep(ACCEPT_RETRY_DELAY).await;
                }
            }
        }
    }

    /// Hand an accepted socket to a tracked handler task, or turn it
    /// away with 503 when the connection cap is reached.
    fn dispatch(&self, socket: TcpStream, peer: SocketAddr) {
        let Ok(permit) = self.permits.clone().try_acquire_owned() else {
            warn!(%peer, limit = MAX_CONNECTIONS, "HLS connection limit reached; rejecting");
            // Best effort, never blocks: one non-blocking write on the
            // fresh socket, whose send buffer has room for this short
            // response. (Tokio's `try_write` would refuse before the
            // reactor has seen the socket become writable.)
            if let Ok(mut std_socket) = socket.into_std() {
                let _ = std::io::Write::write(
                    &mut std_socket,
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\
                      Retry-After: 1\r\nConnection: close\r\n\r\n",
                );
            }
            return;
        };
        let state = self.state.clone();
        let tls = self.tls.clone();
        let mut handlers = self.handlers();
        // Reap finished handlers so the set doesn't grow unbounded.
        while handlers.try_join_next().is_some() {}
        handlers.spawn(async move {
            serve_connection(socket, peer, tls, state).await;
            drop(permit);
        });
    }

    /// Abort every connection handler.
    fn shutdown(&self) {
        self.handlers().abort_all();
    }
}

/// Common per-connection dispatch: best-effort `TCP_NODELAY`, then
/// either pass the raw TCP socket to [`http::handle`] (plain HLS)
/// or wrap it in a `TlsAcceptor::accept` first (HTTPS). All HLS
/// servers in this crate funnel through here so the TLS branch
/// stays in one place.
async fn serve_connection(
    socket: TcpStream,
    peer: SocketAddr,
    tls: Option<TlsAcceptor>,
    state: Arc<http::ServerState>,
) {
    let _ = socket.set_nodelay(true);
    // Detect a peer that vanished without closing (receiver powered
    // off) within ~25 s instead of the kernel's 2 h default.
    let keepalive = socket2::TcpKeepalive::new()
        .with_time(TCP_KEEPALIVE_IDLE)
        .with_interval(TCP_KEEPALIVE_INTERVAL)
        .with_retries(TCP_KEEPALIVE_RETRIES);
    if let Err(e) = socket2::SockRef::from(&socket).set_tcp_keepalive(&keepalive) {
        trace!(%peer, error = %e, "could not enable TCP keepalive");
    }
    let result = match tls {
        Some(acceptor) => {
            match tokio::time::timeout(TLS_HANDSHAKE_TIMEOUT, acceptor.accept(socket)).await {
                Ok(Ok(tls_stream)) => http::handle(peer, tls_stream, state).await,
                Ok(Err(e)) => {
                    trace!(%peer, error = %e, "TLS handshake failed");
                    return;
                }
                Err(_) => {
                    trace!(%peer, "TLS handshake timed out");
                    return;
                }
            }
        }
        None => http::handle(peer, socket, state).await,
    };
    if let Err(e) = result {
        trace!(%peer, error = %e, "HLS connection ended");
    }
}

/// Self-managed HLS endpoint backed by an external frame channel.
///
/// Unlike [`HlsServer`], this handle spawns its own accept loop on
/// construction so the caller doesn't have to drive it; dropping the
/// handle stops the accept loop, every connection handler and the
/// segmenter.
///
/// Used by receiver protocols (Chromecast, …) whose capture+encode
/// loop is already driven by the global stream manager — they don't
/// need a long-running `run()` future, they just need a URL the
/// receiver can pull from.
pub struct HlsFrameSink {
    addr: SocketAddr,
    /// `"http"` or `"https"` depending on whether [`HlsConfig::tls`]
    /// was supplied. Exposed so the caller (chromecast LOAD URL
    /// builder) doesn't have to track the same option twice.
    scheme: &'static str,
    ring: Arc<RwLock<SegmentRing>>,
    connections: Arc<Connections>,
    segmenter: JoinHandle<()>,
    accept: JoinHandle<()>,
}

impl HlsFrameSink {
    /// Bind on `addr`, spawn the segmenter (fed by `frames`) and the
    /// HTTP accept loop, and return immediately. The HLS playlist
    /// won't have any segments yet — call [`Self::wait_first_segment`]
    /// before pointing a player at the URL.
    ///
    /// # Errors
    ///
    /// Invalid configuration or bind failure.
    pub async fn start<A: ToSocketAddrs>(
        addr: A,
        frames: mpsc::Receiver<EncodedFrame>,
        audio_frames: Option<mpsc::Receiver<AudioFrame>>,
        parameter_sets: Vec<u8>,
        config: HlsConfig,
    ) -> Result<Self> {
        let listener = TcpListener::bind(addr).await?;
        Self::start_on_listener(listener, frames, audio_frames, parameter_sets, config).await
    }

    /// Same as [`Self::start`] but accepts an already-bound listener.
    /// Use this when the caller pre-bound the port (e.g. via
    /// [`ferricast_core::bind_in_range`]) to keep the bind inside a
    /// known firewall-friendly range.
    ///
    /// # Errors
    ///
    /// Invalid configuration, or the listener's address cannot be
    /// queried.
    pub async fn start_on_listener(
        listener: TcpListener,
        frames: mpsc::Receiver<EncodedFrame>,
        audio_frames: Option<mpsc::Receiver<AudioFrame>>,
        parameter_sets: Vec<u8>,
        config: HlsConfig,
    ) -> Result<Self> {
        config.validate()?;

        let local = listener.local_addr().map_err(FerricastError::from)?;

        let ring = Arc::new(RwLock::new(SegmentRing::new(
            config.keep_segments,
            config.target_duration(),
        )));
        let state = Arc::new(http::ServerState::new(
            ring.clone(),
            config.adaptive.clone(),
        )?);

        let segmenter_ring = ring.clone();
        let cfg_for_segmenter = config.clone();
        let segmenter = tokio::spawn(async move {
            if let Err(e) = segmenter::run_from_frames(
                frames,
                audio_frames,
                parameter_sets,
                segmenter_ring,
                cfg_for_segmenter,
            )
            .await
            {
                error!(error = %e, "HLS frame-source segmenter exited");
            }
        });

        let connections = Arc::new(Connections::new(state, config.tls.clone()));
        let scheme: &'static str = if config.tls.is_some() {
            "https"
        } else {
            "http"
        };
        let accept_connections = connections.clone();
        let accept = tokio::spawn(async move {
            accept_connections.accept_loop(&listener).await;
        });

        info!(
            listen = %local,
            scheme,
            segment_target_s = config.segment_target_secs,
            playlist_target_s = config.playlist_target_duration,
            keep = config.keep_segments,
            "HLS frame sink ready"
        );

        Ok(Self {
            addr: local,
            scheme,
            ring,
            connections,
            segmenter,
            accept,
        })
    }

    /// Bound socket address (always reflects the actual port even if
    /// the caller asked for `:0`).
    #[must_use]
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// `"http"` when [`HlsConfig::tls`] was `None`, `"https"` when it
    /// was set. The chromecast LOAD URL builder uses this to keep
    /// scheme + TLS toggle in one place.
    #[must_use]
    pub fn scheme(&self) -> &'static str {
        self.scheme
    }

    /// Path of the media playlist, `/<token>/index.m3u8`, where the
    /// token is 128 random bits unique to this sink. Build the player
    /// URL as `{scheme}://{host}:{port}{playlist_path}`; requests
    /// outside the token prefix get 404.
    #[must_use]
    pub fn playlist_path(&self) -> String {
        self.connections.state.token.playlist_path()
    }

    /// Only let `ip` (plus loopback and other allowed peers) fetch from
    /// this server; other peers get 403. Until the first call every
    /// peer may fetch. Applies to connections already open, too.
    pub fn allow_peer(&self, ip: IpAddr) {
        self.connections.state.allow_peer(ip);
    }

    /// Wait until at least one segment has been pushed to the ring.
    /// Cheap polling — segments emit within `segment_target_secs +
    /// keyframe_lag` so this resolves in under a second on typical
    /// streams. Returns `false` if the 30-second deadline passes
    /// without any segment (encoder/capture pipeline failed).
    pub async fn wait_first_segment(&self) -> bool {
        ring::wait_for_first_segment(&self.ring).await
    }

    /// Owned future that resolves when the first segment lands (or
    /// after the 30-second deadline if the encoder never produces one).
    ///
    /// Doesn't borrow `self`, so it can be awaited from a `tokio::spawn`
    /// task. Used by receiver protocols (Chromecast `LOAD`, …) that
    /// need to delay a signaling message until the playlist is actually
    /// playable without blocking the frame-feeding code path.
    pub fn first_segment_ready(&self) -> impl Future<Output = bool> + Send + 'static {
        let ring = self.ring.clone();
        async move { ring::wait_for_first_segment(&ring).await }
    }
}

impl Drop for HlsFrameSink {
    fn drop(&mut self) {
        // Stop accepting (reclaims the listener fd), cut every open
        // connection so keep-alive clients can't keep reading a dead
        // ring, and stop the segmenter even if the caller still holds
        // the frame sender.
        self.accept.abort();
        self.connections.shutdown();
        self.segmenter.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_duration_comes_from_config_and_covers_segment_target() {
        let cfg = HlsConfig {
            segment_target_secs: 2.0,
            playlist_target_duration: 4,
            ..HlsConfig::default()
        };
        assert_eq!(cfg.target_duration(), 4);
        let cfg = HlsConfig {
            segment_target_secs: 4.5,
            playlist_target_duration: 4,
            ..HlsConfig::default()
        };
        assert_eq!(cfg.target_duration(), 5);
    }

    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn sink() -> (HlsFrameSink, mpsc::Sender<EncodedFrame>) {
        let (tx, rx) = mpsc::channel(8);
        let sink = HlsFrameSink::start(
            "127.0.0.1:0",
            rx,
            None,
            vec![0, 0, 0, 1, 0x67],
            HlsConfig {
                segment_target_secs: 2.0,
                playlist_target_duration: 4,
                ..HlsConfig::default()
            },
        )
        .await
        .unwrap();
        (sink, tx)
    }

    async fn get(addr: SocketAddr, path: &str) -> String {
        let mut s = TcpStream::connect(addr).await.unwrap();
        s.write_all(format!("GET {path} HTTP/1.0\r\n\r\n").as_bytes())
            .await
            .unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).await.unwrap();
        out
    }

    #[tokio::test]
    async fn connections_over_the_cap_get_503() {
        let (sink, _tx) = sink().await;
        let addr = sink.local_addr();
        let mut idle = Vec::new();
        for _ in 0..MAX_CONNECTIONS {
            idle.push(TcpStream::connect(addr).await.unwrap());
        }
        // Let the accept loop take all of them.
        tokio::time::sleep(Duration::from_millis(200)).await;
        // Read without sending a request: the server answers 503 and
        // closes right away.
        let mut over = TcpStream::connect(addr).await.unwrap();
        let mut reply = String::new();
        over.read_to_string(&mut reply).await.unwrap();
        assert!(reply.starts_with("HTTP/1.1 503"), "{reply}");
        drop(idle);
        tokio::time::sleep(Duration::from_millis(200)).await;
        let reply = get(addr, &sink.playlist_path()).await;
        assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
    }

    #[tokio::test]
    async fn dropping_the_sink_closes_open_connections() {
        let (sink, _tx) = sink().await;
        let mut conn = TcpStream::connect(sink.local_addr()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(sink);
        let mut buf = [0u8; 16];
        let read = tokio::time::timeout(Duration::from_secs(2), conn.read(&mut buf))
            .await
            .expect("handler must be aborted, not left waiting for a request");
        assert!(matches!(read, Ok(0) | Err(_)));
    }

    #[tokio::test]
    async fn routes_require_the_random_token() {
        let (server, _tx) = sink().await;
        let addr = server.local_addr();
        let path = server.playlist_path();
        let token = path
            .strip_prefix('/')
            .and_then(|p| p.strip_suffix("/index.m3u8"))
            .unwrap();
        assert_eq!(token.len(), 32);
        assert!(token.bytes().all(|b| b.is_ascii_hexdigit()));
        let (other, _tx2) = sink().await;
        assert_ne!(other.playlist_path(), path, "tokens must differ per sink");

        let ok = get(addr, &path).await;
        assert!(ok.starts_with("HTTP/1.1 200"), "{ok}");
        assert!(ok.contains("#EXTM3U"), "{ok}");
        for bad in ["/index.m3u8", "/segment-0.ts", "/0123/index.m3u8", "/"] {
            let reply = get(addr, bad).await;
            assert!(reply.starts_with("HTTP/1.1 404"), "{bad}: {reply}");
        }
        let seg = get(addr, &format!("/{token}/segment-0.ts")).await;
        // No segment yet, but the route is reachable under the token.
        assert!(seg.starts_with("HTTP/1.1 404"), "{seg}");
    }
}
