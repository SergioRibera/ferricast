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

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;

use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, ToSocketAddrs};
use tokio::sync::{RwLock, mpsc};
use tokio::task::JoinHandle;
use tracing::{error, info, trace};

use ferricast_core::{EncodedFrame, FerricastError, Result, ScreenCapture, VideoEncoder};

pub use ring::SegmentRing;

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
            part_target_secs: None,
        }
    }
}

/// Live HLS server. [`Self::start`] spawns the capture/encode loop in
/// the background and returns once the first segment is ready;
/// [`Self::run`] then accepts connections forever.
pub struct HlsServer {
    listener: TcpListener,
    ring: Arc<RwLock<SegmentRing>>,
    adaptive: Option<Arc<ferricast_core::AdaptiveBitrateState>>,
    stats: Arc<stats::SessionStats>,
    /// Cancels the capture loop when the server is dropped.
    _segmenter: JoinHandle<()>,
}

impl HlsServer {
    /// Bind, start segmenting, wait until the first segment lands,
    /// and return. The server is ready to serve as soon as this
    /// resolves — players that connect immediately will see a
    /// non-empty playlist on their first request.
    pub async fn start<S, E, A>(addr: A, capture: S, encoder: E, config: HlsConfig) -> Result<Self>
    where
        S: ScreenCapture + Send + 'static,
        E: VideoEncoder + Send + 'static,
        A: ToSocketAddrs,
    {
        if config.keep_segments < 3 {
            return Err(FerricastError::Hls(format!(
                "keep_segments={} too small (minimum 3)",
                config.keep_segments
            )));
        }
        if (config.playlist_target_duration as f32) < config.segment_target_secs {
            return Err(FerricastError::Hls(format!(
                "playlist_target_duration={}s < segment_target_secs={}s",
                config.playlist_target_duration, config.segment_target_secs
            )));
        }

        let listener = TcpListener::bind(addr).await?;
        let local = listener
            .local_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|_| "?".into());

        let ring = Arc::new(RwLock::new(SegmentRing::new(config.keep_segments)));
        let writer = ring.clone();
        let cfg = config.clone();
        let segmenter = tokio::spawn(async move {
            if let Err(e) = segmenter::run(capture, encoder, writer, cfg).await {
                error!(error = %e, "HLS segmenter loop exited");
            }
        });
    
       ring::wait_for_first_segment(&ring).await;

        info!(
            listen = %local,
            segment_target_s = config.segment_target_secs,
            playlist_target_s = config.playlist_target_duration,
            keep = config.keep_segments,
            "HLS server ready"
        );

        Ok(Self {
            listener,
            ring,
            adaptive: config.adaptive.clone(),
            stats: Arc::new(stats::SessionStats::new()),
            _segmenter: segmenter,
        })
    }

    /// Bound socket address. Useful when [`Self::start`] is given
    /// `0.0.0.0:0` and the caller needs to discover the assigned port.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.listener.local_addr().map_err(FerricastError::from)
    }

    /// Accept a single connection and dispatch it to a per-connection
    /// task. Useful when the caller wants to drive accept manually
    /// (e.g. integrate with a select! loop).
    pub async fn accept_one(&self) -> Result<()> {
        let (socket, peer) = self.listener.accept().await?;
        let ring = self.ring.clone();
        let adaptive = self.adaptive.clone();
        let stats = self.stats.clone();
        tokio::spawn(async move {
            if let Err(e) = http::handle(socket, ring, adaptive, stats).await {
                trace!(peer = %peer, error = %e, "HLS connection ended");
            }
        });
        Ok(())
    }

    /// Accept loop. Runs until the listener errors out (interface
    /// going away, fd exhaustion, …). Per-connection failures are
    /// logged but never bring down the loop.
    pub async fn run(self) -> Result<()> {
        loop {
            match self.listener.accept().await {
                Ok((socket, peer)) => {
                    let ring = self.ring.clone();
                    let adaptive = self.adaptive.clone();
                    let stats = self.stats.clone();
              
                    tokio::spawn(async move {
                        if let Err(e) = http::handle(socket, ring, adaptive, stats).await {
                            trace!(peer = %peer, error = %e, "HLS connection ended");
                        }
                    });
                }
                Err(e) => {
                    error!(error = %e, "HLS listener accept failed");
                    return Err(FerricastError::from(e));
                }
            }
        }
    }
}

/// Self-managed HLS endpoint backed by an external frame channel.
///
/// Unlike [`HlsServer`], this handle spawns its own accept loop on
/// construction so the caller doesn't have to drive it; dropping the
/// handle aborts the accept loop and dropping the [`mpsc::Sender`]
/// the caller still holds shuts the segmenter down.
///
/// Used by receiver protocols (Chromecast, …) whose capture+encode
/// loop is already driven by the global stream manager — they don't
/// need a long-running `run()` future, they just need a URL the
/// receiver can pull from.
pub struct HlsFrameSink {
    addr: SocketAddr,
    ring: Arc<RwLock<SegmentRing>>,
    _segmenter: JoinHandle<()>,
    accept: JoinHandle<()>,
}

impl HlsFrameSink {
    /// Bind on `addr`, spawn the segmenter (fed by `frames`) and the
    /// HTTP accept loop, and return immediately. The HLS playlist
    /// won't have any segments yet — call [`Self::wait_first_segment`]
    /// before pointing a player at the URL.
    pub async fn start<A: ToSocketAddrs>(
        addr: A,
        frames: mpsc::Receiver<EncodedFrame>,
        parameter_sets: Vec<u8>,
        config: HlsConfig,
    ) -> Result<Self> {
        if config.keep_segments < 3 {
            return Err(FerricastError::Hls(format!(
                "keep_segments={} too small (minimum 3)",
                config.keep_segments
            )));
        }
        if (config.playlist_target_duration as f32) < config.segment_target_secs {
            return Err(FerricastError::Hls(format!(
                "playlist_target_duration={}s < segment_target_secs={}s",
                config.playlist_target_duration, config.segment_target_secs
            )));
        }

        let listener = TcpListener::bind(addr).await?;
        let local = listener.local_addr().map_err(FerricastError::from)?;

        let ring = Arc::new(RwLock::new(SegmentRing::new(config.keep_segments)));
        let segmenter_ring = ring.clone();
        let adaptive_for_segmenter = config.adaptive.clone();
        let cfg_for_segmenter = config.clone();
        let segmenter = tokio::spawn(async move {
            if let Err(e) = segmenter::run_from_frames(
                frames,
                parameter_sets,
                segmenter_ring,
                cfg_for_segmenter,
            )
            .await
            {
                error!(error = %e, "HLS frame-source segmenter exited");
            }
            // Drop reference once the segmenter exits so the
            // adaptive state can be freed.
            drop(adaptive_for_segmenter);
        });

        let accept_ring = ring.clone();
        let accept_adaptive = config.adaptive.clone();
        let stats = Arc::new(stats::SessionStats::new());
        let accept_stats = stats.clone();
        let accept = tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((socket, peer)) => {
                        let r = accept_ring.clone();
                        let a = accept_adaptive.clone();
                        let s = accept_stats.clone();
                        tokio::spawn(async move {
                            println!("conx");
                            if let Err(e) = http::handle(socket, r, a, s).await {
                                trace!(peer = %peer, error = %e, "HLS connection ended");
                            }
                        });
                    }
                    Err(e) => {
                        error!(error = %e, "HLS frame-sink accept failed");
                        return;
                    }
                }
            }
        });

        info!(
            listen = %local,
            segment_target_s = config.segment_target_secs,
            playlist_target_s = config.playlist_target_duration,
            keep = config.keep_segments,
            "HLS frame sink ready"
        );

        Ok(Self {
            addr: local,
            ring,
            _segmenter: segmenter,
            accept,
        })
    }

    /// Bound socket address (always reflects the actual port even if
    /// the caller asked for `:0`).
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Wait until at least one segment has been pushed to the ring.
    /// Cheap polling — segments emit within `segment_target_secs +
    /// keyframe_lag` so this resolves in under a second on typical
    /// streams.
    pub async fn wait_first_segment(&self) {
        ring::wait_for_first_segment(&self.ring).await;
    }

    /// Owned future that resolves when the first segment lands.
    ///
    /// Same semantics as [`Self::wait_first_segment`] but doesn't
    /// borrow `self`, so it can be awaited from a `tokio::spawn`
    /// task that doesn't hold a reference to the sink. Used by
    /// receiver protocols (Chromecast `LOAD`, …) that need to delay
    /// a signaling message until the playlist is actually playable
    /// without blocking the frame-feeding code path.
    pub fn first_segment_ready(&self) -> impl Future<Output = ()> + Send + 'static {
        let ring = self.ring.clone();
        async move {
            ring::wait_for_first_segment(&ring).await;
        }
    }
}

impl Drop for HlsFrameSink {
    fn drop(&mut self) {
        // Abort the accept loop so the listener fd is reclaimed.
        // The segmenter exits on its own as soon as the caller drops
        // their `mpsc::Sender`, so we don't have to abort it here —
        // but we abort defensively in case the sender outlives us.
        self.accept.abort();
        self._segmenter.abort();
    }
}
