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

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::{TcpListener, ToSocketAddrs};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::{debug, error, info};

use ferricast_core::{FerricastError, Result, ScreenCapture, VideoEncoder};

pub use ring::SegmentRing;

/// Tunables that govern segment cadence and buffering.
///
/// Sensible defaults for desktop screen casting (2 s segments, 6
/// retained, target duration 4 s on the wire) are exposed via
/// [`HlsConfig::default`]. The playlist target duration is
/// deliberately larger than the segment target so the unavoidable
/// keyframe-delay tail never violates RFC 8216 §4.3.3.1.
#[derive(Debug, Clone, Copy)]
pub struct HlsConfig {
    /// Wallclock target a single segment tries to fit into. Real
    /// segments end at the next keyframe after this elapses.
    pub segment_target_secs: f32,
    /// Value advertised in `#EXT-X-TARGETDURATION`. Must be ≥
    /// `ceil(segment_target_secs)` plus headroom.
    pub playlist_target_duration: u8,
    /// Number of segments retained in the live ring. Must be ≥ 3 so
    /// players can prebuffer.
    pub keep_segments: usize,
}

impl Default for HlsConfig {
    fn default() -> Self {
        Self {
            segment_target_secs: 2.0,
            playlist_target_duration: 4,
            keep_segments: 6,
        }
    }
}

/// Live HLS server. [`Self::start`] spawns the capture/encode loop in
/// the background and returns once the first segment is ready;
/// [`Self::run`] then accepts connections forever.
pub struct HlsServer {
    listener: TcpListener,
    ring: Arc<RwLock<SegmentRing>>,
    /// Cancels the capture loop when the server is dropped.
    _segmenter: JoinHandle<()>,
}

impl HlsServer {
    /// Bind, start segmenting, wait until the first segment lands,
    /// and return. The server is ready to serve as soon as this
    /// resolves — players that connect immediately will see a
    /// non-empty playlist on their first request.
    pub async fn start<S, E, A>(
        addr: A,
        capture: S,
        encoder: E,
        config: HlsConfig,
    ) -> Result<Self>
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
        let cfg = config;
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
        tokio::spawn(async move {
            if let Err(e) = http::handle(socket, ring).await {
                debug!(peer = %peer, error = %e, "HLS connection ended");
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
                    tokio::spawn(async move {
                        if let Err(e) = http::handle(socket, ring).await {
                            debug!(peer = %peer, error = %e, "HLS connection ended");
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
