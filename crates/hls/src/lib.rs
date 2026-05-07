//! Live HLS server backed by a continuous capture/encode pipeline.
//!
//! ## Why this is shaped the way it is
//!
//! The previous version captured + encoded + muxed 180 frames *per
//! HTTP request*. The player therefore had to wait several seconds
//! for the very first byte of every segment, and ffplay's `M-V`
//! drift kept growing. Latency ≈ (segment_duration + capture_lag) ×
//! buffered_segments — minimum 9–18 s.
//!
//! The new design separates *production* from *serving*:
//!
//! * A single background task (`segmenter_loop`) owns `capture` and
//!   `encoder`, runs as fast as the GPU/encoder allow, and emits
//!   complete MPEG-TS segments into a bounded ring buffer
//!   (`SegmentRing`). Segments are closed at the next encoder
//!   keyframe after `target_duration` has elapsed, so every segment
//!   is independently decodable.
//! * `serve()` accepts a TCP connection and dispatches it to a
//!   per-connection task. The handler reads the ring buffer with a
//!   short `RwLock` borrow and writes the playlist or segment bytes
//!   to the client. Multiple concurrent clients are fine — they all
//!   read the same ring.
//!
//! Steady-state latency is now `target_duration × keep_segments`
//! (the player needs ~3 segments to start). With 1s segments and 6
//! retained that's ~3 s end-to-end.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream, ToSocketAddrs};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

use ferricast_core::{FerricastError, ScreenCapture, VideoEncoder};
use ferricast_m3u8::{M3u8Version, M3u8Writer};
use ferricast_muxer::Muxer;
use ferricast_muxer::mpeg_ts::MpegTs;

/// Wallclock window we try to fit a single MPEG-TS segment into.
/// Real segments end on the next encoder keyframe after this elapses,
/// so they're typically 2–3 s.
const SEGMENT_TARGET_SECS: f32 = 2.0;
/// `EXT-X-TARGETDURATION` advertised in the playlist. RFC 8216 §4.3.3.1
/// requires this to be ≥ ceil(any actual segment duration) — we leave
/// room above `SEGMENT_TARGET_SECS` so the keyframe-delay tail never
/// pushes us over.
const PLAYLIST_TARGET_DURATION: u8 = 4;
/// Number of segments retained in the ring. Player needs at least 3
/// to start playback, plus a small reorder cushion. 6 ≈ 12 s of
/// rolling buffer.
const RING_CAPACITY: usize = 6;

#[derive(Clone)]
struct Segment {
    seq: u64,
    /// Wallclock duration spent capturing the segment, in seconds.
    duration_secs: f32,
    data: Bytes,
}

struct SegmentRing {
    segments: VecDeque<Segment>,
    next_seq: u64,
}

impl SegmentRing {
    fn new() -> Self {
        Self {
            segments: VecDeque::with_capacity(RING_CAPACITY),
            next_seq: 0,
        }
    }

    /// Push a freshly produced segment, evicting the oldest entry
    /// when we hit capacity. Returns the assigned sequence number.
    fn push(&mut self, duration_secs: f32, data: Bytes) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.segments.push_back(Segment {
            seq,
            duration_secs,
            data,
        });
        while self.segments.len() > RING_CAPACITY {
            self.segments.pop_front();
        }
        seq
    }

    fn get(&self, seq: u64) -> Option<&Segment> {
        // Linear scan — `RING_CAPACITY` is tiny.
        self.segments.iter().find(|s| s.seq == seq)
    }

    /// Build the live playlist with whatever's currently in the ring.
    fn build_playlist(&self) -> Result<String, FerricastError> {
        let first_seq = self.segments.front().map(|s| s.seq).unwrap_or(0);
        let mut w = M3u8Writer::default()
            .set_version(M3u8Version::V3)
            .set_target_duration(PLAYLIST_TARGET_DURATION)
            .set_media_seq(first_seq);
        for s in &self.segments {
            w = w.add_segment(s.duration_secs, format!("video{}.ts", s.seq))?;
        }
        w.to_string()
    }
}

/// Live HLS server. `listen()` spawns the capture loop in the
/// background; `serve()` accepts a single connection and hands it to
/// a per-client task.
pub struct HlsServer {
    listener: TcpListener,
    ring: Arc<RwLock<SegmentRing>>,
    /// Cancels the capture loop when the server is dropped.
    _segmenter: JoinHandle<()>,
}

impl HlsServer {
    pub async fn listen<S, E, A>(
        addr: A,
        encoder: E,
        capture: S,
    ) -> Result<Self, FerricastError>
    where
        S: ScreenCapture + Send + 'static,
        E: VideoEncoder + Send + 'static,
        A: ToSocketAddrs,
    {
        let listener = TcpListener::bind(addr).await?;
        let ring: Arc<RwLock<SegmentRing>> = Arc::new(RwLock::new(SegmentRing::new()));

        let writer = ring.clone();
        let segmenter = tokio::spawn(async move {
            if let Err(e) = segmenter_loop(capture, encoder, writer).await {
                error!(error = %e, "HLS segmenter loop exited");
            }
        });

        // Block until the first segment has landed in the ring.
        // Without this `serve()` would happily reply to the player's
        // first `GET /` with an empty playlist, and ffplay treats
        // that as a hard error ("Empty segment") rather than
        // retrying.
        wait_for_first_segment(&ring).await;
        info!(
            segment_target_s = SEGMENT_TARGET_SECS,
            playlist_target_s = PLAYLIST_TARGET_DURATION,
            "HLS server ready"
        );

        Ok(Self {
            listener,
            ring,
            _segmenter: segmenter,
        })
    }

    /// Accept one connection and dispatch it to a per-connection
    /// task. Returns when the connection is *accepted*, not when the
    /// response is fully written.
    pub async fn serve(&mut self) -> Result<(), FerricastError> {
        let (socket, _peer) = self.listener.accept().await?;
        let ring = self.ring.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(socket, ring).await {
                debug!(error = %e, "HLS connection handler exited");
            }
        });
        Ok(())
    }
}

/// Poll the ring until the segmenter has pushed at least one
/// segment, then return. Light-weight short-poll; the segmenter
/// emits within `SEGMENT_TARGET_SECS + keyframe_lag`.
async fn wait_for_first_segment(ring: &Arc<RwLock<SegmentRing>>) {
    loop {
        if !ring.read().await.segments.is_empty() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Background loop: pull frames from `capture`, push through
/// `encoder`, mux into MPEG-TS, close a segment whenever a keyframe
/// arrives after `target_duration` has elapsed.
async fn segmenter_loop<S, E>(
    mut capture: S,
    mut encoder: E,
    ring: Arc<RwLock<SegmentRing>>,
) -> Result<(), FerricastError>
where
    S: ScreenCapture + Send,
    E: VideoEncoder + Send,
{
    let target = Duration::from_secs_f32(SEGMENT_TARGET_SECS);
    // SPS/PPS — emitted once by x264 at start; we re-feed the same
    // bytes into every fresh muxer so each segment is independently
    // decodable.
    let headers = encoder.get_headers()?;

    loop {
        let mut muxer = MpegTs::default();
        muxer.config(headers.clone())?;

        let started = Instant::now();
        let mut have_keyframe_in_segment = false;

        loop {
            let frame = capture.next_frame().await?;
            let encoded = match encoder.encode(frame) {
                Ok(e) => e,
                Err(err) => {
                    warn!(error = %err, "encoder.encode failed, dropping frame");
                    continue;
                }
            };
            if encoded.is_keyframe {
                have_keyframe_in_segment = true;
            }
            muxer.add_frame(encoded)?;

            // Close the segment once we've seen at least one keyframe
            // AND the target duration has passed. Closing on the next
            // keyframe (rather than mid-GOP) keeps the next segment
            // independently decodable.
            if have_keyframe_in_segment && started.elapsed() >= target {
                break;
            }
        }

        let elapsed = started.elapsed();
        let bytes = Bytes::from(muxer.drain());
        let mut g = ring.write().await;
        let seq = g.push(elapsed.as_secs_f32(), bytes);
        debug!(
            seq,
            duration_ms = elapsed.as_millis() as u64,
            "segment ready"
        );
    }
}

/// Per-connection HTTP handler. Reads the request line + headers,
/// decides whether the client wants the playlist or a segment, and
/// writes the response. No keep-alive; the server is intentionally
/// per-request.
async fn handle_connection(
    mut socket: TcpStream,
    ring: Arc<RwLock<SegmentRing>>,
) -> Result<(), FerricastError> {
    let mut req_text = String::new();
    {
        let buf = BufReader::new(&mut socket);
        let mut lines = buf.lines();
        while let Some(line) = lines.next_line().await? {
            if line.is_empty() {
                req_text.push_str("\r\n");
                break;
            }
            req_text.push_str(&line);
            req_text.push_str("\r\n");
        }
    }

    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut req = httparse::Request::new(&mut headers);
    if req.parse(req_text.as_bytes()).map(|r| r.is_partial()).unwrap_or(true) {
        let _ = socket
            .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
            .await;
        return Err(FerricastError::Hls("malformed http request".into()));
    }

    let path = req.path.unwrap_or("/");
    match path {
        "/" | "/playlist.m3u8" => {
            let playlist = {
                let g = ring.read().await;
                g.build_playlist()?
            };
            // RFC 8216 §4 specifies `application/vnd.apple.mpegurl`
            // as the canonical MIME type; ffplay 8.x warns
            // ("mime type is not rfc8216 compliant") when it sees
            // the older `application/x-mpegurl`.
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/vnd.apple.mpegurl\r\n")
                .await?;
            socket
                .write_all(
                    format!(
                        "Cache-Control: no-cache\r\nContent-Length: {}\r\n\r\n",
                        playlist.len()
                    )
                    .as_bytes(),
                )
                .await?;
            socket.write_all(playlist.as_bytes()).await?;
        }
        p if p.starts_with("/video") && p.ends_with(".ts") => {
            let seq_str = &p["/video".len()..p.len() - ".ts".len()];
            let Ok(seq) = seq_str.parse::<u64>() else {
                let _ = socket
                    .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
                    .await;
                return Ok(());
            };
            let segment = {
                let g = ring.read().await;
                g.get(seq).cloned()
            };
            match segment {
                Some(s) => {
                    socket
                        .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: video/mp2t\r\n")
                        .await?;
                    socket
                        .write_all(
                            format!(
                                "Cache-Control: no-cache\r\nContent-Length: {}\r\n\r\n",
                                s.data.len()
                            )
                            .as_bytes(),
                        )
                        .await?;
                    socket.write_all(&s.data).await?;
                }
                None => {
                    let _ = socket
                        .write_all(
                            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n",
                        )
                        .await;
                }
            }
        }
        _ => {
            let _ = socket
                .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                .await;
        }
    }

    Ok(())
}
