//! Background loop that turns frames into MPEG-TS segments.
//!
//! Each segment starts on a keyframe and ends right *before* the next
//! keyframe that arrives after the configured target duration. Frames
//! consumed across the segment boundary are stashed in `pending` and
//! replayed as the first frame of the next segment, so no frame is
//! duplicated and every segment is independently decodable.
//!
//! Pacing strategy: the segmenter only synthesises duplicate frames
//! when the upstream `ScreenCapture` actually stalls
//! (`PACE_STALL_THRESHOLD` = ~3× the configured frame period). When
//! the upstream is delivering at a stable rate below `target_fps`
//! (e.g. Mutter's screencast portal commonly settles at 30 fps even
//! when we ask for 60), we just encode whatever it gives us — the
//! player decodes at PTS rate and renders at the natural cadence.
//! Aggressive 1-per-`frame_period` duplication used to make 30 fps
//! sources visually stutter, with each dup sitting between two real
//! frames as a one-frame "freeze" the eye perceives as micro-pause.
//!
//! Once `segment_target_secs` has elapsed inside a segment we ask
//! the encoder to emit an IDR via [`VideoEncoder::request_keyframe`]
//! (a no-op for backends that can't comply, e.g. x264 via the safe
//! crate). That keeps segments anchored to wall clock instead of
//! drifting along the encoder's natural keyint when the actual
//! framerate is below target.
//!
//! PTS is wall-clock based: each frame's PTS is the wall-clock
//! microseconds since the segmenter's first frame. This makes both
//! real frames and the rare pace-padded duplicates carry
//! self-consistent timestamps regardless of upstream delivery rate
//! — the player paces playback by PTS and renders at whatever
//! cadence the source actually produced.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::sync::RwLock;
use tokio::time::timeout;
use tracing::{debug, info, warn};

use ferricast_core::{CapturedFrame, EncodedFrame, FerricastError, ScreenCapture, VideoEncoder};
use ferricast_muxer::Muxer;
use ferricast_muxer::mpeg_ts::MpegTs;

use crate::HlsConfig;
use crate::ring::SegmentRing;

/// Lower bound on how long we wait for a real frame before deciding
/// the upstream has stalled and synthesising a duplicate. Set
/// generously above the largest reasonable capture-source frame
/// period so a stable ≤30 fps source (Mutter's common output) never
/// triggers a dup — only true stalls (>~50 ms gaps) do.
const PACE_STALL_FLOOR: Duration = Duration::from_millis(50);

pub async fn run<S, E>(
    mut capture: S,
    mut encoder: E,
    ring: Arc<RwLock<SegmentRing>>,
    config: HlsConfig,
) -> Result<(), FerricastError>
where
    S: ScreenCapture + Send,
    E: VideoEncoder + Send,
{
    // SPS+PPS in Annex B form. Captured once and injected at every
    // keyframe access unit so any segment is self-describing.
    let parameter_sets = encoder.get_headers()?;
    if parameter_sets.is_empty() {
        return Err(FerricastError::Encoder(
            "encoder produced empty H.264 parameter sets; refusing to start segmenter".into(),
        ));
    }
    let target = Duration::from_secs_f32(config.segment_target_secs);
    let frame_period = Duration::from_secs_f64(1.0 / (config.target_fps.max(1) as f64));
    // Wait at least 3× the frame period before deciding we're stalled
    // (so even a temporarily late frame at the configured fps doesn't
    // trigger a dup), but never less than `PACE_STALL_FLOOR` —
    // upstream sources that settle below the configured fps would
    // otherwise see a dup every `frame_period` exactly between real
    // frames, which is what showed up in the user's logs as
    // dt_ms=21 dup / dt_ms=12 real / dt_ms=21 dup / ... cycles.
    let pace_stall = (frame_period.saturating_mul(3)).max(PACE_STALL_FLOOR);

    // Anchor for wall-clock PTS. Set on the very first frame we push
    // so the first frame lands at PTS≈0 and every later frame is
    // `now - anchor` µs. Monotonic by construction (`Instant`), so
    // PTS is always strictly increasing.
    let mut pts_anchor: Option<Instant> = None;

    // Keyframe carried over from the previous segment. The boundary
    // detector consumes one keyframe to know "the new segment starts
    // here"; we keep that frame as the first frame of the next
    // segment instead of dropping or duplicating it.
    let mut pending: Option<EncodedFrame> = None;

    // Single muxer reused across segments. Its continuity-counter
    // state must persist or ffmpeg's MPEG-TS demuxer flags every
    // boundary as a packet error.
    let mut muxer = MpegTs::default();
    muxer.config(parameter_sets.clone())?;

    // Inter-frame dt for diagnosing capture-side throughput. Tagged
    // with `kind="real"` (fresh PW frame) or `kind="dup"` (pace-pad
    // copy of the previous one). With the relaxed pacer a healthy
    // run shows essentially no `dup` lines except during real
    // capture stalls.
    let mut last_frame_at: Option<Instant> = None;

    // Most recent real frame, kept around so we can re-encode it as
    // pace-padding when the capture source actually stalls past
    // `pace_stall`.
    let mut last_frame: Option<CapturedFrame> = None;

    loop {
        muxer.start_segment();

        // First frame of the segment must be a keyframe. Either we
        // carried one over from the previous iteration, or we drain
        // frames here until the encoder produces one. On cold start
        // request a keyframe up front so we don't waste an entire
        // GOP waiting for the encoder's natural interval to fire.
        let first = match pending.take() {
            Some(f) => f,
            None => {
                encoder.request_keyframe();
                loop {
                    let frame = next_paced_frame(
                        &mut capture,
                        &mut last_frame,
                        &mut last_frame_at,
                        pace_stall,
                    )
                    .await?;
                    match encoder.encode(frame) {
                        Ok(e) if e.is_keyframe => break e,
                        Ok(_) => continue, // pre-IDR frame, drop
                        Err(err) => {
                            warn!(error = %err, "encoder.encode failed, dropping frame");
                            continue;
                        }
                    }
                }
            }
        };

        let started = Instant::now();
        let mut requested_idr = false;
        push_frame(&mut muxer, &first, &mut pts_anchor)?;

        // Continue pulling frames until we see another keyframe past
        // the target duration. Once we've crossed the target, ask
        // the encoder for an IDR — that pulls the segment boundary
        // back onto wall clock instead of waiting for whatever the
        // natural keyint produces (which would make segments far
        // longer than `target` whenever effective fps < target).
        loop {
            let frame = next_paced_frame(
                &mut capture,
                &mut last_frame,
                &mut last_frame_at,
                pace_stall,
            )
            .await?;

            if !requested_idr && started.elapsed() >= target {
                encoder.request_keyframe();
                requested_idr = true;
            }

            let encoded = match encoder.encode(frame) {
                Ok(e) => e,
                Err(err) => {
                    warn!(error = %err, "encoder.encode failed, dropping frame");
                    continue;
                }
            };

            if encoded.is_keyframe && started.elapsed() >= target {
                pending = Some(encoded);
                break;
            }
            push_frame(&mut muxer, &encoded, &mut pts_anchor)?;
        }

        let elapsed = started.elapsed();
        let bytes = Bytes::from(muxer.drain());
        let size_bytes = bytes.len();
        let seq = ring.write().await.push(elapsed.as_secs_f32(), false, bytes);
        // First few segments at INFO so the user can sanity-check
        // cadence (segments far longer than target = encoder keyint
        // too high). Steady-state goes to DEBUG to keep logs quiet.
        if seq < 3 {
            info!(
                seq,
                duration_ms = elapsed.as_millis() as u64,
                size_kb = size_bytes / 1024,
                "segment ready"
            );
        } else {
            debug!(
                seq,
                duration_ms = elapsed.as_millis() as u64,
                size_kb = size_bytes / 1024,
                "segment ready"
            );
        }
    }
}

/// Pull one frame from `capture`. If the next frame doesn't arrive
/// within `stall_threshold`, fall back to a duplicate of the last
/// real frame so the playlist keeps advancing during true capture
/// stalls. The threshold is large on purpose (≥ `PACE_STALL_FLOOR`,
/// well above any reasonable steady-state inter-frame gap) so a
/// source running below the configured `target_fps` doesn't get
/// peppered with dups between every real frame — those dups looked
/// like micro-pauses to the user.
async fn next_paced_frame<S>(
    capture: &mut S,
    last_frame: &mut Option<CapturedFrame>,
    last_frame_at: &mut Option<Instant>,
    stall_threshold: Duration,
) -> Result<CapturedFrame, FerricastError>
where
    S: ScreenCapture + Send,
{
    let (frame, kind) = match timeout(stall_threshold, capture.next_frame()).await {
        Ok(res) => {
            let f = res?;
            *last_frame = Some(f.clone());
            (f, "real")
        }
        Err(_) => {
            // Capture has actually stalled past `stall_threshold`.
            // Re-encode the last real frame so the playlist keeps
            // advancing. Wall-clock PTS in `push_frame` makes the
            // dup land at its true wall-clock position regardless
            // of the gap length.
            match last_frame.as_ref() {
                Some(stored) => (stored.clone(), "dup"),
                None => {
                    // No real frame yet — fall through to an
                    // unbounded await so we don't spin during cold
                    // start. Only happens before the very first
                    // capture frame.
                    let f = capture.next_frame().await?;
                    *last_frame = Some(f.clone());
                    (f, "real")
                }
            }
        }
    };
    record_frame_dt(last_frame_at, kind);
    Ok(frame)
}

/// Records inter-frame wallclock spacing tagged with whether the
/// frame was a fresh capture (`"real"`) or a pace-pad copy of the
/// previous one (`"dup"`). With the loosened pacer a healthy run
/// shows essentially no `dup` lines.
fn record_frame_dt(last: &mut Option<Instant>, kind: &'static str) {
    let now = Instant::now();
    if let Some(prev) = *last {
        debug!(dt_ms = (now - prev).as_millis() as u64, kind, "frame");
    }
    *last = Some(now);
}

/// Convert wall-clock time-since-anchor to a 90 kHz MPEG-TS PTS and
/// hand the frame to the muxer. The first call sets the anchor at
/// `Instant::now()` so the very first frame lands at PTS≈0; every
/// subsequent call computes `now - anchor` in microseconds and
/// rescales. `Instant` is monotonic, so PTS is too.
fn push_frame(
    muxer: &mut MpegTs,
    encoded: &EncodedFrame,
    anchor: &mut Option<Instant>,
) -> Result<(), FerricastError> {
    let now = Instant::now();
    let pts_us = match *anchor {
        Some(start) => now.saturating_duration_since(start).as_micros() as u64,
        None => {
            *anchor = Some(now);
            0
        }
    };
    let pts_90k = pts_us.saturating_mul(9) / 100;
    // Baseline H.264 has no B-frames → DTS == PTS.
    muxer.add_frame(encoded, pts_90k, pts_90k)
}
