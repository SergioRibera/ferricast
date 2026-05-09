//! Background loop that turns frames into MPEG-TS segments.
//!
//! Each segment starts on a keyframe and ends right *before* the next
//! keyframe that arrives after the configured target duration. Frames
//! consumed across the segment boundary are stashed in `pending` and
//! replayed as the first frame of the next segment, so no frame is
//! duplicated and every segment is independently decodable.
//!
//! The capture-pull loop is paced to `config.target_fps`: if the
//! upstream `ScreenCapture` doesn't deliver a fresh frame inside one
//! frame period (PipeWire on idle desktops routinely pauses for
//! hundreds of ms), we re-encode the most recent one with its
//! timestamp bumped forward. Without this padding the playlist would
//! stop advancing whenever nothing on screen was changing — players
//! would stall, exactly the symptom we're fixing.
//!
//! Once `segment_target_secs` has elapsed inside a segment we ask the
//! encoder to emit an IDR via [`VideoEncoder::request_keyframe`] (a
//! no-op for backends that can't comply, e.g. x264 via the safe
//! crate). That keeps segments anchored to wall clock instead of
//! drifting along the encoder's natural keyint when the actual
//! framerate is below target.

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
    let frame_period =
        Duration::from_secs_f64(1.0 / (config.target_fps.max(1) as f64));
    let frame_period_us = frame_period.as_micros() as u64;

    // Anchor capture-clock microseconds → MPEG-TS 90 kHz PTS. Set on
    // the first frame so PTS starts near 0 and the 33-bit field has
    // ~26.5 h of runway before it wraps.
    let mut pts_anchor_us: Option<u64> = None;

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

    // Inter-frame dt for diagnosing capture-side throughput. At 60 fps
    // we expect ~16-17 ms; values closer to 50-60 ms point at the
    // PipeWire/Vulkan import path stalling. Logged at DEBUG so it's
    // off by default but easy to flip on with `RUST_LOG=ferricast_hls=debug`.
    let mut last_frame_at: Option<Instant> = None;

    // Most recent real frame, kept around so we can re-encode it as
    // pace-padding when the capture source pauses.
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
                        frame_period,
                        frame_period_us,
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
        push_frame(&mut muxer, &first, &mut pts_anchor_us)?;

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
                frame_period,
                frame_period_us,
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
            push_frame(&mut muxer, &encoded, &mut pts_anchor_us)?;
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

/// Pull one frame, with target-fps pacing. Returns:
/// * The next real frame from `capture` if it arrives within
///   `frame_period`.
/// * A duplicate of the last real frame (with timestamp bumped
///   forward by one period) otherwise — but only after we've seen at
///   least one real frame. On cold start the pacer falls through to
///   an unbounded `await` so we never spin before capture is up.
async fn next_paced_frame<S>(
    capture: &mut S,
    last_frame: &mut Option<CapturedFrame>,
    last_frame_at: &mut Option<Instant>,
    frame_period: Duration,
    frame_period_us: u64,
) -> Result<CapturedFrame, FerricastError>
where
    S: ScreenCapture + Send,
{
    let frame = match timeout(frame_period, capture.next_frame()).await {
        Ok(res) => {
            let f = res?;
            *last_frame = Some(f.clone());
            f
        }
        Err(_) => {
            // Period elapsed without a fresh frame. If we have a
            // previous one, re-encode it (with bumped timestamp) so
            // the playlist keeps advancing. Otherwise fall through
            // to an unbounded await — this only happens before the
            // very first real frame.
            match last_frame.as_mut() {
                Some(stored) => {
                    bump_timestamp_us(stored, frame_period_us);
                    stored.clone()
                }
                None => {
                    let f = capture.next_frame().await?;
                    *last_frame = Some(f.clone());
                    f
                }
            }
        }
    };
    record_frame_dt(last_frame_at);
    Ok(frame)
}

fn bump_timestamp_us(frame: &mut CapturedFrame, delta_us: u64) {
    match frame {
        CapturedFrame::Cpu(r) => r.timestamp_us = r.timestamp_us.saturating_add(delta_us),
        CapturedFrame::Gpu(g) => g.timestamp_us = g.timestamp_us.saturating_add(delta_us),
    }
}

fn record_frame_dt(last: &mut Option<Instant>) {
    let now = Instant::now();
    if let Some(prev) = *last {
        debug!(dt_ms = (now - prev).as_millis() as u64, "frame");
    }
    *last = Some(now);
}

fn push_frame(
    muxer: &mut MpegTs,
    encoded: &EncodedFrame,
    anchor: &mut Option<u64>,
) -> Result<(), FerricastError> {
    if anchor.is_none() {
        *anchor = Some(encoded.timestamp_us);
    }
    let base = anchor.unwrap();
    // Saturating: timestamps are monotonic in normal operation, but
    // a clock jump (e.g. capture restart) would otherwise underflow.
    let elapsed_us = encoded.timestamp_us.saturating_sub(base);
    let pts_90k = elapsed_us.saturating_mul(9) / 100;
    // Baseline H.264 has no B-frames → DTS == PTS.
    muxer.add_frame(encoded, pts_90k, pts_90k)
}
