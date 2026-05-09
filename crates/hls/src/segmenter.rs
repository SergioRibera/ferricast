//! Background loop that turns frames into MPEG-TS segments.
//!
//! Each segment starts on a keyframe and ends right *before* the next
//! keyframe that arrives after the configured target duration. Frames
//! consumed across the segment boundary are stashed in `pending` and
//! replayed as the first frame of the next segment, so no frame is
//! duplicated and every segment is independently decodable.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use ferricast_core::{EncodedFrame, FerricastError, ScreenCapture, VideoEncoder};
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

    loop {
        muxer.start_segment();

        // First frame of the segment must be a keyframe. Either we
        // carried one over from the previous iteration, or we drain
        // frames here until the encoder produces one.
        let first = match pending.take() {
            Some(f) => f,
            None => loop {
                let frame = capture.next_frame().await?;
                record_frame_dt(&mut last_frame_at);
                match encoder.encode(frame) {
                    Ok(e) if e.is_keyframe => break e,
                    Ok(_) => continue, // pre-IDR frame, drop
                    Err(err) => {
                        warn!(error = %err, "encoder.encode failed, dropping frame");
                        continue;
                    }
                }
            },
        };

        let started = Instant::now();
        push_frame(&mut muxer, &first, &mut pts_anchor_us)?;

        // Continue pulling frames until we see another keyframe past
        // the target duration. That next keyframe is *not* part of
        // this segment — it gets stashed in `pending` and starts the
        // next one.
        loop {
            let frame = capture.next_frame().await?;
            record_frame_dt(&mut last_frame_at);
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
