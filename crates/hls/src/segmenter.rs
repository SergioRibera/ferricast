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
use tokio::sync::{mpsc, RwLock};
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

    // Adaptive PTS rate. Start at the configured frame period, then
    // refine after each segment using `measured = wall_span /
    // frames_pushed`. Per-frame PTS increments stay uniform inside
    // any one segment (so a single slow encode doesn't surface as a
    // visible freeze the way wall-clock PTS does), but the rate
    // gradually converges to actual production cadence so PTS span
    // ≈ wall span and the player consumes segments at the rate we
    // produce them — no drift, no segment eviction race.
    //
    // Why not pure synthetic (`frame_count * frame_period_us`):
    // each segment took ~2.05 s of wall clock to produce but a
    // synthetic 60 fps stamp gave it PTS span = exactly 2.000 s.
    // Players paced consumption by PTS, not EXTINF, so they ran
    // 50 ms slower than the producer per segment — the "expired
    // from playlists" + "Packet corrupt" tail.
    //
    // Why not pure wall-clock (`now - anchor`): exposes encoder
    // time variance directly. NVENC encodes most frames in ~5 ms
    // but occasionally takes 30-40 ms (complex frame, GPU
    // contention); each slow encode became a visible freeze of
    // the preceding frame because the player held it until the
    // late frame's PTS arrived.
    let mut pts_us: u64 = 0;
    let mut effective_frame_period_us: u64 = frame_period_us;

    // Keyframe carried over from the previous segment. The boundary
    // detector consumes one keyframe to know "the new segment starts
    // here"; we keep that frame as the first frame of the next
    // segment instead of dropping or duplicating it.
    let mut pending: Option<EncodedFrame> = None;

    // Single muxer reused across segments. Its continuity-counter
    // state must persist or ffmpeg's MPEG-TS demuxer flags every
    // boundary as a packet error.
    let mut muxer = MpegTs::default().with_silent_audio(config.inject_silent_audio);
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
        let mut frames_in_segment: u64 = 0;
        push_frame(&mut muxer, &first, pts_us)?;
        pts_us = pts_us.saturating_add(effective_frame_period_us);
        frames_in_segment += 1;

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
            push_frame(&mut muxer, &encoded, pts_us)?;
            pts_us = pts_us.saturating_add(effective_frame_period_us);
            frames_in_segment += 1;
        }

        let elapsed = started.elapsed();

        // Refine the per-frame PTS increment based on this segment's
        // actual production: `wall / frames`. EMA-smoothed (7/8 old,
        // 1/8 new) so a one-off slow segment doesn't yank the rate
        // around. After ~5-10 segments the value converges to the
        // real per-frame wall cost (~17 ms typical at 60 fps target
        // with NVENC), which makes PTS span ≈ wall span and stops
        // the player drifting away from the live edge.
        //
        // Skip the very first segment (frames_in_segment can be
        // dominated by warmup).
        if frames_in_segment >= 30 {
            let measured =
                (elapsed.as_micros() as u64).saturating_div(frames_in_segment);
            // Clamp to avoid pathological measurements (e.g. a long
            // capture stall during the segment) hijacking the rate.
            // Stay within ±50 % of the configured frame period.
            let lo = frame_period_us / 2;
            let hi = frame_period_us.saturating_mul(3) / 2;
            let measured_clamped = measured.clamp(lo, hi);
            effective_frame_period_us =
                (effective_frame_period_us.saturating_mul(7) + measured_clamped) / 8;
        }

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
                effective_frame_period_us,
                "segment ready"
            );
        } else {
            debug!(
                seq,
                duration_ms = elapsed.as_millis() as u64,
                size_kb = size_bytes / 1024,
                effective_frame_period_us,
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
    let (frame, kind) = match timeout(frame_period, capture.next_frame()).await {
        Ok(res) => {
            let f = res?;
            *last_frame = Some(f.clone());
            (f, "real")
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
                    (stored.clone(), "dup")
                }
                None => {
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

fn bump_timestamp_us(frame: &mut CapturedFrame, delta_us: u64) {
    match frame {
        CapturedFrame::Cpu(r) => r.timestamp_us = r.timestamp_us.saturating_add(delta_us),
        CapturedFrame::Gpu(g) => g.timestamp_us = g.timestamp_us.saturating_add(delta_us),
    }
}

/// Records inter-frame wallclock spacing tagged with whether the
/// frame was a fresh capture (`"real"`) or a pace-pad copy of the
/// previous one (`"dup"`). Useful for spotting capture-side stalls
/// vs encoder-side throughput limits at a glance — e.g. a long run
/// of `kind="dup"` means the encoder/PW pipeline can't keep up with
/// `target_fps`, while a cluster of `kind="real"` with low `dt_ms`
/// means PW is delivering bursts.
fn record_frame_dt(last: &mut Option<Instant>, kind: &'static str) {
    let now = Instant::now();
    if let Some(prev) = *last {
        debug!(dt_ms = (now - prev).as_millis() as u64, kind, "frame");
    }
    *last = Some(now);
}

/// Like [`run`], but the source of encoded frames is an external
/// channel rather than an owned capture/encoder pair. Used by
/// receiver protocols (Chromecast HLS, …) whose capture+encode is
/// already driven by [`crate::HlsFrameSink`]'s caller (typically
/// `StreamManager`) and just need segments served over HTTP.
///
/// The receiver dropping (sender closed) cleanly drains the current
/// segment and returns, which is how the chromecast session signals
/// shutdown.
///
/// PTS accounting follows the same adaptive scheme as [`run`]:
/// uniform per-frame increment of `effective_frame_period_us` inside
/// each segment, refined after each segment using observed wall span
/// ÷ frames pushed. Without this the player drifts away from the
/// live edge and segments expire underneath it.
pub async fn run_from_frames(
    mut frames: mpsc::Receiver<EncodedFrame>,
    parameter_sets: Vec<u8>,
    ring: Arc<RwLock<SegmentRing>>,
    config: HlsConfig,
) -> Result<(), FerricastError> {
    if parameter_sets.is_empty() {
        return Err(FerricastError::Encoder(
            "empty H.264 parameter sets; refusing to start segmenter".into(),
        ));
    }
    let target = Duration::from_secs_f32(config.segment_target_secs);
    let frame_period_us =
        (1_000_000_f64 / (config.target_fps.max(1) as f64)) as u64;

    let mut pts_us: u64 = 0;
    let mut effective_frame_period_us: u64 = frame_period_us;
    let mut pending: Option<EncodedFrame> = None;

    let mut muxer = MpegTs::default().with_silent_audio(config.inject_silent_audio);
    muxer.config(parameter_sets.clone())?;

    loop {
        muxer.start_segment();

        // Wait for a keyframe to start the segment. Pre-IDR frames
        // are dropped by the upstream session, but if any sneak
        // through we drop them here for the same reason `run` does:
        // every segment must begin at a random-access point.
        let first = match pending.take() {
            Some(f) => f,
            None => loop {
                match frames.recv().await {
                    Some(f) if f.is_keyframe => break f,
                    Some(_) => continue,
                    None => return Ok(()),
                }
            },
        };

        let started = Instant::now();
        let mut frames_in_segment: u64 = 0;
        push_frame(&mut muxer, &first, pts_us)?;
        pts_us = pts_us.saturating_add(effective_frame_period_us);
        frames_in_segment += 1;

        loop {
            let encoded = match frames.recv().await {
                Some(f) => f,
                None => {
                    // Sender hung up — drain whatever we have so the
                    // player gets a final segment and can stop
                    // cleanly.
                    let elapsed = started.elapsed();
                    let bytes = Bytes::from(muxer.drain());
                    if !bytes.is_empty() {
                        ring.write()
                            .await
                            .push(elapsed.as_secs_f32(), false, bytes);
                    }
                    return Ok(());
                }
            };

            if encoded.is_keyframe && started.elapsed() >= target {
                pending = Some(encoded);
                break;
            }
            push_frame(&mut muxer, &encoded, pts_us)?;
            pts_us = pts_us.saturating_add(effective_frame_period_us);
            frames_in_segment += 1;
        }

        let elapsed = started.elapsed();

        // Same EMA refinement as `run`: skip the warmup segment, then
        // pull the per-frame PTS increment toward the actual wall
        // cost so PTS span ≈ wall span over time.
        if frames_in_segment >= 30 {
            let measured =
                (elapsed.as_micros() as u64).saturating_div(frames_in_segment);
            let lo = frame_period_us / 2;
            let hi = frame_period_us.saturating_mul(3) / 2;
            let measured_clamped = measured.clamp(lo, hi);
            effective_frame_period_us =
                (effective_frame_period_us.saturating_mul(7) + measured_clamped) / 8;
        }

        let bytes = Bytes::from(muxer.drain());
        let size_bytes = bytes.len();
        let seq = ring
            .write()
            .await
            .push(elapsed.as_secs_f32(), false, bytes);
        if seq < 3 {
            info!(
                seq,
                duration_ms = elapsed.as_millis() as u64,
                size_kb = size_bytes / 1024,
                effective_frame_period_us,
                "segment ready"
            );
        } else {
            debug!(
                seq,
                duration_ms = elapsed.as_millis() as u64,
                size_kb = size_bytes / 1024,
                effective_frame_period_us,
                "segment ready"
            );
        }
    }
}

fn push_frame(
    muxer: &mut MpegTs,
    encoded: &EncodedFrame,
    pts_us: u64,
) -> Result<(), FerricastError> {
    // Caller-provided monotonic PTS in microseconds. Uniform
    // increments inside any one segment keep the player's display
    // intervals smooth; the increment value itself is refined per
    // segment so PTS span tracks wall-clock production over time.
    let pts_90k = pts_us.saturating_mul(9) / 100;
    // Baseline H.264 has no B-frames → DTS == PTS.
    muxer.add_frame(encoded, pts_90k, pts_90k)
}
