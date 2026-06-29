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
use tokio::sync::{RwLock, mpsc};
use tokio::time::timeout;
use tracing::{debug, info, trace, warn};

use ferricast_core::{
    AudioFrame, CapturedFrame, EncodedFrame, FerricastError, ScreenCapture, VideoEncoder,
};
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
    let frame_period = Duration::from_secs_f64(1.0 / (config.target_fps.max(1) as f64));
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
        //
        // The clamp is asymmetric: 1/2 the configured period at the
        // bottom (a hard "never claim more than 2× target_fps" guard
        // that catches absurd `elapsed` measurements), 3× at the
        // top so we *can* track production well below target without
        // pinning the rate. The previous symmetric ±50 % cap pinned
        // PTS at ~40 fps when target was 60 even when real
        // production was ~30 fps; that made each segment's PTS span
        // shorter than its wall span, the player consumed faster
        // than we produced, and the receiver hit BUFFERING every
        // segment.
        if frames_in_segment >= 30 {
            let measured = (elapsed.as_micros() as u64).saturating_div(frames_in_segment);
            let lo = frame_period_us / 2;
            let hi = frame_period_us.saturating_mul(3);
            let measured_clamped = measured.clamp(lo, hi);
            effective_frame_period_us =
                (effective_frame_period_us.saturating_mul(7) + measured_clamped) / 8;
            if measured > hi {
                warn!(
                    measured_us = measured,
                    cap_us = hi,
                    effective_frame_period_us,
                    "segment production below 1/3 of target_fps; PTS pacing capped (player will drift)"
                );
            }
        }

        let bytes = Bytes::from(muxer.drain());
        let size_bytes = bytes.len();
        let seq = ring.write().await.push(elapsed.as_secs_f32(), false, bytes);
        // First few segments at INFO so the user can sanity-check
        // cadence (segments far longer than target = encoder keyint
        // too high). Steady-state goes to DEBUG to keep logs quiet.
        // Encoded bitrate per segment. Pairs naturally with the
        // HLS handler's per-segment "encoded_kbps" log so the user
        // can see "segmenter produced 1800 kbps, receiver pulled it
        // at X mbps over Y seconds" all in one timeline.
        let encoded_kbps =
            ((size_bytes as f64 * 8.0) / 1000.0 / elapsed.as_secs_f64().max(0.001)) as u32;
        if seq < 3 {
            info!(
                seq,
                duration_ms = elapsed.as_millis() as u64,
                size_kb = size_bytes / 1024,
                encoded_kbps,
                effective_frame_period_us,
                "segment ready"
            );
        } else {
            debug!(
                seq,
                duration_ms = elapsed.as_millis() as u64,
                size_kb = size_bytes / 1024,
                encoded_kbps,
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
    mut audio_frames: Option<mpsc::Receiver<AudioFrame>>,
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
    let frame_period_us = (1_000_000_f64 / (config.target_fps.max(1) as f64)) as u64;
    // LL-HLS: when `part_target_secs` is set, we publish partial
    // segments ("parts") into the ring as they're ready, instead of
    // waiting for the segment to close. This keeps clients fed at
    // sub-second granularity, which on flaky-firmware receivers
    // (1st/2nd-gen Chromecast) reduces the chance the receiver's
    // own prefetch state machine wedges on stale TCP connections.
    let part_target: Option<Duration> = config.part_target_secs.map(Duration::from_secs_f32);
    if let Some(pt) = config.part_target_secs {
        ring.write().await.enable_low_latency(pt);
    }

    // Video PTS uses the encoded frame's *capture-time* wall
    // timestamp, offset from the first video frame's wall ts. Both
    // audio and video PTS now share a single wall-time origin, so
    // A/V sync is mechanical: no synthetic ramp can drift away from
    // wall, no EMA needs to converge.
    //
    // The previous synthetic ramp (`pts_us += effective_frame_
    // period_us` per frame) was introduced to absorb encoder
    // jitter — but the receiver's playback buffer (≥3 s on
    // chromecast) already handles jitter at presentation, and the
    // ramp's mismatch with the wall-anchored audio side caused a
    // permanently-growing audio delay (audio.wall accumulated
    // faster than synth_pts on any host where real fps fell short
    // of `target_fps`, which is common when the encoder's per-frame
    // latency adds onto the capture cadence).
    let mut video_wall_anchor: Option<u64> = None;
    let mut pending: Option<EncodedFrame> = None;

    // When real audio is wired in, take over the audio PID from the
    // silent-injection path. Silent injection used to be necessary
    // to placate the old chromecast firmware; now that we have real
    // PCM coming through, that placeholder traffic would collide on
    // PID_AUDIO with the real PES packets we mux below.
    let external_audio_enabled = audio_frames.is_some();
    let mut muxer = MpegTs::default()
        .with_silent_audio(config.inject_silent_audio && !external_audio_enabled)
        .with_external_audio(external_audio_enabled);
    muxer.config(parameter_sets.clone())?;

    // Both audio and video timestamps come from `SystemTime::UNIX_EPOCH`
    // (see `pipewire_audio::now_us` and `wayland_direct::now_us`), so
    // they share a clock origin. Audio PTS is therefore
    // `audio.ts.saturating_sub(video_wall_anchor)`, identical to the
    // video formula. Any audio captured BEFORE the first video frame
    // saturates to 0 and gets dropped downstream by the muxer's
    // silent-AAC watermark — exactly what we want, since the receiver
    // timeline starts at the first video keyframe.
    //
    // Previous attempts to introduce a second `audio_origin_us` anchor
    // injected a permanent A/V lag equal to the time between first
    // video keyframe and first audio frame the segmenter processed
    // (typically ~1.4 s on Wayland because xdg-screencopy takes longer
    // to negotiate than the PipeWire audio graph). The shared
    // `video_wall_anchor` avoids that class of bug entirely.
    let mut audio_frames_pushed: u64 = 0;
    // Throughput watchdog — operator-visible audio/video consume rate.
    let mut last_throughput_log = Instant::now();
    let mut audio_pushed_window: u64 = 0;
    let mut video_pushed_window: u64 = 0;

    loop {
        muxer.start_segment();

        let first = match pending.take() {
            Some(f) => f,
            None => loop {
                // Drain audio while we're waiting for the first
                // keyframe so its channel doesn't backlog. Pre-IDR
                // audio is dropped because the segmenter hasn't
                // anchored its timeline yet — the first audio frame
                // that survives is the one arriving after the first
                // video frame, which is when we latch
                // `video_wall_anchor`.
                tokio::select! {
                    biased;
                    f = frames.recv() => match f {
                        Some(f) if f.is_keyframe => break f,
                        Some(_) => continue,
                        None => return Ok(()),
                    },
                    a = recv_audio(&mut audio_frames), if audio_frames.is_some() => {
                        // Audio that arrived before the first video
                        // frame: drop it. The mute / capture path
                        // keeps pushing frames so the channel never
                        // closes, and warm-up audio is < 100 ms of
                        // content.
                        let _ = a;
                    }
                }
            },
        };


        let started = Instant::now();
        let mut frames_in_segment: u64 = 0;
        let first_wall_ts = first.timestamp_us;
        // Latch the global wall-time origin on the very first video
        // frame the segmenter ever sees. All subsequent video and
        // audio PTS values are computed as `wall - video_wall_anchor`,
        // so the receiver's reference clock starts at 0 and advances
        // exactly with capture wall time.
        if video_wall_anchor.is_none() {
            video_wall_anchor = Some(first_wall_ts);
            debug!(
                wall_ts_us = first_wall_ts,
                "A/V timeline wall anchor latched"
            );
        }
        let origin = video_wall_anchor.unwrap_or(first_wall_ts);
        let video_pts_us = first_wall_ts.saturating_sub(origin);
        push_frame(&mut muxer, &first, video_pts_us)?;
        frames_in_segment += 1;

        // LL-HLS state for this segment.
        // `local_parts` collects each part's bytes locally so we can
        // splice them into the full segment body at close time —
        // because clients that didn't fetch parts (or arrived after
        // the segment closed) still need the regular `/segment-N.ts`
        // to serve a complete contiguous TS body. The ring also
        // already has these parts published via `push_pending_part`,
        // but those are slices the muxer drained piecemeal; we keep
        // a parallel copy here so reconstructing the segment doesn't
        // require holding the ring write lock for a long time.
        let mut local_parts: Vec<Bytes> = Vec::new();
        let mut part_start = Instant::now();
        let mut first_part_of_segment = true;

        let mut sender_hung_up = false;
        loop {
            // Interleave audio and video on the way to the muxer.
            // `biased`-free select! lets either side win the next
            // iteration based on arrival order, which keeps the
            // 21 ms-cadence audio frames closely interleaved with
            // the ~16 ms-cadence video frames inside the same TS
            // segment — the receiver's audio decoder fills its
            // jitter buffer from PID 0x101 without us having to
            // emit a burst of audio at segment close.
            let event = tokio::select! {
                v = frames.recv() => SegEvent::Video(v),
                a = recv_audio(&mut audio_frames), if audio_frames.is_some() => {
                    SegEvent::Audio(a)
                }
            };

            match event {
                SegEvent::Video(None) => {
                    sender_hung_up = true;
                    break;
                }
                SegEvent::Video(Some(encoded)) => {
                    if encoded.is_keyframe && started.elapsed() >= target {
                        pending = Some(encoded);
                        break;
                    }
                    let wall_ts = encoded.timestamp_us;
                    let video_pts_us = wall_ts.saturating_sub(origin);
                    push_frame(&mut muxer, &encoded, video_pts_us)?;
                    frames_in_segment += 1;
                    video_pushed_window = video_pushed_window.saturating_add(1);
                }
                SegEvent::Audio(None) => {
                    // Audio side closed mid-stream — likely the
                    // user disabled the audio capture. Stop trying
                    // to read it; the rest of the stream is video-
                    // only. We do *not* clear `has_external_audio`
                    // on the muxer mid-segment because the PMT for
                    // this segment was already emitted with the
                    // audio PID; clearing it would create a PMT
                    // discontinuity the receiver doesn't expect.
                    audio_frames = None;
                    continue;
                }
                SegEvent::Audio(Some(audio)) => {
                    if let Err(e) =
                        push_audio_frame(&mut muxer, &audio, video_wall_anchor)
                    {
                        warn!(error = %e, "audio PES push failed");
                    } else {
                        audio_frames_pushed =
                            audio_frames_pushed.saturating_add(1);
                        audio_pushed_window = audio_pushed_window.saturating_add(1);
                    }
                    // Burst-drain to keep the upstream queue empty
                    // regardless of video cadence.
                    if let Some(rx) = audio_frames.as_mut() {
                        loop {
                            match rx.try_recv() {
                                Ok(extra) => {
                                    if let Err(e) = push_audio_frame(
                                        &mut muxer,
                                        &extra,
                                        video_wall_anchor,
                                    ) {
                                        warn!(error = %e, "audio PES push failed (drain)");
                                    } else {
                                        audio_frames_pushed =
                                            audio_frames_pushed.saturating_add(1);
                                        audio_pushed_window =
                                            audio_pushed_window.saturating_add(1);
                                    }
                                }
                                Err(mpsc::error::TryRecvError::Empty) => break,
                                Err(mpsc::error::TryRecvError::Disconnected) => {
                                    audio_frames = None;
                                    break;
                                }
                            }
                        }
                    }
                    // Audio frames must not trigger LL-HLS part
                    // flushes on their own — the part target is
                    // a *time* budget; let video close them.
                    continue;
                }
            }

            if last_throughput_log.elapsed() >= Duration::from_secs(5) {
                let secs = last_throughput_log.elapsed().as_secs_f64();
                tracing::info!(
                    audio_per_s = (audio_pushed_window as f64 / secs) as u32,
                    video_per_s = (video_pushed_window as f64 / secs) as u32,
                    audio_total = audio_frames_pushed,
                    "segmenter throughput"
                );
                last_throughput_log = Instant::now();
                audio_pushed_window = 0;
                video_pushed_window = 0;
            }

            // LL-HLS: if we've held bytes long enough, flush them as
            // a part. The muxer's continuity counters carry across
            // the drain (they live in `self`, not in `out`), so the
            // concatenation of all parts within this segment is a
            // valid TS stream identical to a single drain at close.
            if let Some(pt) = part_target {
                if part_start.elapsed() >= pt {
                    let part_bytes = Bytes::from(muxer.drain());
                    if !part_bytes.is_empty() {
                        let part_dur = part_start.elapsed().as_secs_f32();
                        local_parts.push(part_bytes.clone());
                        ring.write().await.push_pending_part(
                            part_dur,
                            first_part_of_segment,
                            part_bytes,
                        );
                        first_part_of_segment = false;
                        part_start = Instant::now();
                    }
                }
            }
        }

        let elapsed = started.elapsed();

        // Diagnostic: report the per-segment measured wall cadence
        // against the configured frame period. Big delta → encoder
        // / capture is slower than `target_fps`. Doesn't drive PTS
        // anymore (video PTS is wall-anchored), so the warn is
        // informational only — the receiver stays in A/V sync
        // regardless.
        let measured_period_us = if frames_in_segment > 0 {
            (elapsed.as_micros() as u64).saturating_div(frames_in_segment)
        } else {
            0
        };
        if frames_in_segment >= 30 && measured_period_us > frame_period_us.saturating_mul(3) {
            warn!(
                measured_us = measured_period_us,
                cap_us = frame_period_us.saturating_mul(3),
                "segment production below 1/3 of target_fps (capture / encoder bottleneck)"
            );
        }

        // Drain whatever's left in the muxer at segment close. In
        // LL-HLS mode this becomes the final part of the segment;
        // in classic mode it's the entire segment body.
        let final_drain = Bytes::from(muxer.drain());

        let (full_segment_bytes, seq) = if part_target.is_some() {
            // Push the trailing part if we have leftover bytes.
            if !final_drain.is_empty() {
                let part_dur = part_start.elapsed().as_secs_f32();
                local_parts.push(final_drain.clone());
                ring.write()
                    .await
                    .push_pending_part(part_dur, first_part_of_segment, final_drain);
            }
            // Concatenate all parts to form the segment body the
            // `/segment-N.ts` endpoint will serve.
            let total: usize = local_parts.iter().map(|b| b.len()).sum();
            let mut full = bytes::BytesMut::with_capacity(total);
            for p in &local_parts {
                full.extend_from_slice(p);
            }
            let body = Bytes::from(full);
            let size = body.len();
            let seq =
                ring.write()
                    .await
                    .complete_pending_segment(elapsed.as_secs_f32(), false, body);
            (size, seq)
        } else {
            // Classic HLS — single drain → one push.
            let size = final_drain.len();
            let seq = ring
                .write()
                .await
                .push(elapsed.as_secs_f32(), false, final_drain);
            (size, seq)
        };

        let size_bytes = full_segment_bytes;
        let encoded_kbps =
            ((size_bytes as f64 * 8.0) / 1000.0 / elapsed.as_secs_f64().max(0.001)) as u32;
        let parts_count = local_parts.len();
        if seq < 3 {
            info!(
                seq,
                duration_ms = elapsed.as_millis() as u64,
                size_kb = size_bytes / 1024,
                encoded_kbps,
                measured_period_us,
                parts = parts_count,
                "segment ready"
            );
        } else {
            debug!(
                seq,
                duration_ms = elapsed.as_millis() as u64,
                size_kb = size_bytes / 1024,
                encoded_kbps,
                measured_period_us,
                parts = parts_count,
                "segment ready"
            );
        }

        if sender_hung_up {
            return Ok(());
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

/// One event the segment inner loop dispatches on.
enum SegEvent {
    Video(Option<EncodedFrame>),
    Audio(Option<AudioFrame>),
}

/// Helper around `Option<Receiver<AudioFrame>>::recv()` that's
/// `await`-safe inside a `tokio::select!` arm — the inner `Option`
/// can't appear inside `select!`'s pattern matcher directly, so we
/// resolve it here and return `None` when no receiver is wired up.
async fn recv_audio(
    chan: &mut Option<mpsc::Receiver<AudioFrame>>,
) -> Option<AudioFrame> {
    match chan {
        Some(rx) => rx.recv().await,
        // Never resolves — but `select!`'s `, if cond` guard above
        // makes sure this arm is disabled when the option is `None`,
        // so the future never actually polls.
        None => std::future::pending().await,
    }
}

/// Translate an audio frame's capture-clock `timestamp_us` into the
/// receiver-side PTS timeline and forward to the MPEG-TS muxer.
///
/// Both audio and video timestamps come from `SystemTime::UNIX_EPOCH`
/// (see `now_us()` in `pipewire_audio.rs` and `wayland_direct.rs`),
/// so audio PTS uses the same formula as video:
///   `pts_us = wall_ts.saturating_sub(video_wall_anchor)`
///
/// Audio whose wall ts is older than the video anchor (PipeWire audio
/// started capturing before the first video keyframe arrived — common
/// on Wayland where xdg-screencopy negotiation takes ~1 s longer than
/// PipeWire audio graph setup) saturates to PTS=0 and gets dropped by
/// the muxer's silent-AAC watermark, which is exactly what we want:
/// the receiver timeline starts at the first video frame, anything
/// captured before that is uninteresting.
///
/// Frames whose `timestamp_us` is more than `MAX_AUDIO_STALENESS_US`
/// behind current wall time are dropped before they hit the muxer.
/// Without this, when any link in the
/// `PipeWire → AAC encoder → manager tx → session audio_tx`
/// chain (each 64-deep) gets backlogged by a slow consumer, the
/// segmenter eventually drains a wad of frames whose PTS lands behind
/// the muxer's silent-fallback watermark — silent injection has
/// already advanced past those slots, so they'd be rejected one by
/// one and clutter the log with `dropping real AAC frame` lines
/// for the rest of the session. Better to catch them here, log the
/// staleness once per gap, and keep the trace useful.
fn push_audio_frame(
    muxer: &mut MpegTs,
    audio: &AudioFrame,
    video_wall_anchor: Option<u64>,
) -> Result<(), FerricastError> {
    let Some(anchor) = video_wall_anchor else {
        // First video frame hasn't been seen yet — drop quietly.
        return Ok(());
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0);
    let staleness_us = now.saturating_sub(audio.timestamp_us);
    if staleness_us > MAX_AUDIO_STALENESS_US {
        debug!(
            staleness_us,
            audio_wall = audio.timestamp_us,
            now_us = now,
            "segmenter: dropping stale AAC frame (upstream backlog)"
        );
        return Ok(());
    }
    let pts_us = audio.timestamp_us.saturating_sub(anchor);
    let pts_90k = pts_us.saturating_mul(9) / 100;
    trace!(
        audio_wall = audio.timestamp_us,
        video_wall_anchor = anchor,
        audio_pts_us = pts_us,
        pts_90k,
        bytes = audio.data.len(),
        "audio PES push"
    );
    muxer.add_audio_frame(audio.data.as_ref(), pts_90k)
}

/// Maximum age (capture-wall to processing-wall) a real audio frame
/// is allowed to have. Past this we treat it as backlogged garbage
/// and drop. 300 ms is comfortable headroom for normal pipeline
/// jitter (tens of ms) while being well under the muxer's 500 ms
/// silent-injection lag threshold, so a real frame that survives
/// this filter is still in the window the muxer will accept.
const MAX_AUDIO_STALENESS_US: u64 = 300_000;
