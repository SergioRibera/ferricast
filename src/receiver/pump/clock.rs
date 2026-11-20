//! The pump's presentation clock: maps PTS to wall-clock instants and,
//! when the sink reports how much audio it holds, follows the sound
//! card instead of the system clock.
//!
//! Audio is pushed `lead` ahead of its presentation time into a sink
//! that plays samples back to back, so right before each push the sink
//! should hold exactly `lead` of audio. A sound card that runs slower
//! than the system clock makes that backlog grow (audio is heard later
//! and later), a faster one makes it shrink. The clock low-pass filters
//! the error and slews its anchor towards it: video is then presented
//! when its audio is actually heard, and audio pushes slow down or speed
//! up to keep the backlog at `lead`. Corrections are bounded to a small
//! fraction of real time, which no viewer notices, and the audio itself
//! is never cut or stretched, so nothing is audible.

use std::time::{Duration, Instant};

/// Output due further ahead than this is a timestamp jump.
const MAX_AHEAD: Duration = Duration::from_secs(5);
/// Time constant of the backlog error filter: long enough to average out
/// the sawtooth of chunked pushes and scheduling jitter.
const FILTER_TAU_SECS: f64 = 1.0;
/// Time constant of the correction once the filtered error is known.
const CORRECTION_TAU_SECS: f64 = 2.0;
/// Fastest slew, as a fraction of elapsed time (0.5%: 5 ms per second).
const MAX_SLEW: f64 = 0.005;
/// Errors below this are left alone.
const DEADBAND_US: f64 = 2_000.0;
/// Backlog errors beyond this are not drift (a gap in the audio, a stall,
/// a device that started late) and are not followed.
const MAX_FOLLOWED_ERROR_US: f64 = 100_000.0;

/// Maps PTS to wall-clock presentation instants.
pub(super) struct Clock {
    lead: Duration,
    anchor: Option<(Instant, u64)>,
    follow: Option<Follow>,
}

/// Audio-master state for the current timeline.
struct Follow {
    /// Filtered backlog error, µs: positive when audio is heard late.
    error_us: f64,
    last: Instant,
}

impl Clock {
    pub(super) fn new(lead: Duration) -> Self {
        Self {
            lead,
            anchor: None,
            follow: None,
        }
    }

    pub(super) fn is_anchored(&self) -> bool {
        self.anchor.is_some()
    }

    pub(super) fn reset(&mut self) {
        self.anchor = None;
        self.follow = None;
    }

    /// Starts a timeline: `pts_us` is presented `lead` from `now`.
    pub(super) fn anchor_at(&mut self, pts_us: u64, now: Instant) {
        self.anchor = Some((now + self.lead, pts_us));
        self.follow = None;
    }

    /// Presentation instant of `pts_us`, re-anchoring on jumps.
    pub(super) fn due(&mut self, pts_us: u64, now: Instant) -> Instant {
        let (t0, p0) = match self.anchor {
            Some(anchor) => anchor,
            None => {
                self.anchor_at(pts_us, now);
                (now + self.lead, pts_us)
            }
        };
        let due = if pts_us >= p0 {
            t0.checked_add(Duration::from_micros(pts_us - p0))
        } else {
            t0.checked_sub(Duration::from_micros(p0 - pts_us))
        };
        match due {
            Some(d) if d <= now + MAX_AHEAD => d,
            _ => {
                tracing::debug!(pts_us, "pump: timestamp jump — re-anchoring");
                self.anchor_at(pts_us, now);
                now + self.lead
            }
        }
    }

    /// Feeds the sink's audio backlog, measured right before an audio
    /// push, and slews the timeline towards the sound card's clock.
    pub(super) fn follow_audio(&mut self, backlog: Duration, now: Instant) {
        let Some((t0, p0)) = self.anchor else {
            return;
        };
        let error_us = backlog.as_secs_f64() * 1e6 - self.lead.as_secs_f64() * 1e6;
        if error_us.abs() > MAX_FOLLOWED_ERROR_US {
            return;
        }
        let Some(follow) = self.follow.as_mut() else {
            self.follow = Some(Follow {
                error_us,
                last: now,
            });
            return;
        };
        let dt = now.saturating_duration_since(follow.last).as_secs_f64();
        follow.last = now;
        follow.error_us += (error_us - follow.error_us) * (dt / FILTER_TAU_SECS).min(1.0);
        if follow.error_us.abs() < DEADBAND_US {
            return;
        }
        let max_step = MAX_SLEW * dt * 1e6;
        let step =
            (follow.error_us * (dt / CORRECTION_TAU_SECS).min(1.0)).clamp(-max_step, max_step);
        // Moving the anchor later delays every later push by `step`, which
        // shrinks the backlog by the same amount: account for it now
        // rather than waiting for the filter to see it.
        follow.error_us -= step;
        let shift = Duration::from_secs_f64(step.abs() / 1e6);
        let t0 = if step >= 0.0 {
            t0 + shift
        } else {
            t0.checked_sub(shift).unwrap_or(t0)
        };
        self.anchor = Some((t0, p0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEAD: Duration = Duration::from_millis(200);

    #[test]
    fn clock_maps_pts_offsets_to_wall_offsets() {
        let now = Instant::now();
        let mut clock = Clock::new(LEAD);
        let t0 = clock.due(10_000_000, now);
        assert_eq!(t0, now + LEAD);
        let t1 = clock.due(10_500_000, now);
        assert_eq!(t1 - t0, Duration::from_millis(500));
        // Slightly earlier PTS (audio before the first video) maps
        // before the anchor.
        let t_early = clock.due(9_900_000, now);
        assert_eq!(t0 - t_early, Duration::from_millis(100));
    }

    #[test]
    fn clock_reanchors_on_forward_jumps() {
        let now = Instant::now();
        let mut clock = Clock::new(LEAD);
        let t0 = clock.due(0, now);
        let jumped = clock.due(3_600_000_000, now);
        assert!(jumped < t0 + MAX_AHEAD);
        let next = clock.due(3_600_040_000, now);
        assert_eq!(next - jumped, Duration::from_millis(40));
    }

    /// Plays an hour of audio in 1024-sample chunks into a simulated sink
    /// whose sound card runs `ppm` fast (negative: slow) against the
    /// system clock. Returns the worst audio-vs-video error, the error at
    /// the end, and the largest single correction of the presentation
    /// time, all in milliseconds.
    fn simulate(ppm: f64, follow: bool) -> (f64, f64, f64) {
        let start = Instant::now();
        let at = |us: f64| start + Duration::from_secs_f64(us / 1e6);
        let since = |t: Instant| t.duration_since(start).as_secs_f64() * 1e6;
        let rate = 1.0 + ppm * 1e-6;
        let chunk_us = 1024.0 / 48_000.0 * 1e6;
        let lead_us = LEAD.as_secs_f64() * 1e6;
        let mut clock = Clock::new(LEAD);
        clock.anchor_at(0, at(0.0));
        // Priming silence of `lead`, pushed when the timeline starts.
        let mut pushed_us = lead_us;
        let (mut worst, mut last_error, mut largest_jump) = (0.0f64, 0.0, 0.0f64);
        let mut previous_due: Option<f64> = None;
        let mut sim_now = 0.0f64;
        let chunks = (3600.0 * 1e6 / chunk_us) as u64;
        for k in 0..chunks {
            let pts_us = k as f64 * chunk_us;
            let pts = pts_us as u64;
            let push_at = (since(clock.due(pts, at(sim_now))) - lead_us).max(sim_now);
            sim_now = push_at;
            let now = at(push_at);
            let played_us = push_at * rate;
            let backlog = (pushed_us - played_us).max(0.0);
            if follow {
                clock.follow_audio(Duration::from_secs_f64(backlog / 1e6), now);
            }
            pushed_us += chunk_us;
            // The device plays the priming silence, then every chunk
            // back to back, at its own rate.
            let heard = (lead_us + pts_us) / rate;
            let shown = since(clock.due(pts, now));
            let error_ms = (heard - shown) / 1000.0;
            worst = worst.max(error_ms.abs());
            last_error = error_ms;
            if let Some(previous) = previous_due {
                let expected = previous + chunk_us;
                largest_jump = largest_jump.max((shown - expected).abs() / 1000.0);
            }
            previous_due = Some(shown);
        }
        (worst, last_error, largest_jump)
    }

    #[test]
    fn without_following_100_ppm_drifts_hundreds_of_milliseconds_in_an_hour() {
        let (_, end, _) = simulate(-100.0, false);
        assert!(end > 300.0, "{end}");
    }

    #[test]
    fn audio_master_keeps_100_ppm_of_drift_within_40_ms_over_an_hour() {
        for ppm in [100.0, -100.0, 30.0, -250.0] {
            let (worst, end, jump) = simulate(ppm, true);
            eprintln!(
                "{ppm:+} ppm: worst A/V error {worst:.2} ms, at the end {end:+.2} ms, \
                 largest step {jump:.3} ms"
            );
            assert!(worst < 40.0, "{ppm} ppm: worst {worst} ms");
            assert!(end.abs() < 10.0, "{ppm} ppm: ends {end} ms off");
            // Corrections stay far below a frame per step.
            assert!(jump < 1.0, "{ppm} ppm: step {jump} ms");
        }
    }

    #[test]
    fn a_matching_sound_card_needs_no_correction() {
        let (worst, end, jump) = simulate(0.0, true);
        assert!(
            worst < 1.0 && end.abs() < 1.0 && jump < 0.01,
            "{worst} {end} {jump}"
        );
    }

    #[test]
    fn large_backlog_errors_are_not_followed() {
        let now = Instant::now();
        let mut clock = Clock::new(LEAD);
        clock.anchor_at(0, now);
        let before = clock.due(1_000_000, now);
        for i in 1..100u64 {
            clock.follow_audio(
                Duration::from_millis(600),
                now + Duration::from_millis(20 * i),
            );
        }
        assert_eq!(clock.due(1_000_000, now), before);
    }
}
