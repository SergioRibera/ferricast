//! Output timestamps for block-coded audio encoders (AAC, Opus).
//!
//! Encoders re-frame capture chunks into fixed-size codec frames, so
//! output timestamps are derived from a sample count rather than copied
//! from the input: `anchor + samples / sample_rate`. That keeps frame
//! deltas exact, but on its own it never notices a hole in the input —
//! after the capture source idles (PipeWire suspends a monitor source
//! whose sink went quiet) every later frame would carry a timestamp
//! from before the gap, and the HLS segmenter drops frames that are
//! more than 300 ms stale: permanent silence.
//!
//! [`AudioClock`] therefore compares every input chunk's timestamp
//! with where the sample count says it should start, and re-anchors on
//! a forward jump larger than [`RESYNC_THRESHOLD_US`].
//!
//! The opposite deviation comes from a sound card that runs faster
//! than the wall clock (~50 ppm ≈ 180 ms/h): the sample count drifts
//! ahead of the capture timestamps, and the capture side steps its
//! timestamps back once the drift passes its own threshold. Output PTS
//! must stay strictly monotonic (MPEG-TS/HLS), so instead of moving the
//! anchor back the clock has the encoder discard the excess input
//! samples once it passes [`TRIM_THRESHOLD_US`]: output time stands
//! still against the input until both agree again.

/// Forward discontinuity that re-anchors the clock.
///
/// Capture timestamps are themselves sample-count based, so in steady
/// state they agree with ours to the microsecond and only a real gap
/// moves them. 50 ms is above two 1024-sample PipeWire quanta at
/// 48 kHz (≈ 43 ms) — wall-clock jitter when the capture side
/// re-anchors never trips it — and far below the segmenter's 300 ms
/// staleness cut-off, so we resync long before frames would be dropped.
pub(crate) const RESYNC_THRESHOLD_US: u64 = 50_000;

/// How far the sample count may run ahead of the input before input
/// samples are discarded to pull it back.
///
/// Like [`RESYNC_THRESHOLD_US`] it sits above two 1024-sample quanta at
/// 48 kHz (≈ 43 ms), so the wall-clock jitter of a capture re-anchor
/// alone never costs audio, and it is the ITU-R BT.1359 detectability
/// limit for audio leading video (+45 ms), so lip sync is corrected
/// before the lead becomes noticeable.
pub(crate) const TRIM_THRESHOLD_US: u64 = 45_000;

/// Minimum output time between two reported corrections; corrections
/// in between still happen, they are just not reported. Card drift
/// needs ~15 min to cross [`TRIM_THRESHOLD_US`], so this only quiets an
/// input whose timestamps keep stepping back.
const REPORT_INTERVAL_US: u64 = 10_000_000;

/// What [`AudioClock::observe`] concluded about an input chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClockEvent {
    /// First chunk: the clock now starts at its timestamp.
    Anchored,
    /// The chunk continues the running sample count.
    Continuous,
    /// The chunk starts `gap_us` later than the sample count predicts;
    /// the clock jumped forward to it.
    Resynced { gap_us: u64 },
    /// The sample count runs `ahead_us` ahead of the chunk: discard its
    /// first `skip` samples (per channel) before queueing the rest.
    /// `report` is set on the chunk that starts a correction, at most
    /// once per [`REPORT_INTERVAL_US`] of output time.
    Trimmed {
        ahead_us: u64,
        skip: u64,
        report: bool,
    },
}

impl ClockEvent {
    /// Leading samples (per channel) of the chunk to discard.
    pub(crate) fn skip(self) -> u64 {
        match self {
            Self::Trimmed { skip, .. } => skip,
            Self::Anchored | Self::Continuous | Self::Resynced { .. } => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AudioClock {
    sample_rate: u64,
    anchor_us: Option<u64>,
    /// Samples (per channel) emitted since `anchor_us`.
    emitted: u64,
    /// A correction started but its excess did not fit in one chunk;
    /// keep trimming, below [`TRIM_THRESHOLD_US`] too, until it is gone.
    trimming: bool,
    /// Output time of the last reported correction.
    reported_at_us: Option<u64>,
}

impl AudioClock {
    pub(crate) fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate: u64::from(sample_rate.max(1)),
            anchor_us: None,
            emitted: 0,
            trimming: false,
            reported_at_us: None,
        }
    }

    /// Account for an input chunk of `chunk` samples (per channel)
    /// captured at `timestamp_us`, arriving while `buffered` samples
    /// are queued but not yet encoded. Call before queueing the
    /// chunk's own samples, and drop the first [`ClockEvent::skip`]
    /// of them.
    ///
    /// Output timestamps never move backwards: when the chunk starts
    /// more than [`TRIM_THRESHOLD_US`] before the sample count, its
    /// samples are discarded instead (never more than the chunk, the
    /// rest from the following chunks) until the count is back on the
    /// input's timeline.
    pub(crate) fn observe(&mut self, timestamp_us: u64, buffered: u64, chunk: u64) -> ClockEvent {
        let buffered_us = self.duration_us(buffered);
        let Some(anchor) = self.anchor_us else {
            self.anchor_us = Some(timestamp_us.saturating_sub(buffered_us));
            self.emitted = 0;
            return ClockEvent::Anchored;
        };
        let expected = anchor
            .saturating_add(self.duration_us(self.emitted))
            .saturating_add(buffered_us);
        let gap_us = timestamp_us.saturating_sub(expected);
        if gap_us > RESYNC_THRESHOLD_US {
            // Keep the queued samples: they now end exactly where the new
            // chunk begins, so nothing is dropped and PTS stays monotonic
            // (`expected - buffered_us` < `timestamp_us - buffered_us`).
            self.anchor_us = Some(timestamp_us.saturating_sub(buffered_us));
            self.emitted = 0;
            self.trimming = false;
            return ClockEvent::Resynced { gap_us };
        }
        let ahead_us = expected.saturating_sub(timestamp_us);
        let starts = !self.trimming && ahead_us > TRIM_THRESHOLD_US;
        if !starts && !self.trimming {
            return ClockEvent::Continuous;
        }
        // Discarded samples never reach the count, so the next chunk's
        // `expected` already reflects them: the remainder of the excess
        // carries over by being measured again, not by being remembered.
        let excess = self.samples(ahead_us);
        let skip = excess.min(chunk);
        self.trimming = skip < excess;
        let now_us = self.next_timestamp_us();
        let report = starts
            && self
                .reported_at_us
                .is_none_or(|at| now_us.saturating_sub(at) >= REPORT_INTERVAL_US);
        if report {
            self.reported_at_us = Some(now_us);
        }
        ClockEvent::Trimmed {
            ahead_us,
            skip,
            report,
        }
    }

    /// Timestamp of the next sample to be emitted.
    pub(crate) fn next_timestamp_us(&self) -> u64 {
        self.anchor_us
            .unwrap_or(0)
            .saturating_add(self.duration_us(self.emitted))
    }

    /// Record that `samples` (per channel) were emitted.
    pub(crate) fn advance(&mut self, samples: u64) {
        self.emitted = self.emitted.saturating_add(samples);
    }

    fn duration_us(&self, samples: u64) -> u64 {
        samples.saturating_mul(1_000_000) / self.sample_rate
    }

    /// Whole samples in `duration_us`, rounded down so a trim never
    /// overshoots the input timeline.
    fn samples(&self, duration_us: u64) -> u64 {
        duration_us.saturating_mul(self.sample_rate) / 1_000_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;
    const FRAME: u64 = 1024;
    /// 1024 samples at 48 kHz, truncated like the encoder computes it.
    const FRAME_US: u64 = 21_333;
    /// Opus-sized codec frame, so chunks and frames do not line up.
    const CODEC_FRAME: u64 = 960;

    /// Drives the clock the way the encoders do: observe, drop the
    /// skipped samples, queue the rest, emit whole codec frames.
    struct Encoder {
        clock: AudioClock,
        buffered: u64,
        pts: Vec<u64>,
        trimmed: u64,
        reports: usize,
    }

    impl Encoder {
        fn new() -> Self {
            Self {
                clock: AudioClock::new(SR),
                buffered: 0,
                pts: Vec::new(),
                trimmed: 0,
                reports: 0,
            }
        }

        /// Feeds a chunk and returns how far (µs) the end of the queued
        /// audio is stamped ahead of where it ends on the input timeline.
        fn push(&mut self, timestamp_us: u64, chunk: u64) -> i64 {
            let event = self.clock.observe(timestamp_us, self.buffered, chunk);
            let skip = event.skip();
            assert!(skip <= chunk, "{event:?} exceeds a {chunk}-sample chunk");
            if let ClockEvent::Trimmed { report: true, .. } = event {
                self.reports += 1;
            }
            self.trimmed += skip;
            self.buffered += chunk - skip;
            while self.buffered >= CODEC_FRAME {
                self.pts.push(self.clock.next_timestamp_us());
                self.clock.advance(CODEC_FRAME);
                self.buffered -= CODEC_FRAME;
            }
            let stamped_end =
                self.clock.next_timestamp_us() + self.clock.duration_us(self.buffered);
            let input_end = timestamp_us + self.clock.duration_us(chunk);
            i64::try_from(stamped_end).unwrap() - i64::try_from(input_end).unwrap()
        }

        fn assert_strictly_monotonic(&self) {
            assert!(self.pts.windows(2).all(|w| w[0] < w[1]));
        }
    }

    /// Microseconds at which sample `n` is captured when the card runs
    /// `ppm` parts per million faster than the wall clock.
    fn card_time_us(start_us: u64, n: u64, ppm: u64) -> u64 {
        let ticks = u128::from(n) * 1_000_000 * 1_000_000;
        let rate = u128::from(SR) * u128::from(1_000_000 + ppm);
        start_us + u64::try_from(ticks / rate).unwrap()
    }

    #[test]
    fn continuous_input_matches_sample_count() {
        let mut clock = AudioClock::new(SR);
        assert_eq!(clock.observe(1_000_000, 0, FRAME), ClockEvent::Anchored);
        assert_eq!(clock.next_timestamp_us(), 1_000_000);
        clock.advance(FRAME);
        assert_eq!(clock.next_timestamp_us(), 1_000_000 + FRAME_US);
        // Next chunk arrives where the count says (+ sub-ms jitter).
        assert_eq!(
            clock.observe(1_000_000 + FRAME_US + 400, 0, FRAME),
            ClockEvent::Continuous
        );
        clock.advance(FRAME);
        assert_eq!(clock.next_timestamp_us(), 1_000_000 + 42_666);
    }

    /// Sample-counted input (what capture delivers between re-anchors)
    /// is stamped exactly as before trimming existed: nothing dropped,
    /// every frame at `anchor + n * frame`.
    #[test]
    fn steady_input_is_stamped_from_the_sample_count() {
        let start = 1_700_000_000_000_000;
        let mut encoder = Encoder::new();
        for n in 0..10_000 {
            encoder.push(start + encoder.clock.duration_us(n * FRAME), FRAME);
        }
        assert_eq!(encoder.trimmed, 0);
        for (i, &pts) in encoder.pts.iter().enumerate() {
            let expected = start + encoder.clock.duration_us(i as u64 * CODEC_FRAME);
            assert_eq!(pts, expected, "frame {i}");
        }
    }

    /// Jitter up to the threshold never moves the clock.
    #[test]
    fn small_deviation_is_tolerated() {
        let mut clock = AudioClock::new(SR);
        clock.observe(0, 0, FRAME);
        clock.advance(FRAME);
        let event = clock.observe(FRAME_US + RESYNC_THRESHOLD_US, 0, FRAME);
        assert_eq!(event, ClockEvent::Continuous);
        assert_eq!(clock.next_timestamp_us(), FRAME_US);
    }

    /// After a 2 s capture gap the next frame is stamped at the new
    /// capture time instead of 2 s in the past.
    #[test]
    fn gap_reanchors_forward() {
        let mut clock = AudioClock::new(SR);
        clock.observe(10_000_000, 0, FRAME);
        clock.advance(10 * FRAME);
        let resumed = 12_500_000;
        let event = clock.observe(resumed, 0, FRAME);
        assert_eq!(
            event,
            ClockEvent::Resynced {
                gap_us: resumed - (10_000_000 + 213_333)
            }
        );
        assert_eq!(clock.next_timestamp_us(), resumed);
        clock.advance(FRAME);
        assert_eq!(clock.next_timestamp_us(), resumed + FRAME_US);
    }

    /// Samples still queued from before the gap end exactly where the
    /// post-gap chunk starts, and PTS stays monotonic.
    #[test]
    fn buffered_samples_are_placed_before_the_new_chunk() {
        let mut clock = AudioClock::new(SR);
        clock.observe(0, 0, FRAME);
        clock.advance(FRAME);
        let before = clock.next_timestamp_us();
        let buffered = 480; // 10 ms
        clock.observe(1_000_000, buffered, FRAME);
        let after = clock.next_timestamp_us();
        assert_eq!(after, 1_000_000 - 10_000);
        assert!(after > before);
    }

    /// A lead within the threshold is left alone: no samples dropped.
    #[test]
    fn small_backward_deviation_is_tolerated() {
        let mut clock = AudioClock::new(SR);
        clock.observe(5_000_000, 0, FRAME);
        clock.advance(FRAME);
        let late = 5_000_000 + FRAME_US - TRIM_THRESHOLD_US;
        assert_eq!(clock.observe(late, 0, FRAME), ClockEvent::Continuous);
        assert_eq!(clock.next_timestamp_us(), 5_000_000 + FRAME_US);
    }

    /// The capture side steps its timestamps back by 150 ms when the
    /// card ran ahead of the wall clock: 150 ms of input is discarded
    /// over several chunks, reported once, and output PTS never steps
    /// back.
    #[test]
    fn backward_step_is_trimmed_away() {
        let start = 1_700_000_000_000_000;
        let step = 150_000;
        let mut encoder = Encoder::new();
        let mut deviation = 0;
        for n in 0..2_000 {
            let mut timestamp = start + encoder.clock.duration_us(n * FRAME);
            if n >= 1_000 {
                timestamp -= step;
            }
            deviation = encoder.push(timestamp, FRAME);
        }
        assert_eq!(encoder.trimmed, encoder.clock.samples(step));
        assert_eq!(encoder.reports, 1);
        assert!(deviation.abs() <= 1, "converged to {deviation} µs");
        encoder.assert_strictly_monotonic();
        let deltas: Vec<u64> = encoder.pts.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(
            deltas.iter().all(|&d| d == 20_000),
            "frame spacing is exact"
        );
    }

    /// Stamped against the wall clock, a card 50 ppm fast drifts
    /// ~180 ms ahead per hour; trimming keeps the lead under the
    /// threshold the whole hour while PTS stays strictly monotonic.
    #[test]
    fn fast_card_lead_stays_bounded_over_an_hour() {
        let start = 1_700_000_000_000_000;
        let chunks = u64::from(SR) * 3_600 / FRAME;
        let mut encoder = Encoder::new();
        let mut worst = 0;
        for n in 0..chunks {
            let deviation = encoder.push(card_time_us(start, n * FRAME, 50), FRAME);
            worst = worst.max(deviation.abs());
        }
        let max_lead = i64::try_from(TRIM_THRESHOLD_US + FRAME_US).unwrap();
        assert!(worst <= max_lead, "lead reached {worst} µs");
        encoder.assert_strictly_monotonic();
        // 50 ppm of an hour (180 ms), minus the lead still allowed at the end.
        let trimmed_us = encoder.clock.duration_us(encoder.trimmed);
        assert!(
            (180_000 - max_lead.unsigned_abs()..=180_100).contains(&trimmed_us),
            "trimmed {trimmed_us} µs"
        );
        assert!(encoder.reports >= 3, "{} reports", encoder.reports);
    }

    /// A source whose timestamps keep stepping back is corrected every
    /// time but reported at most once per interval.
    #[test]
    fn reports_are_rate_limited() {
        let mut encoder = Encoder::new();
        for n in 0..10_000 {
            // Timestamps advance at half the rate the audio plays.
            encoder.push(n * FRAME_US / 2, FRAME);
        }
        let output_us = encoder.clock.next_timestamp_us();
        let max_reports = usize::try_from(output_us / REPORT_INTERVAL_US).unwrap() + 1;
        assert!(encoder.trimmed > 0);
        assert!(
            encoder.reports <= max_reports,
            "{} reports",
            encoder.reports
        );
        encoder.assert_strictly_monotonic();
    }

    /// Hostile timestamps never ask for more than the chunk holds.
    #[test]
    fn skip_never_exceeds_the_chunk() {
        let mut clock = AudioClock::new(SR);
        clock.observe(u64::MAX / 2, 0, FRAME);
        clock.advance(FRAME);
        for chunk in [0, 1, FRAME] {
            let event = clock.observe(0, u64::MAX, chunk);
            assert!(event.skip() <= chunk, "{event:?}");
        }
    }
}
