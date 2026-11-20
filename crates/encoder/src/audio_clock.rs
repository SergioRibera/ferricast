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

/// Forward discontinuity that re-anchors the clock.
///
/// Capture timestamps are themselves sample-count based, so in steady
/// state they agree with ours to the microsecond and only a real gap
/// moves them. 50 ms is above two 1024-sample PipeWire quanta at
/// 48 kHz (≈ 43 ms) — wall-clock jitter when the capture side
/// re-anchors never trips it — and far below the segmenter's 300 ms
/// staleness cut-off, so we resync long before frames would be dropped.
pub(crate) const RESYNC_THRESHOLD_US: u64 = 50_000;

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
}

#[derive(Debug, Clone)]
pub(crate) struct AudioClock {
    sample_rate: u64,
    anchor_us: Option<u64>,
    /// Samples (per channel) emitted since `anchor_us`.
    emitted: u64,
}

impl AudioClock {
    pub(crate) fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate: u64::from(sample_rate.max(1)),
            anchor_us: None,
            emitted: 0,
        }
    }

    /// Account for an input chunk captured at `timestamp_us`, arriving
    /// while `buffered` samples (per channel) are queued but not yet
    /// encoded. Call before queueing the chunk's own samples.
    ///
    /// Backward deviations are ignored: output timestamps must stay
    /// monotonic, and frames stamped ahead of their capture time are
    /// never dropped as stale.
    pub(crate) fn observe(&mut self, timestamp_us: u64, buffered: u64) -> ClockEvent {
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
        if gap_us <= RESYNC_THRESHOLD_US {
            return ClockEvent::Continuous;
        }
        // Keep the queued samples: they now end exactly where the new
        // chunk begins, so nothing is dropped and PTS stays monotonic
        // (`expected - buffered_us` < `timestamp_us - buffered_us`).
        self.anchor_us = Some(timestamp_us.saturating_sub(buffered_us));
        self.emitted = 0;
        ClockEvent::Resynced { gap_us }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;
    const FRAME: u64 = 1024;
    /// 1024 samples at 48 kHz, truncated like the encoder computes it.
    const FRAME_US: u64 = 21_333;

    #[test]
    fn continuous_input_matches_sample_count() {
        let mut clock = AudioClock::new(SR);
        assert_eq!(clock.observe(1_000_000, 0), ClockEvent::Anchored);
        assert_eq!(clock.next_timestamp_us(), 1_000_000);
        clock.advance(FRAME);
        assert_eq!(clock.next_timestamp_us(), 1_000_000 + FRAME_US);
        // Next chunk arrives where the count says (+ sub-ms jitter).
        assert_eq!(
            clock.observe(1_000_000 + FRAME_US + 400, 0),
            ClockEvent::Continuous
        );
        clock.advance(FRAME);
        assert_eq!(clock.next_timestamp_us(), 1_000_000 + 42_666);
    }

    /// Jitter up to the threshold never moves the clock.
    #[test]
    fn small_deviation_is_tolerated() {
        let mut clock = AudioClock::new(SR);
        clock.observe(0, 0);
        clock.advance(FRAME);
        let event = clock.observe(FRAME_US + RESYNC_THRESHOLD_US, 0);
        assert_eq!(event, ClockEvent::Continuous);
        assert_eq!(clock.next_timestamp_us(), FRAME_US);
    }

    /// After a 2 s capture gap the next frame is stamped at the new
    /// capture time instead of 2 s in the past.
    #[test]
    fn gap_reanchors_forward() {
        let mut clock = AudioClock::new(SR);
        clock.observe(10_000_000, 0);
        clock.advance(10 * FRAME);
        let resumed = 12_500_000;
        let event = clock.observe(resumed, 0);
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
        clock.observe(0, 0);
        clock.advance(FRAME);
        let before = clock.next_timestamp_us();
        let buffered = 480; // 10 ms
        clock.observe(1_000_000, buffered);
        let after = clock.next_timestamp_us();
        assert_eq!(after, 1_000_000 - 10_000);
        assert!(after > before);
    }

    #[test]
    fn backward_jump_is_ignored() {
        let mut clock = AudioClock::new(SR);
        clock.observe(5_000_000, 0);
        clock.advance(FRAME);
        assert_eq!(clock.observe(4_000_000, 0), ClockEvent::Continuous);
        assert_eq!(clock.next_timestamp_us(), 5_000_000 + FRAME_US);
    }
}
