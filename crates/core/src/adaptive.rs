//! Shared state for receiver-feedback adaptive bitrate.
//!
//! Live streaming to receivers we don't control (Chromecast, AirPlay,
//! generic HLS players) needs a feedback loop the receiver doesn't
//! explicitly provide: we have to *infer* its bandwidth from the rate
//! at which it pulls segments and shrink the encode budget when the
//! link is stressed so the receiver doesn't run dry. This module is
//! the cross-component wiring for that loop.
//!
//! Producers (`ferricast-hls`'s HTTP handler) update [`AdaptiveBitrateState`]
//! with per-segment delivery measurements. Consumers (`ferricast`'s
//! stream manager) poll the same state and call
//! [`crate::VideoEncoder::set_bitrate_kbps`] when the recommended
//! target moves.
//!
//! The state is intentionally simple: an atomic target the encoder
//! should aim for, plus an EMA of recent pressure used to decide
//! when to step that target up or down. No async, no channels —
//! contention is cheap and there's at most one updater and one
//! reader on the hot path.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Adaptive bitrate controller shared between an HLS server (or any
/// other component with delivery telemetry) and the encoder loop.
///
/// Construct once per stream with [`AdaptiveBitrateState::new`] and
/// hand `Arc` clones to every component that needs to read or write
/// it. Lifetime is the lifetime of the stream — drop happens
/// implicitly when the manager teardown clears its references.
#[derive(Debug)]
pub struct AdaptiveBitrateState {
    /// Target bitrate the encoder should currently aim for, in kbps.
    /// The HLS observer mutates this on sustained pressure / slack;
    /// the encoder loop reads it and reconfigures when it changes.
    /// Stored as `u32` because that's NVENC's native unit (multiplied
    /// by 1000 to get bps).
    target_kbps: AtomicU32,

    /// Hard upper bound — never raise the target above this. Set
    /// from the original `StreamConfig::bitrate_kbps` (already capped
    /// to the device's `DeviceCapabilities::max_bitrate_kbps`) so we
    /// can recover toward the configured ceiling but never overshoot
    /// the receiver's declared decoder cap.
    pub ceiling_kbps: u32,

    /// Hard lower bound — never drop the target below this. Default
    /// is 500 kbps; below that quality is unusable at any resolution.
    pub floor_kbps: u32,

    /// EMA of the most recent `budget_used_pct` samples (0..=100).
    /// 100 = a single segment took exactly its EXTINF to deliver
    /// (player has zero headroom and will hit BUFFERING on the next
    /// jitter spike); 0 = instant delivery. Stored as a u32 to keep
    /// the whole thing lock-free; only the producer ever writes.
    pressure_pct_x100: AtomicU32,

    /// Last time (monotonic ns since some epoch) we changed
    /// `target_kbps`. Used for hysteresis: don't oscillate at every
    /// segment, wait `MIN_ADJUST_INTERVAL` between adjustments.
    last_adjust_ns: AtomicU64,
}

impl AdaptiveBitrateState {
    /// Minimum gap between two target-bitrate adjustments. Stops the
    /// controller from yo-yoing every segment when delivery time
    /// hovers near a threshold.
    pub const MIN_ADJUST_INTERVAL_MS: u64 = 4_000;

    /// Above this EMA we consider the link saturated and step the
    /// target bitrate down.
    pub const PRESSURE_DOWN_PCT: u32 = 70;

    /// Below this EMA we consider the link slack and step the target
    /// up.
    pub const PRESSURE_UP_PCT: u32 = 35;

    /// Multiplicative step on downshift. 0.75 = 25 % cut, conservative
    /// enough to actually relieve pressure on a slow link without
    /// dropping to unwatchable quality in one move.
    pub const DOWNSHIFT_NUM: u32 = 3;
    pub const DOWNSHIFT_DEN: u32 = 4;

    /// Additive step on upshift, in kbps. Slow recovery — bandwidth
    /// returning isn't a license to spike right back up, the receiver
    /// just paid the cost of catching up.
    pub const UPSHIFT_KBPS: u32 = 200;

    pub fn new(initial_kbps: u32) -> Arc<Self> {
        Arc::new(Self {
            target_kbps: AtomicU32::new(initial_kbps),
            ceiling_kbps: initial_kbps,
            floor_kbps: 500,
            pressure_pct_x100: AtomicU32::new(0),
            last_adjust_ns: AtomicU64::new(0),
        })
    }

    /// Encoder loop's read path. Cheap: a single relaxed atomic
    /// load. Call every frame; only act when the value differs from
    /// the last applied.
    pub fn target_kbps(&self) -> u32 {
        self.target_kbps.load(Ordering::Relaxed)
    }

    /// HLS server's write path. Feeds a fresh `budget_used_pct`
    /// observation in (0..=100 typically; clamped) and runs the
    /// controller. Returns `Some(new_target_kbps)` when the target
    /// changed, so the caller can log a transition; `None` otherwise.
    pub fn record_pressure(&self, sample_pct: u32) -> Option<u32> {
        // EMA: 7/8 history + 1/8 new sample. Same weighting as the
        // segmenter's PTS rate EMA, so users have one mental model
        // for how fast either loop adapts.
        let sample_x100 = sample_pct.min(200).saturating_mul(100);
        let prev = self.pressure_pct_x100.load(Ordering::Relaxed);
        let ema = (prev.saturating_mul(7) + sample_x100) / 8;
        self.pressure_pct_x100.store(ema, Ordering::Relaxed);

        let now_ns = monotonic_ns();
        let last = self.last_adjust_ns.load(Ordering::Relaxed);
        // last == 0 is the sentinel for "never adjusted yet" so the
        // very first eligible move fires without waiting out a full
        // interval against the process start time. Once we record a
        // real adjustment timestamp the normal cool-down kicks in.
        if last != 0
            && now_ns.saturating_sub(last) < Self::MIN_ADJUST_INTERVAL_MS * 1_000_000
        {
            return None;
        }

        let current = self.target_kbps.load(Ordering::Relaxed);
        let ema_pct = ema / 100;
        let new_target = if ema_pct >= Self::PRESSURE_DOWN_PCT {
            // Downshift: multiplicative. Round toward floor.
            (current.saturating_mul(Self::DOWNSHIFT_NUM) / Self::DOWNSHIFT_DEN).max(self.floor_kbps)
        } else if ema_pct <= Self::PRESSURE_UP_PCT && current < self.ceiling_kbps {
            (current.saturating_add(Self::UPSHIFT_KBPS)).min(self.ceiling_kbps)
        } else {
            return None;
        };

        if new_target == current {
            return None;
        }
        self.target_kbps.store(new_target, Ordering::Relaxed);
        self.last_adjust_ns.store(now_ns, Ordering::Relaxed);
        Some(new_target)
    }

    /// Force-clamp the target to its floor. Used when an explicit
    /// fatal signal (Chromecast `detailedErrorCode=301`, repeated
    /// BUFFERING) tells us a soft step isn't enough.
    pub fn drop_to_floor(&self) -> u32 {
        self.target_kbps.store(self.floor_kbps, Ordering::Relaxed);
        self.last_adjust_ns
            .store(monotonic_ns(), Ordering::Relaxed);
        self.floor_kbps
    }

    /// Feed a freshly-measured *available link capacity* sample in
    /// kbps (i.e. how fast the receiver actually pulled a segment,
    /// not how big the segment was). Used during bring-up to seed
    /// the controller with a realistic target instead of starting
    /// at the configured ceiling and discovering the real link
    /// the hard way (one BUFFERING later).
    ///
    /// Strategy:
    /// * If the measured throughput is comfortably above the current
    ///   target (≥2×), leave the target alone — there's plenty of
    ///   headroom, no point disturbing the encoder.
    /// * If the measured throughput is close to the target (1× to
    ///   2×), pull the target down to ~70 % of the measurement so
    ///   we ride the curve with a margin for variance.
    /// * If the measurement is *below* the target, slam the target
    ///   down to the measurement immediately — we're already
    ///   over-budget and the next jitter spike will trigger 301.
    ///
    /// Returns `Some(new_target)` if the target moved, `None`
    /// otherwise. Bypasses the normal hysteresis because this is a
    /// calibration step that should converge quickly.
    pub fn observe_link_capacity_kbps(&self, measured_kbps: u32) -> Option<u32> {
        let current = self.target_kbps.load(Ordering::Relaxed);
        let new_target = if measured_kbps >= current.saturating_mul(2) {
            return None;
        } else if measured_kbps >= current {
            // 70 % of measured, clamped to the controller's bounds.
            ((measured_kbps as u64 * 7 / 10) as u32)
                .clamp(self.floor_kbps, self.ceiling_kbps)
        } else {
            // Already over-budget. Drop to measured.
            measured_kbps.clamp(self.floor_kbps, self.ceiling_kbps)
        };
        if new_target == current {
            return None;
        }
        self.target_kbps.store(new_target, Ordering::Relaxed);
        self.last_adjust_ns.store(monotonic_ns(), Ordering::Relaxed);
        Some(new_target)
    }

    pub fn pressure_pct(&self) -> u32 {
        self.pressure_pct_x100.load(Ordering::Relaxed) / 100
    }
}

fn monotonic_ns() -> u64 {
    // `Instant` doesn't expose nanos directly across platforms; use
    // a process-local anchor and return ns since it. Wraps on
    // overflow after ~584 years — fine.
    use std::sync::OnceLock;
    use std::time::Instant;
    static ANCHOR: OnceLock<Instant> = OnceLock::new();
    let anchor = ANCHOR.get_or_init(Instant::now);
    anchor.elapsed().as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to fast-forward the controller's hysteresis. Tests
    /// can't rely on wall-clock spacing because they need to be
    /// fast; we manually reset `last_adjust_ns` to 0 between
    /// samples to simulate enough elapsed time for the next
    /// adjustment to be eligible.
    fn force_adjustable(state: &AdaptiveBitrateState) {
        state.last_adjust_ns.store(0, Ordering::Relaxed);
    }

    #[test]
    fn ema_starts_at_zero() {
        let s = AdaptiveBitrateState::new(3000);
        assert_eq!(s.pressure_pct(), 0);
        assert_eq!(s.target_kbps(), 3000);
    }

    #[test]
    fn sustained_pressure_steps_target_down() {
        let s = AdaptiveBitrateState::new(3000);
        // First sample with high pressure passes the hysteresis
        // sentinel (last_adjust_ns == 0) and the EMA jumps from 0
        // to (95 * 100) / 8 = 1187 (12 %) — below the 70 %
        // threshold, so no adjustment yet.
        // Feed enough samples for the EMA to climb past 70.
        for _ in 0..16 {
            s.record_pressure(95);
        }
        // After 16 hits the EMA is near 95 %; one of those calls
        // already fired the first downshift (target was 3000 →
        // 2250) and subsequent ones were blocked by the 4 s
        // hysteresis. Result is a single step down.
        let v = s.target_kbps();
        assert!(v < 3000, "target should have decreased, got {v}");
        // Each downshift is a *4 multiplicative step (×3/4); the
        // first lands exactly on 2250.
        assert_eq!(v, 2250);
    }

    #[test]
    fn slack_steps_target_up_but_caps_at_ceiling() {
        let s = AdaptiveBitrateState::new(3000);
        // First force it down so there's room to recover.
        s.target_kbps.store(1000, Ordering::Relaxed);
        // First slack sample fires immediately (last_adjust_ns
        // sentinel), bumping 1000 → 1200. Subsequent samples in
        // this loop are blocked by the cool-down.
        s.record_pressure(10);
        assert_eq!(s.target_kbps(), 1200, "+200 kbps step on first slack sample");
        // Now drive it all the way back up and confirm the ceiling.
        for _ in 0..100 {
            force_adjustable(&s);
            s.record_pressure(10);
        }
        assert_eq!(s.target_kbps(), 3000);
    }

    #[test]
    fn floor_is_respected_on_repeated_pressure() {
        let s = AdaptiveBitrateState::new(1000);
        for _ in 0..200 {
            force_adjustable(&s);
            s.record_pressure(100);
        }
        assert_eq!(s.target_kbps(), s.floor_kbps);
    }

    #[test]
    fn hysteresis_blocks_back_to_back_adjustments() {
        let s = AdaptiveBitrateState::new(3000);
        for _ in 0..16 {
            s.record_pressure(95);
        }
        force_adjustable(&s);
        let _ = s.record_pressure(95); // first adjust
        // Immediately follow with another high sample; without
        // force_adjustable the hysteresis should silently swallow
        // it.
        let again = s.record_pressure(95);
        assert!(
            again.is_none(),
            "second adjustment within MIN_ADJUST_INTERVAL must be suppressed"
        );
    }

    #[test]
    fn drop_to_floor_is_immediate() {
        let s = AdaptiveBitrateState::new(3000);
        let v = s.drop_to_floor();
        assert_eq!(v, s.floor_kbps);
        assert_eq!(s.target_kbps(), s.floor_kbps);
    }

    #[test]
    fn link_probe_leaves_target_alone_when_link_is_2x_plus_capacity() {
        let s = AdaptiveBitrateState::new(2500);
        // Link can pull at 8 Mbps; encoder at 2.5 Mbps; ratio > 2,
        // nothing to do.
        assert!(s.observe_link_capacity_kbps(8000).is_none());
        assert_eq!(s.target_kbps(), 2500);
    }

    #[test]
    fn link_probe_settles_at_70pct_when_link_is_tight() {
        let s = AdaptiveBitrateState::new(2500);
        // Link can only pull at 3 Mbps; encoder at 2.5 Mbps; ratio
        // is 1.2, we want to step down to ~2.1 Mbps for headroom.
        let new = s.observe_link_capacity_kbps(3000).unwrap();
        assert_eq!(new, 2100, "70 % of measured");
        assert_eq!(s.target_kbps(), 2100);
    }

    #[test]
    fn link_probe_slams_target_down_when_link_below_target() {
        let s = AdaptiveBitrateState::new(2500);
        let new = s.observe_link_capacity_kbps(1500).unwrap();
        assert_eq!(new, 1500, "drop straight to measured");
    }

    #[test]
    fn link_probe_respects_floor() {
        let s = AdaptiveBitrateState::new(2500);
        let new = s.observe_link_capacity_kbps(200).unwrap();
        assert_eq!(new, s.floor_kbps);
    }
}
