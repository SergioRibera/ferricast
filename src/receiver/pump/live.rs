//! Keeps a live stream at its live edge.
//!
//! A pushed stream (screen mirroring) is presented as soon as it can be,
//! not buffered like a file. Whatever queued while the pipeline came up
//! (a hardware decoder starting, a window opening) or while the receiver
//! stalled must not become a permanent delay, so the presentation
//! schedule is measured against how early packets actually arrive.
//!
//! For each stream the edge keeps the smallest `arrival - pts` seen over
//! the last second or two: the transit of the packets that came through
//! fastest. Backlog read in a burst only shows larger values, so it does
//! not count. Presenting `pts` at `pts + transit + target` keeps every
//! packet on time that is at most `target` slower than the fastest one.
//! The schedule must suit the later of the two streams, so the larger
//! transit wins. When the pump's clock drifts more than a threshold
//! away from that schedule, in either direction, the pump moves it back
//! and drops what has become too late to present.

use std::time::{Duration, Instant};

use super::latency::Kind;

/// How long a transit minimum stands for: the minimum is taken over the
/// current and the previous window.
const WINDOW: Duration = Duration::from_secs(1);

/// Smallest `arrival - pts` of one stream, µs, over the recent windows.
#[derive(Debug, Default, Clone, Copy)]
struct Transit {
    current: Option<i64>,
    previous: Option<i64>,
}

impl Transit {
    fn observe(&mut self, offset_us: i64) {
        self.current = Some(self.current.map_or(offset_us, |c| c.min(offset_us)));
    }

    /// Starts a new window; a stream that sent nothing keeps what it had.
    fn rotate(&mut self) {
        if self.current.is_some() {
            self.previous = self.current.take();
        }
    }

    fn min(&self) -> Option<i64> {
        match (self.current, self.previous) {
            (Some(c), Some(p)) => Some(c.min(p)),
            (c, p) => c.or(p),
        }
    }
}

/// How far the pump's schedule is from the live edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Drift {
    /// Within the thresholds.
    Fine,
    /// Presenting this much later than needed: move the schedule
    /// earlier and drop what is then stale.
    Behind(Duration),
    /// Presenting this much earlier than packets arrive: move the
    /// schedule later.
    Ahead(Duration),
}

/// Estimates the live edge from packet arrivals.
#[derive(Debug)]
pub(super) struct LiveEdge {
    /// Time kept between the fastest arrival and presentation.
    target: Duration,
    /// Drift tolerated before the schedule is moved.
    tolerance: Duration,
    /// Origin of the µs scale arrivals are measured on.
    origin: Instant,
    window_start: Instant,
    video: Transit,
    audio: Transit,
}

impl LiveEdge {
    pub(super) fn new(target: Duration, tolerance: Duration) -> Self {
        let now = Instant::now();
        Self {
            target,
            tolerance,
            origin: now,
            window_start: now,
            video: Transit::default(),
            audio: Transit::default(),
        }
    }

    /// `at` on the edge's µs scale; negative before the origin.
    fn micros_since_origin(&self, at: Instant) -> i64 {
        let micros = |d: Duration| i64::try_from(d.as_micros()).unwrap_or(i64::MAX);
        match at.checked_duration_since(self.origin) {
            Some(after) => micros(after),
            None => -micros(self.origin.duration_since(at)),
        }
    }

    /// A packet of `kind` stamped `pts_us` was read at `at`.
    pub(super) fn observe(&mut self, kind: Kind, pts_us: u64, at: Instant) {
        if at.saturating_duration_since(self.window_start) >= WINDOW {
            self.video.rotate();
            self.audio.rotate();
            self.window_start = at;
        }
        let pts = i64::try_from(pts_us).unwrap_or(i64::MAX);
        let offset = self.micros_since_origin(at).saturating_sub(pts);
        match kind {
            Kind::Video => self.video.observe(offset),
            Kind::Audio => self.audio.observe(offset),
        }
    }

    /// Forgets what was measured (the stream restarted).
    pub(super) fn reset(&mut self) {
        self.video = Transit::default();
        self.audio = Transit::default();
        self.window_start = Instant::now();
    }

    /// When `pts_us` should be presented, once any packet arrived.
    /// `near` is any instant close to the answer; instants are only
    /// formed relative to it, so the answer may well precede the edge's
    /// origin.
    pub(super) fn due(&self, pts_us: u64, near: Instant) -> Option<Instant> {
        let transit = match (self.video.min(), self.audio.min()) {
            (Some(v), Some(a)) => v.max(a),
            (v, a) => v.or(a)?,
        };
        let target = i64::try_from(self.target.as_micros()).unwrap_or(i64::MAX);
        let pts = i64::try_from(pts_us).unwrap_or(i64::MAX);
        let at_us = pts.saturating_add(transit).saturating_add(target);
        let from_near = at_us.saturating_sub(self.micros_since_origin(near));
        let by = Duration::from_micros(from_near.unsigned_abs());
        if from_near >= 0 {
            near.checked_add(by)
        } else {
            near.checked_sub(by)
        }
    }

    /// How a schedule that presents `pts_us` at `scheduled` compares
    /// with the live edge.
    pub(super) fn drift(&self, pts_us: u64, scheduled: Instant) -> Drift {
        let Some(due) = self.due(pts_us, scheduled) else {
            return Drift::Fine;
        };
        if let Some(late) = scheduled.checked_duration_since(due)
            && late > self.tolerance
        {
            return Drift::Behind(late);
        }
        if let Some(early) = due.checked_duration_since(scheduled)
            && early > self.tolerance
        {
            return Drift::Ahead(early);
        }
        Drift::Fine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TARGET: Duration = Duration::from_millis(200);
    const TOLERANCE: Duration = Duration::from_millis(100);

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// An edge whose origin is `now`, with pts counted from 10 s.
    fn edge() -> (LiveEdge, Instant) {
        let edge = LiveEdge::new(TARGET, TOLERANCE);
        let origin = edge.origin;
        (edge, origin)
    }

    const PTS0: u64 = 10_000_000;

    #[test]
    fn nothing_observed_means_no_opinion() {
        let (edge, origin) = edge();
        assert_eq!(edge.due(PTS0, origin), None);
        assert_eq!(edge.drift(PTS0, origin), Drift::Fine);
    }

    #[test]
    fn a_startup_backlog_read_in_a_burst_is_not_the_edge() {
        let (mut edge, origin) = edge();
        // 600 ms of video queued while the pipeline came up, read at once
        // 1 s in: the newest packet sets the edge.
        let read_at = origin + ms(1_000);
        for i in 0..=36 {
            edge.observe(Kind::Video, PTS0 + i * 16_667, read_at);
        }
        let newest = PTS0 + 36 * 16_667;
        assert_eq!(edge.due(newest, origin), Some(read_at + TARGET));
        // A clock that started on the oldest packet is 600 ms behind.
        let scheduled_oldest = read_at + TARGET;
        match edge.drift(PTS0, scheduled_oldest) {
            Drift::Behind(by) => assert!(by.abs_diff(ms(600)) < ms(1), "{by:?}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn jitter_within_the_tolerance_moves_nothing() {
        let (mut edge, origin) = edge();
        for i in 0..60u64 {
            // Every packet is 20-80 ms in transit.
            let transit = ms(20 + (i * 37) % 60);
            let pts = PTS0 + i * 16_667;
            let at = origin + Duration::from_micros(i * 16_667) + transit;
            edge.observe(Kind::Video, pts, at);
        }
        let scheduled = origin + ms(20) + TARGET + ms(50);
        assert_eq!(edge.drift(PTS0, scheduled), Drift::Fine);
    }

    #[test]
    fn the_later_stream_sets_the_schedule() {
        let (mut edge, origin) = edge();
        edge.observe(Kind::Video, PTS0, origin + ms(30));
        // Audio comes through 120 ms slower than video.
        edge.observe(Kind::Audio, PTS0, origin + ms(150));
        assert_eq!(edge.due(PTS0, origin), Some(origin + ms(150) + TARGET));
    }

    #[test]
    fn a_stall_moves_the_edge_later_once_the_window_passes() {
        let (mut edge, origin) = edge();
        edge.observe(Kind::Video, PTS0, origin + ms(10));
        let on_time = origin + ms(10) + TARGET;
        assert_eq!(edge.drift(PTS0, on_time), Drift::Fine);
        // From now on everything is 500 ms slower; after two windows the
        // old minimum is gone.
        for s in 1..=3u64 {
            let pts = PTS0 + s * 1_000_000;
            edge.observe(Kind::Video, pts, origin + ms(10 + 500) + ms(s * 1_000));
        }
        match edge.drift(PTS0, on_time) {
            Drift::Ahead(by) => assert!(by.abs_diff(ms(500)) < ms(1), "{by:?}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn backlog_from_before_the_stream_started_is_due_in_the_past() {
        let (mut edge, origin) = edge();
        let now = origin + ms(5);
        // The newest packet is from now; the oldest queued one is 800 ms
        // older, from before the edge existed.
        edge.observe(Kind::Video, PTS0 + 800_000, now);
        let due = edge.due(PTS0, now).unwrap();
        assert_eq!(now - due, ms(600));
    }

    #[test]
    fn reset_forgets_the_transit() {
        let (mut edge, origin) = edge();
        edge.observe(Kind::Audio, PTS0, origin);
        edge.reset();
        assert_eq!(edge.due(PTS0, origin), None);
    }
}
