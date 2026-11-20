use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// What a render loop should do with a frame after [`Clock::wait_for`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Due {
    /// The frame's time has come: present it now.
    Show,
    /// The frame is too far behind the timeline: drop it and fetch the next.
    Late,
    /// [`Clock::stop`] was called: end the render loop.
    Stopped,
}

/// Longest a frame may wait for its turn; a frame due further ahead than
/// this is treated as a timeline jump and re-anchors the clock instead.
const MAX_WAIT: Duration = Duration::from_secs(2);

#[derive(Debug, PartialEq, Eq)]
enum Plan {
    Anchor,
    Wait(Duration),
    Show,
    Late,
}

#[derive(Default)]
struct State {
    anchor: Option<(Instant, u64)>,
    last_shown: Option<Instant>,
    paused: bool,
    stopped: bool,
}

/// Paces decoded frames against wall-clock time for an external render loop
/// (the desktop shell drives it; nothing inside this crate does).
///
/// Contract: one render thread calls [`Clock::wait_for`] with each frame's
/// timestamp, in presentation order, and acts on the returned [`Due`]. Any
/// other thread may call [`Clock::pause`], [`Clock::resume`] and
/// [`Clock::stop`]. The first frame, a frame earlier than the current anchor
/// (loop restart), a frame due more than two seconds ahead (timestamp jump)
/// and a frame after a long stall all re-anchor the timeline.
pub struct Clock {
    state: Mutex<State>,
    changed: Condvar,
    max_lateness: Duration,
}

impl Clock {
    /// Creates a clock that still shows frames up to `max_lateness` behind
    /// their due time and reports later ones as [`Due::Late`].
    pub fn new(max_lateness: Duration) -> Self {
        Self {
            state: Mutex::new(State::default()),
            changed: Condvar::new(),
            max_lateness,
        }
    }

    /// Holds the render thread inside [`Clock::wait_for`] until
    /// [`Clock::resume`]; the timeline re-anchors on resume.
    pub fn pause(&self) {
        let mut state = self.lock();
        state.paused = true;
        state.anchor = None;
        drop(state);
        // Wake a waiter sleeping on a frame's due time so it parks on the
        // pause instead of showing the frame once its timeout expires.
        self.changed.notify_all();
    }

    /// Releases a paused render thread.
    pub fn resume(&self) {
        self.lock().paused = false;
        self.changed.notify_all();
    }

    /// Makes every current and future [`Clock::wait_for`] return
    /// [`Due::Stopped`].
    pub fn stop(&self) {
        self.lock().stopped = true;
        self.changed.notify_all();
    }

    /// Blocks until the frame stamped `pts_us` is due (or the clock is
    /// stopped) and says what to do with it.
    pub fn wait_for(&self, pts_us: u64) -> Due {
        let mut state = self.lock();
        loop {
            if state.stopped {
                return Due::Stopped;
            }
            if state.paused {
                state = self.changed.wait(state).unwrap_or_else(|e| e.into_inner());
                continue;
            }
            let now = Instant::now();
            match plan(
                state.anchor,
                state.last_shown,
                now,
                pts_us,
                self.max_lateness,
            ) {
                Plan::Anchor => {
                    state.anchor = Some((now, pts_us));
                    state.last_shown = Some(now);
                    return Due::Show;
                }
                Plan::Wait(wait) => {
                    state = self
                        .changed
                        .wait_timeout(state, wait)
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                }
                Plan::Show => {
                    state.last_shown = Some(now);
                    return Due::Show;
                }
                Plan::Late => return Due::Late,
            }
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

fn plan(
    anchor: Option<(Instant, u64)>,
    last_shown: Option<Instant>,
    now: Instant,
    pts_us: u64,
    max_lateness: Duration,
) -> Plan {
    let Some((wall, base)) = anchor else {
        return Plan::Anchor;
    };
    if pts_us < base {
        return Plan::Anchor;
    }
    // A forward jump in the timestamps (a gap in the file, a broken PTS)
    // would otherwise park the render thread for the whole gap.
    let Some(due) = wall.checked_add(Duration::from_micros(pts_us - base)) else {
        return Plan::Anchor;
    };
    if due > now {
        let wait = due - now;
        return if wait > MAX_WAIT {
            Plan::Anchor
        } else {
            Plan::Wait(wait)
        };
    }
    if now - due <= max_lateness {
        return Plan::Show;
    }
    let starved = last_shown.is_none_or(|shown| now - shown >= max_lateness * 2);
    if starved { Plan::Anchor } else { Plan::Late }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::mpsc;

    use super::*;

    const LATENESS: Duration = Duration::from_millis(50);

    fn anchored(now: Instant) -> Option<(Instant, u64)> {
        Some((now, 1_000_000))
    }

    #[test]
    fn the_first_frame_sets_the_timeline() {
        assert_eq!(plan(None, None, Instant::now(), 5, LATENESS), Plan::Anchor);
    }

    #[test]
    fn a_frame_ahead_of_the_timeline_waits_for_its_turn() {
        let start = Instant::now();
        assert_eq!(
            plan(anchored(start), Some(start), start, 1_040_000, LATENESS),
            Plan::Wait(Duration::from_millis(40))
        );
    }

    #[test]
    fn a_forward_jump_resynchronises_instead_of_freezing() {
        let start = Instant::now();
        assert_eq!(
            plan(
                anchored(start),
                Some(start),
                start,
                1_000_000 + 60_000_000,
                LATENESS
            ),
            Plan::Anchor
        );
        assert_eq!(
            plan(anchored(start), Some(start), start, u64::MAX, LATENESS),
            Plan::Anchor
        );
    }

    #[test]
    fn pausing_holds_a_frame_that_was_waiting() {
        let clock = Arc::new(Clock::new(LATENESS));
        assert_eq!(clock.wait_for(0), Due::Show);
        let (sent, received) = mpsc::channel();
        let waiter = Arc::clone(&clock);
        std::thread::spawn(move || sent.send(waiter.wait_for(1_500_000)));
        std::thread::sleep(Duration::from_millis(50));
        clock.pause();
        assert!(received.recv_timeout(Duration::from_secs(2)).is_err());
        clock.stop();
        assert_eq!(
            received.recv_timeout(Duration::from_secs(5)),
            Ok(Due::Stopped)
        );
    }

    #[test]
    fn a_frame_slightly_behind_is_still_shown() {
        let start = Instant::now();
        let now = start + Duration::from_millis(60);
        assert_eq!(
            plan(anchored(start), Some(now), now, 1_020_000, LATENESS),
            Plan::Show
        );
    }

    #[test]
    fn a_frame_too_far_behind_is_late_when_another_was_just_shown() {
        let start = Instant::now();
        let now = start + Duration::from_millis(200);
        let shown = now - Duration::from_millis(10);
        assert_eq!(
            plan(anchored(start), Some(shown), now, 1_020_000, LATENESS),
            Plan::Late
        );
    }

    #[test]
    fn a_late_frame_is_shown_when_nothing_was_shown_lately() {
        let start = Instant::now();
        let now = start + Duration::from_millis(200);
        assert_eq!(
            plan(anchored(start), Some(start), now, 1_020_000, LATENESS),
            Plan::Anchor
        );
    }

    #[test]
    fn a_long_stall_resynchronises() {
        let start = Instant::now();
        let now = start + Duration::from_secs(3);
        assert_eq!(
            plan(anchored(start), Some(start), now, 1_020_000, LATENESS),
            Plan::Anchor
        );
    }

    fn shown_times(arrival: Duration, frame: Duration, count: u32) -> Vec<Duration> {
        let start = Instant::now();
        let (mut anchor, mut last_shown, mut now) = (None, None, start);
        let mut shown = Vec::new();
        for index in 0..count {
            now = now.max(start + arrival * index);
            let pts = frame.as_micros() as u64 * u64::from(index);
            loop {
                match plan(anchor, last_shown, now, pts, LATENESS) {
                    Plan::Anchor => {
                        anchor = Some((now, pts));
                        last_shown = Some(now);
                        shown.push(now - start);
                        break;
                    }
                    Plan::Wait(wait) => now += wait,
                    Plan::Show => {
                        last_shown = Some(now);
                        shown.push(now - start);
                        break;
                    }
                    Plan::Late => break,
                }
            }
        }
        shown
    }

    #[test]
    fn a_decoder_slower_than_the_clip_keeps_showing_frames_without_long_freezes() {
        let shown = shown_times(
            Duration::from_millis(50),
            Duration::from_micros(33_333),
            120,
        );
        let longest_gap = shown.windows(2).map(|w| w[1] - w[0]).max().unwrap();
        assert!(
            shown.len() >= 90,
            "only {} of 120 frames shown",
            shown.len()
        );
        assert!(longest_gap <= Duration::from_millis(150), "{longest_gap:?}");
    }

    #[test]
    fn a_decoder_faster_than_the_clip_shows_every_frame_on_time() {
        let shown = shown_times(Duration::from_millis(20), Duration::from_micros(33_333), 90);
        assert_eq!(shown.len(), 90);
    }

    #[test]
    fn a_loop_restart_resynchronises() {
        let start = Instant::now();
        assert_eq!(
            plan(anchored(start), Some(start), start, 0, LATENESS),
            Plan::Anchor
        );
    }

    #[test]
    fn a_paused_clock_holds_the_frame_until_it_resumes() {
        let clock = Arc::new(Clock::new(LATENESS));
        clock.pause();
        let (sent, received) = mpsc::channel();
        let waiter = Arc::clone(&clock);
        std::thread::spawn(move || sent.send(waiter.wait_for(0)));
        assert!(received.recv_timeout(Duration::from_millis(100)).is_err());
        clock.resume();
        assert_eq!(received.recv_timeout(Duration::from_secs(5)), Ok(Due::Show));
    }

    #[test]
    fn stopping_releases_a_paused_waiter() {
        let clock = Arc::new(Clock::new(LATENESS));
        clock.pause();
        let (sent, received) = mpsc::channel();
        let waiter = Arc::clone(&clock);
        std::thread::spawn(move || sent.send(waiter.wait_for(0)));
        clock.stop();
        assert_eq!(
            received.recv_timeout(Duration::from_secs(5)),
            Ok(Due::Stopped)
        );
    }
}
