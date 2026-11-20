/// Presentation times of packets handed to a reordering decoder that has not
/// output their pictures yet, earliest first.
#[derive(Default)]
pub(crate) struct PtsQueue(Vec<u64>);

/// Deepest reorder the supported decoders can hold in flight: H.264 and HEVC
/// keep at most 16 pictures in their decoded picture buffer. A packet the
/// decoder swallowed without ever outputting a picture would otherwise leave
/// its PTS queued forever and shift every later frame by one.
const MAX_REORDER_DEPTH: usize = 16;

impl PtsQueue {
    pub(crate) fn push(&mut self, pts: u64) {
        let at = self.0.partition_point(|&queued| queued <= pts);
        self.0.insert(at, pts);
        if self.0.len() > MAX_REORDER_DEPTH {
            // The earliest entry is the stale one: the decoder outputs in
            // presentation order and would already have claimed it.
            self.0.remove(0);
        }
    }

    pub(crate) fn pop_earliest(&mut self) -> Option<u64> {
        (!self.0.is_empty()).then(|| self.0.remove(0))
    }

    pub(crate) fn remove(&mut self, pts: u64) {
        if let Ok(at) = self.0.binary_search(&pts) {
            self.0.remove(at);
        }
    }

    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }
}

/// Frame spacing assumed until the stream reports packet durations.
const DEFAULT_FRAME_US: u64 = 33_333;

/// Decides which decoded frames are shown and rebases their timestamps so
/// the first displayed frame of the stream sits at 0.
pub(crate) struct Presentation {
    /// Timestamp (demuxer timeline) of the first frame meant to be shown;
    /// anything earlier is pre-roll.
    start_us: u64,
    /// Demuxer timestamp of the last frame handed out.
    last_us: Option<u64>,
    /// After restarting the decoder mid-stream, frames up to and including
    /// this timestamp were already shown and must not be replayed.
    shown_through_us: Option<u64>,
    frame_us: u64,
}

impl Presentation {
    pub(crate) fn new(start_us: u64) -> Self {
        Self {
            start_us,
            last_us: None,
            shown_through_us: None,
            frame_us: DEFAULT_FRAME_US,
        }
    }

    /// Records a packet's duration as the spacing used to extrapolate
    /// timestamps the decoder emits without a matching packet.
    pub(crate) fn observe_duration(&mut self, duration_us: Option<u64>) {
        if let Some(duration) = duration_us.filter(|&d| d > 0) {
            self.frame_us = duration;
        }
    }

    /// Timestamp to give a decoded frame that has no queued packet PTS left
    /// (a flush emitting more pictures than packets are pending).
    pub(crate) fn extrapolated(&self) -> u64 {
        self.last_us
            .map_or(self.start_us, |last| last.saturating_add(self.frame_us))
    }

    /// Returns the rebased timestamp when the frame at `pts_us` (demuxer
    /// timeline) should be shown, or `None` to drop it.
    pub(crate) fn admit(&mut self, pts_us: u64) -> Option<u64> {
        if pts_us < self.start_us {
            return None;
        }
        if let Some(shown) = self.shown_through_us {
            if pts_us <= shown {
                return None;
            }
            self.shown_through_us = None;
        }
        self.last_us = Some(pts_us);
        Some(pts_us - self.start_us)
    }

    /// Restarting from the beginning: everything may be shown again.
    pub(crate) fn restart(&mut self) {
        self.last_us = None;
        self.shown_through_us = None;
    }

    /// Restarting the decoder mid-stream: hide every frame already shown.
    pub(crate) fn resume(&mut self) {
        self.shown_through_us = self.last_us;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earliest_pts_comes_out_first_whatever_the_decode_order() {
        let mut queue = PtsQueue::default();
        for pts in [0, 300, 100, 200] {
            queue.push(pts);
        }
        let order: Vec<_> = std::iter::from_fn(|| queue.pop_earliest()).collect();
        assert_eq!(order, [0, 100, 200, 300]);
    }

    #[test]
    fn equal_pts_are_kept_once_each() {
        let mut queue = PtsQueue::default();
        queue.push(5);
        queue.push(5);
        assert_eq!(queue.pop_earliest(), Some(5));
        assert_eq!(queue.pop_earliest(), Some(5));
        assert_eq!(queue.pop_earliest(), None);
    }

    #[test]
    fn a_failed_packet_gives_its_pts_back() {
        let mut queue = PtsQueue::default();
        queue.push(10);
        queue.push(20);
        queue.remove(10);
        queue.remove(99);
        assert_eq!(queue.pop_earliest(), Some(20));
    }

    #[test]
    fn the_queue_never_holds_more_than_the_reorder_depth() {
        let mut queue = PtsQueue::default();
        let pushed = MAX_REORDER_DEPTH as u64 + 4;
        for pts in 0..pushed {
            queue.push(pts);
        }
        let kept: Vec<_> = std::iter::from_fn(|| queue.pop_earliest()).collect();
        assert_eq!(kept.len(), MAX_REORDER_DEPTH);
        assert_eq!(kept.first(), Some(&4));
        assert_eq!(kept.last(), Some(&(pushed - 1)));
    }

    #[test]
    fn pre_roll_frames_are_hidden_and_the_rest_rebased() {
        let mut presentation = Presentation::new(1_000);
        assert_eq!(presentation.admit(0), None);
        assert_eq!(presentation.admit(500), None);
        assert_eq!(presentation.admit(1_000), Some(0));
        assert_eq!(presentation.admit(1_500), Some(500));
    }

    #[test]
    fn resuming_hides_frames_already_shown() {
        let mut presentation = Presentation::new(0);
        for pts in [0, 10, 20] {
            assert!(presentation.admit(pts).is_some());
        }
        presentation.resume();
        assert_eq!(presentation.admit(0), None);
        assert_eq!(presentation.admit(20), None);
        assert_eq!(presentation.admit(30), Some(30));
        assert_eq!(presentation.admit(25), Some(25));
    }

    #[test]
    fn restarting_shows_everything_again() {
        let mut presentation = Presentation::new(0);
        assert!(presentation.admit(50).is_some());
        presentation.resume();
        presentation.restart();
        assert_eq!(presentation.admit(0), Some(0));
    }

    #[test]
    fn missing_timestamps_continue_from_the_last_frame() {
        let mut presentation = Presentation::new(100);
        assert_eq!(presentation.extrapolated(), 100);
        presentation.observe_duration(Some(40));
        presentation.admit(200);
        assert_eq!(presentation.extrapolated(), 240);
    }
}
