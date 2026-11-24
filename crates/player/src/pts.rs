/// Decides which decoded frames are shown and rebases their timestamps so
/// the first displayed frame of the stream sits at 0.
#[expect(
    clippy::struct_field_names,
    reason = "the `_us` suffix names the unit of every timestamp"
)]
pub(crate) struct Presentation {
    /// Timestamp (demuxer timeline) of the first frame meant to be shown;
    /// anything earlier is pre-roll.
    start_us: u64,
    /// Demuxer timestamp of the last frame handed out.
    last_us: Option<u64>,
    /// After restarting the decoder mid-stream, frames up to and including
    /// this timestamp were already shown and must not be replayed.
    shown_through_us: Option<u64>,
}

impl Presentation {
    pub(crate) fn new(start_us: u64) -> Self {
        Self {
            start_us,
            last_us: None,
            shown_through_us: None,
        }
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
}
