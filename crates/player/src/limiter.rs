/// Thins a decoded stream to at most `fps` frames a second for an external
/// consumer (the desktop shell) that cannot present faster.
///
/// Contract: call [`FrameLimiter::accepts`] with each frame's timestamp in
/// presentation order and present only the accepted ones. Spacing is checked
/// at 75% of the target period so timestamp jitter never drops frames from a
/// stream already at the target rate; a timestamp going backwards (loop
/// restart) is always accepted.
pub struct FrameLimiter {
    min_spacing_us: u64,
    last: Option<u64>,
}

impl FrameLimiter {
    /// Creates a limiter for `fps` frames a second (0 is treated as 1).
    pub fn new(fps: u32) -> Self {
        Self {
            min_spacing_us: 750_000 / u64::from(fps.max(1)),
            last: None,
        }
    }

    /// Whether the frame stamped `pts_us` should be presented.
    pub fn accepts(&mut self, pts_us: u64) -> bool {
        let due = self
            .last
            .is_none_or(|last| pts_us < last || pts_us - last >= self.min_spacing_us);
        if due {
            self.last = Some(pts_us);
        }
        due
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sixty_frames_a_second_are_thinned_to_thirty() {
        let mut limiter = FrameLimiter::new(30);
        let kept = (0..60)
            .filter(|frame| limiter.accepts(frame * 1_000_000 / 60))
            .count();
        assert_eq!(kept, 30);
    }

    #[test]
    fn thirty_frames_a_second_keep_every_frame_despite_jitter() {
        let mut limiter = FrameLimiter::new(30);
        let kept = (0..30_u64)
            .filter(|frame| {
                let jitter = frame % 2 * 1_000;
                limiter.accepts((frame * 33_333).saturating_sub(jitter))
            })
            .count();
        assert_eq!(kept, 30);
    }

    #[test]
    fn a_loop_restart_is_not_dropped() {
        let mut limiter = FrameLimiter::new(30);
        assert!(limiter.accepts(2_000_000));
        assert!(limiter.accepts(0));
        assert!(!limiter.accepts(10_000));
    }
}
