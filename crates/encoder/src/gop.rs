//! IPPP… GOP bookkeeping shared by the VA-API H.264 and HEVC encoders.
//!
//! Every picture is a reference picture and IDRs can land anywhere —
//! at the configured interval, on [`GopState::request_idr`] (the HLS
//! segmenter forces one per segment) or after a failed encode. The
//! per-picture numbering therefore derives from the distance to the
//! *last emitted* IDR, never from multiples of the interval.
//!
//! State only advances in [`GopState::commit`], after the driver
//! accepted the picture, so a failed submission never consumes an IDR
//! request or a `frame_num`.

/// Numbering for one picture, computed by [`GopState::plan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PicturePlan {
    pub is_idr: bool,
    /// Pictures since the last IDR (0 for the IDR itself).
    pub position: u32,
}

impl PicturePlan {
    /// H.264 `frame_num` (§7.4.3): +1 per reference picture since the
    /// IDR, modulo `MaxFrameNum = 2^log2_max_frame_num`.
    pub(crate) fn frame_num(self, log2_max_frame_num: u32) -> u32 {
        self.position & ((1 << log2_max_frame_num) - 1)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GopState {
    keyframe_interval: u32,
    /// Position the next picture would take if it isn't an IDR;
    /// `None` until the first IDR has been committed.
    next_position: Option<u32>,
    force_idr: bool,
}

impl GopState {
    pub(crate) fn new(keyframe_interval: u32) -> Self {
        Self {
            keyframe_interval: keyframe_interval.max(1),
            next_position: None,
            force_idr: false,
        }
    }

    /// Ask for the next picture to be an IDR.
    pub(crate) fn request_idr(&mut self) {
        self.force_idr = true;
    }

    /// Plan the next picture without changing any state.
    #[must_use]
    pub(crate) fn plan(&self) -> PicturePlan {
        match self.next_position {
            Some(position) if !self.force_idr && position < self.keyframe_interval => PicturePlan {
                is_idr: false,
                position,
            },
            _ => PicturePlan {
                is_idr: true,
                position: 0,
            },
        }
    }

    /// Record that `plan` was encoded successfully.
    pub(crate) fn commit(&mut self, plan: PicturePlan) {
        self.next_position = Some(plan.position + 1);
        if plan.is_idr {
            self.force_idr = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(gop: &mut GopState, n: usize) -> Vec<PicturePlan> {
        (0..n)
            .map(|_| {
                let p = gop.plan();
                gop.commit(p);
                p
            })
            .collect()
    }

    #[test]
    fn natural_interval() {
        let mut gop = GopState::new(3);
        let idr: Vec<bool> = run(&mut gop, 7).iter().map(|p| p.is_idr).collect();
        assert_eq!(idr, [true, false, false, true, false, false, true]);
    }

    /// A forced IDR mid-GOP restarts `frame_num` at 0 and the next
    /// natural IDR is a full interval after it, not at the old multiple.
    #[test]
    fn forced_idr_restarts_numbering() {
        let mut gop = GopState::new(4);
        run(&mut gop, 2);
        gop.request_idr();
        let plans = run(&mut gop, 6);
        let frame_nums: Vec<u32> = plans.iter().map(|p| p.frame_num(8)).collect();
        assert_eq!(frame_nums, [0, 1, 2, 3, 0, 1]);
        assert!(plans[0].is_idr && plans[4].is_idr);
        assert!(plans[1..4].iter().all(|p| !p.is_idr));
    }

    /// Planning is pure: an uncommitted (failed) picture leaves the IDR
    /// request and numbering intact.
    #[test]
    fn failed_picture_keeps_state() {
        let mut gop = GopState::new(10);
        run(&mut gop, 3);
        gop.request_idr();
        let failed = gop.plan();
        assert!(failed.is_idr);
        let retry = gop.plan();
        assert_eq!(failed, retry);
        gop.commit(retry);
        assert_eq!(gop.plan().position, 1);
    }

    #[test]
    fn frame_num_wraps_at_max_frame_num() {
        let mut gop = GopState::new(1000);
        let plans = run(&mut gop, 258);
        assert_eq!(plans[255].frame_num(8), 255);
        assert_eq!(plans[256].frame_num(8), 0);
        assert_eq!(plans[257].frame_num(8), 1);
    }
}
