//! Presentation timestamps for NVDEC pictures.
//!
//! `shiguredo_nvcodec` hard-codes the CUVID packet timestamp to 0 and
//! hands decoded pictures over without their POC or surface index, so a
//! picture cannot say which packet it came from. Its timestamp is worked
//! out from what is known instead:
//!
//! * NVDEC shows pictures in POC order within a coded video sequence,
//!   and POC order is timestamp order, so the next picture out is the
//!   pending one with the earliest timestamp of the earliest sequence.
//! * A new sequence empties the DPB. The parser only finds a picture's
//!   end when the next one starts (the binding never sets
//!   `CUVID_PKT_ENDOFPICTURE`), so once the picture after a sequence
//!   start has been parsed, nothing older can come out any more.
//! * Once a picture is decoded, no more pictures wait for output than the
//!   SPS bounds: `sps_max_num_reorder_pics` for HEVC, whose DPB bumps
//!   pictures out on it (H.265 C.5.2.2), `max_dec_frame_buffering` for
//!   H.264, whose DPB only bumps when full (H.264 C.4.5.3). Add the one
//!   picture the parser has not finished: anything pending beyond that
//!   was dropped, and the earliest pending one is the one that took its
//!   place.
//!
//! The codec headers never note pictures NVDEC is known not to show
//! (before the first IDR / IRAP picture, RASL pictures of the IRAP that
//! starts decoding, `pic_output_flag` 0). A picture dropped for any other
//! reason, such as a decode error, lends its timestamp to the next picture
//! out; the pictures after it are put right as soon as the pending ones
//! outgrow that bound or their sequence ends.

/// DPB ceiling of both H.264 (`MaxDpbFrames`) and HEVC (`MaxDpbSize`):
/// the most pictures that can wait for output while no SPS bounds it.
const MAX_DPB_SIZE: u32 = 16;

/// Pictures the parser holds besides the DPB: the one whose end it has
/// not seen yet.
const PARSER_LAG: usize = 1;

/// A picture in a packet that NVDEC is expected to show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Picture {
    /// Coded video sequence, counting up; each one empties the DPB.
    pub sequence: u32,
    /// Most pictures left waiting for output once a picture is decoded,
    /// when the SPS bounds it.
    pub max_waiting: Option<u32>,
}

/// A picture's place in output order: sequence by sequence, then
/// timestamp order within one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Slot {
    sequence: u32,
    timestamp_us: u64,
}

#[derive(Debug, Default)]
pub(super) struct Timestamps {
    pending: Vec<Slot>,
    /// Sequence of the latest picture noted.
    latest: Option<u32>,
    /// Sequence of the picture noted before the latest one, which the
    /// parser has finished: older sequences are over.
    finished: Option<u32>,
    max_waiting: Option<u32>,
    /// The last picture out; nothing that sorts before it can follow.
    released: Option<Slot>,
    last_us: u64,
}

impl Timestamps {
    /// Notes a picture fed to NVDEC with its packet's timestamp.
    pub(super) fn push(&mut self, picture: Picture, timestamp_us: u64) {
        self.finished = self.latest;
        self.latest = Some(picture.sequence);
        self.max_waiting = picture.max_waiting;
        let slot = Slot {
            sequence: picture.sequence,
            timestamp_us,
        };
        // Its turn is over: a later picture is out already.
        if self.released.is_some_and(|released| slot < released) {
            return;
        }
        self.pending.push(slot);
    }

    /// The timestamp of the picture NVDEC just showed. A picture nothing
    /// was noted for keeps the previous picture's timestamp.
    pub(super) fn claim(&mut self) -> u64 {
        if let Some(slot) = self.take_earliest() {
            self.released = Some(slot);
            self.last_us = slot.timestamp_us;
        }
        self.last_us
    }

    /// Forgets the pictures NVDEC can no longer show, once a packet's
    /// pictures have been claimed.
    pub(super) fn settle(&mut self) {
        if let Some(finished) = self.finished {
            self.pending.retain(|slot| slot.sequence >= finished);
        }
        let max_waiting = self.max_waiting.unwrap_or(MAX_DPB_SIZE);
        let limit = usize::try_from(max_waiting)
            .unwrap_or(usize::MAX)
            .saturating_add(PARSER_LAG);
        while self.pending.len() > limit {
            self.take_earliest();
        }
    }

    fn take_earliest(&mut self) -> Option<Slot> {
        let (index, _) = self
            .pending
            .iter()
            .enumerate()
            .min_by_key(|(_, slot)| **slot)?;
        Some(self.pending.swap_remove(index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One packet: the picture it carries, if NVDEC is expected to show
    /// one, and how many pictures NVDEC shows while parsing it.
    struct Step {
        picture: Option<(u32, u64)>,
        shown: usize,
    }

    const fn step(sequence: u32, timestamp_us: u64, shown: usize) -> Step {
        Step {
            picture: Some((sequence, timestamp_us)),
            shown,
        }
    }

    /// A packet whose picture is known not to be shown.
    const fn dropped(shown: usize) -> Step {
        Step {
            picture: None,
            shown,
        }
    }

    /// Feeds the steps as `NvdecDecoder::decode` does, then flushes the
    /// `remaining` pictures, returning every timestamp handed out. Each
    /// step's `shown` follows NVDEC: a picture is decoded while the next
    /// packet is parsed, then bumped out in display order.
    fn run(max_waiting: Option<u32>, steps: &[Step], remaining: usize) -> Vec<u64> {
        let mut timestamps = Timestamps::default();
        let mut out = Vec::new();
        for step in steps {
            if let Some((sequence, timestamp_us)) = step.picture {
                timestamps.push(
                    Picture {
                        sequence,
                        max_waiting,
                    },
                    timestamp_us,
                );
            }
            out.extend((0..step.shown).map(|_| timestamps.claim()));
            timestamps.settle();
        }
        out.extend((0..remaining).map(|_| timestamps.claim()));
        out
    }

    #[test]
    fn reordered_pictures_take_their_timestamps_in_display_order() {
        // I0 P3 B1 B2 P6 B4 B5 in decode order; one picture may wait.
        let steps = [
            step(1, 0, 0),
            step(1, 3, 0),
            step(1, 1, 1),
            step(1, 2, 1),
            step(1, 6, 1),
            step(1, 4, 1),
            step(1, 5, 1),
        ];
        assert_eq!(run(Some(1), &steps, 2), [0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn pictures_dropped_before_the_first_idr_shift_nothing() {
        let steps = [
            dropped(0),
            dropped(0),
            step(1, 2, 0),
            step(1, 3, 1),
            step(1, 4, 1),
        ];
        assert_eq!(run(Some(0), &steps, 1), [2, 3, 4]);
    }

    #[test]
    fn skipped_rasl_pictures_shift_nothing() {
        // CRA 8 starts decoding: its RASL pictures 6 and 7 are skipped,
        // the trailing pictures 9 and 10 follow.
        let steps = [
            step(1, 8, 0),
            dropped(1),
            dropped(0),
            step(1, 9, 0),
            step(1, 10, 1),
        ];
        assert_eq!(run(Some(0), &steps, 1), [8, 9, 10]);
    }

    #[test]
    fn a_picture_lost_to_a_decode_error_shifts_nothing_after_it() {
        // Picture 1 never comes out: once picture 2 is pending too, more
        // pictures wait than the SPS allows and the earliest is forgotten.
        let steps = [
            step(1, 0, 0),
            step(1, 1, 1),
            step(1, 2, 0),
            step(1, 3, 1),
            step(1, 4, 1),
            step(1, 5, 1),
        ];
        assert_eq!(run(Some(0), &steps, 1), [0, 2, 3, 4, 5]);
    }

    #[test]
    fn pictures_of_a_finished_sequence_are_forgotten() {
        // Picture 1 is lost just before an IDR NVDEC holds back; with no
        // SPS bound only the end of its sequence gives it away.
        let steps = [
            step(1, 0, 0),
            step(1, 1, 1),
            step(2, 100, 0),
            step(2, 101, 0),
            step(2, 102, 1),
        ];
        assert_eq!(run(None, &steps, 2), [0, 100, 101, 102]);
    }

    #[test]
    fn an_earlier_sequence_comes_out_before_a_later_one_restarting_its_clock() {
        let steps = [
            step(1, 1_000, 0),
            step(1, 1_033, 1),
            step(2, 0, 1),
            step(2, 33, 1),
        ];
        assert_eq!(run(None, &steps, 1), [1_000, 1_033, 0, 33]);
    }

    #[test]
    fn a_flush_hands_out_the_rest_in_display_order() {
        let steps = [step(1, 0, 0), step(1, 3, 0), step(1, 1, 0), step(1, 2, 0)];
        assert_eq!(run(Some(4), &steps, 4), [0, 1, 2, 3]);
    }

    #[test]
    fn pending_pictures_beyond_the_bound_are_forgotten_earliest_first() {
        let steps: Vec<Step> = (0..10).map(|pts| step(1, pts, 0)).collect();
        assert_eq!(run(Some(4), &steps, 5), [5, 6, 7, 8, 9]);
    }

    #[test]
    fn a_picture_noted_after_its_turn_is_ignored_and_running_out_repeats_the_last() {
        let steps = [step(1, 5, 0), step(1, 6, 1), step(1, 4, 1)];
        assert_eq!(run(None, &steps, 1), [5, 6, 6]);
    }
}
