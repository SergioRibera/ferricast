use super::slice::{Mmco, PocKind, RefMod, SliceHeader, SliceType, Sps};

#[derive(Debug, Clone)]
pub(super) struct Frame {
    pub handle: usize,
    pub frame_num: u32,
    pub top_poc: i32,
    pub bottom_poc: i32,
    pub long_term_idx: Option<u32>,
    pub referenced: bool,
    pub awaiting_output: bool,
    pub timestamp_us: u64,
}

impl Frame {
    pub(super) fn poc(&self) -> i32 {
        self.top_poc.min(self.bottom_poc)
    }

    fn is_short(&self) -> bool {
        self.referenced && self.long_term_idx.is_none()
    }

    fn is_long(&self) -> bool {
        self.referenced && self.long_term_idx.is_some()
    }

    fn unmark(&mut self) {
        self.referenced = false;
        self.long_term_idx = None;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Output {
    pub handle: usize,
    pub timestamp_us: u64,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Released {
    pub output: Vec<Output>,
    pub free: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Current {
    pub frame_num: u32,
    pub top_poc: i32,
    pub bottom_poc: i32,
    poc_msb: i32,
    frame_num_offset: u32,
}

pub(super) struct Dpb {
    frames: Vec<Frame>,
    max_frame_num: u32,
    max_num_ref_frames: u32,
    reorder: u32,
    poc: PocKind,
    prev_poc_msb: i32,
    prev_poc_lsb: u32,
    prev_frame_num: u32,
    prev_frame_num_offset: u32,
    max_long_term_idx: Option<u32>,
}

pub(super) fn dpb_size(sps: &Sps) -> u32 {
    if let Some(size) = sps.max_dec_frame_buffering {
        return size.clamp(sps.max_num_ref_frames.max(1), 16);
    }
    let level_limit = match sps.level_idc {
        9 | 10 => 396,
        11 if sps.constraint_set3 && matches!(sps.profile_idc, 66 | 77 | 88) => 396,
        11 => 900,
        12 | 13 | 20 => 2376,
        21 => 4752,
        22 | 30 => 8100,
        31 => 18000,
        32 => 20480,
        40 | 41 => 32768,
        42 => 34816,
        50 => 110_400,
        51 | 52 => 184_320,
        _ => 696_320,
    };
    (level_limit / (sps.width_mbs * sps.height_mbs).max(1)).clamp(sps.max_num_ref_frames.max(1), 16)
}

impl Dpb {
    pub(super) fn new(sps: &Sps, low_latency: bool) -> Self {
        let reorder = if low_latency {
            0
        } else {
            sps.num_reorder_frames
                .unwrap_or_else(|| dpb_size(sps))
                .min(dpb_size(sps))
        };
        Self {
            frames: Vec::new(),
            max_frame_num: 1 << sps.log2_max_frame_num,
            max_num_ref_frames: sps.max_num_ref_frames,
            reorder,
            poc: sps.poc.clone(),
            prev_poc_msb: 0,
            prev_poc_lsb: 0,
            prev_frame_num: 0,
            prev_frame_num_offset: 0,
            max_long_term_idx: None,
        }
    }

    pub(super) fn capacity(&self) -> usize {
        (self.max_num_ref_frames.max(1) + self.reorder + 2) as usize
    }

    pub(super) fn frame(&self, index: usize) -> &Frame {
        &self.frames[index]
    }

    pub(super) fn reference_frames(&self) -> impl Iterator<Item = &Frame> {
        self.frames.iter().filter(|frame| frame.referenced)
    }

    pub(super) fn start(&mut self, header: &SliceHeader) -> (Current, Released) {
        let mut released = Released::default();
        if header.idr {
            self.flush_into(&mut released);
        }
        let mut current = Current {
            frame_num: header.frame_num,
            top_poc: 0,
            bottom_poc: 0,
            poc_msb: 0,
            frame_num_offset: 0,
        };
        match &self.poc {
            PocKind::Lsb { log2_max_lsb } => {
                let max_lsb = 1i32 << log2_max_lsb;
                let (prev_msb, prev_lsb) = if header.idr {
                    (0, 0)
                } else {
                    (self.prev_poc_msb, self.prev_poc_lsb as i32)
                };
                let lsb = header.pic_order_cnt_lsb as i32;
                current.poc_msb = if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
                    prev_msb + max_lsb
                } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
                    prev_msb - max_lsb
                } else {
                    prev_msb
                };
                current.top_poc = current.poc_msb + lsb;
                current.bottom_poc = current.top_poc + header.delta_pic_order_cnt_bottom;
            }
            PocKind::Cycle {
                offset_for_non_ref_pic,
                offset_for_top_to_bottom,
                offsets_for_ref_frame,
                ..
            } => {
                current.frame_num_offset = self.frame_num_offset(header);
                let cycle_len = offsets_for_ref_frame.len() as i32;
                let mut absolute = if cycle_len != 0 {
                    (current.frame_num_offset + header.frame_num) as i32
                } else {
                    0
                };
                if header.nal_ref_idc == 0 && absolute > 0 {
                    absolute -= 1;
                }
                let mut expected = 0;
                if absolute > 0 {
                    let cycle = (absolute - 1) / cycle_len;
                    let in_cycle = ((absolute - 1) % cycle_len) as usize;
                    let per_cycle: i32 = offsets_for_ref_frame.iter().sum();
                    expected =
                        cycle * per_cycle + offsets_for_ref_frame[..=in_cycle].iter().sum::<i32>();
                }
                if header.nal_ref_idc == 0 {
                    expected += offset_for_non_ref_pic;
                }
                current.top_poc = expected + header.delta_pic_order_cnt[0];
                current.bottom_poc =
                    current.top_poc + offset_for_top_to_bottom + header.delta_pic_order_cnt[1];
            }
            PocKind::FrameNum => {
                current.frame_num_offset = self.frame_num_offset(header);
                let base = 2 * (current.frame_num_offset + header.frame_num) as i32;
                let poc = if header.idr {
                    0
                } else if header.nal_ref_idc == 0 {
                    base - 1
                } else {
                    base
                };
                current.top_poc = poc;
                current.bottom_poc = poc;
            }
        }
        (current, released)
    }

    fn frame_num_offset(&self, header: &SliceHeader) -> u32 {
        if header.idr {
            0
        } else if self.prev_frame_num > header.frame_num {
            self.prev_frame_num_offset + self.max_frame_num
        } else {
            self.prev_frame_num_offset
        }
    }

    pub(super) fn finish(
        &mut self,
        current: Current,
        header: &SliceHeader,
        handle: usize,
        timestamp_us: u64,
    ) -> Released {
        let mut released = Released::default();
        let mut frame = Frame {
            handle,
            frame_num: header.frame_num,
            top_poc: current.top_poc,
            bottom_poc: current.bottom_poc,
            long_term_idx: None,
            referenced: header.nal_ref_idc != 0,
            awaiting_output: true,
            timestamp_us,
        };
        let cleared = header.nal_ref_idc != 0
            && header
                .marking
                .adaptive
                .as_ref()
                .is_some_and(|operations| operations.contains(&Mmco::ClearAll));
        if header.nal_ref_idc != 0 {
            self.mark(header, &mut frame);
        }
        if cleared {
            self.bump_all(&mut released);
            let base = frame.poc();
            frame.top_poc -= base;
            frame.bottom_poc -= base;
            frame.frame_num = 0;
        }
        if header.nal_ref_idc != 0 {
            (self.prev_poc_msb, self.prev_poc_lsb) = if cleared {
                (0, frame.top_poc as u32)
            } else {
                (current.poc_msb, header.pic_order_cnt_lsb)
            };
        }
        (self.prev_frame_num, self.prev_frame_num_offset) = if cleared {
            (0, 0)
        } else {
            (header.frame_num, current.frame_num_offset)
        };
        self.frames.push(frame);
        while self.awaiting() > self.reorder as usize {
            self.bump_one(&mut released);
        }
        self.collect_free(&mut released);
        released
    }

    pub(super) fn flush(&mut self) -> Released {
        let mut released = Released::default();
        self.flush_into(&mut released);
        self.prev_poc_msb = 0;
        self.prev_poc_lsb = 0;
        self.prev_frame_num = 0;
        self.prev_frame_num_offset = 0;
        released
    }

    pub(super) fn make_room(&mut self) -> Released {
        let mut released = Released::default();
        self.bump_one(&mut released);
        self.collect_free(&mut released);
        released
    }

    fn flush_into(&mut self, released: &mut Released) {
        self.bump_all(released);
        for frame in &mut self.frames {
            frame.unmark();
        }
        self.max_long_term_idx = None;
        self.collect_free(released);
    }

    fn awaiting(&self) -> usize {
        self.frames
            .iter()
            .filter(|frame| frame.awaiting_output)
            .count()
    }

    fn bump_all(&mut self, released: &mut Released) {
        while self.awaiting() > 0 {
            self.bump_one(released);
        }
    }

    fn bump_one(&mut self, released: &mut Released) {
        let next = self
            .frames
            .iter_mut()
            .filter(|frame| frame.awaiting_output)
            .min_by_key(|frame| frame.poc());
        if let Some(frame) = next {
            frame.awaiting_output = false;
            released.output.push(Output {
                handle: frame.handle,
                timestamp_us: frame.timestamp_us,
            });
        }
    }

    fn collect_free(&mut self, released: &mut Released) {
        self.frames.retain(|frame| {
            let keep = frame.referenced || frame.awaiting_output;
            if !keep {
                released.free.push(frame.handle);
            }
            keep
        });
    }

    fn frame_num_wrap(&self, frame_num: u32, current_frame_num: u32) -> i32 {
        if frame_num > current_frame_num {
            frame_num as i32 - self.max_frame_num as i32
        } else {
            frame_num as i32
        }
    }

    fn mark(&mut self, header: &SliceHeader, current: &mut Frame) {
        if header.idr {
            if header.marking.long_term_reference {
                current.long_term_idx = Some(0);
                self.max_long_term_idx = Some(0);
            } else {
                self.max_long_term_idx = None;
            }
            return;
        }
        let Some(operations) = &header.marking.adaptive else {
            self.slide(header.frame_num);
            return;
        };
        for operation in operations {
            self.apply(*operation, header.frame_num, current);
        }
    }

    fn slide(&mut self, current_frame_num: u32) {
        let referenced = self.frames.iter().filter(|frame| frame.referenced).count() as u32;
        if referenced < self.max_num_ref_frames.max(1) {
            return;
        }
        let oldest = self
            .frames
            .iter()
            .enumerate()
            .filter(|(_, frame)| frame.is_short())
            .min_by_key(|(_, frame)| self.frame_num_wrap(frame.frame_num, current_frame_num))
            .map(|(index, _)| index);
        if let Some(index) = oldest {
            self.frames[index].unmark();
        }
    }

    fn short_term(&self, pic_num: i32, current_frame_num: u32) -> Option<usize> {
        self.frames.iter().position(|frame| {
            frame.is_short() && self.frame_num_wrap(frame.frame_num, current_frame_num) == pic_num
        })
    }

    fn free_long_term_idx(&mut self, idx: u32) {
        for frame in &mut self.frames {
            if frame.long_term_idx == Some(idx) {
                frame.unmark();
            }
        }
    }

    fn apply(&mut self, operation: Mmco, current_frame_num: u32, current: &mut Frame) {
        let pic_num = |diff_minus1: u32| current_frame_num as i32 - (diff_minus1 as i32 + 1);
        match operation {
            Mmco::UnmarkShort { diff_minus1 } => {
                if let Some(index) = self.short_term(pic_num(diff_minus1), current_frame_num) {
                    self.frames[index].unmark();
                }
            }
            Mmco::UnmarkLong { long_term_pic_num } => {
                if let Some(frame) = self
                    .frames
                    .iter_mut()
                    .find(|frame| frame.is_long() && frame.long_term_idx == Some(long_term_pic_num))
                {
                    frame.unmark();
                }
            }
            Mmco::ShortToLong {
                diff_minus1,
                long_term_idx,
            } => {
                let target = self.short_term(pic_num(diff_minus1), current_frame_num);
                self.free_long_term_idx(long_term_idx);
                if let Some(index) = target {
                    self.frames[index].long_term_idx = Some(long_term_idx);
                }
            }
            Mmco::MaxLongTermIdxPlus1(plus1) => {
                self.max_long_term_idx = plus1.checked_sub(1);
                let limit = self.max_long_term_idx;
                for frame in &mut self.frames {
                    if frame
                        .long_term_idx
                        .is_some_and(|idx| limit.is_none_or(|limit| idx > limit))
                    {
                        frame.unmark();
                    }
                }
            }
            Mmco::ClearAll => {
                for frame in &mut self.frames {
                    frame.unmark();
                }
                self.max_long_term_idx = None;
            }
            Mmco::CurrentToLong { long_term_idx } => {
                self.free_long_term_idx(long_term_idx);
                current.long_term_idx = Some(long_term_idx);
            }
        }
    }

    pub(super) fn ref_lists(
        &self,
        header: &SliceHeader,
        current: &Current,
    ) -> [Vec<Option<usize>>; 2] {
        let mut lists: [Vec<Option<usize>>; 2] = Default::default();
        if !header.slice_type.is_inter() {
            return lists;
        }
        let current_poc = current.top_poc.min(current.bottom_poc);
        let mut short: Vec<usize> = (0..self.frames.len())
            .filter(|&index| self.frames[index].is_short())
            .collect();
        let mut long: Vec<usize> = (0..self.frames.len())
            .filter(|&index| self.frames[index].is_long())
            .collect();
        long.sort_by_key(|&index| self.frames[index].long_term_idx);
        let mut initial: [Vec<usize>; 2] = Default::default();
        if header.slice_type == SliceType::B {
            let mut before: Vec<usize> = short
                .iter()
                .copied()
                .filter(|&index| self.frames[index].poc() < current_poc)
                .collect();
            let mut after: Vec<usize> = short
                .iter()
                .copied()
                .filter(|&index| self.frames[index].poc() > current_poc)
                .collect();
            before.sort_by_key(|&index| std::cmp::Reverse(self.frames[index].poc()));
            after.sort_by_key(|&index| self.frames[index].poc());
            initial[0] = [before.clone(), after.clone(), long.clone()].concat();
            initial[1] = [after, before, long].concat();
            if initial[1].len() > 1 && initial[0] == initial[1] {
                initial[1].swap(0, 1);
            }
        } else {
            short.sort_by_key(|&index| {
                std::cmp::Reverse(
                    self.frame_num_wrap(self.frames[index].frame_num, current.frame_num),
                )
            });
            initial[0] = [short, long].concat();
        }
        let list_count = if header.slice_type == SliceType::B {
            2
        } else {
            1
        };
        for list in 0..list_count {
            let active = header.num_ref_idx_active[list] as usize;
            lists[list] = initial[list].iter().map(|&index| Some(index)).collect();
            lists[list].resize(active, None);
            self.modify(&mut lists[list], &header.ref_mods[list], current.frame_num);
        }
        lists
    }

    fn modify(&self, list: &mut Vec<Option<usize>>, operations: &[RefMod], current_frame_num: u32) {
        let active = list.len();
        if active == 0 {
            return;
        }
        let max_pic_num = self.max_frame_num as i32;
        let current_pic_num = current_frame_num as i32;
        let mut predicted = current_pic_num;
        let mut position = 0;
        list.push(None);
        for operation in operations {
            let target = match *operation {
                RefMod::ShortMinus(abs_diff_minus1) | RefMod::ShortPlus(abs_diff_minus1) => {
                    let diff = abs_diff_minus1 as i32 + 1;
                    let no_wrap = if matches!(operation, RefMod::ShortMinus(_)) {
                        if predicted - diff < 0 {
                            predicted - diff + max_pic_num
                        } else {
                            predicted - diff
                        }
                    } else if predicted + diff >= max_pic_num {
                        predicted + diff - max_pic_num
                    } else {
                        predicted + diff
                    };
                    predicted = no_wrap;
                    let pic_num = if no_wrap > current_pic_num {
                        no_wrap - max_pic_num
                    } else {
                        no_wrap
                    };
                    self.short_term(pic_num, current_frame_num)
                }
                RefMod::Long(long_term_pic_num) => self.frames.iter().position(|frame| {
                    frame.is_long() && frame.long_term_idx == Some(long_term_pic_num)
                }),
            };
            if position >= active {
                break;
            }
            for slot in (position + 1..=active).rev() {
                list[slot] = list[slot - 1];
            }
            list[position] = target;
            position += 1;
            let mut kept = position;
            for slot in position..=active {
                if list[slot].is_none() || list[slot] != target {
                    list[kept] = list[slot];
                    kept += 1;
                }
            }
        }
        list.truncate(active);
    }
}

#[cfg(test)]
mod tests {
    use super::super::slice::tests::sps;
    use super::*;

    fn header(frame_num: u32, lsb: u32, nal_ref_idc: u8) -> SliceHeader {
        SliceHeader {
            nal_ref_idc,
            frame_num,
            pic_order_cnt_lsb: lsb,
            slice_type: SliceType::P,
            num_ref_idx_active: [4, 0],
            ..SliceHeader::default()
        }
    }

    fn idr() -> SliceHeader {
        SliceHeader {
            idr: true,
            nal_ref_idc: 3,
            ..header(0, 0, 3)
        }
    }

    fn decode(dpb: &mut Dpb, header: &SliceHeader, handle: usize) -> (Current, Released) {
        let (current, mut released) = dpb.start(header);
        let finished = dpb.finish(current, header, handle, handle as u64 * 1000);
        released.output.extend(finished.output);
        released.free.extend(finished.free);
        (current, released)
    }

    fn handles(outputs: &[Output]) -> Vec<usize> {
        outputs.iter().map(|output| output.handle).collect()
    }

    fn lsb_sps(log2_max_lsb: u32) -> Sps {
        Sps {
            poc: PocKind::Lsb { log2_max_lsb },
            ..sps()
        }
    }

    #[test]
    fn poc_type_0_carries_the_msb_over_the_lsb_wrap() {
        let mut dpb = Dpb::new(&lsb_sps(4), false);
        let mut pocs = Vec::new();
        for (index, lsb) in [0, 6, 12, 2, 8].into_iter().enumerate() {
            let header = if index == 0 {
                idr()
            } else {
                header(index as u32, lsb, 1)
            };
            let (current, _) = decode(&mut dpb, &header, index);
            pocs.push(current.top_poc);
        }
        assert_eq!(pocs, [0, 6, 12, 18, 24]);
    }

    #[test]
    fn poc_type_0_goes_backwards_for_a_picture_displayed_before_the_previous_one() {
        let mut dpb = Dpb::new(&lsb_sps(4), false);
        decode(&mut dpb, &idr(), 0);
        decode(&mut dpb, &header(1, 14, 1), 1);
        let (current, _) = decode(&mut dpb, &header(2, 12, 0), 2);
        assert_eq!(current.top_poc, -4);
    }

    #[test]
    fn poc_type_2_follows_frame_num_and_wraps_its_offset() {
        let sps = Sps {
            poc: PocKind::FrameNum,
            log2_max_frame_num: 2,
            ..sps()
        };
        let mut dpb = Dpb::new(&sps, false);
        let mut pocs = Vec::new();
        for frame_num in [0, 1, 2, 3, 0, 1] {
            let header = if pocs.is_empty() {
                idr()
            } else {
                header(frame_num, 0, 1)
            };
            let (current, _) = decode(&mut dpb, &header, pocs.len());
            pocs.push(current.top_poc);
        }
        assert_eq!(pocs, [0, 2, 4, 6, 8, 10]);
    }

    #[test]
    fn poc_type_1_adds_the_cycle_offsets_and_the_non_reference_offset() {
        let sps = Sps {
            poc: PocKind::Cycle {
                delta_always_zero: true,
                offset_for_non_ref_pic: -1,
                offset_for_top_to_bottom: 0,
                offsets_for_ref_frame: vec![2, 4],
            },
            ..sps()
        };
        let mut dpb = Dpb::new(&sps, false);
        let (first, _) = decode(&mut dpb, &idr(), 0);
        let (second, _) = decode(&mut dpb, &header(1, 0, 1), 1);
        let (third, _) = decode(&mut dpb, &header(2, 0, 1), 2);
        let (fourth, _) = decode(&mut dpb, &header(3, 0, 0), 3);
        assert_eq!(
            [first.top_poc, second.top_poc, third.top_poc, fourth.top_poc],
            [0, 2, 6, 5]
        );
    }

    #[test]
    fn pictures_come_out_in_display_order_through_a_reorder_depth_of_two() {
        let mut dpb = Dpb::new(&lsb_sps(8), false);
        let mut out = Vec::new();
        for (index, (lsb, reference)) in [(0, 3), (6, 2), (2, 0), (4, 0)].into_iter().enumerate() {
            let header = if index == 0 {
                idr()
            } else {
                header(index as u32 % 2 + 1, lsb, reference)
            };
            let (_, released) = decode(&mut dpb, &header, index);
            out.extend(handles(&released.output));
        }
        out.extend(handles(&dpb.flush().output));
        assert_eq!(out, [0, 2, 3, 1]);
    }

    #[test]
    fn output_timestamps_follow_their_pictures() {
        let mut dpb = Dpb::new(&lsb_sps(8), false);
        decode(&mut dpb, &idr(), 0);
        decode(&mut dpb, &header(1, 4, 1), 1);
        let outputs = dpb.flush().output;
        assert_eq!(
            outputs,
            [
                Output {
                    handle: 0,
                    timestamp_us: 0
                },
                Output {
                    handle: 1,
                    timestamp_us: 1000
                }
            ]
        );
    }

    #[test]
    fn low_latency_outputs_every_picture_as_it_is_decoded() {
        let mut dpb = Dpb::new(&lsb_sps(8), true);
        let (_, first) = decode(&mut dpb, &idr(), 0);
        let (_, second) = decode(&mut dpb, &header(1, 2, 1), 1);
        assert_eq!(handles(&first.output), [0]);
        assert_eq!(handles(&second.output), [1]);
    }

    #[test]
    fn an_idr_outputs_every_waiting_picture_and_frees_every_surface() {
        let mut dpb = Dpb::new(&lsb_sps(8), false);
        decode(&mut dpb, &idr(), 0);
        decode(&mut dpb, &header(1, 4, 1), 1);
        decode(&mut dpb, &header(2, 2, 1), 2);
        let (_, released) = dpb.start(&idr());
        assert_eq!(handles(&released.output), [2, 1]);
        let mut free = released.free;
        free.sort_unstable();
        assert_eq!(free, [0, 1, 2]);
        assert_eq!(dpb.reference_frames().count(), 0);
    }

    #[test]
    fn the_sliding_window_drops_the_smallest_frame_num_wrap_across_the_wrap() {
        let sps = Sps {
            log2_max_frame_num: 4,
            max_num_ref_frames: 3,
            num_reorder_frames: Some(0),
            ..sps()
        };
        let mut dpb = Dpb::new(&sps, true);
        for (handle, frame_num) in [14, 15, 0].into_iter().enumerate() {
            let header = SliceHeader {
                idr: handle == 0,
                ..header(frame_num, handle as u32 * 2, 1)
            };
            decode(&mut dpb, &header, handle);
        }
        assert_eq!(dpb.reference_frames().count(), 3);
        let (_, released) = decode(&mut dpb, &header(1, 6, 1), 3);
        let mut kept: Vec<u32> = dpb.reference_frames().map(|f| f.frame_num).collect();
        kept.sort_unstable();
        assert_eq!(kept, [0, 1, 15]);
        assert_eq!(released.free, [0]);
    }

    fn filled(count: u32) -> Dpb {
        let sps = Sps {
            max_num_ref_frames: 4,
            ..lsb_sps(8)
        };
        let mut dpb = Dpb::new(&sps, true);
        decode(&mut dpb, &idr(), 0);
        for index in 1..count {
            decode(&mut dpb, &header(index, index * 2, 1), index as usize);
        }
        dpb
    }

    fn list_frame_nums(dpb: &Dpb, list: &[Option<usize>]) -> Vec<Option<u32>> {
        list.iter()
            .map(|entry| entry.map(|index| dpb.frame(index).frame_num))
            .collect()
    }

    #[test]
    fn a_p_list_starts_with_the_latest_short_term_picture() {
        let dpb = filled(4);
        let probe = header(4, 8, 1);
        let current = Current {
            frame_num: 4,
            top_poc: 8,
            bottom_poc: 8,
            poc_msb: 0,
            frame_num_offset: 0,
        };
        let lists = dpb.ref_lists(&probe, &current);
        assert_eq!(
            list_frame_nums(&dpb, &lists[0]),
            [Some(3), Some(2), Some(1), Some(0)]
        );
        assert!(lists[1].is_empty());
    }

    #[test]
    fn a_p_list_is_cut_to_the_active_count_and_padded_with_nothing() {
        let dpb = filled(2);
        let mut probe = header(2, 4, 1);
        probe.num_ref_idx_active = [3, 0];
        let current = Current {
            frame_num: 2,
            top_poc: 4,
            bottom_poc: 4,
            poc_msb: 0,
            frame_num_offset: 0,
        };
        let lists = dpb.ref_lists(&probe, &current);
        assert_eq!(list_frame_nums(&dpb, &lists[0]), [Some(1), Some(0), None]);
        probe.num_ref_idx_active = [1, 0];
        let lists = dpb.ref_lists(&probe, &current);
        assert_eq!(list_frame_nums(&dpb, &lists[0]), [Some(1)]);
    }

    fn b_probe(poc: i32, active: [u32; 2]) -> (SliceHeader, Current) {
        let probe = SliceHeader {
            slice_type: SliceType::B,
            num_ref_idx_active: active,
            ..header(9, 0, 0)
        };
        let current = Current {
            frame_num: 9,
            top_poc: poc,
            bottom_poc: poc,
            poc_msb: 0,
            frame_num_offset: 0,
        };
        (probe, current)
    }

    #[test]
    fn b_lists_split_the_references_around_the_current_display_position() {
        let mut dpb = Dpb::new(&lsb_sps(8), true);
        decode(&mut dpb, &idr(), 0);
        decode(&mut dpb, &header(1, 16, 1), 1);
        let (probe, current) = b_probe(8, [2, 2]);
        let lists = dpb.ref_lists(&probe, &current);
        assert_eq!(list_frame_nums(&dpb, &lists[0]), [Some(0), Some(1)]);
        assert_eq!(list_frame_nums(&dpb, &lists[1]), [Some(1), Some(0)]);
    }

    #[test]
    fn b_list_one_swaps_its_first_two_entries_when_it_equals_list_zero() {
        let mut dpb = Dpb::new(&lsb_sps(8), true);
        decode(&mut dpb, &idr(), 0);
        decode(&mut dpb, &header(1, 4, 1), 1);
        let (probe, current) = b_probe(12, [2, 2]);
        let lists = dpb.ref_lists(&probe, &current);
        assert_eq!(list_frame_nums(&dpb, &lists[0]), [Some(1), Some(0)]);
        assert_eq!(list_frame_nums(&dpb, &lists[1]), [Some(0), Some(1)]);
    }

    #[test]
    fn a_list_modification_moves_the_requested_picture_to_the_front() {
        let dpb = filled(4);
        let mut probe = header(4, 8, 1);
        probe.num_ref_idx_active = [3, 0];
        probe.ref_mods[0] = vec![RefMod::ShortMinus(2)];
        let current = Current {
            frame_num: 4,
            top_poc: 8,
            bottom_poc: 8,
            poc_msb: 0,
            frame_num_offset: 0,
        };
        let lists = dpb.ref_lists(&probe, &current);
        assert_eq!(
            list_frame_nums(&dpb, &lists[0]),
            [Some(1), Some(3), Some(2)]
        );
    }

    #[test]
    fn long_term_pictures_follow_the_short_term_ones_and_can_be_pulled_forward() {
        let mut dpb = filled(3);
        let mut convert = header(3, 6, 1);
        convert.marking.adaptive = Some(vec![Mmco::ShortToLong {
            diff_minus1: 2,
            long_term_idx: 0,
        }]);
        decode(&mut dpb, &convert, 3);
        let mut probe = header(4, 8, 1);
        probe.num_ref_idx_active = [4, 0];
        let current = Current {
            frame_num: 4,
            top_poc: 8,
            bottom_poc: 8,
            poc_msb: 0,
            frame_num_offset: 0,
        };
        let lists = dpb.ref_lists(&probe, &current);
        assert_eq!(
            list_frame_nums(&dpb, &lists[0]),
            [Some(3), Some(2), Some(1), Some(0)]
        );
        probe.ref_mods[0] = vec![RefMod::Long(0)];
        let lists = dpb.ref_lists(&probe, &current);
        assert_eq!(
            list_frame_nums(&dpb, &lists[0]),
            [Some(0), Some(3), Some(2), Some(1)]
        );
    }

    #[test]
    fn marking_operations_unmark_convert_and_clear() {
        let mut dpb = filled(3);
        let mut unmark = header(3, 6, 1);
        unmark.marking.adaptive = Some(vec![Mmco::UnmarkShort { diff_minus1: 1 }]);
        let (_, released) = decode(&mut dpb, &unmark, 3);
        let mut kept: Vec<u32> = dpb.reference_frames().map(|f| f.frame_num).collect();
        kept.sort_unstable();
        assert_eq!(kept, [0, 2, 3]);
        assert!(released.free.contains(&1));
        let mut clear = header(4, 8, 1);
        clear.marking.adaptive = Some(vec![Mmco::ClearAll]);
        let (_, released) = decode(&mut dpb, &clear, 4);
        let kept: Vec<_> = dpb
            .reference_frames()
            .map(|f| (f.frame_num, f.poc()))
            .collect();
        assert_eq!(kept, [(0, 0)]);
        assert_eq!(released.free.len(), 3);
    }

    #[test]
    fn an_idr_can_keep_itself_as_a_long_term_reference() {
        let mut dpb = Dpb::new(&lsb_sps(8), true);
        let mut start = idr();
        start.marking.long_term_reference = true;
        decode(&mut dpb, &start, 0);
        decode(&mut dpb, &header(1, 2, 1), 1);
        decode(&mut dpb, &header(2, 4, 1), 2);
        let kept: Vec<_> = dpb.reference_frames().map(|f| f.long_term_idx).collect();
        assert_eq!(kept, [Some(0), None, None]);
    }

    #[test]
    fn the_dpb_size_follows_the_level_table_and_the_picture_size() {
        let mut uhd = sps();
        uhd.width_mbs = 240;
        uhd.height_mbs = 135;
        uhd.level_idc = 51;
        uhd.max_dec_frame_buffering = None;
        assert_eq!(dpb_size(&uhd), 5);
        let mut small = sps();
        small.width_mbs = 20;
        small.height_mbs = 12;
        small.max_dec_frame_buffering = None;
        assert_eq!(dpb_size(&small), 16);
        assert_eq!(dpb_size(&sps()), 4);
    }

    #[test]
    fn making_room_outputs_the_earliest_waiting_picture() {
        let mut dpb = Dpb::new(&lsb_sps(8), false);
        decode(&mut dpb, &idr(), 0);
        decode(&mut dpb, &header(1, 4, 0), 1);
        let released = dpb.make_room();
        assert_eq!(handles(&released.output), [0]);
    }
}
