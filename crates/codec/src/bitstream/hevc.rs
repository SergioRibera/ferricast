//! What decoders need to know about an H.265 stream that its backend may
//! not tell them: the sample bit depth its SPS declares (to hand the
//! stream to a backend that keeps it) and the colour its VUI signals.

use super::vui::VuiColour;

use super::bits::{Bits, unescape};
use super::split_annex_b;

/// `nal_unit_type` of an H.265 sequence parameter set.
const SPS: u8 = 33;

/// Most short-term reference picture sets an SPS lists (H.265 §7.4.3.2.1).
const MAX_SHORT_TERM_SETS: u32 = 64;

/// Most pictures one short-term set references (`sps_max_dec_pic_buffering`).
const MAX_SET_PICTURES: u32 = 16;

/// Most long-term reference pictures an SPS lists (H.265 §7.4.3.2.1).
const MAX_LONG_TERM_PICTURES: u32 = 32;

/// Luma and chroma bit depths an H.265 SPS NAL unit (2-byte header
/// included, emulation prevention in place) declares; `None` when `nal`
/// is not an SPS or is truncated.
#[must_use]
pub fn sps_bit_depth(nal: &[u8]) -> Option<(u8, u8)> {
    let rbsp = sps_rbsp(nal)?;
    let head = SpsHead::read(&rbsp)?;
    Some((head.luma_depth, head.chroma_depth))
}

/// The bit depths of the last SPS in an Annex B buffer, if it has one.
#[must_use]
pub fn annex_b_bit_depth(buf: &[u8]) -> Option<(u8, u8)> {
    split_annex_b(buf).into_iter().rev().find_map(sps_bit_depth)
}

/// The RBSP of `nal` when it is an SPS.
fn sps_rbsp(nal: &[u8]) -> Option<Vec<u8>> {
    ((nal.first()? >> 1) & 0x3f == SPS).then(|| unescape(nal.get(2..).unwrap_or_default()))
}

/// The VUI colour fields of an H.265 SPS NAL unit; `None` when `nal` is not
/// a readable SPS.
pub(super) fn sps_colour(nal: &[u8]) -> Option<VuiColour> {
    SpsHead::read(&sps_rbsp(nal)?)?.colour()
}

/// An SPS read up to its bit depths, the reader left right after them.
struct SpsHead<'a> {
    bits: Bits<'a>,
    sub_layers: usize,
    luma_depth: u8,
    chroma_depth: u8,
}

impl<'a> SpsHead<'a> {
    /// `rbsp` is the SPS payload after the 2-byte NAL header.
    fn read(rbsp: &'a [u8]) -> Option<Self> {
        let mut bits = Bits::new(rbsp);
        // sps_video_parameter_set_id.
        bits.skip(4)?;
        let sub_layers = usize::try_from(bits.read(3)?).ok()?;
        // sps_temporal_id_nesting_flag.
        bits.skip(1)?;
        skip_profile_tier_level(&mut bits, sub_layers)?;
        bits.ue()?; // sps_seq_parameter_set_id
        if bits.ue()? == 3 {
            bits.skip(1)?; // separate_colour_plane_flag
        }
        bits.ue()?; // pic_width_in_luma_samples
        bits.ue()?; // pic_height_in_luma_samples
        if bits.flag()? {
            for _ in 0..4 {
                bits.ue()?; // conformance window offsets
            }
        }
        let luma_depth = u8::try_from(bits.ue()?.checked_add(8)?).ok()?;
        let chroma_depth = u8::try_from(bits.ue()?.checked_add(8)?).ok()?;
        Some(Self {
            bits,
            sub_layers,
            luma_depth,
            chroma_depth,
        })
    }

    /// Reads on to the VUI's colour fields (H.265 §7.3.2.2.1); `None`
    /// when the SPS is truncated or out of range.
    fn colour(mut self) -> Option<VuiColour> {
        let bits = &mut self.bits;
        let log2_max_poc_lsb = bits.ue()?.checked_add(4)?;
        if log2_max_poc_lsb > 16 {
            return None;
        }
        let ordered = if bits.flag()? { self.sub_layers + 1 } else { 1 };
        for _ in 0..ordered {
            bits.ue()?; // sps_max_dec_pic_buffering_minus1
            bits.ue()?; // sps_max_num_reorder_pics
            bits.ue()?; // sps_max_latency_increase_plus1
        }
        for _ in 0..6 {
            // Coding and transform block sizes, transform hierarchy depths.
            bits.ue()?;
        }
        // scaling_list_enabled_flag, then sps_scaling_list_data_present_flag.
        if bits.flag()? && bits.flag()? {
            skip_scaling_list_data(bits)?;
        }
        // amp_enabled_flag, sample_adaptive_offset_enabled_flag.
        bits.skip(2)?;
        if bits.flag()? {
            // pcm_sample_bit_depth_{luma,chroma}_minus1.
            bits.skip(8)?;
            bits.ue()?; // log2_min_pcm_luma_coding_block_size_minus3
            bits.ue()?; // log2_diff_max_min_pcm_luma_coding_block_size
            bits.skip(1)?; // pcm_loop_filter_disabled_flag
        }
        skip_short_term_ref_pic_sets(bits)?;
        if bits.flag()? {
            let pictures = bits.ue()?;
            if pictures > MAX_LONG_TERM_PICTURES {
                return None;
            }
            for _ in 0..pictures {
                // lt_ref_pic_poc_lsb_sps, used_by_curr_pic_lt_sps_flag.
                bits.skip(usize::try_from(log2_max_poc_lsb).ok()? + 1)?;
            }
        }
        // sps_temporal_mvp_enabled_flag, strong_intra_smoothing_enabled_flag.
        bits.skip(2)?;
        if bits.flag()? {
            VuiColour::read(bits)
        } else {
            Some(VuiColour::default())
        }
    }
}

/// `profile_tier_level(1, sps_max_sub_layers_minus1)` (ITU-T H.265
/// §7.3.3): the general profile (88 bits) and level (8 bits), then per
/// sub-layer presence flags padded to eight entries, then each present
/// sub-layer part.
fn skip_profile_tier_level(bits: &mut Bits<'_>, sub_layers: usize) -> Option<()> {
    bits.skip(96)?;
    let mut present = Vec::with_capacity(sub_layers);
    for _ in 0..sub_layers {
        present.push((bits.flag()?, bits.flag()?));
    }
    if sub_layers > 0 {
        bits.skip(2 * (8 - sub_layers))?;
    }
    for (profile, level) in present {
        if profile {
            bits.skip(88)?;
        }
        if level {
            bits.skip(8)?;
        }
    }
    Some(())
}

/// `scaling_list_data()` (H.265 §7.3.4).
fn skip_scaling_list_data(bits: &mut Bits<'_>) -> Option<()> {
    for size in 0..4_u32 {
        let step = if size == 3 { 3 } else { 1 };
        for _ in (0..6).step_by(step) {
            if bits.flag()? {
                let coefficients = 64.min(1 << (4 + (size << 1)));
                if size > 1 {
                    bits.se()?; // scaling_list_dc_coef_minus8
                }
                for _ in 0..coefficients {
                    bits.se()?; // scaling_list_delta_coef
                }
            } else {
                bits.ue()?; // scaling_list_pred_matrix_id_delta
            }
        }
    }
    Some(())
}

/// `num_short_term_ref_pic_sets` and each `st_ref_pic_set()` (H.265
/// §7.3.7). A set predicted from the one before it carries a flag pair
/// per picture of that set (plus one), so each set's picture count is
/// kept for the next.
fn skip_short_term_ref_pic_sets(bits: &mut Bits<'_>) -> Option<()> {
    let sets = bits.ue()?;
    if sets > MAX_SHORT_TERM_SETS {
        return None;
    }
    let mut previous = 0_u32;
    for index in 0..sets {
        let predicted = index != 0 && bits.flag()?;
        previous = if predicted {
            // delta_rps_sign, abs_delta_rps_minus1.
            bits.skip(1)?;
            bits.ue()?;
            let mut kept = 0;
            for _ in 0..=previous {
                // used_by_curr_pic_flag, else use_delta_flag.
                if bits.flag()? || bits.flag()? {
                    kept += 1;
                }
            }
            kept
        } else {
            let pictures = bits.ue()?.checked_add(bits.ue()?)?;
            if pictures > MAX_SET_PICTURES {
                return None;
            }
            for _ in 0..pictures {
                bits.ue()?; // delta_poc_s{0,1}_minus1
                bits.skip(1)?; // used_by_curr_pic_s{0,1}_flag
            }
            pictures
        };
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use ferricast_core::{
        ChromaLocation, ColourDescription, ColourPrimaries, ColourRange, MatrixCoefficients,
        TransferCharacteristics,
    };

    use super::super::NalSyntax;
    use super::super::bits::Writer;
    use super::super::vui::ColourTracker;
    use super::*;

    /// The head of an SPS NAL unit for a 1920x1080 stream of `depth`-bit
    /// samples with `sub_layers` temporal sub-layers above the base one,
    /// each with its profile and level present; `window` adds a
    /// conformance window. Ends right after the bit depths.
    fn sps_head(depth: u32, sub_layers: u32, window: bool) -> Writer {
        let mut rbsp = Writer::default();
        rbsp.put(0, 4).put(sub_layers, 3).put(1, 1);
        // General profile: Main10 (2) or Main (1), level 4 (120).
        rbsp.put(0, 3).put(if depth > 8 { 2 } else { 1 }, 5);
        rbsp.put(0x6000_0000, 32)
            .put(0x9000_0000, 32)
            .put(0, 16)
            .put(120, 8);
        for _ in 0..sub_layers {
            rbsp.put(1, 1).put(1, 1);
        }
        if sub_layers > 0 {
            rbsp.put(0, 2 * (8 - sub_layers as usize));
        }
        for _ in 0..sub_layers {
            rbsp.put(0, 32).put(0, 32).put(0, 24).put(90, 8);
        }
        rbsp.ue(0)
            .ue(1)
            .ue(1920)
            .ue(if window { 1088 } else { 1080 });
        if window {
            rbsp.put(1, 1).ue(0).ue(0).ue(0).ue(4);
        } else {
            rbsp.put(0, 1);
        }
        rbsp.ue(depth - 8).ue(depth - 8);
        rbsp
    }

    fn sps(depth: u32, sub_layers: u32, window: bool) -> Vec<u8> {
        sps_head(depth, sub_layers, window).nal(&[SPS << 1, 1])
    }

    /// What the SPS of a 10-bit stream carries between its bit depths and
    /// its VUI, with the parts that make the walk hard: per-sub-layer
    /// ordering, a scaling list in every size, PCM, explicit and
    /// predicted short-term sets and long-term pictures.
    fn sps_body(rbsp: &mut Writer) {
        // log2_max_pic_order_cnt_lsb_minus4 = 4, ordering for 2 sub-layers.
        rbsp.ue(4).flag(true);
        for _ in 0..2 {
            rbsp.ue(4).ue(2).ue(0);
        }
        for value in [0, 3, 0, 3, 1, 1] {
            rbsp.ue(value);
        }
        // Scaling lists present: explicit for matrix 0 of every size.
        rbsp.flag(true).flag(true);
        for size in 0..4_u32 {
            let step = if size == 3 { 3 } else { 1 };
            for matrix in (0..6).step_by(step) {
                if matrix == 0 {
                    rbsp.flag(true);
                    if size > 1 {
                        rbsp.se(-3);
                    }
                    for index in 0..64.min(1 << (4 + (size << 1))) {
                        rbsp.se(if index % 2 == 0 { 2 } else { -1 });
                    }
                } else {
                    rbsp.flag(false).ue(0);
                }
            }
        }
        // AMP, SAO, then PCM.
        rbsp.flag(true)
            .flag(true)
            .flag(true)
            .put(0x77, 8)
            .ue(0)
            .ue(1)
            .flag(false);
        // Three short-term sets: two explicit pictures, then predicted
        // twice (each keeping all three flags of the set before).
        rbsp.ue(3);
        rbsp.ue(1).ue(1).ue(0).flag(true).ue(1).flag(true);
        rbsp.flag(true).flag(false).ue(0);
        rbsp.flag(true)
            .flag(false)
            .flag(true)
            .flag(false)
            .flag(true);
        rbsp.flag(true).flag(false).ue(1);
        rbsp.flag(true).flag(false).flag(true).flag(true).flag(true);
        // Two long-term pictures of 8-bit POC LSBs plus their used flag.
        rbsp.flag(true).ue(2).put(0x1ff, 9).put(0x0aa, 9);
        // Temporal MVP, strong intra smoothing.
        rbsp.flag(true).flag(true);
    }

    #[test]
    fn the_declared_bit_depth_is_read_past_any_sub_layers() {
        for sub_layers in [0, 1, 3, 7] {
            for window in [false, true] {
                assert_eq!(
                    sps_bit_depth(&sps(10, sub_layers, window)),
                    Some((10, 10)),
                    "{sub_layers} {window}"
                );
                assert_eq!(
                    sps_bit_depth(&sps(8, sub_layers, window)),
                    Some((8, 8)),
                    "{sub_layers} {window}"
                );
            }
        }
        let mut annex_b = vec![0, 0, 0, 1];
        annex_b.extend(sps(10, 0, false));
        annex_b.extend([0, 0, 1, 0x26, 1]);
        assert_eq!(annex_b_bit_depth(&annex_b), Some((10, 10)));
    }

    #[test]
    fn other_units_and_truncated_sps_are_not_read() {
        assert_eq!(sps_bit_depth(&[0x40, 0x01, 0x0C]), None, "a VPS");
        let full = sps(10, 2, true);
        // The last bytes are the depths' trailing bits and the stop bit.
        for len in 0..full.len() - 2 {
            assert_eq!(sps_bit_depth(&full[..len]), None, "{len} bytes");
        }
    }

    fn annex_b(nal: &[u8]) -> Vec<u8> {
        let mut out = vec![0, 0, 0, 1];
        out.extend_from_slice(nal);
        out.extend([0, 0, 1, 0x26, 1]);
        out
    }

    /// An HDR10 SPS: BT.2020 primaries and matrix, PQ, limited range,
    /// top-left chroma, behind an extended sample aspect ratio.
    fn hdr10_sps() -> Vec<u8> {
        let mut rbsp = sps_head(10, 1, true);
        sps_body(&mut rbsp);
        rbsp.flag(true);
        rbsp.flag(true).put(255, 8).put(4, 16).put(3, 16);
        rbsp.flag(true).flag(false);
        rbsp.flag(true).put(5, 3).flag(false).flag(true);
        rbsp.put(9, 8).put(16, 8).put(9, 8);
        rbsp.flag(true).ue(2).ue(2);
        rbsp.nal(&[SPS << 1, 1])
    }

    #[test]
    fn the_vui_colour_is_read_past_every_sps_part() {
        let mut tracker = ColourTracker::default();
        tracker.observe_annex_b(&annex_b(&hdr10_sps()), NalSyntax::H265);
        let colour = tracker.current_or(None).expect("signalled colour");
        assert_eq!(colour.primaries, ColourPrimaries::Bt2020);
        assert_eq!(colour.transfer, TransferCharacteristics::Pq);
        assert_eq!(colour.matrix, MatrixCoefficients::Bt2020Ncl);
        assert_eq!(colour.range, ColourRange::Limited);
        assert_eq!(colour.chroma_location, Some(ChromaLocation::TopLeft));
    }

    #[test]
    fn the_bitstream_wins_over_the_container_and_a_new_sps_replaces_it() {
        let mut tracker = ColourTracker::default();
        let container = Some(ColourDescription::BT709_LIMITED);
        assert_eq!(tracker.current_or(container), container, "nothing seen");
        tracker.observe_annex_b(&annex_b(&hdr10_sps()), NalSyntax::H265);
        let hdr = tracker.current_or(container).expect("colour");
        assert_eq!(hdr.transfer, TransferCharacteristics::Pq);
        tracker.observe_annex_b(&[0, 0, 0, 1, 0x26, 1, 0xaf], NalSyntax::H265);
        assert_eq!(tracker.current_or(container), Some(hdr), "a slice keeps it");
        // An SPS without a VUI leaves the container's say.
        let mut bare = sps_head(8, 0, false);
        sps_body(&mut bare);
        bare.flag(false);
        tracker.observe_annex_b(&annex_b(&bare.nal(&[SPS << 1, 1])), NalSyntax::H265);
        assert_eq!(tracker.current_or(container), container);
    }

    #[test]
    fn a_truncated_vui_is_not_read() {
        let full = hdr10_sps();
        let mut tracker = ColourTracker::default();
        // The last byte is the stop bit; the one before ends the VUI fields.
        for len in 0..full.len() - 1 {
            tracker.observe_annex_b(&annex_b(&full[..len]), NalSyntax::H265);
            assert_eq!(tracker.current_or(None), None, "{len} bytes");
        }
    }
}
