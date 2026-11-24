//! What decoders need to know about an H.264 stream that its backend may
//! not tell them: the luma bit depth its SPS declares and the colour its
//! VUI signals.

use super::bits::{Bits, unescape};
use super::vui::VuiColour;

/// H.264 `profile_idc`s whose SPS carries `chroma_format_idc` and the bit
/// depths (ITU-T H.264 §7.3.2.1.1).
const HIGH_PROFILES: [u32; 13] = [100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135];

/// `nal_unit_type` of an H.264 sequence parameter set.
const SPS: u8 = 7;

/// Most entries of an SPS's `offset_for_ref_frame` cycle (§7.4.2.1.1).
const MAX_REF_FRAMES_IN_CYCLE: u32 = 255;

/// An SPS read up to its luma bit depth, the reader left right after it.
struct SpsHead<'a> {
    bits: Bits<'a>,
    /// Whether the profile signals chroma format and bit depths (and so
    /// the fields after them).
    high: bool,
    /// `chroma_format_idc`; 1 (4:2:0) for profiles that cannot signal it.
    chroma_format: u32,
    luma_depth: u8,
}

impl<'a> SpsHead<'a> {
    /// `rbsp` is the SPS payload after the 1-byte NAL header.
    fn read(rbsp: &'a [u8]) -> Option<Self> {
        let mut bits = Bits::new(rbsp);
        let profile = bits.read(8)?;
        // constraint flags, level_idc.
        bits.skip(16)?;
        bits.ue()?; // seq_parameter_set_id
        if !HIGH_PROFILES.contains(&profile) {
            return Some(Self {
                bits,
                high: false,
                chroma_format: 1,
                luma_depth: 8,
            });
        }
        let chroma_format = bits.ue()?;
        if chroma_format == 3 {
            bits.skip(1)?; // separate_colour_plane_flag
        }
        let luma_depth = u8::try_from(bits.ue()?.checked_add(8)?).ok()?;
        Some(Self {
            bits,
            high: true,
            chroma_format,
            luma_depth,
        })
    }

    /// Reads on to the VUI's colour fields (§7.3.2.1.1); `None` when the
    /// SPS is truncated or out of range.
    fn colour(mut self) -> Option<VuiColour> {
        let bits = &mut self.bits;
        // A high profile's chroma depth, transform bypass and scaling
        // matrices sit between the luma depth and the frame number.
        if self.high {
            bits.ue()?; // bit_depth_chroma_minus8
            bits.skip(1)?; // qpprime_y_zero_transform_bypass_flag
            if bits.flag()? {
                let lists = if self.chroma_format == 3 { 12 } else { 8 };
                for index in 0..lists {
                    if bits.flag()? {
                        skip_scaling_list(bits, if index < 6 { 16 } else { 64 })?;
                    }
                }
            }
        }
        bits.ue()?; // log2_max_frame_num_minus4
        match bits.ue()? {
            0 => {
                bits.ue()?; // log2_max_pic_order_cnt_lsb_minus4
            }
            1 => {
                bits.skip(1)?; // delta_pic_order_always_zero_flag
                bits.se()?; // offset_for_non_ref_pic
                bits.se()?; // offset_for_top_to_bottom_field
                let cycle = bits.ue()?;
                if cycle > MAX_REF_FRAMES_IN_CYCLE {
                    return None;
                }
                for _ in 0..cycle {
                    bits.se()?; // offset_for_ref_frame
                }
            }
            2 => {}
            _ => return None,
        }
        bits.ue()?; // max_num_ref_frames
        bits.skip(1)?; // gaps_in_frame_num_value_allowed_flag
        bits.ue()?; // pic_width_in_mbs_minus1
        bits.ue()?; // pic_height_in_map_units_minus1
        if !bits.flag()? {
            bits.skip(1)?; // mb_adaptive_frame_field_flag
        }
        bits.skip(1)?; // direct_8x8_inference_flag
        if bits.flag()? {
            for _ in 0..4 {
                bits.ue()?; // frame_crop_*_offset
            }
        }
        if bits.flag()? {
            VuiColour::read(bits)
        } else {
            Some(VuiColour::default())
        }
    }
}

/// `scaling_list()` (§7.3.2.1.1.1): delta-coded entries until one
/// predicts a zero, which ends the list.
fn skip_scaling_list(bits: &mut Bits<'_>, size: usize) -> Option<()> {
    let (mut last, mut next) = (8_i32, 8_i32);
    for _ in 0..size {
        if next != 0 {
            next = (last + bits.se()? + 256).rem_euclid(256);
        }
        if next != 0 {
            last = next;
        }
    }
    Some(())
}

/// The RBSP of `nal` when it is an SPS.
fn sps_rbsp(nal: &[u8]) -> Option<Vec<u8>> {
    (nal.first()? & 0x1f == SPS).then(|| unescape(nal.get(1..).unwrap_or_default()))
}

/// The luma bit depth of an H.264 SPS NAL unit (header byte included,
/// emulation prevention in place): 8 for profiles that cannot signal
/// another; `None` when truncated.
#[must_use]
pub fn sps_bit_depth(nal: &[u8]) -> Option<u8> {
    let rbsp = unescape(nal.get(1..)?);
    SpsHead::read(&rbsp).map(|head| head.luma_depth)
}

/// The VUI colour fields of an H.264 SPS NAL unit; `None` when `nal` is
/// not a readable SPS.
pub(super) fn sps_colour(nal: &[u8]) -> Option<VuiColour> {
    SpsHead::read(&sps_rbsp(nal)?)?.colour()
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

    #[test]
    fn baseline_and_high_are_eight_bit_and_high10_says_so() {
        let baseline = [0x67, 0x42, 0xc0, 0x1e, 0xf4, 0x41, 0x04, 0x60];
        assert_eq!(sps_bit_depth(&baseline), Some(8));
        let high = [0x67, 0x64, 0x00, 0x1e, 0xac, 0xd9, 0x41, 0x04, 0x60];
        assert_eq!(sps_bit_depth(&high), Some(8));
        let mut high10 = Writer::default();
        high10
            .put(110, 8)
            .put(0, 8)
            .put(30, 8)
            .ue(0)
            .ue(1)
            .ue(2)
            .ue(2);
        assert_eq!(sps_bit_depth(&high10.nal(&[0x67])), Some(10));
        assert_eq!(sps_bit_depth(&[0x67]), None);
    }

    /// A High 10 SPS with every optional part before the VUI: scaling
    /// lists (one ending early on a zero), the POC type 1 cycle, field
    /// coding and cropping; its VUI signals HLG BT.2020, full range.
    fn high10_hlg_sps() -> Vec<u8> {
        let mut rbsp = Writer::default();
        rbsp.put(110, 8).put(0, 8).put(40, 8).ue(0);
        rbsp.ue(1).ue(2).ue(2).flag(false);
        rbsp.flag(true);
        for index in 0..8 {
            if index == 0 {
                // 8 → 0 ends the list after one entry.
                rbsp.flag(true).se(-8);
            } else if index == 6 {
                rbsp.flag(true);
                for entry in 0..64 {
                    rbsp.se(if entry % 2 == 0 { 3 } else { -2 });
                }
            } else {
                rbsp.flag(false);
            }
        }
        rbsp.ue(0).ue(1).flag(false).se(-1).se(2).ue(2).se(4).se(-4);
        rbsp.ue(4)
            .flag(false)
            .ue(119)
            .ue(33)
            .flag(false)
            .flag(true)
            .flag(true);
        rbsp.flag(true).ue(0).ue(0).ue(0).ue(4);
        // VUI: square pixels, no overscan info, then the signal type.
        rbsp.flag(true).flag(true).put(1, 8).flag(false);
        rbsp.flag(true).put(5, 3).flag(true).flag(true);
        rbsp.put(9, 8).put(18, 8).put(9, 8);
        rbsp.flag(true).ue(2).ue(2);
        rbsp.nal(&[0x67])
    }

    fn annex_b(nal: &[u8]) -> Vec<u8> {
        let mut out = vec![0, 0, 0, 1];
        out.extend_from_slice(nal);
        out.extend([0, 0, 1, 0x65, 0x88]);
        out
    }

    #[test]
    fn the_vui_colour_is_read_past_every_sps_part() {
        let mut tracker = ColourTracker::default();
        tracker.observe_annex_b(&annex_b(&high10_hlg_sps()), NalSyntax::H264);
        let colour = tracker.current_or(None).expect("signalled colour");
        assert_eq!(colour.primaries, ColourPrimaries::Bt2020);
        assert_eq!(colour.transfer, TransferCharacteristics::Hlg);
        assert_eq!(colour.matrix, MatrixCoefficients::Bt2020Ncl);
        assert_eq!(colour.range, ColourRange::Full);
        assert_eq!(colour.chroma_location, Some(ChromaLocation::TopLeft));
    }

    #[test]
    fn a_main_profile_sps_without_a_vui_leaves_the_container_colour() {
        let mut rbsp = Writer::default();
        rbsp.put(77, 8).put(0, 8).put(30, 8).ue(0);
        rbsp.ue(0)
            .ue(2)
            .ue(1)
            .flag(false)
            .ue(79)
            .ue(44)
            .flag(true)
            .flag(true);
        rbsp.flag(false).flag(false);
        let container = Some(ColourDescription::BT709_LIMITED);
        let mut tracker = ColourTracker::default();
        tracker.observe_annex_b(&annex_b(&high10_hlg_sps()), NalSyntax::H264);
        tracker.observe_annex_b(&annex_b(&rbsp.nal(&[0x67])), NalSyntax::H264);
        assert_eq!(tracker.current_or(container), container);
    }

    #[test]
    fn a_truncated_sps_is_not_read() {
        let full = high10_hlg_sps();
        let mut tracker = ColourTracker::default();
        // The last byte is the stop bit; the one before ends the VUI fields.
        for len in 0..full.len() - 1 {
            tracker.observe_annex_b(&annex_b(&full[..len]), NalSyntax::H264);
            assert_eq!(tracker.current_or(None), None, "{len} bytes");
        }
    }
}
