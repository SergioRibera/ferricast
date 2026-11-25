//! The left / top crop of H.264 and HEVC sequences, read from the SPS.
//!
//! The cuvid parser reports a sequence's display area with `left` and `top`
//! always 0 (checked on driver 595: an H.264 `frame_crop_left_offset` or an
//! HEVC `conf_win_left_offset` only shows up as a smaller width, with the
//! picture still starting at column 0). So the decoder reads the cropping
//! window of each SPS the caller feeds, just that: it looks at the NAL units
//! in front of a packet's first slice, decodes nothing else, and does not
//! affect timing. The result is used only when its right and bottom edges
//! agree with what the parser reports, so a misread can only fall back to
//! the parser's values.

use super::Codec;
use super::format::Rect;
use super::nal::{Bits, unescape};

/// The cropping window of the newest SPS, in luma samples from each edge of
/// the coded picture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Crop {
    pub(super) coded_width: u32,
    pub(super) coded_height: u32,
    pub(super) left: u32,
    pub(super) right: u32,
    pub(super) top: u32,
    pub(super) bottom: u32,
}

impl Crop {
    /// The parser's display area with this crop's left and top edges, when
    /// the crop belongs to the same coded size and its right and bottom
    /// edges match the area; otherwise the area unchanged.
    pub(super) fn refine(&self, area: Rect, coded_width: u32, coded_height: u32) -> Rect {
        let matches = (self.coded_width, self.coded_height) == (coded_width, coded_height)
            && area.left == 0
            && area.top == 0
            && coded_width.checked_sub(self.right) == Some(area.right)
            && coded_height.checked_sub(self.bottom) == Some(area.bottom)
            && self.left < area.right
            && self.top < area.bottom;
        if matches {
            Rect {
                left: self.left,
                top: self.top,
                ..area
            }
        } else {
            area
        }
    }
}

/// Reads the SPS NAL units in front of a packet's first slice (handed over
/// by [`super::nal::screen`]) and keeps the newest crop.
#[derive(Debug, Default)]
pub(super) struct CropScanner {
    /// A NAL unit's payload without emulation prevention bytes, reused.
    rbsp: Vec<u8>,
    pub(super) latest: Option<Crop>,
}

impl CropScanner {
    /// Reads `unit` (a NAL unit from its header on) when it is an SPS:
    /// any SPS for H.264, a base-layer one for HEVC.
    pub(super) fn read(&mut self, codec: Codec, unit: &[u8]) {
        let Some(&header) = unit.first() else {
            return;
        };
        let payload = if codec == Codec::H264 {
            if header & 0x1f != 7 {
                return;
            }
            unit.get(1..)
        } else {
            let layer = unit.get(1).map_or(0, |second| {
                (u16::from(header & 1) << 5) | u16::from(second >> 3)
            });
            if (header >> 1) & 0x3f != 33 || layer != 0 {
                return;
            }
            unit.get(2..)
        };
        self.parse(codec, payload.unwrap_or_default());
    }

    /// The scratch buffer, for other RBSP readers of the same walk.
    pub(super) fn rbsp_buffer(&mut self) -> &mut Vec<u8> {
        &mut self.rbsp
    }

    fn parse(&mut self, codec: Codec, payload: &[u8]) {
        // A payload with forbidden bytes is no SPS the parser accepts: its
        // crop would not apply anyway.
        if !unescape(payload, &mut self.rbsp) {
            return;
        }
        let mut bits = Bits::new(&self.rbsp);
        let crop = match codec {
            Codec::H264 => h264_crop(&mut bits),
            _ => hevc_crop(&mut bits),
        };
        if let Some(crop) = crop {
            self.latest = Some(crop);
        }
    }
}

/// Crop units per chroma format: (`SubWidthC`, `SubHeightC`), 1 for
/// monochrome or separately coded colour planes.
fn crop_units(chroma_format_idc: u32, separate_planes: bool) -> Option<(u32, u32)> {
    match (chroma_format_idc, separate_planes) {
        (0 | 3, _) => Some((1, 1)),
        (1, false) => Some((2, 2)),
        (2, false) => Some((2, 1)),
        _ => None,
    }
}

/// `seq_parameter_set_data()` up to `frame_crop_*_offset` (H.264 7.3.2.1.1).
fn h264_crop(bits: &mut Bits<'_>) -> Option<Crop> {
    let profile = bits.bits(8)?;
    bits.skip(16)?; // constraint flags, level_idc
    bits.ue()?; // seq_parameter_set_id
    let mut chroma_format_idc = 1;
    let mut separate_planes = false;
    if matches!(
        profile,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        chroma_format_idc = bits.ue()?;
        if chroma_format_idc == 3 {
            separate_planes = bits.flag()?;
        }
        bits.ue()?; // bit_depth_luma_minus8
        bits.ue()?; // bit_depth_chroma_minus8
        bits.skip(1)?; // qpprime_y_zero_transform_bypass_flag
        if bits.flag()? {
            let lists = if chroma_format_idc == 3 { 12 } else { 8 };
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
            bits.skip_se()?; // offset_for_non_ref_pic
            bits.skip_se()?; // offset_for_top_to_bottom_field
            for _ in 0..bits.ue()?.min(255) {
                bits.skip_se()?;
            }
        }
        _ => {}
    }
    bits.ue()?; // max_num_ref_frames
    bits.skip(1)?; // gaps_in_frame_num_value_allowed_flag
    let width_mbs = bits.ue()?.checked_add(1)?;
    let height_units = bits.ue()?.checked_add(1)?;
    let frame_mbs_only = bits.flag()?;
    if !frame_mbs_only {
        bits.skip(1)?; // mb_adaptive_frame_field_flag
    }
    bits.skip(1)?; // direct_8x8_inference_flag
    let field_factor = if frame_mbs_only { 1 } else { 2 };
    let (unit_x, unit_y) = crop_units(chroma_format_idc, separate_planes)?;
    let mut crop = Crop {
        coded_width: width_mbs.checked_mul(16)?,
        coded_height: height_units.checked_mul(16 * field_factor)?,
        ..Crop::default()
    };
    if bits.flag()? {
        let unit_y = unit_y * field_factor;
        crop.left = bits.ue()?.checked_mul(unit_x)?;
        crop.right = bits.ue()?.checked_mul(unit_x)?;
        crop.top = bits.ue()?.checked_mul(unit_y)?;
        crop.bottom = bits.ue()?.checked_mul(unit_y)?;
    }
    Some(crop)
}

fn skip_scaling_list(bits: &mut Bits<'_>, size: usize) -> Option<()> {
    let (mut last, mut next) = (8_i64, 8_i64);
    for _ in 0..size {
        if next != 0 {
            let code = i64::from(bits.ue()?);
            let delta = if code % 2 == 1 {
                (code + 1) / 2
            } else {
                -code / 2
            };
            next = (last + delta + 256).rem_euclid(256);
        }
        if next != 0 {
            last = next;
        }
    }
    Some(())
}

/// `seq_parameter_set_rbsp()` up to `conf_win_*_offset` (HEVC 7.3.2.2.1),
/// for a base-layer SPS.
fn hevc_crop(bits: &mut Bits<'_>) -> Option<Crop> {
    bits.skip(4)?; // sps_video_parameter_set_id
    let sub_layers = bits.bits(3)?;
    bits.skip(1)?; // sps_temporal_id_nesting_flag
    // profile_tier_level(1, sps_max_sub_layers_minus1)
    bits.skip(88 + 8)?; // general profile and level
    let mut present = [(false, false); 8];
    for layer in present.iter_mut().take(sub_layers as usize) {
        *layer = (bits.flag()?, bits.flag()?);
    }
    if sub_layers > 0 {
        bits.skip(2 * (8 - sub_layers as usize))?;
    }
    for (profile, level) in present.iter().take(sub_layers as usize) {
        if *profile {
            bits.skip(88)?;
        }
        if *level {
            bits.skip(8)?;
        }
    }
    bits.ue()?; // sps_seq_parameter_set_id
    let chroma_format_idc = bits.ue()?;
    let separate_planes = chroma_format_idc == 3 && bits.flag()?;
    let width = bits.ue()?;
    let height = bits.ue()?;
    let (unit_x, unit_y) = crop_units(chroma_format_idc, separate_planes)?;
    let mut crop = Crop {
        coded_width: width,
        coded_height: height,
        ..Crop::default()
    };
    if bits.flag()? {
        crop.left = bits.ue()?.checked_mul(unit_x)?;
        crop.right = bits.ue()?.checked_mul(unit_x)?;
        crop.top = bits.ue()?.checked_mul(unit_y)?;
        crop.bottom = bits.ue()?.checked_mul(unit_y)?;
    }
    Some(crop)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::nal::screen;

    #[test]
    fn exp_golomb() {
        // 1, 010, 011, 00100, 00111 -> 0, 1, 2, 3, 6
        let data = [0b1010_0110, 0b0100_0011, 0b1000_0000];
        let mut bits = Bits::new(&data);
        let values: Vec<_> = (0..5).map(|_| bits.ue().unwrap()).collect();
        assert_eq!(values, [0, 1, 2, 3, 6]);
        assert_eq!(Bits::new(&[0, 0, 0, 0, 0]).ue(), None);
    }

    #[test]
    fn emulation_prevention_is_removed_and_slices_end_the_scan() {
        // A garbage SPS (not parseable) must leave the crop alone.
        let mut scanner = CropScanner::default();
        screen(
            Codec::H264,
            &[0, 0, 0, 1, 0x67, 0x64, 0, 0, 3, 1],
            true,
            &mut scanner,
        )
        .unwrap();
        assert_eq!(scanner.latest, None);
        assert_eq!(scanner.rbsp, [0x64, 0, 0, 1]);
        screen(
            Codec::H264,
            &[0, 0, 1, 0x65, 0, 0, 1, 0x67, 0x42],
            true,
            &mut scanner,
        )
        .unwrap();
        assert_eq!(
            scanner.rbsp,
            [0x64, 0, 0, 1],
            "nothing after a slice is read"
        );
    }

    #[test]
    fn refine_only_when_the_edges_agree() {
        let crop = Crop {
            coded_width: 400,
            coded_height: 304,
            left: 8,
            right: 0,
            top: 6,
            bottom: 4,
        };
        let area = Rect {
            left: 0,
            top: 0,
            right: 400,
            bottom: 300,
        };
        let refined = crop.refine(area, 400, 304);
        assert_eq!(
            (refined.left, refined.top, refined.width(), refined.height()),
            (8, 6, 392, 294)
        );
        assert_eq!(crop.refine(area, 416, 304), area);
        let other = Rect {
            bottom: 298,
            ..area
        };
        assert_eq!(crop.refine(other, 400, 304), other);
    }

    #[test]
    fn crop_units_per_chroma_format() {
        assert_eq!(crop_units(1, false), Some((2, 2)));
        assert_eq!(crop_units(2, false), Some((2, 1)));
        assert_eq!(crop_units(3, true), Some((1, 1)));
        assert_eq!(crop_units(4, false), None);
    }
}
