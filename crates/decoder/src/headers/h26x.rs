//! Luma bit depth from H.264 / H.265 sequence parameter sets.

use super::bits::{Bits, unescape};

/// H.264 `profile_idc`s whose SPS carries `chroma_format_idc` and the bit
/// depths (ITU-T H.264 §7.3.2.1.1).
const HIGH_PROFILES: [u32; 13] = [100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135];

/// The luma bit depth of an H.264 SPS NAL unit (header byte included).
pub(crate) fn h264_bit_depth(sps: &[u8]) -> Option<u8> {
    let rbsp = unescape(sps.get(1..)?);
    let mut bits = Bits::new(&rbsp);
    let profile = bits.read(8)?;
    // constraint flags, level_idc.
    bits.skip(16)?;
    bits.ue()?;
    if !HIGH_PROFILES.contains(&profile) {
        return Some(8);
    }
    if bits.ue()? == 3 {
        // separate_colour_plane_flag.
        bits.skip(1)?;
    }
    u8::try_from(bits.ue()?.checked_add(8)?).ok()
}

/// The luma bit depth of an H.265 SPS NAL unit (2-byte header included).
pub(crate) fn h265_bit_depth(sps: &[u8]) -> Option<u8> {
    let rbsp = unescape(sps.get(2..)?);
    let mut bits = Bits::new(&rbsp);
    // sps_video_parameter_set_id.
    bits.skip(4)?;
    let sub_layers = usize::try_from(bits.read(3)?).ok()?;
    // sps_temporal_id_nesting_flag.
    bits.skip(1)?;
    skip_profile_tier_level(&mut bits, sub_layers)?;
    bits.ue()?;
    if bits.ue()? == 3 {
        // separate_colour_plane_flag.
        bits.skip(1)?;
    }
    // pic_width_in_luma_samples, pic_height_in_luma_samples.
    bits.ue()?;
    bits.ue()?;
    if bits.flag()? {
        // conformance window offsets.
        for _ in 0..4 {
            bits.ue()?;
        }
    }
    u8::try_from(bits.ue()?.checked_add(8)?).ok()
}

/// `profile_tier_level(1, sps_max_sub_layers_minus1)` (ITU-T H.265 §7.3.3).
fn skip_profile_tier_level(bits: &mut Bits<'_>, sub_layers: usize) -> Option<()> {
    // General profile (88 bits) and level (8 bits).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headers::bits::Writer;

    #[test]
    fn baseline_and_high_h264_are_eight_bit_and_high10_says_so() {
        let baseline = [0x67, 0x42, 0xc0, 0x1e, 0xf4, 0x41, 0x04, 0x60];
        assert_eq!(h264_bit_depth(&baseline), Some(8));
        let high = [0x67, 0x64, 0x00, 0x1e, 0xac, 0xd9, 0x41, 0x04, 0x60];
        assert_eq!(h264_bit_depth(&high), Some(8));
        let mut high10 = Writer::default();
        high10
            .put(110, 8)
            .put(0, 8)
            .put(30, 8)
            .ue(0)
            .ue(1)
            .ue(2)
            .ue(2);
        assert_eq!(h264_bit_depth(&high10.nal(&[0x67])), Some(10));
        assert_eq!(h264_bit_depth(&[0x67]), None);
    }

    /// An H.265 SPS up to the bit depths: `sub_layers` sub-layers, each
    /// with its profile and level present, 1920x1080 with a conformance
    /// window.
    fn h265_sps(sub_layers: u32, depth_minus8: u32) -> Vec<u8> {
        let mut sps = Writer::default();
        sps.put(0, 4).put(sub_layers, 3).put(1, 1);
        // General profile space / tier / Main10 idc, compatibility and
        // constraint flags; then general_level_idc.
        sps.put(2, 8).put(0x2000_0000, 32).put(0, 32).put(0, 16);
        sps.put(120, 8);
        for _ in 0..sub_layers {
            sps.put(0b11, 2);
        }
        if sub_layers > 0 {
            sps.put(0, 2 * (8 - sub_layers as usize));
        }
        for _ in 0..sub_layers {
            sps.put(0, 32).put(0, 32).put(0, 24).put(90, 8);
        }
        sps.ue(0)
            .ue(1)
            .ue(1920)
            .ue(1088)
            .put(1, 1)
            .ue(0)
            .ue(0)
            .ue(0)
            .ue(4);
        sps.ue(depth_minus8).ue(depth_minus8);
        sps.nal(&[0x42, 0x01])
    }

    #[test]
    fn h265_main_and_main10_depths_are_read() {
        assert_eq!(h265_bit_depth(&h265_sps(0, 0)), Some(8));
        assert_eq!(h265_bit_depth(&h265_sps(0, 2)), Some(10));
        assert_eq!(h265_bit_depth(&h265_sps(2, 2)), Some(10));
        assert_eq!(h265_bit_depth(&h265_sps(0, 0)[..12]), None);
    }
}
