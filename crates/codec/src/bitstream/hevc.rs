//! What decoders need to know about an H.265 stream before handing it to
//! a backend: the sample bit depth its SPS declares.

use super::bits::{Bits, unescape};
use super::split_annex_b;

/// `nal_unit_type` of an H.265 sequence parameter set.
const SPS: u8 = 33;

/// Luma and chroma bit depths an H.265 SPS NAL unit (2-byte header
/// included, emulation prevention in place) declares; `None` when `nal`
/// is not an SPS or is truncated.
#[must_use]
pub fn sps_bit_depth(nal: &[u8]) -> Option<(u8, u8)> {
    if (nal.first()? >> 1) & 0x3f != SPS {
        return None;
    }
    let rbsp = unescape(nal.get(2..)?);
    let mut bits = Bits::new(&rbsp);
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
    let luma = u8::try_from(bits.ue()?.checked_add(8)?).ok()?;
    let chroma = u8::try_from(bits.ue()?.checked_add(8)?).ok()?;
    Some((luma, chroma))
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

/// The bit depths of the last SPS in an Annex B buffer, if it has one.
#[must_use]
pub fn annex_b_bit_depth(buf: &[u8]) -> Option<(u8, u8)> {
    split_annex_b(buf).into_iter().rev().find_map(sps_bit_depth)
}

#[cfg(test)]
mod tests {
    use super::super::bits::Writer;
    use super::*;

    /// An SPS NAL unit for a 1920x1080 stream of `depth`-bit samples with
    /// `sub_layers` temporal sub-layers above the base one, each with its
    /// profile and level present; `window` adds a conformance window.
    fn sps(depth: u32, sub_layers: u32, window: bool) -> Vec<u8> {
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
        rbsp.nal(&[SPS << 1, 1])
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
}
