//! What decoders need to know about an H.265 stream before handing it to
//! a backend: the sample bit depth its SPS declares.

use super::hdr::unescape;
use super::split_annex_b;

/// `nal_unit_type` of an H.265 sequence parameter set.
const SPS: u8 = 33;

/// Reads bits MSB first out of an RBSP.
struct Bits<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Bits<'_> {
    fn bit(&mut self) -> Option<u32> {
        let byte = self.bytes.get(self.at / 8)?;
        let bit = (byte >> (7 - self.at % 8)) & 1;
        self.at += 1;
        Some(u32::from(bit))
    }

    fn bits(&mut self, count: u32) -> Option<u32> {
        (0..count).try_fold(0, |value, _| Some((value << 1) | self.bit()?))
    }

    fn skip(&mut self, count: usize) -> Option<()> {
        self.at += count;
        (self.at <= self.bytes.len() * 8).then_some(())
    }

    /// An unsigned Exp-Golomb value (`ue(v)`), at most 32 bits.
    fn ue(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return None;
            }
        }
        Some((1 << zeros) - 1 + self.bits(zeros)?)
    }
}

/// Luma and chroma bit depths an H.265 SPS NAL unit (header included,
/// emulation prevention in place) declares; `None` when `nal` is not an
/// SPS or is truncated.
#[must_use]
pub fn sps_bit_depth(nal: &[u8]) -> Option<(u8, u8)> {
    if (nal.first()? >> 1) & 0x3f != SPS {
        return None;
    }
    let rbsp = unescape(nal);
    let mut bits = Bits {
        bytes: rbsp.get(2..)?,
        at: 0,
    };
    // sps_video_parameter_set_id, sps_max_sub_layers_minus1,
    // sps_temporal_id_nesting_flag.
    bits.skip(4)?;
    let sub_layers = bits.bits(3)?;
    bits.skip(1)?;
    // profile_tier_level(1, sps_max_sub_layers_minus1): the general
    // profile (88 bits) and level (8 bits), then per sub-layer presence
    // flags padded to eight entries, then each present sub-layer part.
    bits.skip(96)?;
    let mut present = Vec::new();
    for _ in 0..sub_layers {
        present.push((bits.bit()? == 1, bits.bit()? == 1));
    }
    if sub_layers > 0 {
        bits.skip(2 * (8 - sub_layers as usize))?;
    }
    for (profile, level) in present {
        if profile {
            bits.skip(88)?;
        }
        if level {
            bits.skip(8)?;
        }
    }
    bits.ue()?; // sps_seq_parameter_set_id
    if bits.ue()? == 3 {
        bits.skip(1)?; // separate_colour_plane_flag
    }
    bits.ue()?; // pic_width_in_luma_samples
    bits.ue()?; // pic_height_in_luma_samples
    if bits.bit()? == 1 {
        for _ in 0..4 {
            bits.ue()?; // conformance window offsets
        }
    }
    let luma = u8::try_from(bits.ue()? + 8).ok()?;
    let chroma = u8::try_from(bits.ue()? + 8).ok()?;
    Some((luma, chroma))
}

/// The bit depths of the last SPS in an Annex B buffer, if it has one.
#[must_use]
pub fn annex_b_bit_depth(buf: &[u8]) -> Option<(u8, u8)> {
    split_annex_b(buf)
        .into_iter()
        .rev()
        .find_map(sps_bit_depth)
}

#[cfg(test)]
mod tests {
    use super::super::hdr::escape;
    use super::*;

    /// Writes bits MSB first.
    #[derive(Default)]
    struct Writer {
        bytes: Vec<u8>,
        len: usize,
    }

    impl Writer {
        fn bits(&mut self, value: u32, count: u32) -> &mut Self {
            for shift in (0..count).rev() {
                if self.len.is_multiple_of(8) {
                    self.bytes.push(0);
                }
                let bit = u8::try_from((value >> shift) & 1).unwrap();
                *self.bytes.last_mut().unwrap() |= bit << (7 - self.len % 8);
                self.len += 1;
            }
            self
        }

        fn ue(&mut self, value: u32) -> &mut Self {
            let code = value + 1;
            let width = 32 - code.leading_zeros();
            self.bits(0, width - 1).bits(code, width)
        }
    }

    /// An SPS NAL unit for a 1920x1080 stream of `depth`-bit samples with
    /// `sub_layers` temporal sub-layers above the base one.
    fn sps(depth: u32, sub_layers: u32) -> Vec<u8> {
        let mut rbsp = Writer::default();
        rbsp.bits(0, 4).bits(sub_layers, 3).bits(1, 1);
        // General profile: Main10 (2) or Main (1), level 4 (120).
        rbsp.bits(0, 3).bits(if depth > 8 { 2 } else { 1 }, 5);
        rbsp.bits(0x6000_0000, 32)
            .bits(0x9000_0000, 32)
            .bits(0, 16)
            .bits(120, 8);
        for _ in 0..sub_layers {
            rbsp.bits(1, 1).bits(1, 1);
        }
        if sub_layers > 0 {
            rbsp.bits(0, 2 * (8 - sub_layers));
        }
        for _ in 0..sub_layers {
            rbsp.bits(0, 32).bits(0, 32).bits(0, 24).bits(90, 8);
        }
        rbsp.ue(0).ue(1).ue(1920).ue(1080).bits(0, 1);
        rbsp.ue(depth - 8).ue(depth - 8).bits(1, 1);
        let mut nal = vec![SPS << 1, 1];
        nal.extend(rbsp.bytes);
        escape(&nal)
    }

    #[test]
    fn the_declared_bit_depth_is_read_past_any_sub_layers() {
        for sub_layers in [0, 1, 3, 7] {
            assert_eq!(
                sps_bit_depth(&sps(10, sub_layers)),
                Some((10, 10)),
                "{sub_layers}"
            );
            assert_eq!(
                sps_bit_depth(&sps(8, sub_layers)),
                Some((8, 8)),
                "{sub_layers}"
            );
        }
        let mut annex_b = vec![0, 0, 0, 1];
        annex_b.extend(sps(10, 0));
        annex_b.extend([0, 0, 1, 0x26, 1]);
        assert_eq!(annex_b_bit_depth(&annex_b), Some((10, 10)));
    }

    #[test]
    fn other_units_and_truncated_sps_are_not_read() {
        assert_eq!(sps_bit_depth(&[0x40, 0x01, 0x0C]), None, "a VPS");
        let full = sps(10, 2);
        for len in 0..full.len() - 2 {
            assert_eq!(sps_bit_depth(&full[..len]), None, "{len} bytes");
        }
    }
}
