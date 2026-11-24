//! What decoders need to know about an H.264 stream before handing it to
//! a backend: the luma bit depth its SPS declares.

use super::bits::{Bits, unescape};

/// H.264 `profile_idc`s whose SPS carries `chroma_format_idc` and the bit
/// depths (ITU-T H.264 §7.3.2.1.1).
const HIGH_PROFILES: [u32; 13] = [100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135];

/// The luma bit depth of an H.264 SPS NAL unit (header byte included,
/// emulation prevention in place): 8 for profiles that cannot signal
/// another; `None` when truncated.
#[must_use]
pub fn sps_bit_depth(nal: &[u8]) -> Option<u8> {
    let rbsp = unescape(nal.get(1..)?);
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

#[cfg(test)]
mod tests {
    use super::super::bits::Writer;
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
}
