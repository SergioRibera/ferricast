//! H.264 bitstream writing utilities.
//!
//! Implements just enough of the byte-stream / RBSP / Exp-Golomb
//! machinery from the H.264 spec (Rec. ITU-T H.264 §7.2 / §9.1) to
//! emit valid SPS and PPS NAL units.
//!
//! Why we need this: every VAAPI driver expects us to provide the
//! packed SPS / PPS bitstream alongside the
//! `VAEncSequenceParameterBufferH264` / `VAEncPictureParameterBufferH264`
//! struct. Some drivers can synthesize the bitstream from the structs
//! (`VAConfigAttribEncPackedHeaders == 0` means "I'll do it for you"),
//! but the conservative path that works on every driver — including
//! Mesa AMD — is to send packed headers ourselves.
//!
//! The writer is deliberately minimal: byte-oriented append, single
//! bit / fixed-width / Exp-Golomb writes, RBSP-trailing-bits padding,
//! emulation-prevention byte insertion (`0x000003`).

#![allow(
    dead_code,
    reason = "conditional: the SPS/PPS builders use a different subset per encoder"
)]

use ferricast_codec::bitstream::bits::escape;

/// Writes bits into a backing `Vec<u8>`. Produces an RBSP (Raw Byte
/// Sequence Payload) suitable for wrapping into a NAL unit; call
/// [`Self::finish_rbsp`] to add the trailing `1` bit + zero-pad to
/// the next byte boundary, then run the result through
/// [`escape`] before emitting an Annex B NAL.
pub(super) struct BitWriter {
    buf: Vec<u8>,
    /// Bit offset within the current byte, 0..=7 (0 = MSB).
    bit_pos: u8,
}

impl BitWriter {
    pub(super) fn new() -> Self {
        Self {
            buf: Vec::with_capacity(64),
            bit_pos: 0,
        }
    }

    /// Append a single 0/1 bit.
    pub(super) fn write_bit(&mut self, bit: u8) {
        let bit = bit & 1;
        if self.bit_pos == 0 {
            self.buf.push(0);
        }
        let last = self.buf.last_mut().unwrap();
        *last |= bit << (7 - self.bit_pos);
        self.bit_pos = (self.bit_pos + 1) & 7;
    }

    /// Append `width` bits of `value`, MSB-first. `width <= 32`.
    pub(super) fn write_bits(&mut self, value: u32, width: u8) {
        debug_assert!(width <= 32);
        for i in (0..width).rev() {
            self.write_bit(((value >> i) & 1) as u8);
        }
    }

    /// `u(1)` boolean flag.
    pub(super) fn write_flag(&mut self, flag: bool) {
        self.write_bit(u8::from(flag));
    }

    /// Unsigned Exp-Golomb (§9.1, `ue(v)`).
    ///
    /// Encoding: write `floor(log2(v+1))` zero bits, then `(v+1)`
    /// in binary. So `ue(0) = 1`, `ue(1) = 010`, `ue(2) = 011`,
    /// `ue(3) = 00100`, etc.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "leading is at most 32, and the Exp-Golomb fields written are far below u32::MAX, so v + 1 fits u32"
    )]
    pub(super) fn write_ue(&mut self, value: u32) {
        // Number of leading zeros = floor(log2(value+1)).
        let v_plus_1 = u64::from(value) + 1;
        let leading = v_plus_1.ilog2(); // for u64, equals floor(log2(v+1))
        for _ in 0..leading {
            self.write_bit(0);
        }
        // Then write v+1 with `leading + 1` bits.
        let total_bits = (leading + 1) as u8;
        self.write_bits(v_plus_1 as u32, total_bits);
    }

    /// Signed Exp-Golomb (§9.1, `se(v)`).
    ///
    /// Maps signed -> unsigned: `0 -> 0`, `1 -> 1`, `-1 -> 2`,
    /// `2 -> 3`, `-2 -> 4`, ... and feeds the result to `ue(v)`.
    pub(super) fn write_se(&mut self, value: i32) {
        let mapped = if value <= 0 {
            (-2 * value).cast_unsigned()
        } else {
            (2 * value - 1).cast_unsigned()
        };
        self.write_ue(mapped);
    }

    /// Byte-align with zero bits. Useful for `byte_alignment()` in
    /// trailing rbsp.
    pub(super) fn byte_align(&mut self) {
        if self.bit_pos != 0 {
            // Pad current byte to the boundary.
            self.bit_pos = 0;
        }
    }

    /// `rbsp_trailing_bits()` from §7.3.2.11: append a `1` bit, then
    /// pad with zeros to the next byte boundary.
    pub(super) fn finish_rbsp(mut self) -> Vec<u8> {
        self.write_bit(1);
        self.byte_align();
        self.buf
    }
}

/// Wrap an EBSP into an Annex B (§B.1.1) byte stream NAL unit:
/// `00 00 00 01` start code prefix + `nal_unit_type / nal_ref_idc`
/// header byte + EBSP.
pub(super) fn nal_annexb(nal_ref_idc: u8, nal_unit_type: u8, ebsp: &[u8]) -> Vec<u8> {
    debug_assert!(nal_ref_idc <= 0b11);
    debug_assert!(nal_unit_type <= 0b11111);
    let header = ((nal_ref_idc & 0b11) << 5) | (nal_unit_type & 0b11111);
    let mut out = Vec::with_capacity(4 + 1 + ebsp.len());
    out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
    out.push(header);
    out.extend_from_slice(ebsp);
    out
}

/// Build an Annex B NAL unit from a freshly built RBSP — runs
/// `finish_rbsp` + `emulation_prevent` + `nal_annexb` in sequence.
pub(super) fn finalize_nal(writer: BitWriter, nal_ref_idc: u8, nal_unit_type: u8) -> Vec<u8> {
    let rbsp = writer.finish_rbsp();
    let ebsp = escape(&rbsp);
    nal_annexb(nal_ref_idc, nal_unit_type, &ebsp)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ue(v)` encoding spot-checks from H.264 §9.1.
    /// 0 → "1" (1 bit), 1 → "010" (3 bits), 2 → "011" (3 bits),
    /// 3 → "00100" (5 bits), 4 → "00101" (5 bits).
    #[test]
    fn ue_examples() {
        for (v, expected_bits) in [
            (0, "1"),
            (1, "010"),
            (2, "011"),
            (3, "00100"),
            (4, "00101"),
            (7, "0001000"),
            (8, "0001001"),
        ] {
            let mut w = BitWriter::new();
            w.write_ue(v);
            // Read back the bits we just wrote.
            let mut got = String::new();
            for byte_idx in 0..w.buf.len() {
                let byte = w.buf[byte_idx];
                let valid = if byte_idx == w.buf.len() - 1 && w.bit_pos != 0 {
                    w.bit_pos
                } else {
                    8
                };
                for i in 0..valid {
                    got.push(if (byte >> (7 - i)) & 1 == 1 { '1' } else { '0' });
                }
            }
            assert_eq!(got, expected_bits, "ue({v}) = {got}, want {expected_bits}");
        }
    }

    #[test]
    fn se_examples() {
        // §9.1.1 mapping: 0→0, 1→1, -1→2, 2→3, -2→4 then ue().
        // 0 → ue(0) = "1"
        // 1 → ue(1) = "010"
        // -1 → ue(2) = "011"
        // 2 → ue(3) = "00100"
        // -2 → ue(4) = "00101"
        for (v, expected) in [
            (0, "1"),
            (1, "010"),
            (-1, "011"),
            (2, "00100"),
            (-2, "00101"),
        ] {
            let mut w = BitWriter::new();
            w.write_se(v);
            let mut got = String::new();
            for byte_idx in 0..w.buf.len() {
                let byte = w.buf[byte_idx];
                let valid = if byte_idx == w.buf.len() - 1 && w.bit_pos != 0 {
                    w.bit_pos
                } else {
                    8
                };
                for i in 0..valid {
                    got.push(if (byte >> (7 - i)) & 1 == 1 { '1' } else { '0' });
                }
            }
            assert_eq!(got, expected, "se({v}) = {got}, want {expected}");
        }
    }

    #[test]
    fn emulation_prevention() {
        // Plain pass-through.
        assert_eq!(escape(&[0xAA, 0xBB]), &[0xAA, 0xBB]);
        // 00 00 00 → 00 00 03 00
        assert_eq!(escape(&[0x00, 0x00, 0x00]), &[0x00, 0x00, 0x03, 0x00]);
        // 00 00 01 → 00 00 03 01
        assert_eq!(escape(&[0x00, 0x00, 0x01]), &[0x00, 0x00, 0x03, 0x01]);
        // 00 00 04 → 00 00 04 (above the 0x03 threshold)
        assert_eq!(escape(&[0x00, 0x00, 0x04]), &[0x00, 0x00, 0x04]);
    }

    #[test]
    fn rbsp_trailing_bits() {
        let mut w = BitWriter::new();
        w.write_bits(0b1010, 4);
        let rbsp = w.finish_rbsp();
        // 1010 + 1 + 000 = 10101000 = 0xA8
        assert_eq!(rbsp, &[0xA8]);
    }
}
