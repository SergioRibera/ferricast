//! HEVC bitstream helpers: BitWriter, Annex-B start codes,
//! emulation-prevention-byte insertion, and the codec-specific
//! 2-byte NAL header (`forbidden_zero_bit` + `nal_unit_type[6]` +
//! `nuh_layer_id[6]` + `nuh_temporal_id_plus1[3]`).
//!
//! Largely shaped like the H.264 variant in [`crate::h264::bitstream`]
//! — the only differences are the NAL header width and the prevention
//! byte boundary check that operates on the post-header payload, not
//! the whole NAL.

/// Big-endian bit packer.
pub(super) struct BitWriter {
    out: Vec<u8>,
    cur: u8,
    nbits: u8,
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            out: Vec::with_capacity(128),
            cur: 0,
            nbits: 0,
        }
    }

    pub fn write_bits(&mut self, value: u32, count: u8) {
        debug_assert!(count <= 32);
        for i in (0..count).rev() {
            let bit = ((value >> i) & 1) as u8;
            self.cur = (self.cur << 1) | bit;
            self.nbits += 1;
            if self.nbits == 8 {
                self.out.push(self.cur);
                self.cur = 0;
                self.nbits = 0;
            }
        }
    }

    /// Unsigned Exp-Golomb (ue(v)).
    pub fn write_ue(&mut self, value: u32) {
        let v = value + 1;
        let leading_zeros = (32 - v.leading_zeros() - 1) as u8;
        self.write_bits(0, leading_zeros);
        self.write_bits(v, leading_zeros + 1);
    }

    /// Signed Exp-Golomb (se(v)).
    pub fn write_se(&mut self, value: i32) {
        let mapped: u32 = if value <= 0 {
            (-value as u32) * 2
        } else {
            (value as u32) * 2 - 1
        };
        self.write_ue(mapped);
    }

    /// Pad with `rbsp_trailing_bits`: one `1` bit then zeros.
    pub fn rbsp_trailing(&mut self) {
        self.write_bits(1, 1);
        if self.nbits != 0 {
            let pad = 8 - self.nbits;
            self.write_bits(0, pad);
        }
    }

    pub fn into_inner(self) -> Vec<u8> {
        self.out
    }
}

/// HEVC NAL header (§7.3.1.2): 2 bytes
/// `forbidden_zero_bit(1) | nal_unit_type(6) | nuh_layer_id(6) | nuh_temporal_id_plus1(3)`.
/// Layer = 0 (base) and temporal_id_plus1 = 1 are the right values
/// for every Annex-B IRAP / non-IRAP NAL we emit on the base layer.
pub(super) fn nal_header(nal_unit_type: u8) -> [u8; 2] {
    let layer_id: u16 = 0;
    let temporal_id_plus1: u16 = 1;
    let word: u16 = ((nal_unit_type as u16 & 0x3f) << 9)
        | ((layer_id & 0x3f) << 3)
        | (temporal_id_plus1 & 0x07);
    [(word >> 8) as u8, (word & 0xff) as u8]
}

/// HEVC NAL unit types we emit.
#[allow(dead_code)]
pub(super) mod nal {
    pub const TRAIL_N: u8 = 0;
    pub const TRAIL_R: u8 = 1;
    pub const IDR_W_RADL: u8 = 19;
    pub const VPS_NUT: u8 = 32;
    pub const SPS_NUT: u8 = 33;
    pub const PPS_NUT: u8 = 34;
    pub const AUD_NUT: u8 = 35;
}

/// Wrap a finished RBSP into an Annex-B NAL: 4-byte start code +
/// 2-byte NAL header + emulation-prevention-byte-escaped payload.
pub(super) fn finalize_nal(nal_unit_type: u8, rbsp: Vec<u8>) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + 8);
    out.extend_from_slice(&[0, 0, 0, 1]);
    let hdr = nal_header(nal_unit_type);
    out.extend_from_slice(&hdr);
    write_with_emulation_prevention(&mut out, &rbsp);
    out
}

/// Insert `0x03` between any byte pair that would otherwise form a
/// start-code prefix (`0x00 0x00 0x00` / `0x01` / `0x02` / `0x03`).
/// The check operates on the *payload bytes only* — the NAL header
/// is already written and can't itself form a prefix (top bit
/// guaranteed zero, so its first byte is never 0).
fn write_with_emulation_prevention(out: &mut Vec<u8>, payload: &[u8]) {
    let mut zero_run = 0u8;
    for &b in payload {
        if zero_run >= 2 && b <= 0x03 {
            out.push(0x03);
            zero_run = 0;
        }
        out.push(b);
        if b == 0 {
            zero_run += 1;
        } else {
            zero_run = 0;
        }
    }
}
