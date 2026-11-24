//! Bit-level primitives the header parsers share: an MSB-first reader
//! (fixed fields, Exp-Golomb `ue(v)`, AV1 `uvlc()`), H.264 / H.265
//! emulation prevention, and AV1 `leb128()`.

/// Reads bits, most significant first; every read past the end is `None`.
#[derive(Debug, Clone)]
pub struct Bits<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Bits<'a> {
    /// A reader at the first bit of `bytes`.
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    /// `count` (at most 32) bits as an integer.
    pub fn read(&mut self, count: usize) -> Option<u32> {
        if count > 32 {
            return None;
        }
        let mut value = 0_u64;
        for _ in 0..count {
            let byte = self.bytes.get(self.at / 8)?;
            value = (value << 1) | u64::from((byte >> (7 - self.at % 8)) & 1);
            self.at += 1;
        }
        u32::try_from(value).ok()
    }

    /// One bit as a flag.
    pub fn flag(&mut self) -> Option<bool> {
        self.read(1).map(|bit| bit == 1)
    }

    /// Skips `count` bits; `None` (and no move) when fewer are left.
    pub fn skip(&mut self, count: usize) -> Option<()> {
        let end = self.at.checked_add(count)?;
        (end <= self.bytes.len() * 8).then(|| self.at = end)
    }

    /// An unsigned Exp-Golomb code (H.264 / H.265 `ue(v)`).
    pub fn ue(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while !self.flag()? {
            zeros += 1;
            if zeros > 31 {
                return None;
            }
        }
        let rest = self.read(zeros)?;
        (1_u32 << zeros).checked_sub(1)?.checked_add(rest)
    }

    /// A signed Exp-Golomb code (H.264 / H.265 `se(v)`).
    pub fn se(&mut self) -> Option<i32> {
        let code = i64::from(self.ue()?);
        let magnitude = (code + 1) / 2;
        i32::try_from(if code % 2 == 1 { magnitude } else { -magnitude }).ok()
    }

    /// An AV1 `uvlc()`.
    pub fn uvlc(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while !self.flag()? {
            zeros += 1;
            if zeros >= 32 {
                return Some(u32::MAX);
            }
        }
        let rest = self.read(zeros)?;
        (1_u32 << zeros).checked_sub(1)?.checked_add(rest)
    }
}

/// `nal` with emulation-prevention bytes (`00 00 03`) removed: the RBSP
/// of its payload (header bytes, when included, pass through).
#[must_use]
pub fn unescape(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len());
    let mut zeros = 0;
    for &byte in nal {
        if zeros >= 2 && byte == 3 {
            zeros = 0;
            continue;
        }
        zeros = if byte == 0 { zeros + 1 } else { 0 };
        out.push(byte);
    }
    out
}

/// `rbsp` with emulation-prevention bytes inserted, so no `00 00 0x`
/// (x ≤ 3) sequence appears in the NAL unit.
#[must_use]
pub fn escape(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + rbsp.len() / 64 + 1);
    let mut zeros = 0;
    for &byte in rbsp {
        if zeros >= 2 && byte <= 3 {
            out.push(3);
            zeros = 0;
        }
        zeros = if byte == 0 { zeros + 1 } else { 0 };
        out.push(byte);
    }
    out
}

/// An AV1 `leb128()` (at most 8 bytes) at the front of `bytes`: the
/// value and how many bytes it took.
#[must_use]
pub fn leb128(bytes: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0_u64;
    for (index, &byte) in bytes.iter().take(8).enumerate() {
        value |= u64::from(byte & 0x7f) << (7 * index);
        if byte & 0x80 == 0 {
            return Some((value, index + 1));
        }
    }
    None
}

/// `value` as `leb128()` bytes.
#[must_use]
pub fn to_leb128(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let byte = u8::try_from(value & 0x7f).unwrap_or(0);
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return out;
        }
        out.push(byte | 0x80);
    }
}

/// MSB-first bits for hand-built headers in tests.
#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct Writer {
    bits: Vec<bool>,
}

#[cfg(test)]
impl Writer {
    pub(crate) fn put(&mut self, value: u32, count: usize) -> &mut Self {
        self.bits
            .extend((0..count).rev().map(|shift| (value >> shift) & 1 == 1));
        self
    }

    pub(crate) fn ue(&mut self, value: u32) -> &mut Self {
        let coded = value + 1;
        let len = 32 - coded.leading_zeros() as usize;
        self.put(0, len - 1).put(coded, len)
    }

    pub(crate) fn se(&mut self, value: i32) -> &mut Self {
        let magnitude = value.unsigned_abs();
        self.ue(if value > 0 {
            2 * magnitude - 1
        } else {
            2 * magnitude
        })
    }

    pub(crate) fn flag(&mut self, value: bool) -> &mut Self {
        self.put(u32::from(value), 1)
    }

    /// The bits, zero-padded to whole bytes.
    pub(crate) fn bytes(&self) -> Vec<u8> {
        self.bits
            .chunks(8)
            .map(|byte| {
                byte.iter()
                    .enumerate()
                    .fold(0, |acc, (i, &bit)| acc | (u8::from(bit) << (7 - i)))
            })
            .collect()
    }

    /// A NAL unit: `header`, then the bits with a stop bit, escaped.
    pub(crate) fn nal(&self, header: &[u8]) -> Vec<u8> {
        let mut rbsp = self.bytes();
        rbsp.push(0x80);
        let mut out = header.to_vec();
        out.extend(escape(&rbsp));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_golomb_codes_and_fixed_fields_read_back() {
        // 1 | 010 | 011 | 00100 | 101 → ue 0, 1, 2, 3, then 3 fixed bits.
        let mut bits = Bits::new(&[0b1010_0110, 0b0100_1010]);
        assert_eq!(bits.ue(), Some(0));
        assert_eq!(bits.ue(), Some(1));
        assert_eq!(bits.ue(), Some(2));
        assert_eq!(bits.ue(), Some(3));
        assert_eq!(bits.read(3), Some(0b101));
        assert_eq!(bits.read(1), Some(0));
        assert_eq!(bits.read(1), None);
    }

    #[test]
    fn signed_exp_golomb_codes_alternate_around_zero() {
        let mut writer = Writer::default();
        for code in 0..5 {
            writer.ue(code);
        }
        let bytes = writer.bytes();
        let mut bits = Bits::new(&bytes);
        let read: Vec<_> = std::iter::from_fn(|| bits.se()).take(5).collect();
        assert_eq!(read, [0, 1, -1, 2, -2]);
    }

    #[test]
    fn emulation_prevention_bytes_are_removed() {
        assert_eq!(unescape(&[0, 0, 3, 1, 0, 0, 3, 0]), [0, 0, 1, 0, 0, 0]);
        assert_eq!(unescape(&[3, 0, 3]), [3, 0, 3]);
    }

    #[test]
    fn escaping_removes_start_code_emulation_and_round_trips() {
        let rbsp = [0x4E, 1, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0x80];
        let escaped = escape(&rbsp);
        assert_eq!(escaped, [0x4E, 1, 0, 0, 3, 0, 0, 3, 0, 1, 0, 0, 3, 3, 0x80]);
        assert_eq!(unescape(&escaped), rbsp);
    }

    #[test]
    fn leb128_round_trips() {
        for value in [0, 1, 127, 128, 300, 1 << 35] {
            let bytes = to_leb128(value);
            assert_eq!(leb128(&bytes), Some((value, bytes.len())));
        }
        assert_eq!(leb128(&[0x80]), None);
    }
}
