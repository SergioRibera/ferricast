//! Annex B NAL units (H.264, HEVC): finding them, reading their RBSP, and
//! the screening every packet goes through before the parser sees it.
//!
//! One walk over each packet serves everything the decoder reads itself:
//! the cropping window of the SPS in front of the first slice
//! ([`CropScanner`]), and for HEVC the full check of every VPS, which
//! NVIDIA's parser does not bounds-check (a VPS cut short inside its
//! `profile_tier_level()` faults inside `cuvidParseVideoData`, seen on
//! driver 595).

use super::Codec;
use super::crop::CropScanner;
use super::hevc;
use crate::error::{Error, Result};

/// NAL unit type of an HEVC video parameter set.
const HEVC_VPS: u8 = 32;
/// The most bytes held back between packets: a parameter set split by a
/// packet boundary waits for its end, never more.
const MAX_HELD_BACK: usize = 64 * 1024;

/// Offset of the next `00 00 01` in `data`.
///
/// Skips three bytes at a time where it can: a byte above 1 cannot end a
/// start code nor be one of its two zeros, so the next candidate end is
/// three bytes on.
pub(super) fn find_start_code(data: &[u8]) -> Option<usize> {
    let mut end = 2;
    while let Some(&byte) = data.get(end) {
        match byte {
            0 => end += 1,
            1 if data[end - 2] == 0 && data[end - 1] == 0 => return Some(end - 2),
            _ => end += 3,
        }
    }
    None
}

/// What a packet's screening decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Screened {
    /// Bytes of the packet to hand to the parser now; the rest is a
    /// parameter set (or start code) a packet boundary may have cut, held
    /// back until the next packet completes it.
    pub(super) ready: usize,
}

/// Walks the NAL units of an Annex B packet once: reads the SPS crop in
/// front of the first slice into `crop`, and for HEVC checks every VPS.
///
/// `whole` says the packet ends where a NAL unit ends (it is parsed with
/// `CUVID_PKT_ENDOFPICTURE`). Otherwise the packet may stop inside a NAL
/// unit: a VPS at its end (or a start code cut short) is held back rather
/// than checked, since the parser only reads it once the next start code
/// arrives.
///
/// # Errors
///
/// [`Error::InvalidPacket`] for an HEVC VPS that does not parse to its
/// `rbsp_trailing_bits()` with every value in range.
pub(super) fn screen(
    codec: Codec,
    data: &[u8],
    whole: bool,
    crop: &mut CropScanner,
) -> Result<Screened> {
    let hevc = codec == Codec::Hevc;
    let mut before_slices = true;
    let mut next = find_start_code(data);
    while let Some(start) = next {
        let header_at = start + 3;
        let Some(&header) = data.get(header_at) else {
            // A start code with nothing after it yet.
            return Ok(held_back(data, whole, start));
        };
        let kind = if hevc {
            (header >> 1) & 0x3f
        } else {
            header & 0x1f
        };
        let vcl = if hevc {
            kind < 32
        } else {
            matches!(kind, 1..=5)
        };
        before_slices &= !vcl;
        if !hevc && !before_slices {
            // H.264: nothing past the first slice is read; its body is not
            // scanned.
            break;
        }
        next = data
            .get(header_at..)
            .and_then(find_start_code)
            .map(|offset| header_at + offset);
        let end = next.unwrap_or(data.len());
        let unit = &data[header_at..end];
        if hevc && kind == HEVC_VPS {
            if next.is_none() && !whole {
                return Ok(held_back(data, whole, start));
            }
            hevc::check_vps(unit, crop.rbsp_buffer()).map_err(|reason| Error::InvalidPacket {
                reason: format!("HEVC VPS at byte {start}: {reason}"),
            })?;
        } else if before_slices {
            crop.read(codec, unit);
        }
    }
    if !whole && hevc {
        // A start code may be split across packets: hold back trailing
        // zeros so the next packet's scan sees all of it.
        let zeros = data
            .iter()
            .rev()
            .take(3)
            .take_while(|&&byte| byte == 0)
            .count();
        if zeros > 0 {
            return Ok(held_back(data, whole, data.len() - zeros));
        }
    }
    Ok(Screened { ready: data.len() })
}

/// What to parse now when the bytes from `from` (a start code, with the
/// zero byte of a 4-byte one) may continue in the next packet.
fn held_back(data: &[u8], whole: bool, mut from: usize) -> Screened {
    if from > 0 && data[from - 1] == 0 {
        from -= 1;
    }
    if whole {
        Screened { ready: data.len() }
    } else {
        Screened { ready: from }
    }
}

/// Bytes a non-whole HEVC packet held back, waiting for the next one.
#[derive(Debug, Default)]
pub(super) struct HeldBack {
    bytes: Vec<u8>,
    /// The bytes held back plus the next packet, reused.
    joined: Vec<u8>,
}

impl HeldBack {
    /// `packet` after what the packet before held back, screened by
    /// `screen`: returns the part to parse now and keeps the rest.
    ///
    /// # Errors
    ///
    /// `screen`'s error (what was held back is dropped with the packet), or
    /// [`Error::InvalidPacket`] when more than [`MAX_HELD_BACK`] bytes pile
    /// up: no parameter set is that long.
    pub(super) fn split<'a>(
        &'a mut self,
        packet: &'a [u8],
        screen: impl FnOnce(&[u8]) -> Result<Screened>,
    ) -> Result<&'a [u8]> {
        let Self { bytes, joined } = self;
        let data: &'a [u8] = if bytes.is_empty() {
            packet
        } else {
            joined.clear();
            joined.extend_from_slice(bytes);
            joined.extend_from_slice(packet);
            joined
        };
        bytes.clear();
        let (ready, rest) = data.split_at(screen(data)?.ready.min(data.len()));
        if rest.len() > MAX_HELD_BACK {
            return Err(Error::InvalidPacket {
                reason: format!("an HEVC VPS still unterminated after {} bytes", rest.len()),
            });
        }
        bytes.extend_from_slice(rest);
        Ok(ready)
    }

    /// Drops what was held back (at a discontinuity or the end of the
    /// stream: an unterminated VPS is never parsed).
    pub(super) fn clear(&mut self) {
        if !self.bytes.is_empty() {
            tracing::debug!(
                bytes = self.bytes.len(),
                "dropping an unterminated HEVC VPS"
            );
        }
        self.bytes.clear();
    }
}

/// `nal_unit_payload` without its emulation prevention bytes, into `out`.
/// `false` when the payload holds a byte sequence no NAL unit may contain
/// (`00 00 00`, `00 00 01`, `00 00 02`), which the parser would read as the
/// end of the unit.
pub(super) fn unescape(payload: &[u8], out: &mut Vec<u8>) -> bool {
    out.clear();
    let mut zeros = 0;
    for &byte in payload {
        if zeros >= 2 {
            match byte {
                3 => {
                    zeros = 0;
                    continue;
                }
                0..=2 => return false,
                _ => {}
            }
        }
        zeros = if byte == 0 { zeros + 1 } else { 0 };
        out.push(byte);
    }
    true
}

/// An MSB-first bit reader with Exp-Golomb codes.
pub(super) struct Bits<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> Bits<'a> {
    pub(super) fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    /// Bits read so far.
    pub(super) fn position(&self) -> usize {
        self.position
    }

    pub(super) fn bit(&mut self) -> Option<u32> {
        let byte = self.data.get(self.position / 8)?;
        let bit = (byte >> (7 - self.position % 8)) & 1;
        self.position += 1;
        Some(u32::from(bit))
    }

    /// `u(n)`, n at most 32.
    pub(super) fn bits(&mut self, count: u32) -> Option<u32> {
        (0..count).try_fold(0, |value, _| Some((value << 1) | self.bit()?))
    }

    pub(super) fn skip(&mut self, count: usize) -> Option<()> {
        self.position += count;
        (self.position <= self.data.len() * 8).then_some(())
    }

    pub(super) fn flag(&mut self) -> Option<bool> {
        self.bit().map(|bit| bit == 1)
    }

    /// `ue(v)`, at most 32 bits of value.
    pub(super) fn ue(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return None;
            }
        }
        let suffix = self.bits(zeros)?;
        ((1_u64 << zeros) - 1 + u64::from(suffix)).try_into().ok()
    }

    /// `ue(v)` no larger than `max`.
    pub(super) fn ue_max(&mut self, max: u32) -> Option<u32> {
        self.ue().filter(|value| *value <= max)
    }

    /// `se(v)`; only skipped, so the value is not needed.
    pub(super) fn skip_se(&mut self) -> Option<()> {
        self.ue().map(drop)
    }

    /// Where `rbsp_trailing_bits()` starts: the position of the last set
    /// bit (the stop bit) of the data. `None` when every bit is zero.
    pub(super) fn stop_bit(&self) -> Option<usize> {
        let (index, &byte) = self
            .data
            .iter()
            .enumerate()
            .rev()
            .find(|(_, byte)| **byte != 0)?;
        Some(index * 8 + 7 - byte.trailing_zeros() as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(data: &[u8]) -> Option<usize> {
        data.windows(3).position(|window| window == [0, 0, 1])
    }

    #[test]
    fn start_codes_are_found_like_a_byte_by_byte_search() {
        let cases: [&[u8]; 9] = [
            &[],
            &[0, 0, 1],
            &[0, 0, 0, 1, 5],
            &[7, 0, 0, 1],
            &[1, 0, 1, 0, 0, 2, 0, 0, 1],
            &[0, 1, 0, 0, 0, 0, 1],
            &[9, 9, 9, 9, 0, 0],
            &[0, 0, 3, 0, 0, 1],
            &[5, 0, 0, 2, 0, 0, 0, 1],
        ];
        for case in cases {
            assert_eq!(find_start_code(case), naive(case), "{case:?}");
        }
        // Every 0/1/2 sequence of up to 8 bytes.
        for length in 0..=8_u32 {
            for code in 0..3_u32.pow(length) {
                let bytes: Vec<u8> = (0..length)
                    .map(|digit| u8::try_from(code / 3_u32.pow(digit) % 3).unwrap())
                    .collect();
                assert_eq!(find_start_code(&bytes), naive(&bytes), "{bytes:?}");
            }
        }
    }

    #[test]
    fn unescape_drops_emulation_prevention_and_refuses_forbidden_bytes() {
        let mut out = Vec::new();
        assert!(unescape(&[1, 0, 0, 3, 1, 0, 0, 3], &mut out));
        assert_eq!(out, [1, 0, 0, 1, 0, 0]);
        assert!(!unescape(&[1, 0, 0, 0, 5], &mut out));
        assert!(!unescape(&[1, 0, 0, 2], &mut out));
    }

    #[test]
    fn the_stop_bit_is_the_last_set_bit() {
        assert_eq!(Bits::new(&[0x80]).stop_bit(), Some(0));
        assert_eq!(Bits::new(&[0xff, 0x10, 0]).stop_bit(), Some(11));
        assert_eq!(Bits::new(&[0, 0]).stop_bit(), None);
    }

    /// The VPS of `hevc_main8.hevc` (x265, one layer, no timing).
    const VPS: [u8; 24] = [
        0x40, 0x01, 0x0c, 0x01, 0xff, 0xff, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x03, 0x00, 0x3c, 0x95, 0x94, 0x09,
    ];

    fn annex_b(units: &[&[u8]]) -> Vec<u8> {
        units
            .iter()
            .flat_map(|unit| [0, 0, 0, 1].iter().chain(unit.iter()).copied())
            .collect()
    }

    #[test]
    fn a_truncated_vps_is_refused_only_when_the_packet_is_whole() {
        let mut crop = CropScanner::default();
        let slice: &[u8] = &[0x26, 0x01, 0xaf, 0x80];
        let good = annex_b(&[&VPS, slice]);
        assert_eq!(
            screen(Codec::Hevc, &good, true, &mut crop).unwrap().ready,
            good.len()
        );
        // The 26-byte packet that faulted the parser: a VPS cut inside its
        // profile_tier_level().
        let cut = annex_b(&[&VPS[..22]]);
        assert!(matches!(
            screen(Codec::Hevc, &cut, true, &mut crop),
            Err(Error::InvalidPacket { .. })
        ));
        let cut_then_slice = annex_b(&[&VPS[..22], slice]);
        assert!(screen(Codec::Hevc, &cut_then_slice, false, &mut crop).is_err());
        // Without the end-of-picture flag a VPS at the end may continue in
        // the next packet: it is held back, from its start code.
        assert_eq!(
            screen(Codec::Hevc, &cut, false, &mut crop).unwrap().ready,
            0
        );
        // Fed in two pieces, the VPS waits for the second and then passes
        // with the slice after it.
        let mut held = HeldBack::default();
        let first = held
            .split(&cut, |data| screen(Codec::Hevc, data, false, &mut crop))
            .unwrap();
        assert!(first.is_empty());
        let mut second = VPS[22..].to_vec();
        second.extend_from_slice(&annex_b(&[slice]));
        let ready = held
            .split(&second, |data| screen(Codec::Hevc, data, false, &mut crop))
            .unwrap()
            .to_vec();
        assert_eq!(ready, annex_b(&[&VPS, slice]));
        // A VPS held back and then cut short by the next start code fails.
        let mut held = HeldBack::default();
        held.split(&cut, |data| screen(Codec::Hevc, data, false, &mut crop))
            .unwrap();
        let next = annex_b(&[slice]);
        assert!(
            held.split(&next, |data| screen(Codec::Hevc, data, false, &mut crop))
                .is_err()
        );
    }

    #[test]
    fn split_start_codes_are_held_back() {
        let mut crop = CropScanner::default();
        let packet = [0, 0, 1, 0x26, 0x01, 0xaf, 0, 0];
        assert_eq!(
            screen(Codec::Hevc, &packet, false, &mut crop)
                .unwrap()
                .ready,
            6
        );
        assert_eq!(
            screen(Codec::Hevc, &packet[..3], false, &mut crop)
                .unwrap()
                .ready,
            0
        );
        assert_eq!(
            screen(Codec::Hevc, &packet, true, &mut crop).unwrap().ready,
            8
        );
    }
}
