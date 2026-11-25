//! Elementary-stream parsing shared by the encoders, decoders and the
//! backend hand-over: Annex B framing (here), the bit-level primitives
//! ([`bits`]), H.264 / H.265 SPS bit depths ([`h264`], [`hevc`]), AV1 OBUs
//! and sequence headers ([`av1`]), VP9 keyframe headers ([`vp9`]) and the
//! HDR metadata streams carry ([`hdr`]).

use bytes::Bytes;

pub mod av1;
pub mod bits;
pub mod h264;
pub mod hdr;
pub mod hevc;
pub mod vp9;
pub mod vui;

/// The 4-byte Annex B start code.
pub const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// Offset of the next `00 00 01` start code prefix in `buf`.
///
/// Skips three bytes at a time where it can: a byte above 1 can neither
/// end a start code nor be one of its zeros, so the next candidate end is
/// three bytes on. Slice data rarely holds zeros, so this reads about a
/// third of it.
#[must_use]
pub fn find_start_code(buf: &[u8]) -> Option<usize> {
    let mut end = 2;
    while let Some(&byte) = buf.get(end) {
        match byte {
            0 => end += 1,
            1 if buf[end - 2] == 0 && buf[end - 1] == 0 => return Some(end - 2),
            _ => end += 3,
        }
    }
    None
}

/// The NAL units of an Annex B buffer, start codes stripped, without
/// allocating. Bytes before the first start code are not a unit; the zero
/// byte of a 4-byte start code is not part of the unit before it.
#[must_use]
pub fn nal_units(buf: &[u8]) -> NalUnits<'_> {
    NalUnits {
        buf,
        next: find_start_code(buf).map(|start| start + 3),
    }
}

/// The iterator of [`nal_units`].
#[derive(Debug, Clone)]
pub struct NalUnits<'a> {
    buf: &'a [u8],
    /// Where the next unit begins, after its start code.
    next: Option<usize>,
}

impl<'a> Iterator for NalUnits<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        let begin = self.next?;
        let rest = self.buf.get(begin..).unwrap_or_default();
        let Some(offset) = find_start_code(rest) else {
            self.next = None;
            return Some(rest);
        };
        let start = begin + offset;
        self.next = Some(start + 3);
        let end = if offset > 0 && self.buf[start - 1] == 0 {
            start - 1
        } else {
            start
        };
        self.buf.get(begin..end)
    }
}

/// The NAL units of an Annex B buffer, start codes stripped. Bytes
/// before the first start code are not a unit.
#[must_use]
pub fn split_annex_b(buf: &[u8]) -> Vec<&[u8]> {
    nal_units(buf).collect()
}

/// The NAL units of an Annex B buffer rewritten with a 4-byte big-endian
/// length in front of each (AVCC / HVCC framing, what `VideoToolbox` and
/// MP4 carry).
#[must_use]
pub fn annex_b_to_length_prefixed(buf: &[u8]) -> Vec<u8> {
    let units = split_annex_b(buf);
    let mut out = Vec::with_capacity(buf.len() + units.len());
    for unit in units {
        let len = u32::try_from(unit.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(unit);
    }
    out
}

/// Length-prefixed NAL units (`length_size` bytes each, big-endian:
/// 1, 2 or 4) rewritten as Annex B with 4-byte start codes; `None` when
/// a length runs past the end of `buf` or `length_size` is invalid.
#[must_use]
pub fn length_prefixed_to_annex_b(buf: &[u8], length_size: usize) -> Option<Vec<u8>> {
    if !matches!(length_size, 1 | 2 | 4) {
        return None;
    }
    let mut out = Vec::with_capacity(buf.len() + buf.len() / 64 + 8);
    let mut rest = buf;
    while !rest.is_empty() {
        let (len, tail) = rest.split_at_checked(length_size)?;
        let len = len
            .iter()
            .fold(0usize, |acc, &byte| (acc << 8) | usize::from(byte));
        let (unit, tail) = tail.split_at_checked(len)?;
        out.extend_from_slice(&START_CODE);
        out.extend_from_slice(unit);
        rest = tail;
    }
    Some(out)
}

/// Which NAL unit types carry the parameter sets a decoder needs before
/// it can decode a slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NalSyntax {
    /// H.264: SPS (7), PPS (8); type is the low 5 bits of byte 0.
    H264,
    /// H.265: VPS (32), SPS (33), PPS (34); type is bits 1-6 of byte 0.
    H265,
}

impl NalSyntax {
    /// The parameter-set slot `nal` fills, if it is one.
    fn slot(self, nal: &[u8]) -> Option<usize> {
        let header = *nal.first()?;
        match self {
            Self::H264 => match header & 0x1f {
                7 => Some(0),
                8 => Some(1),
                _ => None,
            },
            Self::H265 => match (header >> 1) & 0x3f {
                32 => Some(0),
                33 => Some(1),
                34 => Some(2),
                _ => None,
            },
        }
    }

    /// What `nal` is to access unit framing (H.264 §7.4.1.2.3, H.265
    /// §7.4.2.4.4); `None` for an empty unit.
    #[must_use]
    pub fn role(self, nal: &[u8]) -> Option<NalRole> {
        let header = *nal.first()?;
        // The first bit of the slice header: H.264 `first_mb_in_slice`
        // is 0 exactly when its ue(v) code is the single bit 1; H.265
        // `first_slice_segment_in_pic_flag`.
        let first_bit_at = |at: usize| nal.get(at).is_some_and(|byte| byte & 0x80 != 0);
        let slice = |first: bool| {
            if first {
                NalRole::FirstSlice
            } else {
                NalRole::Slice
            }
        };
        Some(match self {
            Self::H264 => match header & 0x1f {
                1 | 2 | 5 => slice(first_bit_at(1)),
                3 | 4 | 19..=21 => NalRole::Slice,
                6..=9 | 13..=18 => NalRole::Prefix,
                _ => NalRole::Suffix,
            },
            Self::H265 => {
                let layer = nal.get(1).map_or(0, |second| {
                    (u16::from(header & 1) << 5) | u16::from(second >> 3)
                });
                match (header >> 1) & 0x3f {
                    0..=31 => slice(layer == 0 && first_bit_at(2)),
                    32..=35 | 39 | 41..=44 | 48..=55 => NalRole::Prefix,
                    _ => NalRole::Suffix,
                }
            }
        })
    }
}

/// What a NAL unit is to access unit framing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NalRole {
    /// Starts an access unit when it follows a slice: an access unit
    /// delimiter, a parameter set, a prefix SEI.
    Prefix,
    /// The first slice of a (base layer) picture.
    FirstSlice,
    /// Any other slice.
    Slice,
    /// Belongs to the access unit it follows: end of sequence or stream,
    /// filler data, a suffix SEI.
    Suffix,
}

/// How the NAL units of one packet split into access units, counted while
/// walking them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PacketShape {
    /// The packet's first unit starts an access unit (rather than
    /// continuing one from the packet before).
    pub starts_access_unit: bool,
    /// Access units that start in the packet.
    pub access_units: usize,
    /// Pictures (first slices) in the packet.
    pub pictures: usize,
    /// NAL units in the packet.
    pub units: usize,
    after_slice: bool,
    /// The packet opens with bytes of a NAL unit begun in the packet
    /// before (anything but zeros ahead of its first start code).
    leading: bool,
}

impl PacketShape {
    /// Nothing counted yet of `packet`, whose units are added next.
    #[must_use]
    pub fn new(packet: &[u8]) -> Self {
        let head = find_start_code(packet).map_or(packet, |start| &packet[..start]);
        Self {
            leading: head.iter().any(|&byte| byte != 0),
            ..Self::default()
        }
    }

    /// The packet carries on an access unit of the packet before: it opens
    /// inside a NAL unit, or with a unit that does not start one.
    #[must_use]
    pub fn continues(&self) -> bool {
        self.leading || (self.units > 0 && !self.starts_access_unit)
    }

    /// Counts the next NAL unit of the packet.
    pub fn add(&mut self, role: NalRole) {
        let starts = match role {
            NalRole::FirstSlice => true,
            NalRole::Prefix => self.units == 0 || self.after_slice,
            NalRole::Slice | NalRole::Suffix => false,
        };
        if self.units == 0 {
            self.starts_access_unit = starts && !self.leading;
        }
        // A first slice right after a prefix is the same access unit.
        if starts && (role == NalRole::Prefix || self.units == 0 || self.after_slice) {
            self.access_units += 1;
        }
        self.pictures += usize::from(role == NalRole::FirstSlice);
        self.after_slice = matches!(role, NalRole::FirstSlice | NalRole::Slice);
        self.units += 1;
    }

    /// The shape of `packet`, walked on its own.
    #[must_use]
    pub fn of(packet: &[u8], syntax: NalSyntax) -> Self {
        let mut shape = Self::new(packet);
        for role in nal_units(packet).filter_map(|nal| syntax.role(nal)) {
            shape.add(role);
        }
        shape
    }

    /// The packet is exactly one access unit with one picture, starting
    /// at its first byte. Whether it also ends there no packet can tell on
    /// its own.
    #[must_use]
    pub fn is_one_access_unit(&self) -> bool {
        self.starts_access_unit && self.access_units == 1 && self.pictures == 1
    }
}

/// The latest parameter sets seen in a stream. A packet a backend
/// refuses may rely on parameter sets sent packets earlier, so they are
/// replayed in front of it when another backend takes over.
#[derive(Debug, Clone)]
pub struct ParameterSets {
    syntax: NalSyntax,
    sets: [Option<Bytes>; 3],
}

impl ParameterSets {
    #[must_use]
    /// No parameter sets remembered yet.
    pub fn new(syntax: NalSyntax) -> Self {
        Self {
            syntax,
            sets: [None, None, None],
        }
    }

    /// Records the parameter sets in `packet`.
    pub fn remember(&mut self, packet: &[u8]) {
        for nal in nal_units(packet) {
            self.remember_nal(nal);
        }
    }

    /// Records `nal` when it is a parameter set (copied only when it
    /// differs from the one remembered).
    pub fn remember_nal(&mut self, nal: &[u8]) {
        if let Some(slot) = self.syntax.slot(nal)
            && self.sets[slot].as_deref() != Some(nal)
        {
            self.sets[slot] = Some(Bytes::copy_from_slice(nal));
        }
    }

    /// `packet` with the remembered parameter sets in front of it.
    #[must_use]
    pub fn prepend(&self, packet: &Bytes) -> Bytes {
        if self.sets.iter().all(Option::is_none) {
            return packet.clone();
        }
        let mut out = Vec::with_capacity(packet.len() + 128);
        for set in self.sets.iter().flatten() {
            out.extend_from_slice(&START_CODE);
            out.extend_from_slice(set);
        }
        out.extend_from_slice(packet);
        Bytes::from(out)
    }

    /// The remembered sets as one Annex B buffer (SPS / PPS for H.264,
    /// VPS / SPS / PPS for H.265), empty when none were seen.
    #[must_use]
    pub fn annex_b(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for set in self.sets.iter().flatten() {
            out.extend_from_slice(&START_CODE);
            out.extend_from_slice(set);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annex_b_units_are_split_at_three_and_four_byte_start_codes() {
        let stream = [
            0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 0, 0, 0, 1, 0x65, 4,
        ];
        let units = split_annex_b(&stream);
        assert_eq!(units, [&[0x67, 1, 2][..], &[0x68, 3][..], &[0x65, 4][..]]);
        assert!(split_annex_b(&[1, 2, 3, 4]).is_empty());
    }

    #[test]
    fn the_latest_h264_parameter_sets_are_replayed_in_front_of_a_packet() {
        let mut sets = ParameterSets::new(NalSyntax::H264);
        sets.remember(&[0, 0, 0, 1, 0x67, 1, 0, 0, 1, 0x68, 2, 0, 0, 1, 0x65, 9]);
        sets.remember(&[0, 0, 1, 0x67, 3]);
        sets.remember(&[0, 0, 1, 0x41, 7]);
        let replay = sets.prepend(&Bytes::from_static(&[0, 0, 1, 0x41, 8]));
        assert_eq!(
            &replay[..],
            &[0, 0, 0, 1, 0x67, 3, 0, 0, 0, 1, 0x68, 2, 0, 0, 1, 0x41, 8]
        );
    }

    #[test]
    fn h265_parameter_sets_are_vps_sps_pps() {
        let mut sets = ParameterSets::new(NalSyntax::H265);
        // VPS 0x40, SPS 0x42, PPS 0x44, IDR_W_RADL 0x26.
        sets.remember(&[
            0, 0, 1, 0x40, 1, 0, 0, 1, 0x42, 2, 0, 0, 1, 0x44, 3, 0, 0, 1, 0x26, 9,
        ]);
        assert_eq!(
            sets.annex_b(),
            [
                0, 0, 0, 1, 0x40, 1, 0, 0, 0, 1, 0x42, 2, 0, 0, 0, 1, 0x44, 3
            ]
        );
    }

    #[test]
    fn length_prefixed_and_annex_b_framings_convert_both_ways() {
        let annex_b = [0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x65, 3];
        let avcc = annex_b_to_length_prefixed(&annex_b);
        assert_eq!(avcc, [0, 0, 0, 3, 0x67, 1, 2, 0, 0, 0, 2, 0x65, 3]);
        assert_eq!(
            length_prefixed_to_annex_b(&avcc, 4).unwrap(),
            [0, 0, 0, 1, 0x67, 1, 2, 0, 0, 0, 1, 0x65, 3]
        );
        assert!(length_prefixed_to_annex_b(&[0, 0, 0, 9, 1], 4).is_none());
        assert!(length_prefixed_to_annex_b(&avcc, 3).is_none());
    }

    #[test]
    fn nal_units_match_a_byte_by_byte_split() {
        fn naive(buf: &[u8]) -> Vec<&[u8]> {
            let starts: Vec<usize> = buf
                .windows(3)
                .enumerate()
                .filter(|(_, window)| *window == [0, 0, 1])
                .map(|(at, _)| at)
                .collect();
            let mut units = Vec::new();
            for (index, &start) in starts.iter().enumerate() {
                let begin = start + 3;
                let end = starts.get(index + 1).map_or(buf.len(), |&next| {
                    if next > begin && buf[next - 1] == 0 {
                        next - 1
                    } else {
                        next
                    }
                });
                units.push(&buf[begin..end.max(begin)]);
            }
            units
        }
        // Every 0/1/2 sequence of up to 9 bytes.
        for length in 0..=9_u32 {
            for code in 0..3_u32.pow(length) {
                let bytes: Vec<u8> = (0..length)
                    .map(|digit| u8::try_from(code / 3_u32.pow(digit) % 3).unwrap())
                    .collect();
                assert_eq!(split_annex_b(&bytes), naive(&bytes), "{bytes:?}");
            }
        }
    }

    #[test]
    fn packets_are_counted_in_access_units_and_pictures() {
        let h264 = |units: &[&[u8]]| {
            let mut out = Vec::new();
            for unit in units {
                out.extend_from_slice(&START_CODE);
                out.extend_from_slice(unit);
            }
            PacketShape::of(&out, NalSyntax::H264)
        };
        // AUD, SPS, IDR slice with first_mb_in_slice 0, a second slice.
        let (aud, sps, first, next) = (
            &[0x09, 0xf0][..],
            &[0x67, 0x64][..],
            &[0x65, 0x88][..],
            &[0x65, 0x40][..],
        );
        assert!(h264(&[aud, sps, first, next]).is_one_access_unit());
        assert!(h264(&[first, next]).is_one_access_unit());
        let two = h264(&[aud, first, aud, first]);
        assert_eq!((two.access_units, two.pictures), (2, 2));
        assert!(!two.is_one_access_unit());
        // Without delimiters a second first slice starts a second picture.
        assert!(!h264(&[first, next, first]).is_one_access_unit());
        // The second half of a picture does not start an access unit.
        assert!(!h264(&[next]).starts_access_unit);
        assert!(h264(&[next]).continues());
        let mut tail = vec![0x12, 0x34];
        tail.extend_from_slice(&START_CODE);
        tail.extend_from_slice(first);
        let tail = PacketShape::of(&tail, NalSyntax::H264);
        assert!(tail.continues() && !tail.is_one_access_unit(), "{tail:?}");
        assert!(PacketShape::of(&[0x12, 0x34], NalSyntax::H264).continues());
        assert!(!h264(&[first]).continues());
        // Parameter sets of the next access unit after a picture.
        assert!(!h264(&[first, sps]).is_one_access_unit());
        assert!(!h264(&[sps]).is_one_access_unit(), "no picture");
        // HEVC: VPS, IDR_W_RADL with first_slice_segment_in_pic_flag, a
        // dependent segment, a suffix SEI.
        let mut hevc = Vec::new();
        for unit in [
            &[0x40, 0x01][..],
            &[0x26, 0x01, 0xaf],
            &[0x26, 0x01, 0x20],
            &[0x50, 0x01],
        ] {
            hevc.extend_from_slice(&START_CODE);
            hevc.extend_from_slice(unit);
        }
        assert!(PacketShape::of(&hevc, NalSyntax::H265).is_one_access_unit());
    }

    #[test]
    fn a_packet_is_untouched_when_no_parameter_sets_were_seen() {
        let packet = Bytes::from_static(&[0, 0, 1, 0x65, 1]);
        assert_eq!(ParameterSets::new(NalSyntax::H264).prepend(&packet), packet);
    }
}
