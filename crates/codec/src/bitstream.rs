//! Annex B byte-stream helpers shared by the H.264 / H.265 encoders,
//! decoders and the backend hand-over.

use bytes::Bytes;

/// The 4-byte Annex B start code.
pub const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// The NAL units of an Annex B buffer, start codes stripped. Bytes
/// before the first start code are not a unit.
#[must_use]
pub fn split_annex_b(buf: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut start = None;
    let mut index = 0;
    while index + 3 <= buf.len() {
        if buf[index] == 0 && buf[index + 1] == 0 && buf[index + 2] == 1 {
            if let Some(begin) = start {
                let end = if index > begin && buf[index - 1] == 0 {
                    index - 1
                } else {
                    index
                };
                out.push(&buf[begin..end]);
            }
            index += 3;
            start = Some(index);
            continue;
        }
        index += 1;
    }
    if let Some(begin) = start {
        out.push(&buf[begin..]);
    }
    out
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
        for nal in split_annex_b(packet) {
            if let Some(slot) = self.syntax.slot(nal) {
                self.sets[slot] = Some(Bytes::copy_from_slice(nal));
            }
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
    fn a_packet_is_untouched_when_no_parameter_sets_were_seen() {
        let packet = Bytes::from_static(&[0, 0, 1, 0x65, 1]);
        assert_eq!(ParameterSets::new(NalSyntax::H264).prepend(&packet), packet);
    }
}
