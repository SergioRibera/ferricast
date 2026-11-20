use bytes::{BufMut, Bytes, BytesMut};
use ferricast_core::{Codec, FerricastError, Result};

const START_CODE: [u8; 4] = [0, 0, 0, 1];

pub(crate) enum Framing {
    Raw,
    LengthPrefixed(LengthPrefixed),
}

pub(crate) struct LengthPrefixed {
    codec: Codec,
    length_size: usize,
    parameter_sets: Vec<u8>,
}

impl Framing {
    /// Converts one container sample to Annex B.
    ///
    /// `sync` is the container's random-access flag (or "first sample after
    /// open/rewind"): parameter sets are injected on those samples as well as
    /// on IDR/IRAP NAL units, so open-GOP streams whose sync samples are plain
    /// I-frames with a recovery point still get SPS/PPS before decoding starts.
    /// The returned flag reports whether the sample carried an IDR/IRAP NAL.
    pub(crate) fn annex_b(&self, sample: &[u8], sync: bool) -> Result<(Bytes, bool)> {
        match self {
            Framing::Raw => Ok((Bytes::copy_from_slice(sample), false)),
            Framing::LengthPrefixed(prefixed) => prefixed.annex_b(sample, sync),
        }
    }
}

impl LengthPrefixed {
    pub(crate) fn from_avcc(config: &[u8]) -> Result<Self> {
        parse_avcc(config).ok_or_else(|| FerricastError::Demux("malformed avcC".into()))
    }

    pub(crate) fn from_hvcc(config: &[u8]) -> Result<Self> {
        parse_hvcc(config).ok_or_else(|| FerricastError::Demux("malformed hvcC".into()))
    }

    fn annex_b(&self, sample: &[u8], sync: bool) -> Result<(Bytes, bool)> {
        let nals = self.split(sample)?;
        let idr = nals.iter().any(|nal| self.is_idr(nal));
        let mut out = BytesMut::with_capacity(sample.len() + self.parameter_sets.len() + 8);
        let mut rest = nals.as_slice();
        // An access unit delimiter must stay the first NAL of the access
        // unit, so the parameter sets go right after it.
        if let Some((aud, tail)) = rest.split_first()
            && self.is_aud(aud)
        {
            out.put_slice(&START_CODE);
            out.put_slice(aud);
            rest = tail;
        }
        if idr || sync {
            out.put_slice(&self.parameter_sets);
        }
        for nal in rest {
            out.put_slice(&START_CODE);
            out.put_slice(nal);
        }
        Ok((out.freeze(), idr))
    }

    fn split<'a>(&self, sample: &'a [u8]) -> Result<Vec<&'a [u8]>> {
        let mut nals = Vec::new();
        let mut rest = sample;
        while !rest.is_empty() {
            let truncated = || FerricastError::Demux("truncated NAL unit".into());
            let (prefix, tail) = rest
                .split_at_checked(self.length_size)
                .ok_or_else(truncated)?;
            let len = prefix.iter().fold(0usize, |n, b| n << 8 | usize::from(*b));
            let (nal, tail) = tail.split_at_checked(len).ok_or_else(truncated)?;
            if !nal.is_empty() {
                nals.push(nal);
            }
            rest = tail;
        }
        Ok(nals)
    }

    fn is_idr(&self, nal: &[u8]) -> bool {
        let Some(header) = nal.first() else {
            return false;
        };
        match self.codec {
            Codec::H264 => header & 0x1f == 5,
            Codec::H265 => (16..=21).contains(&((header >> 1) & 0x3f)),
            _ => false,
        }
    }

    fn is_aud(&self, nal: &[u8]) -> bool {
        let Some(header) = nal.first() else {
            return false;
        };
        match self.codec {
            Codec::H264 => header & 0x1f == 9,
            Codec::H265 => (header >> 1) & 0x3f == 35,
            _ => false,
        }
    }
}

fn parse_avcc(config: &[u8]) -> Option<LengthPrefixed> {
    let length_size = usize::from(config.get(4)? & 0x03) + 1;
    let mut rest = config.get(5..)?;
    let mut parameter_sets = Vec::new();
    let sps_count = usize::from(rest.first()? & 0x1f);
    rest = &rest[1..];
    take_sets(&mut rest, sps_count, &mut parameter_sets)?;
    let pps_count = usize::from(*rest.first()?);
    rest = &rest[1..];
    take_sets(&mut rest, pps_count, &mut parameter_sets)?;
    Some(LengthPrefixed {
        codec: Codec::H264,
        length_size,
        parameter_sets,
    })
}

fn parse_hvcc(config: &[u8]) -> Option<LengthPrefixed> {
    let length_size = usize::from(config.get(21)? & 0x03) + 1;
    let array_count = usize::from(*config.get(22)?);
    let mut rest = config.get(23..)?;
    let mut parameter_sets = Vec::new();
    for _ in 0..array_count {
        let count = usize::from(u16::from_be_bytes(rest.get(1..3)?.try_into().ok()?));
        rest = &rest[3..];
        take_sets(&mut rest, count, &mut parameter_sets)?;
    }
    Some(LengthPrefixed {
        codec: Codec::H265,
        length_size,
        parameter_sets,
    })
}

fn take_sets(rest: &mut &[u8], count: usize, out: &mut Vec<u8>) -> Option<()> {
    for _ in 0..count {
        let len = usize::from(u16::from_be_bytes(rest.get(..2)?.try_into().ok()?));
        let set = rest.get(2..2 + len)?;
        out.extend_from_slice(&START_CODE);
        out.extend_from_slice(set);
        *rest = &rest[2 + len..];
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPS: [u8; 3] = [0x67, 0x42, 0x00];
    const PPS: [u8; 2] = [0x68, 0xCE];

    fn avcc(length_size_minus_one: u8) -> Vec<u8> {
        let mut config = vec![1, 0x42, 0, 0x1e, 0xFC | length_size_minus_one, 0xE1, 0, 3];
        config.extend_from_slice(&SPS);
        config.extend_from_slice(&[1, 0, 2]);
        config.extend_from_slice(&PPS);
        config
    }

    fn annex_b(nals: &[&[u8]]) -> Vec<u8> {
        nals.iter()
            .flat_map(|nal| START_CODE.iter().chain(nal.iter()))
            .copied()
            .collect()
    }

    #[test]
    fn avcc_parameter_sets_become_annex_b() {
        let prefixed = LengthPrefixed::from_avcc(&avcc(3)).unwrap();
        assert_eq!(prefixed.length_size, 4);
        assert_eq!(prefixed.parameter_sets, annex_b(&[&SPS, &PPS]));
    }

    #[test]
    fn avcc_without_its_pps_is_rejected() {
        let mut config = avcc(3);
        config.truncate(config.len() - 4);
        assert!(LengthPrefixed::from_avcc(&config).is_err());
    }

    #[test]
    fn non_idr_sample_is_converted_without_parameter_sets() {
        let prefixed = LengthPrefixed::from_avcc(&avcc(3)).unwrap();
        let slice = [0x41, 9, 9];
        let sample = [&[0, 0, 0, 3][..], &slice].concat();
        let (data, idr) = prefixed.annex_b(&sample, false).unwrap();
        assert!(!idr);
        assert_eq!(data, annex_b(&[&slice]));
    }

    #[test]
    fn container_sync_sample_gets_parameter_sets_without_an_idr() {
        let prefixed = LengthPrefixed::from_avcc(&avcc(3)).unwrap();
        let i_slice = [0x41, 9, 9];
        let sample = [&[0, 0, 0, 3][..], &i_slice].concat();
        let (data, idr) = prefixed.annex_b(&sample, true).unwrap();
        assert!(!idr);
        assert_eq!(data, annex_b(&[&SPS, &PPS, &i_slice]));
    }

    #[test]
    fn idr_sample_is_preceded_by_parameter_sets() {
        let prefixed = LengthPrefixed::from_avcc(&avcc(3)).unwrap();
        let idr_slice = [0x65, 1, 2, 3];
        let sample = [&[0, 0, 0, 4][..], &idr_slice].concat();
        let (data, idr) = prefixed.annex_b(&sample, false).unwrap();
        assert!(idr);
        assert_eq!(data, annex_b(&[&SPS, &PPS, &idr_slice]));
    }

    #[test]
    fn parameter_sets_follow_a_leading_aud() {
        let prefixed = LengthPrefixed::from_avcc(&avcc(3)).unwrap();
        let aud = [0x09, 0xF0];
        let idr_slice = [0x65, 1, 2, 3];
        let sample = [&[0, 0, 0, 2][..], &aud, &[0, 0, 0, 4], &idr_slice].concat();
        let (data, idr) = prefixed.annex_b(&sample, false).unwrap();
        assert!(idr);
        assert_eq!(data, annex_b(&[&aud, &SPS, &PPS, &idr_slice]));
    }

    #[test]
    fn hevc_aud_is_recognised() {
        let prefixed = LengthPrefixed {
            codec: Codec::H265,
            length_size: 4,
            parameter_sets: Vec::new(),
        };
        assert!(prefixed.is_aud(&[35 << 1, 1, 0x50]));
        assert!(!prefixed.is_aud(&[19 << 1, 1]));
    }

    #[test]
    fn two_byte_length_prefixes_are_honoured() {
        let prefixed = LengthPrefixed::from_avcc(&avcc(1)).unwrap();
        let slice = [0x41, 7];
        let sample = [&[0, 2][..], &slice].concat();
        let (data, _) = prefixed.annex_b(&sample, false).unwrap();
        assert_eq!(data, annex_b(&[&slice]));
    }

    #[test]
    fn nal_longer_than_the_sample_is_rejected() {
        let prefixed = LengthPrefixed::from_avcc(&avcc(3)).unwrap();
        assert!(prefixed.annex_b(&[0, 0, 0, 9, 0x41], false).is_err());
    }

    #[test]
    fn hevc_irap_nal_types_count_as_keyframes() {
        let prefixed = LengthPrefixed {
            codec: Codec::H265,
            length_size: 4,
            parameter_sets: Vec::new(),
        };
        assert!(prefixed.is_idr(&[19 << 1, 1]));
        assert!(prefixed.is_idr(&[21 << 1, 1]));
        assert!(!prefixed.is_idr(&[1 << 1, 1]));
        assert!(!prefixed.is_idr(&[32 << 1, 1]));
    }

    #[test]
    fn hvcc_arrays_are_flattened_in_order() {
        let mut config = vec![0; 22];
        config[21] = 0x03;
        config.push(2);
        config.extend_from_slice(&[32, 0, 1, 0, 2, 0x40, 1]);
        config.extend_from_slice(&[33, 0, 1, 0, 2, 0x42, 1]);
        let prefixed = LengthPrefixed::from_hvcc(&config).unwrap();
        assert_eq!(prefixed.length_size, 4);
        assert_eq!(prefixed.parameter_sets, annex_b(&[&[0x40, 1], &[0x42, 1]]));
    }
}
