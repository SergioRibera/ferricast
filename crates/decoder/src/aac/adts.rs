//! ADTS framing (ISO/IEC 13818-7 §6.2).
//!
//! An HLS / MPEG-TS PES payload carries one or more ADTS frames back to
//! back. Symphonia's AAC decoder only understands bare
//! `raw_data_block()`s, so every frame is split off and stripped of its
//! 7-byte (or 9-byte, with CRC) header here.

use std::fmt;

use ferricast_core::frame::AacChannelConfig;

use super::Shape;

const HEADER_LEN: usize = 7;
const CRC_LEN: usize = 2;
const PROFILE_LC: u8 = 1;
const SAMPLE_RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum AdtsError {
    Truncated {
        offset: usize,
        needed: usize,
        available: usize,
    },
    SyncLost {
        offset: usize,
    },
    Layer(u8),
    FrameTooShort {
        offset: usize,
        frame_len: usize,
    },
    SampleRateIndex(u8),
    Profile(u8),
    ChannelConfig(u8),
    RawDataBlocks(u8),
}

impl fmt::Display for AdtsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated {
                offset,
                needed,
                available,
            } => write!(
                f,
                "ADTS frame at byte {offset} needs {needed} bytes, only {available} left"
            ),
            Self::SyncLost { offset } => write!(f, "ADTS sync word missing at byte {offset}"),
            Self::Layer(layer) => write!(f, "ADTS layer {layer} is not 0"),
            Self::FrameTooShort { offset, frame_len } => write!(
                f,
                "ADTS frame at byte {offset} claims {frame_len} bytes, shorter than its header"
            ),
            Self::SampleRateIndex(index) => {
                write!(f, "ADTS sampling_frequency_index {index} is reserved")
            }
            Self::Profile(profile) => write!(
                f,
                "ADTS profile {profile} is not AAC-LC (the only profile the decoder handles)"
            ),
            Self::ChannelConfig(config) => write!(
                f,
                "ADTS channel_configuration {config} is unsupported (the layout is in a PCE)"
            ),
            Self::RawDataBlocks(count) => write!(
                f,
                "ADTS frame packs {count} raw data blocks; only one per frame is supported"
            ),
        }
    }
}

impl std::error::Error for AdtsError {}

/// One AAC access unit ready for the decoder. `shape` is `None` for
/// unframed (raw) input, where the configured shape applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Unit<'a> {
    pub shape: Option<Shape>,
    pub payload: &'a [u8],
}

pub(super) fn has_sync(data: &[u8]) -> bool {
    matches!(data, [0xFF, second, ..] if second & 0xF0 == 0xF0)
}

/// Splits `data` into access units: every ADTS frame when the payload
/// starts with a sync word, otherwise the whole payload as one raw unit.
pub(super) fn units(data: &[u8]) -> Units<'_> {
    Units {
        data,
        offset: 0,
        framed: has_sync(data),
        done: data.is_empty(),
    }
}

pub(super) struct Units<'a> {
    data: &'a [u8],
    offset: usize,
    framed: bool,
    done: bool,
}

impl<'a> Iterator for Units<'a> {
    type Item = Result<Unit<'a>, AdtsError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        if !self.framed {
            self.done = true;
            return Some(Ok(Unit {
                shape: None,
                payload: self.data,
            }));
        }
        let rest = &self.data[self.offset..];
        let unit = parse_frame(rest, self.offset);
        match &unit {
            Ok((_, frame_len)) => {
                self.offset += frame_len;
                self.done = self.offset >= self.data.len();
            }
            Err(_) => self.done = true,
        }
        Some(unit.map(|(unit, _)| unit))
    }
}

fn parse_frame(data: &[u8], offset: usize) -> Result<(Unit<'_>, usize), AdtsError> {
    let truncated = |needed: usize| AdtsError::Truncated {
        offset,
        needed,
        available: data.len(),
    };
    let header: &[u8; HEADER_LEN] = data.first_chunk().ok_or_else(|| truncated(HEADER_LEN))?;
    if !has_sync(header) {
        return Err(AdtsError::SyncLost { offset });
    }
    let layer = (header[1] >> 1) & 0b11;
    if layer != 0 {
        return Err(AdtsError::Layer(layer));
    }
    let protection_absent = header[1] & 1 == 1;
    let header_len = if protection_absent {
        HEADER_LEN
    } else {
        HEADER_LEN + CRC_LEN
    };
    let profile = header[2] >> 6;
    let rate_index = (header[2] >> 2) & 0b1111;
    let channel_config = ((header[2] & 1) << 2) | (header[3] >> 6);
    let frame_len = (usize::from(header[3] & 0b11) << 11)
        | (usize::from(header[4]) << 3)
        | usize::from(header[5] >> 5);
    let raw_blocks = header[6] & 0b11;
    if profile != PROFILE_LC {
        return Err(AdtsError::Profile(profile));
    }
    let sample_rate = *SAMPLE_RATES
        .get(usize::from(rate_index))
        .ok_or(AdtsError::SampleRateIndex(rate_index))?;
    let channels = AacChannelConfig::new(channel_config)
        .ok_or(AdtsError::ChannelConfig(channel_config))?
        .channels();
    if raw_blocks != 0 {
        return Err(AdtsError::RawDataBlocks(raw_blocks + 1));
    }
    if frame_len < header_len {
        return Err(AdtsError::FrameTooShort { offset, frame_len });
    }
    let payload = data
        .get(header_len..frame_len)
        .ok_or_else(|| truncated(frame_len))?;
    let shape = Shape {
        sample_rate,
        channels,
    };
    Ok((
        Unit {
            shape: Some(shape),
            payload,
        },
        frame_len,
    ))
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// Builds an ADTS frame around `payload` (AAC-LC, no CRC unless asked).
    pub(in crate::aac) fn frame(
        rate_index: u8,
        channels: u8,
        crc: bool,
        payload: &[u8],
    ) -> Vec<u8> {
        let header_len = if crc { 9 } else { 7 };
        let frame_len = header_len + payload.len();
        let mut out = vec![
            0xFF,
            0xF0 | u8::from(!crc),
            (PROFILE_LC << 6) | (rate_index << 2) | (channels >> 2),
            ((channels & 0b11) << 6) | ((frame_len >> 11) as u8 & 0b11),
            (frame_len >> 3) as u8,
            (((frame_len & 0b111) as u8) << 5) | 0x1F,
            0xFC,
        ];
        if crc {
            out.extend([0xAB, 0xCD]);
        }
        out.extend_from_slice(payload);
        out
    }

    fn collect(data: &[u8]) -> Result<Vec<Unit<'_>>, AdtsError> {
        units(data).collect()
    }

    #[test]
    fn every_frame_in_a_pes_payload_is_split_and_stripped() {
        let mut data = frame(3, 2, false, &[1, 2, 3]);
        data.extend(frame(3, 2, true, &[4, 5]));
        let units = collect(&data).unwrap();
        let stereo = Some(Shape {
            sample_rate: 48_000,
            channels: 2,
        });
        assert_eq!(
            units,
            [
                Unit {
                    shape: stereo,
                    payload: &[1, 2, 3]
                },
                Unit {
                    shape: stereo,
                    payload: &[4, 5]
                }
            ]
        );
    }

    #[test]
    fn unframed_input_passes_through_as_one_raw_unit() {
        let units = collect(&[0x21, 0x10, 0x05]).unwrap();
        assert_eq!(
            units,
            [Unit {
                shape: None,
                payload: &[0x21, 0x10, 0x05]
            }]
        );
        assert!(collect(&[]).unwrap().is_empty());
    }

    #[test]
    fn the_header_reports_its_sample_rate_and_channels() {
        let data = frame(4, 1, false, &[9]);
        let units = collect(&data).unwrap();
        assert_eq!(
            units[0].shape,
            Some(Shape {
                sample_rate: 44_100,
                channels: 1
            })
        );
    }

    #[test]
    fn multichannel_configurations_report_their_channel_count() {
        for (config, channels) in [(3, 3), (6, 6), (7, 8)] {
            let data = frame(3, config, false, &[9]);
            let shape = collect(&data).unwrap()[0].shape.unwrap();
            assert_eq!(shape.channels, channels, "configuration {config}");
        }
    }

    #[test]
    fn a_frame_running_past_the_payload_is_truncated() {
        let mut data = frame(3, 2, false, &[1, 2, 3, 4]);
        data.truncate(9);
        assert_eq!(
            collect(&data),
            Err(AdtsError::Truncated {
                offset: 0,
                needed: 11,
                available: 9
            })
        );
    }

    #[test]
    fn garbage_after_a_frame_is_a_lost_sync() {
        let mut data = frame(3, 2, false, &[1]);
        data.extend([0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(collect(&data), Err(AdtsError::SyncLost { offset: 8 }));
    }

    #[test]
    fn unsupported_headers_are_refused() {
        let mut reserved_rate = frame(13, 2, false, &[1]);
        assert_eq!(collect(&reserved_rate), Err(AdtsError::SampleRateIndex(13)));
        reserved_rate[2] = (reserved_rate[2] & 0x3F) | (2 << 6);
        assert_eq!(collect(&reserved_rate), Err(AdtsError::Profile(2)));
        assert_eq!(
            collect(&frame(3, 0, false, &[1])),
            Err(AdtsError::ChannelConfig(0))
        );
        let mut packed = frame(3, 2, false, &[1]);
        packed[6] |= 0b01;
        assert_eq!(collect(&packed), Err(AdtsError::RawDataBlocks(2)));
        let mut short = frame(3, 2, false, &[]);
        short[4] = 0;
        short[5] = (5 << 5) | 0x1F;
        assert_eq!(
            collect(&short),
            Err(AdtsError::FrameTooShort {
                offset: 0,
                frame_len: 5
            })
        );
    }
}
