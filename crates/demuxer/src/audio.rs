//! Audio track descriptions shared by the container parsers: AAC
//! configuration (`AudioSpecificConfig`) and the ADTS framing the AAC
//! decoder reads, and Opus identification headers.

use bytes::{BufMut, Bytes, BytesMut};
use ferricast_core::frame::AacChannelConfig;
use ferricast_core::{AudioCodec, FerricastError, Result};

/// Sampling frequencies indexed by `samplingFrequencyIndex`
/// (ISO/IEC 14496-3 §1.6.3.4).
const AAC_RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];
/// Escape value of `samplingFrequencyIndex`: a 24-bit rate follows.
const AAC_EXPLICIT_RATE: u8 = 15;
/// Audio object types that carry SBR / PS on top of a core codec.
const AOT_SBR: u8 = 5;
const AOT_PS: u8 = 29;
/// ADTS header length without CRC.
const ADTS_HEADER: usize = 7;
/// Largest frame an ADTS header can describe (13-bit length).
const ADTS_MAX_FRAME: usize = (1 << 13) - 1;
/// Samples in one AAC-LC access unit.
#[cfg_attr(
    not(feature = "matroska"),
    allow(
        dead_code,
        reason = "only the Matroska demuxer times packets by their content"
    )
)]
pub(crate) const AAC_FRAME_SAMPLES: u64 = 1024;
/// Opus always decodes at 48 kHz (RFC 7845 §5.1).
pub(crate) const OPUS_RATE: u32 = 48_000;

/// An audio elementary stream a demuxer found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioStream {
    pub codec: AudioCodec,
    /// Output sample rate of the decoder: the SBR rate for HE-AAC that
    /// signals it explicitly (implicitly signalled SBR is only found by
    /// the decoder), always 48 kHz for Opus.
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_us: Option<u64>,
}

/// How the packets of an audio track are handed out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AudioFormat {
    /// Raw AAC access units, wrapped into ADTS frames.
    Aac(AacConfig),
    /// Opus packets, passed through.
    Opus,
}

impl AudioFormat {
    pub(crate) fn codec(self) -> AudioCodec {
        match self {
            AudioFormat::Aac(_) => AudioCodec::Aac,
            AudioFormat::Opus => AudioCodec::Opus,
        }
    }

    /// The bytes the decoder receives for one container sample.
    pub(crate) fn packet(self, sample: &[u8]) -> Result<Bytes> {
        match self {
            AudioFormat::Aac(config) => config.adts(sample),
            AudioFormat::Opus => Ok(Bytes::copy_from_slice(sample)),
        }
    }

    /// Duration of one packet, when it can be told from its bytes.
    #[cfg_attr(
        not(feature = "matroska"),
        allow(
            dead_code,
            reason = "only the Matroska demuxer times packets by their content"
        )
    )]
    pub(crate) fn packet_duration_us(self, sample: &[u8]) -> Option<u64> {
        match self {
            AudioFormat::Aac(config) => {
                Some(AAC_FRAME_SAMPLES * 1_000_000 / u64::from(config.sample_rate.max(1)))
            }
            AudioFormat::Opus => opus_packet_duration_us(sample),
        }
    }
}

/// The parts of an `AudioSpecificConfig` an ADTS header carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AacConfig {
    /// Audio object type of the core codec (2 = AAC-LC).
    object_type: u8,
    rate_index: u8,
    pub(crate) sample_rate: u32,
    pub(crate) channel_config: AacChannelConfig,
    /// What a decoder that handles SBR and PS outputs: twice the core
    /// rate for HE-AAC, stereo for HE-AAC v2 (whose core is mono).
    output_rate: u32,
    output_channels: u16,
}

impl AacConfig {
    /// Parses an `AudioSpecificConfig` (ISO/IEC 14496-3 §1.6.2.1).
    /// HE-AAC configurations resolve to their AAC-LC core, which is
    /// what an ADTS stream of the same audio declares.
    pub(crate) fn from_asc(asc: &[u8]) -> Result<Self> {
        let malformed = || FerricastError::Demux("malformed AAC AudioSpecificConfig".into());
        let mut bits = Bits::new(asc);
        let mut object_type = bits.object_type().ok_or_else(malformed)?;
        let (mut rate_index, mut sample_rate) = bits.rate().ok_or_else(malformed)?;
        let code = bits.read(4).ok_or_else(malformed)? as u8;
        let channel_config = AacChannelConfig::new(code).ok_or_else(|| {
            FerricastError::Demux(format!("AAC channel configuration {code} is not supported"))
        })?;
        let mut output_rate = None;
        let mut output_channels = channel_config.channels();
        if object_type == AOT_SBR || object_type == AOT_PS {
            // The extension rate belongs to the SBR layer; the core
            // keeps the rate read above.
            output_rate = Some(bits.rate().ok_or_else(malformed)?.1);
            if object_type == AOT_PS {
                output_channels = 2;
            }
            object_type = bits.object_type().ok_or_else(malformed)?;
        }
        if !(1..=4).contains(&object_type) {
            return Err(FerricastError::Demux(format!(
                "AAC object type {object_type} is not supported"
            )));
        }
        if rate_index == AAC_EXPLICIT_RATE {
            rate_index = nearest_rate_index(sample_rate);
            sample_rate = AAC_RATES[usize::from(rate_index)];
        }
        Ok(Self {
            object_type,
            rate_index,
            sample_rate,
            channel_config,
            output_rate: output_rate.unwrap_or(sample_rate),
            output_channels,
        })
    }

    /// AAC-LC at `sample_rate` with `channels`, for containers that
    /// name the codec without carrying its configuration.
    #[cfg_attr(
        not(feature = "matroska"),
        allow(
            dead_code,
            reason = "only Matroska names AAC without its configuration"
        )
    )]
    pub(crate) fn low_complexity(sample_rate: u32, channels: u16) -> Result<Self> {
        let channel_config = AacChannelConfig::from_channels(channels).ok_or_else(|| {
            FerricastError::Demux(format!("AAC with {channels} channels is not supported"))
        })?;
        let rate_index = nearest_rate_index(sample_rate);
        let sample_rate = AAC_RATES[usize::from(rate_index)];
        Ok(Self {
            object_type: 2,
            rate_index,
            sample_rate,
            channel_config,
            output_rate: sample_rate,
            output_channels: channels,
        })
    }

    pub(crate) fn stream(self, duration_us: Option<u64>) -> AudioStream {
        AudioStream {
            codec: AudioCodec::Aac,
            sample_rate: self.output_rate,
            channels: self.output_channels,
            duration_us,
        }
    }

    /// Prefixes one raw access unit with its ADTS header.
    fn adts(self, payload: &[u8]) -> Result<Bytes> {
        let len = payload.len() + ADTS_HEADER;
        if len > ADTS_MAX_FRAME {
            return Err(FerricastError::Demux(format!(
                "AAC access unit of {} bytes does not fit an ADTS frame",
                payload.len()
            )));
        }
        let profile = self.object_type - 1;
        let channels = self.channel_config.code();
        let mut out = BytesMut::with_capacity(len);
        out.put_u8(0xFF);
        // MPEG-4, layer 0, no CRC.
        out.put_u8(0xF1);
        out.put_u8((profile << 6) | (self.rate_index << 2) | (channels >> 2));
        out.put_u8(((channels & 3) << 6) | ((len >> 11) & 3) as u8);
        out.put_u8((len >> 3) as u8);
        // Low length bits, then a buffer fullness of 0x7FF (variable rate).
        out.put_u8((((len & 7) as u8) << 5) | 0x1F);
        out.put_u8(0xFC);
        out.put_slice(payload);
        Ok(out.freeze())
    }
}

fn nearest_rate_index(rate: u32) -> u8 {
    (0u8..)
        .zip(AAC_RATES)
        .min_by_key(|&(_, r)| r.abs_diff(rate))
        .map_or(3, |(i, _)| i)
}

/// Channel count of an Opus identification header: `OpusHead` in
/// Matroska `CodecPrivate` (RFC 7845 §5.1), or the body of an MP4
/// `dOps` box, which carries the same fields after a version byte.
pub(crate) fn opus_channels(header: &[u8]) -> Option<u16> {
    let fields = header.strip_prefix(b"OpusHead").unwrap_or(header);
    fields.get(1).map(|&c| u16::from(c)).filter(|&c| c > 0)
}

/// Duration of one Opus packet from its TOC byte (RFC 6716 §3.1).
#[cfg_attr(
    not(feature = "matroska"),
    allow(
        dead_code,
        reason = "only the Matroska demuxer times packets by their content"
    )
)]
pub(crate) fn opus_packet_duration_us(packet: &[u8]) -> Option<u64> {
    let toc = *packet.first()?;
    let config = toc >> 3;
    // Frame sizes in units of 2.5 ms.
    let frame = match config {
        0..=11 => [4, 8, 16, 24][usize::from(config % 4)],
        12..=15 => [4, 8][usize::from(config % 2)],
        _ => [1, 2, 4, 8][usize::from(config % 4)],
    };
    let frames = match toc & 3 {
        0 => 1,
        1 | 2 => 2,
        _ => u64::from(packet.get(1)? & 0x3F),
    };
    Some(frames * frame * 2_500)
}

/// MSB-first bit reader over a byte slice.
struct Bits<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, at: 0 }
    }

    fn read(&mut self, count: usize) -> Option<u32> {
        let mut value = 0;
        for _ in 0..count {
            let byte = self.data.get(self.at / 8)?;
            let bit = (byte >> (7 - self.at % 8)) & 1;
            value = (value << 1) | u32::from(bit);
            self.at += 1;
        }
        Some(value)
    }

    fn object_type(&mut self) -> Option<u8> {
        match self.read(5)? as u8 {
            31 => Some(32 + self.read(6)? as u8),
            aot => Some(aot),
        }
    }

    fn rate(&mut self) -> Option<(u8, u32)> {
        let index = self.read(4)? as u8;
        if index == AAC_EXPLICIT_RATE {
            return Some((index, self.read(24)?));
        }
        Some((index, *AAC_RATES.get(usize::from(index))?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_complexity_config_is_read() {
        // AAC-LC, 48 kHz (index 3), stereo.
        let config = AacConfig::from_asc(&[0x11, 0x90]).unwrap();
        assert_eq!(config.object_type, 2);
        assert_eq!(config.sample_rate, 48_000);
        assert_eq!(config.channel_config.channels(), 2);
    }

    #[test]
    fn he_aac_resolves_to_its_core() {
        // SBR (5), core 24 kHz (index 6), stereo, extension 48 kHz
        // (index 3), core object type 2.
        let asc = [0x2B, 0x11, 0x88, 0x00];
        let config = AacConfig::from_asc(&asc).unwrap();
        assert_eq!(config.object_type, 2);
        assert_eq!(config.sample_rate, 24_000);
        assert_eq!(config.channel_config.channels(), 2);
        let stream = config.stream(None);
        assert_eq!((stream.sample_rate, stream.channels), (48_000, 2));
    }

    #[test]
    fn he_aac_v2_outputs_stereo_at_the_sbr_rate() {
        // PS (29), core 24 kHz (index 6), mono, extension 48 kHz
        // (index 3), core object type 2.
        let asc = [0xEB, 0x09, 0x88, 0x00];
        let config = AacConfig::from_asc(&asc).unwrap();
        assert_eq!(
            (config.sample_rate, config.channel_config.channels()),
            (24_000, 1)
        );
        let stream = config.stream(None);
        assert_eq!((stream.sample_rate, stream.channels), (48_000, 2));
    }

    #[test]
    fn surround_configurations_report_their_channel_count() {
        // AAC-LC, 48 kHz, channel configuration 6 (5.1), then 7 (7.1).
        let surround = AacConfig::from_asc(&[0x11, 0xB0]).unwrap();
        assert_eq!(surround.stream(None).channels, 6);
        let wide = AacConfig::from_asc(&[0x11, 0xB8]).unwrap();
        assert_eq!(wide.stream(None).channels, 8);
        let frame = wide.adts(&[0]).unwrap();
        assert_eq!(((frame[2] & 1) << 2) | (frame[3] >> 6), 7);
        let named = AacConfig::low_complexity(48_000, 8).unwrap();
        assert_eq!(named.channel_config.code(), 7);
        assert!(AacConfig::low_complexity(48_000, 7).is_err());
    }

    #[test]
    fn program_config_element_channels_are_rejected() {
        assert!(AacConfig::from_asc(&[0x11, 0x80]).is_err());
        assert!(AacConfig::from_asc(&[0x11]).is_err());
    }

    #[test]
    fn adts_header_describes_the_frame() {
        let config = AacConfig::from_asc(&[0x11, 0x90]).unwrap();
        let frame = config.adts(&[0xAB; 100]).unwrap();
        assert_eq!(frame.len(), 107);
        assert_eq!(&frame[..2], &[0xFF, 0xF1]);
        // Profile LC (1), rate index 3, channel configuration 2.
        assert_eq!(frame[2] >> 6, 1);
        assert_eq!((frame[2] >> 2) & 0xF, 3);
        let channels = ((frame[2] & 1) << 2) | (frame[3] >> 6);
        assert_eq!(channels, 2);
        let len = (usize::from(frame[3] & 3) << 11)
            | (usize::from(frame[4]) << 3)
            | usize::from(frame[5] >> 5);
        assert_eq!(len, 107);
        assert_eq!(frame[7], 0xAB);
    }

    #[test]
    fn oversized_access_units_are_rejected() {
        let config = AacConfig::low_complexity(44_100, 2).unwrap();
        assert!(config.adts(&vec![0; ADTS_MAX_FRAME]).is_err());
    }

    #[test]
    fn unlisted_rates_snap_to_the_nearest_index() {
        let config = AacConfig::low_complexity(47_000, 1).unwrap();
        assert_eq!(config.sample_rate, 48_000);
    }

    #[test]
    fn opus_headers_give_their_channel_count() {
        let mut head = b"OpusHead".to_vec();
        head.extend_from_slice(&[1, 2, 0x38, 0x01, 0x80, 0xBB, 0, 0, 0, 0, 0]);
        assert_eq!(opus_channels(&head), Some(2));
        assert_eq!(opus_channels(&[0, 1, 0x38, 0x01]), Some(1));
        assert_eq!(opus_channels(&[0, 0]), None);
    }

    #[test]
    fn opus_packet_durations_follow_the_toc() {
        // CELT 20 ms, one frame.
        assert_eq!(opus_packet_duration_us(&[31 << 3]), Some(20_000));
        // SILK 60 ms, two frames.
        assert_eq!(opus_packet_duration_us(&[(3 << 3) | 1]), Some(120_000));
        // Hybrid 10 ms, code 3 with 3 frames.
        assert_eq!(opus_packet_duration_us(&[(12 << 3) | 3, 3]), Some(30_000));
        assert_eq!(opus_packet_duration_us(&[]), None);
    }
}
