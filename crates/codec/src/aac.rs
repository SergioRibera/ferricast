//! AAC framing shared by the platform AAC encoders and decoders:
//! `AudioSpecificConfig` (ISO/IEC 14496-3 §1.6.2.1) and ADTS headers
//! (ISO/IEC 13818-7 §6.2), for AAC-LC.

use ferricast_core::AacChannelConfig;

/// The sampling frequencies `samplingFrequencyIndex` codes, by index.
pub const SAMPLE_RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

/// The `audioObjectType` of AAC-LC.
pub const OBJECT_TYPE_AAC_LC: u8 = 2;

/// Length of an ADTS header without CRC.
pub const ADTS_HEADER_LEN: usize = 7;

/// The `samplingFrequencyIndex` of `rate`, when it has one.
#[must_use]
pub fn sample_rate_index(rate: u32) -> Option<u8> {
    SAMPLE_RATES
        .iter()
        .position(|&known| known == rate)
        .and_then(|index| u8::try_from(index).ok())
}

/// The shape of an AAC-LC stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AacLc {
    sample_rate_index: u8,
    channels: AacChannelConfig,
}

impl AacLc {
    /// `None` when `sample_rate` has no index or `channels` no channel
    /// configuration.
    #[must_use]
    pub fn new(sample_rate: u32, channels: u16) -> Option<Self> {
        Some(Self {
            sample_rate_index: sample_rate_index(sample_rate)?,
            channels: AacChannelConfig::from_channels(channels)?,
        })
    }

    /// The 2-byte `AudioSpecificConfig`: object type (5 bits), sample
    /// rate index (4), channel configuration (4), then three zero bits
    /// of `GASpecificConfig`.
    #[must_use]
    pub fn audio_specific_config(self) -> [u8; 2] {
        let packed = (u16::from(OBJECT_TYPE_AAC_LC) << 11)
            | (u16::from(self.sample_rate_index) << 7)
            | (u16::from(self.channels.code()) << 3);
        packed.to_be_bytes()
    }

    /// The 7-byte ADTS header (no CRC) of a frame whose raw payload is
    /// `payload_len` bytes; `None` when the frame would exceed the
    /// 13-bit frame length.
    #[must_use]
    pub fn adts_header(self, payload_len: usize) -> Option<[u8; ADTS_HEADER_LEN]> {
        let frame_len = u16::try_from(payload_len.checked_add(ADTS_HEADER_LEN)?).ok()?;
        if frame_len >= 1 << 13 {
            return None;
        }
        let profile = OBJECT_TYPE_AAC_LC - 1;
        let channels = self.channels.code();
        // Buffer fullness 0x7FF: variable bitrate.
        Some([
            0xFF,
            0xF1,
            (profile << 6) | (self.sample_rate_index << 2) | (channels >> 2),
            ((channels & 0b11) << 6) | u8::try_from(frame_len >> 11).ok()?,
            u8::try_from((frame_len >> 3) & 0xFF).ok()?,
            (u8::try_from(frame_len & 0b111).ok()? << 5) | 0x1F,
            0xFC,
        ])
    }

    /// `payload` behind its ADTS header.
    #[must_use]
    pub fn adts_frame(self, payload: &[u8]) -> Option<Vec<u8>> {
        let header = self.adts_header(payload.len())?;
        let mut frame = Vec::with_capacity(header.len() + payload.len());
        frame.extend_from_slice(&header);
        frame.extend_from_slice(payload);
        Some(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_audio_specific_config_of_48k_stereo_is_0x1190() {
        assert_eq!(
            AacLc::new(48_000, 2).unwrap().audio_specific_config(),
            [0x11, 0x90]
        );
        assert_eq!(
            AacLc::new(44_100, 1).unwrap().audio_specific_config(),
            [0x12, 0x08]
        );
        assert!(AacLc::new(44_000, 2).is_none());
    }

    #[test]
    fn adts_headers_carry_shape_and_length() {
        let header = AacLc::new(48_000, 2).unwrap().adts_header(100).unwrap();
        // Sync, MPEG-4, no CRC; LC, index 3, channel config 2; length 107.
        assert_eq!(header, [0xFF, 0xF1, 0x4C, 0x80, 0x0D, 0x7F, 0xFC]);
        assert!(AacLc::new(48_000, 2).unwrap().adts_header(9000).is_none());
    }
}
