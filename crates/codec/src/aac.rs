//! AAC framing shared by the platform AAC encoders and decoders:
//! `AudioSpecificConfig` (ISO/IEC 14496-3 §1.6.2.1) and ADTS headers
//! (ISO/IEC 13818-7 §6.2) for AAC-LC, and the `AudioSpecificConfig`s of
//! HE-AAC v1 / v2 and AAC-ELD ([`AudioSpecificConfig`]).

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

/// The `audioObjectType` of SBR (HE-AAC v1).
pub const OBJECT_TYPE_SBR: u8 = 5;
/// The `audioObjectType` of parametric stereo (HE-AAC v2).
pub const OBJECT_TYPE_PS: u8 = 29;
/// The `audioObjectType` of ER AAC-ELD.
pub const OBJECT_TYPE_ELD: u8 = 39;

/// The parts of an `AudioSpecificConfig` a decoder is set up from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioSpecificConfig {
    /// The core coder: [`OBJECT_TYPE_AAC_LC`] under SBR / PS, or
    /// [`OBJECT_TYPE_ELD`].
    pub object_type: u8,
    /// Sample rate of the core coder.
    pub sample_rate: u32,
    /// `channelConfiguration` (0 = program config element).
    pub channel_config: u8,
    /// Output sample rate of the SBR tool, when SBR is signalled (for ELD,
    /// LD-SBR: the core rate when single-rate).
    pub sbr_sample_rate: Option<u32>,
    /// Parametric stereo is signalled (mono core, stereo output).
    pub ps: bool,
    /// `frameLengthFlag`: 960-sample AAC frames, 480-sample ELD frames.
    pub short_frames: bool,
}

impl AudioSpecificConfig {
    /// HE-AAC with explicit hierarchical signalling: SBR doubling
    /// `core_rate`, plus parametric stereo with `ps` (which needs a mono
    /// core). `None` when a rate has no index, the channel count no
    /// configuration, or `ps` is asked of a multichannel core.
    #[must_use]
    pub fn he_aac(core_rate: u32, channels: u16, ps: bool) -> Option<Self> {
        let config = AacChannelConfig::from_channels(channels)?;
        sample_rate_index(core_rate)?;
        sample_rate_index(core_rate.checked_mul(2)?)?;
        if ps && config.code() != 1 {
            return None;
        }
        Some(Self {
            object_type: OBJECT_TYPE_AAC_LC,
            sample_rate: core_rate,
            channel_config: config.code(),
            sbr_sample_rate: Some(core_rate * 2),
            ps,
            short_frames: false,
        })
    }

    /// AAC-ELD without SBR, 480-sample frames (`short_frames`) or 512.
    #[must_use]
    pub fn eld(sample_rate: u32, channels: u16, short_frames: bool) -> Option<Self> {
        sample_rate_index(sample_rate)?;
        Some(Self {
            object_type: OBJECT_TYPE_ELD,
            sample_rate,
            channel_config: AacChannelConfig::from_channels(channels)?.code(),
            sbr_sample_rate: None,
            ps: false,
            short_frames,
        })
    }

    /// Samples per channel the core decodes from one access unit.
    #[must_use]
    pub fn core_frame_samples(self) -> u32 {
        match (self.object_type, self.short_frames) {
            (OBJECT_TYPE_ELD, true) => 480,
            (OBJECT_TYPE_ELD, false) => 512,
            (_, true) => 960,
            (_, false) => 1024,
        }
    }

    /// The decoded sample rate.
    #[must_use]
    pub fn output_sample_rate(self) -> u32 {
        self.sbr_sample_rate.unwrap_or(self.sample_rate)
    }

    /// Samples per channel decoded from one access unit.
    #[must_use]
    pub fn output_frame_samples(self) -> u32 {
        let factor = self.output_sample_rate() / self.sample_rate.max(1);
        self.core_frame_samples() * factor.max(1)
    }

    /// The decoded channel count; `None` for a program config element.
    #[must_use]
    pub fn output_channels(self) -> Option<u16> {
        let channels = AacChannelConfig::new(self.channel_config)?.channels();
        Some(if self.ps && channels == 1 {
            2
        } else {
            channels
        })
    }

    /// Reads an `AudioSpecificConfig`: plain AAC-LC, explicit
    /// hierarchical SBR / PS signalling (object types 5 / 29), the
    /// backward-compatible extension after `GASpecificConfig`, and ELD.
    /// `None` for truncated or other configurations.
    #[must_use]
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        let mut bits = Bits::new(bytes);
        let mut object_type = bits.object_type()?;
        let sample_rate = bits.sample_rate()?;
        let channel_config = u8::try_from(bits.read(4)?).ok()?;
        let mut sbr_sample_rate = None;
        let mut ps = false;
        if matches!(object_type, OBJECT_TYPE_SBR | OBJECT_TYPE_PS) {
            ps = object_type == OBJECT_TYPE_PS;
            sbr_sample_rate = Some(bits.sample_rate()?);
            object_type = bits.object_type()?;
        }
        let short_frames = match object_type {
            OBJECT_TYPE_AAC_LC => {
                let short_frames = bits.read(1)? == 1;
                if bits.read(1)? == 1 {
                    // dependsOnCoreCoder: coreCoderDelay.
                    bits.read(14)?;
                }
                // extensionFlag, 0 for AAC-LC.
                bits.read(1)?;
                if channel_config != 0 && sbr_sample_rate.is_none() {
                    (sbr_sample_rate, ps) = backward_compatible_extension(&mut bits, sample_rate);
                }
                short_frames
            }
            OBJECT_TYPE_ELD => {
                let short_frames = bits.read(1)? == 1;
                // Section / scalefactor / spectral data resilience.
                bits.read(3)?;
                if bits.read(1)? == 1 {
                    // ldSbrPresentFlag; ldSbrSamplingRate picks dual-rate SBR.
                    let dual = bits.read(1)? == 1;
                    sbr_sample_rate = Some(if dual {
                        sample_rate.checked_mul(2)?
                    } else {
                        sample_rate
                    });
                }
                short_frames
            }
            _ => return None,
        };
        Some(Self {
            object_type,
            sample_rate,
            channel_config,
            sbr_sample_rate,
            ps,
            short_frames,
        })
    }

    /// The configuration's bytes: explicit hierarchical signalling for
    /// SBR / PS, an `ELDSpecificConfig` without extensions for ELD.
    /// `None` when a rate has no index, and for ELD with LD-SBR.
    #[must_use]
    pub fn to_bytes(self) -> Option<Vec<u8>> {
        if self.object_type == OBJECT_TYPE_ELD && self.sbr_sample_rate.is_some() {
            return None;
        }
        let mut out = BitWriter::default();
        let index = |rate| sample_rate_index(rate).map(u32::from);
        if let Some(sbr_rate) = self.sbr_sample_rate {
            let signalled = if self.ps {
                OBJECT_TYPE_PS
            } else {
                OBJECT_TYPE_SBR
            };
            out.object_type(signalled);
            out.put(index(self.sample_rate)?, 4);
            out.put(u32::from(self.channel_config), 4);
            out.put(index(sbr_rate)?, 4);
            out.object_type(self.object_type);
        } else {
            out.object_type(self.object_type);
            out.put(index(self.sample_rate)?, 4);
            out.put(u32::from(self.channel_config), 4);
        }
        out.put(u32::from(self.short_frames), 1);
        if self.object_type == OBJECT_TYPE_ELD {
            // Section / scalefactor / spectral data resilience off, no
            // LD-SBR, ELDEXT_TERM, then epConfig 0.
            out.put(0, 3);
            out.put(0, 1);
            out.put(0, 4);
            out.put(0, 2);
        } else {
            // dependsOnCoreCoder, extensionFlag.
            out.put(0, 2);
        }
        Some(out.finish())
    }
}

/// The SBR / PS signalling some encoders append after a plain AAC-LC
/// `GASpecificConfig` (sync extension types 0x2b7 and 0x548).
fn backward_compatible_extension(bits: &mut Bits<'_>, core_rate: u32) -> (Option<u32>, bool) {
    let none = (None, false);
    if bits.peek(11) != Some(0x2b7) {
        return none;
    }
    bits.read(11);
    if bits.object_type() != Some(OBJECT_TYPE_SBR) || bits.read(1) != Some(1) {
        return none;
    }
    let Some(sbr_rate) = bits.sample_rate() else {
        return none;
    };
    let ps = bits.peek(11) == Some(0x548) && {
        bits.read(11);
        bits.read(1) == Some(1)
    };
    (Some(sbr_rate.max(core_rate)), ps)
}

/// An MSB-first reader over a configuration.
struct Bits<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Bits<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn peek(&self, count: usize) -> Option<u32> {
        let mut value = 0;
        for bit in self.at..self.at + count {
            let byte = self.bytes.get(bit / 8)?;
            value = (value << 1) | u32::from((byte >> (7 - bit % 8)) & 1);
        }
        Some(value)
    }

    fn read(&mut self, count: usize) -> Option<u32> {
        let value = self.peek(count)?;
        self.at += count;
        Some(value)
    }

    fn object_type(&mut self) -> Option<u8> {
        let short = self.read(5)?;
        let full = if short == 31 {
            32 + self.read(6)?
        } else {
            short
        };
        u8::try_from(full).ok()
    }

    fn sample_rate(&mut self) -> Option<u32> {
        match self.read(4)? {
            0xF => self.read(24),
            index => SAMPLE_RATES.get(usize::try_from(index).ok()?).copied(),
        }
    }
}

/// An MSB-first writer, zero-padded to whole bytes.
#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    bits: usize,
}

impl BitWriter {
    fn put(&mut self, value: u32, count: usize) {
        for shift in (0..count).rev() {
            if self.bits.is_multiple_of(8) {
                self.bytes.push(0);
            }
            if (value >> shift) & 1 == 1
                && let Some(last) = self.bytes.last_mut()
            {
                *last |= 0x80 >> (self.bits % 8);
            }
            self.bits += 1;
        }
    }

    fn object_type(&mut self, object_type: u8) {
        if object_type >= 32 {
            self.put(31, 5);
            self.put(u32::from(object_type - 32), 6);
        } else {
            self.put(u32::from(object_type), 5);
        }
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn he_aac_configurations_are_written_and_read_hierarchically() {
        let he = AudioSpecificConfig::he_aac(22_050, 2, false).unwrap();
        assert_eq!(he.to_bytes().unwrap(), [0x2B, 0x92, 0x08, 0x00]);
        assert_eq!(
            AudioSpecificConfig::parse(&[0x2B, 0x92, 0x08, 0x00]),
            Some(he)
        );
        assert_eq!(he.output_sample_rate(), 44_100);
        assert_eq!(he.output_frame_samples(), 2048);

        let v2 = AudioSpecificConfig::he_aac(24_000, 1, true).unwrap();
        let parsed = AudioSpecificConfig::parse(&v2.to_bytes().unwrap()).unwrap();
        assert_eq!(parsed, v2);
        assert_eq!(parsed.output_channels(), Some(2));
        assert!(AudioSpecificConfig::he_aac(24_000, 2, true).is_none());
    }

    #[test]
    fn eld_configurations_match_what_airplay_receivers_use() {
        let eld = AudioSpecificConfig::eld(44_100, 2, true).unwrap();
        assert_eq!(eld.to_bytes().unwrap(), [0xF8, 0xE8, 0x50, 0x00]);
        assert_eq!(
            AudioSpecificConfig::parse(&[0xF8, 0xE8, 0x50, 0x00]),
            Some(eld)
        );
        assert_eq!(eld.core_frame_samples(), 480);
        // The same with ldSbrPresentFlag and dual-rate SBR.
        let sbr = AudioSpecificConfig::parse(&[0xF8, 0xE8, 0x51, 0x80]).unwrap();
        assert_eq!(sbr.sbr_sample_rate, Some(88_200));
        assert_eq!(sbr.output_frame_samples(), 960);
        assert!(sbr.to_bytes().is_none());
    }

    #[test]
    fn plain_lc_and_backward_compatible_sbr_are_told_apart() {
        let lc = AudioSpecificConfig::parse(&[0x11, 0x90]).unwrap();
        assert_eq!(
            (lc.sample_rate, lc.sbr_sample_rate, lc.ps),
            (48_000, None, false)
        );
        // LC 24 kHz mono, then sync extension 0x2b7, SBR, 48 kHz.
        let mut out = BitWriter::default();
        out.put(2, 5);
        out.put(6, 4);
        out.put(1, 4);
        out.put(0, 3);
        out.put(0x2b7, 11);
        out.put(5, 5);
        out.put(1, 1);
        out.put(3, 4);
        let implicit = AudioSpecificConfig::parse(&out.finish()).unwrap();
        assert_eq!(implicit.sbr_sample_rate, Some(48_000));
        assert!(AudioSpecificConfig::parse(&[0x11]).is_none());
        assert!(AudioSpecificConfig::parse(&[]).is_none());
    }

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
