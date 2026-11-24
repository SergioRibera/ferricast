//! AAC (LC, HE-AAC v1 / v2), AAC-ELD and ALAC decode through
//! `AudioToolbox`'s `AudioConverter`.
//!
//! Each access unit is fed to a converter built for its [`Format`]: the
//! compressed format, its magic cookie (the `AudioSpecificConfig` for the
//! AAC family, the `ALACSpecificConfig` for ALAC) and the PCM shape it
//! decodes to; a format change rebuilds the converter. Output is
//! interleaved 16-bit PCM.
//!
//! - AAC: ADTS (one or more frames per payload) or raw access units. The
//!   stream's `AudioSpecificConfig` comes from `extra_data` when the
//!   container has one (explicit SBR / PS signalling decodes as HE-AAC);
//!   otherwise ADTS headers only say AAC-LC, and a core rate of 24 kHz or
//!   less is decoded as HE-AAC v1 (v2 for mono), since SBR (implicit
//!   signalling) is what such rates carry in practice. A plain LC stream
//!   at those rates then comes out upsampled 2x, which is still correct
//!   audio.
//! - AAC-ELD: raw access units; `extra_data` or the `AirPlay` mirroring
//!   default (480-sample frames at the configured rate and channels).
//! - ALAC: raw frames; `extra_data` or the `AirPlay` default cookie.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr::{self, NonNull};

use ferricast_codec::aac::{AacLc, AudioSpecificConfig, OBJECT_TYPE_ELD};
use ferricast_codec::alac::AlacConfig;
use ferricast_codec::{AudioDecoderRegistry, Candidate};
use ferricast_core::{
    AacChannelConfig, Acceleration, AudioCodec, AudioDecoder, AudioDecoderConfig, AudioFrame,
    BackendId, CodecError, CodecKind, CodecResult, CodecTarget, DecodedAudio, Operation,
};
use objc2_audio_toolbox::{
    AudioConverterDispose, AudioConverterFillComplexBuffer, AudioConverterNew, AudioConverterRef,
    AudioConverterSetProperty, kAudioConverterDecompressionMagicCookie,
};
use objc2_core_audio_types::{
    AudioBuffer, AudioBufferList, AudioFormatID, AudioStreamBasicDescription,
    AudioStreamPacketDescription, kAppleLosslessFormatFlag_16BitSourceData,
    kAppleLosslessFormatFlag_20BitSourceData, kAppleLosslessFormatFlag_24BitSourceData,
    kAppleLosslessFormatFlag_32BitSourceData, kAudioFormatAppleLossless, kAudioFormatFlagIsPacked,
    kAudioFormatFlagIsSignedInteger, kAudioFormatLinearPCM, kAudioFormatMPEG4AAC,
    kAudioFormatMPEG4AAC_ELD, kAudioFormatMPEG4AAC_ELD_SBR, kAudioFormatMPEG4AAC_HE,
    kAudioFormatMPEG4AAC_HE_V2,
};
use tracing::debug;

use super::os_status;
use crate::adts::{self, Shape};

const BACKEND: BackendId = BackendId::AudioToolbox;
/// What the input callback returns once its packet is consumed.
const NO_MORE_INPUT: i32 = i32::from_be_bytes(*b"!dat");
const MICROS_PER_SECOND: u64 = 1_000_000;
/// The highest AAC core rate treated as carrying implicit SBR.
const IMPLICIT_SBR_MAX_RATE: u32 = 24_000;

/// Adds the `AudioToolbox` AAC, AAC-ELD and ALAC decoders.
pub(crate) fn audio(registry: AudioDecoderRegistry) -> AudioDecoderRegistry {
    match registry.target().codec {
        CodecKind::Audio(AudioCodec::Aac | AudioCodec::AacEld | AudioCodec::Alac) => registry
            .candidate(Candidate::new(
                BACKEND,
                Acceleration::Software,
                |config: &AudioDecoderConfig| {
                    AtDecoder::open(config).map(|decoder| Box::new(decoder) as _)
                },
            )),
        _ => registry,
    }
}

fn fail(what: impl std::fmt::Display) -> CodecError {
    CodecError::backend(BACKEND, Operation::Decode, what)
}

fn unsupported(reason: impl Into<String>) -> CodecError {
    CodecError::UnsupportedStream {
        backend: BACKEND,
        reason: reason.into(),
    }
}

/// Channels a `channelConfiguration` code carries.
fn channels_of(code: u8) -> Option<u16> {
    AacChannelConfig::new(code).map(AacChannelConfig::channels)
}

/// What one converter decodes: the compressed side, its cookie, and the
#[expect(
    clippy::struct_field_names,
    reason = "`format_id` / `format_flags` keep the AudioStreamBasicDescription names"
)]
/// PCM it produces.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Format {
    format_id: AudioFormatID,
    format_flags: u32,
    /// The rate the compressed description declares (for HE-AAC, the SBR
    /// output rate, as `AudioToolbox` describes it).
    sample_rate: u32,
    channels: u16,
    frames_per_packet: u32,
    cookie: Vec<u8>,
    output_rate: u32,
    output_channels: u16,
    /// Samples per channel one packet decodes to, at most.
    output_frames: u32,
}

impl Format {
    /// An AAC-family stream described by `config`.
    fn aac(config: AudioSpecificConfig) -> CodecResult<Self> {
        let (Some(channels), Some(output_channels)) =
            (channels_of(config.channel_config), config.output_channels())
        else {
            return Err(unsupported(
                "AAC with a program config element instead of a channel configuration",
            ));
        };
        let cookie = config.to_bytes().ok_or_else(|| {
            unsupported(format!(
                "AAC at {} Hz with no standard sample rate index, or ELD with LD-SBR",
                config.sample_rate
            ))
        })?;
        let format_id = match (config.object_type, config.sbr_sample_rate, config.ps) {
            (OBJECT_TYPE_ELD, None, _) => kAudioFormatMPEG4AAC_ELD,
            (OBJECT_TYPE_ELD, Some(_), _) => kAudioFormatMPEG4AAC_ELD_SBR,
            (_, None, _) => kAudioFormatMPEG4AAC,
            (_, Some(_), false) => kAudioFormatMPEG4AAC_HE,
            (_, Some(_), true) => kAudioFormatMPEG4AAC_HE_V2,
        };
        Ok(Self {
            format_id,
            format_flags: 0,
            sample_rate: config.output_sample_rate(),
            channels,
            frames_per_packet: config.output_frame_samples(),
            cookie,
            output_rate: config.output_sample_rate(),
            output_channels,
            output_frames: config.output_frame_samples(),
        })
    }

    /// An ALAC stream described by `config`.
    fn alac(config: AlacConfig) -> CodecResult<Self> {
        let format_flags = match config.bit_depth {
            16 => kAppleLosslessFormatFlag_16BitSourceData,
            20 => kAppleLosslessFormatFlag_20BitSourceData,
            24 => kAppleLosslessFormatFlag_24BitSourceData,
            32 => kAppleLosslessFormatFlag_32BitSourceData,
            other => return Err(unsupported(format!("{other}-bit ALAC"))),
        };
        if config.channels == 0 || config.frame_length == 0 {
            return Err(unsupported("an ALAC cookie with no channels or frames"));
        }
        Ok(Self {
            format_id: kAudioFormatAppleLossless,
            format_flags,
            sample_rate: config.sample_rate,
            channels: u16::from(config.channels),
            frames_per_packet: config.frame_length,
            cookie: config.cookie().to_vec(),
            output_rate: config.sample_rate,
            output_channels: u16::from(config.channels),
            output_frames: config.frame_length,
        })
    }

    /// The format of an ADTS / raw AAC unit of `shape`: `stream`'s
    /// configuration when it is for that shape, else AAC-LC, or HE-AAC
    /// for core rates that imply SBR.
    fn aac_unit(shape: Shape, stream: Option<AudioSpecificConfig>) -> CodecResult<Self> {
        if let Some(config) = stream
            && config.sample_rate == shape.sample_rate
            && channels_of(config.channel_config) == Some(shape.channels)
        {
            return Self::aac(config);
        }
        let config = if shape.sample_rate <= IMPLICIT_SBR_MAX_RATE {
            AudioSpecificConfig::he_aac(shape.sample_rate, shape.channels, shape.channels == 1)
        } else {
            AacLc::new(shape.sample_rate, shape.channels)
                .and_then(|lc| AudioSpecificConfig::parse(&lc.audio_specific_config()))
        };
        let config = config.ok_or_else(|| {
            unsupported(format!(
                "AAC at {} Hz with {} channels has no configuration",
                shape.sample_rate, shape.channels
            ))
        })?;
        Self::aac(config)
    }
}

/// An owned `AudioConverterRef`.
struct Converter(AudioConverterRef);

// SAFETY: an AudioConverter may be used from any thread as long as calls
// do not overlap; it is only reached through `&mut AtDecoder`.
unsafe impl Send for Converter {}

impl Drop for Converter {
    fn drop(&mut self) {
        // SAFETY: the converter is live and disposed exactly once.
        let status = unsafe { AudioConverterDispose(self.0) };
        if status != 0 {
            debug!(status = %os_status(status), "AudioConverterDispose failed");
        }
    }
}

impl Converter {
    fn open(format: &Format) -> CodecResult<Self> {
        let compressed = AudioStreamBasicDescription {
            mSampleRate: f64::from(format.sample_rate),
            mFormatID: format.format_id,
            mFormatFlags: format.format_flags,
            mBytesPerPacket: 0,
            mFramesPerPacket: format.frames_per_packet,
            mBytesPerFrame: 0,
            mChannelsPerFrame: u32::from(format.channels),
            mBitsPerChannel: 0,
            mReserved: 0,
        };
        let channels = u32::from(format.output_channels);
        let pcm = AudioStreamBasicDescription {
            mSampleRate: f64::from(format.output_rate),
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
            mBytesPerPacket: 2 * channels,
            mFramesPerPacket: 1,
            mBytesPerFrame: 2 * channels,
            mChannelsPerFrame: channels,
            mBitsPerChannel: 16,
            mReserved: 0,
        };
        let mut raw: AudioConverterRef = ptr::null_mut();
        // SAFETY: both descriptions and the out pointer are valid locals.
        let status = unsafe {
            AudioConverterNew(
                NonNull::from(&compressed),
                NonNull::from(&pcm),
                NonNull::from(&mut raw),
            )
        };
        if status != 0 || raw.is_null() {
            return Err(unsupported(format!(
                "no decoder for {format:?}: {}",
                os_status(status)
            )));
        }
        let converter = Self(raw);
        let cookie_len =
            u32::try_from(format.cookie.len()).map_err(|_| fail("the magic cookie is too long"))?;
        if let Some(cookie) = NonNull::new(format.cookie.as_ptr().cast_mut()) {
            // SAFETY: the converter is live; the cookie is `cookie_len`
            // bytes, only read.
            let status = unsafe {
                AudioConverterSetProperty(
                    converter.0,
                    kAudioConverterDecompressionMagicCookie,
                    cookie_len,
                    cookie.cast(),
                )
            };
            if status != 0 {
                return Err(unsupported(format!(
                    "the decoder refused the stream configuration: {}",
                    os_status(status)
                )));
            }
        }
        Ok(converter)
    }
}

/// The packet one fill call may pull.
struct Feed {
    data: *const u8,
    len: u32,
    description: AudioStreamPacketDescription,
    consumed: bool,
}

unsafe extern "C-unwind" fn feed_packet(
    _converter: AudioConverterRef,
    packets: NonNull<u32>,
    buffers: NonNull<AudioBufferList>,
    descriptions: *mut *mut AudioStreamPacketDescription,
    user: *mut c_void,
) -> i32 {
    // SAFETY: `user` is the `Feed` `decode_unit` passed for this fill call,
    // alive until it returns; the other pointers are the converter's out
    // parameters.
    unsafe {
        let feed = &mut *user.cast::<Feed>();
        if feed.consumed {
            *packets.as_ptr() = 0;
            return NO_MORE_INPUT;
        }
        let list = buffers.as_ptr();
        (*list).mNumberBuffers = 1;
        (*list).mBuffers[0] = AudioBuffer {
            mNumberChannels: 0,
            mDataByteSize: feed.len,
            mData: feed.data.cast_mut().cast(),
        };
        if !descriptions.is_null() {
            *descriptions = &raw mut feed.description;
        }
        *packets.as_ptr() = 1;
        feed.consumed = true;
    }
    0
}

/// What the configured stream is.
enum Source {
    /// ADTS or raw AAC; the stream's `AudioSpecificConfig` when known, and
    /// the shape raw access units have.
    Aac {
        config: Option<AudioSpecificConfig>,
        raw: Option<Shape>,
    },
    /// One fixed format of raw access units (AAC-ELD, ALAC).
    Fixed(Format),
}

impl Source {
    fn of(config: &AudioDecoderConfig) -> CodecResult<Self> {
        let extra = config.extra_data.as_deref();
        let bad_extra = |what: &str| CodecError::UnsupportedConfig {
            backend: BACKEND,
            what: format!(
                "{} extra_data of {} bytes that is not {what}",
                config.codec,
                extra.map_or(0, <[u8]>::len)
            ),
        };
        match config.codec {
            AudioCodec::Aac => {
                let asc = extra
                    .map(|bytes| {
                        AudioSpecificConfig::parse(bytes)
                            .filter(|asc| asc.object_type != OBJECT_TYPE_ELD)
                            .ok_or_else(|| bad_extra("an AAC AudioSpecificConfig"))
                    })
                    .transpose()?;
                let raw = asc
                    .and_then(|asc| {
                        Some(Shape {
                            sample_rate: asc.sample_rate,
                            channels: channels_of(asc.channel_config)?,
                        })
                    })
                    .or_else(|| {
                        (config.sample_rate > 0 && config.channels > 0).then_some(Shape {
                            sample_rate: config.sample_rate,
                            channels: config.channels,
                        })
                    });
                Ok(Self::Aac { config: asc, raw })
            }
            AudioCodec::AacEld => {
                let asc = match extra {
                    Some(bytes) => AudioSpecificConfig::parse(bytes)
                        .filter(|asc| asc.object_type == OBJECT_TYPE_ELD)
                        .ok_or_else(|| bad_extra("an AAC-ELD AudioSpecificConfig"))?,
                    None => AudioSpecificConfig::eld(config.sample_rate, config.channels, true)
                        .ok_or_else(|| CodecError::UnsupportedConfig {
                            backend: BACKEND,
                            what: format!(
                                "AAC-ELD at {} Hz with {} channels",
                                config.sample_rate, config.channels
                            ),
                        })?,
                };
                Ok(Self::Fixed(Format::aac(asc)?))
            }
            AudioCodec::Alac => {
                let cookie = if let Some(bytes) = extra {
                    AlacConfig::parse(bytes).ok_or_else(|| bad_extra("an ALACSpecificConfig"))?
                } else {
                    let channels = u8::try_from(config.channels)
                        .ok()
                        .filter(|channels| (1..=8).contains(channels))
                        .ok_or_else(|| CodecError::UnsupportedConfig {
                            backend: BACKEND,
                            what: format!("ALAC with {} channels", config.channels),
                        })?;
                    AlacConfig::airplay(config.sample_rate, channels)
                };
                Ok(Self::Fixed(Format::alac(cookie)?))
            }
            other => Err(CodecError::BackendCannotHandle {
                backend: BACKEND,
                target: CodecTarget::decode(other),
            }),
        }
    }
}

/// AAC / AAC-ELD / ALAC decoder over `AudioToolbox`.
pub(crate) struct AtDecoder {
    source: Source,
    active: Option<(Format, Converter)>,
    pcm: Vec<i16>,
    pending: VecDeque<DecodedAudio>,
}

impl AtDecoder {
    fn open(config: &AudioDecoderConfig) -> CodecResult<Self> {
        Ok(Self {
            source: Source::of(config)?,
            active: None,
            pcm: Vec::new(),
            pending: VecDeque::new(),
        })
    }

    /// Decodes one access unit of `format`, appending its PCM to `pcm`;
    /// returns the frames decoded.
    fn decode_unit(&mut self, format: Format, payload: &[u8]) -> CodecResult<u64> {
        if self
            .active
            .as_ref()
            .is_none_or(|(current, _)| *current != format)
        {
            let converter = Converter::open(&format)?;
            self.active = Some((format, converter));
        }
        let Some((format, converter)) = &self.active else {
            return Err(fail("invariant: a converter was just opened"));
        };
        let len = u32::try_from(payload.len()).map_err(|_| fail("access unit too long"))?;
        let mut feed = Feed {
            data: payload.as_ptr(),
            len,
            description: AudioStreamPacketDescription {
                mStartOffset: 0,
                mVariableFramesInPacket: 0,
                mDataByteSize: len,
            },
            consumed: false,
        };
        let channels = u32::from(format.output_channels);
        let max_frames = format.output_frames;
        let start = self.pcm.len();
        self.pcm.resize(start + (max_frames * channels) as usize, 0);
        let mut output = AudioBufferList {
            mNumberBuffers: 1,
            mBuffers: [AudioBuffer {
                mNumberChannels: channels,
                mDataByteSize: max_frames * channels * 2,
                mData: self.pcm[start..].as_mut_ptr().cast(),
            }],
        };
        let mut frames = max_frames;
        // SAFETY: `feed` points into `payload`, alive for the call; the
        // output list points at `max_frames` frames of `pcm`.
        let status = unsafe {
            AudioConverterFillComplexBuffer(
                converter.0,
                Some(feed_packet),
                ptr::from_mut(&mut feed).cast(),
                NonNull::from(&mut frames),
                NonNull::from(&mut output),
                ptr::null_mut(),
            )
        };
        let decoded = frames.min(max_frames);
        self.pcm.truncate(start + (decoded * channels) as usize);
        if status != 0 && status != NO_MORE_INPUT {
            return Err(fail(format!(
                "AudioConverterFillComplexBuffer: {}",
                os_status(status)
            )));
        }
        Ok(u64::from(decoded))
    }

    /// Queues the PCM decoded so far as `rate` / `channels` audio stamped
    /// `timestamp_us`.
    fn finish_run(&mut self, rate: u32, channels: u16, timestamp_us: u64) {
        if self.pcm.is_empty() {
            return;
        }
        self.pending.push_back(DecodedAudio {
            pcm: std::mem::take(&mut self.pcm),
            sample_rate: rate,
            channels,
            timestamp_us,
        });
    }

    /// One payload of ADTS frames or raw AAC access units.
    fn decode_aac(
        &mut self,
        frame: &AudioFrame,
        config: Option<AudioSpecificConfig>,
        raw: Option<Shape>,
    ) -> CodecResult<()> {
        let mut base_us = frame.timestamp_us;
        // The output rate and channels of the run being decoded, and its
        // frames so far.
        let mut run: Option<((u32, u16), u64)> = None;
        let mut result = Ok(());
        for unit in adts::units(&frame.data) {
            let unit = match unit {
                Ok(unit) => unit,
                Err(error) => {
                    result = Err(fail(format!("AAC: {error}")));
                    break;
                }
            };
            let Some(shape) = unit.shape.or(raw) else {
                result = Err(fail(
                    "raw (non-ADTS) AAC with no AudioSpecificConfig or sample rate / channels",
                ));
                break;
            };
            let format = match Format::aac_unit(shape, config) {
                Ok(format) => format,
                Err(error) => {
                    result = Err(error);
                    break;
                }
            };
            let output = (format.output_rate, format.output_channels);
            if let Some((previous, samples)) = run
                && previous != output
            {
                self.finish_run(previous.0, previous.1, base_us);
                base_us += samples * MICROS_PER_SECOND / u64::from(previous.0.max(1));
                run = None;
            }
            match self.decode_unit(format, unit.payload) {
                Ok(decoded) => {
                    let samples = run.map_or(0, |(_, samples)| samples);
                    run = Some((output, samples + decoded));
                }
                Err(error) => {
                    result = Err(error);
                    break;
                }
            }
        }
        if let Some(((rate, channels), _)) = run {
            self.finish_run(rate, channels, base_us);
        }
        // Frames decoded before a bad one are still good audio.
        result
    }
}

impl AudioDecoder for AtDecoder {
    const CODEC: AudioCodec = AudioCodec::Aac;

    fn configure(&mut self, config: &AudioDecoderConfig) -> CodecResult<()> {
        self.source = Source::of(config)?;
        self.pending.clear();
        self.pcm.clear();
        Ok(())
    }

    fn decode(&mut self, frame: AudioFrame) -> CodecResult<Option<DecodedAudio>> {
        if frame.data.is_empty() {
            return Ok(self.pending.pop_front());
        }
        match &self.source {
            Source::Aac { config, raw } => {
                let (config, raw) = (*config, *raw);
                self.decode_aac(&frame, config, raw)?;
            }
            Source::Fixed(format) => {
                let format = format.clone();
                let (rate, channels) = (format.output_rate, format.output_channels);
                if let Err(error) = self.decode_unit(format, &frame.data) {
                    self.pcm.clear();
                    return Err(error);
                }
                self.finish_run(rate, channels, frame.timestamp_us);
            }
        }
        Ok(self.pending.pop_front())
    }

    fn flush(&mut self) -> CodecResult<Vec<DecodedAudio>> {
        Ok(self.pending.drain(..).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_core_rates_decode_as_he_aac_and_explicit_configs_win() {
        let shape = |sample_rate, channels| Shape {
            sample_rate,
            channels,
        };
        let lc = Format::aac_unit(shape(48_000, 2), None).unwrap();
        assert_eq!(
            (lc.format_id, lc.output_rate, lc.output_frames),
            (kAudioFormatMPEG4AAC, 48_000, 1024)
        );
        let he = Format::aac_unit(shape(24_000, 2), None).unwrap();
        assert_eq!(
            (he.format_id, he.output_rate, he.output_frames),
            (kAudioFormatMPEG4AAC_HE, 48_000, 2048)
        );
        let v2 = Format::aac_unit(shape(22_050, 1), None).unwrap();
        assert_eq!(
            (v2.format_id, v2.output_channels),
            (kAudioFormatMPEG4AAC_HE_V2, 2)
        );
        // An explicit plain AAC-LC configuration at 22.05 kHz stereo.
        let plain = AudioSpecificConfig::parse(&[0x13, 0x90]).unwrap();
        let explicit = Format::aac_unit(shape(22_050, 2), Some(plain)).unwrap();
        assert_eq!(explicit.format_id, kAudioFormatMPEG4AAC);
    }

    #[test]
    fn eld_and_alac_formats_follow_their_configuration() {
        let eld = Format::aac(AudioSpecificConfig::eld(44_100, 2, true).unwrap()).unwrap();
        assert_eq!(
            (eld.format_id, eld.frames_per_packet, eld.cookie.as_slice()),
            (kAudioFormatMPEG4AAC_ELD, 480, &[0xF8, 0xE8, 0x50, 0x00][..])
        );
        let alac = Format::alac(AlacConfig::airplay(44_100, 2)).unwrap();
        assert_eq!(
            (alac.format_flags, alac.frames_per_packet, alac.cookie.len()),
            (kAppleLosslessFormatFlag_16BitSourceData, 352, 24)
        );
        let odd = AlacConfig {
            bit_depth: 12,
            ..AlacConfig::airplay(44_100, 2)
        };
        assert!(matches!(
            Format::alac(odd),
            Err(CodecError::UnsupportedStream { .. })
        ));
    }
}
