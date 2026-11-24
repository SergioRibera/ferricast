//! AAC-LC and AAC-ELD encode through `AudioToolbox`'s `AudioConverter`.
//!
//! The converter pulls PCM through an input callback; each call feeds it
//! exactly one frame (1024 samples per channel for AAC-LC, 480 for
//! AAC-ELD) and asks for one packet. The encoder's priming delay means
//! the first calls yield nothing.
//!
//! - AAC-LC packets are framed as ADTS, the shape the MPEG-TS / HLS
//!   pipeline carries.
//! - AAC-ELD is `AirPlay` mirroring audio: raw access units at 44.1 kHz,
//!   480-sample frames, no SBR (`AudioSpecificConfig` `f8 e8 50 00` for
//!   stereo). Input at other rates is resampled to 44.1 kHz first, as the
//!   fdk-aac ELD encoder does.

use std::ffi::c_void;
use std::ptr::{self, NonNull};

use bytes::Bytes;
use ferricast_codec::aac::{AacLc, AudioSpecificConfig};
use ferricast_codec::{AudioEncoderRegistry, Candidate};
use ferricast_core::{
    Acceleration, AudioCodec, AudioEncoder, AudioEncoderConfig, AudioFrame, BackendId, CodecError,
    CodecKind, CodecResult, Operation,
};
use objc2_audio_toolbox::{
    AudioConverterDispose, AudioConverterFillComplexBuffer, AudioConverterGetProperty,
    AudioConverterNew, AudioConverterRef, AudioConverterSetProperty, kAudioConverterEncodeBitRate,
    kAudioConverterPropertyMaximumOutputPacketSize,
};
use objc2_core_audio_types::{
    AudioBuffer, AudioBufferList, AudioStreamBasicDescription, AudioStreamPacketDescription,
    kAudioFormatFlagIsPacked, kAudioFormatFlagIsSignedInteger, kAudioFormatLinearPCM,
    kAudioFormatMPEG4AAC, kAudioFormatMPEG4AAC_ELD,
};
use tracing::{debug, info};

use super::os_status;
use crate::aac::resample::Resampler;
use crate::audio_clock::{AudioClock, ClockEvent};

const BACKEND: BackendId = BackendId::AudioToolbox;
/// What the input callback returns once its frame is consumed, so the
/// converter stops pulling; not an error.
const NO_MORE_INPUT: i32 = i32::from_be_bytes(*b"!dat");
/// Sample rate of `AirPlay`'s AAC-ELD.
const ELD_SAMPLE_RATE: u32 = 44_100;

/// Adds the `AudioToolbox` AAC-LC and AAC-ELD encoders.
pub(crate) fn audio(registry: AudioEncoderRegistry) -> AudioEncoderRegistry {
    if !matches!(
        registry.target().codec,
        CodecKind::Audio(AudioCodec::Aac | AudioCodec::AacEld)
    ) {
        return registry;
    }
    registry.candidate(Candidate::new(
        BACKEND,
        Acceleration::Software,
        |config: &AudioEncoderConfig| {
            AtAacEncoder::open(config).map(|encoder| Box::new(encoder) as _)
        },
    ))
}

fn fail(operation: Operation, what: impl std::fmt::Display) -> CodecError {
    CodecError::backend(BACKEND, operation, what)
}

/// The AAC flavour an encoder produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Profile {
    /// AAC-LC, ADTS-framed.
    Lc(AacLc),
    /// AAC-ELD, raw access units.
    Eld,
}

impl Profile {
    fn codec(self) -> AudioCodec {
        match self {
            Self::Lc(_) => AudioCodec::Aac,
            Self::Eld => AudioCodec::AacEld,
        }
    }

    /// Samples per channel in one frame.
    fn frame_samples(self) -> usize {
        match self {
            Self::Lc(_) => 1024,
            Self::Eld => 480,
        }
    }
}

/// An owned `AudioConverterRef`.
struct Converter(AudioConverterRef);

// SAFETY: an AudioConverter may be used from any thread as long as calls
// do not overlap; it is only reached through `&mut AtAacEncoder`.
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

/// The PCM one fill call may pull: one frame.
struct Feed {
    data: *const i16,
    frames: u32,
    channels: u32,
}

unsafe extern "C-unwind" fn feed(
    _converter: AudioConverterRef,
    packets: NonNull<u32>,
    buffers: NonNull<AudioBufferList>,
    _descriptions: *mut *mut AudioStreamPacketDescription,
    user: *mut c_void,
) -> i32 {
    // SAFETY: `user` is the `Feed` `encode_frame` passed for this fill
    // call, alive until it returns; `packets` and `buffers` are the
    // converter's out parameters.
    unsafe {
        let feed = &mut *user.cast::<Feed>();
        if feed.frames == 0 {
            *packets.as_ptr() = 0;
            return NO_MORE_INPUT;
        }
        let list = buffers.as_ptr();
        (*list).mNumberBuffers = 1;
        (*list).mBuffers[0] = AudioBuffer {
            mNumberChannels: feed.channels,
            mDataByteSize: feed.frames * feed.channels * 2,
            mData: feed.data.cast_mut().cast(),
        };
        *packets.as_ptr() = feed.frames;
        feed.frames = 0;
    }
    0
}

/// AAC-LC / AAC-ELD encoder over `AudioToolbox`.
pub(crate) struct AtAacEncoder {
    converter: Converter,
    profile: Profile,
    /// The stream's `AudioSpecificConfig`.
    audio_specific_config: Vec<u8>,
    /// The rate PCM arrives at.
    input_rate: u32,
    /// The rate the encoder codes (the input rate for AAC-LC, 44.1 kHz
    /// for AAC-ELD).
    sample_rate: u32,
    channels: u16,
    resampler: Option<Resampler>,
    resampled: Vec<i16>,
    pcm: Vec<i16>,
    clock: AudioClock,
    packet: Vec<u8>,
    pending: Vec<AudioFrame>,
}

impl AtAacEncoder {
    #[expect(
        clippy::too_many_lines,
        reason = "one linear AudioToolbox converter bring-up"
    )]
    fn open(config: &AudioEncoderConfig) -> CodecResult<Self> {
        let unsupported = |what: String| CodecError::UnsupportedConfig {
            backend: BACKEND,
            what,
        };
        let (profile, sample_rate, audio_specific_config) = match config.codec {
            AudioCodec::Aac => {
                let lc = AacLc::new(config.sample_rate, config.channels).ok_or_else(|| {
                    unsupported(format!(
                        "AAC-LC at {} Hz with {} channels",
                        config.sample_rate, config.channels
                    ))
                })?;
                (
                    Profile::Lc(lc),
                    config.sample_rate,
                    lc.audio_specific_config().to_vec(),
                )
            }
            AudioCodec::AacEld => {
                let asc = AudioSpecificConfig::eld(ELD_SAMPLE_RATE, config.channels, true)
                    .and_then(AudioSpecificConfig::to_bytes)
                    .ok_or_else(|| {
                        unsupported(format!("AAC-ELD with {} channels", config.channels))
                    })?;
                (Profile::Eld, ELD_SAMPLE_RATE, asc)
            }
            other => return Err(unsupported(format!("{other} in the AAC encoder"))),
        };
        if config.sample_rate == 0 {
            return Err(unsupported("PCM input at 0 Hz".to_owned()));
        }
        let channels = u32::from(config.channels);
        let pcm = AudioStreamBasicDescription {
            mSampleRate: f64::from(sample_rate),
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
            mBytesPerPacket: 2 * channels,
            mFramesPerPacket: 1,
            mBytesPerFrame: 2 * channels,
            mChannelsPerFrame: channels,
            mBitsPerChannel: 16,
            mReserved: 0,
        };
        let frame_samples = u32::try_from(profile.frame_samples()).unwrap_or(u32::MAX);
        let aac = AudioStreamBasicDescription {
            mSampleRate: f64::from(sample_rate),
            mFormatID: match profile {
                Profile::Lc(_) => kAudioFormatMPEG4AAC,
                Profile::Eld => kAudioFormatMPEG4AAC_ELD,
            },
            mFormatFlags: 0,
            mBytesPerPacket: 0,
            mFramesPerPacket: frame_samples,
            mBytesPerFrame: 0,
            mChannelsPerFrame: channels,
            mBitsPerChannel: 0,
            mReserved: 0,
        };
        let mut raw: AudioConverterRef = ptr::null_mut();
        // SAFETY: both descriptions and the out pointer are valid locals.
        let status = unsafe {
            AudioConverterNew(
                NonNull::from(&pcm),
                NonNull::from(&aac),
                NonNull::from(&mut raw),
            )
        };
        if status != 0 || raw.is_null() {
            return Err(CodecError::BackendUnavailable {
                backend: BACKEND,
                reason: format!(
                    "no {} encoder for {sample_rate} Hz / {} channels: {}",
                    profile.codec(),
                    config.channels,
                    os_status(status)
                ),
            });
        }
        let converter = Converter(raw);
        let bits = config.bitrate_kbps.saturating_mul(1000);
        // SAFETY: the converter is live; the property takes a `u32`.
        let status = unsafe {
            AudioConverterSetProperty(
                converter.0,
                kAudioConverterEncodeBitRate,
                4,
                NonNull::from(&bits).cast(),
            )
        };
        if status != 0 {
            return Err(unsupported(format!(
                "{} at {} kbps, {sample_rate} Hz, {} channels ({})",
                profile.codec(),
                config.bitrate_kbps,
                config.channels,
                os_status(status)
            )));
        }
        let mut max_packet = 0u32;
        let mut size = 4u32;
        // SAFETY: the converter is live; the property is a `u32`.
        let status = unsafe {
            AudioConverterGetProperty(
                converter.0,
                kAudioConverterPropertyMaximumOutputPacketSize,
                NonNull::from(&mut size),
                NonNull::from(&mut max_packet).cast(),
            )
        };
        if status != 0 || max_packet == 0 {
            return Err(fail(
                Operation::Open,
                format!("reading the maximum packet size: {}", os_status(status)),
            ));
        }
        info!(
            codec = %profile.codec(),
            input_rate = config.sample_rate,
            sample_rate,
            channels = config.channels,
            bitrate_kbps = config.bitrate_kbps,
            "AAC encoder backend: AudioToolbox"
        );
        Ok(Self {
            converter,
            profile,
            audio_specific_config,
            input_rate: config.sample_rate,
            sample_rate,
            channels: config.channels,
            resampler: (config.sample_rate != sample_rate)
                .then(|| Resampler::new(config.sample_rate, sample_rate, config.channels)),
            resampled: Vec::new(),
            pcm: Vec::with_capacity(profile.frame_samples() * usize::from(config.channels) * 4),
            clock: AudioClock::new(sample_rate),
            packet: vec![0; max_packet as usize],
            pending: Vec::new(),
        })
    }

    /// Feeds one frame of PCM from the front of `pcm` (or, with `drain`,
    /// none, to pull out what the encoder still holds) and queues the
    /// packet it yields; returns whether one came out.
    fn encode_frame(&mut self, drain: bool) -> CodecResult<bool> {
        let channels = u32::from(self.channels);
        let frame_samples = self.profile.frame_samples();
        let mut feed = Feed {
            data: self.pcm.as_ptr(),
            frames: if drain {
                0
            } else {
                u32::try_from(frame_samples).unwrap_or(u32::MAX)
            },
            channels,
        };
        let mut output = AudioBufferList {
            mNumberBuffers: 1,
            mBuffers: [AudioBuffer {
                mNumberChannels: channels,
                mDataByteSize: u32::try_from(self.packet.len()).unwrap_or(u32::MAX),
                mData: self.packet.as_mut_ptr().cast(),
            }],
        };
        let mut packets = 1u32;
        let mut description = AudioStreamPacketDescription {
            mStartOffset: 0,
            mVariableFramesInPacket: 0,
            mDataByteSize: 0,
        };
        // SAFETY: `feed` points into `pcm`, which holds a full frame and is
        // not touched until the call returns; the output list points into
        // `packet`, sized to the converter's maximum packet.
        let status = unsafe {
            AudioConverterFillComplexBuffer(
                self.converter.0,
                Some(feed_fn()),
                ptr::from_mut(&mut feed).cast(),
                NonNull::from(&mut packets),
                NonNull::from(&mut output),
                &raw mut description,
            )
        };
        if !drain {
            self.pcm.drain(..frame_samples * usize::from(self.channels));
        }
        if status != 0 && status != NO_MORE_INPUT {
            return Err(fail(
                Operation::Encode,
                format!("AudioConverterFillComplexBuffer: {}", os_status(status)),
            ));
        }
        if packets == 0 {
            // Still priming (or, draining, empty).
            return Ok(false);
        }
        let len = (output.mBuffers[0].mDataByteSize as usize).min(self.packet.len());
        let data = match self.profile {
            Profile::Lc(lc) => lc
                .adts_frame(&self.packet[..len])
                .ok_or_else(|| fail(Operation::Encode, "the AAC packet is too long for ADTS"))?,
            Profile::Eld => self.packet[..len].to_vec(),
        };
        self.pending.push(AudioFrame {
            codec: self.profile.codec(),
            data: Bytes::from(data),
            timestamp_us: self.clock.next_timestamp_us(),
            sample_rate: self.sample_rate,
            channels: self.channels,
        });
        self.clock.advance(frame_samples as u64);
        Ok(true)
    }
}

/// The input callback, as the converter's function-pointer type.
fn feed_fn() -> unsafe extern "C-unwind" fn(
    AudioConverterRef,
    NonNull<u32>,
    NonNull<AudioBufferList>,
    *mut *mut AudioStreamPacketDescription,
    *mut c_void,
) -> i32 {
    feed
}

impl AudioEncoder for AtAacEncoder {
    fn configure(&mut self, config: &AudioEncoderConfig) -> CodecResult<()> {
        *self = Self::open(config)?;
        Ok(())
    }

    fn encode(&mut self, frame: &AudioFrame) -> CodecResult<()> {
        if frame.codec != AudioCodec::Pcm {
            return Err(fail(
                Operation::Encode,
                format!("expected PCM input, got {}", frame.codec),
            ));
        }
        if frame.sample_rate != self.input_rate || frame.channels != self.channels {
            return Err(fail(
                Operation::Encode,
                format!(
                    "input shape changed mid-stream: configured {} Hz / {} ch, got {} Hz / {} ch",
                    self.input_rate, self.channels, frame.sample_rate, frame.channels
                ),
            ));
        }
        let channels = usize::from(self.channels.max(1));
        let samples: Vec<i16> = frame
            .data
            .chunks_exact(2)
            .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        let samples = &samples[..samples.len() / channels * channels];
        let chunk: &[i16] = match self.resampler.as_mut() {
            Some(resampler) => {
                self.resampled.clear();
                resampler.process(samples, &mut self.resampled);
                &self.resampled
            }
            None => samples,
        };
        let buffered = (self.pcm.len() / channels) as u64;
        let event = self.clock.observe(
            frame.timestamp_us,
            buffered,
            (chunk.len() / channels) as u64,
        );
        match event {
            ClockEvent::Resynced { gap_us } => {
                debug!(
                    gap_us,
                    "AudioToolbox AAC: capture gap, re-anchoring output timestamps"
                );
            }
            ClockEvent::Trimmed {
                ahead_us,
                report: true,
                ..
            } => info!(
                ahead_us,
                "AudioToolbox AAC: output clock ahead of capture, discarding input"
            ),
            ClockEvent::Anchored | ClockEvent::Continuous | ClockEvent::Trimmed { .. } => {}
        }
        let skipped = usize::try_from(event.skip()).map_or(chunk.len(), |skip| {
            skip.saturating_mul(channels).min(chunk.len())
        });
        self.pcm.extend_from_slice(&chunk[skipped..]);
        while self.pcm.len() >= self.profile.frame_samples() * channels {
            self.encode_frame(false)?;
        }
        Ok(())
    }

    fn take_output(&mut self) -> Vec<AudioFrame> {
        std::mem::take(&mut self.pending)
    }

    fn codec_config(&self) -> Vec<u8> {
        self.audio_specific_config.clone()
    }

    fn flush(mut self) -> CodecResult<Vec<AudioFrame>> {
        // The encoder holds back its priming delay; bounded so a
        // misbehaving converter cannot spin.
        for _ in 0..8 {
            if !self.encode_frame(true)? {
                break;
            }
        }
        Ok(std::mem::take(&mut self.pending))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_frame_their_codec() {
        let lc = Profile::Lc(AacLc::new(48_000, 2).unwrap());
        assert_eq!((lc.codec(), lc.frame_samples()), (AudioCodec::Aac, 1024));
        assert_eq!(
            (Profile::Eld.codec(), Profile::Eld.frame_samples()),
            (AudioCodec::AacEld, 480)
        );
    }
}
