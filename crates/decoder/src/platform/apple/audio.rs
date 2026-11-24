//! AAC-LC decode through AudioToolbox's `AudioConverter`.
//!
//! Input is ADTS (one or more frames per payload) or raw access units
//! with the shape from the stream metadata. Each access unit is fed to a
//! converter built for its shape, with the matching AAC-LC
//! `AudioSpecificConfig` as its magic cookie; a shape change rebuilds the
//! converter. Output is interleaved 16-bit PCM.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr::{self, NonNull};

use ferricast_codec::aac::AacLc;
use ferricast_codec::{AudioDecoderRegistry, Candidate};
use ferricast_core::{
    Acceleration, AudioCodec, AudioDecoder, AudioDecoderConfig, AudioFrame, BackendId, CodecError,
    CodecKind, CodecResult, CodecTarget, DecodedAudio, Operation,
};
use objc2_audio_toolbox::{
    AudioConverterDispose, AudioConverterFillComplexBuffer, AudioConverterNew, AudioConverterRef,
    AudioConverterSetProperty, kAudioConverterDecompressionMagicCookie,
};
use objc2_core_audio_types::{
    AudioBuffer, AudioBufferList, AudioStreamBasicDescription, AudioStreamPacketDescription,
    kAudioFormatFlagIsPacked, kAudioFormatFlagIsSignedInteger, kAudioFormatLinearPCM,
    kAudioFormatMPEG4AAC,
};
use tracing::debug;

use super::os_status;
use crate::adts::{self, Shape};

const BACKEND: BackendId = BackendId::AudioToolbox;
/// Samples per channel in one AAC-LC frame.
const FRAME_SAMPLES: u32 = 1024;
/// Room for one decoded frame even if the decoder upsamples (SBR).
const MAX_OUTPUT_FRAMES: u32 = 2 * FRAME_SAMPLES;
/// What the input callback returns once its packet is consumed.
const NO_MORE_INPUT: i32 = i32::from_be_bytes(*b"!dat");
const MICROS_PER_SECOND: u64 = 1_000_000;

/// Adds the AudioToolbox AAC decoder.
pub(crate) fn audio(registry: AudioDecoderRegistry) -> AudioDecoderRegistry {
    if registry.target().codec != CodecKind::Audio(AudioCodec::Aac) {
        return registry;
    }
    registry.candidate(Candidate::new(
        BACKEND,
        Acceleration::Software,
        ferricast_codec::open_audio_decoder::<AtAacDecoder>,
    ))
}

fn fail(what: impl std::fmt::Display) -> CodecError {
    CodecError::backend(BACKEND, Operation::Decode, what)
}

/// An owned `AudioConverterRef`.
struct Converter(AudioConverterRef);

// SAFETY: an AudioConverter may be used from any thread as long as calls
// do not overlap; it is only reached through `&mut AtAacDecoder`.
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
    fn open(shape: Shape) -> CodecResult<Self> {
        let lc = AacLc::new(shape.sample_rate, shape.channels).ok_or_else(|| {
            CodecError::UnsupportedStream {
                backend: BACKEND,
                reason: format!(
                    "AAC at {} Hz with {} channels has no AAC-LC configuration",
                    shape.sample_rate, shape.channels
                ),
            }
        })?;
        let channels = u32::from(shape.channels);
        let aac = AudioStreamBasicDescription {
            mSampleRate: f64::from(shape.sample_rate),
            mFormatID: kAudioFormatMPEG4AAC,
            mFormatFlags: 0,
            mBytesPerPacket: 0,
            mFramesPerPacket: FRAME_SAMPLES,
            mBytesPerFrame: 0,
            mChannelsPerFrame: channels,
            mBitsPerChannel: 0,
            mReserved: 0,
        };
        let pcm = AudioStreamBasicDescription {
            mSampleRate: f64::from(shape.sample_rate),
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
                NonNull::from(&aac),
                NonNull::from(&pcm),
                NonNull::from(&mut raw),
            )
        };
        if status != 0 || raw.is_null() {
            return Err(CodecError::UnsupportedStream {
                backend: BACKEND,
                reason: format!(
                    "no AAC decoder for {} Hz / {} channels: {}",
                    shape.sample_rate,
                    shape.channels,
                    os_status(status)
                ),
            });
        }
        let converter = Self(raw);
        let cookie = lc.audio_specific_config();
        // SAFETY: the converter is live; the cookie is `cookie.len()` bytes.
        let status = unsafe {
            AudioConverterSetProperty(
                converter.0,
                kAudioConverterDecompressionMagicCookie,
                cookie.len() as u32,
                NonNull::from(&cookie).cast(),
            )
        };
        if status != 0 {
            return Err(fail(format!(
                "setting the AudioSpecificConfig: {}",
                os_status(status)
            )));
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

/// AAC decoder over AudioToolbox.
#[derive(Default)]
pub(crate) struct AtAacDecoder {
    configured: Option<Shape>,
    ready: bool,
    active: Option<(Shape, Converter)>,
    pcm: Vec<i16>,
    pending: VecDeque<DecodedAudio>,
}

impl AtAacDecoder {
    /// Decodes one raw access unit, appending its PCM to `pcm`; returns
    /// the frames decoded.
    fn decode_unit(&mut self, shape: Shape, payload: &[u8]) -> CodecResult<u64> {
        if self
            .active
            .as_ref()
            .is_none_or(|(current, _)| *current != shape)
        {
            self.active = Some((shape, Converter::open(shape)?));
        }
        let Some((_, converter)) = &self.active else {
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
        let channels = u32::from(shape.channels);
        let start = self.pcm.len();
        self.pcm
            .resize(start + (MAX_OUTPUT_FRAMES * channels) as usize, 0);
        let mut output = AudioBufferList {
            mNumberBuffers: 1,
            mBuffers: [AudioBuffer {
                mNumberChannels: channels,
                mDataByteSize: MAX_OUTPUT_FRAMES * channels * 2,
                mData: self.pcm[start..].as_mut_ptr().cast(),
            }],
        };
        let mut frames = MAX_OUTPUT_FRAMES;
        // SAFETY: `feed` points into `payload`, alive for the call; the
        // output list points at `MAX_OUTPUT_FRAMES` frames of `pcm`.
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
        let decoded = frames.min(MAX_OUTPUT_FRAMES);
        self.pcm.truncate(start + (decoded * channels) as usize);
        if status != 0 && status != NO_MORE_INPUT {
            return Err(fail(format!(
                "AudioConverterFillComplexBuffer: {}",
                os_status(status)
            )));
        }
        Ok(u64::from(decoded))
    }
}

impl AudioDecoder for AtAacDecoder {
    const CODEC: AudioCodec = AudioCodec::Aac;

    fn configure(&mut self, config: &AudioDecoderConfig) -> CodecResult<()> {
        if config.codec != AudioCodec::Aac {
            return Err(CodecError::UnsupportedConfig {
                backend: BACKEND,
                what: format!("{} in the AAC decoder", config.codec),
            });
        }
        self.configured = (config.sample_rate > 0 && config.channels > 0).then_some(Shape {
            sample_rate: config.sample_rate,
            channels: config.channels,
        });
        self.ready = true;
        self.pending.clear();
        Ok(())
    }

    fn decode(&mut self, frame: AudioFrame) -> CodecResult<Option<DecodedAudio>> {
        if !self.ready {
            return Err(CodecError::NotConfigured {
                target: CodecTarget::decode(AudioCodec::Aac),
                operation: Operation::Decode,
            });
        }
        let mut base_us = frame.timestamp_us;
        let mut run: Option<(Shape, u64)> = None;
        let mut result = Ok(());
        for unit in adts::units(&frame.data) {
            let unit = match unit {
                Ok(unit) => unit,
                Err(error) => {
                    result = Err(fail(format!("AAC: {error}")));
                    break;
                }
            };
            let Some(shape) = unit.shape.or(self.configured) else {
                result = Err(fail(
                    "raw (non-ADTS) AAC with no sample rate / channels configured",
                ));
                break;
            };
            if let Some((previous, samples)) = run
                && previous != shape
            {
                self.finish_run(previous, base_us);
                base_us += samples * MICROS_PER_SECOND / u64::from(previous.sample_rate);
                run = None;
            }
            match self.decode_unit(shape, unit.payload) {
                Ok(decoded) => {
                    let samples = run.map_or(0, |(_, samples)| samples);
                    run = Some((shape, samples + decoded));
                }
                Err(error) => {
                    result = Err(error);
                    break;
                }
            }
        }
        if let Some((shape, _)) = run {
            self.finish_run(shape, base_us);
        }
        // Frames decoded before a bad one are still good audio.
        result?;
        Ok(self.pending.pop_front())
    }

    fn flush(&mut self) -> CodecResult<Vec<DecodedAudio>> {
        Ok(self.pending.drain(..).collect())
    }
}

impl AtAacDecoder {
    /// Queues the PCM decoded so far as one run of `shape`.
    fn finish_run(&mut self, shape: Shape, timestamp_us: u64) {
        if self.pcm.is_empty() {
            return;
        }
        self.pending.push_back(DecodedAudio {
            pcm: std::mem::take(&mut self.pcm),
            sample_rate: shape.sample_rate,
            channels: shape.channels,
            timestamp_us,
        });
    }
}
