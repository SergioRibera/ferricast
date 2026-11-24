//! AAC-LC encode through AudioToolbox's `AudioConverter`.
//!
//! The converter pulls PCM through an input callback; each call feeds it
//! exactly one AAC frame (1024 samples per channel) and asks for one
//! packet. The encoder's priming delay means the first calls yield
//! nothing. Packets come out raw and are framed as ADTS, the shape the
//! MPEG-TS / HLS pipeline carries.

use std::ffi::c_void;
use std::ptr::{self, NonNull};

use bytes::Bytes;
use ferricast_codec::aac::AacLc;
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
    kAudioFormatMPEG4AAC,
};
use tracing::{debug, info};

use super::os_status;
use crate::audio_clock::{AudioClock, ClockEvent};

const BACKEND: BackendId = BackendId::AudioToolbox;
/// Samples per channel in one AAC-LC frame.
const FRAME_SAMPLES: usize = 1024;
/// What the input callback returns once its frame is consumed, so the
/// converter stops pulling; not an error.
const NO_MORE_INPUT: i32 = i32::from_be_bytes(*b"!dat");

/// Adds the AudioToolbox AAC-LC encoder.
pub(crate) fn audio(registry: AudioEncoderRegistry) -> AudioEncoderRegistry {
    if registry.target().codec != CodecKind::Audio(AudioCodec::Aac) {
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

/// The PCM one fill call may pull: one AAC frame.
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

/// AAC-LC encoder over AudioToolbox.
pub(crate) struct AtAacEncoder {
    converter: Converter,
    shape: AacLc,
    sample_rate: u32,
    channels: u16,
    pcm: Vec<i16>,
    clock: AudioClock,
    packet: Vec<u8>,
    pending: Vec<AudioFrame>,
}

impl AtAacEncoder {
    fn open(config: &AudioEncoderConfig) -> CodecResult<Self> {
        if config.codec != AudioCodec::Aac {
            return Err(CodecError::UnsupportedConfig {
                backend: BACKEND,
                what: format!("{} in the AAC-LC encoder", config.codec),
            });
        }
        let shape = AacLc::new(config.sample_rate, config.channels).ok_or_else(|| {
            CodecError::UnsupportedConfig {
                backend: BACKEND,
                what: format!(
                    "AAC-LC at {} Hz with {} channels",
                    config.sample_rate, config.channels
                ),
            }
        })?;
        let channels = u32::from(config.channels);
        let pcm = AudioStreamBasicDescription {
            mSampleRate: f64::from(config.sample_rate),
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
            mBytesPerPacket: 2 * channels,
            mFramesPerPacket: 1,
            mBytesPerFrame: 2 * channels,
            mChannelsPerFrame: channels,
            mBitsPerChannel: 16,
            mReserved: 0,
        };
        let aac = AudioStreamBasicDescription {
            mSampleRate: f64::from(config.sample_rate),
            mFormatID: kAudioFormatMPEG4AAC,
            mFormatFlags: 0,
            mBytesPerPacket: 0,
            mFramesPerPacket: FRAME_SAMPLES as u32,
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
                reason: format!("no AAC encoder for this input: {}", os_status(status)),
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
            return Err(CodecError::UnsupportedConfig {
                backend: BACKEND,
                what: format!(
                    "{} kbps at {} Hz with {} channels ({})",
                    config.bitrate_kbps,
                    config.sample_rate,
                    config.channels,
                    os_status(status)
                ),
            });
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
            sample_rate = config.sample_rate,
            channels = config.channels,
            bitrate_kbps = config.bitrate_kbps,
            "AAC encoder backend: AudioToolbox"
        );
        Ok(Self {
            converter,
            shape,
            sample_rate: config.sample_rate,
            channels: config.channels,
            pcm: Vec::with_capacity(FRAME_SAMPLES * usize::from(config.channels) * 4),
            clock: AudioClock::new(config.sample_rate),
            packet: vec![0; max_packet as usize],
            pending: Vec::new(),
        })
    }

    /// Feeds one AAC frame of PCM from the front of `pcm` (or, with
    /// `drain`, none, to pull out what the encoder still holds) and
    /// queues the packet it yields; returns whether one came out.
    fn encode_frame(&mut self, drain: bool) -> CodecResult<bool> {
        let channels = u32::from(self.channels);
        let mut feed = Feed {
            data: self.pcm.as_ptr(),
            frames: if drain { 0 } else { FRAME_SAMPLES as u32 },
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
            self.pcm.drain(..FRAME_SAMPLES * usize::from(self.channels));
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
        let adts = self
            .shape
            .adts_frame(&self.packet[..len])
            .ok_or_else(|| fail(Operation::Encode, "the AAC packet is too long for ADTS"))?;
        self.pending.push(AudioFrame {
            codec: AudioCodec::Aac,
            data: Bytes::from(adts),
            timestamp_us: self.clock.next_timestamp_us(),
            sample_rate: self.sample_rate,
            channels: self.channels,
        });
        self.clock.advance(FRAME_SAMPLES as u64);
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
        if frame.sample_rate != self.sample_rate || frame.channels != self.channels {
            return Err(fail(
                Operation::Encode,
                format!(
                    "input shape changed mid-stream: configured {} Hz / {} ch, got {} Hz / {} ch",
                    self.sample_rate, self.channels, frame.sample_rate, frame.channels
                ),
            ));
        }
        let channels = usize::from(self.channels.max(1));
        let bytes = frame.data.as_ref();
        let usable = bytes.len() & !1;
        let buffered = (self.pcm.len() / channels) as u64;
        let chunk = (usable / 2 / channels) as u64;
        let event = self.clock.observe(frame.timestamp_us, buffered, chunk);
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
        let skipped = usize::try_from(event.skip())
            .map_or(usable, |skip| skip.saturating_mul(channels * 2).min(usable));
        self.pcm.extend(
            bytes[skipped..usable]
                .chunks_exact(2)
                .map(|pair| i16::from_le_bytes([pair[0], pair[1]])),
        );
        while self.pcm.len() >= FRAME_SAMPLES * channels {
            self.encode_frame(false)?;
        }
        Ok(())
    }

    fn take_output(&mut self) -> Vec<AudioFrame> {
        std::mem::take(&mut self.pending)
    }

    fn codec_config(&self) -> Vec<u8> {
        self.shape.audio_specific_config().to_vec()
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
