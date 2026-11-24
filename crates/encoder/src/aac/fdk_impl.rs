//! `fdk-aac` backed AAC-LC encoder.
//!
//! Wraps the `fdk-aac` Rust binding (which itself wraps Fraunhofer's
//! reference libfdk-aac C library) into the [`AudioEncoder`] trait.
//! Output is ADTS-framed AAC-LC, ready to drop straight into an
//! MPEG-TS PES payload (the chromecast HLS path's only audio
//! consumer today). Each emitted [`AudioFrame`] carries the standard
//! 1024-sample AAC-LC frame size with a PTS derived from the
//! upstream capture clock so audio shares a timeline with video.
//!
//! Timestamp scheme: see [`AudioClock`] — anchored on the first PCM
//! chunk, advanced by exactly `1024 / sample_rate × 1e6` µs per
//! emitted AAC frame (no quantisation drift, co-monotonic with video),
//! and re-anchored when the capture timestamps jump forward past a gap.

use bytes::Bytes;
use fdk_aac::enc::{
    AudioObjectType, BitRate, ChannelMode, EncodeInfo, Encoder, EncoderParams, Transport,
};
use ferricast_core::{
    AudioCodec, AudioEncoder, AudioEncoderConfig, AudioFrame, BackendId, CodecError, CodecResult,
    FerricastError, Operation, Result,
};
use tracing::{debug, info};

use crate::audio_clock::{AudioClock, ClockEvent};

/// AAC-LC frame size in samples per channel. Fixed by the codec —
/// every other parameter (bitrate, profile) can change but this is
/// always 1024 for AAC-LC.
const AAC_LC_FRAME_SAMPLES: usize = 1024;

/// Generous scratch buffer for one ADTS-framed AAC frame. A
/// stereo 320 kbps frame is ~900 bytes; 8 KiB leaves margin for VBR
/// peaks and future high-bitrate configs.
const ENC_OUTPUT_SCRATCH: usize = 8 * 1024;

pub struct FdkAacEncoder {
    enc: Encoder,
    sample_rate: u32,
    channels: u16,
    /// Interleaved S16LE input the encoder hasn't consumed yet. We
    /// keep accumulating until we have at least one full AAC frame
    /// (`AAC_LC_FRAME_SAMPLES × channels` i16s), then drain in one
    /// or more `enc.encode()` calls.
    pcm_buffer: Vec<i16>,
    /// Output timestamps, anchored on the capture clock.
    clock: AudioClock,
    /// Scratch buffer reused across `aacEncEncode` calls to avoid
    /// per-frame allocations.
    out_scratch: Vec<u8>,
    /// Frames produced by the most recent `encode()` (and any
    /// leftover from prior calls) waiting to be drained by
    /// `take_output()`.
    pending: Vec<AudioFrame>,
}

impl FdkAacEncoder {
    /// Opens libfdk-aac for `config`.
    ///
    /// # Errors
    /// [`FerricastError::Encoder`] for an unsupported channel count or a
    /// library failure.
    pub fn new(config: &AudioEncoderConfig) -> Result<Self> {
        let channel_mode = match config.channels {
            1 => ChannelMode::Mono,
            2 => ChannelMode::Stereo,
            other => {
                return Err(FerricastError::Encoder(format!(
                    "fdk-aac: unsupported channel count {other} (1 or 2 supported)"
                )));
            }
        };
        let params = EncoderParams {
            bit_rate: BitRate::Cbr(config.bitrate_kbps.saturating_mul(1000)),
            sample_rate: config.sample_rate,
            transport: Transport::Adts,
            channels: channel_mode,
            audio_object_type: AudioObjectType::Mpeg4LowComplexity,
        };
        let enc = Encoder::new(params)
            .map_err(|e| FerricastError::Encoder(format!("fdk-aac: Encoder::new failed: {e}")))?;
        Ok(Self {
            enc,
            sample_rate: config.sample_rate,
            channels: config.channels,
            pcm_buffer: Vec::with_capacity(
                AAC_LC_FRAME_SAMPLES * config.channels.max(1) as usize * 4,
            ),
            clock: AudioClock::new(config.sample_rate),
            out_scratch: vec![0u8; ENC_OUTPUT_SCRATCH],
            pending: Vec::new(),
        })
    }

    /// Feeds one PCM chunk; finished frames are taken with `take_output`.
    ///
    /// # Errors
    /// [`FerricastError::Encoder`] for non-PCM input or a library failure.
    pub fn encode(&mut self, frame: &AudioFrame) -> Result<()> {
        if frame.codec != AudioCodec::Pcm {
            return Err(FerricastError::Encoder(format!(
                "fdk-aac: expected PCM input, got {:?}",
                frame.codec
            )));
        }
        if frame.sample_rate != self.sample_rate || frame.channels != self.channels {
            return Err(FerricastError::Encoder(format!(
                "fdk-aac: input shape changed mid-stream: configured {}Hz/{}ch, got {}Hz/{}ch",
                self.sample_rate, self.channels, frame.sample_rate, frame.channels
            )));
        }
        // Anchor on the first chunk (same clock origin as video), jump
        // forward after capture gaps so frames never go stale, and
        // trim input when the sample count runs ahead of capture.
        let channels = usize::from(self.channels.max(1));
        let pcm_bytes = frame.data.as_ref();
        let usable = pcm_bytes.len() & !1; // round down to i16 boundary
        let buffered = (self.pcm_buffer.len() / channels) as u64;
        let chunk = (usable / 2 / channels) as u64;
        let event = self.clock.observe(frame.timestamp_us, buffered, chunk);
        match event {
            ClockEvent::Resynced { gap_us } => debug!(
                gap_us,
                "fdk-aac: capture gap, re-anchoring output timestamps"
            ),
            ClockEvent::Trimmed {
                ahead_us,
                report: true,
                ..
            } => info!(
                ahead_us,
                "fdk-aac: output clock ahead of capture, discarding input samples"
            ),
            ClockEvent::Anchored | ClockEvent::Continuous | ClockEvent::Trimmed { .. } => {}
        }

        // Reinterpret the raw S16LE bytes as i16 slices, append to
        // the working PCM buffer. Doing it byte-by-byte lets us cope
        // with chunk lengths that aren't a multiple of 2 (shouldn't
        // happen with PipeWire but cheap to be safe).
        let skipped_bytes = usize::try_from(event.skip())
            .map_or(usable, |skip| skip.saturating_mul(channels * 2).min(usable));
        let mut i = skipped_bytes;
        while i < usable {
            let s = i16::from_le_bytes([pcm_bytes[i], pcm_bytes[i + 1]]);
            self.pcm_buffer.push(s);
            i += 2;
        }

        // Drain as many full AAC frames as we can.
        let samples_per_frame = AAC_LC_FRAME_SAMPLES * self.channels.max(1) as usize;
        while self.pcm_buffer.len() >= samples_per_frame {
            // Snapshot the slice we feed in; copy out once the
            // encoder confirms consumption so we don't truncate
            // before the AAC library has actually used the bytes.
            let input_view = &self.pcm_buffer[..samples_per_frame];
            let info: EncodeInfo = self
                .enc
                .encode(input_view, &mut self.out_scratch)
                .map_err(|e| FerricastError::Encoder(format!("fdk-aac: encode failed: {e}")))?;

            if info.input_consumed == 0 && info.output_size == 0 {
                // Encoder asked for more input but didn't consume
                // any of what we gave it — shouldn't happen for
                // AAC-LC at one-frame-aligned input boundaries, but
                // bail out rather than spin.
                break;
            }

            // Drop the samples the encoder actually consumed.
            if info.input_consumed > 0 {
                self.pcm_buffer.drain(..info.input_consumed);
            }

            if info.output_size > 0 {
                let adts = Bytes::copy_from_slice(&self.out_scratch[..info.output_size]);
                let timestamp_us = self.clock.next_timestamp_us();

                self.pending.push(AudioFrame {
                    codec: AudioCodec::Aac,
                    data: adts,
                    timestamp_us,
                    sample_rate: self.sample_rate,
                    channels: self.channels,
                });
                self.clock.advance(AAC_LC_FRAME_SAMPLES as u64);
            }
        }

        Ok(())
    }

    pub fn take_output(&mut self) -> Vec<AudioFrame> {
        std::mem::take(&mut self.pending)
    }

    /// AAC has no out-of-band parameter sets when ADTS is the
    /// transport — every frame is self-describing. Return empty to
    /// keep the trait contract honest.
    #[must_use]
    pub fn codec_config(&self) -> Vec<u8> {
        Vec::new()
    }

    /// Drains the encoder.
    ///
    /// # Errors
    /// Never in practice: a library failure while draining ends the drain.
    pub fn flush(mut self) -> Result<Vec<AudioFrame>> {
        // Drain any final encoder state. fdk-aac's "flush" pattern
        // is to call encode() with an empty input slice repeatedly
        // until output_size hits zero; we cap iterations so a buggy
        // library version can't hang us.
        for _ in 0..8 {
            let Ok(info) = self.enc.encode(&[], &mut self.out_scratch) else {
                break;
            };
            if info.output_size == 0 {
                break;
            }
            let adts = Bytes::copy_from_slice(&self.out_scratch[..info.output_size]);
            let timestamp_us = self.clock.next_timestamp_us();
            self.pending.push(AudioFrame {
                codec: AudioCodec::Aac,
                data: adts,
                timestamp_us,
                sample_rate: self.sample_rate,
                channels: self.channels,
            });
            self.clock.advance(AAC_LC_FRAME_SAMPLES as u64);
        }
        Ok(std::mem::take(&mut self.pending))
    }
}

impl FdkAacEncoder {
    /// Opens an encoder for `config`, which must ask for AAC-LC.
    ///
    /// # Errors
    /// [`CodecError::UnsupportedConfig`] for another codec or a channel
    /// count libfdk-aac is not set up for here; a backend error when
    /// libfdk-aac refuses the parameters.
    pub fn open(config: &AudioEncoderConfig) -> CodecResult<Self> {
        if config.codec != AudioCodec::Aac {
            return Err(CodecError::UnsupportedConfig {
                backend: BackendId::FdkAac,
                what: format!("{} in the AAC-LC encoder", config.codec),
            });
        }
        if !matches!(config.channels, 1 | 2) {
            return Err(CodecError::UnsupportedConfig {
                backend: BackendId::FdkAac,
                what: format!("{} channels (1 or 2 are supported)", config.channels),
            });
        }
        Self::new(config).map_err(CodecError::from_backend(BackendId::FdkAac, Operation::Open))
    }
}

impl AudioEncoder for FdkAacEncoder {
    fn configure(&mut self, config: &AudioEncoderConfig) -> CodecResult<()> {
        *self = Self::open(config)?;
        Ok(())
    }

    fn encode(&mut self, frame: &AudioFrame) -> CodecResult<()> {
        FdkAacEncoder::encode(self, frame).map_err(CodecError::from_backend(
            BackendId::FdkAac,
            Operation::Encode,
        ))
    }

    fn take_output(&mut self) -> Vec<AudioFrame> {
        FdkAacEncoder::take_output(self)
    }

    fn codec_config(&self) -> Vec<u8> {
        FdkAacEncoder::codec_config(self)
    }

    fn flush(self) -> CodecResult<Vec<AudioFrame>> {
        FdkAacEncoder::flush(self).map_err(CodecError::from_backend(
            BackendId::FdkAac,
            Operation::Flush,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    fn pcm(timestamp_us: u64) -> AudioFrame {
        // One AAC frame of stereo silence, S16LE.
        AudioFrame {
            codec: AudioCodec::Pcm,
            data: Bytes::from(vec![0u8; AAC_LC_FRAME_SAMPLES * 2 * 2]),
            timestamp_us,
            sample_rate: SR,
            channels: 2,
        }
    }

    /// A capture gap must move output timestamps forward instead of
    /// continuing the pre-gap sample count (which the HLS segmenter
    /// would drop as stale forever).
    #[test]
    fn output_resyncs_after_capture_gap() {
        let mut enc = FdkAacEncoder::new(&AudioEncoderConfig {
            codec: AudioCodec::Aac,
            sample_rate: SR,
            channels: 2,
            bitrate_kbps: 128,
            backend: ferricast_core::BackendSelection::default(),
        })
        .expect("fdk-aac");
        let frame_us = 1_000_000 * AAC_LC_FRAME_SAMPLES as u64 / u64::from(SR);
        let start = 1_000_000;
        let mut before = Vec::new();
        for i in 0..8 {
            enc.encode(&pcm(start + i * frame_us)).expect("encode");
            before.extend(enc.take_output());
        }
        assert!(!before.is_empty());
        assert!(before.iter().all(|f| f.timestamp_us < start + 8 * frame_us));

        let resumed = start + 5_000_000;
        enc.encode(&pcm(resumed)).expect("encode");
        let after = enc.take_output();
        assert!(!after.is_empty());
        assert_eq!(after[0].timestamp_us, resumed);
    }
}
