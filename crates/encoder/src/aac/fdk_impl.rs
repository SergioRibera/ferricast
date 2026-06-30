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
//! Timestamp scheme: we anchor on the first PCM chunk's
//! `timestamp_us`, then advance the per-output PTS by exactly
//! `1024 / sample_rate × 1e6` µs per emitted AAC frame. That keeps
//! audio frame deltas exact (no quantisation drift) and co-monotonic
//! with the video PTS source.

use bytes::Bytes;
use fdk_aac::enc::{
    AudioObjectType, BitRate, ChannelMode, EncodeInfo, Encoder, EncoderParams, Transport,
};
use ferricast_core::{AudioCodec, AudioEncoderConfig, AudioFrame, FerricastError, Result};

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
    /// PTS anchor — set from the first PCM chunk's `timestamp_us`.
    /// All subsequent output frames derive their PTS from this
    /// plus `samples_emitted / sample_rate`.
    pts_anchor_us: Option<u64>,
    /// Sample count (per channel) we've already emitted as
    /// AAC frames. Used to compute the next output PTS.
    samples_emitted: u64,
    /// Scratch buffer reused across `aacEncEncode` calls to avoid
    /// per-frame allocations.
    out_scratch: Vec<u8>,
    /// Frames produced by the most recent `encode()` (and any
    /// leftover from prior calls) waiting to be drained by
    /// `take_output()`.
    pending: Vec<AudioFrame>,
}

impl FdkAacEncoder {
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
            pts_anchor_us: None,
            samples_emitted: 0,
            out_scratch: vec![0u8; ENC_OUTPUT_SCRATCH],
            pending: Vec::new(),
        })
    }

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
        // Anchor PTS on the very first sample we ever see so the
        // audio elementary stream starts from the same clock origin
        // as the video capture path.
        if self.pts_anchor_us.is_none() {
            self.pts_anchor_us = Some(frame.timestamp_us);
        }

        // Reinterpret the raw S16LE bytes as i16 slices, append to
        // the working PCM buffer. Doing it byte-by-byte lets us cope
        // with chunk lengths that aren't a multiple of 2 (shouldn't
        // happen with PipeWire but cheap to be safe).
        let pcm_bytes = frame.data.as_ref();
        let usable = pcm_bytes.len() & !1; // round down to i16 boundary
        let mut i = 0;
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
                // PTS = anchor + samples_emitted / sample_rate.
                // Using µs ticks here (1e6) keeps the math in u64
                // with no rounding error worth worrying about at
                // 48 kHz; downstream MPEG-TS conversion scales by
                // 9/100 to land in 90 kHz ticks.
                let pts_offset_us =
                    self.samples_emitted.saturating_mul(1_000_000) / self.sample_rate.max(1) as u64;
                let timestamp_us = self
                    .pts_anchor_us
                    .unwrap_or(0)
                    .saturating_add(pts_offset_us);

                self.pending.push(AudioFrame {
                    codec: AudioCodec::Aac,
                    data: adts,
                    timestamp_us,
                    sample_rate: self.sample_rate,
                    channels: self.channels,
                });
                self.samples_emitted = self
                    .samples_emitted
                    .saturating_add(AAC_LC_FRAME_SAMPLES as u64);
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
    pub fn codec_config(&self) -> Vec<u8> {
        Vec::new()
    }

    pub fn flush(mut self) -> Result<Vec<AudioFrame>> {
        // Drain any final encoder state. fdk-aac's "flush" pattern
        // is to call encode() with an empty input slice repeatedly
        // until output_size hits zero; we cap iterations so a buggy
        // library version can't hang us.
        for _ in 0..8 {
            let info = match self.enc.encode(&[], &mut self.out_scratch) {
                Ok(i) => i,
                Err(_) => break,
            };
            if info.output_size == 0 {
                break;
            }
            let adts = Bytes::copy_from_slice(&self.out_scratch[..info.output_size]);
            let pts_offset_us =
                self.samples_emitted.saturating_mul(1_000_000) / self.sample_rate.max(1) as u64;
            let timestamp_us = self
                .pts_anchor_us
                .unwrap_or(0)
                .saturating_add(pts_offset_us);
            self.pending.push(AudioFrame {
                codec: AudioCodec::Aac,
                data: adts,
                timestamp_us,
                sample_rate: self.sample_rate,
                channels: self.channels,
            });
            self.samples_emitted = self
                .samples_emitted
                .saturating_add(AAC_LC_FRAME_SAMPLES as u64);
        }
        Ok(std::mem::take(&mut self.pending))
    }
}
