use bytes::Bytes;
use ferricast_core::{
    AudioCodec, AudioEncoder, AudioEncoderConfig, AudioFrame, FerricastError, Result,
};
use ferricast_core::{BackendId, CodecError, CodecResult, Operation};
use opus_rs::{Application, OpusEncoder as RawEncoder};
use tracing::{debug, info};

use crate::audio_clock::{AudioClock, ClockEvent};

// 20 ms frames (frame_rate = 50, valid for Hybrid mode): 960 samples
// at 48 kHz. Chromecast Cast Streaming receivers expect RFC 7587
// packets at this granularity; it is the best quality/latency
// trade-off for desktop mirroring.
const FRAMES_PER_SECOND: u32 = 50;

// Input rates Opus encodes natively (RFC 6716 §2). Anything else
// (44.1 kHz in particular) needs resampling, which this crate lacks.
const SUPPORTED_RATES: [u32; 5] = [8_000, 12_000, 16_000, 24_000, 48_000];

// Generous upper bound for one encoded Opus packet (RFC 7587 §2.1.3
// permits up to ~1275 bytes for the highest bitrate; 4 KiB is safe).
const OUT_BUF: usize = 4096;

#[derive(Default)]
pub struct OpusRsEncoder {
    enc: Option<RawEncoder>,
    sample_rate: u32,
    channels: u16,
    // Samples per channel in one 20 ms frame at `sample_rate`.
    frame_samples: usize,
    // Interleaved f32 PCM accumulated across PipeWire chunks until
    // we have at least frame_samples × channels samples to give Opus.
    pcm_buf: Vec<f32>,
    // Output timestamps; created by `configure()` with the input rate.
    clock: Option<AudioClock>,
    pending: Vec<AudioFrame>,
}

impl OpusRsEncoder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl OpusRsEncoder {
    fn try_configure(&mut self, config: &AudioEncoderConfig) -> Result<()> {
        if config.codec != AudioCodec::Opus {
            return Err(FerricastError::Encoder(format!(
                "OpusRsEncoder: only Opus supported, got {:?}",
                config.codec
            )));
        }
        if !matches!(config.channels, 1 | 2) {
            return Err(FerricastError::Encoder(format!(
                "opus-rs: unsupported channel count {}",
                config.channels
            )));
        }
        if !SUPPORTED_RATES.contains(&config.sample_rate) {
            return Err(FerricastError::Encoder(format!(
                "opus-rs: {} Hz input is not an Opus rate (8/12/16/24/48 kHz); \
                 capture at 48 kHz or use AAC",
                config.sample_rate
            )));
        }
        let frame_samples = (config.sample_rate / FRAMES_PER_SECOND) as usize;
        let mut enc = RawEncoder::new(
            i32::try_from(config.sample_rate).map_err(|_| {
                FerricastError::Encoder(format!("opus-rs: {} Hz rate", config.sample_rate))
            })?,
            config.channels as usize,
            Application::Audio,
        )
        .map_err(|e| FerricastError::Encoder(format!("opus-rs: new failed: {e}")))?;
        enc.bitrate_bps =
            i32::try_from(config.bitrate_kbps.saturating_mul(1000)).unwrap_or(i32::MAX);
        tracing::info!(
            sample_rate = config.sample_rate,
            channels = config.channels,
            bitrate_kbps = config.bitrate_kbps,
            "Opus encoder backend: opus-rs (pure Rust)"
        );
        self.enc = Some(enc);
        self.sample_rate = config.sample_rate;
        self.channels = config.channels;
        self.frame_samples = frame_samples;
        self.pcm_buf = Vec::with_capacity(frame_samples * config.channels as usize * 4);
        self.clock = Some(AudioClock::new(config.sample_rate));
        Ok(())
    }

    fn try_encode(&mut self, frame: &AudioFrame) -> Result<()> {
        if frame.codec != AudioCodec::Pcm {
            return Err(FerricastError::Encoder(format!(
                "opus-rs: expected PCM input, got {:?}",
                frame.codec
            )));
        }
        let enc = self.enc.as_mut().ok_or_else(|| {
            FerricastError::Encoder("opus-rs: encode called before configure".into())
        })?;

        let clock = self.clock.as_mut().ok_or_else(|| {
            FerricastError::Encoder("opus-rs: encode called before configure".into())
        })?;
        // Anchor on the first chunk, jump forward after capture gaps and
        // trim input when the sample count runs ahead of capture.
        let channels = usize::from(self.channels.max(1));
        let pcm_bytes = frame.data.as_ref();
        let usable = pcm_bytes.len() & !1;
        let buffered = (self.pcm_buf.len() / channels) as u64;
        let chunk = (usable / 2 / channels) as u64;
        let event = clock.observe(frame.timestamp_us, buffered, chunk);
        match event {
            ClockEvent::Resynced { gap_us } => debug!(
                gap_us,
                "opus-rs: capture gap, re-anchoring output timestamps"
            ),
            ClockEvent::Trimmed {
                ahead_us,
                report: true,
                ..
            } => info!(
                ahead_us,
                "opus-rs: output clock ahead of capture, discarding input samples"
            ),
            ClockEvent::Anchored | ClockEvent::Continuous | ClockEvent::Trimmed { .. } => {}
        }

        // S16LE bytes → interleaved f32 in [-1.0, 1.0].
        let skipped_bytes = usize::try_from(event.skip())
            .map_or(usable, |skip| skip.saturating_mul(channels * 2).min(usable));
        let mut i = skipped_bytes;
        while i < usable {
            let s = i16::from_le_bytes([pcm_bytes[i], pcm_bytes[i + 1]]);
            self.pcm_buf.push(f32::from(s) / 32768.0);
            i += 2;
        }

        let samples_per_frame = self.frame_samples * self.channels as usize;
        let mut out = vec![0u8; OUT_BUF];
        while self.pcm_buf.len() >= samples_per_frame {
            let input_view = &self.pcm_buf[..samples_per_frame];
            let n = enc
                .encode(input_view, self.frame_samples, &mut out)
                .map_err(|e| FerricastError::Encoder(format!("opus-rs: encode failed: {e}")))?;
            self.pcm_buf.drain(..samples_per_frame);

            if n > 0 {
                let timestamp_us = clock.next_timestamp_us();
                self.pending.push(AudioFrame {
                    codec: AudioCodec::Opus,
                    data: Bytes::copy_from_slice(&out[..n]),
                    timestamp_us,
                    sample_rate: self.sample_rate,
                    channels: self.channels,
                });
                clock.advance(self.frame_samples as u64);
            }
        }
        Ok(())
    }
}

impl AudioEncoder for OpusRsEncoder {
    fn configure(&mut self, config: &AudioEncoderConfig) -> CodecResult<()> {
        self.try_configure(config).map_err(CodecError::from_backend(
            BackendId::OpusRs,
            Operation::Configure,
        ))
    }

    fn encode(&mut self, frame: &AudioFrame) -> CodecResult<()> {
        self.try_encode(frame).map_err(CodecError::from_backend(
            BackendId::OpusRs,
            Operation::Encode,
        ))
    }

    fn flush(self) -> CodecResult<Vec<AudioFrame>> {
        Ok(self.pending)
    }

    fn take_output(&mut self) -> Vec<AudioFrame> {
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(sample_rate: u32) -> AudioEncoderConfig {
        AudioEncoderConfig {
            codec: AudioCodec::Opus,
            sample_rate,
            channels: 2,
            bitrate_kbps: 96,
            backend: ferricast_core::BackendSelection::default(),
        }
    }

    #[test]
    fn cd_rate_is_rejected_up_front() {
        let err = OpusRsEncoder::new().configure(&config(44_100)).unwrap_err();
        assert!(err.to_string().contains("44100"), "{err}");
    }

    /// 20 ms of input at a non-48 kHz rate yields exactly one packet.
    #[test]
    fn frame_size_follows_sample_rate() {
        let mut enc = OpusRsEncoder::new();
        enc.configure(&config(24_000)).expect("configure");
        let pcm = AudioFrame {
            codec: AudioCodec::Pcm,
            data: Bytes::from(vec![0u8; 480 * 2 * 2]),
            timestamp_us: 0,
            sample_rate: 24_000,
            channels: 2,
        };
        enc.encode(&pcm).expect("encode");
        assert_eq!(enc.take_output().len(), 1);
    }
}
