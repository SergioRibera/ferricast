use bytes::Bytes;
use ferricast_core::{AudioCodec, AudioEncoder, AudioEncoderConfig, AudioFrame, FerricastError, Result};
use opus_rs::{Application, OpusEncoder as RawEncoder};

// 20 ms at 48 kHz — valid for Hybrid mode (frame_rate = 50).
// Chromecast Cast Streaming receivers expect RFC 7587 packets at
// this granularity; smaller sizes (e.g. 480) are also valid but
// 960 gives the best quality/latency trade-off for desktop mirroring.
const FRAME_SAMPLES: usize = 960;

// Generous upper bound for one encoded Opus packet (RFC 7587 §2.1.3
// permits up to ~1275 bytes for the highest bitrate; 4 KiB is safe).
const OUT_BUF: usize = 4096;

pub struct OpusEncoder {
    enc: Option<RawEncoder>,
    sample_rate: u32,
    channels: u16,
    // Interleaved f32 PCM accumulated across PipeWire chunks until
    // we have at least FRAME_SAMPLES × channels samples to give Opus.
    pcm_buf: Vec<f32>,
    pts_anchor_us: Option<u64>,
    samples_emitted: u64,
    pending: Vec<AudioFrame>,
}

impl OpusEncoder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for OpusEncoder {
    fn default() -> Self {
        Self {
            enc: None,
            sample_rate: 0,
            channels: 0,
            pcm_buf: Vec::new(),
            pts_anchor_us: None,
            samples_emitted: 0,
            pending: Vec::new(),
        }
    }
}

impl AudioEncoder for OpusEncoder {
    fn configure(&mut self, config: &AudioEncoderConfig) -> Result<()> {
        if config.codec != AudioCodec::Opus {
            return Err(FerricastError::Encoder(format!(
                "OpusEncoder: only Opus supported, got {:?}",
                config.codec
            )));
        }
        if !matches!(config.channels, 1 | 2) {
            return Err(FerricastError::Encoder(format!(
                "opus-rs: unsupported channel count {}",
                config.channels
            )));
        }
        let mut enc = RawEncoder::new(
            config.sample_rate as i32,
            config.channels as usize,
            Application::Audio,
        )
        .map_err(|e| FerricastError::Encoder(format!("opus-rs: new failed: {e}")))?;
        enc.bitrate_bps = config.bitrate_kbps.saturating_mul(1000) as i32;
        tracing::info!(
            sample_rate = config.sample_rate,
            channels = config.channels,
            bitrate_kbps = config.bitrate_kbps,
            "Opus encoder backend: opus-rs (pure Rust)"
        );
        self.enc = Some(enc);
        self.sample_rate = config.sample_rate;
        self.channels = config.channels;
        self.pcm_buf = Vec::with_capacity(FRAME_SAMPLES * config.channels as usize * 4);
        Ok(())
    }

    fn encode(&mut self, frame: &AudioFrame) -> Result<()> {
        if frame.codec != AudioCodec::Pcm {
            return Err(FerricastError::Encoder(format!(
                "opus-rs: expected PCM input, got {:?}",
                frame.codec
            )));
        }
        let enc = self.enc.as_mut().ok_or_else(|| {
            FerricastError::Encoder("opus-rs: encode called before configure".into())
        })?;

        if self.pts_anchor_us.is_none() {
            self.pts_anchor_us = Some(frame.timestamp_us);
        }

        // S16LE bytes → interleaved f32 in [-1.0, 1.0].
        let pcm_bytes = frame.data.as_ref();
        let usable = pcm_bytes.len() & !1;
        let mut i = 0;
        while i < usable {
            let s = i16::from_le_bytes([pcm_bytes[i], pcm_bytes[i + 1]]);
            self.pcm_buf.push(s as f32 / 32768.0);
            i += 2;
        }

        let samples_per_frame = FRAME_SAMPLES * self.channels as usize;
        let mut out = vec![0u8; OUT_BUF];
        while self.pcm_buf.len() >= samples_per_frame {
            let input_view = &self.pcm_buf[..samples_per_frame];
            let n = enc
                .encode(input_view, FRAME_SAMPLES, &mut out)
                .map_err(|e| FerricastError::Encoder(format!("opus-rs: encode failed: {e}")))?;
            self.pcm_buf.drain(..samples_per_frame);

            if n > 0 {
                let pts_offset_us = self
                    .samples_emitted
                    .saturating_mul(1_000_000)
                    / self.sample_rate.max(1) as u64;
                let timestamp_us = self
                    .pts_anchor_us
                    .unwrap_or(0)
                    .saturating_add(pts_offset_us);
                self.pending.push(AudioFrame {
                    codec: AudioCodec::Opus,
                    data: Bytes::copy_from_slice(&out[..n]),
                    timestamp_us,
                    sample_rate: self.sample_rate,
                    channels: self.channels,
                });
                self.samples_emitted =
                    self.samples_emitted.saturating_add(FRAME_SAMPLES as u64);
            }
        }
        Ok(())
    }

    fn take_output(&mut self) -> Vec<AudioFrame> {
        std::mem::take(&mut self.pending)
    }

    fn flush(mut self) -> Result<Vec<AudioFrame>> {
        Ok(std::mem::take(&mut self.pending))
    }
}
