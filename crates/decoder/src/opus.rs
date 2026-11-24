//! Pure-Rust Opus decoder over `opus-rs`, for Opus tracks in `WebM`,
//! Matroska and MP4.

use ferricast_core::{
    AudioCodec, AudioDecoder, AudioDecoderConfig, AudioFrame, DecodedAudio, FerricastError, Result,
};
use ferricast_core::{BackendId, CodecError, CodecResult, Operation};

/// Opus always decodes at 48 kHz (RFC 7845 §5.1).
const RATE: u32 = 48_000;
/// Longest Opus packet: 120 ms at 48 kHz (RFC 6716 §3.2.5).
const MAX_FRAME: usize = 5_760;

#[derive(Default)]
pub struct OpusRsDecoder {
    channels: u16,
    /// Decoder for packets coded with the stream's channel count.
    native: Option<opus_rs::OpusDecoder>,
    /// Stereo encoders may code quiet stretches as mono packets, which a
    /// stereo `opus-rs` decoder rejects; those go through a mono decoder
    /// and are duplicated to both channels.
    mono: Option<opus_rs::OpusDecoder>,
}

impl OpusRsDecoder {
    fn try_configure(&mut self, config: &AudioDecoderConfig) -> Result<()> {
        if config.codec != AudioCodec::Opus {
            return Err(FerricastError::Decode(format!(
                "OpusRsDecoder asked to decode {:?}; only Opus supported",
                config.codec
            )));
        }
        if !(1..=2).contains(&config.channels) {
            return Err(FerricastError::Decode(format!(
                "Opus with {} channels is not supported (mono and stereo only)",
                config.channels
            )));
        }
        self.channels = config.channels;
        self.native = Some(new_decoder(config.channels)?);
        self.mono = None;
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "samples are clamped to [-1, 1] before scaling to i16"
    )]
    fn try_decode(&mut self, frame: &AudioFrame) -> Result<Option<DecodedAudio>> {
        let Some(native) = self.native.as_mut() else {
            return Err(FerricastError::Decode(
                "OpusRsDecoder::decode before configure()".into(),
            ));
        };
        let Some(samples) = packet_samples(&frame.data) else {
            return Err(FerricastError::Decode("Opus: malformed packet".into()));
        };
        let packet_stereo = frame.data.first().is_some_and(|toc| toc & 0x04 != 0);
        let channels = usize::from(self.channels);
        let mut pcm = vec![0f32; samples * channels];
        if packet_stereo || channels == 1 {
            native
                .decode(&frame.data, samples, &mut pcm)
                .map_err(|e| FerricastError::Decode(format!("Opus decode: {e}")))?;
        } else {
            let mono = match self.mono.as_mut() {
                Some(mono) => mono,
                None => self.mono.insert(new_decoder(1)?),
            };
            let mut single = vec![0f32; samples];
            mono.decode(&frame.data, samples, &mut single)
                .map_err(|e| FerricastError::Decode(format!("Opus decode: {e}")))?;
            for (pair, sample) in pcm.chunks_exact_mut(2).zip(single) {
                pair.fill(sample);
            }
        }
        Ok(Some(DecodedAudio {
            pcm: pcm
                .iter()
                .map(|s| (s.clamp(-1.0, 1.0) * 32767.0).round() as i16)
                .collect(),
            sample_rate: RATE,
            channels: self.channels,
            timestamp_us: frame.timestamp_us,
        }))
    }
}

impl AudioDecoder for OpusRsDecoder {
    const CODEC: AudioCodec = AudioCodec::Opus;

    fn configure(&mut self, config: &AudioDecoderConfig) -> CodecResult<()> {
        self.try_configure(config).map_err(CodecError::from_backend(
            BackendId::OpusRs,
            Operation::Configure,
        ))
    }

    fn decode(&mut self, frame: AudioFrame) -> CodecResult<Option<DecodedAudio>> {
        self.try_decode(&frame).map_err(CodecError::from_backend(
            BackendId::OpusRs,
            Operation::Decode,
        ))
    }

    fn flush(&mut self) -> CodecResult<Vec<DecodedAudio>> {
        Ok(Vec::new())
    }
}

fn new_decoder(channels: u16) -> Result<opus_rs::OpusDecoder> {
    opus_rs::OpusDecoder::new(RATE.cast_signed(), usize::from(channels))
        .map_err(|e| FerricastError::Decode(format!("Opus decoder: {e}")))
}

/// Samples per channel in one packet, from its TOC byte (RFC 6716 §3.1).
fn packet_samples(packet: &[u8]) -> Option<usize> {
    let toc = *packet.first()?;
    let config = toc >> 3;
    // Frame sizes in units of 2.5 ms (120 samples at 48 kHz).
    let frame = match config {
        0..=11 => [4, 8, 16, 24][usize::from(config % 4)],
        12..=15 => [4, 8][usize::from(config % 2)],
        _ => [1, 2, 4, 8][usize::from(config % 4)],
    };
    let frames = match toc & 3 {
        0 => 1,
        1 | 2 => 2,
        _ => usize::from(packet.get(1)? & 0x3F),
    };
    let samples = frames * frame * 120;
    (1..=MAX_FRAME).contains(&samples).then_some(samples)
}

#[cfg(test)]
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    reason = "test signal generation over small positive packet indices"
)]
mod tests {
    use bytes::Bytes;

    use super::*;

    fn config(channels: u16) -> AudioDecoderConfig {
        AudioDecoderConfig {
            extra_data: None,
            codec: AudioCodec::Opus,
            sample_rate: RATE,
            channels,
            backend: ferricast_core::BackendSelection::default(),
        }
    }

    #[test]
    fn packet_sizes_follow_the_toc() {
        assert_eq!(packet_samples(&[31 << 3]), Some(960));
        assert_eq!(packet_samples(&[(3 << 3) | 1]), Some(5_760));
        assert_eq!(packet_samples(&[(12 << 3) | 3, 3]), Some(1_440));
        assert_eq!(packet_samples(&[(12 << 3) | 3, 0]), None);
        assert_eq!(packet_samples(&[]), None);
    }

    #[test]
    fn encoded_stereo_audio_round_trips() {
        let mut encoder =
            opus_rs::OpusEncoder::new(48_000, 2, opus_rs::Application::Audio).expect("encoder");
        let mut decoder = OpusRsDecoder::default();
        decoder.configure(&config(2)).unwrap();
        let mut energy = 0f64;
        for n in 0..20 {
            let pcm: Vec<f32> = (0..960)
                .flat_map(|i| {
                    let t = (n * 960 + i) as f32 / 48_000.0;
                    let s = (t * 440.0 * std::f32::consts::TAU).sin() * 0.5;
                    [s, s]
                })
                .collect();
            let mut packet = vec![0u8; 1500];
            let len = encoder.encode(&pcm, 960, &mut packet).expect("encode");
            packet.truncate(len);
            let decoded = decoder
                .decode(AudioFrame {
                    codec: AudioCodec::Opus,
                    data: Bytes::from(packet),
                    timestamp_us: n as u64 * 20_000,
                    sample_rate: RATE,
                    channels: 2,
                })
                .unwrap()
                .unwrap();
            assert_eq!(decoded.pcm.len(), 960 * 2);
            assert_eq!(decoded.timestamp_us, n as u64 * 20_000);
            if n > 2 {
                energy += decoded
                    .pcm
                    .iter()
                    .map(|&s| f64::from(s).powi(2))
                    .sum::<f64>();
            }
        }
        assert!(energy > 0.0, "decoded only silence");
    }

    #[test]
    fn more_than_two_channels_are_refused() {
        assert!(OpusRsDecoder::default().configure(&config(6)).is_err());
    }
}
