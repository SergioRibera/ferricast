//! "Decoder" for [`AudioCodec::Pcm`] packets: interleaved signed 16-bit
//! little-endian samples, as the demuxer emits for audio files it
//! decodes itself (MP3, FLAC, Vorbis, WAV). Only reinterprets the bytes.

use ferricast_core::{
    AudioCodec, AudioDecoder, AudioDecoderConfig, AudioFrame, DecodedAudio, FerricastError, Result,
};

#[derive(Debug, Default)]
pub struct PcmDecoder {
    configured: bool,
}

impl AudioDecoder for PcmDecoder {
    const CODEC: AudioCodec = AudioCodec::Pcm;

    fn configure(&mut self, config: &AudioDecoderConfig) -> Result<()> {
        if config.codec != AudioCodec::Pcm {
            return Err(FerricastError::Decode(format!(
                "PcmDecoder asked to decode {:?}",
                config.codec
            )));
        }
        self.configured = true;
        Ok(())
    }

    fn decode(&mut self, frame: AudioFrame) -> Result<Option<DecodedAudio>> {
        if !self.configured {
            return Err(FerricastError::Decode(
                "PcmDecoder::decode before configure()".into(),
            ));
        }
        if frame.channels == 0 || frame.sample_rate == 0 {
            return Err(FerricastError::Decode(
                "PCM packet without a sample rate or channel count".into(),
            ));
        }
        let (samples, _) = frame.data.as_chunks::<2>();
        let pcm = samples.iter().map(|s| i16::from_le_bytes(*s)).collect();
        Ok(Some(DecodedAudio {
            pcm,
            sample_rate: frame.sample_rate,
            channels: frame.channels,
            timestamp_us: frame.timestamp_us,
        }))
    }

    fn flush(&mut self) -> Result<Vec<DecodedAudio>> {
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;

    #[test]
    fn little_endian_samples_pass_through() {
        let mut decoder = PcmDecoder::default();
        decoder
            .configure(&AudioDecoderConfig {
                codec: AudioCodec::Pcm,
                sample_rate: 44_100,
                channels: 2,
            })
            .unwrap();
        let audio = decoder
            .decode(AudioFrame {
                codec: AudioCodec::Pcm,
                data: Bytes::from_static(&[0x01, 0x00, 0xFF, 0xFF]),
                timestamp_us: 7,
                sample_rate: 44_100,
                channels: 2,
            })
            .unwrap()
            .unwrap();
        assert_eq!(audio.pcm, [1, -1]);
        assert_eq!(audio.timestamp_us, 7);
    }
}
