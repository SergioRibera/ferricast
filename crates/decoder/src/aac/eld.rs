//! AAC-ELD decoder (libfdk-aac) for AirPlay screen-mirroring audio.
//!
//! Mirroring senders send raw AAC-ELD access units, 44.1 kHz stereo with
//! 480-sample frames, and never signal the configuration in-band: it is
//! implied by the stream's `SETUP` (`ct = 8`). The decoder is configured
//! with that `AudioSpecificConfig` up front and fed one access unit per
//! packet.

use fdk_aac_sys as sys;
use ferricast_core::{
    AudioCodec, AudioDecoder, AudioDecoderConfig, AudioFrame, DecodedAudio, FerricastError, Result,
};
use ferricast_core::{BackendId, CodecError, CodecResult, Operation};

use super::fdk::{RawDecoder, fdk_error};

/// `AudioSpecificConfig` of AAC-ELD at 44.1 kHz, stereo, 480-sample
/// frames: object type 39 (escaped), rate index 4, channel configuration
/// 2, frame length flag set.
const ELD_44100_STEREO_480: [u8; 4] = [0xf8, 0xe8, 0x50, 0x00];
/// Output buffer size: the widest frame libfdk-aac can produce, so a
/// stream that is not what `SETUP` announced still fits.
const MAX_FRAME_SAMPLES: usize = 2048 * 8;

/// Decodes raw AAC-ELD access units into interleaved 16-bit PCM.
#[derive(Default)]
pub struct FdkEldDecoder {
    decoder: Option<RawDecoder>,
    pcm: Vec<i16>,
}

impl FdkEldDecoder {
    fn try_configure(&mut self, config: &AudioDecoderConfig) -> Result<()> {
        if config.codec != AudioCodec::AacEld {
            return Err(FerricastError::Decode(format!(
                "FdkEldDecoder asked to decode {:?}",
                config.codec
            )));
        }
        if (config.sample_rate, config.channels) != (44_100, 2) {
            return Err(FerricastError::Decode(format!(
                "AAC-ELD at {} Hz with {} channels is not supported; AirPlay sends 44100 Hz stereo",
                config.sample_rate, config.channels
            )));
        }
        self.decoder = Some(RawDecoder::open(&ELD_44100_STEREO_480)?);
        self.pcm = vec![0; MAX_FRAME_SAMPLES];
        Ok(())
    }

    fn try_decode(&mut self, frame: AudioFrame) -> Result<Option<DecodedAudio>> {
        let decoder = self.decoder.as_mut().ok_or_else(|| {
            FerricastError::Decode("FdkEldDecoder::decode before configure()".into())
        })?;
        if frame.data.is_empty() {
            return Ok(None);
        }
        let consumed = decoder.fill(&frame.data)?;
        if consumed < frame.data.len() {
            return Err(FerricastError::Decode(format!(
                "libfdk-aac took {consumed} of {} bytes of an AAC-ELD access unit",
                frame.data.len()
            )));
        }
        match decoder.decode_frame(&mut self.pcm) {
            Ok(()) => {}
            Err(sys::AAC_DECODER_ERROR_AAC_DEC_NOT_ENOUGH_BITS) => return Ok(None),
            Err(e) => return Err(fdk_error(e)),
        }
        let Some((rate, channels, samples, _)) = decoder.stream_info() else {
            return Ok(None);
        };
        let (Ok(sample_rate), Ok(channels), Ok(samples)) = (
            u32::try_from(rate),
            u16::try_from(channels),
            usize::try_from(samples),
        ) else {
            return Err(FerricastError::Decode(
                "libfdk-aac reported a negative stream shape".into(),
            ));
        };
        if sample_rate == 0 || channels == 0 {
            return Ok(None);
        }
        let len = (samples * usize::from(channels)).min(self.pcm.len());
        Ok(Some(DecodedAudio {
            pcm: self.pcm[..len].to_vec(),
            sample_rate,
            channels,
            timestamp_us: frame.timestamp_us,
        }))
    }

    fn try_flush(&mut self) -> Result<Vec<DecodedAudio>> {
        Ok(Vec::new())
    }
}

impl AudioDecoder for FdkEldDecoder {
    const CODEC: AudioCodec = AudioCodec::AacEld;

    fn configure(&mut self, config: &AudioDecoderConfig) -> CodecResult<()> {
        self.try_configure(config).map_err(CodecError::from_backend(
            BackendId::FdkAac,
            Operation::Configure,
        ))
    }

    fn decode(&mut self, frame: AudioFrame) -> CodecResult<Option<DecodedAudio>> {
        self.try_decode(frame).map_err(CodecError::from_backend(
            BackendId::FdkAac,
            Operation::Decode,
        ))
    }

    fn flush(&mut self) -> CodecResult<Vec<DecodedAudio>> {
        self.try_flush().map_err(CodecError::from_backend(
            BackendId::FdkAac,
            Operation::Flush,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(codec: AudioCodec, sample_rate: u32) -> AudioDecoderConfig {
        AudioDecoderConfig {
            codec,
            sample_rate,
            channels: 2,
            backend: ferricast_core::BackendSelection::default(),
        }
    }

    #[test]
    fn configures_only_for_airplay_eld() {
        let mut d = FdkEldDecoder::default();
        d.configure(&config(AudioCodec::AacEld, 44_100)).unwrap();
        assert!(d.configure(&config(AudioCodec::Aac, 44_100)).is_err());
        assert!(d.configure(&config(AudioCodec::AacEld, 48_000)).is_err());
    }

    #[test]
    fn empty_units_decode_to_nothing() {
        let mut d = FdkEldDecoder::default();
        d.configure(&config(AudioCodec::AacEld, 44_100)).unwrap();
        let frame = AudioFrame {
            codec: AudioCodec::AacEld,
            data: bytes::Bytes::new(),
            timestamp_us: 0,
            sample_rate: 44_100,
            channels: 2,
        };
        assert!(d.decode(frame).unwrap().is_none());
    }
}
