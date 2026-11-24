//! Apple Lossless (ALAC) decoder over `symphonia`, for AirPlay audio
//! streams set up with `ct = 2` (music playback from iOS / macOS).
//!
//! AirPlay never sends the ALAC "magic cookie" in-band: senders use the
//! fixed configuration of the RAOP `fmtp` line (352 samples per frame,
//! 16-bit, 44.1 kHz stereo, `pb = 40`, `mb = 10`, `kb = 14`). The cookie
//! is built from the configured sample rate and channel count with those
//! parameters.

use ferricast_core::{
    AudioCodec, AudioDecoder, AudioDecoderConfig, AudioFrame, DecodedAudio, FerricastError, Result,
};
use ferricast_core::{BackendId, CodecError, CodecResult, Operation};
use symphonia::core::codecs::{CODEC_TYPE_ALAC, CodecParameters, Decoder as _, DecoderOptions};
use symphonia::core::formats::Packet;
use symphonia::default::codecs::AlacDecoder as SymphoniaAlac;

use crate::aac::interleave_s16;

/// Samples per channel in one AirPlay ALAC frame.
pub const AIRPLAY_ALAC_FRAME_SAMPLES: u32 = 352;

/// The 24-byte `ALACSpecificConfig` AirPlay senders assume.
fn magic_cookie(sample_rate: u32, channels: u8) -> [u8; 24] {
    let mut cookie = [0_u8; 24];
    cookie[..4].copy_from_slice(&AIRPLAY_ALAC_FRAME_SAMPLES.to_be_bytes());
    // compatible version 0, bit depth, pb, mb, kb, channels
    cookie[4..10].copy_from_slice(&[0, 16, 40, 10, 14, channels]);
    cookie[10..12].copy_from_slice(&255_u16.to_be_bytes());
    // max frame bytes and average bit rate unknown (0).
    cookie[20..24].copy_from_slice(&sample_rate.to_be_bytes());
    cookie
}

/// Decodes AirPlay ALAC frames into interleaved 16-bit PCM.
#[derive(Default)]
pub struct SymphoniaAlacDecoder {
    decoder: Option<SymphoniaAlac>,
}

impl SymphoniaAlacDecoder {
    fn try_configure(&mut self, config: &AudioDecoderConfig) -> Result<()> {
        if config.codec != AudioCodec::Alac {
            return Err(FerricastError::Decode(format!(
                "SymphoniaAlacDecoder asked to decode {:?}",
                config.codec
            )));
        }
        let channels = u8::try_from(config.channels)
            .ok()
            .filter(|c| (1..=8).contains(c))
            .ok_or_else(|| {
                FerricastError::Decode(format!("ALAC with {} channels", config.channels))
            })?;
        let mut params = CodecParameters::new();
        params
            .for_codec(CODEC_TYPE_ALAC)
            .with_sample_rate(config.sample_rate)
            .with_extra_data(Box::new(magic_cookie(config.sample_rate, channels)));
        self.decoder = Some(
            SymphoniaAlac::try_new(&params, &DecoderOptions::default())
                .map_err(|e| FerricastError::Decode(format!("symphonia ALAC init: {e}")))?,
        );
        Ok(())
    }

    fn try_decode(&mut self, frame: AudioFrame) -> Result<Option<DecodedAudio>> {
        let decoder = self.decoder.as_mut().ok_or_else(|| {
            FerricastError::Decode("SymphoniaAlacDecoder::decode before configure()".into())
        })?;
        let packet = Packet::new_from_slice(0, 0, 0, &frame.data);
        let buf = decoder
            .decode(&packet)
            .map_err(|e| FerricastError::Decode(format!("symphonia ALAC decode: {e}")))?;
        let spec = *buf.spec();
        let channels = u16::try_from(spec.channels.count())
            .map_err(|_| FerricastError::Decode("ALAC channel count out of range".into()))?;
        let pcm = interleave_s16(buf);
        if pcm.is_empty() {
            return Ok(None);
        }
        Ok(Some(DecodedAudio {
            pcm,
            sample_rate: spec.rate,
            channels,
            timestamp_us: frame.timestamp_us,
        }))
    }

    fn try_flush(&mut self) -> Result<Vec<DecodedAudio>> {
        Ok(Vec::new())
    }
}

impl AudioDecoder for SymphoniaAlacDecoder {
    const CODEC: AudioCodec = AudioCodec::Alac;

    fn configure(&mut self, config: &AudioDecoderConfig) -> CodecResult<()> {
        self.try_configure(config).map_err(CodecError::from_backend(
            BackendId::Symphonia,
            Operation::Configure,
        ))
    }

    fn decode(&mut self, frame: AudioFrame) -> CodecResult<Option<DecodedAudio>> {
        self.try_decode(frame).map_err(CodecError::from_backend(
            BackendId::Symphonia,
            Operation::Decode,
        ))
    }

    fn flush(&mut self) -> CodecResult<Vec<DecodedAudio>> {
        self.try_flush().map_err(CodecError::from_backend(
            BackendId::Symphonia,
            Operation::Flush,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MSB-first bit writer.
    #[derive(Default)]
    struct Bits {
        out: Vec<u8>,
        used: u32,
    }

    impl Bits {
        fn put(&mut self, value: u32, bits: u32) {
            for i in (0..bits).rev() {
                if self.used.is_multiple_of(8) {
                    self.out.push(0);
                }
                let bit = (value >> i) & 1;
                let last = self.out.last_mut().unwrap();
                *last |= (bit as u8) << (7 - self.used % 8);
                self.used += 1;
            }
        }
    }

    /// One stereo frame stored uncompressed (escape flag), as encoders
    /// emit for incompressible input.
    fn escaped_frame(samples: &[(i16, i16)]) -> Vec<u8> {
        let mut b = Bits::default();
        b.put(1, 3); // channel pair element
        b.put(0, 4); // element instance tag
        b.put(0, 12); // unused
        b.put(1, 1); // partial frame: sample count follows
        b.put(0, 2); // no shift
        b.put(1, 1); // uncompressed
        b.put(samples.len() as u32, 32);
        for (l, r) in samples {
            b.put(u32::from(*l as u16), 16);
            b.put(u32::from(*r as u16), 16);
        }
        b.put(7, 3); // end
        b.out
    }

    #[test]
    fn decodes_an_uncompressed_stereo_frame() {
        let mut d = SymphoniaAlacDecoder::default();
        d.configure(&AudioDecoderConfig {
            extra_data: None,
            codec: AudioCodec::Alac,
            sample_rate: 44_100,
            channels: 2,
            backend: ferricast_core::BackendSelection::default(),
        })
        .unwrap();
        let samples: Vec<(i16, i16)> = (0..100).map(|i| (i * 100, -i * 50)).collect();
        let out = d
            .decode(AudioFrame {
                codec: AudioCodec::Alac,
                data: escaped_frame(&samples).into(),
                timestamp_us: 42,
                sample_rate: 44_100,
                channels: 2,
            })
            .unwrap()
            .unwrap();
        assert_eq!(
            (out.sample_rate, out.channels, out.timestamp_us),
            (44_100, 2, 42)
        );
        assert_eq!(out.pcm.len(), 200);
        for (i, (l, r)) in samples.iter().enumerate() {
            assert_eq!(out.pcm[2 * i], *l);
            assert_eq!(out.pcm[2 * i + 1], *r);
        }
    }
}
