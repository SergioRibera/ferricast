//! AAC-LC decoder using `symphonia`'s pure-Rust implementation.
//!
//! No hardware AAC decode exists on consumer drivers (VA-API and
//! NVDEC are video-only; Wayland audio paths don't expose codec
//! acceleration to userspace). CPU decode is the universal answer
//! and symphonia's perf is fine — a single AAC stream is well
//! under 1% of a modern core.
//!
//! Input format: either ADTS-framed AAC as it comes out of an MPEG-TS
//! demuxer or HLS segmenter (one PES payload may hold several ADTS
//! frames back to back), or bare `raw_data_block()`s as carried by
//! RTP / MP4. Symphonia's decoder only accepts raw blocks, so ADTS
//! headers are parsed and stripped here, and a header that disagrees
//! with the configured sample rate / channels reconfigures the decoder.
//!
//! Output format: interleaved signed-16-bit PCM, the format every
//! Linux audio sink (PipeWire, ALSA, PulseAudio) accepts without
//! further conversion.

use crate::adts::{self, Shape};
#[cfg(feature = "fdk-aac-decode")]
mod eld;
#[cfg(feature = "fdk-aac-decode")]
mod fdk;

#[cfg(feature = "fdk-aac-decode")]
pub use eld::FdkEldDecoder;
#[cfg(feature = "fdk-aac-decode")]
pub use fdk::FdkAacDecoder;

use std::collections::VecDeque;

use ferricast_core::{
    AudioCodec, AudioDecoder, AudioDecoderConfig, AudioFrame, DecodedAudio, FerricastError, Result,
};
use ferricast_core::{BackendId, CodecError, CodecResult, Operation};
use symphonia::core::{
    audio::{AudioBufferRef, Channels, Signal},
    codecs::{CODEC_TYPE_AAC, CodecParameters, Decoder as SymphoniaDecoder, DecoderOptions},
    formats::Packet,
};
use symphonia::default::codecs::AacDecoder as SymphoniaCodec;

const MICROS_PER_SECOND: u64 = 1_000_000;

impl Shape {
    /// The configured shape, or `None` when the stream metadata left it
    /// unknown (zero rate or channels) and the first ADTS header decides.
    fn from_config(config: &AudioDecoderConfig) -> Result<Option<Self>> {
        if config.sample_rate == 0 || config.channels == 0 {
            return Ok(None);
        }
        let shape = Self {
            sample_rate: config.sample_rate,
            channels: config.channels,
        };
        shape.channel_mask()?;
        Ok(Some(shape))
    }

    fn channel_mask(self) -> Result<Channels> {
        match self.channels {
            1 => Ok(Channels::FRONT_LEFT),
            2 => Ok(Channels::FRONT_LEFT | Channels::FRONT_RIGHT),
            other => Err(FerricastError::Decode(format!(
                "AAC with {other} channels needs libfdk-aac (the `fdk-aac-decode` feature): \
                 symphonia decodes mono and stereo AAC-LC only"
            ))),
        }
    }

    /// Without an AudioSpecificConfig symphonia takes the shape from the
    /// codec parameters, and refuses to build when channels are missing.
    fn decoder(self) -> Result<SymphoniaCodec> {
        let mut params = CodecParameters::new();
        params
            .for_codec(CODEC_TYPE_AAC)
            .with_sample_rate(self.sample_rate)
            .with_channels(self.channel_mask()?);
        SymphoniaCodec::try_new(&params, &DecoderOptions::default())
            .map_err(|e| FerricastError::Decode(format!("symphonia AAC init: {e}")))
    }

    fn micros(self, samples: u64) -> u64 {
        samples * MICROS_PER_SECOND / u64::from(self.sample_rate)
    }
}

struct Active {
    decoder: SymphoniaCodec,
    shape: Shape,
}

#[derive(Default)]
pub struct SymphoniaAacDecoder {
    configured: bool,
    active: Option<Active>,
    /// Decoded runs not yet handed out: a payload whose frames change
    /// shape mid-way yields one run per shape, but `decode` returns one.
    pending: VecDeque<DecodedAudio>,
}

impl SymphoniaAacDecoder {
    fn decoder_for(&mut self, shape: Option<Shape>) -> Result<&mut Active> {
        if let Some(wanted) = shape
            && self
                .active
                .as_ref()
                .is_none_or(|active| active.shape != wanted)
        {
            if let Some(active) = &self.active {
                tracing::info!(
                    from = ?active.shape,
                    to = ?wanted,
                    "AAC stream shape differs from the configured one; reconfiguring"
                );
            }
            self.active = Some(Active {
                decoder: wanted.decoder()?,
                shape: wanted,
            });
        }
        self.active.as_mut().ok_or_else(|| {
            FerricastError::Decode(
                "AAC: raw (non-ADTS) input with no sample rate / channels configured".into(),
            )
        })
    }

    /// Decodes every access unit of one payload into `runs`, one run per
    /// stretch of identically shaped frames. Each frame after the first
    /// is stamped `n` samples after the payload's timestamp.
    fn decode_units(&mut self, frame: &AudioFrame, runs: &mut Vec<DecodedAudio>) -> Result<()> {
        let mut base_us = frame.timestamp_us;
        let mut samples = 0u64;
        let mut run_shape: Option<Shape> = None;
        for unit in adts::units(&frame.data) {
            let unit = unit.map_err(|e| FerricastError::Decode(format!("AAC: {e}")))?;
            let active = self.decoder_for(unit.shape)?;
            let shape = active.shape;
            if let Some(previous) = run_shape
                && previous != shape
            {
                base_us += previous.micros(samples);
                samples = 0;
            }
            let packet = Packet::new_from_slice(0, 0, 0, unit.payload);
            let buf = active
                .decoder
                .decode(&packet)
                .map_err(|e| FerricastError::Decode(format!("symphonia AAC decode: {e}")))?;
            let decoded = buf.frames() as u64;
            let pcm = interleave_s16(buf);
            match runs.last_mut() {
                Some(run) if run_shape == Some(shape) => run.pcm.extend(pcm),
                _ => runs.push(DecodedAudio {
                    pcm,
                    sample_rate: shape.sample_rate,
                    channels: shape.channels,
                    timestamp_us: base_us,
                }),
            }
            samples += decoded;
            run_shape = Some(shape);
        }
        Ok(())
    }
}

impl SymphoniaAacDecoder {
    fn try_configure(&mut self, config: &AudioDecoderConfig) -> Result<()> {
        if config.codec != AudioCodec::Aac {
            return Err(FerricastError::Decode(format!(
                "SymphoniaAacDecoder asked to decode {:?}; only AAC supported",
                config.codec
            )));
        }
        self.active = match Shape::from_config(config)? {
            Some(shape) => Some(Active {
                decoder: shape.decoder()?,
                shape,
            }),
            None => None,
        };
        self.pending.clear();
        self.configured = true;
        Ok(())
    }

    fn try_decode(&mut self, frame: AudioFrame) -> Result<Option<DecodedAudio>> {
        if !self.configured {
            return Err(FerricastError::Decode(
                "SymphoniaAacDecoder::decode before configure()".into(),
            ));
        }
        let mut runs = Vec::new();
        let decoded = self.decode_units(&frame, &mut runs);
        // Frames decoded before a bad one in the same payload are still
        // good audio; queue them before reporting the failure.
        self.pending.extend(runs);
        decoded?;
        Ok(self.pending.pop_front())
    }

    fn try_flush(&mut self) -> Result<Vec<DecodedAudio>> {
        // Symphonia decoders don't buffer across packets; only runs split
        // off by a mid-payload shape change can still be waiting.
        Ok(self.pending.drain(..).collect())
    }
}

impl AudioDecoder for SymphoniaAacDecoder {
    const CODEC: AudioCodec = AudioCodec::Aac;

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

/// Convert a symphonia `AudioBufferRef` (whatever sample format the
/// decoder picked internally) into interleaved i16 samples. AAC's
/// natural output is f32; the conversion clips at ±1.0 and rounds
/// to the nearest integer.
pub(crate) fn interleave_s16(buf: AudioBufferRef<'_>) -> Vec<i16> {
    let frames = buf.frames();
    let channels = buf.spec().channels.count();
    let mut out = Vec::with_capacity(frames * channels);

    // The macro-like helper that handles every sample variant
    // symphonia might hand us. F32 is the common path for AAC; the
    // others are here so adding other codecs (Opus on f32, FLAC on
    // s32) doesn't reopen this file.
    macro_rules! pack {
        ($buf:expr, $convert:expr) => {{
            for frame_idx in 0..frames {
                for ch_idx in 0..channels {
                    let s: f32 = $convert($buf.chan(ch_idx)[frame_idx]);
                    out.push((s.clamp(-1.0, 1.0) * 32767.0).round() as i16);
                }
            }
        }};
    }

    match buf {
        AudioBufferRef::F32(b) => pack!(b, |x: f32| x),
        AudioBufferRef::F64(b) => pack!(b, |x: f64| x as f32),
        AudioBufferRef::S32(b) => pack!(b, |x: i32| x as f32 / i32::MAX as f32),
        AudioBufferRef::S24(b) => pack!(b, |x: symphonia::core::sample::i24| {
            x.inner() as f32 / (1 << 23) as f32
        }),
        AudioBufferRef::S16(b) => pack!(b, |x: i16| x as f32 / i16::MAX as f32),
        AudioBufferRef::S8(b) => pack!(b, |x: i8| x as f32 / i8::MAX as f32),
        AudioBufferRef::U32(b) => {
            pack!(b, |x: u32| (x as f32 / u32::MAX as f32) * 2.0 - 1.0)
        }
        AudioBufferRef::U24(b) => pack!(b, |x: symphonia::core::sample::u24| {
            (x.inner() as f32 / (1 << 24) as f32) * 2.0 - 1.0
        }),
        AudioBufferRef::U16(b) => {
            pack!(b, |x: u16| (x as f32 / u16::MAX as f32) * 2.0 - 1.0)
        }
        AudioBufferRef::U8(b) => pack!(b, |x: u8| (x as f32 / u8::MAX as f32) * 2.0 - 1.0),
    }

    out
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;
    use crate::adts::tests::frame;

    /// A silent mono `raw_data_block()`: one SCE with max_sfb = 0, then END.
    const SILENT_MONO: [u8; 4] = [0x00, 0xC8, 0x00, 0x07];
    const RATE_48K: u8 = 3;
    const RATE_44K1: u8 = 4;

    fn config(sample_rate: u32, channels: u16) -> AudioDecoderConfig {
        AudioDecoderConfig {
            extra_data: None,
            codec: AudioCodec::Aac,
            sample_rate,
            channels,
            backend: ferricast_core::BackendSelection::default(),
        }
    }

    fn packet(data: Vec<u8>, timestamp_us: u64) -> AudioFrame {
        AudioFrame {
            codec: AudioCodec::Aac,
            data: Bytes::from(data),
            timestamp_us,
            sample_rate: 48_000,
            channels: 1,
        }
    }

    #[test]
    fn configure_succeeds_for_mono_and_stereo_and_with_unknown_shape() {
        for (rate, channels) in [(48_000, 1), (44_100, 2), (0, 0)] {
            SymphoniaAacDecoder::default()
                .configure(&config(rate, channels))
                .unwrap();
        }
        // symphonia decodes mono and stereo only (libfdk-aac goes up to 7.1).
        assert!(
            SymphoniaAacDecoder::default()
                .configure(&config(48_000, 6))
                .is_err()
        );
    }

    #[test]
    fn every_adts_frame_of_a_payload_is_decoded_into_one_run() {
        let mut decoder = SymphoniaAacDecoder::default();
        decoder.configure(&config(48_000, 1)).unwrap();
        let mut data = frame(RATE_48K, 1, false, &SILENT_MONO);
        data.extend(frame(RATE_48K, 1, true, &SILENT_MONO));
        let audio = decoder.decode(packet(data, 5_000)).unwrap().unwrap();
        assert_eq!(audio.pcm.len(), 2048);
        assert_eq!((audio.sample_rate, audio.channels), (48_000, 1));
        assert_eq!(audio.timestamp_us, 5_000);
        assert!(decoder.flush().unwrap().is_empty());
    }

    #[test]
    fn raw_blocks_decode_with_the_configured_shape() {
        let mut decoder = SymphoniaAacDecoder::default();
        decoder.configure(&config(48_000, 1)).unwrap();
        let audio = decoder
            .decode(packet(SILENT_MONO.to_vec(), 0))
            .unwrap()
            .unwrap();
        assert_eq!(audio.pcm.len(), 1024);
    }

    #[test]
    fn raw_blocks_without_a_known_shape_are_refused() {
        let mut decoder = SymphoniaAacDecoder::default();
        decoder.configure(&config(0, 0)).unwrap();
        assert!(decoder.decode(packet(SILENT_MONO.to_vec(), 0)).is_err());
    }

    #[test]
    fn an_adts_header_that_disagrees_with_the_config_reconfigures() {
        let mut decoder = SymphoniaAacDecoder::default();
        decoder.configure(&config(48_000, 2)).unwrap();
        let audio = decoder
            .decode(packet(frame(RATE_44K1, 1, false, &SILENT_MONO), 0))
            .unwrap()
            .unwrap();
        assert_eq!((audio.sample_rate, audio.channels), (44_100, 1));
    }

    #[test]
    fn a_shape_change_inside_a_payload_starts_a_new_run_at_the_right_time() {
        let mut decoder = SymphoniaAacDecoder::default();
        decoder.configure(&config(48_000, 1)).unwrap();
        let mut data = frame(RATE_48K, 1, false, &SILENT_MONO);
        data.extend(frame(RATE_44K1, 1, false, &SILENT_MONO));
        let first = decoder.decode(packet(data, 1_000_000)).unwrap().unwrap();
        assert_eq!((first.sample_rate, first.timestamp_us), (48_000, 1_000_000));
        let rest = decoder.flush().unwrap();
        assert_eq!(rest.len(), 1);
        assert_eq!(
            (rest[0].sample_rate, rest[0].timestamp_us),
            (44_100, 1_021_333)
        );
    }

    #[test]
    fn decode_before_configure_is_an_error() {
        assert!(
            SymphoniaAacDecoder::default()
                .decode(packet(SILENT_MONO.to_vec(), 0))
                .is_err()
        );
    }
}
