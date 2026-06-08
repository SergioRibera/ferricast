//! AAC-LC decoder using `symphonia`'s pure-Rust implementation.
//!
//! No hardware AAC decode exists on consumer drivers (VA-API and
//! NVDEC are video-only; Wayland audio paths don't expose codec
//! acceleration to userspace). CPU decode is the universal answer
//! and symphonia's perf is fine — a single AAC stream is well
//! under 1% of a modern core.
//!
//! Input format: ADTS-framed AAC packets as they come out of an
//! MPEG-TS demuxer or HLS segmenter. Each [`AudioFrame`] carries
//! exactly one ADTS frame in [`AudioFrame::data`]; the decoder
//! parses the ADTS header to discover sample rate / channel
//! configuration if it wasn't supplied at `configure` time.
//!
//! Output format: interleaved signed-16-bit little-endian PCM, the
//! format every Linux audio sink (PipeWire, ALSA, PulseAudio)
//! accepts without further conversion.

use bytes::Bytes;
use ferricast_core::{
    AudioCodec, AudioDecoder, AudioDecoderConfig, AudioFrame, DecodedAudio, FerricastError, Result,
};
use symphonia::core::{
    audio::{AudioBufferRef, Signal},
    codecs::{CODEC_TYPE_AAC, CodecParameters, Decoder as SymphoniaDecoder, DecoderOptions},
    formats::Packet,
};
use symphonia::default::codecs::AacDecoder as SymphoniaAacDecoder;

#[derive(Default)]
pub struct AacDecoder {
    inner: Option<SymphoniaAacDecoder>,
    sample_rate: u32,
    channels: u16,
}

impl AudioDecoder for AacDecoder {
    const CODEC: AudioCodec = AudioCodec::Aac;

    fn configure(&mut self, config: &AudioDecoderConfig) -> Result<()> {
        if config.codec != AudioCodec::Aac {
            return Err(FerricastError::Decode(format!(
                "AacDecoder asked to decode {:?}; only AAC supported",
                config.codec
            )));
        }
        // Symphonia's AAC decoder discovers sample rate / channels
        // from the ADTS header in the first packet — the
        // CodecParameters we supply here are advisory. We still
        // pass them through so the decoder reports the expected
        // shape in its returned `Spec` immediately, which keeps the
        // first push to the sink from re-allocating its ring buffer
        // when the bitstream confirms what we already knew.
        let mut params = CodecParameters::new();
        params.for_codec(CODEC_TYPE_AAC);
        params.with_sample_rate(config.sample_rate);

        let inner = SymphoniaAacDecoder::try_new(&params, &DecoderOptions::default())
            .map_err(|e| FerricastError::Decode(format!("symphonia AAC init: {e}")))?;
        self.inner = Some(inner);
        self.sample_rate = config.sample_rate;
        self.channels = config.channels;
        Ok(())
    }

    fn decode(&mut self, frame: AudioFrame) -> Result<Option<DecodedAudio>> {
        let decoder = self
            .inner
            .as_mut()
            .ok_or_else(|| FerricastError::Decode("AacDecoder::decode before configure()".into()))?;

        let ts = frame.timestamp_us;
        let packet = Packet::new_from_slice(0, ts, 0, &frame.data);
        let buf = decoder
            .decode(&packet)
            .map_err(|e| FerricastError::Decode(format!("symphonia AAC decode: {e}")))?;

        // The decoder's first successful decode is when the bitstream
        // confirms its real sample rate / channel layout; refresh
        // our cached values so they're correct on the returned
        // DecodedAudio (the configure hint can be wrong if the
        // upstream sniffed the stream differently than we did).
        let spec = *buf.spec();
        self.sample_rate = spec.rate;
        self.channels = spec.channels.count() as u16;

        let pcm = interleave_s16(buf);

        Ok(Some(DecodedAudio {
            pcm: Bytes::from(pcm),
            sample_rate: self.sample_rate,
            channels: self.channels,
            timestamp_us: ts,
        }))
    }

    fn flush(&mut self) -> Result<Vec<DecodedAudio>> {
        // Symphonia decoders don't buffer across packets — every
        // input frame produces exactly one decoded buffer (or an
        // error). Nothing to drain at end-of-stream.
        Ok(Vec::new())
    }
}

/// Convert a symphonia `AudioBufferRef` (whatever sample format the
/// decoder picked internally) into interleaved s16le bytes. AAC's
/// natural output is f32; the conversion clips at ±1.0 and rounds
/// to the nearest integer.
fn interleave_s16(buf: AudioBufferRef<'_>) -> Vec<u8> {
    let frames = buf.frames();
    let channels = buf.spec().channels.count();
    let mut out = Vec::with_capacity(frames * channels * 2);

    // The macro-like helper that handles every sample variant
    // symphonia might hand us. F32 is the common path for AAC; the
    // others are here so adding other codecs (Opus on f32, FLAC on
    // s32) doesn't reopen this file.
    macro_rules! pack {
        ($buf:expr, $convert:expr) => {{
            for frame_idx in 0..frames {
                for ch_idx in 0..channels {
                    let s: f32 = $convert($buf.chan(ch_idx)[frame_idx]);
                    let i16_sample = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
                    out.extend_from_slice(&i16_sample.to_le_bytes());
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
