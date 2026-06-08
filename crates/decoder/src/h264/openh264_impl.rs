//! Software H.264 decode via the openh264 crate.
//!
//! Always available when the `openh264-decode` feature is on. Used
//! as the no-GPU fallback in the auto-select chain. CPU-only — every
//! frame ends up in a `CapturedFrame::Cpu` after a YUV→BGRA convert
//! that runs on a worker pool internal to openh264.

use bytes::Bytes;
use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, PixelFormat, RawFrame,
    Result, VideoDecoder,
};
use openh264::{
    OpenH264API,
    decoder::{Decoder, DecoderConfig as Oh264DecoderConfig},
    formats::YUVSource,
};

#[derive(Default)]
pub struct OpenH264Decoder {
    decoder: Option<Decoder>,
}

impl VideoDecoder for OpenH264Decoder {
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, config: &DecoderConfig) -> Result<()> {
        if config.codec != Codec::H264 {
            return Err(FerricastError::Decode(format!(
                "OpenH264Decoder asked to decode {:?}; only H.264 supported",
                config.codec
            )));
        }
        let api = OpenH264API::from_source();
        let cfg = Oh264DecoderConfig::new();
        let decoder = Decoder::with_api_config(api, cfg)
            .map_err(|e| FerricastError::Decode(format!("openh264 decoder init: {e:?}")))?;
        self.decoder = Some(decoder);
        Ok(())
    }

    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        let decoder = self.decoder.as_mut().ok_or_else(|| {
            FerricastError::Decode("OpenH264Decoder::decode before configure()".into())
        })?;

        // `decode()` accepts a contiguous Annex-B buffer. Multi-NALU
        // packets (typical for AVC1 over MPEG-TS once converted) are
        // fine — the decoder consumes one or more NALUs and only
        // returns a frame once the slice that completes a picture
        // arrives. Returns Ok(None) for partial input.
        let decoded = decoder
            .decode(&frame.data)
            .map_err(|e| FerricastError::Decode(format!("openh264 decode: {e:?}")))?;
        let Some(yuv) = decoded else {
            return Ok(None);
        };

        let (w, h) = yuv.dimensions();
        // Allocate one BGRA buffer and let openh264 fill it. The
        // crate's `write_rgba8` produces big-endian RGBA; we want
        // BGRA little-endian for the rest of the pipeline, so we
        // copy through a temporary RGBA buffer and swap channels.
        // The extra pass is ~one memcpy at decode resolution; the
        // measurable hit is rounding error vs. the decode itself.
        let mut rgba = vec![0u8; w * h * 4];
        yuv.write_rgba8(&mut rgba);
        // RGBA → BGRA in place: swap byte 0 (R) and byte 2 (B) of
        // every pixel. The alpha byte stays untouched.
        for px in rgba.chunks_exact_mut(4) {
            px.swap(0, 2);
        }

        Ok(Some(CapturedFrame::Cpu(RawFrame {
            width: w as u32,
            height: h as u32,
            stride: (w * 4) as u32,
            format: PixelFormat::Bgra,
            data: Bytes::from(rgba),
            timestamp_us: frame.timestamp_us,
        })))
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        // openh264 doesn't expose a drain API. The Annex-B decoder
        // emits a picture as soon as the slice that completes it is
        // fed in, so steady-state flush returns nothing — anything
        // left in the decoder is by definition a non-decodable
        // partial frame.
        Ok(Vec::new())
    }
}
