use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, PixelFormat, RawFrame,
    Result, VideoDecoder,
};
use vp8_decoder::DecodeError;

use crate::nv12::{self, Planar};

#[derive(Default)]
pub struct Vp8Decoder {
    decoder: Option<vp8_decoder::Decoder>,
}

impl VideoDecoder for Vp8Decoder {
    const CODEC: Codec = Codec::Vp8;

    fn configure(&mut self, config: &DecoderConfig) -> Result<()> {
        if config.codec != Codec::Vp8 {
            return Err(FerricastError::Decode(format!(
                "Vp8Decoder asked to decode {:?}; only VP8 supported",
                config.codec
            )));
        }
        self.decoder = Some(vp8_decoder::Decoder::new());
        Ok(())
    }

    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        let decoder = self.decoder.as_mut().ok_or_else(|| {
            FerricastError::Decode("Vp8Decoder::decode before configure()".into())
        })?;
        let picture = match decoder.decode(&frame.data) {
            Ok(Some(picture)) => picture,
            Ok(None) | Err(DecodeError::NeedKeyframe) => return Ok(None),
            Err(e) => return Err(FerricastError::Decode(format!("vp8 decode: {e}"))),
        };
        let (data, stride) = nv12::from_i420(&Planar {
            width: picture.width as usize,
            height: picture.height as usize,
            y: picture.y,
            y_stride: picture.y_stride,
            u: picture.u,
            v: picture.v,
            uv_stride: picture.uv_stride,
        })
        .ok_or_else(|| FerricastError::Decode("vp8 returned short planes".into()))?;
        Ok(Some(CapturedFrame::Cpu(RawFrame {
            width: picture.width,
            height: picture.height,
            stride: stride as u32,
            format: PixelFormat::Nv12,
            data,
            timestamp_us: frame.timestamp_us,
        })))
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        Ok(Vec::new())
    }
}
