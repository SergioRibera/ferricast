use ferricast_core::{
    CapturedFrame, Codec, ColourDescription, DecoderConfig, EncodedFrame, FerricastError, Result,
    VideoDecoder,
};
use vp8_decoder::DecodeError;

use crate::nv12::{self, OutputFormat, Planar};

#[derive(Default)]
pub struct Vp8Decoder {
    decoder: Option<vp8_decoder::Decoder>,
    format: OutputFormat,
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
        self.format = OutputFormat(config.pixel_format);
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
        let short = || FerricastError::Decode("vp8 returned short planes".into());
        let (width, height) = (picture.width as usize, picture.height as usize);
        let (data, stride) = nv12::from_i420(&Planar {
            width,
            height,
            y: picture.y,
            y_stride: picture.y_stride,
            u: picture.u,
            v: picture.v,
            uv_stride: picture.uv_stride,
        })
        .ok_or_else(short)?;
        // VP8 is always BT.601 (RFC 6386 §9.2).
        // VP8 is always BT.601 (RFC 6386 §9.2).
        let colour = ColourDescription::BT601_LIMITED;
        nv12::deliver(
            data,
            stride,
            (width, height),
            frame.timestamp_us,
            self.format.0,
            colour,
        )
        .map(|raw| Some(CapturedFrame::Cpu(raw)))
        .ok_or_else(short)
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        Ok(Vec::new())
    }
}
