use std::collections::VecDeque;

use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, PixelFormat, RawFrame,
    Result, VideoDecoder,
};
use rusty_vp9::DecodedFrame;

use crate::nv12::{self, Planar};

#[derive(Default)]
pub struct Vp9Decoder {
    decoder: Option<rusty_vp9::Vp9Decoder>,
    ready: VecDeque<CapturedFrame>,
}

impl VideoDecoder for Vp9Decoder {
    const CODEC: Codec = Codec::Vp9;

    fn configure(&mut self, config: &DecoderConfig) -> Result<()> {
        if config.codec != Codec::Vp9 {
            return Err(FerricastError::Decode(format!(
                "Vp9Decoder asked to decode {:?}; only VP9 supported",
                config.codec
            )));
        }
        self.decoder = Some(rusty_vp9::Vp9Decoder::new());
        self.ready.clear();
        Ok(())
    }

    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        let decoder = self.decoder.as_mut().ok_or_else(|| {
            FerricastError::Decode("Vp9Decoder::decode before configure()".into())
        })?;
        decoder
            .push(&frame.data, Some(frame.timestamp_us as i64))
            .map_err(|e| FerricastError::Decode(format!("vp9 decode: {e:?}")))?;
        while let Ok(decoded) = decoder.next_frame() {
            self.ready.push_back(picture(&decoded)?);
        }
        Ok(self.ready.pop_front())
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        if let Some(decoder) = self.decoder.as_mut() {
            decoder.flush();
            while let Ok(decoded) = decoder.next_frame() {
                self.ready.push_back(picture(&decoded)?);
            }
        }
        Ok(self.ready.drain(..).collect())
    }
}

fn picture(decoded: &DecodedFrame) -> Result<CapturedFrame> {
    let short = || FerricastError::Decode("vp9 returned short planes".into());
    if (decoded.subsampling_x, decoded.subsampling_y) != (1, 1) {
        return Err(FerricastError::Decode(format!(
            "vp9 chroma layout {}:{} is not 4:2:0",
            decoded.subsampling_x, decoded.subsampling_y
        )));
    }
    let (width, height) = (decoded.width as usize, decoded.height as usize);
    let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
    let (data, stride) = if decoded.bit_depth == 8 {
        nv12::from_i420(&Planar {
            width,
            height,
            y: decoded.planes.first().ok_or_else(short)?,
            y_stride: decoded.strides[0],
            u: decoded.planes.get(1).ok_or_else(short)?,
            v: decoded.planes.get(2).ok_or_else(short)?,
            uv_stride: decoded.strides[1],
        })
    } else {
        let narrowed = |plane: usize, w: usize, h: usize| {
            nv12::narrow_le_bytes(
                decoded.planes.get(plane)?,
                *decoded.strides.get(plane)?,
                w,
                h,
                decoded.bit_depth,
            )
        };
        let (y, u, v) = (
            narrowed(0, width, height),
            narrowed(1, chroma_width, chroma_height),
            narrowed(2, chroma_width, chroma_height),
        );
        match (y, u, v) {
            (Some(y), Some(u), Some(v)) => nv12::from_i420(&Planar {
                width,
                height,
                y: &y,
                y_stride: width,
                u: &u,
                v: &v,
                uv_stride: chroma_width,
            }),
            _ => None,
        }
    }
    .ok_or_else(short)?;
    Ok(CapturedFrame::Cpu(RawFrame {
        width: decoded.width,
        height: decoded.height,
        stride: stride as u32,
        format: PixelFormat::Nv12,
        data,
        timestamp_us: decoded.pts.map_or(0, |pts| pts as u64),
    }))
}
