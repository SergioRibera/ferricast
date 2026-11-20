use std::collections::VecDeque;

use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, Result, VideoDecoder,
};
use rusty_vp9::{DecodedFrame, Error as Vp9Error};

use crate::nv12::{self, Colour, OutputFormat, Planar};

/// A VP9 superframe index holds at most eight frames (Annex B.3).
const SUPERFRAME_MAX_FRAMES: usize = 8;

#[derive(Default)]
pub struct Vp9Decoder {
    decoder: Option<rusty_vp9::Vp9Decoder>,
    ready: VecDeque<CapturedFrame>,
    format: OutputFormat,
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
        self.format = OutputFormat(config.pixel_format);
        Ok(())
    }

    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        let decoder = self.decoder.as_mut().ok_or_else(|| {
            FerricastError::Decode("Vp9Decoder::decode before configure()".into())
        })?;
        decoder
            .push(&frame.data, Some(frame.timestamp_us as i64))
            .map_err(|e| FerricastError::Decode(format!("vp9 decode: {e:?}")))?;
        // `Again` means "hidden frame decoded" as well as "queue empty",
        // and the two cannot be told apart, so every frame one superframe
        // can carry gets a turn; an empty queue answers instantly.
        for _ in 0..=SUPERFRAME_MAX_FRAMES {
            match decoder.next_frame() {
                Ok(decoded) => self.ready.push_back(picture(&decoded, self.format)?),
                Err(Vp9Error::Again) => {}
                Err(Vp9Error::Eof) => break,
                Err(e) => return Err(FerricastError::Decode(format!("vp9 decode: {e:?}"))),
            }
        }
        Ok(self.ready.pop_front())
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        if let Some(decoder) = self.decoder.as_mut() {
            decoder.flush();
            // Every call consumes one queued frame, so this ends at `Eof`
            // even when some frames fail.
            loop {
                match decoder.next_frame() {
                    Ok(decoded) => self.ready.push_back(picture(&decoded, self.format)?),
                    Err(Vp9Error::Again) => {}
                    Err(Vp9Error::Eof) => break,
                    Err(e) => tracing::warn!(error = ?e, "vp9 frame failed while flushing"),
                }
            }
        }
        Ok(self.ready.drain(..).collect())
    }
}

fn picture(decoded: &DecodedFrame, format: OutputFormat) -> Result<CapturedFrame> {
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
    // rusty_vp9 does not expose the header's color_space.
    nv12::deliver(
        data,
        stride,
        (width, height),
        decoded.pts.map_or(0, |pts| pts as u64),
        format.0,
        Colour::guess(height),
    )
    .map(CapturedFrame::Cpu)
    .ok_or_else(short)
}
