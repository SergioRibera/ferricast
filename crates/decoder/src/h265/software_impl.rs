use std::collections::VecDeque;

use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, PixelFormat, RawFrame,
    Result, VideoDecoder,
};
use rust_h265::{Decoder, Frame, PixelData, parse_annex_b};

use crate::nv12::{self, Planar};

const MAX_THREADS: usize = 4;

#[derive(Default)]
pub struct SoftwareH265Decoder {
    decoder: Option<Decoder>,
    ready: VecDeque<CapturedFrame>,
}

impl VideoDecoder for SoftwareH265Decoder {
    const CODEC: Codec = Codec::H265;

    fn configure(&mut self, config: &DecoderConfig) -> Result<()> {
        if config.codec != Codec::H265 {
            return Err(FerricastError::Decode(format!(
                "SoftwareH265Decoder asked to decode {:?}; only H.265 supported",
                config.codec
            )));
        }
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(MAX_THREADS));
        self.decoder = Some(Decoder::new().with_display_order().with_threads(threads));
        self.ready.clear();
        Ok(())
    }

    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        let decoder = self.decoder.as_mut().ok_or_else(|| {
            FerricastError::Decode("SoftwareH265Decoder::decode before configure()".into())
        })?;
        for nal in parse_annex_b(&frame.data) {
            decoder
                .decode_nal(&nal)
                .map_err(|e| FerricastError::Decode(format!("h265 decode: {e}")))?;
        }
        while let Some(decoded) = decoder.next_output() {
            self.ready.push_back(picture(&decoded, frame.timestamp_us)?);
        }
        Ok(self.ready.pop_front())
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        if let Some(decoder) = self.decoder.as_mut() {
            while let Some(decoded) = decoder.flush() {
                self.ready.push_back(picture(&decoded, 0)?);
            }
        }
        Ok(self.ready.drain(..).collect())
    }
}

fn picture(decoded: &Frame, timestamp_us: u64) -> Result<CapturedFrame> {
    let short = || FerricastError::Decode("h265 returned short planes".into());
    if decoded.chroma_format_idc != 1 {
        return Err(FerricastError::Decode(format!(
            "h265 chroma_format_idc {} is not 4:2:0",
            decoded.chroma_format_idc
        )));
    }
    let (width, height) = (decoded.width as usize, decoded.height as usize);
    let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
    let depth = u32::from(decoded.bit_depth);
    let plane = |plane: &PixelData, w: usize, h: usize| match plane {
        PixelData::U8(samples) => Some(samples.clone()),
        PixelData::U16(samples) => nv12::narrow_u16(samples, w, w, h, depth),
    };
    let (y, u, v) = (
        plane(&decoded.y, width, height).ok_or_else(short)?,
        plane(&decoded.u, chroma_width, chroma_height).ok_or_else(short)?,
        plane(&decoded.v, chroma_width, chroma_height).ok_or_else(short)?,
    );
    let (data, stride) = nv12::from_i420(&Planar {
        width,
        height,
        y: &y,
        y_stride: width,
        u: &u,
        v: &v,
        uv_stride: chroma_width,
    })
    .ok_or_else(short)?;
    Ok(CapturedFrame::Cpu(RawFrame {
        width: decoded.width,
        height: decoded.height,
        stride: stride as u32,
        format: PixelFormat::Nv12,
        data,
        timestamp_us,
    }))
}
