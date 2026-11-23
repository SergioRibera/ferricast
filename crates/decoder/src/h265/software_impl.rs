use std::collections::VecDeque;

use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, Result, VideoDecoder,
    YuvColour,
};
use rust_h265::{Decoder, Frame, PixelData, parse_annex_b};

use super::timestamps::Timestamps;
use crate::nv12::{self, OutputFormat, Planar};

const MAX_THREADS: usize = 4;

#[derive(Default)]
pub struct SoftwareH265Decoder {
    decoder: Option<Decoder>,
    ready: VecDeque<CapturedFrame>,
    format: OutputFormat,
    timestamps: Timestamps,
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
        self.timestamps = Timestamps::default();
        self.format = OutputFormat(config.pixel_format);
        Ok(())
    }

    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        let decoder = self.decoder.as_mut().ok_or_else(|| {
            FerricastError::Decode("SoftwareH265Decoder::decode before configure()".into())
        })?;
        let nals = parse_annex_b(&frame.data);
        self.timestamps.observe(&nals, frame.timestamp_us);
        for nal in &nals {
            decoder
                .decode_nal(nal)
                .map_err(|e| FerricastError::Decode(format!("h265 decode: {e}")))?;
        }
        while let Some(decoded) = decoder.next_output() {
            // A picture nothing was noted for (an unreadable packet) keeps
            // the time of the packet that released it.
            let timestamp_us = self
                .timestamps
                .claim(decoded.pic_order_cnt)
                .unwrap_or(frame.timestamp_us);
            self.ready
                .push_back(picture(&decoded, timestamp_us, self.format)?);
        }
        Ok(self.ready.pop_front())
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        if let Some(decoder) = self.decoder.as_mut() {
            let mut last = 0;
            while let Some(decoded) = decoder.flush() {
                last = self.timestamps.claim(decoded.pic_order_cnt).unwrap_or(last);
                self.ready.push_back(picture(&decoded, last, self.format)?);
            }
            self.timestamps = Timestamps::default();
        }
        Ok(self.ready.drain(..).collect())
    }
}

fn picture(decoded: &Frame, timestamp_us: u64, format: OutputFormat) -> Result<CapturedFrame> {
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
    // rust_h265 does not expose the VUI colour description.
    nv12::deliver(
        data,
        stride,
        (width, height),
        timestamp_us,
        format.0,
        YuvColour::guess(height),
    )
    .map(CapturedFrame::Cpu)
    .ok_or_else(short)
}
