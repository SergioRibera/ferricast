use std::collections::VecDeque;

use ferricast_core::{BackendId, CodecError, CodecResult, Operation};
use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, Result, VideoDecoder,
};
use rust_h265::{Decoder, Frame, PixelData, parse_annex_b};

use super::timestamps::Timestamps;
use crate::nv12::{self, OutputFormat, Planar, WidePicture, WidePlane, WideSamples};
use crate::signalling::NalSignalling;
use ferricast_codec::bitstream::NalSyntax;

const MAX_THREADS: usize = 4;

#[derive(Default)]
pub struct SoftwareH265Decoder {
    decoder: Option<Decoder>,
    ready: VecDeque<CapturedFrame>,
    format: OutputFormat,
    timestamps: Timestamps,
    /// `rust_h265` reports neither the VUI colour nor SEI messages.
    signalling: NalSignalling,
}

impl SoftwareH265Decoder {
    fn try_configure(&mut self, config: &DecoderConfig) -> Result<()> {
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
        self.signalling = NalSignalling::of(config);
        Ok(())
    }

    fn try_decode(&mut self, frame: &EncodedFrame) -> Result<Option<CapturedFrame>> {
        let decoder = self.decoder.as_mut().ok_or_else(|| {
            FerricastError::Decode("SoftwareH265Decoder::decode before configure()".into())
        })?;
        let nals = parse_annex_b(&frame.data);
        self.timestamps.observe(&nals, frame.timestamp_us);
        self.signalling.observe(&frame.data, NalSyntax::H265);
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
            self.ready.push_back(picture(
                &decoded,
                timestamp_us,
                self.format,
                &self.signalling,
            )?);
        }
        Ok(self.ready.pop_front())
    }

    fn try_flush(&mut self) -> Result<Vec<CapturedFrame>> {
        if let Some(decoder) = self.decoder.as_mut() {
            let mut last = 0;
            while let Some(decoded) = decoder.flush() {
                last = self.timestamps.claim(decoded.pic_order_cnt).unwrap_or(last);
                self.ready
                    .push_back(picture(&decoded, last, self.format, &self.signalling)?);
            }
            self.timestamps = Timestamps::default();
        }
        Ok(self.ready.drain(..).collect())
    }
}

impl VideoDecoder for SoftwareH265Decoder {
    const CODEC: Codec = Codec::H265;

    fn configure(&mut self, config: &DecoderConfig) -> CodecResult<()> {
        self.try_configure(config).map_err(CodecError::from_backend(
            BackendId::RustHevc,
            Operation::Configure,
        ))
    }

    fn decode(&mut self, frame: EncodedFrame) -> CodecResult<Option<CapturedFrame>> {
        self.try_decode(&frame).map_err(CodecError::from_backend(
            BackendId::RustHevc,
            Operation::Decode,
        ))
    }

    fn flush(&mut self) -> CodecResult<Vec<CapturedFrame>> {
        self.try_flush().map_err(CodecError::from_backend(
            BackendId::RustHevc,
            Operation::Flush,
        ))
    }
}

fn picture(
    decoded: &Frame,
    timestamp_us: u64,
    format: OutputFormat,
    signalling: &NalSignalling,
) -> Result<CapturedFrame> {
    let short = || FerricastError::Decode("h265 returned short planes".into());
    if decoded.chroma_format_idc != 1 {
        return Err(FerricastError::Decode(format!(
            "h265 chroma_format_idc {} is not 4:2:0",
            decoded.chroma_format_idc
        )));
    }
    let (width, height) = (decoded.width as usize, decoded.height as usize);
    let chroma_width = width.div_ceil(2);
    let colour = signalling.colour();
    let raw = match (&decoded.y, &decoded.u, &decoded.v) {
        (PixelData::U8(y), PixelData::U8(u), PixelData::U8(v)) => {
            let (data, stride) = nv12::from_i420(&Planar {
                width,
                height,
                y,
                y_stride: width,
                u,
                v,
                uv_stride: chroma_width,
            })
            .ok_or_else(short)?;
            nv12::deliver(
                data,
                stride,
                (width, height),
                timestamp_us,
                format.0,
                colour,
            )
        }
        (PixelData::U16(y), PixelData::U16(u), PixelData::U16(v)) => {
            fn plane(samples: &[u16], stride: usize) -> WidePlane<'_> {
                WidePlane {
                    samples: WideSamples::Words(samples),
                    stride,
                }
            }
            WidePicture {
                width,
                height,
                depth: u32::from(decoded.bit_depth),
                y: plane(y, width),
                u: plane(u, chroma_width),
                v: plane(v, chroma_width),
            }
            .deliver(timestamp_us, format.0, colour)
        }
        _ => {
            return Err(FerricastError::Decode(
                "h265 returned planes of mixed sample widths".into(),
            ));
        }
    };
    let mut raw = raw.ok_or_else(short)?;
    raw.hdr = signalling.hdr();
    Ok(CapturedFrame::Cpu(raw))
}
