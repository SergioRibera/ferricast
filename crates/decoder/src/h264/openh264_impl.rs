use bytes::Bytes;
use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, PixelFormat, RawFrame,
    Result, VideoDecoder,
};
use openh264::{
    OpenH264API,
    decoder::{DecodedYUV, Decoder, DecoderConfig as Oh264DecoderConfig, Flush},
    formats::YUVSource,
};

use crate::nv12::{self, Planar};

#[derive(Default)]
pub struct OpenH264Decoder {
    decoder: Option<Decoder>,
    nv12: bool,
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
        let cfg = Oh264DecoderConfig::new().flush_after_decode(if config.low_latency {
            Flush::Flush
        } else {
            Flush::NoFlush
        });
        let decoder = Decoder::with_api_config(api, cfg)
            .map_err(|e| FerricastError::Decode(format!("openh264 decoder init: {e:?}")))?;
        self.decoder = Some(decoder);
        self.nv12 = config.pixel_format == PixelFormat::Nv12;
        Ok(())
    }

    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        let decoder = self.decoder.as_mut().ok_or_else(|| {
            FerricastError::Decode("OpenH264Decoder::decode before configure()".into())
        })?;
        let decoded = decoder
            .decode(&frame.data)
            .map_err(|e| FerricastError::Decode(format!("openh264 decode: {e:?}")))?;
        decoded
            .map(|yuv| picture(&yuv, self.nv12, frame.timestamp_us))
            .transpose()
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        let Some(decoder) = self.decoder.as_mut() else {
            return Ok(Vec::new());
        };
        decoder
            .flush_remaining()
            .map_err(|e| FerricastError::Decode(format!("openh264 flush: {e:?}")))?
            .iter()
            .map(|yuv| picture(yuv, self.nv12, 0))
            .collect()
    }
}

fn picture(yuv: &DecodedYUV, nv12: bool, timestamp_us: u64) -> Result<CapturedFrame> {
    let (w, h) = yuv.dimensions();
    if nv12 {
        let (y_stride, uv_stride, _) = yuv.strides();
        let (data, stride) = nv12::from_i420(&Planar {
            width: w,
            height: h,
            y: yuv.y(),
            y_stride,
            u: yuv.u(),
            v: yuv.v(),
            uv_stride,
        })
        .ok_or_else(|| FerricastError::Decode("openh264 returned short planes".into()))?;
        return Ok(CapturedFrame::Cpu(RawFrame {
            width: w as u32,
            height: h as u32,
            stride: stride as u32,
            format: PixelFormat::Nv12,
            data,
            timestamp_us,
        }));
    }
    let mut rgba = vec![0u8; w * h * 4];
    yuv.write_rgba8(&mut rgba);
    for px in rgba.as_chunks_mut::<4>().0 {
        px.swap(0, 2);
    }
    Ok(CapturedFrame::Cpu(RawFrame {
        width: w as u32,
        height: h as u32,
        stride: (w * 4) as u32,
        format: PixelFormat::Bgra,
        data: Bytes::from(rgba),
        timestamp_us,
    }))
}
