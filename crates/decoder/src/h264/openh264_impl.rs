use std::cmp::Reverse;
use std::collections::BinaryHeap;

use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, PixelFormat, Result,
    VideoDecoder,
};
use openh264::{
    OpenH264API,
    decoder::{DecodedYUV, Decoder, DecoderConfig as Oh264DecoderConfig, Flush},
    formats::YUVSource,
};

use crate::nv12::{self, Colour, OutputFormat, Planar};

#[derive(Default)]
pub struct OpenH264Decoder {
    decoder: Option<Decoder>,
    format: OutputFormat,
    /// Timestamps of packets whose pictures have not come out yet.
    /// openh264 releases pictures in display order, so each takes the
    /// smallest pending one; that also stamps pictures drained by
    /// `flush` with their real times instead of zero.
    pending: BinaryHeap<Reverse<u64>>,
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
        self.format = OutputFormat(config.pixel_format);
        self.pending.clear();
        Ok(())
    }

    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        let format = self.format.0;
        let decoder = self.decoder.as_mut().ok_or_else(|| {
            FerricastError::Decode("OpenH264Decoder::decode before configure()".into())
        })?;
        let decoded = decoder
            .decode(&frame.data)
            .map_err(|e| FerricastError::Decode(format!("openh264 decode: {e:?}")))?;
        // Parameter-set-only packets never yield a picture; queuing their
        // timestamp would shift every later picture by one.
        if carries_slice(&frame.data) {
            self.pending.push(Reverse(frame.timestamp_us));
        }
        let Some(yuv) = decoded else {
            return Ok(None);
        };
        let timestamp_us = take_smallest(&mut self.pending).unwrap_or(frame.timestamp_us);
        picture(&yuv, format, timestamp_us).map(Some)
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        let format = self.format.0;
        let Some(decoder) = self.decoder.as_mut() else {
            return Ok(Vec::new());
        };
        let remaining = decoder
            .flush_remaining()
            .map_err(|e| FerricastError::Decode(format!("openh264 flush: {e:?}")))?;
        let mut last = 0;
        let frames = remaining
            .iter()
            .map(|yuv| {
                last = take_smallest(&mut self.pending).unwrap_or(last);
                picture(yuv, format, last)
            })
            .collect();
        self.pending.clear();
        frames
    }
}

fn take_smallest(pending: &mut BinaryHeap<Reverse<u64>>) -> Option<u64> {
    pending.pop().map(|Reverse(timestamp_us)| timestamp_us)
}

/// Whether an Annex B packet holds a coded slice (NAL types 1–5).
fn carries_slice(data: &[u8]) -> bool {
    data.windows(4)
        .any(|window| matches!(window, [0, 0, 1, header] if (1..=5).contains(&(header & 0x1f))))
}

fn picture(yuv: &DecodedYUV, format: PixelFormat, timestamp_us: u64) -> Result<CapturedFrame> {
    let short = || FerricastError::Decode("openh264 returned short planes".into());
    let (width, height) = yuv.dimensions();
    let (y_stride, uv_stride, _) = yuv.strides();
    let (data, stride) = nv12::from_i420(&Planar {
        width,
        height,
        y: yuv.y(),
        y_stride,
        u: yuv.u(),
        v: yuv.v(),
        uv_stride,
    })
    .ok_or_else(short)?;
    // openh264 does not expose the VUI colour description.
    nv12::deliver(
        data,
        stride,
        (width, height),
        timestamp_us,
        format,
        Colour::guess(height),
    )
    .map(CapturedFrame::Cpu)
    .ok_or_else(short)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_packets_with_coded_slices_count_as_pictures() {
        assert!(carries_slice(&[0, 0, 0, 1, 0x67, 1, 0, 0, 1, 0x65, 2]));
        assert!(carries_slice(&[0, 0, 1, 0x41, 9]));
        assert!(!carries_slice(&[0, 0, 0, 1, 0x67, 1, 0, 0, 1, 0x68, 2]));
        assert!(!carries_slice(&[0x65, 0x41]));
    }

    #[test]
    fn pictures_take_the_smallest_pending_timestamp() {
        let mut pending: BinaryHeap<Reverse<u64>> = [0, 66, 33].into_iter().map(Reverse).collect();
        assert_eq!(take_smallest(&mut pending), Some(0));
        assert_eq!(take_smallest(&mut pending), Some(33));
        assert_eq!(take_smallest(&mut pending), Some(66));
        assert_eq!(take_smallest(&mut pending), None);
    }
}
