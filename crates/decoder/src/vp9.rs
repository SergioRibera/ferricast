use std::collections::VecDeque;

use ferricast_core::{BackendId, CodecError, CodecResult, Operation};
use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, Result, VideoDecoder,
};
use rusty_vp9::{DecodedFrame, Error as Vp9Error};

use crate::nv12::{self, Container, OutputFormat, Planar, WidePicture, WidePlane, WideSamples};

/// A VP9 superframe index holds at most eight frames (Annex B.3).
const SUPERFRAME_MAX_FRAMES: usize = 8;

#[derive(Default)]
pub struct RustVp9Decoder {
    decoder: Option<rusty_vp9::Vp9Decoder>,
    ready: VecDeque<CapturedFrame>,
    format: OutputFormat,
    container: Container,
}

impl RustVp9Decoder {
    fn try_configure(&mut self, config: &DecoderConfig) -> Result<()> {
        if config.codec != Codec::Vp9 {
            return Err(FerricastError::Decode(format!(
                "RustVp9Decoder asked to decode {:?}; only VP9 supported",
                config.codec
            )));
        }
        self.decoder = Some(rusty_vp9::Vp9Decoder::new());
        self.ready.clear();
        self.format = OutputFormat(config.pixel_format);
        self.container = Container::of(config);
        Ok(())
    }

    fn try_decode(&mut self, frame: &EncodedFrame) -> Result<Option<CapturedFrame>> {
        let decoder = self.decoder.as_mut().ok_or_else(|| {
            FerricastError::Decode("RustVp9Decoder::decode before configure()".into())
        })?;
        decoder
            .push(&frame.data, Some(frame.timestamp_us.cast_signed()))
            .map_err(|e| FerricastError::Decode(format!("vp9 decode: {e:?}")))?;
        // `Again` means "hidden frame decoded" as well as "queue empty",
        // and the two cannot be told apart, so every frame one superframe
        // can carry gets a turn; an empty queue answers instantly.
        for _ in 0..=SUPERFRAME_MAX_FRAMES {
            match decoder.next_frame() {
                Ok(output) => {
                    self.ready
                        .push_back(picture(&output, self.format, self.container)?);
                }
                Err(Vp9Error::Again) => {}
                Err(Vp9Error::Eof) => break,
                Err(e) => return Err(FerricastError::Decode(format!("vp9 decode: {e:?}"))),
            }
        }
        Ok(self.ready.pop_front())
    }

    fn try_flush(&mut self) -> Result<Vec<CapturedFrame>> {
        if let Some(decoder) = self.decoder.as_mut() {
            decoder.flush();
            // Every call consumes one queued frame, so this ends at `Eof`
            // even when some frames fail.
            loop {
                match decoder.next_frame() {
                    Ok(decoded) => {
                        self.ready
                            .push_back(picture(&decoded, self.format, self.container)?);
                    }
                    Err(Vp9Error::Again) => {}
                    Err(Vp9Error::Eof) => break,
                    Err(e) => tracing::warn!(error = ?e, "vp9 frame failed while flushing"),
                }
            }
        }
        Ok(self.ready.drain(..).collect())
    }
}

impl VideoDecoder for RustVp9Decoder {
    const CODEC: Codec = Codec::Vp9;

    fn configure(&mut self, config: &DecoderConfig) -> CodecResult<()> {
        self.try_configure(config).map_err(CodecError::from_backend(
            BackendId::RustVp9,
            Operation::Configure,
        ))
    }

    fn decode(&mut self, frame: EncodedFrame) -> CodecResult<Option<CapturedFrame>> {
        self.try_decode(&frame).map_err(CodecError::from_backend(
            BackendId::RustVp9,
            Operation::Decode,
        ))
    }

    fn flush(&mut self) -> CodecResult<Vec<CapturedFrame>> {
        self.try_flush().map_err(CodecError::from_backend(
            BackendId::RustVp9,
            Operation::Flush,
        ))
    }
}

fn picture(
    decoded: &DecodedFrame,
    format: OutputFormat,
    container: Container,
) -> Result<CapturedFrame> {
    let short = || FerricastError::Decode("vp9 returned short planes".into());
    if (decoded.subsampling_x, decoded.subsampling_y) != (1, 1) {
        return Err(FerricastError::Decode(format!(
            "vp9 chroma layout {}:{} is not 4:2:0",
            decoded.subsampling_x, decoded.subsampling_y
        )));
    }
    let (width, height) = (decoded.width as usize, decoded.height as usize);
    let timestamp_us = decoded.pts.map_or(0, |pts| u64::try_from(pts).unwrap_or(0));
    // rusty_vp9 does not expose the header's color_space: the container's
    // colour (vpcC / Matroska Colour) is all there is.
    let colour = container.colour();
    let plane = |index: usize| decoded.planes.get(index).map(Vec::as_slice);
    let stride = |index: usize| decoded.strides.get(index).copied();
    let raw = if decoded.bit_depth == 8 {
        let (data, nv12_stride) = nv12::from_i420(&Planar {
            width,
            height,
            y: plane(0).ok_or_else(short)?,
            y_stride: stride(0).ok_or_else(short)?,
            u: plane(1).ok_or_else(short)?,
            v: plane(2).ok_or_else(short)?,
            uv_stride: stride(1).ok_or_else(short)?,
        })
        .ok_or_else(short)?;
        nv12::deliver(
            data,
            nv12_stride,
            (width, height),
            timestamp_us,
            format.0,
            colour,
        )
    } else {
        let wide = |index: usize| {
            Some(WidePlane {
                samples: WideSamples::LeBytes(plane(index)?),
                stride: stride(index)?,
            })
        };
        WidePicture {
            width,
            height,
            depth: decoded.bit_depth,
            y: wide(0).ok_or_else(short)?,
            u: wide(1).ok_or_else(short)?,
            v: wide(2).ok_or_else(short)?,
        }
        .deliver(timestamp_us, format.0, colour)
    };
    let mut raw = raw.ok_or_else(short)?;
    raw.hdr = container.hdr;
    Ok(CapturedFrame::Cpu(raw))
}
