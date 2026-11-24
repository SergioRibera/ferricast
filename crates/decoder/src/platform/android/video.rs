//! H.264 / H.265 / VP8 / VP9 / AV1 decode through `AMediaCodec` into
//! byte buffers.
//!
//! Packets are queued with their `timestamp_us` as presentation time;
//! `MediaCodec` reorders and hands each picture back with the time of
//! the packet it was decoded from. H.264 / H.265 parameter sets travel
//! in-band (no `csd-*`): the codec reads them from the stream. Pictures
//! leave as NV12 whatever layout the codec wrote (NV12, Qualcomm's
//! aligned NV12, or I420), cropped to the visible rectangle.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use bytes::Bytes;
use ferricast_core::{
    BackendId, CapturedFrame, ChromaLocation, Codec, CodecError, CodecResult, ColourDescription,
    ColourPrimaries, ColourRange, DecoderConfig, EncodedFrame, MatrixCoefficients, Operation,
    PixelFormat, TransferCharacteristics, VideoDecoder,
};
use ndk::media::media_codec::MediaFormat;
use tracing::{info, warn};

use super::codec::{
    Codec as MediaCodec, FLAG_END_OF_STREAM, Output, Pick, WAIT, color_format, copy_into, fail, key,
};
use crate::nv12;

/// How long `decode` waits for a picture in low-latency mode: a
/// hardware decoder finishes a 1080p picture well within it, so the
/// picture leaves with its packet instead of one packet later.
const LOW_LATENCY_WAIT: Duration = Duration::from_millis(8);

/// How long `flush` waits for the end of the stream at most.
const FLUSH_DEADLINE: Duration = Duration::from_secs(2);

/// The MIME type of `codec`.
pub(crate) fn mime(codec: Codec) -> &'static str {
    match codec {
        Codec::H264 => "video/avc",
        Codec::H265 => "video/hevc",
        Codec::Vp8 => "video/x-vnd.on2.vp8",
        Codec::Vp9 => "video/x-vnd.on2.vp9",
        Codec::Av1 => "video/av01",
    }
}

/// The Google software decoder of `codec`.
pub(crate) fn software_name(codec: Codec) -> &'static str {
    match codec {
        Codec::H264 => "c2.android.avc.decoder",
        Codec::H265 => "c2.android.hevc.decoder",
        Codec::Vp8 => "c2.android.vp8.decoder",
        Codec::Vp9 => "c2.android.vp9.decoder",
        Codec::Av1 => "c2.android.av1.decoder",
    }
}

/// Where the codec's output pictures sit in their buffers.
#[derive(Debug, Clone, Copy)]
struct Layout {
    planar: bool,
    stride: usize,
    slice_height: usize,
    /// Visible rectangle: left, top, width, height.
    crop: (usize, usize, usize, usize),
    colour: ColourDescription,
}

struct Session {
    codec: MediaCodec,
    kind: Codec,
    format: PixelFormat,
    low_latency: bool,
    container_colour: Option<ColourDescription>,
    layout: Option<Layout>,
    ready: VecDeque<CapturedFrame>,
    /// A picture has come out: from here a failure is the stream's.
    decoded: bool,
}

/// A `MediaCodec` video decoder.
pub(crate) struct MediaCodecVideoDecoder {
    pick: Pick,
    session: Option<Session>,
}

/// Opens a decoder for `config`; the candidate's open function.
pub(crate) fn open(pick: Pick, config: &DecoderConfig) -> CodecResult<MediaCodecVideoDecoder> {
    let mut decoder = MediaCodecVideoDecoder {
        pick,
        session: None,
    };
    VideoDecoder::configure(&mut decoder, config)?;
    Ok(decoder)
}

fn int(format: &MediaFormat, key: &str) -> Option<usize> {
    format
        .i32(key)
        .and_then(|value| usize::try_from(value).ok())
}

/// The colour `format` signals (`MediaFormat.COLOR_*`), over `fallback`.
fn signalled_colour(format: &MediaFormat, fallback: ColourDescription) -> ColourDescription {
    let mut colour = fallback;
    match format.i32(key::COLOR_STANDARD) {
        Some(1) => {
            colour.primaries = ColourPrimaries::Bt709;
            colour.matrix = MatrixCoefficients::Bt709;
        }
        Some(2 | 3) => {
            colour.primaries = ColourPrimaries::Bt470Bg;
            colour.matrix = MatrixCoefficients::Bt470Bg;
        }
        Some(4 | 5) => {
            colour.primaries = ColourPrimaries::Smpte170M;
            colour.matrix = MatrixCoefficients::Smpte170M;
        }
        Some(6) => {
            colour.primaries = ColourPrimaries::Bt2020;
            colour.matrix = MatrixCoefficients::Bt2020Ncl;
        }
        Some(7) => {
            colour.primaries = ColourPrimaries::Bt2020;
            colour.matrix = MatrixCoefficients::Bt2020Cl;
        }
        _ => {}
    }
    match format.i32(key::COLOR_RANGE) {
        Some(1) => colour.range = ColourRange::Full,
        Some(2) => colour.range = ColourRange::Limited,
        _ => {}
    }
    match format.i32(key::COLOR_TRANSFER) {
        Some(1) => colour.transfer = TransferCharacteristics::Linear,
        Some(3) => colour.transfer = TransferCharacteristics::Bt709,
        Some(6) => colour.transfer = TransferCharacteristics::Pq,
        Some(7) => colour.transfer = TransferCharacteristics::Hlg,
        _ => {}
    }
    colour
}

impl Session {
    fn open(pick: Pick, config: &DecoderConfig) -> CodecResult<Self> {
        let mime = mime(config.codec);
        let codec = MediaCodec::create(mime, pick)?;
        let dimension = |value: u32, fallback: i32| {
            i32::try_from(value)
                .ok()
                .filter(|&v| v > 0)
                .unwrap_or(fallback)
        };
        let base = || {
            let mut format = MediaFormat::new();
            format.set_str(key::MIME, mime);
            // A size hint only; the stream's own headers decide.
            format.set_i32(key::WIDTH, dimension(config.width, 1920));
            format.set_i32(key::HEIGHT, dimension(config.height, 1080));
            if config.low_latency {
                format.set_i32(key::LOW_LATENCY, 1);
                format.set_i32(key::PRIORITY, 0);
            }
            format
        };
        // Ask for NV12; a codec that will not take the request decodes
        // in its own layout.
        let mut nv12 = base();
        nv12.set_i32(key::COLOR_FORMAT, color_format::NV12);
        let started = codec
            .start_decoder(&nv12)
            .or_else(|_| codec.start_decoder(&base()));
        if let Err(error) = started {
            return Err(CodecError::UnsupportedConfig {
                backend: BackendId::MediaCodec,
                what: format!("{} decode on {} ({error})", config.codec, codec.label()),
            });
        }
        info!(codec = %codec.label(), "MediaCodec decoder opened");
        Ok(Self {
            codec,
            kind: config.codec,
            format: config.pixel_format,
            low_latency: config.low_latency,
            container_colour: config.colour,
            layout: None,
            ready: VecDeque::new(),
            decoded: false,
        })
    }

    /// Reads the output layout from the codec's current output format.
    fn read_layout(&mut self) -> CodecResult<Layout> {
        let layout = Self::layout_of(&self.codec, self.kind, self.container_colour)?;
        self.layout = Some(layout);
        Ok(layout)
    }

    /// The layout `codec`'s current output format describes.
    fn layout_of(
        codec: &MediaCodec,
        kind: Codec,
        container_colour: Option<ColourDescription>,
    ) -> CodecResult<Layout> {
        let format = codec.output_format();
        let width = int(&format, key::WIDTH).unwrap_or(0);
        let height = int(&format, key::HEIGHT).unwrap_or(0);
        let colour_format = format.i32(key::COLOR_FORMAT).unwrap_or(color_format::NV12);
        let planar = match colour_format {
            color_format::NV12 | color_format::QCOM_NV12_32M => false,
            color_format::I420 => true,
            color_format::FLEXIBLE => {
                warn!("MediaCodec decodes to flexible YUV; reading it as I420");
                true
            }
            other => {
                return Err(CodecError::UnsupportedStream {
                    backend: BackendId::MediaCodec,
                    reason: format!(
                        "{} writes pictures in colour format {other:#x}, which this backend cannot read",
                        codec.label()
                    ),
                });
            }
        };
        let stride = int(&format, key::STRIDE)
            .filter(|&s| s >= width)
            .unwrap_or(width);
        let slice_height = int(&format, key::SLICE_HEIGHT)
            .filter(|&h| h >= height)
            .unwrap_or(height);
        let crop = match (
            int(&format, key::CROP_LEFT),
            int(&format, key::CROP_TOP),
            int(&format, key::CROP_RIGHT),
            int(&format, key::CROP_BOTTOM),
        ) {
            (Some(left), Some(top), Some(right), Some(bottom))
                if right >= left && bottom >= top =>
            {
                (left, top, right - left + 1, bottom - top + 1)
            }
            _ => (0, 0, width, height),
        };
        let fallback = container_colour
            .unwrap_or_else(|| ColourDescription::unspecified(ColourRange::Limited));
        let mut colour = signalled_colour(&format, fallback);
        if colour.chroma_location.is_none() && matches!(kind, Codec::H264 | Codec::H265) {
            colour = colour.with_chroma_location(ChromaLocation::Left);
        }
        let layout = Layout {
            planar,
            stride,
            slice_height,
            crop,
            colour,
        };
        if layout.crop.2 == 0 || layout.crop.3 == 0 {
            return Err(fail(
                Operation::Decode,
                format!("output format has no picture size: {layout:?}"),
            ));
        }
        Ok(layout)
    }

    /// Files one output buffer as a picture.
    fn accept(&mut self, output: Output) -> CodecResult<()> {
        if output.data.is_empty() {
            return Ok(());
        }
        let layout = match self.layout {
            Some(layout) => layout,
            None => self.read_layout()?,
        };
        let (left, top, width, height) = layout.crop;
        let short = || {
            fail(
                Operation::Decode,
                format!(
                    "picture buffer of {} bytes does not hold {layout:?}",
                    output.data.len()
                ),
            )
        };
        let luma_len = layout.stride * layout.slice_height;
        let (nv12, stride) = if layout.planar {
            let chroma_stride = layout.stride / 2;
            let chroma_rows = layout.slice_height / 2;
            let u = output.data.get(luma_len..).ok_or_else(short)?;
            let v = u.get(chroma_stride * chroma_rows..).ok_or_else(short)?;
            let planes = nv12::Planar {
                width,
                height,
                y: output
                    .data
                    .get(top * layout.stride + left..)
                    .ok_or_else(short)?,
                y_stride: layout.stride,
                u: u.get(top / 2 * chroma_stride + left / 2..)
                    .ok_or_else(short)?,
                v: v.get(top / 2 * chroma_stride + left / 2..)
                    .ok_or_else(short)?,
                uv_stride: chroma_stride,
            };
            nv12::from_i420(&planes).ok_or_else(short)?
        } else {
            let mut out = vec![0; width * height + width.div_ceil(2) * 2 * height.div_ceil(2)];
            let chroma_width = width.div_ceil(2) * 2;
            for (row, line) in out.chunks_exact_mut(width).take(height).enumerate() {
                let start = (top + row) * layout.stride + left;
                line.copy_from_slice(output.data.get(start..start + width).ok_or_else(short)?);
            }
            let chroma = &mut out[width * height..];
            for (row, line) in chroma.chunks_exact_mut(chroma_width).enumerate() {
                let start = luma_len + (top / 2 + row) * layout.stride + left / 2 * 2;
                line.copy_from_slice(
                    output
                        .data
                        .get(start..start + chroma_width)
                        .ok_or_else(short)?,
                );
            }
            // Chroma rows are `chroma_width` long; luma rows `width`:
            // re-pitch luma to the chroma pitch when width is odd.
            if chroma_width == width {
                (Bytes::from(out), width)
            } else {
                let mut pitched = vec![0; chroma_width * (height + height.div_ceil(2))];
                for (dst, src) in pitched
                    .chunks_exact_mut(chroma_width)
                    .zip(out.chunks_exact(width))
                    .take(height)
                {
                    dst[..width].copy_from_slice(src);
                }
                pitched[chroma_width * height..].copy_from_slice(&out[width * height..]);
                (Bytes::from(pitched), chroma_width)
            }
        };
        let picture = nv12::deliver(
            nv12,
            stride,
            (width, height),
            output.pts_us,
            self.format,
            layout.colour,
        )
        .ok_or_else(|| fail(Operation::Decode, "NV12 conversion failed"))?;
        self.decoded = true;
        self.ready.push_back(CapturedFrame::Cpu(picture));
        Ok(())
    }

    fn drain(&mut self, timeout: Duration) -> CodecResult<bool> {
        let mut ended = false;
        loop {
            let before = self.ready.len();
            // A format change means a new layout for the next pictures.
            let Some(output) = self.next(timeout)? else {
                return Ok(ended);
            };
            ended |= output.is(FLAG_END_OF_STREAM);
            self.accept(output)?;
            if ended || (timeout > Duration::ZERO && self.ready.len() > before) {
                return Ok(ended);
            }
        }
    }

    fn next(&mut self, timeout: Duration) -> CodecResult<Option<Output>> {
        let output = self
            .codec
            .next_output(Operation::Decode, timeout, || self.layout = None)?;
        Ok(output)
    }

    fn decode(&mut self, frame: EncodedFrame) -> CodecResult<Option<CapturedFrame>> {
        // Output drained while waiting for an input buffer is filed after
        // the queue; each buffer keeps the layout current when it was
        // dequeued, so a format change part-way through the batch only
        // applies to the buffers after it.
        let mut early = Vec::new();
        let mut layout = self.layout;
        let (kind, container_colour) = (self.kind, self.container_colour);
        self.codec
            .queue(
                Operation::Decode,
                frame.timestamp_us,
                0,
                |buffer| copy_into(buffer, &frame.data, Operation::Decode),
                |codec| {
                    loop {
                        let mut changed = false;
                        let output =
                            codec.next_output(Operation::Decode, Duration::ZERO, || {
                                changed = true
                            })?;
                        if changed {
                            layout = Some(Self::layout_of(codec, kind, container_colour)?);
                        }
                        let Some(output) = output else {
                            return Ok(());
                        };
                        early.push((output, layout));
                    }
                },
            )
            .map_err(|error| self.stream_error(error))?;
        for (output, snapshot) in early {
            self.layout = snapshot;
            self.accept(output)?;
        }
        self.layout = layout;
        let wait = if self.low_latency && self.ready.is_empty() {
            LOW_LATENCY_WAIT
        } else {
            Duration::ZERO
        };
        self.drain(wait).map_err(|error| self.stream_error(error))?;
        self.drain(Duration::ZERO)
            .map_err(|error| self.stream_error(error))?;
        Ok(self.ready.pop_front())
    }

    /// A codec failure after pictures came out is the stream's fault
    /// (a feature this codec lacks): another backend may decode it.
    fn stream_error(&self, error: CodecError) -> CodecError {
        match error {
            CodecError::Backend { reason, .. } if self.decoded => CodecError::UnsupportedStream {
                backend: BackendId::MediaCodec,
                reason,
            },
            other => other,
        }
    }

    fn flush(&mut self) -> CodecResult<Vec<CapturedFrame>> {
        self.codec.queue(
            Operation::Flush,
            0,
            FLAG_END_OF_STREAM,
            |_| Ok(0),
            |_| Ok(()),
        )?;
        // Drain until the end-of-stream buffer; the deadline only guards
        // against a codec that never signals it.
        let deadline = Instant::now() + FLUSH_DEADLINE;
        while !self.drain(WAIT)? {
            if Instant::now() >= deadline {
                warn!("MediaCodec decoder did not signal the end of the stream within 2 s");
                break;
            }
        }
        // Back to the flushed state, ready for the next stream.
        self.codec
            .flush()
            .map_err(super::codec::media(Operation::Flush, "AMediaCodec_flush"))?;
        Ok(self.ready.drain(..).collect())
    }
}

impl VideoDecoder for MediaCodecVideoDecoder {
    // The facade reports the job's codec; a session decodes the one its
    // configuration names.
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, config: &DecoderConfig) -> CodecResult<()> {
        if let Some(session) = &mut self.session
            && session.kind == config.codec
        {
            session.format = config.pixel_format;
            session.low_latency = config.low_latency;
            session.container_colour = config.colour;
            session.layout = None;
            return Ok(());
        }
        self.session = None;
        self.session = Some(Session::open(self.pick, config)?);
        Ok(())
    }

    fn decode(&mut self, frame: EncodedFrame) -> CodecResult<Option<CapturedFrame>> {
        self.session
            .as_mut()
            .ok_or_else(|| fail(Operation::Decode, "the decoder is not configured"))?
            .decode(frame)
    }

    fn flush(&mut self) -> CodecResult<Vec<CapturedFrame>> {
        self.session
            .as_mut()
            .map_or_else(|| Ok(Vec::new()), Session::flush)
    }
}
