//! H.264 / H.265 / VP8 / VP9 / AV1 decode through `AMediaCodec`.
//!
//! Packets are queued with their `timestamp_us` as presentation time;
//! `MediaCodec` reorders and hands each picture back with the time of
//! the packet it was decoded from. H.264 / H.265 parameter sets travel
//! in-band (no `csd-*`): the codec reads them from the stream.
//!
//! Pictures leave one of two ways:
//! - [`FrameMemory::Cpu`]: byte buffers, repacked into NV12 (or P010
//!   for a 10-bit stream when the configuration asks for P010; 10-bit
//!   pictures narrow to NV12 otherwise) and cropped to the visible
//!   rectangle, whatever layout the codec wrote (NV12, Qualcomm's
//!   aligned NV12, I420, P010).
//! - [`FrameMemory::Native`]: the codec renders into an `AImageReader`
//!   and each picture is a [`HardwareBuffer`]
//!   ([`CapturedFrame::Native`]) with no copy; P010 images when the
//!   configuration asks for P010 and the device allocates them. A
//!   device that cannot render to a reader decodes to byte buffers.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use ferricast_codec::android::{HardwareBuffer, ImageOutput};
use ferricast_codec::yuv420::{self, Crop, SemiPlanar};
use ferricast_core::{
    BackendId, BitDepth, CapturedFrame, ChromaLocation, Codec, CodecError, CodecResult,
    ColourDescription, ColourPrimaries, ColourRange, DecoderConfig, EncodedFrame, FrameMemory,
    HdrMetadata, MatrixCoefficients, NativeFrame, Operation, PixelFormat, RawFrame,
    TransferCharacteristics, VideoDecoder,
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

/// Pictures the consumer may hold at once in native mode, on top of the
/// ones the codec renders ahead.
const MAX_IMAGES: u32 = 8;

/// How long a rendered picture may take to reach the image reader.
const RENDER_WAIT: Duration = Duration::from_millis(30);

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

/// Where byte-buffer pictures sit in their buffers.
#[derive(Debug, Clone, Copy)]
struct Pixels {
    planar: bool,
    depth: BitDepth,
    /// Row stride as the format reports it (see [`Session::accept_bytes`]
    /// for the unit of a P010 stride).
    stride: usize,
    slice_height: usize,
    crop: Crop,
}

/// What the current output format says about the pictures.
#[derive(Debug, Clone, Copy)]
struct Layout {
    /// `None` when the codec renders to a surface.
    pixels: Option<Pixels>,
    colour: ColourDescription,
    hdr: Option<HdrMetadata>,
}

/// How pictures leave the codec.
enum Delivery {
    Buffers,
    Surface {
        output: ImageOutput,
        /// Pictures rendered whose image has not been taken yet.
        pending: usize,
    },
}

struct Session {
    codec: MediaCodec,
    kind: Codec,
    /// The memory the configuration asked for (the session may have
    /// fallen back to byte buffers).
    memory: FrameMemory,
    format: PixelFormat,
    low_latency: bool,
    container_colour: Option<ColourDescription>,
    container_hdr: Option<HdrMetadata>,
    layout: Option<Layout>,
    delivery: Delivery,
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

/// The `MediaFormat` of a decoder for `config`, asking for `colour_format`
/// byte buffers when given.
fn decoder_format(config: &DecoderConfig, colour_format: Option<i32>) -> MediaFormat {
    let dimension = |value: u32, fallback: i32| {
        i32::try_from(value)
            .ok()
            .filter(|&v| v > 0)
            .unwrap_or(fallback)
    };
    let mut format = MediaFormat::new();
    format.set_str(key::MIME, mime(config.codec));
    // A size hint only; the stream's own headers decide.
    format.set_i32(key::WIDTH, dimension(config.width, 1920));
    format.set_i32(key::HEIGHT, dimension(config.height, 1080));
    if config.low_latency {
        format.set_i32(key::LOW_LATENCY, 1);
        format.set_i32(key::PRIORITY, 0);
    }
    if let Some(colour_format) = colour_format {
        format.set_i32(key::COLOR_FORMAT, colour_format);
    }
    format
}

/// An image reader for `config`'s pictures: P010 when asked for and the
/// device allocates it, NV12 otherwise.
fn image_output(config: &DecoderConfig) -> CodecResult<ImageOutput> {
    let (width, height) = (config.width.max(16), config.height.max(16));
    if config.pixel_format == PixelFormat::P010 {
        match ImageOutput::new(width, height, PixelFormat::P010, MAX_IMAGES) {
            Ok(output) => return Ok(output),
            Err(error) => warn!(%error, "no P010 image reader; rendering 8-bit pictures"),
        }
    }
    ImageOutput::new(width, height, PixelFormat::Nv12, MAX_IMAGES)
}

impl Session {
    fn open(pick: Pick, config: &DecoderConfig) -> CodecResult<Self> {
        let codec = MediaCodec::create(mime(config.codec), pick)?;
        let delivery = match config.memory {
            FrameMemory::Native => Self::start_surface(&codec, config),
            FrameMemory::Cpu => None,
        };
        let delivery = if let Some(delivery) = delivery {
            delivery
        } else {
            Self::start_buffers(&codec, config)?;
            Delivery::Buffers
        };
        let native = matches!(delivery, Delivery::Surface { .. });
        info!(codec = %codec.label(), native, "MediaCodec decoder opened");
        Ok(Self {
            codec,
            kind: config.codec,
            memory: config.memory,
            format: config.pixel_format,
            low_latency: config.low_latency,
            container_colour: config.colour,
            container_hdr: config.hdr,
            layout: None,
            delivery,
            ready: VecDeque::new(),
            decoded: false,
        })
    }

    /// Starts `codec` rendering into an image reader; `None` (logged)
    /// when the device can't, so the caller decodes to byte buffers.
    fn start_surface(codec: &MediaCodec, config: &DecoderConfig) -> Option<Delivery> {
        let output = image_output(config)
            .inspect_err(|error| warn!(%error, "no image reader; decoding to CPU memory"))
            .ok()?;
        codec
            .start_decoder(&decoder_format(config, None), Some(output.window()))
            .inspect_err(|error| {
                warn!(%error, "MediaCodec cannot render to an image reader; decoding to CPU memory");
            })
            .ok()?;
        Some(Delivery::Surface { output, pending: 0 })
    }

    /// Starts `codec` decoding to byte buffers in NV12 (P010 when the
    /// configuration asks for it); a codec that will not take the
    /// request decodes in its own layout.
    fn start_buffers(codec: &MediaCodec, config: &DecoderConfig) -> CodecResult<()> {
        let preferred = if config.pixel_format == PixelFormat::P010 {
            color_format::P010
        } else {
            color_format::NV12
        };
        codec
            .start_decoder(&decoder_format(config, Some(preferred)), None)
            .or_else(|_| codec.start_decoder(&decoder_format(config, None), None))
            .map_err(|error| CodecError::UnsupportedConfig {
                backend: BackendId::MediaCodec,
                what: format!("{} decode on {} ({error})", config.codec, codec.label()),
            })
    }

    fn native(&self) -> bool {
        matches!(self.delivery, Delivery::Surface { .. })
    }

    /// Reads the layout from the codec's current output format.
    fn read_layout(&mut self) -> CodecResult<Layout> {
        let layout = Self::layout_of(
            &self.codec,
            self.kind,
            self.container_colour,
            self.container_hdr,
            self.native(),
        )?;
        self.layout = Some(layout);
        Ok(layout)
    }

    /// The layout `codec`'s current output format describes.
    fn layout_of(
        codec: &MediaCodec,
        kind: Codec,
        container_colour: Option<ColourDescription>,
        container_hdr: Option<HdrMetadata>,
        native: bool,
    ) -> CodecResult<Layout> {
        let format = codec.output_format();
        let fallback = container_colour
            .unwrap_or_else(|| ColourDescription::unspecified(ColourRange::Limited));
        let mut colour = signalled_colour(&format, fallback);
        if colour.chroma_location.is_none() && matches!(kind, Codec::H264 | Codec::H265) {
            colour = colour.with_chroma_location(ChromaLocation::Left);
        }
        let hdr = format
            .buffer(key::HDR_STATIC_INFO)
            .and_then(HdrMetadata::from_cta861)
            .or(container_hdr);
        let pixels = if native {
            None
        } else {
            Some(Self::pixels_of(codec, &format)?)
        };
        Ok(Layout {
            pixels,
            colour,
            hdr,
        })
    }

    /// Where `format`'s byte-buffer pictures sit.
    fn pixels_of(codec: &MediaCodec, format: &MediaFormat) -> CodecResult<Pixels> {
        let width = int(format, key::WIDTH).unwrap_or(0);
        let height = int(format, key::HEIGHT).unwrap_or(0);
        let colour_format = format.i32(key::COLOR_FORMAT).unwrap_or(color_format::NV12);
        let (planar, depth) = match colour_format {
            color_format::NV12 | color_format::QCOM_NV12_32M => (false, BitDepth::Eight),
            color_format::P010 => (false, BitDepth::Ten),
            color_format::I420 => (true, BitDepth::Eight),
            color_format::FLEXIBLE => {
                warn!("MediaCodec decodes to flexible YUV; reading it as I420");
                (true, BitDepth::Eight)
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
        let stride = int(format, key::STRIDE)
            .filter(|&s| s >= width)
            .unwrap_or(width);
        let slice_height = int(format, key::SLICE_HEIGHT)
            .filter(|&h| h >= height)
            .unwrap_or(height);
        let crop = match (
            int(format, key::CROP_LEFT),
            int(format, key::CROP_TOP),
            int(format, key::CROP_RIGHT),
            int(format, key::CROP_BOTTOM),
        ) {
            (Some(left), Some(top), Some(right), Some(bottom))
                if right >= left && bottom >= top =>
            {
                Crop {
                    left,
                    top,
                    width: right - left + 1,
                    height: bottom - top + 1,
                }
            }
            _ => Crop::full(width, height),
        };
        if crop.width == 0 || crop.height == 0 {
            return Err(fail(
                Operation::Decode,
                format!("output format has no picture size: {crop:?}"),
            ));
        }
        Ok(Pixels {
            planar,
            depth,
            stride,
            slice_height,
            crop,
        })
    }

    /// Files one output buffer as a picture.
    fn accept(&mut self, output: &Output) -> CodecResult<()> {
        if output.rendered {
            if let Delivery::Surface { pending, .. } = &mut self.delivery {
                *pending += 1;
            }
            return self.take_images(RENDER_WAIT);
        }
        if output.data.is_empty() {
            return Ok(());
        }
        let layout = match self.layout {
            Some(layout) => layout,
            None => self.read_layout()?,
        };
        let pixels = layout.pixels.ok_or_else(|| {
            fail(
                Operation::Decode,
                "a byte-buffer picture from a codec rendering to a surface",
            )
        })?;
        let picture = self.accept_bytes(&output.data, output.pts_us, pixels, layout)?;
        self.decoded = true;
        self.ready.push_back(CapturedFrame::Cpu(picture));
        Ok(())
    }

    /// Repacks one byte-buffer picture.
    ///
    /// The unit of a P010 `stride` is not pinned down by the platform
    /// (`KEY_STRIDE` documents a pixel-index difference; some codecs
    /// report bytes): it is read as samples when the buffer is long
    /// enough for that, as bytes otherwise.
    fn accept_bytes(
        &self,
        data: &[u8],
        pts_us: u64,
        pixels: Pixels,
        layout: Layout,
    ) -> CodecResult<RawFrame> {
        let bad = |reason: String| {
            fail(
                Operation::Decode,
                format!(
                    "picture buffer of {} bytes does not hold {pixels:?}: {reason}",
                    data.len()
                ),
            )
        };
        let crop = pixels.crop;
        let planes = if pixels.planar {
            yuv420::planar(data, pixels.stride, pixels.slice_height)
        } else {
            let bytes = yuv420::sample_bytes(pixels.depth);
            let rows = pixels.slice_height + crop.height.div_ceil(2);
            let fits = |stride: usize| stride * (rows - 1) + crop.width * bytes <= data.len();
            let stride = if bytes > 1 && fits(pixels.stride * bytes) {
                pixels.stride * bytes
            } else {
                pixels.stride
            };
            yuv420::semi_planar(data, stride, stride * pixels.slice_height, pixels.depth)
        }
        .map_err(|error| bad(error.to_string()))?;
        let to = if pixels.depth == BitDepth::Ten && self.format == PixelFormat::P010 {
            BitDepth::Ten
        } else {
            BitDepth::Eight
        };
        let packed = SemiPlanar::packed(crop.width, crop.height, to);
        let mut out = vec![0; packed.len(crop.height)];
        yuv420::pack(planes, crop, pixels.depth, to, packed, &mut out)
            .map_err(|error| bad(error.to_string()))?;
        let size = |value: usize| {
            u32::try_from(value).map_err(|_| bad(format!("dimension {value} exceeds u32")))
        };
        if to == BitDepth::Ten {
            return Ok(RawFrame {
                width: size(crop.width)?,
                height: size(crop.height)?,
                stride: size(packed.stride)?,
                format: PixelFormat::P010,
                data: Bytes::from(out),
                timestamp_us: pts_us,
                colour: Some(layout.colour),
                hdr: layout.hdr,
            });
        }
        let mut picture = nv12::deliver(
            Bytes::from(out),
            packed.stride,
            (crop.width, crop.height),
            pts_us,
            self.format,
            layout.colour,
        )
        .ok_or_else(|| bad("NV12 conversion failed".into()))?;
        picture.hdr = layout.hdr;
        Ok(picture)
    }

    /// Takes the images of rendered pictures, waiting up to `wait` for
    /// the first one; the rest of the rendered ones are taken later.
    fn take_images(&mut self, mut wait: Duration) -> CodecResult<()> {
        let layout = match self.layout {
            Some(layout) => layout,
            None => self.read_layout()?,
        };
        let Delivery::Surface { output, pending } = &mut self.delivery else {
            return Ok(());
        };
        while *pending > 0 {
            let Some((surface, timestamp_us)) = output.next(wait)? else {
                break;
            };
            *pending -= 1;
            wait = Duration::ZERO;
            let crop = surface.crop();
            let size = |value: usize| {
                u32::try_from(value).map_err(|_| {
                    fail(
                        Operation::Decode,
                        format!("picture dimension {value} exceeds u32"),
                    )
                })
            };
            let surface: Arc<HardwareBuffer> = Arc::new(surface);
            self.ready.push_back(CapturedFrame::Native(NativeFrame {
                width: size(crop.width)?,
                height: size(crop.height)?,
                timestamp_us,
                colour: Some(layout.colour),
                hdr: layout.hdr,
                surface,
            }));
            self.decoded = true;
        }
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
            self.accept(&output)?;
            if ended || (timeout > Duration::ZERO && self.ready.len() > before) {
                return Ok(ended);
            }
        }
    }

    fn next(&mut self, timeout: Duration) -> CodecResult<Option<Output>> {
        let native = self.native();
        let output =
            self.codec
                .next_output(Operation::Decode, timeout, || self.layout = None, native)?;
        Ok(output)
    }

    fn decode(&mut self, frame: EncodedFrame) -> CodecResult<Option<CapturedFrame>> {
        // Output drained while waiting for an input buffer is filed after
        // the queue; each buffer keeps the layout current when it was
        // dequeued, so a format change part-way through the batch only
        // applies to the buffers after it.
        let mut early = Vec::new();
        let mut layout = self.layout;
        let (kind, colour, hdr, native) = (
            self.kind,
            self.container_colour,
            self.container_hdr,
            self.native(),
        );
        self.codec
            .queue(
                Operation::Decode,
                frame.timestamp_us,
                0,
                |buffer| copy_into(buffer, &frame.data, Operation::Decode),
                |codec| {
                    loop {
                        let mut changed = false;
                        let output = codec.next_output(
                            Operation::Decode,
                            Duration::ZERO,
                            || changed = true,
                            native,
                        )?;
                        if changed {
                            layout = Some(Self::layout_of(codec, kind, colour, hdr, native)?);
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
            self.accept(&output)?;
        }
        self.layout = layout;
        // Pictures rendered earlier whose image was late.
        self.take_images(Duration::ZERO)?;
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
        // The last rendered pictures may still be on their way.
        while matches!(self.delivery, Delivery::Surface { pending, .. } if pending > 0)
            && Instant::now() < deadline
        {
            let before = self.ready.len();
            self.take_images(RENDER_WAIT)?;
            if self.ready.len() == before {
                warn!("rendered pictures never reached the image reader");
                break;
            }
        }
        if let Delivery::Surface { pending, .. } = &mut self.delivery {
            *pending = 0;
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
            && session.memory == config.memory
        {
            session.format = config.pixel_format;
            session.low_latency = config.low_latency;
            session.container_colour = config.colour;
            session.container_hdr = config.hdr;
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
