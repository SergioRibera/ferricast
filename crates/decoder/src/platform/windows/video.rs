//! Video decoding through Media Foundation transforms.
//!
//! Each packet goes in as one sample stamped with its timestamp, which
//! the MFT carries to the picture decoded from it through its own
//! reordering. A call returns at most one picture; pictures that came
//! out together wait for the next calls, and `flush` drains the rest.
//!
//! Hardware transforms decode into Direct3D 11 textures (a device of
//! their own, handed over with `MFT_MESSAGE_SET_D3D_MANAGER`): those
//! go out as [`CapturedFrame::Native`] for [`FrameMemory::Native`] and
//! are read back otherwise. Software transforms decode to system memory.
//! 10-bit streams come out as P010 when the configuration asks for it
//! (`pixel_format`), downshifted to NV12 on the CPU path otherwise.

use std::collections::VecDeque;
use std::sync::Arc;

use bytes::Bytes;
use ferricast_codec::{OpenFn, ProbeFn, VideoDecoderSession};
use ferricast_core::{
    CapturedFrame, Codec, CodecError, CodecResult, CodecTarget, ColourDescription, ColourRange,
    DecoderConfig, EncodedFrame, FrameMemory, HdrMetadata, NativeFrame, Operation, PixelFormat,
    RawFrame,
};
use windows::Win32::Media::MediaFoundation::{
    IMFMediaType, IMFTransform, MF_E_INVALIDMEDIATYPE, MF_E_UNSUPPORTED_D3D_TYPE,
    MF_E_UNSUPPORTED_FORMAT, MF_MT_DEFAULT_STRIDE, MF_MT_FRAME_SIZE,
    MF_MT_MINIMUM_DISPLAY_APERTURE, MF_MT_SUBTYPE, MFMediaType_Video, MFT_CATEGORY_VIDEO_DECODER,
    MFVideoFormat_AV1, MFVideoFormat_H264, MFVideoFormat_HEVC, MFVideoFormat_NV12,
    MFVideoFormat_P010, MFVideoFormat_VP80, MFVideoFormat_VP90,
};
use windows::core::GUID;

use super::mf::{
    self, BACKEND, D3d11Texture, Device, Kind, MfResult, Output, Rect, Search, Transform, Worker,
    colour_of, hdr_of, hns, media_type, micros, pack, unpack,
};
use crate::nv12;
use crate::nv12::downshift_p010;

/// The open and probe functions of one (codec, kind) candidate.
#[derive(Clone, Copy)]
pub(super) struct Opener {
    pub(super) open: OpenFn<DecoderConfig, dyn VideoDecoderSession>,
    pub(super) probe: ProbeFn,
}

/// Codec indices for the const-generic openers.
const H264: u8 = 0;
const H265: u8 = 1;
const VP8: u8 = 2;
const VP9: u8 = 3;
const AV1: u8 = 4;

fn codec_of(index: u8) -> Codec {
    match index {
        H264 => Codec::H264,
        H265 => Codec::H265,
        VP8 => Codec::Vp8,
        VP9 => Codec::Vp9,
        _ => Codec::Av1,
    }
}

/// The hardware and software openers of `codec`.
pub(super) fn openers(codec: Codec) -> Option<(Opener, Opener)> {
    fn pair<const CODEC: u8>() -> (Opener, Opener) {
        (
            Opener {
                open: open::<CODEC, true>,
                probe: probe::<CODEC, true>,
            },
            Opener {
                open: open::<CODEC, false>,
                probe: probe::<CODEC, false>,
            },
        )
    }
    Some(match codec {
        Codec::H264 => pair::<H264>(),
        Codec::H265 => pair::<H265>(),
        Codec::Vp8 => pair::<VP8>(),
        Codec::Vp9 => pair::<VP9>(),
        Codec::Av1 => pair::<AV1>(),
    })
}

fn search(codec: Codec, hardware: bool) -> Search {
    let (subtype, job) = match codec {
        Codec::H264 => (MFVideoFormat_H264, "H.264 decoder"),
        Codec::H265 => (MFVideoFormat_HEVC, "H.265 decoder"),
        Codec::Vp8 => (MFVideoFormat_VP80, "VP8 decoder"),
        Codec::Vp9 => (MFVideoFormat_VP90, "VP9 decoder"),
        Codec::Av1 => (MFVideoFormat_AV1, "AV1 decoder"),
    };
    Search {
        category: MFT_CATEGORY_VIDEO_DECODER,
        kind: if hardware {
            Kind::Hardware
        } else {
            Kind::Software
        },
        input: Some((MFMediaType_Video, subtype)),
        output: None,
        job,
    }
}

fn probe<const CODEC: u8, const HARDWARE: bool>() -> CodecResult<()> {
    search(codec_of(CODEC), HARDWARE).probe()
}

fn open<const CODEC: u8, const HARDWARE: bool>(
    config: &DecoderConfig,
) -> CodecResult<Box<dyn VideoDecoderSession>> {
    let codec = codec_of(CODEC);
    if config.codec != codec {
        return Err(CodecError::BackendCannotHandle {
            backend: BACKEND,
            target: CodecTarget::decode(config.codec),
        });
    }
    Ok(Box::new(MfVideoDecoder::open(
        search(codec, HARDWARE),
        config,
    )?))
}

/// A Media Foundation video decoder session.
struct MfVideoDecoder {
    search: Search,
    worker: Worker<Session>,
}

impl MfVideoDecoder {
    fn open(search: Search, config: &DecoderConfig) -> CodecResult<Self> {
        let config = config.clone();
        let worker = Worker::spawn("ferricast-mf-video-decoder", move || {
            search.open(|transform| Session::setup(transform, &config, search.kind))
        })?;
        Ok(Self { search, worker })
    }
}

impl VideoDecoderSession for MfVideoDecoder {
    fn configure(&mut self, config: &DecoderConfig) -> CodecResult<()> {
        *self = Self::open(self.search, config)?;
        Ok(())
    }

    fn decode(&mut self, frame: EncodedFrame) -> CodecResult<Option<CapturedFrame>> {
        self.worker
            .run(Operation::Decode, move |session| session.decode(&frame))
    }

    fn flush(&mut self) -> CodecResult<Vec<CapturedFrame>> {
        self.worker.run(Operation::Flush, Session::flush)
    }
}

/// Where the planes of the MFT's NV12 / P010 output are.
#[derive(Debug, Clone, Copy)]
struct Picture {
    /// `Nv12` or `P010`: the output subtype.
    format: PixelFormat,
    /// Visible size.
    width: usize,
    height: usize,
    /// Visible origin inside the decoded surface.
    x: usize,
    y: usize,
    /// Rows of the luma plane (`MF_MT_FRAME_SIZE`; 1088 for 1080p
    /// H.264): the chroma plane starts after them.
    coded_height: usize,
    /// Bytes per row, when the type states it.
    stride: Option<usize>,
    colour: Option<ColourDescription>,
    hdr: Option<HdrMetadata>,
}

impl Picture {
    /// Bytes per sample.
    fn sample_bytes(&self) -> usize {
        if self.format == PixelFormat::P010 {
            2
        } else {
            1
        }
    }

    /// The visible area, for a texture.
    fn visible(&self) -> Option<Rect> {
        let to = |v: usize| u32::try_from(v).ok();
        Some(Rect {
            x: to(self.x)?,
            y: to(self.y)?,
            width: to(self.width)?,
            height: to(self.height)?,
        })
    }
}

struct Session {
    transform: Transform,
    format: PixelFormat,
    memory: FrameMemory,
    fallback_colour: ColourDescription,
    fallback_hdr: Option<HdrMetadata>,
    picture: Picture,
    /// The Direct3D device the transform decodes on, when it took one;
    /// its outputs are then textures.
    device: Option<Device>,
    ready: VecDeque<CapturedFrame>,
    /// Timestamps of packets fed but not yet decoded into a picture.
    in_flight: Timestamps,
}

/// Output samples a Direct3D decoder keeps spare for frames consumers
/// still hold.
const SPARE_SURFACES: u32 = 6;

/// Picks the first offered output type among `preferred`, in that order.
fn select_output(mft: &IMFTransform, preferred: [GUID; 2]) -> windows::core::Result<()> {
    let mut offered = Vec::new();
    for index in 0.. {
        // SAFETY: plain COM calls on the live transform; the loop ends
        // when the MFT runs out of types (an error).
        let Ok(media) = (unsafe { mft.GetOutputAvailableType(0, index) }) else {
            break;
        };
        // SAFETY: plain attribute read.
        let subtype = unsafe { media.GetGUID(&MF_MT_SUBTYPE)? };
        offered.push((subtype, media));
    }
    for wanted in preferred {
        if let Some((_, media)) = offered.iter().find(|(subtype, _)| *subtype == wanted) {
            // SAFETY: plain COM call on the live transform.
            return unsafe { mft.SetOutputType(0, media, 0) };
        }
    }
    Err(MF_E_INVALIDMEDIATYPE.into())
}

/// What the transform reports, each half completed from the container's
/// `fallback`.
fn merge_hdr(reported: Option<HdrMetadata>, fallback: Option<HdrMetadata>) -> Option<HdrMetadata> {
    let (reported, fallback) = (reported.unwrap_or_default(), fallback.unwrap_or_default());
    let merged = HdrMetadata {
        mastering_display: reported.mastering_display.or(fallback.mastering_display),
        content_light: reported.content_light.or(fallback.content_light),
    };
    (!merged.is_empty()).then_some(merged)
}

/// NV12, else P010 (downshifted to 8 bits on the CPU path).
fn select_nv12(mft: &IMFTransform) -> windows::core::Result<()> {
    select_output(mft, [MFVideoFormat_NV12, MFVideoFormat_P010])
}

/// P010 for 10-bit streams (an 8-bit stream offers no P010), else NV12.
fn select_p010(mft: &IMFTransform) -> windows::core::Result<()> {
    select_output(mft, [MFVideoFormat_P010, MFVideoFormat_NV12])
}

/// A failed decode call: a stream the MFT does not implement can go to
/// another backend.
fn decode_failure(call: &str, error: &windows::core::Error) -> CodecError {
    let code = error.code();
    if code == MF_E_INVALIDMEDIATYPE
        || code == MF_E_UNSUPPORTED_FORMAT
        || code == MF_E_UNSUPPORTED_D3D_TYPE
    {
        return CodecError::UnsupportedStream {
            backend: BACKEND,
            reason: format!(
                "{call} failed with HRESULT {:#010X}: {}",
                code.0,
                error.message()
            ),
        };
    }
    mf::failure(Operation::Decode, call, error)
}

impl Session {
    /// `kind` decides the memory: a hardware transform decodes into
    /// Direct3D 11 textures when it can (handed out as they are for
    /// [`FrameMemory::Native`], read back otherwise), a software one into
    /// system memory.
    fn setup(mut transform: Transform, config: &DecoderConfig, kind: Kind) -> CodecResult<Self> {
        let op = Operation::Open;
        let subtype = search(config.codec, false)
            .input
            .map_or(GUID::zeroed(), |(_, subtype)| subtype);
        if config.low_latency {
            transform.request_low_latency();
        }
        let device = match kind {
            Kind::Software => None,
            Kind::Hardware => match Device::hardware() {
                Ok(device) => transform.attach_d3d11(&device)?.then_some(device),
                Err(error) => {
                    tracing::info!(transform = transform.name(), %error, "decoding to system memory");
                    None
                }
            },
        };
        if device.is_some() {
            transform.reserve_output_samples(SPARE_SURFACES);
        }
        let (select, wanted): (mf::Renegotiate, &str) = if config.pixel_format == PixelFormat::P010
        {
            (select_p010, "P010 or NV12")
        } else {
            (select_nv12, "NV12 or P010")
        };
        let input = media_type(&MFMediaType_Video, &subtype, op)?;
        if config.width > 0 && config.height > 0 {
            // SAFETY: plain attribute write.
            unsafe { input.SetUINT64(&MF_MT_FRAME_SIZE, pack(config.width, config.height)) }
                .mf(op, "setting MF_MT_FRAME_SIZE")?;
        }
        // SAFETY: plain COM call on the live transform.
        unsafe { transform.mft().SetInputType(0, &input, 0) }
            .mf(op, "IMFTransform::SetInputType")?;
        select(transform.mft()).map_err(|error| CodecError::UnsupportedConfig {
            backend: BACKEND,
            what: format!(
                "{wanted} {} output from {} (HRESULT {:#010X})",
                if device.is_some() {
                    "Direct3D"
                } else {
                    "system-memory"
                },
                transform.name(),
                error.code().0
            ),
        })?;
        transform.set_renegotiate(select);
        // P010 worst case: two bytes per sample, 4:2:0.
        transform.set_default_output_size(
            config
                .width
                .max(64)
                .saturating_mul(config.height.max(64))
                .saturating_mul(3),
        );
        transform.start()?;
        let picture = picture_of(transform.mft(), (config.width, config.height))?;
        Ok(Self {
            transform,
            format: config.pixel_format,
            memory: config.memory,
            fallback_colour: config
                .colour
                .unwrap_or(ColourDescription::unspecified(ColourRange::Limited)),
            fallback_hdr: config.hdr,
            picture,
            device,
            ready: VecDeque::new(),
            in_flight: Timestamps::default(),
        })
    }

    fn decode(&mut self, frame: &EncodedFrame) -> CodecResult<Option<CapturedFrame>> {
        let op = Operation::Decode;
        let sample = mf::sample_with(frame.data.len(), 16, op, |dst| {
            dst.copy_from_slice(&frame.data);
            Ok(())
        })?;
        // SAFETY: plain attribute writes on the new sample.
        unsafe {
            sample
                .SetSampleTime(hns(frame.timestamp_us))
                .mf(op, "IMFSample::SetSampleTime")?;
            if let Some(duration) = frame.duration_us {
                sample
                    .SetSampleDuration(hns(duration))
                    .mf(op, "IMFSample::SetSampleDuration")?;
            }
        }
        let mut outputs = Vec::new();
        self.transform
            .push(&sample, &mut outputs)
            .map_err(|error| decode_failure("IMFTransform::ProcessInput", &error))?;
        self.transform
            .pull(&mut outputs, None)
            .map_err(|error| decode_failure("IMFTransform::ProcessOutput", &error))?;
        self.in_flight.push(frame.timestamp_us);
        self.collect(outputs)?;
        Ok(self.ready.pop_front())
    }

    fn flush(&mut self) -> CodecResult<Vec<CapturedFrame>> {
        let mut outputs = Vec::new();
        self.transform
            .drain(&mut outputs)
            .mf(Operation::Flush, "draining the decoder")?;
        self.collect(outputs)?;
        Ok(self.ready.drain(..).collect())
    }

    fn collect(&mut self, outputs: Vec<Output>) -> CodecResult<()> {
        let op = Operation::Decode;
        for output in outputs {
            let sample = match output {
                Output::Sample(sample) => sample,
                Output::FormatChanged => {
                    let known = (
                        self.picture.x + self.picture.width,
                        self.picture.coded_height,
                    );
                    let known = (
                        u32::try_from(known.0).unwrap_or(0),
                        u32::try_from(known.1).unwrap_or(0),
                    );
                    self.picture = picture_of(self.transform.mft(), known)?;
                    continue;
                }
            };
            // SAFETY: plain attribute read on the output sample.
            let timestamp_us = self
                .in_flight
                .take(unsafe { sample.GetSampleTime() }.ok().map(micros));
            let picture = self.picture;
            let colour = picture.colour.unwrap_or(self.fallback_colour);
            let hdr = merge_hdr(picture.hdr, self.fallback_hdr);
            let texture = match (&self.device, picture.visible()) {
                (Some(_), Some(visible)) => D3d11Texture::from_sample(&sample, visible)?,
                _ => None,
            };
            let (data, stride, format) = match texture {
                Some(texture) => {
                    let frame = NativeFrame {
                        width: u32::try_from(picture.width).unwrap_or(u32::MAX),
                        height: u32::try_from(picture.height).unwrap_or(u32::MAX),
                        timestamp_us,
                        colour: Some(colour),
                        hdr,
                        surface: Arc::new(texture),
                    };
                    if self.memory == FrameMemory::Native {
                        self.ready.push_back(CapturedFrame::Native(frame));
                        continue;
                    }
                    let raw = frame
                        .into_cpu()
                        .map_err(|error| CodecError::backend(BACKEND, op, error))?;
                    (raw.data.to_vec(), raw.stride as usize, raw.format)
                }
                None => {
                    let (packed, stride) = self.pack(&mf::sample_bytes(&sample, op)?)?;
                    (packed, stride, picture.format)
                }
            };
            let raw = self.deliver(data, stride, format, timestamp_us, colour, hdr)?;
            self.ready.push_back(CapturedFrame::Cpu(raw));
        }
        Ok(())
    }

    /// A CPU picture (NV12 or P010, chroma right after the luma rows) in
    /// the format the configuration asks for: P010 stays P010 only when
    /// asked for, and is downshifted to 8 bits otherwise.
    fn deliver(
        &self,
        data: Vec<u8>,
        stride: usize,
        format: PixelFormat,
        timestamp_us: u64,
        colour: ColourDescription,
        hdr: Option<HdrMetadata>,
    ) -> CodecResult<RawFrame> {
        let (width, height) = (self.picture.width, self.picture.height);
        let failed = || {
            CodecError::backend(
                BACKEND,
                Operation::Decode,
                format!("converting the decoded {format:?} picture failed"),
            )
        };
        if format == PixelFormat::P010 && self.format == PixelFormat::P010 {
            return Ok(RawFrame {
                width: u32::try_from(width).map_err(|_| failed())?,
                height: u32::try_from(height).map_err(|_| failed())?,
                stride: u32::try_from(stride).map_err(|_| failed())?,
                format,
                data: Bytes::from(data),
                timestamp_us,
                colour: Some(colour),
                hdr,
            });
        }
        let (data, stride) = if format == PixelFormat::P010 {
            downshift_p010(&data, stride, width, height)
        } else {
            (data, stride)
        };
        let mut raw = nv12::deliver(
            Bytes::from(data),
            stride,
            (width, height),
            timestamp_us,
            self.format,
            colour,
        )
        .ok_or_else(failed)?;
        raw.hdr = hdr;
        Ok(raw)
    }

    /// The visible picture of a decoded surface as NV12 / P010 with the
    /// chroma plane right after the luma rows, and its stride.
    fn pack(&self, surface: &[u8]) -> CodecResult<(Vec<u8>, usize)> {
        let p = self.picture;
        let fail = |what: &str| {
            CodecError::backend(
                BACKEND,
                Operation::Decode,
                format!(
                    "{what} (surface of {} bytes for {}x{})",
                    surface.len(),
                    p.width,
                    p.height
                ),
            )
        };
        let bytes = p.sample_bytes();
        let coded_width = (p.x + p.width) * bytes;
        let stride = p.stride.unwrap_or(coded_width).max(coded_width);
        if stride == 0 || p.width == 0 || p.height == 0 || p.coded_height < p.y + p.height {
            return Err(fail(
                "the decoder reported an empty or inconsistent picture size",
            ));
        }
        // The chroma plane follows the coded luma rows, whatever padding
        // the buffer carries after it.
        let chroma = stride * p.coded_height;
        if surface.len() < chroma + stride * p.coded_height.div_ceil(2) {
            return Err(fail("decoded surface smaller than its picture"));
        }
        let width = p.width * bytes;
        let row_bytes = p.width.div_ceil(2) * 2 * bytes;
        let x = p.x / 2 * 2 * bytes;
        let mut out = vec![0; row_bytes * p.height + row_bytes * p.height.div_ceil(2)];
        let (luma_out, chroma_out) = out.split_at_mut(row_bytes * p.height);
        for (row, line) in luma_out.chunks_exact_mut(row_bytes).enumerate() {
            let start = (p.y + row) * stride + x;
            let src = surface
                .get(start..start + width)
                .ok_or_else(|| fail("luma row out of range"))?;
            line[..width].copy_from_slice(src);
        }
        for (row, line) in chroma_out.chunks_exact_mut(row_bytes).enumerate() {
            let start = chroma + (p.y / 2 + row) * stride + x;
            let src = surface
                .get(start..start + row_bytes)
                .ok_or_else(|| fail("chroma row out of range"))?;
            line.copy_from_slice(src);
        }
        Ok((out, row_bytes))
    }
}

/// The picture geometry of the MFT's current output type; the
/// `fallback` size stands in until the decoder knows the stream's.
fn picture_of(mft: &IMFTransform, fallback: (u32, u32)) -> CodecResult<Picture> {
    // SAFETY: plain COM call on the live transform.
    let current: IMFMediaType = unsafe { mft.GetOutputCurrentType(0) }
        .mf(Operation::Decode, "IMFTransform::GetOutputCurrentType")?;
    // SAFETY: plain attribute reads on the live media type.
    let (size, stride, subtype) = unsafe {
        (
            current.GetUINT64(&MF_MT_FRAME_SIZE).ok().map(unpack),
            current.GetUINT32(&MF_MT_DEFAULT_STRIDE).ok(),
            current.GetGUID(&MF_MT_SUBTYPE).ok(),
        )
    };
    let (coded_width, coded_height) = size.unwrap_or(fallback);
    let mut picture = Picture {
        format: if subtype == Some(MFVideoFormat_P010) {
            PixelFormat::P010
        } else {
            PixelFormat::Nv12
        },
        width: coded_width as usize,
        height: coded_height as usize,
        x: 0,
        y: 0,
        coded_height: coded_height as usize,
        // A negative stride means a bottom-up image, which NV12 / P010
        // never is.
        stride: stride
            .and_then(|stride| i32::from_ne_bytes(stride.to_ne_bytes()).checked_abs())
            .and_then(|stride| usize::try_from(stride).ok())
            .filter(|&stride| stride > 0),
        colour: colour_of(&current),
        hdr: hdr_of(&current),
    };
    if let Some(area) = aperture(&current) {
        picture.x = area.0;
        picture.y = area.1;
        picture.width = area.2.min(picture.width.saturating_sub(area.0));
        picture.height = area.3.min(picture.height.saturating_sub(area.1));
    }
    Ok(picture)
}

/// `MF_MT_MINIMUM_DISPLAY_APERTURE` as `(x, y, width, height)`: two
/// `MFOffset`s (16-bit fraction, 16-bit integer) and a `SIZE`.
fn aperture(media: &IMFMediaType) -> Option<(usize, usize, usize, usize)> {
    let mut blob = [0u8; 16];
    // SAFETY: the blob is read into a 16-byte buffer, its declared size.
    unsafe { media.GetBlob(&MF_MT_MINIMUM_DISPLAY_APERTURE, &mut blob, None) }.ok()?;
    let offset = |at: usize| usize::try_from(i16::from_le_bytes([blob[at + 2], blob[at + 3]])).ok();
    let size = |at: usize| {
        usize::try_from(i32::from_le_bytes([
            blob[at],
            blob[at + 1],
            blob[at + 2],
            blob[at + 3],
        ]))
        .ok()
    };
    let (x, y, width, height) = (offset(0)?, offset(4)?, size(8)?, size(12)?);
    (width > 0 && height > 0).then_some((x, y, width, height))
}

/// Timestamps of the packets an MFT holds. MFTs carry each packet's
/// sample time to its picture; one that drops it gets the earliest
/// pending timestamp (pictures leave in presentation order) instead of
/// a made-up zero.
#[derive(Debug, Default)]
struct Timestamps {
    pending: std::collections::BTreeMap<u64, usize>,
    /// The last timestamp handed out.
    last: u64,
    warned: bool,
}

impl Timestamps {
    /// More than any decoder's reorder window; beyond it the oldest
    /// entries belong to packets the MFT dropped.
    const LIMIT: usize = 64;

    fn push(&mut self, timestamp_us: u64) {
        *self.pending.entry(timestamp_us).or_default() += 1;
        if self.pending.values().sum::<usize>() > Self::LIMIT {
            self.pending.pop_first();
        }
    }

    /// The timestamp of a picture whose sample says `reported`.
    fn take(&mut self, reported: Option<u64>) -> u64 {
        let key = match reported {
            Some(time) if self.pending.contains_key(&time) => Some(time),
            Some(time) => {
                self.last = time;
                return time;
            }
            None => {
                if !self.warned {
                    self.warned = true;
                    tracing::warn!(
                        "Media Foundation decoder output carries no sample time; using the \
                         earliest pending packet timestamp"
                    );
                }
                self.pending.keys().next().copied()
            }
        };
        let Some(key) = key else {
            tracing::warn!(
                last_us = self.last,
                "Media Foundation decoder output matches no pending packet timestamp; repeating \
                 the last one"
            );
            return self.last;
        };
        if let Some(count) = self.pending.get_mut(&key) {
            *count -= 1;
            if *count == 0 {
                self.pending.remove(&key);
            }
        }
        self.last = key;
        key
    }
}
