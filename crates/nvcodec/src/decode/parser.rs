//! The cuvid parser, its callbacks, and the NVDEC decoder they drive.
//!
//! `cuvidParseVideoData` calls the callbacks synchronously, on the calling
//! thread, before it returns. They reach the decoder's [`State`] through the
//! parser's user-data pointer and nothing else: [`Decoder`](super::Decoder)
//! holds no reference into the state while the parser runs, so the callbacks
//! own it exclusively without any lock. A callback never unwinds into the
//! driver (every body runs under `catch_unwind`); its error is stored in the
//! state, the callback reports failure, and the `decode` call that drove the
//! parser returns that error.

use std::collections::VecDeque;
use std::ffi::{c_int, c_void};
use std::fmt;
use std::panic::{self, AssertUnwindSafe};
use std::ptr::{self, NonNull};
use std::sync::Arc;

use super::caps::DecoderCaps;
use super::crop::CropScanner;
use super::format::{DecodeStatus, Fields, FrameInfo, Layout, PixelFormat, Rect, SequenceInfo};
use super::frame::{DeviceFrame, DeviceSlot, HostFrame, HostSlot};
use super::pool::Pool;
use super::sei::SeiStore;
use super::{Codec, CuvidCheck as _, DecoderConfig};
use crate::cuda::{Context, DeviceBuffer, Event, Stream};
use crate::error::{CheckStatus as _, Error, Result};
use crate::loader::{CudaLibrary, CuvidLibrary};
use crate::sys::cuda as cu;
use crate::sys::cuvid as sys;

/// Surfaces to fall back on when a driver reports no minimum (the most any
/// of these codecs needs: a full H.264 / HEVC DPB plus the current picture
/// and headroom).
const FALLBACK_DECODE_SURFACES: u32 = 20;

/// Most SEI messages read for one picture; more is a corrupt report.
const MAX_SEI_MESSAGES: usize = 256;

/// Where decoded pictures go.
pub(super) enum Output {
    Host {
        pool: Pool<HostSlot>,
        queue: VecDeque<HostFrame>,
    },
    Device {
        pool: Pool<DeviceSlot>,
        queue: VecDeque<DeviceFrame>,
    },
}

impl Output {
    fn retire(&self) {
        match self {
            Self::Host { pool, .. } => pool.retire(),
            Self::Device { pool, .. } => pool.retire(),
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Host { queue, .. } => queue.len(),
            Self::Device { queue, .. } => queue.len(),
        }
    }
}

/// Everything the callbacks touch.
pub(super) struct State {
    pub(super) config: DecoderConfig,
    cuvid: &'static CuvidLibrary,
    cuda: &'static CudaLibrary,
    context: Arc<Context>,
    /// Owned by the `Decoder`, which destroys it after this state.
    lock: sys::CUvideoctxlock,
    /// Every NVDEC map and every copy out of a mapped surface runs here.
    pub(super) stream: Stream,
    decoder: Option<HwDecoder>,
    pub(super) sequence: Option<SequenceInfo>,
    /// Left / top crop of the newest SPS (H.264, HEVC).
    pub(super) crop: CropScanner,
    /// SEI of each decode surface, from the SEI callback until display.
    sei: Vec<SeiStore>,
    pub(super) output: Output,
    /// The first error a callback hit during the current parse.
    pub(super) error: Option<Error>,
}

impl State {
    pub(super) fn new(
        config: DecoderConfig,
        context: &Arc<Context>,
        lock: sys::CUvideoctxlock,
        device_output: bool,
    ) -> Result<Self> {
        let output = if device_output {
            Output::Device {
                pool: Pool::new(config.max_frames),
                queue: VecDeque::with_capacity(config.max_frames),
            }
        } else {
            Output::Host {
                pool: Pool::new(config.max_frames),
                queue: VecDeque::with_capacity(config.max_frames),
            }
        };
        Ok(Self {
            config,
            cuvid: super::cuvid_library()?,
            cuda: CudaLibrary::get()?,
            context: Arc::clone(context),
            lock,
            stream: Stream::new(context)?,
            decoder: None,
            crop: CropScanner::default(),
            sequence: None,
            sei: Vec::new(),
            output,
            error: None,
        })
    }

    /// A new sequence (or the first): checks it against the GPU and creates,
    /// reconfigures or keeps the decoder. Returns the number of decode
    /// surfaces, which overrides the parser's.
    fn on_sequence(&mut self, raw: &sys::CUVIDEOFORMAT) -> Result<c_int> {
        let mut sequence = SequenceInfo::from_raw(self.config.codec, raw)?;
        if let Some(crop) = self.crop.latest {
            sequence.display_area = crop.refine(
                sequence.display_area,
                sequence.coded_width,
                sequence.coded_height,
            );
        }
        let needed = match sequence.min_decode_surfaces {
            0 => FALLBACK_DECODE_SURFACES,
            minimum => minimum,
        }
        .saturating_add(self.config.max_display_delay);
        let geometry = |info: &SequenceInfo| {
            (
                info.format,
                info.display_area.width(),
                info.display_area.height(),
            )
        };
        if self.sequence.as_ref().map(geometry) != Some(geometry(&sequence)) {
            self.output.retire();
        }
        let reusable = self.decoder.as_ref().is_some_and(|decoder| {
            decoder.fits(&sequence, needed)
                && (self.cuvid.reconfigure_decoder.is_some() || decoder.same_size(&sequence))
        });
        let surfaces = match self.decoder.as_mut() {
            Some(decoder) if reusable => {
                if !decoder.same_size(&sequence) {
                    decoder.reconfigure(&sequence)?;
                }
                decoder.surfaces
            }
            _ => {
                // Free the old decoder's surfaces before allocating new ones.
                self.decoder = None;
                let decoder = HwDecoder::create(
                    self.cuvid,
                    &self.context,
                    self.lock,
                    &sequence,
                    needed,
                    &self.config,
                )?;
                let surfaces = decoder.surfaces;
                self.decoder = Some(decoder);
                surfaces
            }
        };
        let slots = surfaces as usize;
        if self.sei.len() < slots {
            self.sei.resize_with(slots, SeiStore::default);
        }
        tracing::debug!(?sequence, surfaces, "NVDEC sequence");
        self.sequence = Some(sequence);
        c_int::try_from(surfaces).map_err(|_| Error::SizeOverflow {
            what: "the decode surface count",
        })
    }

    fn on_decode(&mut self, params: *mut sys::CUVIDPICPARAMS) -> Result<c_int> {
        let decoder = self.decoder.as_ref().ok_or_else(no_sequence)?;
        // SAFETY: the parser passes a valid picture for the callback's
        // duration.
        let index = unsafe { (*params).CurrPicIdx };
        decoder.check_index(index)?;
        let _current = self.context.guard()?;
        // SAFETY: a live decoder, its context current, and the parser's
        // picture parameters for a surface index inside the decoder's range.
        unsafe { (self.cuvid.decode_picture)(decoder.raw.as_ptr(), params) }
            .check_cuvid("cuvidDecodePicture")?;
        Ok(1)
    }

    fn on_display(&mut self, info: &sys::CUVIDPARSERDISPINFO) -> Result<c_int> {
        let Self {
            cuda,
            decoder,
            sequence,
            sei,
            output,
            stream,
            context,
            ..
        } = self;
        let (Some(decoder), Some(sequence)) = (decoder.as_ref(), sequence.as_ref()) else {
            return Err(no_sequence());
        };
        let index = decoder.check_index(info.picture_index)?;
        let _current = context.guard()?;
        let mut params = sys::CUVIDPROCPARAMS {
            progressive_frame: info.progressive_frame,
            top_field_first: info.top_field_first,
            unpaired_field: c_int::from(info.repeat_first_field < 0),
            output_stream: stream.raw().cast(),
            ..Default::default()
        };
        let mapped = decoder.map(info.picture_index, &mut params)?;
        let area = sequence.display_area;
        let frame = FrameInfo {
            timestamp: info.timestamp,
            width: area.width(),
            height: area.height(),
            format: sequence.format,
            bit_depth: sequence.bit_depth,
            colour: sequence.colour,
            display_area: area,
            coded_width: sequence.coded_width,
            coded_height: sequence.coded_height,
            fields: Fields {
                progressive: info.progressive_frame != 0,
                top_field_first: info.top_field_first != 0,
                repeat_first_field: info.repeat_first_field,
            },
            status: decoder.status(info.picture_index),
        };
        let source = Source {
            ptr: mapped.ptr,
            pitch: mapped.pitch,
            plane_rows: decoder.target_height as usize,
            area,
            format: sequence.format,
        };
        let picture_sei = sei.get_mut(index);
        match output {
            Output::Host { pool, queue } => {
                let layout = Layout::packed(frame.format, frame.width, frame.height);
                let mut slot = pool.take(|| Ok(HostSlot::default()))?;
                slot.data.resize(layout.total_bytes(), 0);
                let destination = Destination::Host(&mut slot.data);
                source.copy(cuda, stream, &layout, destination)?;
                // Pageable host memory: the copy has finished when the call
                // returns, but only a sync orders it with this function's
                // return on every driver.
                stream.synchronize()?;
                mapped.unmap()?;
                take_sei(&mut slot.sei, picture_sei);
                queue.push_back(HostFrame {
                    info: frame,
                    layout,
                    slot,
                });
            }
            Output::Device { pool, queue } => {
                let mut slot = pool
                    .take(|| DeviceSlot::new(context, frame.format, frame.width, frame.height))?;
                let layout =
                    Layout::pitched(frame.format, frame.width, frame.height, slot.buffer.pitch());
                if slot.release_pending {
                    stream.wait(&slot.released)?;
                    slot.release_pending = false;
                }
                let destination = Destination::Device(slot.buffer.device_ptr());
                source.copy(cuda, stream, &layout, destination)?;
                slot.ready.record(stream)?;
                // The copy is queued on `stream`, where any later map of this
                // output surface also runs, so the surface is released
                // without waiting for the copy.
                mapped.unmap()?;
                take_sei(&mut slot.sei, picture_sei);
                queue.push_back(DeviceFrame {
                    info: frame,
                    layout,
                    slot,
                });
            }
        }
        Ok(1)
    }

    fn on_sei(&mut self, info: &sys::CUVIDSEIMESSAGEINFO) -> c_int {
        let Ok(index) = usize::try_from(info.picIdx) else {
            return 1;
        };
        let count = (info.sei_message_count as usize).min(MAX_SEI_MESSAGES);
        let messages = if info.pSEIMessage.is_null() || count == 0 {
            &[][..]
        } else {
            // SAFETY: the parser passes `sei_message_count` descriptors for
            // the callback's duration.
            unsafe { std::slice::from_raw_parts(info.pSEIMessage, count) }
        };
        let total = messages.iter().try_fold(0_usize, |total, message| {
            total.checked_add(usize::try_from(message.sei_message_size).ok()?)
        });
        let data = match total {
            Some(total) if total > 0 && !info.pSEIData.is_null() => {
                // SAFETY: the parser's SEI buffer holds the payloads of the
                // messages described, back to back, for the callback's
                // duration.
                unsafe { std::slice::from_raw_parts(info.pSEIData.cast::<u8>(), total) }
            }
            _ => &[][..],
        };
        if index >= self.sei.len() {
            self.sei.resize_with(index + 1, SeiStore::default);
        }
        if let Some(store) = self.sei.get_mut(index) {
            store.fill(data, messages);
        }
        1
    }

    fn on_operating_point(&self, info: &sys::CUVIDOPERATINGPOINTINFO) -> c_int {
        if info.codec != Codec::Av1.raw() {
            return 0;
        }
        // SAFETY: for AV1 the parser fills the `av1` member; every member is
        // plain integers, so any bytes are a valid value.
        let count = unsafe { info.__bindgen_anon_1.av1.operating_points_cnt };
        let wanted = self.config.av1_operating_point;
        // Bits 0-9: the operating point; bit 10 (output all layers) clear.
        if wanted < u16::from(count) {
            c_int::from(wanted)
        } else {
            0
        }
    }
}

/// Moves a picture's SEI into its frame, leaving the frame's old (cleared)
/// buffers in the surface slot, so neither side allocates once warm.
fn take_sei(frame: &mut SeiStore, picture: Option<&mut SeiStore>) {
    frame.clear();
    if let Some(picture) = picture {
        std::mem::swap(frame, picture);
    }
}

fn no_sequence() -> Error {
    Error::UnexpectedValue {
        call: "cuvidParseVideoData",
        what: "a picture",
        value: "before any sequence the decoder accepted".into(),
    }
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("State")
            .field("config", &self.config)
            .field("decoder", &self.decoder)
            .field("sequence", &self.sequence)
            .field("queued", &self.output.len())
            .finish_non_exhaustive()
    }
}

impl DeviceSlot {
    fn new(context: &Arc<Context>, format: PixelFormat, width: u32, height: u32) -> Result<Self> {
        let layout = Layout::pitched(format, width, height, 0);
        Ok(Self {
            buffer: DeviceBuffer::pitched(context, layout.max_width_bytes(), layout.total_rows())?,
            ready: Event::new(context)?,
            released: Event::new(context)?,
            release_pending: false,
            sei: SeiStore::default(),
        })
    }
}

/// The displayed part of a mapped NVDEC output surface.
struct Source {
    ptr: cu::CUdeviceptr,
    pitch: usize,
    /// Rows of the surface's luma plane; each further plane starts this many
    /// rows (rounded up to even) after the previous one.
    plane_rows: usize,
    area: Rect,
    format: PixelFormat,
}

enum Destination<'a> {
    Host(&'a mut Vec<u8>),
    Device(cu::CUdeviceptr),
}

impl Source {
    /// Queues the copy of every plane into `destination`, laid out as
    /// `layout`, on `stream`.
    fn copy(
        &self,
        cuda: &CudaLibrary,
        stream: &Stream,
        layout: &Layout,
        mut destination: Destination<'_>,
    ) -> Result<()> {
        let plane_stride = self.plane_rows.next_multiple_of(2);
        for (index, plane) in layout.planes().iter().enumerate() {
            let (x, y) = self.area.plane_origin(self.format, index);
            if x + plane.width_bytes > self.pitch {
                return Err(Error::CopyOutOfBounds {
                    side: "source",
                    unit: "bytes per row",
                    needed: x + plane.width_bytes,
                    available: self.pitch,
                });
            }
            let mut copy = cu::CUDA_MEMCPY2D {
                srcXInBytes: x,
                srcY: index * plane_stride + y,
                srcMemoryType: cu::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                srcDevice: self.ptr,
                srcPitch: self.pitch,
                dstPitch: plane.pitch,
                WidthInBytes: plane.width_bytes,
                Height: plane.rows,
                ..Default::default()
            };
            match &mut destination {
                Destination::Host(data) => {
                    let end = plane.offset + plane.len();
                    let Some(target) = data.get_mut(plane.offset..end) else {
                        return Err(Error::CopyOutOfBounds {
                            side: "destination",
                            unit: "bytes",
                            needed: end,
                            available: data.len(),
                        });
                    };
                    copy.dstMemoryType = cu::CUmemorytype::CU_MEMORYTYPE_HOST;
                    copy.dstHost = target.as_mut_ptr().cast();
                }
                Destination::Device(base) => {
                    copy.dstMemoryType = cu::CUmemorytype::CU_MEMORYTYPE_DEVICE;
                    copy.dstDevice = *base + plane.offset as u64;
                }
            }
            // SAFETY: the source rectangle lies inside the mapped surface
            // (the display area is within the coded size the surface was
            // created with, checked above for the row width); the destination
            // is a host slice of the plane's length, borrowed until the caller
            // synchronises the stream, or a pooled device buffer allocated for
            // this layout. The context is current and the stream live.
            unsafe { (cuda.memcpy_2d_async)(&raw const copy, stream.raw()) }
                .check("cuMemcpy2DAsync_v2")?;
        }
        Ok(())
    }
}

/// An NVDEC decoder (`cuvidCreateDecoder`), destroyed on drop.
pub(super) struct HwDecoder {
    raw: NonNull<c_void>,
    pub(super) surfaces: u32,
    format: PixelFormat,
    codec: Codec,
    bit_depth: u8,
    chroma: super::ChromaFormat,
    max_width: u32,
    max_height: u32,
    coded_width: u32,
    coded_height: u32,
    pub(super) target_height: u32,
    cuvid: &'static CuvidLibrary,
    context: Arc<Context>,
}

impl HwDecoder {
    fn create(
        cuvid: &'static CuvidLibrary,
        context: &Arc<Context>,
        lock: sys::CUvideoctxlock,
        sequence: &SequenceInfo,
        surfaces: u32,
        config: &DecoderConfig,
    ) -> Result<Self> {
        let caps =
            DecoderCaps::query(context, sequence.codec, sequence.chroma, sequence.bit_depth)?;
        let (width, height) = (sequence.coded_width, sequence.coded_height);
        caps.check(width, height, sequence.format)?;
        let (wanted_width, wanted_height) = config.max_coded_size.unwrap_or((width, height));
        let max_width = wanted_width.clamp(width, caps.max_width.max(width));
        let max_height = wanted_height.clamp(height, caps.max_height.max(height));
        let mut info = sys::CUVIDDECODECREATEINFO {
            ulWidth: width.into(),
            ulHeight: height.into(),
            ulNumDecodeSurfaces: surfaces.into(),
            CodecType: sequence.codec.raw(),
            ChromaFormat: sequence.chroma.raw(),
            ulCreationFlags: sys::cudaVideoCreateFlags::cudaVideoCreate_PreferCUVID
                .0
                .into(),
            bitDepthMinus8: u32::from(sequence.bit_depth - 8).into(),
            ulMaxWidth: max_width.into(),
            ulMaxHeight: max_height.into(),
            OutputFormat: sequence.format.surface_format(),
            DeinterlaceMode: sys::cudaVideoDeinterlaceMode::cudaVideoDeinterlaceMode_Weave,
            ulTargetWidth: width.into(),
            ulTargetHeight: height.into(),
            ulNumOutputSurfaces: config.output_surfaces.into(),
            vidLock: lock,
            ..Default::default()
        };
        (info.display_area.right, info.display_area.bottom) = area_edges(width, height)?;
        let mut raw = ptr::null_mut();
        {
            let _current = context.guard()?;
            // SAFETY: the context is current; `info` is fully initialised with
            // reserved fields zeroed, and `lock` is a live context lock of the
            // same context that outlives the decoder.
            unsafe { (cuvid.create_decoder)(&raw mut raw, &raw mut info) }
                .check_cuvid("cuvidCreateDecoder")?;
        }
        let raw = NonNull::new(raw).ok_or_else(|| Error::UnexpectedValue {
            call: "cuvidCreateDecoder",
            what: "a decoder",
            value: "null".into(),
        })?;
        Ok(Self {
            raw,
            surfaces,
            format: sequence.format,
            codec: sequence.codec,
            bit_depth: sequence.bit_depth,
            chroma: sequence.chroma,
            max_width,
            max_height,
            coded_width: width,
            coded_height: height,
            target_height: height,
            cuvid,
            context: Arc::clone(context),
        })
    }

    /// Whether `sequence` can run on this decoder, at most resized.
    fn fits(&self, sequence: &SequenceInfo, surfaces: u32) -> bool {
        self.codec == sequence.codec
            && self.chroma == sequence.chroma
            && self.bit_depth == sequence.bit_depth
            && self.format == sequence.format
            && sequence.coded_width <= self.max_width
            && sequence.coded_height <= self.max_height
            && surfaces <= self.surfaces
    }

    fn same_size(&self, sequence: &SequenceInfo) -> bool {
        (self.coded_width, self.coded_height) == (sequence.coded_width, sequence.coded_height)
    }

    /// Resizes within the maximum size (`cuvidReconfigureDecoder`).
    fn reconfigure(&mut self, sequence: &SequenceInfo) -> Result<()> {
        let Some(reconfigure) = self.cuvid.reconfigure_decoder else {
            return Err(Error::Unsupported {
                what: "resizing an NVDEC decoder",
                reason: "the driver has no cuvidReconfigureDecoder".into(),
            });
        };
        let (width, height) = (sequence.coded_width, sequence.coded_height);
        let mut info = sys::CUVIDRECONFIGUREDECODERINFO {
            ulWidth: width,
            ulHeight: height,
            ulTargetWidth: width,
            ulTargetHeight: height,
            ulNumDecodeSurfaces: self.surfaces,
            ..Default::default()
        };
        (info.display_area.right, info.display_area.bottom) = area_edges(width, height)?;
        let _current = self.context.guard()?;
        // SAFETY: a live decoder, its context current, `info` fully
        // initialised within the decoder's maximum size.
        unsafe { reconfigure(self.raw.as_ptr(), &raw mut info) }
            .check_cuvid("cuvidReconfigureDecoder")?;
        self.coded_width = width;
        self.coded_height = height;
        self.target_height = height;
        Ok(())
    }

    fn check_index(&self, index: c_int) -> Result<usize> {
        usize::try_from(index)
            .ok()
            .filter(|index| *index < self.surfaces as usize)
            .ok_or_else(|| Error::UnexpectedValue {
                call: "cuvidParseVideoData",
                what: "a picture index",
                value: format!("{index} of {} surfaces", self.surfaces),
            })
    }

    /// Maps decoded picture `index` for reading; the context must be
    /// current.
    fn map(&self, index: c_int, params: &mut sys::CUVIDPROCPARAMS) -> Result<Mapped<'_>> {
        let (mut ptr, mut pitch) = (0, 0);
        // SAFETY: a live decoder, its context current (caller), valid
        // out-pointers and processing parameters.
        unsafe {
            (self.cuvid.map_video_frame)(
                self.raw.as_ptr(),
                index,
                &raw mut ptr,
                &raw mut pitch,
                params,
            )
        }
        .check_cuvid("cuvidMapVideoFrame64")?;
        Ok(Mapped {
            decoder: self,
            ptr,
            pitch: pitch as usize,
            unmapped: false,
        })
    }

    /// The decode status of picture `index`, when the driver reports it.
    fn status(&self, index: c_int) -> DecodeStatus {
        let Some(get_status) = self.cuvid.get_decode_status else {
            return DecodeStatus::Unknown;
        };
        // SAFETY: an all-zero `CUVIDGETDECODESTATUS` is valid (integers and
        // null pointers).
        let mut status: sys::CUVIDGETDECODESTATUS = unsafe { std::mem::zeroed() };
        // SAFETY: a live decoder, its context current (caller), a valid
        // out-struct.
        let result = unsafe { get_status(self.raw.as_ptr(), index, &raw mut status) };
        if result == cu::CUresult::CUDA_SUCCESS {
            DecodeStatus::from_raw(status.decodeStatus)
        } else {
            DecodeStatus::Unknown
        }
    }
}

/// The right and bottom edges of a display area covering a whole coded
/// picture (NVDEC takes them as 16-bit integers).
fn area_edges(width: u32, height: u32) -> Result<(i16, i16)> {
    let overflow = |_| Error::SizeOverflow {
        what: "a coded size as a 16-bit display area",
    };
    Ok((
        i16::try_from(width).map_err(overflow)?,
        i16::try_from(height).map_err(overflow)?,
    ))
}

impl Drop for HwDecoder {
    fn drop(&mut self) {
        let destroyed = self.context.guard().and_then(|_current| {
            // SAFETY: a live decoder with nothing mapped (`Mapped` borrows
            // it), its context current; not used again.
            unsafe { (self.cuvid.destroy_decoder)(self.raw.as_ptr()) }
                .check_cuvid("cuvidDestroyDecoder")
        });
        if let Err(error) = destroyed {
            tracing::warn!(%error, "destroying an NVDEC decoder");
        }
    }
}

impl fmt::Debug for HwDecoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HwDecoder")
            .field("surfaces", &self.surfaces)
            .field("format", &self.format)
            .field("coded", &(self.coded_width, self.coded_height))
            .field("max", &(self.max_width, self.max_height))
            .finish_non_exhaustive()
    }
}

/// A mapped output surface, unmapped on drop.
struct Mapped<'a> {
    decoder: &'a HwDecoder,
    ptr: cu::CUdeviceptr,
    pitch: usize,
    unmapped: bool,
}

impl Mapped<'_> {
    fn unmap(mut self) -> Result<()> {
        self.unmapped = true;
        // SAFETY: this surface was mapped by `decoder`, which is live, and is
        // not used again; the context is current (caller).
        unsafe { (self.decoder.cuvid.unmap_video_frame)(self.decoder.raw.as_ptr(), self.ptr) }
            .check_cuvid("cuvidUnmapVideoFrame64")
    }
}

impl Drop for Mapped<'_> {
    fn drop(&mut self) {
        if self.unmapped {
            return;
        }
        let unmapped = self.decoder.context.guard().and_then(|_current| {
            // SAFETY: as in `unmap`, with the context pushed here.
            unsafe { (self.decoder.cuvid.unmap_video_frame)(self.decoder.raw.as_ptr(), self.ptr) }
                .check_cuvid("cuvidUnmapVideoFrame64")
        });
        if let Err(error) = unmapped {
            tracing::warn!(%error, "unmapping an NVDEC surface");
        }
    }
}

/// A cuvid context lock (`cuvidCtxLockCreate`), destroyed on drop.
pub(super) struct CtxLock {
    pub(super) raw: sys::CUvideoctxlock,
    cuvid: &'static CuvidLibrary,
}

impl CtxLock {
    pub(super) fn new(context: &Context) -> Result<Self> {
        let cuvid = super::cuvid_library()?;
        let mut raw = ptr::null_mut();
        // SAFETY: valid out-pointer and a retained context.
        unsafe { (cuvid.ctx_lock_create)(&raw mut raw, context.raw()) }
            .check_cuvid("cuvidCtxLockCreate")?;
        Ok(Self { raw, cuvid })
    }
}

impl Drop for CtxLock {
    fn drop(&mut self) {
        // SAFETY: a live lock no decoder uses any more (the decoder state is
        // dropped first); not used again.
        let destroyed =
            unsafe { (self.cuvid.ctx_lock_destroy)(self.raw) }.check_cuvid("cuvidCtxLockDestroy");
        if let Err(error) = destroyed {
            tracing::warn!(%error, "destroying an NVDEC context lock");
        }
    }
}

/// The decoder state, at a fixed heap address the parser holds as its user
/// data. A raw pointer rather than a `Box`: moving a `Box` asserts unique
/// ownership of its contents, which the parser's copy of the pointer would
/// contradict.
pub(super) struct StateBox(NonNull<State>);

impl StateBox {
    pub(super) fn new(state: State) -> Self {
        Self(NonNull::from(Box::leak(Box::new(state))))
    }

    pub(super) fn user_data(&self) -> *mut c_void {
        self.0.as_ptr().cast()
    }

    pub(super) fn get(&self) -> &State {
        // SAFETY: allocated by `new`, freed only on drop; while `&self` lives
        // no parse runs (parsing takes `&mut self` of the decoder), so no
        // callback holds a mutable reference.
        unsafe { self.0.as_ref() }
    }

    pub(super) fn get_mut(&mut self) -> &mut State {
        // SAFETY: as in `get`, exclusively borrowed.
        unsafe { self.0.as_mut() }
    }
}

impl Drop for StateBox {
    fn drop(&mut self) {
        // SAFETY: allocated by `Box` in `new` and freed only here.
        drop(unsafe { Box::from_raw(self.0.as_ptr()) });
    }
}

/// A cuvid video parser (`cuvidCreateVideoParser`), destroyed on drop.
pub(super) struct Parser {
    pub(super) raw: sys::CUvideoparser,
    cuvid: &'static CuvidLibrary,
}

impl Parser {
    /// A parser calling back into `state`.
    pub(super) fn new(state: &StateBox) -> Result<Self> {
        let config = state.get().config;
        let cuvid = super::cuvid_library()?;
        let mut params = sys::CUVIDPARSERPARAMS {
            CodecType: config.codec.raw(),
            // Raised by the sequence callback's return value.
            ulMaxNumDecodeSurfaces: 1,
            ulClockRate: config.clock_rate,
            ulErrorThreshold: 0,
            ulMaxDisplayDelay: config.max_display_delay,
            pUserData: state.user_data(),
            pfnSequenceCallback: Some(on_sequence),
            pfnDecodePicture: Some(on_decode),
            pfnDisplayPicture: Some(on_display),
            pfnGetOperatingPoint: Some(on_operating_point),
            // NVDEC API 12.0+; older drivers ignore the field.
            pfnGetSEIMsg: Some(on_sei),
            ..Default::default()
        };
        let mut raw = ptr::null_mut();
        // SAFETY: valid out-pointer and parameters; `state` stays at its
        // address until after the parser is destroyed (`Decoder` drops the
        // state first, and no parse runs during drop).
        unsafe { (cuvid.create_video_parser)(&raw mut raw, &raw mut params) }
            .check_cuvid("cuvidCreateVideoParser")?;
        Ok(Self { raw, cuvid })
    }
}

impl Drop for Parser {
    fn drop(&mut self) {
        // SAFETY: a live parser, not used again. Destroying it calls no
        // callbacks.
        let destroyed = unsafe { (self.cuvid.destroy_video_parser)(self.raw) }
            .check_cuvid("cuvidDestroyVideoParser");
        if let Err(error) = destroyed {
            tracing::warn!(%error, "destroying an NVDEC parser");
        }
    }
}

/// Runs a callback body on the state behind `user`, catching panics and
/// storing errors for the parse call to return. `failure` is what the driver
/// reads as failure.
fn callback(
    user: *mut c_void,
    name: &'static str,
    failure: c_int,
    body: impl FnOnce(&mut State) -> Result<c_int>,
) -> c_int {
    let state = user.cast::<State>();
    let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: `user` is the `StateBox` pointer the parser was created
        // with. The parser calls back only from inside `cuvidParseVideoData`,
        // on the calling thread, one callback at a time, while the decoder
        // holds no reference into the state: this is the only reference.
        let state = unsafe { &mut *state };
        body(state)
    }));
    let error = match outcome {
        Ok(Ok(value)) => return value,
        Ok(Err(error)) => error,
        Err(_) => Error::CallbackPanicked { callback: name },
    };
    tracing::debug!(%error, callback = name, "NVDEC callback failed");
    // SAFETY: as above; the reference made inside the closure is gone.
    let state = unsafe { &mut *state };
    state.error.get_or_insert(error);
    failure
}

unsafe extern "C" fn on_sequence(user: *mut c_void, format: *mut sys::CUVIDEOFORMAT) -> c_int {
    callback(user, "sequence", 0, |state| {
        // SAFETY: the parser passes a valid format for the call's duration.
        let format = unsafe { format.as_ref() }.ok_or_else(no_sequence)?;
        state.on_sequence(format)
    })
}

unsafe extern "C" fn on_decode(user: *mut c_void, params: *mut sys::CUVIDPICPARAMS) -> c_int {
    callback(user, "decode", 0, |state| {
        if params.is_null() {
            return Err(no_sequence());
        }
        state.on_decode(params)
    })
}

unsafe extern "C" fn on_display(user: *mut c_void, info: *mut sys::CUVIDPARSERDISPINFO) -> c_int {
    callback(user, "display", 0, |state| {
        // SAFETY: the parser passes valid display info for the call's
        // duration, or null for an end-of-stream notification.
        match unsafe { info.as_ref() } {
            Some(info) => state.on_display(info),
            None => Ok(1),
        }
    })
}

unsafe extern "C" fn on_sei(user: *mut c_void, info: *mut sys::CUVIDSEIMESSAGEINFO) -> c_int {
    callback(user, "SEI", 0, |state| {
        // SAFETY: the parser passes valid SEI info for the call's duration.
        match unsafe { info.as_ref() } {
            Some(info) => Ok(state.on_sei(info)),
            None => Ok(1),
        }
    })
}

unsafe extern "C" fn on_operating_point(
    user: *mut c_void,
    info: *mut sys::CUVIDOPERATINGPOINTINFO,
) -> c_int {
    // A failure here (-1) makes the parser pick operating point 0, which is
    // also what an error falls back to.
    callback(user, "operating point", -1, |state| {
        // SAFETY: the parser passes valid info for the call's duration.
        Ok(unsafe { info.as_ref() }.map_or(0, |info| state.on_operating_point(info)))
    })
}

#[cfg(test)]
mod tests {
    use crate::decode::{Colour, FrameInfo};

    #[test]
    fn colour_and_fields_are_plain_data() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<Colour>();
        send_sync::<FrameInfo>();
    }
}
