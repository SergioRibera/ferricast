//! GPU image processing for pictures that do not match the encoder: scale,
//! crop, rotate / flip, letterbox, BGRA / RGBA → NV12 and P016 → NV12,
//! without leaving the GPU.
//!
//! The device code (`kernels/kernels.cu`) ships as committed PTX that the
//! driver JIT-compiles once per context ([`Kernels::get`]); building the
//! crate needs no CUDA toolkit. A [`Converter`] queues one kernel per output
//! plane on a caller's stream, into a pooled buffer ([`FramePool`]) that
//! NVENC takes with [`Encoder::encode_device`](crate::encode::Encoder::encode_device):
//!
//! ```no_run
//! use std::sync::Arc;
//! use ferricast_nvcodec::cuda::{Context, Device, Event, Stream};
//! use ferricast_nvcodec::kernels::{
//!     Converter, FramePool, Geometry, Input, Output, OutputFormat, RgbOrder, Size,
//! };
//!
//! # fn main() -> ferricast_nvcodec::Result<()> {
//! let context = Context::primary(Device::get(0)?)?;
//! let stream = Stream::new(&context)?;
//! let mut converter = Converter::new(&context)?;
//! let format = OutputFormat::Rgb(RgbOrder::Bgra);
//! let mut pool = FramePool::new(&context, format, Size::new(1280, 720), 3);
//! let source = ferricast_nvcodec::cuda::DeviceBuffer::pitched(&context, 1920 * 4, 1080)?;
//! let target = pool.acquire()?;
//! converter.convert(
//!     Input::Packed { view: source.as_view(), order: RgbOrder::Bgra, size: Size::new(1920, 1080) },
//!     Output { buffer: &target, format, chroma_row: None },
//!     &Geometry::new(Size::new(1920, 1080), Size::new(1280, 720)),
//!     &stream,
//! )?;
//! let done = Event::new(&context)?;
//! done.record(&stream)?; // `ready` for NVENC's `DeviceFrame`
//! # Ok(())
//! # }
//! ```
//!
//! Ordering is the caller's, with events: the stream must wait for the
//! input's producer (NVDEC's `ready`, a DMA-BUF fence), and the consumer of
//! the output for the stream. Nothing here waits on the host.

mod colour;
mod geometry;
mod launch;

use std::collections::VecDeque;
use std::sync::Arc;

pub use colour::{Matrix, Range, YuvColour};
pub use geometry::{Filter, Geometry, Orientation, Rect, Rotation, Size};
pub use launch::Kernels;
use launch::{
    Channels, Nv12Params, PlaneKind, PlaneParams, PlaneRef, RgbParams, RgbSource, SourceRef,
};

use crate::cuda::{ArrayView, Context, DeviceBuffer, Event, PitchedView, Stream, TextureObject};
use crate::error::{Error, Result};
use crate::sys::cuda as sys;

/// The byte order of packed 8-bit RGB pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RgbOrder {
    /// B, G, R, A in memory (DRM `ARGB8888`, NVENC `ARGB`).
    #[default]
    Bgra,
    /// R, G, B, A in memory (DRM `ABGR8888`, NVENC `ABGR`).
    Rgba,
}

/// A picture the kernels read.
#[derive(Clone, Copy, Debug)]
pub enum Input<'a> {
    /// Packed 8-bit RGB in a CUDA array of 4 × u8 elements (a tiled
    /// DMA-BUF imported through EGL), read through a texture.
    Array {
        /// The array; its size is the picture's.
        array: ArrayView<'a>,
        /// Byte order.
        order: RgbOrder,
    },
    /// Packed 8-bit RGB in pitched memory (a LINEAR DMA-BUF, an uploaded
    /// frame), 4-byte aligned.
    Packed {
        /// The memory.
        view: PitchedView<'a>,
        /// Byte order.
        order: RgbOrder,
        /// Picture size.
        size: Size,
    },
    /// NV12: luma from row 0, interleaved chroma from `chroma_row`, both at
    /// the view's pitch (NVDEC's 8-bit output).
    Nv12 {
        /// The memory.
        view: PitchedView<'a>,
        /// First chroma row.
        chroma_row: usize,
        /// Picture size.
        size: Size,
    },
    /// P016 (P010 included): as [`Input::Nv12`] with 16-bit little-endian
    /// samples, the value in the high bits (NVDEC's output above 8 bits).
    P016 {
        /// The memory.
        view: PitchedView<'a>,
        /// First chroma row.
        chroma_row: usize,
        /// Picture size.
        size: Size,
    },
}

impl Input<'_> {
    fn size(&self) -> Size {
        match *self {
            Self::Array { array, .. } => Size::new(array.width(), array.height()),
            Self::Packed { size, .. } | Self::Nv12 { size, .. } | Self::P016 { size, .. } => size,
        }
    }

    fn context(&self) -> &Arc<Context> {
        match self {
            Self::Array { array, .. } => array.context(),
            Self::Packed { view, .. } | Self::Nv12 { view, .. } | Self::P016 { view, .. } => {
                view.context()
            }
        }
    }

    fn describe(&self) -> &'static str {
        match self {
            Self::Array { .. } => "RGB (CUDA array)",
            Self::Packed { .. } => "RGB (pitched)",
            Self::Nv12 { .. } => "NV12",
            Self::P016 { .. } => "P016",
        }
    }
}

/// The layout the kernels write.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OutputFormat {
    /// Packed 8-bit RGB (from RGB input), letterboxed with opaque black.
    Rgb(RgbOrder),
    /// NV12 (from RGB, NV12 or P016 input): RGB is converted with the
    /// colour; the letterbox is black in its range. P016 is narrowed to
    /// 8 bits with rounding.
    Nv12(YuvColour),
    /// P010 (from P016 input): 10 significant bits in the high bits of each
    /// 16-bit sample.
    P010(YuvColour),
}

impl OutputFormat {
    /// Bytes per row and rows of a `size` picture with the chroma right
    /// after the luma: what [`FramePool`] allocates and NVENC reads.
    #[must_use]
    pub const fn buffer_layout(self, size: Size) -> (usize, usize) {
        let (width, height) = (size.width as usize, size.height as usize);
        let chroma_rows = size.height.div_ceil(2) as usize;
        let pairs = size.width.div_ceil(2) as usize;
        match self {
            Self::Rgb(_) => (width * 4, height),
            Self::Nv12(_) => (pairs * 2, height + chroma_rows),
            Self::P010(_) => (pairs * 4, height + chroma_rows),
        }
    }

    const fn describe(self) -> &'static str {
        match self {
            Self::Rgb(_) => "RGB",
            Self::Nv12(_) => "NV12",
            Self::P010(_) => "P010",
        }
    }
}

/// Where the kernels write: a device buffer of the converter's context.
///
/// The kernels write through `&DeviceBuffer` (device memory is not guarded
/// by Rust's aliasing rules; see [`DeviceBuffer`]): whoever else reads or
/// writes the buffer must be ordered with the stream by events.
#[derive(Clone, Copy, Debug)]
pub struct Output<'a> {
    /// The buffer; the canvas starts at row 0.
    pub buffer: &'a DeviceBuffer,
    /// The layout.
    pub format: OutputFormat,
    /// First chroma row of a YUV format; `None` is right after the luma.
    pub chroma_row: Option<usize>,
}

/// Queues image kernels. Owns nothing per frame: texture objects for array
/// input are destroyed once the kernel that read them has run (checked
/// without waiting at the next call).
#[derive(Debug)]
pub struct Converter {
    kernels: Arc<Kernels>,
    /// Textures still read by queued kernels, with the event recorded after
    /// the last one.
    retired: VecDeque<(TextureObject, Event)>,
    /// Completed events, reused.
    spare_events: Vec<Event>,
}

/// More textures in flight than any pipeline keeps: the converter waits for
/// the oldest before queueing more.
const MAX_RETIRED: usize = 16;

impl Converter {
    /// A converter on `context` (loads the kernels on first use).
    ///
    /// # Errors
    ///
    /// As [`Kernels::get`].
    pub fn new(context: &Arc<Context>) -> Result<Self> {
        Ok(Self {
            kernels: Kernels::get(context)?,
            retired: VecDeque::new(),
            spare_events: Vec::new(),
        })
    }

    /// The kernels' context.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.kernels.context()
    }

    /// Queues the conversion of `input` into `output` on `stream`, as
    /// `geometry` describes (its canvas is the output picture's size).
    ///
    /// Conversions: RGB → RGB (either order), RGB → NV12, NV12 → NV12,
    /// P016 → NV12, P016 → P010.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidKernelInput`] for an unsupported conversion, a
    /// geometry outside either picture, a buffer too small or misaligned,
    /// an array that is not 4 × u8, or objects of another context;
    /// [`Error::Cuda`].
    pub fn convert(
        &mut self,
        input: Input<'_>,
        output: Output<'_>,
        geometry: &Geometry,
        stream: &Stream,
    ) -> Result<()> {
        let context = self.kernels.context();
        for (what, other) in [
            ("the input", input.context()),
            ("the output buffer", output.buffer.context()),
            ("the stream", stream.context()),
        ] {
            if other.raw() != context.raw() {
                return Err(invalid(format!("{what} belongs to another CUDA context")));
            }
        }
        self.reap()?;
        let source = input.size();
        let lowered = geometry.lower(source)?;
        let target = Target::check(output, geometry.canvas)?;
        let yuv = |view, chroma_row, size, sample_bytes, kind, colour| {
            let planes = SemiPlanar::check(view, chroma_row, size, sample_bytes)?;
            self.planes(kind, &planes, &target, geometry, colour, stream)
        };
        match (input, output.format) {
            (
                Input::Array { .. } | Input::Packed { .. },
                format @ (OutputFormat::Rgb(_) | OutputFormat::Nv12(_)),
            ) => self.rgb(input, format, lowered, &target, stream),
            (
                Input::Nv12 {
                    view,
                    chroma_row,
                    size,
                },
                OutputFormat::Nv12(colour),
            ) => yuv(view, chroma_row, size, 1, PlaneKind::U8ToU8, colour),
            (
                Input::P016 {
                    view,
                    chroma_row,
                    size,
                },
                OutputFormat::Nv12(colour),
            ) => yuv(view, chroma_row, size, 2, PlaneKind::U16ToU8, colour),
            (
                Input::P016 {
                    view,
                    chroma_row,
                    size,
                },
                OutputFormat::P010(colour),
            ) => yuv(view, chroma_row, size, 2, PlaneKind::U16ToU16, colour),
            (
                Input::Array { .. } | Input::Packed { .. } | Input::Nv12 { .. },
                OutputFormat::P010(_),
            )
            | (Input::Nv12 { .. } | Input::P016 { .. }, OutputFormat::Rgb(_)) => {
                Err(invalid(format!(
                    "no conversion from {} to {}",
                    input.describe(),
                    output.format.describe()
                )))
            }
        }
    }

    /// RGB input into RGB or NV12.
    fn rgb(
        &mut self,
        input: Input<'_>,
        format: OutputFormat,
        geometry: geometry::Lowered,
        target: &Target,
        stream: &Stream,
    ) -> Result<()> {
        let (source, src, order, texture) = match input {
            Input::Array { array, order } => {
                let texture = Self::texture(array)?;
                let src = SourceRef {
                    handle: texture.raw(),
                    pitch: 0,
                };
                (RgbSource::Array, src, order, Some(texture))
            }
            Input::Packed { view, order, size } => {
                (RgbSource::Pitched, packed_source(view, size)?, order, None)
            }
            Input::Nv12 { .. } | Input::P016 { .. } => {
                return Err(invalid(format!("{} is not RGB", input.describe())));
            }
        };
        // SAFETY (both arms): the source is a live texture (kept until the
        // kernel has run, below), which cannot read outside its array, or
        // pitched memory `packed_source` checked to hold the whole picture,
        // aligned, with `lower` having checked the crop inside it;
        // `Target::check` sized and aligned the destination for the canvas.
        let queued = match format {
            OutputFormat::Rgb(out) => {
                let params = rgb_params(geometry, src, target.luma, order, out);
                // SAFETY: see above.
                unsafe { self.kernels.rgb_to_rgb(source, &params, stream) }
            }
            OutputFormat::Nv12(colour) => {
                let params = nv12_params(geometry, src, target, order, colour);
                // SAFETY: see above.
                unsafe { self.kernels.rgb_to_nv12(source, &params, stream) }
            }
            OutputFormat::P010(_) => Err(invalid("no conversion from RGB to P010".into())),
        };
        if let Some(texture) = texture {
            self.retire(texture, stream)?;
        }
        queued
    }

    /// Luma then chroma of a semi-planar picture.
    fn planes(
        &self,
        kind: PlaneKind,
        source: &SemiPlanar,
        target: &Target,
        geometry: &Geometry,
        colour: YuvColour,
        stream: &Stream,
    ) -> Result<()> {
        let chroma_geometry = geometry.chroma_420()?;
        let luma = geometry.lower(source.size)?;
        let chroma = chroma_geometry.lower(source.size.chroma_420())?;
        let (out_scale, out_max, out_shift, bits) = match kind {
            PlaneKind::U8ToU8 => (1.0, 255, 0, 8),
            // Keep 10 significant bits (P010).
            PlaneKind::U16ToU16 => (1.0 / 64.0, 1023, 6, 16),
            // Round to the top 8 bits.
            PlaneKind::U16ToU8 => (1.0 / 256.0, 255, 0, 8),
        };
        let (black_y, black_uv) = colour.black(bits);
        let plane = |geometry, src, dst, fill0, fill1| PlaneParams {
            geometry,
            src,
            dst,
            out_scale,
            out_max,
            out_shift,
            fill0,
            fill1,
            reserved: 0,
        };
        let luma = plane(luma, source.luma, target.luma, black_y, 0);
        let chroma = plane(chroma, source.chroma, target.chroma, black_uv, black_uv);
        // SAFETY: `SemiPlanar::check` checked that both source planes hold
        // the whole picture at their sample size and alignment, `lower` that
        // the crops lie inside them, and `Target::check` sized and aligned
        // both destination planes for the canvas.
        unsafe {
            self.kernels.plane(kind, Channels::One, &luma, stream)?;
            self.kernels.plane(kind, Channels::Two, &chroma, stream)
        }
    }

    fn texture(array: ArrayView<'_>) -> Result<TextureObject> {
        let descriptor = array.descriptor()?;
        if descriptor.format != sys::CUarray_format::CU_AD_FORMAT_UNSIGNED_INT8
            || descriptor.channels != 4
        {
            return Err(invalid(format!(
                "the CUDA array holds {} channel(s) of format {:?}, not 4 x u8",
                descriptor.channels, descriptor.format
            )));
        }
        if descriptor.width != array.width() as usize
            || descriptor.height != array.height() as usize
        {
            return Err(invalid(format!(
                "the CUDA array is {}x{}, its view says {}x{}",
                descriptor.width,
                descriptor.height,
                array.width(),
                array.height()
            )));
        }
        TextureObject::point_sampled(array)
    }

    /// Keeps `texture` until the work queued on `stream` so far has run.
    fn retire(&mut self, texture: TextureObject, stream: &Stream) -> Result<()> {
        let event = match self.spare_events.pop() {
            Some(event) => event,
            None => Event::new(self.kernels.context())?,
        };
        let recorded = event.record(stream);
        if let Err(error) = recorded {
            // Without the event there is no telling when the kernel is
            // done: wait for it before the texture goes.
            stream.synchronize()?;
            return Err(error);
        }
        self.retired.push_back((texture, event));
        if self.retired.len() > MAX_RETIRED
            && let Some((texture, event)) = self.retired.pop_front()
        {
            event.synchronize()?;
            drop(texture);
            self.spare_events.push(event);
        }
        Ok(())
    }

    /// Destroys the textures whose kernels have run.
    fn reap(&mut self) -> Result<()> {
        while let Some((_, event)) = self.retired.front() {
            if !event.is_complete()? {
                break;
            }
            if let Some((texture, event)) = self.retired.pop_front() {
                drop(texture);
                self.spare_events.push(event);
            }
        }
        Ok(())
    }
}

impl Drop for Converter {
    fn drop(&mut self) {
        for (texture, event) in self.retired.drain(..) {
            if let Err(error) = event.synchronize() {
                tracing::warn!(%error, "waiting for an image kernel before freeing its texture");
            }
            drop(texture);
        }
    }
}

fn invalid(reason: String) -> Error {
    Error::InvalidKernelInput { reason }
}

fn rgb_params(
    geometry: geometry::Lowered,
    src: SourceRef,
    dst: PlaneRef,
    input: RgbOrder,
    output: RgbOrder,
) -> RgbParams {
    RgbParams {
        geometry,
        src,
        dst,
        swap_rb: i32::from(input != output),
        reserved: 0,
    }
}

fn nv12_params(
    geometry: geometry::Lowered,
    src: SourceRef,
    target: &Target,
    order: RgbOrder,
    colour: YuvColour,
) -> Nv12Params {
    let coefficients = colour.coefficients();
    Nv12Params {
        geometry,
        src,
        luma: target.luma,
        chroma: target.chroma,
        coeff: coefficients.matrix,
        bias_y: coefficients.bias_y,
        bias_uv: coefficients.bias_uv,
        src_rgba: i32::from(order == RgbOrder::Rgba),
        reserved: [0; 2],
    }
}

/// Packed RGB input, checked to hold `size` pixels at 4-byte alignment.
fn packed_source(view: PitchedView<'_>, size: Size) -> Result<SourceRef> {
    check_plane("the RGB input", view.device_ptr(), view.pitch(), 4)?;
    let row_bytes = size.width as usize * 4;
    if row_bytes > view.width_bytes() || size.height as usize > view.rows() {
        return Err(invalid(format!(
            "a {}x{} RGB picture needs {row_bytes} bytes x {} rows, the memory has {} x {}",
            size.width,
            size.height,
            size.height,
            view.width_bytes(),
            view.rows()
        )));
    }
    Ok(SourceRef {
        handle: view.device_ptr(),
        pitch: view.pitch() as u64,
    })
}

/// Checks that a plane at `ptr` with rows `pitch` apart can be accessed in
/// `align`-byte units.
fn check_plane(what: &str, ptr: sys::CUdeviceptr, pitch: usize, align: usize) -> Result<()> {
    if !ptr.is_multiple_of(align as u64) || !pitch.is_multiple_of(align) {
        return Err(invalid(format!(
            "{what} at {ptr:#x} with pitch {pitch} is not {align}-byte aligned"
        )));
    }
    Ok(())
}

/// A semi-planar 4:2:0 input, checked.
struct SemiPlanar {
    size: Size,
    luma: PlaneRef,
    chroma: PlaneRef,
}

impl SemiPlanar {
    fn check(
        view: PitchedView<'_>,
        chroma_row: usize,
        size: Size,
        sample_bytes: usize,
    ) -> Result<Self> {
        let chroma = size.chroma_420();
        let luma_bytes = size.width as usize * sample_bytes;
        let chroma_bytes = chroma.width as usize * 2 * sample_bytes;
        let height = size.height as usize;
        let needed_rows = chroma_row.checked_add(chroma.height as usize);
        let fits = chroma_row >= height
            && needed_rows.is_some_and(|rows| rows <= view.rows())
            && luma_bytes.max(chroma_bytes) <= view.width_bytes();
        if !fits {
            return Err(invalid(format!(
                "a {}x{} semi-planar picture with chroma from row {chroma_row} does not fit {} bytes x {} rows",
                size.width,
                size.height,
                view.width_bytes(),
                view.rows()
            )));
        }
        check_plane(
            "the YUV input",
            view.device_ptr(),
            view.pitch(),
            sample_bytes,
        )?;
        let pitch = view.pitch() as u64;
        Ok(Self {
            size,
            luma: PlaneRef {
                ptr: view.device_ptr(),
                pitch,
            },
            chroma: PlaneRef {
                ptr: view.device_ptr() + chroma_row as u64 * pitch,
                pitch,
            },
        })
    }
}

/// The output buffer, checked for the canvas.
struct Target {
    luma: PlaneRef,
    chroma: PlaneRef,
}

impl Target {
    fn check(output: Output<'_>, canvas: Size) -> Result<Self> {
        let buffer = output.buffer;
        let (width_bytes, rows) = output.format.buffer_layout(canvas);
        let height = canvas.height as usize;
        let (sample_bytes, chroma_row) = match output.format {
            OutputFormat::Rgb(_) => (4, None),
            OutputFormat::Nv12(_) => (1, Some(output.chroma_row.unwrap_or(height))),
            OutputFormat::P010(_) => (2, Some(output.chroma_row.unwrap_or(height))),
        };
        let needed_rows = match chroma_row {
            None => Some(rows),
            Some(row) if row >= height => row.checked_add(rows - height),
            Some(row) => {
                return Err(invalid(format!(
                    "chroma row {row} overlaps the {height} luma rows"
                )));
            }
        };
        if width_bytes > buffer.width_bytes()
            || needed_rows.is_none_or(|needed| needed > buffer.rows())
        {
            return Err(invalid(format!(
                "a {}x{} {} picture needs {width_bytes} bytes x {} rows, the buffer has {} x {}",
                canvas.width,
                canvas.height,
                output.format.describe(),
                needed_rows.unwrap_or(usize::MAX),
                buffer.width_bytes(),
                buffer.rows()
            )));
        }
        check_plane(
            "the output",
            buffer.device_ptr(),
            buffer.pitch(),
            sample_bytes,
        )?;
        let pitch = buffer.pitch() as u64;
        let luma = PlaneRef {
            ptr: buffer.device_ptr(),
            pitch,
        };
        let chroma = PlaneRef {
            ptr: buffer.device_ptr() + chroma_row.unwrap_or(0) as u64 * pitch,
            pitch,
        };
        Ok(Self { luma, chroma })
    }
}

/// Output buffers for one format and size, reused: a buffer is handed out
/// again once only the pool holds it. Allocates up to `capacity` buffers,
/// the first time each is needed.
///
/// A consumer that reads a buffer asynchronously (NVENC) must hold its `Arc`
/// until that read has completed; NVENC's encoder does (a pending picture
/// holds its buffer until the bitstream is locked).
#[derive(Debug)]
pub struct FramePool {
    context: Arc<Context>,
    format: OutputFormat,
    size: Size,
    capacity: usize,
    buffers: Vec<Arc<DeviceBuffer>>,
}

impl FramePool {
    /// An empty pool of `format` `size` buffers.
    #[must_use]
    pub fn new(context: &Arc<Context>, format: OutputFormat, size: Size, capacity: usize) -> Self {
        Self {
            context: Arc::clone(context),
            format,
            size,
            capacity: capacity.max(1),
            buffers: Vec::new(),
        }
    }

    /// The format the buffers are laid out for.
    #[must_use]
    pub fn format(&self) -> OutputFormat {
        self.format
    }

    /// The picture size the buffers hold.
    #[must_use]
    pub fn size(&self) -> Size {
        self.size
    }

    /// A buffer nobody else holds.
    ///
    /// # Errors
    ///
    /// [`Error::FramesInFlight`] when all `capacity` buffers are held,
    /// [`Error::Cuda`] when allocating fails.
    pub fn acquire(&mut self) -> Result<Arc<DeviceBuffer>> {
        if let Some(free) = self
            .buffers
            .iter()
            .find(|buffer| Arc::strong_count(buffer) == 1)
        {
            return Ok(Arc::clone(free));
        }
        if self.buffers.len() >= self.capacity {
            return Err(Error::FramesInFlight {
                capacity: self.capacity,
            });
        }
        let (width_bytes, rows) = self.format.buffer_layout(self.size);
        let buffer = Arc::new(DeviceBuffer::pitched(&self.context, width_bytes, rows)?);
        self.buffers.push(Arc::clone(&buffer));
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts() {
        let size = Size::new(5, 3);
        assert_eq!(
            OutputFormat::Rgb(RgbOrder::Bgra).buffer_layout(size),
            (20, 3)
        );
        assert_eq!(
            OutputFormat::Nv12(YuvColour::default()).buffer_layout(size),
            (6, 5)
        );
        assert_eq!(
            OutputFormat::P010(YuvColour::default()).buffer_layout(size),
            (12, 5)
        );
    }
}
