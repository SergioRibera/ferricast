//! Zero-copy import of DMA-BUFs (a Wayland compositor's or PipeWire's screen
//! buffers) into CUDA, so NVENC encodes them without the CPU touching a
//! pixel.
//!
//! A DMA-BUF becomes an EGL image (`EGL_LINUX_DMA_BUF_EXT`, with its DRM
//! format modifier) on the EGL device of the CUDA device's GPU, which CUDA
//! registers (`cuGraphicsEGLRegisterImage`) as an alias of the same memory:
//! a CUDA array for tiled (block-linear) buffers, pitched device memory for
//! LINEAR ones. No GL context is needed and nothing is copied.
//!
//! [`DmaBufImporter`] caches one registration per buffer under the caller's
//! [`BufferKey`] (ferricast's `DmaBufId`), so a pool of capture buffers is
//! imported once and every later frame costs a fence wait and a pointer
//! lookup. [`DmaBufImporter::map`] waits for the producer's fence and lends
//! a [`MappedFrame`]: a [`cuda::ArrayView`] for NVENC's
//! `NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY`, or a [`cuda::PitchedView`], and a
//! device copy into a pooled buffer ([`FramePool`]) for consumers that need
//! one. While a frame is lent the importer cannot forget the buffer: the
//! borrow checker enforces it.
//!
//! Only the CUDA device's own GPU works: [`DmaBufImporter::new`] checks the
//! producer's DRM node against it and fails with
//! [`Error::DeviceMismatch`] otherwise (hybrid laptops render on the
//! integrated GPU); the caller then copies through the CPU.
//!
//! See `design/NVCODEC.md` ("DMA-BUF interop decisions") for what was
//! measured and rejected.
//!
//! ```no_run
//! use std::os::fd::AsFd as _;
//! use std::time::Duration;
//!
//! use ferricast_nvcodec::cuda::{Context, Device};
//! use ferricast_nvcodec::interop::{
//!     BufferKey, DmaBuf, DmaBufImporter, DrmFormat, DrmNode, FenceWait, Modifier,
//! };
//!
//! # fn main() -> ferricast_nvcodec::Result<()> {
//! # let file = std::fs::File::open("/dev/null").unwrap();
//! let context = Context::primary(Device::get(0)?)?;
//! let node = DrmNode::from_path("/dev/dri/renderD128")?;
//! let mut importer = DmaBufImporter::new(&context, Some(node))?;
//! let buffer = DmaBuf {
//!     fd: file.as_fd(),
//!     width: 1920,
//!     height: 1080,
//!     format: DrmFormat::Xrgb8888,
//!     offset: 0,
//!     stride: 7680,
//!     modifier: Modifier::LINEAR,
//! };
//! let key = BufferKey(42);
//! importer.import(key, &buffer)?;
//! let frame = importer.map(key, FenceWait::Implicit(Duration::from_millis(100)))?;
//! if let Some(array) = frame.array() {
//!     // Hand `array` to NVENC as a CUDA array input.
//!     println!("{}x{} array", array.width(), array.height());
//! }
//! # Ok(())
//! # }
//! ```

mod drm;
mod egl;
mod fence;
mod pool;

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt;
use std::os::fd::{AsFd as _, AsRawFd as _, BorrowedFd, OwnedFd};
use std::ptr::{self, NonNull};
use std::sync::Arc;

pub use self::drm::DrmNode;
pub use self::fence::FenceWait;
pub use self::pool::{FramePool, PooledBuffer};

use self::egl::{Egl, EglDisplay, EglImage, EglInt};
use crate::cuda::{
    self, ArrayView, Context, DeviceBuffer, Event, PitchedView, Point, Region, Stream,
};
use crate::error::{CheckStatus as _, Error, Result};
use crate::loader::{CudaLibrary, Required};
use crate::sys::cuda as sys;

/// Bytes per pixel of every supported format.
const BYTES_PER_PIXEL: u32 = 4;

/// `CU_EGL_COLOR_FORMAT_ARGB` from NVIDIA's `cudaEGL.h` (not in the
/// generated subset): the memory order of DRM's `ARGB8888`.
const CU_EGL_COLOR_FORMAT_ARGB: u32 = 0x06;
/// `CU_EGL_COLOR_FORMAT_ABGR`: the memory order of DRM's `ABGR8888`.
const CU_EGL_COLOR_FORMAT_ABGR: u32 = 0x0E;

/// The pixel formats screen capture produces, all 32-bit packed single
/// plane. Names follow DRM (`drm_fourcc.h`): the channel order of a
/// little-endian `u32`, so `Xrgb8888` is stored as bytes B, G, R, X.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DrmFormat {
    /// `DRM_FORMAT_XRGB8888` (`XR24`).
    Xrgb8888,
    /// `DRM_FORMAT_ARGB8888` (`AR24`).
    Argb8888,
    /// `DRM_FORMAT_XBGR8888` (`XB24`).
    Xbgr8888,
    /// `DRM_FORMAT_ABGR8888` (`AB24`).
    Abgr8888,
}

impl DrmFormat {
    /// Every supported format.
    pub const ALL: [Self; 4] = [
        Self::Xrgb8888,
        Self::Argb8888,
        Self::Xbgr8888,
        Self::Abgr8888,
    ];

    /// The DRM fourcc code.
    #[must_use]
    pub const fn fourcc(self) -> u32 {
        u32::from_le_bytes(match self {
            Self::Xrgb8888 => *b"XR24",
            Self::Argb8888 => *b"AR24",
            Self::Xbgr8888 => *b"XB24",
            Self::Abgr8888 => *b"AB24",
        })
    }

    /// The format EGL is told the buffer has. NVIDIA's CUDA refuses EGL
    /// images of the X variants (`CUDA_ERROR_INVALID_VALUE`) but takes the
    /// alpha variant of the same memory layout; the X byte then reads as
    /// alpha, which NVENC's ARGB / ABGR input ignores.
    #[must_use]
    pub const fn registered_as(self) -> Self {
        match self {
            Self::Xrgb8888 | Self::Argb8888 => Self::Argb8888,
            Self::Xbgr8888 | Self::Abgr8888 => Self::Abgr8888,
        }
    }

    /// Whether the fourth byte is alpha (rather than padding).
    #[must_use]
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::Argb8888 | Self::Abgr8888)
    }

    const fn egl_color_format(self) -> u32 {
        match self.registered_as() {
            Self::Abgr8888 | Self::Xbgr8888 => CU_EGL_COLOR_FORMAT_ABGR,
            Self::Argb8888 | Self::Xrgb8888 => CU_EGL_COLOR_FORMAT_ARGB,
        }
    }
}

impl TryFrom<u32> for DrmFormat {
    type Error = Error;

    /// The format of a DRM fourcc.
    fn try_from(fourcc: u32) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|format| format.fourcc() == fourcc)
            .ok_or(Error::UnsupportedFormat { fourcc })
    }
}

/// A DRM format modifier: how the pixels are laid out in memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Modifier(pub u64);

impl Modifier {
    /// `DRM_FORMAT_MOD_LINEAR`: rows one after the other.
    pub const LINEAR: Self = Self(0);
    /// `DRM_FORMAT_MOD_INVALID`: no explicit modifier; the layout was agreed
    /// out of band (implicit modifier).
    pub const INVALID: Self = Self(0x00ff_ffff_ffff_ffff);
}

/// The caller's identity of a DMA-BUF: ferricast's `DmaBufId` (device and
/// inode of the buffer) packed into 128 bits, or any value that is stable
/// for the buffer's life and unique among live buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferKey(pub u128);

impl From<(u64, u64)> for BufferKey {
    /// From a (device, inode) pair.
    fn from((dev, ino): (u64, u64)) -> Self {
        Self((u128::from(dev) << 64) | u128::from(ino))
    }
}

/// One single-plane DMA-BUF, as the producer describes it.
#[derive(Clone, Copy, Debug)]
pub struct DmaBuf<'fd> {
    /// The buffer. Duplicated on import; the caller keeps its own.
    pub fd: BorrowedFd<'fd>,
    /// Width in pixels.
    pub width: u32,
    /// Height in rows.
    pub height: u32,
    /// Pixel format.
    pub format: DrmFormat,
    /// Byte offset of the first pixel in the buffer.
    pub offset: u32,
    /// Bytes from one row to the next (for tiled layouts, as the producer
    /// reports it).
    pub stride: u32,
    /// Layout.
    pub modifier: Modifier,
}

/// What [`DmaBufImporter::import`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Import {
    /// The buffer was registered.
    Registered,
    /// The key was already registered with the same description: nothing
    /// was done.
    Cached,
    /// The key was registered with a different description (the producer
    /// reused the identity); the old registration was replaced.
    Replaced,
}

/// The parts of a [`DmaBuf`] a registration depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Description {
    width: u32,
    height: u32,
    format: DrmFormat,
    offset: u32,
    stride: u32,
    modifier: Modifier,
}

impl From<&DmaBuf<'_>> for Description {
    fn from(buffer: &DmaBuf<'_>) -> Self {
        Self {
            width: buffer.width,
            height: buffer.height,
            format: buffer.format,
            offset: buffer.offset,
            stride: buffer.stride,
            modifier: buffer.modifier,
        }
    }
}

impl Description {
    /// Bytes per row of pixels.
    fn row_bytes(self) -> u32 {
        // `validate` bounds `width` so that this cannot overflow.
        self.width.saturating_mul(BYTES_PER_PIXEL)
    }

    /// Checks the geometry against itself and the buffer's size.
    fn validate(self, buffer_size: Option<u64>) -> Result<()> {
        let invalid = |reason: String| Err(Error::InvalidDmaBuf { reason });
        if self.width == 0 || self.height == 0 {
            return invalid(format!("{}x{} is empty", self.width, self.height));
        }
        // EGL takes sizes as signed 32-bit attributes.
        let limit = u32::try_from(i32::MAX).unwrap_or(u32::MAX) / BYTES_PER_PIXEL;
        if self.width > limit || self.height > limit {
            return invalid(format!("{}x{} is too large", self.width, self.height));
        }
        if self.stride < self.row_bytes() {
            return invalid(format!(
                "stride {} is smaller than a {}-pixel row ({} bytes)",
                self.stride,
                self.width,
                self.row_bytes()
            ));
        }
        // The smallest buffer any layout of this geometry needs (exact for
        // LINEAR; tiled layouts pad further).
        let needed = u64::from(self.offset)
            + u64::from(self.stride) * u64::from(self.height - 1)
            + u64::from(self.row_bytes());
        if let Some(size) = buffer_size
            && needed > size
        {
            return invalid(format!(
                "{}x{} at offset {} with stride {} needs {needed} bytes, the buffer has {size}",
                self.width, self.height, self.offset, self.stride
            ));
        }
        Ok(())
    }

    /// The `eglCreateImageKHR` attributes, terminated.
    fn egl_attributes(self, fd: BorrowedFd<'_>) -> Vec<EglInt> {
        let signed = |value: u32| EglInt::from_ne_bytes(value.to_ne_bytes());
        let mut attributes = vec![
            egl::EGL_WIDTH,
            signed(self.width),
            egl::EGL_HEIGHT,
            signed(self.height),
            egl::EGL_LINUX_DRM_FOURCC_EXT,
            signed(self.format.registered_as().fourcc()),
            egl::EGL_DMA_BUF_PLANE0_FD_EXT,
            fd.as_raw_fd(),
            egl::EGL_DMA_BUF_PLANE0_OFFSET_EXT,
            signed(self.offset),
            egl::EGL_DMA_BUF_PLANE0_PITCH_EXT,
            signed(self.stride),
        ];
        if self.modifier != Modifier::INVALID {
            let [low, high] = [self.modifier.0, self.modifier.0 >> 32]
                .map(|half| signed(u32::try_from(half & 0xffff_ffff).unwrap_or(0)));
            attributes.extend([
                egl::EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
                low,
                egl::EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
                high,
            ]);
        }
        attributes.push(egl::EGL_NONE);
        attributes
    }
}

/// The CUDA entry points of EGL interop, all present (they are optional in
/// the loader).
#[derive(Clone, Copy)]
struct CudaEgl {
    register: Required<sys::tcuGraphicsEGLRegisterImage>,
    mapped_frame: Required<sys::tcuGraphicsResourceGetMappedEglFrame>,
    unregister: Required<sys::tcuGraphicsUnregisterResource>,
}

impl CudaEgl {
    fn get(cuda: &CudaLibrary) -> Result<Self> {
        let missing = |symbol: &str| Error::Unsupported {
            what: "CUDA EGL interop",
            reason: format!("the CUDA driver has no {symbol}"),
        };
        Ok(Self {
            register: cuda
                .graphics_egl_register_image
                .ok_or_else(|| missing("cuGraphicsEGLRegisterImage"))?,
            mapped_frame: cuda
                .graphics_resource_get_mapped_egl_frame
                .ok_or_else(|| missing("cuGraphicsResourceGetMappedEglFrame"))?,
            unregister: cuda
                .graphics_unregister_resource
                .ok_or_else(|| missing("cuGraphicsUnregisterResource"))?,
        })
    }
}

/// Imports DMA-BUFs into one CUDA context and keeps them registered.
///
/// Bound to a CUDA [`Context`] and the EGL display of the same GPU. Every
/// registration owns what it needs (a duplicate of the DMA-BUF descriptor,
/// the EGL image, the CUDA resource) and releases it in that order's
/// reverse; the context outlives them all.
pub struct DmaBufImporter {
    registrations: HashMap<BufferKey, Registration>,
    display: EglDisplay,
    /// Importable modifiers of `Argb8888` and `Abgr8888` (what every format
    /// is registered as), from `eglQueryDmaBufModifiersEXT`.
    modifiers: [Vec<Modifier>; 2],
    interop: CudaEgl,
    context: Arc<Context>,
}

// SAFETY: the importer's handles (EGL display, CUDA resources) have no thread
// affinity: EGL and the CUDA driver API are thread-safe, and every CUDA call
// pushes the context on the calling thread. Not `Sync`: nothing needs shared
// access, and `map` returning views into registrations stays single-owner.
unsafe impl Send for DmaBufImporter {}

impl DmaBufImporter {
    /// An importer for `context`'s GPU. With `node` (the DRM device the
    /// producer renders on: `linux-dmabuf` feedback's main device, or
    /// PipeWire's), checks that it is the same GPU.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] when the CUDA driver or EGL lacks DMA-BUF
    /// interop, [`Error::LibraryMissing`] without `libEGL.so.1`,
    /// [`Error::NoEglDevice`] when no NVIDIA EGL device drives the CUDA
    /// device, [`Error::DeviceMismatch`] when `node` is another GPU,
    /// [`Error::Egl`] / [`Error::Cuda`].
    pub fn new(context: &Arc<Context>, node: Option<DrmNode>) -> Result<Self> {
        let interop = CudaEgl::get(CudaLibrary::get()?)?;
        let egl = Egl::get()?;
        let device = context.device();
        let cuda_pci = device.pci_bus_id()?;
        let devices = egl.devices()?;
        let ordinal = device.raw();
        let egl_device = devices
            .iter()
            .copied()
            .find(|candidate| candidate.cuda_device() == Some(ordinal))
            .ok_or_else(|| Error::NoEglDevice {
                cuda: cuda_pci.to_string(),
                reason: format!(
                    "none of the {} EGL devices reports it through EGL_NV_device_cuda",
                    devices.len()
                ),
            })?;
        if let Some(node) = node {
            check_same_gpu(node, egl_device, cuda_pci)?;
        }
        let display = egl_device.display()?;
        let extensions = display.extensions();
        if !egl::has_extension(&extensions, "EGL_EXT_image_dma_buf_import") {
            return Err(Error::Unsupported {
                what: "DMA-BUF import through EGL",
                reason: "the NVIDIA EGL display lacks EGL_EXT_image_dma_buf_import".into(),
            });
        }
        let modifiers = [DrmFormat::Argb8888, DrmFormat::Abgr8888].map(|format| {
            display
                .modifiers(format.fourcc())
                .map(|list| list.into_iter().map(Modifier).collect())
        });
        let [argb, abgr] = modifiers;
        let importer = Self {
            registrations: HashMap::new(),
            display,
            modifiers: [argb?, abgr?],
            interop,
            context: Arc::clone(context),
        };
        tracing::debug!(
            cuda = %cuda_pci,
            node = ?node,
            argb_modifiers = importer.modifiers[0].len(),
            abgr_modifiers = importer.modifiers[1].len(),
            "DMA-BUF importer ready"
        );
        Ok(importer)
    }

    /// The CUDA context the buffers are imported into.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// The modifiers this GPU imports for `format` (to offer the producer
    /// during format negotiation). [`Modifier::LINEAR`] is among them;
    /// [`Modifier::INVALID`] (implicit) is always accepted too.
    #[must_use]
    pub fn modifiers(&self, format: DrmFormat) -> &[Modifier] {
        &self.modifiers[usize::from(format.registered_as() == DrmFormat::Abgr8888)]
    }

    /// Registers `buffer` under `key`, or does nothing when `key` already
    /// holds the same description. A key holding another description is
    /// re-registered: the old registration is dropped first.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidDmaBuf`] for an impossible description (checked
    /// against the buffer's size), [`Error::UnsupportedModifier`],
    /// [`Error::Os`] when the descriptor cannot be duplicated,
    /// [`Error::Egl`] when EGL refuses the buffer, [`Error::Cuda`] when CUDA
    /// refuses the image, [`Error::UnexpectedValue`] when CUDA maps it in a
    /// shape this importer does not handle.
    pub fn import(&mut self, key: BufferKey, buffer: &DmaBuf<'_>) -> Result<Import> {
        let description = Description::from(buffer);
        let replaced = match self.registrations.entry(key) {
            Entry::Occupied(entry) if entry.get().description == description => {
                return Ok(Import::Cached);
            }
            Entry::Occupied(entry) => {
                entry.remove();
                true
            }
            Entry::Vacant(_) => false,
        };
        description.validate(buffer_size(buffer.fd))?;
        if description.modifier != Modifier::INVALID
            && !self
                .modifiers(description.format)
                .contains(&description.modifier)
        {
            return Err(Error::UnsupportedModifier {
                fourcc: description.format.fourcc(),
                modifier: description.modifier.0,
            });
        }
        let registration = self.register(buffer.fd, description)?;
        self.registrations.insert(key, registration);
        Ok(if replaced {
            Import::Replaced
        } else {
            Import::Registered
        })
    }

    fn register(&self, fd: BorrowedFd<'_>, description: Description) -> Result<Registration> {
        // Kept for the fence wait in `map`; EGL dups its own copy.
        let fd = fd.try_clone_to_owned().map_err(|error| Error::Os {
            call: "fcntl(F_DUPFD_CLOEXEC)",
            errno: error.raw_os_error().unwrap_or(0),
        })?;
        let image = self
            .display
            .create_dmabuf_image(&description.egl_attributes(fd.as_fd()))?;
        let resource = {
            let _current = self.context.guard()?;
            let mut raw = ptr::null_mut();
            let flags = sys::CUgraphicsRegisterFlags::CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY.0;
            // SAFETY: the context is current; `image` is a live EGL image of
            // the CUDA device's EGL display; valid out-pointer.
            unsafe { (self.interop.register)(&raw mut raw, image.raw(), flags) }
                .check("cuGraphicsEGLRegisterImage")?;
            NonNull::new(raw).ok_or(Error::UnexpectedValue {
                call: "cuGraphicsEGLRegisterImage",
                what: "a resource",
                value: "null".into(),
            })?
        };
        let registration = Registration {
            resource,
            _image: image,
            fd,
            description,
            interop: self.interop,
            context: Arc::clone(&self.context),
            synchronized: false,
        };
        // Fails early (and unregisters) when CUDA maps it in an unexpected
        // shape.
        registration.frame()?;
        Ok(registration)
    }

    /// Whether `key` is registered.
    #[must_use]
    pub fn contains(&self, key: BufferKey) -> bool {
        self.registrations.contains_key(&key)
    }

    /// How many buffers are registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.registrations.len()
    }

    /// Whether no buffer is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.registrations.is_empty()
    }

    /// Waits for the producer as `fence` says and lends the buffer under
    /// `key` to CUDA work (NVENC input, a device copy).
    ///
    /// The returned frame borrows the importer, so the buffer cannot be
    /// forgotten while it is lent. Work queued on it must finish before the
    /// producer writes the buffer again: in ferricast, keep the frame's
    /// `BufferLease` until the encoder (or the stream the copy ran on) is
    /// done with it.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownBuffer`], [`Error::FenceTimeout`], [`Error::Os`],
    /// [`Error::Cuda`].
    pub fn map(&self, key: BufferKey, fence: FenceWait) -> Result<MappedFrame<'_>> {
        let registration = self
            .registrations
            .get(&key)
            .ok_or(Error::UnknownBuffer { key: key.0 })?;
        fence.wait(registration.fd.as_fd())?;
        let memory = registration.frame()?;
        Ok(MappedFrame {
            key,
            description: registration.description,
            memory,
            context: &self.context,
        })
    }

    /// Unregisters `key` once `quiesce` says the work reading it is done;
    /// `false` when it was not registered.
    pub fn forget(&mut self, key: BufferKey, quiesce: Quiesce<'_>) -> bool {
        self.forget_all([key], quiesce) == 1
    }

    /// Unregisters every buffer (the producer's pool changed), waiting for
    /// all queued work of the context once. [`DmaBufImporter::clear_with`]
    /// waits for less. Dropping the importer does this when buffers are
    /// still registered: a user that knows the work is done clears with
    /// [`Quiesce::idle`] first, as ferricast's NVENC path does.
    pub fn clear(&mut self) {
        self.clear_with(Quiesce::CONTEXT);
    }

    /// Unregisters every buffer once `quiesce` says the work reading them
    /// is done.
    pub fn clear_with(&mut self, quiesce: Quiesce<'_>) {
        if self.registrations.is_empty() {
            return;
        }
        quiesce.wait(&self.context);
        for (_, mut registration) in self.registrations.drain() {
            registration.synchronized = true;
            drop(registration);
        }
    }

    /// Unregisters the buffers under `keys` with a single wait, as
    /// `quiesce` says; returns how many were registered.
    pub fn forget_all(
        &mut self,
        keys: impl IntoIterator<Item = BufferKey>,
        quiesce: Quiesce<'_>,
    ) -> usize {
        let removed: Vec<Registration> = keys
            .into_iter()
            .filter_map(|key| self.registrations.remove(&key))
            .collect();
        if removed.is_empty() {
            return 0;
        }
        quiesce.wait(&self.context);
        let count = removed.len();
        for mut registration in removed {
            registration.synchronized = true;
            drop(registration);
        }
        count
    }
}

/// What the importer waits for before unregistering buffers: CUDA work
/// still reading an unregistered EGL image is undefined behaviour.
#[derive(Clone, Copy, Debug)]
pub struct Quiesce<'a>(QuiesceKind<'a>);

#[derive(Clone, Copy, Debug)]
enum QuiesceKind<'a> {
    Context,
    Event(&'a Event),
    Idle,
}

impl<'a> Quiesce<'a> {
    /// All queued work of the context (`cuCtxSynchronize`). Always
    /// correct, but it stalls every session sharing the context (the
    /// process-wide primary context in ferricast).
    pub const CONTEXT: Quiesce<'static> = Quiesce(QuiesceKind::Context);

    /// The work before `event`: record it on the stream that last read the
    /// buffers.
    #[must_use]
    pub fn after(event: &'a Event) -> Self {
        Self(QuiesceKind::Event(event))
    }

    /// Nothing: the caller knows the work reading the buffers completed.
    ///
    /// # Safety
    ///
    /// No CUDA work that reads the buffers may be queued or running, e.g.
    /// every NVENC picture encoded from them has been locked (with no frames
    /// in flight, when each encode call returned).
    #[must_use]
    pub unsafe fn idle() -> Self {
        Self(QuiesceKind::Idle)
    }

    fn wait(self, context: &Context) {
        match self.0 {
            QuiesceKind::Context => synchronize(context),
            QuiesceKind::Event(event) => {
                if let Err(error) = event.synchronize() {
                    tracing::warn!(%error, "waiting for an event before unregistering DMA-BUFs");
                    synchronize(context);
                }
            }
            QuiesceKind::Idle => {}
        }
    }
}

impl Drop for DmaBufImporter {
    fn drop(&mut self) {
        self.clear();
    }
}

impl fmt::Debug for DmaBufImporter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DmaBufImporter")
            .field("registered", &self.registrations.len())
            .field("display", &self.display)
            .field("context", &self.context)
            .finish_non_exhaustive()
    }
}

/// Checks that `node` is the GPU behind `egl_device` (and the CUDA device).
fn check_same_gpu(
    node: DrmNode,
    egl_device: egl::EglDevice,
    cuda_pci: cuda::PciBusId,
) -> Result<()> {
    let egl_nodes = [
        egl::EGL_DRM_RENDER_NODE_FILE_EXT,
        egl::EGL_DRM_DEVICE_FILE_EXT,
    ]
    .into_iter()
    .filter_map(|name| egl_device.string(name))
    .filter_map(|path| DrmNode::from_path(path).ok());
    let same_node = egl_nodes.into_iter().any(|egl_node| egl_node == node);
    let node_pci = node.pci_bus_id();
    if same_node || node_pci == Some(cuda_pci) {
        return Ok(());
    }
    Err(Error::DeviceMismatch {
        drm: node.to_string(),
        drm_pci: node_pci.map_or_else(|| "unknown".into(), |pci| pci.to_string()),
        cuda: cuda_pci.to_string(),
    })
}

/// The size of the buffer behind `fd` (a DMA-BUF reports it through
/// `lseek(SEEK_END)`); `None` when the descriptor cannot seek.
fn buffer_size(fd: BorrowedFd<'_>) -> Option<u64> {
    // SAFETY: plain syscall on a borrowed descriptor; a DMA-BUF's position is
    // not used by anything (reads and writes go through mmap or the GPU).
    let end = unsafe { libc::lseek(fd.as_raw_fd(), 0, libc::SEEK_END) };
    u64::try_from(end).ok()
}

/// Waits for all work on `context` (logged, not returned: callers are
/// releasing resources and must go on).
fn synchronize(context: &Context) {
    if let Err(error) = context.synchronize() {
        tracing::warn!(%error, "waiting for CUDA work before unregistering DMA-BUFs");
    }
}

/// One imported DMA-BUF. Fields drop in order after `Drop::drop`
/// unregistered the CUDA resource: the EGL image, then the descriptor.
struct Registration {
    resource: NonNull<sys::CUgraphicsResource_st>,
    _image: EglImage,
    fd: OwnedFd,
    description: Description,
    interop: CudaEgl,
    context: Arc<Context>,
    /// Set when the importer already waited for the context.
    synchronized: bool,
}

impl Registration {
    /// The memory CUDA sees, checked against the description. EGL
    /// resources are mapped for as long as they are registered
    /// (`cuGraphicsMapResources` fails on them), so this is a lookup.
    fn frame(&self) -> Result<FrameMemory> {
        let call = "cuGraphicsResourceGetMappedEglFrame";
        let mut frame = sys::CUeglFrame::default();
        {
            let _current = self.context.guard()?;
            // SAFETY: the context is current; the resource is registered;
            // index and mip level 0 exist for a single-plane image.
            unsafe { (self.interop.mapped_frame)(&raw mut frame, self.resource.as_ptr(), 0, 0) }
                .check(call)?;
        }
        let unexpected =
            |what: &'static str, value: String| Error::UnexpectedValue { call, what, value };
        let description = self.description;
        if (frame.width, frame.height) != (description.width, description.height) {
            return Err(unexpected(
                "the image size",
                format!("{}x{}", frame.width, frame.height),
            ));
        }
        if frame.planeCount != 1
            || frame.numChannels != 4
            || frame.cuFormat != sys::CUarray_format::CU_AD_FORMAT_UNSIGNED_INT8
            || frame.eglColorFormat.0 != description.format.egl_color_format()
        {
            return Err(unexpected(
                "the image format",
                format!(
                    "{} planes, {} channels, array format {}, EGL colour format {}",
                    frame.planeCount, frame.numChannels, frame.cuFormat.0, frame.eglColorFormat.0
                ),
            ));
        }
        match frame.frameType {
            sys::CUeglFrameType::CU_EGL_FRAME_TYPE_ARRAY => {
                // SAFETY: the frame type says the union holds arrays.
                let array = unsafe { frame.frame.pArray }[0];
                if array.is_null() {
                    return Err(unexpected("an array", "null".into()));
                }
                Ok(FrameMemory::Array(array))
            }
            sys::CUeglFrameType::CU_EGL_FRAME_TYPE_PITCH => {
                // SAFETY: the frame type says the union holds pointers.
                let pointer = unsafe { frame.frame.pPitch }[0];
                let pitch = usize::try_from(frame.pitch).unwrap_or(0);
                if pointer.is_null() || pitch < description.row_bytes() as usize {
                    return Err(unexpected(
                        "pitched memory",
                        format!("{pointer:?} with pitch {pitch}"),
                    ));
                }
                Ok(FrameMemory::Pitched {
                    ptr: pointer.addr() as sys::CUdeviceptr,
                    pitch,
                })
            }
            other => Err(unexpected("a frame type", other.0.to_string())),
        }
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        // Work queued on the image (a copy, an encode) must not outlive it.
        if !self.synchronized {
            synchronize(&self.context);
        }
        let unregistered = self.context.guard().and_then(|_current| {
            // SAFETY: the resource is registered on this context and not
            // used again; the importer waited for the work using it.
            unsafe { (self.interop.unregister)(self.resource.as_ptr()) }
                .check("cuGraphicsUnregisterResource")
        });
        if let Err(error) = unregistered {
            tracing::warn!(%error, "unregistering a DMA-BUF from CUDA");
        }
    }
}

/// Where CUDA sees a registered buffer's pixels.
#[derive(Clone, Copy, Debug)]
enum FrameMemory {
    /// A CUDA array (tiled DMA-BUFs).
    Array(sys::CUarray),
    /// Pitched device memory (LINEAR DMA-BUFs).
    Pitched { ptr: sys::CUdeviceptr, pitch: usize },
}

/// A view of a mapped frame's memory.
#[derive(Clone, Copy, Debug)]
pub enum FrameView<'a> {
    /// A CUDA array: NVENC's `NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY`.
    Array(ArrayView<'a>),
    /// Pitched device memory: NVENC's `NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR`.
    Pitched(PitchedView<'a>),
}

/// An imported DMA-BUF lent to CUDA work, after the producer's fence.
///
/// Borrows the importer: the registration stays alive (and mapped) at least
/// as long as this frame. Dropping it ends the loan; there is nothing to
/// unmap (EGL resources stay mapped while registered).
pub struct MappedFrame<'a> {
    key: BufferKey,
    description: Description,
    memory: FrameMemory,
    context: &'a Arc<Context>,
}

impl<'a> MappedFrame<'a> {
    /// The buffer's key.
    #[must_use]
    pub fn key(&self) -> BufferKey {
        self.key
    }

    /// Width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.description.width
    }

    /// Height in rows.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.description.height
    }

    /// The pixel format the producer reported.
    #[must_use]
    pub fn format(&self) -> DrmFormat {
        self.description.format
    }

    /// The memory layout CUDA sees: an array for tiled buffers, pitched
    /// memory for LINEAR ones.
    #[must_use]
    pub fn view(&self) -> FrameView<'a> {
        let (width, height) = (self.description.width, self.description.height);
        match self.memory {
            FrameMemory::Array(raw) => {
                // SAFETY: the array belongs to a registration the importer
                // keeps (mapped) while `'a` borrows it, on `self.context`;
                // it is `width` × `height` (checked at registration).
                let array = unsafe { ArrayView::from_raw(raw, self.context, width, height) };
                FrameView::Array(array)
            }
            FrameMemory::Pitched { ptr, pitch } => {
                // SAFETY: as above, for pitched memory of `height` rows of at
                // least `width * 4` bytes, `pitch` apart (checked).
                let pitched = unsafe {
                    PitchedView::from_raw(
                        ptr,
                        pitch,
                        self.description.row_bytes() as usize,
                        height as usize,
                        self.context,
                    )
                };
                FrameView::Pitched(pitched)
            }
        }
    }

    /// The CUDA array, for tiled buffers.
    #[must_use]
    pub fn array(&self) -> Option<ArrayView<'a>> {
        match self.view() {
            FrameView::Array(array) => Some(array),
            FrameView::Pitched(_) => None,
        }
    }

    /// The pixels as one copy region (`width * 4` bytes × `height` rows).
    #[must_use]
    pub fn region(&self) -> Region {
        Region::new(
            self.description.row_bytes() as usize,
            self.description.height as usize,
        )
    }

    /// Queues a copy of the whole frame to the top-left of `dst` on
    /// `stream`.
    ///
    /// # Errors
    ///
    /// [`Error::CopyOutOfBounds`] when `dst` is too small, [`Error::Cuda`].
    pub fn copy_into(&self, dst: &mut DeviceBuffer, stream: &Stream) -> Result<()> {
        let origin = Point::default();
        match self.view() {
            FrameView::Array(array) => dst.copy_from_array_async(
                origin,
                array,
                BYTES_PER_PIXEL as usize,
                self.region(),
                stream,
            ),
            FrameView::Pitched(pitched) => {
                dst.copy_from_pitched_async(origin, pitched, self.region(), stream)
            }
        }
    }

    /// Queues a copy of the frame into a buffer from `pool` on `stream` and
    /// returns the buffer. The copy has run once `stream` gets past it;
    /// order work on other streams after it with an
    /// [`Event`](crate::cuda::Event).
    ///
    /// # Errors
    ///
    /// As [`FramePool::take`] and [`MappedFrame::copy_into`].
    pub fn copy_to_pool(&self, pool: &FramePool, stream: &Stream) -> Result<PooledBuffer> {
        let mut buffer = pool.take()?;
        self.copy_into(&mut buffer, stream)?;
        Ok(buffer)
    }
}

impl fmt::Debug for MappedFrame<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedFrame")
            .field("key", &self.key)
            .field("description", &self.description)
            .field("memory", &self.memory)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fourcc_codes() {
        assert_eq!(DrmFormat::Xrgb8888.fourcc(), 0x3432_5258);
        assert_eq!(DrmFormat::Argb8888.fourcc(), 0x3432_5241);
        assert_eq!(DrmFormat::Xbgr8888.fourcc(), 0x3432_4258);
        assert_eq!(DrmFormat::Abgr8888.fourcc(), 0x3432_4241);
        for format in DrmFormat::ALL {
            assert_eq!(DrmFormat::try_from(format.fourcc()).unwrap(), format);
            assert!(format.registered_as().has_alpha());
        }
        let nv12 = u32::from_le_bytes(*b"NV12");
        assert!(matches!(
            DrmFormat::try_from(nv12),
            Err(Error::UnsupportedFormat { fourcc }) if fourcc == nv12
        ));
        assert_eq!(
            Error::UnsupportedFormat { fourcc: nv12 }.to_string(),
            "DRM format NV12 (0x3231564e) is not supported for import"
        );
    }

    #[test]
    fn buffer_key_packs_dev_and_inode() {
        assert_eq!(BufferKey::from((1, 2)), BufferKey((1 << 64) | 2));
        assert_ne!(BufferKey::from((1, 2)), BufferKey::from((2, 1)));
    }

    fn description() -> Description {
        Description {
            width: 1920,
            height: 1080,
            format: DrmFormat::Xrgb8888,
            offset: 0,
            stride: 7680,
            modifier: Modifier::LINEAR,
        }
    }

    #[test]
    fn validation() {
        let exact = 7680 * 1080;
        assert!(description().validate(Some(exact)).is_ok());
        assert!(description().validate(None).is_ok());
        let error = description().validate(Some(exact - 1)).unwrap_err();
        assert!(error.to_string().contains("needs 8294400 bytes"), "{error}");
        let narrow = Description {
            stride: 7679,
            ..description()
        };
        assert!(matches!(
            narrow.validate(None),
            Err(Error::InvalidDmaBuf { .. })
        ));
        let empty = Description {
            height: 0,
            ..description()
        };
        assert!(empty.validate(None).is_err());
        let huge = Description {
            width: u32::MAX,
            stride: u32::MAX,
            ..description()
        };
        assert!(huge.validate(None).is_err());
    }

    #[test]
    fn egl_attributes() {
        // SAFETY: test-only borrowed descriptor that is never used as one.
        let fd = unsafe { BorrowedFd::borrow_raw(7) };
        let tiled = Description {
            modifier: Modifier(0x0300_0000_0060_6010),
            ..description()
        };
        let attributes = tiled.egl_attributes(fd);
        assert_eq!(
            attributes,
            [
                egl::EGL_WIDTH,
                1920,
                egl::EGL_HEIGHT,
                1080,
                egl::EGL_LINUX_DRM_FOURCC_EXT,
                // XRGB is registered as ARGB.
                0x3432_5241,
                egl::EGL_DMA_BUF_PLANE0_FD_EXT,
                7,
                egl::EGL_DMA_BUF_PLANE0_OFFSET_EXT,
                0,
                egl::EGL_DMA_BUF_PLANE0_PITCH_EXT,
                7680,
                egl::EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
                0x0060_6010,
                egl::EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
                0x0300_0000,
                egl::EGL_NONE,
            ]
        );
        let implicit = Description {
            modifier: Modifier::INVALID,
            ..description()
        };
        assert!(
            !implicit
                .egl_attributes(fd)
                .contains(&egl::EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT)
        );
    }

    #[test]
    fn importer_crosses_threads() {
        fn send<T: Send>() {}
        send::<DmaBufImporter>();
        send::<FramePool>();
        send::<PooledBuffer>();
    }
}
