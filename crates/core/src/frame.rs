use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd, RawFd};
use std::sync::Arc;

use bytes::Bytes;

use crate::error::{FerricastError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Bgra,
    Rgba,
    Nv12,
    I420,
}

impl PixelFormat {
    /// Bytes per row of a tightly packed (no padding) image `width`
    /// pixels wide. For the planar YUV formats this is the row size
    /// of the luma plane, which is what `RawFrame::stride` describes.
    #[must_use]
    pub const fn packed_row_bytes(self, width: u32) -> u32 {
        match self {
            PixelFormat::Bgra | PixelFormat::Rgba => width.saturating_mul(4),
            PixelFormat::Nv12 | PixelFormat::I420 => width,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RawFrame {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: PixelFormat,
    pub data: Bytes,
    pub timestamp_us: u64,
}

/// A frame that lives on the GPU as a DMA-BUF.
///
/// Carries everything needed to:
/// * Hand the fd straight to a GPU-aware encoder (e.g. VA-API, which
///   imports it as a `VASurface` with no host copy).
/// * Read it back to CPU memory on demand via [`importer`], for
///   encoders that only consume `RawFrame` (x264).
///
/// Cloning a `GpuFrame` is cheap — the plane's fd is an
/// `Arc`-shared owned duplicate (see [`DmaBufFd`]) and the importer
/// handle is `Arc`-shared too.
#[derive(Clone)]
pub struct GpuFrame {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: PixelFormat,
    pub timestamp_us: u64,
    pub plane: DmaBufPlane,
    /// `None` means readback isn't possible (this frame came from a
    /// path that didn't have a GPU importer attached). In practice
    /// always populated when the PipeWire DmaBuf path produces a
    /// `GpuFrame`.
    pub importer: Option<Arc<dyn DmaBufImporter>>,
}

impl std::fmt::Debug for GpuFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuFrame")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("stride", &self.stride)
            .field("format", &self.format)
            .field("timestamp_us", &self.timestamp_us)
            .field("plane", &self.plane)
            .field("has_importer", &self.importer.is_some())
            .finish()
    }
}

/// One DMA-BUF plane. Single-plane formats (BGRA / BGRx / RGBA /
/// RGBx) are the common case; multi-plane (NV12, I420) would carry
/// a `Vec<DmaBufPlane>` once those formats are wired through.
#[derive(Debug, Clone)]
pub struct DmaBufPlane {
    pub fd: DmaBufFd,
    pub offset: u32,
    pub stride: u32,
    pub modifier: u64,
    pub size: u32,
}

/// Owned, shareable handle to a DMA-BUF file descriptor.
///
/// Producers hand frames to consumers on other threads, often after
/// the producer has already returned the buffer to its pool (PipeWire
/// re-queues it as soon as the `process` callback returns). Carrying
/// the producer's *borrowed* fd number would let the consumer import
/// a closed or recycled descriptor — i.e. a different object. Each
/// frame therefore owns a `dup` of the descriptor, which keeps the
/// underlying DMA-BUF object alive until the last clone drops.
///
/// Note: owning the fd guarantees *object* lifetime, not *content*
/// stability — a producer that re-queues the buffer may let the
/// compositor render into it while a consumer still reads it.
#[derive(Debug, Clone)]
pub struct DmaBufFd(Arc<OwnedFd>);

impl DmaBufFd {
    /// Duplicate a borrowed descriptor (`F_DUPFD_CLOEXEC`) into an
    /// owned, shareable handle.
    ///
    /// # Errors
    /// Returns the OS error when the descriptor can't be duplicated
    /// (invalid fd, `EMFILE`).
    pub fn dup(fd: BorrowedFd<'_>) -> std::io::Result<Self> {
        fd.try_clone_to_owned().map(Self::from)
    }

    /// Raw descriptor number, valid for as long as `self` (or any
    /// clone of it) is alive. Callers must not close it.
    #[must_use]
    pub fn as_raw_fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }

    /// Stable identity of the underlying DMA-BUF object. Every `dup`
    /// of the same buffer shares it, unlike the fd number, so it is
    /// the right key for per-buffer import caches.
    ///
    /// # Errors
    /// Returns the OS error from `fstat`.
    pub fn buffer_id(&self) -> std::io::Result<DmaBufId> {
        let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: `self.0` is a live owned descriptor and `st` points
        // to writable storage of the right size; `fstat` fully
        // initialises it on success, and we only read it then.
        let rc = unsafe { libc::fstat(self.0.as_raw_fd(), st.as_mut_ptr()) };
        if rc != 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: `fstat` returned 0, so `st` is initialised.
        let st = unsafe { st.assume_init() };
        #[allow(
            clippy::useless_conversion,
            reason = "dev_t / ino_t widths differ across targets"
        )]
        let id = DmaBufId {
            dev: u64::from(st.st_dev),
            ino: u64::from(st.st_ino),
        };
        Ok(id)
    }
}

impl From<OwnedFd> for DmaBufFd {
    fn from(fd: OwnedFd) -> Self {
        Self(Arc::new(fd))
    }
}

impl AsFd for DmaBufFd {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

/// `(st_dev, st_ino)` of a DMA-BUF: identical for every `dup` of the
/// same buffer object. See [`DmaBufFd::buffer_id`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DmaBufId {
    pub dev: u64,
    pub ino: u64,
}

/// Object-safe trait that lets a `GpuFrame` be read back to CPU
/// memory on demand. Implemented by the Vulkan importer; held inside
/// `GpuFrame` as `Arc<dyn DmaBufImporter>`.
pub trait DmaBufImporter: Send + Sync {
    /// Copy the plane's pixels into CPU memory.
    ///
    /// Contract: the returned bytes are **tightly packed** —
    /// `height` rows of exactly
    /// [`PixelFormat::packed_row_bytes(width)`](PixelFormat::packed_row_bytes)
    /// bytes each, regardless of the DMA-BUF's own pitch
    /// (`plane.stride`). [`CapturedFrame::into_cpu`] relies on this
    /// to label the resulting `RawFrame::stride`.
    ///
    /// # Errors
    /// Implementation-defined import / copy failures.
    fn readback(
        &self,
        plane: &DmaBufPlane,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<Bytes>;
}

/// Captured frame in whichever shape the source produced it. The
/// consumer (encoder) decides what to do with it: x264 always reads
/// CPU bytes (calling [`CapturedFrame::into_cpu`] forces a readback
/// when needed); a future VA-API encoder consumes the `Gpu` variant
/// directly.
#[derive(Debug, Clone)]
pub enum CapturedFrame {
    Cpu(RawFrame),
    Gpu(GpuFrame),
}

impl CapturedFrame {
    pub fn width(&self) -> u32 {
        match self {
            CapturedFrame::Cpu(r) => r.width,
            CapturedFrame::Gpu(g) => g.width,
        }
    }

    pub fn height(&self) -> u32 {
        match self {
            CapturedFrame::Cpu(r) => r.height,
            CapturedFrame::Gpu(g) => g.height,
        }
    }

    pub fn timestamp_us(&self) -> u64 {
        match self {
            CapturedFrame::Cpu(r) => r.timestamp_us,
            CapturedFrame::Gpu(g) => g.timestamp_us,
        }
    }

    pub fn pixel_format(&self) -> PixelFormat {
        match self {
            CapturedFrame::Cpu(r) => r.format,
            CapturedFrame::Gpu(g) => g.format,
        }
    }

    /// Force the frame into a CPU-resident `RawFrame`. No-op for
    /// `Cpu`; performs a readback through the carried importer for
    /// `Gpu`. Errors out if the GPU frame has no importer attached.
    pub fn into_cpu(self) -> Result<RawFrame> {
        match self {
            CapturedFrame::Cpu(r) => Ok(r),
            CapturedFrame::Gpu(g) => {
                let importer = g.importer.as_ref().ok_or_else(|| {
                    FerricastError::Encoder(
                        "GpuFrame has no DmaBuf importer attached; can't read back to CPU".into(),
                    )
                })?;
                let bytes = importer.readback(&g.plane, g.width, g.height, g.format)?;
                Ok(RawFrame {
                    width: g.width,
                    height: g.height,
                    // `readback` returns packed rows, not the
                    // DMA-BUF pitch (`g.stride` / `g.plane.stride`).
                    stride: g.format.packed_row_bytes(g.width),
                    format: g.format,
                    data: bytes,
                    timestamp_us: g.timestamp_us,
                })
            }
        }
    }
}

impl From<RawFrame> for CapturedFrame {
    fn from(r: RawFrame) -> Self {
        CapturedFrame::Cpu(r)
    }
}

impl From<GpuFrame> for CapturedFrame {
    fn from(g: GpuFrame) -> Self {
        CapturedFrame::Gpu(g)
    }
}

#[derive(Debug, Clone)]
pub struct EncodedFrame {
    pub codec: crate::Codec,
    pub data: Bytes,
    pub timestamp_us: u64,
    pub is_keyframe: bool,
    pub duration_us: Option<u64>,
    pub pts_dts: (u64, u64),
}

#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub codec: AudioCodec,
    pub data: Bytes,
    pub timestamp_us: u64,
    pub sample_rate: u32,
    pub channels: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioCodec {
    Aac,
    Opus,
    Pcm,
    Alac,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memfd(name: &std::ffi::CStr) -> OwnedFd {
        // SAFETY: `name` is a valid NUL-terminated string.
        let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
        assert!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
        // SAFETY: `fd` was just returned by `memfd_create` and nothing else owns it.
        unsafe { <OwnedFd as std::os::fd::FromRawFd>::from_raw_fd(fd) }
    }

    #[test]
    fn dmabuf_fd_dup_outlives_source_and_keeps_identity() {
        let source = memfd(c"ferricast-dmabuf-fd-a");
        let owned = DmaBufFd::dup(source.as_fd()).expect("dup");
        assert_ne!(owned.as_raw_fd(), source.as_raw_fd());
        let source_id = DmaBufFd::dup(source.as_fd())
            .and_then(|d| d.buffer_id())
            .expect("dup + fstat");
        drop(source);
        // The duplicate stays valid after the producer closes its fd.
        assert_eq!(owned.buffer_id().expect("fstat after close"), source_id);
        let other = DmaBufFd::from(memfd(c"ferricast-dmabuf-fd-b"));
        assert_ne!(other.buffer_id().expect("fstat"), source_id);
    }

    /// Honours the `readback` contract: packed rows.
    struct PackedImporter;

    impl DmaBufImporter for PackedImporter {
        fn readback(
            &self,
            _plane: &DmaBufPlane,
            width: u32,
            height: u32,
            format: PixelFormat,
        ) -> Result<Bytes> {
            let len = format.packed_row_bytes(width) as usize * height as usize;
            Ok(Bytes::from(vec![0u8; len]))
        }
    }

    #[test]
    fn into_cpu_labels_packed_stride_not_dmabuf_pitch() {
        let (width, height, pitch) = (1366, 4, 5632);
        let frame = CapturedFrame::Gpu(GpuFrame {
            width,
            height,
            stride: pitch,
            format: PixelFormat::Bgra,
            timestamp_us: 7,
            plane: DmaBufPlane {
                fd: DmaBufFd::from(memfd(c"ferricast-into-cpu")),
                offset: 0,
                stride: pitch,
                modifier: 0,
                size: pitch * height,
            },
            importer: Some(Arc::new(PackedImporter)),
        });
        let raw = frame.into_cpu().expect("readback");
        assert_eq!(raw.stride, width * 4);
        assert_eq!(raw.data.len(), (raw.stride * raw.height) as usize);
        assert_eq!(raw.timestamp_us, 7);
    }

    #[test]
    fn packed_row_bytes_per_format() {
        assert_eq!(PixelFormat::Bgra.packed_row_bytes(10), 40);
        assert_eq!(PixelFormat::Rgba.packed_row_bytes(10), 40);
        assert_eq!(PixelFormat::Nv12.packed_row_bytes(10), 10);
        assert_eq!(PixelFormat::I420.packed_row_bytes(10), 10);
        assert_eq!(PixelFormat::Bgra.packed_row_bytes(u32::MAX), u32::MAX);
    }
}
