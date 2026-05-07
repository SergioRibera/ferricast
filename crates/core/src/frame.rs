use std::os::fd::RawFd;
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
/// Cloning a `GpuFrame` is cheap — the fd is borrowed (Vulkan keeps
/// the cached `VkImage` alive via the importer's internal cache) and
/// the importer handle is `Arc`-shared.
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
#[derive(Debug, Clone, Copy)]
pub struct DmaBufPlane {
    pub fd: RawFd,
    pub offset: u32,
    pub stride: u32,
    pub modifier: u64,
    pub size: u32,
}

/// Object-safe trait that lets a `GpuFrame` be read back to CPU
/// memory on demand. Implemented by the Vulkan importer; held inside
/// `GpuFrame` as `Arc<dyn DmaBufImporter>`.
pub trait DmaBufImporter: Send + Sync {
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
                    stride: g.stride,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodec {
    Aac,
    Opus,
    Pcm,
    Alac,
}
