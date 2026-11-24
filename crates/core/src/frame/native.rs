//! Pictures that stay in a platform's own media surface (a
//! `CVPixelBuffer`, an `AHardwareBuffer`, a D3D11 texture) between a
//! decoder or capture source and an encoder or renderer that takes it
//! directly, with no CPU copy.
//!
//! Core only carries the surface: the platform types live in the crates
//! that create them (`ferricast-codec` for the codec backends), and a
//! consumer that knows the type reaches it with
//! [`NativeFrame::surface`]. Everything else reads the picture back
//! with [`NativeFrame::into_cpu`].

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use bytes::Bytes;

use super::{FrameError, PixelFormat, RawFrame};
use crate::colour::{ColourDescription, HdrMetadata};

/// Which platform object a [`NativeSurface`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub enum SurfaceKind {
    /// Core Video `CVPixelBuffer` (IOSurface-backed), Apple platforms.
    CvPixelBuffer,
    /// NDK `AHardwareBuffer`, Android.
    HardwareBuffer,
    /// `ID3D11Texture2D` (one array slice of it), Windows.
    D3d11Texture,
}

impl fmt::Display for SurfaceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::CvPixelBuffer => "CVPixelBuffer",
            Self::HardwareBuffer => "AHardwareBuffer",
            Self::D3d11Texture => "D3D11 texture",
        })
    }
}

/// Pixels read back from a surface: rows `stride` bytes apart, laid out
/// as the surface's [`PixelFormat`] (chroma right after the luma rows).
#[derive(Debug, Clone)]
pub struct SurfacePixels {
    pub stride: u32,
    pub data: Bytes,
}

/// A platform media surface. Implementations hold a retained reference
/// to the platform object and release it on drop.
pub trait NativeSurface: Send + Sync + fmt::Debug + 'static {
    fn kind(&self) -> SurfaceKind;

    /// The layout of the surface's samples.
    fn format(&self) -> PixelFormat;

    /// Copies the picture into CPU memory.
    ///
    /// # Errors
    /// [`FrameError::Readback`] when the platform refuses to map or copy
    /// the surface.
    fn readback(&self) -> Result<SurfacePixels, FrameError>;

    /// `self`, for [`NativeFrame::surface`]'s downcast.
    fn as_any(&self) -> &dyn Any;
}

/// A picture held in a [`NativeSurface`]. Cloning shares the surface.
#[derive(Debug, Clone)]
pub struct NativeFrame {
    pub width: u32,
    pub height: u32,
    pub timestamp_us: u64,
    pub colour: Option<ColourDescription>,
    pub hdr: Option<HdrMetadata>,
    pub surface: Arc<dyn NativeSurface>,
}

impl NativeFrame {
    #[must_use]
    pub fn kind(&self) -> SurfaceKind {
        self.surface.kind()
    }

    #[must_use]
    pub fn format(&self) -> PixelFormat {
        self.surface.format()
    }

    /// The surface as its concrete type, when it is a `T`.
    #[must_use]
    pub fn surface<T: NativeSurface>(&self) -> Option<&T> {
        self.surface.as_any().downcast_ref()
    }

    /// Reads the picture back into a CPU-resident [`RawFrame`].
    ///
    /// # Errors
    /// The surface's [`NativeSurface::readback`] failure.
    pub fn into_cpu(self) -> Result<RawFrame, FrameError> {
        let pixels = self.surface.readback()?;
        Ok(RawFrame {
            width: self.width,
            height: self.height,
            stride: pixels.stride,
            format: self.surface.format(),
            data: pixels.data,
            timestamp_us: self.timestamp_us,
            colour: self.colour,
            hdr: self.hdr,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Fake;

    impl NativeSurface for Fake {
        fn kind(&self) -> SurfaceKind {
            SurfaceKind::HardwareBuffer
        }
        fn format(&self) -> PixelFormat {
            PixelFormat::Nv12
        }
        fn readback(&self) -> Result<SurfacePixels, FrameError> {
            Ok(SurfacePixels {
                stride: 2,
                data: Bytes::from_static(&[1, 2, 3, 4, 5, 6]),
            })
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn a_native_frame_downcasts_and_reads_back() {
        let frame = NativeFrame {
            width: 2,
            height: 2,
            timestamp_us: 7,
            colour: None,
            hdr: None,
            surface: Arc::new(Fake),
        };
        assert!(frame.surface::<Fake>().is_some());
        let raw = frame.into_cpu().unwrap();
        assert_eq!(
            (raw.stride, raw.format, raw.timestamp_us),
            (2, PixelFormat::Nv12, 7)
        );
        assert_eq!(raw.data.len(), 6);
    }
}
