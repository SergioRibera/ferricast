//! Core Video pixel buffers as native frames.
//!
//! The `VideoToolbox` decoders hand their output buffers out as
//! [`PixelBuffer`]s (`CapturedFrame::Native`) and the `VideoToolbox`
//! encoders take them straight back, so a decode → encode or
//! decode → Metal path never copies the picture through CPU memory.
//! Anything else reads the picture back with
//! [`ferricast_core::NativeFrame::into_cpu`].

use std::any::Any;
use std::fmt;
use std::ptr;

use bytes::Bytes;
use ferricast_core::{
    ChromaLocation, ColourDescription, ColourRange, ContentLightLevel, FrameError, HdrMetadata,
    MasteringDisplay, NativeSurface, PixelFormat, SurfaceKind, SurfacePixels,
};
use objc2_core_foundation::{CFData, CFRetained, CFString};
use objc2_core_video::{
    CVColorPrimariesGetIntegerCodePointForString, CVPixelBuffer,
    CVPixelBufferGetBaseAddressOfPlane, CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferGetHeight,
    CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress,
    CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
    CVTransferFunctionGetIntegerCodePointForString, CVYCbCrMatrixGetIntegerCodePointForString,
    kCVImageBufferChromaLocation_Bottom, kCVImageBufferChromaLocation_BottomLeft,
    kCVImageBufferChromaLocation_Center, kCVImageBufferChromaLocation_Left,
    kCVImageBufferChromaLocation_Top, kCVImageBufferChromaLocation_TopLeft,
    kCVImageBufferChromaLocationTopFieldKey, kCVImageBufferColorPrimariesKey,
    kCVImageBufferContentLightLevelInfoKey, kCVImageBufferMasteringDisplayColorVolumeKey,
    kCVImageBufferTransferFunctionKey, kCVImageBufferYCbCrMatrixKey,
    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
    kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
    kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
    kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
};

/// The Core Video pixel format of a biplanar 4:2:0 buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BiPlanarFormat {
    /// NV12 (8-bit) or P010 (10-bit).
    pub format: PixelFormat,
    /// Video (limited) or full range.
    pub range: ColourRange,
}

/// The biplanar 4:2:0 formats, by Core Video `OSType`.
const BIPLANAR: [(u32, PixelFormat, ColourRange); 4] = [
    (
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        PixelFormat::Nv12,
        ColourRange::Limited,
    ),
    (
        kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        PixelFormat::Nv12,
        ColourRange::Full,
    ),
    (
        kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
        PixelFormat::P010,
        ColourRange::Limited,
    ),
    (
        kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
        PixelFormat::P010,
        ColourRange::Full,
    ),
];

impl BiPlanarFormat {
    /// The format a Core Video `OSType` names; `None` for anything but
    /// `420v`, `420f`, `x420` and `xf20`.
    #[must_use]
    pub fn from_os_type(code: u32) -> Option<Self> {
        BIPLANAR
            .iter()
            .find(|(os_type, _, _)| *os_type == code)
            .map(|&(_, format, range)| Self { format, range })
    }

    /// The Core Video `OSType` of this format; `None` for a pixel format
    /// that is not biplanar 4:2:0.
    #[must_use]
    pub fn os_type(self) -> Option<u32> {
        BIPLANAR
            .iter()
            .find(|(_, format, range)| *format == self.format && *range == self.range)
            .map(|&(os_type, _, _)| os_type)
    }

    /// Bytes per sample.
    fn sample_bytes(self) -> usize {
        match self.format {
            PixelFormat::P010 => 2,
            PixelFormat::Nv12 | PixelFormat::Bgra | PixelFormat::Rgba | PixelFormat::I420 => 1,
        }
    }
}

/// A Core Video pixel format that is not biplanar 4:2:0 NV12 / P010.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsupportedPixelBuffer(pub u32);

impl fmt::Display for UnsupportedPixelBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = self.0.to_be_bytes();
        if bytes.iter().all(u8::is_ascii_graphic) {
            let fourcc: String = bytes.iter().map(|&byte| char::from(byte)).collect();
            write!(f, "pixel format '{fourcc}' is not NV12 or P010")
        } else {
            write!(f, "pixel format {:#x} is not NV12 or P010", self.0)
        }
    }
}

impl std::error::Error for UnsupportedPixelBuffer {}

/// A retained NV12 or P010 `CVPixelBuffer` (IOSurface-backed when its
/// producer asked for it), shared as a [`NativeSurface`].
pub struct PixelBuffer {
    buffer: CFRetained<CVPixelBuffer>,
    format: BiPlanarFormat,
}

// SAFETY: Core Video buffers are reference-counted CF objects whose
// retain / release are thread-safe. This type never writes to the
// buffer: it only reads its geometry and attachments, and maps it with
// a read-only base-address lock, which Core Video counts per lock and
// allows from several threads at once.
unsafe impl Send for PixelBuffer {}
// SAFETY: as above; every method takes `&self` and only reads.
unsafe impl Sync for PixelBuffer {}

impl fmt::Debug for PixelBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PixelBuffer")
            .field("width", &self.width())
            .field("height", &self.height())
            .field("format", &self.format)
            .finish_non_exhaustive()
    }
}

impl PixelBuffer {
    /// Wraps `buffer`.
    ///
    /// # Errors
    /// [`UnsupportedPixelBuffer`] when the buffer is not biplanar 4:2:0
    /// (`420v`, `420f`, `x420`, `xf20`).
    pub fn new(buffer: CFRetained<CVPixelBuffer>) -> Result<Self, UnsupportedPixelBuffer> {
        let code = CVPixelBufferGetPixelFormatType(&buffer);
        let format = BiPlanarFormat::from_os_type(code).ok_or(UnsupportedPixelBuffer(code))?;
        Ok(Self { buffer, format })
    }

    /// The Core Video buffer, for APIs that take one (an encoder, a
    /// Metal texture cache).
    #[must_use]
    pub fn buffer(&self) -> &CVPixelBuffer {
        &self.buffer
    }

    /// The buffer's sample layout and range.
    #[must_use]
    pub fn format(&self) -> BiPlanarFormat {
        self.format
    }

    /// Width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        u32::try_from(CVPixelBufferGetWidth(&self.buffer)).unwrap_or(u32::MAX)
    }

    /// Height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        u32::try_from(CVPixelBufferGetHeight(&self.buffer)).unwrap_or(u32::MAX)
    }

    /// The colour the buffer's attachments signal, in the buffer's own
    /// range; `None` when it carries no primaries, transfer or matrix.
    #[must_use]
    pub fn colour(&self) -> Option<ColourDescription> {
        // SAFETY: Core Video exports these keys for the life of the process.
        let keys = unsafe {
            [
                kCVImageBufferColorPrimariesKey,
                kCVImageBufferTransferFunctionKey,
                kCVImageBufferYCbCrMatrixKey,
            ]
        };
        let [primaries, transfer, matrix] = keys.map(|key| self.string_attachment(key));
        if primaries.is_none() && transfer.is_none() && matrix.is_none() {
            return None;
        }
        // Codes Core Video does not know read as 2, "unspecified".
        let code = |value: i32| u8::try_from(value).unwrap_or(2);
        let colour = ColourDescription::from_codes(
            code(CVColorPrimariesGetIntegerCodePointForString(
                primaries.as_deref(),
            )),
            code(CVTransferFunctionGetIntegerCodePointForString(
                transfer.as_deref(),
            )),
            code(CVYCbCrMatrixGetIntegerCodePointForString(matrix.as_deref())),
            self.format.range.is_full(),
        );
        // SAFETY: as above.
        let location_key = unsafe { kCVImageBufferChromaLocationTopFieldKey };
        let location = self
            .string_attachment(location_key)
            .and_then(|location| chroma_location(&location));
        Some(location.map_or(colour, |location| colour.with_chroma_location(location)))
    }

    /// The static HDR metadata the buffer's attachments carry.
    #[must_use]
    pub fn hdr(&self) -> Option<HdrMetadata> {
        // SAFETY: Core Video exports these keys for the life of the process.
        let (display_key, light_key) = unsafe {
            (
                kCVImageBufferMasteringDisplayColorVolumeKey,
                kCVImageBufferContentLightLevelInfoKey,
            )
        };
        let metadata = HdrMetadata {
            mastering_display: self
                .data_attachment(display_key)
                .and_then(|bytes| MasteringDisplay::from_sei(&bytes)),
            content_light: self
                .data_attachment(light_key)
                .and_then(|bytes| ContentLightLevel::from_sei(&bytes)),
        };
        (!metadata.is_empty()).then_some(metadata)
    }

    fn string_attachment(&self, key: &CFString) -> Option<CFRetained<CFString>> {
        // SAFETY: the buffer is live; the attachment mode out pointer may
        // be null.
        let value = unsafe { self.buffer.attachment(key, ptr::null_mut()) }?;
        value.downcast::<CFString>().ok()
    }

    fn data_attachment(&self, key: &CFString) -> Option<Vec<u8>> {
        // SAFETY: as above.
        let value = unsafe { self.buffer.attachment(key, ptr::null_mut()) }?;
        let data = value.downcast::<CFData>().ok()?;
        Some(data.to_vec())
    }

    /// Copies the picture into a packed layout: luma rows, then the
    /// interleaved chroma rows, both `stride` bytes apart (the luma row
    /// rounded up to an even sample count, so an odd-width chroma row
    /// fits).
    fn copy_out(&self) -> Result<SurfacePixels, String> {
        let width = CVPixelBufferGetWidth(&self.buffer);
        let height = CVPixelBufferGetHeight(&self.buffer);
        let sample = self.format.sample_bytes();
        let stride = width.div_ceil(2) * 2 * sample;
        let mut out = vec![0; stride * (height + height.div_ceil(2))];
        // SAFETY: the buffer is live and only read.
        let status =
            unsafe { CVPixelBufferLockBaseAddress(&self.buffer, CVPixelBufferLockFlags::ReadOnly) };
        if status != 0 {
            return Err(format!("CVPixelBufferLockBaseAddress: OSStatus {status}"));
        }
        let copied = self.copy_planes(&mut out, stride, width * sample, height);
        // SAFETY: locked just above with the same flags.
        unsafe { CVPixelBufferUnlockBaseAddress(&self.buffer, CVPixelBufferLockFlags::ReadOnly) };
        copied?;
        let stride = u32::try_from(stride).map_err(|_| "the picture is too wide".to_owned())?;
        Ok(SurfacePixels {
            stride,
            data: Bytes::from(out),
        })
    }

    /// Copies both planes of the locked buffer into `out`.
    fn copy_planes(
        &self,
        out: &mut [u8],
        stride: usize,
        luma_bytes: usize,
        height: usize,
    ) -> Result<(), String> {
        let (luma, chroma) = out.split_at_mut(stride * height);
        for (plane, rows, row_bytes, dst) in [
            (0, height, luma_bytes, luma),
            (1, height.div_ceil(2), stride, chroma),
        ] {
            let base = CVPixelBufferGetBaseAddressOfPlane(&self.buffer, plane).cast::<u8>();
            let pitch = CVPixelBufferGetBytesPerRowOfPlane(&self.buffer, plane);
            if base.is_null() || pitch < row_bytes {
                return Err(format!(
                    "plane {plane} is unusable ({pitch} bytes per row, {row_bytes} needed)"
                ));
            }
            for (row, line) in dst.chunks_exact_mut(stride).take(rows).enumerate() {
                // SAFETY: the buffer is locked; plane `plane` holds `rows`
                // rows of `pitch >= row_bytes` bytes.
                let src = unsafe { std::slice::from_raw_parts(base.add(row * pitch), row_bytes) };
                line[..row_bytes].copy_from_slice(src);
            }
        }
        Ok(())
    }
}

impl NativeSurface for PixelBuffer {
    fn kind(&self) -> SurfaceKind {
        SurfaceKind::CvPixelBuffer
    }

    fn format(&self) -> PixelFormat {
        self.format.format
    }

    fn readback(&self) -> Result<SurfacePixels, FrameError> {
        self.copy_out()
            .map_err(|reason| FrameError::Readback { reason })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The H.273 chroma location a Core Video chroma location string names.
fn chroma_location(value: &CFString) -> Option<ChromaLocation> {
    // SAFETY: Core Video exports these strings for the life of the process.
    let known = unsafe {
        [
            (kCVImageBufferChromaLocation_Left, ChromaLocation::Left),
            (kCVImageBufferChromaLocation_Center, ChromaLocation::Center),
            (
                kCVImageBufferChromaLocation_TopLeft,
                ChromaLocation::TopLeft,
            ),
            (kCVImageBufferChromaLocation_Top, ChromaLocation::Top),
            (
                kCVImageBufferChromaLocation_BottomLeft,
                ChromaLocation::BottomLeft,
            ),
            (kCVImageBufferChromaLocation_Bottom, ChromaLocation::Bottom),
        ]
    };
    known
        .into_iter()
        .find(|(name, _)| *name == value)
        .map(|(_, location)| location)
}

/// The HDR metadata Core Video attachments and `VideoToolbox` properties
/// take: the SEI-layout mastering display (24 bytes) and content light
/// level (4 bytes), each when present.
#[must_use]
pub fn hdr_attachments(hdr: HdrMetadata) -> (Option<[u8; 24]>, Option<[u8; 4]>) {
    (
        hdr.mastering_display.map(MasteringDisplay::to_sei),
        hdr.content_light.map(ContentLightLevel::to_sei),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn biplanar_formats_round_trip_through_their_os_type() {
        for format in [PixelFormat::Nv12, PixelFormat::P010] {
            for range in [ColourRange::Limited, ColourRange::Full] {
                let planar = BiPlanarFormat { format, range };
                assert_eq!(
                    BiPlanarFormat::from_os_type(planar.os_type().unwrap()),
                    Some(planar)
                );
            }
        }
        assert_eq!(
            BiPlanarFormat::from_os_type(u32::from_be_bytes(*b"BGRA")),
            None
        );
        assert_eq!(
            UnsupportedPixelBuffer(u32::from_be_bytes(*b"BGRA")).to_string(),
            "pixel format 'BGRA' is not NV12 or P010"
        );
    }
}
