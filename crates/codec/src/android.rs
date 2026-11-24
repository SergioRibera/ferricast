//! The Android media surfaces the `MediaCodec` encoders and decoders
//! share: [`HardwareBuffer`], the [`NativeSurface`] of a picture held in
//! an `AHardwareBuffer`, and [`ImageOutput`], the `AImageReader` a
//! decoder renders into to hand pictures out that way.
//!
//! Pictures are NV12 (`AIMAGE_FORMAT_YUV_420_888`) or P010
//! (`AIMAGE_FORMAT_YCBCR_P010`) buffers allocated GPU-sampleable and
//! CPU-readable: a renderer imports them (EGL / Vulkan external memory)
//! with no copy, everything else reads them back.

use std::any::Any;
use std::ffi::{c_int, c_void};
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use bytes::Bytes;
use ferricast_core::{
    BackendId, BitDepth, CodecError, CodecResult, FrameError, NativeSurface, Operation,
    PixelFormat, SurfaceKind, SurfacePixels,
};
use ndk::hardware_buffer::{HardwareBufferRef, HardwareBufferUsage};
use ndk::media::image_reader::{AcquireResult, Image, ImageFormat, ImageReader};
use ndk::native_window::NativeWindow;

use crate::yuv420::{self, Crop, Plane, SemiPlanar};

/// `AIMAGE_FORMAT_YCBCR_P010`, also `AHARDWAREBUFFER_FORMAT_YCbCr_P010`.
const FORMAT_P010: i32 = 0x36;
/// `AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420`.
const FORMAT_YUV420: i32 = 0x23;

/// The pixel format of a native image / buffer format code.
fn pixel_format(code: i32) -> Option<PixelFormat> {
    match code {
        FORMAT_P010 => Some(PixelFormat::P010),
        FORMAT_YUV420 => Some(PixelFormat::Nv12),
        code if code == i32::from(ImageFormat::YUV_420_888) => Some(PixelFormat::Nv12),
        _ => None,
    }
}

fn depth(format: PixelFormat) -> BitDepth {
    format.bit_depth()
}

/// An `AImageReader`, shared by the reader side and every picture still
/// holding one of its images (the reader must outlive them).
struct Reader(Mutex<ImageReader>);

// SAFETY: an `AImageReader` is not bound to the thread that created it.
// Every call on it goes through the mutex; `AImage_delete` from the
// thread that drops a picture takes the reader's own internal lock
// before returning the image (NdkImage.cpp `AImage::lockReader`).
unsafe impl Send for Reader {}
// SAFETY: as above; `&Reader` only reaches the reader through the mutex.
unsafe impl Sync for Reader {}

/// What a [`HardwareBuffer`] holds its buffer through.
enum Source {
    /// An image acquired from an [`ImageOutput`]: dropping it returns the
    /// slot to the reader. Declared before `reader` so it drops first.
    Image { image: Image, _reader: Arc<Reader> },
    /// A buffer someone else produced, with a reference of its own.
    Buffer(HardwareBufferRef),
}

// SAFETY: `AImage` and `AHardwareBuffer` handles are not bound to the
// thread that created them (an `AHardwareBuffer` is reference counted
// and documented thread-safe; `AImage` calls lock the image's reader).
// A `Source` is only reached through the `Mutex` in `HardwareBuffer`, so
// no two threads use one at the same time.
unsafe impl Send for Source {}

/// A decoded or captured picture in an `AHardwareBuffer`: NV12 or P010,
/// with its visible rectangle.
pub struct HardwareBuffer {
    source: Mutex<Source>,
    format: PixelFormat,
    crop: Crop,
}

impl fmt::Debug for HardwareBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HardwareBuffer")
            .field("format", &self.format)
            .field("crop", &self.crop)
            .finish_non_exhaustive()
    }
}

/// A frame error naming the step that failed.
fn readback(reason: impl fmt::Display) -> FrameError {
    FrameError::Readback {
        reason: format!("AHardwareBuffer: {reason}"),
    }
}

impl HardwareBuffer {
    /// Wraps a buffer a producer allocated (camera, `MediaProjection`, a
    /// renderer), keeping a reference to it.
    ///
    /// # Errors
    /// [`CodecError::UnsupportedInput`] when the buffer is neither
    /// `Y8Cb8Cr8_420` nor `YCbCr_P010`.
    pub fn from_buffer(buffer: HardwareBufferRef) -> CodecResult<Self> {
        let desc = buffer.describe();
        let code = i32::from(desc.format);
        let format = pixel_format(code).ok_or_else(|| CodecError::UnsupportedInput {
            backend: BackendId::MediaCodec,
            expected: format!(
                "a Y8Cb8Cr8_420 or YCbCr_P010 AHardwareBuffer (format {code:#x} given)"
            ),
            actual: PixelFormat::Nv12,
        })?;
        Ok(Self {
            source: Mutex::new(Source::Buffer(buffer)),
            format,
            crop: Crop::full(desc.width as usize, desc.height as usize),
        })
    }

    /// The visible rectangle, in luma samples.
    #[must_use]
    pub fn crop(&self) -> Crop {
        self.crop
    }

    /// A new reference to the buffer, for a GPU import
    /// (`eglGetNativeClientBufferANDROID`,
    /// `VK_ANDROID_external_memory_android_hardware_buffer`). Keep this
    /// picture alive while the GPU reads it: an image's buffer goes back
    /// to its decoder when the picture is dropped.
    ///
    /// # Errors
    /// [`FrameError::Readback`] when the image has no buffer.
    pub fn buffer(&self) -> Result<HardwareBufferRef, FrameError> {
        match &*self.source.lock().unwrap_or_else(PoisonError::into_inner) {
            Source::Image { image, .. } => image
                .hardware_buffer()
                .map(|buffer| buffer.acquire())
                .map_err(|error| readback(format!("AImage_getHardwareBuffer: {error}"))),
            Source::Buffer(buffer) => Ok(buffer.clone()),
        }
    }

    /// Writes the visible picture into `dst`, laid out as `layout` at
    /// depth `to` (P010 narrows to 8 bits or 8 bits widen, as asked):
    /// the one copy of a picture going into an encoder input buffer.
    ///
    /// # Errors
    /// [`FrameError::Readback`] when the buffer cannot be mapped, or
    /// `dst` / `layout` are too small.
    pub fn write_to(
        &self,
        to: BitDepth,
        layout: SemiPlanar,
        dst: &mut [u8],
    ) -> Result<(), FrameError> {
        let from = depth(self.format);
        let mut pack =
            |planes| yuv420::pack(planes, self.crop, from, to, layout, dst).map_err(readback);
        match &*self.source.lock().unwrap_or_else(PoisonError::into_inner) {
            Source::Image { image, .. } => {
                let plane = |index: i32| -> Result<Plane<'_>, FrameError> {
                    let stride = |value: ndk::media_error::Result<i32>, what: &str| {
                        value
                            .map_err(|error| readback(format!("AImage {what}: {error}")))
                            .and_then(|v| {
                                usize::try_from(v)
                                    .map_err(|_| readback(format!("AImage {what} {v}")))
                            })
                    };
                    Ok(Plane {
                        data: image
                            .plane_data(index)
                            .map_err(|error| readback(format!("AImage_getPlaneData: {error}")))?,
                        row_stride: stride(image.plane_row_stride(index), "row stride")?,
                        pixel_stride: stride(image.plane_pixel_stride(index), "pixel stride")?,
                    })
                };
                pack([plane(0)?, plane(1)?, plane(2)?])
            }
            Source::Buffer(buffer) => {
                let locked = LockedPlanes::lock(buffer, self.format)?;
                pack(locked.planes())
            }
        }
    }
}

impl NativeSurface for HardwareBuffer {
    fn kind(&self) -> SurfaceKind {
        SurfaceKind::HardwareBuffer
    }

    fn format(&self) -> PixelFormat {
        self.format
    }

    fn readback(&self) -> Result<SurfacePixels, FrameError> {
        let to = depth(self.format);
        let layout = SemiPlanar::packed(self.crop.width, self.crop.height, to);
        let mut data = vec![0; layout.len(self.crop.height)];
        self.write_to(to, layout, &mut data)?;
        Ok(SurfacePixels {
            stride: u32::try_from(layout.stride)
                .map_err(|_| readback("the picture is too wide"))?,
            data: Bytes::from(data),
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `AHardwareBuffer_lockPlanes`, API 29+.
type LockPlanesFn = unsafe extern "C" fn(
    *mut ndk_sys::AHardwareBuffer,
    u64,
    i32,
    *const ndk_sys::ARect,
    *mut ndk_sys::AHardwareBuffer_Planes,
) -> c_int;

/// `AHardwareBuffer_lockPlanes`, looked up at run time: linking it
/// directly would keep the library from loading below API 29.
fn lock_planes_fn() -> Option<LockPlanesFn> {
    static LOCK_PLANES: OnceLock<Option<LockPlanesFn>> = OnceLock::new();
    *LOCK_PLANES.get_or_init(|| {
        // SAFETY: `dlopen` / `dlsym` with NUL-terminated literals; the
        // library handle is deliberately never closed, so the symbol
        // stays valid for the life of the process.
        unsafe {
            let library = libc::dlopen(c"libnativewindow.so".as_ptr(), libc::RTLD_NOW);
            if library.is_null() {
                return None;
            }
            let symbol = libc::dlsym(library, c"AHardwareBuffer_lockPlanes".as_ptr());
            // SAFETY: the NDK declares the symbol with exactly
            // `LockPlanesFn`'s signature (`AHardwareBuffer.h`).
            (!symbol.is_null()).then(|| std::mem::transmute::<*mut c_void, LockPlanesFn>(symbol))
        }
    })
}

/// A buffer locked for CPU reads, unlocked on drop.
struct LockedPlanes<'a> {
    buffer: &'a HardwareBufferRef,
    planes: [(*const u8, usize, usize, usize); 3],
}

impl<'a> LockedPlanes<'a> {
    fn lock(buffer: &'a HardwareBufferRef, format: PixelFormat) -> Result<Self, FrameError> {
        let lock = lock_planes_fn().ok_or_else(|| {
            readback("reading a YUV buffer needs AHardwareBuffer_lockPlanes (Android 10, API 29)")
        })?;
        let desc = buffer.describe();
        let (width, height) = (desc.width as usize, desc.height as usize);
        let mut planes = std::mem::MaybeUninit::<ndk_sys::AHardwareBuffer_Planes>::uninit();
        // SAFETY: `buffer` is a live reference; no fence (-1) and no
        // rect (whole buffer); `planes` is written by the call on success.
        let status = unsafe {
            lock(
                buffer.as_ptr(),
                HardwareBufferUsage::CPU_READ_OFTEN.bits(),
                -1,
                std::ptr::null(),
                planes.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(readback(format!(
                "AHardwareBuffer_lockPlanes failed ({status})"
            )));
        }
        // SAFETY: the call succeeded, so it filled `planes`.
        let planes = unsafe { planes.assume_init() };
        let locked = |planes| Self { buffer, planes };
        if planes.planeCount < 3 {
            // Unlock through the guard before reporting.
            drop(locked([(std::ptr::null(), 0, 0, 0); 3]));
            return Err(readback(format!(
                "a YUV buffer locked into {} planes",
                planes.planeCount
            )));
        }
        let bytes = yuv420::sample_bytes(depth(format));
        let plane = |index: usize, columns: usize, rows: usize| {
            let p = planes.planes[index];
            let (row_stride, pixel_stride) = (p.rowStride as usize, p.pixelStride as usize);
            // Bytes from the first sample to the end of the last one.
            let len = row_stride * rows.saturating_sub(1)
                + pixel_stride * columns.saturating_sub(1)
                + bytes;
            (
                p.data.cast_const().cast::<u8>(),
                len,
                row_stride,
                pixel_stride,
            )
        };
        let (chroma_columns, chroma_rows) = (width.div_ceil(2), height.div_ceil(2));
        Ok(locked([
            plane(0, width, height),
            plane(1, chroma_columns, chroma_rows),
            plane(2, chroma_columns, chroma_rows),
        ]))
    }

    fn planes(&self) -> [Plane<'_>; 3] {
        self.planes
            .map(|(data, len, row_stride, pixel_stride)| Plane {
                data: if data.is_null() {
                    &[]
                } else {
                    // SAFETY: the buffer stays locked for CPU reads while
                    // `self` lives, and the lock maps every sample of the
                    // plane: `len` ends at its last sample (see `lock`).
                    unsafe { std::slice::from_raw_parts(data, len) }
                },
                row_stride,
                pixel_stride,
            })
    }
}

impl Drop for LockedPlanes<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.buffer.unlock() {
            tracing::warn!(%error, "AHardwareBuffer_unlock failed");
        }
    }
}

/// An `AImageReader` a decoder renders into: its window is the codec's
/// output surface, its images are the pictures.
pub struct ImageOutput {
    reader: Arc<Reader>,
    window: NativeWindow,
    format: PixelFormat,
    max_images: u32,
}

impl fmt::Debug for ImageOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageOutput")
            .field("format", &self.format)
            .field("max_images", &self.max_images)
            .finish_non_exhaustive()
    }
}

/// How long [`ImageOutput::next`] sleeps between polls.
const POLL: Duration = Duration::from_millis(1);

impl ImageOutput {
    /// A reader of `width` × `height` [`PixelFormat::Nv12`] or
    /// [`PixelFormat::P010`] images, of which the consumer may hold
    /// `max_images` at once.
    ///
    /// # Errors
    /// [`CodecError::UnsupportedConfig`] for another format or a device
    /// that cannot allocate such images.
    pub fn new(width: u32, height: u32, format: PixelFormat, max_images: u32) -> CodecResult<Self> {
        let unsupported = |what: String| CodecError::UnsupportedConfig {
            backend: BackendId::MediaCodec,
            what,
        };
        let image_format = match format {
            PixelFormat::Nv12 => ImageFormat::YUV_420_888,
            PixelFormat::P010 => ImageFormat::from(FORMAT_P010),
            other @ (PixelFormat::Bgra | PixelFormat::Rgba | PixelFormat::I420) => {
                return Err(unsupported(format!(
                    "AHardwareBuffer pictures in {other:?}"
                )));
            }
        };
        let dimension = |value: u32| i32::try_from(value.max(1)).unwrap_or(i32::MAX);
        let reader = ImageReader::new_with_usage(
            dimension(width),
            dimension(height),
            image_format,
            HardwareBufferUsage::GPU_SAMPLED_IMAGE | HardwareBufferUsage::CPU_READ_OFTEN,
            i32::try_from(max_images.max(1)).unwrap_or(i32::MAX),
        )
        .map_err(|error| {
            unsupported(format!(
                "a {width}x{height} {format:?} AImageReader ({error})"
            ))
        })?;
        let window = reader
            .window()
            .map_err(|error| unsupported(format!("the AImageReader window ({error})")))?;
        Ok(Self {
            reader: Arc::new(Reader(Mutex::new(reader))),
            window,
            format,
            max_images,
        })
    }

    /// The surface to configure the codec with.
    #[must_use]
    pub fn window(&self) -> &NativeWindow {
        &self.window
    }

    /// The format of the pictures.
    #[must_use]
    pub fn format(&self) -> PixelFormat {
        self.format
    }

    /// The next picture the codec rendered, waiting up to `timeout`
    /// (rendering is asynchronous), with its presentation time in µs.
    ///
    /// # Errors
    /// [`CodecError::Backend`] when the consumer holds every image the
    /// reader has (pictures must be dropped for decoding to go on) or
    /// the reader fails.
    pub fn next(&self, timeout: Duration) -> CodecResult<Option<(HardwareBuffer, u64)>> {
        let fail =
            |reason: String| CodecError::backend(BackendId::MediaCodec, Operation::Decode, reason);
        let deadline = Instant::now() + timeout;
        loop {
            let acquired = self
                .reader
                .0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .acquire_next_image()
                .map_err(|error| fail(format!("AImageReader_acquireNextImage: {error}")))?;
            match acquired {
                AcquireResult::Image(image) => return self.wrap(image).map(Some),
                AcquireResult::MaxImagesAcquired => {
                    return Err(fail(format!(
                        "the consumer holds all {} decoded pictures; drop them sooner or \
                         decode to CPU memory",
                        self.max_images
                    )));
                }
                AcquireResult::NoBufferAvailable if Instant::now() >= deadline => {
                    return Ok(None);
                }
                AcquireResult::NoBufferAvailable => std::thread::sleep(POLL),
            }
        }
    }

    fn wrap(&self, image: Image) -> CodecResult<(HardwareBuffer, u64)> {
        let fail =
            |reason: String| CodecError::backend(BackendId::MediaCodec, Operation::Decode, reason);
        let code = image
            .format()
            .map(i32::from)
            .map_err(|error| fail(format!("AImage_getFormat: {error}")))?;
        let format = pixel_format(code).unwrap_or(self.format);
        let rect = image
            .crop_rect()
            .map_err(|error| fail(format!("AImage_getCropRect: {error}")))?;
        let extent = |from: i32, to: i32| {
            usize::try_from(to.saturating_sub(from))
                .ok()
                .filter(|&n| n > 0)
        };
        let crop = if let (Ok(left), Ok(top), Some(width), Some(height)) = (
            usize::try_from(rect.left),
            usize::try_from(rect.top),
            extent(rect.left, rect.right),
            extent(rect.top, rect.bottom),
        ) {
            Crop {
                left,
                top,
                width,
                height,
            }
        } else {
            let size = |value: ndk::media_error::Result<i32>| {
                value
                    .ok()
                    .and_then(|v| usize::try_from(v).ok())
                    .unwrap_or(0)
            };
            Crop::full(size(image.width()), size(image.height()))
        };
        if crop.width == 0 || crop.height == 0 {
            return Err(fail("the codec rendered an empty picture".into()));
        }
        let timestamp_us = image
            .timestamp()
            .ok()
            .and_then(|ns| u64::try_from(ns / 1000).ok())
            .unwrap_or(0);
        Ok((
            HardwareBuffer {
                source: Mutex::new(Source::Image {
                    image,
                    _reader: Arc::clone(&self.reader),
                }),
                format,
                crop,
            },
            timestamp_us,
        ))
    }
}
