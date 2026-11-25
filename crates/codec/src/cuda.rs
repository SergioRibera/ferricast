//! NVIDIA plumbing the NVDEC decoders and NVENC encoders share (Linux and
//! Windows, through `ferricast-nvcodec`):
//!
//! - [`context`]: the CUDA context of the GPU both use, so a decoded
//!   picture and the encoder that takes it live on the same context.
//! - [`CudaSurface`]: an NVDEC picture kept in CUDA device memory, handed
//!   out as a [`NativeFrame`](ferricast_core::NativeFrame) of kind
//!   [`SurfaceKind::CudaDevice`]. NVENC encodes it without a copy;
//!   everything else reads it back ([`NativeSurface::readback`]).
//! - [`codec_error`]: how a `ferricast-nvcodec` error becomes a
//!   [`CodecError`], so selection falls back on a missing driver or a full
//!   GPU and hands a stream the GPU cannot decode to the next backend.

use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use bytes::Bytes;
use ferricast_core::{
    BackendId, CodecError, FrameError, NativeSurface, Operation, PixelFormat, SurfaceKind,
    SurfacePixels,
};
pub use ferricast_nvcodec as nvcodec;
use ferricast_nvcodec::cuda::{Context, Device};
use ferricast_nvcodec::decode::{self, DeviceFrame};

/// The primary CUDA context of the GPU ferricast's NVIDIA backends use
/// (device 0), retained for the life of the process. The first call loads
/// the driver; its outcome, failure included, is kept.
///
/// # Errors
///
/// What loading the driver or opening the device failed with
/// ([`nvcodec::Error::LibraryMissing`] without an NVIDIA driver).
pub fn context() -> Result<Arc<Context>, nvcodec::Error> {
    static CONTEXT: OnceLock<Result<Arc<Context>, nvcodec::Error>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| Device::get(0).and_then(Context::primary))
        .clone()
}

/// Host readbacks of [`CudaSurface`]s since the process started.
static READBACKS: AtomicU64 = AtomicU64::new(0);

/// A decoded picture in a pooled CUDA device buffer (NVDEC output):
/// NV12, or P010 for streams deeper than 8 bits (NVDEC's P016, whose
/// samples sit in the high bits like P010's).
///
/// The buffer goes back to the decoder's pool when the last clone of the
/// frame holding it drops. An encoder that reads it must keep the frame
/// until the encode has completed.
pub struct CudaSurface {
    frame: DeviceFrame,
    format: PixelFormat,
}

impl CudaSurface {
    /// Wraps an NVDEC device frame; `None` for layouts ferricast has no
    /// [`PixelFormat`] for (4:2:2, 4:4:4).
    #[must_use]
    pub fn new(frame: DeviceFrame) -> Option<Self> {
        let format = match frame.info().format {
            decode::PixelFormat::Nv12 => PixelFormat::Nv12,
            decode::PixelFormat::P016 => PixelFormat::P010,
            decode::PixelFormat::Nv16
            | decode::PixelFormat::P216
            | decode::PixelFormat::Yuv444
            | decode::PixelFormat::Yuv444P16 => return None,
        };
        Some(Self { frame, format })
    }

    /// The NVDEC frame: its buffer ([`DeviceFrame::shared_buffer`]), the
    /// event that completes when the picture is in it
    /// ([`DeviceFrame::ready`]) and its layout.
    #[must_use]
    pub fn frame(&self) -> &DeviceFrame {
        &self.frame
    }

    /// How many times any `CudaSurface` was read back to host memory in
    /// this process: a GPU-only pipeline keeps it unchanged.
    #[must_use]
    pub fn readbacks() -> u64 {
        READBACKS.load(Ordering::Relaxed)
    }
}

impl fmt::Debug for CudaSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaSurface")
            .field("format", &self.format)
            .field("frame", &self.frame)
            .finish()
    }
}

impl NativeSurface for CudaSurface {
    fn kind(&self) -> SurfaceKind {
        SurfaceKind::CudaDevice
    }

    fn format(&self) -> PixelFormat {
        self.format
    }

    fn readback(&self) -> Result<SurfacePixels, FrameError> {
        READBACKS.fetch_add(1, Ordering::Relaxed);
        let readback = |reason: String| FrameError::Readback { reason };
        let mut packed = Vec::new();
        let layout = self
            .frame
            .download(&mut packed)
            .map_err(|error| readback(format!("CUDA download: {error}")))?;
        let [luma, chroma] = layout.planes() else {
            return Err(readback(format!(
                "{} planes in an NV12 / P010 picture",
                layout.planes().len()
            )));
        };
        // One stride for both planes: they only differ for odd widths,
        // where the chroma row carries one sample more than the luma row.
        let stride = luma.pitch.max(chroma.pitch);
        let data = if luma.pitch == chroma.pitch {
            packed
        } else {
            let mut data = vec![0; stride * (luma.rows + chroma.rows)];
            for (plane, start) in [(luma, 0), (chroma, luma.rows)] {
                let source = packed.get(plane.offset..plane.offset + plane.len());
                let source = source.ok_or_else(|| readback("a short download".into()))?;
                for (row, line) in source.chunks_exact(plane.pitch).enumerate() {
                    let at = (start + row) * stride;
                    data[at..at + plane.width_bytes].copy_from_slice(&line[..plane.width_bytes]);
                }
            }
            data
        };
        Ok(SurfacePixels {
            stride: u32::try_from(stride)
                .map_err(|_| readback("the picture is too wide".into()))?,
            data: Bytes::from(data),
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A `ferricast-nvcodec` error from `backend` while doing `operation`, as
/// a [`CodecError`]:
///
/// - no driver, a driver too old, no device, an NVENC session limit
///   reached, and any CUDA failure while probing or opening:
///   [`CodecError::BackendUnavailable`], so selection moves on;
/// - something the GPU or driver cannot do: [`CodecError::UnsupportedStream`]
///   while decoding (the next decoder takes the stream over),
///   [`CodecError::UnsupportedConfig`] otherwise;
/// - everything else (a corrupt packet, a failed copy):
///   [`CodecError::Backend`].
#[must_use]
pub fn codec_error(backend: BackendId, operation: Operation, error: nvcodec::Error) -> CodecError {
    use nvcodec::Error as E;
    let unavailable = |error: E| CodecError::BackendUnavailable {
        backend,
        reason: error.to_string(),
    };
    match error {
        E::LibraryMissing { .. }
        | E::SymbolMissing { .. }
        | E::DriverTooOld { .. }
        | E::ApiNotSupportedByDriver { .. }
        | E::InvalidApiOverride { .. }
        | E::NvencFunctionMissing { .. }
        | E::DeviceOutOfRange { .. }
        | E::NvencSessionLimit { .. }
        | E::NoEglDevice { .. } => unavailable(error),
        E::Cuda { .. } | E::Cuvid { .. } | E::Nvenc { .. }
            if matches!(operation, Operation::Probe | Operation::Open) =>
        {
            unavailable(error)
        }
        E::Unsupported { .. } if matches!(operation, Operation::Decode | Operation::Flush) => {
            CodecError::UnsupportedStream {
                backend,
                reason: error.to_string(),
            }
        }
        E::Unsupported { .. } | E::InvalidEncodeConfig { .. } => CodecError::UnsupportedConfig {
            backend,
            what: error.to_string(),
        },
        other => CodecError::backend(backend, operation, other),
    }
}

/// `move |error| codec_error(backend, operation, error)`, for `map_err`.
pub fn map_error(
    backend: BackendId,
    operation: Operation,
) -> impl FnOnce(nvcodec::Error) -> CodecError {
    move |error| codec_error(backend, operation, error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_driver_lets_selection_fall_back() {
        let missing = nvcodec::Error::LibraryMissing {
            library: "libcuda.so.1",
            reason: "not found".into(),
        };
        assert!(matches!(
            codec_error(BackendId::Nvdec, Operation::Decode, missing),
            CodecError::BackendUnavailable { .. }
        ));
    }

    #[test]
    fn what_the_gpu_cannot_decode_hands_the_stream_over() {
        let unsupported = || nvcodec::Error::Unsupported {
            what: "H.264 10-bit 4:2:0",
            reason: "not on this GPU".into(),
        };
        assert!(matches!(
            codec_error(BackendId::Nvdec, Operation::Decode, unsupported()),
            CodecError::UnsupportedStream { .. }
        ));
        assert!(matches!(
            codec_error(BackendId::Nvenc, Operation::Open, unsupported()),
            CodecError::UnsupportedConfig { .. }
        ));
    }

    #[test]
    fn a_corrupt_packet_is_a_decode_error() {
        let invalid = nvcodec::Error::InvalidPacket {
            reason: "OBU past the end".into(),
        };
        assert!(matches!(
            codec_error(BackendId::Nvdec, Operation::Decode, invalid),
            CodecError::Backend {
                operation: Operation::Decode,
                ..
            }
        ));
    }
}
