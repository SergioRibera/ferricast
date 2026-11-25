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
use std::sync::{Arc, Mutex, PoisonError, Weak};

use bytes::Bytes;
use ferricast_core::{
    BackendId, CodecError, FrameError, GpuAffinity, NativeSurface, Operation, PixelFormat,
    SurfaceKind, SurfacePixels,
};
pub use ferricast_nvcodec as nvcodec;

use crate::gpu::GpuInput;
use ferricast_nvcodec::cuda::{
    Context, Device, PciBusId, PinnedBuffer, PinnedCopy, Point, Region, Stream,
};
use ferricast_nvcodec::decode::{self, DeviceFrame};
use ferricast_nvcodec::loader::Backoff;

/// The CUDA context of the default GPU (CUDA's device 0); see
/// [`context_on`].
///
/// # Errors
///
/// As [`context_on`].
pub fn context() -> Result<Arc<Context>, nvcodec::Error> {
    context_on(CudaDevice::Default)
}

/// Which GPU an NVIDIA backend runs on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CudaDevice {
    /// CUDA's device 0 (the machine's GPUs are unknown: no sysfs).
    #[default]
    Default,
    /// The CUDA device at this PCI address.
    Pci(PciBusId),
}

impl CudaDevice {
    /// The device `config`'s GPU preference puts `backend` on: the first
    /// NVIDIA GPU it ranks.
    ///
    /// # Errors
    /// [`CodecError::BackendUnavailable`] when the preference pins a GPU
    /// that is missing or not NVIDIA's (selection moves on).
    pub fn for_config(config: &impl GpuInput, backend: BackendId) -> Result<Self, CodecError> {
        Ok(config
            .gpu_for(backend, GpuAffinity::Nvidia)?
            .and_then(|gpu| gpu.pci)
            .map_or(Self::Default, |pci| {
                Self::Pci(PciBusId {
                    domain: pci.domain,
                    bus: pci.bus,
                    device: pci.device,
                    function: pci.function,
                })
            }))
    }

    fn open(self) -> Result<Device, nvcodec::Error> {
        match self {
            Self::Default => Device::get(0),
            Self::Pci(pci) => Device::by_pci_bus_id(pci),
        }
    }
}

impl fmt::Display for CudaDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => f.write_str("CUDA device 0"),
            Self::Pci(pci) => write!(f, "the GPU at {pci}"),
        }
    }
}

/// The CUDA context of `device`, shared by every decoder and encoder open
/// on it at the same time, so a decoded picture and the encoder that takes
/// it live on the same context.
///
/// - It is the device's primary context, retained while someone holds the
///   returned `Arc` and released when the last holder drops it (about
///   100 MiB of VRAM, and on a hybrid laptop what keeps the discrete GPU
///   awake).
/// - A context found lost ([`Context::is_lost`]: a sticky error, an Xid, a
///   resume) is not handed out again: the next call retains the primary
///   context afresh or, while the lost one is still held, creates a context
///   of its own.
/// - A failure to start the driver or open the device is retried after a
///   delay ([`Backoff`]); loading failures (no driver library) are kept for
///   the process.
///
/// # Errors
///
/// What loading the driver or opening the device failed with
/// ([`nvcodec::Error::LibraryMissing`] without an NVIDIA driver;
/// `CUDA_ERROR_INVALID_DEVICE` when no CUDA device sits at the address).
pub fn context_on(device: CudaDevice) -> Result<Arc<Context>, nvcodec::Error> {
    let mut shared = SHARED.lock().unwrap_or_else(PoisonError::into_inner);
    let index = if let Some(index) = shared.iter().position(|entry| entry.device == device) {
        index
    } else {
        shared.push(SharedContext {
            device,
            current: Weak::new(),
            backoff: Backoff::new(),
        });
        shared.len() - 1
    };
    shared[index].context()
}

/// The contexts [`context_on`] hands out, one per device asked for.
static SHARED: Mutex<Vec<SharedContext>> = Mutex::new(Vec::new());

struct SharedContext {
    device: CudaDevice,
    current: Weak<Context>,
    /// Spaces out attempts after a failed one.
    backoff: Backoff,
}

impl SharedContext {
    fn context(&mut self) -> Result<Arc<Context>, nvcodec::Error> {
        let held = self.current.upgrade();
        if let Some(context) = &held
            && !context.is_lost()
        {
            return Ok(Arc::clone(context));
        }
        let lost_and_held = held.is_some();
        drop(held);
        let device = self.device;
        let context = self.backoff.retry(|| {
            let device = device.open()?;
            // The driver would hand a lost primary context that someone
            // still holds out again: take a context of its own instead.
            if lost_and_held {
                Context::create(device)
            } else {
                Context::primary(device)
            }
        })?;
        if lost_and_held {
            tracing::warn!(
                "the shared CUDA context is lost and still held: new NVIDIA sessions get a context of their own"
            );
        }
        self.current = Arc::downgrade(&context);
        Ok(context)
    }
}

/// Forgets the shared contexts if `error` says the device is lost, so the
/// next [`context_on`] call opens a new one. [`codec_error`] does this for
/// every error it maps; the encoders and decoders mark their own context
/// too.
fn note_lost(error: &nvcodec::Error) {
    if !error.is_device_lost() {
        return;
    }
    let shared = SHARED.lock().unwrap_or_else(PoisonError::into_inner);
    // The error does not name its device: every shared context is marked
    // (on a multi-GPU machine a healthy one is only replaced by a fresh
    // context, not lost).
    for context in shared.iter().filter_map(|entry| entry.current.upgrade()) {
        // Marks it lost (the error already says so; nothing is queried).
        let _ = context.note_error(error);
    }
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
        let fail = |reason: String| FrameError::Readback { reason };
        let cuda = |error: nvcodec::Error| fail(format!("CUDA download: {error}"));
        let frame = &self.frame;
        let [luma, chroma] = frame.layout().planes() else {
            return Err(fail(format!(
                "{} planes in an NV12 / P010 picture",
                frame.layout().planes().len()
            )));
        };
        // One stride for both planes: they only differ for odd widths,
        // where the chroma row carries one sample more than the luma row.
        let stride = luma.width_bytes.max(chroma.width_bytes);
        let len = stride * (luma.rows + chroma.rows);
        let buffer = frame.buffer();
        let mut staging = Staging::take(buffer.context(), len).map_err(cuda)?;
        // Both planes are copied on the staging stream once the picture is
        // in the buffer, straight into page-locked memory at `stride`; one
        // wait for both.
        let copies = [(luma, 0), (chroma, luma.rows)].map(|(plane, start)| PinnedCopy {
            src: Region::new(plane.width_bytes, plane.rows).at(Point {
                x_bytes: 0,
                row: plane.offset.checked_div(plane.pitch).unwrap_or(0),
            }),
            offset: start * stride,
            pitch: stride,
        });
        let Staging { pinned, stream } = &mut staging;
        pinned
            .download(buffer, &copies, stream, Some(frame.ready()))
            .map_err(cuda)?;
        Ok(SurfacePixels {
            stride: u32::try_from(stride).map_err(|_| fail("the picture is too wide".into()))?,
            data: Bytes::from_owner(Readback {
                staging: Some(staging),
                len,
            }),
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Page-locked memory and a stream for readbacks, pooled: a readback costs
/// two queued copies and one wait, no allocation in steady state.
struct Staging {
    pinned: PinnedBuffer,
    stream: Stream,
}

/// Idle staging buffers.
static STAGING: Mutex<Vec<Staging>> = Mutex::new(Vec::new());

impl Staging {
    /// Readbacks lent out at once beyond which buffers are freed on return.
    const POOLED: usize = 4;

    /// A pooled staging buffer of at least `len` bytes on `context`, or a
    /// new one.
    fn take(context: &Arc<Context>, len: usize) -> Result<Self, nvcodec::Error> {
        let pooled = {
            let mut pool = STAGING.lock().unwrap_or_else(PoisonError::into_inner);
            pool.iter()
                .position(|staging| {
                    Arc::ptr_eq(staging.pinned.context(), context)
                        && staging.pinned.as_slice().len() >= len
                })
                .map(|index| pool.swap_remove(index))
        };
        match pooled {
            Some(staging) => Ok(staging),
            None => Ok(Self {
                pinned: PinnedBuffer::new(context, len)?,
                stream: Stream::new(context)?,
            }),
        }
    }

    fn give_back(self) {
        let mut pool = STAGING.lock().unwrap_or_else(PoisonError::into_inner);
        // Buffers of a context nobody else uses would keep it alive.
        pool.retain(|staging| Arc::strong_count(staging.pinned.context()) > 2);
        if pool.len() < Self::POOLED && !self.pinned.context().is_lost() {
            pool.push(self);
        }
    }
}

/// A readback lent to `Bytes`: its staging buffer goes back to the pool
/// when the last clone drops.
struct Readback {
    staging: Option<Staging>,
    len: usize,
}

impl AsRef<[u8]> for Readback {
    fn as_ref(&self) -> &[u8] {
        self.staging
            .as_ref()
            .and_then(|staging| staging.pinned.as_slice().get(..self.len))
            .unwrap_or_default()
    }
}

impl Drop for Readback {
    fn drop(&mut self) {
        if let Some(staging) = self.staging.take() {
            staging.give_back();
        }
    }
}

/// A `ferricast-nvcodec` error from `backend` while doing `operation`, as
/// a [`CodecError`]:
///
/// - no driver library, a driver too old, no such device, an unusable
///   NVENC API version: [`CodecError::BackendUnavailable`] (for the
///   process; selection moves on);
/// - an NVENC session limit reached, a closed encoder, and any CUDA / NVDEC
///   / NVENC failure while probing or opening: [`CodecError::BackendBusy`]
///   (selection moves on now and tries again next time);
/// - a lost device (sticky CUDA error): [`CodecError::UnsupportedStream`]
///   while decoding (the next decoder takes the stream over),
///   [`CodecError::BackendBusy`] otherwise (an encoder hands over); the
///   shared context is forgotten so the next open gets a new one;
/// - something the GPU or driver cannot do: [`CodecError::UnsupportedStream`]
///   while decoding, [`CodecError::UnsupportedConfig`] otherwise;
/// - everything else (a corrupt packet, a failed copy):
///   [`CodecError::Backend`].
#[must_use]
pub fn codec_error(backend: BackendId, operation: Operation, error: nvcodec::Error) -> CodecError {
    use nvcodec::Error as E;
    note_lost(&error);
    let decoding = matches!(operation, Operation::Decode | Operation::Flush);
    let reason = error.to_string();
    if error.is_permanent() || matches!(error, E::NoEglDevice { .. }) {
        return CodecError::BackendUnavailable { backend, reason };
    }
    if error.is_device_lost() {
        return if decoding {
            CodecError::UnsupportedStream { backend, reason }
        } else {
            CodecError::BackendBusy { backend, reason }
        };
    }
    match error {
        E::NvencSessionLimit { .. } | E::EncoderClosed { .. } => {
            CodecError::BackendBusy { backend, reason }
        }
        E::Cuda { .. } | E::Cuvid { .. } | E::Nvenc { .. }
            if matches!(operation, Operation::Probe | Operation::Open) =>
        {
            CodecError::BackendBusy { backend, reason }
        }
        E::Unsupported { .. } if decoding => CodecError::UnsupportedStream { backend, reason },
        E::Unsupported { .. } | E::InvalidEncodeConfig { .. } => CodecError::UnsupportedConfig {
            backend,
            what: reason,
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

    fn cuda(code: u32) -> nvcodec::Error {
        nvcodec::Error::Cuda {
            call: "cuMemcpy2DAsync_v2",
            status: nvcodec::error::CudaStatus(nvcodec::sys::cuda::CUresult(code)),
        }
    }

    #[test]
    fn a_full_gpu_is_busy_not_unavailable() {
        let limit = nvcodec::Error::NvencSessionLimit {
            call: "nvEncOpenEncodeSessionEx",
            status: nvcodec::error::NvencStatus(21),
            detail: None,
        };
        assert!(matches!(
            codec_error(BackendId::Nvenc, Operation::Probe, limit),
            CodecError::BackendBusy { .. }
        ));
        assert!(matches!(
            codec_error(BackendId::Nvenc, Operation::Open, cuda(2)),
            CodecError::BackendBusy { .. }
        ));
    }

    /// Tests that report a lost device (which marks the shared context lost)
    /// run one at a time.
    static SHARED_CONTEXT: Mutex<()> = Mutex::new(());

    #[test]
    fn a_lost_device_hands_the_stream_over() {
        let _shared = SHARED_CONTEXT
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        // CUDA_ERROR_ILLEGAL_ADDRESS (700) is sticky.
        assert!(matches!(
            codec_error(BackendId::Nvdec, Operation::Decode, cuda(700)),
            CodecError::UnsupportedStream { .. }
        ));
        assert!(matches!(
            codec_error(BackendId::Nvenc, Operation::Encode, cuda(999)),
            CodecError::BackendBusy { .. }
        ));
        // CUDA_ERROR_INVALID_VALUE (1) is not.
        assert!(matches!(
            codec_error(BackendId::Nvenc, Operation::Encode, cuda(1)),
            CodecError::Backend { .. }
        ));
    }

    #[test]
    fn a_lost_context_is_replaced_and_the_last_holder_releases_it() {
        let _shared = SHARED_CONTEXT
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let Ok(first) = context() else {
            eprintln!("skipping: no usable NVIDIA driver");
            return;
        };
        assert!(
            Arc::ptr_eq(&first, &context().unwrap()),
            "shared while held"
        );
        assert!(first.is_primary());
        // A sticky error seen on it (as an encoder or decoder would report).
        assert!(first.note_error(&cuda(700)).is_some());
        assert!(first.is_lost());
        let second = context().unwrap();
        assert!(
            !Arc::ptr_eq(&first, &second),
            "a lost context is not handed out"
        );
        assert!(
            !second.is_primary(),
            "the lost primary is still held: a context of its own"
        );
        assert!(!second.is_lost());
        drop((first, second));
        // Nobody holds one: the primary context is retained afresh.
        let third = context().unwrap();
        assert!(third.is_primary());
        assert!(!third.is_lost());
    }
}
