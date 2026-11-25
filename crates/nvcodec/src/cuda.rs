//! The safe CUDA layer: devices, the primary context, streams, events and
//! pitched device memory with 2D copies.
//!
//! The CUDA driver API is thread-safe, but which context a call acts on is
//! per-thread state (the current-context stack). Every call here therefore
//! runs inside a [`ContextGuard`], which pushes the object's context and
//! pops it on drop, so objects can move between threads freely and other
//! CUDA users in the process (the renderer, another decoder) keep their own
//! current context.
//!
//! Every object that must be destroyed on a context holds an `Arc` of its
//! [`Context`], so the context cannot be released under a live buffer,
//! stream or event, whatever the drop order.

use std::ffi::{CStr, c_char, c_uint};
use std::fmt;
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::{CheckStatus as _, CudaStatus, Error, Result};
use crate::loader::CudaLibrary;
use crate::sys::cuda as sys;

/// A CUDA device (a GPU).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device {
    raw: sys::CUdevice,
}

impl Device {
    /// How many CUDA devices the driver exposes.
    ///
    /// # Errors
    ///
    /// Loading errors ([`CudaLibrary::get`]) and [`Error::Cuda`].
    pub fn count() -> Result<u32> {
        let cuda = CudaLibrary::get()?;
        let mut count = 0;
        // SAFETY: the out-pointer is a valid local.
        unsafe { (cuda.device_get_count)(&raw mut count) }.check("cuDeviceGetCount")?;
        Ok(count.unsigned_abs())
    }

    /// Every CUDA device, by ordinal.
    ///
    /// # Errors
    ///
    /// As [`Device::count`] and [`Device::get`].
    pub fn all() -> Result<Vec<Self>> {
        (0..Self::count()?).map(Self::get).collect()
    }

    /// The device with this ordinal.
    ///
    /// # Errors
    ///
    /// Loading errors, [`Error::DeviceOutOfRange`], [`Error::Cuda`].
    pub fn get(ordinal: u32) -> Result<Self> {
        let count = Self::count()?;
        let in_range = i32::try_from(ordinal).ok().filter(|_| ordinal < count);
        let ordinal_c = in_range.ok_or(Error::DeviceOutOfRange { ordinal, count })?;
        let cuda = CudaLibrary::get()?;
        let mut raw = 0;
        // SAFETY: the out-pointer is a valid local, the ordinal in range.
        unsafe { (cuda.device_get)(&raw mut raw, ordinal_c) }.check("cuDeviceGet")?;
        Ok(Self { raw })
    }

    /// The device at this PCI address.
    ///
    /// # Errors
    ///
    /// Loading errors and [`Error::Cuda`] (`CUDA_ERROR_INVALID_DEVICE` when
    /// no CUDA device sits there).
    pub fn by_pci_bus_id(bus_id: PciBusId) -> Result<Self> {
        let cuda = CudaLibrary::get()?;
        let text = format!("{bus_id}\0");
        let mut raw = 0;
        // SAFETY: valid out-pointer; `text` is NUL-terminated and outlives
        // the call.
        unsafe { (cuda.device_get_by_pci_bus_id)(&raw mut raw, text.as_ptr().cast()) }
            .check("cuDeviceGetByPCIBusId")?;
        Ok(Self { raw })
    }

    /// The driver's handle.
    #[must_use]
    pub const fn raw(self) -> sys::CUdevice {
        self.raw
    }

    /// The marketing name, e.g. `NVIDIA GeForce RTX 3060`.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`], or [`Error::UnexpectedValue`] for a name without a
    /// terminator.
    pub fn name(self) -> Result<String> {
        let mut buffer: [c_char; 256] = [0; 256];
        let length = i32::try_from(buffer.len()).unwrap_or(i32::MAX);
        let cuda = CudaLibrary::get()?;
        // SAFETY: the buffer holds `length` bytes.
        unsafe { (cuda.device_get_name)(buffer.as_mut_ptr(), length, self.raw) }
            .check("cuDeviceGetName")?;
        c_string(&buffer, "cuDeviceGetName", "a device name")
    }

    /// The PCI address.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`], or [`Error::UnexpectedValue`] for text that is not
    /// a PCI address.
    pub fn pci_bus_id(self) -> Result<PciBusId> {
        // "dddddddd:bb:dd.f" plus the terminator, with room to spare.
        let mut buffer: [c_char; 32] = [0; 32];
        let length = i32::try_from(buffer.len()).unwrap_or(i32::MAX);
        let cuda = CudaLibrary::get()?;
        // SAFETY: the buffer holds `length` bytes.
        unsafe { (cuda.device_get_pci_bus_id)(buffer.as_mut_ptr(), length, self.raw) }
            .check("cuDeviceGetPCIBusId")?;
        let text = c_string(&buffer, "cuDeviceGetPCIBusId", "a PCI bus id")?;
        text.parse().map_err(|_| Error::UnexpectedValue {
            call: "cuDeviceGetPCIBusId",
            what: "a PCI bus id",
            value: text,
        })
    }

    /// The device UUID, stable across reboots and driver reloads.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`].
    pub fn uuid(self) -> Result<DeviceUuid> {
        let mut uuid = sys::CUuuid::default();
        let cuda = CudaLibrary::get()?;
        // SAFETY: valid out-pointer.
        unsafe { (cuda.device_get_uuid)(&raw mut uuid, self.raw) }.check("cuDeviceGetUuid")?;
        Ok(DeviceUuid(uuid.bytes.map(byte_of)))
    }

    /// The compute capability (SM version), e.g. 8.6 for an RTX 3060.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`].
    pub fn compute_capability(self) -> Result<ComputeCapability> {
        let major =
            self.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor =
            self.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        Ok(ComputeCapability {
            major: major.unsigned_abs(),
            minor: minor.unsigned_abs(),
        })
    }

    /// Any device attribute (`cuDeviceGetAttribute`). The binding only names
    /// a subset; others can be passed by value (`CUdevice_attribute(n)`).
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`] (`CUDA_ERROR_INVALID_VALUE` for an attribute the
    /// driver does not know).
    pub fn attribute(self, attribute: sys::CUdevice_attribute) -> Result<i32> {
        let mut value = 0;
        let cuda = CudaLibrary::get()?;
        // SAFETY: valid out-pointer.
        unsafe { (cuda.device_get_attribute)(&raw mut value, attribute, self.raw) }
            .check("cuDeviceGetAttribute")?;
        Ok(value)
    }

    /// Total device memory in bytes.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`].
    pub fn total_memory(self) -> Result<usize> {
        let mut bytes = 0;
        let cuda = CudaLibrary::get()?;
        // SAFETY: valid out-pointer.
        unsafe { (cuda.device_total_mem)(&raw mut bytes, self.raw) }
            .check("cuDeviceTotalMem_v2")?;
        Ok(bytes)
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Device({})", self.raw)
    }
}

/// A C `char` as a byte; `char` is signed on x86 and unsigned on Arm.
fn byte_of(c: c_char) -> u8 {
    u8::from_ne_bytes(c.to_ne_bytes())
}

/// Reads a NUL-terminated string the driver wrote into `buffer`.
fn c_string(buffer: &[c_char], call: &'static str, what: &'static str) -> Result<String> {
    let bytes: Vec<u8> = buffer.iter().copied().map(byte_of).collect();
    CStr::from_bytes_until_nul(&bytes)
        .map(|text| text.to_string_lossy().into_owned())
        .map_err(|_| Error::UnexpectedValue {
            call,
            what,
            value: String::from_utf8_lossy(&bytes).into_owned(),
        })
}

/// A PCI address (`domain:bus:device.function`), the key to match a CUDA
/// device with a DRM or Vulkan device of the same GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PciBusId {
    /// PCI domain (segment).
    pub domain: u32,
    /// Bus number.
    pub bus: u8,
    /// Device (slot) number.
    pub device: u8,
    /// Function number.
    pub function: u8,
}

/// Text that is not a `domain:bus:device.function` PCI address.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0:?} is not a PCI address (expected domain:bus:device.function in hex)")]
pub struct ParsePciBusIdError(String);

impl FromStr for PciBusId {
    type Err = ParsePciBusIdError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let invalid = || ParsePciBusIdError(text.to_owned());
        let (domain, rest) = text.split_once(':').ok_or_else(invalid)?;
        let (bus, rest) = rest.split_once(':').ok_or_else(invalid)?;
        let (device, function) = rest.split_once('.').ok_or_else(invalid)?;
        Ok(Self {
            domain: u32::from_str_radix(domain, 16).map_err(|_| invalid())?,
            bus: u8::from_str_radix(bus, 16).map_err(|_| invalid())?,
            device: u8::from_str_radix(device, 16).map_err(|_| invalid())?,
            function: u8::from_str_radix(function, 16).map_err(|_| invalid())?,
        })
    }
}

impl fmt::Display for PciBusId {
    /// The sysfs spelling, e.g. `0000:01:00.0`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04x}:{:02x}:{:02x}.{:x}",
            self.domain, self.bus, self.device, self.function
        )
    }
}

/// A device UUID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceUuid(pub [u8; 16]);

impl fmt::Display for DeviceUuid {
    /// The `nvidia-smi -L` spelling: `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("GPU")?;
        self.0.iter().enumerate().try_for_each(|(index, byte)| {
            if matches!(index, 0 | 4 | 6 | 8 | 10) {
                f.write_str("-")?;
            }
            write!(f, "{byte:02x}")
        })
    }
}

/// A CUDA compute capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeCapability {
    /// Major (architecture generation).
    pub major: u32,
    /// Minor.
    pub minor: u32,
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// Free and total device memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryInfo {
    /// Free bytes.
    pub free: usize,
    /// Total bytes.
    pub total: usize,
}

/// A CUDA context: a device's primary context, retained for as long as this
/// value lives, or (only to replace a lost primary context that others still
/// hold) a context of its own.
///
/// The primary context is the one context per device that the CUDA runtime,
/// other libraries and other parts of ferricast share, so retaining it never
/// creates a competing context (each costs hundreds of MB and is not
/// interoperable with the others' memory).
pub struct Context {
    raw: NonNull<sys::CUctx_st>,
    device: Device,
    cuda: &'static CudaLibrary,
    kind: ContextKind,
    /// Set once a sticky error showed the context is unusable.
    lost: AtomicBool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContextKind {
    /// Retained with `cuDevicePrimaryCtxRetain`, released on drop.
    Primary,
    /// Created with `cuCtxCreate`, destroyed on drop.
    Owned,
}

// SAFETY: a `CUcontext` has no thread affinity: the driver API is
// thread-safe and every use pushes the context on the calling thread
// (`ContextGuard`), as NVIDIA documents for sharing a context between
// threads. `Context` exposes no interior state besides the handle and an
// atomic flag.
unsafe impl Send for Context {}
// SAFETY: as for `Send`; a context may be current on several threads at
// once, so concurrent `&Context` use is allowed by the driver.
unsafe impl Sync for Context {}

impl Context {
    /// Retains `device`'s primary context.
    ///
    /// # Errors
    ///
    /// Loading errors and [`Error::Cuda`].
    pub fn primary(device: Device) -> Result<Arc<Self>> {
        let cuda = CudaLibrary::get()?;
        let mut raw = ptr::null_mut();
        // SAFETY: valid out-pointer and device handle.
        unsafe { (cuda.device_primary_ctx_retain)(&raw mut raw, device.raw) }
            .check("cuDevicePrimaryCtxRetain")?;
        let Some(raw) = NonNull::new(raw) else {
            return Err(Error::UnexpectedValue {
                call: "cuDevicePrimaryCtxRetain",
                what: "a context",
                value: "null".into(),
            });
        };
        Ok(Arc::new(Self {
            raw,
            device,
            cuda,
            kind: ContextKind::Primary,
            lost: AtomicBool::new(false),
        }))
    }

    /// Creates a context of its own on `device`, destroyed on drop. Only for
    /// replacing a lost primary context while something still holds it
    /// (memory of one context cannot be used on another); otherwise use
    /// [`Context::primary`].
    ///
    /// # Errors
    ///
    /// Loading errors and [`Error::Cuda`].
    pub fn create(device: Device) -> Result<Arc<Self>> {
        let cuda = CudaLibrary::get()?;
        let mut raw = ptr::null_mut();
        // SAFETY: valid out-pointer and device handle; flags 0 (automatic
        // scheduling).
        unsafe { (cuda.ctx_create)(&raw mut raw, 0, device.raw) }.check("cuCtxCreate_v2")?;
        let Some(raw) = NonNull::new(raw) else {
            return Err(Error::UnexpectedValue {
                call: "cuCtxCreate_v2",
                what: "a context",
                value: "null".into(),
            });
        };
        // `cuCtxCreate` makes the new context current; every use pushes it.
        let mut popped = ptr::null_mut();
        // SAFETY: the context was just pushed on this thread; valid
        // out-pointer.
        let popped_status = unsafe { (cuda.ctx_pop_current)(&raw mut popped) };
        if let Err(error) = popped_status.check("cuCtxPopCurrent_v2") {
            tracing::warn!(%error, "popping a new CUDA context");
        }
        Ok(Arc::new(Self {
            raw,
            device,
            cuda,
            kind: ContextKind::Owned,
            lost: AtomicBool::new(false),
        }))
    }

    /// The device.
    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Whether this is the device's primary context.
    #[must_use]
    pub fn is_primary(&self) -> bool {
        self.kind == ContextKind::Primary
    }

    /// The driver's handle, for APIs that take a `CUcontext` (NVENC, NVDEC).
    #[must_use]
    pub fn raw(&self) -> sys::CUcontext {
        self.raw.as_ptr()
    }

    /// The handle as a non-null pointer, for APIs that take a context as
    /// `void*` (NVENC).
    #[must_use]
    pub fn as_non_null(&self) -> NonNull<std::ffi::c_void> {
        self.raw.cast()
    }

    /// Whether a sticky error was seen on the context ([`Context::note_error`]):
    /// nothing on it works any more, and new work belongs on a new context.
    #[must_use]
    pub fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    /// Records `error` if it says the context is unusable
    /// ([`Error::is_device_lost`]). For an NVENC or NVDEC failure, which
    /// does not say, the context is asked (`cuStreamQuery` on its default
    /// stream, which returns a sticky error without waiting).
    ///
    /// Returns [`Error::DeviceLost`] when the context is (now) lost, so the
    /// caller can hand the failure over as such.
    pub fn note_error(&self, error: &Error) -> Option<Error> {
        let lost = match error {
            Error::DeviceLost { call, status } => Some((*call, *status)),
            Error::Cuda { call, status } | Error::Cuvid { call, status } if status.is_sticky() => {
                Some((*call, *status))
            }
            Error::Nvenc { .. } | Error::NvencSessionLimit { .. } | Error::Cuvid { .. } => {
                self.sticky_status().map(|status| ("cuStreamQuery", status))
            }
            _ => None,
        };
        let (call, status) = lost?;
        if !self.lost.swap(true, Ordering::AcqRel) {
            tracing::warn!(%status, call, device = ?self.device, "the CUDA context is lost");
        }
        Some(Error::DeviceLost { call, status })
    }

    /// The sticky error the context is in, if any (`cuStreamQuery` on the
    /// legacy default stream: no wait).
    fn sticky_status(&self) -> Option<CudaStatus> {
        let _current = self.guard().ok()?;
        // SAFETY: the context is current; the null stream is its default
        // stream.
        let status = CudaStatus(unsafe { (self.cuda.stream_query)(ptr::null_mut()) });
        status.is_sticky().then_some(status)
    }

    /// Makes the context current on this thread until the guard drops.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`] if the push fails.
    pub fn guard(&self) -> Result<ContextGuard<'_>> {
        // SAFETY: `raw` is a retained context.
        unsafe { (self.cuda.ctx_push_current)(self.raw.as_ptr()) }.check("cuCtxPushCurrent_v2")?;
        Ok(ContextGuard {
            context: self,
            _not_send: PhantomData,
        })
    }

    /// Waits for all work on the context.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`], including errors of earlier asynchronous work.
    pub fn synchronize(&self) -> Result<()> {
        let _current = self.guard()?;
        // SAFETY: the context is current.
        unsafe { (self.cuda.ctx_synchronize)() }.check("cuCtxSynchronize")
    }

    /// Free and total memory of the device.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`].
    pub fn memory_info(&self) -> Result<MemoryInfo> {
        let _current = self.guard()?;
        let (mut free, mut total) = (0, 0);
        // SAFETY: the context is current; valid out-pointers.
        unsafe { (self.cuda.mem_get_info)(&raw mut free, &raw mut total) }
            .check("cuMemGetInfo_v2")?;
        Ok(MemoryInfo { free, total })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let released = match self.kind {
            ContextKind::Primary => {
                // SAFETY: this value retained the primary context once.
                unsafe { (self.cuda.device_primary_ctx_release)(self.device.raw) }
                    .check("cuDevicePrimaryCtxRelease")
            }
            ContextKind::Owned => {
                // SAFETY: this value created the context and nothing uses it
                // any more (every user holds the `Arc`).
                unsafe { (self.cuda.ctx_destroy)(self.raw.as_ptr()) }.check("cuCtxDestroy_v2")
            }
        };
        if let Err(error) = released {
            tracing::warn!(%error, kind = ?self.kind, "releasing a CUDA context");
        }
    }
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context")
            .field("raw", &self.raw)
            .field("device", &self.device)
            .field("kind", &self.kind)
            .field("lost", &self.is_lost())
            .finish_non_exhaustive()
    }
}

/// Keeps a [`Context`] current on this thread; pops it on drop.
///
/// Not `Send`: the current-context stack is per thread, so the pop must
/// happen on the thread that pushed.
#[must_use = "the context is only current while the guard lives"]
pub struct ContextGuard<'a> {
    context: &'a Context,
    _not_send: PhantomData<*const ()>,
}

impl ContextGuard<'_> {
    /// The context made current.
    #[must_use]
    pub fn context(&self) -> &Context {
        self.context
    }
}

impl Drop for ContextGuard<'_> {
    fn drop(&mut self) {
        let mut popped = ptr::null_mut();
        // SAFETY: this guard pushed a context on this thread (it is not
        // `Send`), so the stack is not empty; valid out-pointer.
        let result = unsafe { (self.context.cuda.ctx_pop_current)(&raw mut popped) }
            .check("cuCtxPopCurrent_v2");
        match result {
            Err(error) => tracing::warn!(%error, "popping the CUDA context"),
            Ok(()) if popped != self.context.raw.as_ptr() => tracing::warn!(
                expected = ?self.context.raw,
                ?popped,
                "popped a CUDA context this guard did not push: unbalanced push/pop on this thread"
            ),
            Ok(()) => {}
        }
    }
}

impl fmt::Debug for ContextGuard<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContextGuard")
            .field("context", &self.context.raw)
            .finish()
    }
}

/// A CUDA stream: work queued on it runs in order, concurrently with other
/// streams. Created non-blocking, so it never synchronises with the legacy
/// default stream other CUDA users may queue on.
pub struct Stream {
    raw: sys::CUstream,
    context: Arc<Context>,
}

// SAFETY: a `CUstream` has no thread affinity; every call pushes the owning
// context first.
unsafe impl Send for Stream {}
// SAFETY: the driver serialises concurrent submissions to one stream;
// `&Stream` only queues work or waits.
unsafe impl Sync for Stream {}

impl Stream {
    /// A new non-blocking stream on `context`.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`].
    pub fn new(context: &Arc<Context>) -> Result<Self> {
        let _current = context.guard()?;
        let mut raw = ptr::null_mut();
        // SAFETY: the context is current; valid out-pointer.
        unsafe { (context.cuda.stream_create)(&raw mut raw, sys::CU_STREAM_NON_BLOCKING) }
            .check("cuStreamCreate")?;
        Ok(Self {
            raw,
            context: Arc::clone(context),
        })
    }

    /// The driver's handle.
    #[must_use]
    pub fn raw(&self) -> sys::CUstream {
        self.raw
    }

    /// The owning context.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Waits until all queued work has run.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`], including errors of the queued work.
    pub fn synchronize(&self) -> Result<()> {
        let _current = self.context.guard()?;
        // SAFETY: live stream, its context current.
        unsafe { (self.context.cuda.stream_synchronize)(self.raw) }.check("cuStreamSynchronize")
    }

    /// Whether all queued work has run, without waiting.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`], including errors of the queued work.
    pub fn is_idle(&self) -> Result<bool> {
        let _current = self.context.guard()?;
        // SAFETY: live stream, its context current.
        let status = unsafe { (self.context.cuda.stream_query)(self.raw) };
        not_ready_as_false(status, "cuStreamQuery")
    }

    /// Makes work queued after this call wait until `event` completes.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`].
    pub fn wait(&self, event: &Event) -> Result<()> {
        let _current = self.context.guard()?;
        // SAFETY: live stream and event, the stream's context current.
        unsafe {
            (self.context.cuda.stream_wait_event)(self.raw, event.raw, sys::CU_EVENT_WAIT_DEFAULT)
        }
        .check("cuStreamWaitEvent")
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // Work still queued completes before the driver frees the stream.
        let destroyed = self.context.guard().and_then(|_current| {
            // SAFETY: live stream, its context current; not used again.
            unsafe { (self.context.cuda.stream_destroy)(self.raw) }.check("cuStreamDestroy_v2")
        });
        if let Err(error) = destroyed {
            tracing::warn!(%error, "destroying a CUDA stream");
        }
    }
}

impl fmt::Debug for Stream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stream")
            .field("raw", &self.raw)
            .finish_non_exhaustive()
    }
}

/// `CUDA_ERROR_NOT_READY` means "not yet": `Ok(false)`.
fn not_ready_as_false(status: sys::CUresult, call: &'static str) -> Result<bool> {
    if CudaStatus(status) == CudaStatus::NOT_READY {
        Ok(false)
    } else {
        status.check(call).map(|()| true)
    }
}

/// A CUDA event: a marker in a stream that other streams or the host can
/// wait for. Created without timing, which makes it cheaper to record and
/// wait on.
pub struct Event {
    raw: sys::CUevent,
    context: Arc<Context>,
}

// SAFETY: a `CUevent` has no thread affinity; every call pushes the owning
// context first.
unsafe impl Send for Event {}
// SAFETY: the driver serialises concurrent record / query / wait calls on one
// event.
unsafe impl Sync for Event {}

impl Event {
    /// A new event on `context`.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`].
    pub fn new(context: &Arc<Context>) -> Result<Self> {
        let _current = context.guard()?;
        let mut raw = ptr::null_mut();
        // SAFETY: the context is current; valid out-pointer.
        unsafe { (context.cuda.event_create)(&raw mut raw, sys::CU_EVENT_DISABLE_TIMING) }
            .check("cuEventCreate")?;
        Ok(Self {
            raw,
            context: Arc::clone(context),
        })
    }

    /// The driver's handle.
    #[must_use]
    pub fn raw(&self) -> sys::CUevent {
        self.raw
    }

    /// Marks the current end of `stream`'s queue; the event completes when
    /// the stream gets there.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`] (also when the stream belongs to another context).
    pub fn record(&self, stream: &Stream) -> Result<()> {
        let _current = self.context.guard()?;
        // SAFETY: live event and stream, the event's context current.
        unsafe { (self.context.cuda.event_record)(self.raw, stream.raw) }.check("cuEventRecord")
    }

    /// Waits on the host until the event completes.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`], including errors of the work before the event.
    pub fn synchronize(&self) -> Result<()> {
        let _current = self.context.guard()?;
        // SAFETY: live event, its context current.
        unsafe { (self.context.cuda.event_synchronize)(self.raw) }.check("cuEventSynchronize")
    }

    /// Whether the event has completed, without waiting. An event never
    /// recorded counts as complete.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`].
    pub fn is_complete(&self) -> Result<bool> {
        let _current = self.context.guard()?;
        // SAFETY: live event, its context current.
        let status = unsafe { (self.context.cuda.event_query)(self.raw) };
        not_ready_as_false(status, "cuEventQuery")
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        let destroyed = self.context.guard().and_then(|_current| {
            // SAFETY: live event, its context current; not used again.
            unsafe { (self.context.cuda.event_destroy)(self.raw) }.check("cuEventDestroy_v2")
        });
        if let Err(error) = destroyed {
            tracing::warn!(%error, "destroying a CUDA event");
        }
    }
}

impl fmt::Debug for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Event")
            .field("raw", &self.raw)
            .finish_non_exhaustive()
    }
}

/// A position inside a 2D buffer: a byte offset in the row and a row index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Point {
    /// Byte offset from the start of the row.
    pub x_bytes: usize,
    /// Row index.
    pub row: usize,
}

/// A rectangle inside a 2D buffer, in bytes per row and rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Region {
    /// Top-left corner.
    pub origin: Point,
    /// Bytes per row.
    pub width_bytes: usize,
    /// Number of rows.
    pub rows: usize,
}

impl Region {
    /// A `width_bytes` × `rows` region at the origin.
    #[must_use]
    pub const fn new(width_bytes: usize, rows: usize) -> Self {
        Self {
            origin: Point { x_bytes: 0, row: 0 },
            width_bytes,
            rows,
        }
    }

    /// The same size at `origin`.
    #[must_use]
    pub const fn at(self, origin: Point) -> Self {
        Self { origin, ..self }
    }

    const fn is_empty(self) -> bool {
        self.width_bytes == 0 || self.rows == 0
    }

    /// Checks that the region lies inside a `width_bytes` × `rows` buffer.
    fn check_within(self, side: &'static str, width_bytes: usize, rows: usize) -> Result<()> {
        let right = self.origin.x_bytes.checked_add(self.width_bytes);
        let bottom = self.origin.row.checked_add(self.rows);
        let (Some(right), Some(bottom)) = (right, bottom) else {
            return Err(Error::SizeOverflow {
                what: "a copy region",
            });
        };
        if right > width_bytes {
            return Err(Error::CopyOutOfBounds {
                side,
                unit: "bytes per row",
                needed: right,
                available: width_bytes,
            });
        }
        if bottom > rows {
            return Err(Error::CopyOutOfBounds {
                side,
                unit: "rows",
                needed: bottom,
                available: rows,
            });
        }
        Ok(())
    }

    /// Checks that host memory of `length` bytes with rows `pitch` apart
    /// holds a region of this size starting at its first byte.
    fn check_host(self, side: &'static str, length: usize, pitch: usize) -> Result<()> {
        if pitch < self.width_bytes {
            return Err(Error::PitchTooSmall {
                pitch,
                width_bytes: self.width_bytes,
            });
        }
        let Some(last_row) = self.rows.checked_sub(1) else {
            return Ok(());
        };
        let needed = last_row
            .checked_mul(pitch)
            .and_then(|bytes| bytes.checked_add(self.width_bytes))
            .ok_or(Error::SizeOverflow {
                what: "a host image size",
            })?;
        if needed > length {
            return Err(Error::CopyOutOfBounds {
                side,
                unit: "bytes",
                needed,
                available: length,
            });
        }
        Ok(())
    }
}

/// Host pixels read by a copy: rows `pitch` bytes apart, starting at the
/// first byte of `data`.
#[derive(Clone, Copy, Debug)]
pub struct HostImage<'a> {
    /// The pixels.
    pub data: &'a [u8],
    /// Bytes from one row to the next.
    pub pitch: usize,
}

impl HostImage<'_> {
    /// Copies its first `rows` rows of `width_bytes` into `dst`, packed
    /// (`width_bytes` apart).
    ///
    /// # Errors
    ///
    /// [`Error::PitchTooSmall`] for a pitch under `width_bytes`,
    /// [`Error::CopyOutOfBounds`] when either side is too short.
    pub(crate) fn copy_rows_into(
        &self,
        dst: &mut [u8],
        width_bytes: usize,
        rows: usize,
    ) -> Result<()> {
        if rows == 0 || width_bytes == 0 {
            return Ok(());
        }
        if self.pitch < width_bytes {
            return Err(Error::PitchTooSmall {
                pitch: self.pitch,
                width_bytes,
            });
        }
        let needed = self.pitch * (rows - 1) + width_bytes;
        let out_of_bounds = |side, needed, available| Error::CopyOutOfBounds {
            side,
            unit: "bytes",
            needed,
            available,
        };
        if self.data.len() < needed {
            return Err(out_of_bounds("source", needed, self.data.len()));
        }
        if dst.len() < width_bytes * rows {
            return Err(out_of_bounds("destination", width_bytes * rows, dst.len()));
        }
        if self.pitch == width_bytes {
            dst[..needed].copy_from_slice(&self.data[..needed]);
        } else {
            for (target, source) in dst
                .chunks_exact_mut(width_bytes)
                .zip(self.data.chunks(self.pitch))
                .take(rows)
            {
                target.copy_from_slice(&source[..width_bytes]);
            }
        }
        Ok(())
    }
}

/// Host pixels written by a copy: rows `pitch` bytes apart, starting at the
/// first byte of `data`.
#[derive(Debug)]
pub struct HostImageMut<'a> {
    /// The pixels.
    pub data: &'a mut [u8],
    /// Bytes from one row to the next.
    pub pitch: usize,
}

/// A CUDA array owned elsewhere (an EGL image of a DMA-BUF mapped through
/// graphics interop), lent for `'a`: the owner keeps it mapped, and its
/// context alive, at least that long. NVENC takes one as input
/// (`NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY`).
#[derive(Debug, Clone, Copy)]
pub struct ArrayView<'a> {
    raw: sys::CUarray,
    context: &'a Arc<Context>,
    width: u32,
    height: u32,
}

impl<'a> ArrayView<'a> {
    /// Lends `raw`.
    ///
    /// # Safety
    ///
    /// `raw` must be a CUDA array of `context`, at least `width` × `height`
    /// elements, that stays allocated (and, for an interop array, mapped)
    /// for all of `'a`, and past it until the GPU work queued on it while
    /// lent has completed. A consumer that keeps a registration of it
    /// (NVENC's [`Encoder::encode_array`](crate::encode::Encoder::encode_array)
    /// cache) extends that to the end of the registration.
    #[must_use]
    pub const unsafe fn from_raw(
        raw: sys::CUarray,
        context: &'a Arc<Context>,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            raw,
            context,
            width,
            height,
        }
    }

    /// The raw handle, valid for `'a`.
    #[must_use]
    pub const fn raw(self) -> sys::CUarray {
        self.raw
    }

    /// The context the array belongs to.
    #[must_use]
    pub const fn context(self) -> &'a Arc<Context> {
        self.context
    }

    /// Width in elements (pixels).
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Height in rows.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }
}

/// Page-locked host memory (`cuMemHostAlloc`): the copy engine reads and
/// writes it directly, so copies to and from it run asynchronously and at
/// full bus speed (pageable memory is staged through a driver buffer and
/// makes the copy synchronous). Freed on drop. Not zero-filled.
pub struct PinnedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    context: Arc<Context>,
}

// SAFETY: the buffer owns its allocation exclusively; page-locked host memory
// has no thread affinity, and `cuMemFreeHost` may run on any thread.
unsafe impl Send for PinnedBuffer {}
// SAFETY: shared access only reads the bytes.
unsafe impl Sync for PinnedBuffer {}

impl PinnedBuffer {
    /// Allocates `len` bytes (at least one) of page-locked memory, usable
    /// from every context (`CU_MEMHOSTALLOC_PORTABLE`).
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`] (`CUDA_ERROR_OUT_OF_MEMORY`).
    pub fn new(context: &Arc<Context>, len: usize) -> Result<Self> {
        /// `CU_MEMHOSTALLOC_PORTABLE`.
        const PORTABLE: c_uint = 0x01;
        let _current = context.guard()?;
        let mut ptr = ptr::null_mut();
        // SAFETY: the context is current; valid out-pointer.
        unsafe { (context.cuda.mem_host_alloc)(&raw mut ptr, len.max(1), PORTABLE) }
            .check("cuMemHostAlloc")?;
        let Some(ptr) = NonNull::new(ptr.cast::<u8>()) else {
            return Err(Error::UnexpectedValue {
                call: "cuMemHostAlloc",
                what: "a host pointer",
                value: "null".into(),
            });
        };
        Ok(Self {
            ptr,
            len,
            context: Arc::clone(context),
        })
    }

    /// The context it was allocated on.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// The bytes. Reading them while an asynchronous copy into them is
    /// queued reads whatever the copy engine has written so far.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `len` bytes allocated by `cuMemHostAlloc` and owned by
        // this value; bytes are always initialised for `u8` purposes
        // (the allocation is plain host memory).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// The bytes, writable.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as in `as_slice`, and `&mut self` makes the access
        // exclusive on the host side.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Copies regions of `src` into this buffer, queued on `stream` (after
    /// `after` completes, when given) and waited for once: each lands at
    /// [`PinnedCopy::offset`] with rows [`PinnedCopy::pitch`] apart.
    ///
    /// # Errors
    ///
    /// [`Error::CopyOutOfBounds`] / [`Error::PitchTooSmall`] for a region
    /// outside either side, [`Error::Cuda`]. The copies queued before a
    /// failure are waited for too.
    pub fn download(
        &mut self,
        src: &DeviceBuffer,
        copies: &[PinnedCopy],
        stream: &Stream,
        after: Option<&Event>,
    ) -> Result<()> {
        let queued = after
            .map_or(Ok(()), |event| stream.wait(event))
            .and_then(|()| {
                copies.iter().try_for_each(|copy| {
                    let data = self
                        .as_mut_slice()
                        .get_mut(copy.offset..)
                        .unwrap_or_default();
                    // SAFETY: this buffer is borrowed mutably until the stream
                    // is synchronised below, so nothing reads, writes or frees
                    // it while the copy engine writes it.
                    unsafe {
                        src.download_async(
                            copy.src,
                            HostImageMut {
                                data,
                                pitch: copy.pitch,
                            },
                            stream,
                        )
                    }
                })
            });
        let waited = stream.synchronize();
        queued.and(waited)
    }
}

/// One copy of [`PinnedBuffer::download`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PinnedCopy {
    /// The region of the device buffer.
    pub src: Region,
    /// Where its first row lands in the pinned buffer.
    pub offset: usize,
    /// Bytes from one row to the next in the pinned buffer.
    pub pitch: usize,
}

impl AsRef<[u8]> for PinnedBuffer {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        let cuda = self.context.cuda;
        let freed = self.context.guard().and_then(|_current| {
            // SAFETY: this value owns the allocation and it is not used
            // again; every copy into it was waited for by its user.
            unsafe { (cuda.mem_free_host)(self.ptr.as_ptr().cast()) }.check("cuMemFreeHost")
        });
        if let Err(error) = freed {
            tracing::warn!(%error, "freeing page-locked host memory");
        }
    }
}

impl fmt::Debug for PinnedBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PinnedBuffer")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

/// A pitched 2D allocation in device memory (`cuMemAllocPitch`): `rows` rows
/// of `width_bytes`, each starting `pitch` bytes after the previous one, the
/// pitch chosen by the driver for coalesced access. Freed on drop, with its
/// context current.
///
/// Rust's aliasing rules do not reach device memory, so `&mut self` on the
/// writing methods documents intent rather than preventing GPU-side races:
/// work queued on different streams must be ordered with [`Event`]s.
pub struct DeviceBuffer {
    ptr: sys::CUdeviceptr,
    pitch: usize,
    width_bytes: usize,
    rows: usize,
    context: Arc<Context>,
    allocation: Arc<Allocation>,
}

/// The memory behind a [`DeviceBuffer`], freed when the last holder goes:
/// the buffer, or someone that registered it with a driver and must
/// unregister it before it is freed ([`DeviceBuffer::keep_allocated`]).
pub(crate) struct Allocation {
    ptr: sys::CUdeviceptr,
    context: Arc<Context>,
}

impl Drop for Allocation {
    fn drop(&mut self) {
        let cuda = self.context.cuda;
        let freed = self.context.guard().and_then(|_current| {
            // SAFETY: the last holder of the allocation is gone, its context
            // is current. `cuMemFree` is not stream-ordered: it waits for
            // queued work that uses the memory.
            unsafe { (cuda.mem_free)(self.ptr) }.check("cuMemFree_v2")
        });
        if let Err(error) = freed {
            tracing::warn!(%error, "freeing a CUDA device buffer");
        }
    }
}

// SAFETY: the buffer owns its allocation exclusively; a device pointer has no
// thread affinity and every call pushes the owning context.
unsafe impl Send for DeviceBuffer {}
// SAFETY: `&DeviceBuffer` only reads device memory (as a copy source) or
// exposes the pointer; host memory is never shared through it.
unsafe impl Sync for DeviceBuffer {}

/// The widest access the pitch is optimised for (`cuMemAllocPitch` accepts
/// 4, 8 or 16): 16-byte vector loads, which also gives the 256-byte (or
/// larger) row alignment NVENC and NVDEC surfaces want.
const PITCH_ELEMENT_BYTES: c_uint = 16;

impl DeviceBuffer {
    /// Allocates `rows` rows of `width_bytes`.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`] (`CUDA_ERROR_OUT_OF_MEMORY`, `CUDA_ERROR_INVALID_VALUE`
    /// for a zero size).
    pub fn pitched(context: &Arc<Context>, width_bytes: usize, rows: usize) -> Result<Self> {
        let _current = context.guard()?;
        let mut ptr = 0;
        let mut pitch = 0;
        // SAFETY: the context is current; valid out-pointers.
        unsafe {
            (context.cuda.mem_alloc_pitch)(
                &raw mut ptr,
                &raw mut pitch,
                width_bytes,
                rows,
                PITCH_ELEMENT_BYTES,
            )
        }
        .check("cuMemAllocPitch_v2")?;
        Ok(Self {
            ptr,
            pitch,
            width_bytes,
            rows,
            context: Arc::clone(context),
            allocation: Arc::new(Allocation {
                ptr,
                context: Arc::clone(context),
            }),
        })
    }

    /// The device address of row 0.
    #[must_use]
    pub fn device_ptr(&self) -> sys::CUdeviceptr {
        self.ptr
    }

    /// Bytes from one row to the next.
    #[must_use]
    pub fn pitch(&self) -> usize {
        self.pitch
    }

    /// Usable bytes per row.
    #[must_use]
    pub fn width_bytes(&self) -> usize {
        self.width_bytes
    }

    /// Number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Keeps the memory allocated past the buffer's drop, until the
    /// returned handle goes too: for a registration (NVENC) that must be
    /// undone before the memory is freed. Does not count as a holder of the
    /// buffer (`Arc::strong_count` of an `Arc<DeviceBuffer>` is unchanged).
    pub(crate) fn keep_allocated(&self) -> Arc<Allocation> {
        Arc::clone(&self.allocation)
    }

    /// The whole buffer as a copy region.
    #[must_use]
    pub fn region(&self) -> Region {
        Region::new(self.width_bytes, self.rows)
    }

    /// The owning context.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Copies `src` into `dst` of this buffer, ordered after all work
    /// queued on the buffer's context (on any stream), and waits for the
    /// copy to complete: work queued afterwards, on any stream, sees it.
    ///
    /// Synchronises the whole context twice (the copy runs on the legacy
    /// default stream, which the crate's non-blocking streams do not order
    /// against): for set-up and tests. Pipelines use
    /// [`DeviceBuffer::upload_async`] on their own stream.
    ///
    /// # Errors
    ///
    /// [`Error::CopyOutOfBounds`] / [`Error::PitchTooSmall`] when either
    /// side is too small, [`Error::Cuda`] (including failures of earlier
    /// work on the context).
    pub fn upload(&mut self, dst: Region, src: HostImage<'_>) -> Result<()> {
        let copy = self.upload_params(dst, src)?;
        self.context.synchronize()?;
        self.run(copy.as_ref(), None)?;
        // A copy from pageable memory may return once the source is staged.
        self.context.synchronize()
    }

    /// Queues the copy of `src` into `dst` of this buffer on `stream`.
    ///
    /// # Errors
    ///
    /// As [`DeviceBuffer::upload`].
    ///
    /// # Safety
    ///
    /// `src` must stay valid and unmodified until `stream` has executed the
    /// copy (e.g. until [`Stream::synchronize`]). The driver stages pageable
    /// memory before returning, but reads page-locked memory (registered or
    /// allocated by any CUDA user in the process) asynchronously.
    pub unsafe fn upload_async(
        &mut self,
        dst: Region,
        src: HostImage<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let copy = self.upload_params(dst, src)?;
        self.run(copy.as_ref(), Some(stream))
    }

    /// Copies `src` of this buffer into `dst`, waiting for completion.
    ///
    /// # Errors
    ///
    /// As [`DeviceBuffer::upload`].
    pub fn download(&self, src: Region, mut dst: HostImageMut<'_>) -> Result<()> {
        let copy = self.download_params(src, &mut dst)?;
        self.run(copy.as_ref(), None)
    }

    /// Queues the copy of `src` of this buffer into `dst` on `stream`.
    ///
    /// # Errors
    ///
    /// As [`DeviceBuffer::upload`].
    ///
    /// # Safety
    ///
    /// `dst` must stay valid, and be neither read nor written, until
    /// `stream` has executed the copy: page-locked destinations are written
    /// asynchronously.
    pub unsafe fn download_async(
        &self,
        src: Region,
        mut dst: HostImageMut<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let copy = self.download_params(src, &mut dst)?;
        self.run(copy.as_ref(), Some(stream))
    }

    /// Copies `src_region` of `src` to `dst` in this buffer, waiting for
    /// completion.
    ///
    /// # Errors
    ///
    /// [`Error::CopyOutOfBounds`], [`Error::Cuda`].
    pub fn copy_from(&mut self, dst: Point, src: &DeviceBuffer, src_region: Region) -> Result<()> {
        let copy = self.device_params(dst, src, src_region)?;
        self.run(copy.as_ref(), None)
    }

    /// Queues the copy of `src_region` of `src` to `dst` in this buffer on
    /// `stream`. Safe: both sides are device memory owned by buffers whose
    /// drop frees through `cuMemFree`, which waits for work using them.
    ///
    /// # Errors
    ///
    /// As [`DeviceBuffer::copy_from`].
    pub fn copy_from_async(
        &mut self,
        dst: Point,
        src: &DeviceBuffer,
        src_region: Region,
        stream: &Stream,
    ) -> Result<()> {
        let copy = self.device_params(dst, src, src_region)?;
        self.run(copy.as_ref(), Some(stream))
    }

    /// Sets every byte of the buffer to `value`, ordered after all work
    /// queued on the buffer's context (on any stream), and waits for it to
    /// complete: work queued afterwards, on any stream, sees it.
    ///
    /// Synchronises the whole context twice (a memset of device memory runs
    /// on the legacy default stream and returns before it completes): for
    /// set-up and tests. Pipelines use [`DeviceBuffer::fill_async`].
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`] (including failures of earlier work on the context).
    pub fn fill(&mut self, value: u8) -> Result<()> {
        let cuda = self.context.cuda;
        let _current = self.context.guard()?;
        self.context.synchronize()?;
        // SAFETY: the allocation spans `rows` rows of `pitch` bytes, of which
        // `width_bytes` are set; its context is current.
        unsafe { (cuda.memset_d2d8)(self.ptr, self.pitch, value, self.width_bytes, self.rows) }
            .check("cuMemsetD2D8_v2")?;
        self.context.synchronize()
    }

    /// Queues setting every byte of the buffer to `value` on `stream`.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`].
    pub fn fill_async(&mut self, value: u8, stream: &Stream) -> Result<()> {
        let cuda = self.context.cuda;
        let _current = self.context.guard()?;
        // SAFETY: as in `fill`; the stream is live.
        unsafe {
            (cuda.memset_d2d8_async)(
                self.ptr,
                self.pitch,
                value,
                self.width_bytes,
                self.rows,
                stream.raw,
            )
        }
        .check("cuMemsetD2D8Async")
    }

    fn upload_params(&self, dst: Region, src: HostImage<'_>) -> Result<Option<sys::CUDA_MEMCPY2D>> {
        if dst.is_empty() {
            return Ok(None);
        }
        dst.check_within("destination", self.width_bytes, self.rows)?;
        dst.check_host("source", src.data.len(), src.pitch)?;
        Ok(Some(sys::CUDA_MEMCPY2D {
            srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_HOST,
            srcHost: src.data.as_ptr().cast(),
            srcPitch: src.pitch,
            ..self.device_side(Side::Destination, dst)
        }))
    }

    fn download_params(
        &self,
        src: Region,
        dst: &mut HostImageMut<'_>,
    ) -> Result<Option<sys::CUDA_MEMCPY2D>> {
        if src.is_empty() {
            return Ok(None);
        }
        src.check_within("source", self.width_bytes, self.rows)?;
        src.check_host("destination", dst.data.len(), dst.pitch)?;
        Ok(Some(sys::CUDA_MEMCPY2D {
            dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_HOST,
            dstHost: dst.data.as_mut_ptr().cast(),
            dstPitch: dst.pitch,
            ..self.device_side(Side::Source, src)
        }))
    }

    fn device_params(
        &self,
        dst: Point,
        src: &DeviceBuffer,
        src_region: Region,
    ) -> Result<Option<sys::CUDA_MEMCPY2D>> {
        if src_region.is_empty() {
            return Ok(None);
        }
        src_region.check_within("source", src.width_bytes, src.rows)?;
        let dst_region = src_region.at(dst);
        dst_region.check_within("destination", self.width_bytes, self.rows)?;
        let source = src.device_side(Side::Source, src_region);
        Ok(Some(sys::CUDA_MEMCPY2D {
            srcXInBytes: source.srcXInBytes,
            srcY: source.srcY,
            srcMemoryType: source.srcMemoryType,
            srcDevice: source.srcDevice,
            srcPitch: source.srcPitch,
            ..self.device_side(Side::Destination, dst_region)
        }))
    }

    /// A copy description with this buffer on one side and the size of
    /// `region`; the other side is left for the caller to fill.
    fn device_side(&self, side: Side, region: Region) -> sys::CUDA_MEMCPY2D {
        let mut copy = sys::CUDA_MEMCPY2D {
            WidthInBytes: region.width_bytes,
            Height: region.rows,
            ..Default::default()
        };
        match side {
            Side::Source => {
                copy.srcXInBytes = region.origin.x_bytes;
                copy.srcY = region.origin.row;
                copy.srcMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_DEVICE;
                copy.srcDevice = self.ptr;
                copy.srcPitch = self.pitch;
            }
            Side::Destination => {
                copy.dstXInBytes = region.origin.x_bytes;
                copy.dstY = region.origin.row;
                copy.dstMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_DEVICE;
                copy.dstDevice = self.ptr;
                copy.dstPitch = self.pitch;
            }
        }
        copy
    }

    /// Runs a validated copy, synchronously or queued on `stream`. `None` is
    /// an empty copy.
    fn run(&self, copy: Option<&sys::CUDA_MEMCPY2D>, stream: Option<&Stream>) -> Result<()> {
        let Some(copy) = copy else {
            return Ok(());
        };
        let cuda = self.context.cuda;
        let _current = self.context.guard()?;
        match stream {
            // SAFETY: both sides were bounds-checked against their
            // allocations; host memory is borrowed for the call (sync) or
            // kept alive by the caller (async, an `unsafe fn` contract).
            None => unsafe { (cuda.memcpy_2d)(copy) }.check("cuMemcpy2D_v2"),
            Some(stream) => {
                // SAFETY: as above; the stream is live.
                unsafe { (cuda.memcpy_2d_async)(copy, stream.raw) }.check("cuMemcpy2DAsync_v2")
            }
        }
    }
}

/// Which side of a copy a buffer is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Source,
    Destination,
}

impl fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("ptr", &format_args!("{:#x}", self.ptr))
            .field("pitch", &self.pitch)
            .field("width_bytes", &self.width_bytes)
            .field("rows", &self.rows)
            .finish_non_exhaustive()
    }
}

/// Pitched device memory owned elsewhere (a linear EGL image of a DMA-BUF),
/// lent for `'a`: `rows` rows of `width_bytes`, `pitch` bytes apart. NVENC
/// takes one as input (`NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR`).
#[derive(Debug, Clone, Copy)]
pub struct PitchedView<'a> {
    ptr: sys::CUdeviceptr,
    pitch: usize,
    width_bytes: usize,
    rows: usize,
    context: &'a Arc<Context>,
}

impl<'a> PitchedView<'a> {
    /// Lends `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must be device memory of `context` holding `rows` rows of
    /// `width_bytes` bytes, `pitch` bytes apart (`pitch >= width_bytes`),
    /// that stays allocated (and, for interop memory, mapped) for all of
    /// `'a`, and past it until the GPU work queued on it while lent has
    /// completed. A consumer that keeps a registration of it (NVENC's
    /// [`Encoder::encode_pitched`](crate::encode::Encoder::encode_pitched)
    /// cache) extends that to the end of the registration.
    #[must_use]
    pub const unsafe fn from_raw(
        ptr: sys::CUdeviceptr,
        pitch: usize,
        width_bytes: usize,
        rows: usize,
        context: &'a Arc<Context>,
    ) -> Self {
        Self {
            ptr,
            pitch,
            width_bytes,
            rows,
            context,
        }
    }

    /// The device address of row 0, valid for `'a`.
    #[must_use]
    pub const fn device_ptr(self) -> sys::CUdeviceptr {
        self.ptr
    }

    /// Bytes from one row to the next.
    #[must_use]
    pub const fn pitch(self) -> usize {
        self.pitch
    }

    /// Usable bytes per row.
    #[must_use]
    pub const fn width_bytes(self) -> usize {
        self.width_bytes
    }

    /// Number of rows.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// The context the memory belongs to.
    #[must_use]
    pub const fn context(self) -> &'a Arc<Context> {
        self.context
    }
}

/// Copies from borrowed CUDA arrays and pitched views into a [`DeviceBuffer`].
///
/// Queued copies stay safe after the view's lifetime ends because owners of
/// lent memory synchronise before freeing it (the DMA-BUF importer waits for
/// the context before unregistering an image).
impl DeviceBuffer {
    /// Queues the copy of `src_region` of the CUDA array `src`, whose
    /// elements are `element_bytes` wide, to `dst` in this buffer on
    /// `stream`. The region is in bytes per row and rows.
    ///
    /// # Errors
    ///
    /// [`Error::CopyOutOfBounds`] when the region leaves either side
    /// (`element_bytes` only feeds this check: the driver bounds-checks
    /// arrays itself), [`Error::SizeOverflow`], [`Error::Cuda`].
    pub fn copy_from_array_async(
        &mut self,
        dst: Point,
        src: ArrayView<'_>,
        element_bytes: usize,
        src_region: Region,
        stream: &Stream,
    ) -> Result<()> {
        if src_region.is_empty() {
            return Ok(());
        }
        let src_width_bytes = usize::try_from(src.width())
            .ok()
            .and_then(|width| width.checked_mul(element_bytes))
            .ok_or(Error::SizeOverflow {
                what: "a CUDA array row",
            })?;
        let src_rows = usize::try_from(src.height()).map_err(|_| Error::SizeOverflow {
            what: "a CUDA array height",
        })?;
        src_region.check_within("source", src_width_bytes, src_rows)?;
        let dst_region = src_region.at(dst);
        dst_region.check_within("destination", self.width_bytes, self.rows)?;
        let copy = sys::CUDA_MEMCPY2D {
            srcXInBytes: src_region.origin.x_bytes,
            srcY: src_region.origin.row,
            srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_ARRAY,
            srcArray: src.raw(),
            ..self.device_side(Side::Destination, dst_region)
        };
        self.run(Some(&copy), Some(stream))
    }

    /// Queues the copy of `src_region` of the pitched memory `src` to `dst`
    /// in this buffer on `stream`.
    ///
    /// # Errors
    ///
    /// [`Error::CopyOutOfBounds`], [`Error::SizeOverflow`], [`Error::Cuda`].
    pub fn copy_from_pitched_async(
        &mut self,
        dst: Point,
        src: PitchedView<'_>,
        src_region: Region,
        stream: &Stream,
    ) -> Result<()> {
        if src_region.is_empty() {
            return Ok(());
        }
        src_region.check_within("source", src.width_bytes, src.rows)?;
        let dst_region = src_region.at(dst);
        dst_region.check_within("destination", self.width_bytes, self.rows)?;
        let copy = sys::CUDA_MEMCPY2D {
            srcXInBytes: src_region.origin.x_bytes,
            srcY: src_region.origin.row,
            srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
            srcDevice: src.ptr,
            srcPitch: src.pitch,
            ..self.device_side(Side::Destination, dst_region)
        };
        self.run(Some(&copy), Some(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pci_bus_id_round_trip() {
        let id: PciBusId = "0000:01:00.0".parse().unwrap();
        assert_eq!(
            id,
            PciBusId {
                domain: 0,
                bus: 1,
                device: 0,
                function: 0
            }
        );
        assert_eq!(id.to_string(), "0000:01:00.0");
        let id: PciBusId = "00000000:AF:1F.7".parse().unwrap();
        assert_eq!(id.to_string(), "0000:af:1f.7");
        assert!("01:00.0".parse::<PciBusId>().is_err());
        assert!("0000:01:00".parse::<PciBusId>().is_err());
        assert!("0000:100:00.0".parse::<PciBusId>().is_err());
    }

    #[test]
    fn uuid_display() {
        let uuid = DeviceUuid([
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
            0xcd, 0xef,
        ]);
        assert_eq!(uuid.to_string(), "GPU-12345678-9abc-def0-0123-456789abcdef");
    }

    #[test]
    fn region_bounds() {
        let region = Region::new(64, 4).at(Point {
            x_bytes: 16,
            row: 2,
        });
        assert!(region.check_within("destination", 80, 6).is_ok());
        assert!(matches!(
            region.check_within("destination", 79, 6),
            Err(Error::CopyOutOfBounds {
                unit: "bytes per row",
                needed: 80,
                available: 79,
                ..
            })
        ));
        assert!(matches!(
            region.check_within("destination", 80, 5),
            Err(Error::CopyOutOfBounds { unit: "rows", .. })
        ));
        let huge = Region::new(usize::MAX, 1).at(Point { x_bytes: 1, row: 0 });
        assert!(matches!(
            huge.check_within("source", usize::MAX, 1),
            Err(Error::SizeOverflow { .. })
        ));
    }

    #[test]
    fn host_bounds() {
        let region = Region::new(10, 3);
        // The last row needs only its width, not a whole pitch.
        assert!(region.check_host("source", 2 * 16 + 10, 16).is_ok());
        assert!(matches!(
            region.check_host("source", 2 * 16 + 9, 16),
            Err(Error::CopyOutOfBounds {
                unit: "bytes",
                needed: 42,
                available: 41,
                ..
            })
        ));
        assert!(matches!(
            region.check_host("source", 100, 9),
            Err(Error::PitchTooSmall {
                pitch: 9,
                width_bytes: 10
            })
        ));
    }

    #[test]
    fn objects_cross_threads() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<Context>();
        send_sync::<Stream>();
        send_sync::<Event>();
        send_sync::<DeviceBuffer>();
    }
}

/// The module / launch / texture entry points: every driver has them, but
/// the loader resolves them as optional so that only their users fail
/// without them.
#[derive(Clone, Copy)]
struct ModuleApi {
    load_data_ex: crate::loader::Required<sys::tcuModuleLoadDataEx>,
    unload: crate::loader::Required<sys::tcuModuleUnload>,
    get_function: crate::loader::Required<sys::tcuModuleGetFunction>,
    launch_kernel: crate::loader::Required<sys::tcuLaunchKernel>,
    tex_object_create: crate::loader::Required<sys::tcuTexObjectCreate>,
    tex_object_destroy: crate::loader::Required<sys::tcuTexObjectDestroy>,
    array_get_descriptor: crate::loader::Required<sys::tcuArrayGetDescriptor_v2>,
}

impl ModuleApi {
    fn get(cuda: &CudaLibrary) -> Result<Self> {
        fn need<F>(entry: Option<F>, symbol: &'static str) -> Result<F> {
            entry.ok_or_else(|| Error::SymbolMissing {
                library: CudaLibrary::file_name(),
                symbol,
                reason: "not exported by this driver".into(),
            })
        }
        Ok(Self {
            load_data_ex: need(cuda.module_load_data_ex, "cuModuleLoadDataEx")?,
            unload: need(cuda.module_unload, "cuModuleUnload")?,
            get_function: need(cuda.module_get_function, "cuModuleGetFunction")?,
            launch_kernel: need(cuda.launch_kernel, "cuLaunchKernel")?,
            tex_object_create: need(cuda.tex_object_create, "cuTexObjectCreate")?,
            tex_object_destroy: need(cuda.tex_object_destroy, "cuTexObjectDestroy")?,
            array_get_descriptor: need(cuda.array_get_descriptor, "cuArrayGetDescriptor_v2")?,
        })
    }
}

/// A module of device code, JIT-compiled from PTX by the driver for the
/// context's GPU, unloaded on drop.
pub struct Module {
    raw: NonNull<sys::CUmod_st>,
    api: ModuleApi,
    context: Arc<Context>,
}

// SAFETY: a `CUmodule` has no thread affinity; every call pushes the owning
// context first.
unsafe impl Send for Module {}
// SAFETY: function lookups and launches on one module are thread-safe in the
// driver API; `&Module` exposes nothing else.
unsafe impl Sync for Module {}

/// A kernel of a [`Module`]: a handle valid while the module is loaded,
/// launched through [`Module::launch`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Function {
    raw: sys::CUfunction,
    module: sys::CUmodule,
}

// SAFETY: a function handle has no thread affinity; it is only used through
// `Module::launch`, which pushes the module's context.
unsafe impl Send for Function {}
// SAFETY: as for `Send`; the handle is never mutated.
unsafe impl Sync for Function {}

/// Grid and block dimensions of a kernel launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    /// Blocks per grid (x, y, z).
    pub grid: [u32; 3],
    /// Threads per block (x, y, z).
    pub block: [u32; 3],
    /// Dynamic shared memory per block, in bytes.
    pub shared_bytes: u32,
}

/// How much of the JIT compiler's error log is kept.
const JIT_LOG_BYTES: usize = 4096;

impl Module {
    /// JIT-compiles `ptx` (a NUL-terminated PTX image) on `context`.
    ///
    /// # Errors
    ///
    /// [`Error::KernelLoad`] with the compiler's log when the driver
    /// rejects the PTX (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` from a driver
    /// older than its ISA), [`Error::SymbolMissing`].
    pub fn load_ptx(context: &Arc<Context>, ptx: &CStr) -> Result<Self> {
        let api = ModuleApi::get(context.cuda)?;
        let _current = context.guard()?;
        let mut log = vec![0_u8; JIT_LOG_BYTES];
        let mut options = [
            sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER,
            sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        ];
        // The size option is a `void*`-sized slot holding an integer, as
        // the driver API documents for scalar JIT options.
        let mut values = [
            log.as_mut_ptr().cast::<std::ffi::c_void>(),
            ptr::without_provenance_mut(JIT_LOG_BYTES),
        ];
        let mut raw = ptr::null_mut();
        // SAFETY: the context is current; `ptx` is NUL-terminated; the two
        // options name a writable buffer of `JIT_LOG_BYTES` and its size,
        // both alive for the call; valid out-pointer.
        let status = unsafe {
            (api.load_data_ex)(
                &raw mut raw,
                ptx.as_ptr().cast(),
                2,
                options.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        };
        if status != sys::CUresult::CUDA_SUCCESS {
            let end = log.iter().position(|&byte| byte == 0).unwrap_or(log.len());
            return Err(Error::KernelLoad {
                status: CudaStatus(status),
                log: String::from_utf8_lossy(&log[..end]).trim().to_owned(),
            });
        }
        let Some(raw) = NonNull::new(raw) else {
            return Err(Error::UnexpectedValue {
                call: "cuModuleLoadDataEx",
                what: "a module",
                value: "null".into(),
            });
        };
        Ok(Self {
            raw,
            api,
            context: Arc::clone(context),
        })
    }

    /// The kernel named `name` (its `extern "C"` name).
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`] (`CUDA_ERROR_NOT_FOUND` for an unknown name).
    pub fn function(&self, name: &CStr) -> Result<Function> {
        let _current = self.context.guard()?;
        let mut raw = ptr::null_mut();
        // SAFETY: live module, its context current; NUL-terminated name;
        // valid out-pointer.
        unsafe { (self.api.get_function)(&raw mut raw, self.raw.as_ptr(), name.as_ptr()) }
            .check("cuModuleGetFunction")?;
        Ok(Function {
            raw,
            module: self.raw.as_ptr(),
        })
    }

    /// The owning context.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Queues `function` on `stream`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidKernelInput`] for a function of another module or a
    /// stream of another context, [`Error::Cuda`].
    ///
    /// # Safety
    ///
    /// `params` must hold one pointer per kernel parameter, each pointing to
    /// a value of exactly the parameter's type and layout, alive for the
    /// call (the driver copies them before returning). Every memory access
    /// the kernel makes with these parameters must stay inside memory that
    /// is allocated, and not freed or reused, until the kernel has run.
    pub unsafe fn launch(
        &self,
        function: Function,
        launch: Launch,
        stream: &Stream,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        if function.module != self.raw.as_ptr() {
            return Err(Error::InvalidKernelInput {
                reason: "the function belongs to another module".into(),
            });
        }
        if stream.context().raw() != self.context.raw() {
            return Err(Error::InvalidKernelInput {
                reason: "the stream belongs to another CUDA context".into(),
            });
        }
        let [gx, gy, gz] = launch.grid;
        let [bx, by, bz] = launch.block;
        let _current = self.context.guard()?;
        // SAFETY: live function of this module and live stream, the context
        // current; the parameter contract is the caller's.
        unsafe {
            (self.api.launch_kernel)(
                function.raw,
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
                launch.shared_bytes,
                stream.raw(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            )
        }
        .check("cuLaunchKernel")
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // Kernels already queued keep running: the driver defers freeing
        // the module's code until they complete.
        let unloaded = self.context.guard().and_then(|_current| {
            // SAFETY: live module, its context current; not used again.
            unsafe { (self.api.unload)(self.raw.as_ptr()) }.check("cuModuleUnload")
        });
        if let Err(error) = unloaded {
            tracing::warn!(%error, "unloading a CUDA module");
        }
    }
}

impl fmt::Debug for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Module")
            .field("raw", &self.raw)
            .finish_non_exhaustive()
    }
}

/// A texture object reading a CUDA array element by element: point
/// filtering, clamped addressing, unnormalised coordinates, values as
/// stored (no conversion to float). Destroyed on drop.
///
/// The texture does not keep the array alive. Reading through it after the
/// array is freed or unmapped is a use after free; destroying it is not.
pub struct TextureObject {
    raw: sys::CUtexObject,
    destroy: crate::loader::Required<sys::tcuTexObjectDestroy>,
    context: Arc<Context>,
}

// SAFETY: a texture object is a plain handle without thread affinity; every
// call pushes the owning context first.
unsafe impl Send for TextureObject {}
// SAFETY: `&TextureObject` only exposes the handle.
unsafe impl Sync for TextureObject {}

impl TextureObject {
    /// A point-sampled texture over all of `array`.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`], [`Error::SymbolMissing`].
    pub fn point_sampled(array: ArrayView<'_>) -> Result<Self> {
        let context = array.context();
        let api = ModuleApi::get(context.cuda)?;
        let resource = sys::CUDA_RESOURCE_DESC {
            resType: sys::CUresourcetype::CU_RESOURCE_TYPE_ARRAY,
            res: sys::CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
                array: sys::CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
                    hArray: array.raw(),
                },
            },
            flags: 0,
        };
        let clamp = sys::CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP;
        let texture = sys::CUDA_TEXTURE_DESC {
            addressMode: [clamp; 3],
            filterMode: sys::CUfilter_mode::CU_TR_FILTER_MODE_POINT,
            // Integer element values; unnormalised coordinates (no
            // CU_TRSF_NORMALIZED_COORDINATES).
            flags: sys::CU_TRSF_READ_AS_INTEGER,
            maxAnisotropy: 0,
            mipmapFilterMode: sys::CUfilter_mode::CU_TR_FILTER_MODE_POINT,
            mipmapLevelBias: 0.0,
            minMipmapLevelClamp: 0.0,
            maxMipmapLevelClamp: 0.0,
            borderColor: [0.0; 4],
            reserved: [0; 12],
        };
        let _current = context.guard()?;
        let mut raw = 0;
        // SAFETY: the context is current; the descriptors are fully
        // initialised and live for the call; the array is live for `'a`
        // (`ArrayView` contract); a null view descriptor means "the whole
        // array"; valid out-pointer.
        unsafe {
            (api.tex_object_create)(
                &raw mut raw,
                &raw const resource,
                &raw const texture,
                ptr::null(),
            )
        }
        .check("cuTexObjectCreate")?;
        Ok(Self {
            raw,
            destroy: api.tex_object_destroy,
            context: Arc::clone(context),
        })
    }

    /// The handle kernels take (`cudaTextureObject_t`).
    #[must_use]
    pub fn raw(&self) -> sys::CUtexObject {
        self.raw
    }
}

impl Drop for TextureObject {
    fn drop(&mut self) {
        let destroyed = self.context.guard().and_then(|_current| {
            // SAFETY: live texture object, its context current; not used
            // again.
            unsafe { (self.destroy)(self.raw) }.check("cuTexObjectDestroy")
        });
        if let Err(error) = destroyed {
            tracing::warn!(%error, "destroying a CUDA texture object");
        }
    }
}

impl fmt::Debug for TextureObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TextureObject")
            .field("raw", &self.raw)
            .finish_non_exhaustive()
    }
}

/// The element layout of a CUDA array (`cuArrayGetDescriptor`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArrayDescriptor {
    /// Width in elements.
    pub width: usize,
    /// Height in rows.
    pub height: usize,
    /// Channel format.
    pub format: sys::CUarray_format,
    /// Channels per element.
    pub channels: u32,
}

impl ArrayView<'_> {
    /// The array's size and element format, as the driver reports them.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`], [`Error::SymbolMissing`].
    pub fn descriptor(self) -> Result<ArrayDescriptor> {
        let api = ModuleApi::get(self.context.cuda)?;
        let _current = self.context.guard()?;
        let mut descriptor = sys::CUDA_ARRAY_DESCRIPTOR {
            Width: 0,
            Height: 0,
            Format: sys::CUarray_format(0),
            NumChannels: 0,
        };
        // SAFETY: the array is live for `'a`, its context current; valid
        // out-pointer.
        unsafe { (api.array_get_descriptor)(&raw mut descriptor, self.raw) }
            .check("cuArrayGetDescriptor_v2")?;
        Ok(ArrayDescriptor {
            width: descriptor.Width,
            height: descriptor.Height,
            format: descriptor.Format,
            channels: descriptor.NumChannels,
        })
    }
}

impl DeviceBuffer {
    /// The buffer lent as pitched memory, for APIs that take memory owned
    /// elsewhere (image kernels).
    #[must_use]
    pub fn as_view(&self) -> PitchedView<'_> {
        PitchedView {
            ptr: self.ptr,
            pitch: self.pitch,
            width_bytes: self.width_bytes,
            rows: self.rows,
            context: &self.context,
        }
    }
}
