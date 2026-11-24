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

/// A device's primary context, retained for as long as this value lives.
///
/// The primary context is the one context per device that the CUDA runtime,
/// other libraries and other parts of ferricast share, so retaining it never
/// creates a competing context (each costs hundreds of MB and is not
/// interoperable with the others' memory).
pub struct Context {
    raw: NonNull<sys::CUctx_st>,
    device: Device,
    cuda: &'static CudaLibrary,
}

// SAFETY: a `CUcontext` has no thread affinity: the driver API is
// thread-safe and every use pushes the context on the calling thread
// (`ContextGuard`), as NVIDIA documents for sharing a context between
// threads. `Context` exposes no interior state besides the handle.
unsafe impl Send for Context {}
// SAFETY: as for `Send`; a primary context may be current on several threads
// at once, so concurrent `&Context` use is allowed by the driver.
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
        Ok(Arc::new(Self { raw, device, cuda }))
    }

    /// The device.
    #[must_use]
    pub fn device(&self) -> Device {
        self.device
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
        // SAFETY: this value retained the primary context once.
        let released = unsafe { (self.cuda.device_primary_ctx_release)(self.device.raw) }
            .check("cuDevicePrimaryCtxRelease");
        if let Err(error) = released {
            tracing::warn!(%error, "releasing the primary CUDA context");
        }
    }
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context")
            .field("raw", &self.raw)
            .field("device", &self.device)
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
    /// for all of `'a`.
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

    /// Copies `src` into `dst` of this buffer, waiting for completion.
    ///
    /// # Errors
    ///
    /// [`Error::CopyOutOfBounds`] / [`Error::PitchTooSmall`] when either
    /// side is too small, [`Error::Cuda`].
    pub fn upload(&mut self, dst: Region, src: HostImage<'_>) -> Result<()> {
        let copy = self.upload_params(dst, src)?;
        self.run(copy.as_ref(), None)
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

    /// Sets every byte of the buffer to `value`, waiting for completion.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`].
    pub fn fill(&mut self, value: u8) -> Result<()> {
        let cuda = self.context.cuda;
        let _current = self.context.guard()?;
        // SAFETY: the allocation spans `rows` rows of `pitch` bytes, of which
        // `width_bytes` are set; its context is current.
        unsafe { (cuda.memset_d2d8)(self.ptr, self.pitch, value, self.width_bytes, self.rows) }
            .check("cuMemsetD2D8_v2")
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

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        let cuda = self.context.cuda;
        let freed = self.context.guard().and_then(|_current| {
            // SAFETY: this buffer owns the allocation, its context is
            // current, and it is not used again. `cuMemFree` is not
            // stream-ordered: it waits for queued work that uses the memory.
            unsafe { (cuda.mem_free)(self.ptr) }.check("cuMemFree_v2")
        });
        if let Err(error) = freed {
            tracing::warn!(%error, "freeing a CUDA device buffer");
        }
    }
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
    /// `'a`.
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
