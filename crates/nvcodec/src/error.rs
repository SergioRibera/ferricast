//! The crate's error type. Every driver failure names the call that failed
//! and the status the driver returned, by name.

use std::fmt;

use crate::sys;
use crate::version::{ApiRequest, ApiVersion, NvencVersion};

/// Shorthand for results carrying [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Everything that can go wrong talking to the NVIDIA driver.
///
/// `Clone` because the outcome of loading a driver library is cached for the
/// whole process and handed to every caller.
#[derive(Clone, Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A driver library could not be loaded: usually no NVIDIA driver is
    /// installed (or, on Linux, its user-space part is missing).
    #[error("cannot load {library}: {reason}")]
    LibraryMissing {
        /// File name of the library.
        library: &'static str,
        /// The loader's explanation.
        reason: String,
    },
    /// A driver library lacks an entry point this crate requires, which means
    /// the driver predates it.
    #[error("{library} has no {symbol} (driver too old?): {reason}")]
    SymbolMissing {
        /// File name of the library.
        library: &'static str,
        /// The missing entry point.
        symbol: &'static str,
        /// The loader's explanation.
        reason: String,
    },
    /// A CUDA driver API call failed.
    #[error("{call} failed: {status}")]
    Cuda {
        /// The driver entry point.
        call: &'static str,
        /// What it returned.
        status: CudaStatus,
    },
    /// An NVDEC (cuvid) call failed. cuvid reports `CUresult` codes.
    #[error("{call} failed: {status}")]
    Cuvid {
        /// The driver entry point.
        call: &'static str,
        /// What it returned.
        status: CudaStatus,
    },
    /// An NVENC call failed.
    #[error("{call} failed: {status}{}", detail.as_deref().map(|d| format!(" ({d})")).unwrap_or_default())]
    Nvenc {
        /// The NVENC entry point.
        call: &'static str,
        /// What it returned.
        status: NvencStatus,
        /// `nvEncGetLastErrorString` for the session, when there is one.
        detail: Option<String>,
    },
    /// The driver's NVENC API is older than the oldest version this crate
    /// carries bindings for.
    #[error(
        "the NVIDIA driver supports NVENC API {found}, ferricast needs {minimum} or newer: install driver {driver} or newer"
    )]
    DriverTooOld {
        /// The newest NVENC API the driver accepts.
        found: NvencVersion,
        /// The oldest NVENC API this crate supports.
        minimum: ApiVersion,
        /// The driver release that brings `minimum` on this OS.
        driver: &'static str,
    },
    /// A specific NVENC API version was requested (by `FERRICAST_NVENC_API`
    /// or the caller) that the driver does not accept.
    #[error(
        "NVENC API {requested} was requested by {origin}, but the driver supports {driver_max} at most: install driver {driver} or newer"
    )]
    ApiNotSupportedByDriver {
        /// The requested version.
        requested: ApiVersion,
        /// Who requested it.
        origin: ApiRequest,
        /// The newest NVENC API the driver accepts.
        driver_max: NvencVersion,
        /// The driver release that brings `requested` on this OS.
        driver: &'static str,
    },
    /// `FERRICAST_NVENC_API` holds something other than a supported
    /// `<major>.<minor>`.
    #[error("FERRICAST_NVENC_API={value:?} is not one of {supported}")]
    InvalidApiOverride {
        /// The variable's value.
        value: String,
        /// The versions this build supports, for the message.
        supported: SupportedVersions,
    },
    /// The NVENC function list a driver returned lacks an entry point.
    #[error("the NVENC {api} function list has no {function}")]
    NvencFunctionMissing {
        /// The API version the list was requested for.
        api: ApiVersion,
        /// The missing function.
        function: &'static str,
    },
    /// No CUDA device has this ordinal.
    #[error("CUDA device {ordinal} does not exist ({count} devices)")]
    DeviceOutOfRange {
        /// The requested ordinal.
        ordinal: u32,
        /// How many devices the driver reports.
        count: u32,
    },
    /// The driver returned text this crate could not interpret.
    #[error("{call} returned {what} that cannot be parsed: {value:?}")]
    UnexpectedValue {
        /// The driver entry point.
        call: &'static str,
        /// What was being read.
        what: &'static str,
        /// The value as returned.
        value: String,
    },
    /// A 2D copy names a region outside one of its two sides.
    #[error("2D copy out of bounds: {side} needs {needed} {unit}, has {available}")]
    CopyOutOfBounds {
        /// Which side of the copy (`"source"`, `"destination"`).
        side: &'static str,
        /// What is being measured (`"bytes per row"`, `"rows"`, `"bytes"`).
        unit: &'static str,
        /// How much the copy touches.
        needed: usize,
        /// How much there is.
        available: usize,
    },
    /// A host image pitch is smaller than the row it must hold.
    #[error("host pitch {pitch} is smaller than the {width_bytes}-byte rows copied")]
    PitchTooSmall {
        /// The pitch given.
        pitch: usize,
        /// The row width copied.
        width_bytes: usize,
    },
    /// A size computation overflowed `usize`.
    #[error("{what} overflows")]
    SizeOverflow {
        /// What was being computed.
        what: &'static str,
    },
    /// The hardware, driver or API version cannot do what was asked.
    #[error("{what} is not supported: {reason}")]
    Unsupported {
        /// The feature.
        what: &'static str,
        /// Why (which version, GPU or driver lacks it).
        reason: String,
    },
    /// An EGL call failed (DMA-BUF interop).
    #[error("{call} failed: {status}")]
    Egl {
        /// The EGL entry point.
        call: &'static str,
        /// `eglGetError` after the call.
        status: EglStatus,
    },
    /// An operating system call failed (DMA-BUF interop).
    #[error("{call} failed: {}", std::io::Error::from_raw_os_error(*errno))]
    Os {
        /// The system call.
        call: &'static str,
        /// `errno`.
        errno: i32,
    },
    /// No EGL device of the NVIDIA driver drives the CUDA device, so its
    /// DMA-BUFs cannot be imported.
    #[error("no NVIDIA EGL device drives CUDA device {cuda}: {reason}")]
    NoEglDevice {
        /// The CUDA device's PCI address.
        cuda: String,
        /// What was found instead.
        reason: String,
    },
    /// The DMA-BUFs come from a GPU other than the CUDA device (a hybrid
    /// laptop's integrated GPU, a second card): they cannot be imported
    /// without a copy through the CPU.
    #[error(
        "DMA-BUFs of DRM device {drm} ({drm_pci}) cannot be imported into CUDA device {cuda}: different GPUs"
    )]
    DeviceMismatch {
        /// The DRM node the buffers come from.
        drm: String,
        /// Its PCI address, or `"unknown"`.
        drm_pci: String,
        /// The CUDA device's PCI address.
        cuda: String,
    },
    /// A DMA-BUF pixel format the importer does not handle.
    #[error("DRM format {} ({fourcc:#010x}) is not supported for import", fourcc_text(*fourcc))]
    UnsupportedFormat {
        /// The DRM fourcc.
        fourcc: u32,
    },
    /// A DRM format modifier the EGL driver cannot import for this format.
    #[error("DRM modifier {modifier:#018x} is not importable for {}", fourcc_text(*fourcc))]
    UnsupportedModifier {
        /// The DRM fourcc.
        fourcc: u32,
        /// The modifier.
        modifier: u64,
    },
    /// A DMA-BUF description that cannot be right (zero size, stride too
    /// small, plane past the end of the buffer).
    #[error("invalid DMA-BUF: {reason}")]
    InvalidDmaBuf {
        /// What is wrong.
        reason: String,
    },
    /// No DMA-BUF is imported under this key.
    #[error("no DMA-BUF is imported under key {key:#x}")]
    UnknownBuffer {
        /// The caller's key.
        key: u128,
    },
    /// The producer had not finished writing the DMA-BUF in time.
    #[error("the DMA-BUF producer's fence did not signal within {waited:?}")]
    FenceTimeout {
        /// How long the wait lasted.
        waited: std::time::Duration,
    },
    /// Every pooled output frame of a decoder is still held by the caller, so
    /// the next picture has nowhere to go.
    #[error(
        "all {capacity} decoded frames are still held: drop or drain some before decoding more"
    )]
    FramesInFlight {
        /// The decoder's limit on frames handed out and not yet dropped.
        capacity: usize,
    },
    /// A packet is malformed in a way the driver would not survive, so it was
    /// not passed on.
    #[error("invalid packet: {reason}")]
    InvalidPacket {
        /// What is wrong with it.
        reason: String,
    },
    /// Code this crate runs inside a driver callback panicked. A bug in this
    /// crate; the panic was caught before it could unwind into the driver.
    #[error("the NVDEC {callback} callback panicked")]
    CallbackPanicked {
        /// Which parser callback.
        callback: &'static str,
    },
    /// The driver refused another NVENC session, most likely because the
    /// GPU's limit on concurrent sessions (consumer GPUs have one) is reached.
    /// Fall back to another encoder or retry once a session closes.
    #[error(
        "{call} failed: {status}{}: NVENC session limit reached?",
        detail.as_deref().map(|d| format!(" ({d})")).unwrap_or_default()
    )]
    NvencSessionLimit {
        /// The NVENC entry point.
        call: &'static str,
        /// What it returned.
        status: NvencStatus,
        /// `nvEncGetLastErrorString`, when there is one.
        detail: Option<String>,
    },
    /// An encoder configuration that cannot work on any GPU.
    #[error("invalid encoder configuration ({field}): {reason}")]
    InvalidEncodeConfig {
        /// The offending field.
        field: &'static str,
        /// What is wrong with it.
        reason: String,
    },
    /// A picture that does not match the encoder it is given to.
    #[error("invalid encoder input: {reason}")]
    InvalidEncodeInput {
        /// What does not match.
        reason: String,
    },
    /// A picture refers to a reference picture the decoder does not have
    /// (lost with an earlier packet, or sent before the first key frame),
    /// so it was not passed to NVDEC, which faults on those instead of
    /// failing. Decoding resumes at the next key frame.
    #[error("missing reference: {reason}")]
    MissingReference {
        /// Which picture and which reference.
        reason: String,
    },
}

/// A DRM fourcc as its four characters, for messages.
fn fourcc_text(fourcc: u32) -> String {
    fourcc
        .to_le_bytes()
        .iter()
        .map(|&byte| {
            if byte.is_ascii_graphic() {
                char::from(byte)
            } else {
                '?'
            }
        })
        .collect()
}

/// An EGL error code (`eglGetError`), shown by name.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct EglStatus(pub i32);

impl EglStatus {
    /// `EGL_SUCCESS`.
    pub const SUCCESS: Self = Self(0x3000);

    /// The enumerator name from `egl.h` / `eglext.h`, when known.
    #[must_use]
    pub fn name(self) -> Option<&'static str> {
        /// `EGL_SUCCESS` (0x3000) to `EGL_CONTEXT_LOST` (0x300E), in order.
        const CORE: [&str; 15] = [
            "EGL_SUCCESS",
            "EGL_NOT_INITIALIZED",
            "EGL_BAD_ACCESS",
            "EGL_BAD_ALLOC",
            "EGL_BAD_ATTRIBUTE",
            "EGL_BAD_CONFIG",
            "EGL_BAD_CONTEXT",
            "EGL_BAD_CURRENT_SURFACE",
            "EGL_BAD_DISPLAY",
            "EGL_BAD_MATCH",
            "EGL_BAD_NATIVE_PIXMAP",
            "EGL_BAD_NATIVE_WINDOW",
            "EGL_BAD_PARAMETER",
            "EGL_BAD_SURFACE",
            "EGL_CONTEXT_LOST",
        ];
        match self.0 {
            0x322B => Some("EGL_BAD_DEVICE_EXT"),
            code => code
                .checked_sub(Self::SUCCESS.0)
                .and_then(|offset| usize::try_from(offset).ok())
                .and_then(|index| CORE.get(index))
                .copied(),
        }
    }
}

impl fmt::Display for EglStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(name) => write!(f, "{name} ({:#x})", self.0),
            None => write!(f, "EGL error {:#x}", self.0),
        }
    }
}

impl fmt::Debug for EglStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// The list of NVENC API versions this build carries, for messages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SupportedVersions;

impl fmt::Display for SupportedVersions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ApiVersion::ALL
            .iter()
            .enumerate()
            .try_for_each(|(index, version)| {
                if index > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{version}")
            })
    }
}

/// A `CUresult` returned by the CUDA driver or NVDEC, shown by name.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaStatus(pub sys::cuda::CUresult);

impl CudaStatus {
    /// `CUDA_ERROR_NOT_READY`: the queried work has not completed yet.
    pub const NOT_READY: Self = Self(sys::cuda::CUresult::CUDA_ERROR_NOT_READY);

    /// The numeric code.
    #[must_use]
    pub const fn code(self) -> u32 {
        self.0.0
    }

    /// The `CUresult` enumerator name from NVIDIA's `cuda.h`, when known.
    #[must_use]
    pub fn name(self) -> Option<&'static str> {
        CUDA_RESULT_NAMES
            .binary_search_by_key(&self.code(), |&(code, _)| code)
            .ok()
            .and_then(|index| CUDA_RESULT_NAMES.get(index))
            .map(|&(_, name)| name)
    }
}

impl fmt::Display for CudaStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(name) => write!(f, "{name} ({})", self.code()),
            None => write!(f, "CUresult {}", self.code()),
        }
    }
}

impl fmt::Debug for CudaStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// `CUresult` names from NVIDIA's `cuda.h` (FFmpeg's `dynlink_cuda.h` only
/// declares five of them), sorted by code for the binary search.
const CUDA_RESULT_NAMES: &[(u32, &str)] = &[
    (0, "CUDA_SUCCESS"),
    (1, "CUDA_ERROR_INVALID_VALUE"),
    (2, "CUDA_ERROR_OUT_OF_MEMORY"),
    (3, "CUDA_ERROR_NOT_INITIALIZED"),
    (4, "CUDA_ERROR_DEINITIALIZED"),
    (5, "CUDA_ERROR_PROFILER_DISABLED"),
    (6, "CUDA_ERROR_PROFILER_NOT_INITIALIZED"),
    (7, "CUDA_ERROR_PROFILER_ALREADY_STARTED"),
    (8, "CUDA_ERROR_PROFILER_ALREADY_STOPPED"),
    (34, "CUDA_ERROR_STUB_LIBRARY"),
    (46, "CUDA_ERROR_DEVICE_UNAVAILABLE"),
    (100, "CUDA_ERROR_NO_DEVICE"),
    (101, "CUDA_ERROR_INVALID_DEVICE"),
    (102, "CUDA_ERROR_DEVICE_NOT_LICENSED"),
    (200, "CUDA_ERROR_INVALID_IMAGE"),
    (201, "CUDA_ERROR_INVALID_CONTEXT"),
    (202, "CUDA_ERROR_CONTEXT_ALREADY_CURRENT"),
    (205, "CUDA_ERROR_MAP_FAILED"),
    (206, "CUDA_ERROR_UNMAP_FAILED"),
    (207, "CUDA_ERROR_ARRAY_IS_MAPPED"),
    (208, "CUDA_ERROR_ALREADY_MAPPED"),
    (209, "CUDA_ERROR_NO_BINARY_FOR_GPU"),
    (210, "CUDA_ERROR_ALREADY_ACQUIRED"),
    (211, "CUDA_ERROR_NOT_MAPPED"),
    (212, "CUDA_ERROR_NOT_MAPPED_AS_ARRAY"),
    (213, "CUDA_ERROR_NOT_MAPPED_AS_POINTER"),
    (214, "CUDA_ERROR_ECC_UNCORRECTABLE"),
    (215, "CUDA_ERROR_UNSUPPORTED_LIMIT"),
    (216, "CUDA_ERROR_CONTEXT_ALREADY_IN_USE"),
    (217, "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED"),
    (218, "CUDA_ERROR_INVALID_PTX"),
    (219, "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT"),
    (220, "CUDA_ERROR_NVLINK_UNCORRECTABLE"),
    (221, "CUDA_ERROR_JIT_COMPILER_NOT_FOUND"),
    (222, "CUDA_ERROR_UNSUPPORTED_PTX_VERSION"),
    (223, "CUDA_ERROR_JIT_COMPILATION_DISABLED"),
    (224, "CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY"),
    (225, "CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC"),
    (300, "CUDA_ERROR_INVALID_SOURCE"),
    (301, "CUDA_ERROR_FILE_NOT_FOUND"),
    (302, "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND"),
    (303, "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED"),
    (304, "CUDA_ERROR_OPERATING_SYSTEM"),
    (400, "CUDA_ERROR_INVALID_HANDLE"),
    (401, "CUDA_ERROR_ILLEGAL_STATE"),
    (402, "CUDA_ERROR_LOSSY_QUERY"),
    (500, "CUDA_ERROR_NOT_FOUND"),
    (600, "CUDA_ERROR_NOT_READY"),
    (700, "CUDA_ERROR_ILLEGAL_ADDRESS"),
    (701, "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES"),
    (702, "CUDA_ERROR_LAUNCH_TIMEOUT"),
    (703, "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING"),
    (704, "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED"),
    (705, "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED"),
    (708, "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE"),
    (709, "CUDA_ERROR_CONTEXT_IS_DESTROYED"),
    (710, "CUDA_ERROR_ASSERT"),
    (711, "CUDA_ERROR_TOO_MANY_PEERS"),
    (712, "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED"),
    (713, "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED"),
    (714, "CUDA_ERROR_HARDWARE_STACK_ERROR"),
    (715, "CUDA_ERROR_ILLEGAL_INSTRUCTION"),
    (716, "CUDA_ERROR_MISALIGNED_ADDRESS"),
    (717, "CUDA_ERROR_INVALID_ADDRESS_SPACE"),
    (718, "CUDA_ERROR_INVALID_PC"),
    (719, "CUDA_ERROR_LAUNCH_FAILED"),
    (720, "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE"),
    (800, "CUDA_ERROR_NOT_PERMITTED"),
    (801, "CUDA_ERROR_NOT_SUPPORTED"),
    (802, "CUDA_ERROR_SYSTEM_NOT_READY"),
    (803, "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH"),
    (804, "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE"),
    (805, "CUDA_ERROR_MPS_CONNECTION_FAILED"),
    (806, "CUDA_ERROR_MPS_RPC_FAILURE"),
    (807, "CUDA_ERROR_MPS_SERVER_NOT_READY"),
    (808, "CUDA_ERROR_MPS_MAX_CLIENTS_REACHED"),
    (809, "CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED"),
    (810, "CUDA_ERROR_MPS_CLIENT_TERMINATED"),
    (811, "CUDA_ERROR_CDP_NOT_SUPPORTED"),
    (812, "CUDA_ERROR_CDP_VERSION_MISMATCH"),
    (900, "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED"),
    (901, "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED"),
    (902, "CUDA_ERROR_STREAM_CAPTURE_MERGE"),
    (903, "CUDA_ERROR_STREAM_CAPTURE_UNMATCHED"),
    (904, "CUDA_ERROR_STREAM_CAPTURE_UNJOINED"),
    (905, "CUDA_ERROR_STREAM_CAPTURE_ISOLATION"),
    (906, "CUDA_ERROR_STREAM_CAPTURE_IMPLICIT"),
    (907, "CUDA_ERROR_CAPTURED_EVENT"),
    (908, "CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD"),
    (909, "CUDA_ERROR_TIMEOUT"),
    (910, "CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE"),
    (911, "CUDA_ERROR_EXTERNAL_DEVICE"),
    (912, "CUDA_ERROR_INVALID_CLUSTER_SIZE"),
    (913, "CUDA_ERROR_FUNCTION_NOT_LOADED"),
    (914, "CUDA_ERROR_INVALID_RESOURCE_TYPE"),
    (915, "CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION"),
    (999, "CUDA_ERROR_UNKNOWN"),
];

/// An `NVENCSTATUS` returned by NVENC, shown by name. The numeric values
/// are the same in every API version this crate supports.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NvencStatus(pub u32);

macro_rules! nvenc_status_names {
    ($($name:ident),* $(,)?) => {
        impl NvencStatus {
            /// The `NVENCSTATUS` enumerator name from `nvEncodeAPI.h`, when
            /// known.
            #[must_use]
            pub fn name(self) -> Option<&'static str> {
                use sys::nvenc::v13_1::NVENCSTATUS;
                match self.0 {
                    $(code if code == NVENCSTATUS::$name.0 => Some(stringify!($name)),)*
                    _ => None,
                }
            }
        }
    };
}

nvenc_status_names!(
    NV_ENC_SUCCESS,
    NV_ENC_ERR_NO_ENCODE_DEVICE,
    NV_ENC_ERR_UNSUPPORTED_DEVICE,
    NV_ENC_ERR_INVALID_ENCODERDEVICE,
    NV_ENC_ERR_INVALID_DEVICE,
    NV_ENC_ERR_DEVICE_NOT_EXIST,
    NV_ENC_ERR_INVALID_PTR,
    NV_ENC_ERR_INVALID_EVENT,
    NV_ENC_ERR_INVALID_PARAM,
    NV_ENC_ERR_INVALID_CALL,
    NV_ENC_ERR_OUT_OF_MEMORY,
    NV_ENC_ERR_ENCODER_NOT_INITIALIZED,
    NV_ENC_ERR_UNSUPPORTED_PARAM,
    NV_ENC_ERR_LOCK_BUSY,
    NV_ENC_ERR_NOT_ENOUGH_BUFFER,
    NV_ENC_ERR_INVALID_VERSION,
    NV_ENC_ERR_MAP_FAILED,
    NV_ENC_ERR_NEED_MORE_INPUT,
    NV_ENC_ERR_ENCODER_BUSY,
    NV_ENC_ERR_EVENT_NOT_REGISTERD,
    NV_ENC_ERR_GENERIC,
    NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY,
    NV_ENC_ERR_UNIMPLEMENTED,
    NV_ENC_ERR_RESOURCE_REGISTER_FAILED,
    NV_ENC_ERR_RESOURCE_NOT_REGISTERED,
    NV_ENC_ERR_RESOURCE_NOT_MAPPED,
    NV_ENC_ERR_NEED_MORE_OUTPUT,
);

impl NvencStatus {
    /// `NV_ENC_SUCCESS`.
    pub const SUCCESS: Self = Self(0);
}

impl fmt::Display for NvencStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(name) => write!(f, "{name} ({})", self.0),
            None => write!(f, "NVENCSTATUS {}", self.0),
        }
    }
}

impl fmt::Debug for NvencStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Turns a driver status into a `Result`, naming the call.
pub(crate) trait CheckStatus {
    /// `Ok(())` on success, otherwise the error for `call`.
    fn check(self, call: &'static str) -> Result<()>;
}

impl CheckStatus for sys::cuda::CUresult {
    fn check(self, call: &'static str) -> Result<()> {
        if self == Self::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(Error::Cuda {
                call,
                status: CudaStatus(self),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_names_are_sorted_and_unique() {
        assert!(
            CUDA_RESULT_NAMES
                .windows(2)
                .all(|pair| pair[0].0 < pair[1].0),
            "the binary search needs strictly increasing codes"
        );
    }

    #[test]
    fn cuda_status_display() {
        let oom = CudaStatus(sys::cuda::CUresult(2));
        assert_eq!(oom.to_string(), "CUDA_ERROR_OUT_OF_MEMORY (2)");
        assert_eq!(CudaStatus::NOT_READY.name(), Some("CUDA_ERROR_NOT_READY"));
        assert_eq!(
            CudaStatus(sys::cuda::CUresult(12345)).to_string(),
            "CUresult 12345"
        );
    }

    #[test]
    fn nvenc_status_display() {
        assert_eq!(
            NvencStatus(15).to_string(),
            "NV_ENC_ERR_INVALID_VERSION (15)"
        );
        assert_eq!(NvencStatus::SUCCESS.name(), Some("NV_ENC_SUCCESS"));
        assert_eq!(NvencStatus(999).to_string(), "NVENCSTATUS 999");
    }

    #[test]
    fn error_messages_name_the_call() {
        let error = Error::Cuda {
            call: "cuMemAllocPitch_v2",
            status: CudaStatus(sys::cuda::CUresult(2)),
        };
        assert_eq!(
            error.to_string(),
            "cuMemAllocPitch_v2 failed: CUDA_ERROR_OUT_OF_MEMORY (2)"
        );
        let error = Error::Nvenc {
            call: "nvEncInitializeEncoder",
            status: NvencStatus(8),
            detail: Some("bad preset".into()),
        };
        assert_eq!(
            error.to_string(),
            "nvEncInitializeEncoder failed: NV_ENC_ERR_INVALID_PARAM (8) (bad preset)"
        );
    }
}
