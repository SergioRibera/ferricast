//! Runtime loading of the NVIDIA driver libraries.
//!
//! Nothing links against them at build time: each library is opened on
//! first use, once per process, and its entry points are resolved into a
//! table of function pointers. A required entry point that is missing fails
//! the load with [`Error::SymbolMissing`]; an optional one (newer drivers,
//! platform-specific interop) is an `Option`. The outcome of loading, success
//! or error, is cached, so a machine without the driver pays for one failed
//! `dlopen`; a failed `cuInit` is retried after a delay ([`Backoff`]).
//!
//! The tables live in `static`s and are never dropped, so the libraries stay
//! mapped and every function pointer stays valid for the rest of the
//! process.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use libloading::Library;

use crate::error::{CheckStatus as _, Error, NvencStatus, Result};
use crate::sys;
use crate::version::NvencVersion;

#[cfg(all(windows, not(target_pointer_width = "64")))]
compile_error!(
    "ferricast-nvcodec calls __stdcall entry points through C function pointers, which is only \
     correct on 64-bit Windows"
);

/// Projects the function pointer type out of bindgen's
/// `Option<unsafe extern "C" fn ..>` typedefs.
pub trait FunctionType {
    /// The bare function pointer type.
    type Fn: Copy;
}

impl<F: Copy> FunctionType for Option<F> {
    type Fn = F;
}

/// The bare function pointer type of a bindgen function typedef.
pub type Required<T> = <T as FunctionType>::Fn;

/// One driver library: its file name per OS.
struct LibraryName {
    linux: &'static str,
    windows: &'static str,
}

impl LibraryName {
    fn file_name(&self) -> &'static str {
        if cfg!(windows) {
            self.windows
        } else {
            self.linux
        }
    }

    fn open(&self) -> Result<Library> {
        let library = self.file_name();
        let opened = {
            #[cfg(windows)]
            {
                // SAFETY: the NVIDIA driver DLLs run no initialisation with
                // requirements on the caller. Searching System32 only keeps a
                // DLL planted next to the executable or in the working
                // directory from being picked up instead.
                unsafe {
                    libloading::os::windows::Library::load_with_flags(
                        library,
                        libloading::os::windows::LOAD_LIBRARY_SEARCH_SYSTEM32,
                    )
                }
                .map(Library::from)
            }
            #[cfg(not(windows))]
            {
                // SAFETY: the NVIDIA driver libraries run no initialisation
                // with requirements on the caller.
                unsafe { Library::new(library) }
            }
        };
        opened.map_err(|error| Error::LibraryMissing {
            library,
            reason: error.to_string(),
        })
    }
}

/// Resolves `names` in order (newer `_v2` entry points first) and returns the
/// first one found.
///
/// # Safety
///
/// `F` must be the function pointer type of every symbol in `names`.
unsafe fn resolve<F: Copy>(
    library: &Library,
    library_name: &'static str,
    names: &[&'static str],
) -> Result<F> {
    let mut last_error = None;
    for name in names {
        // SAFETY: the caller guarantees `F` is the symbol's type. The pointer
        // is copied out of the `Symbol`; it stays valid because the library
        // is kept loaded for the rest of the process (see the module docs).
        match unsafe { library.get::<F>(*name) } {
            Ok(symbol) => return Ok(*symbol),
            Err(error) => last_error = Some(error),
        }
    }
    let symbol = names.first().copied().unwrap_or_default();
    Err(Error::SymbolMissing {
        library: library_name,
        symbol,
        reason: last_error
            .map(|error| error.to_string())
            .unwrap_or_default(),
    })
}

/// Declares a driver library table: its file names, its required and
/// optional entry points (each with fallback names), and a process-wide
/// loader.
macro_rules! driver_library {
    (
        $(#[$meta:meta])*
        $name:ident {
            linux: $linux:literal,
            windows: $windows:literal,
            required {
                $($(#[$req_meta:meta])* $req:ident: $req_ty:ty = [$req_sym:literal $(, $req_alt:literal)*],)*
            }
            optional {
                $($(#[$opt_meta:meta])* $opt:ident: $opt_ty:ty = [$opt_sym:literal $(, $opt_alt:literal)*],)*
            }
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug)]
        pub struct $name {
            $(#[doc = concat!("`", $req_sym, "`.")] $(#[$req_meta])* pub $req: Required<$req_ty>,)*
            $(#[doc = concat!("`", $opt_sym, "`, if the driver has it.")] $(#[$opt_meta])* pub $opt: Option<Required<$opt_ty>>,)*
            _library: Library,
        }

        impl $name {
            const NAME: LibraryName = LibraryName { linux: $linux, windows: $windows };

            /// The library's file name on this OS.
            #[must_use]
            pub fn file_name() -> &'static str {
                Self::NAME.file_name()
            }

            fn load() -> Result<Self> {
                let file_name = Self::NAME.file_name();
                let library = Self::NAME.open()?;
                // SAFETY: each type comes from the bindgen typedef generated
                // from the header that declares the symbol.
                unsafe {
                    Ok(Self {
                        $($req: resolve::<Required<$req_ty>>(&library, file_name, &[$req_sym $(, $req_alt)*])?,)*
                        $($opt: resolve::<Required<$opt_ty>>(&library, file_name, &[$opt_sym $(, $opt_alt)*]).ok(),)*
                        _library: library,
                    })
                }
            }
        }
    };
}

driver_library! {
    /// `libcuda.so.1` / `nvcuda.dll`: the CUDA driver API.
    ///
    /// The required set exists in every driver that has NVENC API 11.1
    /// (R470); interop entry points are optional because WSL and some
    /// platforms lack them.
    CudaLibrary {
        linux: "libcuda.so.1",
        windows: "nvcuda.dll",
        required {
            init: sys::cuda::tcuInit = ["cuInit"],
            driver_get_version: sys::cuda::tcuDriverGetVersion = ["cuDriverGetVersion"],
            get_error_name: sys::cuda::tcuGetErrorName = ["cuGetErrorName"],
            device_get_count: sys::cuda::tcuDeviceGetCount = ["cuDeviceGetCount"],
            device_get: sys::cuda::tcuDeviceGet = ["cuDeviceGet"],
            device_get_attribute: sys::cuda::tcuDeviceGetAttribute = ["cuDeviceGetAttribute"],
            device_get_name: sys::cuda::tcuDeviceGetName = ["cuDeviceGetName"],
            device_get_uuid: sys::cuda::tcuDeviceGetUuid = ["cuDeviceGetUuid_v2", "cuDeviceGetUuid"],
            device_get_pci_bus_id: sys::cuda::tcuDeviceGetPCIBusId = ["cuDeviceGetPCIBusId"],
            device_get_by_pci_bus_id: sys::cuda::tcuDeviceGetByPCIBusId = ["cuDeviceGetByPCIBusId"],
            device_total_mem: sys::cuda::tcuDeviceTotalMem_v2 = ["cuDeviceTotalMem_v2"],
            device_primary_ctx_retain: sys::cuda::tcuDevicePrimaryCtxRetain = ["cuDevicePrimaryCtxRetain"],
            device_primary_ctx_release: sys::cuda::tcuDevicePrimaryCtxRelease = ["cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease"],
            ctx_push_current: sys::cuda::tcuCtxPushCurrent_v2 = ["cuCtxPushCurrent_v2"],
            ctx_pop_current: sys::cuda::tcuCtxPopCurrent_v2 = ["cuCtxPopCurrent_v2"],
            ctx_get_current: sys::cuda::tcuCtxGetCurrent = ["cuCtxGetCurrent"],
            ctx_synchronize: sys::cuda::tcuCtxSynchronize = ["cuCtxSynchronize"],
            ctx_create: sys::cuda::tcuCtxCreate_v2 = ["cuCtxCreate_v2"],
            ctx_destroy: sys::cuda::tcuCtxDestroy_v2 = ["cuCtxDestroy_v2"],
            mem_get_info: sys::cuda::tcuMemGetInfo_v2 = ["cuMemGetInfo_v2"],
            mem_alloc: sys::cuda::tcuMemAlloc_v2 = ["cuMemAlloc_v2"],
            mem_alloc_pitch: sys::cuda::tcuMemAllocPitch_v2 = ["cuMemAllocPitch_v2"],
            mem_free: sys::cuda::tcuMemFree_v2 = ["cuMemFree_v2"],
            mem_host_alloc: sys::cuda::tcuMemHostAlloc = ["cuMemHostAlloc"],
            mem_free_host: sys::cuda::tcuMemFreeHost = ["cuMemFreeHost"],
            memcpy_2d: sys::cuda::tcuMemcpy2D_v2 = ["cuMemcpy2D_v2"],
            memcpy_2d_async: sys::cuda::tcuMemcpy2DAsync_v2 = ["cuMemcpy2DAsync_v2"],
            memcpy_htod: sys::cuda::tcuMemcpyHtoD_v2 = ["cuMemcpyHtoD_v2"],
            memcpy_htod_async: sys::cuda::tcuMemcpyHtoDAsync_v2 = ["cuMemcpyHtoDAsync_v2"],
            memcpy_dtoh: sys::cuda::tcuMemcpyDtoH_v2 = ["cuMemcpyDtoH_v2"],
            memcpy_dtoh_async: sys::cuda::tcuMemcpyDtoHAsync_v2 = ["cuMemcpyDtoHAsync_v2"],
            memcpy_dtod: sys::cuda::tcuMemcpyDtoD_v2 = ["cuMemcpyDtoD_v2"],
            memcpy_dtod_async: sys::cuda::tcuMemcpyDtoDAsync_v2 = ["cuMemcpyDtoDAsync_v2"],
            memset_d8: sys::cuda::tcuMemsetD8_v2 = ["cuMemsetD8_v2"],
            memset_d2d8: sys::cuda::tcuMemsetD2D8_v2 = ["cuMemsetD2D8_v2"],
            memset_d2d16: sys::cuda::tcuMemsetD2D16_v2 = ["cuMemsetD2D16_v2"],
            memset_d2d32: sys::cuda::tcuMemsetD2D32_v2 = ["cuMemsetD2D32_v2"],
            memset_d8_async: sys::cuda::tcuMemsetD8Async = ["cuMemsetD8Async"],
            memset_d2d8_async: sys::cuda::tcuMemsetD2D8Async = ["cuMemsetD2D8Async"],
            memset_d2d16_async: sys::cuda::tcuMemsetD2D16Async = ["cuMemsetD2D16Async"],
            memset_d2d32_async: sys::cuda::tcuMemsetD2D32Async = ["cuMemsetD2D32Async"],
            array_create: sys::cuda::tcuArrayCreate = ["cuArrayCreate_v2"],
            array_3d_create: sys::cuda::tcuArray3DCreate = ["cuArray3DCreate_v2"],
            array_destroy: sys::cuda::tcuArrayDestroy = ["cuArrayDestroy"],
            stream_create: sys::cuda::tcuStreamCreate = ["cuStreamCreate"],
            stream_destroy: sys::cuda::tcuStreamDestroy_v2 = ["cuStreamDestroy_v2"],
            stream_query: sys::cuda::tcuStreamQuery = ["cuStreamQuery"],
            stream_synchronize: sys::cuda::tcuStreamSynchronize = ["cuStreamSynchronize"],
            stream_wait_event: sys::cuda::tcuStreamWaitEvent = ["cuStreamWaitEvent"],
            event_create: sys::cuda::tcuEventCreate = ["cuEventCreate"],
            event_destroy: sys::cuda::tcuEventDestroy_v2 = ["cuEventDestroy_v2"],
            event_record: sys::cuda::tcuEventRecord = ["cuEventRecord"],
            event_query: sys::cuda::tcuEventQuery = ["cuEventQuery"],
            event_synchronize: sys::cuda::tcuEventSynchronize = ["cuEventSynchronize"],
            event_elapsed_time: sys::cuda::tcuEventElapsedTime = ["cuEventElapsedTime"],
        }
        optional {
            /// Windows only.
            device_get_luid: sys::cuda::tcuDeviceGetLuid = ["cuDeviceGetLuid"],
            /// CUDA 11.2+.
            array_get_plane: sys::cuda::tcuArrayGetPlane = ["cuArrayGetPlane"],
            import_external_memory: sys::cuda::tcuImportExternalMemory = ["cuImportExternalMemory"],
            destroy_external_memory: sys::cuda::tcuDestroyExternalMemory = ["cuDestroyExternalMemory"],
            external_memory_get_mapped_buffer: sys::cuda::tcuExternalMemoryGetMappedBuffer = ["cuExternalMemoryGetMappedBuffer"],
            external_memory_get_mapped_mipmapped_array: sys::cuda::tcuExternalMemoryGetMappedMipmappedArray = ["cuExternalMemoryGetMappedMipmappedArray"],
            mipmapped_array_get_level: sys::cuda::tcuMipmappedArrayGetLevel = ["cuMipmappedArrayGetLevel"],
            mipmapped_array_destroy: sys::cuda::tcuMipmappedArrayDestroy = ["cuMipmappedArrayDestroy"],
            import_external_semaphore: sys::cuda::tcuImportExternalSemaphore = ["cuImportExternalSemaphore"],
            destroy_external_semaphore: sys::cuda::tcuDestroyExternalSemaphore = ["cuDestroyExternalSemaphore"],
            signal_external_semaphores_async: sys::cuda::tcuSignalExternalSemaphoresAsync = ["cuSignalExternalSemaphoresAsync"],
            wait_external_semaphores_async: sys::cuda::tcuWaitExternalSemaphoresAsync = ["cuWaitExternalSemaphoresAsync"],
            graphics_unregister_resource: sys::cuda::tcuGraphicsUnregisterResource = ["cuGraphicsUnregisterResource"],
            graphics_map_resources: sys::cuda::tcuGraphicsMapResources = ["cuGraphicsMapResources"],
            graphics_unmap_resources: sys::cuda::tcuGraphicsUnmapResources = ["cuGraphicsUnmapResources"],
            graphics_sub_resource_get_mapped_array: sys::cuda::tcuGraphicsSubResourceGetMappedArray = ["cuGraphicsSubResourceGetMappedArray"],
            graphics_resource_get_mapped_pointer: sys::cuda::tcuGraphicsResourceGetMappedPointer = ["cuGraphicsResourceGetMappedPointer_v2"],
            graphics_resource_set_map_flags: sys::cuda::tcuGraphicsResourceSetMapFlags_v2 = ["cuGraphicsResourceSetMapFlags_v2"],
            /// Linux (EGL) only.
            graphics_egl_register_image: sys::cuda::tcuGraphicsEGLRegisterImage = ["cuGraphicsEGLRegisterImage"],
            /// Linux (EGL) only.
            graphics_resource_get_mapped_egl_frame: sys::cuda::tcuGraphicsResourceGetMappedEglFrame = ["cuGraphicsResourceGetMappedEglFrame"],
            /// Windows only.
            graphics_d3d11_register_resource: sys::cuda::tcuGraphicsD3D11RegisterResource = ["cuGraphicsD3D11RegisterResource"],
        }
    }
}

driver_library! {
    /// `libnvcuvid.so.1` / `nvcuvid.dll`: NVDEC and its bitstream parser.
    CuvidLibrary {
        linux: "libnvcuvid.so.1",
        windows: "nvcuvid.dll",
        required {
            get_decoder_caps: sys::cuvid::tcuvidGetDecoderCaps = ["cuvidGetDecoderCaps"],
            create_decoder: sys::cuvid::tcuvidCreateDecoder = ["cuvidCreateDecoder"],
            destroy_decoder: sys::cuvid::tcuvidDestroyDecoder = ["cuvidDestroyDecoder"],
            decode_picture: sys::cuvid::tcuvidDecodePicture = ["cuvidDecodePicture"],
            map_video_frame: sys::cuvid::tcuvidMapVideoFrame64 = ["cuvidMapVideoFrame64"],
            unmap_video_frame: sys::cuvid::tcuvidUnmapVideoFrame64 = ["cuvidUnmapVideoFrame64"],
            ctx_lock_create: sys::cuvid::tcuvidCtxLockCreate = ["cuvidCtxLockCreate"],
            ctx_lock_destroy: sys::cuvid::tcuvidCtxLockDestroy = ["cuvidCtxLockDestroy"],
            create_video_parser: sys::cuvid::tcuvidCreateVideoParser = ["cuvidCreateVideoParser"],
            parse_video_data: sys::cuvid::tcuvidParseVideoData = ["cuvidParseVideoData"],
            destroy_video_parser: sys::cuvid::tcuvidDestroyVideoParser = ["cuvidDestroyVideoParser"],
        }
        optional {
            get_decode_status: sys::cuvid::tcuvidGetDecodeStatus = ["cuvidGetDecodeStatus"],
            reconfigure_decoder: sys::cuvid::tcuvidReconfigureDecoder = ["cuvidReconfigureDecoder"],
            ctx_lock: sys::cuvid::tcuvidCtxLock = ["cuvidCtxLock"],
            ctx_unlock: sys::cuvid::tcuvidCtxUnlock = ["cuvidCtxUnlock"],
            /// NVDEC API 13.1+.
            decode_picture_async: sys::cuvid::tcuvidDecodePictureAsync = ["cuvidDecodePictureAsync"],
            /// NVDEC API 13.1+.
            register_decode_surfaces: sys::cuvid::tcuvidRegisterDecodeSurfaces = ["cuvidRegisterDecodeSurfaces"],
        }
    }
}

driver_library! {
    /// `libnvidia-encode.so.1` / `nvEncodeAPI64.dll`: NVENC's two exported
    /// entry points. Everything else comes through the per-version function
    /// list (see [`crate::version`]).
    ///
    /// Both typedefs come from the newest header: `NvEncodeAPIGetMaxSupportedVersion`
    /// has the same signature in every version, and `NvEncodeAPICreateInstance`
    /// takes a pointer whose pointee the driver reads by its `version` field,
    /// so each version passes its own list through a pointer cast.
    NvencLibrary {
        linux: "libnvidia-encode.so.1",
        windows: "nvEncodeAPI64.dll",
        required {
            get_max_supported_version: sys::nvenc::v13_1::tNvEncodeAPIGetMaxSupportedVersion = ["NvEncodeAPIGetMaxSupportedVersion"],
            create_instance: sys::nvenc::v13_1::tNvEncodeAPICreateInstance = ["NvEncodeAPICreateInstance"],
        }
        optional {}
    }
}

/// Retries of a failing driver call, spaced out: after a failure the same
/// error is returned without calling again until a delay has passed, which
/// doubles with every consecutive failure (from [`Backoff::FIRST`] up to
/// [`Backoff::LONGEST`]). A success starts over.
#[derive(Debug, Default)]
pub struct Backoff {
    failed: Option<Failure>,
}

#[derive(Debug)]
struct Failure {
    error: Error,
    retry_at: Instant,
    delay: Duration,
}

impl Backoff {
    /// The delay after the first failure.
    pub const FIRST: Duration = Duration::from_millis(250);
    /// The longest delay.
    pub const LONGEST: Duration = Duration::from_secs(30);

    /// Nothing failed yet.
    #[must_use]
    pub const fn new() -> Self {
        Self { failed: None }
    }

    /// Runs `call`, unless the last call failed less than the current delay
    /// ago: then its error is returned again without calling.
    ///
    /// # Errors
    ///
    /// `call`'s error, fresh or remembered.
    pub fn retry<T>(&mut self, call: impl FnOnce() -> Result<T>) -> Result<T> {
        let now = Instant::now();
        if let Some(failed) = &self.failed
            && now < failed.retry_at
        {
            return Err(failed.error.clone());
        }
        match call() {
            Ok(value) => {
                self.failed = None;
                Ok(value)
            }
            Err(error) => {
                let delay = self
                    .failed
                    .as_ref()
                    .map_or(Self::FIRST, |failed| (failed.delay * 2).min(Self::LONGEST));
                tracing::debug!(%error, ?delay, "driver call failed; retrying after a delay");
                self.failed = Some(Failure {
                    error: error.clone(),
                    retry_at: now + delay,
                    delay,
                });
                Err(error)
            }
        }
    }
}

static CUDA: OnceLock<Result<CudaLibrary>> = OnceLock::new();
static CUVID: OnceLock<Result<CuvidLibrary>> = OnceLock::new();
static NVENC: OnceLock<Result<NvencLibrary>> = OnceLock::new();

impl CudaLibrary {
    /// The process-wide CUDA driver library, loaded and initialised
    /// (`cuInit`) on first use.
    ///
    /// # Errors
    ///
    /// [`Error::LibraryMissing`] without an NVIDIA driver,
    /// [`Error::SymbolMissing`] with one too old: kept for the process.
    /// [`Error::Cuda`] when `cuInit` fails (no device yet, a driver / kernel
    /// module mismatch, a GPU still waking up): returned again until
    /// [`Backoff`]'s delay has passed, then `cuInit` is retried.
    pub fn get() -> Result<&'static Self> {
        static INIT: Mutex<Backoff> = Mutex::new(Backoff::new());
        static INITIALISED: AtomicBool = AtomicBool::new(false);
        let library = CUDA
            .get_or_init(Self::load)
            .as_ref()
            .map_err(Clone::clone)?;
        if !INITIALISED.load(Ordering::Acquire) {
            INIT.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .retry(|| {
                    // SAFETY: `cuInit` takes no pointers; flags must be 0.
                    unsafe { (library.init)(0) }.check("cuInit")
                })?;
            INITIALISED.store(true, Ordering::Release);
        }
        Ok(library)
    }

    /// The CUDA version the driver implements (`cuDriverGetVersion`), e.g.
    /// 13.0 for a 580-series driver.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`] if the driver call fails.
    pub fn driver_version(&self) -> Result<CudaVersion> {
        let mut version = 0;
        // SAFETY: the out-pointer is a valid local.
        unsafe { (self.driver_get_version)(&raw mut version) }.check("cuDriverGetVersion")?;
        Ok(CudaVersion::from_raw(version))
    }
}

impl CuvidLibrary {
    /// The process-wide NVDEC library, loaded on first use. Callers also
    /// need [`CudaLibrary::get`]: NVDEC runs on a CUDA context.
    ///
    /// # Errors
    ///
    /// [`Error::LibraryMissing`] / [`Error::SymbolMissing`]; the first
    /// outcome is returned to every later caller.
    pub fn get() -> Result<&'static Self> {
        CUVID.get_or_init(Self::load).as_ref().map_err(Clone::clone)
    }
}

impl NvencLibrary {
    /// The process-wide NVENC library, loaded on first use.
    ///
    /// # Errors
    ///
    /// [`Error::LibraryMissing`] / [`Error::SymbolMissing`]; the first
    /// outcome is returned to every later caller.
    pub fn get() -> Result<&'static Self> {
        NVENC.get_or_init(Self::load).as_ref().map_err(Clone::clone)
    }

    /// The newest NVENC API version the installed driver accepts
    /// (`NvEncodeAPIGetMaxSupportedVersion`).
    ///
    /// # Errors
    ///
    /// [`Error::Nvenc`] if the driver call fails.
    pub fn max_supported_version(&self) -> Result<NvencVersion> {
        let mut raw = 0;
        // SAFETY: the out-pointer is a valid local.
        let status = unsafe { (self.get_max_supported_version)(&raw mut raw) };
        if status.0 != NvencStatus::SUCCESS.0 {
            return Err(Error::Nvenc {
                call: "NvEncodeAPIGetMaxSupportedVersion",
                status: NvencStatus(status.0),
                detail: None,
            });
        }
        Ok(NvencVersion::from_driver(raw))
    }
}

/// A CUDA version as `cuDriverGetVersion` reports it (`1000 * major + 10 *
/// minor`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CudaVersion {
    /// Major version.
    pub major: u32,
    /// Minor version.
    pub minor: u32,
}

impl CudaVersion {
    fn from_raw(raw: i32) -> Self {
        let raw = raw.unsigned_abs();
        Self {
            major: raw / 1000,
            minor: raw % 1000 / 10,
        }
    }
}

impl std::fmt::Display for CudaVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_version_decoding() {
        assert_eq!(
            CudaVersion::from_raw(13000),
            CudaVersion {
                major: 13,
                minor: 0
            }
        );
        assert_eq!(CudaVersion::from_raw(12080).to_string(), "12.8");
        assert_eq!(CudaVersion::from_raw(11040).to_string(), "11.4");
    }

    #[test]
    fn library_names_follow_the_os() {
        let expected = if cfg!(windows) {
            ("nvcuda.dll", "nvcuvid.dll", "nvEncodeAPI64.dll")
        } else {
            ("libcuda.so.1", "libnvcuvid.so.1", "libnvidia-encode.so.1")
        };
        assert_eq!(
            (
                CudaLibrary::file_name(),
                CuvidLibrary::file_name(),
                NvencLibrary::file_name()
            ),
            expected
        );
    }

    #[test]
    fn backoff_spaces_out_retries_and_forgets_on_success() {
        let failure = || Error::Cuda {
            call: "cuInit",
            status: crate::error::CudaStatus(sys::cuda::CUresult(999)),
        };
        let mut backoff = Backoff::new();
        let mut calls = 0;
        let mut attempt = |backoff: &mut Backoff, succeed: bool| {
            backoff.retry(|| {
                calls += 1;
                if succeed { Ok(()) } else { Err(failure()) }
            })
        };
        assert!(attempt(&mut backoff, false).is_err());
        // Within the delay: the remembered error, no call.
        assert!(attempt(&mut backoff, true).is_err());
        std::thread::sleep(Backoff::FIRST + Duration::from_millis(50));
        assert!(attempt(&mut backoff, true).is_ok());
        assert!(attempt(&mut backoff, true).is_ok());
        assert_eq!(calls, 3);
    }
}
