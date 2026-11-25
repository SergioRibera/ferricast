//! NVENC API version negotiation and the per-version plumbing.
//!
//! A driver accepts every NVENC API version up to the one it reports through
//! `NvEncodeAPIGetMaxSupportedVersion`, but struct layouts and struct version
//! numbers change between API versions. The crate carries bindings for each
//! version in [`ApiVersion::ALL`] and, per session, uses the newest one the
//! driver accepts ([`ApiVersion::negotiate`]). `FERRICAST_NVENC_API=<major>.<minor>`
//! forces a version, which is also how every version is tested on one
//! machine.
//!
//! Code that depends on the layouts is written once, generic over
//! [`NvencApi`] (implemented by the marker types in [`api`], one per
//! generated module), and instantiated for the negotiated version through
//! [`ApiVersion::dispatch`] or the [`AnyApi`] enum: an `enum`, not `dyn`.

use std::ffi::{CStr, c_void};
use std::fmt;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::str::FromStr;
use std::sync::{Mutex, OnceLock, PoisonError};

use crate::error::{Error, NvencStatus, Result, SupportedVersions};
use crate::loader::NvencLibrary;

/// The environment variable that forces an NVENC API version.
pub const API_OVERRIDE_VAR: &str = "FERRICAST_NVENC_API";

/// An NVENC API version as the driver reports it; not necessarily one this
/// crate supports (see [`ApiVersion`] for those).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NvencVersion {
    /// Major version.
    pub major: u32,
    /// Minor version.
    pub minor: u32,
}

impl NvencVersion {
    /// A version from its parts.
    #[must_use]
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// Decodes `NvEncodeAPIGetMaxSupportedVersion`'s `(major << 4) | minor`.
    #[must_use]
    pub const fn from_driver(raw: u32) -> Self {
        Self {
            major: raw >> 4,
            minor: raw & 0xf,
        }
    }
}

impl fmt::Display for NvencVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// `<major>.<minor>` could not be parsed as an NVENC API version.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0:?} is not an NVENC API version (expected <major>.<minor>)")]
pub struct ParseVersionError(String);

impl FromStr for NvencVersion {
    type Err = ParseVersionError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let invalid = || ParseVersionError(text.to_owned());
        let (major, minor) = text.trim().split_once('.').ok_or_else(invalid)?;
        Ok(Self {
            major: major.parse().map_err(|_| invalid())?,
            minor: minor.parse().map_err(|_| invalid())?,
        })
    }
}

/// The driver release that first ships an NVENC API version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DriverRelease {
    /// Linux driver version.
    pub linux: &'static str,
    /// Windows driver version.
    pub windows: &'static str,
}

impl DriverRelease {
    /// The release for the OS this build targets.
    #[must_use]
    pub const fn for_this_os(self) -> &'static str {
        if cfg!(windows) {
            self.windows
        } else {
            self.linux
        }
    }
}

/// Who asked for a specific NVENC API version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApiRequest {
    /// The `FERRICAST_NVENC_API` environment variable.
    Environment,
    /// The code calling into this crate ([`Api::load`], [`AnyApi::load`]).
    Caller,
}

impl fmt::Display for ApiRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Environment => write!(f, "{API_OVERRIDE_VAR}"),
            Self::Caller => f.write_str("the caller"),
        }
    }
}

/// An NVENC API version this crate carries bindings for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ApiVersion {
    /// 11.1 (SDK 11.1, R470).
    V11_1,
    /// 12.0 (SDK 12.0, R520): AV1 encode.
    V12_0,
    /// 12.1 (SDK 12.1, R530).
    V12_1,
    /// 12.2 (SDK 12.2, R550).
    V12_2,
    /// 13.0 (SDK 13.0, R570).
    V13_0,
    /// 13.1 (SDK 13.1, R610).
    V13_1,
}

impl ApiVersion {
    /// Every supported version, oldest first.
    pub const ALL: [Self; 6] = [
        Self::V11_1,
        Self::V12_0,
        Self::V12_1,
        Self::V12_2,
        Self::V13_0,
        Self::V13_1,
    ];

    /// The oldest supported version.
    pub const OLDEST: Self = Self::V11_1;

    /// The version number.
    #[must_use]
    pub const fn version(self) -> NvencVersion {
        match self {
            Self::V11_1 => NvencVersion::new(11, 1),
            Self::V12_0 => NvencVersion::new(12, 0),
            Self::V12_1 => NvencVersion::new(12, 1),
            Self::V12_2 => NvencVersion::new(12, 2),
            Self::V13_0 => NvencVersion::new(13, 0),
            Self::V13_1 => NvencVersion::new(13, 1),
        }
    }

    /// The oldest driver that accepts this version.
    #[must_use]
    pub const fn minimum_driver(self) -> DriverRelease {
        let (linux, windows) = match self {
            Self::V11_1 => ("470.57.02", "471.41"),
            Self::V12_0 => ("520.56.06", "522.25"),
            Self::V12_1 => ("530.41.03", "531.61"),
            Self::V12_2 => ("550.54.14", "551.76"),
            Self::V13_0 => ("570.0", "570.0"),
            Self::V13_1 => ("610.0", "610.0"),
        };
        DriverRelease { linux, windows }
    }

    /// The supported version with this number, if any.
    #[must_use]
    pub fn from_version(version: NvencVersion) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|candidate| candidate.version() == version)
    }

    /// Picks the version to use against a driver whose newest NVENC API is
    /// `driver_max`: `requested` if given, otherwise the newest supported
    /// version the driver accepts.
    ///
    /// # Errors
    ///
    /// [`Error::ApiNotSupportedByDriver`] when the driver does not accept
    /// `requested`; [`Error::DriverTooOld`] when it accepts none of
    /// [`ApiVersion::ALL`].
    pub fn select(driver_max: NvencVersion, requested: Option<Self>) -> Result<Self> {
        if let Some(requested) = requested {
            return if requested.version() <= driver_max {
                Ok(requested)
            } else {
                Err(Error::ApiNotSupportedByDriver {
                    requested,
                    origin: ApiRequest::Caller,
                    driver_max,
                    driver: requested.minimum_driver().for_this_os(),
                })
            };
        }
        Self::ALL
            .into_iter()
            .rev()
            .find(|candidate| candidate.version() <= driver_max)
            .ok_or(Error::DriverTooOld {
                found: driver_max,
                minimum: Self::OLDEST,
                driver: Self::OLDEST.minimum_driver().for_this_os(),
            })
    }

    /// The version `FERRICAST_NVENC_API` forces, if it is set (an empty value
    /// counts as unset). Read once per process: the variable cannot switch a
    /// running process to another version (see [`AnyApi::load`]).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidApiOverride`] when the value is not a supported
    /// `<major>.<minor>`.
    pub fn requested() -> Result<Option<Self>> {
        static REQUESTED: OnceLock<Result<Option<ApiVersion>>> = OnceLock::new();
        REQUESTED
            .get_or_init(|| {
                std::env::var_os(API_OVERRIDE_VAR)
                    .filter(|value| !value.is_empty())
                    .map(|value| Self::parse_override(&value.to_string_lossy()))
                    .transpose()
            })
            .clone()
    }

    fn parse_override(value: &str) -> Result<Self> {
        value
            .parse::<NvencVersion>()
            .ok()
            .and_then(Self::from_version)
            .ok_or_else(|| Error::InvalidApiOverride {
                value: value.to_owned(),
                supported: SupportedVersions,
            })
    }

    /// Queries the driver and picks the version to use, honouring
    /// `FERRICAST_NVENC_API`. The process keeps the version once a function
    /// list was loaded ([`AnyApi::load`]): that one is returned from then on.
    ///
    /// # Errors
    ///
    /// Loading NVENC ([`NvencLibrary::get`]), [`ApiVersion::requested`] and
    /// [`ApiVersion::select`] errors.
    pub fn negotiate() -> Result<Self> {
        if let Some(loaded) = AnyApi::loaded() {
            return Ok(loaded.version());
        }
        let requested = Self::requested()?;
        let driver_max = NvencLibrary::get()?.max_supported_version()?;
        let selected = Self::select(driver_max, requested).map_err(|error| match error {
            Error::ApiNotSupportedByDriver {
                requested,
                driver_max,
                driver,
                origin: _,
            } => Error::ApiNotSupportedByDriver {
                requested,
                origin: ApiRequest::Environment,
                driver_max,
                driver,
            },
            other => other,
        })?;
        tracing::debug!(%driver_max, %selected, forced = requested.is_some(), "NVENC API negotiated");
        Ok(selected)
    }

    /// Runs `visitor` with the marker type of this version.
    pub fn dispatch<V: ApiVisitor>(self, visitor: V) -> V::Output {
        match self {
            Self::V11_1 => visitor.visit::<api::V11_1>(),
            Self::V12_0 => visitor.visit::<api::V12_0>(),
            Self::V12_1 => visitor.visit::<api::V12_1>(),
            Self::V12_2 => visitor.visit::<api::V12_2>(),
            Self::V13_0 => visitor.visit::<api::V13_0>(),
            Self::V13_1 => visitor.visit::<api::V13_1>(),
        }
    }
}

impl fmt::Display for ApiVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.version().fmt(f)
    }
}

/// Code generic over the API version, run for a runtime [`ApiVersion`] by
/// [`ApiVersion::dispatch`].
pub trait ApiVisitor {
    /// What the visit returns.
    type Output;

    /// Runs with `A`, the negotiated version's marker type.
    fn visit<A: NvencApi>(self) -> Self::Output;
}

mod sealed {
    pub trait Sealed {}
}

/// The NVENC device kinds a session can be opened on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceType {
    /// A `CUcontext`.
    Cuda,
    /// An `ID3D11Device` (Windows).
    DirectX,
}

/// One NVENC API version: its constants, its function list and the calls
/// whose argument structs differ between versions. Implemented by the
/// marker types in [`api`] only.
///
/// Every constant comes from the version's own header (the `*_VER` macros
/// evaluated by bindgen), so a struct is always tagged with the version
/// number of the layout it was built with.
pub trait NvencApi: sealed::Sealed + Sized + 'static {
    /// This version.
    const VERSION: ApiVersion;
    /// `NVENCAPI_VERSION`: `major | (minor << 24)`.
    const API_VERSION: u32;

    /// `NV_ENCODE_API_FUNCTION_LIST_VER`.
    const FUNCTION_LIST_VER: u32;
    /// `NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER`.
    const OPEN_ENCODE_SESSION_EX_PARAMS_VER: u32;
    /// `NV_ENC_CAPS_PARAM_VER`.
    const CAPS_PARAM_VER: u32;
    /// `NV_ENC_ENCODE_OUT_PARAMS_VER`.
    const ENCODE_OUT_PARAMS_VER: u32;
    /// `NV_ENC_CREATE_INPUT_BUFFER_VER`.
    const CREATE_INPUT_BUFFER_VER: u32;
    /// `NV_ENC_CREATE_BITSTREAM_BUFFER_VER`.
    const CREATE_BITSTREAM_BUFFER_VER: u32;
    /// `NV_ENC_CREATE_MV_BUFFER_VER`.
    const CREATE_MV_BUFFER_VER: u32;
    /// `NV_ENC_RC_PARAMS_VER`.
    const RC_PARAMS_VER: u32;
    /// `NV_ENC_CONFIG_VER`.
    const CONFIG_VER: u32;
    /// `NV_ENC_INITIALIZE_PARAMS_VER`.
    const INITIALIZE_PARAMS_VER: u32;
    /// `NV_ENC_RECONFIGURE_PARAMS_VER`.
    const RECONFIGURE_PARAMS_VER: u32;
    /// `NV_ENC_PRESET_CONFIG_VER`.
    const PRESET_CONFIG_VER: u32;
    /// `NV_ENC_PIC_PARAMS_VER`.
    const PIC_PARAMS_VER: u32;
    /// `NV_ENC_MEONLY_PARAMS_VER`.
    const MEONLY_PARAMS_VER: u32;
    /// `NV_ENC_LOCK_BITSTREAM_VER`.
    const LOCK_BITSTREAM_VER: u32;
    /// `NV_ENC_LOCK_INPUT_BUFFER_VER`.
    const LOCK_INPUT_BUFFER_VER: u32;
    /// `NV_ENC_MAP_INPUT_RESOURCE_VER`.
    const MAP_INPUT_RESOURCE_VER: u32;
    /// `NV_ENC_REGISTER_RESOURCE_VER`.
    const REGISTER_RESOURCE_VER: u32;
    /// `NV_ENC_STAT_VER`.
    const STAT_VER: u32;
    /// `NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER`.
    const SEQUENCE_PARAM_PAYLOAD_VER: u32;
    /// `NV_ENC_EVENT_PARAMS_VER`.
    const EVENT_PARAMS_VER: u32;

    /// This version's `NV_ENCODE_API_FUNCTION_LIST`.
    type FunctionList: Copy + fmt::Debug;

    /// `NVENCAPI_STRUCT_VERSION(ver)` for this version: the API version in
    /// the low bits, the struct's revision in bits 16..28, `0x7` above. A few
    /// structs also set bit 31; their `*_VER` constants include it.
    #[must_use]
    fn struct_version(revision: u32) -> u32 {
        Self::API_VERSION | (revision << 16) | (0x7 << 28)
    }

    /// Fills this version's function list (`NvEncodeAPICreateInstance`).
    ///
    /// # Errors
    ///
    /// [`Error::Nvenc`] when the driver rejects the version (an older driver
    /// answers `NV_ENC_ERR_INVALID_VERSION`).
    fn create_function_list(library: &NvencLibrary) -> Result<Self::FunctionList>;

    /// `nvEncOpenEncodeSessionEx`.
    ///
    /// # Errors
    ///
    /// [`Error::NvencFunctionMissing`] / [`Error::Nvenc`].
    ///
    /// # Safety
    ///
    /// `device` must be a live device of `device_type` (a `CUcontext` for
    /// [`DeviceType::Cuda`]) that outlives the returned encoder, and
    /// `functions` must come from [`NvencApi::create_function_list`].
    unsafe fn open_encode_session(
        functions: &Self::FunctionList,
        device: NonNull<c_void>,
        device_type: DeviceType,
    ) -> Result<NonNull<c_void>>;

    /// `nvEncDestroyEncoder`.
    ///
    /// # Errors
    ///
    /// [`Error::NvencFunctionMissing`] / [`Error::Nvenc`].
    ///
    /// # Safety
    ///
    /// `encoder` must be a session opened with `functions` and not destroyed
    /// yet; it is invalid afterwards whatever the outcome.
    unsafe fn destroy_encoder(
        functions: &Self::FunctionList,
        encoder: NonNull<c_void>,
    ) -> Result<()>;

    /// `nvEncGetLastErrorString`: the driver's explanation of the session's
    /// last failure, if it gave one.
    ///
    /// # Safety
    ///
    /// `encoder` must be a live session opened with `functions`.
    unsafe fn last_error(
        functions: &Self::FunctionList,
        encoder: NonNull<c_void>,
    ) -> Option<String>;

    /// Wraps this version's [`Api`] in [`AnyApi`].
    fn wrap(api: Api<Self>) -> AnyApi;

    /// This version's [`Api`] out of an [`AnyApi`] holding it.
    fn unwrap(api: AnyApi) -> Option<Api<Self>>;
}

/// The marker types of the supported NVENC API versions, one per generated
/// module in [`crate::sys::nvenc`].
pub mod api {
    use std::ffi::c_void;
    use std::ptr::NonNull;

    use super::{ApiVersion, DeviceType, NvencApi, sealed};
    use crate::error::{Error, Result};
    use crate::loader::NvencLibrary;
    use crate::sys::nvenc;

    macro_rules! nvenc_api {
        ($($marker:ident => $module:ident, $version:ident;)*) => {$(
            #[doc = concat!("NVENC API ", stringify!($version), " (`sys::nvenc::", stringify!($module), "`).")]
            #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
            pub enum $marker {}

            impl sealed::Sealed for $marker {}

            impl NvencApi for $marker {
                const VERSION: ApiVersion = ApiVersion::$version;
                const API_VERSION: u32 = nvenc::$module::NVENCAPI_VERSION;
                const FUNCTION_LIST_VER: u32 = nvenc::$module::NV_ENCODE_API_FUNCTION_LIST_VER;
                const OPEN_ENCODE_SESSION_EX_PARAMS_VER: u32 =
                    nvenc::$module::NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
                const CAPS_PARAM_VER: u32 = nvenc::$module::NV_ENC_CAPS_PARAM_VER;
                const ENCODE_OUT_PARAMS_VER: u32 = nvenc::$module::NV_ENC_ENCODE_OUT_PARAMS_VER;
                const CREATE_INPUT_BUFFER_VER: u32 = nvenc::$module::NV_ENC_CREATE_INPUT_BUFFER_VER;
                const CREATE_BITSTREAM_BUFFER_VER: u32 =
                    nvenc::$module::NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
                const CREATE_MV_BUFFER_VER: u32 = nvenc::$module::NV_ENC_CREATE_MV_BUFFER_VER;
                const RC_PARAMS_VER: u32 = nvenc::$module::NV_ENC_RC_PARAMS_VER;
                const CONFIG_VER: u32 = nvenc::$module::NV_ENC_CONFIG_VER;
                const INITIALIZE_PARAMS_VER: u32 = nvenc::$module::NV_ENC_INITIALIZE_PARAMS_VER;
                const RECONFIGURE_PARAMS_VER: u32 = nvenc::$module::NV_ENC_RECONFIGURE_PARAMS_VER;
                const PRESET_CONFIG_VER: u32 = nvenc::$module::NV_ENC_PRESET_CONFIG_VER;
                const PIC_PARAMS_VER: u32 = nvenc::$module::NV_ENC_PIC_PARAMS_VER;
                const MEONLY_PARAMS_VER: u32 = nvenc::$module::NV_ENC_MEONLY_PARAMS_VER;
                const LOCK_BITSTREAM_VER: u32 = nvenc::$module::NV_ENC_LOCK_BITSTREAM_VER;
                const LOCK_INPUT_BUFFER_VER: u32 = nvenc::$module::NV_ENC_LOCK_INPUT_BUFFER_VER;
                const MAP_INPUT_RESOURCE_VER: u32 = nvenc::$module::NV_ENC_MAP_INPUT_RESOURCE_VER;
                const REGISTER_RESOURCE_VER: u32 = nvenc::$module::NV_ENC_REGISTER_RESOURCE_VER;
                const STAT_VER: u32 = nvenc::$module::NV_ENC_STAT_VER;
                const SEQUENCE_PARAM_PAYLOAD_VER: u32 =
                    nvenc::$module::NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;
                const EVENT_PARAMS_VER: u32 = nvenc::$module::NV_ENC_EVENT_PARAMS_VER;

                type FunctionList = nvenc::$module::NV_ENCODE_API_FUNCTION_LIST;

                fn create_function_list(library: &NvencLibrary) -> Result<Self::FunctionList> {
                    let mut functions = nvenc::$module::NV_ENCODE_API_FUNCTION_LIST {
                        version: Self::FUNCTION_LIST_VER,
                        ..Default::default()
                    };
                    // SAFETY: the list is a valid, zeroed
                    // `NV_ENCODE_API_FUNCTION_LIST` of this version with its
                    // `version` set; the driver reads that field to know the
                    // layout and fills the entries. The pointer cast only
                    // changes the nominal type (see `NvencLibrary`).
                    let status = unsafe {
                        (library.create_instance)(std::ptr::from_mut(&mut functions).cast())
                    };
                    super::check(status.0, "NvEncodeAPICreateInstance")?;
                    Ok(functions)
                }

                unsafe fn open_encode_session(
                    functions: &Self::FunctionList,
                    device: NonNull<c_void>,
                    device_type: DeviceType,
                ) -> Result<NonNull<c_void>> {
                    let open = functions.nvEncOpenEncodeSessionEx.ok_or(Error::NvencFunctionMissing {
                        api: Self::VERSION,
                        function: "nvEncOpenEncodeSessionEx",
                    })?;
                    let mut params = nvenc::$module::NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS {
                        version: Self::OPEN_ENCODE_SESSION_EX_PARAMS_VER,
                        deviceType: match device_type {
                            DeviceType::Cuda => nvenc::$module::NV_ENC_DEVICE_TYPE::NV_ENC_DEVICE_TYPE_CUDA,
                            DeviceType::DirectX => nvenc::$module::NV_ENC_DEVICE_TYPE::NV_ENC_DEVICE_TYPE_DIRECTX,
                        },
                        device: device.as_ptr(),
                        apiVersion: Self::API_VERSION,
                        ..Default::default()
                    };
                    let mut encoder = std::ptr::null_mut();
                    // SAFETY: `params` is a valid struct of this version, the
                    // caller guarantees `device`, and `encoder` is a valid
                    // out-pointer.
                    let status = unsafe { open(&raw mut params, &raw mut encoder) };
                    let opened = super::check(status.0, "nvEncOpenEncodeSessionEx");
                    match (opened, NonNull::new(encoder)) {
                        (Ok(()), Some(encoder)) => Ok(encoder),
                        (Ok(()), None) => Err(Error::Nvenc {
                            call: "nvEncOpenEncodeSessionEx",
                            status: crate::error::NvencStatus(
                                nvenc::$module::NVENCSTATUS::NV_ENC_ERR_INVALID_PTR.0,
                            ),
                            detail: Some("the driver returned a null encoder".into()),
                        }),
                        (Err(error), encoder) => {
                            // NVENC requires destroying a half-open session.
                            if let Some(encoder) = encoder {
                                // SAFETY: the driver handed out `encoder`,
                                // nothing else holds it.
                                if let Err(destroy) = unsafe { Self::destroy_encoder(functions, encoder) } {
                                    tracing::warn!(%destroy, "destroying a failed NVENC session");
                                }
                            }
                            Err(error)
                        }
                    }
                }

                unsafe fn destroy_encoder(
                    functions: &Self::FunctionList,
                    encoder: NonNull<c_void>,
                ) -> Result<()> {
                    let destroy = functions.nvEncDestroyEncoder.ok_or(Error::NvencFunctionMissing {
                        api: Self::VERSION,
                        function: "nvEncDestroyEncoder",
                    })?;
                    // SAFETY: the caller guarantees `encoder` is live and
                    // from `functions`.
                    let status = unsafe { destroy(encoder.as_ptr()) };
                    super::check(status.0, "nvEncDestroyEncoder")
                }

                unsafe fn last_error(
                    functions: &Self::FunctionList,
                    encoder: NonNull<c_void>,
                ) -> Option<String> {
                    let last_error = functions.nvEncGetLastErrorString?;
                    // SAFETY: the caller guarantees `encoder` is live.
                    let text = unsafe { last_error(encoder.as_ptr()) };
                    if text.is_null() {
                        return None;
                    }
                    // SAFETY: non-null, so a NUL-terminated string owned by
                    // the session, read before any other call on it.
                    super::message(unsafe { std::ffi::CStr::from_ptr(text) })
                }

                fn wrap(api: super::Api<Self>) -> super::AnyApi {
                    super::AnyApi::$marker(api)
                }

                fn unwrap(api: super::AnyApi) -> Option<super::Api<Self>> {
                    match api {
                        super::AnyApi::$marker(api) => Some(api),
                        _ => None,
                    }
                }
            }
        )*};
    }

    nvenc_api! {
        V11_1 => v11_1, V11_1;
        V12_0 => v12_0, V12_0;
        V12_1 => v12_1, V12_1;
        V12_2 => v12_2, V12_2;
        V13_0 => v13_0, V13_0;
        V13_1 => v13_1, V13_1;
    }
}

fn check(status: u32, call: &'static str) -> Result<()> {
    if status == NvencStatus::SUCCESS.0 {
        Ok(())
    } else {
        Err(Error::Nvenc {
            call,
            status: NvencStatus(status),
            detail: None,
        })
    }
}

fn message(text: &CStr) -> Option<String> {
    Some(text.to_string_lossy().trim().to_owned()).filter(|text| !text.is_empty())
}

/// The function list of one NVENC API version, filled by the driver.
pub struct Api<A: NvencApi> {
    functions: A::FunctionList,
    _version: PhantomData<A>,
}

// SAFETY: the list holds driver entry points, which any thread may call, and
// reserved pointers that are never dereferenced; it is never written after
// `NvEncodeAPICreateInstance` filled it.
unsafe impl<A: NvencApi> Send for Api<A> {}
// SAFETY: as for `Send`; shared access only reads the table.
unsafe impl<A: NvencApi> Sync for Api<A> {}

impl<A: NvencApi> Clone for Api<A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A: NvencApi> Copy for Api<A> {}

impl<A: NvencApi> fmt::Debug for Api<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Api")
            .field("version", &A::VERSION)
            .finish_non_exhaustive()
    }
}

impl<A: NvencApi> Api<A> {
    /// Version `A`'s function list: the process's ([`AnyApi::load`]).
    ///
    /// # Errors
    ///
    /// As [`AnyApi::load`].
    pub fn load() -> Result<Self> {
        let loaded = AnyApi::load(A::VERSION)?;
        A::unwrap(loaded).ok_or(Error::ApiVersionInUse {
            in_use: loaded.version(),
            requested: A::VERSION,
        })
    }

    /// Fills version `A`'s function list (`NvEncodeAPICreateInstance`).
    fn create() -> Result<Self> {
        let library = NvencLibrary::get()?;
        ApiVersion::select(library.max_supported_version()?, Some(A::VERSION))?;
        Ok(Self {
            functions: A::create_function_list(library)?,
            _version: PhantomData,
        })
    }

    /// The driver's entry points for this version.
    #[must_use]
    pub fn functions(&self) -> &A::FunctionList {
        &self.functions
    }

    /// This version.
    #[must_use]
    pub fn version(&self) -> ApiVersion {
        A::VERSION
    }
}

/// The one NVENC function list of the process ([`AnyApi::load`]).
static PROCESS_API: Mutex<Option<AnyApi>> = Mutex::new(None);

/// [`Api`] for whichever version was negotiated at runtime.
#[derive(Clone, Copy, Debug)]
pub enum AnyApi {
    /// API 11.1.
    V11_1(Api<api::V11_1>),
    /// API 12.0.
    V12_0(Api<api::V12_0>),
    /// API 12.1.
    V12_1(Api<api::V12_1>),
    /// API 12.2.
    V12_2(Api<api::V12_2>),
    /// API 13.0.
    V13_0(Api<api::V13_0>),
    /// API 13.1.
    V13_1(Api<api::V13_1>),
}

impl AnyApi {
    /// Negotiates the version ([`ApiVersion::negotiate`]) and loads it.
    ///
    /// # Errors
    ///
    /// See [`ApiVersion::negotiate`] and [`Api::load`].
    pub fn negotiate() -> Result<Self> {
        Self::load(ApiVersion::negotiate()?)
    }

    /// The process's function list, of `version`: filled by the driver the
    /// first time, the same list afterwards.
    ///
    /// One version per process: the driver keeps the API version of the
    /// last `NvEncodeAPICreateInstance` for the whole process, and sessions
    /// opened through another version's list then fail with
    /// `NV_ENC_ERR_INVALID_VERSION`. A process that needs several versions
    /// (the tests) runs each in a child process.
    ///
    /// # Errors
    ///
    /// [`Error::ApiVersionInUse`] when the process already uses another
    /// version; loading errors, [`Error::ApiNotSupportedByDriver`] when the
    /// driver does not accept `version`, [`Error::Nvenc`] if it still
    /// rejects it.
    pub fn load(version: ApiVersion) -> Result<Self> {
        struct Create;
        impl ApiVisitor for Create {
            type Output = Result<AnyApi>;
            fn visit<A: NvencApi>(self) -> Result<AnyApi> {
                Api::<A>::create().map(AnyApi::from)
            }
        }
        let mut loaded = PROCESS_API.lock().unwrap_or_else(PoisonError::into_inner);
        match *loaded {
            Some(api) if api.version() == version => Ok(api),
            Some(api) => Err(Error::ApiVersionInUse {
                in_use: api.version(),
                requested: version,
            }),
            None => {
                let api = version.dispatch(Create)?;
                *loaded = Some(api);
                Ok(api)
            }
        }
    }

    /// The process's function list, if one was loaded.
    #[must_use]
    pub fn loaded() -> Option<Self> {
        *PROCESS_API.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The loaded version.
    #[must_use]
    pub fn version(&self) -> ApiVersion {
        match self {
            Self::V11_1(api) => api.version(),
            Self::V12_0(api) => api.version(),
            Self::V12_1(api) => api.version(),
            Self::V12_2(api) => api.version(),
            Self::V13_0(api) => api.version(),
            Self::V13_1(api) => api.version(),
        }
    }
}

impl<A: NvencApi> From<Api<A>> for AnyApi {
    fn from(api: Api<A>) -> Self {
        A::wrap(api)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_version_decoding() {
        assert_eq!(NvencVersion::from_driver(0xd0), NvencVersion::new(13, 0));
        assert_eq!(NvencVersion::from_driver(0xb1), NvencVersion::new(11, 1));
        assert_eq!(NvencVersion::from_driver(0xc2).to_string(), "12.2");
    }

    #[test]
    fn version_parsing() {
        assert_eq!("13.0".parse(), Ok(NvencVersion::new(13, 0)));
        assert_eq!(" 12.2 ".parse(), Ok(NvencVersion::new(12, 2)));
        assert!("13".parse::<NvencVersion>().is_err());
        assert!("x.1".parse::<NvencVersion>().is_err());
        assert_eq!(
            ApiVersion::parse_override("12.1").ok(),
            Some(ApiVersion::V12_1)
        );
        let error = ApiVersion::parse_override("10.0").map(|_| ()).unwrap_err();
        assert_eq!(
            error.to_string(),
            "FERRICAST_NVENC_API=\"10.0\" is not one of 11.1, 12.0, 12.1, 12.2, 13.0, 13.1"
        );
    }

    #[test]
    fn all_is_sorted_and_matches_version() {
        assert!(ApiVersion::ALL.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(
            ApiVersion::ALL
                .windows(2)
                .all(|pair| pair[0].version() < pair[1].version())
        );
        assert!(
            ApiVersion::ALL
                .into_iter()
                .all(|version| ApiVersion::from_version(version.version()) == Some(version))
        );
    }

    #[test]
    fn selection_picks_the_newest_the_driver_accepts() {
        let pick = |major, minor| ApiVersion::select(NvencVersion::new(major, minor), None);
        assert_eq!(pick(13, 0).ok(), Some(ApiVersion::V13_0));
        assert_eq!(pick(13, 1).ok(), Some(ApiVersion::V13_1));
        assert_eq!(pick(14, 0).ok(), Some(ApiVersion::V13_1));
        assert_eq!(pick(12, 9).ok(), Some(ApiVersion::V12_2));
        assert_eq!(pick(11, 1).ok(), Some(ApiVersion::OLDEST));
        assert!(matches!(
            pick(11, 0),
            Err(Error::DriverTooOld {
                found: NvencVersion {
                    major: 11,
                    minor: 0
                },
                minimum: ApiVersion::V11_1,
                ..
            })
        ));
    }

    #[test]
    fn selection_honours_a_forced_version() {
        let driver = NvencVersion::new(13, 0);
        assert_eq!(
            ApiVersion::select(driver, Some(ApiVersion::V11_1)).ok(),
            Some(ApiVersion::V11_1)
        );
        let error = ApiVersion::select(driver, Some(ApiVersion::V13_1)).unwrap_err();
        assert!(matches!(
            error,
            Error::ApiNotSupportedByDriver {
                requested: ApiVersion::V13_1,
                ..
            }
        ));
        assert!(error.to_string().contains("610.0"), "{error}");
    }

    #[test]
    fn struct_versions_follow_the_header_macro() {
        fn check<A: NvencApi>() {
            assert_eq!(A::CAPS_PARAM_VER, A::struct_version(1), "{}", A::VERSION);
            assert_eq!(
                A::OPEN_ENCODE_SESSION_EX_PARAMS_VER,
                A::struct_version(1),
                "{}",
                A::VERSION
            );
            assert_eq!(A::FUNCTION_LIST_VER, A::struct_version(2), "{}", A::VERSION);
            assert_eq!(A::CONFIG_VER >> 31, 1, "{}", A::VERSION);
            let version = A::VERSION.version();
            assert_eq!(
                A::API_VERSION,
                version.major | (version.minor << 24),
                "{}",
                A::VERSION
            );
        }
        struct Check;
        impl ApiVisitor for Check {
            type Output = ();
            fn visit<A: NvencApi>(self) {
                check::<A>();
            }
        }
        for version in ApiVersion::ALL {
            version.dispatch(Check);
        }
        // Spot checks against the headers.
        assert_eq!(
            api::V11_1::CONFIG_VER,
            api::V11_1::struct_version(7) | (1 << 31)
        );
        assert_eq!(
            api::V12_0::CONFIG_VER,
            api::V12_0::struct_version(8) | (1 << 31)
        );
        assert_eq!(
            api::V13_1::CONFIG_VER,
            api::V13_1::struct_version(9) | (1 << 31)
        );
        assert_eq!(
            api::V12_0::LOCK_BITSTREAM_VER,
            api::V12_0::struct_version(2)
        );
        assert_eq!(
            api::V12_1::LOCK_BITSTREAM_VER,
            api::V12_1::struct_version(1) | (1 << 31)
        );
    }
}
