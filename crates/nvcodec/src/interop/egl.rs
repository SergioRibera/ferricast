//! The slice of EGL the importer needs, loaded at runtime from `libEGL.so.1`
//! (glvnd, which dispatches to the NVIDIA vendor library for NVIDIA devices).
//!
//! Only the device platform is used (`EGL_EXT_platform_device`): no window
//! system, no GL context. Displays are initialised once and never
//! terminated: an `EGLDisplay` is a process-wide singleton per device, and
//! `eglTerminate` would invalidate the images every other user of the same
//! display (a renderer) holds.

use std::ffi::{CStr, c_char, c_void};
use std::fmt;
use std::sync::OnceLock;

use libloading::Library;

use crate::error::{EglStatus, Error, Result};

pub(super) type EglInt = i32;
type EglBoolean = u32;
type EglEnum = u32;
type EglAttrib = isize;
pub(super) type EglDisplayHandle = *mut c_void;
pub(super) type EglDeviceHandle = *mut c_void;
pub(super) type EglImageHandle = *mut c_void;

const EGL_TRUE: EglBoolean = 1;
const EGL_EXTENSIONS: EglInt = 0x3055;
const EGL_PLATFORM_DEVICE_EXT: EglEnum = 0x313F;
pub(super) const EGL_DRM_DEVICE_FILE_EXT: EglInt = 0x3233;
pub(super) const EGL_DRM_RENDER_NODE_FILE_EXT: EglInt = 0x3377;
const EGL_CUDA_DEVICE_NV: EglInt = 0x323A;
const EGL_LINUX_DMA_BUF_EXT: EglEnum = 0x3270;
pub(super) const EGL_NONE: EglInt = 0x3038;
pub(super) const EGL_WIDTH: EglInt = 0x3057;
pub(super) const EGL_HEIGHT: EglInt = 0x3056;
pub(super) const EGL_LINUX_DRM_FOURCC_EXT: EglInt = 0x3271;
pub(super) const EGL_DMA_BUF_PLANE0_FD_EXT: EglInt = 0x3272;
pub(super) const EGL_DMA_BUF_PLANE0_OFFSET_EXT: EglInt = 0x3273;
pub(super) const EGL_DMA_BUF_PLANE0_PITCH_EXT: EglInt = 0x3274;
pub(super) const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: EglInt = 0x3443;
pub(super) const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: EglInt = 0x3444;

const LIBRARY: &str = "libEGL.so.1";

type GetProcAddress = unsafe extern "C" fn(name: *const c_char) -> *mut c_void;
type GetError = unsafe extern "C" fn() -> EglInt;
type QueryString = unsafe extern "C" fn(display: EglDisplayHandle, name: EglInt) -> *const c_char;
type Initialize = unsafe extern "C" fn(
    display: EglDisplayHandle,
    major: *mut EglInt,
    minor: *mut EglInt,
) -> EglBoolean;
type QueryDevices = unsafe extern "C" fn(
    max_devices: EglInt,
    devices: *mut EglDeviceHandle,
    num_devices: *mut EglInt,
) -> EglBoolean;
type QueryDeviceString =
    unsafe extern "C" fn(device: EglDeviceHandle, name: EglInt) -> *const c_char;
type QueryDeviceAttrib = unsafe extern "C" fn(
    device: EglDeviceHandle,
    attribute: EglInt,
    value: *mut EglAttrib,
) -> EglBoolean;
type GetPlatformDisplay = unsafe extern "C" fn(
    platform: EglEnum,
    native_display: *mut c_void,
    attributes: *const EglInt,
) -> EglDisplayHandle;
type CreateImage = unsafe extern "C" fn(
    display: EglDisplayHandle,
    context: *mut c_void,
    target: EglEnum,
    buffer: *mut c_void,
    attributes: *const EglInt,
) -> EglImageHandle;
type DestroyImage =
    unsafe extern "C" fn(display: EglDisplayHandle, image: EglImageHandle) -> EglBoolean;
type QueryDmaBufModifiers = unsafe extern "C" fn(
    display: EglDisplayHandle,
    format: EglInt,
    max_modifiers: EglInt,
    modifiers: *mut u64,
    external_only: *mut EglBoolean,
    num_modifiers: *mut EglInt,
) -> EglBoolean;

/// The loaded EGL library and the entry points the importer calls.
pub(super) struct Egl {
    get_error: GetError,
    query_string: QueryString,
    initialize: Initialize,
    query_devices: QueryDevices,
    query_device_string: QueryDeviceString,
    query_device_attrib: QueryDeviceAttrib,
    get_platform_display: GetPlatformDisplay,
    create_image: CreateImage,
    destroy_image: DestroyImage,
    query_dmabuf_modifiers: Option<QueryDmaBufModifiers>,
    _library: Library,
}

static EGL: OnceLock<Result<Egl>> = OnceLock::new();

impl Egl {
    /// The process-wide EGL library, loaded on first use and never unloaded
    /// (the copied function pointers stay valid).
    pub(super) fn get() -> Result<&'static Self> {
        EGL.get_or_init(Self::load).as_ref().map_err(Clone::clone)
    }

    fn load() -> Result<Self> {
        // SAFETY: libEGL (glvnd) runs no initialisation with requirements on
        // the caller.
        let library = unsafe { Library::new(LIBRARY) }.map_err(|error| Error::LibraryMissing {
            library: LIBRARY,
            reason: error.to_string(),
        })?;
        // SAFETY: the type is `eglGetProcAddress`'s, from `egl.h`.
        let get_proc_address: GetProcAddress = unsafe { symbol(&library, "eglGetProcAddress")? };
        // SAFETY: each type is the entry point's, from `egl.h`.
        let (get_error, query_string, initialize) = unsafe {
            (
                symbol::<GetError>(&library, "eglGetError")?,
                symbol::<QueryString>(&library, "eglQueryString")?,
                symbol::<Initialize>(&library, "eglInitialize")?,
            )
        };
        let client = client_extensions(query_string);
        for required in ["EGL_EXT_device_enumeration", "EGL_EXT_platform_device"] {
            if !has_extension(&client, required) {
                return Err(Error::Unsupported {
                    what: "DMA-BUF import through EGL",
                    reason: format!("{LIBRARY} lacks the client extension {required}"),
                });
            }
        }
        // SAFETY: each type is the extension function's, from `eglext.h`.
        unsafe {
            Ok(Self {
                get_error,
                query_string,
                initialize,
                query_devices: extension(get_proc_address, "eglQueryDevicesEXT")?,
                query_device_string: extension(get_proc_address, "eglQueryDeviceStringEXT")?,
                query_device_attrib: extension(get_proc_address, "eglQueryDeviceAttribEXT")?,
                get_platform_display: extension(get_proc_address, "eglGetPlatformDisplayEXT")?,
                create_image: extension(get_proc_address, "eglCreateImageKHR")?,
                destroy_image: extension(get_proc_address, "eglDestroyImageKHR")?,
                query_dmabuf_modifiers: extension(get_proc_address, "eglQueryDmaBufModifiersEXT")
                    .ok(),
                _library: library,
            })
        }
    }

    /// The error of the last failed EGL call on this thread.
    fn error(&self, call: &'static str) -> Error {
        // SAFETY: no arguments; reads thread-local EGL state.
        let status = unsafe { (self.get_error)() };
        Error::Egl {
            call,
            status: EglStatus(status),
        }
    }

    /// Every EGL device glvnd exposes, all vendors included.
    pub(super) fn devices(&'static self) -> Result<Vec<EglDevice>> {
        let mut count = 0;
        // SAFETY: a null array with 0 entries asks for the count only.
        let ok = unsafe { (self.query_devices)(0, std::ptr::null_mut(), &raw mut count) };
        if ok != EGL_TRUE {
            return Err(self.error("eglQueryDevicesEXT"));
        }
        let mut handles = vec![std::ptr::null_mut(); usize::try_from(count).unwrap_or(0)];
        // SAFETY: the array holds `count` entries.
        let ok = unsafe { (self.query_devices)(count, handles.as_mut_ptr(), &raw mut count) };
        if ok != EGL_TRUE {
            return Err(self.error("eglQueryDevicesEXT"));
        }
        handles.truncate(usize::try_from(count).unwrap_or(0));
        Ok(handles
            .into_iter()
            .map(|raw| EglDevice { raw, egl: self })
            .collect())
    }
}

/// Resolves an exported symbol.
///
/// # Safety
///
/// `F` must be the symbol's function pointer type.
unsafe fn symbol<F: Copy>(library: &Library, name: &'static str) -> Result<F> {
    // SAFETY: the caller guarantees the type; the pointer is copied out and
    // stays valid because the library is never unloaded.
    unsafe { library.get::<F>(name) }
        .map(|symbol| *symbol)
        .map_err(|error| Error::SymbolMissing {
            library: LIBRARY,
            symbol: name,
            reason: error.to_string(),
        })
}

/// Resolves an extension entry point through `eglGetProcAddress`.
///
/// # Safety
///
/// `F` must be the function pointer type of `name`.
unsafe fn extension<F: Copy>(get_proc_address: GetProcAddress, name: &'static str) -> Result<F> {
    const { assert!(size_of::<F>() == size_of::<*mut c_void>()) };
    let mut text = name.as_bytes().to_vec();
    text.push(0);
    // SAFETY: `text` is NUL-terminated and outlives the call.
    let pointer = unsafe { get_proc_address(text.as_ptr().cast()) };
    if pointer.is_null() {
        return Err(Error::SymbolMissing {
            library: LIBRARY,
            symbol: name,
            reason: "eglGetProcAddress returned null".into(),
        });
    }
    // SAFETY: a non-null pointer from `eglGetProcAddress` is the entry point
    // named, of type `F` (the caller's guarantee); sizes match (asserted).
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&pointer) })
}

fn client_extensions(query_string: QueryString) -> String {
    // SAFETY: `EGL_NO_DISPLAY` with `EGL_EXTENSIONS` returns the client
    // extensions (`EGL_EXT_client_extensions`) or null.
    let text = unsafe { query_string(std::ptr::null_mut(), EGL_EXTENSIONS) };
    // SAFETY: EGL returns a static NUL-terminated string or null.
    unsafe { c_text(text) }.unwrap_or_default()
}

/// Copies a string EGL returned.
///
/// # Safety
///
/// `text` is null or a NUL-terminated string valid for the call.
unsafe fn c_text(text: *const c_char) -> Option<String> {
    (!text.is_null()).then(|| {
        // SAFETY: non-null, and NUL-terminated as the caller guarantees.
        unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned()
    })
}

pub(super) fn has_extension(list: &str, name: &str) -> bool {
    list.split_ascii_whitespace().any(|item| item == name)
}

/// An EGL device (`EGLDeviceEXT`): a GPU as one vendor library sees it.
#[derive(Clone, Copy)]
pub(super) struct EglDevice {
    raw: EglDeviceHandle,
    egl: &'static Egl,
}

impl EglDevice {
    /// A device string (`eglQueryDeviceStringEXT`), `None` if the device
    /// does not have it.
    pub(super) fn string(self, name: EglInt) -> Option<String> {
        // SAFETY: live device handle; EGL returns a static string or null.
        unsafe { c_text((self.egl.query_device_string)(self.raw, name)) }
    }

    /// The device's extensions.
    pub(super) fn extensions(self) -> String {
        self.string(EGL_EXTENSIONS).unwrap_or_default()
    }

    /// The CUDA device ordinal (`EGL_NV_device_cuda`), for NVIDIA devices.
    pub(super) fn cuda_device(self) -> Option<i32> {
        if !has_extension(&self.extensions(), "EGL_NV_device_cuda") {
            return None;
        }
        let mut value: EglAttrib = 0;
        // SAFETY: live device handle, valid out-pointer.
        let ok =
            unsafe { (self.egl.query_device_attrib)(self.raw, EGL_CUDA_DEVICE_NV, &raw mut value) };
        (ok == EGL_TRUE)
            .then(|| i32::try_from(value).ok())
            .flatten()
    }

    /// The display of this device, initialised.
    pub(super) fn display(self) -> Result<EglDisplay> {
        let attributes = [EGL_NONE];
        // SAFETY: live device handle; the attribute list is terminated.
        let raw = unsafe {
            (self.egl.get_platform_display)(EGL_PLATFORM_DEVICE_EXT, self.raw, attributes.as_ptr())
        };
        if raw.is_null() {
            return Err(self.egl.error("eglGetPlatformDisplayEXT"));
        }
        let (mut major, mut minor) = (0, 0);
        // SAFETY: a display handle EGL returned; valid out-pointers.
        // Initialising an initialised display is a no-op.
        let ok = unsafe { (self.egl.initialize)(raw, &raw mut major, &raw mut minor) };
        if ok != EGL_TRUE {
            return Err(self.egl.error("eglInitialize"));
        }
        Ok(EglDisplay { raw, egl: self.egl })
    }
}

/// An initialised EGL display of one device; never terminated (see the
/// module docs).
#[derive(Clone, Copy)]
pub(super) struct EglDisplay {
    raw: EglDisplayHandle,
    egl: &'static Egl,
}

// SAFETY: EGL is thread-safe (EGL 1.5 §3.9: every call may be made from any
// thread) and the handle is a process-wide singleton that is never
// terminated, so it is valid on every thread.
unsafe impl Send for EglDisplay {}
// SAFETY: as for `Send`; the type exposes no interior state.
unsafe impl Sync for EglDisplay {}

impl EglDisplay {
    /// The display's extensions.
    pub(super) fn extensions(self) -> String {
        // SAFETY: initialised display; EGL returns a static string or null.
        unsafe { c_text((self.egl.query_string)(self.raw, EGL_EXTENSIONS)) }.unwrap_or_default()
    }

    /// The modifiers EGL can import for `fourcc`
    /// (`eglQueryDmaBufModifiersEXT`), empty without
    /// `EGL_EXT_image_dma_buf_import_modifiers`.
    pub(super) fn modifiers(self, fourcc: u32) -> Result<Vec<u64>> {
        let Some(query) = self.egl.query_dmabuf_modifiers else {
            return Ok(Vec::new());
        };
        let format = EglInt::from_ne_bytes(fourcc.to_ne_bytes());
        let mut count = 0;
        // SAFETY: initialised display; null arrays with 0 entries ask for the
        // count only.
        let ok = unsafe {
            query(
                self.raw,
                format,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &raw mut count,
            )
        };
        if ok != EGL_TRUE {
            return Err(self.egl.error("eglQueryDmaBufModifiersEXT"));
        }
        let mut modifiers = vec![0_u64; usize::try_from(count).unwrap_or(0)];
        let mut external_only = vec![0; modifiers.len()];
        // SAFETY: both arrays hold `count` entries.
        let ok = unsafe {
            query(
                self.raw,
                format,
                count,
                modifiers.as_mut_ptr(),
                external_only.as_mut_ptr(),
                &raw mut count,
            )
        };
        if ok != EGL_TRUE {
            return Err(self.egl.error("eglQueryDmaBufModifiersEXT"));
        }
        // `external_only` restricts GL to `GL_TEXTURE_EXTERNAL_OES`; CUDA's
        // EGL interop is not GL and imports those too (LINEAR is
        // external-only on NVIDIA and registers fine).
        modifiers.truncate(usize::try_from(count).unwrap_or(0));
        Ok(modifiers)
    }

    /// Creates an EGL image of a DMA-BUF (`EGL_LINUX_DMA_BUF_EXT`) from a
    /// terminated attribute list.
    pub(super) fn create_dmabuf_image(self, attributes: &[EglInt]) -> Result<EglImage> {
        debug_assert_eq!(attributes.last(), Some(&EGL_NONE));
        // SAFETY: initialised display; `EGL_LINUX_DMA_BUF_EXT` takes no
        // context and a null client buffer; the attribute list is terminated
        // and outlives the call. EGL dups the file descriptors it is given.
        let raw = unsafe {
            (self.egl.create_image)(
                self.raw,
                std::ptr::null_mut(),
                EGL_LINUX_DMA_BUF_EXT,
                std::ptr::null_mut(),
                attributes.as_ptr(),
            )
        };
        if raw.is_null() {
            return Err(self.egl.error("eglCreateImageKHR"));
        }
        Ok(EglImage { raw, display: self })
    }
}

impl fmt::Debug for EglDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("EglDisplay").field(&self.raw).finish()
    }
}

/// An EGL image, destroyed on drop.
pub(super) struct EglImage {
    raw: EglImageHandle,
    display: EglDisplay,
}

// SAFETY: an `EGLImageKHR` is a handle into a thread-safe API with no thread
// affinity; this value owns it exclusively.
unsafe impl Send for EglImage {}

impl EglImage {
    /// The handle, valid while `self` lives.
    pub(super) fn raw(&self) -> EglImageHandle {
        self.raw
    }
}

impl Drop for EglImage {
    fn drop(&mut self) {
        // SAFETY: this value created the image on this display and owns it.
        let ok = unsafe { (self.display.egl.destroy_image)(self.display.raw, self.raw) };
        if ok != EGL_TRUE {
            let error = self.display.egl.error("eglDestroyImageKHR");
            tracing::warn!(%error, "destroying an EGL image");
        }
    }
}

impl fmt::Debug for EglImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("EglImage").field(&self.raw).finish()
    }
}
