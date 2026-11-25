//! GBM, EGL and GLES test support, loaded at runtime like the crate loads
//! its drivers: allocates DMA-BUFs on the NVIDIA GPU and writes them with a
//! writer independent of the importer. Shared with ferricast's NVENC
//! end-to-end tests (`#[path]`-included).

use std::ffi::{CStr, c_char, c_int, c_void};
use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use ferricast_nvcodec::cuda::{Context, Device};
use ferricast_nvcodec::interop::{DmaBufImporter, DrmFormat, DrmNode, Modifier};
use ferricast_nvcodec::loader::CudaLibrary;
use ferricast_nvcodec::sys::cuda as sys;
use libloading::Library;

const GBM_BO_USE_RENDERING: u32 = 1 << 2;
const EGL_EXTENSIONS: i32 = 0x3055;
const EGL_NONE: i32 = 0x3038;
const EGL_PLATFORM_DEVICE_EXT: u32 = 0x313F;
const EGL_OPENGL_ES_API: u32 = 0x30A0;
const EGL_CONTEXT_CLIENT_VERSION: i32 = 0x3098;
const EGL_LINUX_DMA_BUF_EXT: u32 = 0x3270;
const GL_TEXTURE_2D: u32 = 0x0DE1;
const GL_RGBA: u32 = 0x1908;
const GL_UNSIGNED_BYTE: u32 = 0x1401;
const GL_UNPACK_ALIGNMENT: u32 = 0x0CF5;

/// Loads `name` from `library` as `F`.
fn symbol<F: Copy>(library: &Library, name: &str) -> Result<F, String> {
    // SAFETY: test-only; every call site names the C prototype it uses.
    unsafe { library.get::<F>(name.as_bytes()) }
        .map(|symbol| *symbol)
        .map_err(|error| format!("{name}: {error}"))
}

type GbmCreateDevice = unsafe extern "C" fn(c_int) -> *mut c_void;
type GbmBoCreateWithModifiers2 =
    unsafe extern "C" fn(*mut c_void, u32, u32, u32, *const u64, u32, u32) -> *mut c_void;
type GbmBoGetFd = unsafe extern "C" fn(*mut c_void) -> c_int;
type GbmBoGetU32 = unsafe extern "C" fn(*mut c_void) -> u32;
type GbmBoGetOffset = unsafe extern "C" fn(*mut c_void, c_int) -> u32;
type GbmBoGetModifier = unsafe extern "C" fn(*mut c_void) -> u64;
type GbmBoDestroy = unsafe extern "C" fn(*mut c_void);

struct Gbm {
    device: usize,
    create: GbmBoCreateWithModifiers2,
    get_fd: GbmBoGetFd,
    get_stride: GbmBoGetU32,
    get_offset: GbmBoGetOffset,
    get_modifier: GbmBoGetModifier,
    destroy: GbmBoDestroy,
    _node: std::fs::File,
    _library: Library,
}

impl Gbm {
    fn open(path: &PathBuf) -> Result<Self, String> {
        // SAFETY: libgbm runs no initialisation with requirements.
        let library = unsafe { Library::new("libgbm.so.1") }.map_err(|e| e.to_string())?;
        let node = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let create_device: GbmCreateDevice = symbol(&library, "gbm_create_device")?;
        // SAFETY: a valid DRM node descriptor, kept open by `_node`.
        let device = unsafe { create_device(node.as_raw_fd()) };
        if device.is_null() {
            return Err("gbm_create_device failed".into());
        }
        Ok(Self {
            device: device.expose_provenance(),
            create: symbol(&library, "gbm_bo_create_with_modifiers2")?,
            get_fd: symbol(&library, "gbm_bo_get_fd")?,
            get_stride: symbol(&library, "gbm_bo_get_stride")?,
            get_offset: symbol(&library, "gbm_bo_get_offset")?,
            get_modifier: symbol(&library, "gbm_bo_get_modifier")?,
            destroy: symbol(&library, "gbm_bo_destroy")?,
            _node: node,
            _library: library,
        })
    }
}

type GetProcAddress = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type QueryDevices = unsafe extern "C" fn(i32, *mut *mut c_void, *mut i32) -> u32;
type QueryDeviceString = unsafe extern "C" fn(*mut c_void, i32) -> *const c_char;
type GetPlatformDisplay = unsafe extern "C" fn(u32, *mut c_void, *const i32) -> *mut c_void;
type Initialize = unsafe extern "C" fn(*mut c_void, *mut i32, *mut i32) -> u32;
type BindApi = unsafe extern "C" fn(u32) -> u32;
type CreateContext =
    unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *const i32) -> *mut c_void;
type MakeCurrent = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> u32;
type CreateImage =
    unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *mut c_void, *const i32) -> *mut c_void;
type DestroyImage = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
type ImageTargetTexture = unsafe extern "C" fn(u32, *mut c_void);
type GenTextures = unsafe extern "C" fn(i32, *mut u32);
type BindTexture = unsafe extern "C" fn(u32, u32);
type PixelStorei = unsafe extern "C" fn(u32, i32);
type TexSubImage = unsafe extern "C" fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void);
type Finish = unsafe extern "C" fn();
type GetError = unsafe extern "C" fn() -> u32;

/// A GLES context on the NVIDIA EGL device, current on the thread that
/// uses it (one per call: tests run on several threads).
struct Gles {
    display: usize,
    bind_api: BindApi,
    create_context: CreateContext,
    make_current: MakeCurrent,
    create_image: CreateImage,
    destroy_image: DestroyImage,
    image_target: ImageTargetTexture,
    gen_textures: GenTextures,
    bind_texture: BindTexture,
    pixel_store: PixelStorei,
    tex_sub_image: TexSubImage,
    finish: Finish,
    get_error: GetError,
    _egl_library: Library,
    _gles_library: Library,
}

impl Gles {
    fn open() -> Result<Self, String> {
        // SAFETY: glvnd's libraries run no initialisation with
        // requirements.
        let egl = unsafe { Library::new("libEGL.so.1") }.map_err(|e| e.to_string())?;
        // SAFETY: as above.
        let gles = unsafe { Library::new("libGLESv2.so.2") }.map_err(|e| e.to_string())?;
        let get_proc: GetProcAddress = symbol(&egl, "eglGetProcAddress")?;
        let proc_address = |name: &CStr| {
            // SAFETY: NUL-terminated name.
            let pointer = unsafe { get_proc(name.as_ptr()) };
            (!pointer.is_null())
                .then_some(pointer)
                .ok_or(format!("no {name:?}"))
        };
        // SAFETY (all transmutes below): `eglGetProcAddress` returns the
        // entry point named, whose C prototype is the target type.
        let query_devices: QueryDevices =
            unsafe { std::mem::transmute(proc_address(c"eglQueryDevicesEXT")?) };
        let query_string: QueryDeviceString =
            unsafe { std::mem::transmute(proc_address(c"eglQueryDeviceStringEXT")?) };
        let platform_display: GetPlatformDisplay =
            unsafe { std::mem::transmute(proc_address(c"eglGetPlatformDisplayEXT")?) };
        let mut devices = [std::ptr::null_mut(); 16];
        let mut count = 0;
        // SAFETY: the array holds 16 entries.
        unsafe { query_devices(16, devices.as_mut_ptr(), &raw mut count) };
        let device = devices[..count as usize]
            .iter()
            .copied()
            .find(|&device| {
                // SAFETY: a device EGL returned; static string or null.
                let extensions = unsafe { query_string(device, EGL_EXTENSIONS) };
                !extensions.is_null()
                    // SAFETY: non-null NUL-terminated string.
                    && unsafe { CStr::from_ptr(extensions) }
                        .to_string_lossy()
                        .contains("EGL_NV_device_cuda")
            })
            .ok_or("no NVIDIA EGL device")?;
        let attributes = [EGL_NONE];
        // SAFETY: a device handle and a terminated list.
        let display =
            unsafe { platform_display(EGL_PLATFORM_DEVICE_EXT, device, attributes.as_ptr()) };
        let initialize: Initialize = symbol(&egl, "eglInitialize")?;
        // SAFETY: a display EGL returned.
        if unsafe { initialize(display, std::ptr::null_mut(), std::ptr::null_mut()) } != 1 {
            return Err("eglInitialize failed".into());
        }
        Ok(Self {
            display: display.expose_provenance(),
            bind_api: symbol(&egl, "eglBindAPI")?,
            create_context: symbol(&egl, "eglCreateContext")?,
            make_current: symbol(&egl, "eglMakeCurrent")?,
            create_image: unsafe {
                std::mem::transmute::<*mut c_void, CreateImage>(proc_address(c"eglCreateImageKHR")?)
            },
            destroy_image: unsafe {
                std::mem::transmute::<*mut c_void, DestroyImage>(proc_address(
                    c"eglDestroyImageKHR",
                )?)
            },
            image_target: unsafe {
                std::mem::transmute::<*mut c_void, ImageTargetTexture>(proc_address(
                    c"glEGLImageTargetTexture2DOES",
                )?)
            },
            gen_textures: symbol(&gles, "glGenTextures")?,
            bind_texture: symbol(&gles, "glBindTexture")?,
            pixel_store: symbol(&gles, "glPixelStorei")?,
            tex_sub_image: symbol(&gles, "glTexSubImage2D")?,
            finish: symbol(&gles, "glFinish")?,
            get_error: symbol(&gles, "glGetError")?,
            _egl_library: egl,
            _gles_library: gles,
        })
    }

    fn display(&self) -> *mut c_void {
        std::ptr::with_exposed_provenance_mut(self.display)
    }

    /// An EGL image of `bo` (as the alpha variant of its format, like
    /// the importer registers it).
    fn image(&self, bo: &Bo) -> *mut c_void {
        let modifier = bo.modifier.0;
        let attributes = [
            0x3057,
            bo.width as i32,
            0x3056,
            bo.height as i32,
            0x3271,
            bo.format.registered_as().fourcc() as i32,
            0x3272,
            bo.fd.as_raw_fd(),
            0x3273,
            bo.offset as i32,
            0x3274,
            bo.stride as i32,
            0x3443,
            (modifier & 0xffff_ffff) as u32 as i32,
            0x3444,
            (modifier >> 32) as u32 as i32,
            EGL_NONE,
        ];
        // SAFETY: initialised display, terminated attributes.
        let image = unsafe {
            (self.create_image)(
                self.display(),
                std::ptr::null_mut(),
                EGL_LINUX_DMA_BUF_EXT,
                std::ptr::null_mut(),
                attributes.as_ptr(),
            )
        };
        assert!(!image.is_null(), "test writer: eglCreateImageKHR failed");
        image
    }

    /// Writes `pixels` (memory order) into a tiled `bo` through GLES.
    fn write(&self, bo: &Bo, pixels: &[u8]) {
        // GL uploads RGBA in logical order; ARGB8888 memory is B, G, R,
        // A, so swap R and B for the ARGB/XRGB formats.
        let swap = bo.format.registered_as() == DrmFormat::Argb8888;
        let upload: Vec<u8> = pixels
            .chunks_exact(4)
            .flat_map(|p| {
                if swap {
                    [p[2], p[1], p[0], p[3]]
                } else {
                    [p[0], p[1], p[2], p[3]]
                }
            })
            .collect();
        let context_attributes = [EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE];
        let null = std::ptr::null_mut();
        // SAFETY: test-only GLES calls on a context made current on this
        // thread, released before returning; every pointer is live for
        // the call.
        unsafe {
            assert_eq!((self.bind_api)(EGL_OPENGL_ES_API), 1);
            let context =
                (self.create_context)(self.display(), null, null, context_attributes.as_ptr());
            assert!(!context.is_null(), "eglCreateContext");
            assert_eq!((self.make_current)(self.display(), null, null, context), 1);
            let image = self.image(bo);
            let mut texture = 0;
            (self.gen_textures)(1, &raw mut texture);
            (self.bind_texture)(GL_TEXTURE_2D, texture);
            (self.image_target)(GL_TEXTURE_2D, image);
            (self.pixel_store)(GL_UNPACK_ALIGNMENT, 1);
            (self.tex_sub_image)(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                bo.width as i32,
                bo.height as i32,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                upload.as_ptr().cast(),
            );
            (self.finish)();
            assert_eq!((self.get_error)(), 0, "GL error writing the buffer");
            (self.make_current)(self.display(), null, null, null);
            (self.destroy_image)(self.display(), image);
        }
    }
}

/// The machine's NVIDIA GPU, its render node and the writers.
pub(super) struct Gpu {
    pub(super) context: Arc<Context>,
    pub(super) node: DrmNode,
    gbm: Gbm,
    gles: Gles,
}

// SAFETY: test-only; GBM and EGL are used from several test threads with
// handles that have no thread affinity (GBM allocation takes no thread
// state; each GLES write makes its own context current).
unsafe impl Send for Gpu {}
// SAFETY: as above.
unsafe impl Sync for Gpu {}

static GPU: OnceLock<Result<Gpu, String>> = OnceLock::new();

impl Gpu {
    /// The shared setup, or `None` (with the reason printed) when the
    /// machine cannot run these tests.
    pub(super) fn get() -> Option<&'static Self> {
        match GPU.get_or_init(Self::open) {
            Ok(gpu) => Some(gpu),
            Err(reason) => {
                eprintln!("skipping DMA-BUF interop test: {reason}");
                None
            }
        }
    }

    fn open() -> Result<Self, String> {
        let device = Device::get(0).map_err(|e| format!("no CUDA device: {e}"))?;
        let context = Context::primary(device).map_err(|e| e.to_string())?;
        let pci = device.pci_bus_id().map_err(|e| e.to_string())?;
        let node = DrmNode::render_node_of(pci).map_err(|e| e.to_string())?;
        let name = node.name().ok_or("render node without a name")?;
        let gbm = Gbm::open(&PathBuf::from(format!("/dev/{name}")))?;
        let gles = Gles::open()?;
        DmaBufImporter::new(&context, Some(node)).map_err(|e| format!("no interop: {e}"))?;
        Ok(Self {
            context,
            node,
            gbm,
            gles,
        })
    }

    /// A tiled modifier the importer accepts.
    pub(super) fn tiled_modifier(importer: &DmaBufImporter) -> Modifier {
        importer
            .modifiers(DrmFormat::Argb8888)
            .iter()
            .copied()
            .find(|modifier| *modifier != Modifier::LINEAR)
            .expect("the NVIDIA driver offers tiled modifiers")
    }

    /// Allocates a buffer with exactly `modifier`; `None` (printed) when
    /// GBM refuses that combination.
    pub(super) fn allocate(
        &self,
        width: u32,
        height: u32,
        format: DrmFormat,
        modifier: Modifier,
    ) -> Option<Bo> {
        let flags = if modifier == Modifier::LINEAR {
            0
        } else {
            GBM_BO_USE_RENDERING
        };
        let device = std::ptr::with_exposed_provenance_mut(self.gbm.device);
        // SAFETY: a live GBM device; one modifier in the list.
        let raw = unsafe {
            (self.gbm.create)(
                device,
                width,
                height,
                format.fourcc(),
                &raw const modifier.0,
                1,
                flags,
            )
        };
        if raw.is_null() {
            eprintln!(
                "GBM cannot allocate {format:?} with modifier {:#x}",
                modifier.0
            );
            return None;
        }
        // SAFETY: a live buffer object; plane 0 exists.
        let (fd, stride, offset, actual) = unsafe {
            (
                (self.gbm.get_fd)(raw),
                (self.gbm.get_stride)(raw),
                (self.gbm.get_offset)(raw, 0),
                (self.gbm.get_modifier)(raw),
            )
        };
        assert!(fd >= 0, "gbm_bo_get_fd");
        assert_eq!(actual, modifier.0);
        Some(Bo {
            raw: raw.expose_provenance(),
            destroy: self.gbm.destroy,
            // SAFETY: GBM returned a new descriptor the caller owns.
            fd: unsafe { OwnedFd::from_raw_fd(fd) },
            width,
            height,
            format,
            offset,
            stride,
            modifier,
        })
    }

    /// Writes `pixels` (memory order, tightly packed) into `bo` with a
    /// writer independent of the importer: GLES for tiled layouts; for
    /// LINEAR (which GLES cannot render to on NVIDIA: external-only) a
    /// separate, writable CUDA registration of its own EGL image. (CPU
    /// `mmap` of NVIDIA LINEAR GBM buffers does not see the GPU's view
    /// past the first pages on driver 595, so it is no witness.)
    pub(super) fn write(&self, bo: &Bo, pixels: &[u8]) {
        if bo.modifier == Modifier::LINEAR {
            self.write_with_cuda(bo, pixels);
        } else {
            self.gles.write(bo, pixels);
        }
    }

    fn write_with_cuda(&self, bo: &Bo, pixels: &[u8]) {
        let cuda = CudaLibrary::get().unwrap();
        let image = self.gles.image(bo);
        let _current = self.context.guard().unwrap();
        let mut resource = std::ptr::null_mut();
        let mut frame = sys::CUeglFrame::default();
        // SAFETY: test-only driver calls with live handles and valid
        // out-pointers; the copy writes `height` rows of `width * 4`
        // bytes into the pitched image CUDA reports.
        unsafe {
            assert_eq!(
                (cuda.graphics_egl_register_image.unwrap())(&raw mut resource, image, 0),
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                (cuda.graphics_resource_get_mapped_egl_frame.unwrap())(
                    &raw mut frame,
                    resource,
                    0,
                    0
                ),
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                frame.frameType,
                sys::CUeglFrameType::CU_EGL_FRAME_TYPE_PITCH
            );
            let copy = sys::CUDA_MEMCPY2D {
                srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_HOST,
                srcHost: pixels.as_ptr().cast(),
                srcPitch: bo.width as usize * 4,
                dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                dstDevice: frame.frame.pPitch[0].addr() as u64,
                dstPitch: frame.pitch as usize,
                WidthInBytes: bo.width as usize * 4,
                Height: bo.height as usize,
                ..Default::default()
            };
            assert_eq!(
                (cuda.memcpy_2d)(&raw const copy),
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!((cuda.ctx_synchronize)(), sys::CUresult::CUDA_SUCCESS);
            (cuda.graphics_unregister_resource.unwrap())(resource);
            (self.gles.destroy_image)(self.gles.display(), image);
        }
    }
}

/// A GBM buffer object and its DMA-BUF.
pub(super) struct Bo {
    raw: usize,
    destroy: GbmBoDestroy,
    pub(super) fd: OwnedFd,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) format: DrmFormat,
    pub(super) offset: u32,
    pub(super) stride: u32,
    pub(super) modifier: Modifier,
}

impl Drop for Bo {
    fn drop(&mut self) {
        // SAFETY: this value owns the buffer object.
        unsafe { (self.destroy)(std::ptr::with_exposed_provenance_mut(self.raw)) };
    }
}
