//! DMA-BUF → CUDA import on real hardware, without a compositor: buffers are
//! allocated with GBM on the CUDA device's render node, filled by an
//! independent writer (GLES for tiled layouts, a separate CUDA registration
//! for LINEAR ones), imported, mapped, copied
//! to host memory and compared bit-exact.
//!
//! Every test prints why it skips when the machine has no NVIDIA GPU, EGL,
//! GBM or render node.

#![cfg(target_os = "linux")]
#![allow(
    clippy::missing_panics_doc,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::print_stderr,
    reason = "hardware tests: panics are failures, sizes are small and positive"
)]

use std::os::fd::{AsFd as _, FromRawFd as _, OwnedFd};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ferricast_nvcodec::Error;
use ferricast_nvcodec::cuda::{ArrayView, HostImageMut, Stream};
use ferricast_nvcodec::interop::{
    BufferKey, DmaBuf, DmaBufImporter, DrmFormat, DrmNode, FenceWait, FramePool, FrameView, Import,
    Modifier,
};

use support::{Bo, Gpu};

const FENCE: FenceWait = FenceWait::Implicit(Duration::from_secs(1));

/// The test pattern: four bytes per pixel, in memory order, distinct per
/// position and seed (including the fourth byte).
fn pattern(width: u32, height: u32, seed: u8) -> Vec<u8> {
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .flat_map(|(x, y)| {
            [
                (x as u8) ^ seed,
                (y as u8).wrapping_mul(3),
                ((x >> 8) as u8) ^ ((y >> 8) as u8).wrapping_add(seed),
                (x.wrapping_add(y) as u8) ^ 0x5a,
            ]
        })
        .collect()
}

/// Maps `key`, copies it through a pooled device buffer and returns the
/// pixels, tightly packed.
fn read_back(importer: &DmaBufImporter, key: BufferKey, stream: &Stream) -> Vec<u8> {
    let frame = importer.map(key, FENCE).unwrap();
    let pool = FramePool::new(
        importer.context(),
        frame.width() as usize * 4,
        frame.height() as usize,
    );
    let buffer = frame.copy_to_pool(&pool, stream).unwrap();
    stream.synchronize().unwrap();
    let mut host = vec![0; buffer.width_bytes() * buffer.rows()];
    let pitch = buffer.width_bytes();
    buffer
        .download(
            buffer.region(),
            HostImageMut {
                data: &mut host,
                pitch,
            },
        )
        .unwrap();
    host
}

fn assert_same(got: &[u8], want: &[u8], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(index) = got.iter().zip(want).position(|(a, b)| a != b) {
        let wrong = got.iter().zip(want).filter(|(a, b)| a != b).count();
        panic!(
            "{what}: {wrong} bytes differ, first at byte {index} (pixel {}): got {:?}, want {:?}",
            index / 4,
            &got[index / 4 * 4..index / 4 * 4 + 4],
            &want[index / 4 * 4..index / 4 * 4 + 4],
        );
    }
}

fn dma_buf(bo: &Bo) -> DmaBuf<'_> {
    DmaBuf {
        fd: bo.fd.as_fd(),
        width: bo.width,
        height: bo.height,
        format: bo.format,
        offset: bo.offset,
        stride: bo.stride,
        modifier: bo.modifier,
    }
}

/// Imports a buffer of every supported format and layout, written by an
/// independent writer, and reads it back bit-exact.
#[test]
fn imports_every_format_and_layout_bit_exact() {
    let Some(gpu) = Gpu::get() else { return };
    let mut importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
    let stream = Stream::new(&gpu.context).unwrap();
    let (width, height) = (1920, 1080);
    let mut key = 0;
    for format in DrmFormat::ALL {
        let tiled = importer
            .modifiers(format)
            .iter()
            .copied()
            .find(|modifier| *modifier != Modifier::LINEAR);
        assert!(
            importer.modifiers(format).contains(&Modifier::LINEAR),
            "LINEAR is importable for {format:?}"
        );
        for modifier in [Some(Modifier::LINEAR), tiled].into_iter().flatten() {
            let Some(bo) = gpu.allocate(width, height, format, modifier) else {
                continue;
            };
            let want = pattern(width, height, 7);
            gpu.write(&bo, &want);
            key += 1;
            assert_eq!(
                importer.import(BufferKey(key), &dma_buf(&bo)).unwrap(),
                Import::Registered
            );
            {
                let frame = importer.map(BufferKey(key), FENCE).unwrap();
                let tiled_layout = modifier != Modifier::LINEAR;
                assert_eq!(
                    matches!(frame.view(), FrameView::Array(_)),
                    tiled_layout,
                    "{format:?} {modifier:x?}: arrays for tiled layouts, pitched memory for LINEAR"
                );
                if let Some(array) = frame.array() {
                    assert_eq!((array.width(), array.height()), (width, height));
                }
            }
            let got = read_back(&importer, BufferKey(key), &stream);
            assert_same(&got, &want, &format!("{format:?} {modifier:x?}"));
            eprintln!("{format:?} modifier {:#018x}: bit-exact", modifier.0);
        }
    }
    assert!(key >= 4, "at least LINEAR of every format was tested");
}

/// A cached import is reused and aliases the buffer: new content shows
/// through without registering again.
#[test]
fn cached_import_is_reused_and_zero_copy() {
    let Some(gpu) = Gpu::get() else { return };
    let mut importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
    let stream = Stream::new(&gpu.context).unwrap();
    let (width, height) = (640, 360);
    for modifier in [Modifier::LINEAR, Gpu::tiled_modifier(&importer)] {
        let bo = gpu
            .allocate(width, height, DrmFormat::Xbgr8888, modifier)
            .unwrap();
        let key = BufferKey::from((1, modifier.0));
        gpu.write(&bo, &pattern(width, height, 1));
        assert_eq!(
            importer.import(key, &dma_buf(&bo)).unwrap(),
            Import::Registered
        );
        let array_before = importer
            .map(key, FENCE)
            .unwrap()
            .array()
            .map(ArrayView::raw);
        assert_same(
            &read_back(&importer, key, &stream),
            &pattern(width, height, 1),
            "first frame",
        );
        // The producer renders the next frame into the same buffer.
        gpu.write(&bo, &pattern(width, height, 2));
        assert_eq!(importer.import(key, &dma_buf(&bo)).unwrap(), Import::Cached);
        let array_after = importer
            .map(key, FENCE)
            .unwrap()
            .array()
            .map(ArrayView::raw);
        assert_eq!(array_before, array_after, "same registration");
        assert_same(
            &read_back(&importer, key, &stream),
            &pattern(width, height, 2),
            "second frame through the cached registration",
        );
        // The same key with another description is re-registered.
        let mut moved = dma_buf(&bo);
        moved.height -= 8;
        assert_eq!(importer.import(key, &moved).unwrap(), Import::Replaced);
        assert_eq!(importer.map(key, FENCE).unwrap().height(), height - 8);
    }
    assert_eq!(importer.len(), 2);
}

#[test]
fn forget_and_clear() {
    let Some(gpu) = Gpu::get() else { return };
    let mut importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
    let bos: Vec<Bo> = (0..3)
        .map(|_| {
            gpu.allocate(64, 64, DrmFormat::Argb8888, Modifier::LINEAR)
                .unwrap()
        })
        .collect();
    for (index, bo) in (0_u128..).zip(&bos) {
        importer.import(BufferKey(index), &dma_buf(bo)).unwrap();
    }
    assert_eq!(importer.len(), 3);
    assert!(importer.forget(BufferKey(1)));
    assert!(!importer.forget(BufferKey(1)));
    assert!(!importer.contains(BufferKey(1)));
    assert!(matches!(
        importer.map(BufferKey(1), FENCE),
        Err(Error::UnknownBuffer { key: 1 })
    ));
    assert!(importer.map(BufferKey(0), FENCE).is_ok());
    importer.clear();
    assert!(importer.is_empty());
    assert!(matches!(
        importer.map(BufferKey(0), FENCE),
        Err(Error::UnknownBuffer { .. })
    ));
    // Re-importing after a clear registers again.
    assert_eq!(
        importer.import(BufferKey(0), &dma_buf(&bos[0])).unwrap(),
        Import::Registered
    );
}

/// The importer keeps what it needs: the producer may free its buffer, and
/// the context, importer, pool and pooled buffers may go in any order.
#[test]
fn drop_order_is_free() {
    let Some(gpu) = Gpu::get() else { return };
    let (width, height) = (320, 240);
    let want = pattern(width, height, 9);
    let context = Arc::clone(&gpu.context);
    let mut importer = DmaBufImporter::new(&context, Some(gpu.node)).unwrap();
    let stream = Stream::new(&context).unwrap();
    let key = BufferKey(1);
    {
        let bo = gpu
            .allocate(
                width,
                height,
                DrmFormat::Abgr8888,
                Gpu::tiled_modifier(&importer),
            )
            .unwrap();
        gpu.write(&bo, &want);
        importer.import(key, &dma_buf(&bo)).unwrap();
        // The producer frees its buffer object and descriptor here.
    }
    assert_same(
        &read_back(&importer, key, &stream),
        &want,
        "after the producer freed the buffer",
    );
    let pool = FramePool::new(&context, width as usize * 4, height as usize);
    let pooled = importer
        .map(key, FENCE)
        .unwrap()
        .copy_to_pool(&pool, &stream)
        .unwrap();
    drop(context);
    drop(importer);
    drop(pool);
    stream.synchronize().unwrap();
    drop(pooled);
    drop(stream);
}

#[test]
fn rejects_bad_buffers() {
    let Some(gpu) = Gpu::get() else { return };
    let mut importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
    let bo = gpu
        .allocate(256, 128, DrmFormat::Xrgb8888, Modifier::LINEAR)
        .unwrap();

    // Larger than the buffer.
    let mut too_big = dma_buf(&bo);
    too_big.height = 4096;
    let error = importer.import(BufferKey(1), &too_big).unwrap_err();
    assert!(matches!(error, Error::InvalidDmaBuf { .. }), "{error}");

    // Stride smaller than a row.
    let mut narrow = dma_buf(&bo);
    narrow.stride = 256 * 4 - 4;
    assert!(matches!(
        importer.import(BufferKey(1), &narrow),
        Err(Error::InvalidDmaBuf { .. })
    ));

    // A modifier the GPU does not know.
    let mut odd = dma_buf(&bo);
    odd.modifier = Modifier(0x0123_4567);
    assert!(matches!(
        importer.import(BufferKey(1), &odd),
        Err(Error::UnsupportedModifier {
            modifier: 0x0123_4567,
            ..
        })
    ));

    // A descriptor that is not a DMA-BUF: a memfd of the right size.
    // SAFETY: plain syscalls; the name is NUL-terminated.
    let memfd = unsafe {
        let raw = libc::memfd_create(c"not-a-dmabuf".as_ptr(), libc::MFD_CLOEXEC);
        assert!(raw >= 0);
        assert_eq!(libc::ftruncate(raw, 256 * 4 * 128), 0);
        OwnedFd::from_raw_fd(raw)
    };
    let mut bogus = dma_buf(&bo);
    bogus.fd = memfd.as_fd();
    let error = importer.import(BufferKey(1), &bogus).unwrap_err();
    assert!(matches!(error, Error::Egl { .. }), "{error}");

    // `/dev/null` reports no size: refused before EGL sees it.
    let null = std::fs::File::open("/dev/null").unwrap();
    let mut empty = dma_buf(&bo);
    empty.fd = null.as_fd();
    assert!(matches!(
        importer.import(BufferKey(1), &empty),
        Err(Error::InvalidDmaBuf { .. })
    ));

    // Nothing failed half-way into the cache.
    assert!(importer.is_empty());
    assert!(matches!(
        DrmFormat::try_from(u32::from_le_bytes(*b"NV12")),
        Err(Error::UnsupportedFormat { .. })
    ));
}

#[test]
fn refuses_another_gpu() {
    let Some(gpu) = Gpu::get() else { return };
    // A DRM node number no GPU uses (major 226, minor 250).
    let elsewhere = DrmNode::from_dev(libc::makedev(226, 250));
    let error = DmaBufImporter::new(&gpu.context, Some(elsewhere)).unwrap_err();
    assert!(matches!(error, Error::DeviceMismatch { .. }), "{error}");
    eprintln!("{error}");
    // The primary node of the same GPU is accepted like its render node.
    let pci = gpu.context.device().pci_bus_id().unwrap();
    if let Some(card) = std::fs::read_dir(format!("/sys/bus/pci/devices/{pci}/drm"))
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .find(|name| name.starts_with("card"))
        && let Ok(node) = DrmNode::from_path(format!("/dev/dri/{card}"))
    {
        DmaBufImporter::new(&gpu.context, Some(node)).unwrap();
    }
}

#[test]
fn fence_wait_times_out_on_unsignalled_buffers_only() {
    let Some(gpu) = Gpu::get() else { return };
    let mut importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
    let bo = gpu
        .allocate(64, 64, DrmFormat::Argb8888, Modifier::LINEAR)
        .unwrap();
    importer.import(BufferKey(1), &dma_buf(&bo)).unwrap();
    // No producer work pending: an immediate wait succeeds.
    importer
        .map(BufferKey(1), FenceWait::Implicit(Duration::ZERO))
        .unwrap();
    importer
        .map(BufferKey(1), FenceWait::AlreadySignalled)
        .unwrap();
}

/// Per-frame cost of the steady state at 1920x1080: map (fence poll + frame
/// lookup), and map + device copy into a pooled buffer.
#[test]
fn per_frame_cost() {
    let Some(gpu) = Gpu::get() else { return };
    let mut importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
    let stream = Stream::new(&gpu.context).unwrap();
    let (width, height) = (1920, 1080);
    let pool = FramePool::new(&gpu.context, width as usize * 4, height as usize);
    for modifier in [Modifier::LINEAR, Gpu::tiled_modifier(&importer)] {
        let bo = gpu
            .allocate(width, height, DrmFormat::Xrgb8888, modifier)
            .unwrap();
        let key = BufferKey(u128::from(modifier.0));
        let start = Instant::now();
        importer.import(key, &dma_buf(&bo)).unwrap();
        let import = start.elapsed();

        let rounds = 1000;
        let start = Instant::now();
        for _ in 0..rounds {
            let frame = importer.map(key, FENCE).unwrap();
            std::hint::black_box(frame.view());
        }
        let map = start.elapsed() / rounds;

        let rounds = 300;
        let start = Instant::now();
        for _ in 0..rounds {
            let frame = importer.map(key, FENCE).unwrap();
            let buffer = frame.copy_to_pool(&pool, &stream).unwrap();
            stream.synchronize().unwrap();
            drop(buffer);
        }
        let copy = start.elapsed() / rounds;
        eprintln!(
            "1920x1080 modifier {:#018x}: import {import:?}, map {map:?}, map + device copy + sync {copy:?}",
            modifier.0
        );
    }
}

/// GBM, EGL and GLES, loaded at runtime like the crate loads its drivers.
mod support {
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
    type MakeCurrent =
        unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> u32;
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
                    std::mem::transmute::<*mut c_void, CreateImage>(proc_address(
                        c"eglCreateImageKHR",
                    )?)
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
        fn image(&self, bo: &super::Bo) -> *mut c_void {
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
        fn write(&self, bo: &super::Bo, pixels: &[u8]) {
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
}
