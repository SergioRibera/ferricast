//! One-shot Wayland thumbnail capture.
//!
//! Two protocols, two flows, one file because they share most of
//! the support code (wl_shm pool allocation, format conversion,
//! PNG encode, output lookup).
//!
//! ### Monitor thumbnails — `zwlr_screencopy_unstable_v1`
//!
//! Bind the screencopy manager, find the wl_output that matches
//! the picker-issued id, call `capture_output`, wait for the
//! `buffer` event to know the pixel format, allocate a wl_shm
//! buffer of that size, `copy()` into it, wait for `ready` (or
//! `failed`), mmap the fd, swizzle to RGBA, downscale, PNG-encode.
//!
//! Supported on every wlroots compositor, niri, KDE Plasma 6.
//!
//! ### Window thumbnails — `ext-image-capture-source-v1` + `ext-image-copy-capture-v1`
//!
//! Bind `ext_foreign_toplevel_list_v1` to find the toplevel handle
//! whose `identifier` matches our id, then bind the foreign-toplevel
//! image-capture-source manager and turn that handle into a capture
//! source. Bind the image-copy-capture manager, create a session
//! from the source, wait for `buffer_size` + `shm_format` + `done`,
//! allocate the shm buffer, `create_frame`, `attach_buffer`,
//! `capture`, wait for `ready`, read back same as above.
//!
//! Supported today on niri and Hyprland ≥ 0.46. On compositors
//! that don't expose it we return [`SourceError::Unsupported`] and
//! the daemon translates that to an empty `ay` so pickers can show
//! a placeholder.
//!
//! Each call opens its own wayland connection. Cheaper than
//! marshalling work onto the enumerator's queue thread and avoids
//! contention with the live-snapshot dispatch.

use std::collections::HashMap;
use std::os::fd::{AsFd, AsRawFd, OwnedFd};

use ferricast_core::SourceError;
use tracing::debug;

use wayland_client::{
    Connection, Dispatch, Proxy, QueueHandle, WEnum,
    globals::{GlobalListContents, registry_queue_init},
    protocol::{
        wl_buffer::WlBuffer,
        wl_output::{self, WlOutput},
        wl_registry::WlRegistry,
        wl_shm::{self, WlShm},
        wl_shm_pool::WlShmPool,
    },
};
use wayland_protocols::ext::foreign_toplevel_list::v1::client::{
    ext_foreign_toplevel_handle_v1::{self, ExtForeignToplevelHandleV1},
    ext_foreign_toplevel_list_v1::{self, ExtForeignToplevelListV1},
};
use wayland_protocols::ext::image_capture_source::v1::client::{
    ext_foreign_toplevel_image_capture_source_manager_v1::ExtForeignToplevelImageCaptureSourceManagerV1,
    ext_image_capture_source_v1::ExtImageCaptureSourceV1,
    ext_output_image_capture_source_manager_v1::ExtOutputImageCaptureSourceManagerV1,
};
use wayland_protocols::ext::image_copy_capture::v1::client::{
    ext_image_copy_capture_frame_v1::{self, ExtImageCopyCaptureFrameV1},
    ext_image_copy_capture_manager_v1::{ExtImageCopyCaptureManagerV1, Options as CaptureOptions},
    ext_image_copy_capture_session_v1::{self, ExtImageCopyCaptureSessionV1},
};
use wayland_protocols_wlr::screencopy::v1::client::{
    zwlr_screencopy_frame_v1::{self, ZwlrScreencopyFrameV1},
    zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1,
};

/// Buffer constraints announced by either capture protocol. We
/// normalise them so the downstream shm-allocation + readback path
/// is one piece of code instead of two.
#[derive(Debug, Clone, Copy)]
struct BufferInfo {
    format: u32, // wl_shm format enum
    width: u32,
    height: u32,
    stride: u32,
}

impl BufferInfo {
    fn size(&self) -> usize {
        self.stride as usize * self.height as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameStatus {
    Pending,
    Ready,
    Failed,
}

// ── Monitor thumbnails (wlr-screencopy) ───────────────────────────

pub fn monitor_png(id: &str, max_w: u32, max_h: u32) -> Result<Vec<u8>, SourceError> {
    let conn = Connection::connect_to_env()
        .map_err(|e| SourceError::Backend(format!("wayland connect: {e}")))?;
    let (globals, mut queue) = registry_queue_init::<MonitorState>(&conn)
        .map_err(|e| SourceError::Backend(format!("registry: {e}")))?;
    let qh = queue.handle();

    let shm: WlShm = globals
        .bind(&qh, 1..=1, ())
        .map_err(|e| SourceError::Backend(format!("wl_shm: {e}")))?;
    let screencopy: ZwlrScreencopyManagerV1 = globals.bind(&qh, 1..=3, ()).map_err(|e| {
        SourceError::Backend(format!(
            "zwlr_screencopy_manager_v1 missing on this compositor: {e}"
        ))
    })?;

    let mut state = MonitorState {
        outputs: Vec::new(),
        buffer: None,
        status: FrameStatus::Pending,
        frame_flags: 0,
        target_transform: WlTransform::Normal,
    };
    for g in globals.contents().clone_list() {
        if g.interface == "wl_output" {
            let out = globals
                .registry()
                .bind::<WlOutput, _, _>(g.name, g.version.min(4), &qh, ());
            state.outputs.push(OutputEntry {
                proxy: out,
                name: None,
                transform: WlTransform::Normal,
            });
        }
    }
    // First roundtrip: wait for every wl_output to emit `name` /
    // `geometry` / `done`. We need both `name` (to match the picker
    // id) and `transform` (so the captured framebuffer is rotated /
    // flipped to match what the user sees) before screen-capturing.
    queue
        .roundtrip(&mut state)
        .map_err(|e| SourceError::Backend(format!("output roundtrip: {e}")))?;

    let entry = state
        .outputs
        .iter()
        .find(|e| e.name.as_deref() == Some(id))
        .ok_or_else(|| SourceError::NotFound(id.to_owned()))?;
    let output = entry.proxy.clone();
    state.target_transform = entry.transform;

    let frame = screencopy.capture_output(0, &output, &qh, ());
    // Wait for the `buffer` event so we know which shm format to
    // allocate. Some compositors also emit `linux_dmabuf` and
    // `buffer_done`; we only need the wl_shm side.
    while state.buffer.is_none() {
        queue
            .blocking_dispatch(&mut state)
            .map_err(|e| SourceError::Backend(format!("buffer wait: {e}")))?;
    }
    let info = state.buffer.unwrap();

    let (fd, mmap) = alloc_shm(info.size())?;
    let pool = shm.create_pool(fd.as_fd(), info.size() as i32, &qh, ());
    let buffer = pool.create_buffer(
        0,
        info.width as i32,
        info.height as i32,
        info.stride as i32,
        wl_shm_format(info.format)?,
        &qh,
        (),
    );
    frame.copy(&buffer);

    while state.status == FrameStatus::Pending {
        queue
            .blocking_dispatch(&mut state)
            .map_err(|e| SourceError::Backend(format!("frame wait: {e}")))?;
    }
    if state.status == FrameStatus::Failed {
        return Err(SourceError::Backend(
            "compositor returned `failed` for screencopy frame".into(),
        ));
    }

    let png = encode_png(
        info,
        mmap,
        max_w,
        max_h,
        Orientation {
            transform: state.target_transform,
            y_invert: state.frame_flags & 1 != 0,
        },
    )?;
    // Explicit cleanup: drop wayland objects + buffer first so the
    // server can release its end before we close the fd.
    buffer.destroy();
    pool.destroy();
    drop(fd);
    Ok(png)
}

// `wl_output::Transform` doesn't impl `Default` upstream — derive
// our own constructor so the auto-`#[derive(Default)]` on
// `MonitorState` would have to chain a manual init anyway. Cleaner
// to just write `Default` ourselves.
struct MonitorState {
    outputs: Vec<OutputEntry>,
    buffer: Option<BufferInfo>,
    status: FrameStatus,
    /// Bit 0 = `Y_INVERT`. Set from the screencopy `flags` event;
    /// applied as a vertical flip on the captured buffer before
    /// the output transform.
    frame_flags: u32,
    /// The wl_output transform of the chosen target. Captured pixels
    /// are in the framebuffer's pre-transform orientation; we apply
    /// this transform on the way out so the thumbnail matches what
    /// the user sees on the monitor (rotated displays come out
    /// upright, not sideways).
    target_transform: WlTransform,
}

impl Default for MonitorState {
    fn default() -> Self {
        Self {
            outputs: Vec::new(),
            buffer: None,
            status: FrameStatus::default(),
            frame_flags: 0,
            target_transform: WlTransform::Normal,
        }
    }
}

#[derive(Clone)]
struct OutputEntry {
    proxy: WlOutput,
    name: Option<String>,
    transform: WlTransform,
}

/// Re-export so the `encode_png` orientation arg has a stable type
/// even though wayland-client's `WEnum<Transform>` is `!Copy`. Keeps
/// the per-orientation match arms in one place too.
type WlTransform = wl_output::Transform;

impl Default for FrameStatus {
    fn default() -> Self {
        Self::Pending
    }
}

impl Dispatch<WlRegistry, GlobalListContents> for MonitorState {
    fn event(
        _: &mut Self,
        _: &WlRegistry,
        _: <WlRegistry as Proxy>::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlOutput, ()> for MonitorState {
    fn event(
        state: &mut Self,
        proxy: &WlOutput,
        event: <WlOutput as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wl_output::Event::Name { name } => {
                for entry in &mut state.outputs {
                    if &entry.proxy == proxy {
                        entry.name = Some(name);
                        break;
                    }
                }
            }
            wl_output::Event::Geometry { transform, .. } => {
                // `WEnum::Value(t)` has the variant; `Unknown(_)`
                // is something the protocol added we don't know
                // about — fall back to Normal which is a safe
                // no-op orientation.
                let t = match transform {
                    WEnum::Value(v) => v,
                    WEnum::Unknown(_) => WlTransform::Normal,
                };
                for entry in &mut state.outputs {
                    if &entry.proxy == proxy {
                        entry.transform = t;
                        break;
                    }
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<WlShm, ()> for MonitorState {
    fn event(
        _: &mut Self,
        _: &WlShm,
        _: <WlShm as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<WlShmPool, ()> for MonitorState {
    fn event(
        _: &mut Self,
        _: &WlShmPool,
        _: <WlShmPool as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<WlBuffer, ()> for MonitorState {
    fn event(
        _: &mut Self,
        _: &WlBuffer,
        _: <WlBuffer as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<ZwlrScreencopyManagerV1, ()> for MonitorState {
    fn event(
        _: &mut Self,
        _: &ZwlrScreencopyManagerV1,
        _: <ZwlrScreencopyManagerV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ZwlrScreencopyFrameV1, ()> for MonitorState {
    fn event(
        state: &mut Self,
        _: &ZwlrScreencopyFrameV1,
        event: <ZwlrScreencopyFrameV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        use zwlr_screencopy_frame_v1::Event;
        match event {
            Event::Buffer {
                format,
                width,
                height,
                stride,
            } => {
                let format = match format {
                    WEnum::Value(v) => v as u32,
                    WEnum::Unknown(v) => v,
                };
                state.buffer = Some(BufferInfo {
                    format,
                    width,
                    height,
                    stride,
                });
            }
            Event::Flags { flags } => {
                // Bitfield with `y_invert = 1`. We surface the raw
                // bits so the encoder can apply the flip uniformly
                // even if future protocol versions add more flags.
                state.frame_flags = match flags {
                    WEnum::Value(v) => v.bits(),
                    WEnum::Unknown(v) => v,
                };
            }
            Event::Ready { .. } => state.status = FrameStatus::Ready,
            Event::Failed => state.status = FrameStatus::Failed,
            _ => {}
        }
    }
}

// ── Window thumbnails (ext-image-copy-capture) ────────────────────

pub fn window_png(id: &str, max_w: u32, max_h: u32) -> Result<Vec<u8>, SourceError> {
    let conn = Connection::connect_to_env()
        .map_err(|e| SourceError::Backend(format!("wayland connect: {e}")))?;
    let (globals, mut queue) = registry_queue_init::<WindowState>(&conn)
        .map_err(|e| SourceError::Backend(format!("registry: {e}")))?;
    let qh = queue.handle();

    let shm: WlShm = globals
        .bind(&qh, 1..=1, ())
        .map_err(|e| SourceError::Backend(format!("wl_shm: {e}")))?;
    let toplevel_list: ExtForeignToplevelListV1 = globals
        .bind(&qh, 1..=1, ())
        .map_err(|e| SourceError::Backend(format!("ext_foreign_toplevel_list_v1 missing: {e}")))?;
    let toplevel_src_mgr: ExtForeignToplevelImageCaptureSourceManagerV1 =
        globals.bind(&qh, 1..=1, ()).map_err(|e| {
            SourceError::Backend(format!(
                "ext_foreign_toplevel_image_capture_source_manager_v1 missing — compositor doesn't expose window thumbnails: {e}"
            ))
        })?;
    let capture_mgr: ExtImageCopyCaptureManagerV1 = globals
        .bind(&qh, 1..=1, ())
        .map_err(|e| SourceError::Backend(format!("ext_image_copy_capture_manager_v1 missing — compositor doesn't expose window thumbnails: {e}")))?;
    let _ = toplevel_list;

    let mut state = WindowState {
        target_id: id.to_owned(),
        handle: None,
        identifiers: HashMap::new(),
        buffer: None,
        formats_done: false,
        status: FrameStatus::Pending,
    };

    // Roundtrip until we've seen `done` on the matching toplevel
    // handle (which guarantees `identifier` was delivered first).
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
    while state.handle.is_none() && std::time::Instant::now() < deadline {
        queue
            .blocking_dispatch(&mut state)
            .map_err(|e| SourceError::Backend(format!("toplevel wait: {e}")))?;
    }
    let handle = state
        .handle
        .clone()
        .ok_or_else(|| SourceError::NotFound(id.to_owned()))?;

    let source = toplevel_src_mgr.create_source(&handle, &qh, ());
    let session = capture_mgr.create_session(&source, CaptureOptions::empty(), &qh, ());
    while !state.formats_done {
        queue
            .blocking_dispatch(&mut state)
            .map_err(|e| SourceError::Backend(format!("session wait: {e}")))?;
    }
    let info = state.buffer.ok_or_else(|| {
        SourceError::Backend("session emitted `done` without buffer_size/shm_format".into())
    })?;

    let (fd, mmap) = alloc_shm(info.size())?;
    let pool = shm.create_pool(fd.as_fd(), info.size() as i32, &qh, ());
    let buffer = pool.create_buffer(
        0,
        info.width as i32,
        info.height as i32,
        info.stride as i32,
        wl_shm_format(info.format)?,
        &qh,
        (),
    );

    let frame = session.create_frame(&qh, ());
    frame.attach_buffer(&buffer);
    frame.capture();

    while state.status == FrameStatus::Pending {
        queue
            .blocking_dispatch(&mut state)
            .map_err(|e| SourceError::Backend(format!("frame wait: {e}")))?;
    }
    if state.status == FrameStatus::Failed {
        return Err(SourceError::Backend(
            "compositor returned `failed` for ext-image-copy frame".into(),
        ));
    }

    // Window thumbnails come from `ext-image-copy-capture-v1`, which
    // delivers pixels in the standard top-down framebuffer order
    // and doesn't expose `Y_INVERT`. Output transforms don't apply
    // either — the source is a toplevel surface, not an output.
    // So `Orientation::default()` (Normal, no flip) is correct.
    let png = encode_png(info, mmap, max_w, max_h, Orientation::default())?;
    frame.destroy();
    session.destroy();
    source.destroy();
    handle.destroy();
    buffer.destroy();
    pool.destroy();
    drop(fd);
    Ok(png)
}

struct WindowState {
    target_id: String,
    handle: Option<ExtForeignToplevelHandleV1>,
    identifiers: HashMap<u32, String>,
    buffer: Option<BufferInfo>,
    formats_done: bool,
    status: FrameStatus,
}

impl Dispatch<WlRegistry, GlobalListContents> for WindowState {
    fn event(
        _: &mut Self,
        _: &WlRegistry,
        _: <WlRegistry as Proxy>::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<WlShm, ()> for WindowState {
    fn event(
        _: &mut Self,
        _: &WlShm,
        _: <WlShm as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<WlShmPool, ()> for WindowState {
    fn event(
        _: &mut Self,
        _: &WlShmPool,
        _: <WlShmPool as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<WlBuffer, ()> for WindowState {
    fn event(
        _: &mut Self,
        _: &WlBuffer,
        _: <WlBuffer as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<ExtForeignToplevelImageCaptureSourceManagerV1, ()> for WindowState {
    fn event(
        _: &mut Self,
        _: &ExtForeignToplevelImageCaptureSourceManagerV1,
        _: <ExtForeignToplevelImageCaptureSourceManagerV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<ExtOutputImageCaptureSourceManagerV1, ()> for WindowState {
    fn event(
        _: &mut Self,
        _: &ExtOutputImageCaptureSourceManagerV1,
        _: <ExtOutputImageCaptureSourceManagerV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<ExtImageCaptureSourceV1, ()> for WindowState {
    fn event(
        _: &mut Self,
        _: &ExtImageCaptureSourceV1,
        _: <ExtImageCaptureSourceV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<ExtImageCopyCaptureManagerV1, ()> for WindowState {
    fn event(
        _: &mut Self,
        _: &ExtImageCopyCaptureManagerV1,
        _: <ExtImageCopyCaptureManagerV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ExtForeignToplevelListV1, ()> for WindowState {
    fn event(
        state: &mut Self,
        _: &ExtForeignToplevelListV1,
        event: <ExtForeignToplevelListV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let ext_foreign_toplevel_list_v1::Event::Toplevel { toplevel } = event {
            // ToplevelData lives entirely in `identifiers` keyed by
            // protocol_id; once `identifier` arrives we check it
            // against `target_id`.
            let pid = toplevel.id().protocol_id();
            state.identifiers.entry(pid).or_default();
            // Stash the proxy so we can match later. We can't keep
            // it in `handle` yet because we haven't seen its id.
            // We'll match in the handle dispatch.
            let _ = toplevel;
        }
    }
    wayland_client::event_created_child!(WindowState, ExtForeignToplevelListV1, [
        0 => (ExtForeignToplevelHandleV1, ()),
    ]);
}

impl Dispatch<ExtForeignToplevelHandleV1, ()> for WindowState {
    fn event(
        state: &mut Self,
        proxy: &ExtForeignToplevelHandleV1,
        event: <ExtForeignToplevelHandleV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let pid = proxy.id().protocol_id();
        if let ext_foreign_toplevel_handle_v1::Event::Identifier { identifier } = event {
            if identifier == state.target_id && state.handle.is_none() {
                state.handle = Some(proxy.clone());
            }
            state.identifiers.insert(pid, identifier);
        }
    }
}

impl Dispatch<ExtImageCopyCaptureSessionV1, ()> for WindowState {
    fn event(
        state: &mut Self,
        _: &ExtImageCopyCaptureSessionV1,
        event: <ExtImageCopyCaptureSessionV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        use ext_image_copy_capture_session_v1::Event;
        match event {
            Event::BufferSize { width, height } => {
                let entry = state.buffer.get_or_insert(BufferInfo {
                    format: 0,
                    width: 0,
                    height: 0,
                    stride: 0,
                });
                entry.width = width;
                entry.height = height;
                // We don't know stride yet; estimate as `width * 4`
                // (every wl_shm format we accept is 32-bit). The
                // compositor doesn't advertise stride for this
                // protocol — clients pick it.
                entry.stride = width.saturating_mul(4);
            }
            Event::ShmFormat { format } => {
                let format = match format {
                    WEnum::Value(v) => v as u32,
                    WEnum::Unknown(v) => v,
                };
                if let Some(b) = state.buffer.as_mut() {
                    b.format = format;
                } else {
                    state.buffer = Some(BufferInfo {
                        format,
                        width: 0,
                        height: 0,
                        stride: 0,
                    });
                }
            }
            Event::Done => state.formats_done = true,
            Event::Stopped => {
                state.formats_done = true;
                state.status = FrameStatus::Failed;
            }
            _ => {}
        }
    }
}

impl Dispatch<ExtImageCopyCaptureFrameV1, ()> for WindowState {
    fn event(
        state: &mut Self,
        _: &ExtImageCopyCaptureFrameV1,
        event: <ExtImageCopyCaptureFrameV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        use ext_image_copy_capture_frame_v1::Event;
        match event {
            Event::Ready => state.status = FrameStatus::Ready,
            Event::Failed { .. } => state.status = FrameStatus::Failed,
            _ => {}
        }
    }
}

// ── Shared support: shm allocation + format → RGBA + PNG ──────────

fn alloc_shm(size: usize) -> Result<(OwnedFd, MmapRegion), SourceError> {
    use std::os::fd::FromRawFd;
    let name = b"ferricast-thumb\0";
    // Linux 3.17+ has memfd_create; no fallback needed for any
    // distribution we'd realistically target.
    let raw = unsafe {
        libc::syscall(
            libc::SYS_memfd_create,
            name.as_ptr() as *const libc::c_char,
            libc::MFD_CLOEXEC,
        )
    };
    if raw < 0 {
        return Err(SourceError::Backend(format!(
            "memfd_create failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    let fd = unsafe { OwnedFd::from_raw_fd(raw as i32) };
    if unsafe { libc::ftruncate(fd.as_raw_fd(), size as libc::off_t) } != 0 {
        return Err(SourceError::Backend(format!(
            "ftruncate failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd.as_raw_fd(),
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(SourceError::Backend(format!(
            "mmap failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok((
        fd,
        MmapRegion {
            ptr: ptr as *mut u8,
            len: size,
        },
    ))
}

struct MmapRegion {
    ptr: *mut u8,
    len: usize,
}
unsafe impl Send for MmapRegion {}
unsafe impl Sync for MmapRegion {}

impl MmapRegion {
    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for MmapRegion {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut _, self.len);
        }
    }
}

fn wl_shm_format(raw: u32) -> Result<wl_shm::Format, SourceError> {
    // wl_shm::Format is a closed enum at compile time; if the
    // compositor offers a format we don't enumerate we still need
    // a valid variant for `create_buffer`. The values we accept
    // here (Xrgb8888 / Argb8888 / Xbgr8888 / Abgr8888) cover every
    // sane compositor; anything exotic falls through as a backend
    // error so we don't silently corrupt the PNG.
    match raw {
        0 => Ok(wl_shm::Format::Argb8888),
        1 => Ok(wl_shm::Format::Xrgb8888),
        0x34325258 => Ok(wl_shm::Format::Xbgr8888),
        0x34324241 => Ok(wl_shm::Format::Abgr8888),
        other => Err(SourceError::Backend(format!(
            "unsupported wl_shm format 0x{other:08x} for thumbnails"
        ))),
    }
}

/// Orientation hints for [`encode_png`]. `transform` is the wl_output
/// transform of the captured monitor (rotated displays come out
/// sideways from the framebuffer otherwise); `y_invert` is the
/// `Y_INVERT` bit from the screencopy `flags` event (some
/// compositors emit pixels bottom-up).
#[derive(Clone, Copy)]
struct Orientation {
    transform: wl_output::Transform,
    y_invert: bool,
}

impl Default for Orientation {
    fn default() -> Self {
        Orientation {
            transform: wl_output::Transform::Normal,
            y_invert: false,
        }
    }
}

fn encode_png(
    info: BufferInfo,
    mmap: MmapRegion,
    max_w: u32,
    max_h: u32,
    orientation: Orientation,
) -> Result<Vec<u8>, SourceError> {
    let src = mmap.as_slice();
    if src.len() < info.size() {
        return Err(SourceError::Backend(format!(
            "shm mmap shorter than declared ({} < {})",
            src.len(),
            info.size()
        )));
    }
    let w = info.width as usize;
    let h = info.height as usize;
    let stride = info.stride as usize;
    let mut rgba = vec![0u8; w * h * 4];
    // Swizzle each row from the source format to RGBA. The four
    // formats we accept are all 4 bytes/pixel, just with different
    // channel orderings — translate at the byte level.
    let (r, g, b, a, opaque) = channels(info.format);
    for y in 0..h {
        let row_src = &src[y * stride..y * stride + w * 4];
        let row_dst = &mut rgba[y * w * 4..(y + 1) * w * 4];
        for (px_src, px_dst) in row_src.chunks_exact(4).zip(row_dst.chunks_exact_mut(4)) {
            px_dst[0] = px_src[r];
            px_dst[1] = px_src[g];
            px_dst[2] = px_src[b];
            px_dst[3] = if opaque { 0xFF } else { px_src[a] };
        }
    }
    drop(mmap);

    let mut img = image::RgbaImage::from_raw(w as u32, h as u32, rgba)
        .ok_or_else(|| SourceError::Backend("rgba from_raw failed".into()))?;

    // Apply Y_INVERT first (it's defined on the buffer's framebuffer
    // axis), then the wl_output transform — which is the
    // composition the compositor would apply before scanning out to
    // the user's eye. Doing them in the same order the spec describes
    // means rotated + inverted captures (some VR / mobile-ish
    // displays) come out the way the user sees them.
    if orientation.y_invert {
        img = image::imageops::flip_vertical(&img);
    }
    img = apply_wl_transform(img, orientation.transform);

    let (final_w, final_h) = (img.width(), img.height());
    let (tw, th) = crate::fit_box(final_w, final_h, max_w, max_h);
    let thumb = image::imageops::thumbnail(&img, tw, th);

    let mut out = Vec::with_capacity(8 * 1024);
    thumb
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .map_err(|e| SourceError::Backend(format!("png encode: {e}")))?;
    debug!(
        w = tw,
        h = th,
        bytes = out.len(),
        transform = ?orientation.transform,
        y_invert = orientation.y_invert,
        "wayland thumbnail encoded"
    );
    Ok(out)
}

/// Apply the wl_output transform to a captured framebuffer so the
/// thumbnail matches what the user sees on the monitor. The
/// `Flipped*` variants are "flip horizontal then rotate by N"
/// per the wayland spec — same operation order as wlroots' renderer.
fn apply_wl_transform(img: image::RgbaImage, transform: wl_output::Transform) -> image::RgbaImage {
    use image::imageops as ops;
    use wl_output::Transform::*;
    match transform {
        Normal => img,
        _90 => ops::rotate90(&img),
        _180 => ops::rotate180(&img),
        _270 => ops::rotate270(&img),
        Flipped => ops::flip_horizontal(&img),
        Flipped90 => ops::rotate90(&ops::flip_horizontal(&img)),
        Flipped180 => ops::rotate180(&ops::flip_horizontal(&img)),
        Flipped270 => ops::rotate270(&ops::flip_horizontal(&img)),
        // Future-proof: an unknown variant lands here. Treat as
        // Normal so the thumbnail is still readable, just at the
        // wrong orientation — better than dropping the frame.
        _ => img,
    }
}

/// Byte offsets within a 4-byte wl_shm pixel for `(R, G, B, A)`,
/// plus whether the format ignores alpha (force `0xFF` on output).
///
/// `Argb8888` little-endian wire = `[B, G, R, A]` in memory.
/// `Xrgb8888` little-endian wire = `[B, G, R, X]` in memory.
/// `Abgr8888` little-endian wire = `[R, G, B, A]` in memory.
/// `Xbgr8888` little-endian wire = `[R, G, B, X]` in memory.
fn channels(format: u32) -> (usize, usize, usize, usize, bool) {
    match format {
        0 => (2, 1, 0, 3, false),          // Argb8888
        1 => (2, 1, 0, 3, true),           // Xrgb8888
        0x34324241 => (0, 1, 2, 3, false), // Abgr8888
        0x34325258 => (0, 1, 2, 3, true),  // Xbgr8888
        _ => (0, 1, 2, 3, true),
    }
}
