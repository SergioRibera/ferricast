//! One-shot Wayland thumbnail capture.
//!
//! Two protocols, two flows, one file because they share most of
//! the support code (`wl_shm` pool allocation, format conversion,
//! PNG encode, output lookup).
//!
//! ### Monitor thumbnails — `zwlr_screencopy_unstable_v1`
//!
//! Bind the screencopy manager, find the `wl_output` that matches
//! the picker-issued id, call `capture_output`, wait for the
//! `buffer` event to know the pixel format, allocate a `wl_shm`
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
use std::time::{Duration, Instant};

use ferricast_core::SourceError;
use tracing::debug;

use wayland_client::{
    Connection, Dispatch, DispatchError, EventQueue, Proxy, QueueHandle, WEnum,
    backend::WaylandError,
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum FrameStatus {
    #[default]
    Pending,
    Ready,
    Failed,
}

/// Upper bound for each compositor round-trip of a thumbnail capture.
/// A healthy compositor answers within a frame; this only bounds the
/// thread when the source vanishes mid-capture.
const THUMB_EVENT_TIMEOUT: Duration = Duration::from_secs(2);

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
    queue
        .dispatch_until(&mut state, THUMB_EVENT_TIMEOUT, |s| {
            s.buffer.is_some() || s.status == FrameStatus::Failed
        })
        .map_err(|e| SourceError::Backend(format!("buffer wait: {e}")))?;
    let Some(info) = state.buffer else {
        frame.destroy();
        return Err(SourceError::Backend(
            "compositor didn't announce a wl_shm buffer for the screencopy frame".into(),
        ));
    };

    let (fd, mmap) = alloc_shm(info.size())?;
    let pool = shm.create_pool(fd.as_fd(), pool_size(info.size())?, &qh, ());
    let buffer = pool.create_buffer(
        0,
        info.width.cast_signed(),
        info.height.cast_signed(),
        info.stride.cast_signed(),
        wl_shm_format(info.format)?,
        &qh,
        (),
    );
    frame.copy(&buffer);

    queue
        .dispatch_until(&mut state, THUMB_EVENT_TIMEOUT, |s| {
            s.status != FrameStatus::Pending
        })
        .map_err(|e| SourceError::Backend(format!("frame wait: {e}")))?;
    match state.status {
        FrameStatus::Ready => {}
        FrameStatus::Failed => {
            return Err(SourceError::Backend(
                "compositor returned `failed` for screencopy frame".into(),
            ));
        }
        FrameStatus::Pending => {
            return Err(SourceError::Backend(
                "screencopy frame not ready in time".into(),
            ));
        }
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
    /// The `wl_output` transform of the chosen target. Captured pixels
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
        (): &(),
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
        (): &(),
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
        (): &(),
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
        (): &(),
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
        (): &(),
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
        (): &(),
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

#[expect(
    clippy::too_many_lines,
    reason = "one linear setup or event sequence over shared state; splitting it would scatter that state"
)]
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
        shm_format: None,
        formats_done: false,
        status: FrameStatus::Pending,
    };

    // Roundtrip until we've seen `done` on the matching toplevel
    // handle (which guarantees `identifier` was delivered first).
    queue
        .dispatch_until(&mut state, Duration::from_millis(500), |s| {
            s.handle.is_some()
        })
        .map_err(|e| SourceError::Backend(format!("toplevel wait: {e}")))?;
    let handle = state
        .handle
        .clone()
        .ok_or_else(|| SourceError::NotFound(id.to_owned()))?;

    let source = toplevel_src_mgr.create_source(&handle, &qh, ());
    let session = capture_mgr.create_session(&source, CaptureOptions::empty(), &qh, ());
    let formats_done = queue
        .dispatch_until(&mut state, THUMB_EVENT_TIMEOUT, |s| s.formats_done)
        .map_err(|e| SourceError::Backend(format!("session wait: {e}")))?;
    if !formats_done {
        return Err(SourceError::Backend(
            "ext-image-copy session didn't announce buffer constraints in time".into(),
        ));
    }
    if state.status == FrameStatus::Failed {
        // `stopped` before `done`: the window went away.
        return Err(SourceError::NotFound(id.to_owned()));
    }
    let size = state
        .buffer
        .ok_or_else(|| SourceError::Backend("session emitted `done` without buffer_size".into()))?;
    let format = state.shm_format.ok_or_else(|| {
        SourceError::Backend(
            "session advertised no supported wl_shm format (need 8-bit ARGB/XRGB/ABGR/XBGR)".into(),
        )
    })?;
    let info = BufferInfo { format, ..size };

    let (fd, mmap) = alloc_shm(info.size())?;
    let pool = shm.create_pool(fd.as_fd(), pool_size(info.size())?, &qh, ());
    let buffer = pool.create_buffer(
        0,
        info.width.cast_signed(),
        info.height.cast_signed(),
        info.stride.cast_signed(),
        wl_shm_format(info.format)?,
        &qh,
        (),
    );

    let frame = session.create_frame(&qh, ());
    frame.attach_buffer(&buffer);
    frame.capture();

    queue
        .dispatch_until(&mut state, THUMB_EVENT_TIMEOUT, |s| {
            s.status != FrameStatus::Pending
        })
        .map_err(|e| SourceError::Backend(format!("frame wait: {e}")))?;
    match state.status {
        FrameStatus::Ready => {}
        FrameStatus::Failed => {
            return Err(SourceError::Backend(
                "compositor returned `failed` for ext-image-copy frame".into(),
            ));
        }
        FrameStatus::Pending => {
            return Err(SourceError::Backend(
                "ext-image-copy frame not ready in time (window closed?)".into(),
            ));
        }
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
    /// Size announced by `buffer_size`; `format` is filled from
    /// `shm_format` once the session's `done` arrives.
    buffer: Option<BufferInfo>,
    /// First advertised `wl_shm` format we support.
    shm_format: Option<u32>,
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
        (): &(),
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
        (): &(),
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
        (): &(),
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
        (): &(),
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
        (): &(),
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
        (): &(),
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
        (): &(),
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
        (): &(),
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
        (): &(),
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
        (): &(),
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
                // One event per advertised format, in the
                // compositor's preference order: keep the first one we
                // can read back instead of letting a later (e.g.
                // 10-bit) entry overwrite it.
                if state.shm_format.is_none() && wl_shm_format(format).is_ok() {
                    state.shm_format = Some(format);
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
        (): &(),
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

// ── Shared support: bounded event-queue dispatch ──────────────────

/// Bounded waits on a Wayland [`EventQueue`].
///
/// `EventQueue::blocking_dispatch` blocks until the compositor sends
/// *something*, so a `while !done && now < deadline { blocking_dispatch }`
/// loop never notices its deadline when the compositor goes quiet (a
/// window closed mid-capture, a hung compositor). This polls the
/// connection fd with the remaining budget instead, so the deadline
/// is honoured. Shared with `wayland_direct`.
pub(crate) trait EventQueueExt<S> {
    /// Dispatch events until `done(state)` holds or `timeout` elapses.
    ///
    /// Returns `Ok(true)` when `done` became true, `Ok(false)` on
    /// timeout.
    ///
    /// # Errors
    ///
    /// Protocol / socket errors from flushing, reading or dispatching.
    fn dispatch_until(
        &mut self,
        state: &mut S,
        timeout: Duration,
        done: impl FnMut(&S) -> bool,
    ) -> Result<bool, DispatchError>;
}

impl<S> EventQueueExt<S> for EventQueue<S> {
    fn dispatch_until(
        &mut self,
        state: &mut S,
        timeout: Duration,
        mut done: impl FnMut(&S) -> bool,
    ) -> Result<bool, DispatchError> {
        let start = Instant::now();
        loop {
            self.dispatch_pending(state)?;
            if done(state) {
                return Ok(true);
            }
            let remaining = timeout.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                return Ok(false);
            }
            self.flush()?;
            // `None` means events are already queued: loop back to
            // `dispatch_pending`.
            let Some(guard) = self.prepare_read() else {
                continue;
            };
            let mut pfd = libc::pollfd {
                fd: guard.connection_fd().as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // Round up so a sub-millisecond remainder still sleeps
            // instead of busy-looping on a zero timeout.
            let timeout_ms =
                i32::try_from(remaining.as_micros().div_ceil(1000)).unwrap_or(i32::MAX);
            // SAFETY: `pfd` is a valid, exclusively borrowed pollfd for
            // the duration of the call and `nfds` is 1. The fd stays
            // open while `guard` (which borrows the connection) lives.
            let rc = unsafe { libc::poll(&raw mut pfd, 1, timeout_ms) };
            if rc < 0 {
                let err = std::io::Error::last_os_error();
                if err.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(WaylandError::Io(err).into());
            }
            if rc == 0 {
                // Timed out: dropping the guard cancels the read; the
                // next iteration sees `remaining == 0` and returns.
                continue;
            }
            match guard.read() {
                Ok(_) => {}
                Err(WaylandError::Io(e)) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => return Err(e.into()),
            }
        }
    }
}

// ── Shared support: shm allocation + format → RGBA + PNG ──────────

/// `size` as the `i32` a `wl_shm_pool` takes.
fn pool_size(size: usize) -> Result<i32, SourceError> {
    i32::try_from(size)
        .map_err(|_| SourceError::Backend(format!("shm buffer of {size} bytes exceeds i32")))
}

fn alloc_shm(size: usize) -> Result<(OwnedFd, MmapRegion), SourceError> {
    use std::os::fd::FromRawFd;
    let name = b"ferricast-thumb\0";
    // Linux 3.17+ has memfd_create; no fallback needed for any
    // distribution we'd realistically target.
    let raw = unsafe {
        libc::syscall(
            libc::SYS_memfd_create,
            name.as_ptr().cast::<libc::c_char>(),
            libc::MFD_CLOEXEC,
        )
    };
    if raw < 0 {
        return Err(SourceError::Backend(format!(
            "memfd_create failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    let raw = i32::try_from(raw)
        .map_err(|_| SourceError::Backend(format!("memfd_create returned fd {raw}")))?;
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    let length = libc::off_t::try_from(size)
        .map_err(|_| SourceError::Backend(format!("memfd of {size} bytes exceeds off_t")))?;
    if unsafe { libc::ftruncate(fd.as_raw_fd(), length) } != 0 {
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
            ptr: ptr.cast::<u8>(),
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
            libc::munmap(self.ptr.cast(), self.len);
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
        0x3432_5258 => Ok(wl_shm::Format::Xbgr8888),
        0x3432_4241 => Ok(wl_shm::Format::Abgr8888),
        other => Err(SourceError::Backend(format!(
            "unsupported wl_shm format 0x{other:08x} for thumbnails"
        ))),
    }
}

/// How a captured output framebuffer must be re-oriented to match
/// what the user sees. `transform` is the `wl_output` transform of the
/// captured monitor (rotated displays come out sideways from the
/// framebuffer otherwise); `y_invert` is the `Y_INVERT` bit from the
/// screencopy `flags` event (some compositors emit pixels bottom-up).
///
/// Shared with `wayland_direct`, whose CPU path applies the same
/// correction to every streamed frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Orientation {
    pub(crate) transform: wl_output::Transform,
    pub(crate) y_invert: bool,
}

impl Default for Orientation {
    fn default() -> Self {
        Orientation {
            transform: wl_output::Transform::Normal,
            y_invert: false,
        }
    }
}

impl Orientation {
    /// `true` when [`Self::apply`] would return the image unchanged.
    pub(crate) fn is_identity(self) -> bool {
        !self.y_invert && self.transform == wl_output::Transform::Normal
    }

    /// `true` when [`Self::apply`] swaps width and height.
    pub(crate) fn swaps_axes(self) -> bool {
        use wl_output::Transform;
        matches!(
            self.transform,
            Transform::_90 | Transform::_270 | Transform::Flipped90 | Transform::Flipped270
        )
    }

    /// Apply `Y_INVERT` first (it's defined on the buffer's
    /// framebuffer axis), then the `wl_output` transform — the
    /// composition the compositor applies before scanning out to the
    /// user's eye, so rotated + inverted captures come out the way
    /// the user sees them.
    ///
    /// Channel order is irrelevant here (pure pixel moves), so the
    /// `RgbaImage` may just as well hold BGRA bytes.
    pub(crate) fn apply(self, img: image::RgbaImage) -> image::RgbaImage {
        let img = if self.y_invert {
            image::imageops::flip_vertical(&img)
        } else {
            img
        };
        apply_wl_transform(img, self.transform)
    }
}

#[expect(
    clippy::many_single_char_names,
    reason = "w, h and the r, g, b, a channel offsets read as the pixel math they are"
)]
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

    let img = image::RgbaImage::from_raw(info.width, info.height, rgba)
        .ok_or_else(|| SourceError::Backend("rgba from_raw failed".into()))?;
    let img = orientation.apply(img);

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

/// Apply the `wl_output` transform to a captured framebuffer so the
/// thumbnail matches what the user sees on the monitor. The
/// `Flipped*` variants are "flip horizontal then rotate by N"
/// per the wayland spec — same operation order as wlroots' renderer.
#[expect(
    clippy::match_same_arms,
    reason = "unknown future variants deliberately render like `Normal`"
)]
fn apply_wl_transform(img: image::RgbaImage, transform: wl_output::Transform) -> image::RgbaImage {
    use image::imageops as ops;
    use wl_output::Transform;
    match transform {
        Transform::Normal => img,
        Transform::_90 => ops::rotate90(&img),
        Transform::_180 => ops::rotate180(&img),
        Transform::_270 => ops::rotate270(&img),
        Transform::Flipped => ops::flip_horizontal(&img),
        Transform::Flipped90 => ops::rotate90(&ops::flip_horizontal(&img)),
        Transform::Flipped180 => ops::rotate180(&ops::flip_horizontal(&img)),
        Transform::Flipped270 => ops::rotate270(&ops::flip_horizontal(&img)),
        // Future-proof: an unknown variant lands here. Treat as
        // Normal so the thumbnail is still readable, just at the
        // wrong orientation — better than dropping the frame.
        _ => img,
    }
}

/// Byte offsets within a 4-byte `wl_shm` pixel for `(R, G, B, A)`,
/// plus whether the format ignores alpha (force `0xFF` on output).
///
/// `Argb8888` little-endian wire = `[B, G, R, A]` in memory.
/// `Xrgb8888` little-endian wire = `[B, G, R, X]` in memory.
/// `Abgr8888` little-endian wire = `[R, G, B, A]` in memory.
/// `Xbgr8888` little-endian wire = `[R, G, B, X]` in memory.
fn channels(format: u32) -> (usize, usize, usize, usize, bool) {
    match format {
        0 => (2, 1, 0, 3, false),           // Argb8888
        1 => (2, 1, 0, 3, true),            // Xrgb8888
        0x3432_4241 => (0, 1, 2, 3, false), // Abgr8888
        // Xbgr8888 (0x3432_5258), and the opaque RGBA reading for any other code.
        _ => (0, 1, 2, 3, true),
    }
}
