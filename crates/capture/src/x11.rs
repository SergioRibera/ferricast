use bytes::Bytes;
use ferricast_core::{CaptureSource, FerricastError, ScreenCapture, WindowIdentifier};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicPtr, Ordering},
    },
    time::Instant,
};
use tracing::{debug, info, warn};
use xcb::{
    Xid, XidNew,
    shm::Seg,
    x::{self, Drawable, Format},
};

/// Rect resolved at `start()` and stable for the lifetime of the
/// capture. We deliberately freeze it at start because the SHM
/// segment is sized to match — re-resolving mid-stream (e.g. window
/// moves to a different monitor) is a problem for a later commit.
#[derive(Debug, Clone, Copy)]
struct ResolvedSource {
    drawable: Drawable,
    origin_x: i16,
    origin_y: i16,
    width: u16,
    height: u16,
}

pub struct X11Capture {
    seg_id: i32,
    segment: Option<Seg>,
    conn: Option<Arc<xcb::Connection>>,
    pixmap: Option<Format>,
    is_running: AtomicBool,
    size: (usize, usize),
    /// Source `next_frame` reads from. `None` only between `new()`
    /// and `start()`.
    source: Option<ResolvedSource>,
    /// fps the caller asked for in `start()`. X11 has no real
    /// negotiation — we just poll on demand — so we surface the
    /// configured value so downstream paces correctly.
    fps: u32,
    buffer_ptr: AtomicPtr<u8>,
    time: Instant,
}

impl X11Capture {
    pub fn new() -> Self {
        Self {
            seg_id: 0,
            segment: None,
            conn: None,
            pixmap: None,
            is_running: AtomicBool::new(false),
            size: (0, 0),
            source: None,
            fps: 0,
            time: Instant::now(),
            buffer_ptr: AtomicPtr::new(core::ptr::null_mut()),
        }
    }
}

/// Resolve a [`CaptureSource`] to the X11 drawable + rect to read
/// from. Errors are returned as `FerricastError::Capture` so the
/// daemon can surface a clean D-Bus error instead of a panic.
fn resolve_source(
    conn: &xcb::Connection,
    screen_root: x::Window,
    source: CaptureSource,
) -> ferricast_core::Result<ResolvedSource> {
    match source {
        CaptureSource::FullScreen { monitor: None } => {
            // Whole root drawable. Origin (0,0); size from the
            // screen's `width_in_pixels` / `height_in_pixels`.
            // We re-resolve via GetGeometry so we don't have to
            // thread the Screen struct through every code path.
            let g = conn
                .wait_for_reply(conn.send_request(&x::GetGeometry {
                    drawable: Drawable::Window(screen_root),
                }))
                .map_err(|e| FerricastError::Capture(format!("GetGeometry(root): {e}")))?;
            Ok(ResolvedSource {
                drawable: Drawable::Window(screen_root),
                origin_x: 0,
                origin_y: 0,
                width: g.width(),
                height: g.height(),
            })
        }
        CaptureSource::FullScreen {
            monitor: Some(name),
        } => {
            // Find the XRandR monitor whose output-name atom matches.
            let reply = conn
                .wait_for_reply(conn.send_request(&xcb::randr::GetMonitors {
                    window: screen_root,
                    get_active: true,
                }))
                .map_err(|e| FerricastError::Capture(format!("GetMonitors: {e}")))?;

            for m in reply.monitors() {
                let name_reply = conn
                    .wait_for_reply(conn.send_request(&x::GetAtomName { atom: m.name() }))
                    .map_err(|e| FerricastError::Capture(format!("GetAtomName: {e}")))?;
                let mon_name =
                    String::from_utf8_lossy(name_reply.name().to_utf8().as_bytes()).into_owned();
                if mon_name == name {
                    return Ok(ResolvedSource {
                        drawable: Drawable::Window(screen_root),
                        origin_x: m.x(),
                        origin_y: m.y(),
                        width: m.width(),
                        height: m.height(),
                    });
                }
            }
            Err(FerricastError::Capture(format!(
                "no XRandR monitor named {name:?}"
            )))
        }
        CaptureSource::Window {
            identifier: Some(WindowIdentifier::Id(xid)),
        } => window_drawable(conn, xid as u32),
        CaptureSource::Window {
            identifier: Some(WindowIdentifier::Title(title)),
        } => {
            // Look up by title against _NET_CLIENT_LIST + _NET_WM_NAME.
            // Multiple windows can share a title; we pick the first
            // match (deterministic by EWMH stacking order). Same
            // policy as `SourceDto::window_by_title`.
            let xid = lookup_window_by_title(conn, screen_root, &title)?;
            window_drawable(conn, xid)
        }
        CaptureSource::Window { identifier: None } => Err(FerricastError::Capture(
            "Window capture requires a Window identifier (id or title); none provided".into(),
        )),
    }
}

fn window_drawable(conn: &xcb::Connection, xid: u32) -> ferricast_core::Result<ResolvedSource> {
    let window: x::Window = <x::Window as XidNew>::new(xid);
    let g = conn
        .wait_for_reply(conn.send_request(&x::GetGeometry {
            drawable: Drawable::Window(window),
        }))
        .map_err(|_| FerricastError::Capture(format!("window {xid} not found")))?;
    Ok(ResolvedSource {
        drawable: Drawable::Window(window),
        origin_x: 0,
        origin_y: 0,
        width: g.width(),
        height: g.height(),
    })
}

fn lookup_window_by_title(
    conn: &xcb::Connection,
    root: x::Window,
    title: &str,
) -> ferricast_core::Result<u32> {
    // Two interned atoms: the property list + the UTF-8 title.
    let client_list_atom = intern(conn, b"_NET_CLIENT_LIST")?;
    let name_atom = intern(conn, b"_NET_WM_NAME")?;
    let utf8_atom = intern(conn, b"UTF8_STRING")?;

    let list_reply = conn
        .wait_for_reply(conn.send_request(&x::GetProperty {
            delete: false,
            window: root,
            property: client_list_atom,
            r#type: x::ATOM_WINDOW,
            long_offset: 0,
            long_length: 4096,
        }))
        .map_err(|e| FerricastError::Capture(format!("GetProperty(_NET_CLIENT_LIST): {e}")))?;
    let windows: &[x::Window] = list_reply.value();

    for &w in windows {
        let Ok(name_reply) = conn.wait_for_reply(conn.send_request(&x::GetProperty {
            delete: false,
            window: w,
            property: name_atom,
            r#type: utf8_atom,
            long_offset: 0,
            long_length: 1024,
        })) else {
            continue;
        };
        let got = String::from_utf8_lossy(name_reply.value::<u8>());
        if got == title {
            return Ok(w.resource_id());
        }
    }
    Err(FerricastError::Capture(format!(
        "no X11 window with _NET_WM_NAME == {title:?}"
    )))
}

fn intern(conn: &xcb::Connection, name: &[u8]) -> ferricast_core::Result<x::Atom> {
    conn.wait_for_reply(conn.send_request(&x::InternAtom {
        only_if_exists: false,
        name,
    }))
    .map(|r| r.atom())
    .map_err(|e| FerricastError::Capture(format!("InternAtom({name:?}): {e}")))
}

impl ScreenCapture for X11Capture {
    async fn start(
        &mut self,
        source: CaptureSource,
        config: ferricast_core::CaptureConfig,
    ) -> ferricast_core::Result<()> {
        info!(?source, "Connecting to Xserver");
        let (conn, screen_num) = xcb::Connection::connect(None)
            .map_err(|_| FerricastError::Capture("Cannot connect to server".to_string()))?;

        let screen = conn.get_setup().roots().nth(screen_num as usize).ok_or_else(|| {
            FerricastError::Capture("X server returned no screen on this connection".into())
        })?;

        let pixmap = conn
            .get_setup()
            .pixmap_formats()
            .iter()
            .find(|f| f.depth() == f.bits_per_pixel())
            .ok_or_else(|| {
                FerricastError::Capture("no matching pixmap format on this visual".into())
            })?
            .to_owned();

        let resolved = resolve_source(&conn, screen.root(), source)?;

        // Honour caller-provided width/height as the *output* size
        // (== the SHM segment size). The capture origin/extent always
        // come from `resolved`; if a caller asks for a different
        // size we'd need to either crop or scale, which is out of
        // scope here. For now `config.width`/`height` are only used
        // when no source-driven size is available (kept for
        // back-compat with callers that pre-date the source plumbing).
        let w = config.width.unwrap_or(resolved.width as u32) as usize;
        let h = config.height.unwrap_or(resolved.height as u32) as usize;
        if (w, h) != (resolved.width as usize, resolved.height as usize) {
            debug!(
                requested_w = w,
                requested_h = h,
                source_w = resolved.width,
                source_h = resolved.height,
                "capture size differs from source rect; using source size for SHM and ignoring config override"
            );
        }
        let w = resolved.width as usize;
        let h = resolved.height as usize;

        let segment = conn.generate_id();

        info!(w, h, "Creating shared memory");
        let seg_id =
            unsafe { libc::shmget(libc::IPC_PRIVATE, w * h * 4, libc::IPC_CREAT | 0o600) };

        if seg_id == -1 {
            return Err(FerricastError::Capture(
                "Cannot create shared memory".to_string(),
            ));
        }

        let buffer = unsafe { libc::shmat(seg_id, core::ptr::null(), 0) } as *mut u8;

        if buffer as i32 == -1 {
            return Err(FerricastError::Capture(
                "Cannot map shared memory".to_string(),
            ));
        }

        conn.send_request(&xcb::shm::Attach {
            shmseg: segment,
            shmid: seg_id as u32,
            read_only: false,
        });

        conn.flush()
            .map_err(|_| FerricastError::Capture("Cannot flush x11 server".to_string()))?;

        info!(
            origin_x = resolved.origin_x,
            origin_y = resolved.origin_y,
            w,
            h,
            "Connected"
        );

        self.buffer_ptr = AtomicPtr::new(buffer);
        self.seg_id = seg_id;
        self.segment = Some(segment);
        self.conn = Some(Arc::new(conn));
        self.is_running = AtomicBool::new(true);
        self.size = (w, h);
        self.source = Some(resolved);
        self.fps = config.fps;
        self.time = Instant::now();
        self.pixmap = Some(pixmap);
        Ok(())
    }
    async fn stop(&mut self) -> ferricast_core::Result<()> {
        info!("Closing connection");
        if !self.is_running.load(Ordering::Acquire) {
            return Err(FerricastError::Capture(
                "Trying to close recorder without starting it".to_string(),
            ));
        }

        let conn = self.conn.as_ref().unwrap();

        unsafe {
            if libc::shmctl(self.seg_id, libc::IPC_RMID, core::ptr::null_mut()) == -1 {
                return Err(FerricastError::Capture("Cannot clean segment".to_string()));
            }
        }

        conn.send_request(&xcb::shm::Detach {
            shmseg: self.segment.unwrap(),
        });

        conn.flush()
            .map_err(|_| FerricastError::Capture("Cannot flush x11 server".to_string()))?;

        self.is_running.store(false, Ordering::SeqCst);

        Ok(())
    }
    fn is_running(&self) -> bool {
        self.is_running.load(Ordering::SeqCst)
    }
    async fn next_frame(&mut self) -> ferricast_core::Result<ferricast_core::CapturedFrame> {
        if !self.is_running.load(Ordering::Acquire) {
            return Err(FerricastError::Capture(
                "Trying to close recorder without starting it".to_string(),
            ));
        }

        let buffer = unsafe {
            std::slice::from_raw_parts(
                self.buffer_ptr.load(Ordering::Relaxed),
                self.size.0 * self.size.1 * 4,
            )
        };

        let conn = self.conn.as_ref().unwrap();
        let format = self.pixmap.as_ref().unwrap();
        let src = self
            .source
            .as_ref()
            .expect("source must be set after start()");

        let cookie = conn.send_request(&xcb::shm::GetImage {
            drawable: src.drawable,
            x: src.origin_x,
            y: src.origin_y,
            width: src.width,
            height: src.height,
            plane_mask: !0,
            format: xcb::x::ImageFormat::ZPixmap as u8,
            shmseg: self.segment.unwrap(),
            offset: 0,
        });

        // BadWindow / BadMatch on a window source typically means the
        // window was closed mid-stream. Surface that as a capture
        // error so the manager can stop the stream cleanly instead
        // of silently shipping stale pixels.
        if let Err(e) = conn.wait_for_reply(cookie) {
            warn!(%e, "shm::GetImage failed; source likely disappeared");
            return Err(FerricastError::Capture(format!("GetImage: {e}")));
        }

        let bytes_per_pixel = (format.bits_per_pixel() / 8) as u32;

        Ok(ferricast_core::CapturedFrame::Cpu(
            ferricast_core::RawFrame {
                width: self.size.0 as u32,
                height: self.size.1 as u32,
                stride: (self.size.0 as u32) * bytes_per_pixel,
                format: ferricast_core::PixelFormat::Bgra,
                data: Bytes::from(buffer.to_vec()),
                timestamp_us: self.time.elapsed().as_micros() as u64,
            },
        ))
    }
    fn get_pixel_format(&self) -> ferricast_core::PixelFormat {
        ferricast_core::PixelFormat::Bgra
    }
    fn get_screen_size(&self) -> (usize, usize) {
        self.size
    }
    fn get_framerate(&self) -> u32 {
        self.fps
    }
}
