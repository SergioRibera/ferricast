//! Source enumeration on X11.
//!
//! Monitors come from the **XRandR 1.5** `GetMonitors` request,
//! which returns the active layout (already merged across CRTCs).
//! Windows come from the **EWMH** root-window properties
//! (`_NET_CLIENT_LIST` for the list, `_NET_WM_NAME` etc. for the
//! metadata). Both are universally supported on any X11 session
//! shipped after ~2015.
//!
//! Change detection rides on two separate event streams kept alive
//! on a dedicated thread:
//!
//! - `xcb::randr::ScreenChangeNotifyEvent` for the monitor layout.
//! - `PropertyNotifyEvent` on the root window for
//!   `_NET_CLIENT_LIST` / `_NET_ACTIVE_WINDOW` (window churn).
//!
//! `list_*` calls are run via `tokio::task::spawn_blocking` because
//! `xcb` is synchronous. Each call opens its own short-lived
//! connection — cheaper than a `Mutex<Connection>` that the event
//! thread would contend for on every roundtrip.

use async_trait::async_trait;
use ferricast_core::{
    EnumerationCapability, Geometry, MonitorInfo, SourceChange, SourceEnumerator, SourceError,
    WindowInfo,
};
use tokio::sync::broadcast;
use tracing::{debug, warn};
use xcb::{x, randr, Xid};

/// Atom names we need to read window metadata. Cached per-connection
/// because intern is cheap (one round-trip) but a list-of-200-windows
/// call would do 200 * 4 round-trips otherwise.
struct Atoms {
    net_client_list: x::Atom,
    net_wm_name: x::Atom,
    net_wm_pid: x::Atom,
    net_wm_window_type: x::Atom,
    net_wm_window_type_normal: x::Atom,
    net_active_window: x::Atom,
    utf8_string: x::Atom,
}

impl Atoms {
    fn intern(conn: &xcb::Connection) -> xcb::Result<Self> {
        let names: [&[u8]; 7] = [
            b"_NET_CLIENT_LIST",
            b"_NET_WM_NAME",
            b"_NET_WM_PID",
            b"_NET_WM_WINDOW_TYPE",
            b"_NET_WM_WINDOW_TYPE_NORMAL",
            b"_NET_ACTIVE_WINDOW",
            b"UTF8_STRING",
        ];
        let cookies: Vec<_> = names
            .iter()
            .map(|n| {
                conn.send_request(&x::InternAtom {
                    only_if_exists: false,
                    name: n,
                })
            })
            .collect();
        let mut atoms = Vec::with_capacity(cookies.len());
        for c in cookies {
            atoms.push(conn.wait_for_reply(c)?.atom());
        }
        Ok(Self {
            net_client_list: atoms[0],
            net_wm_name: atoms[1],
            net_wm_pid: atoms[2],
            net_wm_window_type: atoms[3],
            net_wm_window_type_normal: atoms[4],
            net_active_window: atoms[5],
            utf8_string: atoms[6],
        })
    }
}

pub struct X11SourceEnumerator {
    change_tx: broadcast::Sender<SourceChange>,
}

impl X11SourceEnumerator {
    /// Open the event-listener connection and spawn its thread.
    /// Returns [`SourceError::Backend`] if no X server is reachable
    /// (no `DISPLAY`, refused connection, etc.) so callers know to
    /// fall back to a stub enumerator instead of treating us as a
    /// valid backend that just happens to be empty.
    pub fn try_new() -> Result<Self, SourceError> {
        // Probe the connection on the calling thread so failures
        // surface synchronously — the listener thread would
        // otherwise eat them and look like a healthy-but-empty
        // backend.
        let (probe, _) = xcb::Connection::connect(None)
            .map_err(|e| SourceError::Backend(format!("connect: {e}")))?;
        drop(probe);

        let (tx, _) = broadcast::channel(16);
        let tx2 = tx.clone();
        std::thread::Builder::new()
            .name("ferricast-x11-enum-events".into())
            .spawn(move || {
                if let Err(e) = event_loop(tx2) {
                    warn!(%e, "x11 enumerator event loop exited");
                }
            })
            .map_err(|e| SourceError::Backend(format!("spawn: {e}")))?;
        Ok(Self { change_tx: tx })
    }
}

#[async_trait]
impl SourceEnumerator for X11SourceEnumerator {
    fn backend_name(&self) -> &'static str {
        "x11"
    }

    fn capabilities(&self) -> Vec<EnumerationCapability> {
        vec![
            EnumerationCapability::Monitors,
            EnumerationCapability::Windows,
        ]
    }

    async fn list_monitors(&self) -> Result<Vec<MonitorInfo>, SourceError> {
        tokio::task::spawn_blocking(query_monitors)
            .await
            .map_err(|e| SourceError::Backend(format!("join: {e}")))?
    }

    async fn list_windows(&self) -> Result<Vec<WindowInfo>, SourceError> {
        tokio::task::spawn_blocking(query_windows)
            .await
            .map_err(|e| SourceError::Backend(format!("join: {e}")))?
    }

    fn subscribe(&self) -> broadcast::Receiver<SourceChange> {
        self.change_tx.subscribe()
    }
}

fn query_monitors() -> Result<Vec<MonitorInfo>, SourceError> {
    let (conn, screen_num) = xcb::Connection::connect(None)
        .map_err(|e| SourceError::Backend(format!("connect: {e}")))?;
    let setup = conn.get_setup();
    let screen = setup
        .roots()
        .nth(screen_num as usize)
        .ok_or_else(|| SourceError::Backend("no screen".into()))?;
    let root = screen.root();

    let cookie = conn.send_request(&randr::GetMonitors {
        window: root,
        get_active: true,
    });
    let reply = conn
        .wait_for_reply(cookie)
        .map_err(|e| SourceError::Backend(format!("GetMonitors: {e}")))?;

    let mut out = Vec::new();
    for m in reply.monitors() {
        // The name atom needs a GetAtomName round-trip; do them all
        // in flight so we don't pay N*RTT for N outputs.
        let name_cookie = conn.send_request(&x::GetAtomName { atom: m.name() });
        let name = conn
            .wait_for_reply(name_cookie)
            .ok()
            .map(|r| String::from_utf8_lossy(r.name().to_utf8().as_bytes()).into_owned())
            .unwrap_or_default();
        out.push(MonitorInfo {
            id: name.clone(),
            name,
            make: None,
            model: None,
            geometry: Geometry {
                x: m.x() as i32,
                y: m.y() as i32,
                width: m.width() as u32,
                height: m.height() as u32,
            },
            scale: 1.0,
            refresh_mhz: None,
            primary: m.primary(),
        });
    }
    Ok(out)
}

fn query_windows() -> Result<Vec<WindowInfo>, SourceError> {
    let (conn, screen_num) = xcb::Connection::connect(None)
        .map_err(|e| SourceError::Backend(format!("connect: {e}")))?;
    let setup = conn.get_setup();
    let screen = setup
        .roots()
        .nth(screen_num as usize)
        .ok_or_else(|| SourceError::Backend("no screen".into()))?;
    let root = screen.root();

    let atoms = Atoms::intern(&conn).map_err(|e| SourceError::Backend(format!("atoms: {e}")))?;

    // _NET_CLIENT_LIST: array of WINDOW ids (32-bit each).
    let list_cookie = conn.send_request(&x::GetProperty {
        delete: false,
        window: root,
        property: atoms.net_client_list,
        r#type: x::ATOM_WINDOW,
        long_offset: 0,
        long_length: 4096,
    });
    let reply = conn
        .wait_for_reply(list_cookie)
        .map_err(|e| SourceError::Backend(format!("GetProperty(_NET_CLIENT_LIST): {e}")))?;

    let windows: &[x::Window] = reply.value();
    if windows.is_empty() {
        return Ok(Vec::new());
    }

    // Compute monitor coverage once so we can attribute each window
    // to its dominant output without re-querying RandR per window.
    let monitors = query_monitors().unwrap_or_default();

    let mut out = Vec::with_capacity(windows.len());
    for &w in windows {
        // Window-type filter: only Normal/Dialog windows are
        // streamable in the picker sense — skip docks, menus,
        // splashes, dropdowns, tooltips, notifications.
        let type_cookie = conn.send_request(&x::GetProperty {
            delete: false,
            window: w,
            property: atoms.net_wm_window_type,
            r#type: x::ATOM_ATOM,
            long_offset: 0,
            long_length: 16,
        });
        let title_cookie = conn.send_request(&x::GetProperty {
            delete: false,
            window: w,
            property: atoms.net_wm_name,
            r#type: atoms.utf8_string,
            long_offset: 0,
            long_length: 1024,
        });
        let class_cookie = conn.send_request(&x::GetProperty {
            delete: false,
            window: w,
            property: x::ATOM_WM_CLASS,
            r#type: x::ATOM_STRING,
            long_offset: 0,
            long_length: 256,
        });
        let pid_cookie = conn.send_request(&x::GetProperty {
            delete: false,
            window: w,
            property: atoms.net_wm_pid,
            r#type: x::ATOM_CARDINAL,
            long_offset: 0,
            long_length: 1,
        });
        let geom_cookie = conn.send_request(&x::GetGeometry {
            drawable: x::Drawable::Window(w),
        });
        let trans_cookie = conn.send_request(&x::TranslateCoordinates {
            src_window: w,
            dst_window: root,
            src_x: 0,
            src_y: 0,
        });

        // `_NET_WM_WINDOW_TYPE` is optional; absence means "treat as
        // Normal" per the EWMH spec, so we keep the window when the
        // property is missing or empty.
        let keep = match conn.wait_for_reply(type_cookie) {
            Ok(r) if !r.value::<x::Atom>().is_empty() => r
                .value::<x::Atom>()
                .iter()
                .any(|a| *a == atoms.net_wm_window_type_normal),
            _ => true,
        };
        if !keep {
            // Drain the remaining cookies so xcb doesn't complain;
            // we don't need their replies.
            let _ = conn.wait_for_reply(title_cookie);
            let _ = conn.wait_for_reply(class_cookie);
            let _ = conn.wait_for_reply(pid_cookie);
            let _ = conn.wait_for_reply(geom_cookie);
            let _ = conn.wait_for_reply(trans_cookie);
            continue;
        }

        let title = conn
            .wait_for_reply(title_cookie)
            .ok()
            .map(|r| String::from_utf8_lossy(r.value::<u8>()).into_owned())
            .unwrap_or_default();
        let app_id = conn.wait_for_reply(class_cookie).ok().and_then(|r| {
            let raw: &[u8] = r.value();
            // WM_CLASS is two NUL-terminated strings: instance, class.
            // Prefer class (second one); fall back to instance.
            let mut parts = raw.split(|b| *b == 0).filter(|p| !p.is_empty());
            let first = parts.next();
            let second = parts.next();
            second
                .or(first)
                .map(|p| String::from_utf8_lossy(p).into_owned())
        });
        let pid = conn
            .wait_for_reply(pid_cookie)
            .ok()
            .and_then(|r| r.value::<u32>().first().copied());

        let (geometry, on_monitor) = match (
            conn.wait_for_reply(geom_cookie),
            conn.wait_for_reply(trans_cookie),
        ) {
            (Ok(g), Ok(t)) => {
                let geo = Geometry {
                    x: t.dst_x() as i32,
                    y: t.dst_y() as i32,
                    width: g.width() as u32,
                    height: g.height() as u32,
                };
                let mon = monitor_for(&monitors, &geo);
                (Some(geo), mon)
            }
            _ => (None, None),
        };

        out.push(WindowInfo {
            id: w.resource_id().to_string(),
            title,
            app_id,
            pid,
            geometry,
            on_monitor,
        });
    }
    debug!(count = out.len(), "x11 windows enumerated");
    Ok(out)
}

/// Pick the monitor with the largest intersection with `geo`.
/// Returns `None` if the window doesn't overlap any output (off-screen
/// / minimized).
fn monitor_for(monitors: &[MonitorInfo], geo: &Geometry) -> Option<String> {
    let mut best: Option<(&MonitorInfo, i64)> = None;
    for m in monitors {
        let ix = geo.x.max(m.geometry.x);
        let iy = geo.y.max(m.geometry.y);
        let ax = (geo.x + geo.width as i32).min(m.geometry.x + m.geometry.width as i32);
        let ay = (geo.y + geo.height as i32).min(m.geometry.y + m.geometry.height as i32);
        let area = ((ax - ix).max(0) as i64) * ((ay - iy).max(0) as i64);
        if area > 0 && best.map(|(_, a)| area > a).unwrap_or(true) {
            best = Some((m, area));
        }
    }
    best.map(|(m, _)| m.id.clone())
}

fn event_loop(tx: broadcast::Sender<SourceChange>) -> Result<(), SourceError> {
    let (conn, screen_num) = xcb::Connection::connect_with_extensions(
        None,
        &[xcb::Extension::RandR],
        &[],
    )
    .map_err(|e| SourceError::Backend(format!("connect: {e}")))?;
    let setup = conn.get_setup();
    let screen = setup
        .roots()
        .nth(screen_num as usize)
        .ok_or_else(|| SourceError::Backend("no screen".into()))?;
    let root = screen.root();
    let atoms = Atoms::intern(&conn).map_err(|e| SourceError::Backend(format!("atoms: {e}")))?;

    conn.send_request(&randr::SelectInput {
        window: root,
        enable: randr::NotifyMask::SCREEN_CHANGE,
    });
    conn.send_request(&x::ChangeWindowAttributes {
        window: root,
        value_list: &[x::Cw::EventMask(x::EventMask::PROPERTY_CHANGE)],
    });
    conn.flush()
        .map_err(|e| SourceError::Backend(format!("flush: {e}")))?;

    loop {
        let ev = conn
            .wait_for_event()
            .map_err(|e| SourceError::Disconnected(e.to_string()))?;
        match ev {
            xcb::Event::RandR(randr::Event::ScreenChangeNotify(_)) => {
                let _ = tx.send(SourceChange::Monitors);
            }
            xcb::Event::X(x::Event::PropertyNotify(pn)) => {
                let a = pn.atom();
                if a == atoms.net_client_list || a == atoms.net_active_window {
                    let _ = tx.send(SourceChange::Windows);
                } else if a == atoms.net_wm_name {
                    // Title change on some window — coalesce as
                    // Windows so pickers re-list and pick up the
                    // new title.
                    let _ = tx.send(SourceChange::Windows);
                }
            }
            _ => {}
        }
    }
}

