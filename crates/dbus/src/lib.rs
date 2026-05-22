//! D-Bus contract for the ferricast daemon.
//!
//! This crate is the only thing that needs to be in sync between
//! the daemon (in `crates/gui`) and any client — Rust, Python,
//! TypeScript, etc. The Rust side gets the strongly-typed [`Manager`]
//! proxy + the data-transfer structs; foreign languages can consume
//! the introspection XML at [`INTROSPECTION_XML`] (also shipped as a
//! file under `crates/dbus/xml/`).
//!
//! ## Wire layout
//!
//! - Bus: **session**
//! - Bus name: [`BUS_NAME`] (`rs.sergioribera.ferricast`)
//! - Object path: [`OBJECT_PATH`] (`/rs/sergioribera/ferricast`)
//! - Interface: [`INTERFACE`] (`rs.sergioribera.ferricast.Manager1`)
//!
//! ## Real-time listings
//!
//! `ListDevices` returns whatever the daemon has seen so far; for a
//! live view, subscribe to [`Manager::receive_device_added`] /
//! [`Manager::receive_device_removed`] and merge into a local map.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use zbus::zvariant::{self, OwnedValue, Type};

/// Well-known session-bus name owned by the daemon while it's running.
pub const BUS_NAME: &str = "rs.sergioribera.ferricast";

/// Root object exposed by the daemon.
pub const OBJECT_PATH: &str = "/rs/sergioribera/ferricast";

/// Primary interface name. Versioned (`Manager1`) so we can ship
/// breaking changes as a new interface on the same object.
pub const INTERFACE: &str = "rs.sergioribera.ferricast.Manager1";

/// Static D-Bus introspection XML for the interface. Use with
/// `gdbus-codegen`, `pydbus`, etc. Kept in sync with the proxy by
/// `crates/dbus/xml/<interface>.xml`.
pub const INTROSPECTION_XML: &str =
    include_str!("../xml/rs.sergioribera.ferricast.Manager1.xml");

/// Snapshot of a device exposed over D-Bus. Mirrors
/// [`ferricast_core::Device`] minus runtime-only fields (icon bytes,
/// socket addresses) that don't make sense to ship over the bus.
///
/// `capabilities` is a free-form `a{sv}` so we can add new fields
/// (max_fps, max_bitrate_kbps, max_h264_profile, requires_audio…)
/// without bumping the interface version every time.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct DeviceDto {
    pub id: String,
    pub name: String,
    pub protocol: String,
    pub model: String,
    pub host: String,
    pub capabilities: HashMap<String, OwnedValue>,
}

/// Capture-source request over the bus.
///
/// `kind` selects the variant; `args` carries the kind-specific
/// arguments as `a{sv}` so each backend can grow new options
/// without a wire break.
///
/// | kind        | args                          | meaning                                       |
/// |-------------|-------------------------------|-----------------------------------------------|
/// | `""`        | (empty)                       | Let the daemon pick: PipeWire portal on       |
/// |             |                               | Wayland; on X11 a picker dialog (TODO).       |
/// | `"screen"`  | `monitor: s?`                 | Full-screen capture. `monitor` is a string id |
/// |             |                               | matching `MonitorInfoDto.id`. Legacy alias    |
/// |             |                               | for `"monitor"` — prefer the latter.          |
/// | `"monitor"` | `id: s`                       | Capture the monitor whose id matches.         |
/// | `"window"`  | `id: s` *or* `title: s`       | Capture a specific window. `id` is the id     |
/// |             |                               | the daemon previously returned in             |
/// |             |                               | `ListWindows`; `title` is a fallback for      |
/// |             |                               | callers without access to the enumerator.     |
#[derive(Debug, Clone, Default, Serialize, Deserialize, Type)]
pub struct SourceDto {
    pub kind: String,
    pub args: HashMap<String, OwnedValue>,
}

impl SourceDto {
    /// Daemon-chosen default. The daemon decides what this means
    /// based on the active session.
    pub fn auto() -> Self {
        Self::default()
    }

    /// Full-screen, daemon-chosen monitor.
    pub fn screen() -> Self {
        Self {
            kind: "screen".into(),
            args: HashMap::new(),
        }
    }

    /// Capture the monitor whose [`MonitorInfoDto::id`] equals `id`.
    /// The daemon validates the id against its enumerator before
    /// dispatching to the capture backend, so passing a stale id
    /// produces a clean `InvalidArgs` error instead of a generic
    /// capture failure.
    pub fn monitor(id: impl Into<String>) -> Self {
        let mut args = HashMap::new();
        if let Ok(v) = OwnedValue::try_from(zvariant::Value::from(id.into())) {
            args.insert("id".into(), v);
        }
        Self {
            kind: "monitor".into(),
            args,
        }
    }

    /// Empty `kind="window"` — daemon picks (portal flow).
    pub fn window() -> Self {
        Self {
            kind: "window".into(),
            args: HashMap::new(),
        }
    }

    /// Capture the window whose [`WindowInfoDto::id`] equals `id`.
    /// Same resolution + validation flow as [`SourceDto::monitor`].
    pub fn window_by_id(id: impl Into<String>) -> Self {
        let mut args = HashMap::new();
        if let Ok(v) = OwnedValue::try_from(zvariant::Value::from(id.into())) {
            args.insert("id".into(), v);
        }
        Self {
            kind: "window".into(),
            args,
        }
    }

    /// Capture the first window whose title matches. Used by clients
    /// that don't (or can't) talk to the enumerator — e.g. quick
    /// shell scripts. Less precise than `window_by_id` because two
    /// windows can share a title.
    pub fn window_by_title(title: impl Into<String>) -> Self {
        let mut args = HashMap::new();
        if let Ok(v) = OwnedValue::try_from(zvariant::Value::from(title.into())) {
            args.insert("title".into(), v);
        }
        Self {
            kind: "window".into(),
            args,
        }
    }

    /// Set the `audio` flag in `args`. The daemon reads this when
    /// starting the stream to decide whether to mux a captured
    /// audio track alongside the video. Callable on any source
    /// (`SourceDto::monitor("DP-1").with_audio(true)` etc.) —
    /// chains fluently with the constructors above.
    ///
    /// Defaults to absent → daemon treats it as `false` (no audio).
    pub fn with_audio(mut self, audio: bool) -> Self {
        if let Ok(v) = OwnedValue::try_from(zvariant::Value::from(audio)) {
            self.args.insert("audio".into(), v);
        }
        self
    }

    /// Read the `audio` flag previously set with `with_audio`. Used
    /// by the daemon to decide whether to wire an audio capture
    /// pipeline alongside video. Defaults to `false` when absent.
    pub fn audio(&self) -> bool {
        self.args
            .get("audio")
            .and_then(|v| v.downcast_ref::<bool>().ok())
            .unwrap_or(false)
    }
}

/// Summary entry from `ListActiveStreams`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct ActiveStreamDto {
    pub device_id: String,
    pub device_name: String,
}

/// Monitor entry on the wire.
///
/// `Option<T>` fields are flattened with sentinel values so the
/// signature stays a plain struct (`gdbus-codegen`-friendly) and we
/// don't pay the cost of D-Bus `maybe` types:
///
/// - `make` / `model` / `on_monitor` empty string → unknown.
/// - `refresh_mhz == 0` → unknown.
///
/// `extra` is an `a{sv}` tail so backends can publish backend-specific
/// fields (e.g. transform, EDID hash, HDR support) without a wire
/// break.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct MonitorInfoDto {
    pub id: String,
    pub name: String,
    pub make: String,
    pub model: String,
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub scale: f64,
    pub refresh_mhz: u32,
    pub primary: bool,
    pub extra: HashMap<String, OwnedValue>,
}

/// Window entry on the wire. Same flattening conventions as
/// [`MonitorInfoDto`].
///
/// `has_geometry` exists because some backends (wlroots) only report
/// window geometry once the window enters an output; we want to keep
/// the window in the list even before that point. When `false`, the
/// x/y/width/height fields are meaningless.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct WindowInfoDto {
    pub id: String,
    pub title: String,
    pub app_id: String,
    pub pid: u32,
    pub has_geometry: bool,
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub on_monitor: String,
    pub extra: HashMap<String, OwnedValue>,
}

/// Strongly-typed proxy for the daemon's manager interface.
///
/// ```no_run
/// # async fn ex() -> zbus::Result<()> {
/// let conn = zbus::Connection::session().await?;
/// let proxy = ferricast_dbus::ManagerProxy::new(&conn).await?;
/// for d in proxy.list_devices().await? {
///     println!("{} ({})", d.name, d.protocol);
/// }
/// # Ok(()) }
/// ```
#[zbus::proxy(
    interface = "rs.sergioribera.ferricast.Manager1",
    default_service = "rs.sergioribera.ferricast",
    default_path = "/rs/sergioribera/ferricast"
)]
pub trait Manager {
    /// All devices the daemon currently knows about.
    fn list_devices(&self) -> zbus::Result<Vec<DeviceDto>>;

    /// Devices that currently have an active stream.
    fn list_active_streams(&self) -> zbus::Result<Vec<ActiveStreamDto>>;

    /// Start a stream on `device_id`. `source` selects what to share;
    /// pass [`SourceDto::auto()`] to let the daemon choose.
    fn start_stream(&self, device_id: &str, source: SourceDto) -> zbus::Result<()>;

    /// Stop the active stream on `device_id`. Errors if nothing is
    /// streaming to that device.
    fn stop_stream(&self, device_id: &str) -> zbus::Result<()>;

    /// Names of receiver protocols compiled into the running daemon
    /// (e.g. `["chromecast"]`). Useful for clients that want to grey
    /// out features without a separate capability handshake.
    #[zbus(property)]
    fn protocols(&self) -> zbus::Result<Vec<String>>;

    /// Snapshot of every monitor the local session can stream. Fails
    /// with `org.freedesktop.DBus.Error.NotSupported` when the daemon
    /// is running on a backend that cannot enumerate monitors
    /// (typically: Wayland on GNOME/KDE without extensions). Use
    /// [`Manager::enumeration_capabilities`] to detect that up-front
    /// and present the right UI.
    fn list_monitors(&self) -> zbus::Result<Vec<MonitorInfoDto>>;

    /// Snapshot of every streamable top-level window in the local
    /// session. Same NotSupported semantics as `ListMonitors`.
    fn list_windows(&self) -> zbus::Result<Vec<WindowInfoDto>>;

    /// What the daemon's enumerator can publish. Each entry is one
    /// of `"monitors"` / `"windows"`. An empty list means the daemon
    /// is on a backend that supports neither — clients should fall
    /// back to the OS portal picker for source selection.
    #[zbus(property)]
    fn enumeration_capabilities(&self) -> zbus::Result<Vec<String>>;

    /// What the daemon can actually *capture* end-to-end without
    /// re-asking the user via xdg-desktop-portal. Each entry is one
    /// of `"monitor"` / `"window"`. Pickers should use this to
    /// disable tabs whose selection wouldn't be honoured by the
    /// streaming path — e.g. on niri today this returns `["monitor"]`
    /// because window streaming requires
    /// `ext_foreign_toplevel_image_capture_source_manager_v1` which
    /// the compositor doesn't expose yet.
    ///
    /// `enumeration_capabilities` and `capture_capabilities` are
    /// independent: a backend can enumerate windows it can't stream
    /// (the picker can show them as informational, but selecting
    /// one would have to fall back to the portal).
    #[zbus(property)]
    fn capture_capabilities(&self) -> zbus::Result<Vec<String>>;

    /// Capture a one-shot preview of the monitor `id`, downscaled
    /// to fit in `max_width × max_height` (aspect-preserving), and
    /// return it as PNG bytes ready to feed into an image widget.
    ///
    /// Returns:
    /// - PNG bytes on success.
    /// - Empty byte array `[]` when the backend doesn't have a
    ///   capture path *for that specific item* — typical case is
    ///   Wayland-on-niri asking for a window thumbnail before the
    ///   compositor exposes `ext-image-copy-capture-v1`. Clients
    ///   should render a placeholder.
    /// - `org.freedesktop.DBus.Error.NotSupported` when no
    ///   thumbnail capability is wired up at all (Stub backends).
    /// - `org.freedesktop.DBus.Error.InvalidArgs` when `id` doesn't
    ///   match anything in the latest `ListMonitors` snapshot.
    fn get_monitor_thumbnail(
        &self,
        id: &str,
        max_width: u32,
        max_height: u32,
    ) -> zbus::Result<Vec<u8>>;

    /// Same as [`Manager::get_monitor_thumbnail`] but for a window.
    /// On Wayland this needs `ext-image-copy-capture-v1`; without it
    /// the daemon returns an empty array (NOT `NotSupported`) so the
    /// picker can show a placeholder per-window instead of disabling
    /// previews entirely.
    fn get_window_thumbnail(
        &self,
        id: &str,
        max_width: u32,
        max_height: u32,
    ) -> zbus::Result<Vec<u8>>;

    #[zbus(signal)]
    fn device_added(&self, device: DeviceDto) -> zbus::Result<()>;

    #[zbus(signal)]
    fn device_removed(&self, device_id: String) -> zbus::Result<()>;

    #[zbus(signal)]
    fn stream_started(&self, device_id: String, device_name: String) -> zbus::Result<()>;

    #[zbus(signal)]
    fn stream_stopped(&self, device_id: String) -> zbus::Result<()>;

    #[zbus(signal)]
    fn stream_reconnecting(
        &self,
        device_id: String,
        attempt: u32,
        reason: String,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    fn stream_error(&self, device_id: String, message: String) -> zbus::Result<()>;

    #[zbus(signal)]
    fn discovery_error(&self, protocol: String, message: String) -> zbus::Result<()>;

    /// Fired when at least one monitor was added, removed, moved or
    /// rescaled. Coarse on purpose: re-call [`Manager::list_monitors`]
    /// to get a fresh snapshot. Same contract as the wlroots /
    /// portal protocols this backend mirrors.
    #[zbus(signal)]
    fn monitors_changed(&self) -> zbus::Result<()>;

    /// Fired when at least one window was added, removed, retitled
    /// or moved between outputs. Re-call [`Manager::list_windows`]
    /// to resync.
    #[zbus(signal)]
    fn windows_changed(&self) -> zbus::Result<()>;
}
