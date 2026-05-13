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
use zbus::zvariant::{OwnedValue, Type};

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
/// | kind       | args                          | meaning                                   |
/// |------------|-------------------------------|-------------------------------------------|
/// | `""`       | (empty)                       | Let the daemon pick: PipeWire portal on   |
/// |            |                               | Wayland; on X11 a picker dialog (TODO).   |
/// | `"screen"` | `monitor: s?`                 | Full-screen capture, optional monitor id. |
/// | `"window"` | `identifier: s?`              | Window capture; identifier is portal-     |
/// |            |                               | specific.                                 |
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

    pub fn screen() -> Self {
        Self {
            kind: "screen".into(),
            args: HashMap::new(),
        }
    }

    pub fn window() -> Self {
        Self {
            kind: "window".into(),
            args: HashMap::new(),
        }
    }
}

/// Summary entry from `ListActiveStreams`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct ActiveStreamDto {
    pub device_id: String,
    pub device_name: String,
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
}
