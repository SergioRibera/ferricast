//! Source enumeration — listing the monitors and windows that the
//! local desktop can stream.
//!
//! This is **separate** from [`crate::capture`]: capturing a source
//! and *picking* one to capture are different problems. The capture
//! API needs to support OS portals (Wayland's PipeWire portal, the
//! XDG ScreenCast portal) where the *user* picks via a portal-owned
//! dialog and the app never sees the candidate list. That's a
//! deliberate sandboxing guarantee and shouldn't change.
//!
//! `SourceEnumerator` exists for the *other* case: in-process
//! pickers, third-party widgets, status-bar applets, sidebar UIs,
//! anything that wants to render its own list of monitors / windows
//! before handing the chosen source back to ferricast. Concretely:
//!
//! - **X11** can always enumerate (XRandR + EWMH).
//! - **wlroots-based Wayland compositors** (Hyprland, sway, river,
//!   Wayfire) expose `zwlr_foreign_toplevel_management_v1` and
//!   `xdg-output`, so we can enumerate there too.
//! - **GNOME, KDE/Plasma, Mutter without extensions** do not expose
//!   a public protocol for either; the only correct answer is to
//!   delegate to the portal picker. The [`StubEnumerator`] reports
//!   no capabilities so clients can detect this case and switch UI.
//!
//! Implementations live in `ferricast-capture` so the heavy system
//! dependencies (xcb, wayland-client) stay out of `ferricast-core`.

use async_trait::async_trait;
use tokio::sync::broadcast;

/// Rectangle in compositor pixels. `(x, y)` is the top-left corner
/// in the global desktop layout; `width` / `height` are extents.
/// For X11 this matches the root-window coordinate space; for
/// wlroots it matches the logical layout reported by `xdg-output`
/// (i.e. after scaling, not raw output mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// A monitor (output) attached to the local session.
#[derive(Debug, Clone, PartialEq)]
pub struct MonitorInfo {
    /// Stable identifier. On X11 this is the XRandR output name
    /// (`HDMI-1`, `DP-2`…). On wlroots it's the wl_output `name`
    /// event payload, which compositors guarantee is stable across
    /// reconnects for the same physical output.
    pub id: String,
    /// Display label. Falls back to `id` when the compositor /
    /// server doesn't provide a description.
    pub name: String,
    /// EDID-derived manufacturer, when available.
    pub make: Option<String>,
    /// EDID-derived model name, when available.
    pub model: Option<String>,
    /// Position + extents in the compositor's logical pixel space.
    pub geometry: Geometry,
    /// Scale factor (1.0, 1.25, 1.5, 2.0…). Logical pixels =
    /// physical / `scale`.
    pub scale: f32,
    /// Refresh rate in millihertz (60_000 = 60 Hz). `None` when
    /// the backend can't report it.
    pub refresh_mhz: Option<u32>,
    /// Whether this monitor is the user's primary. On wlroots
    /// there's no canonical "primary" concept; backends MAY pick
    /// the first output or leave this `false` for everything.
    pub primary: bool,
}

/// A streamable top-level window.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowInfo {
    /// Stable handle. On X11 this is the decimal X11 window id
    /// (`u32`). On wlroots it's a backend-internal handle string
    /// (the toplevel object's id) that stays valid as long as the
    /// window exists.
    pub id: String,
    /// Window title (`_NET_WM_NAME` on X11, `title` on wlroots).
    pub title: String,
    /// Application identifier — `WM_CLASS` instance/class on X11,
    /// `app_id` on wlroots. Useful for grouping or showing icons.
    pub app_id: Option<String>,
    /// Owning PID, when the backend can resolve it (`_NET_WM_PID`
    /// on X11; wlroots doesn't expose it).
    pub pid: Option<u32>,
    /// Frame geometry, when the backend can compute it. Optional
    /// because wlroots only reports geometry once the window is on
    /// an output, and we'd rather report partial data than skip
    /// the window entirely.
    pub geometry: Option<Geometry>,
    /// `MonitorInfo::id` of the output this window is currently on,
    /// when known. Multi-monitor windows pick the dominant one.
    pub on_monitor: Option<String>,
}

/// Coarse-grained change notification. Clients re-call `list_*` on
/// receipt — same pattern as wlr/xdg portals, robust against missed
/// events because the next List call resyncs the world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceChange {
    /// At least one monitor was added/removed/moved/rescaled.
    Monitors,
    /// At least one window was added/removed/retitled/refocused.
    Windows,
}

/// What a backend can enumerate. Surfaced verbatim over D-Bus so
/// clients can grey out picker tabs without round-tripping the
/// actual list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnumerationCapability {
    Monitors,
    Windows,
}

impl EnumerationCapability {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Monitors => "monitors",
            Self::Windows => "windows",
        }
    }
}

/// Enumeration failure. Backends use [`SourceError::Unsupported`]
/// when the capability is fundamentally unavailable (e.g. asking
/// for windows on GNOME-Wayland) — clients can treat that as a
/// permanent "no" without retrying.
#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    #[error("backend does not support enumerating {0:?}")]
    Unsupported(EnumerationCapability),
    #[error("backend transport disconnected: {0}")]
    Disconnected(String),
    #[error("backend error: {0}")]
    Backend(String),
}

/// In-process enumeration API. The daemon holds one of these via
/// `Arc<dyn SourceEnumerator>` and bridges it to D-Bus; other
/// consumers (a Rust GUI widget, an embedded picker) can use the
/// same trait without going through the bus.
///
/// All `list_*` calls are snapshots — for a live view, subscribe
/// via [`SourceEnumerator::subscribe`] and re-list on each change.
#[async_trait]
pub trait SourceEnumerator: Send + Sync {
    /// Identifier of the concrete backend (e.g. `"x11"`,
    /// `"wlroots"`, `"stub"`). Mostly diagnostic.
    fn backend_name(&self) -> &'static str;

    /// Capabilities advertised to clients. Methods MUST return
    /// [`SourceError::Unsupported`] for anything not in this list,
    /// and SHOULD NOT return success for anything that IS — even
    /// if "success" means an empty list.
    fn capabilities(&self) -> Vec<EnumerationCapability>;

    async fn list_monitors(&self) -> Result<Vec<MonitorInfo>, SourceError>;
    async fn list_windows(&self) -> Result<Vec<WindowInfo>, SourceError>;

    /// Coarse change stream. The returned receiver is independent
    /// per call — multiple consumers can subscribe; lagging
    /// consumers may see [`broadcast::error::RecvError::Lagged`]
    /// (always safe to ignore and re-list).
    fn subscribe(&self) -> broadcast::Receiver<SourceChange>;
}

/// Fallback enumerator that reports no capabilities. Returned by
/// the auto-detect factory when nothing supported is available
/// (typically: Wayland on a non-wlroots compositor without X11
/// fallback). Lets daemon code treat enumeration as always-present
/// and capability-gated, instead of `Option<_>` everywhere.
pub struct StubEnumerator {
    tx: broadcast::Sender<SourceChange>,
}

impl StubEnumerator {
    pub fn new() -> Self {
        // Capacity 1 is plenty — stub never emits — but
        // `broadcast::channel` requires > 0.
        let (tx, _) = broadcast::channel(1);
        Self { tx }
    }
}

impl Default for StubEnumerator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SourceEnumerator for StubEnumerator {
    fn backend_name(&self) -> &'static str {
        "stub"
    }
    fn capabilities(&self) -> Vec<EnumerationCapability> {
        Vec::new()
    }
    async fn list_monitors(&self) -> Result<Vec<MonitorInfo>, SourceError> {
        Err(SourceError::Unsupported(EnumerationCapability::Monitors))
    }
    async fn list_windows(&self) -> Result<Vec<WindowInfo>, SourceError> {
        Err(SourceError::Unsupported(EnumerationCapability::Windows))
    }
    fn subscribe(&self) -> broadcast::Receiver<SourceChange> {
        self.tx.subscribe()
    }
}
