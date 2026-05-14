//! D-Bus daemon: wraps a [`StreamManager`] with the
//! `rs.sergioribera.ferricast.Manager1` interface and re-emits
//! manager events as D-Bus signals.
//!
//! The daemon is a thin façade on purpose — every method either
//! delegates to `StreamManager` directly or maps types between the
//! wire DTOs in `ferricast-dbus` and the rich Rust types in
//! `ferricast-core`. Anything that does real work (capture,
//! encoding, network) lives downstream of the manager.

use std::collections::HashMap;
use std::sync::Arc;

use ferricast::prelude::*;
use ferricast::{ManagerEvent, SourceChange, SourceEnumerator};
use ferricast_dbus::{
    ActiveStreamDto, BUS_NAME, DeviceDto, MonitorInfoDto, OBJECT_PATH, SourceDto, WindowInfoDto,
};
use tokio::sync::{Mutex, mpsc};
use uuid::Uuid;
use zbus::object_server::SignalEmitter;
use zbus::zvariant::{OwnedValue, Value};

/// Build a [`DeviceDto`] from a core [`Device`]. The `protocol_icon`
/// is dropped — sending raw bytes over D-Bus for every signal is
/// expensive and most clients want to load icons by protocol name
/// from their own asset set anyway.
fn device_to_dto(d: &Device) -> DeviceDto {
    let mut caps: HashMap<String, OwnedValue> = HashMap::new();
    if let Some(v) = d.capabilities.max_fps {
        caps.insert("max_fps".into(), OwnedValue::from(v));
    }
    if let Some(v) = d.capabilities.max_bitrate_kbps {
        caps.insert("max_bitrate_kbps".into(), OwnedValue::from(v));
    }
    if let Some(p) = d.capabilities.max_h264_profile {
        if let Ok(v) = Value::Str(format!("{p:?}").into()).try_to_owned() {
            caps.insert("max_h264_profile".into(), v);
        }
    }
    caps.insert("requires_audio".into(), OwnedValue::from(d.capabilities.requires_audio));

    DeviceDto {
        id: d.id.to_string(),
        name: d.name.clone(),
        protocol: d.protocol.to_string(),
        model: d.model.clone().unwrap_or_default(),
        host: format!("{}:{}", d.addr, d.port),
        capabilities: caps,
    }
}

/// Pull a string field out of a `SourceDto.args` map. Quietly
/// ignores fields whose value isn't a string — the wire format is
/// `a{sv}` precisely so we can grow new field types without breaking
/// existing daemons, and a future client sending e.g. a number where
/// we expect a string shouldn't crash discovery.
fn str_arg(s: &SourceDto, key: &str) -> Option<String> {
    let v = s.args.get(key)?;
    v.downcast_ref::<&str>().ok().map(|s| s.to_owned())
}

/// Translate a wire [`SourceDto`] into a [`CaptureSource`], resolving
/// any picker-issued ids against the live enumerator first. The
/// kinds we accept:
///
/// - `""` → full-screen, daemon-chosen monitor.
/// - `"screen"` with optional `monitor: s` (legacy) → full-screen,
///   that monitor.
/// - `"monitor"` with `id: s` (preferred) → full-screen, monitor
///   with the given id. Errors `InvalidArgs` if no such monitor
///   exists *and* the enumerator can actually enumerate (stub
///   backends fall through to "let the capture backend try its
///   own picker").
/// - `"window"` with `id: s` → look the window up in the enumerator,
///   prefer `WindowIdentifier::Id` when the id parses as a u64
///   (X11 case), fall back to its current title (wlroots case
///   where ids are wayland protocol_ids and don't survive a fresh
///   wl_display connection — title is the only handle the capture
///   backend can re-resolve).
/// - `"window"` with `title: s` → exact title; no enumerator lookup.
///
/// Returning `zbus::fdo::Error` so unknown ids surface as proper bus
/// errors, not a generic capture failure later.
async fn resolve_source(
    enumerator: &dyn SourceEnumerator,
    s: &SourceDto,
) -> zbus::fdo::Result<CaptureSource> {
    match s.kind.as_str() {
        "" | "screen" => Ok(CaptureSource::FullScreen {
            monitor: str_arg(s, "monitor").or_else(|| str_arg(s, "id")),
        }),
        "monitor" => {
            let Some(id) = str_arg(s, "id") else {
                return Err(zbus::fdo::Error::InvalidArgs(
                    "`monitor` source requires args.id".into(),
                ));
            };
            // Validate against the enumerator only when it actually
            // supports monitor enumeration — otherwise we'd reject
            // legitimate ids the capture backend would have accepted
            // on its own (e.g. a portal-issued id).
            if enumerator
                .capabilities()
                .contains(&ferricast::EnumerationCapability::Monitors)
            {
                let mons = enumerator
                    .list_monitors()
                    .await
                    .map_err(|e| zbus::fdo::Error::Failed(format!("list_monitors: {e}")))?;
                if !mons.iter().any(|m| m.id == id) {
                    return Err(zbus::fdo::Error::InvalidArgs(format!(
                        "no monitor with id {id:?} (try ListMonitors to see current ids)"
                    )));
                }
            }
            Ok(CaptureSource::FullScreen { monitor: Some(id) })
        }
        "window" => {
            if let Some(id) = str_arg(s, "id") {
                let supports = enumerator
                    .capabilities()
                    .contains(&ferricast::EnumerationCapability::Windows);
                if supports {
                    let ws = enumerator
                        .list_windows()
                        .await
                        .map_err(|e| zbus::fdo::Error::Failed(format!("list_windows: {e}")))?;
                    let Some(w) = ws.into_iter().find(|w| w.id == id) else {
                        return Err(zbus::fdo::Error::InvalidArgs(format!(
                            "no window with id {id:?} (try ListWindows to see current ids)"
                        )));
                    };
                    // X11 ids are decimal XIDs — pass them through as
                    // numeric so the capture backend doesn't have to
                    // re-parse. Anything that doesn't parse (wlroots
                    // wayland protocol_id) falls back to the current
                    // title; the capture backend will re-resolve.
                    let identifier = if let Ok(num) = w.id.parse::<u64>() {
                        ferricast::WindowIdentifier::Id(num)
                    } else {
                        ferricast::WindowIdentifier::Title(w.title)
                    };
                    return Ok(CaptureSource::Window {
                        identifier: Some(identifier),
                    });
                }
                // No enumerator → trust the caller, treat the id as
                // a literal numeric XID or pass it as a title if it
                // doesn't parse.
                let identifier = id
                    .parse::<u64>()
                    .map(ferricast::WindowIdentifier::Id)
                    .unwrap_or(ferricast::WindowIdentifier::Title(id));
                return Ok(CaptureSource::Window {
                    identifier: Some(identifier),
                });
            }
            Ok(CaptureSource::Window {
                identifier: str_arg(s, "title")
                    .or_else(|| str_arg(s, "identifier"))
                    .map(ferricast::WindowIdentifier::Title),
            })
        }
        other => {
            tracing::warn!(
                kind = other,
                "unknown source kind, defaulting to full-screen"
            );
            Ok(CaptureSource::FullScreen { monitor: None })
        }
    }
}

fn monitor_to_dto(m: ferricast::MonitorInfo) -> MonitorInfoDto {
    MonitorInfoDto {
        id: m.id,
        name: m.name,
        make: m.make.unwrap_or_default(),
        model: m.model.unwrap_or_default(),
        x: m.geometry.x,
        y: m.geometry.y,
        width: m.geometry.width,
        height: m.geometry.height,
        scale: m.scale as f64,
        refresh_mhz: m.refresh_mhz.unwrap_or(0),
        primary: m.primary,
        extra: HashMap::new(),
    }
}

/// Hard upper bound on the thumbnail box the picker can ask for.
/// 4096×4096 is well above what any reasonable picker UI renders
/// and stops a misbehaving client from asking the daemon to
/// allocate a 200 MB buffer.
fn clamp_thumbnail_box(max_w: u32, max_h: u32) -> (u32, u32) {
    const HARD_CAP: u32 = 4096;
    (max_w.min(HARD_CAP).max(1), max_h.min(HARD_CAP).max(1))
}

fn window_to_dto(w: ferricast::WindowInfo) -> WindowInfoDto {
    let (has_geometry, x, y, width, height) = match w.geometry {
        Some(g) => (true, g.x, g.y, g.width, g.height),
        None => (false, 0, 0, 0, 0),
    };
    WindowInfoDto {
        id: w.id,
        title: w.title,
        app_id: w.app_id.unwrap_or_default(),
        pid: w.pid.unwrap_or(0),
        has_geometry,
        x,
        y,
        width,
        height,
        on_monitor: w.on_monitor.unwrap_or_default(),
        extra: HashMap::new(),
    }
}

/// A picker-pop request issued by the daemon when a D-Bus client
/// calls `StartStream` with no concrete source. The Freya app side
/// listens on the corresponding `mpsc::Receiver`, opens the picker
/// window, and replies through the embedded oneshot. `None` reply
/// means the user cancelled.
pub struct PickerRequest {
    pub device_id: Uuid,
    pub reply: tokio::sync::oneshot::Sender<Option<SourceDto>>,
}

/// Object exported on the bus. The manager is shared with the rest
/// of the app (the Freya window holds the same `Arc`), which is why
/// it's behind a mutex. The enumerator is read-only — its own
/// internal locking handles concurrent access.
///
/// `picker_req_tx` is only `Some` when the binary was launched in
/// a mode that has a Freya runtime alive (the foreground GUI or
/// `--background` with a hidden main window). When it's `None` —
/// no Freya at all — `StartStream` calls with empty source fail
/// fast instead of hanging waiting for a picker that can never
/// open.
pub struct ManagerService {
    pub manager: Arc<Mutex<StreamManager>>,
    pub enumerator: Arc<dyn SourceEnumerator>,
    pub picker_req_tx: Option<tokio::sync::mpsc::Sender<PickerRequest>>,
}

/// Heuristic: a `SourceDto` is "abstract" — caller wants help —
/// when its kind is empty, or when it's a `screen` / `monitor` /
/// `window` shape with no concrete id/title to dispatch on. The
/// daemon delegates abstract requests to the in-app picker so the
/// user gets to choose interactively.
fn needs_picker(s: &SourceDto) -> bool {
    fn has_str(s: &SourceDto, key: &str) -> bool {
        s.args
            .get(key)
            .and_then(|v| v.downcast_ref::<&str>().ok())
            .is_some()
    }
    match s.kind.as_str() {
        "" => true,
        "screen" => !has_str(s, "monitor") && !has_str(s, "id"),
        "monitor" => !has_str(s, "id"),
        "window" => !has_str(s, "id") && !has_str(s, "title") && !has_str(s, "identifier"),
        _ => false,
    }
}

impl ManagerService {
    /// Send a picker request to the Freya app and await the reply.
    /// Returns the chosen `SourceDto` or a `zbus::fdo::Error` when
    /// the user cancelled / the picker channel collapsed.
    async fn run_picker(
        &self,
        tx: tokio::sync::mpsc::Sender<PickerRequest>,
        device_id: Uuid,
    ) -> zbus::fdo::Result<SourceDto> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel::<Option<SourceDto>>();
        tx.send(PickerRequest {
            device_id,
            reply: reply_tx,
        })
        .await
        .map_err(|_| {
            zbus::fdo::Error::Failed(
                "picker channel closed — Freya app is not listening for requests".into(),
            )
        })?;
        match reply_rx.await {
            Ok(Some(dto)) => Ok(dto),
            Ok(None) => Err(zbus::fdo::Error::Failed("picker cancelled by user".into())),
            Err(_) => Err(zbus::fdo::Error::Failed(
                "picker oneshot dropped without reply".into(),
            )),
        }
    }

    /// Resolve a `device_id` argument coming from D-Bus. Accepts
    /// either the canonical UUID or a case-insensitive device name.
    /// Returning `zbus::fdo::Error` so the message goes back to the
    /// caller as a proper D-Bus error and not a panic-style trace.
    async fn resolve(&self, ident: &str) -> zbus::fdo::Result<Uuid> {
        if let Ok(uuid) = Uuid::parse_str(ident) {
            return Ok(uuid);
        }
        let needle = ident.to_lowercase();
        let m = self.manager.lock().await;
        for d in m.devices().await {
            if d.name.to_lowercase() == needle {
                return Ok(d.id);
            }
        }
        Err(zbus::fdo::Error::InvalidArgs(format!(
            "no device matches {ident:?} (try `list` to see ids)"
        )))
    }
}

#[zbus::interface(name = "rs.sergioribera.ferricast.Manager1")]
impl ManagerService {
    async fn list_devices(&self) -> zbus::fdo::Result<Vec<DeviceDto>> {
        let m = self.manager.lock().await;
        Ok(m.devices().await.iter().map(device_to_dto).collect())
    }

    async fn list_active_streams(&self) -> zbus::fdo::Result<Vec<ActiveStreamDto>> {
        // `StreamManager` doesn't expose the active-stream map yet;
        // we keep this method around so the wire contract is stable
        // and add the backing accessor in a follow-up. For now,
        // return empty — clients should rely on StreamStarted /
        // StreamStopped signals to track liveness.
        let _ = self.manager.lock().await;
        Ok(Vec::new())
    }

    async fn start_stream(&self, device_id: String, source: SourceDto) -> zbus::fdo::Result<()> {
        let id = self.resolve(&device_id).await?;

        // Empty source = "user wants help picking". Pop the
        // in-process picker via the Freya app and use its reply.
        // Without picker support (no Freya runtime) the call fails
        // explicitly so the caller knows it has to supply a source
        // itself instead of waiting on a window that won't open.
        let source = if needs_picker(&source) {
            match self.picker_req_tx.as_ref() {
                Some(tx) => self.run_picker(tx.clone(), id).await?,
                None => {
                    return Err(zbus::fdo::Error::Failed(
                        "StartStream needs a concrete source: no Freya runtime is up to host \
                         the picker (start the daemon with `ferricast-gui` or \
                         `ferricast-gui --background`, then call StartStream with \
                         SourceDto::monitor(...) or window_by_id(...))"
                            .into(),
                    ));
                }
            }
        } else {
            source
        };

        // Resolve the source *before* taking the manager lock: id
        // lookup hits the enumerator and we don't want capture-side
        // contention to back up behind a slow enumerator call.
        let cap_source = resolve_source(self.enumerator.as_ref(), &source).await?;

        // Wayland honesty gap: the only capture path today is the
        // xdg-desktop-portal flow (PipeWire via ashpd), and the
        // portal owns source selection — it always pops its own
        // picker dialog, regardless of what id the daemon resolved.
        // Warn the caller once so the second dialog doesn't look
        // like a bug. Direct wayland capture-by-id (wlr-screencopy
        // / ext-image-copy-capture as a streaming source) is a
        // follow-up.
        let on_wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
        let picker_id = matches!(source.kind.as_str(), "monitor" | "window");
        if on_wayland && picker_id {
            tracing::warn!(
                kind = %source.kind,
                "Wayland capture today goes through xdg-desktop-portal — \
                 the portal will ignore the picker-issued id and show its own \
                 selection dialog. Direct id capture is a follow-up."
            );
        }

        let m = self.manager.lock().await;
        let capture = NativeCapture::new();
        let encoder = H264Encoder::default();
        let config = StreamConfig::default();
        m.start_stream(id, cap_source, capture, encoder, config)
            .await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        Ok(())
    }

    async fn stop_stream(&self, device_id: String) -> zbus::fdo::Result<()> {
        let id = self.resolve(&device_id).await?;
        let m = self.manager.lock().await;
        m.stop_stream(id)
            .await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        Ok(())
    }

    #[zbus(property)]
    async fn protocols(&self) -> Vec<String> {
        let m = self.manager.lock().await;
        m.registered_protocols()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    async fn list_monitors(&self) -> zbus::fdo::Result<Vec<MonitorInfoDto>> {
        match self.enumerator.list_monitors().await {
            Ok(mons) => Ok(mons.into_iter().map(monitor_to_dto).collect()),
            Err(ferricast::SourceError::Unsupported(_)) => {
                Err(zbus::fdo::Error::NotSupported(format!(
                    "backend `{}` cannot enumerate monitors — use the OS portal picker",
                    self.enumerator.backend_name()
                )))
            }
            Err(e) => Err(zbus::fdo::Error::Failed(e.to_string())),
        }
    }

    async fn list_windows(&self) -> zbus::fdo::Result<Vec<WindowInfoDto>> {
        match self.enumerator.list_windows().await {
            Ok(ws) => Ok(ws.into_iter().map(window_to_dto).collect()),
            Err(ferricast::SourceError::Unsupported(_)) => {
                Err(zbus::fdo::Error::NotSupported(format!(
                    "backend `{}` cannot enumerate windows — use the OS portal picker",
                    self.enumerator.backend_name()
                )))
            }
            Err(e) => Err(zbus::fdo::Error::Failed(e.to_string())),
        }
    }

    #[zbus(property)]
    async fn enumeration_capabilities(&self) -> Vec<String> {
        self.enumerator
            .capabilities()
            .into_iter()
            .map(|c| c.as_str().to_string())
            .collect()
    }

    /// Reports which kinds of sources the *capture* path can actually
    /// stream end-to-end without falling back to the
    /// xdg-desktop-portal picker. Used by the in-app picker to grey
    /// out tabs whose selection wouldn't actually be honoured.
    ///
    /// Resolution rules (no probe — derived from session type):
    ///
    /// - `XDG_SESSION_TYPE=x11`     → `["monitor", "window"]`
    ///   (X11Capture supports both via XID + XRandR rect)
    /// - `WAYLAND_DISPLAY` set      → `["monitor"]` only.
    ///   WaylandDirect can stream a chosen monitor via wlr-screencopy
    ///   but doesn't capture toplevels yet — needs
    ///   `ext_foreign_toplevel_image_capture_source_manager_v1` which
    ///   no major compositor exposes today.
    /// - Otherwise (no session info) → `[]`.
    ///
    /// Once a Wayland compositor adds the toplevel image-capture
    /// protocol, this method gains a real probe (try-bind from the
    /// auto_capture path) and returns `"window"` accordingly without
    /// any picker change needed on the client side.
    #[zbus(property)]
    async fn capture_capabilities(&self) -> Vec<String> {
        let on_x11 = std::env::var_os("DISPLAY").is_some()
            && std::env::var_os("WAYLAND_DISPLAY").is_none();
        let on_wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
        match (on_x11, on_wayland) {
            (true, _) => vec!["monitor".into(), "window".into()],
            (_, true) => vec!["monitor".into()],
            _ => Vec::new(),
        }
    }

    /// Capture a one-shot PNG preview of the monitor `id`. See the
    /// proxy doc for the empty-vs-NotSupported convention.
    async fn get_monitor_thumbnail(
        &self,
        id: String,
        max_width: u32,
        max_height: u32,
    ) -> zbus::fdo::Result<Vec<u8>> {
        // Belt + suspenders: cap the box. A picker that requests a
        // 16k×16k thumbnail would force a few hundred MB allocation
        // for nothing. 4k is plenty for any UI surface.
        let (w, h) = clamp_thumbnail_box(max_width, max_height);
        match self.enumerator.monitor_thumbnail(&id, w, h).await {
            Ok(bytes) => Ok(bytes),
            Err(ferricast::SourceError::Unsupported(_)) => Err(zbus::fdo::Error::NotSupported(
                format!(
                    "backend `{}` has no thumbnail capability",
                    self.enumerator.backend_name()
                ),
            )),
            Err(ferricast::SourceError::NotFound(_)) => Err(zbus::fdo::Error::InvalidArgs(
                format!("no monitor with id {id:?}"),
            )),
            // Treat backend errors as "no preview available right
            // now" — return an empty array so the picker shows a
            // placeholder instead of failing the whole listing.
            // The reason still goes into the daemon log for debug.
            Err(e) => {
                tracing::warn!(%e, id, "monitor thumbnail failed; returning empty");
                Ok(Vec::new())
            }
        }
    }

    async fn get_window_thumbnail(
        &self,
        id: String,
        max_width: u32,
        max_height: u32,
    ) -> zbus::fdo::Result<Vec<u8>> {
        let (w, h) = clamp_thumbnail_box(max_width, max_height);
        match self.enumerator.window_thumbnail(&id, w, h).await {
            Ok(bytes) => Ok(bytes),
            Err(ferricast::SourceError::Unsupported(_)) => Err(zbus::fdo::Error::NotSupported(
                format!(
                    "backend `{}` has no thumbnail capability",
                    self.enumerator.backend_name()
                ),
            )),
            Err(ferricast::SourceError::NotFound(_)) => Err(zbus::fdo::Error::InvalidArgs(
                format!("no window with id {id:?}"),
            )),
            Err(e) => {
                tracing::warn!(%e, id, "window thumbnail failed; returning empty");
                Ok(Vec::new())
            }
        }
    }

    #[zbus(signal)]
    async fn device_added(emitter: &SignalEmitter<'_>, device: DeviceDto) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn device_removed(emitter: &SignalEmitter<'_>, device_id: String) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn stream_started(
        emitter: &SignalEmitter<'_>,
        device_id: String,
        device_name: String,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn stream_stopped(emitter: &SignalEmitter<'_>, device_id: String) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn stream_reconnecting(
        emitter: &SignalEmitter<'_>,
        device_id: String,
        attempt: u32,
        reason: String,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn stream_error(
        emitter: &SignalEmitter<'_>,
        device_id: String,
        message: String,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn discovery_error(
        emitter: &SignalEmitter<'_>,
        protocol: String,
        message: String,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn monitors_changed(emitter: &SignalEmitter<'_>) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn windows_changed(emitter: &SignalEmitter<'_>) -> zbus::Result<()>;
}

/// Acquire the well-known name and export the manager service.
///
/// Returns the live `Connection` — drop it (or let the process exit)
/// to release the bus name. The caller is responsible for keeping it
/// alive for the lifetime of the daemon.
///
/// `event_rx` is the receiver returned by
/// [`StreamManagerBuilder::build_with_events`] (or
/// [`StreamManager::take_event_rx`]); the daemon spawns a task that
/// drains it and converts each [`ManagerEvent`] into the matching
/// D-Bus signal, optionally also forwarding it to `forward_tx` for
/// in-process consumers like the GUI.
pub async fn start(
    manager: Arc<Mutex<StreamManager>>,
    enumerator: Arc<dyn SourceEnumerator>,
    event_rx: mpsc::Receiver<ManagerEvent>,
    forward_tx: Option<mpsc::Sender<ManagerEvent>>,
    picker_req_tx: Option<tokio::sync::mpsc::Sender<PickerRequest>>,
) -> zbus::Result<zbus::Connection> {
    // Subscribe BEFORE serve_at so we can't lose initial events.
    // Re-subscribing inside the spawned task would race the
    // first burst of monitors_dirty / windows_dirty from the
    // enumerator's startup roundtrip.
    let source_rx = enumerator.subscribe();

    let conn = zbus::connection::Builder::session()?
        .name(BUS_NAME)?
        .serve_at(
            OBJECT_PATH,
            ManagerService {
                manager,
                enumerator,
                picker_req_tx,
            },
        )?
        .build()
        .await?;

    let signal_conn = conn.clone();
    tokio::spawn(async move {
        signal_loop(signal_conn, event_rx, forward_tx).await;
    });

    let source_conn = conn.clone();
    tokio::spawn(async move {
        source_signal_loop(source_conn, source_rx).await;
    });

    Ok(conn)
}

async fn source_signal_loop(
    conn: zbus::Connection,
    mut source_rx: tokio::sync::broadcast::Receiver<SourceChange>,
) {
    let iface_ref = match conn
        .object_server()
        .interface::<_, ManagerService>(OBJECT_PATH)
        .await
    {
        Ok(i) => i,
        Err(e) => {
            tracing::error!(%e, "could not look up Manager interface for source signal emission");
            return;
        }
    };

    loop {
        match source_rx.recv().await {
            Ok(SourceChange::Monitors) => {
                if let Err(e) = ManagerService::monitors_changed(iface_ref.signal_emitter()).await {
                    tracing::warn!(%e, "failed to emit MonitorsChanged");
                }
            }
            Ok(SourceChange::Windows) => {
                if let Err(e) = ManagerService::windows_changed(iface_ref.signal_emitter()).await {
                    tracing::warn!(%e, "failed to emit WindowsChanged");
                }
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                // We've fallen behind the enumerator. Coalesce: emit
                // both signals so any subscriber re-syncs, then keep
                // going. Same recovery the enumerator's own clients
                // are expected to do.
                tracing::debug!(skipped = n, "source change channel lagged; re-syncing");
                let _ = ManagerService::monitors_changed(iface_ref.signal_emitter()).await;
                let _ = ManagerService::windows_changed(iface_ref.signal_emitter()).await;
            }
            Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                tracing::info!("source enumerator dropped its sender; signal loop exiting");
                return;
            }
        }
    }
}

async fn signal_loop(
    conn: zbus::Connection,
    mut event_rx: mpsc::Receiver<ManagerEvent>,
    forward_tx: Option<mpsc::Sender<ManagerEvent>>,
) {
    // The `InterfaceRef` is the idiomatic way to emit signals from
    // outside an interface method: it lets us hand zbus the same
    // `SignalEmitter` it would build internally, with lifetimes that
    // are guaranteed to outlive the spawned task.
    let iface_ref = match conn
        .object_server()
        .interface::<_, ManagerService>(OBJECT_PATH)
        .await
    {
        Ok(i) => i,
        Err(e) => {
            tracing::error!(%e, "could not look up Manager interface for signal emission");
            return;
        }
    };

    while let Some(ev) = event_rx.recv().await {
        // Forward to the in-process listener first so the window
        // updates in lockstep with the bus — otherwise a UI watcher
        // reading the bus would race the local one and we'd lose
        // ordering guarantees inside the same process.
        if let Some(tx) = &forward_tx {
            let _ = tx.send(ev.clone()).await;
        }
        let emitter = iface_ref.signal_emitter();
        let r = match ev {
            ManagerEvent::DeviceFound(d) => {
                ManagerService::device_added(emitter, device_to_dto(&d)).await
            }
            ManagerEvent::DeviceLost(id) => {
                ManagerService::device_removed(emitter, id.to_string()).await
            }
            ManagerEvent::StreamStarted {
                device_id,
                device_name,
            } => ManagerService::stream_started(emitter, device_id.to_string(), device_name).await,
            ManagerEvent::StreamStopped { device_id } => {
                ManagerService::stream_stopped(emitter, device_id.to_string()).await
            }
            ManagerEvent::StreamReconnecting {
                device_id,
                attempt,
                reason,
            } => {
                ManagerService::stream_reconnecting(emitter, device_id.to_string(), attempt, reason)
                    .await
            }
            ManagerEvent::StreamError { device_id, message } => {
                ManagerService::stream_error(emitter, device_id.to_string(), message).await
            }
            ManagerEvent::DiscoveryError { protocol, message } => {
                ManagerService::discovery_error(emitter, protocol.to_string(), message).await
            }
        };
        if let Err(e) = r {
            tracing::warn!(%e, "failed to emit D-Bus signal");
        }
    }
}
