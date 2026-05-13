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
    ActiveStreamDto, DeviceDto, MonitorInfoDto, SourceDto, WindowInfoDto, BUS_NAME, OBJECT_PATH,
};
use tokio::sync::{mpsc, Mutex};
use uuid::Uuid;
use zbus::object_server::SignalEmitter;
use zbus::zvariant::OwnedValue;

/// Build a [`DeviceDto`] from a core [`Device`]. The `protocol_icon`
/// is dropped — sending raw bytes over D-Bus for every signal is
/// expensive and most clients want to load icons by protocol name
/// from their own asset set anyway.
fn device_to_dto(d: &Device) -> DeviceDto {
    let mut caps: HashMap<String, OwnedValue> = HashMap::new();
    if let Some(v) = d.capabilities.max_fps {
        if let Ok(v) = OwnedValue::try_from(v) {
            caps.insert("max_fps".into(), v);
        }
    }
    if let Some(v) = d.capabilities.max_bitrate_kbps {
        if let Ok(v) = OwnedValue::try_from(v) {
            caps.insert("max_bitrate_kbps".into(), v);
        }
    }
    if let Some(p) = d.capabilities.max_h264_profile {
        if let Ok(v) = OwnedValue::try_from(format!("{p:?}")) {
            caps.insert("max_h264_profile".into(), v);
        }
    }
    if let Ok(v) = OwnedValue::try_from(d.capabilities.requires_audio) {
        caps.insert("requires_audio".into(), v);
    }

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

/// Translate a wire [`SourceDto`] into a [`CaptureSource`]. Empty
/// `kind` means "let the daemon decide", which today is PipeWire's
/// full-screen path on both Wayland and X11 (the bespoke X11 picker
/// dialog is still a TODO).
fn dto_to_source(s: &SourceDto) -> CaptureSource {
    match s.kind.as_str() {
        "window" => CaptureSource::Window {
            identifier: str_arg(s, "identifier").map(ferricast::WindowIdentifier::Title),
        },
        "screen" | "" => CaptureSource::FullScreen {
            monitor: str_arg(s, "monitor"),
        },
        other => {
            tracing::warn!(kind = other, "unknown source kind, defaulting to full-screen");
            CaptureSource::FullScreen { monitor: None }
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

/// Object exported on the bus. The manager is shared with the rest
/// of the app (the Freya window holds the same `Arc`), which is why
/// it's behind a mutex. The enumerator is read-only — its own
/// internal locking handles concurrent access.
pub struct ManagerService {
    pub manager: Arc<Mutex<StreamManager>>,
    pub enumerator: Arc<dyn SourceEnumerator>,
}

impl ManagerService {
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
        let cap_source = dto_to_source(&source);
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
            Err(ferricast::SourceError::Unsupported(_)) => Err(zbus::fdo::Error::NotSupported(
                format!(
                    "backend `{}` cannot enumerate monitors — use the OS portal picker",
                    self.enumerator.backend_name()
                ),
            )),
            Err(e) => Err(zbus::fdo::Error::Failed(e.to_string())),
        }
    }

    async fn list_windows(&self) -> zbus::fdo::Result<Vec<WindowInfoDto>> {
        match self.enumerator.list_windows().await {
            Ok(ws) => Ok(ws.into_iter().map(window_to_dto).collect()),
            Err(ferricast::SourceError::Unsupported(_)) => Err(zbus::fdo::Error::NotSupported(
                format!(
                    "backend `{}` cannot enumerate windows — use the OS portal picker",
                    self.enumerator.backend_name()
                ),
            )),
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
                if let Err(e) =
                    ManagerService::monitors_changed(iface_ref.signal_emitter()).await
                {
                    tracing::warn!(%e, "failed to emit MonitorsChanged");
                }
            }
            Ok(SourceChange::Windows) => {
                if let Err(e) =
                    ManagerService::windows_changed(iface_ref.signal_emitter()).await
                {
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
                ManagerService::stream_reconnecting(
                    emitter,
                    device_id.to_string(),
                    attempt,
                    reason,
                )
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
