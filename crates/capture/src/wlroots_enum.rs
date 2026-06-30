//! Source enumeration on **Wayland**. Supports any compositor that
//! exposes at least one of two foreign-toplevel protocols, which in
//! practice covers everything except GNOME / Mutter:
//!
//! - `zwlr_foreign_toplevel_management_v1` (wlroots family: Hyprland,
//!   sway, river, Wayfire, labwc, …). Richer — has `output_enter` /
//!   `output_leave` / window state.
//! - `ext_foreign_toplevel_list_v1` (niri, KDE Plasma 6, future
//!   wlroots). Standard upstream protocol, intentionally minimal:
//!   title / app_id / a globally-stable identifier and nothing else.
//!
//! `try_new` tries them in that order — wlr first because the extra
//! metadata is genuinely useful for pickers — and falls back to the
//! ext variant when the wlr binding fails. Monitor geometry uses
//! `zxdg_output_manager_v1` either way.
//!
//! On compositors that expose neither (today: GNOME / Mutter),
//! [`WaylandSourceEnumerator::try_new`] reports an error and the
//! upper layer falls back to the stub. That's the intended detection
//! mechanism for "use the OS portal picker instead".
//!
//! Threading model: a dedicated `std::thread` owns the wayland
//! `EventQueue` and a `Mutex<Snapshot>` of the current world. The
//! async trait methods clone the snapshot under the lock — fast,
//! no IO, no blocking. Change events go out via `tokio::sync::broadcast`.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ferricast_core::{
    EnumerationCapability, Geometry, MonitorInfo, SourceChange, SourceEnumerator, SourceError,
    WindowInfo,
};
use tokio::sync::broadcast;
use tracing::{debug, warn};

use wayland_client::{
    Connection, Dispatch, Proxy, QueueHandle,
    globals::{GlobalListContents, registry_queue_init},
    protocol::{
        wl_output::{self, WlOutput},
        wl_registry::WlRegistry,
    },
};
use wayland_protocols::ext::foreign_toplevel_list::v1::client::{
    ext_foreign_toplevel_handle_v1::{self, ExtForeignToplevelHandleV1},
    ext_foreign_toplevel_list_v1::{self, ExtForeignToplevelListV1},
};
use wayland_protocols::xdg::xdg_output::zv1::client::{
    zxdg_output_manager_v1::ZxdgOutputManagerV1,
    zxdg_output_v1::{self, ZxdgOutputV1},
};
use wayland_protocols_wlr::foreign_toplevel::v1::client::{
    zwlr_foreign_toplevel_handle_v1::{self, ZwlrForeignToplevelHandleV1},
    zwlr_foreign_toplevel_manager_v1::{self, ZwlrForeignToplevelManagerV1},
};

#[derive(Default)]
struct OutputData {
    /// wl_output `name` (since v4) — when missing falls back to
    /// the xdg-output `name` event. Stable per compositor restart.
    name: Option<String>,
    description: Option<String>,
    make: Option<String>,
    model: Option<String>,
    /// Logical position from xdg_output. We deliberately ignore the
    /// wl_output `geometry` x/y (which is in raw compositor pixels)
    /// because pickers always want logical coords.
    logical_x: i32,
    logical_y: i32,
    logical_w: i32,
    logical_h: i32,
    scale: i32,
    refresh_mhz: i32,
    /// Whether we've received enough events to call this output
    /// "ready". Compositors batch initial state behind a `done`
    /// event; we wait for it before publishing.
    ready: bool,
}

#[derive(Default)]
struct ToplevelData {
    title: String,
    app_id: Option<String>,
    /// wl_output protocol_ids this toplevel is currently on. Empty
    /// on the `ext-foreign-toplevel-list-v1` backend because that
    /// protocol intentionally omits output tracking — clients are
    /// expected to compose it with `ext-foreign-toplevel-state-v1`
    /// (still draft) or query the compositor IPC.
    outputs: HashSet<u32>,
    /// Stable string id from `ext_foreign_toplevel_handle_v1.identifier`.
    /// Only populated on the ext backend; wlr toplevels fall back to
    /// the wayland protocol_id stringified. We prefer this when
    /// publishing because the ext spec guarantees it's a globally
    /// unique, compositor-stable id, while protocol_id is per-
    /// connection.
    identifier: Option<String>,
    closed: bool,
    ready: bool,
}

#[derive(Default)]
struct Snapshot {
    monitors: Vec<MonitorInfo>,
    windows: Vec<WindowInfo>,
}

struct State {
    /// wl_output object id → its data. We key on `id()` (a u32)
    /// rather than the `WlOutput` proxy itself because it's `Copy`
    /// and lets us cross-reference from toplevel `output_enter`
    /// events without juggling proxy lifetimes.
    outputs: HashMap<u32, (WlOutput, OutputData)>,
    xdg_outputs: HashMap<u32, ZxdgOutputV1>,
    /// Toplevels announced via `zwlr_foreign_toplevel_management_v1`
    /// (wlroots / Hyprland / sway / Wayfire / …). Richer than the
    /// ext variant — has output tracking + state.
    wlr_toplevels: HashMap<u32, (ZwlrForeignToplevelHandleV1, ToplevelData)>,
    /// Toplevels announced via the standard `ext-foreign-toplevel-list-v1`
    /// (niri, KDE Plasma 6, future wlroots). Less metadata but
    /// universally supported across modern compositors.
    ext_toplevels: HashMap<u32, (ExtForeignToplevelHandleV1, ToplevelData)>,

    xdg_output_manager: Option<ZxdgOutputManagerV1>,
    /// Held for their side effect: dropping the manager proxy makes
    /// the compositor stop sending `toplevel` events. Only one of
    /// the two is `Some` at any time — try_new picks wlr first
    /// (richer info) and falls back to ext.
    #[allow(dead_code)]
    foreign_manager: Option<ZwlrForeignToplevelManagerV1>,
    #[allow(dead_code)]
    ext_manager: Option<ExtForeignToplevelListV1>,

    snapshot: Arc<Mutex<Snapshot>>,
    change_tx: broadcast::Sender<SourceChange>,

    /// Bumped on every event that might affect the snapshot; the
    /// post-dispatch tick uses it to decide whether to recompute +
    /// broadcast. Avoids firing 200 `Windows` signals when the
    /// compositor floods us with the initial state.
    monitors_dirty: bool,
    windows_dirty: bool,
}

impl State {
    fn recompute(&mut self) {
        if !self.monitors_dirty && !self.windows_dirty {
            return;
        }
        let mut snap = Snapshot::default();
        for (_, (_, o)) in &self.outputs {
            if !o.ready {
                continue;
            }
            let name = o.name.clone().unwrap_or_else(|| "wl_output".to_string());
            snap.monitors.push(MonitorInfo {
                id: name.clone(),
                name: o.description.clone().unwrap_or(name),
                make: o.make.clone(),
                model: o.model.clone(),
                geometry: Geometry {
                    x: o.logical_x,
                    y: o.logical_y,
                    width: o.logical_w.max(0) as u32,
                    height: o.logical_h.max(0) as u32,
                },
                scale: o.scale.max(1) as f32,
                refresh_mhz: if o.refresh_mhz > 0 {
                    Some(o.refresh_mhz as u32)
                } else {
                    None
                },
                // wlroots has no canonical primary — leave it false
                // for every output. Clients that want a "primary"
                // can pick by position (e.g. the one at 0,0) or by
                // their own config.
                primary: false,
            });
        }
        let publish = |snap: &mut Snapshot,
                       handle_id: u32,
                       t: &ToplevelData,
                       outputs: &HashMap<u32, (WlOutput, OutputData)>| {
            if !t.ready || t.closed {
                return;
            }
            let on_monitor = t
                .outputs
                .iter()
                .next()
                .and_then(|wl_id| outputs.get(wl_id).and_then(|(_, o)| o.name.clone()));
            snap.windows.push(WindowInfo {
                id: t
                    .identifier
                    .clone()
                    .unwrap_or_else(|| handle_id.to_string()),
                title: t.title.clone(),
                app_id: t.app_id.clone(),
                pid: None,
                geometry: None,
                on_monitor,
            });
        };
        for (id, (_, t)) in &self.wlr_toplevels {
            publish(&mut snap, *id, t, &self.outputs);
        }
        for (id, (_, t)) in &self.ext_toplevels {
            publish(&mut snap, *id, t, &self.outputs);
        }
        *self.snapshot.lock().unwrap() = snap;

        if self.monitors_dirty {
            let _ = self.change_tx.send(SourceChange::Monitors);
        }
        if self.windows_dirty {
            let _ = self.change_tx.send(SourceChange::Windows);
        }
        self.monitors_dirty = false;
        self.windows_dirty = false;
    }
}

pub struct WaylandSourceEnumerator {
    snapshot: Arc<Mutex<Snapshot>>,
    change_tx: broadcast::Sender<SourceChange>,
}

impl WaylandSourceEnumerator {
    /// Connect to the running Wayland compositor and bind the
    /// available foreign-toplevel + xdg-output globals. Returns an
    /// error if:
    ///
    /// - `WAYLAND_DISPLAY` isn't set / can't connect, or
    /// - the compositor exposes **neither**
    ///   `zwlr_foreign_toplevel_management_v1` (wlroots family) nor
    ///   `ext_foreign_toplevel_list_v1` (niri / KDE Plasma 6 /
    ///   modern compositors). In that case the caller falls back to
    ///   the stub and the picker should defer to the OS portal.
    ///
    /// Binding precedence is wlr first because it carries strictly
    /// more information (per-output presence + window state). The
    /// ext fallback is intentionally minimal: title / app_id /
    /// stable identifier and nothing else.
    pub fn try_new() -> Result<Self, SourceError> {
        let conn = Connection::connect_to_env()
            .map_err(|e| SourceError::Backend(format!("wayland connect: {e}")))?;
        let (globals, mut event_queue) = registry_queue_init::<State>(&conn)
            .map_err(|e| SourceError::Backend(format!("registry init: {e}")))?;
        let qh = event_queue.handle();

        let foreign_manager = globals
            .bind::<ZwlrForeignToplevelManagerV1, _, _>(&qh, 1..=3, ())
            .ok();
        let ext_manager = if foreign_manager.is_none() {
            globals
                .bind::<ExtForeignToplevelListV1, _, _>(&qh, 1..=1, ())
                .ok()
        } else {
            None
        };
        if foreign_manager.is_none() && ext_manager.is_none() {
            return Err(SourceError::Backend(
                "neither zwlr_foreign_toplevel_management_v1 nor \
                 ext_foreign_toplevel_list_v1 is available — compositor \
                 is GNOME / Mutter or an older stack; use the portal picker"
                    .into(),
            ));
        }
        tracing::info!(
            wlr = foreign_manager.is_some(),
            ext = ext_manager.is_some(),
            "wayland enumerator: bound foreign-toplevel protocol"
        );
        let xdg_output_manager = globals
            .bind::<ZxdgOutputManagerV1, _, _>(&qh, 2..=3, ())
            .ok();
        if xdg_output_manager.is_none() {
            warn!(
                "wayland enumerator: xdg-output missing; monitor geometry will be 0×0. \
                 (unusually old compositor — every modern wlroots / niri / KWin ships it.)"
            );
        }

        // Bind every existing wl_output upfront so we don't miss
        // events for monitors that were present before we
        // connected. New ones come in via the registry handler.
        for g in globals.contents().clone_list() {
            if g.interface == "wl_output" {
                let output =
                    globals
                        .registry()
                        .bind::<WlOutput, _, _>(g.name, g.version.min(4), &qh, ());
                let mut data = OutputData::default();
                // Trigger xdg-output binding for this wl_output if
                // possible — populated lazily inside dispatch.
                let _ = (output, &mut data);
            }
        }

        let (change_tx, _) = broadcast::channel(16);
        let snapshot = Arc::new(Mutex::new(Snapshot::default()));

        let mut state = State {
            outputs: HashMap::new(),
            xdg_outputs: HashMap::new(),
            wlr_toplevels: HashMap::new(),
            ext_toplevels: HashMap::new(),
            xdg_output_manager,
            foreign_manager,
            ext_manager,
            snapshot: snapshot.clone(),
            change_tx: change_tx.clone(),
            monitors_dirty: false,
            windows_dirty: false,
        };

        // Pump once so the manager's `toplevel` events for the
        // pre-existing windows actually arrive before we return.
        event_queue
            .roundtrip(&mut state)
            .map_err(|e| SourceError::Backend(format!("initial roundtrip: {e}")))?;
        state.recompute();

        std::thread::Builder::new()
            .name("ferricast-wlroots-enum".into())
            .spawn(move || run_loop(conn, event_queue, state))
            .map_err(|e| SourceError::Backend(format!("spawn: {e}")))?;

        Ok(Self {
            snapshot,
            change_tx,
        })
    }
}

fn run_loop(_conn: Connection, mut queue: wayland_client::EventQueue<State>, mut state: State) {
    // wayland-client's Dispatch trait panics by default on
    // unhandled object-creating events (see the explanation around
    // `event_created_child!` above). If a panic ever escapes
    // dispatch the thread would die silently and the daemon would
    // keep advertising "wlroots" with stale snapshots — wrap each
    // tick in `catch_unwind` so we log it loudly instead.
    use std::panic::AssertUnwindSafe;

    loop {
        let result =
            std::panic::catch_unwind(AssertUnwindSafe(|| queue.blocking_dispatch(&mut state)));
        match result {
            Ok(Ok(_)) => state.recompute(),
            Ok(Err(e)) => {
                warn!(%e, "wlroots event loop exited cleanly with error");
                return;
            }
            Err(panic) => {
                let msg = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&'static str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "<non-string panic>".into());
                warn!(%msg, "wlroots dispatch panicked — exiting enumerator thread");
                return;
            }
        }
    }
}

#[async_trait]
impl SourceEnumerator for WaylandSourceEnumerator {
    fn backend_name(&self) -> &'static str {
        // Historically "wlroots"; broadened to cover any compositor
        // that exposes either foreign-toplevel protocol — niri,
        // KDE Plasma 6, etc. — so the name now reflects the
        // protocol family rather than a specific compositor stack.
        "wayland"
    }

    fn capabilities(&self) -> Vec<EnumerationCapability> {
        vec![
            EnumerationCapability::Monitors,
            EnumerationCapability::Windows,
        ]
    }

    async fn list_monitors(&self) -> Result<Vec<MonitorInfo>, SourceError> {
        Ok(self.snapshot.lock().unwrap().monitors.clone())
    }
    async fn list_windows(&self) -> Result<Vec<WindowInfo>, SourceError> {
        Ok(self.snapshot.lock().unwrap().windows.clone())
    }

    async fn monitor_thumbnail(
        &self,
        id: &str,
        max_width: u32,
        max_height: u32,
    ) -> Result<Vec<u8>, SourceError> {
        let id = id.to_owned();
        tokio::task::spawn_blocking(move || {
            crate::wayland_thumb::monitor_png(&id, max_width, max_height)
        })
        .await
        .map_err(|e| SourceError::Backend(format!("join: {e}")))?
    }

    async fn window_thumbnail(
        &self,
        id: &str,
        max_width: u32,
        max_height: u32,
    ) -> Result<Vec<u8>, SourceError> {
        let id = id.to_owned();
        tokio::task::spawn_blocking(move || {
            crate::wayland_thumb::window_png(&id, max_width, max_height)
        })
        .await
        .map_err(|e| SourceError::Backend(format!("join: {e}")))?
    }

    fn subscribe(&self) -> broadcast::Receiver<SourceChange> {
        self.change_tx.subscribe()
    }
}

// ── Dispatch implementations ──────────────────────────────────────
//
// Every protocol we touch needs a Dispatch impl. They're terse but
// numerous — wayland-client requires one per (interface, user_data)
// pair. Logic lives in the per-event match arms; State only carries
// data.

impl Dispatch<WlRegistry, GlobalListContents> for State {
    fn event(
        state: &mut Self,
        registry: &WlRegistry,
        event: <WlRegistry as wayland_client::Proxy>::Event,
        _: &GlobalListContents,
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        use wayland_client::protocol::wl_registry::Event;
        if let Event::Global {
            name,
            interface,
            version,
        } = event
        {
            if interface == "wl_output" {
                let out = registry.bind::<WlOutput, _, _>(name, version.min(4), qh, ());
                state
                    .outputs
                    .insert(out.id().protocol_id(), (out, OutputData::default()));
                state.monitors_dirty = true;
            }
        }
    }
}

impl Dispatch<WlOutput, ()> for State {
    fn event(
        state: &mut Self,
        proxy: &WlOutput,
        event: <WlOutput as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        let id = proxy.id().protocol_id();
        let entry = state
            .outputs
            .entry(id)
            .or_insert_with(|| (proxy.clone(), OutputData::default()));
        match event {
            wl_output::Event::Geometry { make, model, .. } => {
                if !make.is_empty() {
                    entry.1.make = Some(make);
                }
                if !model.is_empty() {
                    entry.1.model = Some(model);
                }
            }
            wl_output::Event::Mode { refresh, flags, .. } => {
                // Only the *current* mode counts. wlroots flags the
                // current one with `current` in the mode flags.
                if flags
                    .into_result()
                    .map(|f| f.contains(wl_output::Mode::Current))
                    .unwrap_or(false)
                {
                    entry.1.refresh_mhz = refresh;
                }
            }
            wl_output::Event::Scale { factor } => {
                entry.1.scale = factor;
            }
            wl_output::Event::Name { name } => {
                entry.1.name = Some(name);
            }
            wl_output::Event::Description { description } => {
                entry.1.description = Some(description);
            }
            wl_output::Event::Done => {
                entry.1.ready = true;
                state.monitors_dirty = true;
            }
            _ => {}
        }
        // Bind xdg-output lazily once the manager is around.
        if !state.xdg_outputs.contains_key(&id) {
            if let Some(mgr) = state.xdg_output_manager.as_ref() {
                let xo = mgr.get_xdg_output(proxy, qh, id);
                state.xdg_outputs.insert(id, xo);
            }
        }
    }
}

impl Dispatch<ZxdgOutputManagerV1, ()> for State {
    fn event(
        _: &mut Self,
        _: &ZxdgOutputManagerV1,
        _: <ZxdgOutputManagerV1 as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ZxdgOutputV1, u32> for State {
    fn event(
        state: &mut Self,
        _: &ZxdgOutputV1,
        event: <ZxdgOutputV1 as wayland_client::Proxy>::Event,
        wl_id: &u32,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let Some(entry) = state.outputs.get_mut(wl_id) else {
            return;
        };
        match event {
            zxdg_output_v1::Event::LogicalPosition { x, y } => {
                entry.1.logical_x = x;
                entry.1.logical_y = y;
            }
            zxdg_output_v1::Event::LogicalSize { width, height } => {
                entry.1.logical_w = width;
                entry.1.logical_h = height;
            }
            zxdg_output_v1::Event::Name { name } => {
                if entry.1.name.is_none() {
                    entry.1.name = Some(name);
                }
            }
            zxdg_output_v1::Event::Description { description } => {
                if entry.1.description.is_none() {
                    entry.1.description = Some(description);
                }
            }
            zxdg_output_v1::Event::Done => {
                state.monitors_dirty = true;
            }
            _ => {}
        }
    }
}

impl Dispatch<ZwlrForeignToplevelManagerV1, ()> for State {
    fn event(
        state: &mut Self,
        _: &ZwlrForeignToplevelManagerV1,
        event: <ZwlrForeignToplevelManagerV1 as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        use zwlr_foreign_toplevel_manager_v1::Event;
        match event {
            Event::Toplevel { toplevel } => {
                let id = toplevel.id().protocol_id();
                debug!(handle_id = id, "wlr: new toplevel announced");
                state
                    .wlr_toplevels
                    .insert(id, (toplevel, ToplevelData::default()));
                state.windows_dirty = true;
            }
            Event::Finished => {
                debug!("wlr foreign-toplevel manager finished; pickers should re-list");
                state.windows_dirty = true;
            }
            _ => {}
        }
    }

    // The `toplevel` event creates a new `zwlr_foreign_toplevel_handle_v1`
    // server-side. wayland-client's default `event_created_child` panics
    // — silently killing the event-loop thread — unless we tell it which
    // dispatch user-data to attach. Opcode `0` is the `toplevel` event:
    // it's the first event declared in the protocol XML, and wayland
    // assigns event opcodes in declaration order. Without this override
    // the daemon comes up, advertises the wlroots backend, but the
    // first incoming `toplevel` aborts the thread — exactly the "no
    // windows ever appear" symptom.
    wayland_client::event_created_child!(State, ZwlrForeignToplevelManagerV1, [
        0 => (ZwlrForeignToplevelHandleV1, ()),
    ]);
}

impl Dispatch<ZwlrForeignToplevelHandleV1, ()> for State {
    fn event(
        state: &mut Self,
        proxy: &ZwlrForeignToplevelHandleV1,
        event: <ZwlrForeignToplevelHandleV1 as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let id = proxy.id().protocol_id();
        let Some((_, data)) = state.wlr_toplevels.get_mut(&id) else {
            return;
        };
        use zwlr_foreign_toplevel_handle_v1::Event;
        match event {
            Event::Title { title } => {
                data.title = title;
                data.ready = true;
                state.windows_dirty = true;
            }
            Event::AppId { app_id } => {
                data.app_id = Some(app_id);
                state.windows_dirty = true;
            }
            Event::OutputEnter { output } => {
                data.outputs.insert(output.id().protocol_id());
                state.windows_dirty = true;
            }
            Event::OutputLeave { output } => {
                data.outputs.remove(&output.id().protocol_id());
                state.windows_dirty = true;
            }
            Event::Done => {
                data.ready = true;
                state.windows_dirty = true;
            }
            Event::Closed => {
                data.closed = true;
                state.wlr_toplevels.remove(&id);
                state.windows_dirty = true;
            }
            _ => {}
        }
    }
}

// ── ext-foreign-toplevel-list-v1 ──────────────────────────────────
//
// Minimal upstream alternative to the wlr protocol. Fired by niri,
// KDE Plasma 6, and (eventually) every modern compositor. Carries
// title / app_id / a stable identifier but no output presence or
// state — clients that need those have to compose this with other
// protocols (or the compositor's IPC).

impl Dispatch<ExtForeignToplevelListV1, ()> for State {
    fn event(
        state: &mut Self,
        _: &ExtForeignToplevelListV1,
        event: <ExtForeignToplevelListV1 as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        use ext_foreign_toplevel_list_v1::Event;
        match event {
            Event::Toplevel { toplevel } => {
                let id = toplevel.id().protocol_id();
                debug!(handle_id = id, "ext: new toplevel announced");
                state
                    .ext_toplevels
                    .insert(id, (toplevel, ToplevelData::default()));
                state.windows_dirty = true;
            }
            Event::Finished => {
                debug!("ext foreign-toplevel list finished; pickers should re-list");
                state.windows_dirty = true;
            }
            _ => {}
        }
    }

    // Same panic-by-default reasoning as the wlr manager — opcode 0
    // is `toplevel` per the protocol declaration order.
    wayland_client::event_created_child!(State, ExtForeignToplevelListV1, [
        0 => (ExtForeignToplevelHandleV1, ()),
    ]);
}

impl Dispatch<ExtForeignToplevelHandleV1, ()> for State {
    fn event(
        state: &mut Self,
        proxy: &ExtForeignToplevelHandleV1,
        event: <ExtForeignToplevelHandleV1 as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let id = proxy.id().protocol_id();
        let Some((_, data)) = state.ext_toplevels.get_mut(&id) else {
            return;
        };
        use ext_foreign_toplevel_handle_v1::Event;
        match event {
            Event::Title { title } => {
                data.title = title;
                state.windows_dirty = true;
            }
            Event::AppId { app_id } => {
                data.app_id = Some(app_id);
                state.windows_dirty = true;
            }
            Event::Identifier { identifier } => {
                data.identifier = Some(identifier);
                state.windows_dirty = true;
            }
            Event::Done => {
                data.ready = true;
                state.windows_dirty = true;
            }
            Event::Closed => {
                data.closed = true;
                state.ext_toplevels.remove(&id);
                state.windows_dirty = true;
            }
            _ => {}
        }
    }
}
