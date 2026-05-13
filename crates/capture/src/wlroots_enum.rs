//! Source enumeration on **wlroots-based Wayland compositors**
//! (Hyprland, sway, river, Wayfire, labwc, …).
//!
//! Uses two protocols that wlroots compositors expose but GNOME's
//! Mutter and KDE's KWin currently don't:
//!
//! - `zwlr_foreign_toplevel_management_v1` — every top-level window,
//!   with `title` / `app_id` / `output_enter` / `state` events.
//! - `zxdg_output_manager_v1` — stable output name + logical
//!   position/size (post-scale, post-transform), which is what a
//!   picker actually wants to show.
//!
//! When binding fails (`zwlr_foreign_toplevel_management_v1` absent),
//! [`WlrootsSourceEnumerator::try_new`] reports an error and the
//! upper layer falls back to the stub. That's the intended detection
//! mechanism for "this is GNOME/KDE, use the portal picker instead".
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
    globals::{registry_queue_init, GlobalListContents},
    protocol::{
        wl_output::{self, WlOutput},
        wl_registry::WlRegistry,
    },
    Connection, Dispatch, Proxy, QueueHandle,
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
    outputs: HashSet<u32>,
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
    toplevels: HashMap<u32, (ZwlrForeignToplevelHandleV1, ToplevelData)>,

    xdg_output_manager: Option<ZxdgOutputManagerV1>,
    /// Held for its side effect: dropping the manager proxy makes
    /// the compositor stop sending `toplevel` events. Not read
    /// directly anywhere.
    #[allow(dead_code)]
    foreign_manager: Option<ZwlrForeignToplevelManagerV1>,

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
        for (handle_id, (_, t)) in &self.toplevels {
            if !t.ready || t.closed {
                continue;
            }
            let on_monitor = t.outputs.iter().next().and_then(|wl_id| {
                self.outputs
                    .get(wl_id)
                    .and_then(|(_, o)| o.name.clone())
            });
            snap.windows.push(WindowInfo {
                id: handle_id.to_string(),
                title: t.title.clone(),
                app_id: t.app_id.clone(),
                pid: None,
                geometry: None,
                on_monitor,
            });
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

pub struct WlrootsSourceEnumerator {
    snapshot: Arc<Mutex<Snapshot>>,
    change_tx: broadcast::Sender<SourceChange>,
}

impl WlrootsSourceEnumerator {
    /// Connect to the running Wayland compositor and bind the
    /// foreign-toplevel + xdg-output globals. Returns an error if:
    ///
    /// - `WAYLAND_DISPLAY` isn't set / can't connect
    /// - the compositor doesn't expose
    ///   `zwlr_foreign_toplevel_management_v1` (i.e. it's GNOME or
    ///   KDE). xdg-output is optional but every modern wlroots
    ///   compositor has it.
    ///
    /// The check is the whole point of this backend: a successful
    /// `try_new` means we can enumerate; failure means the caller
    /// should drop us and use a stub.
    pub fn try_new() -> Result<Self, SourceError> {
        let conn = Connection::connect_to_env()
            .map_err(|e| SourceError::Backend(format!("wayland connect: {e}")))?;
        let (globals, mut event_queue) = registry_queue_init::<State>(&conn)
            .map_err(|e| SourceError::Backend(format!("registry init: {e}")))?;
        let qh = event_queue.handle();

        let foreign_manager = globals
            .bind::<ZwlrForeignToplevelManagerV1, _, _>(&qh, 1..=3, ())
            .map_err(|e| {
                SourceError::Backend(format!(
                    "zwlr_foreign_toplevel_management_v1 missing — not a wlroots compositor: {e}"
                ))
            })?;
        let xdg_output_manager = globals
            .bind::<ZxdgOutputManagerV1, _, _>(&qh, 2..=3, ())
            .ok();
        if xdg_output_manager.is_none() {
            warn!(
                "wlroots: xdg-output missing; monitor geometry will be 0×0. \
                 (compositor is wlroots-ish but unusually old.)"
            );
        }

        // Bind every existing wl_output upfront so we don't miss
        // events for monitors that were present before we
        // connected. New ones come in via the registry handler.
        for g in globals.contents().clone_list() {
            if g.interface == "wl_output" {
                let output = globals
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
            toplevels: HashMap::new(),
            xdg_output_manager,
            foreign_manager: Some(foreign_manager),
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
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| queue.blocking_dispatch(&mut state)));
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
impl SourceEnumerator for WlrootsSourceEnumerator {
    fn backend_name(&self) -> &'static str {
        "wlroots"
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
                state.outputs.insert(out.id().protocol_id(), (out, OutputData::default()));
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
        let entry = state.outputs.entry(id).or_insert_with(|| {
            (proxy.clone(), OutputData::default())
        });
        match event {
            wl_output::Event::Geometry {
                make, model, ..
            } => {
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
                if flags.into_result().map(|f| f.contains(wl_output::Mode::Current)).unwrap_or(false) {
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
                debug!(handle_id = id, "wlroots: new toplevel announced");
                state
                    .toplevels
                    .insert(id, (toplevel, ToplevelData::default()));
                state.windows_dirty = true;
            }
            Event::Finished => {
                debug!("foreign-toplevel manager finished; pickers should re-list");
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
        let Some((_, data)) = state.toplevels.get_mut(&id) else {
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
                state.toplevels.remove(&id);
                state.windows_dirty = true;
            }
            _ => {}
        }
    }
}
