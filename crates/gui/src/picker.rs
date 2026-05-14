//! Out-of-process source picker — runs in its own OS-level window.
//!
//! Why a separate window (vs an in-app `Popup` overlay): the user
//! wants the picker reachable even when the main Ferricast window is
//! minimised / backgrounded. A `Popup` is just a child element of
//! the host window — when that window isn't on screen, the popup
//! isn't either. A standalone window has its own surface and the
//! compositor / window manager treats it independently.
//!
//! Stay-on-top: we request `WindowLevel::AlwaysOnTop` via winit's
//! `WindowAttributes`. On X11 / Windows / macOS that's honoured;
//! on Wayland the protocol doesn't expose "always on top" to apps,
//! so it's a best-effort soft-request that the compositor may
//! ignore. The window is still independent (top-level), reachable
//! through Alt-Tab and the compositor's task switcher.
//!
//! ## Wiring
//!
//! [`open_picker`] is the entry point used by [`crate::app`]:
//!
//! 1. Allocates a `tokio::sync::oneshot::channel`.
//! 2. Builds a [`PickerWindowApp`] holding the sender, then calls
//!    `Platform::launch_window` to spawn the window asynchronously.
//! 3. Awaits the user's selection on the receiver.
//! 4. Closes the window via `Platform::close_window` and dispatches
//!    the chosen [`SourceDto`] (or no-op on cancel).
//!
//! The picker UI itself is the [`SourcePicker`] component — same
//! component the prior in-window popup used. Now it lives at the
//! root of [`PickerWindowApp::render`] instead of inside a `Popup`.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use bytes::Bytes;
use ferricast::WindowIdentifier;
use ferricast::prelude::*;
use ferricast_dbus::{MonitorInfoDto, SourceDto, WindowInfoDto};
use freya::prelude::*;
use freya::winit::window::WindowLevel;
use tokio::sync::{Mutex, oneshot};
use tracing::warn;
use uuid::Uuid;

use crate::client;

// ── Entry point used by the main app ──────────────────────────────

/// Launch the picker in a new top-level window and start the
/// chosen stream when the user selects something.
///
/// Returns immediately — the actual window lifecycle runs on a
/// freya-local task spawned here. Multiple concurrent calls open
/// independent windows; cancelling one doesn't affect the others.
pub fn open_picker(platform: Platform, stream_manager: Arc<Mutex<StreamManager>>, device_id: Uuid) {
    spawn(async move {
        let (tx, rx) = oneshot::channel::<Option<SourceDto>>();
        let app = PickerWindowApp {
            sender: Rc::new(RefCell::new(Some(tx))),
        };

        let config = WindowConfig::new_app(app)
            .with_title("Choose what to share")
            .with_size(820., 580.)
            .with_background((20, 20, 28))
            // Always-on-top: honoured on X11 / Windows / macOS. On
            // Wayland the protocol doesn't expose this knob; the
            // compositor will ignore the request and the window
            // will behave like any other top-level (still reachable
            // via Alt-Tab, just not pinned above).
            .with_window_attributes(|attrs, _| attrs.with_window_level(WindowLevel::AlwaysOnTop));

        let wid = platform.launch_window(config).await;
        // Focus on creation so the user sees it on top even when
        // AlwaysOnTop is a no-op. Wayland still respects activation
        // when it comes from a user gesture (the share-button click
        // that triggered this).
        platform.focus_window(Some(wid));

        match rx.await {
            Ok(Some(dto)) => {
                platform.close_window(wid);
                if let Err(e) = start_stream_with_dto(stream_manager, device_id, dto).await {
                    tracing::error!(%e, ?device_id, "start_stream");
                }
            }
            // Cancel button, escape, or window dismissed: close
            // the window if it's still alive (it might already be
            // gone if the user clicked the close button).
            Ok(None) | Err(_) => platform.close_window(wid),
        }
    });
}

/// Translate the wire-shape `SourceDto` chosen in the picker into
/// the in-process `CaptureSource`. Mirrors what
/// `daemon::resolve_source` does over D-Bus but stays in-process
/// since we already hold the `StreamManager`.
async fn start_stream_with_dto(
    stream_manager: Arc<Mutex<StreamManager>>,
    device_id: Uuid,
    source: SourceDto,
) -> Result<()> {
    let cap_source = dto_to_capture_source(&source);
    let sm = stream_manager.lock().await;
    let capture = NativeCapture::new();
    let encoder = H264Encoder::default();
    let config = StreamConfig::default();
    sm.start_stream(device_id, cap_source, capture, encoder, config)
        .await
}

fn dto_to_capture_source(s: &SourceDto) -> CaptureSource {
    fn str_arg(s: &SourceDto, key: &str) -> Option<String> {
        let v = s.args.get(key)?;
        v.downcast_ref::<&str>().ok().map(|s| s.to_string())
    }
    match s.kind.as_str() {
        "monitor" | "screen" => CaptureSource::FullScreen {
            monitor: str_arg(s, "id").or_else(|| str_arg(s, "monitor")),
        },
        "window" => {
            let id = str_arg(s, "id");
            if let Some(id) = id {
                let identifier = id
                    .parse::<u64>()
                    .map(WindowIdentifier::Id)
                    .unwrap_or_else(|_| WindowIdentifier::Title(id));
                return CaptureSource::Window {
                    identifier: Some(identifier),
                };
            }
            CaptureSource::Window {
                identifier: str_arg(s, "title").map(WindowIdentifier::Title),
            }
        }
        _ => CaptureSource::FullScreen { monitor: None },
    }
}

// ── PickerWindowApp ───────────────────────────────────────────────

/// The freya `App` that runs inside the picker window. Holds a
/// one-shot sender that the picker's select/cancel callbacks fire
/// to communicate the user's choice back to [`open_picker`].
pub struct PickerWindowApp {
    /// `RefCell<Option<_>>` because `oneshot::Sender::send` takes
    /// `self`, but `App::render` only sees `&self`. Wrapped in
    /// `Rc` so both `on_select` and `on_cancel` can hold a clone;
    /// whichever fires first `.take()`s the sender and the other
    /// becomes a no-op.
    sender: Rc<RefCell<Option<oneshot::Sender<Option<SourceDto>>>>>,
}

impl App for PickerWindowApp {
    fn render(&self) -> impl IntoElement {
        let sender_select = self.sender.clone();
        let sender_cancel = self.sender.clone();

        let on_select: Arc<dyn Fn(SourceDto)> = Arc::new(move |dto: SourceDto| {
            if let Some(tx) = sender_select.borrow_mut().take() {
                let _ = tx.send(Some(dto));
            }
        });
        let on_cancel: Arc<dyn Fn()> = Arc::new(move || {
            if let Some(tx) = sender_cancel.borrow_mut().take() {
                let _ = tx.send(None);
            }
        });

        SourcePicker {
            on_select,
            on_cancel,
        }
    }
}

// ── SourcePicker (UI body, no Popup wrapper) ──────────────────────

/// Picker UI. Used as the root of the standalone picker window;
/// also reusable from any other surface that wants the same
/// "choose what to share" flow (just instantiate it with the
/// appropriate callbacks).
#[derive(Clone)]
pub struct SourcePicker {
    pub on_select: Arc<dyn Fn(SourceDto)>,
    pub on_cancel: Arc<dyn Fn()>,
}

impl PartialEq for SourcePicker {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.on_select, &other.on_select)
            && Arc::ptr_eq(&self.on_cancel, &other.on_cancel)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum Tab {
    #[default]
    Monitors,
    Windows,
}

#[derive(Default, Clone)]
struct Entries {
    monitors: Vec<MonitorInfoDto>,
    windows: Vec<WindowInfoDto>,
    loaded: bool,
    /// `Some` when the daemon refused enumeration with a hard
    /// error. We render the message instead of an empty grid.
    error: Option<String>,
}

impl Component for SourcePicker {
    fn render(&self) -> impl IntoElement {
        let on_select = self.on_select.clone();
        let on_cancel = self.on_cancel.clone();

        let tab = use_state::<Tab>(Default::default);
        let entries = use_state::<Entries>(Default::default);

        // One-shot fetch when the picker mounts.
        use_side_effect_with_deps(&(), {
            let entries = entries;
            move |_: &()| {
                let mut entries = entries;
                spawn(async move {
                    let loaded = load_entries().await;
                    entries.set(loaded);
                });
            }
        });

        let snapshot = entries.read().clone();
        let current_tab = *tab.read();

        rect()
            .expanded()
            .vertical()
            .background((20, 20, 28))
            .padding(16.)
            .spacing(12.)
            .content(Content::Flex)
            .child(header(on_cancel.clone()))
            .child(tabs_row(current_tab, tab.clone()))
            .child(grid(current_tab, snapshot, on_select.clone()))
            .child(footer(on_cancel))
    }
}

fn header(on_cancel: Arc<dyn Fn()>) -> Rect {
    rect()
        .width(Size::fill())
        .horizontal()
        .cross_align(Alignment::center())
        .content(Content::Flex)
        .child(
            label()
                .text("Choose what to share")
                .font_size(16.)
                .color((230, 230, 240)),
        )
        .child(rect().width(Size::flex(1.)))
        .child(
            rect()
                .padding(Gaps::new(4., 10., 4., 10.))
                .corner_radius(6.)
                .background((50, 50, 65))
                .on_press(move |_| (on_cancel)())
                .child(label().text("✕").color((230, 230, 240)).font_size(13.)),
        )
}

fn footer(on_cancel: Arc<dyn Fn()>) -> Rect {
    rect()
        .width(Size::fill())
        .horizontal()
        .main_align(Alignment::end())
        .child(
            rect()
                .padding(Gaps::new(6., 14., 6., 14.))
                .corner_radius(8.)
                .background((50, 50, 65))
                .on_press(move |_| (on_cancel)())
                .child(label().text("Cancel").color((230, 230, 240)).font_size(13.)),
        )
}

// ── Tabs ──────────────────────────────────────────────────────────

fn tabs_row(current: Tab, tab_state: State<Tab>) -> Rect {
    // `State<T>` is `Copy` (it's a handle into the global signal
    // store), so we capture by-value into the closure and re-bind
    // as `mut` inside the body — Fn closures can't mutably borrow
    // a captured variable directly, but they can shadow it.
    let monitor_btn = tab_button("Monitors", current == Tab::Monitors, move |_| {
        let mut t = tab_state;
        t.set(Tab::Monitors);
    });
    let window_btn = tab_button("Windows", current == Tab::Windows, move |_| {
        let mut t = tab_state;
        t.set(Tab::Windows);
    });
    rect()
        .horizontal()
        .spacing(6.)
        .child(monitor_btn)
        .child(window_btn)
}

fn tab_button(
    text: &'static str,
    active: bool,
    on_press: impl Fn(Event<PressEventData>) + 'static,
) -> Rect {
    let bg = if active { (60, 90, 160) } else { (40, 40, 55) };
    let fg = if active {
        (245, 245, 250)
    } else {
        (200, 200, 210)
    };
    rect()
        .padding(Gaps::new(6., 14., 6., 14.))
        .corner_radius(8.)
        .background(bg)
        .on_press(on_press)
        .child(label().text(text).color(fg).font_size(13.))
}

// ── Grid ──────────────────────────────────────────────────────────

fn grid(current: Tab, entries: Entries, on_select: Arc<dyn Fn(SourceDto)>) -> Rect {
    let body = rect()
        .width(Size::fill())
        .height(Size::flex(1.))
        .background((22, 22, 30))
        .corner_radius(10.)
        .padding(8.);

    if !entries.loaded {
        return body.center().child(
            label()
                .text("Loading sources…")
                .color((150, 150, 165))
                .font_size(13.),
        );
    }
    if let Some(err) = entries.error.as_ref() {
        return body.center().child(
            label()
                .text(format!("Daemon: {err}"))
                .color((220, 130, 130))
                .font_size(13.),
        );
    }

    match current {
        Tab::Monitors => {
            if entries.monitors.is_empty() {
                return body.center().child(empty_msg(
                    "No monitors enumerable on this compositor — picker will fall back to the portal at stream time.",
                ));
            }
            let cards = entries.monitors.into_iter().map(move |m| {
                MonitorCard {
                    info: m,
                    on_select: on_select.clone(),
                }
                .into()
            });
            body.child(
                ScrollView::new()
                    .expanded()
                    .direction(Direction::Vertical)
                    .child(
                        rect()
                            .horizontal()
                            .spacing(8.)
                            .content(Content::wrap_spacing(8.))
                            .children(cards),
                    ),
            )
        }
        Tab::Windows => {
            if entries.windows.is_empty() {
                return body.center().child(empty_msg(
                    "No windows enumerable on this compositor — pick a monitor or fall back to the portal.",
                ));
            }
            let cards = entries.windows.into_iter().map(move |w| {
                WindowCard {
                    info: w,
                    on_select: on_select.clone(),
                }
                .into()
            });
            body.child(
                ScrollView::new()
                    .expanded()
                    .direction(Direction::Vertical)
                    .child(
                        rect()
                            .horizontal()
                            .spacing(8.)
                            .content(Content::wrap_spacing(8.))
                            .children(cards),
                    ),
            )
        }
    }
}

fn empty_msg(text: &str) -> Label {
    label()
        .text(text.to_string())
        .color((180, 180, 195))
        .font_size(13.)
}

// ── Cards ─────────────────────────────────────────────────────────

const THUMB_W: f32 = 220.;
const THUMB_H: f32 = 124.;

#[derive(Clone)]
struct MonitorCard {
    info: MonitorInfoDto,
    on_select: Arc<dyn Fn(SourceDto)>,
}

impl PartialEq for MonitorCard {
    fn eq(&self, other: &Self) -> bool {
        self.info.id == other.info.id
            && self.info.width == other.info.width
            && self.info.height == other.info.height
    }
}

impl Component for MonitorCard {
    fn render(&self) -> impl IntoElement {
        let id = self.info.id.clone();
        let label_text = if self.info.name.is_empty() {
            id.clone()
        } else {
            format!("{} ({})", self.info.name, id)
        };
        let geom = format!("{}×{}", self.info.width, self.info.height);
        let on_select = self.on_select.clone();
        let id_for_select = id.clone();

        card_shell(
            label_text,
            geom,
            id.clone(),
            "monitor",
            THUMB_W as u32,
            THUMB_H as u32,
            move |_| (on_select)(SourceDto::monitor(id_for_select.clone())),
        )
    }
}

#[derive(Clone)]
struct WindowCard {
    info: WindowInfoDto,
    on_select: Arc<dyn Fn(SourceDto)>,
}

impl PartialEq for WindowCard {
    fn eq(&self, other: &Self) -> bool {
        self.info.id == other.info.id && self.info.title == other.info.title
    }
}

impl Component for WindowCard {
    fn render(&self) -> impl IntoElement {
        let id = self.info.id.clone();
        let title = if self.info.title.is_empty() {
            "(untitled)".to_string()
        } else {
            self.info.title.clone()
        };
        let app_id = if self.info.app_id.is_empty() {
            "—".to_string()
        } else {
            self.info.app_id.clone()
        };
        let on_select = self.on_select.clone();
        let id_for_select = id.clone();

        card_shell(
            title,
            app_id,
            id,
            "window",
            THUMB_W as u32,
            THUMB_H as u32,
            move |_| (on_select)(SourceDto::window_by_id(id_for_select.clone())),
        )
    }
}

/// Shared visual shell for monitor + window cards. Holds the lazy
/// thumbnail fetch so the layout code stays close to the data.
///
/// The thumbnail is fed into `ImageViewer` which handles async PNG
/// decoding via Freya's asset cache — same path Freya itself uses
/// for any byte-blob image source. Re-renders that re-fetch the same
/// `(cache_key, bytes)` tuple deduplicate via the asset cache.
fn card_shell(
    title: String,
    subtitle: String,
    cache_key: String,
    kind: &'static str,
    thumb_w: u32,
    thumb_h: u32,
    on_press: impl Fn(Event<PressEventData>) + 'static,
) -> Rect {
    let thumb = use_state::<Option<Bytes>>(|| None);
    let cache_key_for_effect = cache_key.clone();
    use_side_effect_with_deps(&cache_key_for_effect, move |key: &String| {
        let mut thumb = thumb;
        let k = key.clone();
        spawn(async move {
            if let Some(bytes) = fetch_thumbnail_bytes(kind, &k, thumb_w, thumb_h).await {
                thumb.set(Some(bytes));
            }
        });
    });

    let preview: Element = match thumb.read().clone() {
        Some(bytes) => {
            // Stable id = (kind, cache_key) so the asset cache
            // dedupes when the same card re-renders.
            let source: ImageSource = ((kind, cache_key.clone()), bytes).into();
            ImageViewer::new(source)
                .width(Size::px(thumb_w as f32))
                .height(Size::px(thumb_h as f32))
                .corner_radius(6.)
                .into()
        }
        None => rect()
            .width(Size::px(thumb_w as f32))
            .height(Size::px(thumb_h as f32))
            .corner_radius(6.)
            .background((35, 35, 48))
            .center()
            .child(label().text("…").color((120, 120, 135)).font_size(18.))
            .into(),
    };

    rect()
        .min_width(Size::px(thumb_w as f32))
        .background((28, 28, 38))
        .corner_radius(10.)
        .padding(6.)
        .vertical()
        .spacing(4.)
        .border(Border::new().fill((50, 50, 65)).width(1.))
        .on_press(on_press)
        .child(preview)
        .child(
            label()
                .text(title)
                .color((230, 230, 240))
                .font_size(13.)
                .max_lines(1)
                .text_overflow(TextOverflow::Ellipsis),
        )
        .child(
            label()
                .text(subtitle)
                .color((140, 140, 160))
                .font_size(11.)
                .max_lines(1)
                .text_overflow(TextOverflow::Ellipsis),
        )
}

// ── Daemon I/O ────────────────────────────────────────────────────

async fn load_entries() -> Entries {
    let proxy = match client::proxy().await {
        Ok(p) => p,
        Err(e) => {
            return Entries {
                monitors: Vec::new(),
                windows: Vec::new(),
                loaded: true,
                error: Some(format!("can't reach daemon: {e}")),
            };
        }
    };
    let monitors = proxy.list_monitors().await.unwrap_or_else(|e| {
        warn!(%e, "list_monitors failed");
        Vec::new()
    });
    let windows = proxy.list_windows().await.unwrap_or_else(|e| {
        warn!(%e, "list_windows failed");
        Vec::new()
    });
    Entries {
        monitors,
        windows,
        loaded: true,
        error: None,
    }
}

async fn fetch_thumbnail_bytes(kind: &str, id: &str, max_w: u32, max_h: u32) -> Option<Bytes> {
    let proxy = client::proxy().await.ok()?;
    let bytes = match kind {
        "monitor" => proxy.get_monitor_thumbnail(id, max_w, max_h).await.ok()?,
        "window" => proxy.get_window_thumbnail(id, max_w, max_h).await.ok()?,
        _ => return None,
    };
    if bytes.is_empty() {
        // Daemon returned `[]` → backend can't preview this item.
        // Card stays on the placeholder.
        return None;
    }
    Some(Bytes::from(bytes))
}
