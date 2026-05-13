//! In-process source picker shown as a Freya `Popup` overlay on
//! the main window.
//!
//! When a `DeviceCard` requests a stream, the app pushes a
//! [`PickerRequest`] into shared state and the top-level `App`
//! mounts a `SourcePicker` over the device list. The picker:
//!
//! 1. Calls `ListMonitors` + `ListWindows` against the running
//!    daemon (via the same `client::proxy()` helper the CLI uses).
//! 2. Renders a tabbed grid: Monitors / Windows. Each entry is a
//!    card with a lazy thumbnail (PNG bytes from the daemon's
//!    `GetMonitorThumbnail` / `GetWindowThumbnail`).
//! 3. On click, builds a [`SourceDto`] (`monitor` + `id` /
//!    `window_by_id`) and fires the `on_select` callback. The app
//!    clears the picker request and starts the stream.
//! 4. Escape / backdrop tap → `on_cancel`.
//!
//! Same component will be reusable from any future "pick a source"
//! flow inside the app.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use bytes::Bytes;
use ferricast_dbus::{MonitorInfoDto, SourceDto, WindowInfoDto};
use freya::prelude::*;
use freya_engine::prelude::{SkData, SkImage};
use tracing::warn;

use crate::client;

// ── Public component ──────────────────────────────────────────────

/// Callbacks from the picker run synchronously on the Freya UI
/// thread (button press / Escape key), so they don't need
/// `Send + Sync` — keeping them as plain `Fn` lets the caller
/// capture non-Send handles (Radio writers, Rc<...>, etc.) without
/// jumping through extra channels.
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

#[derive(Default, Clone, PartialEq)]
struct Entries {
    monitors: Vec<MonitorInfoDto>,
    windows: Vec<WindowInfoDto>,
    loaded: bool,
    /// `Err` when the daemon refused either list with NotSupported
    /// (e.g. GNOME / Mutter where the picker can't enumerate).
    /// We render the message instead of an empty grid.
    error: Option<String>,
}

impl Component for SourcePicker {
    fn render(&self) -> impl IntoElement {
        let on_select = self.on_select.clone();
        let on_cancel_top = self.on_cancel.clone();

        let mut tab = use_state::<Tab>(Default::default);
        let mut entries = use_state::<Entries>(Default::default);

        // One-shot fetch on mount.
        use_side_effect_with_deps((), {
            let entries = entries.clone();
            move |_| {
                let mut entries = entries.clone();
                spawn(async move {
                    let loaded = load_entries().await;
                    entries.set(loaded);
                });
            }
        });

        let snapshot = entries.read().clone();
        let current_tab = *tab.read();

        Popup::new()
            .width(Size::px(720.))
            .on_close_request({
                let cancel = on_cancel_top.clone();
                move |_| (cancel)()
            })
            .child(PopupTitle::new("Choose what to share".to_string()))
            .child(
                PopupContent::new().child(
                    rect()
                        .vertical()
                        .spacing(12.)
                        .child(tabs_row(current_tab, tab.clone()))
                        .child(grid(
                            current_tab,
                            snapshot,
                            on_select.clone(),
                            on_cancel_top.clone(),
                        )),
                ),
            )
            .child(
                PopupButtons::new().child(
                    Button::new()
                        .child("Cancel")
                        .on_press(move |_| (on_cancel_top)()),
                ),
            )
    }
}

// ── Tabs row ──────────────────────────────────────────────────────

fn tabs_row(current: Tab, mut tab_state: State<Tab>) -> Rect {
    let monitor_btn = tab_button("Monitors", current == Tab::Monitors, {
        let mut t = tab_state.clone();
        move |_| t.set(Tab::Monitors)
    });
    let window_btn = tab_button("Windows", current == Tab::Windows, {
        let mut t = tab_state.clone();
        move |_| t.set(Tab::Windows)
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
    let fg = if active { (245, 245, 250) } else { (200, 200, 210) };
    rect()
        .padding(Gaps::new(6., 14., 6., 14.))
        .corner_radius(8.)
        .background(bg)
        .on_press(on_press)
        .child(label().text(text).color(fg).font_size(13.))
}

// ── Grid ──────────────────────────────────────────────────────────

fn grid(
    current: Tab,
    entries: Entries,
    on_select: Arc<dyn Fn(SourceDto) + Send + Sync>,
    on_cancel: Arc<dyn Fn() + Send + Sync>,
) -> Rect {
    let body = rect()
        .width(Size::fill())
        .height(Size::px(420.))
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
                    "No monitors enumerable on this compositor — falling back to portal at stream time.",
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
                    "No windows enumerable on this compositor — fall back to portal or X11 source.",
                ));
            }
            let _cancel = on_cancel; // reserved for future per-card actions
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
    on_select: Arc<dyn Fn(SourceDto) + Send + Sync>,
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
        let id_for_thumb = id.clone();

        card_shell(label_text, geom, id, "monitor", THUMB_W as u32, THUMB_H as u32, move |_| {
            (on_select)(SourceDto::monitor(id_for_select.clone()))
        }, id_for_thumb)
    }
}

#[derive(Clone)]
struct WindowCard {
    info: WindowInfoDto,
    on_select: Arc<dyn Fn(SourceDto) + Send + Sync>,
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
        let id_for_thumb = id.clone();

        card_shell(title, app_id, id, "window", THUMB_W as u32, THUMB_H as u32, move |_| {
            (on_select)(SourceDto::window_by_id(id_for_select.clone()))
        }, id_for_thumb)
    }
}

/// Shared visual shell for monitor + window cards. Holds the lazy
/// thumbnail fetch so the layout code stays close to the data.
fn card_shell(
    title: String,
    subtitle: String,
    cache_key: String,
    kind: &'static str,
    thumb_w: u32,
    thumb_h: u32,
    on_press: impl Fn(Event<PressEventData>) + 'static,
    thumb_id: String,
) -> Rect {
    let mut thumb = use_state::<Option<ImageHolder>>(|| None);
    use_side_effect_with_deps(cache_key.clone(), {
        let thumb = thumb.clone();
        move |key| {
            let mut thumb = thumb.clone();
            let k = key.clone();
            spawn(async move {
                if let Some(holder) = fetch_thumbnail(kind, &k, thumb_w, thumb_h).await {
                    thumb.set(Some(holder));
                }
            });
        }
    });

    let _ = thumb_id; // documented above; debug-only

    let preview = match thumb.read().clone() {
        Some(h) => image(h)
            .width(Size::px(thumb_w as f32))
            .height(Size::px(thumb_h as f32))
            .corner_radius(6.)
            .into_element(),
        None => rect()
            .width(Size::px(thumb_w as f32))
            .height(Size::px(thumb_h as f32))
            .corner_radius(6.)
            .background((35, 35, 48))
            .center()
            .child(
                label()
                    .text("…")
                    .color((120, 120, 135))
                    .font_size(18.),
            )
            .into_element(),
    };

    rect()
        .width(Size::px(thumb_w as f32))
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

async fn fetch_thumbnail(kind: &str, id: &str, max_w: u32, max_h: u32) -> Option<ImageHolder> {
    let proxy = client::proxy().await.ok()?;
    let bytes = match kind {
        "monitor" => proxy.get_monitor_thumbnail(id, max_w, max_h).await.ok()?,
        "window" => proxy.get_window_thumbnail(id, max_w, max_h).await.ok()?,
        _ => return None,
    };
    if bytes.is_empty() {
        // Daemon returned `[]` → backend can't preview this item.
        // Card stays on the placeholder shimmer.
        return None;
    }
    decode_png(Bytes::from(bytes))
}

fn decode_png(bytes: Bytes) -> Option<ImageHolder> {
    // SAFETY: `SkData::new_bytes` borrows the slice for the
    // lifetime of the returned `SkData`. We hand the SkData into
    // `from_encoded`, which copies the decoded image into a new
    // `SkImage` — the SkData (and its borrow) drops at the end of
    // this expression. The Bytes itself we keep around in the
    // ImageHolder so the asset cache can dedupe by content.
    let image = unsafe { SkImage::from_encoded(SkData::new_bytes(&bytes)) }?;
    Some(ImageHolder {
        bytes,
        image: Rc::new(RefCell::new(image)),
    })
}
