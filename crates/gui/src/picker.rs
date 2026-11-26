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
use std::time::Duration;

use bytes::Bytes;
use ferricast::WindowIdentifier;
use ferricast::prelude::*;
use ferricast_dbus::{ManagerProxy, MonitorInfoDto, SourceDto, WindowInfoDto};
use freya::prelude::*;
use freya::winit::window::WindowLevel;
use futures_util::StreamExt;
use tokio::sync::oneshot;
use tracing::warn;
use uuid::Uuid;

use crate::client;
use crate::daemon::SharedManager;

// ── Entry point used by the main app ──────────────────────────────

/// Launch the picker in a new top-level window and start the
/// chosen stream when the user selects something.
///
/// Returns immediately — the actual window lifecycle runs on a
/// freya-local task spawned here. Multiple concurrent calls open
/// independent windows; cancelling one doesn't affect the others.
/// Launch the picker in a new top-level window, await the user's
/// selection, and return the chosen `SourceDto` via the supplied
/// `oneshot::Sender`. Returns immediately — the actual window
/// lifecycle runs on a freya-local task spawned here.
///
/// The caller decides what to do with the picked source:
///
/// - In-process flow (main app share button): see [`open_picker`]
///   which wraps this and dispatches `start_stream_with_dto`
///   directly through the local `StreamManager`.
/// - Daemon-driven flow (D-Bus `StartStream` with empty source):
///   the daemon calls this with a oneshot it awaits in its own
///   request handler, then runs `start_stream` itself.
///
/// `None` on the reply channel means "user cancelled / closed the
/// window"; `Some(dto)` is the picked source.
pub fn open_picker_for_dto(platform: Platform, reply_tx: oneshot::Sender<Option<SourceDto>>) {
    spawn(async move {
        let (tx, rx) = oneshot::channel::<Option<SourceDto>>();
        let app = PickerWindowApp {
            sender: Rc::new(RefCell::new(Some(tx))),
        };

        let config = WindowConfig::new_app(app)
            .with_title("Choose what to share")
            .with_size(522., 530.)
            .with_background((20, 20, 28))
            .with_transparency(true)
            .with_decorations(false)
            .with_app_id("rs.sergioribera.ferricast.Picker")
            .with_window_attributes(|attrs, _| {
                attrs
                    .with_window_level(WindowLevel::AlwaysOnTop)
                    .with_blur(true)
            });

        let wid = platform.launch_window(config).await;
        platform.focus_window(wid);

        let picked = match rx.await {
            Ok(Some(dto)) => Some(dto),
            Ok(None) | Err(_) => None,
        };
        platform.close_window(wid);
        // Best-effort: the caller might have lost interest (e.g.
        // dropped its receiver because of a timeout) — that's fine,
        // we just discard the dto.
        let _ = reply_tx.send(picked);
    });
}

/// Launch the picker and start the stream in-process when the user
/// confirms. Convenience wrapper used by the main app's share
/// buttons. The daemon path uses [`open_picker_for_dto`] directly
/// because it does the `start_stream` work itself.
pub fn open_picker(
    platform: Platform,
    stream_manager: SharedManager,
    device_id: Uuid,
    on_error: impl FnOnce(String) + 'static,
) {
    let (tx, rx) = oneshot::channel::<Option<SourceDto>>();
    open_picker_for_dto(platform, tx);
    spawn(async move {
        let Ok(Some(dto)) = rx.await else {
            return;
        };
        // The start runs on the Tokio runtime, not on this UI task: it
        // captures, initialises the encoder and negotiates with the
        // receiver for seconds, partly with blocking calls, and every
        // window shares the UI thread. The picker is already closed;
        // progress shows on the device card through the stream events.
        let start = tokio::spawn(start_stream_with_dto(stream_manager, device_id, dto));
        let e = match start.await {
            Ok(Ok(())) => return,
            Ok(Err(e)) => e,
            Err(join) => {
                tracing::error!(%join, ?device_id, "start_stream task failed");
                on_error(format!("Could not start: {join}"));
                return;
            }
        };
        tracing::error!(%e, ?device_id, "start_stream");
        // Failures after the start was accepted arrive as
        // StreamError events; only rejections need a notice here.
        if matches!(
            e,
            FerricastError::SessionAlreadyActive(_) | FerricastError::DeviceNotFound(_)
        ) {
            on_error(format!("Could not start: {e}"));
        }
    });
}

/// Translate the wire-shape `SourceDto` chosen in the picker into
/// the in-process `CaptureSource`. Mirrors what
/// `daemon::resolve_source` does over D-Bus but stays in-process
/// since we already hold the `StreamManager`.
async fn start_stream_with_dto(
    stream_manager: SharedManager,
    device_id: Uuid,
    source: SourceDto,
) -> Result<()> {
    let cap_source = dto_to_capture_source(&source);
    let audio = source.audio();
    let sm = stream_manager.read().await;
    let capture = NativeCapture::new();
    let encoder = H264Encoder::default();
    let config = if audio {
        // Opt the new stream into PipeWire audio capture (default
        // sink monitor → AAC-LC). Bitrate comes from the target
        // device's capability table — see
        // `AudioStreamConfig::for_device`.
        // The mute handle is dropped here for now — once the UI
        // grows a mute toggle the handle needs to be threaded back
        // out into AppState. The manager keeps its own clone for
        // the lifetime of the audio task, so dropping ours doesn't
        // disable capture.
        let mute = AudioMuteHandle::default();
        let audio_cfg = match sm.devices().await.into_iter().find(|d| d.id == device_id) {
            Some(d) => AudioStreamConfig::for_device(&d.capabilities, mute.clone()),
            None => AudioStreamConfig {
                mute: mute.clone(),
                ..AudioStreamConfig::default()
            },
        };
        StreamConfig {
            audio: Some(audio_cfg),
            ..StreamConfig::default()
        }
    } else {
        StreamConfig::default()
    };
    sm.start_stream(device_id, cap_source, capture, encoder, config)
        .await
}

fn dto_to_capture_source(s: &SourceDto) -> CaptureSource {
    fn str_arg(s: &SourceDto, key: &str) -> Option<String> {
        let v = s.args.get(key)?;
        v.downcast_ref::<&str>()
            .ok()
            .map(std::string::ToString::to_string)
    }
    match s.kind.as_str() {
        "monitor" | "screen" => CaptureSource::FullScreen {
            monitor: str_arg(s, "id").or_else(|| str_arg(s, "monitor")),
        },
        "window" => {
            let id = str_arg(s, "id");
            if let Some(id) = id {
                // The picker lists the enumerator's ids: carried as is, so
                // windows sharing a title stay apart.
                return CaptureSource::Window {
                    identifier: Some(crate::daemon::window_identifier(id)),
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

        let on_select: Rc<dyn Fn(SourceDto)> = Rc::new(move |dto: SourceDto| {
            if let Some(tx) = sender_select.borrow_mut().take() {
                let _ = tx.send(Some(dto));
            }
        });
        let on_cancel: Rc<dyn Fn()> = Rc::new(move || {
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
    pub on_select: Rc<dyn Fn(SourceDto)>,
    pub on_cancel: Rc<dyn Fn()>,
}

impl PartialEq for SourcePicker {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.on_select, &other.on_select)
            && Rc::ptr_eq(&self.on_cancel, &other.on_cancel)
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
    /// Whether the streaming side can actually honour a window
    /// pick. When `false` the picker hides the Windows tab — there
    /// would be no point in selecting a window the capture path
    /// would just fall back to portal-asking-again on. Driven by
    /// the daemon's `CaptureCapabilities` property.
    can_capture_window: bool,
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
        // Audio toggle. Default off — audio capture isn't wired
        // through StreamManager yet, but the choice rides in
        // `SourceDto.args["audio"]` so the daemon can pick it up
        // once the audio pipeline lands.
        let audio = use_state::<bool>(|| false);

        // List the sources on mount and again whenever the daemon
        // reports a change, for as long as the picker is open.
        use_side_effect_with_deps(&(), move |(): &()| {
            spawn(watch_entries(entries));
        });

        let snapshot = entries.read().clone();
        let current_tab = *tab.read();

        // Wrap `on_select` so every emission carries the current
        // audio choice — the grid cards stay oblivious to it.
        let on_select_with_audio: Rc<dyn Fn(SourceDto)> = {
            let on_select = on_select.clone();
            Rc::new(move |dto: SourceDto| {
                let dto = dto.with_audio(*audio.peek_state());
                (on_select)(dto);
            })
        };

        // Compact-mode rendering is decided by `snapshot.can_capture_window`:
        // - true  → grid with thumbnail cards (the rich view).
        // - false → bare name list (compositor doesn't expose
        //   ext-foreign-toplevel-image-capture-source, so previews
        //   would be empty placeholders anyway — show the data we
        //   *do* have instead of pretending).
        // Selection paths through start_stream as usual; if the
        // streaming side can't honour the pick either it falls
        // back to the portal flow.
        let compact_windows = !snapshot.can_capture_window;

        rect()
            .expanded()
            .vertical()
            .padding(16.)
            .spacing(12.)
            .content(Content::Flex)
            .child(header(on_cancel.clone()))
            .child(tabs_row(current_tab, tab))
            .child(grid(
                current_tab,
                snapshot,
                on_select_with_audio,
                compact_windows,
            ))
            .child(footer(on_cancel, audio))
    }
}

fn header(on_cancel: Rc<dyn Fn()>) -> Rect {
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

fn footer(on_cancel: Rc<dyn Fn()>, audio_state: State<bool>) -> Rect {
    let audio_on = *audio_state.read();
    // Click target: the whole row toggles. Box-shaped indicator on
    // the left mirrors the state so the affordance reads "checkbox".
    let toggle_row = {
        let bg_indicator = if audio_on {
            (60, 90, 160)
        } else {
            (40, 40, 55)
        };
        let fg_indicator = if audio_on {
            (245, 245, 250)
        } else {
            (90, 90, 105)
        };
        rect()
            .horizontal()
            .spacing(10.)
            .cross_align(Alignment::center())
            .on_press(move |_| {
                let mut s = audio_state;
                let cur = *s.peek_state();
                s.set(!cur);
            })
            .child(
                rect()
                    .width(Size::px(18.))
                    .height(Size::px(18.))
                    .corner_radius(4.)
                    .background(bg_indicator)
                    .border(Border::new().fill((90, 90, 105)).width(1.))
                    .center()
                    // U+2713 (check). Renders only when on.
                    .maybe_child(
                        audio_on.then(|| label().text("✓").color(fg_indicator).font_size(13.)),
                    ),
            )
            .child(
                label()
                    .text("Include audio")
                    .color((220, 220, 230))
                    .font_size(13.),
            )
    };

    rect()
        .width(Size::fill())
        .horizontal()
        .cross_align(Alignment::center())
        .content(Content::Flex)
        .child(toggle_row)
        // Filler pushes the cancel button to the far right.
        .child(rect().width(Size::flex(1.)))
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

fn grid(
    current: Tab,
    entries: Entries,
    on_select: Rc<dyn Fn(SourceDto)>,
    compact_windows: bool,
) -> Rect {
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
                rect().key(m.id.clone()).child(MonitorCard {
                    info: m,
                    on_select: on_select.clone(),
                })
            });
            body.child(
                ScrollView::new()
                    .expanded()
                    .direction(Direction::Vertical)
                    .child(
                        rect()
                            .horizontal()
                            .spacing(8.)
                            .max_width(Size::fill())
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
            // Two layouts for the Windows tab, picked on whether
            // the daemon told us window previews are reachable:
            //
            //   - rich: same thumbnail-card grid as Monitors. Each
            //     card lazy-loads `GetWindowThumbnail`.
            //   - compact: vertical list of name + app_id rows, no
            //     preview area. Used when the compositor doesn't
            //     expose `ext_foreign_toplevel_image_capture_source_manager_v1`
            //     — we'd just be rendering placeholder boxes
            //     otherwise. The list is denser, scrolls naturally,
            //     and reads as "this is by design", not "I'm broken".
            if compact_windows {
                let rows = entries.windows.into_iter().map(move |w| {
                    rect()
                        .key(w.id.clone())
                        .width(Size::fill())
                        .child(WindowRow {
                            info: w,
                            on_select: on_select.clone(),
                        })
                });
                body.child(
                    ScrollView::new()
                        .expanded()
                        .direction(Direction::Vertical)
                        .child(
                            rect()
                                .vertical()
                                .spacing(4.)
                                .width(Size::fill())
                                .children(rows),
                        ),
                )
            } else {
                let cards = entries.windows.into_iter().map(move |w| {
                    rect().key(w.id.clone()).child(WindowCard {
                        info: w,
                        on_select: on_select.clone(),
                    })
                });
                body.child(
                    ScrollView::new()
                        .expanded()
                        .direction(Direction::Vertical)
                        .child(
                            rect()
                                .horizontal()
                                .spacing(8.)
                                .max_width(Size::fill())
                                .content(Content::wrap_spacing(8.))
                                .children(cards),
                        ),
                )
            }
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

const THUMB_W: u16 = 220;
const THUMB_H: u16 = 124;

#[derive(Clone)]
struct MonitorCard {
    info: MonitorInfoDto,
    on_select: Rc<dyn Fn(SourceDto)>,
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
            &id,
            "monitor",
            THUMB_W,
            THUMB_H,
            move |_| (on_select)(SourceDto::monitor(id_for_select.clone())),
        )
    }
}

#[derive(Clone)]
struct WindowCard {
    info: WindowInfoDto,
    on_select: Rc<dyn Fn(SourceDto)>,
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

        card_shell(title, app_id, &id, "window", THUMB_W, THUMB_H, move |_| {
            (on_select)(SourceDto::window_by_id(id_for_select.clone()));
        })
    }
}

/// Compact list-row variant of `WindowCard` used when the daemon
/// reports no window-preview capability. No thumbnail, just a row
/// of text the user can scan and click. Selection routes through
/// the same `SourceDto::window_by_id` as the card path, so click
/// behavior is identical — only the visual is different.
#[derive(Clone)]
struct WindowRow {
    info: WindowInfoDto,
    on_select: Rc<dyn Fn(SourceDto)>,
}

impl PartialEq for WindowRow {
    fn eq(&self, other: &Self) -> bool {
        self.info.id == other.info.id && self.info.title == other.info.title
    }
}

impl Component for WindowRow {
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

        rect()
            .width(Size::fill())
            .background((28, 28, 38))
            .corner_radius(8.)
            .border(Border::new().fill((50, 50, 65)).width(1.))
            .padding(Gaps::new(8., 12., 8., 12.))
            .horizontal()
            .cross_align(Alignment::center())
            .spacing(12.)
            .on_press(move |_| (on_select)(SourceDto::window_by_id(id_for_select.clone())))
            .child(
                rect()
                    .vertical()
                    .width(Size::flex(1.))
                    .spacing(2.)
                    .child(
                        label()
                            .text(title)
                            .color((230, 230, 240))
                            .font_size(14.)
                            .max_lines(1)
                            .text_overflow(TextOverflow::Ellipsis),
                    )
                    .child(
                        label()
                            .text(app_id)
                            .color((140, 140, 160))
                            .font_size(11.)
                            .max_lines(1)
                            .text_overflow(TextOverflow::Ellipsis),
                    ),
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
    cache_key: &str,
    kind: &'static str,
    thumb_w: u16,
    thumb_h: u16,
    on_press: impl Fn(Event<PressEventData>) + 'static,
) -> Rect {
    let thumb = use_state(|| Thumbnail::Loading);
    let cache_key_for_effect = cache_key.to_owned();
    use_side_effect_with_deps(&cache_key_for_effect, move |key: &String| {
        let mut thumb = thumb;
        let k = key.clone();
        spawn(async move {
            thumb.set(Thumbnail::fetch(kind, &k, u32::from(thumb_w), u32::from(thumb_h)).await);
        });
    });

    // Outer thumb frame: always exactly `thumb_w × thumb_h` so the
    // grid stays uniform regardless of what came back from the
    // daemon. The ImageViewer / placeholder lives inside, centered.
    //
    // `AspectRatio::Min` + `ImageCover::Center` make the image
    // scale uniformly until it fills the smaller dimension, then
    // letterbox / pillarbox along the other. Portrait monitors
    // become a centered vertical strip inside the landscape card
    // frame instead of being stretched horizontally; landscape
    // monitors fill the frame; square monitors land in the middle.
    let preview_frame = rect()
        .width(Size::px(f32::from(thumb_w)))
        .height(Size::px(f32::from(thumb_h)))
        .corner_radius(6.)
        .background((10, 10, 16))
        .center()
        .overflow(Overflow::Clip)
        .child(match thumb.read().clone() {
            Thumbnail::Ready(bytes) => {
                let source: ImageSource = ((kind, cache_key.to_owned()), bytes).into();
                let viewer: Element = ImageViewer::new(source)
                    .width(Size::px(f32::from(thumb_w)))
                    .height(Size::px(f32::from(thumb_h)))
                    .aspect_ratio(AspectRatio::Min)
                    .image_cover(ImageCover::Center)
                    .into();
                viewer
            }
            state @ (Thumbnail::Loading | Thumbnail::Unavailable) => {
                // No thumbnail bytes yet, or none coming: the backend
                // can't preview this item (e.g. no window capture
                // protocol on the compositor), the request failed or
                // it timed out. For windows we surface the subtitle
                // (app_id) prominently as a fallback identifier; for
                // monitors we show the id text.
                let status = if state == Thumbnail::Loading {
                    "loading preview…"
                } else {
                    "no preview"
                };
                let big = if kind == "window" && subtitle != "—" {
                    subtitle.clone()
                } else {
                    cache_key.to_owned()
                };
                rect()
                    .expanded()
                    .center()
                    .vertical()
                    .spacing(4.)
                    .child(
                        label()
                            .text(big)
                            .color((180, 180, 195))
                            .font_size(18.)
                            .max_lines(1)
                            .text_overflow(TextOverflow::Ellipsis),
                    )
                    .child(label().text(status).color((110, 110, 125)).font_size(11.))
                    .into()
            }
        });

    // Card frame: fixed width = thumb width + 2*padding; height is
    // auto-sized to thumb + subtitle/title rows. Using Size::px (not
    // Size::fill / flex) on width is what stops the wrap-grid from
    // stretching cards when the window resizes — every card stays
    // at the same width and the row reflows by reflowing whole
    // cards, not by resizing them.
    rect()
        .width(Size::px(f32::from(thumb_w) + 12.))
        .background((28, 28, 38))
        .corner_radius(10.)
        .padding(6.)
        .vertical()
        .spacing(4.)
        .border(Border::new().fill((50, 50, 65)).width(1.))
        .on_press(on_press)
        .child(preview_frame)
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

/// Longest the picker waits for one listing call before reporting the
/// daemon as unresponsive instead of showing "Loading" forever.
const DAEMON_CALL_TIMEOUT: Duration = Duration::from_secs(5);

/// Longest a card waits for its preview. Longer than the backend's own
/// capture deadline so a slow capture still lands.
const THUMBNAIL_TIMEOUT: Duration = Duration::from_secs(8);

/// Change signals arrive in bursts (a window retitling, several outputs
/// reconfiguring); wait this long and coalesce before listing again.
const REFRESH_DEBOUNCE: Duration = Duration::from_millis(250);

/// Awaits `call`, giving up after `deadline`.
async fn bounded<T>(
    deadline: Duration,
    what: &str,
    call: impl Future<Output = T>,
) -> std::result::Result<T, String> {
    tokio::time::timeout(deadline, call)
        .await
        .map_err(|_| format!("{what}: no answer within {}s", deadline.as_secs()))
}

impl Entries {
    fn failed(error: String) -> Self {
        Self {
            loaded: true,
            error: Some(error),
            ..Self::default()
        }
    }

    /// Lists monitors, windows and the capture capabilities.
    ///
    /// # Errors
    ///
    /// When the daemon doesn't answer within [`DAEMON_CALL_TIMEOUT`].
    /// A call that answers with an error counts as an empty list.
    async fn load(proxy: &ManagerProxy<'static>) -> std::result::Result<Self, String> {
        let monitors = bounded(DAEMON_CALL_TIMEOUT, "ListMonitors", proxy.list_monitors())
            .await?
            .unwrap_or_else(|e| {
                warn!(%e, "list_monitors failed");
                Vec::new()
            });
        let windows = bounded(DAEMON_CALL_TIMEOUT, "ListWindows", proxy.list_windows())
            .await?
            .unwrap_or_else(|e| {
                warn!(%e, "list_windows failed");
                Vec::new()
            });
        // Whether the capture path can stream a window selection
        // without falling back to a portal dialog. Without it the
        // Windows tab shows a plain list instead of preview cards. A
        // daemon without the property counts as "no".
        let can_capture_window = bounded(
            DAEMON_CALL_TIMEOUT,
            "CaptureCapabilities",
            proxy.capture_capabilities(),
        )
        .await?
        .is_ok_and(|caps| caps.iter().any(|s| s == "window"));
        Ok(Self {
            monitors,
            windows,
            loaded: true,
            can_capture_window,
            error: None,
        })
    }
}

/// Keeps `entries` in sync with the daemon while the picker is open:
/// lists once, then again after each `MonitorsChanged` /
/// `WindowsChanged` signal. Ends with the picker window, which drops
/// the task.
async fn watch_entries(mut entries: State<Entries>) {
    let proxy = match bounded(DAEMON_CALL_TIMEOUT, "connect", client::proxy())
        .await
        .and_then(|proxy| proxy.map_err(|e| e.to_string()))
    {
        Ok(proxy) => proxy,
        Err(e) => {
            entries.set(Entries::failed(format!("can't reach daemon: {e}")));
            return;
        }
    };
    // Subscribe before the first listing so a change in between is not
    // lost. Without signals the picker still lists once.
    let subscribe = async {
        let monitors = proxy.receive_monitors_changed().await?;
        let windows = proxy.receive_windows_changed().await?;
        zbus::Result::Ok(futures_util::stream::select(
            monitors.map(|_| ()),
            windows.map(|_| ()),
        ))
    };
    let subscribed = bounded(DAEMON_CALL_TIMEOUT, "subscribe", subscribe)
        .await
        .and_then(|r| r.map_err(|e| e.to_string()));
    let mut changes = match subscribed {
        Ok(changes) => Some(changes),
        Err(e) => {
            warn!(%e, "can't watch source changes; the picker lists once");
            None
        }
    };
    loop {
        match Entries::load(&proxy).await {
            Ok(loaded) => entries.set(loaded),
            // Keep showing the last good listing on a failed refresh.
            Err(e) if entries.peek().loaded => warn!(%e, "refreshing sources failed"),
            Err(e) => entries.set(Entries::failed(e)),
        }
        let Some(stream) = changes.as_mut() else {
            return;
        };
        if stream.next().await.is_none() {
            return;
        }
        tokio::time::sleep(REFRESH_DEBOUNCE).await;
        while futures_util::FutureExt::now_or_never(stream.next()).is_some_and(|c| c.is_some()) {}
    }
}

/// Preview state of one picker card.
#[derive(Clone, PartialEq)]
enum Thumbnail {
    Loading,
    Ready(Bytes),
    /// The backend can't preview this source, the request failed or it
    /// timed out.
    Unavailable,
}

impl From<Vec<u8>> for Thumbnail {
    /// The daemon answers `[]` when the backend can't preview a source.
    fn from(png: Vec<u8>) -> Self {
        if png.is_empty() {
            Self::Unavailable
        } else {
            Self::Ready(Bytes::from(png))
        }
    }
}

impl Thumbnail {
    async fn fetch(kind: &str, id: &str, max_w: u32, max_h: u32) -> Self {
        let request = async {
            let proxy = client::proxy().await.map_err(|e| e.to_string())?;
            match kind {
                "monitor" => proxy.get_monitor_thumbnail(id, max_w, max_h).await,
                "window" => proxy.get_window_thumbnail(id, max_w, max_h).await,
                other => return Err(format!("unknown source kind {other:?}")),
            }
            .map_err(|e| e.to_string())
        };
        match bounded(THUMBNAIL_TIMEOUT, "thumbnail", request).await {
            Ok(Ok(png)) => png.into(),
            Ok(Err(e)) | Err(e) => {
                tracing::debug!(kind, id, %e, "no thumbnail");
                Self::Unavailable
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn unanswered_call_times_out_instead_of_waiting_forever() {
        let never = std::future::pending::<()>();
        let err = bounded(DAEMON_CALL_TIMEOUT, "ListMonitors", never)
            .await
            .unwrap_err();
        assert!(err.contains("ListMonitors"), "{err}");
    }

    #[test]
    fn a_picked_window_keeps_its_listed_id() {
        for (id, expected) in [
            ("toplevel:2", "Source(\"toplevel:2\")"),
            ("window:0x2a00007", "Source(\"window:0x2a00007\")"),
            ("44040199", "Id(44040199)"),
        ] {
            let CaptureSource::Window {
                identifier: Some(identifier),
            } = dto_to_capture_source(&SourceDto::window_by_id(id))
            else {
                panic!("a window source for {id}");
            };
            assert_eq!(format!("{identifier:?}"), expected);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn answered_call_passes_through() {
        assert_eq!(bounded(DAEMON_CALL_TIMEOUT, "x", async { 7 }).await, Ok(7));
    }

    #[test]
    fn empty_thumbnail_reply_means_no_preview() {
        assert!(Thumbnail::from(Vec::new()) == Thumbnail::Unavailable);
        assert!(matches!(Thumbnail::from(vec![1]), Thumbnail::Ready(_)));
    }
}
