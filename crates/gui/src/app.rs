use ferricast::prelude::*;
use freya::{prelude::*, radio::*};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::daemon::PickerRequest;
use crate::picker;
use crate::receiver_window;

/// Request to spawn a receiver window for an incoming transmission.
/// Pumped from `with_future` (which sees the manager events) to the
/// App's render loop (which holds `Platform`) over a tokio mpsc —
/// same shape the picker uses to bridge daemon → UI.
///
/// Carries the `rx` halves of the per-session video / audio
/// channels: the sink (manager-side) holds the `tx`, the window
/// (Freya-side) holds the `rx`. One window per session.
pub struct ReceiverWindowReq {
    pub remote: ferricast::RemoteSender,
    pub info: ferricast::MediaInfo,
    pub counters: Arc<receiver_window::ReceiverCounters>,
    pub video_rx: tokio::sync::mpsc::Receiver<receiver_window::VideoFrame>,
    pub audio_rx: tokio::sync::mpsc::Receiver<receiver_window::AudioBuf>,
}

#[derive(Default)]
pub struct AppState {
    pub devices: HashMap<Uuid, Device>,
    pub streaming: Vec<Uuid>,
}

#[derive(PartialEq, Eq, Clone, Debug, Copy, Hash)]
pub enum AppChannel {
    Devices,
    Streaming,
}

impl RadioChannel<AppState> for AppChannel {}

pub struct FerricastApp {
    pub stream_manager: Arc<Mutex<StreamManager>>,
    pub radio_station: RadioStation<AppState, AppChannel>,
    /// Receiver for picker requests coming from the daemon's
    /// D-Bus path (e.g. `--background` mode where there's no
    /// in-process share button to trigger the picker). Wrapped in
    /// `Rc<RefCell<Option<_>>>` because `App::render` only sees
    /// `&self` and the receiver isn't `Clone` — the first render
    /// `.take()`s it and starts the listener task.
    pub picker_req_rx: Rc<RefCell<Option<tokio::sync::mpsc::Receiver<PickerRequest>>>>,
    /// Receiver-window requests from the pump. Same one-shot-take
    /// pattern as `picker_req_rx`.
    pub receiver_req_rx: Rc<RefCell<Option<tokio::sync::mpsc::Receiver<ReceiverWindowReq>>>>,
}

impl App for FerricastApp {
    fn render(&self) -> impl IntoElement {
        let stream_manager = self.stream_manager.clone();
        use_share_radio(move || self.radio_station);

        let devices_radio = use_radio::<AppState, AppChannel>(AppChannel::Devices);
        let streaming_radio = use_radio::<AppState, AppChannel>(AppChannel::Streaming);
        let binding = devices_radio.read();

        // Grab the platform once per render so per-device callbacks
        // can clone it cheaply. `Platform::get()` consumes a root
        // context that's only valid from inside a component body —
        // calling it inside the share-button closure would panic.
        let platform = Platform::get();

        // Picker-request listener: drive the daemon→freya channel
        // exactly once, on first mount. `use_hook` runs its init
        // closure a single time per component instance, so the
        // `take()` is safe and the spawned future lives for the
        // lifetime of the app.
        //
        // For each request that arrives (D-Bus `StartStream` with
        // an abstract source), we open the picker and forward the
        // user's choice back through the embedded oneshot. The
        // daemon's request handler awaits the oneshot, then runs
        // `StreamManager::start_stream` itself.
        let picker_req_rx_init = self.picker_req_rx.clone();
        let picker_platform = platform.clone();
        use_hook(move || {
            if let Some(mut rx) = picker_req_rx_init.borrow_mut().take() {
                let platform = picker_platform;
                spawn(async move {
                    while let Some(req) = rx.recv().await {
                        tracing::debug!(
                            device_id = ?req.device_id,
                            "daemon picker request — opening picker window"
                        );
                        let (dto_tx, dto_rx) = tokio::sync::oneshot::channel();
                        picker::open_picker_for_dto(platform.clone(), dto_tx);
                        let _ = req.reply.send(dto_rx.await.ok().flatten());
                    }
                });
            }
        });

        // Receiver-window listener — symmetric to the picker one
        // above. Drains requests off the pump → app channel and
        // opens a new top-level Freya window per receiver session.
        let receiver_req_rx_init = self.receiver_req_rx.clone();
        let receiver_platform = platform.clone();
        use_hook(move || {
            if let Some(mut rx) = receiver_req_rx_init.borrow_mut().take() {
                let platform = receiver_platform;
                spawn(async move {
                    while let Some(req) = rx.recv().await {
                        tracing::info!(
                            sender = %req.remote.addr,
                            audio = req.info.audio.is_some(),
                            video = req.info.video.is_some(),
                            "opening receiver window"
                        );
                        let _wid = receiver_window::open_receiver_window(
                            platform.clone(),
                            req.remote,
                            req.info,
                            req.counters,
                            req.video_rx,
                            req.audio_rx,
                        )
                        .await;
                    }
                });
            }
        });

        rect().expanded().background((18, 18, 24)).vertical().child(
            rect()
                .expanded()
                .padding(24.)
                .vertical()
                .spacing(8.)
                .maybe_child(
                    binding.devices.is_empty().then_some(
                        label()
                            .text("Buscando dispositivos...")
                            .font_size(13.)
                            .color((140, 140, 160)),
                    ),
                )
                .maybe(!binding.devices.is_empty(), |r| {
                    r.child(
                        ScrollView::new()
                            .expanded()
                            .direction(Direction::Vertical)
                            .children(binding.devices.iter().map(|(device_id, device)| {
                                let device_id = *device_id;
                                let is_streaming =
                                    streaming_radio.read().streaming.contains(&device_id);
                                let sm = Arc::clone(&stream_manager);
                                DeviceCard {
                                    device: device.clone(),
                                    is_streaming,
                                    on_request_picker: Arc::new({
                                        let sm = sm.clone();
                                        let platform = platform.clone();
                                        move || {
                                            // Open the picker in a
                                            // standalone OS window
                                            // so it stays reachable
                                            // even when the main
                                            // Ferricast window is
                                            // minimised / hidden.
                                            picker::open_picker(
                                                platform.clone(),
                                                sm.clone(),
                                                device_id,
                                            );
                                        }
                                    }),
                                    on_stop: Arc::new({
                                        let sm = sm.clone();
                                        move || {
                                            let sm = sm.clone();
                                            spawn(async move {
                                                let sm = sm.lock().await;
                                                if let Err(e) = sm.stop_stream(device_id).await {
                                                    tracing::error!(%e, ?device_id, "stop_stream");
                                                }
                                            });
                                        }
                                    }),
                                }
                                .into()
                            })),
                    )
                }),
        )
    }
}

// --- DeviceCard component ---

#[derive(Clone)]
struct DeviceCard {
    device: Device,
    is_streaming: bool,
    /// Open the picker in a separate top-level window. Wired to
    /// the share button when the device isn't currently streaming.
    on_request_picker: Arc<dyn Fn()>,
    /// Stop the in-flight stream. Wired to the share button
    /// (re-used as a stop button) when the device is streaming.
    on_stop: Arc<dyn Fn()>,
}

impl PartialEq for DeviceCard {
    fn eq(&self, other: &Self) -> bool {
        self.device == other.device && self.is_streaming == other.is_streaming
    }
}

impl Component for DeviceCard {
    fn render(&self) -> impl IntoElement {
        let device = self.device.clone();
        let is_streaming = self.is_streaming;
        let on_request_picker = self.on_request_picker.clone();
        let on_stop = self.on_stop.clone();

        let bg = if is_streaming {
            (30, 60, 40)
        } else {
            (28, 28, 38)
        };
        let border_color = if is_streaming {
            (80, 200, 120)
        } else {
            (50, 50, 65)
        };

        rect()
            .width(Size::fill())
            .height(Size::px(72.))
            .background(bg)
            .corner_radius(10.)
            .border(Border::new().fill(border_color).width(1.))
            .padding(Gaps::new(0., 16., 0., 16.))
            .cross_align(Alignment::center())
            .horizontal()
            .spacing(12.)
            .maybe(is_streaming, |r| r.on_press(move |_| (on_stop)()))
            .maybe(!is_streaming, |r| {
                r.on_press(move |_| (on_request_picker)())
            })
            .child(
                rect()
                    .center()
                    .maybe_child((device.protocol == "chromecast").then(|| {
                        svg(device.protocol_icon.clone())
                            .fill(Color::WHITE)
                            .width(Size::px(36.))
                            .height(Size::px(36.))
                    }))
                    .maybe_child((device.protocol != "chromecast").then(|| {
                        svg(device.protocol_icon)
                            .stroke(Color::WHITE)
                            .width(Size::px(36.))
                            .height(Size::px(36.))
                    })),
            )
            .child(
                rect()
                    .width(Size::fill())
                    .vertical()
                    .spacing(3.)
                    .child(
                        label()
                            .text(device.name.clone())
                            .font_size(15.)
                            .color((230, 230, 240)),
                    )
                    .child(
                        label()
                            .text(device.protocol.clone())
                            .font_size(12.)
                            .color((230, 230, 240)),
                    ),
            )
    }
}

fn share_btn(b: impl Into<SvgBytes>) -> Rect {
    rect()
        .width(Size::px(18.))
        .height(Size::px(18.))
        .center()
        .child(svg(b).expanded().stroke(Color::WHITE))
}
