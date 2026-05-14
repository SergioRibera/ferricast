use ferricast::prelude::*;
use freya::{prelude::*, radio::*};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::picker;

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
            .maybe(!is_streaming, |r| r.on_press(move |_| (on_request_picker)()))
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
