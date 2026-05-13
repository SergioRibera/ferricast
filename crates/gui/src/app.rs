use ferricast::prelude::*;
use ferricast::WindowIdentifier;
use ferricast_dbus::SourceDto;
use freya::{prelude::*, radio::*};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::picker::SourcePicker;

#[derive(Default)]
pub struct AppState {
    pub devices: HashMap<Uuid, Device>,
    pub streaming: Vec<Uuid>,
    /// Set when a device card requests source selection. The
    /// top-level `App` mounts the `SourcePicker` popup over the
    /// device list while this is `Some`.
    pub picker: Option<PickerRequest>,
}

/// Open-picker request. Carries everything the picker needs to
/// route the selection back to the right stream — currently just
/// the device id, but kept as a struct so adding `kind` /
/// `prefer_window: bool` later doesn't shift state shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PickerRequest {
    pub device_id: Uuid,
}

#[derive(PartialEq, Eq, Clone, Debug, Copy, Hash)]
pub enum AppChannel {
    Devices,
    Streaming,
    Picker,
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
        let picker_radio = use_radio::<AppState, AppChannel>(AppChannel::Picker);
        let binding = devices_radio.read();

        let pending_picker = picker_radio.read().picker.clone();

        let body = rect().expanded().background((18, 18, 24)).vertical().child(
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
                                DeviceCard {
                                    device: device.clone(),
                                    is_streaming,
                                    on_request_picker: Arc::new({
                                        let mut picker_radio = picker_radio.clone();
                                        move || {
                                            picker_radio
                                                .write_channel(AppChannel::Picker)
                                                .picker = Some(PickerRequest { device_id });
                                        }
                                    }),
                                    on_stop: Arc::new({
                                        let sm = Arc::clone(&stream_manager);
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
        );

        body.maybe_child(pending_picker.map(|req| {
            let stream_manager = stream_manager.clone();
            let device_id = req.device_id;
            let mut picker_radio = picker_radio.clone();
            SourcePicker {
                on_select: Arc::new(move |source: SourceDto| {
                    // Clear the picker request first so the popup
                    // dismisses synchronously — the stream start
                    // happens on the spawned task and shouldn't
                    // block the UI thread.
                    picker_radio.write_channel(AppChannel::Picker).picker = None;
                    let sm = stream_manager.clone();
                    spawn(async move {
                        if let Err(e) = start_stream_with_dto(sm, device_id, source).await {
                            tracing::error!(%e, ?device_id, "start_stream");
                        }
                    });
                }),
                on_cancel: Arc::new({
                    let mut picker_radio = picker_radio.clone();
                    move || {
                        picker_radio.write_channel(AppChannel::Picker).picker = None;
                    }
                }),
            }
        }))
    }
}

/// Start a stream by translating a wire-shape `SourceDto` into the
/// in-process `CaptureSource` and dispatching through the local
/// `StreamManager`. Mirrors what the daemon does over D-Bus, but
/// stays in-process so the picker selection avoids a bus round-trip.
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

// --- DeviceCard component ---

#[derive(Clone)]
struct DeviceCard {
    device: Device,
    is_streaming: bool,
    /// Open the picker for this device. Wired to "share screen" /
    /// "share window" instead of starting the stream blindly so the
    /// user picks from real monitors/windows the daemon enumerated.
    on_request_picker: Arc<dyn Fn() + Send + Sync>,
    on_stop: Arc<dyn Fn() + Send + Sync>,
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

        let primary_btn = if is_streaming {
            // Already streaming — the share button stops it.
            TooltipContainer::new(Tooltip::new("Stop stream")).child(
                share_btn(include_bytes!("../assets/screen.svg"))
                    .on_press(move |_| (on_stop)()),
            )
        } else {
            // Not streaming — open the picker.
            TooltipContainer::new(Tooltip::new("Pick a source to share")).child(
                share_btn(include_bytes!("../assets/screen.svg"))
                    .on_press(move |_| (on_request_picker)()),
            )
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
                        rect()
                            .width(Size::fill())
                            .horizontal()
                            .spacing(5.)
                            .child(primary_btn),
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
