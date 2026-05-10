use ferricast_encoder::h264::H264Encoder;
use freya::{prelude::*, radio::*};
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

use ferricast_capture::NativeCapture;
use ferricast_core::{CaptureSource, Device, StreamConfig};

use crate::manager::StreamManager;

#[derive(Default)]
pub struct AppState {
    pub devices: Vec<Device>,
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
        let devices = binding.devices.as_slice();

        rect().expanded().background((18, 18, 24)).vertical().child(
            // Body
            rect()
                .expanded()
                .padding(24.)
                .vertical()
                .spacing(8.)
                .maybe_child(
                    devices.is_empty().then_some(
                        label()
                            .text("Buscando dispositivos...")
                            .font_size(13.)
                            .color((140, 140, 160)),
                    ),
                )
                .children(devices.iter().map(|device| {
                    let device_id = device.id;
                    let is_streaming = streaming_radio.read().streaming.contains(&device_id);
                    DeviceCard {
                        device: device.clone(),
                        is_streaming,
                        on_click: Arc::new({
                            let sm = Arc::clone(&stream_manager);
                            move || {
                                let sm = sm.clone();
                                spawn(async move {
                                    let capture = NativeCapture::new();
                                    let encoder = H264Encoder::default();
                                    let source = CaptureSource::FullScreen { monitor: None };
                                    let config = StreamConfig::default();
                                    let sm = sm.lock().await;
                                    if let Err(e) = sm
                                        .start_stream(device_id, source, capture, encoder, config)
                                        .await
                                    {
                                        tracing::error!(%e, ?device_id, "Failed to start stream");
                                    }
                                });
                            }
                        }),
                    }
                    .into()
                })),
        )
    }
}

// --- DeviceCard component ---

#[derive(Clone)]
struct DeviceCard {
    device: Device,
    is_streaming: bool,
    on_click: Arc<dyn Fn() + Send + Sync>,
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
        let on_click = self.on_click.clone();

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
        let status_text = if is_streaming {
            "● Transmitiendo"
        } else {
            "○ Disponible"
        };
        let status_color: (u8, u8, u8) = if is_streaming {
            (80, 200, 120)
        } else {
            (100, 100, 120)
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
            .on_press(move |_| (on_click)())
            .child(
                // Icono de protocolo (círculo de color)
                rect()
                    .width(Size::px(36.))
                    .height(Size::px(36.))
                    .corner_radius(18.)
                    .background((247, 109, 39))
                    .center()
                    .child(
                        label()
                            .text(device.protocol[..1].to_uppercase())
                            .font_size(14.)
                            .color((255, 255, 255)),
                    ),
            )
            .child(
                // Info del dispositivo
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
                            .text(format!("{} - {}", device.protocol, device.addr))
                            .font_size(12.)
                            .color((100, 100, 120)),
                    )
                    .child(label().text(status_text).font_size(11.).color(status_color)),
            )
    }
}
