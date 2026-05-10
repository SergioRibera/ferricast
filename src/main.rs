mod manager;

use std::sync::Arc;

use freya::{prelude::*, radio::*};
use tokio::sync::Mutex;
use tracing_subscriber::EnvFilter;

use crate::manager::*;

use crate::app::*;

mod app;


#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new(
                "warn,freya=off,freya_core=off,freya_winit=off,ragnarok=off,ferricast=info,ferricast_capture=info,ferricast_chromecast=trace,ferricast_encoder=info,ferricast_hls=trace",
            )
        }))
        .init();

    tracing::info!("Ferricast starting");

    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls CryptoProvider");

    let mut stream_manager = StreamManager::default();
    stream_manager.register::<ferricast_chromecast::ChromecastHandler>();
    // stream_manager.register::<ferricast_dial::DialHandler>();
    // stream_manager.register::<ferricast_airplay::AirPlayHandler>();
    // stream_manager.register::<ferricast_miracast::MiracastHandler>();

    let mut event_rx = stream_manager
        .take_event_rx()
        .expect("event_rx already taken");

    let stream_manager = Arc::new(Mutex::new(stream_manager));

    let radio_station = RadioStation::create_global(AppState::default());

    // Clon para el future de eventos
    let mut radio_events = radio_station.clone();
    let sm_discovery = Arc::clone(&stream_manager);

    launch(
        LaunchConfig::new()
            // Discovery
            .with_future(move |_| async move {
                if let Err(e) = sm_discovery.lock().await.start_discovery().await {
                    tracing::error!(%e, "Failed to start discovery");
                } else {
                    tracing::info!("Discovery started for all protocols");
                }
            })
            // Loop de eventos del manager -> actualiza RadioStation
            .with_future(move |_| async move {
                loop {
                    match event_rx.recv().await {
                        Some(ManagerEvent::DeviceFound(device)) => {
                            radio_events
                                .write_channel(AppChannel::Devices)
                                .devices
                                .push(device);
                        }
                        Some(ManagerEvent::DeviceLost(id)) => {
                            radio_events
                                .write_channel(AppChannel::Devices)
                                .devices
                                .retain(|d| d.id != id);
                            radio_events
                                .write_channel(AppChannel::Streaming)
                                .streaming
                                .retain(|&s| s != id);
                        }
                        Some(ManagerEvent::StreamStarted { device_id, .. }) => {
                            let mut state = radio_events.write_channel(AppChannel::Streaming);
                            if !state.streaming.contains(&device_id) {
                                state.streaming.push(device_id);
                            }
                        }
                        Some(ManagerEvent::StreamStopped { device_id }) => {
                            radio_events
                                .write_channel(AppChannel::Streaming)
                                .streaming
                                .retain(|&s| s != device_id);
                        }
                        None => break,
                        _ => {}
                    }
                }
            })
            .with_window(
                WindowConfig::new_app(FerricastApp {
                    stream_manager,
                    radio_station,
                })
                .with_title("Ferricast")
                .with_size(800., 600.),
            ),
    );
}
