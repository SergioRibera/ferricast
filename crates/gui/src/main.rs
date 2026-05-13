use std::sync::Arc;

use ferricast::prelude::*;
use freya::{prelude::*, radio::*};
use tokio::sync::Mutex;
use tracing_subscriber::EnvFilter;

mod app;

use crate::app::*;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new(
                "warn,freya=off,freya_core=off,freya_winit=off,ragnarok=off,ferricast_chromecast=info,ferricast_encoder=info",
            )
        }))
        .init();

    tracing::info!("Ferricast starting");

    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls CryptoProvider");

    let (stream_manager, mut event_rx) = StreamManager::builder()
        .with_chromecast()
        .build_with_events();

    let stream_manager = Arc::new(Mutex::new(stream_manager));

    let radio_station = RadioStation::create_global(AppState::default());

    let mut radio_events = radio_station.clone();
    let sm_discovery = Arc::clone(&stream_manager);

    launch(
        LaunchConfig::new()
            .with_future(move |_| async move {
                if let Err(e) = sm_discovery.lock().await.start_discovery().await {
                    tracing::error!(%e, "Failed to start discovery");
                } else {
                    tracing::info!("Discovery started for all protocols");
                }
            })
            .with_future(move |_| async move {
                loop {
                    match event_rx.recv().await {
                        Some(ManagerEvent::DeviceFound(device)) => {
                            radio_events
                                .write_channel(AppChannel::Devices)
                                .devices
                                .entry(device.id)
                                .insert_entry(device);
                        }
                        Some(ManagerEvent::DeviceLost(id)) => {
                            radio_events
                                .write_channel(AppChannel::Devices)
                                .devices
                                .remove(&id);
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
