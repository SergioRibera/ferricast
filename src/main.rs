mod manager;
mod tray;

use std::sync::Arc;

use ferricast_encoder::h264::H264Encoder;
use freya::{prelude::*, radio::*};
use tokio::sync::Mutex;
use tracing_subscriber::EnvFilter;

use ferricast_capture::PipeWireCapture;
use ferricast_core::{CaptureSource, Codec, StreamConfig};

use crate::manager::*;
use crate::tray::{FerricastTray, TrayAction};

use crate::app::*;

mod app;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("error")),
        )
        .init();

    tracing::info!("Ferricast starting");

    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls CryptoProvider");

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime");
    let _rt = rt.enter();

    let mut stream_manager = StreamManager::default();
    stream_manager.register::<ferricast_chromecast::ChromecastHandler>();
    stream_manager.register::<ferricast_dial::DialHandler>();
    stream_manager.register::<ferricast_airplay::AirPlayHandler>();
    stream_manager.register::<ferricast_miracast::MiracastHandler>();

    let mut event_rx = stream_manager
        .take_event_rx()
        .expect("event_rx already taken");

    let tray = match FerricastTray::new() {
        Ok(t) => t,
        Err(e) => {
            tracing::error!(%e, "Failed to create tray icon");
            std::process::exit(1);
        }
    };

    tray.run();

    let stream_manager = Arc::new(Mutex::new(stream_manager));

    let radio_station = RadioStation::create_global(AppState::default());

    // Clon para el future de eventos
    let mut radio_events = radio_station.clone();
    let sm_discovery = Arc::clone(&stream_manager);
    let sm_tray = Arc::clone(&stream_manager);
    let radio_tray = radio_station.clone();

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
            // Loop del tray
            .with_future(move |_| async move {
                loop {
                    // try_recv_action es síncrono, usamos sleep para no bloquear
                    if let Some(action) = tray.try_recv_action() {
                        match action {
                            TrayAction::About => {
                                let _ = std::process::Command::new("xdg-open")
                                    .arg("https://github.com/SergioRibera/ferricast")
                                    .spawn();
                            }
                            TrayAction::Quit => {
                                tracing::info!("Shutting down");
                                let _ = sm_tray.lock().await.shutdown().await;
                                tray.close();
                                break;
                            }
                            TrayAction::Device(device_id) => {
                                let sm = Arc::clone(&sm_tray);
                                tokio::spawn(async move {
                                    let capture = PipeWireCapture::new();
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
                            _ => {}
                        }
                    }

                    // Sincronizar devices del radio al tray
                    {
                        let state = radio_tray.read();
                        for device in &state.devices {
                            tray.handle_manager_event(&ManagerEvent::DeviceFound(device.clone()));
                        }
                    }

                    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
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
