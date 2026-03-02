mod manager;
mod tray;

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tracing_subscriber::EnvFilter;

use ferricast_capture::{PassthroughEncoder, PipeWireCapture};
use ferricast_core::{CaptureSource, Codec, StreamConfig};

use crate::manager::StreamManager;
use crate::tray::{FerricastTray, TrayAction};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
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

    rt.block_on(async {
        if let Err(e) = stream_manager.lock().await.start_discovery().await {
            tracing::error!(%e, "Failed to start discovery");
        } else {
            tracing::info!("Discovery started for all protocols");
        }
    });

    rt.block_on(async {
        loop {
            if let Some(action) = tray.try_recv_action() {
                match action {
                    TrayAction::About => {
                        let _ = std::process::Command::new("xdg-open")
                            .arg("https://github.com/SergioRibera/ferricast")
                            .spawn();
                    }
                    TrayAction::Quit => {
                        tracing::info!("Shutting down");
                        let _ = stream_manager.lock().await.shutdown().await;
                        tray.close();
                        break;
                    }
                    TrayAction::Device(device_id) => {
                        tracing::info!(?device_id, "Device selected, opening screen picker");
                        let sm = Arc::clone(&stream_manager);
                        tokio::spawn(async move {
                            let capture = PipeWireCapture::new();
                            let encoder = PassthroughEncoder::new(Codec::H264);
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

            while let Ok(event) = event_rx.try_recv() {
                tray.handle_manager_event(&event);
            }

            std::thread::sleep(Duration::from_millis(50));
        }
    });
}
