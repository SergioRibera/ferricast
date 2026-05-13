use std::sync::Arc;

use clap::Parser;
use ferricast::prelude::*;
use ferricast::ManagerEvent;
use freya::{prelude::*, radio::*};
use tokio::sync::{mpsc, Mutex};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

mod app;
mod cli;
mod client;
mod daemon;

use crate::app::*;
use crate::cli::{Cli, Command};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Cli::parse();

    // Client subcommands don't need the heavy init that the daemon
    // does (no tls, no rustls provider, no stream-manager build) —
    // they're a thin proxy to the bus, so we short-circuit early.
    if let Some(cmd) = args.command {
        init_tracing_for_client();
        return run_client(cmd).await;
    }

    init_tracing_for_daemon();
    tracing::info!("Ferricast starting");

    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls CryptoProvider");

    let (manager, manager_events) = StreamManager::builder()
        .with_chromecast()
        .build_with_events();
    let manager = Arc::new(Mutex::new(manager));

    // Fan-out: the daemon owns the original receiver and forwards a
    // clone of each event into `ui_rx` so the in-process window sees
    // the same stream the bus does, in the same order.
    let (ui_tx, ui_rx) = mpsc::channel::<ManagerEvent>(256);
    let _conn = daemon::start(manager.clone(), manager_events, Some(ui_tx))
        .await
        .map_err(|e| anyhow::anyhow!("failed to publish D-Bus service: {e}"))?;
    tracing::info!(
        bus = ferricast_dbus::BUS_NAME,
        path = ferricast_dbus::OBJECT_PATH,
        "D-Bus service published"
    );

    // Start discovery (both modes need it).
    {
        let sm = manager.clone();
        tokio::spawn(async move {
            if let Err(e) = sm.lock().await.start_discovery().await {
                tracing::error!(%e, "Failed to start discovery");
            } else {
                tracing::info!("Discovery started for all protocols");
            }
        });
    }

    // Optional `--device` auto-start: watch for the matching device
    // and fire a stream once it shows up.
    if let Some(device_arg) = args.device.clone() {
        let sm = manager.clone();
        let source = args.source;
        tokio::spawn(async move {
            if let Err(e) = auto_stream(sm, device_arg, source).await {
                tracing::error!(%e, "auto-stream failed");
            }
        });
    }

    if args.background {
        // Headless: stay alive until SIGINT/SIGTERM. The daemon's
        // signal-loop task and discovery already run in the
        // background; we just need to keep the process up.
        tracing::info!("running headless — Ctrl-C to exit");
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutdown requested");
        let mut m = manager.lock().await;
        if let Err(e) = m.shutdown().await {
            tracing::warn!(%e, "shutdown error");
        }
        return Ok(());
    }

    // Windowed mode: feed the in-process Freya app from `ui_rx`.
    run_window(manager, ui_rx);
    Ok(())
}

fn run_window(stream_manager: Arc<Mutex<StreamManager>>, mut ui_rx: mpsc::Receiver<ManagerEvent>) {
    let radio_station = RadioStation::create_global(AppState::default());
    let mut radio_events = radio_station.clone();

    launch(
        LaunchConfig::new()
            .with_future(move |_| async move {
                loop {
                    match ui_rx.recv().await {
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

/// Resolve `--device` (UUID or case-insensitive name) and start a
/// stream as soon as the matching device is discovered. Times out
/// after 30s of not seeing it — gives slow mDNS / SSDP discoveries
/// time to settle without hanging the daemon forever.
async fn auto_stream(
    manager: Arc<Mutex<StreamManager>>,
    ident: String,
    source: Option<cli::SourceKind>,
) -> anyhow::Result<()> {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
    let id = loop {
        if let Some(uuid) = match_device(&manager, &ident).await {
            break uuid;
        }
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!("no device matching {ident:?} appeared within 30s");
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    };

    let cap_source = match source {
        Some(cli::SourceKind::Screen) | None => CaptureSource::FullScreen { monitor: None },
        Some(cli::SourceKind::Window) => CaptureSource::Window { identifier: None },
    };

    let m = manager.lock().await;
    let capture = NativeCapture::new();
    let encoder = H264Encoder::default();
    let config = StreamConfig::default();
    m.start_stream(id, cap_source, capture, encoder, config)
        .await?;
    tracing::info!(device = %ident, "auto-stream started");
    Ok(())
}

async fn match_device(manager: &Arc<Mutex<StreamManager>>, ident: &str) -> Option<Uuid> {
    if let Ok(uuid) = Uuid::parse_str(ident) {
        return Some(uuid);
    }
    let needle = ident.to_lowercase();
    let m = manager.lock().await;
    for d in m.devices().await {
        if d.name.to_lowercase() == needle {
            return Some(d.id);
        }
    }
    None
}

async fn run_client(cmd: Command) -> anyhow::Result<()> {
    match cmd {
        Command::List { watch } => client::list(watch).await,
        Command::Stream { device, source } => client::stream(device, source).await,
        Command::Stop { device } => client::stop(device).await,
        Command::Introspect => {
            client::introspect();
            Ok(())
        }
    }
}

fn init_tracing_for_daemon() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new(
                "warn,freya=off,freya_core=off,freya_winit=off,ragnarok=off,\
                 ferricast_chromecast=info,ferricast_encoder=info,ferricast_gui=info",
            )
        }))
        .init();
}

fn init_tracing_for_client() {
    // Quieter default — a CLI client shouldn't paint the terminal
    // green by default. Users can still bump it with RUST_LOG.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("error")))
        .init();
}
