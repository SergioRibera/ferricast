use std::sync::Arc;

use clap::Parser;
use ferricast::ManagerEvent;
use ferricast::prelude::*;
use freya::{prelude::*, radio::*};
use tokio::sync::{Mutex, mpsc};
use tracing_subscriber::EnvFilter;

mod app;
mod cli;
mod client;
mod daemon;

use crate::app::*;
use crate::cli::{Cli, Command};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Cli::parse();

    if args.background && args.command.is_some() {
        return Err(anyhow::Error::msg(
            "Any command has conflicts with --background",
        ));
    }

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

    // Source enumerator: picks wlroots → x11 → stub at runtime. The
    // daemon doesn't need to know which backend it is — the trait
    // hides everything. Capability discovery happens over D-Bus.
    let enumerator = ferricast::capture::auto_enumerator();
    tracing::info!(
        backend = enumerator.backend_name(),
        capabilities = ?enumerator.capabilities(),
        "source enumerator selected"
    );

    // Fan-out: the daemon owns the original receiver and forwards a
    // clone of each event into `ui_rx` so the in-process window sees
    // the same stream the bus does, in the same order.
    let (ui_tx, ui_rx) = mpsc::channel::<ManagerEvent>(256);
    let _conn = daemon::start(manager.clone(), enumerator, manager_events, Some(ui_tx))
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

async fn run_client(cmd: Command) -> anyhow::Result<()> {
    match cmd {
        Command::List { watch } => client::list(watch).await,
        Command::Stream { device, source } => client::stream(device, source).await,
        Command::Stop { device } => client::stop(device).await,
        Command::Monitors { watch } => client::monitors(watch).await,
        Command::Windows { watch } => client::windows(watch).await,
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
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("error")),
        )
        .init();
}
