use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use clap::Parser;
use ferricast::ManagerEvent;
use ferricast::prelude::*;
use freya::{prelude::*, radio::*};
use tokio::sync::{Mutex, mpsc};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

use crate::app::ReceiverWindowReq;
use crate::daemon::PickerRequest;

mod app;
mod cli;
mod client;
mod daemon;
mod picker;
mod receiver_window;

use crate::app::*;
use crate::cli::{Cli, Command};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Cli::parse();

    if args.background && args.command.is_some() {
        // The two modes serve opposite roles — `--background` runs
        // the daemon side that owns the bus name; a subcommand is a
        // client that talks to a daemon that's already up. Trying
        // to combine them would either race the well-known name or
        // pointlessly stand up a second daemon. Bail with a hint
        // pointing at the workflow we actually expect.
        return Err(anyhow::anyhow!(
            "--background and subcommands are mutually exclusive: \
             `--background` starts the headless daemon (and owns the \
             D-Bus name), while subcommands like `list` / `stream` / \
             `monitors` are clients of an already-running daemon. \
             Start the daemon in one terminal with `ferricast-gui \
             --background`, then run the subcommand in another."
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

    // Receiver-window channel: pump → App. App opens windows when
    // requests come in. Capacity 16 because casts come in bursts on
    // a busy LAN but don't realistically queue.
    let (receiver_window_tx, receiver_window_rx) = mpsc::channel::<ReceiverWindowReq>(16);
    let receiver_window_tx_factory = receiver_window_tx.clone();

    let (manager, manager_events) = StreamManager::builder()
        .with_chromecast()
        .with_chromecast_receiver()
        .with_h264_decoder()
        .with_aac_decoder()
        // Sink factory: per-session, allocate video+audio channels
        // and immediately request a window via
        // `receiver_window_tx`. Returns a `WindowSink` that the
        // pipeline pumps into. Pairing the channel allocation with
        // the window open here (instead of via ReceiverStarted
        // event) keeps the rx halves on a single ownership path —
        // sink owns tx, window owns rx, no shared registry.
        .set_sink_factory(move |remote, info| {
            let label = format!(
                "{}{}",
                remote
                    .name
                    .clone()
                    .unwrap_or_else(|| remote.addr.to_string()),
                if info.video.is_some() { " (video)" } else { " (audio)" },
            );
            let counters = Arc::new(receiver_window::ReceiverCounters::default());
            // Channel depths sized to absorb window-startup latency
            // (Freya + Skia + rodio + alsa init takes several
            // hundred ms before the drain tasks start consuming).
            // Video is `try_send` on the sink side — capacity 4 is
            // just a tiny coalescing buffer, the latest frame wins.
            // Audio used to be `send().await` with capacity 64; that
            // path could backpressure the puller into stalling and
            // losing segments to ring eviction. We now `try_send`
            // audio too (with a generous 256-slot buffer ≈ 5 s of
            // playback), trading occasional audio dropouts under
            // sustained overload for a pipeline that never wedges.
            let (video_tx, video_rx) = mpsc::channel::<receiver_window::VideoFrame>(4);
            let (audio_tx, audio_rx) = mpsc::channel::<receiver_window::AudioBuf>(256);
            let req = ReceiverWindowReq {
                remote: remote.clone(),
                info: info.clone(),
                counters: counters.clone(),
                video_rx,
                audio_rx,
            };
            let tx = receiver_window_tx_factory.clone();
            async move {
                if tx.send(req).await.is_err() {
                    tracing::warn!(
                        receiver = %label,
                        "sink_factory: window request channel closed — \
                         pipeline will run without a window"
                    );
                }
                Ok(Box::new(receiver_window::WindowSink::new(
                    label, counters, video_tx, audio_tx,
                )) as Box<dyn ferricast::FrameSink>)
            }
        })
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

    // Picker delegation channel: the daemon's `StartStream` handler
    // pushes a request here when the D-Bus caller didn't pick a
    // concrete source, and the Freya app drains it to open the
    // picker window. Capacity 4 because requests are user-driven —
    // bursts higher than that mean a buggy client.
    let (picker_tx, picker_rx) = mpsc::channel::<PickerRequest>(4);

    let _conn = daemon::start(
        manager.clone(),
        enumerator,
        manager_events,
        Some(ui_tx),
        Some(picker_tx),
    )
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

    // Start receivers (Chromecast advertise + accept). Idempotent;
    // running on its own task so a slow mDNS bind doesn't block the
    // window from coming up.
    {
        let sm = manager.clone();
        tokio::spawn(async move {
            if let Err(e) = sm.lock().await.start_receivers().await {
                tracing::error!(%e, "Failed to start receivers");
            } else {
                tracing::info!("Receivers advertising on the LAN");
            }
        });
    }

    // Both modes mount the Freya runtime so the picker window can
    // be opened on demand from the D-Bus path. `--background` just
    // hides the main window — the user only sees Ferricast UI when
    // the picker pops, which is what they want for headless
    // workflows that drive the daemon over the bus.
    if args.background {
        tracing::info!(
            "running with hidden main window — Freya alive to host the picker; \
             Ctrl-C to exit"
        );
    }
    run_window(
        manager,
        ui_rx,
        picker_rx,
        receiver_window_rx,
        args.background,
    );
    Ok(())
}

fn run_window(
    stream_manager: Arc<Mutex<StreamManager>>,
    mut ui_rx: mpsc::Receiver<ManagerEvent>,
    picker_rx: mpsc::Receiver<PickerRequest>,
    receiver_window_rx: mpsc::Receiver<ReceiverWindowReq>,
    hidden: bool,
) {
    let radio_station = RadioStation::create_global(AppState::default());
    let mut radio_events = radio_station.clone();

    let mut window_config = WindowConfig::new_app(FerricastApp {
        stream_manager,
        radio_station,
        // `Rc<RefCell<Option<_>>>` because the receiver isn't
        // `Clone` and `App::render` takes `&self`. The first render
        // `take()`s it from inside `use_hook` and starts the
        // listener; subsequent renders see `None` and skip.
        picker_req_rx: Rc::new(RefCell::new(Some(picker_rx))),
        receiver_req_rx: Rc::new(RefCell::new(Some(receiver_window_rx))),
    })
    .with_title("Ferricast")
    .with_size(800., 600.);
    if hidden {
        // `--background`: open the main window invisible. winit
        // accepts `with_visible(false)` at creation; the surface
        // never appears in the user's task switcher / task bar.
        // Picker windows opened via `Platform::launch_window` are
        // separate top-levels and DO appear normally — that's the
        // whole point of this mode.
        window_config = window_config.with_window_attributes(|attrs, _| attrs.with_visible(false));
    }

    launch(
        LaunchConfig::new()
            .with_future(move |proxy| async move {
                // Two concurrent jobs on the Freya local runtime:
                // - drain the manager-events channel into the radio
                //   station so the device list updates live;
                // - listen for Ctrl-C and request a clean exit so
                //   `--background` (hidden main window) can still
                //   be terminated from the launching terminal.
                let ctrl_c = async {
                    let _ = tokio::signal::ctrl_c().await;
                    tracing::info!("Ctrl-C received, exiting");
                    let _ = proxy
                        .post_callback(|ctx| {
                            ctx.exit();
                        })
                        .await;
                };
                let pump = async {
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
                            Some(ManagerEvent::ReceiverIncoming { protocol, remote, .. }) => {
                                tracing::info!(
                                    protocol,
                                    sender = %remote.addr,
                                    "receiver: sender connected — waiting for LOAD"
                                );
                            }
                            Some(ManagerEvent::ReceiverStarted { remote, info, .. }) => {
                                // Window has already been requested
                                // from `sink_factory`; this event is
                                // diagnostic only.
                                tracing::info!(
                                    sender = %remote.addr,
                                    audio = info.audio.is_some(),
                                    video = info.video.is_some(),
                                    "receiver: LOAD acknowledged, pipeline running"
                                );
                            }
                            Some(ManagerEvent::ReceiverStateChanged {
                                receiver_id,
                                state,
                            }) => {
                                tracing::debug!(?receiver_id, ?state, "receiver state");
                            }
                            Some(ManagerEvent::ReceiverStopped { receiver_id })
                            | Some(ManagerEvent::ReceiverError { receiver_id, .. }) => {
                                // TODO follow-up: close the
                                // associated receiver window here.
                                // Tracking the WindowId requires the
                                // app listener to bubble it back
                                // (`launch_window` returns the id
                                // on the freya local thread only) —
                                // small wire-up out of scope this
                                // turn. User closes the window
                                // manually in the meantime.
                                let _ = receiver_id;
                            }
                            None => break,
                            _ => {}
                        }
                    }
                };
                tokio::join!(ctrl_c, pump);
            })
            .with_window(window_config),
    );
}

async fn run_client(cmd: Command) -> anyhow::Result<()> {
    match cmd {
        Command::List { watch } => client::list(watch).await,
        Command::Stream { device, source } => client::stream(device, source).await,
        Command::Stop { device } => client::stop(device).await,
        Command::Monitors { watch } => client::monitors(watch).await,
        Command::Windows { watch } => client::windows(watch).await,
        Command::Thumb {
            kind,
            id,
            max_width,
            max_height,
            output,
        } => client::thumb(kind, id, max_width, max_height, output).await,
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
