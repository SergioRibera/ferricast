use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use clap::Parser;
use ferricast::ManagerEvent;
use ferricast::prelude::*;
use freya::{prelude::*, radio::*};
use tokio::sync::mpsc;
use tracing_subscriber::{EnvFilter, Layer, layer::SubscriberExt, util::SubscriberInitExt};

use crate::app::ReceiverWindowReq;
use crate::daemon::{PickerRequest, SharedManager};

mod app;
mod cli;
mod client;
mod daemon;
mod log_file;
mod pairing;
mod picker;
mod receiver_window;

use crate::app::*;
use crate::cli::{Cli, Command};

fn main() -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let Some(daemon) = runtime.block_on(start())? else {
        return Ok(());
    };
    {
        // The window's event loop must run outside `block_on`. Inside
        // it, every tokio resource polled on this thread draws from the
        // one cooperative budget of the `block_on` future, which is
        // never refilled while the loop runs: once spent, each poll
        // returns `Pending` and wakes itself again, and the event pump
        // floods the loop with wake-ups (exponentially, as `join!`
        // wakes on both branches) until memory runs out.
        let _runtime = runtime.enter();
        run_window(
            daemon.manager.clone(),
            daemon.ui_rx,
            daemon.picker_rx,
            daemon.receiver_window_rx,
            daemon.hidden,
        );
    }
    runtime.block_on(shutdown_manager(&daemon.manager));
    Ok(())
}

/// What the window needs from the start-up, which runs on the runtime.
struct Daemon {
    manager: SharedManager,
    ui_rx: mpsc::Receiver<ManagerEvent>,
    picker_rx: mpsc::Receiver<PickerRequest>,
    receiver_window_rx: mpsc::Receiver<ReceiverWindowReq>,
    hidden: bool,
    /// Owns the bus name for as long as the window runs.
    _bus: zbus::Connection,
}

/// Sets up the manager, the D-Bus service, discovery and receivers, or
/// runs a client subcommand (returning `None`).
async fn start() -> anyhow::Result<Option<Daemon>> {
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
        return run_client(cmd).await.map(|()| None);
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

    let manager_builder = StreamManager::builder();
    #[cfg(feature = "chromecast")]
    let manager_builder = manager_builder.with_chromecast();
    #[cfg(feature = "airplay")]
    let manager_builder = manager_builder.with_airplay();
    #[cfg(feature = "chromecast")]
    let manager_builder = manager_builder.with_chromecast_receiver();
    #[cfg(feature = "airplay")]
    let manager_builder = if args.no_airplay_receiver {
        manager_builder
    } else {
        let receiver =
            ferricast::airplay::AirPlayReceiver::new(ferricast::airplay::AirPlayReceiverConfig {
                name: args.receiver_name.clone(),
                require_pin: args.airplay_pin,
                video: !args.no_airplay_video,
                ..Default::default()
            })
            .await;
        manager_builder.with_airplay_receiver(receiver)
    };
    let manager_builder = manager_builder.with_h264_decoder();
    #[cfg(feature = "aac")]
    let manager_builder = manager_builder.with_aac_decoder();
    #[cfg(feature = "hevc-decode")]
    let manager_builder =
        manager_builder.register_video_decoder::<ferricast::decoder::H265Decoder>();
    #[cfg(feature = "vp8-decode")]
    let manager_builder =
        manager_builder.register_video_decoder::<ferricast::decoder::Vp8Decoder>();
    #[cfg(feature = "vp9-decode")]
    let manager_builder =
        manager_builder.register_video_decoder::<ferricast::decoder::Vp9Decoder>();
    #[cfg(feature = "av1-decode")]
    let manager_builder =
        manager_builder.register_video_decoder::<ferricast::decoder::Av1Decoder>();
    #[cfg(feature = "opus-decode")]
    let manager_builder =
        manager_builder.register_audio_decoder::<ferricast::decoder::OpusDecoder>();
    // Audio files the demuxer decodes itself (MP3, FLAC, Vorbis, WAV)
    // arrive as 16-bit PCM.
    let manager_builder =
        manager_builder.register_audio_decoder::<ferricast::decoder::PcmDecoder>();
    let mut manager_builder = manager_builder;

    if let (Some(min), Some(max)) = (args.port_min, args.port_max) {
        if max < min {
            return Err(anyhow::anyhow!(
                "--port-max ({max}) must be ≥ --port-min ({min})"
            ));
        }
        manager_builder = manager_builder.with_port_range(min, max);
        tracing::info!(port_min = min, port_max = max, "protocol port range set");
    }

    let (manager, manager_events) = manager_builder
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
                if info.video.is_some() {
                    " (video)"
                } else {
                    " (audio)"
                },
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
    let manager: SharedManager = Arc::new(tokio::sync::RwLock::new(manager));

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
            if let Err(e) = sm.read().await.start_discovery().await {
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
            if let Err(e) = sm.write().await.start_receivers().await {
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
    Ok(Some(Daemon {
        manager,
        ui_rx,
        picker_rx,
        receiver_window_rx,
        hidden: args.background,
        _bus: _conn,
    }))
}

/// Stop streams (receivers get a STOP, nothing keeps showing our
/// splash), receivers and discovery (mDNS goodbye) before exiting.
async fn shutdown_manager(manager: &SharedManager) {
    const WRITE_LOCK_GRACE: std::time::Duration = std::time::Duration::from_secs(5);
    tracing::info!("shutting down streams, receivers and discovery");
    // Cancels in-flight starts too, which releases their read locks.
    manager.read().await.stop_all_streams().await;
    match tokio::time::timeout(WRITE_LOCK_GRACE, manager.write()).await {
        Ok(mut m) => {
            if let Err(e) = m.shutdown().await {
                tracing::warn!(%e, "manager shutdown");
            }
        }
        Err(_) => {
            tracing::warn!("manager still busy; stopping discovery only");
            if let Err(e) = manager.read().await.stop_discovery().await {
                tracing::warn!(%e, "stop discovery");
            }
        }
    }
}

fn run_window(
    stream_manager: SharedManager,
    mut ui_rx: mpsc::Receiver<ManagerEvent>,
    picker_rx: mpsc::Receiver<PickerRequest>,
    receiver_window_rx: mpsc::Receiver<ReceiverWindowReq>,
    hidden: bool,
) {
    let radio_station = RadioStation::create_global(AppState::default());
    let mut radio_events = radio_station;
    // Pump → window listener: pairing prompts and receiver lifecycle.
    let (ui_cmd_tx, ui_cmd_rx) = mpsc::channel::<UiCommand>(16);

    let mut window_config = WindowConfig::new_app(FerricastApp {
        stream_manager,
        radio_station,
        // `Rc<RefCell<Option<_>>>` because the receiver isn't
        // `Clone` and `App::render` takes `&self`. The first render
        // `take()`s it from inside `use_hook` and starts the
        // listener; subsequent renders see `None` and skip.
        picker_req_rx: Rc::new(RefCell::new(Some(picker_rx))),
        receiver_req_rx: Rc::new(RefCell::new(Some(receiver_window_rx))),
        ui_cmd_rx: Rc::new(RefCell::new(Some(ui_cmd_rx))),
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
            .with_task(move |proxy| async move {
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
                    while let Some(event) = ui_rx.recv().await {
                        let command = match &event {
                            ManagerEvent::PairingRequired {
                                device_id,
                                challenge,
                                respond,
                            } => radio_events
                                .write_channel(AppChannel::Notices)
                                .pairing_prompt(*device_id, challenge, respond),
                            ManagerEvent::ReceiverIncoming {
                                receiver_id,
                                protocol,
                                remote,
                            } => {
                                tracing::info!(
                                    protocol,
                                    sender = %remote.addr,
                                    "receiver: sender connected — waiting for LOAD"
                                );
                                Some(UiCommand::ReceiverSession {
                                    receiver_id: *receiver_id,
                                    remote_id: remote.id,
                                })
                            }
                            ManagerEvent::ReceiverPin { receiver_id, pin } => {
                                Some(UiCommand::ReceiverPin {
                                    receiver_id: *receiver_id,
                                    pin: pin.clone(),
                                })
                            }
                            ManagerEvent::ReceiverStarted { remote, info, .. } => {
                                // The window was already requested by
                                // the sink factory.
                                tracing::info!(
                                    sender = %remote.addr,
                                    audio = info.audio.is_some(),
                                    video = info.video.is_some(),
                                    "receiver: LOAD acknowledged, pipeline running"
                                );
                                None
                            }
                            ManagerEvent::ReceiverStateChanged { receiver_id, state } => {
                                tracing::debug!(?receiver_id, ?state, "receiver state");
                                None
                            }
                            ManagerEvent::ReceiverStopped { receiver_id }
                            | ManagerEvent::ReceiverError { receiver_id, .. } => {
                                Some(UiCommand::ReceiverEnded {
                                    receiver_id: *receiver_id,
                                })
                            }
                            _ => None,
                        };
                        if radio_events.peek().changed_by(&event) {
                            radio_events
                                .write_channel(AppChannel::Devices)
                                .apply(&event);
                        }
                        if let Some(command) = command
                            && ui_cmd_tx.send(command).await.is_err()
                        {
                            tracing::warn!("window listener gone; dropping UI command");
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
    // Shared filter: `RUST_LOG` drives BOTH console and file layers.
    // Without `RUST_LOG` we fall back to a quiet default suitable for
    // interactive use.
    let default_filter = "warn,freya=off,freya_core=off,freya_winit=off,\
                          ragnarok=off,ferricast=info,ferricast_chromecast=info,\
                          ferricast_encoder=info,ferricast_gui=info,\
                          ferricast_hls=info,ferricast_muxer=info,\
                          ferricast_capture=info,ferricast_airplay=info,\
                          ferricast_decoder=info,ferricast_demuxer=info";
    let make_filter =
        || EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter));

    let console_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_filter(make_filter());

    // File path resolution:
    //   * `FERRICAST_TRACE_LOG=` (empty)     → disable file output
    //   * `FERRICAST_TRACE_LOG=/path/file`  → explicit path
    //   * otherwise                          → OS cache dir
    //       Linux:   $XDG_CACHE_HOME or $HOME/.cache,  + /ferricast/ferricast.log
    //       macOS:   $HOME/Library/Caches/ferricast/ferricast.log
    //       Windows: alongside the running executable
    let file_path: Option<std::path::PathBuf> = match std::env::var("FERRICAST_TRACE_LOG") {
        Ok(p) if p.is_empty() => None,
        Ok(p) => Some(std::path::PathBuf::from(p)),
        Err(_) => default_trace_log_path(),
    };

    let file_layer = file_path.as_ref().and_then(|path| {
        let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!(
                "ferricast: cannot create trace log dir {}: {e}",
                parent.display()
            );
            return None;
        }
        let appender = match log_file::RotatingFile::open(path, log_file::MAX_LOG_BYTES) {
            Ok(file) => file,
            Err(e) => {
                eprintln!("ferricast: cannot open trace log {}: {e}", path.display());
                return None;
            }
        };
        let (writer, guard) = tracing_appender::non_blocking(appender);
        // Leak the guard so the worker thread lives for the whole
        // process; otherwise it drops at function exit and flushes
        // stop firing.
        Box::leak(Box::new(guard));
        Some(
            tracing_subscriber::fmt::layer()
                .with_writer(writer)
                .with_ansi(false)
                .with_target(true)
                .with_thread_ids(true)
                .with_filter(make_filter()),
        )
    });

    let registry = tracing_subscriber::registry().with(console_layer);
    if let Some(file) = file_layer {
        registry.with(file).init();
    } else {
        registry.init();
    }

    if let Some(p) = file_path {
        tracing::info!(trace_log = %p.display(), "trace log file enabled");
    }
}

/// Default trace-log path per platform. `None` only on platforms
/// where we couldn't resolve a sensible location.
fn default_trace_log_path() -> Option<std::path::PathBuf> {
    let dir = if cfg!(windows) {
        // Windows: same dir as the running .exe so the log is next to
        // the binary the user launched.
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))?
    } else if cfg!(target_os = "macos") {
        let home = std::env::var_os("HOME")?;
        std::path::PathBuf::from(home).join("Library/Caches/ferricast")
    } else {
        // Linux / BSD / other Unix → XDG cache.
        std::env::var_os("XDG_CACHE_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache"))
            })?
            .join("ferricast")
    };
    Some(dir.join("ferricast.log"))
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
