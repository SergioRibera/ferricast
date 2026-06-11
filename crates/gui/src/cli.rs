//! `clap`-derived CLI surface for `ferricast-gui`.
//!
//! The binary has three modes of operation; which one runs is
//! determined entirely by what was passed on the command line:
//!
//! 1. **GUI + daemon** (no flags, no subcommand) — open the Freya
//!    window and publish the D-Bus service in the same process.
//! 2. **Daemon-only** (`--background`) — same daemon, no window.
//!    Optionally auto-start a stream via `--device` / `--source`.
//! 3. **Client** (any subcommand) — talk to an already-running
//!    daemon over D-Bus. Fails with a clear message if nothing
//!    owns the bus name.

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser)]
#[command(
    name = "ferricast-gui",
    version,
    about = "Ferricast — desktop window + D-Bus daemon",
    long_about = "Without arguments, opens the desktop window and publishes the \
                  ferricast D-Bus service (rs.sergioribera.ferricast on the session bus). \
                  Use --background for a headless daemon, or one of the subcommands \
                  to act as a client of a running daemon."
)]
pub struct Cli {
    /// Run as a headless daemon without opening the desktop window.
    #[arg(long)]
    pub background: bool,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum SourceKind {
    /// Full-screen capture.
    Screen,
    /// Window capture (portal picks the window on Wayland).
    Window,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// List devices currently known to the daemon.
    List {
        /// Stream new/removed device events instead of exiting after
        /// the initial snapshot.
        #[arg(long)]
        watch: bool,
    },

    /// Start streaming to a device via the running daemon.
    Stream {
        /// UUID or case-insensitive device name.
        #[arg(value_name = "DEVICE")]
        device: String,

        /// Capture source. Omit for daemon-chosen default.
        #[arg(long, value_enum)]
        source: Option<SourceKind>,
    },

    /// Stop the active stream on a device.
    Stop {
        /// UUID or case-insensitive device name.
        #[arg(value_name = "DEVICE")]
        device: String,
    },

    /// List monitors visible to the daemon.
    Monitors {
        /// Stream `MonitorsChanged` events instead of exiting.
        #[arg(long)]
        watch: bool,
    },

    /// List top-level windows visible to the daemon. Same caveats
    /// as `monitors`.
    Windows {
        /// Stream `WindowsChanged` events instead of exiting.
        #[arg(long)]
        watch: bool,
    },

    /// Capture a one-shot PNG preview of a monitor or window from
    /// the daemon. Writes to `--output` (or stdout if omitted) so it
    /// composes with `feh -`, `wl-copy`, etc.
    Thumb {
        /// What to capture.
        #[arg(value_enum)]
        kind: ThumbKind,
        /// Id from `ferricast-gui monitors` / `windows`.
        #[arg(value_name = "ID")]
        id: String,
        /// Maximum PNG width — aspect ratio is preserved.
        #[arg(long, default_value_t = 640)]
        max_width: u32,
        #[arg(long, default_value_t = 360)]
        max_height: u32,
        /// Write the PNG to this path. Defaults to stdout (binary).
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
    },

    /// Print the D-Bus introspection XML for the daemon's
    /// `Manager1` interface — feed it into `gdbus-codegen`, `pydbus`,
    /// etc.
    Introspect,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum ThumbKind {
    Monitor,
    Window,
}
