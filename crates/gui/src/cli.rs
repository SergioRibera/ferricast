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
    #[arg(long, conflicts_with = "command")]
    pub background: bool,

    /// Auto-start a stream on this device once it appears in discovery.
    /// Accepts a UUID or a (case-insensitive) device name.
    ///
    /// Only meaningful in daemon modes (`--background` or windowed).
    #[arg(long, value_name = "ID_OR_NAME", conflicts_with = "command")]
    pub device: Option<String>,

    /// What to share when `--device` triggers an auto-start. If
    /// omitted the daemon picks: PipeWire portal on Wayland; an
    /// X11 picker dialog (TODO) on X11.
    #[arg(long, value_enum, conflicts_with = "command")]
    pub source: Option<SourceKind>,

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

impl SourceKind {
    pub fn as_kind_str(self) -> &'static str {
        match self {
            Self::Screen => "screen",
            Self::Window => "window",
        }
    }
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

    /// Print the D-Bus introspection XML for the daemon's
    /// `Manager1` interface — feed it into `gdbus-codegen`, `pydbus`,
    /// etc.
    Introspect,
}
