//! Miracast (Wi-Fi Display) sender + receiver.
//!
//! Three layers:
//!
//! - [`parameters`], [`rtsp`], [`ts`], [`rtp`], [`ie`] — pure protocol
//!   primitives, no I/O. The wire-level reference for every byte that
//!   goes on the air lives in these modules' tests.
//! - [`transport`] — platform-specific bring-up for the control and
//!   media sockets. [`transport::mice`] (MS-MICE over mDNS + TCP:7250)
//!   is portable; [`transport::linux_p2p`] drives wpa_supplicant via
//!   D-Bus; [`transport::windows_native`] bridges to WinRT's
//!   `Windows.Media.Casting` stack.
//! - [`handler`] (sender) and [`receiver`] (sink) — the ferricast-core
//!   trait impls that wire everything into [`StreamManager`].
//!
//! The split lets the pure stack be exercised in loopback tests without
//! any radio hardware and lets macOS participate at all — Apple exposes
//! no Wi-Fi Direct API, so the MICE path is the only option there.
//!
//! [`StreamManager`]: ferricast_core

pub mod discovery;
pub mod handler;
pub mod ie;
pub mod parameters;
pub mod receiver;
pub mod rtp;
pub mod rtsp;
pub mod session;
pub mod transport;
pub mod ts;

pub use discovery::MiracastDiscovery;
pub use handler::MiracastHandler;
pub use receiver::{MiracastReceiver, MiracastReceiverAdvertiser, MiracastReceiverControl};
pub use session::MiracastSession;
