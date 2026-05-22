//! Ferricast — unified screen and video streaming for Linux.
//!
//! `ferricast` is a high-level façade over the protocol, capture and
//! encoder crates that make up the workspace. The intended usage is:
//!
//! 1. Build a [`StreamManager`] with the protocols you want to support.
//! 2. Start discovery and consume [`ManagerEvent`]s from the channel.
//! 3. Call [`StreamManager::start_stream`] when the user picks a device.
//!
//! ```no_run
//! use ferricast::prelude::*;
//!
//! # async fn run() -> Result<()> {
//! let (mut manager, mut events) = StreamManager::builder()
//!     .with_chromecast()
//!     .build_with_events();
//!
//! manager.start_discovery().await?;
//!
//! while let Some(event) = events.recv().await {
//!     match event {
//!         ManagerEvent::DeviceFound(device) => {
//!             println!("found {} ({})", device.name, device.protocol);
//!         }
//!         ManagerEvent::DeviceLost(id) => println!("lost {id}"),
//!         _ => {}
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Modules
//!
//! - [`capture`] — screen-capture backends (PipeWire, X11, native picker).
//! - [`encoder`] — video encoders (openh264, VA-API, NVENC) behind a unified trait.
//! - [`protocols`] — receiver protocols (Chromecast today; DIAL/AirPlay/Miracast pending).
//! - [`prelude`] — the everyday types most callers want in scope.

mod manager;

pub use manager::{ManagerEvent, StreamManager, StreamManagerBuilder};

// Re-export the entirety of `ferricast-core`. This crate is the
// trait + type contract that every other crate in the workspace
// implements against — keeping it at the root means downstream
// users never have to depend on `ferricast-core` directly.
pub use ferricast_core::*;

/// Screen-capture backends.
///
/// Most callers want [`capture::NativeCapture`], which auto-selects
/// PipeWire on Wayland and X11 elsewhere.
pub mod capture {
    pub use ferricast_capture::*;
}

/// Video encoders.
///
/// The default H.264 entry point is [`encoder::h264::H264Encoder`].
/// At runtime it negotiates NVENC → VA-API → openh264 in order, so the
/// same handle works across hardware without any caller changes.
pub mod encoder {
    pub use ferricast_encoder::*;
}

/// HLS server primitives — exposed mainly so protocols outside this
/// workspace can reuse the segmenter / adaptive controller. End
/// users typically don't touch this module directly.
pub mod hls {
    pub use ferricast_hls::*;
}

/// Receiver protocols.
///
/// Each submodule re-exports the protocol's [`ProtocolHandler`]
/// implementation under a short name. To register one with a manager,
/// either pass the handler to [`StreamManager::register`] or use the
/// matching `with_*` method on [`StreamManagerBuilder`].
pub mod protocols {
    #[cfg(feature = "chromecast")]
    pub use ferricast_chromecast::ChromecastHandler as Chromecast;
}

/// Everything you typically want in scope to talk to Ferricast.
///
/// ```
/// use ferricast::prelude::*;
/// ```
pub mod prelude {
    pub use crate::capture::NativeCapture;
    pub use crate::encoder::h264::H264Encoder;
    pub use crate::{
        CaptureSource, Codec, Device, DeviceCapabilities, FerricastError, ManagerEvent,
        ProtocolHandler, Result, StreamConfig, StreamManager, StreamManagerBuilder,
    };
}
