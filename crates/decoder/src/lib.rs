//! Receiver-side decoders.
//!
//! Two unrelated trees:
//! * [`h264`] — backend-agnostic H.264 facade. Auto-selects NVDEC →
//!   VA-API → openh264 at first `configure()`. Mirror of the
//!   encoder's `H264Encoder` facade.
//! * [`aac`] — pure-Rust AAC-LC decoder over symphonia.
//!
//! Each module re-exports its facade type; the typical receiver
//! pipeline registers both with the manager:
//!
//! ```ignore
//! use ferricast::prelude::*;
//! use ferricast_decoder::{H264Decoder, AacDecoder};
//!
//! let manager = StreamManager::builder()
//!     .register_video_decoder::<H264Decoder>()
//!     .register_audio_decoder::<AacDecoder>()
//!     ...;
//! ```

pub mod h264;

#[cfg(feature = "nvdec-decode")]
pub mod nvdec;

#[cfg(feature = "aac")]
pub mod aac;

#[cfg(feature = "aac")]
pub use aac::AacDecoder;
pub use h264::H264Decoder;
