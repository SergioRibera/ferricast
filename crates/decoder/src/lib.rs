//! Receiver-side decoders.
//!
//! Two unrelated trees:
//! * [`h264`] — backend-agnostic H.264 facade. Auto-selects NVDEC →
//!   VA-API → openh264 at first `configure()`. Mirror of the
//!   encoder's `H264Encoder` facade.
//! * [`aac`] — AAC decoder: libfdk-aac (HE-AAC v1/v2 too) when built in,
//!   else pure-Rust AAC-LC over symphonia.
//! * [`PcmDecoder`] — pass-through for 16-bit PCM packets.
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

mod nv12;
mod pcm;

#[cfg(any(
    feature = "nvdec-hevc-decode",
    feature = "vaapi-hevc-decode",
    feature = "hevc-decode"
))]
pub mod h265;

#[cfg(feature = "nvdec-decode")]
pub mod nvdec;

#[cfg(any(feature = "nvdec-decode", feature = "vaapi-decode"))]
mod probation;

#[cfg(feature = "aac")]
pub mod aac;

#[cfg(feature = "alac-decode")]
mod alac;

#[cfg(feature = "av1-decode")]
mod av1;

#[cfg(feature = "opus-decode")]
mod opus;

#[cfg(feature = "vp8-decode")]
mod vp8;

#[cfg(feature = "vp9-decode")]
mod vp9;

#[cfg(feature = "aac")]
pub use aac::AacDecoder;
#[cfg(feature = "fdk-aac-decode")]
pub use aac::AacEldDecoder;
#[cfg(feature = "alac-decode")]
pub use alac::AlacDecoder;
#[cfg(feature = "av1-decode")]
pub use av1::Av1Decoder;
pub use h264::H264Decoder;
#[cfg(any(
    feature = "nvdec-hevc-decode",
    feature = "vaapi-hevc-decode",
    feature = "hevc-decode"
))]
pub use h265::H265Decoder;
#[cfg(feature = "opus-decode")]
pub use opus::OpusDecoder;
pub use pcm::PcmDecoder;
#[cfg(feature = "vp8-decode")]
pub use vp8::Vp8Decoder;
#[cfg(feature = "vp9-decode")]
pub use vp9::Vp9Decoder;
