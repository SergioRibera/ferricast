//! Receiver-side decoders.
//!
//! One facade per codec ([`H264Decoder`], [`H265Decoder`], [`Vp8Decoder`],
//! [`Vp9Decoder`], [`Av1Decoder`], [`AacDecoder`], [`AacEldDecoder`],
//! [`AlacDecoder`], [`OpusDecoder`], [`PcmDecoder`]), each selecting its
//! backend from its configuration's [`ferricast_core::BackendSelection`]
//! (see [`ferricast_codec`]). `Foo::backends()` reports which backends
//! this build has and whether each can run here.
//!
//! The typical receiver pipeline registers them with the manager:
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

#[cfg(feature = "aac")]
pub mod aac;
#[cfg(feature = "alac-decode")]
mod alac;
#[cfg(feature = "av1-decode")]
mod av1;
mod facades;
pub mod h264;
pub mod h265;
mod nv12;
#[cfg(feature = "nvdec-decode")]
pub mod nvdec;
#[cfg(feature = "opus-decode")]
mod opus;
mod pcm;
mod platform;
#[cfg(feature = "vp8-decode")]
mod vp8;
#[cfg(feature = "vp9-decode")]
mod vp9;

pub use facades::{
    AacDecoder, AacEldDecoder, AlacDecoder, Av1Decoder, OpusDecoder, PcmDecoder, Vp8Decoder,
    Vp9Decoder,
};
pub use ferricast_codec::{Availability, BackendReport};
pub use h264::H264Decoder;
pub use h265::H265Decoder;
