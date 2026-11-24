//! Sender-side encoders. Each codec facade ([`h264::H264Encoder`],
//! [`h265::H265Encoder`], [`aac::AacEncoder`], [`opus::OpusEncoder`])
//! selects its backend from its configuration's
//! [`ferricast_core::BackendSelection`]; see [`ferricast_codec`].

pub mod aac;
#[cfg(any(feature = "fdk-aac", feature = "opus-rs"))]
mod audio_clock;
#[cfg(any(feature = "vaapi", feature = "nvenc"))]
mod config_ext;
mod frame_view;
#[cfg(feature = "vaapi")]
mod gop;
pub mod h264;
pub mod h265;
#[cfg(feature = "nvenc")]
pub mod nvenc;
pub mod opus;
mod platform;
pub mod scale;

pub use ferricast_codec::{Availability, BackendReport};
