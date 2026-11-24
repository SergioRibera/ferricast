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

/// Adds every encode job of this crate to an istmo codec catalog.
#[cfg(feature = "istmo")]
#[must_use]
pub fn catalog(catalog: ferricast_codec::plugin::Catalog) -> ferricast_codec::plugin::Catalog {
    use ferricast_core::{AudioCodec, Codec};
    catalog
        .video_encoder(Codec::H264, h264::H264Encoder::registry)
        .video_encoder(Codec::H265, h265::H265Encoder::registry)
        .audio_encoder(AudioCodec::Aac, aac::AacEncoder::registry)
        .audio_encoder(AudioCodec::AacEld, aac::AacEldEncoder::registry)
        .audio_encoder(AudioCodec::Opus, opus::OpusEncoder::registry)
}
