//! Sender-side encoders. Each codec facade ([`h264::H264Encoder`],
//! [`h265::H265Encoder`], [`aac::AacEncoder`], [`opus::OpusEncoder`])
//! selects its backend from its configuration's
//! [`ferricast_core::BackendSelection`]; see [`ferricast_codec`].

pub mod aac;
#[cfg(any(
    feature = "fdk-aac",
    feature = "opus-rs",
    all(target_os = "android", feature = "mediacodec"),
    all(any(target_os = "macos", target_os = "ios"), feature = "audiotoolbox"),
    all(target_os = "windows", feature = "media-foundation")
))]
mod audio_clock;
#[cfg(any(feature = "vaapi", feature = "nvenc"))]
mod config_ext;
mod frame_view;
#[cfg(feature = "vaapi")]
mod gop;
pub mod h264;
pub mod h265;
#[cfg_attr(
    not(any(
        all(target_os = "android", feature = "mediacodec"),
        all(target_os = "windows", feature = "media-foundation")
    )),
    allow(dead_code, reason = "used by the NV12-input platform encoders only")
)]
mod nv12;
#[cfg(feature = "nvenc")]
pub mod nvenc;
pub mod opus;
#[cfg(any(test, all(target_os = "windows", feature = "media-foundation")))]
mod platform;
// The scaler is crate-private: an audio-only build leaves it unused.
#[allow(dead_code, reason = "conditional: used by the video encoders only")]
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
