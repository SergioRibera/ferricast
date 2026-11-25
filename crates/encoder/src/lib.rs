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
#[cfg(any(
    feature = "vaapi",
    all(feature = "nvenc", any(target_os = "linux", target_os = "windows"))
))]
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
#[cfg(all(feature = "nvenc", any(target_os = "linux", target_os = "windows")))]
pub mod nvenc;
pub mod opus;
mod platform;
// The scaler is crate-private: an audio-only build leaves it unused.
#[allow(dead_code, reason = "conditional: used by the video encoders only")]
pub mod scale;

pub use ferricast_codec::{Availability, BackendReport};

/// DRM format modifiers the GPU encoders on this machine encode from
/// without a copy, for screen capture to prefer when it allocates or
/// negotiates DMA-BUFs (`CaptureConfig::consumer_modifiers`). Empty when
/// no such encoder is built in or usable (today: NVENC on Linux).
#[must_use]
pub fn dmabuf_modifiers() -> ferricast_core::DmaBufModifiers {
    #[cfg(all(feature = "nvenc", target_os = "linux"))]
    return nvenc::dmabuf_modifiers();
    #[cfg(not(all(feature = "nvenc", target_os = "linux")))]
    ferricast_core::DmaBufModifiers::default()
}

/// Whether a GPU encoder on this machine turns rotated or flipped GPU
/// frames itself (`CaptureConfig::consumer_orients`): NVENC on Linux,
/// whose GPU fit applies `GpuFrame::orientation`.
#[must_use]
pub fn orients_gpu_frames() -> bool {
    #[cfg(all(feature = "nvenc", target_os = "linux"))]
    return nvenc::orients_gpu_frames();
    #[cfg(not(all(feature = "nvenc", target_os = "linux")))]
    false
}

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

/// Adds NVENC for `codec`: the first candidate of automatic selection
/// (on a laptop with an NVIDIA dGPU and an AMD / Intel iGPU, VA-API would
/// pick the iGPU and move every captured frame across `PCIe` twice).
/// Recorded as missing (naming its feature) when the build leaves it
/// out, and as elsewhere off Linux and Windows.
fn nvenc(
    registry: ferricast_codec::VideoEncoderRegistry,
    codec: ferricast_core::Codec,
) -> ferricast_codec::VideoEncoderRegistry {
    #[cfg(all(feature = "nvenc", any(target_os = "linux", target_os = "windows")))]
    let registry = nvenc::register(registry, codec);
    #[cfg(not(all(feature = "nvenc", any(target_os = "linux", target_os = "windows"))))]
    let registry = {
        let feature = if codec == ferricast_core::Codec::H265 {
            "ferricast-encoder/nvenc-hevc"
        } else {
            "ferricast-encoder/nvenc"
        };
        if cfg!(any(target_os = "linux", target_os = "windows")) {
            registry.missing(ferricast_core::BackendId::Nvenc, feature)
        } else {
            registry.elsewhere(
                ferricast_core::BackendId::Nvenc,
                feature,
                "only on Linux and Windows",
            )
        }
    };
    registry
}
