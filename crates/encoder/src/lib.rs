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
// VA-API plumbing the H.264 and HEVC encoders share (render node, VPP).
pub mod opus;
mod platform;
#[cfg(feature = "vaapi")]
#[allow(
    unsafe_code,
    reason = "vaQueryVideoProcPipelineCaps; the block carries a SAFETY comment"
)]
mod vaapi;
// The scaler is crate-private: an audio-only build leaves it unused.
#[allow(dead_code, reason = "conditional: used by the video encoders only")]
pub mod scale;

pub use ferricast_codec::{Availability, BackendReport};

/// DRM format modifiers the GPU encoder `config` would select encodes
/// from without a copy, for screen capture to prefer when it allocates or
/// negotiates DMA-BUFs (`CaptureConfig::consumer_modifiers`). Empty when
/// that encoder publishes none (VA-API, software) or none is usable
/// (today only NVENC on Linux lists its modifiers).
#[must_use]
pub fn dmabuf_modifiers(config: &ferricast_core::EncoderConfig) -> ferricast_core::DmaBufModifiers {
    match predicted_backend(config) {
        #[cfg(all(feature = "nvenc", target_os = "linux"))]
        Some(ferricast_core::BackendId::Nvenc) => {
            ferricast_codec::cuda::CudaDevice::for_config(config, ferricast_core::BackendId::Nvenc)
                .map(nvenc::dmabuf_modifiers_on)
                .unwrap_or_default()
        }
        _ => ferricast_core::DmaBufModifiers::default(),
    }
}

/// Whether the encoder `config` would select turns rotated or flipped GPU
/// frames itself (`CaptureConfig::consumer_orients`), so capture can hand
/// them over instead of turning them on the CPU: NVENC on Linux when its
/// DMA-BUF import works on its GPU (its GPU fit applies
/// `GpuFrame::orientation`); VA-API when its render node's VPP rotates and
/// mirrors every way. Every encoder stays upright either way: one that
/// cannot turn a frame reads it back and turns it on the CPU.
#[must_use]
pub fn orients_gpu_frames(config: &ferricast_core::EncoderConfig) -> bool {
    match predicted_backend(config) {
        #[cfg(all(feature = "nvenc", target_os = "linux"))]
        Some(ferricast_core::BackendId::Nvenc) => {
            ferricast_codec::cuda::CudaDevice::for_config(config, ferricast_core::BackendId::Nvenc)
                .is_ok_and(nvenc::orients_gpu_frames_on)
        }
        #[cfg(feature = "vaapi")]
        Some(ferricast_core::BackendId::Vaapi) => vaapi::orients_gpu_frames(config),
        _ => false,
    }
}

/// The backend automatic selection would most likely open for `config`
/// (its codec, backend selection and GPU preference), probing but not
/// opening; `None` for a codec without an encoder facade or a selection
/// that does not resolve.
#[must_use]
pub fn predicted_backend(
    config: &ferricast_core::EncoderConfig,
) -> Option<ferricast_core::BackendId> {
    use ferricast_codec::GpuInput as _;
    let registry = match config.codec {
        ferricast_core::Codec::H264 => h264::H264Encoder::registry(),
        ferricast_core::Codec::H265 => h265::H265Encoder::registry(),
        ferricast_core::Codec::Vp8 | ferricast_core::Codec::Vp9 | ferricast_core::Codec::Av1 => {
            return None;
        }
    };
    let placement = config.gpu_placement().ok()?;
    let plan = registry.plan_on(&config.backend, &placement).ok()?;
    registry.first_available(&plan)
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
