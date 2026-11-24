//! H.265 (HEVC) encoder facade.
//!
//! [`H265Encoder`] picks its backend from [`EncoderConfig::backend`].
//! HEVC has no software encoder here (x265 is GPL and slow at
//! real-time rates), so every backend is hardware: NVENC, then VA-API,
//! then the platform encoders. `FERRICAST_H265_ENCODE_BACKEND` (or the
//! older `FERRICAST_H265_BACKEND`) overrides automatic selection.

#[cfg(feature = "vaapi-hevc")]
mod bitstream;
#[cfg(feature = "vaapi-hevc")]
mod headers;
#[cfg(feature = "vaapi-hevc")]
mod vaapi_impl;

#[cfg(any(feature = "nvenc-hevc", feature = "vaapi-hevc"))]
use ferricast_codec::Candidate;
use ferricast_codec::VideoEncoderRegistry;
#[cfg(any(feature = "nvenc-hevc", feature = "vaapi-hevc"))]
use ferricast_core::{Acceleration, EncoderConfig};
use ferricast_core::{BackendId, Codec, CodecTarget};
#[cfg(any(feature = "nvenc-hevc", feature = "vaapi-hevc"))]
use ferricast_core::{CodecError, Operation};

#[cfg(feature = "nvenc-hevc")]
pub use crate::nvenc::NvencH265Encoder;
#[cfg(feature = "vaapi-hevc")]
pub use vaapi_impl::VaapiH265Encoder;

ferricast_codec::video_encoder_facade! {
    /// H.265 encoder over whichever backend its configuration selects.
    pub struct H265Encoder: Codec::H265, registry = registry;
}

/// The H.265 encode backends of this build, in automatic order.
fn registry() -> VideoEncoderRegistry {
    let registry = VideoEncoderRegistry::new(CodecTarget::encode(Codec::H265));
    #[cfg(feature = "nvenc-hevc")]
    let registry = registry.candidate(
        Candidate::new(
            BackendId::Nvenc,
            Acceleration::Hardware,
            |config: &EncoderConfig| {
                NvencH265Encoder::probe_with(config.clone())
                    .map(|encoder| Box::new(encoder) as _)
                    .map_err(CodecError::from_backend(BackendId::Nvenc, Operation::Open))
            },
        )
        .with_probe(|| {
            NvencH265Encoder::probe()
                .map(drop)
                .map_err(CodecError::from_backend(BackendId::Nvenc, Operation::Probe))
        }),
    );
    #[cfg(not(feature = "nvenc-hevc"))]
    let registry = registry.missing(BackendId::Nvenc, "ferricast-encoder/nvenc-hevc");
    #[cfg(feature = "vaapi-hevc")]
    let registry = registry.candidate(
        Candidate::new(
            BackendId::Vaapi,
            Acceleration::Hardware,
            |config: &EncoderConfig| {
                VaapiH265Encoder::probe_with(config)
                    .map(|encoder| Box::new(encoder) as _)
                    .map_err(CodecError::from_backend(BackendId::Vaapi, Operation::Open))
            },
        )
        .with_probe(|| {
            VaapiH265Encoder::probe()
                .map(drop)
                .map_err(CodecError::from_backend(BackendId::Vaapi, Operation::Probe))
        }),
    );
    #[cfg(not(feature = "vaapi-hevc"))]
    let registry = registry.missing(BackendId::Vaapi, "ferricast-encoder/vaapi-hevc");
    crate::platform::video(registry)
}
