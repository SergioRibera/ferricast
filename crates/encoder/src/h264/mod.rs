//! H.264 encoder facade.
//!
//! [`H264Encoder`] picks its backend from [`EncoderConfig::backend`]
//! (see [`ferricast_core::BackendSelection`]). Automatic selection
//! tries, in order:
//!
//! 1. **NVENC** — NVIDIA. First because on the common laptop combo of
//!    an NVIDIA dGPU and an AMD / Intel iGPU, VA-API would pick the
//!    iGPU and move every captured frame across PCIe twice.
//! 2. **VA-API** — Intel / AMD on Linux.
//! 3. **x264** — software; works everywhere it is compiled in.
//!
//! `FERRICAST_H264_ENCODE_BACKEND` (or the older
//! `FERRICAST_H264_BACKEND`) overrides automatic selection without a
//! rebuild: a backend name, a comma-separated list, `hardware` or
//! `software`.

// Helper modules used only by the VA-API backend today. Gated so a
// no-vaapi build doesn't compile them just to drop dead code.
#[cfg(feature = "vaapi")]
mod bitstream;
#[cfg(feature = "vaapi")]
mod headers;
#[cfg(feature = "vaapi")]
pub(crate) mod yuv;

#[cfg(feature = "vaapi")]
mod vaapi_impl;
#[cfg(feature = "x264")]
mod x264_impl;

#[cfg(any(feature = "nvenc", feature = "vaapi", feature = "x264"))]
use ferricast_codec::Candidate;
use ferricast_codec::VideoEncoderRegistry;
#[cfg(any(feature = "nvenc", feature = "vaapi", feature = "x264"))]
use ferricast_core::Acceleration;
#[cfg(any(feature = "nvenc", feature = "vaapi"))]
use ferricast_core::CodecError;
#[cfg(any(feature = "nvenc", feature = "vaapi"))]
use ferricast_core::EncoderConfig;
use ferricast_core::{BackendId, Codec, CodecTarget};

#[cfg(feature = "nvenc")]
pub use crate::nvenc::NvencH264Encoder;
#[cfg(feature = "vaapi")]
pub use vaapi_impl::VaapiH264Encoder;
#[cfg(feature = "x264")]
pub use x264_impl::X264Encoder;

ferricast_codec::video_encoder_facade! {
    /// H.264 encoder over whichever backend its configuration selects.
    pub struct H264Encoder: Codec::H264, registry = registry;
}

/// The H.264 encode backends of this build, in automatic order.
fn registry() -> VideoEncoderRegistry {
    let registry = VideoEncoderRegistry::new(CodecTarget::encode(Codec::H264));
    #[cfg(feature = "nvenc")]
    let registry = registry.candidate(
        Candidate::new(
            BackendId::Nvenc,
            Acceleration::Hardware,
            |config: &EncoderConfig| {
                NvencH264Encoder::probe_with(config.clone())
                    .map(|encoder| Box::new(encoder) as _)
                    .map_err(CodecError::from_backend(
                        BackendId::Nvenc,
                        ferricast_core::Operation::Open,
                    ))
            },
        )
        .with_probe(|| {
            NvencH264Encoder::probe()
                .map(drop)
                .map_err(CodecError::from_backend(
                    BackendId::Nvenc,
                    ferricast_core::Operation::Probe,
                ))
        }),
    );
    #[cfg(not(feature = "nvenc"))]
    let registry = registry.missing(BackendId::Nvenc, "ferricast-encoder/nvenc");
    #[cfg(feature = "vaapi")]
    let registry = registry.candidate(
        Candidate::new(
            BackendId::Vaapi,
            Acceleration::Hardware,
            |config: &EncoderConfig| {
                VaapiH264Encoder::probe_with(config.clone())
                    .map(|encoder| Box::new(encoder) as _)
                    .map_err(CodecError::from_backend(
                        BackendId::Vaapi,
                        ferricast_core::Operation::Open,
                    ))
            },
        )
        .with_probe(|| {
            VaapiH264Encoder::probe()
                .map(drop)
                .map_err(CodecError::from_backend(
                    BackendId::Vaapi,
                    ferricast_core::Operation::Probe,
                ))
        }),
    );
    #[cfg(not(feature = "vaapi"))]
    let registry = registry.missing(BackendId::Vaapi, "ferricast-encoder/vaapi");
    let registry = crate::platform::video(registry);
    #[cfg(feature = "x264")]
    let registry = registry.candidate(Candidate::new(
        BackendId::X264,
        Acceleration::Software,
        ferricast_codec::open_video_encoder::<X264Encoder>,
    ));
    #[cfg(not(feature = "x264"))]
    let registry = registry.missing(BackendId::X264, "ferricast-encoder/x264");
    registry
}
pub mod utils {
    use ferricast_codec::bitstream::{NalSyntax, ParameterSets};

    /// The SPS and PPS of an Annex B access unit, each behind a 4-byte
    /// start code; empty when it carries none.
    #[must_use]
    pub fn extract_sps_pps(annex_b: &[u8]) -> Vec<u8> {
        let mut sets = ParameterSets::new(NalSyntax::H264);
        sets.remember(annex_b);
        sets.annex_b()
    }
}

#[cfg(test)]
mod tests {
    use super::utils::*;

    #[test]
    fn extracts_sps_pps_from_idr() {
        // SPS (type 7), PPS (type 8), IDR slice (type 5) — all
        // prefixed with 4-byte start codes.
        let stream = [
            0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1e, // SPS
            0, 0, 0, 1, 0x68, 0xce, 0x3c, 0x80, // PPS
            0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00, // IDR
        ];
        let out = extract_sps_pps(&stream);
        let expected = [
            0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1e, 0, 0, 0, 1, 0x68, 0xce, 0x3c, 0x80,
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn empty_when_no_params() {
        let stream = [0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00];
        assert!(extract_sps_pps(&stream).is_empty());
    }
}
