//! HEVC (H.265) decoder facade.
//!
//! [`H265Decoder`] picks its backend from [`DecoderConfig::backend`].
//! Automatic selection tries NVDEC, then the platform decoders, then
//! the pure-Rust `rust_h265` software decoder. VA-API HEVC is opt-in
//! (`FERRICAST_H265_DECODE_BACKEND=vaapi` or a selection naming it):
//! its profile probe and surface bring-up are complete but slice
//! submission is not, so automatic selection never picks it.

#[cfg(feature = "hevc-decode")]
mod software_impl;
#[cfg(any(feature = "hevc-decode", feature = "nvdec-hevc-decode"))]
pub(crate) mod timestamps;
#[cfg(feature = "vaapi-hevc-decode")]
mod vaapi_impl;

#[cfg(any(
    feature = "nvdec-hevc-decode",
    feature = "vaapi-hevc-decode",
    feature = "hevc-decode"
))]
use ferricast_codec::Candidate;
use ferricast_codec::VideoDecoderRegistry;
use ferricast_codec::bitstream::NalSyntax;
#[cfg(any(
    feature = "nvdec-hevc-decode",
    feature = "vaapi-hevc-decode",
    feature = "hevc-decode"
))]
use ferricast_core::Acceleration;
#[cfg(feature = "vaapi-hevc-decode")]
use ferricast_core::DecoderConfig;
use ferricast_core::{BackendId, Codec, CodecTarget};
#[cfg(any(feature = "nvdec-hevc-decode", feature = "vaapi-hevc-decode"))]
use ferricast_core::{CodecError, Operation};

#[cfg(feature = "nvdec-hevc-decode")]
pub use crate::nvdec::NvdecH265Decoder;
#[cfg(feature = "hevc-decode")]
pub use software_impl::SoftwareH265Decoder;
#[cfg(feature = "vaapi-hevc-decode")]
pub use vaapi_impl::VaapiH265Decoder;

ferricast_codec::video_decoder_facade! {
    /// H.265 decoder over whichever backend its configuration selects.
    pub struct H265Decoder: Codec::H265, registry = registry, syntax = Some(NalSyntax::H265);
}

/// The H.265 decode backends of this build, in automatic order.
fn registry() -> VideoDecoderRegistry {
    let registry = VideoDecoderRegistry::new(CodecTarget::decode(Codec::H265));
    #[cfg(feature = "nvdec-hevc-decode")]
    let registry = registry.candidate(
        Candidate::new(BackendId::Nvdec, Acceleration::Hardware, |config| {
            // Capability query first: cheap, and it names a missing driver
            // or codec instead of failing later on the first packet.
            NvdecH265Decoder::probe()
                .map_err(CodecError::from_backend(BackendId::Nvdec, Operation::Probe))?;
            ferricast_codec::open_video_decoder::<NvdecH265Decoder>(config)
        })
        .with_probe(|| {
            NvdecH265Decoder::probe()
                .map_err(CodecError::from_backend(BackendId::Nvdec, Operation::Probe))
        }),
    );
    #[cfg(not(feature = "nvdec-hevc-decode"))]
    let registry = registry.missing(BackendId::Nvdec, "ferricast-decoder/nvdec-hevc-decode");
    #[cfg(feature = "vaapi-hevc-decode")]
    let registry = registry.candidate(
        Candidate::new(
            BackendId::Vaapi,
            Acceleration::Hardware,
            |config: &DecoderConfig| {
                let mut decoder = VaapiH265Decoder::probe_with(config)
                    .map_err(CodecError::from_backend(BackendId::Vaapi, Operation::Open))?;
                ferricast_core::VideoDecoder::configure(&mut decoder, config)?;
                Ok(Box::new(decoder) as Box<dyn ferricast_codec::VideoDecoderSession>)
            },
        )
        .opt_in(),
    );
    #[cfg(not(feature = "vaapi-hevc-decode"))]
    let registry = registry.missing(BackendId::Vaapi, "ferricast-decoder/vaapi-hevc-decode");
    let registry = crate::platform::video(registry);
    #[cfg(feature = "hevc-decode")]
    let registry = registry.candidate(Candidate::new(
        BackendId::RustHevc,
        Acceleration::Software,
        ferricast_codec::open_video_decoder::<SoftwareH265Decoder>,
    ));
    #[cfg(not(feature = "hevc-decode"))]
    let registry = registry.missing(BackendId::RustHevc, "ferricast-decoder/hevc-decode");
    registry
}
