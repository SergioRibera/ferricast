//! H.264 decoder facade.
//!
//! [`H264Decoder`] picks its backend from [`DecoderConfig::backend`]
//! (see [`ferricast_core::BackendSelection`]). Automatic selection
//! tries, in order:
//!
//! 1. **NVDEC** — NVIDIA hardware, when a CUDA-capable GPU is present.
//! 2. **VA-API** — Intel / AMD hardware on Linux. A stream using a
//!    feature it does not implement (interlacing, 4:2:2, high bit depth,
//!    scaling matrices, FMO) hands over to the next backend mid-stream.
//! 3. The platform decoders (Media Foundation, VideoToolbox,
//!    `MediaCodec`) where compiled in.
//! 4. **openh264** — software; works everywhere it is compiled in.
//!
//! A hardware backend that fails before producing its first picture
//! hands the stream to the next backend from its first packet, so a
//! driver that accepts the configuration but not the bitstream never
//! breaks playback (see [`ferricast_codec::SelectingVideoDecoder`]).
//! `FERRICAST_H264_DECODE_BACKEND` overrides automatic selection.

#[cfg(feature = "vaapi-decode")]
mod dpb;
#[cfg(feature = "openh264-decode")]
mod openh264_impl;
#[cfg(feature = "vaapi-decode")]
mod slice;
#[cfg(feature = "vaapi-decode")]
mod vaapi_impl;

#[cfg(any(
    feature = "nvdec-decode",
    feature = "vaapi-decode",
    feature = "openh264-decode"
))]
use ferricast_codec::Candidate;
use ferricast_codec::VideoDecoderRegistry;
use ferricast_codec::bitstream::NalSyntax;
#[cfg(any(
    feature = "nvdec-decode",
    feature = "vaapi-decode",
    feature = "openh264-decode"
))]
use ferricast_core::Acceleration;
#[cfg(feature = "vaapi-decode")]
use ferricast_core::DecoderConfig;
use ferricast_core::{BackendId, Codec, CodecTarget};
#[cfg(any(feature = "nvdec-decode", feature = "vaapi-decode"))]
use ferricast_core::{CodecError, Operation};

#[cfg(feature = "nvdec-decode")]
pub use crate::nvdec::NvdecH264Decoder;
#[cfg(feature = "openh264-decode")]
pub use openh264_impl::OpenH264Decoder;
#[cfg(feature = "vaapi-decode")]
pub use vaapi_impl::{VaapiH264Decoder, VaapiH264Error};

ferricast_codec::video_decoder_facade! {
    /// H.264 decoder over whichever backend its configuration selects.
    pub struct H264Decoder: Codec::H264, registry = registry, syntax = Some(NalSyntax::H264);
}

/// The H.264 decode backends of this build, in automatic order.
fn registry() -> VideoDecoderRegistry {
    let registry = VideoDecoderRegistry::new(CodecTarget::decode(Codec::H264));
    #[cfg(feature = "nvdec-decode")]
    let registry = registry.candidate(
        Candidate::new(
            BackendId::Nvdec,
            Acceleration::Hardware,
            ferricast_codec::open_video_decoder::<NvdecH264Decoder>,
        )
        .with_probe(|| {
            NvdecH264Decoder::probe()
                .map_err(CodecError::from_backend(BackendId::Nvdec, Operation::Probe))
        }),
    );
    #[cfg(not(feature = "nvdec-decode"))]
    let registry = registry.missing(BackendId::Nvdec, "ferricast-decoder/nvdec-decode");
    #[cfg(feature = "vaapi-decode")]
    let registry = registry.candidate(Candidate::new(
        BackendId::Vaapi,
        Acceleration::Hardware,
        |config: &DecoderConfig| {
            let mut decoder = VaapiH264Decoder::probe_with(config)
                .map_err(CodecError::from_backend(BackendId::Vaapi, Operation::Open))?;
            ferricast_core::VideoDecoder::configure(&mut decoder, config)?;
            Ok(Box::new(decoder))
        },
    ));
    #[cfg(not(feature = "vaapi-decode"))]
    let registry = registry.missing(BackendId::Vaapi, "ferricast-decoder/vaapi-decode");
    let registry = crate::platform::video(registry);
    #[cfg(feature = "openh264-decode")]
    let registry = registry.candidate(Candidate::new(
        BackendId::OpenH264,
        Acceleration::Software,
        ferricast_codec::open_video_decoder::<OpenH264Decoder>,
    ));
    #[cfg(not(feature = "openh264-decode"))]
    let registry = registry.missing(BackendId::OpenH264, "ferricast-decoder/openh264-decode");
    registry
}
