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
#[cfg_attr(
    not(any(
        feature = "aac",
        all(target_os = "android", feature = "mediacodec"),
        all(target_os = "windows", feature = "media-foundation"),
        all(any(target_os = "macos", target_os = "ios"), feature = "audiotoolbox")
    )),
    allow(dead_code, reason = "used by the AAC decoders only")
)]
mod adts;
#[cfg(feature = "alac-decode")]
mod alac;
#[cfg(feature = "av1-decode")]
mod av1;
mod facades;
// Header fields the VideoToolbox decoder has to spell out itself.
pub mod h264;
pub mod h265;
#[allow(
    dead_code,
    reason = "conditional: shared by the video decoders; each backend set uses a different subset"
)]
mod nv12;
#[cfg(feature = "nvdec-decode")]
pub mod nvdec;
#[cfg(feature = "opus-decode")]
mod opus;
mod pcm;
mod platform;
#[cfg(any(
    feature = "openh264-decode",
    feature = "nvdec-decode",
    feature = "hevc-decode"
))]
mod signalling;
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

/// Adds every decode job of this crate to an istmo codec catalog.
#[cfg(feature = "istmo")]
#[must_use]
pub fn catalog(catalog: ferricast_codec::plugin::Catalog) -> ferricast_codec::plugin::Catalog {
    use ferricast_codec::bitstream::NalSyntax;
    use ferricast_core::{AudioCodec, Codec};
    catalog
        .video_decoder(Codec::H264, H264Decoder::registry, Some(NalSyntax::H264))
        .video_decoder(Codec::H265, H265Decoder::registry, Some(NalSyntax::H265))
        .video_decoder(Codec::Vp8, Vp8Decoder::registry, None)
        .video_decoder(Codec::Vp9, Vp9Decoder::registry, None)
        .video_decoder(Codec::Av1, Av1Decoder::registry, None)
        .audio_decoder(AudioCodec::Aac, AacDecoder::registry)
        .audio_decoder(AudioCodec::AacEld, AacEldDecoder::registry)
        .audio_decoder(AudioCodec::Alac, AlacDecoder::registry)
        .audio_decoder(AudioCodec::Opus, OpusDecoder::registry)
        .audio_decoder(AudioCodec::Pcm, PcmDecoder::registry)
}
