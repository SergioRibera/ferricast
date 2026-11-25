//! Decoder facades for the codecs whose backends live in single-file
//! modules: each picks its backend from the configuration's
//! [`ferricast_core::BackendSelection`] like [`crate::H264Decoder`].
//! Automatic selection tries NVDEC first, then the platform decoders,
//! then the pure-Rust software decoder.

use ferricast_codec::{AudioDecoderRegistry, Candidate, VideoDecoderRegistry};
use ferricast_core::{Acceleration, AudioCodec, BackendId, Codec, CodecTarget};

use crate::platform;

/// Adds NVDEC for `codec`: the first hardware candidate of every video
/// job. Recorded as missing (naming its feature) when the build leaves it
/// out, and as elsewhere off Linux and Windows.
pub(crate) fn nvdec(registry: VideoDecoderRegistry, codec: Codec) -> VideoDecoderRegistry {
    #[cfg(all(
        feature = "nvdec-decode",
        any(target_os = "linux", target_os = "windows")
    ))]
    let registry = crate::nvdec::register(registry, codec);
    #[cfg(not(all(
        feature = "nvdec-decode",
        any(target_os = "linux", target_os = "windows")
    )))]
    let registry = {
        let feature = if codec == Codec::H265 {
            "ferricast-decoder/nvdec-hevc-decode"
        } else {
            "ferricast-decoder/nvdec-decode"
        };
        if cfg!(any(target_os = "linux", target_os = "windows")) {
            registry.missing(BackendId::Nvdec, feature)
        } else {
            registry.elsewhere(BackendId::Nvdec, feature, "only on Linux and Windows")
        }
    };
    registry
}

ferricast_codec::video_decoder_facade! {
    /// VP8 decoder over whichever backend its configuration selects.
    pub struct Vp8Decoder: Codec::Vp8, registry = vp8_registry, syntax = None;
}

fn vp8_registry() -> VideoDecoderRegistry {
    let registry = platform::video(nvdec(
        VideoDecoderRegistry::new(CodecTarget::decode(Codec::Vp8)),
        Codec::Vp8,
    ));
    #[cfg(feature = "vp8-decode")]
    let registry = registry.candidate(Candidate::new(
        BackendId::RustVp8,
        Acceleration::Software,
        ferricast_codec::open_video_decoder::<crate::vp8::RustVp8Decoder>,
    ));
    #[cfg(not(feature = "vp8-decode"))]
    let registry = registry.missing(BackendId::RustVp8, "ferricast-decoder/vp8-decode");
    registry
}

ferricast_codec::video_decoder_facade! {
    /// VP9 decoder over whichever backend its configuration selects.
    pub struct Vp9Decoder: Codec::Vp9, registry = vp9_registry, syntax = None;
}

fn vp9_registry() -> VideoDecoderRegistry {
    let registry = platform::video(nvdec(
        VideoDecoderRegistry::new(CodecTarget::decode(Codec::Vp9)),
        Codec::Vp9,
    ));
    #[cfg(feature = "vp9-decode")]
    let registry = registry.candidate(Candidate::new(
        BackendId::RustVp9,
        Acceleration::Software,
        ferricast_codec::open_video_decoder::<crate::vp9::RustVp9Decoder>,
    ));
    #[cfg(not(feature = "vp9-decode"))]
    let registry = registry.missing(BackendId::RustVp9, "ferricast-decoder/vp9-decode");
    registry
}

ferricast_codec::video_decoder_facade! {
    /// AV1 decoder over whichever backend its configuration selects.
    pub struct Av1Decoder: Codec::Av1, registry = av1_registry, syntax = None;
}

fn av1_registry() -> VideoDecoderRegistry {
    let registry = platform::video(nvdec(
        VideoDecoderRegistry::new(CodecTarget::decode(Codec::Av1)),
        Codec::Av1,
    ));
    #[cfg(feature = "av1-decode")]
    let registry = registry.candidate(Candidate::new(
        BackendId::Rav1d,
        Acceleration::Software,
        ferricast_codec::open_video_decoder::<crate::av1::Rav1dDecoder>,
    ));
    #[cfg(not(feature = "av1-decode"))]
    let registry = registry.missing(BackendId::Rav1d, "ferricast-decoder/av1-decode");
    registry
}

ferricast_codec::audio_decoder_facade! {
    /// AAC (ADTS or raw access units) decoder over whichever backend
    /// its configuration selects. libfdk-aac comes first: it decodes
    /// HE-AAC v1 / v2 in full, where symphonia keeps only the AAC-LC
    /// core.
    pub struct AacDecoder: AudioCodec::Aac, registry = aac_registry;
}

fn aac_registry() -> AudioDecoderRegistry {
    let registry = AudioDecoderRegistry::new(CodecTarget::decode(AudioCodec::Aac));
    #[cfg(feature = "fdk-aac-decode")]
    let registry = registry.candidate(Candidate::new(
        BackendId::FdkAac,
        Acceleration::Software,
        ferricast_codec::open_audio_decoder::<crate::aac::FdkAacDecoder>,
    ));
    #[cfg(not(feature = "fdk-aac-decode"))]
    let registry = registry.missing(BackendId::FdkAac, "ferricast-decoder/fdk-aac-decode");
    let registry = platform::audio(registry);
    #[cfg(feature = "aac")]
    let registry = registry.candidate(Candidate::new(
        BackendId::Symphonia,
        Acceleration::Software,
        ferricast_codec::open_audio_decoder::<crate::aac::SymphoniaAacDecoder>,
    ));
    #[cfg(not(feature = "aac"))]
    let registry = registry.missing(BackendId::Symphonia, "ferricast-decoder/aac");
    registry
}

ferricast_codec::audio_decoder_facade! {
    /// AAC-ELD (AirPlay mirroring audio) decoder over whichever backend
    /// its configuration selects.
    pub struct AacEldDecoder: AudioCodec::AacEld, registry = aac_eld_registry;
}

fn aac_eld_registry() -> AudioDecoderRegistry {
    let registry = AudioDecoderRegistry::new(CodecTarget::decode(AudioCodec::AacEld));
    #[cfg(feature = "fdk-aac-decode")]
    let registry = registry.candidate(Candidate::new(
        BackendId::FdkAac,
        Acceleration::Software,
        ferricast_codec::open_audio_decoder::<crate::aac::FdkEldDecoder>,
    ));
    #[cfg(not(feature = "fdk-aac-decode"))]
    let registry = registry.missing(BackendId::FdkAac, "ferricast-decoder/fdk-aac-decode");
    platform::audio(registry)
}

ferricast_codec::audio_decoder_facade! {
    /// Apple Lossless (AirPlay audio) decoder over whichever backend its
    /// configuration selects.
    pub struct AlacDecoder: AudioCodec::Alac, registry = alac_registry;
}

fn alac_registry() -> AudioDecoderRegistry {
    let registry = platform::audio(AudioDecoderRegistry::new(CodecTarget::decode(
        AudioCodec::Alac,
    )));
    #[cfg(feature = "alac-decode")]
    let registry = registry.candidate(Candidate::new(
        BackendId::Symphonia,
        Acceleration::Software,
        ferricast_codec::open_audio_decoder::<crate::alac::SymphoniaAlacDecoder>,
    ));
    #[cfg(not(feature = "alac-decode"))]
    let registry = registry.missing(BackendId::Symphonia, "ferricast-decoder/alac-decode");
    registry
}

ferricast_codec::audio_decoder_facade! {
    /// Opus decoder over whichever backend its configuration selects.
    pub struct OpusDecoder: AudioCodec::Opus, registry = opus_registry;
}

fn opus_registry() -> AudioDecoderRegistry {
    let registry = platform::audio(AudioDecoderRegistry::new(CodecTarget::decode(
        AudioCodec::Opus,
    )));
    #[cfg(feature = "opus-decode")]
    let registry = registry.candidate(Candidate::new(
        BackendId::OpusRs,
        Acceleration::Software,
        ferricast_codec::open_audio_decoder::<crate::opus::OpusRsDecoder>,
    ));
    #[cfg(not(feature = "opus-decode"))]
    let registry = registry.missing(BackendId::OpusRs, "ferricast-decoder/opus-decode");
    registry
}

ferricast_codec::audio_decoder_facade! {
    /// 16-bit PCM pass-through.
    pub struct PcmDecoder: AudioCodec::Pcm, registry = pcm_registry;
}

fn pcm_registry() -> AudioDecoderRegistry {
    AudioDecoderRegistry::new(CodecTarget::decode(AudioCodec::Pcm)).candidate(Candidate::new(
        BackendId::Pcm,
        Acceleration::Software,
        ferricast_codec::open_audio_decoder::<crate::pcm::PcmPassthrough>,
    ))
}
