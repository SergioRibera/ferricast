//! Android decoders through the NDK `AMediaCodec` (API 28+): H.264,
//! H.265, VP8, VP9 and AV1 video, AAC and Opus audio.
//!
//! Video gets two candidates: the device's preferred decoder for the
//! type, accepted only when it is a vendor (hardware) codec, then the
//! Google software decoder (`c2.android.*`) by name. Audio decoders are
//! software on every device, so audio gets the Google one only.

mod audio;
mod codec;
mod video;

use ferricast_codec::{AudioDecoderRegistry, Candidate, VideoDecoderRegistry};
use ferricast_core::{
    Acceleration, AudioDecoderConfig, BackendId, Codec, CodecError, CodecKind, CodecResult,
    CodecTarget, DecoderConfig, Operation,
};

use codec::Pick;

pub(crate) fn video(registry: VideoDecoderRegistry) -> VideoDecoderRegistry {
    let CodecKind::Video(kind) = registry.target().codec else {
        return registry;
    };
    let hardware = Candidate::new(
        BackendId::MediaCodec,
        Acceleration::Hardware,
        |config: &DecoderConfig| {
            video::open(Pick::Hardware, config).map(|decoder| Box::new(decoder) as _)
        },
    );
    let hardware = match kind {
        Codec::H264 => hardware.with_probe(|| probe(Codec::H264)),
        Codec::H265 => hardware.with_probe(|| probe(Codec::H265)),
        Codec::Vp8 => hardware.with_probe(|| probe(Codec::Vp8)),
        Codec::Vp9 => hardware.with_probe(|| probe(Codec::Vp9)),
        Codec::Av1 => hardware.with_probe(|| probe(Codec::Av1)),
    };
    let software = Candidate::new(
        BackendId::MediaCodec,
        Acceleration::Software,
        |config: &DecoderConfig| {
            video::open(Pick::Software(video::software_name(config.codec)), config)
                .map(|decoder| Box::new(decoder) as _)
        },
    );
    registry.candidate(hardware).candidate(software)
}

pub(crate) fn audio(registry: AudioDecoderRegistry) -> AudioDecoderRegistry {
    let CodecKind::Audio(kind) = registry.target().codec else {
        return registry;
    };
    if audio::mime(kind).is_none() {
        return registry;
    }
    registry.candidate(Candidate::new(
        BackendId::MediaCodec,
        Acceleration::Software,
        |config: &AudioDecoderConfig| {
            let name =
                audio::software_name(config.codec).ok_or(CodecError::BackendCannotHandle {
                    backend: BackendId::MediaCodec,
                    target: CodecTarget::decode(config.codec),
                })?;
            audio::open(Pick::Software(name), config).map(|decoder| Box::new(decoder) as _)
        },
    ))
}

/// Whether the device's preferred decoder for `codec` is a hardware one.
fn probe(codec: Codec) -> CodecResult<()> {
    codec::Codec::create(video::mime(codec), Pick::Hardware)
        .map(drop)
        .map_err(|error| match error {
            CodecError::BackendUnavailable { .. } => error,
            other => CodecError::backend(BackendId::MediaCodec, Operation::Probe, other),
        })
}
