//! Android encoders through the NDK `AMediaCodec` (API 28+): H.264 and
//! H.265 video, AAC-LC and Opus audio.
//!
//! Video gets two candidates: the device's preferred encoder for the
//! type, accepted only when it is a vendor (hardware) codec, then the
//! Google software encoder (`c2.android.*`) by name. Audio encoders are
//! software on every device, so audio gets the Google one only.

mod audio;
mod codec;
mod video;

use ferricast_codec::{AudioEncoderRegistry, Candidate, VideoEncoderRegistry};
use ferricast_core::{
    Acceleration, AudioEncoderConfig, BackendId, CodecKind, EncoderConfig, Operation,
};

use codec::Pick;

pub(crate) fn video(registry: VideoEncoderRegistry) -> VideoEncoderRegistry {
    let CodecKind::Video(kind) = registry.target().codec else {
        return registry;
    };
    let (Some(mime), Some(_)) = (video::mime(kind), video::software_name(kind)) else {
        return registry;
    };
    let hardware = Candidate::new(
        BackendId::MediaCodec,
        Acceleration::Hardware,
        |config: &EncoderConfig| {
            video::open(Pick::Hardware, config).map(|encoder| Box::new(encoder) as _)
        },
    );
    let hardware = match mime {
        "video/avc" => hardware.with_probe(|| probe("video/avc")),
        _ => hardware.with_probe(|| probe("video/hevc")),
    };
    let software = Candidate::new(
        BackendId::MediaCodec,
        Acceleration::Software,
        |config: &EncoderConfig| {
            let name = video::software_name(config.codec).ok_or_else(|| {
                ferricast_core::CodecError::BackendCannotHandle {
                    backend: BackendId::MediaCodec,
                    target: ferricast_core::CodecTarget::encode(config.codec),
                }
            })?;
            video::open(Pick::Software(name), config).map(|encoder| Box::new(encoder) as _)
        },
    );
    registry.candidate(hardware).candidate(software)
}

pub(crate) fn audio(registry: AudioEncoderRegistry) -> AudioEncoderRegistry {
    let CodecKind::Audio(kind) = registry.target().codec else {
        return registry;
    };
    if audio::mime(kind).is_none() || audio::software_name(kind).is_none() {
        return registry;
    }
    registry.candidate(Candidate::new(
        BackendId::MediaCodec,
        Acceleration::Software,
        |config: &AudioEncoderConfig| {
            let name = audio::software_name(config.codec).ok_or_else(|| {
                ferricast_core::CodecError::BackendCannotHandle {
                    backend: BackendId::MediaCodec,
                    target: ferricast_core::CodecTarget::encode(config.codec),
                }
            })?;
            audio::open(Pick::Software(name), config).map(|encoder| Box::new(encoder) as _)
        },
    ))
}

/// Whether the device's preferred encoder for `mime` is a hardware one.
fn probe(mime: &str) -> ferricast_core::CodecResult<()> {
    codec::Codec::create(mime, Pick::Hardware)
        .map(drop)
        .map_err(|error| match error {
            ferricast_core::CodecError::BackendUnavailable { .. } => error,
            other => {
                ferricast_core::CodecError::backend(BackendId::MediaCodec, Operation::Probe, other)
            }
        })
}
