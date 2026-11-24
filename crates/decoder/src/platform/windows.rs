//! Media Foundation decoders (Windows).
//!
//! Video jobs (H.264, H.265, VP8, VP9, AV1) get a hardware candidate
//! (vendor asynchronous MFTs) and a software one (Microsoft's decoder
//! MFTs, which run on the CPU when no Direct3D device manager is
//! handed to them, as here). H.264 decoders ship with Windows; H.265,
//! VP9 and AV1 come with the Store "Video Extensions", so their
//! candidates report themselves unavailable where those are missing.
//! AAC gets Microsoft's AAC decoder MFT (software).
//!
//! Hardware decoders that only output Direct3D surfaces refuse the
//! system-memory NV12 output type these sessions ask for; they fail to
//! open and selection moves on to the software candidate.
//!
//! The Media Foundation plumbing is shared with `ferricast-encoder`
//! through [`ferricast_codec::mf`].

use ferricast_codec::mf;

mod audio;
mod video;

use ferricast_codec::{AudioDecoderRegistry, Candidate, VideoDecoderRegistry};
use ferricast_core::{Acceleration, AudioCodec, BackendId, CodecKind};

/// Adds the Media Foundation decoders of `registry`'s video job.
pub(crate) fn video(registry: VideoDecoderRegistry) -> VideoDecoderRegistry {
    let CodecKind::Video(codec) = registry.target().codec else {
        return registry;
    };
    let Some((hardware, software)) = video::openers(codec) else {
        return registry;
    };
    registry
        .candidate(
            Candidate::new(
                BackendId::MediaFoundation,
                Acceleration::Hardware,
                hardware.open,
            )
            .with_probe(hardware.probe),
        )
        .candidate(
            Candidate::new(
                BackendId::MediaFoundation,
                Acceleration::Software,
                software.open,
            )
            .with_probe(software.probe),
        )
}

/// Adds the Media Foundation decoders of `registry`'s audio job.
pub(crate) fn audio(registry: AudioDecoderRegistry) -> AudioDecoderRegistry {
    if registry.target().codec != CodecKind::Audio(AudioCodec::Aac) {
        return registry;
    }
    registry.candidate(
        Candidate::new(
            BackendId::MediaFoundation,
            Acceleration::Software,
            audio::open_aac,
        )
        .with_probe(audio::probe_aac),
    )
}
