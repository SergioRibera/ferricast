//! Media Foundation encoders (Windows).
//!
//! H.264 and H.265 each get a hardware candidate (the GPU vendor's
//! asynchronous MFT: Intel Quick Sync, NVIDIA NVENC, AMD AMF behind
//! Media Foundation) and a software one (Microsoft's H.264 encoder
//! MFT; Windows ships no software HEVC encoder, so that candidate
//! reports itself unavailable unless a third party installed one).
//! AAC-LC gets Microsoft's AAC encoder MFT (software; it is the only
//! AAC encoder Windows registers).
//!
//! See [`mf`] for the threading and MFT models.

mod audio;
use ferricast_codec::mf;
mod video;

use ferricast_codec::{AudioEncoderRegistry, Candidate, VideoEncoderRegistry};
use ferricast_core::{Acceleration, AudioCodec, BackendId, Codec, CodecKind};

/// Adds the Media Foundation encoders of `registry`'s video job.
pub(crate) fn video(registry: VideoEncoderRegistry) -> VideoEncoderRegistry {
    let (hardware, software): (video::Opener, video::Opener) = match registry.target().codec {
        CodecKind::Video(Codec::H264) => (video::H264_HARDWARE, video::H264_SOFTWARE),
        CodecKind::Video(Codec::H265) => (video::H265_HARDWARE, video::H265_SOFTWARE),
        _ => return registry,
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

/// Adds the Media Foundation encoders of `registry`'s audio job.
pub(crate) fn audio(registry: AudioEncoderRegistry) -> AudioEncoderRegistry {
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
