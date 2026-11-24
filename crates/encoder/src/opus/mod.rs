//! Opus encoder facade (Cast Streaming mirroring audio).

#[cfg(feature = "opus-rs")]
mod opus_rs;

use ferricast_codec::AudioEncoderRegistry;
#[cfg(feature = "opus-rs")]
use ferricast_codec::Candidate;
#[cfg(feature = "opus-rs")]
use ferricast_core::Acceleration;
use ferricast_core::{AudioCodec, BackendId, CodecTarget};

#[cfg(feature = "opus-rs")]
pub use opus_rs::OpusRsEncoder;

ferricast_codec::audio_encoder_facade! {
    /// Opus encoder over whichever backend its configuration selects.
    pub struct OpusEncoder, registry = registry;
}

/// The Opus encode backends of this build.
fn registry() -> AudioEncoderRegistry {
    let registry = AudioEncoderRegistry::new(CodecTarget::encode(AudioCodec::Opus));
    #[cfg(feature = "opus-rs")]
    let registry = registry.candidate(Candidate::new(
        BackendId::OpusRs,
        Acceleration::Software,
        ferricast_codec::open_audio_encoder::<OpusRsEncoder>,
    ));
    #[cfg(not(feature = "opus-rs"))]
    let registry = registry.missing(BackendId::OpusRs, "ferricast-encoder/opus-rs");
    registry
}
