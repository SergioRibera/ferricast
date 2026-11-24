//! Platform encoders for this OS. Contract: [`video`] / [`audio`] add a
//! candidate for every codec of `registry.target()` this OS can do and
//! return the registry unchanged for the rest.

use ferricast_codec::{AudioEncoderRegistry, VideoEncoderRegistry};

pub(crate) fn video(registry: VideoEncoderRegistry) -> VideoEncoderRegistry {
    registry
}

pub(crate) fn audio(registry: AudioEncoderRegistry) -> AudioEncoderRegistry {
    registry
}
