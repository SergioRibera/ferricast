//! Platform decoders for this OS. Contract: [`video`] / [`audio`] add a
//! candidate for every codec of `registry.target()` this OS can do and
//! return the registry unchanged for the rest.

use ferricast_codec::{AudioDecoderRegistry, VideoDecoderRegistry};

pub(crate) fn video(registry: VideoDecoderRegistry) -> VideoDecoderRegistry {
    registry
}

pub(crate) fn audio(registry: AudioDecoderRegistry) -> AudioDecoderRegistry {
    registry
}
