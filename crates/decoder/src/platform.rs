//! Decoders that come with an operating system: Media Foundation on
//! Windows, VideoToolbox / AudioToolbox on Apple platforms, the NDK
//! `AMediaCodec` on Android. Each function appends this platform's
//! backends for a job (or records the ones that only exist elsewhere,
//! so a selection naming a foreign backend says where it runs).

use ferricast_codec::{AudioDecoderRegistry, VideoDecoderRegistry};
use ferricast_core::BackendId;

const WINDOWS: &str = "only on Windows";
const APPLE: &str = "only on macOS and iOS";
const ANDROID: &str = "only on Android";

/// Adds the platform H.264 decoders.
pub(crate) fn h264(registry: VideoDecoderRegistry) -> VideoDecoderRegistry {
    video(registry)
}

/// Adds the platform H.265 decoders.
pub(crate) fn h265(registry: VideoDecoderRegistry) -> VideoDecoderRegistry {
    video(registry)
}

/// Adds the platform decoders of a video job.
pub(crate) fn video(registry: VideoDecoderRegistry) -> VideoDecoderRegistry {
    registry
        .elsewhere(
            BackendId::MediaFoundation,
            "ferricast-decoder/media-foundation",
            WINDOWS,
        )
        .elsewhere(
            BackendId::VideoToolbox,
            "ferricast-decoder/videotoolbox",
            APPLE,
        )
        .elsewhere(
            BackendId::MediaCodec,
            "ferricast-decoder/mediacodec",
            ANDROID,
        )
}

/// Adds the platform decoders of an audio job.
pub(crate) fn audio(registry: AudioDecoderRegistry) -> AudioDecoderRegistry {
    registry
        .elsewhere(
            BackendId::MediaFoundation,
            "ferricast-decoder/media-foundation",
            WINDOWS,
        )
        .elsewhere(
            BackendId::AudioToolbox,
            "ferricast-decoder/audiotoolbox",
            APPLE,
        )
        .elsewhere(
            BackendId::MediaCodec,
            "ferricast-decoder/mediacodec",
            ANDROID,
        )
}
