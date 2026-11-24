//! Encoders that come with an operating system: Media Foundation on
//! Windows, VideoToolbox / AudioToolbox on Apple platforms, the NDK
//! `AMediaCodec` on Android. Each registry function appends this
//! platform's backends for a job (or records the ones that only exist
//! elsewhere), so a selection naming a foreign backend says where it
//! runs.

use ferricast_codec::{AudioEncoderRegistry, VideoEncoderRegistry};
use ferricast_core::BackendId;

const WINDOWS: &str = "only on Windows";
const APPLE: &str = "only on macOS and iOS";
const ANDROID: &str = "only on Android";

/// Adds the platform H.264 encoders.
pub(crate) fn h264(registry: VideoEncoderRegistry) -> VideoEncoderRegistry {
    video(registry)
}

/// Adds the platform H.265 encoders.
pub(crate) fn h265(registry: VideoEncoderRegistry) -> VideoEncoderRegistry {
    video(registry)
}

/// Adds the platform AAC-LC encoders.
pub(crate) fn aac(registry: AudioEncoderRegistry) -> AudioEncoderRegistry {
    registry
        .elsewhere(
            BackendId::MediaFoundation,
            "ferricast-encoder/media-foundation",
            WINDOWS,
        )
        .elsewhere(
            BackendId::AudioToolbox,
            "ferricast-encoder/audiotoolbox",
            APPLE,
        )
        .elsewhere(
            BackendId::MediaCodec,
            "ferricast-encoder/mediacodec",
            ANDROID,
        )
}

fn video(registry: VideoEncoderRegistry) -> VideoEncoderRegistry {
    registry
        .elsewhere(
            BackendId::MediaFoundation,
            "ferricast-encoder/media-foundation",
            WINDOWS,
        )
        .elsewhere(
            BackendId::VideoToolbox,
            "ferricast-encoder/videotoolbox",
            APPLE,
        )
        .elsewhere(
            BackendId::MediaCodec,
            "ferricast-encoder/mediacodec",
            ANDROID,
        )
}
