//! Decoders that come with an operating system: Media Foundation on
//! Windows, `VideoToolbox` / `AudioToolbox` on Apple platforms, the NDK
//! `AMediaCodec` on Android.
//!
//! [`video`] and [`audio`] append this platform's backends for the job
//! a registry is for ([`ferricast_codec::Registry::target`]), in front
//! of the software fallbacks. A platform backend whose cargo feature is
//! off is recorded as missing (naming the feature), and one for another
//! operating system as elsewhere, so a selection naming it fails saying
//! why.

#[cfg(all(target_os = "android", feature = "mediacodec"))]
mod android;
#[cfg(all(
    any(target_os = "macos", target_os = "ios"),
    any(feature = "videotoolbox", feature = "audiotoolbox")
))]
mod apple;
#[cfg(all(target_os = "windows", feature = "media-foundation"))]
mod windows;

use ferricast_codec::{AudioDecoderRegistry, Registry, VideoDecoderRegistry};
use ferricast_core::BackendId;

/// Records `backend` as absent: missing its feature on its own platform,
/// elsewhere on any other.
fn absent<C, S: ?Sized>(
    registry: Registry<C, S>,
    backend: BackendId,
    feature: &'static str,
    on_its_platform: bool,
    platform: &'static str,
) -> Registry<C, S> {
    if on_its_platform {
        registry.missing(backend, feature)
    } else {
        registry.elsewhere(backend, feature, platform)
    }
}

#[allow(
    dead_code,
    reason = "conditional: unused on the OS it names once its feature is on"
)]
const WINDOWS: &str = "only on Windows";
#[allow(
    dead_code,
    reason = "conditional: unused on the OS it names once its feature is on"
)]
const APPLE: &str = "only on macOS and iOS";
#[allow(
    dead_code,
    reason = "conditional: unused on the OS it names once its feature is on"
)]
const ANDROID: &str = "only on Android";

/// Adds the platform backends of a video job.
pub(crate) fn video(registry: VideoDecoderRegistry) -> VideoDecoderRegistry {
    #[cfg(all(target_os = "windows", feature = "media-foundation"))]
    let registry = windows::video(registry);
    #[cfg(not(all(target_os = "windows", feature = "media-foundation")))]
    let registry = absent(
        registry,
        BackendId::MediaFoundation,
        "ferricast-decoder/media-foundation",
        cfg!(target_os = "windows"),
        WINDOWS,
    );
    #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox"))]
    let registry = apple::video(registry);
    #[cfg(not(all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox")))]
    let registry = absent(
        registry,
        BackendId::VideoToolbox,
        "ferricast-decoder/videotoolbox",
        cfg!(any(target_os = "macos", target_os = "ios")),
        APPLE,
    );
    #[cfg(all(target_os = "android", feature = "mediacodec"))]
    let registry = android::video(registry);
    #[cfg(not(all(target_os = "android", feature = "mediacodec")))]
    let registry = absent(
        registry,
        BackendId::MediaCodec,
        "ferricast-decoder/mediacodec",
        cfg!(target_os = "android"),
        ANDROID,
    );
    registry
}

/// Adds the platform backends of an audio job.
pub(crate) fn audio(registry: AudioDecoderRegistry) -> AudioDecoderRegistry {
    #[cfg(all(target_os = "windows", feature = "media-foundation"))]
    let registry = windows::audio(registry);
    #[cfg(not(all(target_os = "windows", feature = "media-foundation")))]
    let registry = absent(
        registry,
        BackendId::MediaFoundation,
        "ferricast-decoder/media-foundation",
        cfg!(target_os = "windows"),
        WINDOWS,
    );
    #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "audiotoolbox"))]
    let registry = apple::audio(registry);
    #[cfg(not(all(any(target_os = "macos", target_os = "ios"), feature = "audiotoolbox")))]
    let registry = absent(
        registry,
        BackendId::AudioToolbox,
        "ferricast-decoder/audiotoolbox",
        cfg!(any(target_os = "macos", target_os = "ios")),
        APPLE,
    );
    #[cfg(all(target_os = "android", feature = "mediacodec"))]
    let registry = android::audio(registry);
    #[cfg(not(all(target_os = "android", feature = "mediacodec")))]
    let registry = absent(
        registry,
        BackendId::MediaCodec,
        "ferricast-decoder/mediacodec",
        cfg!(target_os = "android"),
        ANDROID,
    );
    registry
}
