//! AAC audio encoder facade.
//!
//! Backends: the platform encoders (Media Foundation, AudioToolbox,
//! `MediaCodec`) where compiled in, then **`fdk-aac`** — Fraunhofer FDK AAC via
//! the C library. Reference encoder for AAC; transparent quality
//! above ~96 kbps stereo. Requires `libfdk-aac` on the link path
//! and the `fdk-aac` cargo feature (on by default).
//!
//! A pure-Rust fallback briefly lived here on top of `oxideav-aac`,
//! but upstream yanked the `encoder` module — leaving no production-
//! grade pure-Rust AAC encoder on crates.io. Without `fdk-aac` the
//! facade returns an error from `configure()`; the chromecast HLS
//! pipeline falls back to silent-AAC injection (no real audio).
//!
//! The `AudioEncoder` trait is the same surface regardless of which
//! backend is wired in; receiver sessions don't know which one is
//! running.

#[cfg(feature = "fdk-aac")]
mod eld;
#[cfg(feature = "fdk-aac")]
mod fdk_impl;
// Shared by the encoders that resample to AAC-ELD's 44.1 kHz themselves.
#[cfg(any(
    feature = "fdk-aac",
    all(any(target_os = "macos", target_os = "ios"), feature = "audiotoolbox")
))]
pub(crate) mod resample;

use ferricast_codec::AudioEncoderRegistry;
#[cfg(feature = "fdk-aac")]
use ferricast_codec::Candidate;
#[cfg(feature = "fdk-aac")]
use ferricast_core::{Acceleration, AudioEncoderConfig};
use ferricast_core::{AudioCodec, BackendId, CodecTarget};

#[cfg(feature = "fdk-aac")]
pub use eld::{ELD_AUDIO_SPECIFIC_CONFIG, ELD_FRAME_SAMPLES, ELD_SAMPLE_RATE, FdkEldEncoder};
#[cfg(feature = "fdk-aac")]
pub use fdk_impl::FdkAacEncoder;

ferricast_codec::audio_encoder_facade! {
    /// AAC-ELD encoder (480-sample granules, AirPlay mirroring audio)
    /// over whichever backend its configuration selects.
    pub struct AacEldEncoder, registry = eld_registry;
}

/// The AAC-ELD encode backends of this build.
fn eld_registry() -> AudioEncoderRegistry {
    let registry = AudioEncoderRegistry::new(CodecTarget::encode(AudioCodec::AacEld));
    let registry = crate::platform::audio(registry);
    #[cfg(feature = "fdk-aac")]
    let registry = registry.candidate(Candidate::new(
        BackendId::FdkAac,
        Acceleration::Software,
        ferricast_codec::open_audio_encoder::<FdkEldEncoder>,
    ));
    #[cfg(not(feature = "fdk-aac"))]
    let registry = registry.missing(BackendId::FdkAac, "ferricast-encoder/fdk-aac");
    registry
}

ferricast_codec::audio_encoder_facade! {
    /// AAC-LC encoder over whichever backend its configuration selects.
    pub struct AacEncoder, registry = registry;
}

/// The AAC-LC encode backends of this build, in automatic order.
fn registry() -> AudioEncoderRegistry {
    let registry = AudioEncoderRegistry::new(CodecTarget::encode(AudioCodec::Aac));
    let registry = crate::platform::audio(registry);
    #[cfg(feature = "fdk-aac")]
    let registry = registry.candidate(Candidate::new(
        BackendId::FdkAac,
        Acceleration::Software,
        |config: &AudioEncoderConfig| {
            FdkAacEncoder::open(config).map(|encoder| Box::new(encoder) as _)
        },
    ));
    #[cfg(not(feature = "fdk-aac"))]
    let registry = registry.missing(BackendId::FdkAac, "ferricast-encoder/fdk-aac");
    registry
}
