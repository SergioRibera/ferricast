//! AAC audio encoder facade.
//!
//! Single backend today: **`fdk-aac`** — Fraunhofer FDK AAC via
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
mod fdk_impl;

use ferricast_core::{AudioCodec, AudioEncoder, AudioEncoderConfig, AudioFrame, FerricastError, Result};

#[cfg(feature = "fdk-aac")]
pub use fdk_impl::FdkAacEncoder;

/// Backend-agnostic AAC encoder. `configure()` picks a backend at
/// first call and the variant stays put until the encoder is
/// dropped.
pub enum AacEncoder {
    Pending,
    #[cfg(feature = "fdk-aac")]
    Fdk(FdkAacEncoder),
}

impl AacEncoder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for AacEncoder {
    fn default() -> Self {
        AacEncoder::Pending
    }
}

impl AudioEncoder for AacEncoder {
    fn configure(&mut self, config: &AudioEncoderConfig) -> Result<()> {
        if config.codec != AudioCodec::Aac {
            return Err(FerricastError::Encoder(format!(
                "AacEncoder: only AAC codec is supported, got {:?}",
                config.codec
            )));
        }

        #[cfg(feature = "fdk-aac")]
        {
            let enc = FdkAacEncoder::new(config)?;
            tracing::info!(
                sample_rate = config.sample_rate,
                channels = config.channels,
                bitrate_kbps = config.bitrate_kbps,
                "AAC encoder backend: fdk-aac (libfdk-aac)"
            );
            *self = AacEncoder::Fdk(enc);
            return Ok(());
        }

        #[cfg(not(feature = "fdk-aac"))]
        {
            let _ = config;
            Err(FerricastError::Encoder(
                "AacEncoder: no backend compiled in (enable the `fdk-aac` feature)".into(),
            ))
        }
    }

    fn encode(&mut self, frame: &AudioFrame) -> Result<()> {
        match self {
            AacEncoder::Pending => Err(FerricastError::Encoder(
                "AacEncoder::encode called before configure()".into(),
            )),
            #[cfg(feature = "fdk-aac")]
            AacEncoder::Fdk(e) => e.encode(frame),
        }
    }

    fn take_output(&mut self) -> Vec<AudioFrame> {
        match self {
            AacEncoder::Pending => Vec::new(),
            #[cfg(feature = "fdk-aac")]
            AacEncoder::Fdk(e) => e.take_output(),
        }
    }

    fn codec_config(&self) -> Vec<u8> {
        match self {
            AacEncoder::Pending => Vec::new(),
            #[cfg(feature = "fdk-aac")]
            AacEncoder::Fdk(e) => e.codec_config(),
        }
    }

    fn flush(self) -> Result<Vec<AudioFrame>> {
        match self {
            AacEncoder::Pending => Ok(Vec::new()),
            #[cfg(feature = "fdk-aac")]
            AacEncoder::Fdk(e) => e.flush(),
        }
    }
}
