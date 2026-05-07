//! H.264 encoder facade.
//!
//! Picks a backend at runtime:
//! 1. **VA-API** via `cros-libva` — preferred when the system has a
//!    working DRM render node + an H.264 encode entrypoint
//!    (Constrained Baseline or Main). Output is a standard H.264
//!    bitstream compatible with Chromecast, Miracast and AirPlay.
//! 2. **x264** — software fallback. Always available because
//!    `libx264` is a build-time dep.
//!
//! Both backends implement the same [`VideoEncoder`] trait so
//! consumers (HLS server, Chromecast handler, etc.) don't need to
//! care which one is active.
//!
//! [`H264Encoder::default`] (and [`H264Encoder::new`]) preserves the
//! previous "just create the encoder, configure later" call shape
//! from when only x264 existed; runtime detection happens inside
//! `configure` so callers don't need to deal with the constructor
//! returning a fallible result.

mod vaapi_impl;
mod x264_impl;

use ferricast_core::{
    CapturedFrame, Codec, EncodedFrame, EncoderConfig, FerricastError, Result, VideoEncoder,
};
use tracing::{info, warn};

pub use vaapi_impl::VaapiH264Encoder;
pub use x264_impl::X264H264Encoder;

/// Backend-agnostic H.264 encoder. Internally an enum of the
/// available implementations; runtime detection happens at
/// [`Self::configure`] time so the constructor stays simple.
pub enum H264Encoder {
    /// Hardware VA-API path. Currently unreachable from the factory
    /// because the encode body is still a TODO; the variant exists
    /// so adding it later is a one-line change in the constructor.
    Vaapi(VaapiH264Encoder),
    /// Software x264 path. Always works.
    X264(X264H264Encoder),
}

impl H264Encoder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for H264Encoder {
    fn default() -> Self {
        // Cheap pre-detection — actual `configure()` will retry. We
        // can't fail the constructor signature, so this returns the
        // x264 fallback whenever VA-API can't come up.
        match VaapiH264Encoder::probe() {
            Ok(enc) => {
                info!("H.264 encoder backend: VA-API");
                H264Encoder::Vaapi(enc)
            }
            Err(e) => {
                // info! once at startup so the user sees why x264 is
                // active; debug! the actual error so it doesn't
                // clutter the steady state.
                info!("H.264 encoder backend: x264 (software)");
                tracing::debug!(error = %e, "VA-API probe failed");
                H264Encoder::X264(X264H264Encoder::default())
            }
        }
    }
}

impl VideoEncoder for H264Encoder {
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, config: &EncoderConfig) -> Result<()> {
        match self {
            H264Encoder::Vaapi(e) => match e.configure(config) {
                Ok(()) => Ok(()),
                Err(err) => {
                    // VA-API rejected the config (resolution / format
                    // not supported by the GPU). Swap to x264 in
                    // place rather than failing the call.
                    warn!(error = %err, "VA-API configure failed; falling back to x264");
                    let mut x = X264H264Encoder::default();
                    x.configure(config)?;
                    *self = H264Encoder::X264(x);
                    Ok(())
                }
            },
            H264Encoder::X264(e) => e.configure(config),
        }
    }

    fn encode(&mut self, frame: CapturedFrame) -> Result<EncodedFrame> {
        match self {
            H264Encoder::Vaapi(e) => e.encode(frame),
            H264Encoder::X264(e) => e.encode(frame),
        }
    }

    fn flush(self) -> Result<Vec<EncodedFrame>> {
        match self {
            H264Encoder::Vaapi(e) => e.flush(),
            H264Encoder::X264(e) => e.flush(),
        }
    }

    fn get_headers(&mut self) -> Result<Vec<u8>> {
        match self {
            H264Encoder::Vaapi(e) => e.get_headers(),
            H264Encoder::X264(e) => e.get_headers(),
        }
    }
}

/// Convenience: explicit error for callers that want to know whether
/// hardware encoding is active.
#[allow(dead_code)]
pub fn vaapi_available() -> bool {
    VaapiH264Encoder::probe().is_ok()
}

// Suppress unused-import lints when the vaapi module's public types
// aren't yet referenced from outside the crate.
const _: fn() = || {
    let _ = std::mem::size_of::<FerricastError>();
};
