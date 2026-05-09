//! H.264 encoder facade.
//!
//! Picks a backend at runtime, in order of preference:
//! 1. **VA-API** via `cros-libva` — Intel iGPU + AMD Mesa. Encode
//!    on the GPU, no CPU readback needed once the dmabuf import
//!    path is wired.
//! 2. **NVENC** via `shiguredo_nvcodec` — NVIDIA proprietary
//!    driver. NVIDIA's VA-API implementation is decode-only, so
//!    this is the path that actually engages the GPU on NVIDIA
//!    hardware.
//! 3. **x264** — software fallback. Always available because
//!    `libx264` is a build-time dep.
//!
//! All three backends implement the same [`VideoEncoder`] trait so
//! consumers (HLS server, Chromecast handler, etc.) don't need to
//! care which one is active. Output is plain H.264 — every
//! receiver protocol we ship (Chromecast / Miracast / AirPlay)
//! decodes any of the three backends' bitstreams identically.
//!
//! Construction has two phases:
//! * `H264Encoder::new()` returns an enum stuck in `Pending` —
//!   detection is deferred so we can size surfaces / SPS / NVENC
//!   session correctly from the user's actual resolution.
//! * `configure(&EncoderConfig)` walks the preference list with
//!   that config; on any failure at any backend (no driver, profile
//!   not supported, session creation refused, ...) it transparently
//!   tries the next one.

mod bitstream;
mod headers;
mod nvenc_impl;
mod vaapi_impl;
mod x264_impl;
mod yuv;

use ferricast_core::{
    CapturedFrame, Codec, EncodedFrame, EncoderConfig, FerricastError, Result, VideoEncoder,
};
use tracing::{info, warn};

pub use nvenc_impl::NvencH264Encoder;
pub use vaapi_impl::VaapiH264Encoder;
pub use x264_impl::X264H264Encoder;

/// Backend-agnostic H.264 encoder. Internal enum picks a backend in
/// `configure()`.
pub enum H264Encoder {
    /// Pre-configure placeholder. Replaced with one of the concrete
    /// variants on first `configure()` call.
    Pending,
    /// Hardware VA-API path — Intel / AMD.
    Vaapi(VaapiH264Encoder),
    /// Hardware NVENC path — NVIDIA.
    Nvenc(NvencH264Encoder),
    /// Software x264 path — always works.
    X264(X264H264Encoder),
}

impl H264Encoder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for H264Encoder {
    fn default() -> Self {
        H264Encoder::Pending
    }
}

impl VideoEncoder for H264Encoder {
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, config: &EncoderConfig) -> Result<()> {
        match self {
            H264Encoder::Pending => {
                // Try VA-API first; if that fails (NVIDIA, no
                // libva, profile mismatch, ...) try NVENC; if that
                // also fails, fall back to x264. We always end up
                // with a working encoder — x264 is the floor.
                match VaapiH264Encoder::probe_with(config.clone()) {
                    Ok(enc) => {
                        info!(
                            width = config.width,
                            height = config.height,
                            fps = config.fps,
                            "H.264 encoder backend: VA-API"
                        );
                        *self = H264Encoder::Vaapi(enc);
                        return Ok(());
                    }
                    Err(e) => info!(error = %e, "VA-API unavailable, trying NVENC"),
                }

                match NvencH264Encoder::probe_with(config.clone()) {
                    Ok(enc) => {
                        info!(
                            width = config.width,
                            height = config.height,
                            fps = config.fps,
                            "H.264 encoder backend: NVENC"
                        );
                        *self = H264Encoder::Nvenc(enc);
                        return Ok(());
                    }
                    Err(e) => info!(
                        error = %e,
                        "NVENC unavailable, falling back to x264. \
                         If you expected NVENC, ensure libcuda.so.1 + \
                         libnvidia-encode.so.1 are on LD_LIBRARY_PATH \
                         (NixOS: /run/opengl-driver/lib)."
                    ),
                }

                info!("H.264 encoder backend: x264 (software)");
                let mut x = X264H264Encoder::default();
                x.configure(config)?;
                *self = H264Encoder::X264(x);
                Ok(())
            }
            H264Encoder::Vaapi(e) => match e.configure(config) {
                Ok(()) => Ok(()),
                Err(err) => {
                    warn!(error = %err, "VA-API reconfigure failed, switching to x264");
                    let mut x = X264H264Encoder::default();
                    x.configure(config)?;
                    *self = H264Encoder::X264(x);
                    Ok(())
                }
            },
            H264Encoder::Nvenc(e) => match e.configure(config) {
                Ok(()) => Ok(()),
                Err(err) => {
                    warn!(error = %err, "NVENC reconfigure failed, switching to x264");
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
            H264Encoder::Pending => Err(FerricastError::Encoder(
                "H264Encoder::encode called before configure()".into(),
            )),
            H264Encoder::Vaapi(e) => e.encode(frame),
            H264Encoder::Nvenc(e) => e.encode(frame),
            H264Encoder::X264(e) => e.encode(frame),
        }
    }

    fn flush(self) -> Result<Vec<EncodedFrame>> {
        match self {
            H264Encoder::Pending => Ok(Vec::new()),
            H264Encoder::Vaapi(e) => e.flush(),
            H264Encoder::Nvenc(e) => e.flush(),
            H264Encoder::X264(e) => e.flush(),
        }
    }

    fn get_headers(&mut self) -> Result<Vec<u8>> {
        match self {
            H264Encoder::Pending => Err(FerricastError::Encoder(
                "H264Encoder::get_headers called before configure()".into(),
            )),
            H264Encoder::Vaapi(e) => e.get_headers(),
            H264Encoder::Nvenc(e) => e.get_headers(),
            H264Encoder::X264(e) => e.get_headers(),
        }
    }

    fn request_keyframe(&mut self) {
        match self {
            // No backend wired up yet; the request is dropped.
            // Once `configure()` runs the next call lands on a
            // real backend.
            H264Encoder::Pending => {}
            H264Encoder::Vaapi(e) => e.request_keyframe(),
            H264Encoder::Nvenc(e) => e.request_keyframe(),
            // x264 via the safe `x264` crate has no exposed knob to
            // override `picture.i_type`; the segmenter pace path
            // still bumps frames at fps but segment cuts will fall
            // on the encoder's natural keyint until/unless we drop
            // to x264 FFI.
            H264Encoder::X264(_) => {}
        }
    }
}

/// Returns true when a VA-API H.264 encoder can be brought up on
/// this system at the given resolution / fps. Useful for telemetry
/// and config UIs.
#[allow(dead_code)]
pub fn vaapi_available(config: &EncoderConfig) -> bool {
    VaapiH264Encoder::probe_with(config.clone()).is_ok()
}
