//! HEVC (H.265) encoder facade.
//!
//! Picks a backend at runtime, in order of preference:
//!
//! 1. **NVENC** via `shiguredo_nvcodec` — NVIDIA hardware. Gated
//!    behind `nvenc-hevc`.
//! 2. **VA-API** via `cros-libva` — Intel iGPU / AMD Mesa hardware.
//!    Gated behind `vaapi-hevc`.
//! 3. *(no software fallback)* — `openh264` doesn't speak HEVC and
//!    no production-grade pure-Rust HEVC encoder exists. If neither
//!    hardware path is available, `configure()` returns an error and
//!    the caller is expected to downgrade to H.264.
//!
//! At least one of `nvenc-hevc` / `vaapi-hevc` MUST be enabled — the
//! `compile_error!` below blocks builds that turn both off, since the
//! facade has nothing to dispatch to.

#[cfg(not(any(feature = "nvenc-hevc", feature = "vaapi-hevc")))]
compile_error!(
    "ferricast-encoder's H.265 facade requires at least one of the `nvenc-hevc` / \
     `vaapi-hevc` features. HEVC has no software fallback — disable the H.265 \
     facade entirely if neither GPU path is available."
);

#[cfg(feature = "vaapi-hevc")]
mod bitstream;
#[cfg(feature = "vaapi-hevc")]
mod headers;
#[cfg(feature = "vaapi-hevc")]
mod vaapi_impl;

use ferricast_core::{
    CapturedFrame, Codec, EncodedFrame, EncoderConfig, FerricastError, Result, VideoEncoder,
};
#[allow(unused_imports)]
use tracing::{info, warn};

#[cfg(feature = "nvenc-hevc")]
pub use crate::nvenc::NvencH265Encoder;
#[cfg(feature = "vaapi-hevc")]
pub use vaapi_impl::VaapiH265Encoder;

const H265_BACKEND_VAR: &'static str = "FERRICAST_H265_BACKEND";

pub enum H265Encoder {
    Pending,
    #[cfg(feature = "vaapi-hevc")]
    Vaapi(VaapiH265Encoder),
    #[cfg(feature = "nvenc-hevc")]
    Nvenc(NvencH265Encoder),
}

impl H265Encoder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for H265Encoder {
    fn default() -> Self {
        H265Encoder::Pending
    }
}

impl VideoEncoder for H265Encoder {
    const CODEC: Codec = Codec::H265;

    fn configure(&mut self, config: &EncoderConfig) -> Result<()> {
        match self {
            H265Encoder::Pending => {
                let var = std::env::var(H265_BACKEND_VAR).unwrap_or_default();

                // NVENC first — same multi-GPU rationale as the H.264
                // facade: discrete NVIDIA + iGPU is a very common laptop
                // combo, and an iGPU VA-API path forces every frame
                // across PCIe twice.
                #[cfg(feature = "nvenc-hevc")]
                {
                    if var.is_empty() || var == "nvenc" {
                        match NvencH265Encoder::probe_with(config.clone()) {
                            Ok(enc) => {
                                info!(
                                    width = config.width,
                                    height = config.height,
                                    fps = config.fps,
                                    "H.265 encoder backend: NVENC"
                                );
                                *self = H265Encoder::Nvenc(enc);
                                return Ok(());
                            }
                            Err(e) => info!(error = %e, "NVENC HEVC unavailable, trying VA-API"),
                        }
                    }
                }

                #[cfg(feature = "vaapi-hevc")]
                {
                    if var.is_empty() || var == "vaapi" {
                        match VaapiH265Encoder::probe_with(config.clone()) {
                            Ok(enc) => {
                                info!(
                                    width = config.width,
                                    height = config.height,
                                    fps = config.fps,
                                    "H.265 encoder backend: VA-API"
                                );
                                *self = H265Encoder::Vaapi(enc);
                                return Ok(());
                            }
                            Err(e) => info!(error = %e, "VA-API HEVC unavailable"),
                        }
                    }
                }

                Err(FerricastError::Encoder(format!(
                    "no H.265 encoder backend available for FERRICAST_H265_BACKEND={var:?} \
                     — HEVC requires GPU hardware (NVENC or VA-API)"
                )))
            }
            #[cfg(feature = "vaapi-hevc")]
            H265Encoder::Vaapi(e) => e.configure(config),
            #[cfg(feature = "nvenc-hevc")]
            H265Encoder::Nvenc(e) => e.configure(config),
        }
    }

    fn encode(&mut self, frame: CapturedFrame) -> Result<EncodedFrame> {
        match self {
            H265Encoder::Pending => Err(FerricastError::Encoder(
                "H265Encoder::encode called before configure()".into(),
            )),
            #[cfg(feature = "vaapi-hevc")]
            H265Encoder::Vaapi(e) => e.encode(frame),
            #[cfg(feature = "nvenc-hevc")]
            H265Encoder::Nvenc(e) => e.encode(frame),
        }
    }

    fn flush(self) -> Result<Vec<EncodedFrame>> {
        match self {
            H265Encoder::Pending => Ok(Vec::new()),
            #[cfg(feature = "vaapi-hevc")]
            H265Encoder::Vaapi(e) => e.flush(),
            #[cfg(feature = "nvenc-hevc")]
            H265Encoder::Nvenc(e) => e.flush(),
        }
    }

    fn get_headers(&mut self) -> Result<Vec<u8>> {
        match self {
            H265Encoder::Pending => Err(FerricastError::Encoder(
                "H265Encoder::get_headers called before configure()".into(),
            )),
            #[cfg(feature = "vaapi-hevc")]
            H265Encoder::Vaapi(e) => e.get_headers(),
            #[cfg(feature = "nvenc-hevc")]
            H265Encoder::Nvenc(e) => e.get_headers(),
        }
    }

    fn request_keyframe(&mut self) {
        match self {
            H265Encoder::Pending => {}
            #[cfg(feature = "vaapi-hevc")]
            H265Encoder::Vaapi(e) => e.request_keyframe(),
            #[cfg(feature = "nvenc-hevc")]
            H265Encoder::Nvenc(e) => e.request_keyframe(),
        }
    }

    fn set_bitrate_kbps(&mut self, kbps: u32) -> Result<()> {
        match self {
            H265Encoder::Pending => Ok(()),
            #[cfg(feature = "vaapi-hevc")]
            H265Encoder::Vaapi(e) => e.set_bitrate_kbps(kbps),
            #[cfg(feature = "nvenc-hevc")]
            H265Encoder::Nvenc(e) => e.set_bitrate_kbps(kbps),
        }
    }
}
