//! HEVC (H.265) decoder facade.
//!
//! Auto-selects a backend at first `configure()`:
//!
//! 1. **NVDEC** via `shiguredo_nvcodec` — full HEVC decode in
//!    hardware on NVIDIA. Gated behind `nvdec-hevc-decode`.
//! 2. **VA-API** — opt-in via `FERRICAST_H265_DECODE_BACKEND=vaapi`.
//!    Profile probe + surface bring-up are complete; slice
//!    submission is the same opt-in pattern as the H.264 VAAPI
//!    backend (engage only if you intend to drive it).
//!
//! No software fallback: HEVC has no production-grade pure-Rust
//! decoder, and `openh264` doesn't speak HEVC. Without one of the
//! two GPU backends compiled in, the facade refuses to build via
//! the `compile_error!` below.

#[cfg(not(any(feature = "nvdec-hevc-decode", feature = "vaapi-hevc-decode")))]
compile_error!(
    "ferricast-decoder's H.265 facade requires at least one of \
     `nvdec-hevc-decode` / `vaapi-hevc-decode`. HEVC has no software \
     fallback — leave the H.265 facade off entirely if neither GPU \
     path is wanted."
);

#[cfg(feature = "vaapi-hevc-decode")]
mod vaapi_impl;

use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, Result, VideoDecoder,
};
#[allow(unused_imports)]
use tracing::info;

#[cfg(feature = "nvdec-hevc-decode")]
pub use crate::nvdec::NvdecH265Decoder;
#[cfg(feature = "vaapi-hevc-decode")]
pub use vaapi_impl::VaapiH265Decoder;

const H265_BACKEND_VAR: &str = "FERRICAST_H265_DECODE_BACKEND";

pub enum H265Decoder {
    Pending,
    #[cfg(feature = "nvdec-hevc-decode")]
    Nvdec(NvdecH265Decoder),
    #[cfg(feature = "vaapi-hevc-decode")]
    Vaapi(VaapiH265Decoder),
}

impl H265Decoder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for H265Decoder {
    fn default() -> Self {
        H265Decoder::Pending
    }
}

impl VideoDecoder for H265Decoder {
    const CODEC: Codec = Codec::H265;

    fn configure(&mut self, config: &DecoderConfig) -> Result<()> {
        if !matches!(self, H265Decoder::Pending) {
            return match self {
                H265Decoder::Pending => unreachable!(),
                #[cfg(feature = "nvdec-hevc-decode")]
                H265Decoder::Nvdec(d) => d.configure(config),
                #[cfg(feature = "vaapi-hevc-decode")]
                H265Decoder::Vaapi(d) => d.configure(config),
            };
        }

        let var = std::env::var(H265_BACKEND_VAR).unwrap_or_default();

        // NVDEC first when present + no explicit override.
        #[cfg(feature = "nvdec-hevc-decode")]
        {
            if var.is_empty() || var == "nvdec" {
                match NvdecH265Decoder::probe() {
                    Ok(()) => {
                        let mut d = NvdecH265Decoder::new();
                        d.configure(config)?;
                        info!(
                            width = config.width,
                            height = config.height,
                            "H.265 decoder backend: NVDEC"
                        );
                        *self = H265Decoder::Nvdec(d);
                        return Ok(());
                    }
                    Err(e) => info!(error = %e, "NVDEC HEVC unavailable, trying VA-API"),
                }
            }
        }

        #[cfg(feature = "vaapi-hevc-decode")]
        {
            if var == "vaapi" || var.is_empty() {
                match VaapiH265Decoder::probe_with(config) {
                    Ok(mut d) => match d.configure(config) {
                        Ok(()) => {
                            info!(
                                width = config.width,
                                height = config.height,
                                "H.265 decoder backend: VA-API"
                            );
                            *self = H265Decoder::Vaapi(d);
                            return Ok(());
                        }
                        Err(e) => info!(error = %e, "VA-API HEVC configure failed"),
                    },
                    Err(e) => info!(error = %e, "VA-API HEVC probe failed"),
                }
            }
        }

        Err(FerricastError::Decode(format!(
            "no H.265 decoder backend available for FERRICAST_H265_DECODE_BACKEND={var:?} \
             — HEVC requires GPU hardware (NVDEC or VA-API)"
        )))
    }

    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        match self {
            H265Decoder::Pending => Err(FerricastError::Decode(
                "H265Decoder::decode called before configure()".into(),
            )),
            #[cfg(feature = "nvdec-hevc-decode")]
            H265Decoder::Nvdec(d) => d.decode(frame),
            #[cfg(feature = "vaapi-hevc-decode")]
            H265Decoder::Vaapi(d) => d.decode(frame),
        }
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        match self {
            H265Decoder::Pending => Ok(Vec::new()),
            #[cfg(feature = "nvdec-hevc-decode")]
            H265Decoder::Nvdec(d) => d.flush(),
            #[cfg(feature = "vaapi-hevc-decode")]
            H265Decoder::Vaapi(d) => d.flush(),
        }
    }
}
