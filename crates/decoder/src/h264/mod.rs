//! H.264 decoder facade.
//!
//! Auto-selects a backend at first `configure()`. Chain:
//!
//! 1. **NVDEC** — NVIDIA hardware. Tried first when `nvdec-decode` is
//!    compiled in and a CUDA-capable GPU is present at runtime.
//! 2. **VA-API** — opt-in only. Engaged when
//!    `FERRICAST_H264_DECODE_BACKEND=vaapi` is set in the environment.
//!    The slice-submission path is incomplete (see `vaapi_impl.rs`);
//!    leaving it out of the default chain prevents accidental
//!    engagement of a half-validated decode path.
//! 3. **openh264** — software CPU decode. Default fallback. Always
//!    works when compiled in.

#[cfg(not(feature = "openh264-decode"))]
compile_error!(
    "ferricast-decoder requires the `openh264-decode` feature today — \
     it's the only universally-working H.264 backend wired up. VA-API \
     is opt-in (off by default); NVDEC needs an NVIDIA GPU at runtime \
     so it can't be the sole backend."
);

#[cfg(feature = "openh264-decode")]
mod openh264_impl;
#[cfg(feature = "vaapi-decode")]
mod vaapi_impl;

use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, Result, VideoDecoder,
};
#[allow(unused_imports)]
use tracing::info;

#[cfg(feature = "nvdec-decode")]
pub use crate::nvdec::NvdecH264Decoder;
#[cfg(feature = "openh264-decode")]
pub use openh264_impl::OpenH264Decoder;
#[cfg(feature = "vaapi-decode")]
pub use vaapi_impl::VaapiH264Decoder;

const H264_BACKEND_VAR: &str = "FERRICAST_H264_DECODE_BACKEND";

pub enum H264Decoder {
    Pending,
    #[cfg(feature = "nvdec-decode")]
    Nvdec(NvdecH264Decoder),
    #[cfg(feature = "vaapi-decode")]
    Vaapi(VaapiH264Decoder),
    #[cfg(feature = "openh264-decode")]
    OpenH264(OpenH264Decoder),
}

impl H264Decoder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for H264Decoder {
    fn default() -> Self {
        H264Decoder::Pending
    }
}

impl VideoDecoder for H264Decoder {
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, config: &DecoderConfig) -> Result<()> {
        if !matches!(self, H264Decoder::Pending) {
            return match self {
                H264Decoder::Pending => unreachable!(),
                #[cfg(feature = "nvdec-decode")]
                H264Decoder::Nvdec(d) => d.configure(config),
                #[cfg(feature = "vaapi-decode")]
                H264Decoder::Vaapi(d) => d.configure(config),
                #[cfg(feature = "openh264-decode")]
                H264Decoder::OpenH264(d) => d.configure(config),
            };
        }

        let var = std::env::var(H264_BACKEND_VAR).unwrap_or_default();

        // NVDEC first: hardware decode on NVIDIA. Same multi-GPU
        // reasoning as the encoder facade — discrete NVIDIA + iGPU is
        // a common laptop combo, and engaging NVDEC keeps the entire
        // pipeline on the discrete GPU.
        #[cfg(feature = "nvdec-decode")]
        {
            if var.is_empty() || var == "nvdec" {
                match NvdecH264Decoder::probe() {
                    Ok(()) => {
                        let mut d = NvdecH264Decoder::new();
                        d.configure(config)?;
                        info!(
                            width = config.width,
                            height = config.height,
                            "H.264 decoder backend: NVDEC"
                        );
                        *self = H264Decoder::Nvdec(d);
                        return Ok(());
                    }
                    Err(e) => info!(error = %e, "NVDEC unavailable, trying VA-API / openh264"),
                }
            }
        }

        // VA-API is opt-in: only engage when the env var explicitly
        // names it. Default chain skips straight to openh264.
        #[cfg(feature = "vaapi-decode")]
        {
            if var == "vaapi" {
                match VaapiH264Decoder::probe_with(config) {
                    Ok(mut d) => {
                        d.configure(config)?;
                        info!(
                            width = config.width,
                            height = config.height,
                            "H.264 decoder backend: VA-API (opt-in via env var)"
                        );
                        *self = H264Decoder::Vaapi(d);
                        return Ok(());
                    }
                    Err(e) => {
                        info!(error = %e, "VA-API probe failed, falling back to openh264");
                    }
                }
            }
        }

        #[cfg(feature = "openh264-decode")]
        {
            if var.is_empty() || var == "openh264" || var == "vaapi" || var == "nvdec" {
                info!("H.264 decoder backend: openh264 (software)");
                let mut d = OpenH264Decoder::default();
                d.configure(config)?;
                *self = H264Decoder::OpenH264(d);
                return Ok(());
            }
        }

        Err(FerricastError::Decode(format!(
            "no H.264 decoder backend available for FERRICAST_H264_DECODE_BACKEND={var:?}"
        )))
    }

    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        match self {
            H264Decoder::Pending => Err(FerricastError::Decode(
                "H264Decoder::decode called before configure()".into(),
            )),
            #[cfg(feature = "nvdec-decode")]
            H264Decoder::Nvdec(d) => d.decode(frame),
            #[cfg(feature = "vaapi-decode")]
            H264Decoder::Vaapi(d) => d.decode(frame),
            #[cfg(feature = "openh264-decode")]
            H264Decoder::OpenH264(d) => d.decode(frame),
        }
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        match self {
            H264Decoder::Pending => Ok(Vec::new()),
            #[cfg(feature = "nvdec-decode")]
            H264Decoder::Nvdec(d) => d.flush(),
            #[cfg(feature = "vaapi-decode")]
            H264Decoder::Vaapi(d) => d.flush(),
            #[cfg(feature = "openh264-decode")]
            H264Decoder::OpenH264(d) => d.flush(),
        }
    }
}
