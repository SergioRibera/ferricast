//! H.264 decoder facade.
//!
//! Auto-selects a backend at first `configure()`. Today's chain:
//!
//! 1. **VA-API** — opt-in only. Engaged when
//!    `FERRICAST_H264_DECODE_BACKEND=vaapi` is set in the environment.
//!    The slice-submission path is incomplete (see
//!    `vaapi_impl.rs`); leaving it out of the default chain prevents
//!    accidental engagement of a half-validated decode path.
//! 2. **openh264** — software CPU decode. Default. Always works
//!    when compiled in.
//!
//! NVDEC slot is reserved for a follow-up — needs `cudarc` +
//! `libnvcuvid` + CUDA→DMA-BUF interop, which is its own
//! investigation.

#[cfg(not(feature = "openh264-decode"))]
compile_error!(
    "ferricast-decoder requires the `openh264-decode` feature today — \
     it's the only universally-working H.264 backend wired up. VA-API \
     is opt-in (off by default) and NVDEC follows in a later change."
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

#[cfg(feature = "openh264-decode")]
pub use openh264_impl::OpenH264Decoder;
#[cfg(feature = "vaapi-decode")]
pub use vaapi_impl::VaapiH264Decoder;

const H264_BACKEND_VAR: &str = "FERRICAST_H264_DECODE_BACKEND";

pub enum H264Decoder {
    Pending,
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
                #[cfg(feature = "vaapi-decode")]
                H264Decoder::Vaapi(d) => d.configure(config),
                #[cfg(feature = "openh264-decode")]
                H264Decoder::OpenH264(d) => d.configure(config),
            };
        }

        let var = std::env::var(H264_BACKEND_VAR).unwrap_or_default();

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
            if var.is_empty() || var == "openh264" || var == "vaapi" {
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
            #[cfg(feature = "vaapi-decode")]
            H264Decoder::Vaapi(d) => d.decode(frame),
            #[cfg(feature = "openh264-decode")]
            H264Decoder::OpenH264(d) => d.decode(frame),
        }
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        match self {
            H264Decoder::Pending => Ok(Vec::new()),
            #[cfg(feature = "vaapi-decode")]
            H264Decoder::Vaapi(d) => d.flush(),
            #[cfg(feature = "openh264-decode")]
            H264Decoder::OpenH264(d) => d.flush(),
        }
    }
}
