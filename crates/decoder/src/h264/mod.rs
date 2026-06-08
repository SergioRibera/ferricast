//! H.264 decoder facade.
//!
//! Auto-selects a backend at first `configure()`. Today there's one
//! backend (openh264, CPU). The facade shape mirrors the encoder's
//! `H264Encoder` so when the GPU backends land — VA-API via
//! cros-libva, NVDEC via cudarc — they slot in as additional
//! variants without changing the public type or its trait impl.
//!
//! Override at runtime with the `FERRICAST_H264_DECODE_BACKEND` env
//! var (values: `openh264` today; `vaapi` / `nvdec` once those land).

#[cfg(not(feature = "openh264-decode"))]
compile_error!(
    "ferricast-decoder requires the `openh264-decode` feature today — \
     it's the only H.264 backend wired up. GPU paths follow in a later \
     change without changing the facade type."
);

#[cfg(feature = "openh264-decode")]
mod openh264_impl;

use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, Result, VideoDecoder,
};
#[allow(unused_imports)]
use tracing::info;

#[cfg(feature = "openh264-decode")]
pub use openh264_impl::OpenH264Decoder;

const H264_BACKEND_VAR: &str = "FERRICAST_H264_DECODE_BACKEND";

pub enum H264Decoder {
    Pending,
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
                #[cfg(feature = "openh264-decode")]
                H264Decoder::OpenH264(d) => d.configure(config),
            };
        }

        let var = std::env::var(H264_BACKEND_VAR).unwrap_or_default();

        #[cfg(feature = "openh264-decode")]
        {
            if var.is_empty() || var == "openh264" {
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
            #[cfg(feature = "openh264-decode")]
            H264Decoder::OpenH264(d) => d.decode(frame),
        }
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        match self {
            H264Decoder::Pending => Ok(Vec::new()),
            #[cfg(feature = "openh264-decode")]
            H264Decoder::OpenH264(d) => d.flush(),
        }
    }
}
