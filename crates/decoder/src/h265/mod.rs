//! HEVC (H.265) decoder facade.
//!
//! Auto-selects a backend at first `configure()`:
//!
//! 1. **NVDEC** via `shiguredo_nvcodec` — full HEVC decode in
//!    hardware on NVIDIA. Gated behind `nvdec-hevc-decode`.
//! 2. **VA-API** — opt-in via `FERRICAST_H265_DECODE_BACKEND=vaapi`.
//!    Profile probe + surface bring-up are complete; slice
//!    submission is not wired yet, so it is never picked unless the
//!    variable names it.
//! 3. **Software** via `rust_h265` behind `hevc-decode`; also the
//!    fallback when a backend named in the variable fails to open.
//!
//! Without any of the three compiled in, the facade refuses to build
//! via the `compile_error!` below.

#[cfg(not(any(
    feature = "nvdec-hevc-decode",
    feature = "vaapi-hevc-decode",
    feature = "hevc-decode"
)))]
compile_error!(
    "ferricast-decoder's H.265 facade requires at least one of \
     `nvdec-hevc-decode` / `vaapi-hevc-decode`. HEVC has no software \
     fallback — leave the H.265 facade off entirely if neither GPU \
     path is wanted."
);

#[cfg(feature = "hevc-decode")]
mod software_impl;
#[cfg(any(feature = "hevc-decode", feature = "nvdec-hevc-decode"))]
pub(crate) mod timestamps;
#[cfg(feature = "vaapi-hevc-decode")]
mod vaapi_impl;

#[cfg(feature = "nvdec-hevc-decode")]
use std::collections::VecDeque;

use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, Result, VideoDecoder,
};
use tracing::info;
#[cfg(feature = "nvdec-hevc-decode")]
use tracing::warn;

#[cfg(feature = "nvdec-hevc-decode")]
pub use crate::nvdec::NvdecH265Decoder;
#[cfg(feature = "nvdec-hevc-decode")]
use crate::nvdec::probation::Probation;
#[cfg(feature = "hevc-decode")]
pub use software_impl::SoftwareH265Decoder;
#[cfg(feature = "vaapi-hevc-decode")]
pub use vaapi_impl::VaapiH265Decoder;

const H265_BACKEND_VAR: &str = "FERRICAST_H265_DECODE_BACKEND";

#[derive(Default)]
pub struct H265Decoder {
    backend: Backend,
    /// Pictures the software decoder produced while replaying a stream
    /// NVDEC rejected; served ahead of anything it produces next.
    #[cfg(feature = "nvdec-hevc-decode")]
    handed_over: VecDeque<CapturedFrame>,
}

#[derive(Default)]
enum Backend {
    #[default]
    Pending,
    #[cfg(feature = "nvdec-hevc-decode")]
    Nvdec(Box<NvdecBackend>),
    #[cfg(feature = "vaapi-hevc-decode")]
    Vaapi(VaapiH265Decoder),
    #[cfg(feature = "hevc-decode")]
    Software(Box<SoftwareH265Decoder>),
}

/// NVDEC plus what it takes to hand a stream it rejects to the software
/// decoder.
#[cfg(feature = "nvdec-hevc-decode")]
struct NvdecBackend {
    decoder: NvdecH265Decoder,
    config: DecoderConfig,
    probation: Probation,
}

impl H265Decoder {
    pub fn new() -> Self {
        Self::default()
    }

    fn select(config: &DecoderConfig) -> Result<Backend> {
        let var = std::env::var(H265_BACKEND_VAR).unwrap_or_default();

        // NVDEC first when present + no explicit override.
        #[cfg(feature = "nvdec-hevc-decode")]
        {
            if var.is_empty() || var == "nvdec" {
                match NvdecH265Decoder::probe() {
                    Ok(()) => {
                        let mut d = NvdecH265Decoder::new();
                        match d.configure(config) {
                            Ok(()) => {
                                info!(
                                    width = config.width,
                                    height = config.height,
                                    "H.265 decoder backend: NVDEC"
                                );
                                return Ok(Backend::Nvdec(Box::new(NvdecBackend {
                                    decoder: d,
                                    config: config.clone(),
                                    probation: Probation::default(),
                                })));
                            }
                            Err(e) => warn!(
                                error = %e,
                                "NVDEC HEVC probed fine but would not open; trying the next backend"
                            ),
                        }
                    }
                    Err(e) => info!(error = %e, "NVDEC HEVC unavailable, trying the next backend"),
                }
            }
        }

        // Opt-in only: slice submission is not wired yet, so engaging it
        // by default would fail every decode.
        #[cfg(feature = "vaapi-hevc-decode")]
        {
            if var == "vaapi" {
                match VaapiH265Decoder::probe_with(config) {
                    Ok(mut d) => match d.configure(config) {
                        Ok(()) => {
                            info!(
                                width = config.width,
                                height = config.height,
                                "H.265 decoder backend: VA-API"
                            );
                            return Ok(Backend::Vaapi(d));
                        }
                        Err(e) => info!(error = %e, "VA-API HEVC configure failed"),
                    },
                    Err(e) => info!(error = %e, "VA-API HEVC probe failed"),
                }
            }
        }

        #[cfg(feature = "hevc-decode")]
        {
            // A named hardware backend that failed above still gets a
            // working decoder, as the H.264 facade does.
            if var.is_empty() || matches!(var.as_str(), "software" | "nvdec" | "vaapi") {
                info!("H.265 decoder backend: software");
                return Ok(Backend::Software(Box::new(software(config)?)));
            }
        }

        Err(FerricastError::Decode(format!(
            "no H.265 decoder backend available for FERRICAST_H265_DECODE_BACKEND={var:?}"
        )))
    }
}

#[cfg(feature = "hevc-decode")]
fn software(config: &DecoderConfig) -> Result<SoftwareH265Decoder> {
    let mut decoder = SoftwareH265Decoder::default();
    decoder.configure(config)?;
    Ok(decoder)
}

#[cfg(feature = "nvdec-hevc-decode")]
impl H265Decoder {
    fn decode_nvdec(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        let Backend::Nvdec(nvdec) = &mut self.backend else {
            return Err(FerricastError::Decode(
                "H.265 facade lost its NVDEC backend".into(),
            ));
        };
        nvdec.probation.record(&frame);
        match nvdec.decoder.decode(frame) {
            Ok(picture) => {
                if picture.is_some() {
                    nvdec.probation.settle();
                }
                Ok(picture)
            }
            Err(error) => {
                #[cfg(feature = "hevc-decode")]
                if let Some(replay) = nvdec.probation.take_replay() {
                    let config = nvdec.config.clone();
                    warn!(%error, "NVDEC rejected the HEVC stream; switching to software");
                    let software = self.replay_into_software(&config, replay)?;
                    self.backend = Backend::Software(Box::new(software));
                    return Ok(self.handed_over.pop_front());
                }
                Err(error)
            }
        }
    }

    fn flush_nvdec(&mut self) -> Result<Vec<CapturedFrame>> {
        let Backend::Nvdec(nvdec) = &mut self.backend else {
            return Err(FerricastError::Decode(
                "H.265 facade lost its NVDEC backend".into(),
            ));
        };
        match nvdec.decoder.flush() {
            Ok(frames) => Ok(frames),
            Err(error) => {
                #[cfg(feature = "hevc-decode")]
                if let Some(replay) = nvdec.probation.take_replay() {
                    let config = nvdec.config.clone();
                    warn!(%error, "NVDEC rejected the HEVC stream at flush; switching to software");
                    let mut software = self.replay_into_software(&config, replay)?;
                    self.handed_over.extend(software.flush()?);
                    self.backend = Backend::Software(Box::new(software));
                    return Ok(self.handed_over.drain(..).collect());
                }
                Err(error)
            }
        }
    }

    /// Decodes the packets NVDEC was fed into a fresh software decoder
    /// and queues its pictures. A packet the software decoder rejects is
    /// skipped, as it would have been had it decoded the stream from the
    /// start.
    #[cfg(feature = "hevc-decode")]
    fn replay_into_software(
        &mut self,
        config: &DecoderConfig,
        replay: Vec<EncodedFrame>,
    ) -> Result<SoftwareH265Decoder> {
        let mut software = software(config)?;
        for packet in replay {
            match software.decode(packet) {
                Ok(picture) => self.handed_over.extend(picture),
                Err(error) => warn!(%error, "software HEVC skipped a replayed packet"),
            }
        }
        Ok(software)
    }
}

impl VideoDecoder for H265Decoder {
    const CODEC: Codec = Codec::H265;

    fn configure(&mut self, config: &DecoderConfig) -> Result<()> {
        match &mut self.backend {
            Backend::Pending => {
                self.backend = Self::select(config)?;
                Ok(())
            }
            #[cfg(feature = "nvdec-hevc-decode")]
            Backend::Nvdec(d) => {
                d.config = config.clone();
                d.decoder.configure(config)
            }
            #[cfg(feature = "vaapi-hevc-decode")]
            Backend::Vaapi(d) => d.configure(config),
            #[cfg(feature = "hevc-decode")]
            Backend::Software(d) => d.configure(config),
        }
    }

    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        let decoded = match &mut self.backend {
            Backend::Pending => Err(FerricastError::Decode(
                "H265Decoder::decode called before configure()".into(),
            )),
            #[cfg(feature = "nvdec-hevc-decode")]
            Backend::Nvdec(_) => return self.decode_nvdec(frame),
            #[cfg(feature = "vaapi-hevc-decode")]
            Backend::Vaapi(d) => d.decode(frame),
            #[cfg(feature = "hevc-decode")]
            Backend::Software(d) => d.decode(frame),
        }?;
        #[cfg(feature = "nvdec-hevc-decode")]
        if !self.handed_over.is_empty() {
            self.handed_over.extend(decoded);
            return Ok(self.handed_over.pop_front());
        }
        Ok(decoded)
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        let flushed = match &mut self.backend {
            Backend::Pending => Vec::new(),
            #[cfg(feature = "nvdec-hevc-decode")]
            Backend::Nvdec(_) => return self.flush_nvdec(),
            #[cfg(feature = "vaapi-hevc-decode")]
            Backend::Vaapi(d) => d.flush()?,
            #[cfg(feature = "hevc-decode")]
            Backend::Software(d) => d.flush()?,
        };
        #[cfg(feature = "nvdec-hevc-decode")]
        if !self.handed_over.is_empty() {
            self.handed_over.extend(flushed);
            return Ok(self.handed_over.drain(..).collect());
        }
        Ok(flushed)
    }
}
