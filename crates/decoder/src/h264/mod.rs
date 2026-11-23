//! H.264 decoder facade.
//!
//! Auto-selects a backend at first `configure()`. Chain:
//!
//! 1. **NVDEC** — NVIDIA hardware. Tried first when `nvdec-decode` is
//!    compiled in and a CUDA-capable GPU is present at runtime.
//! 2. **VA-API** — Intel/AMD hardware, when `vaapi-decode` is compiled in
//!    and a driver accepts the stream's profile. A stream using a feature
//!    it does not implement (interlacing, 4:2:2, high bit depth, scaling
//!    matrices, FMO) is handed over to openh264 mid-stream.
//! 3. **openh264** — software CPU decode. Default fallback. Always
//!    works when compiled in.
//!
//! A hardware backend that fails before producing its first picture
//! hands the stream to openh264 from its first packet, so a driver
//! that accepts the configuration but not the bitstream never breaks
//! playback. `FERRICAST_H264_DECODE_BACKEND` (`nvdec`, `vaapi`,
//! `software`) forces one backend.

#[cfg(not(feature = "openh264-decode"))]
compile_error!(
    "ferricast-decoder requires the `openh264-decode` feature today — \
     it's the only universally-working H.264 backend wired up. VA-API \
     and NVDEC depend on the GPU and driver present at runtime, \
     so neither can be the sole backend."
);

#[cfg(feature = "vaapi-decode")]
mod dpb;
#[cfg(feature = "openh264-decode")]
mod openh264_impl;
#[cfg(feature = "vaapi-decode")]
mod slice;
#[cfg(feature = "vaapi-decode")]
mod vaapi_impl;

#[cfg(any(feature = "nvdec-decode", feature = "vaapi-decode"))]
use std::collections::VecDeque;

#[cfg(feature = "vaapi-decode")]
use bytes::Bytes;
use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, Result, VideoDecoder,
};
use tracing::info;
#[cfg(any(feature = "nvdec-decode", feature = "vaapi-decode"))]
use tracing::warn;

#[cfg(feature = "nvdec-decode")]
pub use crate::nvdec::NvdecH264Decoder;
#[cfg(any(feature = "nvdec-decode", feature = "vaapi-decode"))]
use crate::probation::Probation;
#[cfg(feature = "openh264-decode")]
pub use openh264_impl::OpenH264Decoder;
#[cfg(feature = "vaapi-decode")]
pub use vaapi_impl::{VaapiH264Decoder, VaapiH264Error};

const H264_BACKEND_VAR: &str = "FERRICAST_H264_DECODE_BACKEND";

#[derive(Default)]
pub struct H264Decoder {
    backend: Backend,
    /// Pictures a hardware backend finished (or openh264 produced while
    /// replaying a stream NVDEC rejected) before the stream moved to
    /// openh264; served ahead of anything openh264 produces next.
    #[cfg(any(feature = "nvdec-decode", feature = "vaapi-decode"))]
    handed_over: VecDeque<CapturedFrame>,
}

#[derive(Default)]
enum Backend {
    #[default]
    Pending,
    #[cfg(feature = "nvdec-decode")]
    Nvdec(Box<NvdecBackend>),
    #[cfg(feature = "vaapi-decode")]
    Vaapi(Box<VaapiBackend>),
    #[cfg(feature = "openh264-decode")]
    OpenH264(OpenH264Decoder),
}

impl H264Decoder {
    pub fn new() -> Self {
        Self::default()
    }

    fn select(config: &DecoderConfig) -> Result<Backend> {
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
                        match d.configure(config) {
                            Ok(()) => {
                                info!(
                                    width = config.width,
                                    height = config.height,
                                    "H.264 decoder backend: NVDEC"
                                );
                                return Ok(Backend::Nvdec(Box::new(NvdecBackend {
                                    decoder: d,
                                    config: config.clone(),
                                    probation: Probation::default(),
                                })));
                            }
                            Err(e) => warn!(
                                error = %e,
                                "NVDEC probed fine but would not open; trying VA-API / openh264"
                            ),
                        }
                    }
                    Err(e) => info!(error = %e, "NVDEC unavailable, trying VA-API / openh264"),
                }
            }
        }

        #[cfg(feature = "vaapi-decode")]
        {
            if var.is_empty() || var == "vaapi" {
                match VaapiBackend::open(config) {
                    Ok(d) => {
                        info!(
                            width = config.width,
                            height = config.height,
                            "H.264 decoder backend: VA-API"
                        );
                        return Ok(Backend::Vaapi(Box::new(d)));
                    }
                    Err(e) => {
                        info!(error = %e, "VA-API unavailable, falling back to openh264");
                    }
                }
            }
        }

        #[cfg(feature = "openh264-decode")]
        {
            if var.is_empty() || matches!(var.as_str(), "software" | "openh264" | "vaapi" | "nvdec")
            {
                info!("H.264 decoder backend: openh264 (software)");
                return Ok(Backend::OpenH264(software(config)?));
            }
        }

        Err(FerricastError::Decode(format!(
            "no H.264 decoder backend available for FERRICAST_H264_DECODE_BACKEND={var:?}"
        )))
    }

    #[cfg(feature = "vaapi-decode")]
    fn decode_vaapi(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        let Backend::Vaapi(vaapi) = &mut self.backend else {
            return Err(FerricastError::Decode(
                "H.264 facade lost its VA-API backend".into(),
            ));
        };
        vaapi.parameter_sets.remember(&frame.data);
        vaapi.probation.record(&frame);
        let replay = frame.clone();
        match vaapi.decoder.decode_packet(frame) {
            Ok(picture) => {
                if picture.is_some() {
                    vaapi.probation.settle();
                }
                Ok(picture)
            }
            Err(VaapiH264Error::Unsupported(reason)) => {
                warn!(%reason, "VA-API cannot decode this stream; switching to openh264");
                let mut software = software(&vaapi.config)?;
                match vaapi.decoder.flush() {
                    Ok(frames) => self.handed_over.extend(frames),
                    Err(error) => warn!(%error, "VA-API flush before the handover failed"),
                }
                let replay = EncodedFrame {
                    data: vaapi.parameter_sets.prepend(&replay.data),
                    ..replay
                };
                let first = software.decode(replay);
                self.backend = Backend::OpenH264(software);
                self.handed_over.extend(first?);
                Ok(self.handed_over.pop_front())
            }
            Err(VaapiH264Error::Failed(reason)) => {
                let Some(replay) = vaapi.probation.take_replay() else {
                    return Err(FerricastError::Decode(reason));
                };
                let config = vaapi.config.clone();
                warn!(%reason, "VA-API failed on the stream; switching to openh264");
                let software = self.replay_into_software(&config, replay)?;
                self.backend = Backend::OpenH264(software);
                Ok(self.handed_over.pop_front())
            }
        }
    }
}

#[cfg(any(feature = "nvdec-decode", feature = "vaapi-decode"))]
impl H264Decoder {
    /// Decodes the packets a hardware backend was fed into a fresh
    /// openh264 decoder and queues its pictures. A packet openh264
    /// rejects (e.g. a P slice before the first IDR) is skipped, as it
    /// would have been had openh264 decoded the stream from the start.
    fn replay_into_software(
        &mut self,
        config: &DecoderConfig,
        replay: Vec<EncodedFrame>,
    ) -> Result<OpenH264Decoder> {
        let mut software = software(config)?;
        for packet in replay {
            match software.decode(packet) {
                Ok(picture) => self.handed_over.extend(picture),
                Err(error) => warn!(%error, "openh264 skipped a replayed packet"),
            }
        }
        Ok(software)
    }
}

/// NVDEC plus what it takes to hand a stream it rejects to openh264.
#[cfg(feature = "nvdec-decode")]
struct NvdecBackend {
    decoder: NvdecH264Decoder,
    config: DecoderConfig,
    probation: Probation,
}

#[cfg(feature = "nvdec-decode")]
impl H264Decoder {
    fn decode_nvdec(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        let Backend::Nvdec(nvdec) = &mut self.backend else {
            return Err(FerricastError::Decode(
                "H.264 facade lost its NVDEC backend".into(),
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
                let Some(replay) = nvdec.probation.take_replay() else {
                    return Err(error);
                };
                let config = nvdec.config.clone();
                warn!(%error, "NVDEC rejected the stream; switching to openh264");
                let software = self.replay_into_software(&config, replay)?;
                self.backend = Backend::OpenH264(software);
                Ok(self.handed_over.pop_front())
            }
        }
    }

    fn flush_nvdec(&mut self) -> Result<Vec<CapturedFrame>> {
        let Backend::Nvdec(nvdec) = &mut self.backend else {
            return Err(FerricastError::Decode(
                "H.264 facade lost its NVDEC backend".into(),
            ));
        };
        match nvdec.decoder.flush() {
            Ok(frames) => Ok(frames),
            Err(error) => {
                let Some(replay) = nvdec.probation.take_replay() else {
                    return Err(error);
                };
                let config = nvdec.config.clone();
                warn!(%error, "NVDEC rejected the stream at flush; switching to openh264");
                let mut software = self.replay_into_software(&config, replay)?;
                self.handed_over.extend(software.flush()?);
                self.backend = Backend::OpenH264(software);
                Ok(self.handed_over.drain(..).collect())
            }
        }
    }
}

#[cfg(feature = "openh264-decode")]
fn software(config: &DecoderConfig) -> Result<OpenH264Decoder> {
    let mut decoder = OpenH264Decoder::default();
    decoder.configure(config)?;
    Ok(decoder)
}

impl VideoDecoder for H264Decoder {
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, config: &DecoderConfig) -> Result<()> {
        match &mut self.backend {
            Backend::Pending => {
                self.backend = Self::select(config)?;
                Ok(())
            }
            #[cfg(feature = "nvdec-decode")]
            Backend::Nvdec(d) => {
                d.config = config.clone();
                d.decoder.configure(config)
            }
            #[cfg(feature = "vaapi-decode")]
            Backend::Vaapi(d) => {
                d.config = config.clone();
                d.decoder.configure(config)
            }
            #[cfg(feature = "openh264-decode")]
            Backend::OpenH264(d) => d.configure(config),
        }
    }

    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        let decoded = match &mut self.backend {
            Backend::Pending => {
                return Err(FerricastError::Decode(
                    "H264Decoder::decode called before configure()".into(),
                ));
            }
            #[cfg(feature = "nvdec-decode")]
            Backend::Nvdec(_) => return self.decode_nvdec(frame),
            #[cfg(feature = "vaapi-decode")]
            Backend::Vaapi(_) => return self.decode_vaapi(frame),
            #[cfg(feature = "openh264-decode")]
            Backend::OpenH264(d) => d.decode(frame),
        }?;
        #[cfg(any(feature = "nvdec-decode", feature = "vaapi-decode"))]
        if !self.handed_over.is_empty() {
            self.handed_over.extend(decoded);
            return Ok(self.handed_over.pop_front());
        }
        Ok(decoded)
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        let flushed = match &mut self.backend {
            Backend::Pending => Vec::new(),
            #[cfg(feature = "nvdec-decode")]
            Backend::Nvdec(_) => return self.flush_nvdec(),
            #[cfg(feature = "vaapi-decode")]
            Backend::Vaapi(d) => d.decoder.flush()?,
            #[cfg(feature = "openh264-decode")]
            Backend::OpenH264(d) => d.flush()?,
        };
        #[cfg(any(feature = "nvdec-decode", feature = "vaapi-decode"))]
        if !self.handed_over.is_empty() {
            self.handed_over.extend(flushed);
            return Ok(self.handed_over.drain(..).collect());
        }
        Ok(flushed)
    }
}

/// VA-API plus what it takes to hand a stream to openh264 part-way.
#[cfg(feature = "vaapi-decode")]
struct VaapiBackend {
    decoder: VaapiH264Decoder,
    config: DecoderConfig,
    parameter_sets: ParameterSets,
    probation: Probation,
}

#[cfg(feature = "vaapi-decode")]
impl VaapiBackend {
    fn open(config: &DecoderConfig) -> Result<Self> {
        let mut decoder = VaapiH264Decoder::probe_with(config)?;
        decoder.configure(config)?;
        Ok(Self {
            decoder,
            config: config.clone(),
            parameter_sets: ParameterSets::default(),
            probation: Probation::default(),
        })
    }
}

/// The latest SPS and PPS seen in the bitstream. A packet that VA-API
/// refuses may rely on parameter sets sent packets earlier, so they are
/// replayed in front of it when openh264 takes over.
#[cfg(feature = "vaapi-decode")]
#[derive(Default)]
struct ParameterSets {
    sps: Option<Bytes>,
    pps: Option<Bytes>,
}

#[cfg(feature = "vaapi-decode")]
impl ParameterSets {
    const START_CODE: [u8; 4] = [0, 0, 0, 1];

    fn remember(&mut self, packet: &[u8]) {
        for nal in split_annex_b(packet) {
            match nal.first().map(|header| header & 0x1f) {
                Some(7) => self.sps = Some(Bytes::copy_from_slice(nal)),
                Some(8) => self.pps = Some(Bytes::copy_from_slice(nal)),
                _ => {}
            }
        }
    }

    fn prepend(&self, packet: &Bytes) -> Bytes {
        let sets = [&self.sps, &self.pps];
        if sets.iter().all(|set| set.is_none()) {
            return packet.clone();
        }
        let mut out = Vec::with_capacity(packet.len() + 64);
        for set in sets.into_iter().flatten() {
            out.extend_from_slice(&Self::START_CODE);
            out.extend_from_slice(set);
        }
        out.extend_from_slice(packet);
        Bytes::from(out)
    }
}

#[cfg(all(test, feature = "vaapi-decode"))]
mod tests {
    use super::*;

    #[test]
    fn the_latest_parameter_sets_are_replayed_in_front_of_a_packet() {
        let mut sets = ParameterSets::default();
        sets.remember(&[0, 0, 0, 1, 0x67, 1, 0, 0, 1, 0x68, 2, 0, 0, 1, 0x65, 9]);
        sets.remember(&[0, 0, 1, 0x67, 3]);
        sets.remember(&[0, 0, 1, 0x41, 7]);
        let replay = sets.prepend(&Bytes::from_static(&[0, 0, 1, 0x41, 8]));
        assert_eq!(
            &replay[..],
            &[0, 0, 0, 1, 0x67, 3, 0, 0, 0, 1, 0x68, 2, 0, 0, 1, 0x41, 8]
        );
    }

    #[test]
    fn a_packet_is_untouched_when_no_parameter_sets_were_seen() {
        let packet = Bytes::from_static(&[0, 0, 1, 0x65, 1]);
        assert_eq!(ParameterSets::default().prepend(&packet), packet);
    }
}

/// The NAL units of an Annex B buffer, start codes stripped.
#[cfg(any(feature = "nvdec-decode", feature = "vaapi-decode"))]
pub(crate) fn split_annex_b(buf: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut start = None;
    let mut index = 0;
    while index + 3 <= buf.len() {
        if buf[index] == 0 && buf[index + 1] == 0 && buf[index + 2] == 1 {
            if let Some(begin) = start {
                let end = if index > begin && buf[index - 1] == 0 {
                    index - 1
                } else {
                    index
                };
                out.push(&buf[begin..end]);
            }
            index += 3;
            start = Some(index);
            continue;
        }
        index += 1;
    }
    if let Some(begin) = start {
        out.push(&buf[begin..]);
    }
    out
}

#[cfg(all(test, any(feature = "nvdec-decode", feature = "vaapi-decode")))]
mod annex_b_tests {
    use super::split_annex_b;

    #[test]
    fn annex_b_units_are_split_at_three_and_four_byte_start_codes() {
        let stream = [
            0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 0, 0, 0, 1, 0x65, 4,
        ];
        let units = split_annex_b(&stream);
        assert_eq!(units, [&[0x67, 1, 2][..], &[0x68, 3][..], &[0x65, 4][..]]);
    }

    #[test]
    fn a_stream_without_start_codes_has_no_units() {
        assert!(split_annex_b(&[1, 2, 3, 4]).is_empty());
    }
}
