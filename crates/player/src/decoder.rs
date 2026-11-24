use ferricast_core::{
    Acceleration, AccelerationPolicy, AutoSelect, BackendId, BackendSelection, CapturedFrame,
    Codec, CodecResult, DecoderConfig, EncodedFrame, PixelFormat, RuntimeFallback, VideoDecoder,
};
use ferricast_decoder::{Av1Decoder, H264Decoder, H265Decoder, Vp8Decoder, Vp9Decoder};

pub(crate) enum Decoder {
    H264(H264Decoder),
    H265(H265Decoder),
    Vp8(Box<Vp8Decoder>),
    Vp9(Box<Vp9Decoder>),
    Av1(Av1Decoder),
}

impl Decoder {
    /// Opens a decoder for `codec`. Hardware decode is not yet verified
    /// across drivers for local playback, so it runs only when the
    /// codec's `FERRICAST_<CODEC>_DECODE_BACKEND` asks for it; with
    /// `software_only` (the player's fallback after a hardware decoder
    /// failed) the environment is ignored too.
    pub(crate) fn open(
        codec: Codec,
        width: u32,
        height: u32,
        software_only: bool,
    ) -> CodecResult<Self> {
        let backend = if software_only {
            BackendSelection::Ordered {
                backends: vec![software_backend(codec)],
                runtime_fallback: RuntimeFallback::Disabled,
            }
        } else {
            AutoSelect::default()
                .with_acceleration(AccelerationPolicy::SoftwareOnly)
                .into()
        };
        let config = DecoderConfig {
            hdr: None,
            memory: ferricast_core::FrameMemory::Cpu,
            codec,
            width,
            height,
            pixel_format: PixelFormat::Nv12,
            low_latency: false,
            colour: None,
            backend,
        };
        let decoder = match codec {
            Codec::H264 => configured(&config).map(Self::H264),
            Codec::H265 => configured(&config).map(Self::H265),
            Codec::Vp8 => configured(&config).map(|d| Self::Vp8(Box::new(d))),
            Codec::Vp9 => configured(&config).map(|d| Self::Vp9(Box::new(d))),
            Codec::Av1 => configured(&config).map(Self::Av1),
        }?;
        if let Some((backend, acceleration)) = decoder.backend() {
            tracing::info!(%codec, width, height, %backend, %acceleration, "decoder opened");
        }
        Ok(decoder)
    }

    fn backend(&self) -> Option<(BackendId, Acceleration)> {
        match self {
            Decoder::H264(d) => d.backend(),
            Decoder::H265(d) => d.backend(),
            Decoder::Vp8(d) => d.backend(),
            Decoder::Vp9(d) => d.backend(),
            Decoder::Av1(d) => d.backend(),
        }
    }

    pub(crate) fn is_hardware(&self) -> bool {
        self.backend()
            .is_some_and(|(_, acceleration)| acceleration == Acceleration::Hardware)
    }

    pub(crate) fn decode(&mut self, packet: EncodedFrame) -> CodecResult<Option<CapturedFrame>> {
        match self {
            Decoder::H264(d) => d.decode(packet),
            Decoder::H265(d) => d.decode(packet),
            Decoder::Vp8(d) => d.decode(packet),
            Decoder::Vp9(d) => d.decode(packet),
            Decoder::Av1(d) => d.decode(packet),
        }
    }

    pub(crate) fn flush(&mut self) -> CodecResult<Vec<CapturedFrame>> {
        match self {
            Decoder::H264(d) => d.flush(),
            Decoder::H265(d) => d.flush(),
            Decoder::Vp8(d) => d.flush(),
            Decoder::Vp9(d) => d.flush(),
            Decoder::Av1(d) => d.flush(),
        }
    }
}

/// The software decoder the player falls back to for `codec`.
fn software_backend(codec: Codec) -> BackendId {
    match codec {
        Codec::H264 => BackendId::OpenH264,
        Codec::H265 => BackendId::RustHevc,
        Codec::Vp8 => BackendId::RustVp8,
        Codec::Vp9 => BackendId::RustVp9,
        Codec::Av1 => BackendId::Rav1d,
    }
}

fn configured<D: VideoDecoder + Default>(config: &DecoderConfig) -> CodecResult<D> {
    let mut decoder = D::default();
    decoder.configure(config)?;
    Ok(decoder)
}
