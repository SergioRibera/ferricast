use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, PixelFormat, Result, VideoDecoder,
};
#[cfg(feature = "vaapi")]
use ferricast_decoder::h264::VaapiH264Decoder;
use ferricast_decoder::{Av1Decoder, H264Decoder, H265Decoder, Vp8Decoder, Vp9Decoder};

pub(crate) enum Decoder {
    H264(H264Decoder),
    H265(H265Decoder),
    Vp8(Box<Vp8Decoder>),
    Vp9(Box<Vp9Decoder>),
    Av1(Av1Decoder),
    #[cfg(feature = "vaapi")]
    VaapiH264(Box<VaapiH264Decoder>),
}

impl Decoder {
    pub(crate) fn open(codec: Codec, width: u32, height: u32, software_only: bool) -> Result<Self> {
        let config = DecoderConfig {
            codec,
            width,
            height,
            pixel_format: PixelFormat::Nv12,
            low_latency: false,
        };
        if !software_only && !software_requested(codec) {
            match hardware(codec, &config) {
                Some(Ok(decoder)) => {
                    tracing::info!(%codec, width, height, "decoding on the GPU through VA-API");
                    return Ok(decoder);
                }
                Some(Err(error)) => {
                    tracing::info!(%error, %codec, "no hardware decoder; using software");
                }
                None => {}
            }
        }
        tracing::info!(%codec, width, height, "decoding in software");
        match codec {
            Codec::H264 => configured(&config).map(Self::H264),
            Codec::H265 => configured(&config).map(Self::H265),
            Codec::Vp8 => configured(&config).map(|d| Self::Vp8(Box::new(d))),
            Codec::Vp9 => configured(&config).map(|d| Self::Vp9(Box::new(d))),
            Codec::Av1 => configured(&config).map(Self::Av1),
        }
    }

    pub(crate) fn is_hardware(&self) -> bool {
        match self {
            #[cfg(feature = "vaapi")]
            Decoder::VaapiH264(_) => true,
            _ => false,
        }
    }

    pub(crate) fn reorders(&self) -> bool {
        matches!(self, Decoder::H264(_) | Decoder::H265(_))
    }

    pub(crate) fn decode(&mut self, packet: EncodedFrame) -> Result<Option<CapturedFrame>> {
        match self {
            Decoder::H264(d) => d.decode(packet),
            Decoder::H265(d) => d.decode(packet),
            Decoder::Vp8(d) => d.decode(packet),
            Decoder::Vp9(d) => d.decode(packet),
            Decoder::Av1(d) => d.decode(packet),
            #[cfg(feature = "vaapi")]
            Decoder::VaapiH264(d) => d.decode(packet),
        }
    }

    pub(crate) fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        match self {
            Decoder::H264(d) => d.flush(),
            Decoder::H265(d) => d.flush(),
            Decoder::Vp8(d) => d.flush(),
            Decoder::Vp9(d) => d.flush(),
            Decoder::Av1(d) => d.flush(),
            #[cfg(feature = "vaapi")]
            Decoder::VaapiH264(d) => d.flush(),
        }
    }
}

fn hardware(codec: Codec, config: &DecoderConfig) -> Option<Result<Decoder>> {
    match codec {
        #[cfg(feature = "vaapi")]
        Codec::H264 => Some(
            VaapiH264Decoder::probe_with(config).and_then(|mut decoder| {
                decoder.configure(config)?;
                Ok(Decoder::VaapiH264(Box::new(decoder)))
            }),
        ),
        _ => {
            let _ = config;
            None
        }
    }
}

fn software_requested(codec: Codec) -> bool {
    let variable = match codec {
        Codec::H264 => "FERRICAST_H264_DECODE_BACKEND",
        Codec::H265 => "FERRICAST_H265_DECODE_BACKEND",
        Codec::Vp9 => "FERRICAST_VP9_DECODE_BACKEND",
        Codec::Vp8 | Codec::Av1 => return true,
    };
    matches!(
        std::env::var(variable).as_deref(),
        Ok("software" | "openh264")
    )
}

fn configured<D: VideoDecoder + Default>(config: &DecoderConfig) -> Result<D> {
    let mut decoder = D::default();
    decoder.configure(config)?;
    Ok(decoder)
}
