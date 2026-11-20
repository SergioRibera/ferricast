use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, PixelFormat, Result, VideoDecoder,
};
use ferricast_decoder::h264::OpenH264Decoder;
#[cfg(feature = "vaapi")]
use ferricast_decoder::h264::VaapiH264Decoder;
use ferricast_decoder::{Av1Decoder, H265Decoder, Vp8Decoder, Vp9Decoder};

/// Same switch the decoder crate's H.264 facade reads; VA-API decode is
/// not yet verified across drivers, so it only runs when asked for.
#[cfg(feature = "vaapi")]
const H264_BACKEND_VAR: &str = "FERRICAST_H264_DECODE_BACKEND";

pub(crate) enum Decoder {
    /// openh264 directly rather than the H.264 facade: the facade would
    /// pick VA-API again when the environment asks for it, and this is
    /// the variant the player falls back to after VA-API failed.
    H264(OpenH264Decoder),
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
        if !software_only {
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

#[cfg(feature = "vaapi")]
fn hardware(codec: Codec, config: &DecoderConfig) -> Option<Result<Decoder>> {
    if codec != Codec::H264 || std::env::var(H264_BACKEND_VAR).as_deref() != Ok("vaapi") {
        return None;
    }
    Some(
        VaapiH264Decoder::probe_with(config).and_then(|mut decoder| {
            decoder.configure(config)?;
            Ok(Decoder::VaapiH264(Box::new(decoder)))
        }),
    )
}

#[cfg(not(feature = "vaapi"))]
fn hardware(_codec: Codec, _config: &DecoderConfig) -> Option<Result<Decoder>> {
    None
}

fn configured<D: VideoDecoder + Default>(config: &DecoderConfig) -> Result<D> {
    let mut decoder = D::default();
    decoder.configure(config)?;
    Ok(decoder)
}
