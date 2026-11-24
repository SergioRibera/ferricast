//! What the GPU's NVENC can do, per codec (`nvEncGetEncodeCaps`), and what
//! the negotiated API version adds or lacks.

use std::sync::Arc;

use super::Session;
use super::config::{Codec, InputFormat};
use super::raw::{ApiFeature, Cap, EncodeApi};
use crate::cuda::Context;
use crate::error::{Error, Result};
use crate::version::{AnyApi, Api, ApiVersion};

/// What one codec can do on this GPU.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one flag per NVENC capability, as the driver reports them"
)]
pub struct CodecCapabilities {
    /// The codec.
    pub codec: Codec,
    /// Largest picture width.
    pub max_width: u32,
    /// Largest picture height.
    pub max_height: u32,
    /// Smallest picture width.
    pub min_width: u32,
    /// Smallest picture height.
    pub min_height: u32,
    /// 10-bit encoding (HEVC Main10, AV1 10-bit).
    pub ten_bit: bool,
    /// Most B frames between references; 0 when B frames are unsupported.
    pub max_b_frames: u32,
    /// Rate-control lookahead.
    pub lookahead: bool,
    /// A custom VBV buffer size.
    pub custom_vbv: bool,
    /// Asynchronous encode (event completion), Windows only; this crate
    /// locks synchronously everywhere.
    pub async_encode: bool,
    /// Lossless encoding.
    pub lossless: bool,
    /// 4:4:4 encoding.
    pub yuv444: bool,
    /// Bitrate changes without a new session.
    pub dynamic_bitrate: bool,
    /// Rate-control mode changes without a new session.
    pub dynamic_rate_control_mode: bool,
    /// Temporal adaptive quantisation.
    pub temporal_aq: bool,
    /// Bit mask of `NV_ENC_PARAMS_RC_MODE` values the encoder accepts.
    pub rate_control_modes: u32,
    /// NVENC engines on the GPU.
    pub engines: u32,
    /// Input formats, among those this crate supports.
    pub input_formats: Vec<InputFormat>,
}

/// Whether a codec is available.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodecSupport {
    /// The GPU encodes it.
    Available(CodecCapabilities),
    /// The negotiated API version predates the codec.
    MissingFromApi {
        /// The version that adds it.
        needs: ApiVersion,
    },
    /// The GPU's NVENC has no encoder for it (AV1 before Ada, for one).
    MissingFromGpu,
}

/// What NVENC can do on one GPU through the negotiated API version.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capabilities {
    /// The API version queried (and that an encoder opened now would use).
    pub api: ApiVersion,
    /// H.264.
    pub h264: CodecSupport,
    /// HEVC.
    pub hevc: CodecSupport,
    /// AV1.
    pub av1: CodecSupport,
    /// The version-dependent features [`Capabilities::api`] has.
    pub features: Vec<ApiFeature>,
}

impl Capabilities {
    /// Queries the GPU of `context` through the negotiated API version
    /// ([`AnyApi::negotiate`], honouring `FERRICAST_NVENC_API`). Opens and
    /// closes an NVENC session.
    ///
    /// # Errors
    ///
    /// Loading and negotiation errors, [`Error::NvencSessionLimit`] when no
    /// session can be opened, [`Error::Nvenc`] for failed queries.
    pub fn query(context: &Arc<Context>) -> Result<Self> {
        Self::query_with(AnyApi::negotiate()?, context)
    }

    /// As [`Capabilities::query`] through `api`.
    ///
    /// # Errors
    ///
    /// As [`Capabilities::query`].
    pub fn query_with(api: AnyApi, context: &Arc<Context>) -> Result<Self> {
        match api {
            AnyApi::V11_1(api) => query(api, context),
            AnyApi::V12_0(api) => query(api, context),
            AnyApi::V12_1(api) => query(api, context),
            AnyApi::V12_2(api) => query(api, context),
            AnyApi::V13_0(api) => query(api, context),
            AnyApi::V13_1(api) => query(api, context),
        }
    }

    /// The support of `codec`.
    #[must_use]
    pub fn get(&self, codec: Codec) -> &CodecSupport {
        match codec {
            Codec::H264 => &self.h264,
            Codec::Hevc => &self.hevc,
            Codec::Av1 => &self.av1,
        }
    }

    /// The capabilities of `codec`.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] saying whether the API version or the GPU
    /// lacks it.
    pub fn require(&self, codec: Codec) -> Result<&CodecCapabilities> {
        match self.get(codec) {
            CodecSupport::Available(caps) => Ok(caps),
            CodecSupport::MissingFromApi { .. } => {
                Err(ApiFeature::Av1.unsupported("AV1 encoding", self.api))
            }
            CodecSupport::MissingFromGpu => Err(Error::Unsupported {
                what: codec_name(codec),
                reason: "this GPU's NVENC has no encoder for it".into(),
            }),
        }
    }
}

const fn codec_name(codec: Codec) -> &'static str {
    match codec {
        Codec::H264 => "H.264 encoding",
        Codec::Hevc => "HEVC encoding",
        Codec::Av1 => "AV1 encoding",
    }
}

fn query<A: EncodeApi>(api: Api<A>, context: &Arc<Context>) -> Result<Capabilities> {
    let session = super::open_session(api, context)?;
    let _current = context.guard()?;
    let codecs = A::codecs(&session)?;
    let support = |codec: Codec| -> Result<CodecSupport> {
        if codec == Codec::Av1 && !A::AV1 {
            return Ok(CodecSupport::MissingFromApi {
                needs: ApiFeature::Av1.added_in(),
            });
        }
        if !codecs.contains(&codec) {
            return Ok(CodecSupport::MissingFromGpu);
        }
        codec_caps(&session, codec).map(CodecSupport::Available)
    };
    Ok(Capabilities {
        api: A::VERSION,
        h264: support(Codec::H264)?,
        hevc: support(Codec::Hevc)?,
        av1: support(Codec::Av1)?,
        features: ApiFeature::ALL
            .into_iter()
            .filter(|feature| A::has(*feature))
            .collect(),
    })
}

/// Queries every capability of `codec` on an open session.
pub(crate) fn codec_caps<A: EncodeApi>(
    session: &Session<A>,
    codec: Codec,
) -> Result<CodecCapabilities> {
    let cap = |cap: Cap| A::caps(session, codec, cap);
    let unsigned = |value: Cap| cap(value).map(|value| u32::try_from(value).unwrap_or(0));
    let flag = |value: Cap| cap(value).map(|value| value != 0);
    Ok(CodecCapabilities {
        codec,
        max_width: unsigned(Cap::WIDTH_MAX)?,
        max_height: unsigned(Cap::HEIGHT_MAX)?,
        min_width: unsigned(Cap::WIDTH_MIN)?,
        min_height: unsigned(Cap::HEIGHT_MIN)?,
        ten_bit: flag(Cap::SUPPORT_10BIT_ENCODE)?,
        max_b_frames: unsigned(Cap::NUM_MAX_BFRAMES)?,
        lookahead: flag(Cap::SUPPORT_LOOKAHEAD)?,
        custom_vbv: flag(Cap::SUPPORT_CUSTOM_VBV_BUF_SIZE)?,
        async_encode: flag(Cap::ASYNC_ENCODE_SUPPORT)?,
        lossless: flag(Cap::SUPPORT_LOSSLESS_ENCODE)?,
        yuv444: flag(Cap::SUPPORT_YUV444_ENCODE)?,
        dynamic_bitrate: flag(Cap::SUPPORT_DYN_BITRATE_CHANGE)?,
        dynamic_rate_control_mode: flag(Cap::SUPPORT_DYN_RCMODE_CHANGE)?,
        temporal_aq: flag(Cap::SUPPORT_TEMPORAL_AQ)?,
        rate_control_modes: unsigned(Cap::SUPPORTED_RATECONTROL_MODES)?,
        engines: unsigned(Cap::NUM_ENCODER_ENGINES)?,
        input_formats: A::input_formats(session, codec)?,
    })
}
