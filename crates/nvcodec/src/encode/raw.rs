//! The NVENC calls whose argument structs differ between API versions,
//! written once and instantiated for every generated module.
//!
//! [`EncodeApi`] is implemented for each marker type of
//! [`crate::version::api`] by [`encode_api!`]: the macro body is the single
//! source of every struct the encoder builds, and each instantiation is
//! compiled against that version's bindings, so a field that a version lacks
//! is a compile error, not a silent zero. Where versions differ the macro
//! takes a `yes` / `no` token per difference and [`pick!`] keeps the matching
//! code:
//!
//! | token              | added in | what changes                                              |
//! |--------------------|----------|-----------------------------------------------------------|
//! | `av1`              | 12.0     | AV1 GUIDs, `av1Config`, `av1PicParams`                    |
//! | `typed_vui`        | 12.0     | VUI colour fields become enums (plain `u32` in 11.1)      |
//! | `bit_depth_enum`   | 12.2     | `inputBitDepth` / `outputBitDepth` replace `pixelBitDepthMinus8` |
//! | `native_hdr`       | 13.0     | `outputMasteringDisplay` / `outputMaxCll`, `pMasteringDisplay` / `pMaxCll` (HEVC, AV1) |
//! | `chroma_offset_in` | 13.0     | `NV_ENC_REGISTER_RESOURCE::chromaOffsetIn`                |
//! | `uhq`              | 13.0     | `NV_ENC_TUNING_INFO_ULTRA_HIGH_QUALITY`                   |
//!
//! Struct version numbers (`*_VER`) come from each module's own header
//! through [`NvencApi`]; the layouts of the other structs used here
//! (`NV_ENC_LOCK_BITSTREAM`, `NV_ENC_PIC_PARAMS`, `NV_ENC_REGISTER_RESOURCE`,
//! `NV_ENC_INITIALIZE_PARAMS`, `NV_ENC_RC_PARAMS`, ...) also move between
//! versions, which the per-module types take care of.

#![expect(
    clippy::used_underscore_items,
    reason = "the generated bindings give every enum type a leading underscore (`_NV_ENC_BUFFER_FORMAT`); its newtype constructor has no other name"
)]

use std::ffi::c_void;
use std::ptr::NonNull;

use super::Session;
use super::config::{
    Codec, ColorDescription, Config, Hdr10, InputFormat, Preset, Profile, RateControl, Tuning,
};
use super::sei::{self, HdrSei};
use crate::error::{Error, NvencStatus, Result};
use crate::sys::cuda::{CUarray, CUdeviceptr, CUstream};
use crate::version::{ApiVersion, NvencApi, api};

/// A feature that only some NVENC API versions have.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ApiFeature {
    /// AV1 encoding.
    Av1,
    /// HDR10 metadata written by NVENC itself (HEVC SEI, AV1 metadata OBUs);
    /// older versions get HEVC / H.264 SEI payloads from this crate, AV1
    /// none.
    NativeHdrMetadata,
    /// A device input whose chroma plane does not directly follow the luma
    /// rows (`chromaOffsetIn`), as NVDEC surfaces with padded rows have.
    ChromaPlaneOffset,
    /// [`Tuning::UltraHighQuality`].
    UltraHighQualityTuning,
}

impl ApiFeature {
    /// Every feature.
    pub const ALL: [Self; 4] = [
        Self::Av1,
        Self::NativeHdrMetadata,
        Self::ChromaPlaneOffset,
        Self::UltraHighQualityTuning,
    ];

    /// The first API version that has it.
    #[must_use]
    pub const fn added_in(self) -> ApiVersion {
        match self {
            Self::Av1 => ApiVersion::V12_0,
            Self::NativeHdrMetadata | Self::ChromaPlaneOffset | Self::UltraHighQualityTuning => {
                ApiVersion::V13_0
            }
        }
    }

    /// Whether `version` has it.
    #[must_use]
    pub fn available_in(self, version: ApiVersion) -> bool {
        version >= self.added_in()
    }

    /// What the feature is, for messages.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Av1 => "AV1 encoding",
            Self::NativeHdrMetadata => "HDR10 metadata written by NVENC",
            Self::ChromaPlaneOffset => "a chroma plane offset on device input",
            Self::UltraHighQualityTuning => "ultra-high-quality tuning",
        }
    }

    /// `Error::Unsupported` for using the feature on `version`.
    pub(crate) fn unsupported(self, what: &'static str, version: ApiVersion) -> Error {
        let added = self.added_in();
        Error::Unsupported {
            what,
            reason: format!(
                "{} needs NVENC API {added} (driver {} or newer); this encoder uses API {version}",
                self.name(),
                added.minimum_driver().for_this_os(),
            ),
        }
    }
}

/// An `NV_ENC_CAPS` query; the values are the same in every version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cap(pub(crate) u32);

impl Cap {
    pub(crate) const NUM_MAX_BFRAMES: Self = Self(0);
    pub(crate) const SUPPORTED_RATECONTROL_MODES: Self = Self(1);
    pub(crate) const WIDTH_MAX: Self = Self(16);
    pub(crate) const HEIGHT_MAX: Self = Self(17);
    pub(crate) const SUPPORT_DYN_BITRATE_CHANGE: Self = Self(20);
    pub(crate) const SUPPORT_DYN_RCMODE_CHANGE: Self = Self(22);
    pub(crate) const SUPPORT_CUSTOM_VBV_BUF_SIZE: Self = Self(26);
    pub(crate) const ASYNC_ENCODE_SUPPORT: Self = Self(30);
    pub(crate) const SUPPORT_YUV444_ENCODE: Self = Self(33);
    pub(crate) const SUPPORT_LOSSLESS_ENCODE: Self = Self(34);
    pub(crate) const SUPPORT_LOOKAHEAD: Self = Self(37);
    pub(crate) const SUPPORT_TEMPORAL_AQ: Self = Self(38);
    pub(crate) const SUPPORT_10BIT_ENCODE: Self = Self(39);
    pub(crate) const WIDTH_MIN: Self = Self(45);
    pub(crate) const HEIGHT_MIN: Self = Self(46);
    pub(crate) const NUM_ENCODER_ENGINES: Self = Self(49);
}

/// What an input resource is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Resource {
    /// A pitched device allocation (`NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR`).
    DevicePtr(CUdeviceptr),
    /// A CUDA array (`NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY`).
    Array(CUarray),
}

/// An input resource to register.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Registration {
    pub(crate) resource: Resource,
    pub(crate) format: InputFormat,
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// Bytes per row.
    pub(crate) pitch: u32,
    /// Bytes from the luma plane to the chroma plane when it does not
    /// directly follow the `height` luma rows ([`ApiFeature::ChromaPlaneOffset`]).
    pub(crate) chroma_offset: Option<u32>,
}

/// A mapped input (`nvEncMapInputResource`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Mapped {
    pub(crate) input: NonNull<c_void>,
    /// The `NV_ENC_BUFFER_FORMAT` NVENC reports for it.
    pub(crate) format: u32,
}

/// One picture to encode.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Picture<'a> {
    /// The session's codec.
    pub(crate) codec: Codec,
    pub(crate) input: Mapped,
    pub(crate) output: NonNull<c_void>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pitch: u32,
    pub(crate) timestamp: u64,
    pub(crate) force_idr: bool,
    /// Write the parameter sets with this picture.
    pub(crate) parameter_sets: bool,
    /// The session's HDR10 metadata, on every picture.
    pub(crate) hdr10: Option<&'a Hdr10>,
    /// The picture is expected to be an IDR: where this crate writes the
    /// HDR10 SEI itself, it goes with these pictures.
    pub(crate) expect_idr: bool,
}

/// What `nvEncEncodePicture` said.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Submitted {
    /// Every picture submitted so far can be locked.
    OutputReady,
    /// NVENC holds the picture (B frames, lookahead) until more come.
    NeedMoreInput,
}

/// What a locked bitstream says about its picture.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Locked {
    pub(crate) timestamp: u64,
    pub(crate) picture_type: u32,
}

/// The NVENC calls whose argument structs depend on the API version. See the
/// module docs.
pub(crate) trait EncodeApi: NvencApi {
    /// The version has [`ApiFeature::Av1`].
    const AV1: bool;
    /// The version has [`ApiFeature::NativeHdrMetadata`].
    const NATIVE_HDR: bool;
    /// The version has [`ApiFeature::ChromaPlaneOffset`].
    const CHROMA_OFFSET: bool;
    /// The version has [`ApiFeature::UltraHighQualityTuning`].
    const ULTRA_HIGH_QUALITY: bool;

    /// Whether this version knows `feature`, from the constants above.
    fn has(feature: ApiFeature) -> bool {
        match feature {
            ApiFeature::Av1 => Self::AV1,
            ApiFeature::NativeHdrMetadata => Self::NATIVE_HDR,
            ApiFeature::ChromaPlaneOffset => Self::CHROMA_OFFSET,
            ApiFeature::UltraHighQualityTuning => Self::ULTRA_HIGH_QUALITY,
        }
    }

    /// The codecs the GPU encodes (`nvEncGetEncodeGUIDs`), among those this
    /// version knows.
    fn codecs(session: &Session<Self>) -> Result<Vec<Codec>>;

    /// One capability of `codec` (`nvEncGetEncodeCaps`).
    fn caps(session: &Session<Self>, codec: Codec, cap: Cap) -> Result<i32>;

    /// The input formats `codec` accepts (`nvEncGetInputFormats`).
    fn input_formats(session: &Session<Self>, codec: Codec) -> Result<Vec<InputFormat>>;

    /// `nvEncInitializeEncoder` with the preset's configuration adjusted to
    /// `config` (validated, and checked against this version).
    fn initialize(session: &Session<Self>, config: &Config) -> Result<()>;

    /// `nvEncReconfigureEncoder` to `config` (same codec, size and format).
    fn reconfigure(session: &Session<Self>, config: &Config, force_idr: bool) -> Result<()>;

    /// The parameter sets (`nvEncGetSequenceParams`) into `out`.
    fn sequence_params(session: &Session<Self>, out: &mut Vec<u8>) -> Result<()>;

    /// `nvEncCreateBitstreamBuffer`.
    fn create_bitstream(session: &Session<Self>) -> Result<NonNull<c_void>>;

    /// `nvEncDestroyBitstreamBuffer`.
    ///
    /// # Safety
    ///
    /// `buffer` comes from [`EncodeApi::create_bitstream`] on `session`, is
    /// not locked nor pending, and is not used again.
    unsafe fn destroy_bitstream(session: &Session<Self>, buffer: NonNull<c_void>) -> Result<()>;

    /// `nvEncSetIOCudaStreams` with the same stream for input and output.
    ///
    /// # Safety
    ///
    /// `stream` points to a live stream of the session's context, and both
    /// the pointer and the stream stay valid until the session is destroyed
    /// (the driver may keep the pointer).
    unsafe fn set_io_stream(session: &Session<Self>, stream: NonNull<CUstream>) -> Result<()>;

    /// `nvEncRegisterResource`.
    ///
    /// # Safety
    ///
    /// The resource is a live allocation of the session's context holding
    /// at least what `registration` describes, and stays so until
    /// [`EncodeApi::unregister`].
    unsafe fn register(
        session: &Session<Self>,
        registration: &Registration,
    ) -> Result<NonNull<c_void>>;

    /// `nvEncUnregisterResource`.
    ///
    /// # Safety
    ///
    /// `registered` comes from [`EncodeApi::register`] on `session`, is not
    /// mapped, and is not used again.
    unsafe fn unregister(session: &Session<Self>, registered: NonNull<c_void>) -> Result<()>;

    /// `nvEncMapInputResource`.
    ///
    /// # Safety
    ///
    /// `registered` comes from [`EncodeApi::register`] on `session` and is
    /// not mapped.
    unsafe fn map(session: &Session<Self>, registered: NonNull<c_void>) -> Result<Mapped>;

    /// `nvEncUnmapInputResource`.
    ///
    /// # Safety
    ///
    /// `mapped` comes from [`EncodeApi::map`] on `session`, its picture's
    /// bitstream has been locked (or the picture was never submitted), and
    /// it is not used again.
    unsafe fn unmap(session: &Session<Self>, mapped: Mapped) -> Result<()>;

    /// `nvEncEncodePicture`.
    ///
    /// # Safety
    ///
    /// `picture.input` is mapped on `session` and `picture.output` is one of
    /// its bitstream buffers, neither in use by a pending picture.
    unsafe fn encode(session: &Session<Self>, picture: &Picture<'_>) -> Result<Submitted>;

    /// `nvEncEncodePicture` with `NV_ENC_PIC_FLAG_EOS`: every pending
    /// picture becomes lockable.
    fn end_of_stream(session: &Session<Self>) -> Result<()>;

    /// `nvEncLockBitstream` (blocking), copies the bitstream into `out`,
    /// `nvEncUnlockBitstream`.
    ///
    /// # Safety
    ///
    /// `output` is a bitstream buffer of `session` whose picture was
    /// submitted and reported ready, and not locked yet.
    unsafe fn lock(
        session: &Session<Self>,
        output: NonNull<c_void>,
        out: &mut Vec<u8>,
    ) -> Result<Locked>;
}

/// Keeps the `yes` or the `no` branch: how [`encode_api!`] writes code that
/// exists only in some versions.
macro_rules! pick {
    (yes, { $($yes:tt)* }, { $($no:tt)* }) => { $($yes)* };
    (no, { $($yes:tt)* }, { $($no:tt)* }) => { $($no)* };
}

/// The entry point `$name` of the session's function list.
macro_rules! entry {
    ($session:expr, $name:ident) => {
        $session
            .api()
            .functions()
            .$name
            .ok_or(Error::NvencFunctionMissing {
                api: Self::VERSION,
                function: stringify!($name),
            })?
    };
}

/// `Ok(())` for `NV_ENC_SUCCESS`, otherwise the error for `call` with the
/// session's explanation.
pub(crate) fn check<A: NvencApi>(
    session: &Session<A>,
    status: u32,
    call: &'static str,
) -> Result<()> {
    if status == NvencStatus::SUCCESS.0 {
        Ok(())
    } else {
        Err(Error::Nvenc {
            call,
            status: NvencStatus(status),
            detail: session.last_error(),
        })
    }
}

/// `NV_ENC_ERR_NEED_MORE_INPUT`, identical in every version.
const NEED_MORE_INPUT: u32 = 17;

/// A `NonNull` out of a pointer the driver returned, or an error naming the
/// call.
fn returned(
    pointer: *mut c_void,
    call: &'static str,
    what: &'static str,
) -> Result<NonNull<c_void>> {
    NonNull::new(pointer).ok_or(Error::UnexpectedValue {
        call,
        what,
        value: "null".into(),
    })
}

fn to_u32(value: usize, what: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::SizeOverflow { what })
}

macro_rules! encode_api {
    ($(
        $marker:ident => $module:ident {
            av1: $av1:tt,
            typed_vui: $typed_vui:tt,
            bit_depth_enum: $bit_depth_enum:tt,
            native_hdr: $native_hdr:tt,
            chroma_offset_in: $chroma_offset_in:tt,
            uhq: $uhq:tt $(,)?
        };
    )*) => {$(
        impl api::$marker {
            fn same_guid(a: &nvenc::$module::GUID, b: &nvenc::$module::GUID) -> bool {
                a.Data1 == b.Data1 && a.Data2 == b.Data2 && a.Data3 == b.Data3 && a.Data4 == b.Data4
            }

            fn codec_guid(codec: Codec) -> Option<nvenc::$module::GUID> {
                match codec {
                    Codec::H264 => Some(nvenc::$module::NV_ENC_CODEC_H264_GUID),
                    Codec::Hevc => Some(nvenc::$module::NV_ENC_CODEC_HEVC_GUID),
                    Codec::Av1 => pick!($av1, { Some(nvenc::$module::NV_ENC_CODEC_AV1_GUID) }, { None }),
                }
            }

            fn require_codec_guid(codec: Codec) -> Result<nvenc::$module::GUID> {
                Self::codec_guid(codec)
                    .ok_or_else(|| ApiFeature::Av1.unsupported("AV1 encoding", Self::VERSION))
            }

            fn profile_guid(profile: Profile) -> Result<nvenc::$module::GUID> {
                Ok(match profile {
                    Profile::H264Baseline => nvenc::$module::NV_ENC_H264_PROFILE_BASELINE_GUID,
                    Profile::H264Main => nvenc::$module::NV_ENC_H264_PROFILE_MAIN_GUID,
                    Profile::H264High => nvenc::$module::NV_ENC_H264_PROFILE_HIGH_GUID,
                    Profile::HevcMain => nvenc::$module::NV_ENC_HEVC_PROFILE_MAIN_GUID,
                    Profile::HevcMain10 => nvenc::$module::NV_ENC_HEVC_PROFILE_MAIN10_GUID,
                    Profile::Av1Main => pick!($av1, { nvenc::$module::NV_ENC_AV1_PROFILE_MAIN_GUID }, {
                        return Err(ApiFeature::Av1.unsupported("AV1 encoding", Self::VERSION))
                    }),
                })
            }

            fn preset_guid(preset: Preset) -> nvenc::$module::GUID {
                match preset {
                    Preset::P1 => nvenc::$module::NV_ENC_PRESET_P1_GUID,
                    Preset::P2 => nvenc::$module::NV_ENC_PRESET_P2_GUID,
                    Preset::P3 => nvenc::$module::NV_ENC_PRESET_P3_GUID,
                    Preset::P4 => nvenc::$module::NV_ENC_PRESET_P4_GUID,
                    Preset::P5 => nvenc::$module::NV_ENC_PRESET_P5_GUID,
                    Preset::P6 => nvenc::$module::NV_ENC_PRESET_P6_GUID,
                    Preset::P7 => nvenc::$module::NV_ENC_PRESET_P7_GUID,
                }
            }

            fn tuning(tuning: Tuning) -> Result<nvenc::$module::NV_ENC_TUNING_INFO> {
                type Info = nvenc::$module::NV_ENC_TUNING_INFO;
                Ok(match tuning {
                    Tuning::HighQuality => Info::NV_ENC_TUNING_INFO_HIGH_QUALITY,
                    Tuning::LowLatency => Info::NV_ENC_TUNING_INFO_LOW_LATENCY,
                    Tuning::UltraLowLatency => Info::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
                    Tuning::Lossless => Info::NV_ENC_TUNING_INFO_LOSSLESS,
                    Tuning::UltraHighQuality => pick!($uhq, { Info::NV_ENC_TUNING_INFO_ULTRA_HIGH_QUALITY }, {
                        return Err(ApiFeature::UltraHighQualityTuning
                            .unsupported("ultra-high-quality tuning", Self::VERSION))
                    }),
                })
            }

            fn buffer_format(format: InputFormat) -> nvenc::$module::_NV_ENC_BUFFER_FORMAT {
                type Format = nvenc::$module::_NV_ENC_BUFFER_FORMAT;
                match format {
                    InputFormat::Nv12 => Format::NV_ENC_BUFFER_FORMAT_NV12,
                    InputFormat::P010 => Format::NV_ENC_BUFFER_FORMAT_YUV420_10BIT,
                    InputFormat::Argb => Format::NV_ENC_BUFFER_FORMAT_ARGB,
                    InputFormat::Abgr => Format::NV_ENC_BUFFER_FORMAT_ABGR,
                }
            }

            fn input_format(format: nvenc::$module::_NV_ENC_BUFFER_FORMAT) -> Option<InputFormat> {
                InputFormat::ALL
                    .into_iter()
                    .find(|candidate| Self::buffer_format(*candidate) == format)
            }

            /// The H.264 / HEVC VUI colour description.
            fn set_vui(
                vui: &mut nvenc::$module::NV_ENC_CONFIG_H264_VUI_PARAMETERS,
                color: &ColorDescription,
            ) {
                /// H.273 `video_format` 5: unspecified.
                const UNSPECIFIED_VIDEO_FORMAT: u32 = 5;
                vui.videoSignalTypePresentFlag = 1;
                vui.videoFullRangeFlag = u32::from(color.full_range);
                vui.colourDescriptionPresentFlag = 1;
                pick!($typed_vui, {
                    vui.videoFormat = nvenc::$module::_NV_ENC_VUI_VIDEO_FORMAT(UNSPECIFIED_VIDEO_FORMAT);
                    vui.colourPrimaries =
                        nvenc::$module::_NV_ENC_VUI_COLOR_PRIMARIES(u32::from(color.primaries.0));
                    vui.transferCharacteristics =
                        nvenc::$module::_NV_ENC_VUI_TRANSFER_CHARACTERISTIC(u32::from(color.transfer.0));
                    vui.colourMatrix =
                        nvenc::$module::_NV_ENC_VUI_MATRIX_COEFFS(u32::from(color.matrix.0));
                }, {
                    vui.videoFormat = UNSPECIFIED_VIDEO_FORMAT;
                    vui.colourPrimaries = u32::from(color.primaries.0);
                    vui.transferCharacteristics = u32::from(color.transfer.0);
                    vui.colourMatrix = u32::from(color.matrix.0);
                });
            }

            /// The preset's configuration for `config`, adjusted to it.
            fn build_config(
                session: &Session<Self>,
                config: &Config,
            ) -> Result<nvenc::$module::NV_ENC_CONFIG> {
                let codec = Self::require_codec_guid(config.codec)?;
                let get_preset = entry!(session, nvEncGetEncodePresetConfigEx);
                let mut preset = nvenc::$module::NV_ENC_PRESET_CONFIG {
                    version: Self::PRESET_CONFIG_VER,
                    presetCfg: nvenc::$module::NV_ENC_CONFIG {
                        version: Self::CONFIG_VER,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                // SAFETY: live session; `preset` is a valid struct of this
                // version with its versions set, written by the driver.
                let status = unsafe {
                    get_preset(
                        session.as_ptr(),
                        codec,
                        Self::preset_guid(config.preset),
                        Self::tuning(config.tuning)?,
                        &raw mut preset,
                    )
                };
                check(session, status.0, "nvEncGetEncodePresetConfigEx")?;

                let mut nv = preset.presetCfg;
                Self::apply_gop_and_rate_control(&mut nv, config)?;
                Self::apply_codec_config(&mut nv, config)?;
                Ok(nv)
            }

            /// Profile, GOP and rate control on the preset's configuration.
            fn apply_gop_and_rate_control(
                nv: &mut nvenc::$module::NV_ENC_CONFIG,
                config: &Config,
            ) -> Result<()> {
                type Mode = nvenc::$module::_NV_ENC_PARAMS_RC_MODE;
                nv.version = Self::CONFIG_VER;
                nv.profileGUID = Self::profile_guid(config.effective_profile())?;
                nv.gopLength = config
                    .gop
                    .length
                    .map_or(nvenc::$module::NVENC_INFINITE_GOPLENGTH, |length| length.get());
                nv.frameIntervalP = i32::from(config.b_frames) + 1;

                let rc = &mut nv.rcParams;
                match config.rate_control {
                    RateControl::Cbr { bitrate, .. } => {
                        rc.rateControlMode = Mode::NV_ENC_PARAMS_RC_CBR;
                        rc.averageBitRate = bitrate;
                        rc.maxBitRate = bitrate;
                    }
                    RateControl::Vbr { average, max, .. } => {
                        rc.rateControlMode = Mode::NV_ENC_PARAMS_RC_VBR;
                        rc.averageBitRate = average;
                        rc.maxBitRate = max;
                    }
                    RateControl::ConstQp(qp) => {
                        rc.rateControlMode = Mode::NV_ENC_PARAMS_RC_CONSTQP;
                        rc.constQP = nvenc::$module::NV_ENC_QP {
                            qpInterP: u32::from(qp.inter_p),
                            qpInterB: u32::from(qp.inter_b),
                            qpIntra: u32::from(qp.intra),
                        };
                    }
                }
                if let Some(vbv) = config.rate_control.vbv() {
                    rc.vbvBufferSize = vbv.buffer_bits;
                    rc.vbvInitialDelay = vbv.initial_delay_bits;
                }
                rc.set_enableLookahead(u32::from(config.lookahead.is_some()));
                rc.lookaheadDepth = config.lookahead.unwrap_or(0);
                Ok(())
            }

            /// The codec part of the preset's configuration: IDR period,
            /// parameter sets, chroma format, bit depth, colour, HDR.
            fn apply_codec_config(
                nv: &mut nvenc::$module::NV_ENC_CONFIG,
                config: &Config,
            ) -> Result<()> {
                let idr_period = config
                    .gop
                    .effective_idr_period()
                    .map_or(nvenc::$module::NVENC_INFINITE_GOPLENGTH, |period| period.get());
                let repeat = u32::from(config.repeat_parameter_sets);
                let bit_depth = config.input_format.bit_depth();
                match config.codec {
                    Codec::H264 => {
                        // SAFETY: the preset filled the union for the codec
                        // requested; it is plain data valid for any bits.
                        let mut h264 = unsafe { nv.encodeCodecConfig.h264Config };
                        h264.idrPeriod = idr_period;
                        h264.set_repeatSPSPPS(repeat);
                        h264.chromaFormatIDC = 1;
                        if let Some(color) = &config.color {
                            Self::set_vui(&mut h264.h264VUIParameters, color);
                        }
                        pick!($bit_depth_enum, {
                            h264.inputBitDepth = nvenc::$module::_NV_ENC_BIT_DEPTH(bit_depth);
                            h264.outputBitDepth = nvenc::$module::_NV_ENC_BIT_DEPTH(bit_depth);
                        }, {});
                        nv.encodeCodecConfig.h264Config = h264;
                    }
                    Codec::Hevc => {
                        // SAFETY: as for H.264.
                        let mut hevc = unsafe { nv.encodeCodecConfig.hevcConfig };
                        hevc.idrPeriod = idr_period;
                        hevc.set_repeatSPSPPS(repeat);
                        hevc.set_chromaFormatIDC(1);
                        if let Some(color) = &config.color {
                            Self::set_vui(&mut hevc.hevcVUIParameters, color);
                        }
                        pick!($bit_depth_enum, {
                            hevc.inputBitDepth = nvenc::$module::_NV_ENC_BIT_DEPTH(bit_depth);
                            hevc.outputBitDepth = nvenc::$module::_NV_ENC_BIT_DEPTH(bit_depth);
                        }, {
                            hevc.set_pixelBitDepthMinus8(bit_depth - 8);
                        });
                        pick!($native_hdr, {
                            let hdr10 = config.hdr10.unwrap_or_default();
                            hevc.set_outputMasteringDisplay(u32::from(hdr10.mastering_display.is_some()));
                            hevc.set_outputMaxCll(u32::from(hdr10.content_light.is_some()));
                        }, {});
                        nv.encodeCodecConfig.hevcConfig = hevc;
                    }
                    Codec::Av1 => {
                        pick!($av1, {
                            // SAFETY: as for H.264.
                            let mut av1 = unsafe { nv.encodeCodecConfig.av1Config };
                            av1.idrPeriod = idr_period;
                            av1.set_repeatSeqHdr(repeat);
                            av1.set_chromaFormatIDC(1);
                            if let Some(color) = &config.color {
                                av1.colorPrimaries =
                                    nvenc::$module::_NV_ENC_VUI_COLOR_PRIMARIES(u32::from(color.primaries.0));
                                av1.transferCharacteristics =
                                    nvenc::$module::_NV_ENC_VUI_TRANSFER_CHARACTERISTIC(u32::from(color.transfer.0));
                                av1.matrixCoefficients =
                                    nvenc::$module::_NV_ENC_VUI_MATRIX_COEFFS(u32::from(color.matrix.0));
                                av1.colorRange = u32::from(color.full_range);
                            }
                            pick!($bit_depth_enum, {
                                av1.inputBitDepth = nvenc::$module::_NV_ENC_BIT_DEPTH(bit_depth);
                                av1.outputBitDepth = nvenc::$module::_NV_ENC_BIT_DEPTH(bit_depth);
                            }, {
                                av1.set_inputPixelBitDepthMinus8(bit_depth - 8);
                                av1.set_pixelBitDepthMinus8(bit_depth - 8);
                            });
                            pick!($native_hdr, {
                                let hdr10 = config.hdr10.unwrap_or_default();
                                av1.set_outputMasteringDisplay(u32::from(hdr10.mastering_display.is_some()));
                                av1.set_outputMaxCll(u32::from(hdr10.content_light.is_some()));
                            }, {
                                if config.hdr10.is_some() {
                                    return Err(ApiFeature::NativeHdrMetadata
                                        .unsupported("AV1 HDR10 metadata", Self::VERSION));
                                }
                            });
                            nv.encodeCodecConfig.av1Config = av1;
                        }, {
                            return Err(ApiFeature::Av1.unsupported("AV1 encoding", Self::VERSION));
                        });
                    }
                }
                Ok(())
            }

            fn initialize_params(
                config: &Config,
                nv: &mut nvenc::$module::NV_ENC_CONFIG,
            ) -> Result<nvenc::$module::NV_ENC_INITIALIZE_PARAMS> {
                Ok(nvenc::$module::NV_ENC_INITIALIZE_PARAMS {
                    version: Self::INITIALIZE_PARAMS_VER,
                    encodeGUID: Self::require_codec_guid(config.codec)?,
                    presetGUID: Self::preset_guid(config.preset),
                    encodeWidth: config.width,
                    encodeHeight: config.height,
                    darWidth: config.width,
                    darHeight: config.height,
                    frameRateNum: config.frame_rate.num.get(),
                    frameRateDen: config.frame_rate.den.get(),
                    // Linux has no asynchronous mode; on Windows the encoder
                    // still locks synchronously, which works on both.
                    enableEncodeAsync: 0,
                    enablePTD: 1,
                    encodeConfig: nv,
                    maxEncodeWidth: config.width,
                    maxEncodeHeight: config.height,
                    tuningInfo: Self::tuning(config.tuning)?,
                    bufferFormat: Self::buffer_format(config.input_format),
                    ..Default::default()
                })
            }
        }

        impl EncodeApi for api::$marker {
            const AV1: bool = pick!($av1, { true }, { false });
            const NATIVE_HDR: bool = pick!($native_hdr, { true }, { false });
            const CHROMA_OFFSET: bool = pick!($chroma_offset_in, { true }, { false });
            const ULTRA_HIGH_QUALITY: bool = pick!($uhq, { true }, { false });

            fn codecs(session: &Session<Self>) -> Result<Vec<Codec>> {
                let count = entry!(session, nvEncGetEncodeGUIDCount);
                let list = entry!(session, nvEncGetEncodeGUIDs);
                let mut len = 0;
                // SAFETY: live session, valid out-pointer.
                let status = unsafe { count(session.as_ptr(), &raw mut len) };
                check(session, status.0, "nvEncGetEncodeGUIDCount")?;
                let mut guids = vec![nvenc::$module::GUID::default(); len as usize];
                let mut written = 0;
                // SAFETY: `guids` holds `len` GUIDs; valid out-pointer.
                let status = unsafe { list(session.as_ptr(), guids.as_mut_ptr(), len, &raw mut written) };
                check(session, status.0, "nvEncGetEncodeGUIDs")?;
                guids.truncate(written as usize);
                Ok(Codec::ALL
                    .into_iter()
                    .filter(|codec| Self::codec_guid(*codec).is_some_and(|guid| guids.iter().any(|listed| Self::same_guid(listed, &guid))))
                    .collect())
            }

            fn caps(session: &Session<Self>, codec: Codec, cap: Cap) -> Result<i32> {
                let get = entry!(session, nvEncGetEncodeCaps);
                let mut param = nvenc::$module::NV_ENC_CAPS_PARAM {
                    version: Self::CAPS_PARAM_VER,
                    capsToQuery: nvenc::$module::_NV_ENC_CAPS(cap.0),
                    ..Default::default()
                };
                let mut value = 0;
                // SAFETY: live session; valid struct of this version and
                // out-pointer.
                let status = unsafe {
                    get(session.as_ptr(), Self::require_codec_guid(codec)?, &raw mut param, &raw mut value)
                };
                check(session, status.0, "nvEncGetEncodeCaps")?;
                Ok(value)
            }

            fn input_formats(session: &Session<Self>, codec: Codec) -> Result<Vec<InputFormat>> {
                let guid = Self::require_codec_guid(codec)?;
                let count = entry!(session, nvEncGetInputFormatCount);
                let list = entry!(session, nvEncGetInputFormats);
                let mut len = 0;
                // SAFETY: live session, valid out-pointer.
                let status = unsafe { count(session.as_ptr(), guid, &raw mut len) };
                check(session, status.0, "nvEncGetInputFormatCount")?;
                let mut formats =
                    vec![nvenc::$module::_NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_UNDEFINED; len as usize];
                let mut written = 0;
                // SAFETY: `formats` holds `len` entries; valid out-pointer.
                let status = unsafe {
                    list(session.as_ptr(), guid, formats.as_mut_ptr(), len, &raw mut written)
                };
                check(session, status.0, "nvEncGetInputFormats")?;
                formats.truncate(written as usize);
                Ok(formats.into_iter().filter_map(Self::input_format).collect())
            }

            fn initialize(session: &Session<Self>, config: &Config) -> Result<()> {
                let initialize = entry!(session, nvEncInitializeEncoder);
                let mut nv = Self::build_config(session, config)?;
                let mut params = Self::initialize_params(config, &mut nv)?;
                // SAFETY: live session; `params` and the configuration it
                // points to are valid structs of this version that outlive
                // the call.
                let status = unsafe { initialize(session.as_ptr(), &raw mut params) };
                check(session, status.0, "nvEncInitializeEncoder")
            }

            fn reconfigure(session: &Session<Self>, config: &Config, force_idr: bool) -> Result<()> {
                let reconfigure = entry!(session, nvEncReconfigureEncoder);
                let mut nv = Self::build_config(session, config)?;
                let mut params = nvenc::$module::NV_ENC_RECONFIGURE_PARAMS {
                    version: Self::RECONFIGURE_PARAMS_VER,
                    reInitEncodeParams: Self::initialize_params(config, &mut nv)?,
                    ..Default::default()
                };
                params.set_resetEncoder(0);
                params.set_forceIDR(u32::from(force_idr));
                // SAFETY: as in `initialize`.
                let status = unsafe { reconfigure(session.as_ptr(), &raw mut params) };
                check(session, status.0, "nvEncReconfigureEncoder")
            }

            fn sequence_params(session: &Session<Self>, out: &mut Vec<u8>) -> Result<()> {
                /// Comfortably above SPS + PPS + VPS or an AV1 sequence
                /// header.
                const CAPACITY: usize = 4096;
                let get = entry!(session, nvEncGetSequenceParams);
                out.clear();
                out.resize(CAPACITY, 0);
                let mut size = 0;
                let mut payload = nvenc::$module::NV_ENC_SEQUENCE_PARAM_PAYLOAD {
                    version: Self::SEQUENCE_PARAM_PAYLOAD_VER,
                    inBufferSize: to_u32(CAPACITY, "the sequence parameter buffer")?,
                    spsppsBuffer: out.as_mut_ptr().cast(),
                    outSPSPPSPayloadSize: &raw mut size,
                    ..Default::default()
                };
                // SAFETY: live session; the buffer holds `inBufferSize`
                // bytes and the size out-pointer is valid for the call.
                let status = unsafe { get(session.as_ptr(), &raw mut payload) };
                let checked = check(session, status.0, "nvEncGetSequenceParams");
                out.truncate(if checked.is_ok() { (size as usize).min(CAPACITY) } else { 0 });
                checked
            }

            fn create_bitstream(session: &Session<Self>) -> Result<NonNull<c_void>> {
                let create = entry!(session, nvEncCreateBitstreamBuffer);
                let mut params = nvenc::$module::NV_ENC_CREATE_BITSTREAM_BUFFER {
                    version: Self::CREATE_BITSTREAM_BUFFER_VER,
                    ..Default::default()
                };
                // SAFETY: live session; valid struct of this version.
                let status = unsafe { create(session.as_ptr(), &raw mut params) };
                check(session, status.0, "nvEncCreateBitstreamBuffer")?;
                returned(params.bitstreamBuffer, "nvEncCreateBitstreamBuffer", "a bitstream buffer")
            }

            unsafe fn destroy_bitstream(session: &Session<Self>, buffer: NonNull<c_void>) -> Result<()> {
                let destroy = entry!(session, nvEncDestroyBitstreamBuffer);
                // SAFETY: the caller guarantees `buffer` is this session's
                // and unused.
                let status = unsafe { destroy(session.as_ptr(), buffer.as_ptr()) };
                check(session, status.0, "nvEncDestroyBitstreamBuffer")
            }

            unsafe fn set_io_stream(session: &Session<Self>, stream: NonNull<CUstream>) -> Result<()> {
                let set = entry!(session, nvEncSetIOCudaStreams);
                let stream = stream.as_ptr().cast::<c_void>();
                // SAFETY: the caller guarantees the stream (and the pointer
                // to it) outlive the session.
                let status = unsafe { set(session.as_ptr(), stream, stream) };
                check(session, status.0, "nvEncSetIOCudaStreams")
            }

            unsafe fn register(
                session: &Session<Self>,
                registration: &Registration,
            ) -> Result<NonNull<c_void>> {
                type Kind = nvenc::$module::_NV_ENC_INPUT_RESOURCE_TYPE;
                let register = entry!(session, nvEncRegisterResource);
                let (resource_type, resource) = match registration.resource {
                    Resource::DevicePtr(pointer) => (
                        Kind::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                        std::ptr::without_provenance_mut::<c_void>(
                            usize::try_from(pointer).map_err(|_| Error::SizeOverflow { what: "a device address" })?,
                        ),
                    ),
                    Resource::Array(array) => (Kind::NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY, array.cast::<c_void>()),
                };
                #[allow(unused_mut, reason = "mutated only by the versions with chromaOffsetIn")]
                let mut params = nvenc::$module::NV_ENC_REGISTER_RESOURCE {
                    version: Self::REGISTER_RESOURCE_VER,
                    resourceType: resource_type,
                    width: registration.width,
                    height: registration.height,
                    pitch: registration.pitch,
                    resourceToRegister: resource,
                    bufferFormat: Self::buffer_format(registration.format),
                    bufferUsage: nvenc::$module::_NV_ENC_BUFFER_USAGE::NV_ENC_INPUT_IMAGE,
                    ..Default::default()
                };
                pick!($chroma_offset_in, {
                    if let Some(offset) = registration.chroma_offset {
                        params.chromaOffsetIn = [offset, 0];
                    }
                }, {
                    if registration.chroma_offset.is_some() {
                        return Err(ApiFeature::ChromaPlaneOffset
                            .unsupported("a chroma plane offset", Self::VERSION));
                    }
                });
                // SAFETY: live session; the caller guarantees the resource;
                // valid struct of this version.
                let status = unsafe { register(session.as_ptr(), &raw mut params) };
                check(session, status.0, "nvEncRegisterResource")?;
                returned(params.registeredResource, "nvEncRegisterResource", "a registered resource")
            }

            unsafe fn unregister(session: &Session<Self>, registered: NonNull<c_void>) -> Result<()> {
                let unregister = entry!(session, nvEncUnregisterResource);
                // SAFETY: the caller guarantees `registered` is this
                // session's, unmapped and unused.
                let status = unsafe { unregister(session.as_ptr(), registered.as_ptr()) };
                check(session, status.0, "nvEncUnregisterResource")
            }

            unsafe fn map(session: &Session<Self>, registered: NonNull<c_void>) -> Result<Mapped> {
                let map = entry!(session, nvEncMapInputResource);
                let mut params = nvenc::$module::NV_ENC_MAP_INPUT_RESOURCE {
                    version: Self::MAP_INPUT_RESOURCE_VER,
                    registeredResource: registered.as_ptr(),
                    ..Default::default()
                };
                // SAFETY: live session; the caller guarantees `registered`;
                // valid struct of this version.
                let status = unsafe { map(session.as_ptr(), &raw mut params) };
                check(session, status.0, "nvEncMapInputResource")?;
                Ok(Mapped {
                    input: returned(params.mappedResource, "nvEncMapInputResource", "a mapped resource")?,
                    format: params.mappedBufferFmt.0,
                })
            }

            unsafe fn unmap(session: &Session<Self>, mapped: Mapped) -> Result<()> {
                let unmap = entry!(session, nvEncUnmapInputResource);
                // SAFETY: the caller guarantees `mapped` is this session's
                // and no longer read.
                let status = unsafe { unmap(session.as_ptr(), mapped.input.as_ptr()) };
                check(session, status.0, "nvEncUnmapInputResource")
            }

            unsafe fn encode(session: &Session<Self>, picture: &Picture<'_>) -> Result<Submitted> {
                type Flags = nvenc::$module::_NV_ENC_PIC_FLAGS;
                let encode = entry!(session, nvEncEncodePicture);
                let mut flags = 0;
                if picture.force_idr {
                    flags |= Flags::NV_ENC_PIC_FLAG_FORCEIDR.0;
                }
                if picture.parameter_sets {
                    flags |= Flags::NV_ENC_PIC_FLAG_OUTPUT_SPSPPS.0;
                }
                let mut params = nvenc::$module::NV_ENC_PIC_PARAMS {
                    version: Self::PIC_PARAMS_VER,
                    inputWidth: picture.width,
                    inputHeight: picture.height,
                    inputPitch: picture.pitch,
                    encodePicFlags: flags,
                    inputTimeStamp: picture.timestamp,
                    inputBuffer: picture.input.input.as_ptr(),
                    outputBitstream: picture.output.as_ptr(),
                    bufferFmt: nvenc::$module::_NV_ENC_BUFFER_FORMAT(picture.input.format),
                    pictureStruct: nvenc::$module::_NV_ENC_PIC_STRUCT::NV_ENC_PIC_STRUCT_FRAME,
                    ..Default::default()
                };

                // HDR10 metadata. Everything the parameters point to lives on
                // this frame until the call returns; NVENC copies it.
                let hdr_sei = picture
                    .hdr10
                    .filter(|_| picture.expect_idr)
                    .map(HdrSei::from)
                    .unwrap_or_default();
                let (mut mastering_payload, mut light_payload) = (
                    hdr_sei.mastering_display.unwrap_or_default(),
                    hdr_sei.content_light.unwrap_or_default(),
                );
                let mut sei_payloads = [nvenc::$module::NV_ENC_SEI_PAYLOAD::default(); 2];
                let mut sei_count = 0;
                if hdr_sei.mastering_display.is_some() {
                    sei_payloads[sei_count] = nvenc::$module::NV_ENC_SEI_PAYLOAD {
                        payloadSize: to_u32(mastering_payload.len(), "an SEI payload")?,
                        payloadType: sei::MASTERING_DISPLAY_TYPE,
                        payload: mastering_payload.as_mut_ptr(),
                    };
                    sei_count += 1;
                }
                if hdr_sei.content_light.is_some() {
                    sei_payloads[sei_count] = nvenc::$module::NV_ENC_SEI_PAYLOAD {
                        payloadSize: to_u32(light_payload.len(), "an SEI payload")?,
                        payloadType: sei::CONTENT_LIGHT_TYPE,
                        payload: light_payload.as_mut_ptr(),
                    };
                    sei_count += 1;
                }
                let sei_count = to_u32(sei_count, "the SEI payload count")?;
                let sei_array = if sei_count > 0 { sei_payloads.as_mut_ptr() } else { std::ptr::null_mut() };

                match picture.codec {
                    Codec::H264 => {
                        // SAFETY: the union is zeroed plain data; the H.264
                        // member is the one this session uses.
                        let mut h264 = unsafe { params.codecPicParams.h264PicParams };
                        h264.seiPayloadArrayCnt = sei_count;
                        h264.seiPayloadArray = sei_array;
                        params.codecPicParams.h264PicParams = h264;
                    }
                    Codec::Hevc => {
                        // SAFETY: as for H.264.
                        let mut hevc = unsafe { params.codecPicParams.hevcPicParams };
                        pick!($native_hdr, {
                            let hdr10 = picture.hdr10.copied().unwrap_or_default();
                            let mut mastering = hdr10.mastering_display.map(Self::mastering_display);
                            let mut light = hdr10.content_light.map(Self::content_light);
                            hevc.pMasteringDisplay = mastering
                                .as_mut()
                                .map_or(std::ptr::null_mut(), std::ptr::from_mut);
                            hevc.pMaxCll = light.as_mut().map_or(std::ptr::null_mut(), std::ptr::from_mut);
                            params.codecPicParams.hevcPicParams = hevc;
                            // SAFETY: live session; `params` and everything
                            // it points to are valid for the call.
                            let status = unsafe { encode(session.as_ptr(), &raw mut params) };
                            return Self::submitted(session, status.0);
                        }, {
                            hevc.seiPayloadArrayCnt = sei_count;
                            hevc.seiPayloadArray = sei_array;
                            params.codecPicParams.hevcPicParams = hevc;
                        });
                    }
                    Codec::Av1 => {
                        pick!($av1, {
                            pick!($native_hdr, {
                                // SAFETY: as for H.264.
                                let mut av1 = unsafe { params.codecPicParams.av1PicParams };
                                let hdr10 = picture.hdr10.copied().unwrap_or_default();
                                let mut mastering = hdr10.mastering_display.map(Self::mastering_display);
                                let mut light = hdr10.content_light.map(Self::content_light);
                                av1.pMasteringDisplay = mastering
                                    .as_mut()
                                    .map_or(std::ptr::null_mut(), std::ptr::from_mut);
                                av1.pMaxCll = light.as_mut().map_or(std::ptr::null_mut(), std::ptr::from_mut);
                                params.codecPicParams.av1PicParams = av1;
                                // SAFETY: as for HEVC.
                                let status = unsafe { encode(session.as_ptr(), &raw mut params) };
                                return Self::submitted(session, status.0);
                            }, {});
                        }, {
                            return Err(ApiFeature::Av1.unsupported("AV1 encoding", Self::VERSION));
                        });
                    }
                }
                // SAFETY: live session; the caller guarantees the input and
                // output; `params` and the SEI payloads it points to are
                // valid for the call.
                let status = unsafe { encode(session.as_ptr(), &raw mut params) };
                Self::submitted(session, status.0)
            }

            fn end_of_stream(session: &Session<Self>) -> Result<()> {
                let encode = entry!(session, nvEncEncodePicture);
                let mut params = nvenc::$module::NV_ENC_PIC_PARAMS {
                    version: Self::PIC_PARAMS_VER,
                    encodePicFlags: nvenc::$module::_NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_EOS.0,
                    ..Default::default()
                };
                // SAFETY: live session; an EOS picture carries no buffers.
                let status = unsafe { encode(session.as_ptr(), &raw mut params) };
                Self::submitted(session, status.0).map(|_| ())
            }

            unsafe fn lock(
                session: &Session<Self>,
                output: NonNull<c_void>,
                out: &mut Vec<u8>,
            ) -> Result<Locked> {
                let lock = entry!(session, nvEncLockBitstream);
                let unlock = entry!(session, nvEncUnlockBitstream);
                let mut params = nvenc::$module::NV_ENC_LOCK_BITSTREAM {
                    version: Self::LOCK_BITSTREAM_VER,
                    outputBitstream: output.as_ptr(),
                    ..Default::default()
                };
                params.set_doNotWait(0);
                // SAFETY: live session; the caller guarantees `output` holds
                // a submitted picture; valid struct of this version.
                let status = unsafe { lock(session.as_ptr(), &raw mut params) };
                check(session, status.0, "nvEncLockBitstream")?;
                let size = params.bitstreamSizeInBytes as usize;
                out.clear();
                if size > 0 && !params.bitstreamBufferPtr.is_null() {
                    // SAFETY: while locked, the driver guarantees
                    // `bitstreamSizeInBytes` readable bytes at
                    // `bitstreamBufferPtr`.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(params.bitstreamBufferPtr.cast::<u8>(), size)
                    };
                    out.extend_from_slice(bytes);
                }
                let locked = Locked {
                    timestamp: params.outputTimeStamp,
                    picture_type: params.pictureType.0,
                };
                // SAFETY: locked just above, the slice is no longer used.
                let status = unsafe { unlock(session.as_ptr(), output.as_ptr()) };
                check(session, status.0, "nvEncUnlockBitstream")?;
                Ok(locked)
            }
        }

        impl api::$marker {
            fn submitted(session: &Session<Self>, status: u32) -> Result<Submitted> {
                if status == NEED_MORE_INPUT {
                    Ok(Submitted::NeedMoreInput)
                } else {
                    check(session, status, "nvEncEncodePicture").map(|()| Submitted::OutputReady)
                }
            }

            pick!($native_hdr, {
                fn mastering_display(
                    display: super::config::MasteringDisplay,
                ) -> nvenc::$module::MASTERING_DISPLAY_INFO {
                    let point = |c: super::config::Chromaticity| nvenc::$module::CHROMA_POINTS { x: c.x, y: c.y };
                    nvenc::$module::MASTERING_DISPLAY_INFO {
                        g: point(display.green),
                        b: point(display.blue),
                        r: point(display.red),
                        whitePoint: point(display.white_point),
                        maxLuma: display.max_luminance,
                        minLuma: display.min_luminance,
                    }
                }

                fn content_light(
                    level: super::config::ContentLightLevel,
                ) -> nvenc::$module::CONTENT_LIGHT_LEVEL {
                    nvenc::$module::CONTENT_LIGHT_LEVEL {
                        maxContentLightLevel: level.max_content,
                        maxPicAverageLightLevel: level.max_frame_average,
                    }
                }
            }, {});
        }
    )*};
}

use crate::sys::nvenc;

encode_api! {
    V11_1 => v11_1 { av1: no, typed_vui: no, bit_depth_enum: no, native_hdr: no, chroma_offset_in: no, uhq: no };
    V12_0 => v12_0 { av1: yes, typed_vui: yes, bit_depth_enum: no, native_hdr: no, chroma_offset_in: no, uhq: no };
    V12_1 => v12_1 { av1: yes, typed_vui: yes, bit_depth_enum: no, native_hdr: no, chroma_offset_in: no, uhq: no };
    V12_2 => v12_2 { av1: yes, typed_vui: yes, bit_depth_enum: yes, native_hdr: no, chroma_offset_in: no, uhq: no };
    V13_0 => v13_0 { av1: yes, typed_vui: yes, bit_depth_enum: yes, native_hdr: yes, chroma_offset_in: yes, uhq: yes };
    V13_1 => v13_1 { av1: yes, typed_vui: yes, bit_depth_enum: yes, native_hdr: yes, chroma_offset_in: yes, uhq: yes };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens_match<A: EncodeApi>() {
        for feature in ApiFeature::ALL {
            assert_eq!(
                A::has(feature),
                feature.available_in(A::VERSION),
                "{feature:?} in {}",
                A::VERSION
            );
        }
    }

    #[test]
    fn version_tokens_match_added_in() {
        tokens_match::<api::V11_1>();
        tokens_match::<api::V12_0>();
        tokens_match::<api::V12_1>();
        tokens_match::<api::V12_2>();
        tokens_match::<api::V13_0>();
        tokens_match::<api::V13_1>();
    }
}
