//! What an [`Encoder`](super::Encoder) is opened with: codec, picture size and
//! format, rate control, GOP, profile, colour and HDR metadata.
//!
//! Everything here is plain data. [`Config::validate`] rejects combinations
//! that make no sense whatever the GPU; what the GPU or the negotiated API
//! version lacks is checked when the encoder opens.

use std::fmt;
use std::num::NonZeroU32;

use crate::error::{Error, Result};

/// The video codec.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Codec {
    /// H.264 / AVC.
    H264,
    /// H.265 / HEVC.
    Hevc,
    /// AV1 (NVENC API 12.0+, Ada and newer GPUs).
    Av1,
}

impl Codec {
    /// Every codec, in the order capability queries report them.
    pub const ALL: [Self; 3] = [Self::H264, Self::Hevc, Self::Av1];
}

impl fmt::Display for Codec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::H264 => "H.264",
            Self::Hevc => "HEVC",
            Self::Av1 => "AV1",
        })
    }
}

/// The layout of the pictures fed to the encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InputFormat {
    /// 8-bit 4:2:0: a luma plane, then an interleaved UV plane of half the
    /// rows.
    Nv12,
    /// 10-bit 4:2:0 laid out as NV12 with 16-bit samples, the value in the
    /// high 10 bits.
    P010,
    /// 8-bit packed RGB, one 32-bit word per pixel with blue in the lowest
    /// byte: B, G, R, A in memory (DRM `ARGB8888`, ferricast's BGRA).
    Argb,
    /// 8-bit packed RGB, one 32-bit word per pixel with red in the lowest
    /// byte: R, G, B, A in memory (DRM `ABGR8888`, ferricast's RGBA).
    Abgr,
}

impl InputFormat {
    /// Every format, in the order capability queries report them.
    pub const ALL: [Self; 4] = [Self::Nv12, Self::P010, Self::Argb, Self::Abgr];

    /// Bits per sample of the encoded video.
    #[must_use]
    pub const fn bit_depth(self) -> u32 {
        match self {
            Self::P010 => 10,
            Self::Nv12 | Self::Argb | Self::Abgr => 8,
        }
    }

    /// Whether the format has a separate, half-height chroma plane.
    #[must_use]
    pub const fn is_semi_planar(self) -> bool {
        matches!(self, Self::Nv12 | Self::P010)
    }

    /// Bytes of one row of the first (or only) plane.
    #[must_use]
    pub const fn row_bytes(self, width: u32) -> usize {
        let bytes_per_pixel = match self {
            Self::Nv12 => 1,
            Self::P010 => 2,
            Self::Argb | Self::Abgr => 4,
        };
        width as usize * bytes_per_pixel
    }

    /// Rows of the chroma plane, for semi-planar formats.
    #[must_use]
    pub const fn chroma_rows(self, height: u32) -> usize {
        if self.is_semi_planar() {
            height.div_ceil(2) as usize
        } else {
            0
        }
    }
}

impl fmt::Display for InputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Nv12 => "NV12",
            Self::P010 => "P010",
            Self::Argb => "ARGB",
            Self::Abgr => "ABGR",
        })
    }
}

/// Frames per second as a fraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameRate {
    /// Numerator.
    pub num: NonZeroU32,
    /// Denominator.
    pub den: NonZeroU32,
}

impl FrameRate {
    /// `fps` frames per second.
    #[must_use]
    pub const fn per_second(fps: NonZeroU32) -> Self {
        Self {
            num: fps,
            den: NonZeroU32::MIN,
        }
    }

    /// Frames per second as a float, for size estimates.
    #[must_use]
    pub fn as_f64(self) -> f64 {
        f64::from(self.num.get()) / f64::from(self.den.get())
    }
}

/// NVENC's speed / quality presets: P1 fastest, P7 best.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Preset {
    /// Fastest.
    P1,
    /// Faster.
    P2,
    /// Fast; ferricast's live default.
    #[default]
    P3,
    /// Medium.
    P4,
    /// Slow.
    P5,
    /// Slower.
    P6,
    /// Slowest, best quality.
    P7,
}

/// What the preset is tuned for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Tuning {
    /// Offline quality.
    HighQuality,
    /// Streaming and screen sharing.
    #[default]
    LowLatency,
    /// Interactive, one frame of VBV.
    UltraLowLatency,
    /// Lossless; needs [`RateControl::ConstQp`] (QP 0 for every frame type
    /// is the lossless setting).
    Lossless,
    /// Best quality at a large speed cost (NVENC API 13.0+, HEVC and AV1).
    UltraHighQuality,
}

/// Video buffering verifier: the decoder buffer the rate control respects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Vbv {
    /// Buffer size in bits.
    pub buffer_bits: u32,
    /// Fullness when decoding starts, in bits.
    pub initial_delay_bits: u32,
}

impl Vbv {
    /// One frame's worth of bits at `bitrate`: the lowest-latency buffer,
    /// what NVIDIA recommends for ultra-low-latency streaming.
    #[must_use]
    pub fn one_frame(bitrate: u32, frame_rate: FrameRate) -> Self {
        let bits =
            u64::from(bitrate) * u64::from(frame_rate.den.get()) / u64::from(frame_rate.num.get());
        let bits = u32::try_from(bits).unwrap_or(u32::MAX);
        Self {
            buffer_bits: bits,
            initial_delay_bits: bits,
        }
    }
}

/// Quantiser per frame type, for [`RateControl::ConstQp`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct QpSet {
    /// I / IDR frames.
    pub intra: u8,
    /// P frames.
    pub inter_p: u8,
    /// B frames.
    pub inter_b: u8,
}

/// How the encoder spends bits. Bitrates are in bits per second.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RateControl {
    /// Constant bitrate with the peak locked to the average (`maxBitRate ==
    /// averageBitRate`): scene changes cost quality, not bandwidth.
    Cbr {
        /// Target and peak bitrate.
        bitrate: u32,
        /// `None` keeps the preset's buffer.
        vbv: Option<Vbv>,
    },
    /// Variable bitrate around `average`, peaks up to `max`.
    Vbr {
        /// Long-term average.
        average: u32,
        /// Peak, at least `average`.
        max: u32,
        /// `None` keeps the preset's buffer.
        vbv: Option<Vbv>,
    },
    /// A fixed quantiser per frame type.
    ConstQp(QpSet),
}

impl RateControl {
    /// The average and peak bitrate, `None` for constant QP.
    #[must_use]
    pub const fn bitrates(self) -> Option<(u32, u32)> {
        match self {
            Self::Cbr { bitrate, .. } => Some((bitrate, bitrate)),
            Self::Vbr { average, max, .. } => Some((average, max)),
            Self::ConstQp(_) => None,
        }
    }

    /// The custom VBV, if one is set.
    #[must_use]
    pub const fn vbv(self) -> Option<Vbv> {
        match self {
            Self::Cbr { vbv, .. } | Self::Vbr { vbv, .. } => vbv,
            Self::ConstQp(_) => None,
        }
    }

    /// Whether two rate controls use the same mode (bitrates aside).
    #[must_use]
    pub const fn same_mode(self, other: Self) -> bool {
        matches!(
            (self, other),
            (Self::Cbr { .. }, Self::Cbr { .. })
                | (Self::Vbr { .. }, Self::Vbr { .. })
                | (Self::ConstQp(_), Self::ConstQp(_))
        )
    }
}

/// When key frames come.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Gop {
    /// Pictures from one I frame to the next; `None` never inserts one on
    /// its own (only forced IDRs).
    pub length: Option<NonZeroU32>,
    /// Pictures from one IDR to the next; `None` makes every GOP start with
    /// an IDR.
    pub idr_period: Option<NonZeroU32>,
}

impl Gop {
    /// An IDR every `frames` pictures.
    #[must_use]
    pub const fn idr_every(frames: NonZeroU32) -> Self {
        Self {
            length: Some(frames),
            idr_period: Some(frames),
        }
    }

    /// The IDR period NVENC applies: `idr_period`, else the GOP length,
    /// `None` when IDRs only come on request.
    #[must_use]
    pub const fn effective_idr_period(self) -> Option<NonZeroU32> {
        match self.idr_period {
            Some(period) => Some(period),
            None => self.length,
        }
    }
}

/// The codec profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Profile {
    /// H.264 Constrained Baseline.
    H264Baseline,
    /// H.264 Main.
    H264Main,
    /// H.264 High.
    H264High,
    /// HEVC Main (8-bit 4:2:0).
    HevcMain,
    /// HEVC Main 10 (10-bit 4:2:0).
    HevcMain10,
    /// AV1 Main (8 or 10-bit 4:2:0).
    Av1Main,
}

impl Profile {
    /// The codec this profile belongs to.
    #[must_use]
    pub const fn codec(self) -> Codec {
        match self {
            Self::H264Baseline | Self::H264Main | Self::H264High => Codec::H264,
            Self::HevcMain | Self::HevcMain10 => Codec::Hevc,
            Self::Av1Main => Codec::Av1,
        }
    }

    /// The bit depths the profile allows.
    #[must_use]
    const fn allows_bit_depth(self, bits: u32) -> bool {
        match self {
            Self::H264Baseline | Self::H264Main | Self::H264High | Self::HevcMain => bits == 8,
            Self::HevcMain10 => bits == 10,
            Self::Av1Main => bits == 8 || bits == 10,
        }
    }

    /// The profile used when [`Config::profile`] is `None`.
    #[must_use]
    pub const fn default_for(codec: Codec, format: InputFormat) -> Self {
        match (codec, format.bit_depth()) {
            (Codec::H264, _) => Self::H264High,
            (Codec::Hevc, 10) => Self::HevcMain10,
            (Codec::Hevc, _) => Self::HevcMain,
            (Codec::Av1, _) => Self::Av1Main,
        }
    }
}

/// ITU-T H.273 colour primaries (`colour_primaries`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorPrimaries(pub u8);

impl ColorPrimaries {
    /// BT.709 / sRGB.
    pub const BT709: Self = Self(1);
    /// Unspecified.
    pub const UNSPECIFIED: Self = Self(2);
    /// BT.601 625 lines.
    pub const BT470BG: Self = Self(5);
    /// BT.601 525 lines.
    pub const SMPTE170M: Self = Self(6);
    /// BT.2020 / BT.2100.
    pub const BT2020: Self = Self(9);
}

/// ITU-T H.273 transfer characteristics (`transfer_characteristics`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransferCharacteristics(pub u8);

impl TransferCharacteristics {
    /// BT.709.
    pub const BT709: Self = Self(1);
    /// Unspecified.
    pub const UNSPECIFIED: Self = Self(2);
    /// BT.601.
    pub const SMPTE170M: Self = Self(6);
    /// sRGB / sYCC (IEC 61966-2-1).
    pub const SRGB: Self = Self(13);
    /// SMPTE ST 2084 (PQ), HDR10.
    pub const PQ: Self = Self(16);
    /// ARIB STD-B67 (HLG).
    pub const HLG: Self = Self(18);
}

/// ITU-T H.273 matrix coefficients (`matrix_coefficients`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatrixCoefficients(pub u8);

impl MatrixCoefficients {
    /// Identity (RGB).
    pub const IDENTITY: Self = Self(0);
    /// BT.709.
    pub const BT709: Self = Self(1);
    /// Unspecified.
    pub const UNSPECIFIED: Self = Self(2);
    /// BT.601.
    pub const BT470BG: Self = Self(5);
    /// BT.601 525 lines.
    pub const SMPTE170M: Self = Self(6);
    /// BT.2020 non-constant luminance.
    pub const BT2020_NCL: Self = Self(9);
}

/// The colour description written to the stream (H.264 / HEVC VUI, AV1
/// colour config).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorDescription {
    /// Primaries.
    pub primaries: ColorPrimaries,
    /// Transfer function.
    pub transfer: TransferCharacteristics,
    /// YUV matrix.
    pub matrix: MatrixCoefficients,
    /// Full (0-255) rather than limited (16-235) range.
    pub full_range: bool,
}

impl ColorDescription {
    /// BT.709 limited range, the SDR default.
    pub const BT709: Self = Self {
        primaries: ColorPrimaries::BT709,
        transfer: TransferCharacteristics::BT709,
        matrix: MatrixCoefficients::BT709,
        full_range: false,
    };

    /// BT.2100 PQ limited range, HDR10.
    pub const HDR10: Self = Self {
        primaries: ColorPrimaries::BT2020,
        transfer: TransferCharacteristics::PQ,
        matrix: MatrixCoefficients::BT2020_NCL,
        full_range: false,
    };
}

/// A CIE 1931 xy chromaticity in units of 0.00002 (the SEI / AV1 metadata
/// encoding: 0.3127 is 15635).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Chromaticity {
    /// x.
    pub x: u16,
    /// y.
    pub y: u16,
}

/// The mastering display colour volume (SMPTE ST 2086): HEVC / H.264 SEI
/// 137, AV1 `METADATA_TYPE_HDR_MDCV`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MasteringDisplay {
    /// Red primary.
    pub red: Chromaticity,
    /// Green primary.
    pub green: Chromaticity,
    /// Blue primary.
    pub blue: Chromaticity,
    /// White point.
    pub white_point: Chromaticity,
    /// Peak luminance in units of 0.0001 cd/m².
    pub max_luminance: u32,
    /// Minimum luminance in units of 0.0001 cd/m².
    pub min_luminance: u32,
}

impl MasteringDisplay {
    /// BT.2020 primaries, D65 white, 1000 cd/m² peak, 0.0001 cd/m² black:
    /// the usual HDR10 mastering display.
    pub const BT2020_1000_NITS: Self = Self {
        red: Chromaticity { x: 35400, y: 14600 },
        green: Chromaticity { x: 8500, y: 39850 },
        blue: Chromaticity { x: 6550, y: 2300 },
        white_point: Chromaticity { x: 15635, y: 16450 },
        max_luminance: 10_000_000,
        min_luminance: 1,
    };
}

/// Content light level (CTA-861.3): HEVC / H.264 SEI 144, AV1
/// `METADATA_TYPE_HDR_CLL`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ContentLightLevel {
    /// `MaxCLL` in cd/m².
    pub max_content: u16,
    /// `MaxFALL` in cd/m².
    pub max_frame_average: u16,
}

/// HDR10 static metadata, written with every IDR.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Hdr10 {
    /// Mastering display colour volume.
    pub mastering_display: Option<MasteringDisplay>,
    /// Content light level.
    pub content_light: Option<ContentLightLevel>,
}

/// Everything an encoder is opened with.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Config {
    /// Codec.
    pub codec: Codec,
    /// Picture width in pixels.
    pub width: u32,
    /// Picture height in pixels.
    pub height: u32,
    /// Layout of the input pictures.
    pub input_format: InputFormat,
    /// Frame rate (rate control and VUI timing).
    pub frame_rate: FrameRate,
    /// Rate control.
    pub rate_control: RateControl,
    /// Preset.
    pub preset: Preset,
    /// Tuning.
    pub tuning: Tuning,
    /// Key frame structure.
    pub gop: Gop,
    /// B frames between references (0: none, lowest latency).
    pub b_frames: u8,
    /// Frames of rate-control lookahead (`None`: off). Delays output by that
    /// many frames.
    pub lookahead: Option<u16>,
    /// `None` picks [`Profile::default_for`].
    pub profile: Option<Profile>,
    /// Write the parameter sets (SPS / PPS / VPS, AV1 sequence header) with
    /// every IDR, so a receiver can join at any key frame.
    pub repeat_parameter_sets: bool,
    /// Colour description; `None` writes none (decoders assume BT.709 /
    /// BT.601).
    pub color: Option<ColorDescription>,
    /// HDR10 static metadata.
    pub hdr10: Option<Hdr10>,
    /// Pictures the encoder may hold before handing their packets out: 0
    /// returns each packet from the call that encodes it (lowest latency);
    /// more lets the GPU encode one picture while the next is submitted.
    /// B frames and lookahead add their own delay on top.
    pub frames_in_flight: u8,
}

impl Config {
    /// A low-latency CBR configuration: preset P3, low-latency tuning, no B
    /// frames, parameter sets with every IDR, BT.709 colour.
    #[must_use]
    pub fn low_latency(
        codec: Codec,
        width: u32,
        height: u32,
        input_format: InputFormat,
        frame_rate: FrameRate,
        bitrate: u32,
    ) -> Self {
        Self {
            codec,
            width,
            height,
            input_format,
            frame_rate,
            rate_control: RateControl::Cbr { bitrate, vbv: None },
            preset: Preset::P3,
            tuning: Tuning::LowLatency,
            gop: Gop::default(),
            b_frames: 0,
            lookahead: None,
            profile: None,
            repeat_parameter_sets: true,
            color: Some(ColorDescription::BT709),
            hdr10: None,
            frames_in_flight: 0,
        }
    }

    /// The profile used: [`Config::profile`] or the codec's default.
    #[must_use]
    pub fn effective_profile(&self) -> Profile {
        self.profile
            .unwrap_or_else(|| Profile::default_for(self.codec, self.input_format))
    }

    /// Bitstream buffers (and input slots) the encoder keeps: one per
    /// picture it may hold, plus one being encoded.
    #[must_use]
    pub(crate) fn pipeline_depth(&self) -> usize {
        1 + usize::from(self.frames_in_flight)
            + usize::from(self.b_frames)
            + usize::from(self.lookahead.unwrap_or(0))
    }

    /// Checks what can be checked without a GPU.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidEncodeConfig`] naming the field.
    pub fn validate(&self) -> Result<()> {
        let invalid =
            |field: &'static str, reason: String| Err(Error::InvalidEncodeConfig { field, reason });
        if self.width == 0 || self.height == 0 {
            return invalid("size", format!("{}x{} is empty", self.width, self.height));
        }
        if self.input_format.is_semi_planar()
            && (!self.width.is_multiple_of(2) || !self.height.is_multiple_of(2))
        {
            return invalid(
                "size",
                format!(
                    "{}x{}: {} needs an even width and height",
                    self.width, self.height, self.input_format
                ),
            );
        }
        let profile = self.effective_profile();
        if profile.codec() != self.codec {
            return invalid(
                "profile",
                format!("{profile:?} is not a {} profile", self.codec),
            );
        }
        let bits = self.input_format.bit_depth();
        if !profile.allows_bit_depth(bits) {
            return invalid(
                "profile",
                format!(
                    "{profile:?} cannot encode {bits}-bit {} input",
                    self.input_format
                ),
            );
        }
        match self.rate_control {
            RateControl::Cbr { bitrate: 0, .. } | RateControl::Vbr { average: 0, .. } => {
                return invalid("rate_control", "a bitrate of 0".into());
            }
            RateControl::Vbr { average, max, .. } if max < average => {
                return invalid(
                    "rate_control",
                    format!("peak {max} below average {average}"),
                );
            }
            RateControl::ConstQp(qp) if qp.intra > 63 || qp.inter_p > 63 || qp.inter_b > 63 => {
                return invalid("rate_control", format!("QP above 63: {qp:?}"));
            }
            RateControl::Cbr { .. } | RateControl::Vbr { .. } | RateControl::ConstQp(_) => {}
        }
        if self.tuning == Tuning::Lossless && !matches!(self.rate_control, RateControl::ConstQp(_))
        {
            return invalid(
                "tuning",
                "lossless tuning needs RateControl::ConstQp".into(),
            );
        }
        if self.gop.length.is_none() && self.b_frames > 0 {
            return invalid(
                "b_frames",
                "B frames need a finite GOP (NVENC wants frameIntervalP 1 with an infinite GOP)"
                    .into(),
            );
        }
        if self.lookahead == Some(0) {
            return invalid("lookahead", "a lookahead of 0 frames: use None".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        Config::low_latency(
            Codec::Hevc,
            1920,
            1080,
            InputFormat::Nv12,
            FrameRate::per_second(NonZeroU32::new(60).unwrap()),
            8_000_000,
        )
    }

    #[test]
    fn defaults_validate() {
        config().validate().unwrap();
        assert_eq!(config().effective_profile(), Profile::HevcMain);
        let ten_bit = Config {
            input_format: InputFormat::P010,
            ..config()
        };
        assert_eq!(ten_bit.effective_profile(), Profile::HevcMain10);
        ten_bit.validate().unwrap();
    }

    #[test]
    fn rejects_nonsense() {
        let field = |config: Config| match config.validate() {
            Err(Error::InvalidEncodeConfig { field, .. }) => field,
            other => panic!("{other:?}"),
        };
        assert_eq!(
            field(Config {
                width: 0,
                ..config()
            }),
            "size"
        );
        assert_eq!(
            field(Config {
                height: 1081,
                ..config()
            }),
            "size"
        );
        assert_eq!(
            field(Config {
                profile: Some(Profile::H264High),
                ..config()
            }),
            "profile"
        );
        assert_eq!(
            field(Config {
                profile: Some(Profile::HevcMain10),
                ..config()
            }),
            "profile"
        );
        assert_eq!(
            field(Config {
                rate_control: RateControl::Vbr {
                    average: 10,
                    max: 5,
                    vbv: None
                },
                ..config()
            }),
            "rate_control"
        );
        assert_eq!(
            field(Config {
                tuning: Tuning::Lossless,
                ..config()
            }),
            "tuning"
        );
        assert_eq!(
            field(Config {
                b_frames: 2,
                ..config()
            }),
            "b_frames"
        );
        // An odd size is fine for packed RGB.
        Config {
            width: 641,
            height: 479,
            input_format: InputFormat::Argb,
            ..config()
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn derived_values() {
        let rate = FrameRate::per_second(NonZeroU32::new(30).unwrap());
        assert_eq!(Vbv::one_frame(3_000_000, rate).buffer_bits, 100_000);
        assert_eq!(InputFormat::P010.row_bytes(1920), 3840);
        assert_eq!(InputFormat::Nv12.chroma_rows(1080), 540);
        assert_eq!(InputFormat::Argb.chroma_rows(1080), 0);
        let gop = Gop {
            length: NonZeroU32::new(60),
            idr_period: None,
        };
        assert_eq!(gop.effective_idr_period(), NonZeroU32::new(60));
        let deep = Config {
            frames_in_flight: 2,
            b_frames: 1,
            lookahead: Some(4),
            gop,
            ..config()
        };
        assert_eq!(deep.pipeline_depth(), 8);
    }
}
