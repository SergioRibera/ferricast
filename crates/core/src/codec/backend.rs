//! Codec backend identity.

use std::fmt;
use std::str::FromStr;

use super::Codec;
use crate::frame::AudioCodec;

/// One codec implementation the crates can drive. A backend may serve
/// several codecs and both directions (VideoToolbox encodes and decodes
/// H.264 and H.265); [`CodecTarget`] names the job.
///
/// The textual name ([`Self::name`], [`FromStr`]) is what environment
/// variables and the istmo plugin use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
#[non_exhaustive]
pub enum BackendId {
    /// libx264, software H.264 encode.
    X264,
    /// Cisco openh264, software H.264 decode.
    OpenH264,
    /// VA-API (Intel, AMD on Linux), hardware H.264 / H.265.
    Vaapi,
    /// NVIDIA NVENC, hardware H.264 / H.265 encode.
    Nvenc,
    /// NVIDIA NVDEC, hardware H.264 / H.265 decode.
    Nvdec,
    /// `rust_h265`, pure-Rust software H.265 decode.
    RustHevc,
    /// `vp8-decoder`, pure-Rust software VP8 decode.
    RustVp8,
    /// `rusty_vp9`, pure-Rust software VP9 decode.
    RustVp9,
    /// rav1d (dav1d port), software AV1 decode.
    Rav1d,
    /// Fraunhofer libfdk-aac, AAC-LC / HE-AAC / AAC-ELD.
    FdkAac,
    /// Symphonia, pure-Rust AAC-LC and ALAC decode.
    Symphonia,
    /// `opus-rs`, pure-Rust Opus.
    OpusRs,
    /// Raw PCM pass-through.
    Pcm,
    /// Windows Media Foundation transforms (hardware or software).
    MediaFoundation,
    /// Apple VideoToolbox (macOS, iOS), hardware H.264 / H.265.
    VideoToolbox,
    /// Apple AudioToolbox `AudioConverter` (macOS, iOS), AAC.
    AudioToolbox,
    /// Android NDK `AMediaCodec` (hardware or software).
    MediaCodec,
}

impl BackendId {
    /// Every backend, in declaration order.
    pub const ALL: [Self; 17] = [
        Self::X264,
        Self::OpenH264,
        Self::Vaapi,
        Self::Nvenc,
        Self::Nvdec,
        Self::RustHevc,
        Self::RustVp8,
        Self::RustVp9,
        Self::Rav1d,
        Self::FdkAac,
        Self::Symphonia,
        Self::OpusRs,
        Self::Pcm,
        Self::MediaFoundation,
        Self::VideoToolbox,
        Self::AudioToolbox,
        Self::MediaCodec,
    ];

    /// Stable lowercase name, as environment variables spell it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::X264 => "x264",
            Self::OpenH264 => "openh264",
            Self::Vaapi => "vaapi",
            Self::Nvenc => "nvenc",
            Self::Nvdec => "nvdec",
            Self::RustHevc => "rust-hevc",
            Self::RustVp8 => "rust-vp8",
            Self::RustVp9 => "rust-vp9",
            Self::Rav1d => "rav1d",
            Self::FdkAac => "fdk-aac",
            Self::Symphonia => "symphonia",
            Self::OpusRs => "opus-rs",
            Self::Pcm => "pcm",
            Self::MediaFoundation => "media-foundation",
            Self::VideoToolbox => "videotoolbox",
            Self::AudioToolbox => "audiotoolbox",
            Self::MediaCodec => "mediacodec",
        }
    }

    /// Every valid name, comma separated, for error messages.
    #[must_use]
    pub fn all_names() -> String {
        Self::ALL.map(Self::name).join(", ")
    }
}

impl fmt::Display for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A string that names no [`BackendId`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "{value:?} is not a codec backend; expected one of: {}",
    BackendId::all_names()
)]
pub struct UnknownBackend {
    pub value: String,
}

impl FromStr for BackendId {
    type Err = UnknownBackend;

    /// Case-insensitive; `_` and `-` are interchangeable.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let wanted = value.trim().to_ascii_lowercase().replace('_', "-");
        Self::ALL
            .into_iter()
            .find(|backend| backend.name() == wanted)
            .ok_or_else(|| UnknownBackend {
                value: value.to_owned(),
            })
    }
}

/// Whether a backend instance runs on dedicated media hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub enum Acceleration {
    Hardware,
    Software,
}

impl fmt::Display for Acceleration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Hardware => "hardware",
            Self::Software => "software",
        })
    }
}

/// Encode or decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub enum Direction {
    Encode,
    Decode,
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Encode => "encode",
            Self::Decode => "decode",
        })
    }
}

/// A video or audio codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub enum CodecKind {
    Video(Codec),
    Audio(AudioCodec),
}

impl fmt::Display for CodecKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Video(codec) => codec.fmt(f),
            Self::Audio(codec) => codec.fmt(f),
        }
    }
}

impl From<Codec> for CodecKind {
    fn from(codec: Codec) -> Self {
        Self::Video(codec)
    }
}

impl From<AudioCodec> for CodecKind {
    fn from(codec: AudioCodec) -> Self {
        Self::Audio(codec)
    }
}

/// A job a backend is asked to do: encode or decode one codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub struct CodecTarget {
    pub codec: CodecKind,
    pub direction: Direction,
}

impl CodecTarget {
    #[must_use]
    pub fn encode(codec: impl Into<CodecKind>) -> Self {
        Self {
            codec: codec.into(),
            direction: Direction::Encode,
        }
    }

    #[must_use]
    pub fn decode(codec: impl Into<CodecKind>) -> Self {
        Self {
            codec: codec.into(),
            direction: Direction::Decode,
        }
    }

    /// The environment variable that overrides automatic backend
    /// selection for this job, e.g. `FERRICAST_H264_DECODE_BACKEND`.
    #[must_use]
    pub fn env_var(self) -> String {
        let codec = match self.codec {
            CodecKind::Video(Codec::H264) => "H264",
            CodecKind::Video(Codec::H265) => "H265",
            CodecKind::Video(Codec::Vp8) => "VP8",
            CodecKind::Video(Codec::Vp9) => "VP9",
            CodecKind::Video(Codec::Av1) => "AV1",
            CodecKind::Audio(AudioCodec::Aac) => "AAC",
            CodecKind::Audio(AudioCodec::AacEld) => "AAC_ELD",
            CodecKind::Audio(AudioCodec::Opus) => "OPUS",
            CodecKind::Audio(AudioCodec::Pcm) => "PCM",
            CodecKind::Audio(AudioCodec::Alac) => "ALAC",
        };
        let direction = match self.direction {
            Direction::Encode => "ENCODE",
            Direction::Decode => "DECODE",
        };
        format!("FERRICAST_{codec}_{direction}_BACKEND")
    }

    /// Variables earlier releases read for the same override, still
    /// honoured after [`Self::env_var`].
    #[must_use]
    pub fn legacy_env_vars(self) -> &'static [&'static str] {
        match (self.codec, self.direction) {
            (CodecKind::Video(Codec::H264), Direction::Encode) => &["FERRICAST_H264_BACKEND"],
            (CodecKind::Video(Codec::H265), Direction::Encode) => &["FERRICAST_H265_BACKEND"],
            (CodecKind::Audio(AudioCodec::Aac), Direction::Decode) => &["FERRICAST_AAC_DECODER"],
            _ => &[],
        }
    }
}

impl fmt::Display for CodecTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.codec, self.direction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip_and_parse_loosely() {
        for backend in BackendId::ALL {
            assert_eq!(backend.name().parse::<BackendId>(), Ok(backend));
        }
        assert_eq!("Media_Foundation".parse(), Ok(BackendId::MediaFoundation));
        assert_eq!(" NVENC ".parse(), Ok(BackendId::Nvenc));
        let error = "ffmpeg".parse::<BackendId>().unwrap_err();
        assert!(error.to_string().contains("videotoolbox"), "{error}");
    }

    #[test]
    fn env_vars_name_the_codec_and_direction() {
        assert_eq!(
            CodecTarget::decode(Codec::H264).env_var(),
            "FERRICAST_H264_DECODE_BACKEND"
        );
        assert_eq!(
            CodecTarget::encode(AudioCodec::AacEld).env_var(),
            "FERRICAST_AAC_ELD_ENCODE_BACKEND"
        );
        assert_eq!(
            CodecTarget::encode(Codec::H264).legacy_env_vars(),
            ["FERRICAST_H264_BACKEND"]
        );
        assert_eq!(CodecTarget::encode(Codec::H264).to_string(), "H.264 encode");
    }
}
