//! What a codec is ([`Codec`], [`AudioCodec`]), which implementation
//! runs it ([`BackendId`]), how one is picked ([`BackendSelection`])
//! and how it fails ([`CodecError`]).

mod backend;
mod error;
mod selection;

use std::fmt;

pub use backend::{Acceleration, BackendId, CodecKind, CodecTarget, Direction, UnknownBackend};
pub use error::{BackendAttempt, CodecError, CodecResult, Operation};
pub use selection::{AccelerationPolicy, AutoSelect, BackendSelection, RuntimeFallback};

use crate::frame::AudioCodec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub enum Codec {
    H264,
    H265,
    Vp8,
    Vp9,
    Av1,
}

impl fmt::Display for Codec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::H264 => write!(f, "H.264"),
            Self::H265 => write!(f, "H.265"),
            Self::Vp8 => write!(f, "VP8"),
            Self::Vp9 => write!(f, "VP9"),
            Self::Av1 => write!(f, "AV1"),
        }
    }
}

impl fmt::Display for AudioCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Aac => "AAC-LC",
            Self::AacEld => "AAC-ELD",
            Self::Opus => "Opus",
            Self::Pcm => "PCM",
            Self::Alac => "ALAC",
        })
    }
}
