use crate::discovery::Discovery;
use crate::error::Result;
use crate::session::CastSession;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Codec {
    H264,
    H265,
    Vp8,
    Vp9,
}

impl std::fmt::Display for Codec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::H264 => write!(f, "H.264"),
            Self::H265 => write!(f, "H.265"),
            Self::Vp8 => write!(f, "VP8"),
            Self::Vp9 => write!(f, "VP9"),
        }
    }
}

pub trait ProtocolHandler: Send + Sync {
    const PROTOCOL: &'static str;
    const SUPPORTED_CODECS: &'static [Codec];

    type Discovery: Discovery;
    type Session: CastSession;

    fn create_discovery(&self) -> Self::Discovery;
    fn create_session(&self) -> Result<Self::Session>;
}
