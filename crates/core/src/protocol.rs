use crate::advertiser::{AdvertiseInfo, Advertiser};
use crate::control::ControlSession;
use crate::discovery::Discovery;
use crate::error::Result;
use crate::puller::MediaPuller;
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

/// Receiver-side protocol — advertise the local process as a sink,
/// accept a control connection from a remote sender, and translate
/// its `MediaCommand::Load { url }` into an internal media pull.
///
/// Chromecast is the canonical impl: advertiser publishes
/// `_googlecast._tcp`, control session speaks CASTV2 over TLS, and
/// the puller is the HLS one (Cast LOAD always hands us an HTTP URL).
/// AirPlay would fit the same shape with `_airplay._tcp` + AirPlay
/// HTTP + the same HLS puller.
///
/// Pure pull protocols (a user pastes an HLS URL into the GUI) don't
/// go through this trait — they invoke a [`MediaPuller`] directly
/// without any advertise/control step.
pub trait ReceiverProtocol: Send + Sync {
    const PROTOCOL: &'static str;
    const SUPPORTED_CODECS: &'static [Codec];

    type Advertiser: Advertiser;
    type Control: ControlSession;
    type Puller: MediaPuller;

    fn create_advertiser(&self) -> Self::Advertiser;
    fn create_control(&self) -> Result<Self::Control>;
    fn create_puller(&self) -> Result<Self::Puller>;

    /// What to publish about this receiver. Called once at
    /// registration; impls bake in service-type-specific TXT keys
    /// (`md`, `ca`, `ic` for Cast; `features`, `deviceid` for
    /// AirPlay) on top of the generic fields.
    fn advertise_info(&self) -> AdvertiseInfo;
}
