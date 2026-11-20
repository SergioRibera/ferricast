//! AirPlay screen-mirroring sender.
//!
//! Targets receivers built on the open-source AirPlay server stack (UxPlay,
//! RPiPlay) using legacy pairing, with HomeKit pairing for AirPlay 2
//! receivers.

mod audio;
mod crypto;
mod digest;
pub mod discovery;
mod fairplay;
mod mirror;
mod ntp;
mod pair;
mod rtsp;
pub mod session;
mod srp;
mod store;
mod tlv;

pub use discovery::AirPlayDiscovery;
pub use session::AirPlaySession;

use ferricast_core::{Codec, ProtocolHandler, Result};

/// Top-level protocol handler for AirPlay.
#[derive(Clone, Default)]
pub struct AirPlayHandler;

impl ProtocolHandler for AirPlayHandler {
    const PROTOCOL: &'static str = "airplay";
    const SUPPORTED_CODECS: &'static [Codec] = &[Codec::H264];

    type Discovery = AirPlayDiscovery;
    type Session = AirPlaySession;

    fn create_discovery(&self) -> AirPlayDiscovery {
        AirPlayDiscovery::default()
    }

    fn create_session(&self) -> Result<AirPlaySession> {
        Ok(AirPlaySession::default())
    }
}
