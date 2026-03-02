pub mod discovery;
pub mod session;

pub use discovery::AirPlayDiscovery;
pub use session::AirPlaySession;

use ferricast_core::{Codec, ProtocolHandler, Result};

/// Top-level protocol handler for AirPlay 2.
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
