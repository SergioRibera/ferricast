pub mod discovery;
pub mod session;

pub use discovery::MiracastDiscovery;
pub use session::MiracastSession;

use ferricast_core::{Codec, ProtocolHandler, Result};

/// Miracast protocol handler.
#[derive(Clone, Default)]
pub struct MiracastHandler;

impl ProtocolHandler for MiracastHandler {
    const PROTOCOL: &'static str = "miracast";
    const SUPPORTED_CODECS: &'static [Codec] = &[Codec::H264];

    type Discovery = MiracastDiscovery;
    type Session = MiracastSession;

    fn create_discovery(&self) -> MiracastDiscovery {
        MiracastDiscovery::default()
    }

    fn create_session(&self) -> Result<MiracastSession> {
        Ok(MiracastSession::default())
    }
}
