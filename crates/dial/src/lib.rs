pub mod discovery;
pub mod session;

pub use discovery::DialDiscovery;
pub use session::DialSession;

use ferricast_core::{Codec, ProtocolHandler, Result};

#[derive(Debug, Clone, Copy, Default)]
pub struct DialHandler;

impl ProtocolHandler for DialHandler {
    const PROTOCOL: &'static str = "dial";
    const SUPPORTED_CODECS: &'static [Codec] = &[Codec::H264];

    type Discovery = DialDiscovery;
    type Session = DialSession;

    fn create_discovery(&self) -> DialDiscovery {
        DialDiscovery::default()
    }

    fn create_session(&self) -> Result<DialSession> {
        Ok(DialSession::default())
    }
}
