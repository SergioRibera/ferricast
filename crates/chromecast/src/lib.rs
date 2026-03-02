pub mod castv2;
pub mod discovery;
pub mod session;

pub use castv2::{
    CastMessage, CastV2Error, GenericPayload, PayloadType, ProtocolVersion,
    namespace as cast_namespace,
};
pub use discovery::ChromecastDiscovery;
pub use session::ChromecastSession;

use ferricast_core::{Codec, ProtocolHandler, Result};

#[derive(Clone, Default)]
pub struct ChromecastHandler;

impl ProtocolHandler for ChromecastHandler {
    const PROTOCOL: &'static str = "chromecast";
    const SUPPORTED_CODECS: &'static [Codec] = &[Codec::H264, Codec::Vp8];

    type Discovery = ChromecastDiscovery;
    type Session = ChromecastSession;

    fn create_discovery(&self) -> ChromecastDiscovery {
        ChromecastDiscovery::default()
    }

    fn create_session(&self) -> Result<ChromecastSession> {
        Ok(ChromecastSession::default())
    }
}
