//! `ProtocolHandler` impl: wires [`MiracastDiscovery`] and
//! [`MiracastSession`] into the ferricast-core trait surface so the
//! stream manager can register Miracast alongside Chromecast and
//! AirPlay.

use ferricast_core::{Codec, ProtocolHandler, Result};

use crate::{MiracastDiscovery, MiracastSession};

#[derive(Clone, Default)]
pub struct MiracastHandler;

impl ProtocolHandler for MiracastHandler {
    const PROTOCOL: &'static str = "miracast";
    // WFD mandates H.264 CBP; HEVC/VP9/AV1 are not in-spec. Add
    // AAC-LC on the audio side once the session wires it in.
    const SUPPORTED_CODECS: &'static [Codec] = &[Codec::H264];

    type Discovery = MiracastDiscovery;
    type Session = MiracastSession;

    fn create_discovery(&self) -> MiracastDiscovery {
        MiracastDiscovery::new()
    }

    fn create_session(&self) -> Result<MiracastSession> {
        Ok(MiracastSession::new())
    }
}
