pub mod castv2;
pub mod discovery;
pub mod mirroring;
pub mod receiver;
pub(crate) mod self_filter;
pub mod session;
mod wire;

use std::collections::HashSet;
use std::sync::{Arc, Mutex, PoisonError};

pub use castv2::{
    CastMessage, CastV2Error, GenericPayload, PayloadType, ProtocolVersion,
    namespace as cast_namespace,
};
pub use discovery::ChromecastDiscovery;
pub use mirroring::{ChromecastMirrorSession, MirrorFailure};
pub use receiver::{ChromecastReceiver, ChromecastReceiverAdvertiser, ChromecastReceiverControl};
pub use session::ChromecastSession;

use ferricast_core::{
    AudioCodec, AudioFrame, CastSession, Codec, ConnectOutcome, Device, EncodedFrame,
    FerricastError, ProtocolHandler, Result, StreamConfig,
};
use uuid::Uuid;

/// Devices on which Cast Streaming was tried and did not work during this
/// process; later sessions to them go straight to HLS.
#[derive(Clone, Default, Debug)]
pub struct MirrorBlocklist(Arc<Mutex<HashSet<Uuid>>>);

impl MirrorBlocklist {
    #[must_use]
    pub fn contains(&self, device: &Uuid) -> bool {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .contains(device)
    }

    pub fn insert(&self, device: Uuid) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(device);
    }
}

/// Wraps either the HLS session or the Cast Streaming (RTP/UDP) mirror
/// session. The manager stores `Box<dyn ErasedSession>` so both variants
/// erase identically; this enum is the single `Session` associated type
/// on [`ChromecastHandler`] that makes that work without touching the
/// `ProtocolHandler` trait.
///
/// The mirror variant falls back to HLS when the mirroring app cannot be
/// launched or the OFFER is not answered or accepted. When mirroring sets
/// up but the receiver never acknowledges a frame, the device is
/// blocklisted and the session reports itself dead, so the reconnect
/// builds an HLS session instead.
pub enum ChromecastEitherSession {
    Hls(ChromecastSession),
    Mirror {
        session: Box<ChromecastMirrorSession>,
        /// Captured during `connect` so a fallback can reconnect over HLS.
        connected_device: Option<Device>,
        blocklist: MirrorBlocklist,
    },
}

impl ChromecastEitherSession {
    /// Replaces the mirror session with a connected (and, given a config,
    /// streaming) HLS session.
    async fn fall_back_to_hls(
        &mut self,
        device: &Device,
        config: Option<&StreamConfig>,
    ) -> Result<ConnectOutcome> {
        let mut hls = ChromecastSession::default();
        let outcome = hls.connect(device).await?;
        if let Some(config) = config {
            let mut config = config.clone();
            // The mirror path asked for Opus; HLS carries AAC.
            if let Some(audio) = config.audio.as_mut() {
                audio.codec = AudioCodec::Aac;
            }
            hls.setup_stream(&config).await?;
        }
        *self = Self::Hls(hls);
        Ok(outcome)
    }
}

impl CastSession for ChromecastEitherSession {
    async fn connect(&mut self, device: &Device) -> Result<ConnectOutcome> {
        let error = match self {
            Self::Hls(s) => return s.connect(device).await,
            Self::Mirror {
                session,
                connected_device,
                blocklist,
            } => match session.connect(device).await {
                Ok(outcome) => {
                    *connected_device = Some(device.clone());
                    return Ok(outcome);
                }
                Err(e @ FerricastError::RemoteStopped(_)) => return Err(e),
                Err(e) => {
                    blocklist.insert(device.id);
                    let _ = session.stop().await;
                    e
                }
            },
        };
        tracing::warn!(%error, device = %device.name, "Cast Streaming unavailable — falling back to HLS");
        self.fall_back_to_hls(device, None).await
    }

    async fn setup_stream(&mut self, config: &StreamConfig) -> Result<()> {
        let device = match self {
            Self::Hls(s) => return s.setup_stream(config).await,
            Self::Mirror {
                session,
                connected_device,
                blocklist,
            } => match session.setup_stream(config).await {
                Ok(()) => return Ok(()),
                Err(e @ FerricastError::RemoteStopped(_)) => return Err(e),
                Err(e) => {
                    tracing::warn!(error = %e, "Cast Streaming setup failed — falling back to HLS");
                    let _ = session.stop().await;
                    let Some(device) = connected_device.take() else {
                        return Err(e);
                    };
                    blocklist.insert(device.id);
                    device
                }
            },
        };
        self.fall_back_to_hls(&device, Some(config)).await.map(drop)
    }

    async fn send_frame(&mut self, frame: &EncodedFrame) -> Result<()> {
        match self {
            Self::Hls(s) => s.send_frame(frame).await,
            Self::Mirror { session, .. } => session.send_frame(frame).await,
        }
    }

    async fn send_audio_frame(&mut self, frame: &AudioFrame) -> Result<()> {
        match self {
            Self::Hls(s) => s.send_audio_frame(frame).await,
            Self::Mirror { session, .. } => session.send_audio_frame(frame).await,
        }
    }

    async fn stop(&mut self) -> Result<()> {
        match self {
            Self::Hls(s) => s.stop().await,
            Self::Mirror { session, .. } => session.stop().await,
        }
    }

    fn is_alive(&self) -> bool {
        match self {
            Self::Hls(s) => s.is_alive(),
            Self::Mirror {
                session,
                connected_device,
                blocklist,
            } => {
                if session.failure() == Some(MirrorFailure::NoProgress) {
                    if let Some(device) = connected_device
                        && !blocklist.contains(&device.id)
                    {
                        tracing::warn!(
                            device = %device.name,
                            "receiver never acknowledged Cast Streaming frames — using HLS from now on"
                        );
                        blocklist.insert(device.id);
                    }
                    return false;
                }
                session.is_alive()
            }
        }
    }

    fn poll_keyframe_request(&mut self) -> bool {
        match self {
            Self::Hls(s) => s.poll_keyframe_request(),
            Self::Mirror { session, .. } => session.poll_keyframe_request(),
        }
    }

    fn preferred_audio_codec(&self) -> AudioCodec {
        match self {
            Self::Mirror { session, .. } => session.preferred_audio_codec(),
            Self::Hls(s) => s.preferred_audio_codec(),
        }
    }

    fn feeds_adaptive_bitrate(&self) -> bool {
        match self {
            Self::Mirror { session, .. } => session.feeds_adaptive_bitrate(),
            Self::Hls(s) => s.feeds_adaptive_bitrate(),
        }
    }
}

#[derive(Clone, Default)]
pub struct ChromecastHandler {
    mirror_blocklist: MirrorBlocklist,
}

impl ChromecastHandler {
    /// Cast Streaming sends Opus audio, so it is only offered when the
    /// Opus encoder is compiled in.
    const CAST_STREAMING_AVAILABLE: bool = cfg!(feature = "opus");
}

impl ProtocolHandler for ChromecastHandler {
    const PROTOCOL: &'static str = "chromecast";
    const SUPPORTED_CODECS: &'static [Codec] = &[Codec::H264];

    type Discovery = ChromecastDiscovery;
    type Session = ChromecastEitherSession;

    fn create_discovery(&self) -> ChromecastDiscovery {
        ChromecastDiscovery::default()
    }

    fn create_session(&self) -> Result<ChromecastEitherSession> {
        Ok(ChromecastEitherSession::Hls(ChromecastSession::default()))
    }

    fn create_session_for_device(&self, device: &Device) -> Result<ChromecastEitherSession> {
        let mirror = Self::CAST_STREAMING_AVAILABLE
            && device.capabilities.supports_cast_streaming
            && !self.mirror_blocklist.contains(&device.id);
        if mirror {
            tracing::info!(device = %device.name, "selecting Cast Streaming (RTP/UDP) session");
            Ok(ChromecastEitherSession::Mirror {
                session: Box::default(),
                connected_device: None,
                blocklist: self.mirror_blocklist.clone(),
            })
        } else {
            tracing::debug!(device = %device.name, "selecting HLS session");
            Ok(ChromecastEitherSession::Hls(ChromecastSession::default()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferricast_core::DeviceCapabilities;

    fn device(cast_streaming: bool) -> Device {
        Device {
            id: Uuid::new_v4(),
            name: "tv".into(),
            protocol: "chromecast",
            protocol_icon: bytes::Bytes::new(),
            addr: [192, 168, 1, 2].into(),
            port: 8009,
            model: None,
            capabilities: DeviceCapabilities {
                supports_cast_streaming: cast_streaming,
                ..DeviceCapabilities::default()
            },
            metadata: std::collections::HashMap::new(),
        }
    }

    fn is_mirror(s: &ChromecastEitherSession) -> bool {
        matches!(s, ChromecastEitherSession::Mirror { .. })
    }

    #[test]
    fn session_choice_honours_capability_and_blocklist() {
        let handler = ChromecastHandler::default();
        let capable = device(true);
        assert!(!is_mirror(
            &handler.create_session_for_device(&device(false)).unwrap()
        ));
        assert_eq!(
            is_mirror(&handler.create_session_for_device(&capable).unwrap()),
            ChromecastHandler::CAST_STREAMING_AVAILABLE
        );
        // Clones share the blocklist, as the reconnect path uses a clone.
        handler.clone().mirror_blocklist.insert(capable.id);
        assert!(!is_mirror(
            &handler.create_session_for_device(&capable).unwrap()
        ));
    }

    #[test]
    fn audio_codec_follows_the_active_variant() {
        let handler = ChromecastHandler::default();
        let hls = handler.create_session_for_device(&device(false)).unwrap();
        assert_eq!(hls.preferred_audio_codec(), AudioCodec::Aac);
        if ChromecastHandler::CAST_STREAMING_AVAILABLE {
            let mirror = handler.create_session_for_device(&device(true)).unwrap();
            assert_eq!(mirror.preferred_audio_codec(), AudioCodec::Opus);
        }
    }

    #[test]
    fn vp8_is_not_advertised() {
        assert_eq!(ChromecastHandler::SUPPORTED_CODECS, &[Codec::H264]);
    }
}
