pub mod castv2;
pub mod discovery;
pub mod mirroring;
pub mod receiver;
pub(crate) mod self_filter;
pub mod session;
mod wire;

pub use castv2::{
    CastMessage, CastV2Error, GenericPayload, PayloadType, ProtocolVersion,
    namespace as cast_namespace,
};
pub use discovery::ChromecastDiscovery;
pub use mirroring::ChromecastMirrorSession;
pub use receiver::{ChromecastReceiver, ChromecastReceiverAdvertiser, ChromecastReceiverControl};
pub use session::ChromecastSession;

use ferricast_core::{
    AudioFrame, CastSession, Codec, Device, EncodedFrame, FerricastError, ProtocolHandler, Result,
    StreamConfig,
};

/// Wraps either the HLS session or the Cast Streaming (RTP/UDP) mirror
/// session. The manager stores `Box<dyn ErasedSession>` so both variants
/// erase identically; this enum is the single `Session` associated type
/// on [`ChromecastHandler`] that makes that work without touching the
/// `ProtocolHandler` trait.
///
/// The `Mirror` variant stores the connected `Device` so that
/// `setup_stream` can fall back to HLS transparently when the receiver
/// doesn't respond with a Cast Streaming ANSWER within 10 s — old
/// firmware, Chromecast Audio, and some third-party builds reject the
/// mirroring namespace silently.
pub enum ChromecastEitherSession {
    Hls(ChromecastSession),
    Mirror {
        session: ChromecastMirrorSession,
        /// Captured during `connect` so `setup_stream` can re-connect
        /// to the same device if it needs to fall back to HLS.
        connected_device: Option<Device>,
    },
}

impl CastSession for ChromecastEitherSession {
    async fn connect(&mut self, device: &Device) -> Result<()> {
        match self {
            Self::Hls(s) => s.connect(device).await,
            Self::Mirror { session, connected_device } => {
                let r = session.connect(device).await;
                if r.is_ok() {
                    *connected_device = Some(device.clone());
                }
                r
            }
        }
    }

    async fn setup_stream(&mut self, config: &StreamConfig) -> Result<()> {
        // Mirror path: attempt Cast Streaming OFFER/ANSWER, fall back to
        // HLS on timeout (device doesn't support mirroring namespace).
        let fallback_device: Option<Device> = match self {
            Self::Hls(s) => return s.setup_stream(config).await,
            Self::Mirror { session, connected_device } => {
                match session.setup_stream(config).await {
                    Ok(()) => return Ok(()),
                    Err(FerricastError::Timeout(ref msg)) => {
                        tracing::warn!(
                            %msg,
                            "Cast Streaming ANSWER timed out — falling back to HLS"
                        );
                        let dev = connected_device.take();
                        let _ = session.stop().await;
                        dev
                    }
                    Err(e) => return Err(e),
                }
            }
        };
        // Borrow of `self` ended — replace with a connected HLS session.
        let mut hls = ChromecastSession::default();
        if let Some(dev) = fallback_device {
            hls.connect(&dev).await?;
            hls.setup_stream(config).await?;
        }
        *self = Self::Hls(hls);
        Ok(())
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
            Self::Mirror { session, .. } => session.is_alive(),
        }
    }

    fn poll_keyframe_request(&mut self) -> bool {
        match self {
            Self::Hls(s) => s.poll_keyframe_request(),
            Self::Mirror { session, .. } => session.poll_keyframe_request(),
        }
    }
}

#[derive(Clone, Default)]
pub struct ChromecastHandler;

impl ProtocolHandler for ChromecastHandler {
    const PROTOCOL: &'static str = "chromecast";
    const SUPPORTED_CODECS: &'static [Codec] = &[Codec::H264, Codec::Vp8];

    type Discovery = ChromecastDiscovery;
    type Session = ChromecastEitherSession;

    fn create_discovery(&self) -> ChromecastDiscovery {
        ChromecastDiscovery::default()
    }

    fn create_session(&self) -> Result<ChromecastEitherSession> {
        Ok(ChromecastEitherSession::Hls(ChromecastSession::default()))
    }

    fn create_session_for_device(&self, device: &Device) -> Result<ChromecastEitherSession> {
        if device.capabilities.supports_cast_streaming {
            tracing::info!(
                device = %device.name,
                "selecting Cast Streaming (RTP/UDP) session"
            );
            Ok(ChromecastEitherSession::Mirror {
                session: ChromecastMirrorSession::default(),
                connected_device: None,
            })
        } else {
            tracing::debug!(
                device = %device.name,
                "selecting HLS session (Cast Streaming not supported)"
            );
            Ok(ChromecastEitherSession::Hls(ChromecastSession::default()))
        }
    }
}
