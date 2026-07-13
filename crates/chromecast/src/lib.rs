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
    AudioFrame, CastSession, Codec, Device, EncodedFrame, ProtocolHandler, Result, StreamConfig,
};

/// Wraps either the HLS session or the Cast Streaming (RTP/UDP) mirror
/// session. The manager stores `Box<dyn ErasedSession>` so both variants
/// erase identically; this enum is the single `Session` associated type
/// on [`ChromecastHandler`] that makes that work without touching the
/// `ProtocolHandler` trait.
pub enum ChromecastEitherSession {
    Hls(ChromecastSession),
    Mirror(ChromecastMirrorSession),
}

impl CastSession for ChromecastEitherSession {
    async fn connect(&mut self, device: &Device) -> Result<()> {
        match self {
            Self::Hls(s) => s.connect(device).await,
            Self::Mirror(s) => s.connect(device).await,
        }
    }

    async fn setup_stream(&mut self, config: &StreamConfig) -> Result<()> {
        match self {
            Self::Hls(s) => s.setup_stream(config).await,
            Self::Mirror(s) => s.setup_stream(config).await,
        }
    }

    async fn send_frame(&mut self, frame: &EncodedFrame) -> Result<()> {
        match self {
            Self::Hls(s) => s.send_frame(frame).await,
            Self::Mirror(s) => s.send_frame(frame).await,
        }
    }

    async fn send_audio_frame(&mut self, frame: &AudioFrame) -> Result<()> {
        match self {
            Self::Hls(s) => s.send_audio_frame(frame).await,
            Self::Mirror(s) => s.send_audio_frame(frame).await,
        }
    }

    async fn stop(&mut self) -> Result<()> {
        match self {
            Self::Hls(s) => s.stop().await,
            Self::Mirror(s) => s.stop().await,
        }
    }

    fn is_alive(&self) -> bool {
        match self {
            Self::Hls(s) => s.is_alive(),
            Self::Mirror(s) => s.is_alive(),
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
            Ok(ChromecastEitherSession::Mirror(
                ChromecastMirrorSession::default(),
            ))
        } else {
            tracing::debug!(
                device = %device.name,
                "selecting HLS session (Cast Streaming not supported)"
            );
            Ok(ChromecastEitherSession::Hls(ChromecastSession::default()))
        }
    }
}

