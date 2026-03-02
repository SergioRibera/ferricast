use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tracing::{info, warn};
use uuid::Uuid;

use ferricast_core::{
    CastSession, Codec, Device, EncodedFrame, FerricastError, Result, StreamConfig,
};

/// Internal state of the AirPlay session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionState {
    /// Initial state, not connected.
    Disconnected,
    /// TCP connection established but not yet paired/negotiated.
    Connected,
    /// Pair-Verify completed, encryption keys established.
    Paired,
    /// RTSP SETUP completed, data channel ready.
    Ready,
    /// RTSP RECORD sent, actively streaming.
    Streaming,
    /// Session is being torn down.
    TearingDown,
}

impl std::fmt::Display for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disconnected => write!(f, "Disconnected"),
            Self::Connected => write!(f, "Connected"),
            Self::Paired => write!(f, "Paired"),
            Self::Ready => write!(f, "Ready"),
            Self::Streaming => write!(f, "Streaming"),
            Self::TearingDown => write!(f, "TearingDown"),
        }
    }
}

/// An AirPlay 2 screen mirroring session.
pub struct AirPlaySession {
    state: SessionState,
    session_id: String,
    client_device_id: String,
    alive: Arc<AtomicBool>,
    frame_counter: u64,
}

impl Default for AirPlaySession {
    fn default() -> Self {
        let session_id = Uuid::new_v4().to_string();
        let client_device_id = generate_device_id();
        Self {
            state: SessionState::Disconnected,
            session_id,
            client_device_id,
            alive: Default::default(),
            frame_counter: Default::default(),
        }
    }
}

impl AirPlaySession {
    /// Get the session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

impl CastSession for AirPlaySession {
    async fn connect(&mut self, device: &Device) -> Result<()> {
        if device.protocol != "airplay" {
            return Err(FerricastError::Protocol(format!(
                "Expected AirPlay device, got {:?}",
                device.protocol
            )));
        }

        if self.state != SessionState::Disconnected {
            return Err(FerricastError::SessionAlreadyActive(device.name.clone()));
        }

        Ok(())
    }

    async fn setup_stream(&mut self, config: &StreamConfig) -> Result<()> {
        if self.state != SessionState::Connected && self.state != SessionState::Paired {
            return Err(FerricastError::Protocol(format!(
                "Cannot setup stream in state: {}",
                self.state
            )));
        }

        // Validate codec
        if config.codec != Codec::H264 {
            return Err(FerricastError::UnsupportedCodec {
                codec: config.codec,
                protocol: "airplay",
            });
        }

        info!(
            width = config.width,
            height = config.height,
            fps = config.fps,
            bitrate_kbps = config.bitrate_kbps,
            "Setting up AirPlay stream"
        );

        self.state = SessionState::Streaming;
        info!("AirPlay stream is now active");

        Ok(())
    }

    async fn send_frame(&mut self, frame: &EncodedFrame) -> Result<()> {
        if self.state != SessionState::Streaming {
            return Err(FerricastError::Streaming(format!(
                "Cannot send frame in state: {}",
                self.state
            )));
        }

        if frame.codec != Codec::H264 {
            return Err(FerricastError::UnsupportedCodec {
                codec: frame.codec,
                protocol: "airplay",
            });
        }

        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        if self.state == SessionState::Disconnected {
            return Ok(());
        }

        info!(
            session_id = %self.session_id,
            frames_sent = self.frame_counter,
            "Stopping AirPlay session"
        );

        self.state = SessionState::TearingDown;

        info!("AirPlay session stopped");
        Ok(())
    }

    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
            && matches!(
                self.state,
                SessionState::Connected
                    | SessionState::Paired
                    | SessionState::Ready
                    | SessionState::Streaming
            )
    }
}

impl Drop for AirPlaySession {
    fn drop(&mut self) {
        if self.state != SessionState::Disconnected {
            warn!(
                session_id = %self.session_id,
                state = %self.state,
                "AirPlaySession dropped while still active"
            );
            self.alive.store(false, Ordering::Relaxed);
        }
    }
}

/// Generate a random device ID in MAC address format.
fn generate_device_id() -> String {
    let bytes: [u8; 6] = rand::random();
    format!(
        "{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5]
    )
}
