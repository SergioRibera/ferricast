use ferricast_core::{CastSession, Device, EncodedFrame, Result, StreamConfig};

/// A Miracast (Wi-Fi Display) streaming session.
#[derive(Default)]
pub struct MiracastSession;

impl CastSession for MiracastSession {
    async fn connect(&mut self, device: &Device) -> Result<()> {
        Ok(())
    }

    async fn setup_stream(&mut self, config: &StreamConfig) -> Result<()> {
        Ok(())
    }

    async fn send_frame(&mut self, frame: &EncodedFrame) -> Result<()> {
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        Ok(())
    }

    fn is_alive(&self) -> bool {
        false
    }
}
