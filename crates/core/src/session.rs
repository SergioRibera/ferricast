use crate::device::Device;
use crate::error::Result;
use crate::frame::EncodedFrame;

#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub bitrate_kbps: u32,
    pub codec: crate::Codec,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            // 60 matches typical desktop refresh rates. Some
            // xdg-desktop-portal backends only complete the
            // EnumFormat negotiation when the default framerate the
            // sender advertises matches what the compositor wants to
            // produce — `default: 30, range: 0-1000` was reproducibly
            // failing with `Paused → Error("no more input formats")`
            // on Niri+pipewire-shm. The actual value the encoder
            // ends up using is overridden by `ScreenCapture::
            // get_framerate()` once the format settles.
            fps: 60,
            bitrate_kbps: 5000,
            codec: crate::Codec::H264,
        }
    }
}

pub trait CastSession: Send + Sync {
    fn connect(&mut self, device: &Device) -> impl Future<Output = Result<()>> + Send;
    fn setup_stream(&mut self, config: &StreamConfig) -> impl Future<Output = Result<()>> + Send;
    fn send_frame(&mut self, frame: &EncodedFrame) -> impl Future<Output = Result<()>> + Send;
    fn stop(&mut self) -> impl Future<Output = Result<()>> + Send;
    fn is_alive(&self) -> bool;
}
