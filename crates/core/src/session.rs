use std::sync::Arc;

use crate::adaptive::AdaptiveBitrateState;
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

    /// Optional adaptive bitrate controller. When provided, the
    /// receiver protocol (e.g. Chromecast) plumbs it into its HLS
    /// server, and the stream manager polls it on the hot path to
    /// live-reconfigure the encoder when the receiver's link is
    /// under sustained pressure. `None` (default) keeps the
    /// pre-adaptive behaviour: fixed bitrate from configure, no
    /// runtime feedback.
    pub adaptive: Option<Arc<AdaptiveBitrateState>>,
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
            adaptive: None,
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
