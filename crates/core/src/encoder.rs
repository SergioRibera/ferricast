use crate::error::Result;
use crate::frame::{CapturedFrame, EncodedFrame};
use crate::{Codec, PixelFormat};

#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub codec: Codec,
    pub width: u32,
    pub height: u32,
    pub bitrate_kbps: u32,
    pub fps: u32,
    pub keyframe_interval: u32,
    pub pixel_format: PixelFormat
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            codec: Codec::H264,
            width: 1920,
            height: 1080,
            bitrate_kbps: 5000,
            fps: 60,
            keyframe_interval: 60,
            pixel_format: PixelFormat::Bgra,
        }
    }
}

pub trait VideoEncoder: Send {
    const CODEC: Codec;

    fn configure(&mut self, config: &EncoderConfig) -> Result<()>;

    /// Encode the given captured frame.
    ///
    /// Implementations decide what to do with each variant:
    /// * x264 — always reads CPU bytes, calling
    ///   [`CapturedFrame::into_cpu`] to trigger a Vulkan readback if
    ///   the source produced a `Gpu` frame.
    /// * VA-API — consumes the `Gpu` variant directly (zero-copy)
    ///   and uploads to a `VASurface` for the `Cpu` variant.
    fn encode(&mut self, frame: CapturedFrame) -> Result<EncodedFrame>;

    fn flush(self) -> Result<Vec<EncodedFrame>>;
    fn get_headers(&mut self) -> Result<Vec<u8>>;
}
