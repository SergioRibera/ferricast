use crate::error::Result;
use crate::frame::{EncodedFrame, RawFrame};
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
            fps: 30,
            keyframe_interval: 60,
            pixel_format: PixelFormat::Bgra,
        }
    }
}

pub trait VideoEncoder: Send {
    const CODEC: Codec;

    fn configure(&mut self, config: &EncoderConfig) -> Result<()>;
    fn encode(&mut self, frame: &RawFrame) -> Result<EncodedFrame>;
    fn flush(self) -> Result<Vec<EncodedFrame>>;
    fn get_headers(&mut self) -> Result<Vec<u8>>;
}
