use bytes::Bytes;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Bgra,
    Rgba,
    Nv12,
    I420,
}

#[derive(Debug, Clone)]
pub struct RawFrame {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: PixelFormat,
    pub data: Bytes,
    pub timestamp_us: u64,
}

#[derive(Debug, Clone)]
pub struct EncodedFrame {
    pub codec: crate::Codec,
    pub data: Bytes,
    pub timestamp_us: u64,
    pub is_keyframe: bool,
    pub duration_us: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub codec: AudioCodec,
    pub data: Bytes,
    pub timestamp_us: u64,
    pub sample_rate: u32,
    pub channels: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodec {
    Aac,
    Opus,
    Pcm,
    Alac,
}
