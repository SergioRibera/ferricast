//! Receiver-side decode — turn [`EncodedFrame`] / [`AudioFrame`]
//! coming out of [`crate::puller::MediaPuller`] into something a
//! [`crate::sink::FrameSink`] can render.
//!
//! Mirror of [`crate::encoder`]. Video decoders produce
//! [`CapturedFrame`] (Cpu or Gpu) so a VA-API / NVDEC zero-copy path
//! can hand the GPU surface straight to a compatible sink without
//! readback, the same way the VA-API encoder consumes
//! `CapturedFrame::Gpu` on the sender side.

use bytes::Bytes;

use crate::error::Result;
use crate::frame::{AudioCodec, AudioFrame, CapturedFrame, EncodedFrame, PixelFormat};
use crate::protocol::Codec;

#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub codec: Codec,
    /// Width hint from the stream metadata. Decoder MAY override
    /// after parsing the SPS / sequence header; the value here is
    /// just for buffer pre-allocation.
    pub width: u32,
    pub height: u32,
    /// Preferred output pixel format. The decoder will produce this
    /// when it can (VA-API can output NV12 directly, NVDEC similar);
    /// software decoders typically force NV12 / I420 regardless.
    pub pixel_format: PixelFormat,
}

/// Decode an [`EncodedFrame`] stream to [`CapturedFrame`].
///
/// `decode` returns `Option<CapturedFrame>` because real decoders
/// don't emit one frame per input packet — H.264 with B-frames needs
/// to see future packets before it can emit the current frame in
/// presentation order. `None` = "I consumed the packet, but no
/// output yet; keep feeding". Drain remaining frames at end of
/// stream with `flush`.
pub trait VideoDecoder: Send {
    const CODEC: Codec;

    fn configure(&mut self, config: &DecoderConfig) -> Result<()>;
    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>>;
    fn flush(&mut self) -> Result<Vec<CapturedFrame>>;
}

/// Audio decoder config. Sample rate / channels come from the
/// stream's audio header; impls MAY validate against their supported
/// set in `configure`.
#[derive(Debug, Clone)]
pub struct AudioDecoderConfig {
    pub codec: AudioCodec,
    pub sample_rate: u32,
    pub channels: u16,
}

/// PCM output from an audio decoder. Always interleaved, 16-bit
/// signed little-endian for now — every audio path on Linux
/// (PipeWire, ALSA, PulseAudio) accepts that format without
/// conversion, so we don't bother carrying a sample format tag yet.
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    pub pcm: Bytes,
    pub sample_rate: u32,
    pub channels: u16,
    pub timestamp_us: u64,
}

pub trait AudioDecoder: Send {
    const CODEC: AudioCodec;

    fn configure(&mut self, config: &AudioDecoderConfig) -> Result<()>;
    fn decode(&mut self, frame: AudioFrame) -> Result<Option<DecodedAudio>>;
    fn flush(&mut self) -> Result<Vec<DecodedAudio>>;
}
