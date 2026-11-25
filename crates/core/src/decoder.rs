//! Receiver-side decode — turn [`EncodedFrame`] / [`AudioFrame`]
//! coming out of [`crate::puller::MediaPuller`] into something a
//! [`crate::sink::FrameSink`] can render.
//!
//! Mirror of [`crate::encoder`]. Video decoders produce
//! [`CapturedFrame`] (Cpu or Gpu) so a VA-API / NVDEC zero-copy path
//! can hand the GPU surface straight to a compatible sink without
//! readback, the same way the VA-API encoder consumes
//! `CapturedFrame::Gpu` on the sender side.

use crate::codec::Codec;
use crate::codec::{BackendSelection, CodecResult};
use crate::colour::{ColourDescription, HdrMetadata};
use crate::frame::{AudioCodec, AudioFrame, CapturedFrame, EncodedFrame, PixelFormat};
use crate::gpu::GpuPreference;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
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
    pub low_latency: bool,
    /// The colour signalling of the container, used for pictures whose
    /// bitstream carries none (or whose backend cannot read it).
    pub colour: Option<ColourDescription>,
    /// The container's static HDR metadata, for pictures whose bitstream
    /// carries none.
    pub hdr: Option<HdrMetadata>,
    /// Where decoded pictures are handed out; see [`FrameMemory`].
    pub memory: FrameMemory,
    /// Which backend decodes; see [`BackendSelection`].
    pub backend: BackendSelection,
    /// Which GPU the hardware backends run on; see [`GpuPreference`]
    /// (decoders have no frame producer: `Auto` prefers discrete GPUs).
    pub gpu: GpuPreference,
}

/// Where a video decoder hands out its pictures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub enum FrameMemory {
    /// CPU memory ([`CapturedFrame::Cpu`]): every sink can take it.
    #[default]
    Cpu,
    /// The backend's own media surface when it has one
    /// ([`CapturedFrame::Native`]: `CVPixelBuffer`, `AHardwareBuffer`,
    /// D3D11 texture; [`CapturedFrame::Gpu`] for VA-API), skipping the copy
    /// to CPU memory; CPU memory from backends without one.
    Native,
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

    /// Opens the decoder for `config`.
    ///
    /// # Errors
    /// A [`crate::CodecError`] naming why no backend can decode the stream.
    fn configure(&mut self, config: &DecoderConfig) -> CodecResult<()>;

    /// Decodes one access unit; `None` while no picture is ready.
    ///
    /// # Errors
    /// A [`crate::CodecError`] for a packet the backend cannot decode.
    fn decode(&mut self, frame: EncodedFrame) -> CodecResult<Option<CapturedFrame>>;

    /// Drains the pictures still held at the end of the stream.
    ///
    /// # Errors
    /// A [`crate::CodecError`] when the backend fails to drain.
    fn flush(&mut self) -> CodecResult<Vec<CapturedFrame>>;
}

/// Audio decoder config. Sample rate / channels come from the
/// stream's audio header; impls MAY validate against their supported
/// set in `configure`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub struct AudioDecoderConfig {
    pub codec: AudioCodec,
    pub sample_rate: u32,
    pub channels: u16,
    /// Out-of-band codec configuration from the container or session:
    /// the `AudioSpecificConfig` for AAC / AAC-ELD, the `ALACSpecificConfig`
    /// (magic cookie) for ALAC. `None` derives it from the other fields
    /// (ADTS headers for AAC, the AirPlay defaults for AAC-ELD and ALAC).
    pub extra_data: Option<Vec<u8>>,
    /// Which backend decodes; see [`BackendSelection`].
    pub backend: BackendSelection,
}

/// PCM output from an audio decoder. Always interleaved, 16-bit
/// signed native-endian samples — every audio path on Linux
/// (PipeWire, ALSA, PulseAudio) accepts that format without
/// conversion, so we don't carry a sample format tag yet.
///
/// `pcm` is a `Vec<i16>` (not `Bytes` of LE-encoded bytes) so the
/// sink can hand it straight to `rodio::buffer::SamplesBuffer::new`
/// without a per-sample byte-unpack pass. Keeping the natural shape
/// here also lets a future zero-copy SIMD f32→i16 path land without
/// churning the trait surface again.
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    pub pcm: Vec<i16>,
    pub sample_rate: u32,
    pub channels: u16,
    pub timestamp_us: u64,
}

pub trait AudioDecoder: Send {
    const CODEC: AudioCodec;

    /// Opens the decoder for `config`.
    ///
    /// # Errors
    /// A [`crate::CodecError`] naming why no backend can decode the stream.
    fn configure(&mut self, config: &AudioDecoderConfig) -> CodecResult<()>;

    /// Decodes one packet; `None` while no PCM is ready.
    ///
    /// # Errors
    /// A [`crate::CodecError`] for a packet the backend cannot decode.
    fn decode(&mut self, frame: AudioFrame) -> CodecResult<Option<DecodedAudio>>;

    /// Drains the PCM still held at the end of the stream.
    ///
    /// # Errors
    /// A [`crate::CodecError`] when the backend fails to drain.
    fn flush(&mut self) -> CodecResult<Vec<DecodedAudio>>;
}
