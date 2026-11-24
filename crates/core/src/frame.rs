use std::sync::Arc;

use bytes::Bytes;

use crate::colour::{ColourDescription, HdrMetadata};

#[cfg(any(target_os = "linux", target_os = "android"))]
mod dmabuf;
mod native;

#[cfg(any(target_os = "linux", target_os = "android"))]
pub use dmabuf::{DmaBufFd, DmaBufId, DmaBufImporter, DmaBufPlane, GpuFrame};
pub use native::{NativeFrame, NativeSurface, SurfaceKind, SurfacePixels};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub enum PixelFormat {
    Bgra,
    Rgba,
    Nv12,
    I420,
    /// 10-bit 4:2:0, semi-planar like NV12: little-endian 16-bit samples
    /// with the value in the high 10 bits (what `VideoToolbox`, Media
    /// Foundation, `MediaCodec` and VA-API produce for Main10 / AV1 10-bit).
    P010,
}

impl PixelFormat {
    /// Bytes per row of a tightly packed (no padding) image `width`
    /// pixels wide. For the planar YUV formats this is the row size
    /// of the luma plane, which is what `RawFrame::stride` describes.
    #[must_use]
    pub const fn packed_row_bytes(self, width: u32) -> u32 {
        match self {
            PixelFormat::Bgra | PixelFormat::Rgba => width.saturating_mul(4),
            PixelFormat::Nv12 | PixelFormat::I420 => width,
            PixelFormat::P010 => width.saturating_mul(2),
        }
    }

    /// Bits per sample.
    #[must_use]
    pub const fn bit_depth(self) -> BitDepth {
        match self {
            PixelFormat::Bgra | PixelFormat::Rgba | PixelFormat::Nv12 | PixelFormat::I420 => {
                BitDepth::Eight
            }
            PixelFormat::P010 => BitDepth::Ten,
        }
    }
}

/// Bits per sample of a picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub enum BitDepth {
    #[default]
    Eight,
    Ten,
}

impl BitDepth {
    #[must_use]
    pub const fn bits(self) -> u8 {
        match self {
            Self::Eight => 8,
            Self::Ten => 10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RawFrame {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: PixelFormat,
    pub data: Bytes,
    pub timestamp_us: u64,
    /// The colour signalling of a YCbCr picture, when the
    /// producer knows; `None` for RGB formats or an undescribed picture.
    pub colour: Option<ColourDescription>,
    /// The static HDR metadata the picture carries, when it has any.
    pub hdr: Option<HdrMetadata>,
}

/// Holds a producer's buffer out of its pool while a consumer may
/// still read it.
///
/// The producer wraps a guard whose `Drop` hands the buffer back (for
/// PipeWire: re-queues it to the compositor). Every clone of a frame
/// shares one lease, so the buffer goes back exactly once, when the
/// last clone is gone. A consumer whose GPU work keeps reading the
/// DMA-BUF after it is done with the frame (asynchronous encode) must
/// keep a clone alive until that work has completed.
#[derive(Clone)]
pub struct BufferLease(Arc<dyn Send + Sync>);

impl BufferLease {
    /// Wrap `guard`; its `Drop` runs when the last clone is dropped.
    #[must_use]
    pub fn new<G: Send + Sync + 'static>(guard: G) -> Self {
        Self(Arc::new(guard))
    }
}

impl std::fmt::Debug for BufferLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferLease")
            .field("holders", &Arc::strong_count(&self.0))
            .finish()
    }
}

/// Captured frame in whichever shape the source produced it. The
/// consumer (encoder) decides what to do with it: x264 always reads
/// CPU bytes (calling [`CapturedFrame::into_cpu`] forces a readback
/// when needed); VA-API and NVENC consume the `Gpu` variant directly.
///
/// `Gpu` frames are DMA-BUFs, so the variant only exists on Linux and
/// Android. `Native` frames are platform media surfaces
/// (`CVPixelBuffer`, `AHardwareBuffer`, D3D11 texture); see [`NativeFrame`].
#[derive(Debug, Clone)]
pub enum CapturedFrame {
    Cpu(RawFrame),
    #[cfg(any(target_os = "linux", target_os = "android"))]
    Gpu(GpuFrame),
    Native(NativeFrame),
}

impl CapturedFrame {
    #[must_use]
    pub fn width(&self) -> u32 {
        match self {
            CapturedFrame::Cpu(r) => r.width,
            #[cfg(any(target_os = "linux", target_os = "android"))]
            CapturedFrame::Gpu(g) => g.width,
            CapturedFrame::Native(n) => n.width,
        }
    }

    #[must_use]
    pub fn height(&self) -> u32 {
        match self {
            CapturedFrame::Cpu(r) => r.height,
            #[cfg(any(target_os = "linux", target_os = "android"))]
            CapturedFrame::Gpu(g) => g.height,
            CapturedFrame::Native(n) => n.height,
        }
    }

    #[must_use]
    pub fn timestamp_us(&self) -> u64 {
        match self {
            CapturedFrame::Cpu(r) => r.timestamp_us,
            #[cfg(any(target_os = "linux", target_os = "android"))]
            CapturedFrame::Gpu(g) => g.timestamp_us,
            CapturedFrame::Native(n) => n.timestamp_us,
        }
    }

    /// Re-stamps the picture, keeping it in whichever memory it is in.
    pub fn set_timestamp_us(&mut self, timestamp_us: u64) {
        match self {
            CapturedFrame::Cpu(r) => r.timestamp_us = timestamp_us,
            #[cfg(any(target_os = "linux", target_os = "android"))]
            CapturedFrame::Gpu(g) => g.timestamp_us = timestamp_us,
            CapturedFrame::Native(n) => n.timestamp_us = timestamp_us,
        }
    }

    #[must_use]
    pub fn pixel_format(&self) -> PixelFormat {
        match self {
            CapturedFrame::Cpu(r) => r.format,
            #[cfg(any(target_os = "linux", target_os = "android"))]
            CapturedFrame::Gpu(g) => g.format,
            CapturedFrame::Native(n) => n.format(),
        }
    }

    /// Force the frame into a CPU-resident `RawFrame`. No-op for
    /// `Cpu`; performs a readback through the carried importer for
    /// `Gpu` (see [`GpuFrame::into_cpu`]) and `Native` ([`NativeFrame::into_cpu`]).
    ///
    /// # Errors
    /// [`FrameError::NoImporter`] when a GPU frame carries no importer,
    /// or the importer's readback failure.
    pub fn into_cpu(self) -> Result<RawFrame, FrameError> {
        match self {
            CapturedFrame::Cpu(r) => Ok(r),
            #[cfg(any(target_os = "linux", target_os = "android"))]
            CapturedFrame::Gpu(g) => g.into_cpu(),
            CapturedFrame::Native(n) => n.into_cpu(),
        }
    }
}

impl From<RawFrame> for CapturedFrame {
    fn from(r: RawFrame) -> Self {
        CapturedFrame::Cpu(r)
    }
}

impl From<NativeFrame> for CapturedFrame {
    fn from(n: NativeFrame) -> Self {
        CapturedFrame::Native(n)
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
impl From<GpuFrame> for CapturedFrame {
    fn from(g: GpuFrame) -> Self {
        CapturedFrame::Gpu(g)
    }
}

/// Why a frame could not be turned into CPU-resident pixels.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FrameError {
    /// The GPU frame came from a path without a readback importer.
    #[error("the GPU frame has no importer attached, so it cannot be read back to CPU memory")]
    NoImporter,
    /// The importer failed to copy the GPU buffer to CPU memory.
    #[error("reading the GPU frame back to CPU memory failed: {reason}")]
    Readback { reason: String },
}

/// One compressed video access unit.
#[derive(Debug, Clone)]
pub struct EncodedFrame {
    pub codec: crate::Codec,
    /// Annex B for H.264 / H.265, with the parameter sets in front of
    /// every keyframe when an encoder produced it.
    pub data: Bytes,
    /// Capture (presentation) time of the picture, wall-clock microseconds.
    pub timestamp_us: u64,
    pub is_keyframe: bool,
    pub duration_us: Option<u64>,
    /// Presentation and decode order of the picture in frame-index
    /// units (0, 1, 2, ... in capture order), as an encoder emits them;
    /// equal when the stream has no B-frames.
    pub pts_dts: (u64, u64),
}

#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub codec: AudioCodec,
    pub data: Bytes,
    pub timestamp_us: u64,
    pub sample_rate: u32,
    pub channels: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub enum AudioCodec {
    /// AAC-LC, ADTS-framed.
    Aac,
    /// AAC-ELD (enhanced low delay), raw access units of 480 samples at
    /// 44.1 kHz: the audio format of AirPlay screen mirroring.
    AacEld,
    Opus,
    Pcm,
    Alac,
}

/// The format an audio stream is stored in when the receiver decodes it
/// to PCM before its pipeline (its packets then say
/// [`AudioCodec::Pcm`]): what to show the user instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceAudioFormat {
    Mp3,
    Flac,
    Vorbis,
    /// Uncompressed PCM in a WAV file.
    Wav,
}

impl SourceAudioFormat {
    /// A short human-readable name.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Mp3 => "MP3",
            Self::Flac => "FLAC",
            Self::Vorbis => "Vorbis",
            Self::Wav => "WAV (PCM)",
        }
    }
}

/// An AAC `channel_configuration` (ISO/IEC 14496-3 §1.6.3.5): the
/// 3-bit layout code ADTS headers and `AudioSpecificConfig`s carry.
/// Codes 1 to 6 name as many channels (6 is 5.1); 7 is 7.1, eight
/// channels. Code 0 (layout in a program config element) and the
/// reserved codes are not represented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AacChannelConfig(u8);

impl AacChannelConfig {
    /// The configuration with code `code`; `None` for 0 and reserved codes.
    #[must_use]
    pub const fn new(code: u8) -> Option<Self> {
        match code {
            1..=7 => Some(Self(code)),
            _ => None,
        }
    }

    /// The configuration that carries `channels` channels.
    #[must_use]
    pub const fn from_channels(channels: u16) -> Option<Self> {
        match channels {
            1..=6 => Some(Self(channels as u8)),
            8 => Some(Self(7)),
            _ => None,
        }
    }

    /// The 3-bit code.
    #[must_use]
    pub const fn code(self) -> u8 {
        self.0
    }

    /// Channels decoded from this configuration.
    #[must_use]
    pub const fn channels(self) -> u16 {
        match self.0 {
            7 => 8,
            code => code as u16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_row_bytes_per_format() {
        assert_eq!(PixelFormat::Bgra.packed_row_bytes(10), 40);
        assert_eq!(PixelFormat::Rgba.packed_row_bytes(10), 40);
        assert_eq!(PixelFormat::Nv12.packed_row_bytes(10), 10);
        assert_eq!(PixelFormat::I420.packed_row_bytes(10), 10);
        assert_eq!(PixelFormat::Bgra.packed_row_bytes(u32::MAX), u32::MAX);
    }
}
