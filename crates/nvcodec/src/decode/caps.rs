//! What the GPU's NVDEC can decode (`cuvidGetDecoderCaps`).

use std::fmt;
use std::sync::Arc;

use super::format::PixelFormat;
use super::{CuvidCheck as _, cuvid_library};
use crate::cuda::Context;
use crate::error::{Error, Result};
use crate::sys::cuvid as sys;

/// A codec NVDEC decodes through this crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Codec {
    /// H.264 / AVC.
    H264,
    /// H.265 / HEVC.
    Hevc,
    /// VP8.
    Vp8,
    /// VP9.
    Vp9,
    /// AV1.
    Av1,
}

impl Codec {
    pub(super) fn raw(self) -> sys::cudaVideoCodec {
        match self {
            Self::H264 => sys::cudaVideoCodec::cudaVideoCodec_H264,
            Self::Hevc => sys::cudaVideoCodec::cudaVideoCodec_HEVC,
            Self::Vp8 => sys::cudaVideoCodec::cudaVideoCodec_VP8,
            Self::Vp9 => sys::cudaVideoCodec::cudaVideoCodec_VP9,
            Self::Av1 => sys::cudaVideoCodec::cudaVideoCodec_AV1,
        }
    }

    /// The codec name, for messages.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::H264 => "H.264",
            Self::Hevc => "HEVC",
            Self::Vp8 => "VP8",
            Self::Vp9 => "VP9",
            Self::Av1 => "AV1",
        }
    }
}

impl fmt::Display for Codec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Chroma subsampling of a coded stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChromaFormat {
    /// Luma only. NVDEC outputs it as 4:2:0 with neutral chroma.
    Monochrome,
    /// 4:2:0.
    Yuv420,
    /// 4:2:2.
    Yuv422,
    /// 4:4:4.
    Yuv444,
}

impl ChromaFormat {
    pub(super) fn raw(self) -> sys::cudaVideoChromaFormat {
        match self {
            Self::Monochrome => sys::cudaVideoChromaFormat::cudaVideoChromaFormat_Monochrome,
            Self::Yuv420 => sys::cudaVideoChromaFormat::cudaVideoChromaFormat_420,
            Self::Yuv422 => sys::cudaVideoChromaFormat::cudaVideoChromaFormat_422,
            Self::Yuv444 => sys::cudaVideoChromaFormat::cudaVideoChromaFormat_444,
        }
    }

    /// `None` for a value the header does not define.
    pub(super) fn from_raw(raw: sys::cudaVideoChromaFormat) -> Option<Self> {
        [Self::Monochrome, Self::Yuv420, Self::Yuv422, Self::Yuv444]
            .into_iter()
            .find(|format| format.raw() == raw)
    }
}

/// The output surface formats NVDEC can write for one stream format, from
/// `CUVIDDECODECAPS::nOutputFormatMask`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct OutputFormats(u16);

impl OutputFormats {
    /// Whether `format` is among them.
    #[must_use]
    pub fn contains(self, format: PixelFormat) -> bool {
        let bit = format.surface_format().0;
        bit < u16::BITS && self.0 & (1 << bit) != 0
    }

    /// The raw mask: bit `n` set for `cudaVideoSurfaceFormat` value `n`.
    #[must_use]
    pub const fn bits(self) -> u16 {
        self.0
    }
}

/// NVDEC's limits for one codec, chroma format and bit depth on one GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DecoderCaps {
    /// The codec.
    pub codec: Codec,
    /// The chroma format.
    pub chroma: ChromaFormat,
    /// Bits per sample.
    pub bit_depth: u8,
    /// Whether the GPU decodes this combination at all. Every other field
    /// is meaningless when it does not.
    pub supported: bool,
    /// How many NVDEC engines can decode it.
    pub engines: u8,
    /// The surface formats it can output.
    pub output_formats: OutputFormats,
    /// Smallest coded width, in pixels.
    pub min_width: u32,
    /// Smallest coded height, in pixels.
    pub min_height: u32,
    /// Largest coded width, in pixels.
    pub max_width: u32,
    /// Largest coded height, in pixels.
    pub max_height: u32,
    /// Largest coded size in 16x16 macroblocks (`width * height / 256`).
    pub max_macroblocks: u32,
}

impl DecoderCaps {
    /// Asks the driver what the GPU of `context` can decode.
    ///
    /// # Errors
    ///
    /// Loading errors, [`Error::Cuda`] when the context cannot be made
    /// current, [`Error::Cuvid`] when the query fails.
    pub fn query(
        context: &Arc<Context>,
        codec: Codec,
        chroma: ChromaFormat,
        bit_depth: u8,
    ) -> Result<Self> {
        let cuvid = cuvid_library()?;
        let mut raw = sys::CUVIDDECODECAPS {
            eCodecType: codec.raw(),
            eChromaFormat: chroma.raw(),
            nBitDepthMinus8: u32::from(bit_depth.saturating_sub(8)),
            ..Default::default()
        };
        let _current = context.guard()?;
        // SAFETY: the context is current and `raw` is a valid in/out struct
        // with its inputs set and reserved fields zeroed.
        unsafe { (cuvid.get_decoder_caps)(&raw mut raw) }.check_cuvid("cuvidGetDecoderCaps")?;
        Ok(Self {
            codec,
            chroma,
            bit_depth,
            supported: raw.bIsSupported != 0,
            engines: raw.nNumNVDECs,
            output_formats: OutputFormats(raw.nOutputFormatMask),
            min_width: u32::from(raw.nMinWidth),
            min_height: u32::from(raw.nMinHeight),
            max_width: raw.nMaxWidth,
            max_height: raw.nMaxHeight,
            max_macroblocks: raw.nMaxMBCount,
        })
    }

    /// Checks that a `width` x `height` coded stream fits, and that NVDEC
    /// can write `format` for it.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] naming the limit that is exceeded.
    pub fn check(&self, width: u32, height: u32, format: PixelFormat) -> Result<()> {
        let what = "this NVDEC stream format";
        let stream = format!(
            "{} {}-bit {:?} {width}x{height}",
            self.codec, self.bit_depth, self.chroma
        );
        if !self.supported {
            return Err(Error::Unsupported {
                what,
                reason: format!("the GPU does not decode {stream}"),
            });
        }
        if width < self.min_width || height < self.min_height {
            return Err(Error::Unsupported {
                what,
                reason: format!(
                    "{stream} is smaller than the GPU's minimum {}x{}",
                    self.min_width, self.min_height
                ),
            });
        }
        if width > self.max_width || height > self.max_height {
            return Err(Error::Unsupported {
                what,
                reason: format!(
                    "{stream} is larger than the GPU's maximum {}x{}",
                    self.max_width, self.max_height
                ),
            });
        }
        let macroblocks = u64::from(width.div_ceil(16)) * u64::from(height.div_ceil(16));
        if macroblocks > u64::from(self.max_macroblocks) {
            return Err(Error::Unsupported {
                what,
                reason: format!(
                    "{stream} has {macroblocks} macroblocks, the GPU's maximum is {}",
                    self.max_macroblocks
                ),
            });
        }
        if !self.output_formats.contains(format) {
            return Err(Error::Unsupported {
                what,
                reason: format!(
                    "the GPU cannot output {stream} as {format:?} (output format mask {:#06x})",
                    self.output_formats.bits()
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> DecoderCaps {
        DecoderCaps {
            codec: Codec::Hevc,
            chroma: ChromaFormat::Yuv420,
            bit_depth: 10,
            supported: true,
            engines: 1,
            output_formats: OutputFormats(0b11),
            min_width: 144,
            min_height: 144,
            max_width: 8192,
            max_height: 8192,
            max_macroblocks: 262_144,
        }
    }

    #[test]
    fn output_mask_bits_follow_the_surface_enum() {
        let formats = OutputFormats(0b10);
        assert!(formats.contains(PixelFormat::P016));
        assert!(!formats.contains(PixelFormat::Nv12));
        assert!(!formats.contains(PixelFormat::Yuv444));
    }

    #[test]
    fn check_names_the_limit() {
        let caps = caps();
        assert!(caps.check(1920, 1080, PixelFormat::P016).is_ok());
        let small = caps.check(128, 144, PixelFormat::P016).unwrap_err();
        assert!(small.to_string().contains("minimum 144x144"), "{small}");
        let large = caps.check(8200, 1080, PixelFormat::P016).unwrap_err();
        assert!(large.to_string().contains("maximum 8192x8192"), "{large}");
        let format = caps.check(1920, 1080, PixelFormat::Yuv444).unwrap_err();
        assert!(format.to_string().contains("cannot output"), "{format}");
        let unsupported = DecoderCaps {
            supported: false,
            ..caps
        };
        assert!(matches!(
            unsupported.check(1920, 1080, PixelFormat::P016),
            Err(Error::Unsupported { .. })
        ));
    }

    #[test]
    fn chroma_round_trip() {
        for chroma in [
            ChromaFormat::Monochrome,
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
        ] {
            assert_eq!(ChromaFormat::from_raw(chroma.raw()), Some(chroma));
        }
        assert_eq!(ChromaFormat::from_raw(sys::cudaVideoChromaFormat(9)), None);
    }
}
