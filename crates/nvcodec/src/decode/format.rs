//! Pixel formats, plane layouts and per-picture metadata of decoded frames.

use super::caps::ChromaFormat;
use crate::error::{Error, Result};
use crate::sys::cuvid as sys;

/// The memory layout of a decoded picture. Every format stores luma first,
/// then chroma; 16-bit formats hold `bit_depth` significant bits in the
/// most significant bits of each little-endian sample (the low bits are
/// zero), so 10-bit [`PixelFormat::P016`] is byte-for-byte P010.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 8-bit 4:2:0: Y plane, then interleaved UV at half height.
    Nv12,
    /// 9- to 16-bit 4:2:0: as [`PixelFormat::Nv12`] with 16-bit samples.
    P016,
    /// 8-bit 4:2:2: Y plane, then interleaved UV at full height.
    Nv16,
    /// 9- to 16-bit 4:2:2: as [`PixelFormat::Nv16`] with 16-bit samples.
    P216,
    /// 8-bit 4:4:4: Y, U and V planes, each full size.
    Yuv444,
    /// 9- to 16-bit 4:4:4: as [`PixelFormat::Yuv444`] with 16-bit samples.
    Yuv444P16,
}

/// How one plane samples the picture.
#[derive(Clone, Copy, Debug)]
struct PlaneKind {
    /// Horizontal subsampling (1 or 2).
    sub_x: u32,
    /// Vertical subsampling (1 or 2).
    sub_y: u32,
    /// Samples per pixel position (2 for interleaved UV).
    interleave: u32,
}

const LUMA: PlaneKind = PlaneKind {
    sub_x: 1,
    sub_y: 1,
    interleave: 1,
};

impl PixelFormat {
    /// The output format NVDEC is asked for, for a stream's chroma format and
    /// bit depth: NV12 / NV16 / YUV444 at 8 bits, their 16-bit variants
    /// above. Monochrome decodes as 4:2:0.
    pub(super) fn for_stream(chroma: ChromaFormat, bit_depth: u8) -> Self {
        let deep = bit_depth > 8;
        match (chroma, deep) {
            (ChromaFormat::Monochrome | ChromaFormat::Yuv420, false) => Self::Nv12,
            (ChromaFormat::Monochrome | ChromaFormat::Yuv420, true) => Self::P016,
            (ChromaFormat::Yuv422, false) => Self::Nv16,
            (ChromaFormat::Yuv422, true) => Self::P216,
            (ChromaFormat::Yuv444, false) => Self::Yuv444,
            (ChromaFormat::Yuv444, true) => Self::Yuv444P16,
        }
    }

    pub(super) fn surface_format(self) -> sys::cudaVideoSurfaceFormat {
        use sys::cudaVideoSurfaceFormat as F;
        match self {
            Self::Nv12 => F::cudaVideoSurfaceFormat_NV12,
            Self::P016 => F::cudaVideoSurfaceFormat_P016,
            Self::Nv16 => F::cudaVideoSurfaceFormat_NV16,
            Self::P216 => F::cudaVideoSurfaceFormat_P216,
            Self::Yuv444 => F::cudaVideoSurfaceFormat_YUV444,
            Self::Yuv444P16 => F::cudaVideoSurfaceFormat_YUV444_16Bit,
        }
    }

    /// Bytes per sample: 1 or 2.
    #[must_use]
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::Nv12 | Self::Nv16 | Self::Yuv444 => 1,
            Self::P016 | Self::P216 | Self::Yuv444P16 => 2,
        }
    }

    /// Number of planes: 2 (semi-planar) or 3 (planar 4:4:4).
    #[must_use]
    pub const fn plane_count(self) -> usize {
        match self {
            Self::Nv12 | Self::P016 | Self::Nv16 | Self::P216 => 2,
            Self::Yuv444 | Self::Yuv444P16 => 3,
        }
    }

    fn plane_kind(self, index: usize) -> PlaneKind {
        let chroma = match self {
            Self::Nv12 | Self::P016 => PlaneKind {
                sub_x: 2,
                sub_y: 2,
                interleave: 2,
            },
            Self::Nv16 | Self::P216 => PlaneKind {
                sub_x: 2,
                sub_y: 1,
                interleave: 2,
            },
            Self::Yuv444 | Self::Yuv444P16 => LUMA,
        };
        if index == 0 { LUMA } else { chroma }
    }

    /// Bytes per row and rows of plane `index` of a `width` x `height`
    /// picture. Subsampled planes round up (an odd width keeps its last
    /// chroma sample).
    fn plane_size(self, index: usize, width: u32, height: u32) -> (usize, usize) {
        let kind = self.plane_kind(index);
        let samples = width.div_ceil(kind.sub_x) * kind.interleave;
        (
            samples as usize * self.bytes_per_sample(),
            height.div_ceil(kind.sub_y) as usize,
        )
    }
}

/// One plane of a frame: where it starts in the frame's memory and its
/// shape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Plane {
    /// Byte offset of its first row from the start of the frame's memory.
    pub offset: usize,
    /// Bytes from one row to the next.
    pub pitch: usize,
    /// Bytes of picture per row (at most `pitch`).
    pub width_bytes: usize,
    /// Rows.
    pub rows: usize,
}

impl Plane {
    /// Bytes from the plane's first byte to the end of its last row.
    #[must_use]
    pub const fn len(&self) -> usize {
        if self.rows == 0 {
            0
        } else {
            (self.rows - 1) * self.pitch + self.width_bytes
        }
    }

    /// Whether the plane holds no bytes.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rows == 0 || self.width_bytes == 0
    }
}

/// The planes of a frame in one block of memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Layout {
    planes: [Plane; 3],
    count: usize,
    total: usize,
}

impl Layout {
    /// Planes packed back to back with no padding: each row exactly its
    /// width, as FFmpeg's `rawvideo` writes them.
    pub(super) fn packed(format: PixelFormat, width: u32, height: u32) -> Self {
        Self::build(format, width, height, None)
    }

    /// Planes stacked at one common `pitch`: plane `n + 1` starts on the row
    /// after the last row of plane `n` (NVENC's layout for NV12 / P010 /
    /// YUV444 input).
    pub(super) fn pitched(format: PixelFormat, width: u32, height: u32, pitch: usize) -> Self {
        Self::build(format, width, height, Some(pitch))
    }

    fn build(format: PixelFormat, width: u32, height: u32, pitch: Option<usize>) -> Self {
        let mut layout = Self {
            count: format.plane_count(),
            ..Self::default()
        };
        let mut offset = 0;
        for (index, plane) in layout.planes.iter_mut().take(layout.count).enumerate() {
            let (width_bytes, rows) = format.plane_size(index, width, height);
            let pitch = pitch.unwrap_or(width_bytes);
            *plane = Plane {
                offset,
                pitch,
                width_bytes,
                rows,
            };
            offset += pitch * rows;
        }
        layout.total = offset;
        layout
    }

    /// The planes, luma first.
    #[must_use]
    pub fn planes(&self) -> &[Plane] {
        self.planes.get(..self.count).unwrap_or_default()
    }

    /// Bytes from the start of the first plane to the end of the last one's
    /// last row, including the padding of that row.
    #[must_use]
    pub const fn total_bytes(&self) -> usize {
        self.total
    }

    /// The widest plane row, in bytes.
    pub(super) fn max_width_bytes(&self) -> usize {
        self.planes()
            .iter()
            .map(|plane| plane.width_bytes)
            .max()
            .unwrap_or(0)
    }

    /// Rows of all planes together.
    pub(super) fn total_rows(&self) -> usize {
        self.planes().iter().map(|plane| plane.rows).sum()
    }
}

/// A rectangle in pixels: the part of the coded picture to display.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Rect {
    /// First column.
    pub left: u32,
    /// First row.
    pub top: u32,
    /// One past the last column.
    pub right: u32,
    /// One past the last row.
    pub bottom: u32,
}

impl Rect {
    /// Width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.right.saturating_sub(self.left)
    }

    /// Height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.bottom.saturating_sub(self.top)
    }

    /// The display area of a sequence, validated against its coded size.
    /// An empty or out-of-range area (some streams leave it unset) means
    /// the whole coded picture.
    pub(super) fn display_area(
        area: &sys::CUVIDEOFORMAT__bindgen_ty_2,
        coded_width: u32,
        coded_height: u32,
    ) -> Self {
        let coords = [area.left, area.top, area.right, area.bottom].map(u32::try_from);
        let whole = Self {
            left: 0,
            top: 0,
            right: coded_width,
            bottom: coded_height,
        };
        match coords {
            [Ok(left), Ok(top), Ok(right), Ok(bottom)]
                if left < right
                    && top < bottom
                    && right <= coded_width
                    && bottom <= coded_height =>
            {
                Self {
                    left,
                    top,
                    right,
                    bottom,
                }
            }
            _ => whole,
        }
    }

    /// Byte offset of the rectangle's corner in plane `index` of `format`,
    /// as (bytes into the row, rows). Subsampled planes start at the chroma
    /// sample covering the corner.
    pub(super) fn plane_origin(&self, format: PixelFormat, index: usize) -> (usize, usize) {
        let kind = format.plane_kind(index);
        let x = (self.left / kind.sub_x * kind.interleave) as usize * format.bytes_per_sample();
        (x, (self.top / kind.sub_y) as usize)
    }
}

/// The stream's colour signalling, as ITU-T H.273 code points (the values
/// H.264 / HEVC VUI, AV1 `color_config` and VP9 use). 2 means unspecified.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Colour {
    /// `colour_primaries` (1 BT.709, 9 BT.2020, ...).
    pub primaries: u8,
    /// `transfer_characteristics` (1 BT.709, 16 PQ, 18 HLG, ...).
    pub transfer: u8,
    /// `matrix_coefficients` (1 BT.709, 6 BT.601, 9 BT.2020 non-constant, ...).
    pub matrix: u8,
    /// `video_full_range_flag`: full (0..=255 at 8 bits) rather than
    /// limited (16..=235) range.
    pub full_range: bool,
    /// `video_format` (0 component, ..., 5 unspecified).
    pub video_format: u8,
}

impl Colour {
    pub(super) fn from_raw(raw: sys::CUVIDEOFORMAT__bindgen_ty_4) -> Self {
        Self {
            primaries: raw.color_primaries,
            transfer: raw.transfer_characteristics,
            matrix: raw.matrix_coefficients,
            full_range: raw.video_full_range_flag() != 0,
            video_format: raw.video_format(),
        }
    }
}

/// Whether NVDEC decoded the picture cleanly (`cuvidGetDecodeStatus`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DecodeStatus {
    /// Decoded without error.
    Ok,
    /// Unknown: the driver has no `cuvidGetDecodeStatus`, or does not report
    /// it for this codec / GPU.
    Unknown,
    /// The bitstream had errors; the picture is damaged.
    Corrupted,
    /// The bitstream had errors that NVDEC concealed; the picture is
    /// approximate.
    Concealed,
}

impl DecodeStatus {
    pub(super) fn from_raw(raw: sys::cuvidDecodeStatus) -> Self {
        use sys::cuvidDecodeStatus as S;
        match raw {
            S::cuvidDecodeStatus_Success => Self::Ok,
            S::cuvidDecodeStatus_Error => Self::Corrupted,
            S::cuvidDecodeStatus_Error_Concealed => Self::Concealed,
            _ => Self::Unknown,
        }
    }

    /// Whether the picture is known to be damaged or concealed.
    #[must_use]
    pub const fn is_damaged(self) -> bool {
        matches!(self, Self::Corrupted | Self::Concealed)
    }
}

/// How a picture was coded and is to be shown, from
/// `CUVIDPARSERDISPINFO`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fields {
    /// A progressive frame (both fields from one instant).
    pub progressive: bool,
    /// The top field is shown first.
    pub top_field_first: bool,
    /// Extra fields to show: 1 for 3:2 pulldown, 2 frame doubling, 4 frame
    /// tripling, -1 a single unpaired field.
    pub repeat_first_field: i32,
}

/// Everything known about a decoded picture besides its pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameInfo {
    /// The timestamp given with the packet that carried the picture
    /// (interpolated by the parser for packets without one).
    pub timestamp: i64,
    /// Displayed width in pixels: the frame holds exactly this many.
    pub width: u32,
    /// Displayed height in pixels.
    pub height: u32,
    /// The layout of the samples.
    pub format: PixelFormat,
    /// Significant bits per sample (8, 10, 12).
    pub bit_depth: u8,
    /// Colour signalling of the sequence.
    pub colour: Colour,
    /// The displayed part of the coded picture.
    pub display_area: Rect,
    /// Coded width in pixels.
    pub coded_width: u32,
    /// Coded height in pixels.
    pub coded_height: u32,
    /// Progressive / field information.
    pub fields: Fields,
    /// Whether the picture decoded cleanly.
    pub status: DecodeStatus,
}

/// The format of the sequence being decoded, from the parser's sequence
/// callback (`CUVIDEOFORMAT`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SequenceInfo {
    /// The codec.
    pub codec: super::Codec,
    /// Chroma subsampling.
    pub chroma: ChromaFormat,
    /// Bits per sample.
    pub bit_depth: u8,
    /// Coded width in pixels.
    pub coded_width: u32,
    /// Coded height in pixels.
    pub coded_height: u32,
    /// The displayed part of the coded picture.
    pub display_area: Rect,
    /// Colour signalling.
    pub colour: Colour,
    /// What frames are output as.
    pub format: PixelFormat,
    /// Whether the sequence is progressive.
    pub progressive: bool,
    /// Frame rate numerator / denominator, `0 / 0` when unknown.
    pub frame_rate: (u32, u32),
    /// The minimum decode surfaces the stream needs.
    pub min_decode_surfaces: u32,
}

impl SequenceInfo {
    /// Reads and validates the parser's description of a new sequence.
    pub(super) fn from_raw(codec: super::Codec, raw: &sys::CUVIDEOFORMAT) -> Result<Self> {
        if raw.codec != codec.raw() {
            return Err(Error::UnexpectedValue {
                call: "the NVDEC sequence callback",
                what: "the codec",
                value: format!("{:?} for a {codec} decoder", raw.codec),
            });
        }
        let Some(chroma) = ChromaFormat::from_raw(raw.chroma_format) else {
            return Err(Error::Unsupported {
                what: "this chroma format",
                reason: format!("cudaVideoChromaFormat {}", raw.chroma_format.0),
            });
        };
        let (luma, chroma_depth) = (raw.bit_depth_luma_minus8, raw.bit_depth_chroma_minus8);
        if luma != chroma_depth && chroma != ChromaFormat::Monochrome {
            return Err(Error::Unsupported {
                what: "different luma and chroma bit depths",
                reason: format!(
                    "{} and {} bits",
                    u32::from(luma) + 8,
                    u32::from(chroma_depth) + 8
                ),
            });
        }
        let Some(bit_depth) = luma.checked_add(8).filter(|depth| *depth <= 16) else {
            return Err(Error::Unsupported {
                what: "this bit depth",
                reason: format!("{} bits", u32::from(luma) + 8),
            });
        };
        let (coded_width, coded_height) = (raw.coded_width, raw.coded_height);
        if coded_width == 0 || coded_height == 0 || coded_width > 1 << 15 || coded_height > 1 << 15
        {
            return Err(Error::UnexpectedValue {
                call: "the NVDEC sequence callback",
                what: "the coded size",
                value: format!("{coded_width}x{coded_height}"),
            });
        }
        Ok(Self {
            codec,
            chroma,
            bit_depth,
            coded_width,
            coded_height,
            display_area: Rect::display_area(&raw.display_area, coded_width, coded_height),
            colour: Colour::from_raw(raw.video_signal_description),
            format: PixelFormat::for_stream(chroma, bit_depth),
            progressive: raw.progressive_sequence != 0,
            frame_rate: (raw.frame_rate.numerator, raw.frame_rate.denominator),
            min_decode_surfaces: u32::from(raw.min_num_decode_surfaces),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_nv12_matches_ffmpeg_rawvideo() {
        // Odd sizes round the chroma up, as av_image_copy_to_buffer does.
        let layout = Layout::packed(PixelFormat::Nv12, 5, 3);
        let planes = layout.planes();
        assert_eq!(planes.len(), 2);
        assert_eq!((planes[0].width_bytes, planes[0].rows), (5, 3));
        assert_eq!(
            (planes[1].offset, planes[1].width_bytes, planes[1].rows),
            (15, 6, 2)
        );
        assert_eq!(layout.total_bytes(), 15 + 12);

        let layout = Layout::packed(PixelFormat::P016, 256, 144);
        assert_eq!(layout.total_bytes(), 256 * 144 * 2 * 3 / 2);
        let layout = Layout::packed(PixelFormat::Yuv444, 4, 2);
        assert_eq!(layout.planes().len(), 3);
        assert_eq!(layout.planes()[2].offset, 16);
    }

    #[test]
    fn pitched_planes_share_the_pitch() {
        let layout = Layout::pitched(PixelFormat::Nv16, 6, 4, 64);
        let planes = layout.planes();
        assert_eq!(planes[1].offset, 64 * 4);
        assert_eq!(planes[1].rows, 4);
        assert_eq!(layout.total_rows(), 8);
        assert_eq!(layout.max_width_bytes(), 6);
        assert_eq!(planes[1].len(), 3 * 64 + 6);
    }

    #[test]
    fn crop_origin_per_plane() {
        let rect = Rect {
            left: 8,
            top: 6,
            right: 400,
            bottom: 300,
        };
        assert_eq!(rect.plane_origin(PixelFormat::Nv12, 0), (8, 6));
        assert_eq!(rect.plane_origin(PixelFormat::Nv12, 1), (8, 3));
        assert_eq!(rect.plane_origin(PixelFormat::P016, 1), (16, 3));
        assert_eq!(rect.plane_origin(PixelFormat::Yuv444P16, 2), (16, 6));
        assert_eq!((rect.width(), rect.height()), (392, 294));
    }

    #[test]
    fn display_area_falls_back_to_the_coded_size() {
        let area = sys::CUVIDEOFORMAT__bindgen_ty_2 {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        };
        assert_eq!(Rect::display_area(&area, 64, 32).width(), 64);
        let area = sys::CUVIDEOFORMAT__bindgen_ty_2 {
            left: 2,
            top: 2,
            right: 70,
            bottom: 30,
        };
        assert_eq!(Rect::display_area(&area, 64, 32).right, 64);
        let area = sys::CUVIDEOFORMAT__bindgen_ty_2 {
            left: 2,
            top: 4,
            right: 62,
            bottom: 30,
        };
        assert_eq!(
            Rect::display_area(&area, 64, 32),
            Rect {
                left: 2,
                top: 4,
                right: 62,
                bottom: 30
            }
        );
    }

    #[test]
    fn output_format_follows_depth_and_chroma() {
        assert_eq!(
            PixelFormat::for_stream(ChromaFormat::Yuv420, 8),
            PixelFormat::Nv12
        );
        assert_eq!(
            PixelFormat::for_stream(ChromaFormat::Yuv420, 10),
            PixelFormat::P016
        );
        assert_eq!(
            PixelFormat::for_stream(ChromaFormat::Monochrome, 12),
            PixelFormat::P016
        );
        assert_eq!(
            PixelFormat::for_stream(ChromaFormat::Yuv444, 10),
            PixelFormat::Yuv444P16
        );
    }
}
