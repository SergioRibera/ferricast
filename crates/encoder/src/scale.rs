//! Output-size scaling stage shared by the video backends.
//!
//! A stream's encoded size is fixed when it starts (see
//! [`FrameSize::fit_within`]), but the capture source can change size
//! mid-stream: a window is resized, the monitor changes mode, the
//! compositor toggles fractional scaling. Backends therefore accept
//! input of any size and fit it into the configured output size,
//! keeping the aspect ratio and filling the rest with black
//! (letterbox / pillarbox). The receiver never sees a resolution
//! change.

use bytes::Bytes;
use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use ferricast_core::{FerricastError, PixelFormat, RawFrame, Result};

use crate::frame_view::FrameView;

/// Opaque black in both byte orders the scaler accepts (BGRA and
/// RGBA keep alpha last).
const BLACK_PIXEL: [u8; 4] = [0, 0, 0, 0xff];

/// Width × height of a picture, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameSize {
    pub width: u32,
    pub height: u32,
}

/// Where a picture lands inside a larger canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    pub x: u32,
    pub y: u32,
    pub size: FrameSize,
}

impl FrameSize {
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// The output size to encode a `self`-sized source at, given the
    /// receiver's optional limits: the source size when it already
    /// fits, otherwise the largest size with the source's aspect that
    /// does. Never upscales. Both dimensions are even (4:2:0 chroma)
    /// and at least 2.
    #[must_use]
    pub fn fit_within(self, max_width: Option<u32>, max_height: Option<u32>) -> Self {
        let bound = Self::new(
            max_width.map_or(self.width, |m| m.min(self.width)),
            max_height.map_or(self.height, |m| m.min(self.height)),
        );
        if bound == self {
            return self.even();
        }
        self.letterbox_in(bound).size
    }

    /// Largest aspect-preserving placement of a `self`-sized picture
    /// centred in `canvas`. Size and offsets are even so chroma
    /// samples stay aligned.
    #[must_use]
    pub fn letterbox_in(self, canvas: Self) -> Placement {
        let (sw, sh) = (u64::from(self.width.max(1)), u64::from(self.height.max(1)));
        let (cw, ch) = (u64::from(canvas.width), u64::from(canvas.height));
        // Compare aspect ratios without division: source wider than
        // the canvas → full width, reduced height; else full height.
        let (w, h) = if sw * ch >= sh * cw {
            (cw, sh * cw / sw)
        } else {
            (sw * ch / sh, ch)
        };
        // `w <= cw` and `h <= ch`, both of which came from `u32`.
        let size = Self::new(
            u32::try_from(w).unwrap_or(canvas.width),
            u32::try_from(h).unwrap_or(canvas.height),
        )
        .even();
        let size = Self::new(size.width.min(canvas.width), size.height.min(canvas.height));
        Placement {
            x: (canvas.width.saturating_sub(size.width) / 2) & !1,
            y: (canvas.height.saturating_sub(size.height) / 2) & !1,
            size,
        }
    }

    /// Rounded down to even dimensions, at least 2×2.
    #[must_use]
    pub const fn even(self) -> Self {
        let w = self.width & !1;
        let h = self.height & !1;
        Self::new(if w < 2 { 2 } else { w }, if h < 2 { 2 } else { h })
    }

    fn of(raw: &RawFrame) -> Self {
        Self::new(raw.width, raw.height)
    }
}

impl Placement {
    /// Whether the placement covers the whole `canvas`.
    #[must_use]
    pub fn fills(&self, canvas: FrameSize) -> bool {
        self.x == 0 && self.y == 0 && self.size == canvas
    }
}

/// CPU scaler that fits packed RGB frames into a fixed output size.
///
/// Keeps its resampler and repack buffer across frames; frames that
/// already have the output size pass through untouched.
#[derive(Default)]
pub(crate) struct FrameScaler {
    resizer: Resizer,
    repack: Vec<u8>,
    last_source: Option<FrameSize>,
}

impl FrameScaler {
    /// Return `raw` unchanged when it already is `out`-sized,
    /// otherwise a tightly packed copy in the same pixel format with
    /// the picture letterboxed into `out`.
    ///
    /// # Errors
    ///
    /// [`FerricastError::Encoder`] when the frame needs scaling but is
    /// planar YUV (only BGRA / RGBA are scaled) or its geometry is
    /// inconsistent with its buffer.
    pub(crate) fn fit(&mut self, raw: RawFrame, out: FrameSize) -> Result<RawFrame> {
        let source = FrameSize::of(&raw);
        if source == out {
            return Ok(raw);
        }
        if self.last_source != Some(source) {
            tracing::info!(
                src_width = source.width,
                src_height = source.height,
                out_width = out.width,
                out_height = out.height,
                "encoder: scaling source into the fixed output size"
            );
            self.last_source = Some(source);
        }
        if !matches!(raw.format, PixelFormat::Bgra | PixelFormat::Rgba) {
            return Err(FerricastError::Encoder(format!(
                "cannot scale {:?} {}x{} frame to {}x{}: only BGRA / RGBA input is scaled",
                raw.format, raw.width, raw.height, out.width, out.height
            )));
        }

        let view = FrameView::new(&raw, raw.width, raw.height)?;
        let src_bytes = view.tight_rows(&mut self.repack);
        let src = ImageRef::new(source.width, source.height, src_bytes, PixelType::U8x4)
            .map_err(|e| FerricastError::Encoder(format!("scaler source: {e}")))?;
        let options = ResizeOptions::new()
            .resize_alg(ResizeAlg::Convolution(FilterType::Bilinear))
            // Capture alpha is padding (BGRx) — premultiplying by it
            // would darken or blank the picture.
            .use_alpha(false);

        let placement = source.letterbox_in(out);
        let out_row = out.width as usize * 4;
        let mut canvas = if placement.fills(out) {
            let mut dst = Image::new(out.width, out.height, PixelType::U8x4);
            self.resizer
                .resize(&src, &mut dst, &options)
                .map_err(|e| FerricastError::Encoder(format!("scaler: {e}")))?;
            dst.into_vec()
        } else {
            let mut inner =
                Image::new(placement.size.width, placement.size.height, PixelType::U8x4);
            self.resizer
                .resize(&src, &mut inner, &options)
                .map_err(|e| FerricastError::Encoder(format!("scaler: {e}")))?;
            let mut canvas: Vec<u8> = BLACK_PIXEL
                .iter()
                .copied()
                .cycle()
                .take(out_row * out.height as usize)
                .collect();
            let inner_row = placement.size.width as usize * 4;
            let x_off = placement.x as usize * 4;
            for (dst_row, src_row) in canvas
                .chunks_exact_mut(out_row)
                .skip(placement.y as usize)
                .zip(inner.buffer().chunks_exact(inner_row))
            {
                dst_row[x_off..x_off + inner_row].copy_from_slice(src_row);
            }
            canvas
        };
        canvas.truncate(out_row * out.height as usize);

        Ok(RawFrame {
            width: out.width,
            height: out.height,
            stride: out.width * 4,
            format: raw.format,
            data: Bytes::from(canvas),
            timestamp_us: raw.timestamp_us,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_within_keeps_sources_that_fit() {
        let s = FrameSize::new(1920, 1080);
        assert_eq!(s.fit_within(Some(1920), Some(1080)), s);
        assert_eq!(s.fit_within(None, None), s);
        assert_eq!(s.fit_within(Some(3840), Some(2160)), s);
    }

    #[test]
    fn fit_within_never_upscales() {
        let s = FrameSize::new(1280, 720);
        assert_eq!(s.fit_within(Some(1920), Some(1080)), s);
    }

    #[test]
    fn fit_within_scales_down_keeping_aspect() {
        let s = FrameSize::new(3840, 2160);
        assert_eq!(
            s.fit_within(Some(1920), Some(1080)),
            FrameSize::new(1920, 1080)
        );
        // Ultrawide → height shrinks.
        let s = FrameSize::new(3440, 1440);
        assert_eq!(
            s.fit_within(Some(1920), Some(1080)),
            FrameSize::new(1920, 802)
        );
        // Portrait → width shrinks.
        let s = FrameSize::new(1080, 1920);
        assert_eq!(
            s.fit_within(Some(1920), Some(1080)),
            FrameSize::new(606, 1080)
        );
    }

    #[test]
    fn fit_within_only_one_limit() {
        let s = FrameSize::new(2560, 1440);
        assert_eq!(s.fit_within(Some(1280), None), FrameSize::new(1280, 720));
    }

    #[test]
    fn fit_within_rounds_to_even() {
        let s = FrameSize::new(1919, 1079);
        assert_eq!(s.fit_within(None, None), FrameSize::new(1918, 1078));
        let tiny = FrameSize::new(1, 1);
        assert_eq!(tiny.fit_within(None, None), FrameSize::new(2, 2));
    }

    #[test]
    fn letterbox_centres_with_even_offsets() {
        let p = FrameSize::new(1000, 1000).letterbox_in(FrameSize::new(1920, 1080));
        assert_eq!(p.size, FrameSize::new(1080, 1080));
        assert_eq!((p.x, p.y), (420, 0));
        let p = FrameSize::new(1920, 800).letterbox_in(FrameSize::new(1920, 1080));
        assert_eq!(p.size, FrameSize::new(1920, 800));
        assert_eq!((p.x, p.y), (0, 140));
        assert!(p.x.is_multiple_of(2) && p.y.is_multiple_of(2));
    }

    fn solid(width: u32, height: u32, stride: u32, px: [u8; 4]) -> RawFrame {
        let mut data = vec![0u8; stride as usize * height as usize];
        for row in data.chunks_exact_mut(stride as usize) {
            for p in row[..width as usize * 4].chunks_exact_mut(4) {
                p.copy_from_slice(&px);
            }
        }
        RawFrame {
            width,
            height,
            stride,
            format: PixelFormat::Bgra,
            data: Bytes::from(data),
            timestamp_us: 42,
        }
    }

    #[test]
    fn scaler_passes_matching_frames_through() {
        let mut s = FrameScaler::default();
        let raw = solid(8, 4, 40, [1, 2, 3, 255]);
        let out = s.fit(raw.clone(), FrameSize::new(8, 4)).unwrap();
        assert_eq!(out.data, raw.data);
        assert_eq!(out.stride, 40);
    }

    #[test]
    fn scaler_downscales_padded_frames() {
        let mut s = FrameScaler::default();
        let out = s
            .fit(solid(16, 8, 80, [10, 20, 30, 255]), FrameSize::new(8, 4))
            .unwrap();
        assert_eq!((out.width, out.height, out.stride), (8, 4, 32));
        assert_eq!(out.data.len(), 8 * 4 * 4);
        assert_eq!(out.timestamp_us, 42);
        assert!(out.data.chunks_exact(4).all(|p| p[..3] == [10, 20, 30]));
    }

    #[test]
    fn scaler_letterboxes_other_aspects() {
        let mut s = FrameScaler::default();
        // Square source into a 2:1 canvas → black bars left / right.
        let out = s
            .fit(solid(8, 8, 32, [200, 200, 200, 255]), FrameSize::new(16, 8))
            .unwrap();
        let px = |x: usize, y: usize| &out.data[(y * 16 + x) * 4..(y * 16 + x) * 4 + 4];
        assert_eq!(px(0, 4), BLACK_PIXEL);
        assert_eq!(px(15, 4), BLACK_PIXEL);
        assert_eq!(px(8, 4)[..3], [200, 200, 200]);
    }

    #[test]
    fn scaler_rejects_planar_input_that_needs_scaling() {
        let mut s = FrameScaler::default();
        let raw = RawFrame {
            width: 4,
            height: 4,
            stride: 4,
            format: PixelFormat::Nv12,
            data: Bytes::from(vec![0u8; 24]),
            timestamp_us: 0,
        };
        assert!(s.fit(raw, FrameSize::new(2, 2)).is_err());
    }
}
