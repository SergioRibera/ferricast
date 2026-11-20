//! Packed RGB → NV12 conversion for the VA-API encoder upload path.
//!
//! Thin wrapper over the SIMD-accelerated `yuv` crate's
//! `bgra_to_yuv_nv12`. The previous hand-rolled scalar version ran at
//! ~3-5 ms per 1080p frame; the SIMD path is under 1 ms on any host
//! with SSE4.2/AVX2/RDM, which matters at 60 fps or 1440p+ when
//! capture and encode are competing for the same cores.
//!
//! BT.709 limited range (`Y` ∈ [16,235], `UV` ∈ [16,240]), the same
//! matrix the x264 path uses and what HD receivers (Chromecast
//! included) assume. The SPS VUI signals it explicitly
//! (`colour_primaries` = `transfer_characteristics` =
//! `matrix_coefficients` = 1, `video_full_range_flag` = 0) so no
//! decoder has to guess.

use cros_libva::{
    Image, ProcColorProperties, Surface, VA_LSB_FIRST, VA_SOURCE_RANGE_FULL,
    VA_SOURCE_RANGE_REDUCED, VAImageFormat,
};
use ferricast_core::{FerricastError, PixelFormat, Result};
use yuv::{
    BufferStoreMut, YuvBiPlanarImageMut, YuvConversionMode, YuvRange, YuvStandardMatrix,
    bgra_to_yuv_nv12, rgba_to_yuv_nv12,
};

use crate::frame_view::FrameView;

/// Standard NV12 V4L2 / DRM fourcc reused across libva.
const VA_FOURCC_NV12: u32 = 0x3231_564E; // 'N','V','1','2'

/// `VAProcColorStandardType` (va_vpp.h) for the GPU colour conversion:
/// the captured DMA-BUF is sRGB, the encoder input BT.709 — the same
/// matrix the CPU path and the SPS VUI use.
pub(crate) const VPP_INPUT_STANDARD: u8 = 8; // VAProcColorStandardSRGB
pub(crate) const VPP_OUTPUT_STANDARD: u8 = 2; // VAProcColorStandardBT709

/// Full-range RGB in.
pub(crate) fn vpp_input_colour() -> ProcColorProperties {
    ProcColorProperties::new(0, VA_SOURCE_RANGE_FULL as u8, 0, 0, 0)
}

/// Limited-range BT.709 out (H.273 code 1 for primaries / transfer /
/// matrix), matching the VUI.
pub(crate) fn vpp_output_colour() -> ProcColorProperties {
    ProcColorProperties::new(0, VA_SOURCE_RANGE_REDUCED as u8, 1, 1, 1)
}

/// Limited-range black: luma 16, neutral chroma 128.
const PAD_LUMA: u8 = 16;
const PAD_CHROMA: u8 = 128;

/// Driver-allocated NV12 destination planes (split-borrowed because
/// the Y and UV regions live in the same mapped VAImage buffer).
///
/// * `y` / `y_stride` — destination luma plane.
/// * `uv` / `uv_stride` — destination chroma plane (interleaved
///   `U`/`V`, half height, full byte width).
pub(crate) struct Nv12Planes<'a> {
    pub y: &'a mut [u8],
    pub y_stride: usize,
    pub uv: &'a mut [u8],
    pub uv_stride: usize,
}

impl Nv12Planes<'_> {
    /// Colour-convert `src` into the top-left `src.width × src.height`
    /// region of the planes, honouring `src.stride()`.
    ///
    /// # Errors
    ///
    /// Returns [`FerricastError::Encoder`] for non-packed-RGB input or
    /// when the destination planes are too small for the picture.
    pub(crate) fn fill_from(&mut self, src: &FrameView<'_>) -> Result<()> {
        let mut bi = YuvBiPlanarImageMut::<u8> {
            y_plane: BufferStoreMut::Borrowed(&mut *self.y),
            y_stride: to_u32(self.y_stride)?,
            uv_plane: BufferStoreMut::Borrowed(&mut *self.uv),
            uv_stride: to_u32(self.uv_stride)?,
            width: src.width(),
            height: src.height(),
        };
        let src_stride = to_u32(src.stride())?;
        let converted = match src.format() {
            PixelFormat::Bgra => bgra_to_yuv_nv12(
                &mut bi,
                src.data(),
                src_stride,
                YuvRange::Limited,
                YuvStandardMatrix::Bt709,
                YuvConversionMode::Balanced,
            ),
            PixelFormat::Rgba => rgba_to_yuv_nv12(
                &mut bi,
                src.data(),
                src_stride,
                YuvRange::Limited,
                YuvStandardMatrix::Bt709,
                YuvConversionMode::Balanced,
            ),
            other @ (PixelFormat::Nv12 | PixelFormat::I420) => {
                return Err(FerricastError::Encoder(format!(
                    "VA-API: cannot upload {other:?} frames (need Bgra/Rgba)"
                )));
            }
        };
        converted.map_err(|e| FerricastError::Encoder(format!("RGB→NV12 conversion: {e}")))
    }

    /// Paint everything outside the visible `width × height` region of
    /// a `coded_width × coded_height` picture black, so the encoder's
    /// padding macroblocks / CTUs never carry stale data into the
    /// reference frames.
    pub(crate) fn fill_padding(
        &mut self,
        (width, height): (u32, u32),
        (coded_width, coded_height): (u32, u32),
    ) {
        let (w, h) = (width as usize, height as usize);
        let (cw, ch) = (coded_width as usize, coded_height as usize);
        // Chroma is interleaved U/V at half vertical resolution, so a
        // row of the UV plane has the same byte width as a luma row.
        let (cw_even, w_even) = (cw.next_multiple_of(2), w.next_multiple_of(2));
        paint(self.y, self.y_stride, (w, h), (cw, ch), PAD_LUMA);
        paint(
            self.uv,
            self.uv_stride,
            (w_even, h.div_ceil(2)),
            (cw_even, ch.div_ceil(2)),
            PAD_CHROMA,
        );
    }
}

/// Fill the right strip of the first `visible.1` rows and every row
/// below them (up to `coded`) with `value`.
fn paint(
    plane: &mut [u8],
    stride: usize,
    visible: (usize, usize),
    coded: (usize, usize),
    value: u8,
) {
    for (row, line) in plane.chunks_mut(stride).take(coded.1).enumerate() {
        let start = if row < visible.1 { visible.0 } else { 0 };
        let end = coded.0.min(line.len());
        if start < end {
            line[start..end].fill(value);
        }
    }
}

fn to_u32(v: usize) -> Result<u32> {
    u32::try_from(v).map_err(|_| FerricastError::Encoder(format!("stride {v} exceeds u32")))
}

/// Upload `src` into an NV12 VA surface whose coded (macroblock / CTU
/// aligned) size is `coded`. Uses `vaDeriveImage` for a zero-copy view
/// when the driver supports it; falls back to `vaCreateImage` +
/// `vaPutImage` otherwise.
///
/// # Errors
///
/// Returns [`FerricastError::Encoder`] when the image cannot be mapped,
/// its planes are smaller than `coded`, or the conversion fails.
pub(crate) fn upload_to_surface(
    surface: &Surface<()>,
    coded: (u32, u32),
    src: &FrameView<'_>,
) -> Result<()> {
    let mut image = match Image::derive_from(surface, coded) {
        Ok(img) => img,
        Err(_e) => {
            // Driver doesn't allow derive on this format. Allocate
            // an NV12 image and pay the put-image cost on drop.
            let format = VAImageFormat {
                fourcc: VA_FOURCC_NV12,
                byte_order: VA_LSB_FIRST,
                bits_per_pixel: 12,
                ..Default::default()
            };
            Image::create_from(surface, format, coded, coded)
                .map_err(|e| FerricastError::Encoder(format!("vaCreateImage(NV12): {e}")))?
        }
    };

    // Lay out the NV12 planes inside the mapped buffer using the
    // image's plane offsets and strides.
    let im = image.image();
    let y_off = im.offsets[0] as usize;
    let uv_off = im.offsets[1] as usize;
    let y_stride = im.pitches[0] as usize;
    let uv_stride = im.pitches[1] as usize;
    let coded_h = coded.1 as usize;

    let buf = image.as_mut(); // marks dirty so vaPutImage runs on drop
    let len = buf.len();
    if y_stride < coded.0 as usize
        || uv_stride < coded.0 as usize
        || y_off + y_stride * coded_h > len
        || uv_off + uv_stride * coded_h.div_ceil(2) > len
    {
        return Err(FerricastError::Encoder(
            "VA-API NV12 image buffer smaller than expected planes".into(),
        ));
    }
    // Split-borrow to write Y and UV planes.
    let (y, uv) = if y_off < uv_off {
        let (lo, hi) = buf.split_at_mut(uv_off);
        (&mut lo[y_off..], hi)
    } else {
        let (lo, hi) = buf.split_at_mut(y_off);
        (hi, &mut lo[uv_off..])
    };
    let mut planes = Nv12Planes {
        y,
        y_stride,
        uv,
        uv_stride,
    };
    planes.fill_from(src)?;
    planes.fill_padding((src.width(), src.height()), coded);
    Ok(())
    // Image drops here: if !derived && dirty, vaPutImage writes
    // back to the surface; vaUnmapBuffer + vaDestroyImage either
    // way.
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use ferricast_core::RawFrame;

    use super::*;

    fn frame(format: PixelFormat, w: u32, h: u32, stride: u32, fill: u8) -> RawFrame {
        RawFrame {
            width: w,
            height: h,
            stride,
            format,
            data: Bytes::from(vec![fill; (stride * h) as usize]),
            timestamp_us: 0,
        }
    }

    fn convert(raw: &RawFrame) -> Result<(Vec<u8>, Vec<u8>)> {
        let (w, h) = (raw.width as usize, raw.height as usize);
        let mut y = vec![0u8; w * h];
        let mut uv = vec![0u8; w * h / 2];
        let view = FrameView::new(raw, raw.width, raw.height)?;
        Nv12Planes {
            y: &mut y,
            y_stride: w,
            uv: &mut uv,
            uv_stride: w,
        }
        .fill_from(&view)?;
        Ok((y, uv))
    }

    /// Pure black BGRA → expected NV12: Y plane all 16, UV plane all 128.
    #[test]
    fn black_image() {
        let mut raw = frame(PixelFormat::Bgra, 4, 4, 16, 0);
        raw.data = Bytes::from(vec![0u8; 64]);
        let (y, uv) = convert(&raw).unwrap();
        assert!(y.iter().all(|&v| v == 16), "Y plane should be 16 (black)");
        assert!(uv.iter().all(|&v| v == 128), "UV should be 128 (neutral)");
    }

    /// Pure white BGRA → Y plane near 235.
    #[test]
    fn white_image() {
        let raw = frame(PixelFormat::Bgra, 4, 4, 16, 0xFF);
        let (y, _) = convert(&raw).unwrap();
        assert!(
            y.iter().all(|&v| (233..=237).contains(&v)),
            "Y plane should be ~235 (white): {:?}",
            &y[..4]
        );
    }

    /// The same bytes mean red in RGBA and blue in BGRA; the luma of
    /// red is roughly twice that of blue, so a swapped dispatch shows.
    #[test]
    fn rgba_and_bgra_dispatch_differ() {
        let px = [0xFFu8, 0, 0, 0xFF];
        let mut raw = frame(PixelFormat::Rgba, 4, 4, 16, 0);
        raw.data = Bytes::from(px.repeat(16));
        let (red_y, _) = convert(&raw).unwrap();
        raw.format = PixelFormat::Bgra;
        let (blue_y, _) = convert(&raw).unwrap();
        assert!(
            red_y[0] > blue_y[0] + 20,
            "red {} vs blue {}",
            red_y[0],
            blue_y[0]
        );
    }

    /// Row padding must be skipped, not read as pixels: a white image
    /// with black padding bytes converts to uniform white.
    #[test]
    fn padded_stride_is_honoured() {
        let (w, h, stride) = (4u32, 4u32, 32usize);
        let mut data = vec![0u8; stride * h as usize];
        for row in data.chunks_mut(stride) {
            row[..(w * 4) as usize].fill(0xFF);
        }
        let raw = RawFrame {
            width: w,
            height: h,
            stride: stride as u32,
            format: PixelFormat::Bgra,
            data: Bytes::from(data),
            timestamp_us: 0,
        };
        let (y, _) = convert(&raw).unwrap();
        assert!(y.iter().all(|&v| (233..=237).contains(&v)), "{y:?}");
    }

    #[test]
    fn padding_painted_black() {
        // Visible 2x2 inside a coded 4x4 picture.
        let mut y = vec![0xAAu8; 16];
        let mut uv = vec![0xAAu8; 8];
        Nv12Planes {
            y: &mut y,
            y_stride: 4,
            uv: &mut uv,
            uv_stride: 4,
        }
        .fill_padding((2, 2), (4, 4));
        assert_eq!(&y[..4], &[0xAA, 0xAA, 16, 16]);
        assert_eq!(&y[4..8], &[0xAA, 0xAA, 16, 16]);
        assert!(y[8..].iter().all(|&v| v == 16));
        assert_eq!(&uv[..4], &[0xAA, 0xAA, 128, 128]);
        assert!(uv[4..].iter().all(|&v| v == 128));
    }
}
