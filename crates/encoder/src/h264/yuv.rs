//! BGRA → NV12 conversion for the VA-API encoder upload path.
//!
//! VA-API H.264 encoders consume YUV 4:2:0 surfaces (NV12 layout:
//! one plane of `Y`, then a tightly-interleaved `UV` plane at half
//! width and half height). Capture sources hand us packed BGRA, so
//! we colour-convert before uploading.
//!
//! Uses BT.601 limited-range coefficients — the same convention
//! every H.264 decoder defaults to when `colour_primaries` /
//! `matrix_coefficients` aren't specified in VUI. Limited range
//! (`Y` ∈ [16,235], `UV` ∈ [16,240]) keeps the output decodable on
//! receivers that don't honour full-range signalling.
//!
//! Performance: this is a per-frame CPU loop. For a 1920×1080
//! frame that's 8 MB read + 3 MB write. A scalar implementation
//! runs in ~3-5 ms; SIMD would bring it under 1 ms but the encoder
//! is the bottleneck either way (encode dominates). When/if we
//! switch to a VA-API VPP path the BGRA dmabuf is converted in the
//! GPU and this whole module becomes legacy.

#![allow(dead_code)] // referenced from vaapi_impl

/// Convert a packed BGRA plane to two planar surfaces (`Y`, `UV`)
/// in NV12 layout.
///
/// * `src` — input pixels, `src_stride` bytes per row, `height`
///   rows. The first byte of each pixel is `B`, then `G`, `R`, `A`.
/// * `y_dst` / `y_stride` — destination luma plane.
/// * `uv_dst` / `uv_stride` — destination chroma plane (interleaved
///   `U`/`V`, half width × half height).
/// * `width`, `height` — image dimensions in pixels. Must both be
///   even (the chroma subsampling assumes pairs).
///
/// Stride may exceed `width * bpp` to honour driver alignment.
pub(super) fn bgra_to_nv12(
    src: &[u8],
    src_stride: usize,
    width: u32,
    height: u32,
    y_dst: &mut [u8],
    y_stride: usize,
    uv_dst: &mut [u8],
    uv_stride: usize,
) {
    debug_assert!(width % 2 == 0 && height % 2 == 0, "even dims required");
    let w = width as usize;
    let h = height as usize;

    // Two rows at a time so we can subsample chroma per 2×2 block.
    for y in (0..h).step_by(2) {
        let src_row0 = &src[y * src_stride..y * src_stride + w * 4];
        let src_row1 = &src[(y + 1) * src_stride..(y + 1) * src_stride + w * 4];
        // Split `y_dst` into two non-overlapping row slices so the
        // borrow checker is happy.
        let (y_lo, y_hi) = y_dst.split_at_mut((y + 1) * y_stride);
        let y_row0 = &mut y_lo[y * y_stride..y * y_stride + w];
        let y_row1 = &mut y_hi[..w];
        let uv_off = (y / 2) * uv_stride;

        for x in (0..w).step_by(2) {
            let p00 = &src_row0[x * 4..x * 4 + 4];
            let p01 = &src_row0[(x + 1) * 4..(x + 1) * 4 + 4];
            let p10 = &src_row1[x * 4..x * 4 + 4];
            let p11 = &src_row1[(x + 1) * 4..(x + 1) * 4 + 4];

            // BGRA layout.
            let (b00, g00, r00) = (p00[0] as i32, p00[1] as i32, p00[2] as i32);
            let (b01, g01, r01) = (p01[0] as i32, p01[1] as i32, p01[2] as i32);
            let (b10, g10, r10) = (p10[0] as i32, p10[1] as i32, p10[2] as i32);
            let (b11, g11, r11) = (p11[0] as i32, p11[1] as i32, p11[2] as i32);

            // BT.601 limited range, coefficients ×256:
            //   Y = (66*R + 129*G +  25*B) / 256 + 16
            //   U = (-38*R -  74*G + 112*B) / 256 + 128
            //   V = (112*R -  94*G -  18*B) / 256 + 128
            let y00 = ((66 * r00 + 129 * g00 + 25 * b00 + 128) >> 8) + 16;
            let y01 = ((66 * r01 + 129 * g01 + 25 * b01 + 128) >> 8) + 16;
            let y10 = ((66 * r10 + 129 * g10 + 25 * b10 + 128) >> 8) + 16;
            let y11 = ((66 * r11 + 129 * g11 + 25 * b11 + 128) >> 8) + 16;

            y_row0[x] = y00.clamp(0, 255) as u8;
            y_row0[x + 1] = y01.clamp(0, 255) as u8;
            y_row1[x] = y10.clamp(0, 255) as u8;
            y_row1[x + 1] = y11.clamp(0, 255) as u8;

            // Chroma: average the four pixels' colour components
            // before converting (cheaper and correct for 4:2:0).
            let r_avg = (r00 + r01 + r10 + r11) >> 2;
            let g_avg = (g00 + g01 + g10 + g11) >> 2;
            let b_avg = (b00 + b01 + b10 + b11) >> 2;
            let u = ((-38 * r_avg - 74 * g_avg + 112 * b_avg + 128) >> 8) + 128;
            let v = ((112 * r_avg - 94 * g_avg - 18 * b_avg + 128) >> 8) + 128;

            // NV12: U then V interleaved for each 2×2 block.
            uv_dst[uv_off + x] = u.clamp(0, 255) as u8;
            uv_dst[uv_off + x + 1] = v.clamp(0, 255) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pure black BGRA → expected NV12: Y plane all 16, UV plane all 128.
    #[test]
    fn black_image() {
        let w = 4u32;
        let h = 4u32;
        let src = vec![0u8; (w * h * 4) as usize];
        let mut y = vec![0u8; (w * h) as usize];
        let mut uv = vec![0u8; (w * h / 2) as usize];
        bgra_to_nv12(
            &src,
            (w * 4) as usize,
            w,
            h,
            &mut y,
            w as usize,
            &mut uv,
            w as usize,
        );
        assert!(y.iter().all(|&v| v == 16), "Y plane should be 16 (black)");
        assert!(uv.iter().all(|&v| v == 128), "UV should be 128 (neutral)");
    }

    /// Pure white BGRA → Y plane near 235.
    #[test]
    fn white_image() {
        let w = 4u32;
        let h = 4u32;
        let src = vec![0xFFu8; (w * h * 4) as usize];
        let mut y = vec![0u8; (w * h) as usize];
        let mut uv = vec![0u8; (w * h / 2) as usize];
        bgra_to_nv12(
            &src,
            (w * 4) as usize,
            w,
            h,
            &mut y,
            w as usize,
            &mut uv,
            w as usize,
        );
        assert!(
            y.iter().all(|&v| (233..=237).contains(&v)),
            "Y plane should be ~235 (white): {:?}",
            &y[..4]
        );
    }
}
