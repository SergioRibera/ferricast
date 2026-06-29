//! BGRA → NV12 conversion for the VA-API encoder upload path.
//!
//! Thin wrapper over the SIMD-accelerated `yuv` crate's
//! `bgra_to_yuv_nv12`. The previous hand-rolled scalar version ran at
//! ~3-5 ms per 1080p frame; the SIMD path is under 1 ms on any host
//! with SSE4.2/AVX2/RDM, which matters at 60 fps or 1440p+ when
//! capture and encode are competing for the same cores.
//!
//! BT.601 limited range — the convention every H.264/HEVC decoder
//! defaults to when `colour_primaries` / `matrix_coefficients` aren't
//! signalled in VUI. Limited range (`Y` ∈ [16,235], `UV` ∈ [16,240])
//! is what every receiver in our matrix (Chromecast, Miracast,
//! AirPlay) expects on the wire.

#![allow(dead_code)] // referenced from vaapi_impl

use yuv::{
    BufferStoreMut, YuvBiPlanarImageMut, YuvConversionMode, YuvRange, YuvStandardMatrix,
    bgra_to_yuv_nv12,
};

/// Convert a packed BGRA plane into the driver-allocated NV12
/// destination planes (split-borrowed because the Y and UV regions
/// live in the same mapped VAImage buffer).
///
/// * `src` / `src_stride` — input BGRA bytes; stride may exceed
///   `width * 4` if the source has row padding.
/// * `width`, `height` — image dimensions in pixels. Must both be
///   even (chroma subsampling assumes 2×2 blocks).
/// * `y_dst` / `y_stride` — destination luma plane.
/// * `uv_dst` / `uv_stride` — destination chroma plane (interleaved
///   `U`/`V`, half height, full byte width).
pub(crate) fn bgra_to_nv12(
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

    // The yuv crate borrows both destination planes mutably inside a
    // single `YuvBiPlanarImageMut`. The driver buffer is already
    // split into two non-overlapping slices by the caller's
    // `split_at_mut`, so we wrap each as a `Borrowed` variant —
    // zero copy, same lifetimes.
    let mut bi = YuvBiPlanarImageMut::<u8> {
        y_plane: BufferStoreMut::Borrowed(y_dst),
        y_stride: y_stride as u32,
        uv_plane: BufferStoreMut::Borrowed(uv_dst),
        uv_stride: uv_stride as u32,
        width,
        height,
    };

    // `bgra_to_yuv_nv12` validates plane lengths against (width,
    // height, strides) and returns YuvError on mismatch. The
    // VA-API caller derives all of those from the same surface
    // geometry it allocated, so an error here is a programmer bug
    // not a runtime condition — surface with an `expect` so it's
    // loud if the invariants ever drift.
    bgra_to_yuv_nv12(
        &mut bi,
        src,
        src_stride as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
        YuvConversionMode::Balanced,
    )
    .expect("BGRA→NV12: plane geometry mismatch (programmer bug)");
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
