//! Captured frames to NV12, the input layout platform encoders
//! (Media Foundation, `MediaCodec`) take, converted with the matrix and
//! range the encoder configuration signals.

use ferricast_core::{
    BackendId, CodecError, CodecResult, ColourDescription, ColourRange, MatrixCoefficients,
    PixelFormat,
};
use yuv::{BufferStoreMut, YuvBiPlanarImageMut, YuvConversionMode, YuvRange, YuvStandardMatrix};

use crate::frame_view::FrameView;

/// Where an encoder wants NV12 laid out in its input buffer: luma rows
/// `stride` bytes apart, the interleaved chroma plane starting at
/// `chroma_offset` with the same stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Nv12Layout {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) stride: usize,
    pub(crate) chroma_offset: usize,
}

impl Nv12Layout {
    /// Tightly packed NV12.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "conditional: only the Media Foundation encoder packs tightly")
    )]
    pub(crate) fn packed(width: u32, height: u32) -> Self {
        let stride = width as usize;
        Self {
            width,
            height,
            stride,
            chroma_offset: stride * height as usize,
        }
    }

    /// Bytes the layout needs.
    pub(crate) fn len(&self) -> usize {
        self.chroma_offset + self.stride * self.height.div_ceil(2) as usize
    }
}

/// The `yuv` crate matrix and range that produce `colour`'s YCbCr.
///
/// # Errors
/// [`CodecError::UnsupportedConfig`] for a matrix that is not a YCbCr
/// conversion of RGB this crate implements (identity, YCgCo, ICtCp,
/// chroma-derived, ST 2085).
pub(crate) fn conversion(
    colour: ColourDescription,
    backend: BackendId,
) -> CodecResult<(YuvRange, YuvStandardMatrix)> {
    let matrix = match colour.matrix {
        MatrixCoefficients::Bt709 | MatrixCoefficients::Unspecified => YuvStandardMatrix::Bt709,
        MatrixCoefficients::Smpte170M => YuvStandardMatrix::Bt601,
        MatrixCoefficients::Bt470Bg => YuvStandardMatrix::Bt470_6,
        MatrixCoefficients::Fcc => YuvStandardMatrix::Fcc,
        MatrixCoefficients::Smpte240M => YuvStandardMatrix::Smpte240,
        MatrixCoefficients::Bt2020Ncl => YuvStandardMatrix::Bt2020,
        unsupported @ (MatrixCoefficients::Identity
        | MatrixCoefficients::YCgCo
        | MatrixCoefficients::Bt2020Cl
        | MatrixCoefficients::Smpte2085
        | MatrixCoefficients::ChromaDerivedNcl
        | MatrixCoefficients::ChromaDerivedCl
        | MatrixCoefficients::ICtCp) => {
            return Err(CodecError::UnsupportedConfig {
                backend,
                what: format!("RGB input converted with the {unsupported:?} matrix"),
            });
        }
    };
    let range = match colour.range {
        ColourRange::Limited => YuvRange::Limited,
        ColourRange::Full => YuvRange::Full,
    };
    Ok((range, matrix))
}

/// Writes `src` into `dst` as NV12 laid out as `layout`, converting RGB
/// with `colour`. YUV input is copied as is (its colour is the
/// producer's).
///
/// # Errors
/// [`CodecError::UnsupportedConfig`] for a matrix [`conversion`] lacks;
/// [`CodecError::Backend`] (naming `backend`) when `dst` is shorter than
/// the layout or `src` does not match it.
pub(crate) fn write(
    src: &FrameView<'_>,
    colour: ColourDescription,
    layout: Nv12Layout,
    dst: &mut [u8],
    backend: BackendId,
) -> CodecResult<()> {
    let fail =
        |reason: String| CodecError::backend(backend, ferricast_core::Operation::Encode, reason);
    if src.width() != layout.width || src.height() != layout.height {
        return Err(fail(format!(
            "frame {}x{} does not match the {}x{} input buffer",
            src.width(),
            src.height(),
            layout.width,
            layout.height
        )));
    }
    if dst.len() < layout.len() || layout.chroma_offset < layout.stride * layout.height as usize {
        return Err(fail(format!(
            "NV12 input buffer of {} bytes is too small for {layout:?}",
            dst.len()
        )));
    }
    let (luma, chroma) = dst.split_at_mut(layout.chroma_offset);
    let width = layout.width as usize;
    let chroma_width = layout.width.div_ceil(2) as usize * 2;
    match src.format() {
        PixelFormat::Bgra | PixelFormat::Rgba => {
            let (range, matrix) = conversion(colour, backend)?;
            let stride =
                u32::try_from(layout.stride).map_err(|_| fail("stride exceeds u32".into()))?;
            let mut image = YuvBiPlanarImageMut {
                y_plane: BufferStoreMut::Borrowed(luma),
                y_stride: stride,
                uv_plane: BufferStoreMut::Borrowed(chroma),
                uv_stride: stride,
                width: layout.width,
                height: layout.height,
            };
            let src_stride = u32::try_from(src.stride())
                .map_err(|_| fail("source stride exceeds u32".into()))?;
            let converted = if src.format() == PixelFormat::Bgra {
                yuv::bgra_to_yuv_nv12(
                    &mut image,
                    src.data(),
                    src_stride,
                    range,
                    matrix,
                    YuvConversionMode::Balanced,
                )
            } else {
                yuv::rgba_to_yuv_nv12(
                    &mut image,
                    src.data(),
                    src_stride,
                    range,
                    matrix,
                    YuvConversionMode::Balanced,
                )
            };
            converted.map_err(|error| fail(format!("RGB to NV12 conversion: {error}")))
        }
        PixelFormat::Nv12 => {
            copy_plane(
                luma,
                layout.stride,
                src.data(),
                src.stride(),
                width,
                layout.height as usize,
            );
            copy_plane(
                chroma,
                layout.stride,
                &src.data()[src.chroma_offset()..],
                src.chroma_stride(),
                chroma_width,
                src.chroma_rows(),
            );
            Ok(())
        }
        PixelFormat::I420 => {
            copy_plane(
                luma,
                layout.stride,
                src.data(),
                src.stride(),
                width,
                layout.height as usize,
            );
            let u = &src.data()[src.chroma_offset()..];
            let v = &src.data()[src.v_offset()..];
            let half = layout.width.div_ceil(2) as usize;
            for ((row, u), v) in chroma
                .chunks_mut(layout.stride)
                .zip(u.chunks(src.chroma_stride()))
                .zip(v.chunks(src.chroma_stride()))
                .take(src.chroma_rows())
            {
                for ((pair, &u), &v) in row[..half * 2]
                    .chunks_exact_mut(2)
                    .zip(&u[..half])
                    .zip(&v[..half])
                {
                    pair[0] = u;
                    pair[1] = v;
                }
            }
            Ok(())
        }
    }
}

/// Copies `rows` rows of `row_bytes` bytes between two strides.
fn copy_plane(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    row_bytes: usize,
    rows: usize,
) {
    for (out, line) in dst
        .chunks_mut(dst_stride)
        .zip(src.chunks(src_stride))
        .take(rows)
    {
        out[..row_bytes].copy_from_slice(&line[..row_bytes]);
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use ferricast_core::RawFrame;

    use super::*;

    fn raw(format: PixelFormat, data: Vec<u8>, stride: u32) -> RawFrame {
        RawFrame {
            width: 2,
            height: 2,
            stride,
            format,
            data: Bytes::from(data),
            timestamp_us: 0,
            colour: None,
        }
    }

    #[test]
    fn white_bgra_becomes_limited_range_white_nv12() {
        let frame = raw(PixelFormat::Bgra, vec![255; 16], 8);
        let view = FrameView::new(&frame, 2, 2).unwrap();
        let layout = Nv12Layout::packed(2, 2);
        let mut out = vec![0; layout.len()];
        write(
            &view,
            ColourDescription::BT709_LIMITED,
            layout,
            &mut out,
            BackendId::MediaCodec,
        )
        .unwrap();
        assert!(
            out[..4].iter().all(|&y| (234..=236).contains(&y)),
            "{out:?}"
        );
        assert!(
            out[4..].iter().all(|&c| (127..=129).contains(&c)),
            "{out:?}"
        );
    }

    #[test]
    fn i420_is_interleaved_into_a_padded_layout() {
        let frame = raw(PixelFormat::I420, vec![1, 2, 3, 4, 5, 6], 2);
        let view = FrameView::new(&frame, 2, 2).unwrap();
        let layout = Nv12Layout {
            width: 2,
            height: 2,
            stride: 4,
            chroma_offset: 16,
        };
        let mut out = vec![0; layout.len()];
        write(
            &view,
            ColourDescription::BT709_LIMITED,
            layout,
            &mut out,
            BackendId::MediaCodec,
        )
        .unwrap();
        assert_eq!(&out[..6], &[1, 2, 0, 0, 3, 4]);
        assert_eq!(&out[16..18], &[5, 6]);
    }

    #[test]
    fn an_unsupported_matrix_is_an_explicit_error() {
        let ictcp = ColourDescription {
            matrix: MatrixCoefficients::ICtCp,
            ..ColourDescription::BT709_LIMITED
        };
        assert!(matches!(
            conversion(ictcp, BackendId::MediaFoundation),
            Err(CodecError::UnsupportedConfig { .. })
        ));
    }
}
