//! Captured frames to NV12, the input layout platform encoders
//! (Media Foundation, `MediaCodec`) take, converted with the matrix and
//! range the encoder configuration signals.

use ferricast_core::{
    BackendId, CodecError, CodecResult, ColourDescription, ColourRange, MatrixCoefficients,
    PixelFormat, RawFrame,
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
    /// Tightly packed NV12 (rows rounded up to an even width, the
    /// interleaved chroma row of an odd-width picture is one byte longer).
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "conditional: only the Media Foundation encoder packs tightly"
        )
    )]
    pub(crate) fn packed(width: u32, height: u32) -> Self {
        let stride = width.div_ceil(2) as usize * 2;
        Self {
            width,
            height,
            stride,
            chroma_offset: stride * height as usize,
        }
    }

    /// Tightly packed P010 (16-bit samples): rows of `ceil(width / 2) × 4`
    /// bytes, for [`write_p010`].
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "conditional: only the 10-bit platform encoders take P010"
        )
    )]
    pub(crate) fn packed_p010(width: u32, height: u32) -> Self {
        let stride = width.div_ceil(2) as usize * 4;
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
    let width = layout.width as usize;
    let chroma_width = layout.width.div_ceil(2) as usize * 2;
    if layout.stride < chroma_width {
        return Err(fail(format!(
            "NV12 stride {} is shorter than one {chroma_width}-byte chroma row",
            layout.stride
        )));
    }
    let (luma, chroma) = dst.split_at_mut(layout.chroma_offset);
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
        // An 8-bit encoder keeps the top eight bits of 10-bit input.
        PixelFormat::P010 => {
            let (data, stride) = src
                .narrowed()
                .ok_or_else(|| fail("P010 frame could not be narrowed".into()))?;
            let narrowed = RawFrame {
                width: src.width(),
                height: src.height(),
                stride: u32::try_from(stride).map_err(|_| fail("stride exceeds u32".into()))?,
                format: PixelFormat::Nv12,
                data: data.into(),
                timestamp_us: 0,
                colour: None,
                hdr: None,
            };
            let view = FrameView::new(&narrowed, src.width(), src.height())
                .map_err(|error| fail(error.to_string()))?;
            write(&view, colour, layout, dst, backend)
        }
    }
}

/// Writes `src` into `dst` as P010 laid out as `layout`, for 10-bit
/// encoders: `layout` is read as for NV12 but with 16-bit samples, so
/// `stride` and `chroma_offset` are in bytes and a luma row holds
/// `2 × width` of them ([`Nv12Layout::packed_p010`] packs tightly).
/// P010 input is copied as is; 8-bit input (RGB converted with `colour`,
/// NV12, I420) is widened, each sample moved to the high byte.
///
/// # Errors
/// As [`write`]: [`CodecError::UnsupportedConfig`] for a matrix
/// [`conversion`] lacks; [`CodecError::Backend`] (naming `backend`) when
/// `dst` is shorter than the layout or `src` does not match it.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "conditional: only the 10-bit platform encoders take P010"
    )
)]
pub(crate) fn write_p010(
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
    let row = layout.width.div_ceil(2) as usize * 4;
    if layout.stride < row
        || dst.len() < layout.len()
        || layout.chroma_offset < layout.stride * layout.height as usize
    {
        return Err(fail(format!(
            "P010 input buffer of {} bytes does not fit {layout:?} ({row}-byte rows)",
            dst.len()
        )));
    }
    let (luma, chroma) = dst.split_at_mut(layout.chroma_offset);
    if src.format() == PixelFormat::P010 {
        copy_plane(
            luma,
            layout.stride,
            src.data(),
            src.stride(),
            layout.width as usize * 2,
            layout.height as usize,
        );
        copy_plane(
            chroma,
            layout.stride,
            &src.data()[src.chroma_offset()..],
            src.chroma_stride(),
            row,
            src.chroma_rows(),
        );
        return Ok(());
    }
    let packed = Nv12Layout::packed(layout.width, layout.height);
    let mut nv12 = vec![0; packed.len()];
    write(src, colour, packed, &mut nv12, backend)?;
    let (nv12_luma, nv12_chroma) = nv12.split_at(packed.chroma_offset);
    let widen = |dst: &mut [u8], dst_stride: usize, src: &[u8], bytes: usize| {
        for (out, line) in dst.chunks_mut(dst_stride).zip(src.chunks(packed.stride)) {
            for (word, &sample) in out[..bytes * 2]
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(&line[..bytes])
            {
                *word = [0, sample];
            }
        }
    };
    widen(luma, layout.stride, nv12_luma, layout.width as usize);
    widen(chroma, layout.stride, nv12_chroma, row / 2);
    Ok(())
}

/// Fills `dst` with the little-endian 16-bit `value` (a P010 sample).
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "conditional: only the Media Foundation encoder primes P010 black"
    )
)]
pub(crate) fn fill_u16(dst: &mut [u8], value: u16) {
    for pair in dst.chunks_exact_mut(2) {
        pair.copy_from_slice(&value.to_le_bytes());
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

    use super::*;

    fn raw(format: PixelFormat, data: Vec<u8>, stride: u32) -> RawFrame {
        RawFrame {
            hdr: None,
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
    fn odd_widths_are_converted_or_refused_never_panic() {
        let frame = RawFrame {
            width: 3,
            ..raw(PixelFormat::Nv12, vec![9; 3 * 2 + 4], 3)
        };
        assert!(
            FrameView::new(&frame, 3, 2).is_err(),
            "NV12 rows of an odd width need a padded stride"
        );
        let padded = RawFrame {
            width: 3,
            ..raw(PixelFormat::Nv12, vec![9; 4 * 2 + 4], 4)
        };
        let view = FrameView::new(&padded, 3, 2).unwrap();
        let layout = Nv12Layout::packed(3, 2);
        let mut out = vec![0; layout.len()];
        write(
            &view,
            ColourDescription::BT709_LIMITED,
            layout,
            &mut out,
            BackendId::MediaCodec,
        )
        .unwrap();
        let narrow = Nv12Layout {
            stride: 3,
            chroma_offset: 6,
            ..layout
        };
        assert!(
            write(
                &view,
                ColourDescription::BT709_LIMITED,
                narrow,
                &mut out,
                BackendId::MediaCodec
            )
            .is_err()
        );
    }

    /// A 2x2 P010 frame: luma words 0x1234, 0xFF00, 0x0040, 0x8000 and
    /// one chroma pair 0x4000 / 0xC000.
    fn p010() -> RawFrame {
        let words: [u16; 6] = [0x1234, 0xFF00, 0x0040, 0x8000, 0x4000, 0xC000];
        raw(
            PixelFormat::P010,
            words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            4,
        )
    }

    #[test]
    fn p010_into_an_eight_bit_buffer_keeps_the_high_bytes() {
        let frame = p010();
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
        assert_eq!(out, [0x12, 0xFF, 0x00, 0x80, 0x40, 0xC0]);
    }

    #[test]
    fn p010_is_copied_and_eight_bit_input_widened_for_ten_bit_buffers() {
        let frame = p010();
        let view = FrameView::new(&frame, 2, 2).unwrap();
        let layout = Nv12Layout {
            stride: 6,
            chroma_offset: 14,
            ..Nv12Layout::packed_p010(2, 2)
        };
        let mut out = vec![0xAA; layout.len()];
        write_p010(
            &view,
            ColourDescription::BT709_LIMITED,
            layout,
            &mut out,
            BackendId::MediaFoundation,
        )
        .unwrap();
        assert_eq!(&out[..4], &frame.data[..4]);
        assert_eq!(&out[6..10], &frame.data[4..8]);
        assert_eq!(&out[14..18], &frame.data[8..12]);

        let nv12 = raw(PixelFormat::Nv12, vec![1, 2, 3, 4, 5, 6], 2);
        let view = FrameView::new(&nv12, 2, 2).unwrap();
        let layout = Nv12Layout::packed_p010(2, 2);
        let mut out = vec![0; layout.len()];
        write_p010(
            &view,
            ColourDescription::BT709_LIMITED,
            layout,
            &mut out,
            BackendId::MediaFoundation,
        )
        .unwrap();
        assert_eq!(out, [0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6]);

        let narrow = Nv12Layout {
            stride: 3,
            ..layout
        };
        assert!(
            write_p010(
                &view,
                ColourDescription::BT709_LIMITED,
                narrow,
                &mut out,
                BackendId::MediaFoundation
            )
            .is_err()
        );
    }

    #[test]
    fn fill_writes_little_endian() {
        let mut out = [0; 4];
        fill_u16(&mut out, 512 << 6);
        assert_eq!(out, [0x00, 0x80, 0x00, 0x80]);
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
