//! P010 input for 10-bit platform encoders: 16-bit little-endian
//! samples with the 10-bit code in the high bits, luma rows then
//! interleaved chroma, laid out as an [`Nv12Layout`] whose strides are
//! in bytes.

use ferricast_core::{BackendId, CodecError, CodecResult, Operation, PixelFormat};

use crate::frame_view::FrameView;
use crate::nv12::Nv12Layout;

/// Fills `dst` with the little-endian `value`.
pub(crate) fn fill_u16(dst: &mut [u8], value: u16) {
    for pair in dst.chunks_exact_mut(2) {
        pair.copy_from_slice(&value.to_le_bytes());
    }
}

/// Writes `src` into `dst` as P010 laid out as `layout` (strides in
/// bytes): P010 is copied, NV12 / I420 samples are widened (`v << 8`,
/// the 8-bit code in the high bits). RGB input has no 10-bit
/// conversion here.
pub(crate) fn write_p010(
    src: &FrameView<'_>,
    layout: Nv12Layout,
    dst: &mut [u8],
    backend: BackendId,
) -> CodecResult<()> {
    let fail = |reason: String| CodecError::backend(backend, Operation::Encode, reason);
    if dst.len() < layout.len() {
        return Err(fail(format!(
            "P010 input buffer of {} bytes is too small for {layout:?}",
            dst.len()
        )));
    }
    let width = layout.width as usize;
    let height = layout.height as usize;
    let pairs = layout.width.div_ceil(2) as usize;
    if layout.stride < pairs * 4 {
        return Err(fail(format!(
            "P010 stride {} is shorter than one {}-byte row",
            layout.stride,
            pairs * 4
        )));
    }
    let (luma, chroma) = dst.split_at_mut(layout.chroma_offset);
    let data = src.data();
    let widen = |out: &mut [u8], samples: &mut dyn Iterator<Item = u8>| {
        for (pair, sample) in out.chunks_exact_mut(2).zip(samples) {
            pair.copy_from_slice(&(u16::from(sample) << 8).to_le_bytes());
        }
    };
    match src.format() {
        PixelFormat::P010 => {
            for (out, line) in luma
                .chunks_mut(layout.stride)
                .zip(data.chunks(src.stride()))
                .take(height)
            {
                out[..width * 2].copy_from_slice(&line[..width * 2]);
            }
            for (out, line) in chroma
                .chunks_mut(layout.stride)
                .zip(data[src.chroma_offset()..].chunks(src.chroma_stride()))
                .take(src.chroma_rows())
            {
                out[..pairs * 4].copy_from_slice(&line[..pairs * 4]);
            }
        }
        PixelFormat::Nv12 => {
            for (out, line) in luma
                .chunks_mut(layout.stride)
                .zip(data.chunks(src.stride()))
                .take(height)
            {
                widen(&mut out[..width * 2], &mut line[..width].iter().copied());
            }
            for (out, line) in chroma
                .chunks_mut(layout.stride)
                .zip(data[src.chroma_offset()..].chunks(src.chroma_stride()))
                .take(src.chroma_rows())
            {
                widen(
                    &mut out[..pairs * 4],
                    &mut line[..pairs * 2].iter().copied(),
                );
            }
        }
        PixelFormat::I420 => {
            for (out, line) in luma
                .chunks_mut(layout.stride)
                .zip(data.chunks(src.stride()))
                .take(height)
            {
                widen(&mut out[..width * 2], &mut line[..width].iter().copied());
            }
            let u = &data[src.chroma_offset()..];
            let v = &data[src.v_offset()..];
            for ((out, u), v) in chroma
                .chunks_mut(layout.stride)
                .zip(u.chunks(src.chroma_stride()))
                .zip(v.chunks(src.chroma_stride()))
                .take(src.chroma_rows())
            {
                let mut interleaved = u[..pairs]
                    .iter()
                    .zip(&v[..pairs])
                    .flat_map(|(&u, &v)| [u, v]);
                widen(&mut out[..pairs * 4], &mut interleaved);
            }
        }
        format @ (PixelFormat::Bgra | PixelFormat::Rgba) => {
            return Err(CodecError::UnsupportedInput {
                backend,
                expected: "P010, NV12 or I420 for a 10-bit stream".into(),
                actual: format,
            });
        }
    }
    Ok(())
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
            hdr: None,
        }
    }

    fn layout() -> Nv12Layout {
        Nv12Layout {
            width: 2,
            height: 2,
            stride: 6,
            chroma_offset: 12,
        }
    }

    #[test]
    fn nv12_is_widened_into_the_high_bits() {
        let frame = raw(PixelFormat::Nv12, vec![16, 235, 16, 235, 128, 64], 2);
        let view = FrameView::new(&frame, 2, 2).unwrap();
        let mut out = vec![0xEE; layout().len()];
        write_p010(&view, layout(), &mut out, BackendId::MediaFoundation).unwrap();
        assert_eq!(&out[..4], &[0, 16, 0, 235]);
        assert_eq!(&out[4..6], &[0xEE, 0xEE], "stride padding untouched");
        assert_eq!(&out[12..16], &[0, 128, 0, 64]);
    }

    #[test]
    fn i420_chroma_is_interleaved() {
        let frame = raw(PixelFormat::I420, vec![1, 2, 3, 4, 5, 6], 2);
        let view = FrameView::new(&frame, 2, 2).unwrap();
        let mut out = vec![0; layout().len()];
        write_p010(&view, layout(), &mut out, BackendId::MediaFoundation).unwrap();
        assert_eq!(&out[12..16], &[0, 5, 0, 6]);
    }

    #[test]
    fn p010_is_copied_and_rgb_refused() {
        let samples: Vec<u8> = (0..12).collect();
        let frame = raw(PixelFormat::P010, samples, 4);
        let view = FrameView::new(&frame, 2, 2).unwrap();
        let mut out = vec![0; layout().len()];
        write_p010(&view, layout(), &mut out, BackendId::MediaFoundation).unwrap();
        assert_eq!(&out[..4], &[0, 1, 2, 3]);
        assert_eq!(&out[6..10], &[4, 5, 6, 7]);
        assert_eq!(&out[12..16], &[8, 9, 10, 11]);

        let bgra = raw(PixelFormat::Bgra, vec![0; 16], 8);
        let view = FrameView::new(&bgra, 2, 2).unwrap();
        assert!(matches!(
            write_p010(&view, layout(), &mut out, BackendId::MediaFoundation),
            Err(CodecError::UnsupportedInput { .. })
        ));
    }

    #[test]
    fn fill_writes_little_endian() {
        let mut out = [0; 4];
        fill_u16(&mut out, 512 << 6);
        assert_eq!(out, [0x00, 0x80, 0x00, 0x80]);
    }
}
