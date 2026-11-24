//! Packs YUV 4:2:0 pictures handed out plane by plane (an `AImage`, a
//! locked `AHardwareBuffer`, a `MediaCodec` byte buffer) into the
//! semi-planar layout core documents for [`PixelFormat::Nv12`] and
//! [`PixelFormat::P010`]: luma rows `stride` bytes apart, then the
//! interleaved Cb/Cr rows with the same stride.
//!
//! The sample depth can change on the way: 10-bit samples (P010, value
//! in the high 10 bits of a little-endian `u16`) narrow to their high
//! byte, 8-bit samples widen into the high byte.
//!
//! [`PixelFormat::Nv12`]: ferricast_core::PixelFormat::Nv12
//! [`PixelFormat::P010`]: ferricast_core::PixelFormat::P010

use ferricast_core::BitDepth;

/// One plane as the platform describes it: samples `pixel_stride` bytes
/// apart, rows `row_stride` bytes apart, each sample `1` (8-bit) or `2`
/// (P010) bytes wide.
#[derive(Debug, Clone, Copy)]
pub struct Plane<'a> {
    /// The plane, from its first sample.
    pub data: &'a [u8],
    /// Bytes from one row to the next.
    pub row_stride: usize,
    /// Bytes from one sample to the next in a row.
    pub pixel_stride: usize,
}

impl<'a> Plane<'a> {
    /// The `bytes`-wide sample at `row`, `column`.
    fn sample(
        &self,
        bytes: usize,
        name: &'static str,
        row: usize,
        column: usize,
    ) -> Result<&'a [u8], PackError> {
        let at = row * self.row_stride + column * self.pixel_stride;
        self.data.get(at..at + bytes).ok_or(PackError::ShortPlane {
            plane: name,
            row,
            sample: column,
        })
    }
}

/// The visible rectangle, in luma samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Crop {
    /// First visible column.
    pub left: usize,
    /// First visible row.
    pub top: usize,
    /// Visible columns.
    pub width: usize,
    /// Visible rows.
    pub height: usize,
}

impl Crop {
    /// The whole `width` × `height` picture.
    #[must_use]
    pub fn full(width: usize, height: usize) -> Self {
        Self {
            left: 0,
            top: 0,
            width,
            height,
        }
    }
}

/// Where the packed picture goes: luma rows `stride` bytes apart, the
/// interleaved chroma plane from `chroma_offset` with the same stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemiPlanar {
    /// Bytes from one row to the next, in both planes.
    pub stride: usize,
    /// Byte offset of the first chroma row.
    pub chroma_offset: usize,
}

/// Bytes per sample at `depth`.
#[must_use]
pub fn sample_bytes(depth: BitDepth) -> usize {
    match depth {
        BitDepth::Eight => 1,
        BitDepth::Ten => 2,
    }
}

impl SemiPlanar {
    /// Tightly packed rows for a `width` × `height` picture: one
    /// interleaved chroma row of an odd width is a sample pair longer
    /// than its luma row, so rows hold `ceil(width / 2)` pairs.
    #[must_use]
    pub fn packed(width: usize, height: usize, depth: BitDepth) -> Self {
        let stride = width.div_ceil(2) * 2 * sample_bytes(depth);
        Self {
            stride,
            chroma_offset: stride * height,
        }
    }

    /// Bytes a picture `height` rows tall needs in this layout.
    #[must_use]
    pub fn len(&self, height: usize) -> usize {
        self.chroma_offset + self.stride * height.div_ceil(2)
    }
}

/// Why a picture could not be packed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PackError {
    /// A plane is shorter than its strides and the crop say.
    #[error("the {plane} plane ends before row {row}, sample {sample}")]
    ShortPlane {
        /// `luma`, `Cb`, `Cr` or `chroma`.
        plane: &'static str,
        /// Source row.
        row: usize,
        /// Source sample in the row.
        sample: usize,
    },
    /// The destination does not hold the packed picture.
    #[error("the destination holds {have} bytes, {need} needed for {layout:?}")]
    ShortDestination {
        /// Bytes given.
        have: usize,
        /// Bytes needed.
        need: usize,
        /// The layout asked for.
        layout: SemiPlanar,
    },
    /// The destination rows are narrower than one packed row.
    #[error("destination stride {stride} is shorter than one {need}-byte row")]
    NarrowStride {
        /// The layout stride.
        stride: usize,
        /// Bytes of one packed row.
        need: usize,
    },
}

/// Copies one sample of `from` depth into `to` depth.
fn put(dst: &mut [u8], src: &[u8], from: BitDepth, to: BitDepth) {
    match (from, to) {
        (BitDepth::Eight, BitDepth::Eight) => dst[0] = src[0],
        (BitDepth::Ten, BitDepth::Ten) => dst[..2].copy_from_slice(&src[..2]),
        // P010 keeps the value in the high 10 bits: its high byte is
        // the 8-bit sample.
        (BitDepth::Ten, BitDepth::Eight) => dst[0] = src[1],
        (BitDepth::Eight, BitDepth::Ten) => {
            dst[0] = 0;
            dst[1] = src[0];
        }
    }
}

/// Packs `y`, `u` (Cb) and `v` (Cr) at `from` depth into `dst` at `to`
/// depth, laid out as `layout`, keeping only `crop`.
///
/// # Errors
/// [`PackError`] when a plane or `dst` is too short, or `layout` rows
/// are narrower than one packed row; `dst` is then partly written.
pub fn pack(
    [y, u, v]: [Plane<'_>; 3],
    crop: Crop,
    from: BitDepth,
    to: BitDepth,
    layout: SemiPlanar,
    dst: &mut [u8],
) -> Result<(), PackError> {
    let (src_bytes, dst_bytes) = (sample_bytes(from), sample_bytes(to));
    let pairs = crop.width.div_ceil(2);
    let row_bytes = pairs * 2 * dst_bytes;
    if layout.stride < row_bytes {
        return Err(PackError::NarrowStride {
            stride: layout.stride,
            need: row_bytes,
        });
    }
    let need = layout.len(crop.height);
    if dst.len() < need || layout.chroma_offset < layout.stride * crop.height {
        return Err(PackError::ShortDestination {
            have: dst.len(),
            need,
            layout,
        });
    }
    let (luma, chroma) = dst.split_at_mut(layout.chroma_offset);
    for (row, line) in luma.chunks_mut(layout.stride).take(crop.height).enumerate() {
        let source_row = crop.top + row;
        if from == to && y.pixel_stride == src_bytes {
            let start = source_row * y.row_stride + crop.left * src_bytes;
            let len = crop.width * src_bytes;
            let src = y
                .data
                .get(start..start + len)
                .ok_or(PackError::ShortPlane {
                    plane: "luma",
                    row: source_row,
                    sample: crop.left + crop.width - 1,
                })?;
            line[..len].copy_from_slice(src);
            continue;
        }
        for (column, out) in line
            .chunks_exact_mut(dst_bytes)
            .take(crop.width)
            .enumerate()
        {
            put(
                out,
                y.sample(src_bytes, "luma", source_row, crop.left + column)?,
                from,
                to,
            );
        }
    }
    let (chroma_top, chroma_left) = (crop.top / 2, crop.left / 2);
    for (row, line) in chroma
        .chunks_mut(layout.stride)
        .take(crop.height.div_ceil(2))
        .enumerate()
    {
        let source_row = chroma_top + row;
        for (column, pair) in line.chunks_exact_mut(2 * dst_bytes).take(pairs).enumerate() {
            let (cb, cr) = pair.split_at_mut(dst_bytes);
            put(
                cb,
                u.sample(src_bytes, "Cb", source_row, chroma_left + column)?,
                from,
                to,
            );
            put(
                cr,
                v.sample(src_bytes, "Cr", source_row, chroma_left + column)?,
                from,
                to,
            );
        }
    }
    Ok(())
}

/// The three planes of a semi-planar (NV12 / P010) buffer whose luma
/// rows are `stride` bytes apart and whose interleaved chroma starts at
/// `chroma_offset` with the same stride.
///
/// # Errors
/// [`PackError::ShortPlane`] when `data` ends before `chroma_offset`.
pub fn semi_planar(
    data: &[u8],
    stride: usize,
    chroma_offset: usize,
    depth: BitDepth,
) -> Result<[Plane<'_>; 3], PackError> {
    let bytes = sample_bytes(depth);
    let chroma = data.get(chroma_offset..).ok_or(PackError::ShortPlane {
        plane: "chroma",
        row: 0,
        sample: 0,
    })?;
    let cr = chroma.get(bytes..).unwrap_or_default();
    Ok([
        Plane {
            data,
            row_stride: stride,
            pixel_stride: bytes,
        },
        Plane {
            data: chroma,
            row_stride: stride,
            pixel_stride: 2 * bytes,
        },
        Plane {
            data: cr,
            row_stride: stride,
            pixel_stride: 2 * bytes,
        },
    ])
}

/// The three planes of an 8-bit planar (I420) buffer: luma rows `stride`
/// bytes apart over `slice_height` rows, then Cb and Cr with half the
/// stride and half the rows.
///
/// # Errors
/// [`PackError::ShortPlane`] when `data` ends before the Cr plane.
pub fn planar(
    data: &[u8],
    stride: usize,
    slice_height: usize,
) -> Result<[Plane<'_>; 3], PackError> {
    let short = |plane| PackError::ShortPlane {
        plane,
        row: 0,
        sample: 0,
    };
    let (chroma_stride, chroma_rows) = (stride / 2, slice_height.div_ceil(2));
    let cb = data.get(stride * slice_height..).ok_or(short("Cb"))?;
    let cr = cb.get(chroma_stride * chroma_rows..).ok_or(short("Cr"))?;
    let plane = |data, row_stride| Plane {
        data,
        row_stride,
        pixel_stride: 1,
    };
    Ok([
        plane(data, stride),
        plane(cb, chroma_stride),
        plane(cr, chroma_stride),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4x2 NV12 picture with an 8-byte stride (luma 1..=8 per row).
    fn nv12() -> Vec<u8> {
        let mut data = vec![0; 8 * 3];
        data[..4].copy_from_slice(&[1, 2, 3, 4]);
        data[8..12].copy_from_slice(&[5, 6, 7, 8]);
        data[16..20].copy_from_slice(&[10, 20, 11, 21]);
        data
    }

    #[test]
    fn semi_planar_input_repacks_tightly() {
        let data = nv12();
        let planes = semi_planar(&data, 8, 16, BitDepth::Eight).unwrap();
        let layout = SemiPlanar::packed(4, 2, BitDepth::Eight);
        let mut out = vec![0; layout.len(2)];
        pack(
            planes,
            Crop::full(4, 2),
            BitDepth::Eight,
            BitDepth::Eight,
            layout,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 11, 21]);
    }

    #[test]
    fn eight_bits_widen_to_p010_and_narrow_back() {
        let data = nv12();
        let planes = semi_planar(&data, 8, 16, BitDepth::Eight).unwrap();
        let wide = SemiPlanar::packed(4, 2, BitDepth::Ten);
        let mut p010 = vec![0; wide.len(2)];
        pack(
            planes,
            Crop::full(4, 2),
            BitDepth::Eight,
            BitDepth::Ten,
            wide,
            &mut p010,
        )
        .unwrap();
        assert_eq!(&p010[..4], &[0, 1, 0, 2]);
        assert_eq!(&p010[16..20], &[0, 10, 0, 20]);

        let planes = semi_planar(&p010, wide.stride, wide.chroma_offset, BitDepth::Ten).unwrap();
        let narrow = SemiPlanar::packed(4, 2, BitDepth::Eight);
        let mut back = vec![0; narrow.len(2)];
        pack(
            planes,
            Crop::full(4, 2),
            BitDepth::Ten,
            BitDepth::Eight,
            narrow,
            &mut back,
        )
        .unwrap();
        assert_eq!(back, [1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 11, 21]);
    }

    #[test]
    fn planar_input_interleaves_and_crops() {
        // 4x4 I420: luma 0..16, Cb 100.., Cr 200..
        let mut data: Vec<u8> = (0..16).collect();
        data.extend([100, 101, 102, 103]);
        data.extend([200, 201, 202, 203]);
        let planes = planar(&data, 4, 4).unwrap();
        let crop = Crop {
            left: 2,
            top: 2,
            width: 2,
            height: 2,
        };
        let layout = SemiPlanar::packed(2, 2, BitDepth::Eight);
        let mut out = vec![0; layout.len(2)];
        pack(
            planes,
            crop,
            BitDepth::Eight,
            BitDepth::Eight,
            layout,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [10, 11, 14, 15, 103, 203]);
    }

    #[test]
    fn odd_widths_get_a_full_chroma_pair() {
        let data = vec![7; 4 * 3];
        let planes = semi_planar(&data, 4, 8, BitDepth::Eight).unwrap();
        let layout = SemiPlanar::packed(3, 2, BitDepth::Eight);
        assert_eq!(layout.stride, 4);
        let mut out = vec![0; layout.len(2)];
        pack(
            planes,
            Crop::full(3, 2),
            BitDepth::Eight,
            BitDepth::Eight,
            layout,
            &mut out,
        )
        .unwrap();
        assert_eq!(&out[8..], &[7, 7, 7, 7]);
    }

    #[test]
    fn short_inputs_are_errors_not_panics() {
        let data = nv12();
        let planes = semi_planar(&data[..18], 8, 16, BitDepth::Eight).unwrap();
        let layout = SemiPlanar::packed(4, 2, BitDepth::Eight);
        let mut out = vec![0; layout.len(2)];
        assert!(matches!(
            pack(
                planes,
                Crop::full(4, 2),
                BitDepth::Eight,
                BitDepth::Eight,
                layout,
                &mut out,
            ),
            Err(PackError::ShortPlane { .. })
        ));
        let mut small = vec![0; 3];
        let planes = semi_planar(&data, 8, 16, BitDepth::Eight).unwrap();
        assert!(matches!(
            pack(
                planes,
                Crop::full(4, 2),
                BitDepth::Eight,
                BitDepth::Eight,
                layout,
                &mut small,
            ),
            Err(PackError::ShortDestination { .. })
        ));
        assert!(semi_planar(&data, 8, 99, BitDepth::Eight).is_err());
    }
}
