use bytes::Bytes;

pub(crate) struct Planar<'a> {
    pub width: usize,
    pub height: usize,
    pub y: &'a [u8],
    pub y_stride: usize,
    pub u: &'a [u8],
    pub v: &'a [u8],
    pub uv_stride: usize,
}

pub(crate) fn from_i420(planes: &Planar) -> Option<(Bytes, usize)> {
    let Planar {
        width,
        height,
        y,
        y_stride,
        u,
        v,
        uv_stride,
    } = *planes;
    let chroma_width = width.div_ceil(2);
    let chroma_height = height.div_ceil(2);
    let covers = |plane: &[u8], stride: usize, rows: usize, row_len: usize| {
        rows == 0 || plane.len() >= (rows - 1) * stride + row_len
    };
    if width == 0
        || height == 0
        || !covers(y, y_stride, height, width)
        || !covers(u, uv_stride, chroma_height, chroma_width)
        || !covers(v, uv_stride, chroma_height, chroma_width)
    {
        return None;
    }
    let stride = chroma_width * 2;
    let mut out = vec![0; stride * (height + chroma_height)];
    for (row, line) in out.chunks_exact_mut(stride).take(height).enumerate() {
        line[..width].copy_from_slice(&y[row * y_stride..][..width]);
    }
    for (row, line) in out.chunks_exact_mut(stride).skip(height).enumerate() {
        let u = &u[row * uv_stride..][..chroma_width];
        let v = &v[row * uv_stride..][..chroma_width];
        let (pairs, _) = line.as_chunks_mut::<2>();
        for (pair, (u, v)) in pairs.iter_mut().zip(u.iter().zip(v)) {
            *pair = [*u, *v];
        }
    }
    Some((Bytes::from(out), stride))
}

#[cfg(feature = "hevc-decode")]
pub(crate) fn narrow_u16(
    plane: &[u16],
    stride: usize,
    width: usize,
    height: usize,
    depth: u32,
) -> Option<Vec<u8>> {
    let shift = depth.saturating_sub(8);
    let mut out = Vec::with_capacity(width * height);
    for row in 0..height {
        let line = plane.get(row * stride..)?.get(..width)?;
        out.extend(line.iter().map(|&sample| (sample >> shift) as u8));
    }
    Some(out)
}

#[cfg(any(feature = "vp9-decode", feature = "av1-decode"))]
pub(crate) fn narrow_le_bytes(
    plane: &[u8],
    stride: usize,
    width: usize,
    height: usize,
    depth: u32,
) -> Option<Vec<u8>> {
    let shift = depth.saturating_sub(8);
    let mut out = Vec::with_capacity(width * height);
    for row in 0..height {
        let line = plane.get(row * stride..)?.get(..width * 2)?;
        let (pairs, _) = line.as_chunks::<2>();
        out.extend(
            pairs
                .iter()
                .map(|pair| (u16::from_le_bytes(*pair) >> shift) as u8),
        );
    }
    Some(out)
}

#[cfg(feature = "vaapi-decode")]
pub(crate) struct Nv12Planes<'a> {
    pub y: &'a [u8],
    pub y_pitch: usize,
    pub uv: &'a [u8],
    pub uv_pitch: usize,
}

#[cfg(feature = "vaapi-decode")]
pub(crate) fn crop(src: &Nv12Planes, rect: (usize, usize, usize, usize)) -> Option<(Bytes, usize)> {
    let (x, y, width, height) = rect;
    if width == 0 || height == 0 || x % 2 != 0 || y % 2 != 0 {
        return None;
    }
    let chroma_width = width.div_ceil(2);
    let chroma_height = height.div_ceil(2);
    let stride = chroma_width * 2;
    let mut out = Vec::with_capacity(stride * (height + chroma_height));
    for row in 0..height {
        let start = (y + row) * src.y_pitch + x;
        out.extend_from_slice(src.y.get(start..start + width)?);
        out.resize(out.len() + stride - width, 0);
    }
    for row in 0..chroma_height {
        let start = (y / 2 + row) * src.uv_pitch + x;
        out.extend_from_slice(src.uv.get(start..start + stride)?);
    }
    Some((Bytes::from(out), stride))
}

#[cfg(feature = "vaapi-decode")]
pub(crate) fn to_bgra(data: &[u8], stride: usize, width: usize, height: usize) -> Option<Vec<u8>> {
    let (luma, chroma) = data.split_at_checked(stride * height)?;
    let mut out = vec![0; width * height * 4];
    for (row, line) in out.chunks_exact_mut(width * 4).enumerate() {
        let luma_row = luma.get(row * stride..)?.get(..width)?;
        let chroma_row = chroma.get(row / 2 * stride..)?.get(..stride)?;
        let (pixels, _) = line.as_chunks_mut::<4>();
        for (column, pixel) in pixels.iter_mut().enumerate() {
            let c = i32::from(luma_row[column]) - 16;
            let d = i32::from(chroma_row[column / 2 * 2]) - 128;
            let e = i32::from(chroma_row[column / 2 * 2 + 1]) - 128;
            pixel[0] = ((298 * c + 516 * d + 128) >> 8).clamp(0, 255) as u8;
            pixel[1] = ((298 * c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255) as u8;
            pixel[2] = ((298 * c + 409 * e + 128) >> 8).clamp(0, 255) as u8;
            pixel[3] = 255;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "hevc-decode")]
    #[test]
    fn ten_bit_samples_keep_their_top_eight_bits() {
        let plane = [0u16, 1023, 512, 4, 9, 9];
        assert_eq!(narrow_u16(&plane, 3, 2, 2, 10).unwrap(), [0, 255, 1, 2]);
    }

    #[cfg(feature = "hevc-decode")]
    #[test]
    fn eight_bit_samples_are_left_alone() {
        assert_eq!(narrow_u16(&[7, 200], 2, 2, 1, 8).unwrap(), [7, 200]);
    }

    #[cfg(feature = "hevc-decode")]
    #[test]
    fn a_sample_plane_shorter_than_its_rows_is_rejected() {
        assert!(narrow_u16(&[1, 2, 3], 2, 2, 2, 8).is_none());
    }

    #[cfg(any(feature = "vp9-decode", feature = "av1-decode"))]
    #[test]
    fn little_endian_bytes_are_narrowed_like_samples() {
        let plane = [0xFF, 0x03, 0x00, 0x02, 9, 9, 9, 9];
        assert_eq!(narrow_le_bytes(&plane, 4, 2, 1, 10).unwrap(), [255, 128]);
    }

    #[cfg(any(feature = "vp9-decode", feature = "av1-decode"))]
    #[test]
    fn a_byte_plane_shorter_than_its_rows_is_rejected() {
        assert!(narrow_le_bytes(&[1, 2, 3], 2, 2, 1, 10).is_none());
    }

    #[test]
    fn chroma_planes_are_interleaved_after_the_luma_rows() {
        let y = [1, 2, 3, 4, 5, 6, 7, 8];
        let (data, stride) = from_i420(&Planar {
            width: 4,
            height: 2,
            y: &y,
            y_stride: 4,
            u: &[10, 11],
            v: &[20, 21],
            uv_stride: 2,
        })
        .unwrap();
        assert_eq!(stride, 4);
        assert_eq!(&data[..], &[1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 11, 21]);
    }

    #[test]
    fn row_padding_in_the_source_is_dropped() {
        let y = [1, 2, 9, 9, 3, 4, 9, 9];
        let (data, _) = from_i420(&Planar {
            width: 2,
            height: 2,
            y: &y,
            y_stride: 4,
            u: &[10, 9, 9],
            v: &[20, 9, 9],
            uv_stride: 3,
        })
        .unwrap();
        assert_eq!(&data[..], &[1, 2, 3, 4, 10, 20]);
    }

    #[test]
    fn odd_sizes_round_the_chroma_up() {
        let (data, stride) = from_i420(&Planar {
            width: 3,
            height: 3,
            y: &[1; 9],
            y_stride: 3,
            u: &[10, 11, 12, 13],
            v: &[20, 21, 22, 23],
            uv_stride: 2,
        })
        .unwrap();
        assert_eq!(stride, 4);
        assert_eq!(data.len(), 4 * 3 + 4 * 2);
        assert_eq!(&data[12..], &[10, 20, 11, 21, 12, 22, 13, 23]);
    }

    #[test]
    fn planes_shorter_than_the_picture_are_rejected() {
        assert!(
            from_i420(&Planar {
                width: 4,
                height: 2,
                y: &[0; 7],
                y_stride: 4,
                u: &[0; 2],
                v: &[0; 2],
                uv_stride: 2,
            })
            .is_none()
        );
    }

    #[cfg(feature = "vaapi-decode")]
    #[test]
    fn a_crop_keeps_the_requested_rectangle_of_both_planes() {
        let y: Vec<u8> = (0..24).collect();
        let uv: Vec<u8> = (100..112).collect();
        let (data, stride) = crop(
            &Nv12Planes {
                y: &y,
                y_pitch: 6,
                uv: &uv,
                uv_pitch: 6,
            },
            (2, 2, 2, 2),
        )
        .unwrap();
        assert_eq!(stride, 2);
        assert_eq!(&data[..], &[14, 15, 20, 21, 108, 109]);
    }

    #[cfg(feature = "vaapi-decode")]
    #[test]
    fn a_crop_outside_the_planes_or_on_an_odd_origin_is_rejected() {
        let planes = Nv12Planes {
            y: &[0; 8],
            y_pitch: 4,
            uv: &[0; 4],
            uv_pitch: 4,
        };
        assert!(crop(&planes, (0, 0, 4, 4)).is_none());
        assert!(crop(&planes, (1, 0, 2, 2)).is_none());
    }

    #[cfg(feature = "vaapi-decode")]
    #[test]
    fn limited_range_white_and_black_convert_to_full_range_bgra() {
        let nv12 = [235, 235, 16, 16, 128, 128];
        let pixels = to_bgra(&nv12, 2, 2, 2).unwrap();
        assert_eq!(&pixels[..4], &[255, 255, 255, 255]);
        assert_eq!(&pixels[8..12], &[0, 0, 0, 255]);
    }
}
