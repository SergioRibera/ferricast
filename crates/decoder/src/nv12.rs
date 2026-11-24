use bytes::Bytes;
use ferricast_core::{ColourDescription, PixelFormat, RawFrame};

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

/// The pixel format a decoder was configured to hand out; NV12, the
/// decoders' native layout, until `configure` says otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OutputFormat(pub PixelFormat);

impl Default for OutputFormat {
    fn default() -> Self {
        Self(PixelFormat::Nv12)
    }
}

/// Hands out an NV12 picture in the caller's requested format; planar
/// requests get NV12, the only planar layout the decoders produce, with
/// its colour description so the consumer can convert it.
pub(crate) fn deliver(
    data: Bytes,
    stride: usize,
    (width, height): (usize, usize),
    timestamp_us: u64,
    format: PixelFormat,
    colour: ColourDescription,
) -> Option<RawFrame> {
    let yuv = colour.yuv(height);
    let (data, stride, format, colour) = match format {
        // 8-bit pictures stay NV12 when P010 is asked for: P010 is the
        // preference for 10-bit streams, not an up-conversion.
        PixelFormat::Nv12 | PixelFormat::I420 | PixelFormat::P010 => {
            (data, stride, PixelFormat::Nv12, Some(colour))
        }
        PixelFormat::Bgra => {
            let bgra = yuv.nv12_to_bgra(&data, stride, width, height)?;
            (Bytes::from(bgra), width * 4, PixelFormat::Bgra, None)
        }
        PixelFormat::Rgba => {
            let mut rgba = yuv.nv12_to_bgra(&data, stride, width, height)?;
            for pixel in rgba.as_chunks_mut::<4>().0 {
                pixel.swap(0, 2);
            }
            (Bytes::from(rgba), width * 4, PixelFormat::Rgba, None)
        }
    };
    Some(RawFrame {
        hdr: None,
        width: u32::try_from(width).ok()?,
        height: u32::try_from(height).ok()?,
        stride: u32::try_from(stride).ok()?,
        format,
        data,
        timestamp_us,
        colour,
    })
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

    #[test]
    fn delivery_converts_only_when_a_packed_format_is_asked_for() {
        let nv12 = Bytes::from_static(&[235, 235, 235, 235, 128, 128]);
        let colour = ColourDescription::BT709_LIMITED;
        let kept = deliver(nv12.clone(), 2, (2, 2), 9, PixelFormat::Nv12, colour).unwrap();
        assert_eq!(
            (kept.format, kept.stride, kept.timestamp_us, kept.colour),
            (PixelFormat::Nv12, 2, 9, Some(colour))
        );
        let bgra = deliver(nv12.clone(), 2, (2, 2), 9, PixelFormat::Bgra, colour).unwrap();
        assert_eq!(
            (bgra.format, bgra.stride, bgra.data.len()),
            (PixelFormat::Bgra, 8, 16)
        );
        let rgba = deliver(nv12, 2, (2, 2), 9, PixelFormat::Rgba, colour).unwrap();
        assert_eq!(rgba.format, PixelFormat::Rgba);
    }
}
