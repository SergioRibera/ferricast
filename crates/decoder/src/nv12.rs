use bytes::Bytes;
use ferricast_core::{
    ColourDescription, ColourRange, DecoderConfig, HdrMetadata, PixelFormat, RawFrame,
};

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

/// What the container signals for a stream, for decoders whose bitstream
/// parser does not expose the colour description or HDR metadata.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(
    not(any(
        feature = "hevc-decode",
        feature = "vp9-decode",
        feature = "openh264-decode",
        all(
            feature = "nvdec-decode",
            any(target_os = "linux", target_os = "windows")
        )
    )),
    allow(
        dead_code,
        reason = "conditional: only the openh264, NVDEC, pure-Rust HEVC and VP9 decoders use it"
    )
)]
pub(crate) struct Container {
    pub colour: Option<ColourDescription>,
    pub hdr: Option<HdrMetadata>,
}

#[cfg_attr(
    not(any(
        feature = "hevc-decode",
        feature = "vp9-decode",
        feature = "openh264-decode",
        all(
            feature = "nvdec-decode",
            any(target_os = "linux", target_os = "windows")
        )
    )),
    allow(
        dead_code,
        reason = "conditional: only the openh264, NVDEC, pure-Rust HEVC and VP9 decoders use it"
    )
)]
impl Container {
    pub(crate) fn of(config: &DecoderConfig) -> Self {
        Self {
            colour: config.colour,
            hdr: config.hdr,
        }
    }

    /// The container's colour, unspecified (limited range) without one.
    pub(crate) fn colour(self) -> ColourDescription {
        self.colour
            .unwrap_or(ColourDescription::unspecified(ColourRange::Limited))
    }
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

/// The high byte of every sample of a P010 picture whose luma rows are
/// followed by its interleaved chroma rows, `stride` bytes apart: the
/// same picture as NV12, rows packed. Returns the bytes and their stride.
pub(crate) fn downshift_p010(
    data: &[u8],
    stride: usize,
    width: usize,
    height: usize,
) -> (Vec<u8>, usize) {
    let row = width.div_ceil(2) * 2;
    let rows = height + height.div_ceil(2);
    let mut out = Vec::with_capacity(row * rows);
    for line in data.chunks(stride).take(rows) {
        out.extend(line.iter().skip(1).step_by(2).take(row));
    }
    (out, row)
}

/// Hands out a P010 picture (luma rows, then interleaved chroma rows,
/// `stride` bytes apart) in the caller's requested format: P010 as is,
/// anything else from its top eight bits through [`deliver`].
#[cfg(all(
    feature = "nvdec-decode",
    any(target_os = "linux", target_os = "windows")
))]
pub(crate) fn deliver_p010(
    data: Bytes,
    stride: usize,
    (width, height): (usize, usize),
    timestamp_us: u64,
    format: PixelFormat,
    colour: ColourDescription,
) -> Option<RawFrame> {
    if format != PixelFormat::P010 {
        let (narrow, stride) = downshift_p010(&data, stride, width, height);
        return deliver(
            Bytes::from(narrow),
            stride,
            (width, height),
            timestamp_us,
            format,
            colour,
        );
    }
    let rows = height + height.div_ceil(2);
    if data.len() < stride.checked_mul(rows)? || stride < width.div_ceil(2) * 4 {
        return None;
    }
    Some(RawFrame {
        width: u32::try_from(width).ok()?,
        height: u32::try_from(height).ok()?,
        stride: u32::try_from(stride).ok()?,
        format: PixelFormat::P010,
        data,
        timestamp_us,
        colour: Some(colour),
        hdr: None,
    })
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

/// The samples of a plane of a high-bit-depth (9 to 16-bit) picture,
/// in the shape the decoders hand them out.
#[cfg(any(
    feature = "hevc-decode",
    feature = "vp9-decode",
    feature = "av1-decode"
))]
#[derive(Debug, Clone, Copy)]
pub(crate) enum WideSamples<'a> {
    /// Little-endian 16-bit words (dav1d, `rusty_vp9`); stride in bytes.
    LeBytes(&'a [u8]),
    /// Native 16-bit samples (`rust_h265`); stride in samples.
    Words(&'a [u16]),
}

/// One plane of a [`WidePicture`].
#[cfg(any(
    feature = "hevc-decode",
    feature = "vp9-decode",
    feature = "av1-decode"
))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct WidePlane<'a> {
    pub samples: WideSamples<'a>,
    /// Row pitch, in the unit [`WideSamples`] documents.
    pub stride: usize,
}

#[cfg(any(
    feature = "hevc-decode",
    feature = "vp9-decode",
    feature = "av1-decode"
))]
impl WidePlane<'_> {
    /// Copies the first `out.len()` samples of row `row` into `out`;
    /// `None` when the plane is shorter than that.
    fn row_into(&self, row: usize, out: &mut [u16]) -> Option<()> {
        match self.samples {
            WideSamples::LeBytes(bytes) => {
                let line = bytes.get(row * self.stride..)?.get(..out.len() * 2)?;
                for (sample, pair) in out.iter_mut().zip(line.as_chunks::<2>().0) {
                    *sample = u16::from_le_bytes(*pair);
                }
            }
            WideSamples::Words(words) => {
                out.copy_from_slice(words.get(row * self.stride..)?.get(..out.len())?);
            }
        }
        Some(())
    }
}

/// A decoded 4:2:0 picture with `depth`-bit samples in three planes.
#[cfg(any(
    feature = "hevc-decode",
    feature = "vp9-decode",
    feature = "av1-decode"
))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct WidePicture<'a> {
    pub width: usize,
    pub height: usize,
    pub depth: u32,
    pub y: WidePlane<'a>,
    pub u: WidePlane<'a>,
    pub v: WidePlane<'a>,
}

#[cfg(any(
    feature = "hevc-decode",
    feature = "vp9-decode",
    feature = "av1-decode"
))]
impl WidePicture<'_> {
    /// Hands the picture out in the requested format: P010 keeps every
    /// bit (samples moved to the high bits); any other request gets the
    /// top eight bits as NV12, then [`deliver`]'s conversion. `None` for
    /// short planes or a depth outside 8..=16.
    pub(crate) fn deliver(
        &self,
        timestamp_us: u64,
        format: PixelFormat,
        colour: ColourDescription,
    ) -> Option<RawFrame> {
        if !(8..=16).contains(&self.depth) || self.width == 0 || self.height == 0 {
            return None;
        }
        if format == PixelFormat::P010 {
            let (data, stride) = self.pack(16 - self.depth, 2)?;
            return Some(RawFrame {
                width: u32::try_from(self.width).ok()?,
                height: u32::try_from(self.height).ok()?,
                stride: u32::try_from(stride).ok()?,
                format: PixelFormat::P010,
                data,
                timestamp_us,
                colour: Some(colour),
                hdr: None,
            });
        }
        let (data, stride) = self.pack(self.depth - 8, 1)?;
        deliver(
            data,
            stride,
            (self.width, self.height),
            timestamp_us,
            format,
            colour,
        )
    }

    /// Semi-planar output with `bytes` per sample: luma rows, then
    /// interleaved chroma rows, one stride for both (`ceil(width / 2)`
    /// pairs). `bytes == 2` writes little-endian words shifted left by
    /// `shift`; `bytes == 1` writes each sample shifted right by `shift`.
    fn pack(&self, shift: u32, bytes: usize) -> Option<(Bytes, usize)> {
        let (chroma_width, chroma_height) = (self.width.div_ceil(2), self.height.div_ceil(2));
        let stride = chroma_width * 2 * bytes;
        let mut out = vec![0; stride * (self.height + chroma_height)];
        let put = |line: &mut [u8], samples: &mut dyn Iterator<Item = u16>| {
            if bytes == 2 {
                for (pair, sample) in line.as_chunks_mut::<2>().0.iter_mut().zip(samples) {
                    *pair = (sample << shift).to_le_bytes();
                }
            } else {
                for (byte, sample) in line.iter_mut().zip(samples) {
                    *byte = u8::try_from(sample >> shift).unwrap_or(u8::MAX);
                }
            }
        };
        let (luma, chroma) = out.split_at_mut(stride * self.height);
        let mut row = vec![0; self.width];
        for (index, line) in luma.chunks_exact_mut(stride).enumerate() {
            self.y.row_into(index, &mut row)?;
            put(line, &mut row.iter().copied());
        }
        let (mut u, mut v) = (vec![0; chroma_width], vec![0; chroma_width]);
        for (index, line) in chroma.chunks_exact_mut(stride).enumerate() {
            self.u.row_into(index, &mut u)?;
            self.v.row_into(index, &mut v)?;
            put(line, &mut u.iter().zip(&v).flat_map(|(&u, &v)| [u, v]));
        }
        Some((Bytes::from(out), stride))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_high_byte_of_each_sample_and_drops_padding() {
        // 2x2 P010, stride 6 (2 bytes padding): luma rows, one chroma row.
        let data = [
            0x00, 0x40, 0xC0, 0xEB, 0xFF, 0xFF, //
            0x11, 0x10, 0x22, 0x20, 0xFF, 0xFF, //
            0x00, 0x80, 0x00, 0x7F, 0xFF, 0xFF,
        ];
        let (nv12, stride) = downshift_p010(&data, 6, 2, 2);
        assert_eq!(stride, 2);
        assert_eq!(nv12, [0x40, 0xEB, 0x10, 0x20, 0x80, 0x7F]);
    }

    #[cfg(any(
        feature = "hevc-decode",
        feature = "vp9-decode",
        feature = "av1-decode"
    ))]
    fn wide_picture<'a>(
        y: WideSamples<'a>,
        u: WideSamples<'a>,
        v: WideSamples<'a>,
        strides: (usize, usize),
    ) -> WidePicture<'a> {
        WidePicture {
            width: 2,
            height: 2,
            depth: 10,
            y: WidePlane {
                samples: y,
                stride: strides.0,
            },
            u: WidePlane {
                samples: u,
                stride: strides.1,
            },
            v: WidePlane {
                samples: v,
                stride: strides.1,
            },
        }
    }

    #[cfg(any(
        feature = "hevc-decode",
        feature = "vp9-decode",
        feature = "av1-decode"
    ))]
    #[test]
    fn ten_bit_pictures_become_p010_with_samples_in_the_high_bits() {
        let colour = ColourDescription::BT709_LIMITED;
        let picture = wide_picture(
            WideSamples::Words(&[0, 1023, 9, 512, 4, 9]),
            WideSamples::Words(&[64]),
            WideSamples::Words(&[960]),
            (3, 1),
        );
        let frame = picture.deliver(5, PixelFormat::P010, colour).unwrap();
        assert_eq!(
            (frame.format, frame.stride, frame.colour),
            (PixelFormat::P010, 4, Some(colour))
        );
        let words: Vec<u16> = frame
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u16::from_le_bytes(*p))
            .collect();
        assert_eq!(words, [0, 1023 << 6, 512 << 6, 4 << 6, 64 << 6, 960 << 6]);
    }

    #[cfg(any(
        feature = "hevc-decode",
        feature = "vp9-decode",
        feature = "av1-decode"
    ))]
    #[test]
    fn ten_bit_pictures_keep_their_top_eight_bits_as_nv12() {
        let le = [0xFF, 0x03, 0x00, 0x02, 9, 9, 0x04, 0x00, 0x10, 0x00, 9, 9];
        let picture = wide_picture(
            WideSamples::LeBytes(&le),
            WideSamples::LeBytes(&[0x00, 0x01]),
            WideSamples::LeBytes(&[0xFC, 0x03]),
            (6, 2),
        );
        let frame = picture
            .deliver(5, PixelFormat::Nv12, ColourDescription::BT709_LIMITED)
            .unwrap();
        assert_eq!((frame.format, frame.stride), (PixelFormat::Nv12, 2));
        assert_eq!(&frame.data[..], &[255, 128, 1, 4, 64, 255]);
    }

    #[cfg(any(
        feature = "hevc-decode",
        feature = "vp9-decode",
        feature = "av1-decode"
    ))]
    #[test]
    fn short_planes_or_odd_depths_are_rejected() {
        let short = wide_picture(
            WideSamples::Words(&[1, 2, 3]),
            WideSamples::Words(&[1]),
            WideSamples::Words(&[1]),
            (2, 1),
        );
        assert!(
            short
                .deliver(0, PixelFormat::P010, ColourDescription::BT709_LIMITED)
                .is_none()
        );
        let deep = WidePicture {
            depth: 17,
            ..wide_picture(
                WideSamples::Words(&[0; 4]),
                WideSamples::Words(&[0]),
                WideSamples::Words(&[0]),
                (2, 1),
            )
        };
        assert!(
            deep.deliver(0, PixelFormat::Nv12, ColourDescription::BT709_LIMITED)
                .is_none()
        );
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
