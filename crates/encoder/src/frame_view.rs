//! Validated geometry of a CPU-resident [`RawFrame`].
//!
//! Every capture producer sets `RawFrame::stride` to the number of
//! bytes per row of the first plane *as laid out in `data`* — which
//! may exceed `width × bytes_per_pixel` when the source pads rows
//! (`PipeWire` SHM buffers, VA-API readback). Encoders must walk the
//! buffer row by row with that stride; treating it as tightly packed
//! shears the picture.
//!
//! Planar layouts follow the convention the decoders emit:
//! * **NV12** — UV plane starts at `stride × height`, same stride.
//! * **P010** — as NV12 with 16-bit samples (rows of `2 × width` bytes).
//! * **I420** — U plane starts at `stride × height`, V right after it;
//!   both chroma planes use `stride / 2`.

#[cfg(feature = "vaapi")]
use ferricast_core::{BackendId, BitDepth, CodecError, CodecResult, EncoderConfig};
use ferricast_core::{FerricastError, PixelFormat, RawFrame, Result};

/// A [`RawFrame`] whose dimensions match the encoder configuration
/// and whose buffer is long enough for `stride` × `height`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FrameView<'a> {
    data: &'a [u8],
    format: PixelFormat,
    width: u32,
    height: u32,
    stride: usize,
}

#[allow(
    dead_code,
    reason = "shared by every video backend; each feature combination uses a different subset"
)]
impl<'a> FrameView<'a> {
    /// Validate `raw` against the size the encoder was configured for.
    ///
    /// # Errors
    ///
    /// Returns [`FerricastError::Encoder`] when the frame dimensions
    /// differ from `expected_width × expected_height`, the stride is
    /// shorter than one row of pixels, or `data` is too short for the
    /// declared geometry.
    pub(crate) fn new(
        raw: &'a RawFrame,
        expected_width: u32,
        expected_height: u32,
    ) -> Result<Self> {
        if raw.width != expected_width || raw.height != expected_height {
            return Err(FerricastError::Encoder(format!(
                "frame size {}x{} differs from configured {expected_width}x{expected_height}; \
                 reconfigure the encoder for the new size",
                raw.width, raw.height
            )));
        }
        let view = Self {
            data: &raw.data,
            format: raw.format,
            width: raw.width,
            height: raw.height,
            stride: raw.stride as usize,
        };
        if view.width == 0 || view.height == 0 {
            return Err(FerricastError::Encoder(
                "frame has zero width or height".into(),
            ));
        }
        if view.stride < view.min_stride() {
            return Err(FerricastError::Encoder(format!(
                "{:?} frame stride {} is shorter than one {}-px row ({} bytes)",
                view.format,
                view.stride,
                view.width,
                view.min_stride()
            )));
        }
        let required = view
            .required_len()
            .ok_or_else(|| FerricastError::Encoder("frame geometry overflows usize".into()))?;
        if view.data.len() < required {
            return Err(FerricastError::Encoder(format!(
                "{:?} {}x{} frame with stride {} needs {required} bytes, got {}",
                view.format,
                view.width,
                view.height,
                view.stride,
                view.data.len()
            )));
        }
        Ok(view)
    }

    pub(crate) fn data(&self) -> &'a [u8] {
        self.data
    }

    pub(crate) fn format(&self) -> PixelFormat {
        self.format
    }

    pub(crate) fn width(&self) -> u32 {
        self.width
    }

    pub(crate) fn height(&self) -> u32 {
        self.height
    }

    /// Bytes per row of the first plane.
    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    /// Meaningful bytes per row of the first plane (excludes padding).
    pub(crate) fn row_bytes(&self) -> usize {
        match self.format {
            PixelFormat::Bgra | PixelFormat::Rgba => self.width as usize * 4,
            PixelFormat::Nv12 | PixelFormat::I420 => self.width as usize,
            PixelFormat::P010 => self.width as usize * 2,
        }
    }

    /// Stride of the chroma plane(s); see the module docs for the layout.
    pub(crate) fn chroma_stride(&self) -> usize {
        match self.format {
            PixelFormat::Bgra | PixelFormat::Rgba | PixelFormat::Nv12 | PixelFormat::P010 => {
                self.stride
            }
            PixelFormat::I420 => self.stride / 2,
        }
    }

    /// Offset of the first chroma plane (U for I420, UV for NV12).
    pub(crate) fn chroma_offset(&self) -> usize {
        self.stride * self.height as usize
    }

    /// Rows in each chroma plane.
    pub(crate) fn chroma_rows(&self) -> usize {
        self.height.div_ceil(2) as usize
    }

    /// Offset of the V plane (I420 only).
    pub(crate) fn v_offset(&self) -> usize {
        self.chroma_offset() + self.chroma_stride() * self.chroma_rows()
    }

    /// The first plane's rows packed back to back (`row_bytes` apart).
    /// Borrows `data` directly when it is already tight; otherwise
    /// copies each row into `scratch`.
    pub(crate) fn tight_rows<'s>(&self, scratch: &'s mut Vec<u8>) -> &'s [u8]
    where
        'a: 's,
    {
        let row = self.row_bytes();
        let len = row * self.height as usize;
        if self.stride == row {
            return &self.data[..len];
        }
        scratch.clear();
        scratch.reserve(len);
        for line in self.data.chunks(self.stride).take(self.height as usize) {
            scratch.extend_from_slice(&line[..row]);
        }
        scratch
    }

    /// For a P010 frame, the picture as tightly packed NV12 holding the
    /// top eight bits of every sample (rows `ceil(width / 2) * 2` bytes
    /// apart), for encoders that only take 8-bit input; `None` for any
    /// other format.
    pub(crate) fn narrowed(&self) -> Option<(Vec<u8>, usize)> {
        if self.format != PixelFormat::P010 {
            return None;
        }
        let pairs = self.width.div_ceil(2) as usize;
        let stride = pairs * 2;
        let rows = self.height as usize;
        let mut out = vec![0; stride * (rows + self.chroma_rows())];
        let (luma, chroma) = out.split_at_mut(stride * rows);
        // P010 words are little-endian: the high byte is the second one.
        let narrow = |dst: &mut [u8], src: &[u8]| {
            for (byte, word) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                *byte = word[1];
            }
        };
        for (dst, src) in luma
            .chunks_exact_mut(stride)
            .zip(self.data.chunks(self.stride))
        {
            narrow(&mut dst[..self.width as usize], src);
        }
        let chroma_src = &self.data[self.chroma_offset()..];
        for (dst, src) in chroma
            .chunks_exact_mut(stride)
            .zip(chroma_src.chunks(self.stride))
        {
            narrow(dst, src);
        }
        Some((out, stride))
    }

    fn min_stride(&self) -> usize {
        match self.format {
            // I420 chroma rows are `stride / 2` wide and must still hold
            // `ceil(width / 2)` samples; NV12 chroma rows interleave
            // `ceil(width / 2)` pairs, one byte more than an odd width.
            PixelFormat::I420 | PixelFormat::Nv12 => self.width.div_ceil(2) as usize * 2,
            PixelFormat::P010 => self.width.div_ceil(2) as usize * 4,
            PixelFormat::Bgra | PixelFormat::Rgba => self.row_bytes(),
        }
    }

    /// Smallest buffer that holds every meaningful byte: the last row
    /// of each plane only needs its pixels, not its trailing padding.
    fn required_len(&self) -> Option<usize> {
        let h = self.height as usize;
        let luma = self
            .stride
            .checked_mul(h - 1)?
            .checked_add(self.row_bytes())?;
        match self.format {
            PixelFormat::Bgra | PixelFormat::Rgba => Some(luma),
            PixelFormat::Nv12 | PixelFormat::P010 => {
                let uv_row = self.min_stride();
                let uv = self
                    .chroma_stride()
                    .checked_mul(self.chroma_rows() - 1)?
                    .checked_add(uv_row)?;
                self.chroma_offset().checked_add(uv)
            }
            PixelFormat::I420 => {
                let c_row = self.width.div_ceil(2) as usize;
                let last = self
                    .chroma_stride()
                    .checked_mul(self.chroma_rows() - 1)?
                    .checked_add(c_row)?;
                self.v_offset().checked_add(last)
            }
        }
    }
}

/// `raw` as 8-bit samples for an encoder that only takes 8-bit input:
/// P010 becomes tightly packed NV12 holding the top eight bits of every
/// sample; every other format is returned as it is.
///
/// # Errors
///
/// As [`FrameView::new`], for a P010 frame whose geometry does not
/// match `width × height` or its buffer.
#[allow(
    dead_code,
    reason = "conditional: only the x264, NVENC and platform encoders take P010 down to 8 bits"
)]
pub(crate) fn eight_bit(raw: RawFrame, width: u32, height: u32) -> Result<RawFrame> {
    let Some((data, stride)) = FrameView::new(&raw, width, height)?.narrowed() else {
        return Ok(raw);
    };
    Ok(RawFrame {
        stride: u32::try_from(stride)
            .map_err(|_| FerricastError::Encoder("NV12 stride exceeds u32".into()))?,
        format: PixelFormat::Nv12,
        data: data.into(),
        ..raw
    })
}

/// Refuses what an 8-bit backend without HDR signalling cannot encode:
/// a 10-bit stream, or static HDR metadata it would silently drop.
///
/// # Errors
///
/// [`CodecError::UnsupportedConfig`] naming `backend` and what it lacks.
#[cfg(feature = "vaapi")]
pub(crate) fn eight_bit_sdr_only(config: &EncoderConfig, backend: BackendId) -> CodecResult<()> {
    if config.bit_depth != BitDepth::Eight {
        return Err(CodecError::UnsupportedConfig {
            backend,
            what: format!("{}-bit {}", config.bit_depth.bits(), config.codec),
        });
    }
    if config.hdr.is_some_and(|hdr| !hdr.is_empty()) {
        return Err(CodecError::UnsupportedConfig {
            backend,
            what: "static HDR metadata (mastering display / content light level)".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;

    fn raw(format: PixelFormat, width: u32, height: u32, stride: u32, len: usize) -> RawFrame {
        RawFrame {
            hdr: None,
            colour: None,
            width,
            height,
            stride,
            format,
            data: Bytes::from(vec![0u8; len]),
            timestamp_us: 0,
        }
    }

    #[test]
    fn tight_bgra_accepted() {
        let f = raw(PixelFormat::Bgra, 4, 2, 16, 32);
        let v = FrameView::new(&f, 4, 2).unwrap();
        assert_eq!(v.row_bytes(), 16);
    }

    #[test]
    fn padded_bgra_last_row_needs_no_padding() {
        // stride 32, 3 rows: 32 * 2 + 16 = 80 bytes minimum.
        let ok = raw(PixelFormat::Bgra, 4, 3, 32, 80);
        assert!(FrameView::new(&ok, 4, 3).is_ok());
        let short = raw(PixelFormat::Bgra, 4, 3, 32, 79);
        assert!(FrameView::new(&short, 4, 3).is_err());
    }

    #[test]
    fn stride_shorter_than_row_rejected() {
        let f = raw(PixelFormat::Bgra, 4, 2, 12, 64);
        assert!(FrameView::new(&f, 4, 2).is_err());
    }

    #[test]
    fn size_mismatch_rejected() {
        let f = raw(PixelFormat::Bgra, 4, 2, 16, 32);
        assert!(FrameView::new(&f, 8, 2).is_err());
        assert!(FrameView::new(&f, 4, 4).is_err());
    }

    #[test]
    fn tight_rows_strips_padding() {
        let mut data = Vec::new();
        for row in 0u8..3 {
            data.extend_from_slice(&[row; 8]); // 2 px BGRA
            data.extend_from_slice(&[0xEE; 4]); // padding
        }
        let f = RawFrame {
            hdr: None,
            colour: None,
            width: 2,
            height: 3,
            stride: 12,
            format: PixelFormat::Bgra,
            data: Bytes::from(data),
            timestamp_us: 0,
        };
        let v = FrameView::new(&f, 2, 3).unwrap();
        let mut scratch = Vec::new();
        let tight = v.tight_rows(&mut scratch);
        assert_eq!(tight.len(), 24);
        assert!(tight.iter().all(|&b| b != 0xEE));
        assert_eq!(&tight[16..24], &[2u8; 8]);
    }

    #[test]
    fn tight_rows_borrows_when_already_tight() {
        let f = raw(PixelFormat::Rgba, 2, 2, 8, 16);
        let v = FrameView::new(&f, 2, 2).unwrap();
        let mut scratch = Vec::new();
        assert_eq!(v.tight_rows(&mut scratch).len(), 16);
        assert!(scratch.is_empty());
    }

    #[test]
    fn nv12_layout() {
        // 4x2, stride 8: Y = 8 + 4, UV at 16, one row of 4 bytes.
        let f = raw(PixelFormat::Nv12, 4, 2, 8, 20);
        let v = FrameView::new(&f, 4, 2).unwrap();
        assert_eq!(v.chroma_offset(), 16);
        assert_eq!(v.chroma_stride(), 8);
        let short = raw(PixelFormat::Nv12, 4, 2, 8, 19);
        assert!(FrameView::new(&short, 4, 2).is_err());
    }

    #[test]
    fn p010_layout_is_nv12_with_two_byte_samples() {
        // 3x2, stride 8: luma rows of 6 bytes, chroma at 16 holding two
        // 4-byte pairs (8 bytes): 16 + 8.
        let f = raw(PixelFormat::P010, 3, 2, 8, 24);
        let v = FrameView::new(&f, 3, 2).unwrap();
        assert_eq!(
            (v.row_bytes(), v.chroma_offset(), v.chroma_stride()),
            (6, 16, 8)
        );
        assert!(FrameView::new(&raw(PixelFormat::P010, 3, 2, 8, 23), 3, 2).is_err());
        assert!(FrameView::new(&raw(PixelFormat::P010, 3, 2, 6, 18), 3, 2).is_err());
    }

    #[test]
    fn p010_is_narrowed_to_the_high_bytes_and_other_formats_are_kept() {
        let words: [u16; 6] = [0x1234, 0xFF00, 0x0040, 0x8000, 0x4000, 0xC000];
        let f = RawFrame {
            data: Bytes::from(
                words
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            timestamp_us: 7,
            ..raw(PixelFormat::P010, 2, 2, 4, 12)
        };
        let narrowed = eight_bit(f, 2, 2).unwrap();
        assert_eq!(
            (narrowed.format, narrowed.stride, narrowed.timestamp_us),
            (PixelFormat::Nv12, 2, 7)
        );
        assert_eq!(&narrowed.data[..], &[0x12, 0xFF, 0x00, 0x80, 0x40, 0xC0]);
        let nv12 = raw(PixelFormat::Nv12, 4, 2, 8, 20);
        assert_eq!(eight_bit(nv12, 4, 2).unwrap().stride, 8);
    }

    #[test]
    fn i420_layout() {
        // 4x2, stride 8: Y 16 bytes, U at 16 (stride 4, 1 row), V at 20.
        let f = raw(PixelFormat::I420, 4, 2, 8, 22);
        let v = FrameView::new(&f, 4, 2).unwrap();
        assert_eq!(v.chroma_offset(), 16);
        assert_eq!(v.chroma_stride(), 4);
        assert_eq!(v.v_offset(), 20);
        let short = raw(PixelFormat::I420, 4, 2, 8, 21);
        assert!(FrameView::new(&short, 4, 2).is_err());
    }
}
