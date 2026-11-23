//! Validated geometry of a CPU-resident [`RawFrame`].
//!
//! Every capture producer sets `RawFrame::stride` to the number of
//! bytes per row of the first plane *as laid out in `data`* — which
//! may exceed `width × bytes_per_pixel` when the source pads rows
//! (PipeWire SHM buffers, VA-API readback). Encoders must walk the
//! buffer row by row with that stride; treating it as tightly packed
//! shears the picture.
//!
//! Planar layouts follow the convention the decoders emit:
//! * **NV12** — UV plane starts at `stride × height`, same stride.
//! * **I420** — U plane starts at `stride × height`, V right after it;
//!   both chroma planes use `stride / 2`.

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
        }
    }

    /// Stride of the chroma plane(s); see the module docs for the layout.
    pub(crate) fn chroma_stride(&self) -> usize {
        match self.format {
            PixelFormat::Bgra | PixelFormat::Rgba | PixelFormat::Nv12 => self.stride,
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

    fn min_stride(&self) -> usize {
        match self.format {
            // I420 chroma rows are `stride / 2` wide and must still hold
            // `ceil(width / 2)` samples.
            PixelFormat::I420 => self.width.div_ceil(2) as usize * 2,
            PixelFormat::Bgra | PixelFormat::Rgba | PixelFormat::Nv12 => self.row_bytes(),
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
            PixelFormat::Nv12 => {
                let uv_row = self.width.div_ceil(2) as usize * 2;
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

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;

    fn raw(format: PixelFormat, width: u32, height: u32, stride: u32, len: usize) -> RawFrame {
        RawFrame {
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
