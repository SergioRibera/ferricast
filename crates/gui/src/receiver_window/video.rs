//! Decoded video frames and how a receiver window draws them.
//!
//! The decoders hand the window NV12 (P010 for 10-bit streams) with its
//! colour description.
//! On a GPU renderer (Vulkan or OpenGL, whichever Freya picked) the
//! two planes are uploaded as textures straight from the decoded
//! buffer and Skia converts YCbCr to RGB and scales to the window in
//! the fragment shader, so the CPU never touches a pixel. Freya's
//! software renderer has no GPU context; it gets a CPU conversion
//! into a reused buffer instead.

use bytes::Bytes;
use ferricast::{
    ColourDescription, ColourPrimaries, ColourRange, FerricastError, PixelFormat, RawFrame, Result,
    TransferCharacteristics, YuvColour, YuvMatrix,
};
use skia_safe::gpu::{self, Mipmapped, RecordingContext};
use skia_safe::image::RequiredProperties;
use skia_safe::yuva_info::{PlaneConfig, Subsampling};
use skia_safe::{
    AlphaType, Canvas, ColorSpace, ColorType, Data, FilterMode, ISize, Image, ImageInfo,
    MipmapMode, Paint, Rect, SamplingOptions, YUVAInfo, YUVAPixmapInfo, YUVAPixmaps, YUVColorSpace,
    images,
};

/// How a frame's bytes are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// `stride`-pitched luma rows, then interleaved CbCr rows at half
    /// resolution.
    Nv12(YuvColour),
    /// As NV12 with 16-bit little-endian samples holding 10 bits in their
    /// high bits; the full description, since 10-bit pictures are the
    /// ones that carry wide-gamut / PQ / HLG signalling.
    P010(ColourDescription),
    /// Packed BGRA8888 rows, `stride` bytes apart.
    Bgra,
}

/// One decoded video frame on its way from the pipeline to the
/// window. Plain bytes, because Skia images are tied to the render
/// thread's GPU context; the window uploads them when it draws.
#[derive(Clone, Debug)]
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub layout: Layout,
    pub data: Bytes,
    /// Presentation timestamp. The pipeline already delivers each
    /// frame at its presentation time, so the window shows frames as
    /// they arrive; the PTS is kept for diagnostics.
    pub pts_us: u64,
}

impl TryFrom<RawFrame> for VideoFrame {
    type Error = FerricastError;

    fn try_from(raw: RawFrame) -> Result<Self> {
        let layout = match raw.format {
            PixelFormat::Nv12 => Layout::Nv12(raw.colour.map_or_else(
                || YuvColour::guess(raw.height as usize),
                |colour| colour.yuv(raw.height as usize),
            )),
            PixelFormat::P010 => Layout::P010(
                raw.colour
                    .unwrap_or(ColourDescription::unspecified(ColourRange::Limited)),
            ),
            PixelFormat::Bgra => Layout::Bgra,
            PixelFormat::Rgba | PixelFormat::I420 => {
                return Err(FerricastError::Decode(format!(
                    "receiver window: cannot show {:?} frames",
                    raw.format
                )));
            }
        };
        Ok(Self {
            width: raw.width,
            height: raw.height,
            stride: raw.stride,
            layout,
            data: raw.data,
            pts_us: raw.timestamp_us,
        })
    }
}

impl VideoFrame {
    fn dimensions(&self) -> ISize {
        ISize::new(
            i32::try_from(self.width).unwrap_or(0),
            i32::try_from(self.height).unwrap_or(0),
        )
    }

    /// Uploads an NV12 or P010 frame to the GPU as a YCbCr image Skia
    /// converts to RGB when drawing. A P010 frame signalling a wide gamut
    /// or an HDR transfer (PQ, HLG) is tagged with that colour space and
    /// converted to sRGB on the GPU, so it shows with its colours instead
    /// of washed out; HDR is tone-mapped down to SDR by Skia's colour
    /// management, not shown as HDR.
    fn yuv_texture(&self, context: &mut RecordingContext) -> Option<Image> {
        let (space, planes, image_space) = match self.layout {
            Layout::Nv12(colour) => (
                yuv_colour_space(colour, false),
                [ColorType::Gray8, ColorType::R8G8UNorm],
                None,
            ),
            Layout::P010(colour) => (
                yuv_colour_space(colour.yuv(self.height as usize), true),
                [ColorType::A16UNorm, ColorType::R16G16UNorm],
                managed_colour_space(colour),
            ),
            Layout::Bgra => return None,
        };
        let info = YUVAInfo::new(
            self.dimensions(),
            PlaneConfig::Y_UV,
            Subsampling::S420,
            space,
            None,
            None,
        )?;
        let stride = self.stride as usize;
        let pixmap_info = YUVAPixmapInfo::new(&info, &planes, Some(&[stride, stride]))?;
        if pixmap_info.compute_total_bytes(None) > self.data.len() {
            return None;
        }
        // SAFETY: `from_external_memory` only reads through the
        // pointer (the `*mut` is Skia's signature, not a write), the
        // check above makes `data` cover every plane `pixmap_info`
        // describes, and the pixmaps are dropped at the end of this
        // function while `self.data` stays alive for all of it.
        // `texture_from_yuva_pixmaps` copies the planes into textures
        // before returning, so nothing refers to `data` afterwards.
        let pixmaps = unsafe {
            YUVAPixmaps::from_external_memory(&pixmap_info, self.data.as_ptr().cast_mut().cast())
        }?;
        let image = gpu::images::texture_from_yuva_pixmaps(
            context,
            &pixmaps,
            Mipmapped::No,
            false,
            image_space.clone(),
        )?;
        if image_space.is_none() {
            return Some(image);
        }
        image.make_color_space(
            Some(context.as_recorder()),
            ColorSpace::new_srgb(),
            RequiredProperties::default(),
        )
    }

    /// A CPU image of the frame, converting NV12 into `scratch` (P010
    /// keeps its top eight bits first; the software renderer applies no
    /// colour management).
    fn raster(&self, scratch: &mut Vec<u8>) -> Option<Image> {
        let size = (self.width as usize, self.height as usize);
        let (bytes, row_bytes): (&[u8], usize) = match self.layout {
            Layout::Bgra => (&self.data, self.stride as usize),
            Layout::Nv12(colour) => {
                colour.nv12_to_bgra_into(&self.data, self.stride as usize, size, scratch)?;
                (scratch, self.width as usize * 4)
            }
            Layout::P010(colour) => {
                let (nv12, stride) = narrow_p010(&self.data, self.stride as usize, size)?;
                colour
                    .yuv(size.1)
                    .nv12_to_bgra_into(&nv12, stride, size, scratch)?;
                (scratch, self.width as usize * 4)
            }
        };
        let info = ImageInfo::new(
            self.dimensions(),
            ColorType::BGRA8888,
            AlphaType::Premul,
            None,
        );
        images::raster_from_data(&info, Data::new_copy(bytes), row_bytes)
    }
}

/// Skia's YCbCr conversion for `colour`. P010 samples sit in the high
/// bits of 16-bit words, so Skia reads them normalised like 10-bit
/// values; BT.601 and BT.709 have no 10-bit variants, and their 8-bit
/// range offsets differ from the 10-bit ones by under 0.1 %.
fn yuv_colour_space(colour: YuvColour, ten_bit: bool) -> YUVColorSpace {
    match (colour.matrix, colour.range, ten_bit) {
        (YuvMatrix::Bt601, ColourRange::Limited, _) => YUVColorSpace::Rec601_Limited,
        (YuvMatrix::Bt601, ColourRange::Full, _) => YUVColorSpace::JPEG_Full,
        (YuvMatrix::Bt709, ColourRange::Limited, _) => YUVColorSpace::Rec709_Limited,
        (YuvMatrix::Bt709, ColourRange::Full, _) => YUVColorSpace::Rec709_Full,
        (YuvMatrix::Bt2020, ColourRange::Limited, false) => YUVColorSpace::BT2020_8bit_Limited,
        (YuvMatrix::Bt2020, ColourRange::Full, false) => YUVColorSpace::BT2020_8bit_Full,
        (YuvMatrix::Bt2020, ColourRange::Limited, true) => YUVColorSpace::BT2020_10bit_Limited,
        (YuvMatrix::Bt2020, ColourRange::Full, true) => YUVColorSpace::BT2020_10bit_Full,
    }
}

/// The colour space a P010 picture's RGB lives in, when it is one the
/// window must convert to sRGB: wide-gamut (BT.2020) primaries or an HDR
/// transfer. `None` (drawn as is) for SDR BT.709-class pictures and for
/// codes Skia has no CICP mapping for.
fn managed_colour_space(colour: ColourDescription) -> Option<ColorSpace> {
    use skia_safe::named_primaries::CicpId as Primaries;
    use skia_safe::named_transfer_fn::CicpId as Transfer;
    let hdr_transfer = match colour.transfer {
        TransferCharacteristics::Pq => Some(Transfer::PQ),
        TransferCharacteristics::Hlg => Some(Transfer::HLG),
        _ => None,
    };
    let wide = colour.primaries == ColourPrimaries::Bt2020;
    if hdr_transfer.is_none() && !wide {
        return None;
    }
    let transfer = hdr_transfer.unwrap_or(Transfer::Rec709);
    let primaries = if wide {
        Primaries::Rec2020
    } else {
        Primaries::Rec709
    };
    ColorSpace::new_cicp(primaries, transfer)
}

/// The top eight bits of a P010 picture as tightly packed NV12 (rows of
/// `ceil(width / 2) * 2` bytes); `None` when `data` is short.
fn narrow_p010(
    data: &[u8],
    stride: usize,
    (width, height): (usize, usize),
) -> Option<(Vec<u8>, usize)> {
    let out_stride = width.div_ceil(2) * 2;
    let rows = height + height.div_ceil(2);
    let mut out = vec![0; out_stride * rows];
    for (row, line) in out.chunks_exact_mut(out_stride).enumerate() {
        let bytes = if row < height { width } else { out_stride };
        let source = data.get(row * stride..)?.get(..bytes * 2)?;
        // Little-endian words: the high byte is the second one.
        for (byte, word) in line[..bytes].iter_mut().zip(source.as_chunks::<2>().0) {
            *byte = word[1];
        }
    }
    Some((out, out_stride))
}

/// The newest frame the window has been handed, numbered so the
/// renderer knows when it changed.
#[derive(Default)]
pub struct FrameSlot {
    serial: u64,
    frame: Option<VideoFrame>,
}

impl FrameSlot {
    pub fn replace(&mut self, frame: VideoFrame) {
        self.serial += 1;
        self.frame = Some(frame);
    }
}

/// Draws the latest frame of a [`FrameSlot`], uploading each frame
/// once however often the window redraws.
#[derive(Default)]
pub struct VideoRenderer {
    /// The image of frame `serial`.
    image: Option<(u64, Image)>,
    /// Reused BGRA buffer of the software path.
    scratch: Vec<u8>,
    /// Whether the first drawn frame was logged.
    announced: bool,
}

impl VideoRenderer {
    /// Draws `slot`'s frame letterboxed into a `width` × `height` area
    /// at the canvas origin.
    pub fn draw(&mut self, canvas: &Canvas, slot: &FrameSlot, width: f32, height: f32) {
        let Some(frame) = &slot.frame else { return };
        let mut context = canvas.recording_context();
        let cached = self.image.take().filter(|(serial, image)| {
            *serial == slot.serial && image.is_valid(context.as_mut().map(|c| c.as_recorder() as _))
        });
        let image = match cached {
            Some((_, image)) => Some(image),
            None => match (frame.layout, context.as_mut()) {
                (Layout::Nv12(_) | Layout::P010(_), Some(context)) => frame.yuv_texture(context),
                _ => frame.raster(&mut self.scratch),
            },
        };
        if !self.announced
            && let Some(image) = &image
        {
            self.announced = true;
            tracing::info!(
                width = frame.width,
                height = frame.height,
                layout = ?frame.layout,
                gpu = image.is_texture_backed(),
                "receiver window: first video frame"
            );
        }
        let Some(image) = image else {
            tracing::warn!(
                width = frame.width,
                height = frame.height,
                "receiver window: frame could not be turned into an image"
            );
            return;
        };
        let (x, y, w, h) = aspect_fit(width, height, frame.width as f32, frame.height as f32);
        canvas.draw_image_rect_with_sampling_options(
            &image,
            None,
            Rect::from_xywh(x, y, w, h),
            SamplingOptions::new(FilterMode::Linear, MipmapMode::None),
            &Paint::default(),
        );
        self.image = Some((slot.serial, image));
    }
}

/// Aspect-fit `src` into `dst`, return `(x, y, w, h)` of the inset
/// rectangle.
fn aspect_fit(dst_w: f32, dst_h: f32, src_w: f32, src_h: f32) -> (f32, f32, f32, f32) {
    if src_w <= 0.0 || src_h <= 0.0 || dst_w <= 0.0 || dst_h <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let src_aspect = src_w / src_h;
    let dst_aspect = dst_w / dst_h;
    let (w, h) = if src_aspect > dst_aspect {
        (dst_w, dst_w / src_aspect)
    } else {
        (dst_h * src_aspect, dst_h)
    };
    ((dst_w - w) / 2.0, (dst_h - h) / 2.0, w, h)
}

#[cfg(test)]
mod gpu_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use ferricast::ColourDescription;

    #[test]
    fn wide_frames_are_letterboxed_and_tall_ones_pillarboxed() {
        assert_eq!(
            aspect_fit(200.0, 200.0, 16.0, 9.0),
            (0.0, 43.75, 200.0, 112.5)
        );
        assert_eq!(
            aspect_fit(200.0, 100.0, 1.0, 1.0),
            (50.0, 0.0, 100.0, 100.0)
        );
        assert_eq!(aspect_fit(0.0, 100.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0));
    }

    fn nv12(colour: Option<ColourDescription>) -> RawFrame {
        RawFrame {
            hdr: None,
            width: 2,
            height: 2,
            stride: 2,
            format: PixelFormat::Nv12,
            data: Bytes::from_static(&[235, 235, 235, 235, 128, 128]),
            timestamp_us: 7,
            colour,
        }
    }

    #[test]
    fn nv12_frames_keep_their_colour_or_get_a_guess() {
        let full = ColourDescription::unspecified(ColourRange::Full);
        let frame = VideoFrame::try_from(nv12(Some(full))).unwrap();
        assert_eq!((frame.layout, frame.pts_us), (Layout::Nv12(full.yuv(2)), 7));
        let guessed = VideoFrame::try_from(nv12(None)).unwrap();
        assert_eq!(guessed.layout, Layout::Nv12(YuvColour::guess(2)));
    }

    #[test]
    fn the_software_path_converts_nv12_on_the_cpu() {
        let frame = VideoFrame::try_from(nv12(None)).unwrap();
        let mut scratch = Vec::new();
        let image = frame.raster(&mut scratch).unwrap();
        assert_eq!((image.width(), image.height()), (2, 2));
        assert_eq!(scratch, [255; 16]);
    }

    /// [`nv12`] as P010: every sample in the high byte.
    fn p010(colour: Option<ColourDescription>) -> RawFrame {
        let eight = nv12(colour);
        RawFrame {
            stride: 4,
            format: PixelFormat::P010,
            data: eight.data.iter().flat_map(|&sample| [0, sample]).collect(),
            ..eight
        }
    }

    #[test]
    fn p010_frames_keep_their_full_description() {
        let pq = ColourDescription {
            transfer: ferricast::TransferCharacteristics::Pq,
            ..ColourDescription::BT709_LIMITED
        };
        let frame = VideoFrame::try_from(p010(Some(pq))).unwrap();
        assert_eq!(frame.layout, Layout::P010(pq));
        let unsignalled = VideoFrame::try_from(p010(None)).unwrap();
        assert_eq!(
            unsignalled.layout,
            Layout::P010(ColourDescription::unspecified(ColourRange::Limited))
        );
        assert!(managed_colour_space(pq).is_some(), "PQ is colour managed");
        assert!(managed_colour_space(ColourDescription::BT709_LIMITED).is_none());
    }

    #[test]
    fn the_software_path_draws_p010_from_its_top_eight_bits() {
        let frame = VideoFrame::try_from(p010(None)).unwrap();
        let mut scratch = Vec::new();
        let image = frame.raster(&mut scratch).unwrap();
        assert_eq!((image.width(), image.height()), (2, 2));
        assert_eq!(scratch, [255; 16]);
        assert!(narrow_p010(&[0; 11], 4, (2, 2)).is_none(), "short data");
    }

    #[test]
    fn rgba_frames_are_refused() {
        let mut raw = nv12(None);
        raw.format = PixelFormat::Rgba;
        assert!(VideoFrame::try_from(raw).is_err());
    }
}
