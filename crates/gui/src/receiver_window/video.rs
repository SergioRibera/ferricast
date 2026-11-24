//! Decoded video frames and how a receiver window draws them.
//!
//! The decoders hand the window NV12 with its colour description.
//! On a GPU renderer (Vulkan or OpenGL, whichever Freya picked) the
//! two planes are uploaded as textures straight from the decoded
//! buffer and Skia converts YCbCr to RGB and scales to the window in
//! the fragment shader, so the CPU never touches a pixel. Freya's
//! software renderer has no GPU context; it gets a CPU conversion
//! into a reused buffer instead.

use bytes::Bytes;
use ferricast::{ColourRange, FerricastError, PixelFormat, RawFrame, Result, YuvColour, YuvMatrix};
use skia_safe::gpu::{self, Mipmapped, RecordingContext};
use skia_safe::yuva_info::{PlaneConfig, Subsampling};
use skia_safe::{
    AlphaType, Canvas, ColorType, Data, FilterMode, ISize, Image, ImageInfo, MipmapMode, Paint,
    Rect, SamplingOptions, YUVAInfo, YUVAPixmapInfo, YUVAPixmaps, YUVColorSpace, images,
};

/// How a frame's bytes are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// `stride`-pitched luma rows, then interleaved CbCr rows at half
    /// resolution.
    Nv12(YuvColour),
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

    /// Uploads the frame to the GPU as a YCbCr image Skia converts to
    /// RGB when drawing.
    fn nv12_texture(&self, colour: YuvColour, context: &mut RecordingContext) -> Option<Image> {
        let info = YUVAInfo::new(
            self.dimensions(),
            PlaneConfig::Y_UV,
            Subsampling::S420,
            yuv_colour_space(colour),
            None,
            None,
        )?;
        let stride = self.stride as usize;
        let pixmap_info = YUVAPixmapInfo::new(
            &info,
            &[ColorType::Gray8, ColorType::R8G8UNorm],
            Some(&[stride, stride]),
        )?;
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
        gpu::images::texture_from_yuva_pixmaps(context, &pixmaps, Mipmapped::No, false, None)
    }

    /// A CPU image of the frame, converting NV12 into `scratch`.
    fn raster(&self, scratch: &mut Vec<u8>) -> Option<Image> {
        let (bytes, row_bytes): (&[u8], usize) = match self.layout {
            Layout::Bgra => (&self.data, self.stride as usize),
            Layout::Nv12(colour) => {
                let size = (self.width as usize, self.height as usize);
                colour.nv12_to_bgra_into(&self.data, self.stride as usize, size, scratch)?;
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

fn yuv_colour_space(colour: YuvColour) -> YUVColorSpace {
    match (colour.matrix, colour.range) {
        (YuvMatrix::Bt601, ColourRange::Limited) => YUVColorSpace::Rec601_Limited,
        (YuvMatrix::Bt601, ColourRange::Full) => YUVColorSpace::JPEG_Full,
        (YuvMatrix::Bt709, ColourRange::Limited) => YUVColorSpace::Rec709_Limited,
        (YuvMatrix::Bt709, ColourRange::Full) => YUVColorSpace::Rec709_Full,
        (YuvMatrix::Bt2020, ColourRange::Limited) => YUVColorSpace::BT2020_8bit_Limited,
        (YuvMatrix::Bt2020, ColourRange::Full) => YUVColorSpace::BT2020_8bit_Full,
    }
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
                (Layout::Nv12(colour), Some(context)) => frame.nv12_texture(colour, context),
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

    #[test]
    fn rgba_frames_are_refused() {
        let mut raw = nv12(None);
        raw.format = PixelFormat::Rgba;
        assert!(VideoFrame::try_from(raw).is_err());
    }
}
