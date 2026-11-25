//! Pictures whose size or format differs from the session, fitted into it
//! on the GPU: `ferricast-nvcodec`'s image kernels scale and letterbox
//! (the CPU scaler's geometry, [`FrameSize::letterbox_in`]), convert RGB to
//! NV12 or narrow P010 to NV12 into a pooled device buffer, which NVENC
//! encodes in place. NVDEC pictures and DMA-BUF captures never leave the
//! GPU; CPU pictures are uploaded once and fitted there instead of being
//! scaled on the CPU.
//!
//! Any failure leaves the picture to the CPU path, like the zero-copy path.

use ferricast_codec::cuda::nvcodec::cuda::{DeviceBuffer, Event, HostImage, Point, Region, Stream};
use ferricast_codec::cuda::nvcodec::encode::{DeviceFrame, Encoder, FrameOptions};
use ferricast_codec::cuda::nvcodec::kernels::{
    Converter, FramePool, Geometry, Input as KernelInput, Matrix, Orientation, OutputFormat, Range,
    Rect, RgbOrder, Rotation, Size, YuvColour,
};
use ferricast_codec::cuda::{self, CudaSurface, nvcodec};
use ferricast_core::{
    ColourDescription, ColourRange, MatrixCoefficients, NativeSurface as _, PixelFormat, RawFrame,
};

use super::{Input, NvencCodec, NvencEncoder};
use crate::frame_view::FrameView;
use crate::scale::FrameSize;

/// Pooled output pictures: one being written, one NVENC may still hold,
/// one spare.
const POOL: usize = 3;

/// Why a picture could not be fitted on the GPU.
#[derive(Debug, thiserror::Error)]
pub(super) enum Error {
    #[error("{0} cannot be fitted on the GPU")]
    Unsupported(String),
    #[error(transparent)]
    Nvcodec(#[from] nvcodec::Error),
    #[cfg(target_os = "linux")]
    #[error(transparent)]
    ZeroCopy(#[from] super::zero_copy::Error),
}

/// The GPU fitting state, created on the first picture that needs it.
#[derive(Default)]
pub(super) struct GpuFit {
    ready: Option<Box<Ready>>,
    /// Pictures that fell back to the CPU so far.
    failures: u64,
}

struct Ready {
    converter: Converter,
    stream: Stream,
    /// Recorded after the kernels; NVENC's stream waits for it.
    done: Event,
    pool: Option<FramePool>,
    /// CPU pictures are uploaded here.
    staging: Option<DeviceBuffer>,
}

impl GpuFit {
    fn ready(&mut self) -> Result<&mut Ready, Error> {
        if self.ready.is_none() {
            let context = cuda::context()?;
            self.ready = Some(Box::new(Ready {
                converter: Converter::new(&context)?,
                stream: Stream::new(&context)?,
                done: Event::new(&context)?,
                pool: None,
                staging: None,
            }));
        }
        self.ready
            .as_deref_mut()
            .ok_or_else(|| Error::Unsupported("the GPU fit state".into()))
    }

    /// Logs a failure loudly the first time, then once a minute at 60 fps.
    fn report<T>(&mut self, result: Result<T, Error>) -> Result<T, Error> {
        if let Err(error) = &result {
            self.failures += 1;
            if self.failures % 3600 == 1 {
                tracing::warn!(
                    %error,
                    failures = self.failures,
                    "NVENC: fitting the picture on the GPU failed, using the CPU"
                );
            }
        }
        result
    }
}

/// What the session takes: its layout and size.
#[derive(Clone, Copy)]
struct Session {
    format: OutputFormat,
    canvas: FrameSize,
}

/// The picture a kernel reads: its size, and how it must be turned to
/// stand upright.
#[derive(Clone, Copy)]
struct Source {
    size: FrameSize,
    orientation: Orientation,
}

impl From<FrameSize> for Source {
    fn from(size: FrameSize) -> Self {
        Self {
            size,
            orientation: Orientation::IDENTITY,
        }
    }
}

impl Source {
    /// The size once turned: what is letterboxed into the session.
    fn oriented(self) -> FrameSize {
        let size = self
            .orientation
            .oriented_size(Size::new(self.size.width, self.size.height));
        FrameSize::new(size.width, size.height)
    }
}

impl Ready {
    /// Fits `input` (a `source`-sized picture whose producer completes
    /// `ready`) into a pooled `target` picture of `canvas` and encodes it.
    fn encode(
        &mut self,
        encoder: &mut Encoder,
        input: KernelInput<'_>,
        source: Source,
        ready: Option<&Event>,
        session: Session,
        options: FrameOptions,
    ) -> Result<(), Error> {
        let Session {
            format: target,
            canvas,
        } = session;
        let size = Size::new(canvas.width, canvas.height);
        if self
            .pool
            .as_ref()
            .is_none_or(|pool| pool.format() != target || pool.size() != size)
        {
            self.pool = Some(FramePool::new(self.converter.context(), target, size, POOL));
        }
        let pool = self
            .pool
            .as_mut()
            .ok_or_else(|| Error::Unsupported("the output pool".into()))?;
        let buffer = pool.acquire()?;
        let placement = source.oriented().letterbox_in(canvas);
        let geometry = Geometry::new(Size::new(source.size.width, source.size.height), size)
            .with_orientation(source.orientation)
            .with_placement(Rect::new(
                placement.x,
                placement.y,
                placement.size.width,
                placement.size.height,
            ));
        if let Some(ready) = ready {
            self.stream.wait(ready)?;
        }
        let output = nvcodec::kernels::Output {
            buffer: &buffer,
            format: target,
            chroma_row: None,
        };
        self.converter
            .convert(input, output, &geometry, &self.stream)?;
        self.done.record(&self.stream)?;
        let frame = DeviceFrame {
            buffer: &buffer,
            chroma_row: None,
            ready: Some(&self.done),
        };
        encoder.encode_device(frame, options)?;
        Ok(())
    }
}

/// The kernels' output layout for a session taking `input`.
fn target(input: Input, colour: ColourDescription) -> Result<OutputFormat, Error> {
    let range = match colour.range {
        ColourRange::Limited => Range::Limited,
        ColourRange::Full => Range::Full,
    };
    // The matrix only matters for RGB → YUV; `rgb_matrix` refuses others.
    let yuv = YuvColour {
        matrix: rgb_matrix(colour).unwrap_or_default(),
        range,
    };
    Ok(match input {
        Input::Rgb(PixelFormat::Rgba) => OutputFormat::Rgb(RgbOrder::Rgba),
        Input::Rgb(PixelFormat::Bgra) => OutputFormat::Rgb(RgbOrder::Bgra),
        Input::Rgb(other) => return Err(Error::Unsupported(format!("an {other:?} session"))),
        Input::Nv12 => OutputFormat::Nv12(yuv),
        Input::P010 => OutputFormat::P010(yuv),
    })
}

/// The kernels' matrix for `colour`, when they have it.
fn rgb_matrix(colour: ColourDescription) -> Option<Matrix> {
    match colour.matrix {
        MatrixCoefficients::Bt709 | MatrixCoefficients::Unspecified => Some(Matrix::Bt709),
        MatrixCoefficients::Smpte170M | MatrixCoefficients::Bt470Bg => Some(Matrix::Bt601),
        MatrixCoefficients::Identity
        | MatrixCoefficients::Fcc
        | MatrixCoefficients::Smpte240M
        | MatrixCoefficients::YCgCo
        | MatrixCoefficients::Bt2020Ncl
        | MatrixCoefficients::Bt2020Cl
        | MatrixCoefficients::Smpte2085
        | MatrixCoefficients::ChromaDerivedNcl
        | MatrixCoefficients::ChromaDerivedCl
        | MatrixCoefficients::ICtCp => None,
    }
}

/// A capture's orientation as the kernels take it (the same mirror, then
/// clockwise rotation).
#[cfg(target_os = "linux")]
fn kernel_orientation(orientation: ferricast_core::Orientation) -> Orientation {
    use ferricast_core::Rotation as Turn;
    Orientation {
        mirrored: orientation.mirrored,
        rotation: match orientation.rotation {
            Turn::None => Rotation::None,
            Turn::Clockwise90 => Rotation::Clockwise90,
            Turn::Half => Rotation::Half,
            Turn::Clockwise270 => Rotation::Clockwise270,
        },
    }
}

fn rgb_order(format: PixelFormat) -> Result<RgbOrder, Error> {
    match format {
        PixelFormat::Bgra => Ok(RgbOrder::Bgra),
        PixelFormat::Rgba => Ok(RgbOrder::Rgba),
        PixelFormat::Nv12 | PixelFormat::I420 | PixelFormat::P010 => {
            Err(Error::Unsupported(format!("{format:?} as RGB")))
        }
    }
}

impl<C: NvencCodec> NvencEncoder<C> {
    /// The output layout and size of the session.
    fn fit_target(&self) -> Result<Session, Error> {
        let (width, height) = self.size();
        Ok(Session {
            format: target(self.input, self.requested.colour)?,
            canvas: FrameSize::new(width, height),
        })
    }

    /// Fits an NVDEC picture (NV12 or P016 in CUDA memory) into the session
    /// and encodes it from the GPU.
    pub(super) fn fit_native(
        &mut self,
        surface: &CudaSurface,
        options: FrameOptions,
    ) -> Result<(), Error> {
        let result = self.fit_native_inner(surface, options);
        self.fit.report(result)
    }

    fn fit_native_inner(
        &mut self,
        surface: &CudaSurface,
        options: FrameOptions,
    ) -> Result<(), Error> {
        let session = self.fit_target()?;
        let frame = surface.frame();
        let info = frame.info();
        let size = Size::new(info.width, info.height);
        let [luma, chroma] = frame.layout().planes() else {
            return Err(Error::Unsupported(
                "a picture that is not semi-planar".into(),
            ));
        };
        if luma.offset != 0 || luma.pitch == 0 || !chroma.offset.is_multiple_of(luma.pitch) {
            return Err(Error::Unsupported(format!(
                "an NVDEC layout with planes at {} and {}",
                luma.offset, chroma.offset
            )));
        }
        let view = frame.buffer().as_view();
        let chroma_row = chroma.offset / luma.pitch;
        let input = match surface.format() {
            PixelFormat::Nv12 => KernelInput::Nv12 {
                view,
                chroma_row,
                size,
            },
            PixelFormat::P010 => KernelInput::P016 {
                view,
                chroma_row,
                size,
            },
            other => return Err(Error::Unsupported(format!("an NVDEC {other:?} picture"))),
        };
        let source = Source::from(FrameSize::new(info.width, info.height));
        self.fit.ready()?.encode(
            &mut self.encoder,
            input,
            source,
            Some(frame.ready()),
            session,
            options,
        )
    }

    /// Uploads a CPU picture and fits it into the session on the GPU.
    ///
    /// The caller must keep `raw` alive until the picture's packet is out:
    /// the upload is queued on the fit stream, which NVENC waits for.
    pub(super) fn fit_host(&mut self, raw: &RawFrame, options: FrameOptions) -> Result<(), Error> {
        let result = self.fit_host_inner(raw, options);
        self.fit.report(result)
    }

    fn fit_host_inner(&mut self, raw: &RawFrame, options: FrameOptions) -> Result<(), Error> {
        let session = self.fit_target()?;
        if matches!(session.format, OutputFormat::Nv12(_))
            && matches!(raw.format, PixelFormat::Bgra | PixelFormat::Rgba)
            && rgb_matrix(self.requested.colour).is_none()
        {
            return Err(Error::Unsupported(format!(
                "RGB converted with {:?}",
                self.requested.colour.matrix
            )));
        }
        let view = FrameView::new(raw, raw.width, raw.height)
            .map_err(|error| Error::Unsupported(error.to_string()))?;
        let size = Size::new(raw.width, raw.height);
        let height = raw.height as usize;
        let row_bytes = view.row_bytes();
        let chroma_bytes = match raw.format {
            PixelFormat::Nv12 => raw.width.div_ceil(2) as usize * 2,
            PixelFormat::P010 => raw.width.div_ceil(2) as usize * 4,
            PixelFormat::Bgra | PixelFormat::Rgba => 0,
            PixelFormat::I420 => return Err(Error::Unsupported("I420".into())),
        };
        let chroma_rows = if chroma_bytes == 0 {
            0
        } else {
            view.chroma_rows()
        };
        let width_bytes = row_bytes.max(chroma_bytes);
        let rows = height + chroma_rows;
        let fit = self.fit.ready()?;
        // Out of `fit` while in use; put back below.
        let mut staging = match fit.staging.take() {
            Some(buffer) if buffer.width_bytes() >= width_bytes && buffer.rows() >= rows => buffer,
            _ => DeviceBuffer::pitched(fit.converter.context(), width_bytes, rows)?,
        };
        let data = view.data();
        let luma = HostImage {
            data,
            pitch: view.stride(),
        };
        // SAFETY: `raw` (and with it `data`, immutable `Bytes`) is kept by
        // the caller until the packet is out, and NVENC reads the picture
        // only after the fit stream (this copy included) has run; an error
        // below waits for the stream before returning.
        unsafe { staging.upload_async(Region::new(row_bytes, height), luma, &fit.stream) }?;
        if chroma_rows > 0 {
            let chroma = HostImage {
                data: data.get(view.chroma_offset()..).unwrap_or_default(),
                pitch: view.chroma_stride(),
            };
            let region = Region::new(chroma_bytes, chroma_rows).at(Point {
                x_bytes: 0,
                row: height,
            });
            // SAFETY: as above.
            let queued = unsafe { staging.upload_async(region, chroma, &fit.stream) };
            if let Err(error) = queued {
                fit.stream.synchronize()?;
                return Err(error.into());
            }
        }
        let staged = staging.as_view();
        let input = match raw.format {
            PixelFormat::Bgra | PixelFormat::Rgba => KernelInput::Packed {
                view: staged,
                order: rgb_order(raw.format)?,
                size,
            },
            PixelFormat::Nv12 => KernelInput::Nv12 {
                view: staged,
                chroma_row: height,
                size,
            },
            PixelFormat::P010 => KernelInput::P016 {
                view: staged,
                chroma_row: height,
                size,
            },
            PixelFormat::I420 => return Err(Error::Unsupported("I420".into())),
        };
        let source = Source::from(FrameSize::new(raw.width, raw.height));
        let encoded = fit.encode(&mut self.encoder, input, source, None, session, options);
        fit.staging = Some(staging);
        if encoded.is_err() {
            // The upload may still be reading `raw`, which the caller drops
            // on error.
            fit.stream.synchronize()?;
        }
        encoded
    }

    /// Fits a DMA-BUF capture picture (imported into CUDA through the
    /// zero-copy path's importer) into the session and encodes it.
    #[cfg(target_os = "linux")]
    pub(super) fn fit_gpu(
        &mut self,
        frame: &ferricast_core::GpuFrame,
        options: FrameOptions,
    ) -> Result<(), Error> {
        let result = self.fit_gpu_inner(frame, options);
        self.fit.report(result)
    }

    #[cfg(target_os = "linux")]
    fn fit_gpu_inner(
        &mut self,
        frame: &ferricast_core::GpuFrame,
        options: FrameOptions,
    ) -> Result<(), Error> {
        use ferricast_codec::cuda::nvcodec::interop::FrameView as Mapped;

        let session = self.fit_target()?;
        if matches!(session.format, OutputFormat::Nv12(_))
            && rgb_matrix(self.requested.colour).is_none()
        {
            return Err(Error::Unsupported(format!(
                "RGB converted with {:?}",
                self.requested.colour.matrix
            )));
        }
        let order = rgb_order(frame.format)?;
        let size = Size::new(frame.width, frame.height);
        let source = Source {
            size: FrameSize::new(frame.width, frame.height),
            orientation: kernel_orientation(frame.orientation),
        };
        let fit = self.fit.ready()?;
        self.zero_copy
            .with_view(&mut self.encoder, frame, |encoder, view| {
                let input = match view {
                    Mapped::Array(array) => KernelInput::Array { array, order },
                    Mapped::Pitched(view) => KernelInput::Packed { view, order, size },
                };
                // The importer waited for the producer's fence on the CPU.
                let encoded = fit.encode(encoder, input, source, None, session, options);
                if encoded.is_err() {
                    // Kernels may still read the view; the importer
                    // unregisters it later without waiting.
                    fit.stream.synchronize()?;
                }
                encoded
            })
    }
}

impl<C: NvencCodec> NvencEncoder<C> {
    /// Whether a CPU picture needs fitting: another size, or P010 into an
    /// 8-bit session.
    pub(super) fn needs_fit(&self, width: u32, height: u32, format: PixelFormat) -> bool {
        (width, height) != self.size() || (self.input == Input::Nv12 && format == PixelFormat::P010)
    }
}

/// The GPU path against ferricast's CPU path on the same pictures (needs an
/// NVIDIA GPU; prints why and passes without one).
#[cfg(test)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test pictures and statistics over small, positive sizes"
)]
mod tests {
    use std::time::{Duration, Instant};

    use bytes::Bytes;
    use ferricast_codec::cuda::nvcodec::cuda::{Context, HostImageMut};
    use ferricast_core::BackendId;

    use super::*;
    use crate::nv12::{self, Nv12Layout};
    use crate::scale::FrameScaler;

    fn context() -> Option<std::sync::Arc<Context>> {
        match cuda::context() {
            Ok(context) => Some(context),
            Err(error) => {
                eprintln!("skipping: no NVIDIA GPU: {error}");
                None
            }
        }
    }

    /// A screen-like BGRA picture: gradients with one-pixel text strokes.
    fn screen(width: u32, height: u32) -> RawFrame {
        let mut data = Vec::with_capacity(width as usize * height as usize * 4);
        for y in 0..height {
            for x in 0..width {
                let stroke = ((x / 3) % 7 == 0 || (y / 5) % 9 == 0) && (x / 40 + y / 24) % 3 == 0;
                let pixel = if stroke {
                    [20, 20, 20, 255]
                } else {
                    [
                        (x * 255 / width) as u8,
                        (y * 255 / height) as u8,
                        ((x + y) % 256) as u8,
                        255,
                    ]
                };
                data.extend_from_slice(&pixel);
            }
        }
        RawFrame {
            width,
            height,
            stride: width * 4,
            format: PixelFormat::Bgra,
            data: Bytes::from(data),
            timestamp_us: 0,
            colour: None,
            hdr: None,
        }
    }

    /// Uploads, fits into `canvas` the way the encoder does and downloads,
    /// returning the picture and the time per fit (upload included).
    fn gpu_fit(
        context: &std::sync::Arc<Context>,
        raw: &RawFrame,
        format: OutputFormat,
        canvas: FrameSize,
        rounds: u32,
    ) -> (Vec<u8>, Duration) {
        let mut converter = Converter::new(context).unwrap();
        let stream = Stream::new(context).unwrap();
        let (width, height) = (raw.width as usize, raw.height as usize);
        let (row_bytes, rows, input_format) = match raw.format {
            PixelFormat::P010 => (width * 2, height + height / 2, 2),
            _ => (width * 4, height, 4),
        };
        let mut staging = DeviceBuffer::pitched(context, row_bytes, rows).unwrap();
        let size = Size::new(canvas.width, canvas.height);
        let (out_bytes, out_rows) = format.buffer_layout(size);
        let target = DeviceBuffer::pitched(context, out_bytes, out_rows).unwrap();
        let placement = FrameSize::new(raw.width, raw.height).letterbox_in(canvas);
        let geometry =
            Geometry::new(Size::new(raw.width, raw.height), size).with_placement(Rect::new(
                placement.x,
                placement.y,
                placement.size.width,
                placement.size.height,
            ));
        let started = Instant::now();
        for _ in 0..rounds {
            let host = HostImage {
                data: &raw.data,
                pitch: raw.stride as usize,
            };
            // SAFETY: `raw` outlives the stream synchronisation below.
            unsafe { staging.upload_async(Region::new(row_bytes, rows), host, &stream) }.unwrap();
            let view = staging.as_view();
            let input = if input_format == 2 {
                KernelInput::P016 {
                    view,
                    chroma_row: height,
                    size: Size::new(raw.width, raw.height),
                }
            } else {
                KernelInput::Packed {
                    view,
                    order: RgbOrder::Bgra,
                    size: Size::new(raw.width, raw.height),
                }
            };
            let output = nvcodec::kernels::Output {
                buffer: &target,
                format,
                chroma_row: None,
            };
            converter
                .convert(input, output, &geometry, &stream)
                .unwrap();
            stream.synchronize().unwrap();
        }
        let per_fit = started.elapsed() / rounds;
        let mut host = vec![0; out_bytes * out_rows];
        target
            .download(
                Region::new(out_bytes, out_rows),
                HostImageMut {
                    data: &mut host,
                    pitch: out_bytes,
                },
            )
            .unwrap();
        (host, per_fit)
    }

    fn psnr(a: &[u8], b: &[u8]) -> (f64, u8) {
        assert_eq!(a.len(), b.len());
        let worst = a
            .iter()
            .zip(b)
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap_or(0);
        let mse = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum::<f64>()
            / a.len() as f64;
        let db = if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        };
        (db, worst)
    }

    /// The kernels scale like `FrameScaler` (same filter, same letterbox):
    /// what the receiver sees does not depend on which path ran.
    #[test]
    fn gpu_scaling_matches_the_cpu_scaler() {
        let Some(context) = context() else { return };
        for (source, canvas) in [
            (FrameSize::new(1920, 1080), FrameSize::new(1280, 720)),
            (FrameSize::new(3840, 2160), FrameSize::new(1920, 1080)),
            (FrameSize::new(1080, 1920), FrameSize::new(1920, 1080)),
            (FrameSize::new(1366, 768), FrameSize::new(1280, 720)),
        ] {
            let raw = screen(source.width, source.height);
            let mut scaler = FrameScaler::default();
            let started = Instant::now();
            let rounds = 5;
            let mut cpu = None;
            for _ in 0..rounds {
                cpu = Some(scaler.fit(raw.clone(), canvas).unwrap());
            }
            let cpu_time = started.elapsed() / rounds;
            let cpu = cpu.unwrap();
            let (gpu, gpu_time) = gpu_fit(
                &context,
                &raw,
                OutputFormat::Rgb(RgbOrder::Bgra),
                canvas,
                20,
            );
            let (db, worst) = psnr(&gpu, &cpu.data);
            eprintln!(
                "{}x{} -> {}x{}: GPU vs CPU scaler PSNR {db:.1} dB (max {worst}); CPU {cpu_time:?}, upload + GPU {gpu_time:?}",
                source.width, source.height, canvas.width, canvas.height
            );
            assert!(db > 45.0 && worst <= 3, "{db:.1} dB, max {worst}");
        }
    }

    /// RGB → NV12 on the GPU gives the `yuv` crate's samples (ferricast's
    /// CPU conversion).
    #[test]
    fn gpu_csc_matches_the_cpu_conversion() {
        let Some(context) = context() else { return };
        let raw = screen(1280, 720);
        let colour = ColourDescription::BT709_LIMITED;
        let format = OutputFormat::Nv12(YuvColour::default());
        let (gpu, _) = gpu_fit(&context, &raw, format, FrameSize::new(1280, 720), 1);
        let layout = Nv12Layout::packed(1280, 720);
        let mut cpu = vec![0; layout.len()];
        let view = FrameView::new(&raw, 1280, 720).unwrap();
        nv12::write(&view, colour, layout, &mut cpu, BackendId::Nvenc).unwrap();
        let (db, worst) = psnr(&gpu, &cpu);
        let differ = gpu.iter().zip(&cpu).filter(|(a, b)| a != b).count();
        eprintln!("RGB -> NV12: {differ} samples differ, max {worst}, PSNR {db:.1} dB");
        assert!(worst <= 1, "max difference {worst}");
    }

    /// P010 → NV12: the GPU rounds to 8 bits; the CPU path
    /// (`frame_view::eight_bit`) keeps the top byte (truncates).
    #[test]
    fn gpu_narrowing_rounds_where_the_cpu_truncates() {
        let Some(context) = context() else { return };
        let (width, height) = (640_u32, 360_u32);
        let samples = width as usize * height as usize * 3 / 2;
        let data: Vec<u8> = (0..samples)
            .flat_map(|i| ((((i * 37) % 1024) as u16) << 6).to_le_bytes())
            .collect();
        let raw = RawFrame {
            width,
            height,
            stride: width * 2,
            format: PixelFormat::P010,
            data: Bytes::from(data),
            timestamp_us: 0,
            colour: None,
            hdr: None,
        };
        let format = OutputFormat::Nv12(YuvColour::default());
        let (gpu, _) = gpu_fit(&context, &raw, format, FrameSize::new(width, height), 1);
        let cpu = crate::frame_view::eight_bit(raw.clone(), width, height).unwrap();
        let rounded: Vec<u8> = raw
            .data
            .chunks_exact(2)
            .map(|b| ((u32::from(u16::from_le_bytes([b[0], b[1]])) + 128) >> 8).min(255) as u8)
            .collect();
        assert_eq!(gpu, rounded, "GPU narrowing rounds");
        let (db, worst) = psnr(&gpu, &cpu.data);
        eprintln!("P010 -> NV12: GPU (rounded) vs CPU (truncated) max {worst}, PSNR {db:.1} dB");
        assert!(worst <= 1);
    }
}
