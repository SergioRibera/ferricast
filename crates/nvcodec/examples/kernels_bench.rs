//! Throughput of the GPU image kernels, per frame:
//!
//! ```text
//! cargo run --release -p ferricast-nvcodec --example kernels_bench -- [frames]
//! ```
//!
//! Each case queues `frames` conversions back to back on one stream and
//! waits once: the wall time per frame is the GPU time of the kernel (the
//! host only queues). The host cost of one `convert` call (checks, texture
//! object for array input, launch) is reported separately. Run it on an
//! idle GPU.
#![allow(
    clippy::print_stdout,
    clippy::cast_precision_loss,
    reason = "an example reporting its measurements"
)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use ferricast_nvcodec::cuda::{ArrayView, Context, Device, DeviceBuffer, Stream};
use ferricast_nvcodec::kernels::{
    Converter, Geometry, Input, Orientation, Output, OutputFormat, Rect, RgbOrder, Rotation, Size,
    YuvColour,
};
use ferricast_nvcodec::loader::CudaLibrary;
use ferricast_nvcodec::sys::cuda as sys;

struct Case {
    name: &'static str,
    source: Size,
    input: Source,
    format: OutputFormat,
    geometry: Geometry,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Source {
    Pitched,
    Array,
    P016,
}

/// A 4 × u8 CUDA array, destroyed on drop.
struct Array {
    raw: sys::CUarray,
    context: Arc<Context>,
}

impl Array {
    fn new(context: &Arc<Context>, size: Size) -> Result<Self, Box<dyn std::error::Error>> {
        let cuda = CudaLibrary::get()?;
        let _current = context.guard()?;
        let descriptor = sys::CUDA_ARRAY_DESCRIPTOR {
            Width: size.width as usize,
            Height: size.height as usize,
            Format: sys::CUarray_format::CU_AD_FORMAT_UNSIGNED_INT8,
            NumChannels: 4,
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: valid descriptor and out-pointer, context current.
        let status = unsafe { (cuda.array_create)(&raw mut raw, &raw const descriptor) };
        if status != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuArrayCreate: {status:?}").into());
        }
        Ok(Self {
            raw,
            context: Arc::clone(context),
        })
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        if let (Ok(cuda), Ok(_current)) = (CudaLibrary::get(), self.context.guard()) {
            // SAFETY: owned array; the benchmark synchronised its work.
            unsafe { (cuda.array_destroy)(self.raw) };
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frames: u32 = std::env::args()
        .nth(1)
        .map_or(Ok(500), |frames| frames.parse())?;
    let context = Context::primary(Device::get(0)?)?;
    let stream = Stream::new(&context)?;
    let started = Instant::now();
    let mut converter = Converter::new(&context)?;
    println!("kernels loaded (JIT) in {:?}", started.elapsed());
    for case in &cases() {
        run(&context, &stream, &mut converter, case, frames)?;
    }
    Ok(())
}

fn cases() -> [Case; 11] {
    let hd = Size::new(1920, 1080);
    let uhd = Size::new(3840, 2160);
    let portrait = Size::new(1080, 1920);
    let nv12 = OutputFormat::Nv12(YuvColour::default());
    let bgra = OutputFormat::Rgb(RgbOrder::Bgra);
    let quarter = Orientation {
        mirrored: false,
        rotation: Rotation::Clockwise270,
    };
    [
        Case {
            name: "1080p -> 720p BGRA (pitched)",
            source: hd,
            input: Source::Pitched,
            format: bgra,
            geometry: Geometry::new(hd, Size::new(1280, 720)),
        },
        Case {
            name: "1080p -> 720p BGRA (array)",
            source: hd,
            input: Source::Array,
            format: bgra,
            geometry: Geometry::new(hd, Size::new(1280, 720)),
        },
        Case {
            name: "1080p -> 720p NV12 (scale + CSC, pitched)",
            source: hd,
            input: Source::Pitched,
            format: nv12,
            geometry: Geometry::new(hd, Size::new(1280, 720)),
        },
        Case {
            name: "1080p -> 720p NV12 (scale + CSC, array)",
            source: hd,
            input: Source::Array,
            format: nv12,
            geometry: Geometry::new(hd, Size::new(1280, 720)),
        },
        Case {
            name: "1080p BGRA -> NV12 (CSC only)",
            source: hd,
            input: Source::Pitched,
            format: nv12,
            geometry: Geometry::new(hd, hd),
        },
        Case {
            name: "4K -> 1080p BGRA",
            source: uhd,
            input: Source::Pitched,
            format: bgra,
            geometry: Geometry::new(uhd, hd),
        },
        Case {
            name: "4K -> 1080p NV12",
            source: uhd,
            input: Source::Pitched,
            format: nv12,
            geometry: Geometry::new(uhd, hd),
        },
        Case {
            name: "1080x1920 rotate -> 1920x1080 BGRA",
            source: portrait,
            input: Source::Pitched,
            format: bgra,
            geometry: Geometry::new(portrait, hd).with_orientation(quarter),
        },
        Case {
            name: "1080x1920 letterbox -> 1920x1080 BGRA",
            source: portrait,
            input: Source::Pitched,
            format: bgra,
            geometry: Geometry::new(portrait, hd).with_placement(Rect::new(656, 0, 606, 1080)),
        },
        Case {
            name: "4K P010 -> NV12",
            source: uhd,
            input: Source::P016,
            format: nv12,
            geometry: Geometry::new(uhd, uhd),
        },
        Case {
            name: "4K P010 -> 1080p NV12",
            source: uhd,
            input: Source::P016,
            format: nv12,
            geometry: Geometry::new(uhd, hd),
        },
    ]
}

fn run(
    context: &Arc<Context>,
    stream: &Stream,
    converter: &mut Converter,
    case: &Case,
    frames: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let (width, height) = (case.source.width as usize, case.source.height as usize);
    let pitched = match case.input {
        Source::P016 => DeviceBuffer::pitched(context, width * 2, height * 3 / 2)?,
        Source::Pitched | Source::Array => DeviceBuffer::pitched(context, width * 4, height)?,
    };
    let array = Array::new(context, case.source)?;
    // SAFETY: a live 4 × u8 array of this size, freed after the
    // benchmark synchronised.
    let view =
        unsafe { ArrayView::from_raw(array.raw, context, case.source.width, case.source.height) };
    let input = match case.input {
        Source::Pitched => Input::Packed {
            view: pitched.as_view(),
            order: RgbOrder::Bgra,
            size: case.source,
        },
        Source::Array => Input::Array {
            array: view,
            order: RgbOrder::Bgra,
        },
        Source::P016 => Input::P016 {
            view: pitched.as_view(),
            chroma_row: height,
            size: case.source,
        },
    };
    let (width_bytes, rows) = case.format.buffer_layout(case.geometry.canvas);
    let target = DeviceBuffer::pitched(context, width_bytes, rows)?;
    let output = Output {
        buffer: &target,
        format: case.format,
        chroma_row: None,
    };
    // Warm up.
    for _ in 0..10 {
        converter.convert(input, output, &case.geometry, stream)?;
    }
    stream.synchronize()?;
    let start = Instant::now();
    for _ in 0..frames {
        converter.convert(input, output, &case.geometry, stream)?;
    }
    stream.synchronize()?;
    let total = start.elapsed();
    // One frame at a time, as a pipeline runs: the host cost of a call
    // without a queue to wait behind, and the latency to completion.
    let (mut call, mut latency) = (Duration::ZERO, Duration::ZERO);
    let rounds = frames.min(200);
    for _ in 0..rounds {
        let start = Instant::now();
        converter.convert(input, output, &case.geometry, stream)?;
        call += start.elapsed();
        stream.synchronize()?;
        latency += start.elapsed();
    }
    let per = |time: Duration, count: u32| time.as_secs_f64() * 1e6 / f64::from(count);
    println!(
        "{:<44} {:>7.1} µs/frame GPU (queued), {:>7.1} µs latency, {:>5.1} µs host per call",
        case.name,
        per(total, frames),
        per(latency, rounds),
        per(call, rounds),
    );
    Ok(())
}
