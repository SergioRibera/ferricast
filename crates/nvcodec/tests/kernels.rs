//! The GPU image kernels against CPU reference implementations written
//! independently of the kernels: bit-exact where the operation moves or
//! narrows samples (crop, rotate / flip, letterbox fill, P016 → NV12,
//! RGB → NV12 at the same size), within one level and above a PSNR
//! threshold where it resamples (an f64 implementation of
//! `fast_image_resize`'s convolution, the CPU scaler ferricast uses).
//!
//! Every test prints why and returns early without an NVIDIA driver; the
//! DMA-BUF tests also without GBM / EGL.

#![allow(
    clippy::missing_panics_doc,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::print_stderr,
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    reason = "hardware tests: panics are failures, sizes are small and positive"
)]

use std::sync::Arc;

use ferricast_nvcodec::Error;
use ferricast_nvcodec::cuda::{
    ArrayView, Context, Device, DeviceBuffer, HostImage, HostImageMut, Region, Stream,
};
use ferricast_nvcodec::kernels::{
    Converter, Filter, FramePool, Geometry, Input, Matrix, Orientation, Output, OutputFormat,
    Range, Rect, RgbOrder, Rotation, Size, YuvColour,
};
use ferricast_nvcodec::loader::CudaLibrary;
use ferricast_nvcodec::sys::cuda as sys;

// ---------------------------------------------------------------- setup

struct Setup {
    context: Arc<Context>,
    stream: Stream,
    converter: Converter,
}

fn setup() -> Option<Setup> {
    let opened = CudaLibrary::get()
        .and_then(|_| Device::get(0))
        .and_then(Context::primary);
    let context = match opened {
        Ok(context) => context,
        Err(error) => {
            eprintln!("skipping: no usable NVIDIA CUDA driver: {error}");
            return None;
        }
    };
    let stream = Stream::new(&context).unwrap();
    let converter = Converter::new(&context).unwrap();
    Some(Setup {
        context,
        stream,
        converter,
    })
}

fn upload(context: &Arc<Context>, data: &[u8], width_bytes: usize, rows: usize) -> DeviceBuffer {
    let mut buffer = DeviceBuffer::pitched(context, width_bytes, rows).unwrap();
    buffer
        .upload(
            Region::new(width_bytes, rows),
            HostImage {
                data,
                pitch: width_bytes,
            },
        )
        .unwrap();
    // A synchronous upload from pageable memory may return before the
    // DMA lands, and the kernels run on a non-blocking stream.
    context.synchronize().unwrap();
    buffer
}

fn download(buffer: &DeviceBuffer, width_bytes: usize, rows: usize) -> Vec<u8> {
    let mut host = vec![0; width_bytes * rows];
    buffer
        .download(
            Region::new(width_bytes, rows),
            HostImageMut {
                data: &mut host,
                pitch: width_bytes,
            },
        )
        .unwrap();
    host
}

/// A CUDA array of 4 × u8 elements holding `pixels`, destroyed on drop.
struct Array {
    raw: sys::CUarray,
    width: u32,
    height: u32,
    context: Arc<Context>,
}

impl Array {
    fn new(context: &Arc<Context>, pixels: &[u8], width: u32, height: u32) -> Self {
        let cuda = CudaLibrary::get().unwrap();
        let _current = context.guard().unwrap();
        let descriptor = sys::CUDA_ARRAY_DESCRIPTOR {
            Width: width as usize,
            Height: height as usize,
            Format: sys::CUarray_format::CU_AD_FORMAT_UNSIGNED_INT8,
            NumChannels: 4,
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: test-only driver calls with valid descriptors and
        // out-pointers; the copy reads `height` rows of `width * 4` bytes.
        unsafe {
            assert_eq!(
                (cuda.array_create)(&raw mut raw, &raw const descriptor),
                sys::CUresult::CUDA_SUCCESS
            );
            let copy = sys::CUDA_MEMCPY2D {
                srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_HOST,
                srcHost: pixels.as_ptr().cast(),
                srcPitch: width as usize * 4,
                dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_ARRAY,
                dstArray: raw,
                WidthInBytes: width as usize * 4,
                Height: height as usize,
                ..Default::default()
            };
            assert_eq!(
                (cuda.memcpy_2d)(&raw const copy),
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!((cuda.ctx_synchronize)(), sys::CUresult::CUDA_SUCCESS);
        }
        Self {
            raw,
            width,
            height,
            context: Arc::clone(context),
        }
    }

    fn view(&self) -> ArrayView<'_> {
        // SAFETY: a live 4 × u8 array of this size on this context, kept
        // until `self` drops (after the tests synchronise).
        unsafe { ArrayView::from_raw(self.raw, &self.context, self.width, self.height) }
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        let cuda = CudaLibrary::get().unwrap();
        let _current = self.context.guard().unwrap();
        // SAFETY: this value owns the array; work using it was synchronised.
        unsafe { (cuda.array_destroy)(self.raw) };
    }
}

// ------------------------------------------------------- test pictures

/// Deterministic noise (xorshift), so every sample position differs.
fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        })
        .collect()
}

/// A smooth picture with some text-like edges: what screens look like.
fn screen(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height {
        for x in 0..width {
            let fx = x as f64 / width as f64;
            let fy = y as f64 / height as f64;
            let glyph = ((x / 3) % 7 == 0 || (y / 5) % 9 == 0) && (x / 40 + y / 24) % 3 == 0;
            let base = |v: f64| (v * 255.0) as u8;
            if glyph {
                pixels.extend_from_slice(&[20, 20, 20, 255]);
            } else {
                pixels.extend_from_slice(&[
                    base(fx),
                    base(fy),
                    base(0.5 + 0.5 * (fx * 9.0 + fy * 4.0).sin()),
                    255,
                ]);
            }
        }
    }
    pixels
}

// ------------------------------------------------- CPU reference: geometry

/// Where (x, y) of a `w` × `h` picture goes: a horizontal mirror first, then
/// a clockwise rotation, step by step like the `image` crate.
fn forward(o: Orientation, w: usize, h: usize, x: usize, y: usize) -> (usize, usize) {
    let x = if o.mirrored { w - 1 - x } else { x };
    match o.rotation {
        Rotation::None => (x, y),
        Rotation::Clockwise90 => (h - 1 - y, x),
        Rotation::Half => (w - 1 - x, h - 1 - y),
        Rotation::Clockwise270 => (y, w - 1 - x),
    }
}

/// `channels`-sample pixels cropped and oriented.
fn crop_orient(
    src: &[f64],
    width: usize,
    channels: usize,
    crop: Rect,
    o: Orientation,
) -> (Vec<f64>, usize, usize) {
    let (cw, ch) = (crop.width as usize, crop.height as usize);
    let (ow, oh) = if o.swaps_axes() { (ch, cw) } else { (cw, ch) };
    let mut out = vec![0.0; ow * oh * channels];
    for y in 0..ch {
        for x in 0..cw {
            let (tx, ty) = forward(o, cw, ch, x, y);
            let from = ((y + crop.y as usize) * width + x + crop.x as usize) * channels;
            let to = (ty * ow + tx) * channels;
            out[to..to + channels].copy_from_slice(&src[from..from + channels]);
        }
    }
    (out, ow, oh)
}

fn filter_weight(filter: Filter, t: f64) -> f64 {
    match filter {
        Filter::Bilinear => (1.0 - t.abs()).max(0.0),
        Filter::Box => {
            if t > -0.5 && t <= 0.5 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// `fast_image_resize`'s convolution coefficients (adaptive support), in
/// f64: per output position, the first input index and normalised weights.
fn coefficients(input: usize, output: usize, filter: Filter) -> Vec<(usize, Vec<f64>)> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = match filter {
        Filter::Bilinear => 1.0,
        Filter::Box => 0.5,
    } * filter_scale;
    (0..output)
        .map(|o| {
            let center = (o as f64 + 0.5) * scale;
            let lo = (center - support).floor().max(0.0) as usize;
            let hi = ((center + support).ceil() as usize).min(input);
            let weights: Vec<f64> = (lo..hi)
                .map(|i| filter_weight(filter, (i as f64 - (center - 0.5)) / filter_scale))
                .collect();
            let sum: f64 = weights.iter().sum();
            (lo, weights.into_iter().map(|w| w / sum).collect())
        })
        .collect()
}

/// Separable resampling in f64 (horizontal then vertical, no rounding in
/// between).
fn resample(
    src: &[f64],
    (w, h): (usize, usize),
    channels: usize,
    (dw, dh): (usize, usize),
    filter: Filter,
) -> Vec<f64> {
    let horizontal = coefficients(w, dw, filter);
    let vertical = coefficients(h, dh, filter);
    let mut rows = vec![0.0; dw * h * channels];
    for y in 0..h {
        for (x, (lo, weights)) in horizontal.iter().enumerate() {
            for (k, weight) in weights.iter().enumerate() {
                for c in 0..channels {
                    rows[(y * dw + x) * channels + c] +=
                        weight * src[(y * w + lo + k) * channels + c];
                }
            }
        }
    }
    let mut out = vec![0.0; dw * dh * channels];
    for (y, (lo, weights)) in vertical.iter().enumerate() {
        for (k, weight) in weights.iter().enumerate() {
            for x in 0..dw * channels {
                out[y * dw * channels + x] += weight * rows[(lo + k) * dw * channels + x];
            }
        }
    }
    out
}

/// The whole geometry on `channels`-sample pixels, before rounding: the
/// placement resampled, everything else `fill`.
fn reference(
    src: &[f64],
    source: Size,
    channels: usize,
    geometry: &Geometry,
    fill: &[f64],
) -> Vec<f64> {
    let (oriented, ow, oh) = crop_orient(
        src,
        source.width as usize,
        channels,
        geometry.crop,
        geometry.orientation,
    );
    let place = geometry.placement;
    let (pw, ph) = (place.width as usize, place.height as usize);
    let placed = resample(&oriented, (ow, oh), channels, (pw, ph), geometry.filter);
    let (cw, ch) = (
        geometry.canvas.width as usize,
        geometry.canvas.height as usize,
    );
    let mut out: Vec<f64> = fill
        .iter()
        .copied()
        .cycle()
        .take(cw * ch * channels)
        .collect();
    for y in 0..ph {
        let row = (y + place.y as usize) * cw + place.x as usize;
        out[row * channels..(row + pw) * channels]
            .copy_from_slice(&placed[y * pw * channels..(y + 1) * pw * channels]);
    }
    out
}

fn round_u8(v: f64) -> u8 {
    (v + 0.5).floor().clamp(0.0, 255.0) as u8
}

fn to_f64(bytes: &[u8]) -> Vec<f64> {
    bytes.iter().map(|&b| f64::from(b)).collect()
}

// --------------------------------------------------- CPU reference: colour

/// BT.709 / BT.601 at 13 bits, limited / full: the `yuv` crate's prebuilt
/// tables (what ferricast's CPU conversion uses).
fn csc_table(colour: YuvColour) -> ([i32; 9], i32) {
    let table = match (colour.matrix, colour.range) {
        (Matrix::Bt709, Range::Limited) => [1496, 5032, 508, -824, -2774, 3598, 3598, -3268, -330],
        (Matrix::Bt709, Range::Full) => [1742, 5859, 591, -939, -3157, 4096, 4096, -3720, -376],
        (Matrix::Bt601, Range::Limited) => [2104, 4130, 802, -1214, -2384, 3598, 3598, -3013, -585],
        (Matrix::Bt601, Range::Full) => [2449, 4809, 934, -1382, -2714, 4096, 4096, -3430, -666],
    };
    let black = if colour.range == Range::Limited {
        16
    } else {
        0
    };
    (table, black)
}

/// RGB → NV12 like the `yuv` crate: luma per pixel, chroma from the
/// rounded average of each 2 × 2 block (of the pixels that exist).
fn nv12_reference(
    rgb: &[u8],
    order: RgbOrder,
    width: usize,
    height: usize,
    colour: YuvColour,
) -> Vec<u8> {
    let (k, black) = csc_table(colour);
    let bias_y = (black << 13) + 4095;
    let bias_uv = (128 << 13) + 4095;
    let pairs = width.div_ceil(2);
    let mut luma = vec![0_u8; pairs * 2 * height];
    let mut chroma = vec![0_u8; pairs * 2 * height.div_ceil(2)];
    let pixel = |x: usize, y: usize| {
        let p = &rgb[(y * width + x) * 4..][..4];
        match order {
            RgbOrder::Bgra => [i32::from(p[2]), i32::from(p[1]), i32::from(p[0])],
            RgbOrder::Rgba => [i32::from(p[0]), i32::from(p[1]), i32::from(p[2])],
        }
    };
    for by in 0..height.div_ceil(2) {
        for bx in 0..pairs {
            let mut sum = [0; 3];
            let mut n = 0;
            for y in (2 * by)..(2 * by + 2).min(height) {
                for x in (2 * bx)..(2 * bx + 2).min(width) {
                    let [r, g, b] = pixel(x, y);
                    luma[y * pairs * 2 + x] =
                        ((r * k[0] + g * k[1] + b * k[2] + bias_y) >> 13).clamp(0, 255) as u8;
                    sum = [sum[0] + r, sum[1] + g, sum[2] + b];
                    n += 1;
                }
            }
            let [r, g, b] = sum.map(|s| (s + n / 2) / n);
            let at = by * pairs * 2 + 2 * bx;
            chroma[at] = ((r * k[3] + g * k[4] + b * k[5] + bias_uv) >> 13).clamp(0, 255) as u8;
            chroma[at + 1] = ((r * k[6] + g * k[7] + b * k[8] + bias_uv) >> 13).clamp(0, 255) as u8;
        }
    }
    luma.extend_from_slice(&chroma);
    luma
}

// ------------------------------------------------------------ comparison

fn psnr(got: &[u8], want: &[u8]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mse = got
        .iter()
        .zip(want)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>()
        / got.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

fn assert_exact(got: &[u8], want: &[u8], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(index) = got.iter().zip(want).position(|(a, b)| a != b) {
        let wrong = got.iter().zip(want).filter(|(a, b)| a != b).count();
        panic!(
            "{what}: {wrong} of {} bytes differ, first at {index}: got {}, want {}",
            got.len(),
            got[index],
            want[index]
        );
    }
}

/// Within one level everywhere and above `min_psnr`.
fn assert_close(got: &[u8], want: &[u8], min_psnr: f64, what: &str) {
    let worst = got
        .iter()
        .zip(want)
        .map(|(&a, &b)| a.abs_diff(b))
        .max()
        .unwrap_or(0);
    let quality = psnr(got, want);
    eprintln!("{what}: max difference {worst}, PSNR {quality:.1} dB");
    assert!(worst <= 1, "{what}: a sample is {worst} levels off");
    assert!(
        quality >= min_psnr,
        "{what}: PSNR {quality:.1} dB < {min_psnr}"
    );
}

// --------------------------------------------------------- RGB kernels

/// Runs an RGB → RGB conversion from pitched memory and from a CUDA array,
/// checks both against each other and returns the output.
fn rgb_both(
    setup: &mut Setup,
    pixels: &[u8],
    source: Size,
    order: RgbOrder,
    out: RgbOrder,
    geometry: &Geometry,
) -> Vec<u8> {
    let (w, h) = (source.width as usize, source.height as usize);
    let input = upload(&setup.context, pixels, w * 4, h);
    let array = Array::new(&setup.context, pixels, source.width, source.height);
    let format = OutputFormat::Rgb(out);
    let (width_bytes, rows) = format.buffer_layout(geometry.canvas);
    let mut results = Vec::new();
    for from_array in [false, true] {
        let mut target = DeviceBuffer::pitched(&setup.context, width_bytes, rows).unwrap();
        target.fill_async(0, &setup.stream).unwrap();
        let input = if from_array {
            Input::Array {
                array: array.view(),
                order,
            }
        } else {
            Input::Packed {
                view: input.as_view(),
                order,
                size: source,
            }
        };
        let output = Output {
            buffer: &target,
            format,
            chroma_row: None,
        };
        setup
            .converter
            .convert(input, output, geometry, &setup.stream)
            .unwrap();
        setup.stream.synchronize().unwrap();
        results.push(download(&target, width_bytes, rows));
    }
    assert_exact(&results[1], &results[0], "array vs pitched input");
    results.swap_remove(0)
}

fn rgb_reference(pixels: &[u8], source: Size, geometry: &Geometry, swap: bool) -> Vec<u8> {
    let out = reference(
        &to_f64(pixels),
        source,
        4,
        geometry,
        &[0.0, 0.0, 0.0, 255.0],
    );
    let mut bytes: Vec<u8> = out.into_iter().map(round_u8).collect();
    if swap {
        for pixel in bytes.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
    }
    bytes
}

#[test]
fn rgb_copy_is_bit_exact_at_odd_sizes() {
    let Some(mut setup) = setup() else { return };
    for (w, h) in [(37, 23), (1, 17), (17, 1), (1, 1), (64, 64)] {
        let source = Size::new(w, h);
        let pixels = noise(w as usize * h as usize * 4, u64::from(w * 31 + h));
        let geometry = Geometry::new(source, source);
        let got = rgb_both(
            &mut setup,
            &pixels,
            source,
            RgbOrder::Bgra,
            RgbOrder::Bgra,
            &geometry,
        );
        assert_exact(&got, &pixels, &format!("copy {w}x{h}"));
        let swapped = rgb_both(
            &mut setup,
            &pixels,
            source,
            RgbOrder::Bgra,
            RgbOrder::Rgba,
            &geometry,
        );
        assert_exact(
            &swapped,
            &rgb_reference(&pixels, source, &geometry, true),
            &format!("swap {w}x{h}"),
        );
    }
}

#[test]
fn every_orientation_is_bit_exact() {
    let Some(mut setup) = setup() else { return };
    for (w, h) in [(61, 37), (1, 9), (9, 1), (2, 3)] {
        let source = Size::new(w, h);
        let pixels = noise(w as usize * h as usize * 4, 5);
        for orientation in Orientation::ALL {
            let canvas = orientation.oriented_size(source);
            let geometry = Geometry::new(source, canvas).with_orientation(orientation);
            let got = rgb_both(
                &mut setup,
                &pixels,
                source,
                RgbOrder::Rgba,
                RgbOrder::Rgba,
                &geometry,
            );
            assert_exact(
                &got,
                &rgb_reference(&pixels, source, &geometry, false),
                &format!("{w}x{h} {orientation:?}"),
            );
        }
    }
}

#[test]
fn crop_and_orientation_compose_bit_exact() {
    let Some(mut setup) = setup() else { return };
    let source = Size::new(50, 40);
    let pixels = noise(50 * 40 * 4, 9);
    let crop = Rect::new(7, 3, 31, 22);
    for orientation in Orientation::ALL {
        let canvas = orientation.oriented_size(crop.size());
        let geometry = Geometry::new(source, canvas)
            .with_crop(crop)
            .with_orientation(orientation);
        let got = rgb_both(
            &mut setup,
            &pixels,
            source,
            RgbOrder::Bgra,
            RgbOrder::Bgra,
            &geometry,
        );
        assert_exact(
            &got,
            &rgb_reference(&pixels, source, &geometry, false),
            &format!("crop {crop:?} {orientation:?}"),
        );
    }
}

#[test]
fn scaling_matches_the_reference_filter() {
    let Some(mut setup) = setup() else { return };
    let cases = [
        (Size::new(1920, 1080), Size::new(1280, 720)),
        (Size::new(1280, 720), Size::new(1920, 1080)),
        (Size::new(333, 77), Size::new(41, 19)),
        (Size::new(7, 5), Size::new(23, 3)),
        (Size::new(1, 40), Size::new(1, 7)),
    ];
    for (source, canvas) in cases {
        let pixels = screen(source.width, source.height);
        for filter in [Filter::Bilinear, Filter::Box] {
            let geometry = Geometry::new(source, canvas).with_filter(filter);
            let got = rgb_both(
                &mut setup,
                &pixels,
                source,
                RgbOrder::Bgra,
                RgbOrder::Bgra,
                &geometry,
            );
            assert_close(
                &got,
                &rgb_reference(&pixels, source, &geometry, false),
                50.0,
                &format!("{source:?} -> {canvas:?} {filter:?}"),
            );
        }
    }
}

#[test]
fn rotated_portrait_is_letterboxed() {
    let Some(mut setup) = setup() else { return };
    // A portrait 1080x1920 screen into a 1920x1080 session: ferricast's
    // letterbox puts it 606x1080 at x = 656.
    let source = Size::new(1080, 1920);
    let canvas = Size::new(1920, 1080);
    let pixels = screen(source.width, source.height);
    let geometry = Geometry::new(source, canvas).with_placement(Rect::new(656, 0, 606, 1080));
    let got = rgb_both(
        &mut setup,
        &pixels,
        source,
        RgbOrder::Bgra,
        RgbOrder::Bgra,
        &geometry,
    );
    let want = rgb_reference(&pixels, source, &geometry, false);
    assert_close(&got, &want, 50.0, "1080x1920 letterboxed into 1920x1080");
    for x in [0, 655, 1262, 1919] {
        assert_eq!(
            &got[(500 * 1920 + x) * 4..][..4],
            &[0, 0, 0, 255],
            "bar at x = {x}"
        );
    }
    // And the same picture turned upright first (a rotated output).
    let upright = geometry
        .with_orientation(Orientation {
            mirrored: false,
            rotation: Rotation::Clockwise270,
        })
        .with_placement(Rect::of(canvas));
    let got = rgb_both(
        &mut setup,
        &pixels,
        source,
        RgbOrder::Bgra,
        RgbOrder::Bgra,
        &upright,
    );
    assert_close(
        &got,
        &rgb_reference(&pixels, source, &upright, false),
        50.0,
        "1080x1920 rotated to 1920x1080",
    );
}

#[test]
fn downscaling_text_does_not_alias() {
    let Some(mut setup) = setup() else { return };
    // One-pixel vertical lines: a 4x downscale must average them to grey
    // (25% coverage), not keep or drop whole lines.
    let source = Size::new(256, 16);
    let mut pixels = vec![255_u8; 256 * 16 * 4];
    for (i, pixel) in pixels.chunks_exact_mut(4).enumerate() {
        if i % 4 == 0 {
            pixel[..3].fill(0);
        }
    }
    for filter in [Filter::Bilinear, Filter::Box] {
        let geometry = Geometry::new(source, Size::new(64, 4)).with_filter(filter);
        let got = rgb_both(
            &mut setup,
            &pixels,
            source,
            RgbOrder::Bgra,
            RgbOrder::Bgra,
            &geometry,
        );
        let interior: Vec<u8> = got.chunks_exact(4).skip(1).take(60).map(|p| p[0]).collect();
        let (lo, hi) = (
            interior.iter().min().unwrap(),
            interior.iter().max().unwrap(),
        );
        eprintln!("{filter:?}: interior {lo}..={hi}");
        assert!(
            hi - lo <= 2 && (185..=195).contains(lo),
            "{filter:?}: {interior:?}"
        );
    }
}

// ---------------------------------------------------------- NV12 output

fn rgb_to_nv12(
    setup: &mut Setup,
    pixels: &[u8],
    source: Size,
    order: RgbOrder,
    geometry: &Geometry,
    colour: YuvColour,
) -> Vec<u8> {
    let (w, h) = (source.width as usize, source.height as usize);
    let input = upload(&setup.context, pixels, w * 4, h);
    let array = Array::new(&setup.context, pixels, source.width, source.height);
    let format = OutputFormat::Nv12(colour);
    let (width_bytes, rows) = format.buffer_layout(geometry.canvas);
    let mut results = Vec::new();
    for from_array in [false, true] {
        let mut target = DeviceBuffer::pitched(&setup.context, width_bytes, rows).unwrap();
        target.fill_async(0, &setup.stream).unwrap();
        let input = if from_array {
            Input::Array {
                array: array.view(),
                order,
            }
        } else {
            Input::Packed {
                view: input.as_view(),
                order,
                size: source,
            }
        };
        setup
            .converter
            .convert(
                input,
                Output {
                    buffer: &target,
                    format,
                    chroma_row: None,
                },
                geometry,
                &setup.stream,
            )
            .unwrap();
        setup.stream.synchronize().unwrap();
        results.push(download(&target, width_bytes, rows));
    }
    assert_exact(&results[1], &results[0], "NV12 from array vs pitched input");
    results.swap_remove(0)
}

#[test]
fn rgb_to_nv12_is_bit_exact_at_the_same_size() {
    let Some(mut setup) = setup() else { return };
    for (w, h) in [(64, 32), (37, 23), (1, 1), (3, 1), (1, 5)] {
        let source = Size::new(w, h);
        let pixels = noise(w as usize * h as usize * 4, 3);
        for colour in [
            YuvColour::default(),
            YuvColour {
                matrix: Matrix::Bt601,
                range: Range::Full,
            },
        ] {
            for order in [RgbOrder::Bgra, RgbOrder::Rgba] {
                let geometry = Geometry::new(source, source);
                let mut got = rgb_to_nv12(&mut setup, &pixels, source, order, &geometry, colour);
                mask_luma_padding(&mut got, source, 1);
                let want = nv12_reference(&pixels, order, w as usize, h as usize, colour);
                assert_exact(&got, &want, &format!("{w}x{h} {order:?} {colour:?}"));
            }
        }
    }
}

#[test]
fn scaled_rgb_to_nv12_matches_scale_then_convert() {
    let Some(mut setup) = setup() else { return };
    let source = Size::new(1920, 1080);
    let canvas = Size::new(1280, 720);
    let pixels = screen(1920, 1080);
    let geometry = Geometry::new(source, canvas);
    let got = rgb_to_nv12(
        &mut setup,
        &pixels,
        source,
        RgbOrder::Bgra,
        &geometry,
        YuvColour::default(),
    );
    let scaled = rgb_reference(&pixels, source, &geometry, false);
    let want = nv12_reference(&scaled, RgbOrder::Bgra, 1280, 720, YuvColour::default());
    // One level of RGB rounding can move luma / chroma by one.
    assert_close(&got, &want, 50.0, "1920x1080 -> 1280x720 NV12");
    // Letterbox bars are black in the range.
    let boxed = geometry.with_placement(Rect::new(160, 0, 960, 720));
    let got = rgb_to_nv12(
        &mut setup,
        &pixels,
        source,
        RgbOrder::Bgra,
        &boxed,
        YuvColour::default(),
    );
    assert_eq!(got[100 * 1280 + 10], 16, "limited black luma");
    let chroma = 1280 * 720;
    assert_eq!(
        &got[chroma + 50 * 1280 + 20..][..2],
        &[128, 128],
        "neutral chroma"
    );
}

// ---------------------------------------------------------- YUV planes

/// Semi-planar `width` × `height` (luma rows, then interleaved chroma rows),
/// `sample_bytes` per sample, tightly packed with rows of `ceil(w/2)*2`
/// samples.
fn semi_planar_size(size: Size, sample_bytes: usize) -> (usize, usize) {
    (
        size.width.div_ceil(2) as usize * 2 * sample_bytes,
        size.height as usize + size.height.div_ceil(2) as usize,
    )
}

fn run_planes(
    setup: &mut Setup,
    data: &[u8],
    source: Size,
    input_bytes: usize,
    format: OutputFormat,
    geometry: &Geometry,
) -> Vec<u8> {
    let (in_width, in_rows) = semi_planar_size(source, input_bytes);
    let buffer = upload(&setup.context, data, in_width, in_rows);
    let view = buffer.as_view();
    let chroma_row = source.height as usize;
    let input = if input_bytes == 1 {
        Input::Nv12 {
            view,
            chroma_row,
            size: source,
        }
    } else {
        Input::P016 {
            view,
            chroma_row,
            size: source,
        }
    };
    let (width_bytes, rows) = format.buffer_layout(geometry.canvas);
    let mut target = DeviceBuffer::pitched(&setup.context, width_bytes, rows).unwrap();
    target.fill_async(0, &setup.stream).unwrap();
    setup
        .converter
        .convert(
            input,
            Output {
                buffer: &target,
                format,
                chroma_row: None,
            },
            geometry,
            &setup.stream,
        )
        .unwrap();
    setup.stream.synchronize().unwrap();
    download(&target, width_bytes, rows)
}

/// The luma and chroma planes of tightly packed semi-planar bytes as
/// samples (8 or 16 bit).
fn planes_of(data: &[u8], size: Size, sample_bytes: usize) -> (Vec<f64>, Vec<f64>) {
    let (row_bytes, _) = semi_planar_size(size, sample_bytes);
    let samples = |bytes: &[u8]| -> Vec<f64> {
        if sample_bytes == 1 {
            to_f64(bytes)
        } else {
            bytes
                .chunks_exact(2)
                .map(|b| f64::from(u16::from_le_bytes([b[0], b[1]])))
                .collect()
        }
    };
    let width = size.width as usize * sample_bytes;
    let luma: Vec<f64> = data[..row_bytes * size.height as usize]
        .chunks_exact(row_bytes)
        .flat_map(|row| samples(&row[..width]))
        .collect();
    let chroma = samples(&data[row_bytes * size.height as usize..]);
    (luma, chroma)
}

/// The reference of a plane conversion: both planes through the geometry,
/// then quantised by `quantise`, packed like the kernels' output.
fn planes_reference(
    data: &[u8],
    source: Size,
    input_bytes: usize,
    geometry: &Geometry,
    black: (f64, f64),
    quantise: impl Fn(f64) -> u16,
    output_bytes: usize,
) -> Vec<u8> {
    let (luma, chroma) = planes_of(data, source, input_bytes);
    let chroma_geometry = Geometry {
        crop: Rect::new(
            geometry.crop.x / 2,
            geometry.crop.y / 2,
            geometry.crop.width.div_ceil(2),
            geometry.crop.height.div_ceil(2),
        ),
        canvas: Size::new(
            geometry.canvas.width.div_ceil(2),
            geometry.canvas.height.div_ceil(2),
        ),
        placement: Rect::new(
            geometry.placement.x / 2,
            geometry.placement.y / 2,
            geometry.placement.width.div_ceil(2),
            geometry.placement.height.div_ceil(2),
        ),
        ..*geometry
    };
    let luma = reference(&luma, source, 1, geometry, &[black.0]);
    let chroma_source = Size::new(source.width.div_ceil(2), source.height.div_ceil(2));
    let chroma = reference(
        &chroma,
        chroma_source,
        2,
        &chroma_geometry,
        &[black.1, black.1],
    );
    let canvas = geometry.canvas;
    let (row_bytes, _) = semi_planar_size(canvas, output_bytes);
    let mut out = Vec::new();
    let push = |out: &mut Vec<u8>, value: u16| {
        if output_bytes == 1 {
            out.push(value as u8);
        } else {
            out.extend_from_slice(&value.to_le_bytes());
        }
    };
    for row in luma.chunks_exact(canvas.width as usize) {
        let start = out.len();
        for &v in row {
            push(&mut out, quantise(v));
        }
        out.resize(start + row_bytes, 0);
    }
    for row in chroma.chunks_exact(canvas.width.div_ceil(2) as usize * 2) {
        for &v in row {
            push(&mut out, quantise(v));
        }
    }
    out
}

/// The `width_bytes` of each output row that the picture covers (the
/// padding byte of an odd-width luma row is not written).
fn mask_luma_padding(data: &mut [u8], canvas: Size, sample_bytes: usize) {
    let (row_bytes, _) = semi_planar_size(canvas, sample_bytes);
    let width = canvas.width as usize * sample_bytes;
    for row in data
        .chunks_exact_mut(row_bytes)
        .take(canvas.height as usize)
    {
        row[width..].fill(0);
    }
}

#[test]
fn p016_narrows_to_nv12_with_rounding_bit_exact() {
    let Some(mut setup) = setup() else { return };
    for (w, h) in [(64, 32), (37, 23), (1, 1), (1, 6), (5, 1)] {
        let source = Size::new(w, h);
        let (row_bytes, rows) = semi_planar_size(source, 2);
        let data = noise(row_bytes * rows, 11);
        let colour = YuvColour::default();
        let geometry = Geometry::new(source, source);
        let mut got = run_planes(
            &mut setup,
            &data,
            source,
            2,
            OutputFormat::Nv12(colour),
            &geometry,
        );
        mask_luma_padding(&mut got, source, 1);
        // (v + 128) >> 8, saturating at 255.
        let want = planes_reference(
            &data,
            source,
            2,
            &geometry,
            (16.0, 128.0),
            |v| ((v as u32 + 128) >> 8).min(255) as u16,
            1,
        );
        assert_exact(&got, &want, &format!("P016 -> NV12 {w}x{h}"));
    }
}

#[test]
fn nv12_and_p010_copy_rotate_and_scale() {
    let Some(mut setup) = setup() else { return };
    let source = Size::new(38, 22);
    let (row_bytes, rows) = semi_planar_size(source, 1);
    let nv12 = noise(row_bytes * rows, 21);
    let colour = YuvColour::default();
    let half_turn = Orientation {
        mirrored: true,
        rotation: Rotation::Clockwise90,
    };
    // Bit-exact: copy and orientation.
    for orientation in [Orientation::IDENTITY, half_turn, Orientation::FLIP_VERTICAL] {
        let canvas = orientation.oriented_size(source);
        let geometry = Geometry::new(source, canvas).with_orientation(orientation);
        let got = run_planes(
            &mut setup,
            &nv12,
            source,
            1,
            OutputFormat::Nv12(colour),
            &geometry,
        );
        let want = planes_reference(&nv12, source, 1, &geometry, (16.0, 128.0), |v| v as u16, 1);
        assert_exact(&got, &want, &format!("NV12 {orientation:?}"));
    }
    // P010 (10 significant bits) scaled and letterboxed.
    let (row_bytes, rows) = semi_planar_size(source, 2);
    let p010: Vec<u8> = noise(row_bytes * rows, 4)
        .chunks_exact(2)
        .flat_map(|b| (u16::from_le_bytes([b[0], b[1]]) & 0xffc0).to_le_bytes())
        .collect();
    let canvas = Size::new(30, 20);
    let geometry = Geometry::new(source, canvas).with_placement(Rect::new(2, 4, 26, 12));
    let got = run_planes(
        &mut setup,
        &p010,
        source,
        2,
        OutputFormat::P010(colour),
        &geometry,
    );
    let want = planes_reference(
        &p010,
        source,
        2,
        &geometry,
        (4096.0, 32768.0),
        |v| (((v / 64.0) + 0.5).floor().clamp(0.0, 1023.0) as u16) << 6,
        2,
    );
    let to_10_bits = |bytes: &[u8]| -> Vec<u8> {
        bytes
            .chunks_exact(2)
            .map(|b| (u16::from_le_bytes([b[0], b[1]]) >> 8) as u8)
            .collect()
    };
    assert_close(
        &to_10_bits(&got),
        &to_10_bits(&want),
        45.0,
        "P010 scaled + letterboxed (top 8 bits)",
    );
    // The bars are limited-range black.
    assert_eq!(u16::from_le_bytes([got[0], got[1]]), 64 << 6);
}

#[test]
fn p010_4k_narrows_to_nv12_without_scaling() {
    let Some(mut setup) = setup() else { return };
    let source = Size::new(3840, 2160);
    let (row_bytes, rows) = semi_planar_size(source, 2);
    let data = noise(row_bytes * rows, 99);
    let geometry = Geometry::new(source, source);
    let got = run_planes(
        &mut setup,
        &data,
        source,
        2,
        OutputFormat::Nv12(YuvColour::default()),
        &geometry,
    );
    let want: Vec<u8> = data
        .chunks_exact(2)
        .map(|b| ((u32::from(u16::from_le_bytes([b[0], b[1]])) + 128) >> 8).min(255) as u8)
        .collect();
    assert_exact(&got, &want, "4K P010 -> NV12");
}

// ------------------------------------------------------------- errors

#[test]
fn bad_inputs_are_refused_before_launch() {
    let Some(mut setup) = setup() else { return };
    let source = Size::new(16, 8);
    let input = DeviceBuffer::pitched(&setup.context, 64, 8).unwrap();
    let small = DeviceBuffer::pitched(&setup.context, 16, 4).unwrap();
    let format = OutputFormat::Rgb(RgbOrder::Bgra);
    let packed = Input::Packed {
        view: input.as_view(),
        order: RgbOrder::Bgra,
        size: source,
    };
    let mut refused = |input, output, geometry: Geometry| {
        let result = setup
            .converter
            .convert(input, output, &geometry, &setup.stream);
        assert!(
            matches!(result, Err(Error::InvalidKernelInput { .. })),
            "{result:?}"
        );
    };
    // Destination too small for the canvas.
    refused(
        packed,
        Output {
            buffer: &small,
            format,
            chroma_row: None,
        },
        Geometry::new(source, Size::new(8, 8)),
    );
    let target = DeviceBuffer::pitched(&setup.context, 32, 8).unwrap();
    let output = Output {
        buffer: &target,
        format,
        chroma_row: None,
    };
    // Source smaller than it claims.
    refused(
        Input::Packed {
            view: input.as_view(),
            order: RgbOrder::Bgra,
            size: Size::new(17, 8),
        },
        output,
        Geometry::new(Size::new(17, 8), Size::new(8, 8)),
    );
    // Crop outside the source.
    refused(
        packed,
        output,
        Geometry::new(source, Size::new(8, 8)).with_crop(Rect::new(8, 0, 9, 8)),
    );
    // Odd chroma origin for a YUV picture.
    let nv12 = DeviceBuffer::pitched(&setup.context, 16, 12).unwrap();
    let nv12_out = DeviceBuffer::pitched(&setup.context, 16, 12).unwrap();
    refused(
        Input::Nv12 {
            view: nv12.as_view(),
            chroma_row: 8,
            size: source,
        },
        Output {
            buffer: &nv12_out,
            format: OutputFormat::Nv12(YuvColour::default()),
            chroma_row: None,
        },
        Geometry::new(source, source).with_placement(Rect::new(1, 0, 14, 8)),
    );
    // No conversion from NV12 to RGB.
    refused(
        Input::Nv12 {
            view: nv12.as_view(),
            chroma_row: 8,
            size: source,
        },
        output,
        Geometry::new(source, Size::new(8, 8)),
    );
}

#[test]
fn pool_reuses_released_buffers() {
    let Some(setup) = setup() else { return };
    let mut pool = FramePool::new(
        &setup.context,
        OutputFormat::Nv12(YuvColour::default()),
        Size::new(1280, 720),
        2,
    );
    let first = pool.acquire().unwrap();
    assert!(first.width_bytes() >= 1280 && first.rows() == 1080);
    let second = pool.acquire().unwrap();
    assert!(!Arc::ptr_eq(&first, &second));
    assert!(matches!(
        pool.acquire(),
        Err(Error::FramesInFlight { capacity: 2 })
    ));
    let address = Arc::as_ptr(&first);
    drop(first);
    assert_eq!(Arc::as_ptr(&pool.acquire().unwrap()), address);
}

// ------------------------------------------------------ DMA-BUF input

#[cfg(target_os = "linux")]
mod dmabuf {
    use std::os::fd::AsFd as _;
    use std::time::Duration;

    use ferricast_nvcodec::interop::{
        BufferKey, DmaBuf, DmaBufImporter, DrmFormat, FenceWait, FrameView, Modifier,
    };

    use super::support::Gpu;
    use super::*;

    /// GBM buffers, tiled (CUDA array) and LINEAR (pitched), written by
    /// GLES / CUDA, imported and scaled into a 1280x720 NV12 and BGRA
    /// picture on the GPU.
    #[test]
    fn scales_gbm_dma_bufs() {
        let Some(gpu) = Gpu::get() else { return };
        let mut setup = setup().unwrap();
        let mut importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
        let (width, height) = (1920, 1080);
        let source = Size::new(width, height);
        let canvas = Size::new(1280, 720);
        let pixels = screen(width, height);
        for (key, modifier) in [Gpu::tiled_modifier(&importer), Modifier::LINEAR]
            .into_iter()
            .enumerate()
        {
            let bo = gpu
                .allocate(width, height, DrmFormat::Argb8888, modifier)
                .unwrap();
            gpu.write(&bo, &pixels);
            let key = BufferKey(key as u128 + 1);
            let dma_buf = DmaBuf {
                fd: bo.fd.as_fd(),
                width,
                height,
                format: bo.format,
                offset: bo.offset,
                stride: bo.stride,
                modifier: bo.modifier,
            };
            importer.import(key, &dma_buf).unwrap();
            let frame = importer
                .map(key, FenceWait::Implicit(Duration::from_secs(1)))
                .unwrap();
            let input = match frame.view() {
                FrameView::Array(array) => Input::Array {
                    array,
                    order: RgbOrder::Bgra,
                },
                FrameView::Pitched(view) => Input::Packed {
                    view,
                    order: RgbOrder::Bgra,
                    size: source,
                },
            };
            let geometry = Geometry::new(source, canvas);
            for format in [
                OutputFormat::Rgb(RgbOrder::Bgra),
                OutputFormat::Nv12(YuvColour::default()),
            ] {
                let (width_bytes, rows) = format.buffer_layout(canvas);
                let mut target = DeviceBuffer::pitched(&setup.context, width_bytes, rows).unwrap();
                target.fill_async(0, &setup.stream).unwrap();
                setup
                    .converter
                    .convert(
                        input,
                        Output {
                            buffer: &target,
                            format,
                            chroma_row: None,
                        },
                        &geometry,
                        &setup.stream,
                    )
                    .unwrap();
                setup.stream.synchronize().unwrap();
                let got = download(&target, width_bytes, rows);
                let scaled = rgb_reference(&pixels, source, &geometry, false);
                let want = match format {
                    OutputFormat::Rgb(_) => scaled,
                    _ => nv12_reference(&scaled, RgbOrder::Bgra, 1280, 720, YuvColour::default()),
                };
                assert_close(
                    &got,
                    &want,
                    50.0,
                    &format!("GBM {modifier:x?} -> {format:?}"),
                );
            }
        }
    }
}

#[cfg(target_os = "linux")]
#[path = "support/dmabuf.rs"]
mod support;
