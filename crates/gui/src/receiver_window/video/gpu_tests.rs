//! Tests of the GPU video path against a headless Vulkan Skia context,
//! the backend Freya prefers on Linux. They skip (pass) on machines
//! without a Vulkan device.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::similar_names,
    reason = "test fixtures are a few pixels wide; paired names mirror the planes they hold"
)]

use std::ffi::c_char;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::time::{Duration, Instant};

use ash::vk::{self, Handle};
use bytes::Bytes;
use ferricast::decoder::H264Decoder;
use ferricast::{
    Codec, ColourDescription, ColourPrimaries, ColourRange, DecoderConfig, EncodedFrame,
    MatrixCoefficients, PixelFormat, RawFrame, TransferCharacteristics, VideoDecoder, YuvColour,
    YuvMatrix,
};
use skia_safe::gpu::{self, Budgeted, DirectContext, vk as skvk};
use skia_safe::{AlphaType, ColorType, ImageInfo, Surface};

use super::{FrameSlot, Layout, VideoFrame, VideoRenderer};

/// A Vulkan device and the Skia context on it.
struct Gpu {
    /// Released in `drop` before the device it runs on.
    context: ManuallyDrop<DirectContext>,
    device: ash::Device,
    instance: ash::Instance,
    _entry: ash::Entry,
}

impl Gpu {
    fn new() -> Option<Self> {
        // SAFETY: loading the Vulkan loader runs its initialisation
        // code, which is what every Vulkan program does first.
        let entry = unsafe { ash::Entry::load() }.ok()?;
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
        let create = vk::InstanceCreateInfo::default().application_info(&app);
        // SAFETY: the create info is valid and outlives the call.
        let instance = unsafe { entry.create_instance(&create, None) }.ok()?;
        // SAFETY: `instance` is a live instance.
        let devices = unsafe { instance.enumerate_physical_devices() }.ok()?;
        let (physical, family) = devices.iter().find_map(|&device| {
            // SAFETY: `device` was enumerated from `instance`.
            unsafe { instance.get_physical_device_queue_family_properties(device) }
                .iter()
                .position(|family| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|family| (device, family))
        })?;
        let family_index = u32::try_from(family).ok()?;
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family_index)
            .queue_priorities(&priorities)];
        let create = vk::DeviceCreateInfo::default().queue_create_infos(&queues);
        // SAFETY: `physical` belongs to `instance`; the create info is valid.
        let device = unsafe { instance.create_device(physical, &create, None) }.ok()?;
        // SAFETY: the device was created with one queue of this family.
        let queue = unsafe { device.get_device_queue(family_index, 0) };

        let context = {
            let get_proc = |of: skvk::GetProcOf| {
                // SAFETY: Skia asks for entry points by NUL-terminated
                // name on handles it got from us.
                unsafe {
                    match of {
                        skvk::GetProcOf::Instance(handle, name) => {
                            entry.get_instance_proc_addr(vk::Instance::from_raw(handle as _), name)
                        }
                        skvk::GetProcOf::Device(handle, name) => {
                            (instance.fp_v1_0().get_device_proc_addr)(
                                vk::Device::from_raw(handle as _),
                                name.cast::<c_char>(),
                            )
                        }
                    }
                }
                .map_or(std::ptr::null(), |f| f as _)
            };
            let builder = skvk::BackendContext::new_builder(
                instance.handle().as_raw() as _,
                physical.as_raw() as _,
                device.handle().as_raw() as _,
                (queue.as_raw() as _, family),
                &get_proc,
                None,
            );
            // SAFETY: the handles are live and belong together, and the
            // queue family supports graphics.
            let backend = unsafe { builder.build() };
            gpu::direct_contexts::make_vulkan(&backend, None)?
        };
        Some(Self {
            context: ManuallyDrop::new(context),
            device,
            instance,
            _entry: entry,
        })
    }

    fn surface(&mut self, width: i32, height: i32) -> Surface {
        let info = ImageInfo::new(
            (width, height),
            ColorType::BGRA8888,
            AlphaType::Premul,
            None,
        );
        gpu::surfaces::render_target(
            &mut self.context,
            Budgeted::Yes,
            &info,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("a Vulkan render target")
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: `context` is not used again.
        unsafe { ManuallyDrop::drop(&mut self.context) };
        // SAFETY: Skia released the device above and nothing else uses
        // it; the device goes before the instance it came from.
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn gpu() -> Option<Gpu> {
    let gpu = Gpu::new();
    if gpu.is_none() {
        eprintln!("no Vulkan device; skipping");
    }
    gpu
}

/// An NV12 picture of vertical colour bars, one per (Y, Cb, Cr).
fn bars(width: usize, height: usize, bars: &[(u8, u8, u8)]) -> Vec<u8> {
    let bar = |x: usize| bars[x * bars.len() / width];
    let mut data = Vec::with_capacity(width * height * 3 / 2);
    for _ in 0..height {
        data.extend((0..width).map(|x| bar(x).0));
    }
    for _ in 0..height / 2 {
        data.extend((0..width / 2).flat_map(|x| {
            let (_, cb, cr) = bar(x * 2);
            [cb, cr]
        }));
    }
    data
}

#[test]
fn the_gpu_converts_nv12_like_the_cpu_for_every_colour_space() {
    let Some(mut gpu) = gpu() else { return };
    let (width, height) = (64, 16);
    let data = bars(
        width,
        height,
        &[
            (235, 128, 128),
            (16, 128, 128),
            (173, 42, 26),
            (63, 102, 240),
            (32, 240, 118),
            (128, 128, 128),
            (81, 90, 240),
            (145, 54, 34),
        ],
    );
    for matrix in [YuvMatrix::Bt601, YuvMatrix::Bt709, YuvMatrix::Bt2020] {
        for range in [ColourRange::Limited, ColourRange::Full] {
            let colour = YuvColour { matrix, range };
            let frame = VideoFrame {
                width: width as u32,
                height: height as u32,
                stride: width as u32,
                layout: Layout::Nv12(colour),
                data: Bytes::from(data.clone()),
                pts_us: 0,
            };
            let mut surface = gpu.surface(width as i32, height as i32);
            let mut slot = FrameSlot::default();
            slot.replace(frame);
            VideoRenderer::default().draw(surface.canvas(), &slot, width as f32, height as f32);
            gpu.context.flush_submit_and_sync_cpu().unwrap();
            let info = ImageInfo::new(
                (width as i32, height as i32),
                ColorType::BGRA8888,
                AlphaType::Premul,
                None,
            );
            let mut gpu_pixels = vec![0; width * height * 4];
            assert!(surface.read_pixels(&info, &mut gpu_pixels, width * 4, (0, 0)));
            let cpu_pixels = colour.nv12_to_bgra(&data, width, width, height).unwrap();
            // Compare the middle row of every bar: the bar edges are
            // filtered differently by the GPU's chroma upsampling.
            let row = height / 2 * width * 4;
            for bar in 0..8 {
                let x = (bar * width / 8 + width / 16) * 4;
                let at = row + x;
                let (g, c) = (&gpu_pixels[at..at + 4], &cpu_pixels[at..at + 4]);
                for (g, c) in g.iter().zip(c) {
                    assert!(
                        g.abs_diff(*c) <= 3,
                        "{colour:?} bar {bar}: gpu {g:?} cpu {c:?} ({:?} vs {:?})",
                        &gpu_pixels[at..at + 4],
                        &cpu_pixels[at..at + 4]
                    );
                }
            }
        }
    }
}

/// `nv12` widened to P010: every sample moved to the high byte.
fn widen(nv12: &[u8]) -> Vec<u8> {
    nv12.iter().flat_map(|&sample| [0, sample]).collect()
}

/// Draws `frame` into a fresh `width` × `height` surface and reads it back.
fn draw_and_read(gpu: &mut Gpu, frame: VideoFrame) -> Vec<u8> {
    let (width, height) = (frame.width as usize, frame.height as usize);
    let mut surface = gpu.surface(width as i32, height as i32);
    let mut slot = FrameSlot::default();
    slot.replace(frame);
    let mut renderer = VideoRenderer::default();
    renderer.draw(surface.canvas(), &slot, width as f32, height as f32);
    assert!(
        renderer.image.as_ref().unwrap().1.is_texture_backed(),
        "uploaded, not a raster image"
    );
    gpu.context.flush_submit_and_sync_cpu().unwrap();
    let info = ImageInfo::new(
        (width as i32, height as i32),
        ColorType::BGRA8888,
        AlphaType::Premul,
        None,
    );
    let mut pixels = vec![0; width * height * 4];
    assert!(surface.read_pixels(&info, &mut pixels, width * 4, (0, 0)));
    pixels
}

#[test]
fn the_gpu_draws_sdr_p010_like_its_top_eight_bits() {
    let Some(mut gpu) = gpu() else { return };
    let (width, height) = (64, 16);
    let data = bars(
        width,
        height,
        &[
            (235, 128, 128),
            (16, 128, 128),
            (173, 42, 26),
            (63, 102, 240),
        ],
    );
    let matrices = [
        MatrixCoefficients::Smpte170M,
        MatrixCoefficients::Bt709,
        MatrixCoefficients::Bt2020Ncl,
    ];
    for matrix in matrices {
        let colour = ColourDescription {
            matrix,
            ..ColourDescription::BT709_LIMITED
        };
        let frame = VideoFrame {
            width: width as u32,
            height: height as u32,
            stride: width as u32 * 2,
            layout: Layout::P010(colour),
            data: Bytes::from(widen(&data)),
            pts_us: 0,
        };
        let gpu_pixels = draw_and_read(&mut gpu, frame);
        let cpu_pixels = colour
            .yuv(height)
            .nv12_to_bgra(&data, width, width, height)
            .unwrap();
        let row = height / 2 * width * 4;
        for bar in 0..4 {
            let at = row + (bar * width / 4 + width / 8) * 4;
            let (g, c) = (&gpu_pixels[at..at + 4], &cpu_pixels[at..at + 4]);
            assert!(
                g.iter().zip(c).all(|(g, c)| g.abs_diff(*c) <= 4),
                "{matrix:?} bar {bar}: gpu {g:?} cpu {c:?}"
            );
        }
    }
}

#[test]
fn pq_p010_is_converted_to_srgb_rather_than_drawn_as_sdr() {
    let Some(mut gpu) = gpu() else { return };
    let (width, height) = (16, 4);
    // A BT.2020 limited-range mid grey, chroma neutral.
    let data = widen(&bars(width, height, &[(128, 128, 128)]));
    let frame = |colour: ColourDescription| VideoFrame {
        width: width as u32,
        height: height as u32,
        stride: width as u32 * 2,
        layout: Layout::P010(colour),
        data: Bytes::from(data.clone()),
        pts_us: 0,
    };
    let sdr = ColourDescription {
        matrix: MatrixCoefficients::Bt2020Ncl,
        ..ColourDescription::BT709_LIMITED
    };
    let pq = ColourDescription {
        primaries: ColourPrimaries::Bt2020,
        transfer: TransferCharacteristics::Pq,
        ..sdr
    };
    let as_sdr = draw_and_read(&mut gpu, frame(sdr));
    let as_pq = draw_and_read(&mut gpu, frame(pq));
    let centre = (height / 2 * width + width / 2) * 4;
    assert_ne!(
        &as_sdr[centre..centre + 3],
        &as_pq[centre..centre + 3],
        "the PQ transfer must change how the samples are shown"
    );
}

#[test]
fn a_frame_is_uploaded_once_however_often_it_is_drawn() {
    let Some(mut gpu) = gpu() else { return };
    let frame = VideoFrame {
        width: 4,
        height: 4,
        stride: 4,
        layout: Layout::Nv12(YuvColour::guess(4)),
        data: Bytes::from(bars(4, 4, &[(100, 128, 128)])),
        pts_us: 0,
    };
    let mut surface = gpu.surface(4, 4);
    let mut slot = FrameSlot::default();
    slot.replace(frame.clone());
    let mut renderer = VideoRenderer::default();
    renderer.draw(surface.canvas(), &slot, 4.0, 4.0);
    let image = &renderer.image.as_ref().unwrap().1;
    assert!(image.is_texture_backed(), "uploaded, not a raster image");
    let first = image.unique_id();
    renderer.draw(surface.canvas(), &slot, 4.0, 4.0);
    assert_eq!(renderer.image.as_ref().unwrap().1.unique_id(), first);
    slot.replace(frame);
    renderer.draw(surface.canvas(), &slot, 4.0, 4.0);
    assert_ne!(renderer.image.as_ref().unwrap().1.unique_id(), first);
}

/// Splits an Annex-B stream at its access unit delimiters.
fn access_units(stream: &[u8]) -> Vec<Bytes> {
    const AUD: [u8; 5] = [0, 0, 0, 1, 9];
    let starts: Vec<usize> = stream
        .windows(AUD.len())
        .enumerate()
        .filter_map(|(at, window)| (window == AUD).then_some(at))
        .chain([stream.len()])
        .collect();
    starts
        .windows(2)
        .map(|pair| Bytes::copy_from_slice(&stream[pair[0]..pair[1]]))
        .collect()
}

fn ffmpeg_1080p60(dir: &Path) -> Option<Vec<Bytes>> {
    let ffmpeg = std::env::var_os("FERRICAST_FFMPEG").unwrap_or_else(|| "ffmpeg".into());
    let out = dir.join("bars.h264");
    let status = std::process::Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi"])
        .args(["-i", "testsrc2=size=1920x1080:rate=60", "-t", "5"])
        .args([
            "-c:v", "libx264", "-preset", "veryfast", "-bf", "0", "-g", "60",
        ])
        .args(["-pix_fmt", "yuv420p", "-x264-params", "aud=1", "-f", "h264"])
        .arg(&out)
        .env_remove("LD_LIBRARY_PATH")
        .status()
        .ok()?;
    status
        .success()
        .then(|| access_units(&std::fs::read(&out).expect("ffmpeg output")))
}

#[derive(Default)]
struct Samples(Vec<Duration>);

impl Samples {
    fn report(&mut self, name: &str) -> String {
        self.0.sort();
        let mean = self.0.iter().sum::<Duration>() / self.0.len().max(1) as u32;
        let p95 = self.0[self.0.len() * 95 / 100];
        format!("{name}: mean {mean:.2?} p95 {p95:.2?}")
    }
}

/// Decodes a 1080p60 stream and draws every frame into a 1280×720
/// window-sized Vulkan surface, the way a receiver window does, with
/// the decoders asked for `format`. Prints the per-frame cost of each
/// stage. `cargo test -p ferricast-gui --release -- --ignored
/// --nocapture receiver_video_cost`
fn receiver_video_cost(format: PixelFormat) {
    let Some(mut gpu) = gpu() else { return };
    let dir = tempfile::tempdir().unwrap();
    let Some(units) = ffmpeg_1080p60(dir.path()) else {
        eprintln!("ffmpeg with libx264 not found (set FERRICAST_FFMPEG); skipping");
        return;
    };
    let mut decoder = H264Decoder::new();
    decoder
        .configure(&DecoderConfig {
            gpu: Default::default(),
            hdr: None,
            memory: ferricast::FrameMemory::Cpu,
            codec: Codec::H264,
            width: 1920,
            height: 1080,
            pixel_format: format,
            low_latency: true,
            colour: None,
            backend: ferricast::BackendSelection::default(),
        })
        .unwrap();
    let mut surface = gpu.surface(1280, 720);
    let mut renderer = VideoRenderer::default();
    let mut slot = FrameSlot::default();
    let (mut decode, mut draw) = (Samples::default(), Samples::default());
    for (index, unit) in units.into_iter().enumerate() {
        let timestamp_us = index as u64 * 16_667;
        let started = Instant::now();
        let decoded = decoder
            .decode(EncodedFrame {
                codec: Codec::H264,
                data: unit,
                timestamp_us,
                is_keyframe: index % 60 == 0,
                duration_us: Some(16_667),
                pts_dts: (timestamp_us, timestamp_us),
            })
            .unwrap();
        let Some(decoded) = decoded else { continue };
        let raw: RawFrame = decoded.into_cpu().unwrap();
        decode.0.push(started.elapsed());

        let started = Instant::now();
        slot.replace(VideoFrame::try_from(raw).unwrap());
        renderer.draw(surface.canvas(), &slot, 1280.0, 720.0);
        gpu.context.flush_submit_and_sync_cpu().unwrap();
        draw.0.push(started.elapsed());
    }
    eprintln!(
        "{format:?} ({} frames, {} build): {}; {}",
        decode.0.len(),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        decode.report("decode"),
        draw.report("upload+convert+draw")
    );
}

#[test]
#[ignore = "benchmark: needs ffmpeg with libx264 and a Vulkan device"]
fn receiver_video_cost_bgra() {
    receiver_video_cost(PixelFormat::Bgra);
}

#[test]
#[ignore = "benchmark: needs ffmpeg with libx264 and a Vulkan device"]
fn receiver_video_cost_nv12() {
    receiver_video_cost(PixelFormat::Nv12);
}
