//! NVENC encoder tests against the installed driver. Each test prints why and
//! returns early without an NVIDIA driver.
//!
//! Encoded streams are written under `CARGO_TARGET_TMPDIR/nvenc` and checked
//! with `ffprobe` / `ffmpeg` from `FERRICAST_FFMPEG_DIR` (or `PATH`); without
//! them the stream checks are skipped, with a message.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "synthetic pictures and PSNR maths on small, known ranges; pixel formulas read best with short names"
)]

use std::collections::BTreeSet;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use ferricast_nvcodec::Error;
use ferricast_nvcodec::cuda::{
    ArrayView, Context, Device, DeviceBuffer, HostImage, PitchedView, Point, Region,
};
use ferricast_nvcodec::encode::{
    ApiFeature, Capabilities, Codec, CodecSupport, ColorDescription, Config, ContentLightLevel,
    DeviceFrame, Encoder, FrameOptions, FrameRate, Gop, Hdr10, HostFrame, InputFormat,
    MasteringDisplay, Packet, PictureType, Profile, QpSet, RateControl, Tuning,
};
use ferricast_nvcodec::loader::{CudaLibrary, NvencLibrary};
use ferricast_nvcodec::sys::cuda as sys;
use ferricast_nvcodec::version::{API_OVERRIDE_VAR, AnyApi, ApiVersion, NvencVersion};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const FRAMES: usize = 32;
const IDR_PERIOD: u32 = 15;
const FORCED_IDR: usize = 20;

fn context() -> Option<Arc<Context>> {
    let opened = CudaLibrary::get()
        .and_then(|_| Device::get(0))
        .and_then(Context::primary);
    match opened {
        Ok(context) => Some(context),
        Err(error) => {
            eprintln!("skipping: no usable NVIDIA CUDA driver: {error}");
            None
        }
    }
}

/// Serialises the tests of this binary: several at once on one GPU (one of
/// them filling the session limit) make the driver refuse sessions with
/// errors that are not the test's subject. Children run while their parent
/// holds it.
fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn driver_max() -> Option<NvencVersion> {
    match NvencLibrary::get().and_then(NvencLibrary::max_supported_version) {
        Ok(version) => Some(version),
        Err(error) => {
            eprintln!("skipping: no usable NVENC driver: {error}");
            None
        }
    }
}

fn rate(fps: u32) -> FrameRate {
    FrameRate::per_second(NonZeroU32::new(fps).unwrap())
}

// ---------------------------------------------------------------------------
// Synthetic video

/// One 4:2:0 picture, planar, samples at the video's bit depth.
#[derive(Clone)]
struct Planes {
    y: Vec<u16>,
    u: Vec<u16>,
    v: Vec<u16>,
}

/// Moving sine patterns and a moving square: smooth enough to compress,
/// different in every picture.
fn video(width: u32, height: u32, frames: usize, bits: u32) -> Vec<Planes> {
    let (w, h) = (width as usize, height as usize);
    let scale = f64::from(1u32 << (bits - 8));
    let quantise = |value: f64| (value * scale).round().clamp(0.0, 255.0 * scale) as u16;
    (0..frames)
        .map(|t| {
            let t = t as f64;
            let mut y = Vec::with_capacity(w * h);
            for row in 0..h {
                for col in 0..w {
                    let (x, r) = (col as f64, row as f64);
                    let square =
                        (x - 4.0 * t - 100.0).abs() < 40.0 && (r - 2.0 * t - 80.0).abs() < 40.0;
                    let wave = ((x + 3.0 * t) / 23.0).sin() * ((r - 2.0 * t) / 17.0).cos();
                    y.push(quantise(if square { 220.0 } else { 120.0 + 70.0 * wave }));
                }
            }
            let (cw, ch) = (w / 2, h / 2);
            let mut u = Vec::with_capacity(cw * ch);
            let mut v = Vec::with_capacity(cw * ch);
            for row in 0..ch {
                for col in 0..cw {
                    let (x, r) = (col as f64, row as f64);
                    u.push(quantise(128.0 + 40.0 * ((x + r + t) / 21.0).sin()));
                    v.push(quantise(128.0 + 40.0 * ((x - r) / 19.0 + t / 9.0).cos()));
                }
            }
            Planes { y, u, v }
        })
        .collect()
}

/// `planes` laid out as `format`, rows tightly packed.
fn pack(planes: &Planes, format: InputFormat, width: u32, height: u32) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    match format {
        InputFormat::Nv12 => {
            let mut out: Vec<u8> = planes.y.iter().map(|&s| s as u8).collect();
            for (u, v) in planes.u.iter().zip(&planes.v) {
                out.extend([*u as u8, *v as u8]);
            }
            out
        }
        InputFormat::P010 => {
            let mut out = Vec::with_capacity(w * h * 3);
            for &sample in &planes.y {
                out.extend((sample << 6).to_le_bytes());
            }
            for (u, v) in planes.u.iter().zip(&planes.v) {
                out.extend((u << 6).to_le_bytes());
                out.extend((v << 6).to_le_bytes());
            }
            out
        }
        InputFormat::Argb | InputFormat::Abgr => {
            // BT.709 limited range YUV -> full range RGB.
            let mut out = Vec::with_capacity(w * h * 4);
            for row in 0..h {
                for col in 0..w {
                    let y = f64::from(planes.y[row * w + col]) - 16.0;
                    let chroma = (row / 2) * (w / 2) + col / 2;
                    let u = f64::from(planes.u[chroma]) - 128.0;
                    let v = f64::from(planes.v[chroma]) - 128.0;
                    let r = 1.164 * y + 1.793 * v;
                    let g = 1.164 * y - 0.213 * u - 0.533 * v;
                    let b = 1.164 * y + 2.112 * u;
                    let [r, g, b] = [r, g, b].map(|c| c.round().clamp(0.0, 255.0) as u8);
                    if format == InputFormat::Argb {
                        out.extend([b, g, r, 255]);
                    } else {
                        out.extend([r, g, b, 255]);
                    }
                }
            }
            out
        }
    }
}

/// PSNR of decoded planar 4:2:0 (`bits` per sample, little-endian above 8)
/// against the source pictures.
fn psnr(decoded: &[u8], source: &[Planes], bits: u32) -> f64 {
    let samples: Vec<u16> = if bits > 8 {
        decoded
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect()
    } else {
        decoded.iter().map(|&byte| u16::from(byte)).collect()
    };
    let expected: Vec<u16> = source
        .iter()
        .flat_map(|planes| planes.y.iter().chain(&planes.u).chain(&planes.v).copied())
        .collect();
    assert_eq!(samples.len(), expected.len(), "decoded sample count");
    let mse = samples
        .iter()
        .zip(&expected)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    let peak = f64::from((1u32 << bits) - 1);
    10.0 * (peak * peak / mse.max(1e-9)).log10()
}

// ---------------------------------------------------------------------------
// Encoding helpers

/// Encodes host pictures, forcing IDRs at `forced`, and returns every
/// packet after a flush.
fn encode_host(encoder: &mut Encoder, pictures: &[Vec<u8>], forced: &[usize]) -> Vec<Packet> {
    let config = encoder.config().clone();
    let mut packets = Vec::new();
    for (index, picture) in pictures.iter().enumerate() {
        encoder
            .encode_host(
                HostFrame::tight(picture, config.input_format, config.width, config.height),
                FrameOptions {
                    timestamp: index as u64 * 1000,
                    force_idr: forced.contains(&index),
                },
            )
            .unwrap();
        packets.extend(std::iter::from_fn(|| encoder.next_packet()));
    }
    encoder.flush().unwrap();
    packets.extend(std::iter::from_fn(|| encoder.next_packet()));
    packets
}

/// Device buffers holding `pictures` (tightly packed host layout), chroma at
/// `chroma_row` (`None`: right after the luma).
fn device_pictures(
    context: &Arc<Context>,
    pictures: &[Vec<u8>],
    format: InputFormat,
    width: u32,
    height: u32,
    chroma_row: Option<usize>,
) -> Vec<Arc<DeviceBuffer>> {
    let row_bytes = format.row_bytes(width);
    let luma_rows = height as usize;
    let chroma_rows = format.chroma_rows(height);
    let chroma_at = chroma_row.unwrap_or(luma_rows);
    pictures
        .iter()
        .map(|picture| {
            let rows = if chroma_rows > 0 {
                chroma_at + chroma_rows
            } else {
                luma_rows
            };
            let mut buffer = DeviceBuffer::pitched(context, row_bytes, rows).unwrap();
            buffer.fill(0x40).unwrap();
            let (luma, chroma) = picture.split_at(row_bytes * luma_rows);
            buffer
                .upload(
                    Region::new(row_bytes, luma_rows),
                    HostImage {
                        data: luma,
                        pitch: row_bytes,
                    },
                )
                .unwrap();
            if chroma_rows > 0 {
                buffer
                    .upload(
                        Region::new(row_bytes, chroma_rows).at(Point {
                            x_bytes: 0,
                            row: chroma_at,
                        }),
                        HostImage {
                            data: chroma,
                            pitch: row_bytes,
                        },
                    )
                    .unwrap();
            }
            Arc::new(buffer)
        })
        .collect()
}

fn encode_device(
    encoder: &mut Encoder,
    buffers: &[Arc<DeviceBuffer>],
    chroma_row: Option<usize>,
    forced: &[usize],
) -> ferricast_nvcodec::Result<Vec<Packet>> {
    let mut packets = Vec::new();
    for (index, buffer) in buffers.iter().enumerate() {
        encoder.encode_device(
            DeviceFrame {
                buffer,
                chroma_row,
                ready: None,
            },
            FrameOptions {
                timestamp: index as u64 * 1000,
                force_idr: forced.contains(&index),
            },
        )?;
        packets.extend(std::iter::from_fn(|| encoder.next_packet()));
    }
    encoder.flush()?;
    packets.extend(std::iter::from_fn(|| encoder.next_packet()));
    Ok(packets)
}

fn stream(packets: &[Packet]) -> Vec<u8> {
    packets
        .iter()
        .flat_map(|packet| packet.data.iter().copied())
        .collect()
}

fn keyframes(packets: &[Packet]) -> Vec<usize> {
    packets
        .iter()
        .enumerate()
        .filter(|(_, packet)| packet.is_keyframe())
        .map(|(index, _)| index)
        .collect()
}

/// NAL unit types of an Annex B access unit.
fn nal_types(codec: Codec, data: &[u8]) -> Vec<u8> {
    let mut types = Vec::new();
    let mut index = 0;
    while index + 3 < data.len() {
        if data[index..index + 3] == [0, 0, 1] {
            let header = data[index + 3];
            types.push(match codec {
                Codec::H264 => header & 0x1f,
                Codec::Hevc => (header >> 1) & 0x3f,
                Codec::Av1 => unreachable!("AV1 is not Annex B"),
            });
            index += 3;
        } else {
            index += 1;
        }
    }
    types
}

/// Whether an IDR access unit carries its parameter sets.
fn has_parameter_sets(codec: Codec, data: &[u8]) -> bool {
    let types = nal_types(codec, data);
    match codec {
        Codec::H264 => types.contains(&7) && types.contains(&8),
        Codec::Hevc => types.contains(&32) && types.contains(&33) && types.contains(&34),
        Codec::Av1 => false,
    }
}

fn output_dir(version: ApiVersion) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("nvenc")
        .join(version.to_string());
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ---------------------------------------------------------------------------
// ffprobe / ffmpeg

fn tool(name: &str) -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("FERRICAST_FFMPEG_DIR") {
        let path = Path::new(&dir).join(name);
        return path.exists().then_some(path);
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join(name))
            .find(|path| path.exists())
    })
}

fn tools() -> Option<(PathBuf, PathBuf)> {
    let found = tool("ffprobe").zip(tool("ffmpeg"));
    if found.is_none() {
        eprintln!("ffprobe / ffmpeg not found (set FERRICAST_FFMPEG_DIR): stream checks skipped");
    }
    found
}

#[derive(Debug, Default)]
struct Probe {
    codec_name: String,
    profile: String,
    pix_fmt: String,
    color_transfer: String,
    color_primaries: String,
    color_space: String,
    frames: usize,
    keyframes: Vec<usize>,
    /// Frames carrying each side data type.
    side_data: Vec<(usize, String)>,
}

fn demuxer(codec: Codec) -> &'static str {
    match codec {
        Codec::H264 => "h264",
        Codec::Hevc => "hevc",
        Codec::Av1 => "obu",
    }
}

fn probe(ffprobe: &Path, path: &Path, codec: Codec) -> Probe {
    let output = Command::new(ffprobe)
        .env_remove("LD_LIBRARY_PATH")
        .args([
            "-v",
            "error",
            "-f",
            demuxer(codec),
            "-show_streams",
            "-show_frames",
            "-of",
            "flat",
        ])
        .arg(path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let mut probe = Probe::default();
    let mut frames = BTreeSet::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.trim_matches('"').to_owned();
        if let Some(field) = key.strip_prefix("streams.stream.0.") {
            match field {
                "codec_name" => probe.codec_name = value,
                "profile" => probe.profile = value,
                "pix_fmt" => probe.pix_fmt = value,
                "color_transfer" => probe.color_transfer = value,
                "color_primaries" => probe.color_primaries = value,
                "color_space" => probe.color_space = value,
                _ => {}
            }
        } else if let Some(rest) = key.strip_prefix("frames.frame.") {
            let Some((index, field)) = rest.split_once('.') else {
                continue;
            };
            let index: usize = index.parse().unwrap();
            frames.insert(index);
            if field == "key_frame" && value == "1" {
                probe.keyframes.push(index);
            } else if field.starts_with("side_data_list.") && field.ends_with(".side_data_type") {
                probe.side_data.push((index, value));
            }
        }
    }
    probe.frames = frames.len();
    probe
}

fn decode(ffmpeg: &Path, path: &Path, codec: Codec, pix_fmt: &str) -> Vec<u8> {
    let output = Command::new(ffmpeg)
        .env_remove("LD_LIBRARY_PATH")
        .args(["-v", "error", "-f", demuxer(codec), "-i"])
        .arg(path)
        .args(["-f", "rawvideo", "-pix_fmt", pix_fmt, "-"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

// ---------------------------------------------------------------------------
// The per-version matrix (run in a child per version)

struct Case {
    name: &'static str,
    codec: Codec,
    format: InputFormat,
    hdr: bool,
}

const CASES: [Case; 5] = [
    Case {
        name: "h264-nv12",
        codec: Codec::H264,
        format: InputFormat::Nv12,
        hdr: false,
    },
    Case {
        name: "h264-argb",
        codec: Codec::H264,
        format: InputFormat::Argb,
        hdr: false,
    },
    Case {
        name: "hevc-nv12",
        codec: Codec::Hevc,
        format: InputFormat::Nv12,
        hdr: false,
    },
    Case {
        name: "hevc-abgr",
        codec: Codec::Hevc,
        format: InputFormat::Abgr,
        hdr: false,
    },
    Case {
        name: "hevc-p010-hdr10",
        codec: Codec::Hevc,
        format: InputFormat::P010,
        hdr: true,
    },
];

const HDR10: Hdr10 = Hdr10 {
    mastering_display: Some(MasteringDisplay::BT2020_1000_NITS),
    content_light: Some(ContentLightLevel {
        max_content: 1000,
        max_frame_average: 400,
    }),
};

fn case_config(case: &Case) -> Config {
    Config {
        gop: Gop::idr_every(NonZeroU32::new(IDR_PERIOD).unwrap()),
        color: Some(if case.hdr {
            ColorDescription::HDR10
        } else {
            ColorDescription::BT709
        }),
        hdr10: case.hdr.then_some(HDR10),
        ..Config::low_latency(case.codec, WIDTH, HEIGHT, case.format, rate(30), 6_000_000)
    }
}

/// Encodes every case through the negotiated (forced) version and checks
/// the streams.
fn run_matrix(context: &Arc<Context>, version: ApiVersion) {
    let dir = output_dir(version);
    let tools = tools();
    for case in &CASES {
        let config = case_config(case);
        let mut encoder = Encoder::open(context, &config).unwrap();
        assert_eq!(encoder.api_version(), version);
        let source = video(WIDTH, HEIGHT, FRAMES, case.format.bit_depth());
        let pictures: Vec<Vec<u8>> = source
            .iter()
            .map(|planes| pack(planes, case.format, WIDTH, HEIGHT))
            .collect();
        let packets = encode_host(&mut encoder, &pictures, &[FORCED_IDR]);
        assert_eq!(packets.len(), FRAMES, "{}", case.name);
        let keys = keyframes(&packets);
        eprintln!(
            "API {version} {}: {} bytes, IDR at {keys:?}",
            case.name,
            stream(&packets).len()
        );
        assert!(
            keys.starts_with(&[0])
                && keys.contains(&(IDR_PERIOD as usize))
                && keys.contains(&FORCED_IDR),
            "{keys:?}"
        );
        for index in &keys {
            assert!(
                has_parameter_sets(case.codec, &packets[*index].data),
                "{} IDR {index} lacks parameter sets",
                case.name
            );
        }
        for (index, packet) in packets.iter().enumerate() {
            assert_eq!(packet.index, index as u64);
            assert_eq!(packet.timestamp, index as u64 * 1000);
        }
        let path = dir.join(format!("{}.{}", case.name, demuxer(case.codec)));
        std::fs::write(&path, stream(&packets)).unwrap();
        let Some((ffprobe, ffmpeg)) = &tools else {
            continue;
        };

        let probe = probe(ffprobe, &path, case.codec);
        eprintln!("  {probe:?}");
        assert_eq!(probe.frames, FRAMES, "{}", case.name);
        assert_eq!(probe.keyframes, keys, "{}", case.name);
        let (codec_name, profile, pix_fmt) = match (case.codec, case.format) {
            (Codec::H264, _) => ("h264", "High", "yuv420p"),
            (Codec::Hevc, InputFormat::P010) => ("hevc", "Main 10", "yuv420p10le"),
            (Codec::Hevc, _) => ("hevc", "Main", "yuv420p"),
            (Codec::Av1, _) => unreachable!(),
        };
        assert_eq!(probe.codec_name, codec_name);
        assert_eq!(probe.profile, profile);
        assert_eq!(probe.pix_fmt, pix_fmt);
        if case.hdr {
            assert_eq!(probe.color_transfer, "smpte2084");
            assert_eq!(probe.color_primaries, "bt2020");
            assert_eq!(probe.color_space, "bt2020nc");
            for key in &keys {
                for kind in ["Mastering display metadata", "Content light level metadata"] {
                    assert!(
                        probe
                            .side_data
                            .iter()
                            .any(|(index, seen)| index == key && seen == kind),
                        "{} IDR {key} lacks {kind}: {:?}",
                        case.name,
                        probe.side_data
                    );
                }
            }
        } else {
            assert_eq!(probe.color_transfer, "bt709");
            assert_eq!(probe.color_primaries, "bt709");
            assert_eq!(probe.color_space, "bt709");
        }
        let decoded = decode(ffmpeg, &path, case.codec, pix_fmt);
        let quality = psnr(&decoded, &source, case.format.bit_depth());
        eprintln!("  PSNR {quality:.2} dB");
        assert!(quality > 35.0, "{}: PSNR {quality:.2} dB", case.name);
    }
}

/// Host and device input give the same bitstream; the device registration
/// happens once per buffer.
fn host_and_device_agree(context: &Arc<Context>, version: ApiVersion) {
    let format = InputFormat::Nv12;
    let config = case_config(&CASES[2]);
    let source = video(WIDTH, HEIGHT, 8, 8);
    let pictures: Vec<Vec<u8>> = source
        .iter()
        .map(|planes| pack(planes, format, WIDTH, HEIGHT))
        .collect();
    let mut host = Encoder::open(context, &config).unwrap();
    let from_host = encode_host(&mut host, &pictures, &[5]);
    let buffers = device_pictures(context, &pictures, format, WIDTH, HEIGHT, None);
    let mut device = Encoder::open(context, &config).unwrap();
    let from_device = encode_device(&mut device, &buffers, None, &[5]).unwrap();
    assert_eq!(
        stream(&from_host),
        stream(&from_device),
        "API {version}: host and device bitstreams differ"
    );
    assert_eq!(device.registrations(), buffers.len() as u64);
    // The same buffers again: no new registration.
    encode_device(&mut device, &buffers, None, &[]).unwrap();
    assert_eq!(device.registrations(), buffers.len() as u64);
    eprintln!(
        "API {version}: host and device input give identical {} bytes",
        stream(&from_host).len()
    );
}

/// A device frame whose chroma plane does not follow the luma rows needs
/// `chromaOffsetIn` (13.0): encoded there, refused before.
fn padded_chroma(context: &Arc<Context>, version: ApiVersion) {
    let format = InputFormat::Nv12;
    let config = case_config(&CASES[2]);
    let source = video(WIDTH, HEIGHT, 4, 8);
    let pictures: Vec<Vec<u8>> = source
        .iter()
        .map(|planes| pack(planes, format, WIDTH, HEIGHT))
        .collect();
    let chroma_row = Some(HEIGHT as usize + 24);
    let buffers = device_pictures(context, &pictures, format, WIDTH, HEIGHT, chroma_row);
    let mut encoder = Encoder::open(context, &config).unwrap();
    let encoded = encode_device(&mut encoder, &buffers, chroma_row, &[]);
    if ApiFeature::ChromaPlaneOffset.available_in(version) {
        let packets = encoded.unwrap();
        let mut reference = Encoder::open(context, &config).unwrap();
        let expected = encode_host(&mut reference, &pictures, &[]);
        assert_eq!(
            stream(&packets),
            stream(&expected),
            "API {version}: padded chroma differs"
        );
        eprintln!("API {version}: padded chroma plane encoded like the contiguous one");
    } else {
        let error = encoded.unwrap_err();
        eprintln!("API {version}: padded chroma refused: {error}");
        assert!(matches!(error, Error::Unsupported { .. }), "{error}");
        assert!(error.to_string().contains("NVENC API 13.0"), "{error}");
    }
}

/// Features a version lacks are refused, naming the version that adds them.
fn version_gates(context: &Arc<Context>, version: ApiVersion) {
    let uhq = Config {
        tuning: Tuning::UltraHighQuality,
        preset: ferricast_nvcodec::encode::Preset::P7,
        ..case_config(&CASES[2])
    };
    match Encoder::open(context, &uhq) {
        Ok(encoder) => {
            assert!(ApiFeature::UltraHighQualityTuning.available_in(version));
            eprintln!(
                "API {version}: ultra-high-quality tuning opens ({:?})",
                encoder.api_version()
            );
        }
        Err(error) => {
            eprintln!("API {version}: ultra-high-quality tuning: {error}");
            if ApiFeature::UltraHighQualityTuning.available_in(version) {
                // The GPU may still refuse it (it is Turing+ / HEVC & AV1).
                assert!(matches!(error, Error::Nvenc { .. }), "{error}");
            } else {
                assert!(matches!(error, Error::Unsupported { .. }), "{error}");
                assert!(error.to_string().contains("NVENC API 13.0"), "{error}");
            }
        }
    }

    let av1 = Config::low_latency(
        Codec::Av1,
        WIDTH,
        HEIGHT,
        InputFormat::Nv12,
        rate(30),
        4_000_000,
    );
    let error = Encoder::open(context, &av1).unwrap_err();
    eprintln!("API {version}: AV1: {error}");
    assert!(matches!(error, Error::Unsupported { .. }), "{error}");
    let caps = Capabilities::query(context).unwrap();
    assert_eq!(caps.api, version);
    if ApiFeature::Av1.available_in(version) {
        // An RTX 3060 (Ampere) has no AV1 encoder; newer GPUs would.
        if caps.av1 == CodecSupport::MissingFromGpu {
            assert!(error.to_string().contains("no AV1 encoder"), "{error}");
        }
    } else {
        assert_eq!(
            caps.av1,
            CodecSupport::MissingFromApi {
                needs: ApiVersion::V12_0
            }
        );
        assert!(error.to_string().contains("NVENC API 12.0"), "{error}");
    }
    assert_eq!(
        caps.features,
        ApiFeature::ALL
            .into_iter()
            .filter(|feature| feature.available_in(version))
            .collect::<Vec<_>>()
    );
}

/// Runs in a child process with `FERRICAST_NVENC_API` set; passes trivially
/// otherwise.
#[test]
fn child_encode_matrix() {
    let _gpu = gpu_lock();
    let Some(value) = std::env::var_os(API_OVERRIDE_VAR) else {
        return;
    };
    let Some(context) = context() else { return };
    let version = ApiVersion::negotiate().unwrap();
    assert_eq!(version.to_string(), value.to_string_lossy());
    run_matrix(&context, version);
    host_and_device_agree(&context, version);
    padded_chroma(&context, version);
    version_gates(&context, version);
    eprintln!("API {version}: matrix passed");
}

#[test]
fn encode_matrix_every_api_version() {
    let _gpu = gpu_lock();
    let Some(driver_max) = driver_max() else {
        return;
    };
    if context().is_none() {
        return;
    }
    for version in ApiVersion::ALL
        .into_iter()
        .filter(|version| version.version() <= driver_max)
    {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "child_encode_matrix",
                "--nocapture",
                "--test-threads",
                "1",
            ])
            .env(API_OVERRIDE_VAR, version.to_string())
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("{}", stderr.trim());
        assert!(
            output.status.success(),
            "{API_OVERRIDE_VAR}={version} failed"
        );
        assert!(
            stderr.contains(&format!("API {version}: matrix passed")),
            "the child did not run for {version}"
        );
    }
}

// ---------------------------------------------------------------------------
// Negotiated version only

#[test]
fn capabilities_of_this_gpu() {
    let _gpu = gpu_lock();
    let Some(context) = context() else { return };
    let caps = Capabilities::query(&context).unwrap();
    eprintln!("{caps:#?}");
    let h264 = caps.require(Codec::H264).unwrap();
    assert!(h264.max_width >= 4096 && h264.input_formats.contains(&InputFormat::Nv12));
    assert!(h264.input_formats.contains(&InputFormat::Argb));
    let hevc = caps.require(Codec::Hevc).unwrap();
    assert!(hevc.ten_bit && hevc.input_formats.contains(&InputFormat::P010));
    if let CodecSupport::Available(av1) = &caps.av1 {
        eprintln!("this GPU encodes AV1: {av1:?}");
    } else {
        let error = caps.require(Codec::Av1).unwrap_err();
        eprintln!("AV1: {error}");
        assert!(matches!(error, Error::Unsupported { .. }));
    }
}

#[test]
fn forced_idr_and_parameter_sets() {
    let _gpu = gpu_lock();
    let Some(context) = context() else { return };
    for codec in [Codec::H264, Codec::Hevc] {
        let config = Config {
            gop: Gop::idr_every(NonZeroU32::new(10).unwrap()),
            ..Config::low_latency(codec, WIDTH, HEIGHT, InputFormat::Nv12, rate(30), 4_000_000)
        };
        let mut encoder = Encoder::open(&context, &config).unwrap();
        let source = video(WIDTH, HEIGHT, 30, 8);
        let pictures: Vec<Vec<u8>> = source
            .iter()
            .map(|planes| pack(planes, InputFormat::Nv12, WIDTH, HEIGHT))
            .collect();
        let mut packets = Vec::new();
        for (index, picture) in pictures.iter().enumerate() {
            if index == 13 {
                encoder.request_idr();
            }
            encoder
                .encode_host(
                    HostFrame::tight(picture, InputFormat::Nv12, WIDTH, HEIGHT),
                    FrameOptions {
                        timestamp: index as u64,
                        force_idr: index == 4,
                    },
                )
                .unwrap();
            // Lowest latency: the packet comes out of the call that
            // encoded it.
            let packet = encoder.next_packet().expect("a packet per picture");
            assert!(encoder.next_packet().is_none());
            packets.push(packet);
        }
        let keys = keyframes(&packets);
        eprintln!("{codec}: IDR at {keys:?}");
        // A forced IDR restarts NVENC's IDR period: 4 + 10 would be 14,
        // the request at 13 comes first, then 13 + 10.
        assert_eq!(keys, [0, 4, 13, 23]);
        for (index, packet) in packets.iter().enumerate() {
            assert_eq!(
                packet.is_keyframe(),
                has_parameter_sets(codec, &packet.data),
                "picture {index}"
            );
            if !packet.is_keyframe() {
                assert!(
                    matches!(packet.picture_type, PictureType::P),
                    "{:?}",
                    packet.picture_type
                );
            }
        }
        let headers = encoder.sequence_params().unwrap();
        assert!(has_parameter_sets(codec, &headers), "{headers:02x?}");
        assert!(headers.starts_with(&[0, 0, 0, 1]) || headers.starts_with(&[0, 0, 1]));
    }
}

#[test]
fn bitrate_reconfigure_takes_effect() {
    let _gpu = gpu_lock();
    let Some(context) = context() else { return };
    let fps = 30;
    let (low, high) = (1_000_000, 4_000_000);
    let config = Config {
        rate_control: RateControl::Cbr {
            bitrate: low,
            vbv: None,
        },
        ..Config::low_latency(
            Codec::H264,
            WIDTH,
            HEIGHT,
            InputFormat::Nv12,
            rate(fps),
            low,
        )
    };
    let mut encoder = Encoder::open(&context, &config).unwrap();
    // The moving pattern with grain on top: hard enough that CBR spends
    // what it may (pure noise overshoots any bitrate even at QP 51).
    let base: Vec<Vec<u8>> = video(WIDTH, HEIGHT, 16, 8)
        .iter()
        .map(|planes| pack(planes, InputFormat::Nv12, WIDTH, HEIGHT))
        .collect();
    let mut seed = 0x1234_5678_u32;
    let mut frame = 0;
    let mut noise = || {
        frame += 1;
        base[frame % base.len()]
            .iter()
            .map(|&sample| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                sample.saturating_add((seed >> 29) as u8)
            })
            .collect::<Vec<u8>>()
    };
    let mut second = |encoder: &mut Encoder| -> usize {
        (0..2 * fps)
            .map(|_| {
                let picture = noise();
                encoder
                    .encode_host(
                        HostFrame::tight(&picture, InputFormat::Nv12, WIDTH, HEIGHT),
                        FrameOptions::default(),
                    )
                    .unwrap();
                let packet = encoder.next_packet().unwrap();
                let size = packet.data.len();
                encoder.recycle(packet);
                size
            })
            .sum()
    };
    let low_bytes = second(&mut encoder);
    encoder
        .set_rate_control(RateControl::Cbr {
            bitrate: high,
            vbv: None,
        })
        .unwrap();
    assert_eq!(encoder.config().rate_control.bitrates(), Some((high, high)));
    let high_bytes = second(&mut encoder);
    let (low_rate, high_rate) = (low_bytes as f64 * 8.0 / 2.0, high_bytes as f64 * 8.0 / 2.0);
    eprintln!("CBR {low} b/s -> {low_rate:.0} b/s, {high} b/s -> {high_rate:.0} b/s");
    assert!((low_rate / f64::from(low) - 1.0).abs() < 0.3, "{low_rate}");
    assert!(
        (high_rate / f64::from(high) - 1.0).abs() < 0.3,
        "{high_rate}"
    );
}

#[test]
fn frames_in_flight_b_frames_and_lookahead_keep_order() {
    let _gpu = gpu_lock();
    let Some(context) = context() else { return };
    let caps = Capabilities::query(&context).unwrap();
    let h264 = caps.require(Codec::H264).unwrap();
    let source = video(WIDTH, HEIGHT, 40, 8);
    let pictures: Vec<Vec<u8>> = source
        .iter()
        .map(|planes| pack(planes, InputFormat::Nv12, WIDTH, HEIGHT))
        .collect();
    let base = Config {
        gop: Gop::idr_every(NonZeroU32::new(30).unwrap()),
        ..Config::low_latency(
            Codec::H264,
            WIDTH,
            HEIGHT,
            InputFormat::Nv12,
            rate(30),
            4_000_000,
        )
    };
    let variants = [
        Config {
            frames_in_flight: 3,
            ..base.clone()
        },
        Config {
            b_frames: 2.min(u8::try_from(h264.max_b_frames).unwrap_or(0)),
            tuning: Tuning::HighQuality,
            ..base.clone()
        },
        Config {
            lookahead: h264.lookahead.then_some(8),
            tuning: Tuning::HighQuality,
            rate_control: RateControl::Vbr {
                average: 4_000_000,
                max: 6_000_000,
                vbv: None,
            },
            ..base.clone()
        },
    ];
    let tools = tools();
    for (variant, config) in variants.iter().enumerate() {
        let mut encoder = Encoder::open(&context, config).unwrap();
        let packets = encode_host(&mut encoder, &pictures, &[]);
        assert_eq!(packets.len(), pictures.len(), "variant {variant}");
        let indices: Vec<u64> = packets.iter().map(|packet| packet.index).collect();
        assert_eq!(indices, (0..pictures.len() as u64).collect::<Vec<_>>());
        let types: Vec<_> = packets.iter().map(|packet| packet.picture_type).collect();
        eprintln!("variant {variant}: {types:?}");
        if config.b_frames > 0 {
            assert!(types.contains(&PictureType::B), "{types:?}");
        }
        let path =
            output_dir(ApiVersion::negotiate().unwrap()).join(format!("ordering-{variant}.h264"));
        std::fs::write(&path, stream(&packets)).unwrap();
        if let Some((ffprobe, ffmpeg)) = &tools {
            assert_eq!(probe(ffprobe, &path, Codec::H264).frames, pictures.len());
            let quality = psnr(&decode(ffmpeg, &path, Codec::H264, "yuv420p"), &source, 8);
            eprintln!("variant {variant}: PSNR {quality:.2} dB");
            assert!(quality > 35.0);
        }
    }
}

#[test]
fn constant_qp_and_lossless() {
    let _gpu = gpu_lock();
    let Some(context) = context() else { return };
    let caps = Capabilities::query(&context).unwrap();
    let source = video(WIDTH, HEIGHT, 6, 8);
    let pictures: Vec<Vec<u8>> = source
        .iter()
        .map(|planes| pack(planes, InputFormat::Nv12, WIDTH, HEIGHT))
        .collect();
    let lossless = caps.require(Codec::Hevc).unwrap().lossless;
    let config = Config {
        rate_control: RateControl::ConstQp(QpSet::default()),
        tuning: if lossless {
            Tuning::Lossless
        } else {
            Tuning::HighQuality
        },
        preset: ferricast_nvcodec::encode::Preset::P4,
        profile: if lossless {
            None
        } else {
            Some(Profile::HevcMain)
        },
        ..Config::low_latency(Codec::Hevc, WIDTH, HEIGHT, InputFormat::Nv12, rate(30), 1)
    };
    let mut encoder = Encoder::open(&context, &config).unwrap();
    let packets = encode_host(&mut encoder, &pictures, &[]);
    let path = output_dir(ApiVersion::negotiate().unwrap()).join("lossless.hevc");
    std::fs::write(&path, stream(&packets)).unwrap();
    if let Some((_, ffmpeg)) = tools() {
        let quality = psnr(&decode(&ffmpeg, &path, Codec::Hevc, "yuv420p"), &source, 8);
        eprintln!("constant QP 0 (lossless tuning: {lossless}): PSNR {quality:.2} dB");
        assert!(quality > 60.0, "{quality}");
    }
}

/// A packed ARGB picture in a CUDA array, as EGL interop hands them out.
struct Array {
    raw: sys::CUarray,
}

impl Array {
    fn new(context: &Arc<Context>, width: u32, height: u32, argb: &[u8]) -> Self {
        let cuda = CudaLibrary::get().unwrap();
        let _current = context.guard().unwrap();
        let descriptor = sys::CUDA_ARRAY_DESCRIPTOR {
            Width: width as usize,
            Height: height as usize,
            Format: sys::CUarray_format::CU_AD_FORMAT_UNSIGNED_INT8,
            NumChannels: 4,
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: context current, valid descriptor and out-pointer.
        let status = unsafe { (cuda.array_create)(&raw mut raw, &raw const descriptor) };
        assert_eq!(status, sys::CUresult::CUDA_SUCCESS);
        let copy = sys::CUDA_MEMCPY2D {
            srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_HOST,
            srcHost: argb.as_ptr().cast(),
            srcPitch: width as usize * 4,
            dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_ARRAY,
            dstArray: raw,
            WidthInBytes: width as usize * 4,
            Height: height as usize,
            ..Default::default()
        };
        // SAFETY: the host image holds `Height` rows of `WidthInBytes`, the
        // array is that large; synchronous copy.
        let status = unsafe { (cuda.memcpy_2d)(&raw const copy) };
        assert_eq!(status, sys::CUresult::CUDA_SUCCESS);
        Self { raw }
    }

    fn destroy(self, context: &Arc<Context>) {
        let cuda = CudaLibrary::get().unwrap();
        let _current = context.guard().unwrap();
        // SAFETY: created above, no longer registered anywhere.
        let status = unsafe { (cuda.array_destroy)(self.raw) };
        assert_eq!(status, sys::CUresult::CUDA_SUCCESS);
    }
}

#[test]
fn cuda_array_input() {
    let _gpu = gpu_lock();
    let Some(context) = context() else { return };
    let format = InputFormat::Argb;
    let config = Config {
        gop: Gop::idr_every(NonZeroU32::new(30).unwrap()),
        ..Config::low_latency(Codec::Hevc, WIDTH, HEIGHT, format, rate(30), 6_000_000)
    };
    let source = video(WIDTH, HEIGHT, 6, 8);
    let pictures: Vec<Vec<u8>> = source
        .iter()
        .map(|planes| pack(planes, format, WIDTH, HEIGHT))
        .collect();
    let arrays: Vec<Array> = pictures
        .iter()
        .map(|picture| Array::new(&context, WIDTH, HEIGHT, picture))
        .collect();

    let mut encoder = Encoder::open(&context, &config).unwrap();
    let mut packets = Vec::new();
    for round in 0..2 {
        for (index, array) in arrays.iter().enumerate() {
            // SAFETY: the arrays outlive the encoder's use of them: the
            // caches are cleared before they are destroyed below, and no
            // picture is in flight after each call (frames_in_flight 0).
            let view = unsafe { ArrayView::from_raw(array.raw, &context, WIDTH, HEIGHT) };
            // SAFETY: as above.
            unsafe {
                encoder
                    .encode_array(
                        index as u64,
                        view,
                        FrameOptions {
                            timestamp: index as u64,
                            force_idr: false,
                        },
                    )
                    .unwrap();
            }
            let packet = encoder.next_packet().unwrap();
            if round == 0 {
                packets.push(packet);
            }
        }
        // Every array registered once, in the first round only.
        assert_eq!(encoder.registrations(), arrays.len() as u64);
    }
    let mut reference = Encoder::open(&context, &config).unwrap();
    let expected = encode_host(&mut reference, &pictures, &[]);
    assert_eq!(
        stream(&packets),
        stream(&expected),
        "array and host input differ"
    );

    encoder.invalidate_array(0);
    encoder.clear_input_caches();
    drop(encoder);
    for array in arrays {
        array.destroy(&context);
    }
}

/// Pitched memory owned elsewhere (what a LINEAR DMA-BUF maps to) encodes
/// in place, registered once per key, to the same stream as host input.
#[test]
fn pitched_input() {
    let _gpu = gpu_lock();
    let Some(context) = context() else { return };
    let format = InputFormat::Abgr;
    let config = Config {
        gop: Gop::idr_every(NonZeroU32::new(30).unwrap()),
        ..Config::low_latency(Codec::H264, WIDTH, HEIGHT, format, rate(30), 6_000_000)
    };
    let source = video(WIDTH, HEIGHT, 4, 8);
    let pictures: Vec<Vec<u8>> = source
        .iter()
        .map(|planes| pack(planes, format, WIDTH, HEIGHT))
        .collect();
    let row = WIDTH as usize * 4;
    let buffers: Vec<DeviceBuffer> = pictures
        .iter()
        .map(|picture| {
            let mut buffer = DeviceBuffer::pitched(&context, row, HEIGHT as usize).unwrap();
            buffer
                .upload(
                    buffer.region(),
                    HostImage {
                        data: picture,
                        pitch: row,
                    },
                )
                .unwrap();
            buffer
        })
        .collect();

    let mut encoder = Encoder::open(&context, &config).unwrap();
    let mut packets = Vec::new();
    for round in 0..2 {
        for (index, buffer) in buffers.iter().enumerate() {
            // SAFETY: the buffers outlive the encoder's use of them: the
            // encoder is dropped before them, and no picture is in flight
            // after each call (frames_in_flight 0).
            let view = unsafe {
                PitchedView::from_raw(
                    buffer.device_ptr(),
                    buffer.pitch(),
                    row,
                    HEIGHT as usize,
                    &context,
                )
            };
            // SAFETY: as above.
            unsafe {
                encoder
                    .encode_pitched(
                        index as u64,
                        view,
                        FrameOptions {
                            timestamp: index as u64,
                            force_idr: false,
                        },
                    )
                    .unwrap();
            }
            let packet = encoder.next_packet().unwrap();
            if round == 0 {
                packets.push(packet);
            }
        }
        assert_eq!(encoder.registrations(), buffers.len() as u64);
    }
    let mut reference = Encoder::open(&context, &config).unwrap();
    let expected = encode_host(&mut reference, &pictures, &[]);
    assert_eq!(
        stream(&packets),
        stream(&expected),
        "pitched and host input differ"
    );
    drop(encoder);
}
#[test]
fn steady_state_registers_nothing() {
    let _gpu = gpu_lock();
    let Some(context) = context() else { return };
    let config = Config {
        frames_in_flight: 2,
        ..Config::low_latency(
            Codec::H264,
            WIDTH,
            HEIGHT,
            InputFormat::Nv12,
            rate(30),
            4_000_000,
        )
    };
    let pictures: Vec<Vec<u8>> = video(WIDTH, HEIGHT, 4, 8)
        .iter()
        .map(|planes| pack(planes, InputFormat::Nv12, WIDTH, HEIGHT))
        .collect();
    let mut encoder = Encoder::open(&context, &config).unwrap();
    let depth = 1 + u64::from(config.frames_in_flight);
    for picture in pictures.iter().cycle().take(50) {
        encoder
            .encode_host(
                HostFrame::tight(picture, InputFormat::Nv12, WIDTH, HEIGHT),
                FrameOptions::default(),
            )
            .unwrap();
        while let Some(packet) = encoder.next_packet() {
            encoder.recycle(packet);
        }
    }
    assert_eq!(
        encoder.registrations(),
        depth,
        "one registration per host slot"
    );

    // Device buffers: one registration each, then none; a dropped buffer's
    // registration goes and a new buffer gets its own.
    let mut buffers = device_pictures(&context, &pictures, InputFormat::Nv12, WIDTH, HEIGHT, None);
    for buffer in buffers.iter().cycle().take(40) {
        encoder
            .encode_device(DeviceFrame::new(buffer), FrameOptions::default())
            .unwrap();
        while let Some(packet) = encoder.next_packet() {
            encoder.recycle(packet);
        }
    }
    assert_eq!(encoder.registrations(), depth + buffers.len() as u64);
    buffers.truncate(1);
    buffers.extend(device_pictures(
        &context,
        &pictures[..1],
        InputFormat::Nv12,
        WIDTH,
        HEIGHT,
        None,
    ));
    for buffer in &buffers {
        encoder
            .encode_device(DeviceFrame::new(buffer), FrameOptions::default())
            .unwrap();
    }
    encoder.flush().unwrap();
    assert_eq!(encoder.registrations(), depth + pictures.len() as u64 + 1);

    // A buffer still being encoded is refused rather than mapped twice.
    let busy = &buffers[1];
    encoder
        .encode_device(DeviceFrame::new(busy), FrameOptions::default())
        .unwrap();
    let again = encoder.encode_device(DeviceFrame::new(busy), FrameOptions::default());
    assert!(
        matches!(again, Err(Error::InvalidEncodeInput { .. })),
        "{again:?}"
    );
}

#[test]
fn invalid_inputs_are_refused() {
    let _gpu = gpu_lock();
    let Some(context) = context() else { return };
    let config = Config::low_latency(
        Codec::H264,
        WIDTH,
        HEIGHT,
        InputFormat::Nv12,
        rate(30),
        4_000_000,
    );
    let mut encoder = Encoder::open(&context, &config).unwrap();
    let short = vec![0; 100];
    assert!(matches!(
        encoder.encode_host(
            HostFrame::tight(&short, InputFormat::Nv12, WIDTH, HEIGHT),
            FrameOptions::default()
        ),
        Err(Error::CopyOutOfBounds { .. })
    ));
    let packed = vec![0; InputFormat::Argb.row_bytes(WIDTH) * HEIGHT as usize];
    assert!(matches!(
        encoder.encode_host(
            HostFrame::tight(&packed, InputFormat::Argb, WIDTH, HEIGHT),
            FrameOptions::default()
        ),
        Err(Error::InvalidEncodeInput { .. })
    ));
    let small = Arc::new(DeviceBuffer::pitched(&context, 64, 64).unwrap());
    assert!(matches!(
        encoder.encode_device(DeviceFrame::new(&small), FrameOptions::default()),
        Err(Error::InvalidEncodeInput { .. })
    ));
    let bad = Config { width: 0, ..config };
    assert!(matches!(
        Encoder::open(&context, &bad),
        Err(Error::InvalidEncodeConfig { field: "size", .. })
    ));
    let huge = Config {
        width: 16384,
        height: 16384,
        ..Config::low_latency(Codec::H264, 2, 2, InputFormat::Nv12, rate(30), 1_000_000)
    };
    let error = Encoder::open(&context, &huge).unwrap_err();
    assert!(matches!(error, Error::Unsupported { .. }), "{error}");
}

/// Set in the child processes of [`isolated`].
const ISOLATED_VAR: &str = "FERRICAST_NVENC_TEST_ISOLATED";

/// Re-runs `test` alone in a child process and returns `true` there: for
/// tests that must not share the driver with the others (the session limit
/// starves them; the driver refuses sessions of different API versions open
/// at the same time in one process).
fn isolated(test: &str) -> bool {
    if std::env::var_os(ISOLATED_VAR).is_some() {
        return true;
    }
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture", "--test-threads", "1"])
        .env(ISOLATED_VAR, "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    eprintln!("{}", stderr.trim());
    assert!(
        output.status.success(),
        "{test} failed in its child process"
    );
    false
}

#[test]
fn session_limit_is_recognisable() {
    let _gpu = gpu_lock();
    if !isolated("session_limit_is_recognisable") {
        return;
    }
    let Some(context) = context() else { return };
    let config = Config::low_latency(
        Codec::H264,
        256,
        256,
        InputFormat::Nv12,
        rate(30),
        1_000_000,
    );
    let mut sessions = Vec::new();
    for _ in 0..16 {
        match Encoder::open(&context, &config) {
            Ok(encoder) => sessions.push(encoder),
            Err(error) => {
                eprintln!("after {} sessions: {error}", sessions.len());
                assert!(
                    matches!(error, Error::NvencSessionLimit { .. }),
                    "{error:?}"
                );
                return;
            }
        }
    }
    eprintln!("{} sessions opened without hitting a limit", sessions.len());
}

#[test]
fn throughput_1080p() {
    let _gpu = gpu_lock();
    let Some(context) = context() else { return };
    let (width, height) = (1920, 1080);
    let frames = 240;
    let source = video(width, height, 8, 8);
    let pictures: Vec<Vec<u8>> = source
        .iter()
        .map(|planes| pack(planes, InputFormat::Nv12, width, height))
        .collect();
    let buffers = device_pictures(&context, &pictures, InputFormat::Nv12, width, height, None);
    for in_flight in [0, 3] {
        let config = Config {
            frames_in_flight: in_flight,
            preset: ferricast_nvcodec::encode::Preset::P1,
            tuning: Tuning::UltraLowLatency,
            ..Config::low_latency(
                Codec::H264,
                width,
                height,
                InputFormat::Nv12,
                rate(60),
                20_000_000,
            )
        };
        for device in [false, true] {
            let mut encoder = Encoder::open(&context, &config).unwrap();
            let start = Instant::now();
            for index in 0..frames {
                let options = FrameOptions {
                    timestamp: index as u64,
                    force_idr: false,
                };
                if device {
                    let buffer = &buffers[index % buffers.len()];
                    // Each buffer comes back only after its packet did.
                    while encoder
                        .encode_device(DeviceFrame::new(buffer), options)
                        .is_err()
                    {
                        encoder.flush().unwrap();
                    }
                } else {
                    let picture = &pictures[index % pictures.len()];
                    encoder
                        .encode_host(
                            HostFrame::tight(picture, InputFormat::Nv12, width, height),
                            options,
                        )
                        .unwrap();
                }
                while let Some(packet) = encoder.next_packet() {
                    encoder.recycle(packet);
                }
            }
            encoder.flush().unwrap();
            let fps = frames as f64 / start.elapsed().as_secs_f64();
            eprintln!(
                "1080p NV12 H.264 P1 ULL, {} input, {in_flight} in flight: {fps:.0} fps (API {})",
                if device { "device" } else { "host" },
                encoder.api_version()
            );
            assert!(fps > 30.0);
        }
    }
}

#[test]
fn every_version_loads_through_any_api() {
    let _gpu = gpu_lock();
    if !isolated("every_version_loads_through_any_api") {
        return;
    }
    let Some(driver_max) = driver_max() else {
        return;
    };
    let Some(context) = context() else { return };
    let config = Config::low_latency(
        Codec::Hevc,
        320,
        240,
        InputFormat::Nv12,
        rate(30),
        1_000_000,
    );
    for version in ApiVersion::ALL
        .into_iter()
        .filter(|version| version.version() <= driver_max)
    {
        let encoder =
            Encoder::open_with(AnyApi::load(version).unwrap(), &context, &config).unwrap();
        assert_eq!(encoder.api_version(), version);
    }
}
