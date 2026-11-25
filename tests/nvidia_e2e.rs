//! NVIDIA end to end through ferricast's facades, on real hardware:
//!
//! - DMA-BUF capture frames (GBM buffers on the NVIDIA GPU, written by
//!   GLES or CUDA) encoded by NVENC through the zero-copy path, decoded
//!   back and compared with the pattern that was written;
//! - NVDEC pictures kept in CUDA memory encoded by NVENC without a host
//!   readback;
//! - 10-bit HDR10 HEVC from NVENC, decoded back with its colour and
//!   metadata.
//!
//! Every test prints why it skips on a machine without an NVIDIA GPU (or
//! GBM / EGL for the DMA-BUF one).

#![cfg(all(feature = "nvenc", target_os = "linux"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "test patterns and PSNR over small, positive sizes"
)]

use std::os::fd::AsFd as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ferricast_core::{
    AccelerationPolicy, AutoSelect, BackendId, BackendSelection, BufferLease, CapturedFrame, Codec,
    CodecError, ColourDescription, DecoderConfig, DmaBufFd, DmaBufPlane, EncodedFrame,
    EncoderConfig, FrameMemory, GpuFrame, PixelFormat, RawFrame, VideoDecoder, VideoEncoder,
};
use ferricast_nvcodec::interop::{DmaBufImporter, DrmFormat, Modifier};

#[path = "../crates/nvcodec/tests/support/dmabuf.rs"]
#[allow(
    dead_code,
    clippy::all,
    clippy::pedantic,
    reason = "shared test support of ferricast-nvcodec, linted there"
)]
mod dmabuf;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const FRAMES: usize = 30;

/// A smooth moving gradient (lossy codecs keep it well), BGRA in memory.
fn bgra_pattern(width: u32, height: u32, frame: usize) -> Vec<u8> {
    let shift = frame as u32 * 4;
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .flat_map(|(x, y)| {
            let r = ((x + shift) * 255 / (width + shift)) as u8;
            let g = (y * 255 / height) as u8;
            let b = ((u32::midpoint(x, y) + shift) % 256) as u8;
            [b / 2 + 64, g, r, 0xff]
        })
        .collect()
}

/// PSNR in dB over the colour channels of two BGRA (or RGBA) pictures.
fn psnr_rgb(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "picture sizes");
    let (sum, count) = a
        .chunks_exact(4)
        .zip(b.chunks_exact(4))
        .flat_map(|(a, b)| (0..3).map(move |c| f64::from(a[c]) - f64::from(b[c])))
        .fold((0.0, 0_usize), |(sum, count), d| (sum + d * d, count + 1));
    let mse = sum / count as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

/// PSNR in dB of two equal-length runs of samples with the given peak.
fn psnr(a: impl Iterator<Item = f64>, b: impl Iterator<Item = f64>, peak: f64) -> f64 {
    let (sum, count) = a.zip(b).fold((0.0, 0_usize), |(sum, count), (a, b)| {
        (sum + (a - b) * (a - b), count + 1)
    });
    let mse = sum / count as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (peak * peak / mse).log10()
    }
}

fn nvenc(codec: Codec, format: PixelFormat) -> EncoderConfig {
    EncoderConfig {
        codec,
        width: WIDTH,
        height: HEIGHT,
        bitrate_kbps: 20_000,
        fps: 30,
        pixel_format: format,
        backend: BackendSelection::Fixed(BackendId::Nvenc),
        ..EncoderConfig::default()
    }
}

/// `E` configured for `config`, or `None` (printed) without NVENC.
fn encoder<E: VideoEncoder + Default>(name: &str, config: &EncoderConfig) -> Option<E> {
    let mut encoder = E::default();
    match encoder.configure(config) {
        Ok(()) => Some(encoder),
        Err(error @ CodecError::BackendUnavailable { .. }) => {
            eprintln!("{name}: skipped, {error}");
            None
        }
        Err(error) => panic!("{name}: {error}"),
    }
}

/// Every picture `D` (on `backend`) decodes from `packets`, in host
/// memory, as `format`.
fn decode<D: VideoDecoder + Default>(
    packets: &[EncodedFrame],
    format: PixelFormat,
    backend: BackendSelection,
) -> Vec<RawFrame> {
    let mut decoder = D::default();
    decoder
        .configure(&DecoderConfig {
            codec: D::CODEC,
            width: WIDTH,
            height: HEIGHT,
            pixel_format: format,
            low_latency: true,
            colour: None,
            hdr: None,
            memory: FrameMemory::Cpu,
            backend,
        })
        .unwrap();
    let mut frames: Vec<CapturedFrame> = packets
        .iter()
        .filter_map(|packet| decoder.decode(packet.clone()).unwrap())
        .collect();
    frames.extend(decoder.flush().unwrap());
    frames.into_iter().map(|f| f.into_cpu().unwrap()).collect()
}

fn software() -> BackendSelection {
    BackendSelection::Auto(
        AutoSelect::default().with_acceleration(AccelerationPolicy::SoftwareOnly),
    )
}

/// A decoded NV12 picture as BGRA, with the colour it signals.
fn to_bgra(frame: &RawFrame) -> Vec<u8> {
    assert_eq!(frame.format, PixelFormat::Nv12);
    let colour = frame.colour.expect("decoded pictures carry their colour");
    colour
        .yuv(frame.height as usize)
        .nv12_to_bgra(
            &frame.data,
            frame.stride as usize,
            frame.width as usize,
            frame.height as usize,
        )
        .unwrap()
}

/// Counts the producer getting its buffer back.
struct Returned(Arc<AtomicUsize>);

impl Drop for Returned {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// Encodes DMA-BUF frames of one layout through NVENC's zero-copy path,
/// decodes them back in software and checks every picture against what
/// was written. Frames carry no CPU importer: a readback fallback would
/// fail the encode, so every frame that encodes went zero-copy.
fn dma_buf_round_trip<E: VideoEncoder + Default, D: VideoDecoder + Default>(
    name: &str,
    codec: Codec,
    modifier: impl Fn(&DmaBufImporter) -> Modifier,
    colour: ColourDescription,
) {
    let Some(gpu) = dmabuf::Gpu::get() else {
        return;
    };
    let importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
    let modifier = modifier(&importer);
    let config = EncoderConfig {
        colour,
        ..nvenc(codec, PixelFormat::Bgra)
    };
    let Some(mut encoder) = encoder::<E>(name, &config) else {
        return;
    };
    // A capture pool of three buffers, reused round-robin.
    let pool: Vec<dmabuf::Bo> = (0..3)
        .map_while(|_| gpu.allocate(WIDTH, HEIGHT, DrmFormat::Argb8888, modifier))
        .collect();
    assert_eq!(pool.len(), 3, "{name}: GBM allocation");
    let returned = Arc::new(AtomicUsize::new(0));
    let mut packets = Vec::new();
    let patterns: Vec<Vec<u8>> = (0..FRAMES)
        .map(|index| bgra_pattern(WIDTH, HEIGHT, index))
        .collect();
    for (index, pattern) in patterns.iter().enumerate() {
        let bo = &pool[index % pool.len()];
        gpu.write(bo, pattern);
        let frame = GpuFrame {
            width: WIDTH,
            height: HEIGHT,
            stride: bo.stride,
            format: PixelFormat::Bgra,
            timestamp_us: index as u64 * 33_333,
            plane: DmaBufPlane {
                fd: DmaBufFd::dup(bo.fd.as_fd()).unwrap(),
                offset: bo.offset,
                stride: bo.stride,
                modifier: bo.modifier.0,
                size: bo.stride * HEIGHT,
            },
            importer: None,
            lease: Some(BufferLease::new(Returned(Arc::clone(&returned)))),
        };
        let packet = encoder
            .encode(CapturedFrame::Gpu(frame))
            .unwrap_or_else(|error| panic!("{name}: frame {index}: {error}"));
        // The encoder let go of the capture buffer once NVENC was done.
        assert_eq!(returned.load(Ordering::SeqCst), index + 1, "{name}: lease");
        packets.push(packet);
    }
    assert!(
        packets[0].is_keyframe,
        "{name}: the stream opens with an IDR"
    );
    let decoded = decode::<D>(&packets, PixelFormat::Nv12, software());
    assert_eq!(decoded.len(), FRAMES, "{name}: pictures decoded");
    let worst = decoded
        .iter()
        .zip(&patterns)
        .map(|(frame, pattern)| {
            assert_eq!(
                frame.colour.map(|c| c.matrix),
                Some(colour.matrix),
                "{name}"
            );
            psnr_rgb(&to_bgra(frame), pattern)
        })
        .fold(f64::INFINITY, f64::min);
    eprintln!("{name}: worst PSNR {worst:.1} dB over {FRAMES} frames");
    assert!(worst > 35.0, "{name}: PSNR {worst:.1} dB");
}

#[test]
fn nvenc_h264_encodes_tiled_dma_bufs_zero_copy() {
    dma_buf_round_trip::<ferricast_encoder::h264::H264Encoder, ferricast_decoder::H264Decoder>(
        "H.264 tiled DMA-BUF",
        Codec::H264,
        dmabuf::Gpu::tiled_modifier,
        ColourDescription::BT709_LIMITED,
    );
}

#[test]
fn nvenc_h264_encodes_linear_dma_bufs_zero_copy() {
    dma_buf_round_trip::<ferricast_encoder::h264::H264Encoder, ferricast_decoder::H264Decoder>(
        "H.264 LINEAR DMA-BUF",
        Codec::H264,
        |_| Modifier::LINEAR,
        ColourDescription::BT709_LIMITED,
    );
}

/// NVENC converts RGB with the matrix the configuration (and the VUI)
/// names.
#[test]
fn nvenc_converts_rgb_with_the_configured_matrix() {
    dma_buf_round_trip::<ferricast_encoder::h264::H264Encoder, ferricast_decoder::H264Decoder>(
        "H.264 tiled DMA-BUF, BT.601",
        Codec::H264,
        dmabuf::Gpu::tiled_modifier,
        ColourDescription::BT601_LIMITED,
    );
}

#[cfg(feature = "nvenc-hevc")]
#[test]
fn nvenc_hevc_encodes_tiled_dma_bufs_zero_copy() {
    dma_buf_round_trip::<ferricast_encoder::h265::H265Encoder, ferricast_decoder::H265Decoder>(
        "HEVC tiled DMA-BUF",
        Codec::H265,
        dmabuf::Gpu::tiled_modifier,
        ColourDescription::BT709_LIMITED,
    );
}

/// An NV12 test stream: the pattern, converted on the CPU, encoded by
/// NVENC from host memory.
fn nv12_source(name: &str) -> Option<(Vec<RawFrame>, Vec<EncodedFrame>)> {
    let mut encoder = encoder::<ferricast_encoder::h264::H264Encoder>(
        name,
        &nvenc(Codec::H264, PixelFormat::Bgra),
    )?;
    let mut sources = Vec::new();
    let mut packets = Vec::new();
    for index in 0..FRAMES {
        let raw = RawFrame {
            width: WIDTH,
            height: HEIGHT,
            stride: WIDTH * 4,
            format: PixelFormat::Bgra,
            data: bgra_pattern(WIDTH, HEIGHT, index).into(),
            timestamp_us: index as u64 * 33_333,
            colour: None,
            hdr: None,
        };
        packets.push(encoder.encode(CapturedFrame::Cpu(raw.clone())).unwrap());
        sources.push(raw);
    }
    Some((sources, packets))
}

/// NVDEC → NVENC on the GPU: device pictures are encoded in place, none is
/// read back to host memory, and the result decodes to the source.
#[cfg(feature = "nvdec")]
#[test]
fn nvdec_device_frames_encode_on_nvenc_without_readback() {
    use ferricast_codec::cuda::CudaSurface;

    let name = "NVDEC → NVENC";
    let Some((_, source)) = nv12_source(name) else {
        return;
    };
    // The reference: the same stream decoded to host memory.
    let reference =
        decode::<ferricast_decoder::H264Decoder>(&source, PixelFormat::Nv12, software());

    let mut decoder = ferricast_decoder::H264Decoder::default();
    decoder
        .configure(&DecoderConfig {
            codec: Codec::H264,
            width: WIDTH,
            height: HEIGHT,
            pixel_format: PixelFormat::Nv12,
            low_latency: true,
            colour: None,
            hdr: None,
            memory: FrameMemory::Native,
            backend: BackendSelection::Fixed(BackendId::Nvdec),
        })
        .unwrap();
    let mut encoder = encoder::<ferricast_encoder::h264::H264Encoder>(
        name,
        &nvenc(Codec::H264, PixelFormat::Nv12),
    )
    .unwrap();
    let readbacks = CudaSurface::readbacks();
    let mut packets = Vec::new();
    for packet in &source {
        if let Some(frame) = decoder.decode(packet.clone()).unwrap() {
            let CapturedFrame::Native(native) = &frame else {
                panic!("{name}: NVDEC handed out {frame:?}");
            };
            assert_eq!(native.kind(), ferricast_core::SurfaceKind::CudaDevice);
            packets.push(encoder.encode(frame).unwrap());
        }
    }
    for frame in decoder.flush().unwrap() {
        packets.push(encoder.encode(frame).unwrap());
    }
    assert_eq!(
        CudaSurface::readbacks(),
        readbacks,
        "{name}: a device picture was read back to host memory"
    );
    let round_trip =
        decode::<ferricast_decoder::H264Decoder>(&packets, PixelFormat::Nv12, software());
    assert_eq!(round_trip.len(), reference.len(), "{name}: pictures");
    let worst = round_trip
        .iter()
        .zip(&reference)
        .map(|(got, want)| {
            let samples = |frame: &RawFrame| -> Vec<f64> {
                let rows = (frame.height + frame.height / 2) as usize;
                frame
                    .data
                    .chunks(frame.stride as usize)
                    .take(rows)
                    .flat_map(|row| row[..frame.width as usize].iter().map(|&s| f64::from(s)))
                    .collect()
            };
            psnr(samples(got).into_iter(), samples(want).into_iter(), 255.0)
        })
        .fold(f64::INFINITY, f64::min);
    eprintln!("{name}: worst PSNR {worst:.1} dB against the first decode");
    assert!(worst > 38.0, "{name}: PSNR {worst:.1} dB");
}

/// P010 pattern: 10-bit samples in the high bits.
#[cfg(feature = "nvenc-hevc")]
fn p010_pattern(width: u32, height: u32, frame: usize) -> Vec<u8> {
    let shift = frame as u32 * 8;
    let luma = (0..height).flat_map(|y| {
        (0..width).map(move |x| (64 + (x + shift) * 876 / (width + shift) + y % 2) as u16)
    });
    let chroma = (0..height / 2).flat_map(|y| {
        (0..width / 2).flat_map(move |x| {
            [
                (512 + x * 200 / width) as u16,
                (512 - y * 200 / height + shift % 32) as u16,
            ]
        })
    });
    luma.chain(chroma)
        .flat_map(|sample| (sample << 6).to_le_bytes())
        .collect()
}

/// HDR10 metadata of a 1000 cd/m² BT.2020 mastering display.
#[cfg(feature = "nvenc-hevc")]
fn hdr10() -> ferricast_core::HdrMetadata {
    use ferricast_core::{Chromaticity, ContentLightLevel, HdrMetadata, MasteringDisplay};
    let point = |x, y| Chromaticity { x, y };
    HdrMetadata {
        mastering_display: Some(MasteringDisplay {
            red: point(35_400, 14_600),
            green: point(8_500, 39_850),
            blue: point(6_550, 2_300),
            white_point: point(15_635, 16_450),
            max_luminance: 10_000_000,
            min_luminance: 50,
        }),
        content_light: Some(ContentLightLevel {
            max_cll: 1000,
            max_fall: 400,
        }),
    }
}

/// 10-bit HDR10 HEVC from P010: decoded back as P010 with BT.2020 PQ and
/// the mastering display and light levels it was configured with.
#[cfg(feature = "nvenc-hevc")]
#[test]
fn nvenc_encodes_ten_bit_hdr10_hevc() {
    use ferricast_core::{BitDepth, ColourPrimaries, TransferCharacteristics};

    let name = "HEVC Main10 HDR10";
    let hdr = hdr10();
    let colour = ColourDescription::from_codes(9, 16, 9, false);
    let config = EncoderConfig {
        bit_depth: BitDepth::Ten,
        colour,
        hdr: Some(hdr),
        ..nvenc(Codec::H265, PixelFormat::P010)
    };
    let Some(mut encoder) = encoder::<ferricast_encoder::h265::H265Encoder>(name, &config) else {
        return;
    };
    let sources: Vec<Vec<u8>> = (0..FRAMES)
        .map(|index| p010_pattern(WIDTH, HEIGHT, index))
        .collect();
    let packets: Vec<EncodedFrame> = sources
        .iter()
        .enumerate()
        .map(|(index, data)| {
            let raw = RawFrame {
                width: WIDTH,
                height: HEIGHT,
                stride: WIDTH * 2,
                format: PixelFormat::P010,
                data: data.clone().into(),
                timestamp_us: index as u64 * 33_333,
                colour: Some(colour),
                hdr: Some(hdr),
            };
            encoder.encode(CapturedFrame::Cpu(raw)).unwrap()
        })
        .collect();
    let mut backends = vec![("rust_h265", software())];
    if cfg!(feature = "nvdec") {
        backends.push(("NVDEC", BackendSelection::Fixed(BackendId::Nvdec)));
    }
    for (decoder_name, backend) in backends {
        let decoded =
            decode::<ferricast_decoder::H265Decoder>(&packets, PixelFormat::P010, backend);
        assert_eq!(decoded.len(), FRAMES, "{name} via {decoder_name}: pictures");
        let mut worst = f64::INFINITY;
        for (frame, source) in decoded.iter().zip(&sources) {
            assert_eq!(frame.format, PixelFormat::P010, "{name} via {decoder_name}");
            let colour = frame.colour.expect("colour");
            assert_eq!(colour.primaries, ColourPrimaries::Bt2020, "{decoder_name}");
            assert_eq!(
                colour.transfer,
                TransferCharacteristics::Pq,
                "{decoder_name}"
            );
            assert_eq!(
                frame.hdr,
                Some(hdr),
                "{name} via {decoder_name}: HDR metadata"
            );
            let words = |data: &[u8], stride: usize| -> Vec<f64> {
                data.chunks(stride)
                    .take(HEIGHT as usize)
                    .flat_map(|row| {
                        row[..WIDTH as usize * 2]
                            .chunks_exact(2)
                            .map(|w| f64::from(u16::from_le_bytes([w[0], w[1]]) >> 6))
                    })
                    .collect()
            };
            let got = words(&frame.data, frame.stride as usize);
            let want = words(source, WIDTH as usize * 2);
            worst = worst.min(psnr(got.into_iter(), want.into_iter(), 1023.0));
        }
        eprintln!("{name} via {decoder_name}: worst luma PSNR {worst:.1} dB (10-bit)");
        assert!(
            worst > 40.0,
            "{name} via {decoder_name}: PSNR {worst:.1} dB"
        );
    }
}

/// Reads a fixed picture back, standing in for the capture's Vulkan
/// importer.
struct Readback(Vec<u8>, Arc<AtomicUsize>);

impl ferricast_core::DmaBufImporter for Readback {
    fn readback(
        &self,
        _plane: &DmaBufPlane,
        _width: u32,
        _height: u32,
        _format: PixelFormat,
    ) -> ferricast_core::Result<bytes::Bytes> {
        self.1.fetch_add(1, Ordering::SeqCst);
        Ok(bytes::Bytes::from(self.0.clone()))
    }
}

/// A frame the zero-copy path cannot import goes through the CPU readback;
/// the next importable frame goes zero-copy again.
#[test]
fn a_zero_copy_failure_falls_back_for_that_frame_only() {
    let name = "zero-copy fallback";
    let Some(gpu) = dmabuf::Gpu::get() else {
        return;
    };
    let Some(mut encoder) = encoder::<ferricast_encoder::h264::H264Encoder>(
        name,
        &nvenc(Codec::H264, PixelFormat::Bgra),
    ) else {
        return;
    };
    let pattern = bgra_pattern(WIDTH, HEIGHT, 0);
    let readbacks = Arc::new(AtomicUsize::new(0));
    // Not a DMA-BUF at all: EGL refuses to import it.
    let memfd = memfd();
    let not_importable = GpuFrame {
        width: WIDTH,
        height: HEIGHT,
        stride: WIDTH * 4,
        format: PixelFormat::Bgra,
        timestamp_us: 0,
        plane: DmaBufPlane {
            fd: DmaBufFd::from(memfd),
            offset: 0,
            stride: WIDTH * 4,
            modifier: 0,
            size: WIDTH * 4 * HEIGHT,
        },
        importer: Some(Arc::new(Readback(pattern.clone(), Arc::clone(&readbacks)))),
        lease: None,
    };
    encoder.encode(CapturedFrame::Gpu(not_importable)).unwrap();
    assert_eq!(readbacks.load(Ordering::SeqCst), 1, "{name}: read back");

    let importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
    let bo = gpu
        .allocate(
            WIDTH,
            HEIGHT,
            DrmFormat::Argb8888,
            dmabuf::Gpu::tiled_modifier(&importer),
        )
        .unwrap();
    gpu.write(&bo, &pattern);
    let importable = GpuFrame {
        width: WIDTH,
        height: HEIGHT,
        stride: bo.stride,
        format: PixelFormat::Bgra,
        timestamp_us: 33_333,
        plane: DmaBufPlane {
            fd: DmaBufFd::dup(bo.fd.as_fd()).unwrap(),
            offset: bo.offset,
            stride: bo.stride,
            modifier: bo.modifier.0,
            size: bo.stride * HEIGHT,
        },
        importer: Some(Arc::new(Readback(pattern, Arc::clone(&readbacks)))),
        lease: None,
    };
    encoder.encode(CapturedFrame::Gpu(importable)).unwrap();
    assert_eq!(
        readbacks.load(Ordering::SeqCst),
        1,
        "{name}: the importable frame was read back"
    );
}

/// An anonymous memory file: a descriptor that is not a DMA-BUF.
fn memfd() -> std::os::fd::OwnedFd {
    use std::os::fd::FromRawFd as _;
    // SAFETY: a NUL-terminated name; `memfd_create` returns a new descriptor
    // or -1.
    let fd = unsafe { libc::memfd_create(c"ferricast-not-a-dmabuf".as_ptr(), libc::MFD_CLOEXEC) };
    assert!(fd >= 0, "memfd_create");
    // SAFETY: `fd` was just created and nothing else owns it.
    unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) }
}
