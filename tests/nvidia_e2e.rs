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
    CodecError, ColourDescription, DecoderConfig, DmaBufFd, DmaBufPlane, DrmDevice, EncodedFrame,
    EncoderConfig, FrameMemory, FrameSync, GpuFrame, Orientation, PixelFormat, RawFrame, Rotation,
    VideoDecoder, VideoEncoder,
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
            producer: Some(DrmDevice::from_dev(gpu.node.dev())),
            // GLES finished writing (glFinish) before the frame is built.
            sync: FrameSync::Ready,
            orientation: Orientation::IDENTITY,
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
    let _gpu = one_at_a_time();
    dma_buf_round_trip::<ferricast_encoder::h264::H264Encoder, ferricast_decoder::H264Decoder>(
        "H.264 tiled DMA-BUF",
        Codec::H264,
        dmabuf::Gpu::tiled_modifier,
        ColourDescription::BT709_LIMITED,
    );
}

#[test]
fn nvenc_h264_encodes_linear_dma_bufs_zero_copy() {
    let _gpu = one_at_a_time();
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
    let _gpu = one_at_a_time();
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
    let _gpu = one_at_a_time();
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
    let _gpu = one_at_a_time();
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
    let _gpu = one_at_a_time();
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
    let _gpu = one_at_a_time();
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
        producer: None,
        sync: FrameSync::Implicit,
        orientation: Orientation::IDENTITY,
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
        producer: Some(DrmDevice::from_dev(gpu.node.dev())),
        sync: FrameSync::Ready,
        orientation: Orientation::IDENTITY,
    };
    encoder.encode(CapturedFrame::Gpu(importable)).unwrap();
    assert_eq!(
        readbacks.load(Ordering::SeqCst),
        1,
        "{name}: the importable frame was read back"
    );
}

/// A frame one pixel larger than the session per axis (an odd source
/// size the session rounded down to even) is encoded zero-copy from its
/// top-left corner, and a buffer the importer refused once is read back
/// again without another import attempt.
#[test]
fn odd_sized_frames_go_zero_copy_and_refused_buffers_stay_read_back() {
    let _gpu = one_at_a_time();
    let name = "odd size";
    let Some(gpu) = dmabuf::Gpu::get() else {
        return;
    };
    let Some(mut encoder) = encoder::<ferricast_encoder::h264::H264Encoder>(
        name,
        &nvenc(Codec::H264, PixelFormat::Bgra),
    ) else {
        return;
    };
    let importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
    let (width, height) = (WIDTH + 1, HEIGHT + 1);
    let bo = gpu
        .allocate(
            width,
            height,
            DrmFormat::Argb8888,
            dmabuf::Gpu::tiled_modifier(&importer),
        )
        .unwrap();
    gpu.write(&bo, &bgra_pattern(width, height, 0));
    let odd = GpuFrame {
        width,
        height,
        stride: bo.stride,
        format: PixelFormat::Bgra,
        timestamp_us: 0,
        plane: DmaBufPlane {
            fd: DmaBufFd::dup(bo.fd.as_fd()).unwrap(),
            offset: bo.offset,
            stride: bo.stride,
            modifier: bo.modifier.0,
            size: bo.stride * height,
        },
        // No CPU importer: a readback would fail the encode.
        importer: None,
        lease: None,
        producer: Some(DrmDevice::from_dev(gpu.node.dev())),
        sync: FrameSync::Ready,
        orientation: Orientation::IDENTITY,
    };
    let packet = encoder.encode(CapturedFrame::Gpu(odd)).unwrap();
    assert!(packet.is_keyframe, "{name}: encoded zero-copy");

    let pattern = bgra_pattern(width, height, 1);
    let readbacks = Arc::new(AtomicUsize::new(0));
    let memfd = DmaBufFd::from(memfd());
    for round in 1..=3 {
        let refused = GpuFrame {
            width,
            height,
            stride: width * 4,
            format: PixelFormat::Bgra,
            timestamp_us: round * 33_333,
            plane: DmaBufPlane {
                fd: memfd.clone(),
                offset: 0,
                stride: width * 4,
                modifier: 0,
                size: width * 4 * height,
            },
            importer: Some(Arc::new(Readback(pattern.clone(), Arc::clone(&readbacks)))),
            lease: None,
            producer: None,
            sync: FrameSync::Implicit,
            orientation: Orientation::IDENTITY,
        };
        encoder.encode(CapturedFrame::Gpu(refused)).unwrap();
    }
    assert_eq!(readbacks.load(Ordering::SeqCst), 3, "{name}: read back");
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

// ------------------------------------------------------------------------
// GPU fit: pictures whose size or format differs from the session are
// scaled / letterboxed / narrowed by ferricast-nvcodec's image kernels on
// the GPU, never read back.

/// `bgra` (`sw` × `sh`) resampled to `dw` × `dh` with the triangle filter
/// and adaptive support the CPU scaler (`fast_image_resize`) uses, in f64.
fn resize_reference(
    src: &[f64],
    channels: usize,
    (sw, sh): (usize, usize),
    (dw, dh): (usize, usize),
) -> Vec<f64> {
    let taps = |input: usize, output: usize| -> Vec<(usize, Vec<f64>)> {
        let scale = input as f64 / output as f64;
        let support = scale.max(1.0);
        (0..output)
            .map(|o| {
                let center = (o as f64 + 0.5) * scale;
                let lo = (center - support).floor().max(0.0) as usize;
                let hi = ((center + support).ceil() as usize).min(input);
                let weights: Vec<f64> = (lo..hi)
                    .map(|i| (1.0 - ((i as f64 - (center - 0.5)) / support).abs()).max(0.0))
                    .collect();
                let sum: f64 = weights.iter().sum();
                (lo, weights.into_iter().map(|w| w / sum).collect())
            })
            .collect()
    };
    let (horizontal, vertical) = (taps(sw, dw), taps(sh, dh));
    let mut rows = vec![0.0; dw * sh * channels];
    for y in 0..sh {
        for (x, (lo, weights)) in horizontal.iter().enumerate() {
            for (k, w) in weights.iter().enumerate() {
                for c in 0..channels {
                    rows[(y * dw + x) * channels + c] += w * src[(y * sw + lo + k) * channels + c];
                }
            }
        }
    }
    let mut out = vec![0.0; dw * dh * channels];
    for (y, (lo, weights)) in vertical.iter().enumerate() {
        for (k, w) in weights.iter().enumerate() {
            for x in 0..dw * channels {
                out[y * dw * channels + x] += w * rows[(lo + k) * dw * channels + x];
            }
        }
    }
    out
}

/// The CPU path's picture: `bgra` letterboxed into `canvas` with
/// ferricast's geometry (black bars).
fn letterbox_reference(bgra: &[u8], source: (u32, u32), canvas: (u32, u32)) -> Vec<u8> {
    use ferricast_encoder::scale::FrameSize;
    let placement =
        FrameSize::new(source.0, source.1).letterbox_in(FrameSize::new(canvas.0, canvas.1));
    let (pw, ph) = (
        placement.size.width as usize,
        placement.size.height as usize,
    );
    let src: Vec<f64> = bgra.iter().map(|&b| f64::from(b)).collect();
    let placed = resize_reference(&src, 4, (source.0 as usize, source.1 as usize), (pw, ph));
    let (cw, ch) = (canvas.0 as usize, canvas.1 as usize);
    let mut out: Vec<u8> = [0, 0, 0, 255]
        .into_iter()
        .cycle()
        .take(cw * ch * 4)
        .collect();
    for y in 0..ph {
        for x in 0..pw {
            let at = ((y + placement.y as usize) * cw + x + placement.x as usize) * 4;
            for c in 0..4 {
                out[at + c] = placed[(y * pw + x) * 4 + c].round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Encodes `frames` DMA-BUF captures of `source` size (GBM, `modifier`,
/// written by GLES / CUDA, no CPU importer: a readback would fail the
/// encode) into a `canvas` H.264 session, decodes them in software and
/// compares each with the picture the CPU path would have encoded
/// (turned by `orientation` on the CPU, then letterboxed).
fn gpu_fit_round_trip(
    name: &str,
    source: (u32, u32),
    canvas: (u32, u32),
    modifier: impl Fn(&DmaBufImporter) -> Modifier,
    orientation: Orientation,
) {
    let Some(gpu) = dmabuf::Gpu::get() else {
        return;
    };
    let importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
    let modifier = modifier(&importer);
    let config = EncoderConfig {
        width: canvas.0,
        height: canvas.1,
        ..nvenc(Codec::H264, PixelFormat::Bgra)
    };
    let Some(mut encoder) = encoder::<ferricast_encoder::h264::H264Encoder>(name, &config) else {
        return;
    };
    let pool: Vec<dmabuf::Bo> = (0..3)
        .map_while(|_| gpu.allocate(source.0, source.1, DrmFormat::Argb8888, modifier))
        .collect();
    assert_eq!(pool.len(), 3, "{name}: GBM allocation");
    let frames = 10;
    let patterns: Vec<Vec<u8>> = (0..frames)
        .map(|index| bgra_pattern(source.0, source.1, index))
        .collect();
    let mut packets = Vec::new();
    for (index, pattern) in patterns.iter().enumerate() {
        let bo = &pool[index % pool.len()];
        gpu.write(bo, pattern);
        let frame = GpuFrame {
            width: source.0,
            height: source.1,
            stride: bo.stride,
            format: PixelFormat::Bgra,
            timestamp_us: index as u64 * 33_333,
            plane: DmaBufPlane {
                fd: DmaBufFd::dup(bo.fd.as_fd()).unwrap(),
                offset: bo.offset,
                stride: bo.stride,
                modifier: bo.modifier.0,
                size: bo.stride * source.1,
            },
            importer: None,
            lease: None,
            producer: Some(DrmDevice::from_dev(gpu.node.dev())),
            sync: FrameSync::Ready,
            orientation,
        };
        packets.push(
            encoder
                .encode(CapturedFrame::Gpu(frame))
                .unwrap_or_else(|error| panic!("{name}: frame {index}: {error}")),
        );
    }
    let decoded = decode::<ferricast_decoder::H264Decoder>(&packets, PixelFormat::Nv12, software());
    assert_eq!(decoded.len(), frames, "{name}: pictures decoded");
    let worst = decoded
        .iter()
        .zip(&patterns)
        .map(|(frame, pattern)| {
            assert_eq!((frame.width, frame.height), canvas, "{name}: size");
            let upright = RawFrame {
                width: source.0,
                height: source.1,
                stride: source.0 * 4,
                format: PixelFormat::Bgra,
                data: pattern.clone().into(),
                timestamp_us: 0,
                colour: None,
                hdr: None,
            }
            .oriented(orientation)
            .unwrap();
            psnr_rgb(
                &to_bgra(frame),
                &letterbox_reference(&upright.data, (upright.width, upright.height), canvas),
            )
        })
        .fold(f64::INFINITY, f64::min);
    eprintln!("{name}: worst PSNR {worst:.1} dB against the CPU path's picture");
    assert!(worst > 35.0, "{name}: PSNR {worst:.1} dB");
}

#[test]
fn nvenc_scales_tiled_dma_bufs_on_the_gpu() {
    let _gpu = one_at_a_time();
    gpu_fit_round_trip(
        "GBM 1920x1080 tiled -> 1280x720",
        (1920, 1080),
        (1280, 720),
        dmabuf::Gpu::tiled_modifier,
        Orientation::IDENTITY,
    );
}

#[test]
fn nvenc_letterboxes_portrait_linear_dma_bufs_on_the_gpu() {
    let _gpu = one_at_a_time();
    gpu_fit_round_trip(
        "GBM 1080x1920 LINEAR -> 1920x1080 letterboxed",
        (1080, 1920),
        (1920, 1080),
        |_| Modifier::LINEAR,
        Orientation::IDENTITY,
    );
}

/// A capture of an output rotated a quarter turn, upright on the GPU into a
/// portrait session.
#[test]
fn nvenc_turns_rotated_dma_bufs_on_the_gpu() {
    let _gpu = one_at_a_time();
    gpu_fit_round_trip(
        "GBM 1920x1080 tiled turned 90 -> 1080x1920",
        (1920, 1080),
        (1080, 1920),
        dmabuf::Gpu::tiled_modifier,
        Orientation {
            mirrored: false,
            rotation: Rotation::Clockwise90,
        },
    );
}

/// A flipped 270° capture (with screencopy's y-invert composed in),
/// upright and letterboxed into a landscape session.
#[test]
fn nvenc_turns_and_letterboxes_flipped_dma_bufs_on_the_gpu() {
    let _gpu = one_at_a_time();
    gpu_fit_round_trip(
        "GBM 1920x1080 tiled flipped 270 + y-invert -> 1920x1080 letterboxed",
        (1920, 1080),
        (1920, 1080),
        dmabuf::Gpu::tiled_modifier,
        Orientation::FLIP_VERTICAL.then(Orientation {
            mirrored: true,
            rotation: Rotation::Clockwise270,
        }),
    );
}

/// A Main10 stream of the P010 pattern (NVENC, from host memory) decoded by
/// NVDEC into CUDA memory: the sources and the P010 pictures.
#[cfg(all(feature = "nvdec", feature = "nvenc-hevc"))]
fn nvdec_p010_pictures(name: &str) -> Option<(Vec<Vec<u8>>, Vec<CapturedFrame>)> {
    use ferricast_core::BitDepth;

    let config = EncoderConfig {
        bit_depth: BitDepth::Ten,
        colour: ColourDescription::BT709_LIMITED,
        ..nvenc(Codec::H265, PixelFormat::P010)
    };
    let mut main10 = encoder::<ferricast_encoder::h265::H265Encoder>(name, &config)?;
    let sources: Vec<Vec<u8>> = (0..FRAMES)
        .map(|index| p010_pattern(WIDTH, HEIGHT, index))
        .collect();
    let stream: Vec<EncodedFrame> = sources
        .iter()
        .enumerate()
        .map(|(index, data)| {
            main10
                .encode(CapturedFrame::Cpu(RawFrame {
                    width: WIDTH,
                    height: HEIGHT,
                    stride: WIDTH * 2,
                    format: PixelFormat::P010,
                    data: data.clone().into(),
                    timestamp_us: index as u64 * 33_333,
                    colour: Some(ColourDescription::BT709_LIMITED),
                    hdr: None,
                }))
                .unwrap()
        })
        .collect();
    drop(main10);
    let mut decoder = ferricast_decoder::H265Decoder::default();
    decoder
        .configure(&DecoderConfig {
            codec: Codec::H265,
            width: WIDTH,
            height: HEIGHT,
            pixel_format: PixelFormat::P010,
            low_latency: true,
            colour: None,
            hdr: None,
            memory: FrameMemory::Native,
            backend: BackendSelection::Fixed(BackendId::Nvdec),
        })
        .unwrap();
    let mut pictures = Vec::new();
    for packet in &stream {
        pictures.extend(decoder.decode(packet.clone()).unwrap());
    }
    pictures.extend(decoder.flush().unwrap());
    assert_eq!(pictures.len(), FRAMES, "{name}: NVDEC pictures");
    Some((sources, pictures))
}

/// NVDEC P010 pictures (a Main10 stream) into an 8-bit H.264 session, at
/// the same size (narrowing only) and at half size (scale + narrow): none
/// is read back, and the result decodes to the source's top bits.
#[cfg(all(feature = "nvdec", feature = "nvenc-hevc"))]
#[test]
fn nvdec_p010_encodes_on_eight_bit_nvenc_without_readback() {
    let _gpu = one_at_a_time();
    use ferricast_codec::cuda::CudaSurface;

    let name = "NVDEC P010 -> 8-bit NVENC";
    let Some((sources, pictures)) = nvdec_p010_pictures(name) else {
        return;
    };
    for canvas in [(WIDTH, HEIGHT), (WIDTH / 2, HEIGHT / 2)] {
        let config = EncoderConfig {
            width: canvas.0,
            height: canvas.1,
            ..nvenc(Codec::H264, PixelFormat::Nv12)
        };
        let mut encoder = encoder::<ferricast_encoder::h264::H264Encoder>(name, &config).unwrap();
        let readbacks = CudaSurface::readbacks();
        let packets: Vec<EncodedFrame> = pictures
            .iter()
            .map(|picture| {
                let CapturedFrame::Native(native) = picture else {
                    panic!("{name}: NVDEC handed out {picture:?}");
                };
                assert_eq!(native.format(), PixelFormat::P010, "{name}");
                encoder.encode(picture.clone()).unwrap()
            })
            .collect();
        assert_eq!(
            CudaSurface::readbacks(),
            readbacks,
            "{name}: a device picture was read back to host memory"
        );
        let round_trip =
            decode::<ferricast_decoder::H264Decoder>(&packets, PixelFormat::Nv12, software());
        assert_eq!(round_trip.len(), FRAMES, "{name}: pictures");
        let worst = round_trip
            .iter()
            .zip(&sources)
            .map(|(frame, source)| {
                assert_eq!((frame.width, frame.height), canvas, "{name}: size");
                // The source luma narrowed with rounding, resized like the
                // kernels do.
                let luma: Vec<f64> = source[..(WIDTH * HEIGHT * 2) as usize]
                    .chunks_exact(2)
                    .map(|w| f64::from(u16::from_le_bytes([w[0], w[1]])) / 256.0)
                    .collect();
                let want = resize_reference(
                    &luma,
                    1,
                    (WIDTH as usize, HEIGHT as usize),
                    (canvas.0 as usize, canvas.1 as usize),
                );
                let got = frame
                    .data
                    .chunks(frame.stride as usize)
                    .take(canvas.1 as usize)
                    .flat_map(|row| row[..canvas.0 as usize].iter().map(|&s| f64::from(s)));
                psnr(got, want.into_iter(), 255.0)
            })
            .fold(f64::INFINITY, f64::min);
        eprintln!(
            "{name} {}x{}: worst luma PSNR {worst:.1} dB, {} readbacks",
            canvas.0,
            canvas.1,
            CudaSurface::readbacks() - readbacks
        );
        assert!(worst > 38.0, "{name}: PSNR {worst:.1} dB");
    }
}

/// Encode rate and CPU time of 1080p DMA-BUF captures into a 720p session,
/// through the GPU fit and through the CPU path (readback stand-in +
/// `FrameScaler` + upload). Ignored by default (timing):
///
/// ```text
/// cargo test --release --test nvidia_e2e --features nvenc -- --ignored --nocapture fit_throughput
/// ```
#[test]
#[ignore = "timing; needs an NVIDIA GPU with GBM / EGL"]
fn gpu_fit_throughput() {
    let _gpu = one_at_a_time();
    let Some(gpu) = dmabuf::Gpu::get() else {
        return;
    };
    let importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
    let (source, canvas) = ((1920_u32, 1080_u32), (1280_u32, 720_u32));
    let pattern = bgra_pattern(source.0, source.1, 0);
    let bo = gpu
        .allocate(
            source.0,
            source.1,
            DrmFormat::Argb8888,
            dmabuf::Gpu::tiled_modifier(&importer),
        )
        .unwrap();
    gpu.write(&bo, &pattern);
    let memfd = Arc::new(memfd());
    let readbacks = Arc::new(AtomicUsize::new(0));
    for gpu_path in [true, false] {
        let config = EncoderConfig {
            width: canvas.0,
            height: canvas.1,
            fps: 60,
            ..nvenc(Codec::H264, PixelFormat::Bgra)
        };
        let mut encoder =
            encoder::<ferricast_encoder::h264::H264Encoder>("throughput", &config).unwrap();
        let frame = |index: usize| {
            let (fd, stride, modifier) = if gpu_path {
                (
                    DmaBufFd::dup(bo.fd.as_fd()).unwrap(),
                    bo.stride,
                    bo.modifier.0,
                )
            } else {
                // Not importable: the CPU path, with an instant readback.
                (DmaBufFd::dup(memfd.as_fd()).unwrap(), source.0 * 4, 0)
            };
            GpuFrame {
                width: source.0,
                height: source.1,
                stride,
                format: PixelFormat::Bgra,
                timestamp_us: index as u64 * 16_666,
                plane: DmaBufPlane {
                    fd,
                    offset: 0,
                    stride,
                    modifier,
                    size: stride * source.1,
                },
                importer: (!gpu_path)
                    .then(|| Arc::new(Readback(pattern.clone(), Arc::clone(&readbacks))) as _),
                lease: None,
                producer: gpu_path.then(|| DrmDevice::from_dev(gpu.node.dev())),
                sync: FrameSync::Ready,
                orientation: Orientation::IDENTITY,
            }
        };
        for index in 0..10 {
            encoder.encode(CapturedFrame::Gpu(frame(index))).unwrap();
        }
        let count = 300;
        let cpu_before = cpu_time();
        let start = std::time::Instant::now();
        for index in 0..count {
            encoder.encode(CapturedFrame::Gpu(frame(index))).unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64();
        let cpu = cpu_time() - cpu_before;
        eprintln!(
            "{} 1920x1080 -> 1280x720: {:.0} fps, {:.2} ms/frame, CPU {:.0}% of a core ({:.2} ms/frame)",
            if gpu_path { "GPU fit" } else { "CPU path" },
            count as f64 / elapsed,
            elapsed * 1e3 / count as f64,
            cpu / elapsed * 100.0,
            cpu * 1e3 / count as f64,
        );
    }
}

/// User + system CPU time of the process, in seconds.
fn cpu_time() -> f64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: valid out-pointer; `getrusage` fills it on success.
    let ok = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0;
    assert!(ok, "getrusage");
    // SAFETY: initialised by the successful call above.
    let usage = unsafe { usage.assume_init() };
    let seconds = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
    seconds(usage.ru_utime) + seconds(usage.ru_stime)
}

/// Every test here runs alone. Each opens NVDEC / NVENC sessions, the GPU
/// allows about a dozen NVENC sessions across all processes, and driver
/// 595 can crash the process when a session is destroyed while another
/// thread works with NVENC or NVDEC (see `design/NVCODEC.md`).
fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
