//! Throughput of the NVIDIA backends through ferricast's facades, on a
//! clip named by `FERRICAST_BENCH_CLIP` (any container the demuxer
//! reads). Ignored by default: it needs an NVIDIA GPU and a 1080p clip.
//!
//! ```text
//! FERRICAST_BENCH_CLIP=clip.mp4 cargo test --release --test nvidia_throughput -- --ignored --nocapture
//! ```

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test patterns and rates over small, positive numbers"
)]

use std::time::Instant;

use ferricast_core::{
    BackendId, BackendSelection, CapturedFrame, Codec, DecoderConfig, EncoderConfig, FrameMemory,
    PixelFormat, RawFrame, VideoDecoder, VideoEncoder,
};
use ferricast_demuxer::Demuxer;

fn clip() -> Option<(Codec, Vec<ferricast_core::EncodedFrame>, u32, u32)> {
    let path = std::env::var_os("FERRICAST_BENCH_CLIP")?;
    let mut demuxer = Demuxer::open(std::path::Path::new(&path)).unwrap();
    let stream = demuxer.video().unwrap();
    let (codec, width, height) = (stream.codec, stream.width, stream.height);
    let packets = std::iter::from_fn(|| demuxer.next_frame().unwrap()).collect();
    Some((codec, packets, width, height))
}

fn decode_fps<D: VideoDecoder + Default>(memory: FrameMemory, backend: BackendSelection) {
    let Some((codec, packets, width, height)) = clip() else {
        eprintln!("FERRICAST_BENCH_CLIP is not set");
        return;
    };
    if codec != D::CODEC {
        return;
    }
    let mut decoder = D::default();
    decoder
        .configure(&DecoderConfig {
            gpu: ferricast_core::GpuPreference::Auto,
            codec,
            width,
            height,
            pixel_format: PixelFormat::Nv12,
            low_latency: true,
            colour: None,
            hdr: None,
            memory,
            backend,
        })
        .unwrap();
    let start = Instant::now();
    let mut pictures = 0_usize;
    for packet in &packets {
        pictures += usize::from(decoder.decode(packet.clone()).unwrap().is_some());
    }
    pictures += decoder.flush().unwrap().len();
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "{codec:?} {width}x{height} {memory:?}: {pictures} pictures in {elapsed:.3} s = {:.0} fps",
        pictures as f64 / elapsed
    );
}

#[test]
#[ignore = "needs an NVIDIA GPU and FERRICAST_BENCH_CLIP"]
fn nvdec_decode_throughput() {
    let nvdec = || BackendSelection::Fixed(BackendId::Nvdec);
    decode_fps::<ferricast_decoder::H264Decoder>(FrameMemory::Cpu, nvdec());
    decode_fps::<ferricast_decoder::H265Decoder>(FrameMemory::Cpu, nvdec());
    decode_fps::<ferricast_decoder::Vp9Decoder>(FrameMemory::Cpu, nvdec());
    decode_fps::<ferricast_decoder::Av1Decoder>(FrameMemory::Cpu, nvdec());
    decode_fps::<ferricast_decoder::H264Decoder>(FrameMemory::Native, nvdec());
    decode_fps::<ferricast_decoder::H265Decoder>(FrameMemory::Native, nvdec());
    decode_fps::<ferricast_decoder::Vp9Decoder>(FrameMemory::Native, nvdec());
    decode_fps::<ferricast_decoder::Av1Decoder>(FrameMemory::Native, nvdec());
}

#[test]
#[ignore = "needs an NVIDIA GPU and FERRICAST_BENCH_CLIP"]
fn software_decode_throughput() {
    let software = || {
        BackendSelection::Auto(
            ferricast_core::AutoSelect::default()
                .with_acceleration(ferricast_core::AccelerationPolicy::SoftwareOnly),
        )
    };
    decode_fps::<ferricast_decoder::H265Decoder>(FrameMemory::Cpu, software());
    decode_fps::<ferricast_decoder::Vp9Decoder>(FrameMemory::Cpu, software());
    decode_fps::<ferricast_decoder::Av1Decoder>(FrameMemory::Cpu, software());
}

fn encode_fps<E: VideoEncoder + Default>(codec: Codec, format: PixelFormat) {
    let (width, height) = (1920_u32, 1080_u32);
    let mut encoder = E::default();
    let configured = encoder.configure(&EncoderConfig {
        codec,
        width,
        height,
        bitrate_kbps: 8000,
        fps: 60,
        pixel_format: format,
        backend: BackendSelection::Fixed(BackendId::Nvenc),
        ..EncoderConfig::default()
    });
    if let Err(error) = configured {
        println!("{codec:?} encode {format:?}: {error}");
        return;
    }
    let stride = format.packed_row_bytes(width);
    let rows = match format {
        PixelFormat::Bgra | PixelFormat::Rgba => height,
        _ => height + height / 2,
    };
    let frames: Vec<CapturedFrame> = (0..4_u8)
        .map(|seed| {
            let data: Vec<u8> = (0..stride * rows)
                .map(|index| (index as u8).wrapping_mul(seed.wrapping_add(3)))
                .collect();
            CapturedFrame::Cpu(RawFrame {
                width,
                height,
                stride,
                format,
                data: data.into(),
                timestamp_us: 0,
                colour: None,
                hdr: None,
            })
        })
        .collect();
    let count = 600_usize;
    let start = Instant::now();
    for index in 0..count {
        encoder
            .encode(frames[index % frames.len()].clone())
            .unwrap();
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "{codec:?} encode {width}x{height} {format:?}: {count} frames in {elapsed:.3} s = {:.0} fps",
        count as f64 / elapsed
    );
}

#[test]
#[ignore = "needs an NVIDIA GPU"]
fn nvenc_encode_throughput() {
    encode_fps::<ferricast_encoder::h264::H264Encoder>(Codec::H264, PixelFormat::Bgra);
    encode_fps::<ferricast_encoder::h264::H264Encoder>(Codec::H264, PixelFormat::Nv12);
    encode_fps::<ferricast_encoder::h265::H265Encoder>(Codec::H265, PixelFormat::Bgra);
}
