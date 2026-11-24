//! Real 10-bit streams decode to the exact P010 pictures a reference
//! decoder produces, and carry their HDR signalling.
//!
//! The fixtures are 128x72, 30 frames, encoded by FFmpeg from its
//! `testsrc2` source: HEVC Main10 (x265, with HDR10 mastering display
//! and content light level SEI), AV1 Main 10-bit (SVT-AV1) and VP9
//! profile 2 (libvpx). Each `*.p010.fnv` lists the FNV-1a 64 hash of
//! every picture FFmpeg decodes from it, as packed P010, in
//! presentation order. Hardware backends skip when the machine has no
//! device for them.
#![allow(
    dead_code,
    reason = "helpers for tests that each need a decoder feature; a build without them runs none"
)]

use std::path::PathBuf;

use ferricast_core::{
    BackendSelection, CapturedFrame, CodecError, ColourDescription, DecoderConfig, EncodedFrame,
    FrameMemory, HdrMetadata, PixelFormat, RawFrame, TransferCharacteristics, VideoDecoder,
};
use ferricast_demuxer::Demuxer;

const WIDTH: u32 = 128;
const HEIGHT: u32 = 72;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

struct Clip {
    packets: Vec<EncodedFrame>,
    colour: Option<ColourDescription>,
    hdr: Option<HdrMetadata>,
    expected: Vec<String>,
}

fn clip(name: &str) -> Clip {
    let mut demuxer = Demuxer::open(&fixture(name)).unwrap();
    let stream = demuxer.video().unwrap();
    let (colour, hdr) = (stream.colour, stream.hdr);
    let packets = std::iter::from_fn(|| demuxer.next_frame().unwrap()).collect();
    let stem = name.split('.').next().unwrap();
    let expected = std::fs::read_to_string(fixture(&format!("{stem}.p010.fnv")))
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    Clip {
        packets,
        colour,
        hdr,
        expected,
    }
}

fn config<D: VideoDecoder>(clip: &Clip) -> DecoderConfig {
    DecoderConfig {
        codec: D::CODEC,
        width: WIDTH,
        height: HEIGHT,
        pixel_format: PixelFormat::P010,
        low_latency: false,
        colour: clip.colour,
        hdr: clip.hdr,
        memory: FrameMemory::Cpu,
        backend: BackendSelection::default(),
    }
}

/// FNV-1a 64 of `frame` as packed P010 (rows without padding, the
/// interleaved chroma right after the luma).
fn packed_hash(frame: &RawFrame) -> String {
    assert_eq!(frame.format, PixelFormat::P010, "picture format");
    assert_eq!((frame.width, frame.height), (WIDTH, HEIGHT), "picture size");
    let (row, stride) = (WIDTH as usize * 2, frame.stride as usize);
    let rows = HEIGHT as usize + HEIGHT.div_ceil(2) as usize;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for index in 0..rows {
        for &byte in &frame.data[index * stride..index * stride + row] {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

/// Every picture `decoder` produces from `clip`, flush included.
fn decode_all<D: VideoDecoder>(mut decoder: D, clip: &Clip) -> Vec<RawFrame> {
    let mut frames: Vec<CapturedFrame> = clip
        .packets
        .iter()
        .filter_map(|packet| decoder.decode(packet.clone()).unwrap())
        .collect();
    frames.extend(decoder.flush().unwrap());
    frames.into_iter().map(|f| f.into_cpu().unwrap()).collect()
}

fn assert_matches_reference(name: &str, frames: &[RawFrame], clip: &Clip) {
    let hashes: Vec<String> = frames.iter().map(packed_hash).collect();
    assert_eq!(hashes.len(), clip.expected.len(), "{name}: picture count");
    for (index, (got, want)) in hashes.iter().zip(&clip.expected).enumerate() {
        assert_eq!(
            got, want,
            "{name}: picture {index} differs from the reference"
        );
    }
}

fn assert_hdr10(name: &str, frame: &RawFrame) {
    let colour = frame.colour.unwrap_or_else(|| panic!("{name}: no colour"));
    assert_eq!(colour.transfer, TransferCharacteristics::Pq, "{name}");
    let hdr = frame
        .hdr
        .unwrap_or_else(|| panic!("{name}: no HDR metadata"));
    let mastering = hdr.mastering_display.expect("mastering display");
    assert_eq!(mastering.max_luminance, 10_000_000, "{name}");
    assert_eq!(mastering.min_luminance, 1, "{name}");
    let light = hdr.content_light.expect("content light level");
    assert_eq!((light.max_cll, light.max_fall), (1000, 400), "{name}");
}

/// Opens `D` for `clip`, or `None` (and a note) when this machine has
/// no device for it.
fn open<D: VideoDecoder + Default>(name: &str, clip: &Clip) -> Option<D> {
    open_with(name, clip, D::default())
}

fn open_with<D: VideoDecoder>(name: &str, clip: &Clip, mut decoder: D) -> Option<D> {
    match decoder.configure(&config::<D>(clip)) {
        Ok(()) => Some(decoder),
        Err(error @ CodecError::BackendUnavailable { .. }) => {
            eprintln!("{name}: skipped, {error}");
            None
        }
        Err(error) => panic!("{name}: {error}"),
    }
}

fn check_hevc<D: VideoDecoder>(name: &str, make: impl FnOnce(&DecoderConfig) -> D) {
    let clip = clip("hevc_main10.mp4");
    let Some(decoder) = open_with(name, &clip, make(&config::<D>(&clip))) else {
        return;
    };
    let frames = decode_all(decoder, &clip);
    assert_matches_reference(name, &frames, &clip);
    assert_hdr10(name, &frames[0]);
}

#[cfg(feature = "hevc-decode")]
#[test]
fn software_hevc_decodes_main10_bit_exact() {
    check_hevc("software HEVC", |_| {
        ferricast_decoder::h265::SoftwareH265Decoder::default()
    });
}

#[cfg(feature = "nvdec-hevc-decode")]
#[test]
fn nvdec_hevc_decodes_main10_bit_exact() {
    check_hevc("NVDEC HEVC", |_| {
        ferricast_decoder::h265::NvdecH265Decoder::default()
    });
}

#[cfg(feature = "vaapi-hevc-decode")]
#[test]
fn vaapi_hevc_decodes_main10_bit_exact() {
    check_hevc("VA-API HEVC", |config| {
        ferricast_decoder::h265::VaapiH265Decoder::probe_with(config).unwrap()
    });
}

#[cfg(feature = "av1-decode")]
#[test]
fn av1_decodes_ten_bit_bit_exact() {
    let clip = clip("av1_10bit.mp4");
    let decoder = open::<ferricast_decoder::Av1Decoder>("AV1", &clip).unwrap();
    assert_matches_reference("AV1", &decode_all(decoder, &clip), &clip);
}

#[cfg(feature = "vp9-decode")]
#[test]
fn vp9_decodes_profile_2_bit_exact() {
    let clip = clip("vp9_profile2.webm");
    let decoder = open::<ferricast_decoder::Vp9Decoder>("VP9", &clip).unwrap();
    assert_matches_reference("VP9", &decode_all(decoder, &clip), &clip);
}
