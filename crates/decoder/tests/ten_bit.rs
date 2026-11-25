//! Real 10-bit streams decode to the exact P010 pictures a reference
//! decoder produces, and keep their colour and HDR signalling.
//!
//! The fixtures are 256x144 (NVDEC decodes no smaller HEVC), 30 frames,
//! encoded by FFmpeg from its `testsrc2` source, BT.2020 PQ:
//! - HEVC Main10 (x265, B-frames), with HDR10 mastering display and
//!   content light level SEI and no MP4 `colr` box: the colour is only
//!   in the SPS VUI;
//! - AV1 Main 10-bit (SVT-AV1), with HDR metadata OBUs;
//! - VP9 profile 2 (libvpx) in WebM, whose transfer only the Matroska
//!   `Colour` element carries.
//!
//! Each `*.p010.fnv` lists the FNV-1a 64 hash of every picture FFmpeg
//! decodes from its clip, as packed P010, in presentation order.
//! Each clip goes through the facade, each software backend, and NVDEC;
//! hardware backends skip when the machine has no device for them.
#![allow(
    dead_code,
    reason = "helpers for tests that each need a decoder feature; a build without them runs none"
)]

use std::path::PathBuf;

use ferricast_core::{
    BackendSelection, CapturedFrame, CodecError, ColourDescription, ColourPrimaries, DecoderConfig,
    EncodedFrame, FrameMemory, HdrMetadata, PixelFormat, RawFrame, TransferCharacteristics,
    VideoDecoder,
};
use ferricast_demuxer::Demuxer;

const WIDTH: u32 = 256;
const HEIGHT: u32 = 144;

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

impl Clip {
    fn open(name: &str) -> Self {
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
        Self {
            packets,
            colour,
            hdr,
            expected,
        }
    }

    /// The clip as an Annex B transport (MPEG-TS, RTP) delivers it:
    /// whatever the decoder knows of its colour and HDR metadata, it
    /// reads from the bitstream.
    fn without_container_signalling(self) -> Self {
        Self {
            colour: None,
            hdr: None,
            ..self
        }
    }

    fn config<D: VideoDecoder>(&self) -> DecoderConfig {
        DecoderConfig {
            codec: D::CODEC,
            width: WIDTH,
            height: HEIGHT,
            pixel_format: PixelFormat::P010,
            low_latency: false,
            colour: self.colour,
            hdr: self.hdr,
            memory: FrameMemory::Cpu,
            backend: BackendSelection::default(),
        }
    }

    /// Every picture `decoder` produces, flush included.
    fn decode<D: VideoDecoder>(&self, mut decoder: D) -> Vec<RawFrame> {
        let mut frames: Vec<CapturedFrame> = self
            .packets
            .iter()
            .filter_map(|packet| decoder.decode(packet.clone()).unwrap())
            .collect();
        frames.extend(decoder.flush().unwrap());
        let mut presentation: Vec<u64> = self.packets.iter().map(|p| p.timestamp_us).collect();
        presentation.sort_unstable();
        let stamps: Vec<u64> = frames.iter().map(CapturedFrame::timestamp_us).collect();
        assert_eq!(stamps, presentation, "pictures keep their packets' times");
        frames.into_iter().map(|f| f.into_cpu().unwrap()).collect()
    }

    fn assert_matches_reference(&self, name: &str, frames: &[RawFrame]) {
        let hashes: Vec<String> = frames.iter().map(packed_hash).collect();
        assert_eq!(hashes.len(), self.expected.len(), "{name}: picture count");
        for (index, (got, want)) in hashes.iter().zip(&self.expected).enumerate() {
            assert_eq!(
                got, want,
                "{name}: picture {index} differs from the reference"
            );
        }
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

fn assert_pq(name: &str, frame: &RawFrame) {
    let colour = frame.colour.unwrap_or_else(|| panic!("{name}: no colour"));
    assert_eq!(colour.primaries, ColourPrimaries::Bt2020, "{name}");
    assert_eq!(colour.transfer, TransferCharacteristics::Pq, "{name}");
}

/// The mastering display peak (1000 cd/m²) and black level
/// (0.0001 cd/m²), and the content light levels, of every HDR fixture.
fn assert_hdr10(name: &str, frame: &RawFrame) {
    assert_pq(name, frame);
    let hdr = frame
        .hdr
        .unwrap_or_else(|| panic!("{name}: no HDR metadata"));
    let mastering = hdr.mastering_display.expect("mastering display");
    assert_eq!(mastering.max_luminance, 10_000_000, "{name}");
    assert_eq!(mastering.min_luminance, 1, "{name}");
    let light = hdr.content_light.expect("content light level");
    assert_eq!((light.max_cll, light.max_fall), (1000, 400), "{name}");
}

/// `decoder` configured for `clip`, or `None` (and a note) when this
/// machine has no device for it.
fn configured<D: VideoDecoder>(name: &str, clip: &Clip, mut decoder: D) -> Option<D> {
    match decoder.configure(&clip.config::<D>()) {
        Ok(()) => Some(decoder),
        Err(error @ CodecError::BackendUnavailable { .. }) => {
            eprintln!("{name}: skipped, {error}");
            None
        }
        Err(error) => panic!("{name}: {error}"),
    }
}

fn check_hevc<D: VideoDecoder>(name: &str, decoder: D) {
    let clip = Clip::open("hevc_main10.mp4").without_container_signalling();
    let Some(decoder) = configured(name, &clip, decoder) else {
        return;
    };
    let frames = clip.decode(decoder);
    clip.assert_matches_reference(name, &frames);
    for frame in &frames {
        assert_hdr10(name, frame);
    }
}

#[cfg(feature = "hevc-decode")]
#[test]
fn software_hevc_decodes_main10_bit_exact() {
    check_hevc(
        "software HEVC",
        ferricast_decoder::h265::SoftwareH265Decoder::default(),
    );
}

#[cfg(feature = "nvdec-hevc-decode")]
#[test]
fn nvdec_hevc_decodes_main10_bit_exact() {
    check_hevc(
        "NVDEC HEVC",
        ferricast_decoder::h265::NvdecH265Decoder::default(),
    );
}

#[cfg(any(feature = "hevc-decode", feature = "nvdec-hevc-decode"))]
#[test]
fn automatic_hevc_selection_keeps_main10() {
    check_hevc("H.265 facade", ferricast_decoder::H265Decoder::default());
}

fn check_av1<D: VideoDecoder>(name: &str, decoder: D) {
    let clip = Clip::open("av1_10bit.mp4").without_container_signalling();
    let Some(decoder) = configured(name, &clip, decoder) else {
        return;
    };
    let frames = clip.decode(decoder);
    clip.assert_matches_reference(name, &frames);
    for frame in &frames {
        assert_hdr10(name, frame);
    }
}

fn check_vp9<D: VideoDecoder>(name: &str, decoder: D) {
    let clip = Clip::open("vp9_profile2.webm");
    let Some(decoder) = configured(name, &clip, decoder) else {
        return;
    };
    let frames = clip.decode(decoder);
    clip.assert_matches_reference(name, &frames);
    for frame in &frames {
        assert_pq(name, frame);
    }
}

#[cfg(feature = "av1-decode")]
#[test]
fn av1_decodes_ten_bit_bit_exact() {
    check_av1("AV1 facade", ferricast_decoder::Av1Decoder::default());
}

#[cfg(feature = "av1-decode")]
#[test]
fn rav1d_decodes_ten_bit_bit_exact() {
    let software = ferricast_core::AutoSelect::default()
        .with_acceleration(ferricast_core::AccelerationPolicy::SoftwareOnly);
    check_av1(
        "rav1d",
        Software(ferricast_decoder::Av1Decoder::default(), software),
    );
}

#[cfg(feature = "nvdec-decode")]
#[test]
fn nvdec_av1_decodes_ten_bit_bit_exact() {
    check_av1(
        "NVDEC AV1",
        ferricast_decoder::nvdec::NvdecAv1Decoder::default(),
    );
}

#[cfg(feature = "vp9-decode")]
#[test]
fn vp9_decodes_profile_2_bit_exact() {
    check_vp9("VP9 facade", ferricast_decoder::Vp9Decoder::default());
}

#[cfg(feature = "vp9-decode")]
#[test]
fn rusty_vp9_decodes_profile_2_bit_exact() {
    let software = ferricast_core::AutoSelect::default()
        .with_acceleration(ferricast_core::AccelerationPolicy::SoftwareOnly);
    check_vp9(
        "rusty_vp9",
        Software(ferricast_decoder::Vp9Decoder::default(), software),
    );
}

#[cfg(feature = "nvdec-decode")]
#[test]
fn nvdec_vp9_decodes_profile_2_bit_exact() {
    check_vp9(
        "NVDEC VP9",
        ferricast_decoder::nvdec::NvdecVp9Decoder::default(),
    );
}

/// NVDEC pictures kept in CUDA memory read back to the same P010 the host
/// path hands out.
#[cfg(feature = "nvdec-hevc-decode")]
#[test]
fn nvdec_device_frames_read_back_bit_exact() {
    let clip = Clip::open("hevc_main10.mp4").without_container_signalling();
    let name = "NVDEC HEVC (device)";
    let mut decoder = ferricast_decoder::h265::NvdecH265Decoder::default();
    let config = DecoderConfig {
        memory: FrameMemory::Native,
        ..clip.config::<ferricast_decoder::h265::NvdecH265Decoder>()
    };
    match decoder.configure(&config) {
        Ok(()) => {}
        Err(error @ CodecError::BackendUnavailable { .. }) => {
            eprintln!("{name}: skipped, {error}");
            return;
        }
        Err(error) => panic!("{name}: {error}"),
    }
    let mut frames: Vec<CapturedFrame> = clip
        .packets
        .iter()
        .filter_map(|packet| decoder.decode(packet.clone()).unwrap())
        .collect();
    frames.extend(decoder.flush().unwrap());
    for frame in &frames {
        let CapturedFrame::Native(native) = frame else {
            panic!("{name}: a {frame:?} picture, not a native one");
        };
        assert_eq!(native.kind(), ferricast_core::SurfaceKind::CudaDevice);
        assert_eq!(native.format(), PixelFormat::P010);
    }
    let frames: Vec<RawFrame> = frames.into_iter().map(|f| f.into_cpu().unwrap()).collect();
    clip.assert_matches_reference(name, &frames);
    for frame in &frames {
        assert_hdr10(name, frame);
    }
}

/// A facade forced onto its software backends.
struct Software<D>(D, ferricast_core::AutoSelect);

impl<D: VideoDecoder> VideoDecoder for Software<D> {
    const CODEC: ferricast_core::Codec = D::CODEC;

    fn configure(&mut self, config: &DecoderConfig) -> ferricast_core::CodecResult<()> {
        self.0.configure(&DecoderConfig {
            backend: BackendSelection::Auto(self.1.clone()),
            ..config.clone()
        })
    }

    fn decode(
        &mut self,
        frame: EncodedFrame,
    ) -> ferricast_core::CodecResult<Option<CapturedFrame>> {
        self.0.decode(frame)
    }

    fn flush(&mut self) -> ferricast_core::CodecResult<Vec<CapturedFrame>> {
        self.0.flush()
    }
}
