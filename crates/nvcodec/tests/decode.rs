//! NVDEC against the installed driver, on the clips in `tests/fixtures`
//! (made by `scripts/decode_fixtures.sh`).
//!
//! Every decoded picture is compared bit for bit with FFmpeg's decode of the
//! same clip: each `<clip>.fnv` lists the FNV-1a 64 hash of every picture
//! FFmpeg outputs, packed as its `rawvideo` muxer writes it (`nv12`,
//! `p010le`, `yuv444p`), in display order. Tests print why and return early
//! without an NVIDIA driver, or when the GPU lacks a codec.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use ferricast_nvcodec::Error;
use ferricast_nvcodec::cuda::{Context, Device, DeviceBuffer, Point, Region, Stream};
use ferricast_nvcodec::decode::{
    ChromaFormat, Codec, ContentLightLevel, DecodeStatus, Decoder, DecoderCaps, DecoderConfig,
    Device as OnDevice, FrameInfo, Host, MasteringDisplay, Packet, PixelFormat,
};
use ferricast_nvcodec::loader::{CudaLibrary, CuvidLibrary};

/// The primary context of device 0, or `None` (with the reason printed)
/// without a usable NVIDIA driver.
fn context() -> Option<Arc<Context>> {
    let opened = CudaLibrary::get()
        .and_then(|_| CuvidLibrary::get())
        .and_then(|_| Device::get(0))
        .and_then(Context::primary);
    match opened {
        Ok(context) => Some(context),
        Err(error) => {
            eprintln!("skipping: no usable NVIDIA driver: {error}");
            None
        }
    }
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A test clip: its file, codec, and the pictures FFmpeg decodes from it.
struct Clip {
    file: &'static str,
    codec: Codec,
    /// FFmpeg's name for the packed format NVDEC output equals.
    pix_fmt: &'static str,
    /// Display width, height and picture count of each sequence.
    sequences: &'static [(u32, u32, usize)],
}

const H264_BFRAMES: Clip = Clip {
    file: "h264_bframes.h264",
    codec: Codec::H264,
    pix_fmt: "nv12",
    sequences: &[(320, 240, 30)],
};
const H264_CROP: Clip = Clip {
    file: "h264_crop.h264",
    codec: Codec::H264,
    pix_fmt: "nv12",
    sequences: &[(392, 294, 10)],
};
const RES_CHANGE: Clip = Clip {
    file: "res_change.h264",
    codec: Codec::H264,
    pix_fmt: "nv12",
    sequences: &[(320, 240, 10), (176, 144, 10), (640, 360, 10)],
};
const HEVC_MAIN8: Clip = Clip {
    file: "hevc_main8.hevc",
    codec: Codec::Hevc,
    pix_fmt: "nv12",
    sequences: &[(256, 144, 30)],
};
const HEVC_MAIN10_HDR10: Clip = Clip {
    file: "hevc_main10_hdr10.hevc",
    codec: Codec::Hevc,
    pix_fmt: "p010le",
    sequences: &[(256, 144, 30)],
};
const HEVC_444: Clip = Clip {
    file: "hevc_444.hevc",
    codec: Codec::Hevc,
    pix_fmt: "yuv444p",
    sequences: &[(256, 144, 10)],
};
const VP8: Clip = Clip {
    file: "vp8.ivf",
    codec: Codec::Vp8,
    pix_fmt: "nv12",
    sequences: &[(256, 144, 30)],
};
const VP9_PROFILE0: Clip = Clip {
    file: "vp9_profile0.ivf",
    codec: Codec::Vp9,
    pix_fmt: "nv12",
    sequences: &[(256, 144, 30)],
};
const VP9_PROFILE2: Clip = Clip {
    file: "vp9_profile2.ivf",
    codec: Codec::Vp9,
    pix_fmt: "p010le",
    sequences: &[(256, 144, 30)],
};
const AV1_8BIT: Clip = Clip {
    file: "av1_8bit.ivf",
    codec: Codec::Av1,
    pix_fmt: "nv12",
    sequences: &[(256, 144, 30)],
};
const AV1_10BIT: Clip = Clip {
    file: "av1_10bit.ivf",
    codec: Codec::Av1,
    pix_fmt: "p010le",
    sequences: &[(256, 144, 30)],
};

const CLIPS: &[Clip] = &[
    H264_BFRAMES,
    H264_CROP,
    RES_CHANGE,
    HEVC_MAIN8,
    HEVC_MAIN10_HDR10,
    HEVC_444,
    VP8,
    VP9_PROFILE0,
    VP9_PROFILE2,
    AV1_8BIT,
    AV1_10BIT,
];

impl Clip {
    fn bit_depth(&self) -> u8 {
        if self.pix_fmt == "p010le" { 10 } else { 8 }
    }

    fn chroma(&self) -> ChromaFormat {
        if self.pix_fmt == "yuv444p" {
            ChromaFormat::Yuv444
        } else {
            ChromaFormat::Yuv420
        }
    }

    /// Bytes of one packed picture.
    fn picture_bytes(&self, width: usize, height: usize) -> usize {
        match self.pix_fmt {
            "yuv444p" => 3 * width * height,
            format => {
                let sample = if format == "p010le" { 2 } else { 1 };
                sample * (width * height + width.div_ceil(2) * 2 * height.div_ceil(2))
            }
        }
    }

    /// The packets in decode order, with their timestamps.
    fn packets(&self) -> Vec<(Vec<u8>, i64)> {
        let data = std::fs::read(fixture(self.file)).unwrap();
        if std::path::Path::new(self.file)
            .extension()
            .is_some_and(|extension| extension == "ivf")
        {
            return ivf_frames(&data);
        }
        let timestamps: Vec<i64> = std::fs::read_to_string(fixture(&format!("{}.pts", self.file)))
            .unwrap()
            .lines()
            .map(|line| line.trim().parse().unwrap())
            .collect();
        let units = access_units(&data, self.codec);
        assert_eq!(
            units.len(),
            timestamps.len(),
            "{}: access units vs .pts",
            self.file
        );
        units
            .into_iter()
            .map(<[u8]>::to_vec)
            .zip(timestamps)
            .collect()
    }

    /// FFmpeg's hashes, from `<clip>.fnv`.
    fn expected(&self) -> Vec<u64> {
        let path = fixture(&format!("{}.fnv", self.file));
        std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
            .lines()
            .map(|line| u64::from_str_radix(line.trim(), 16).unwrap())
            .collect()
    }

    /// Whether the GPU decodes this clip; prints why not.
    fn supported(&self, context: &Arc<Context>) -> bool {
        let caps =
            DecoderCaps::query(context, self.codec, self.chroma(), self.bit_depth()).unwrap();
        if !caps.supported {
            eprintln!(
                "skipping {}: the GPU does not decode {} {}-bit {:?}",
                self.file,
                self.codec,
                self.bit_depth(),
                self.chroma()
            );
        }
        caps.supported
    }
}

/// IVF: a 32-byte file header (its length at offset 6), then per frame a
/// 4-byte size, an 8-byte timestamp and the frame.
fn ivf_frames(data: &[u8]) -> Vec<(Vec<u8>, i64)> {
    assert_eq!(&data[..4], b"DKIF");
    let mut rest = &data[usize::from(u16::from_le_bytes([data[6], data[7]]))..];
    let mut frames = Vec::new();
    while rest.len() >= 12 {
        let size = u32::from_le_bytes(rest[..4].try_into().unwrap()) as usize;
        let timestamp = i64::from_le_bytes(rest[4..12].try_into().unwrap());
        frames.push((rest[12..12 + size].to_vec(), timestamp));
        rest = &rest[12 + size..];
    }
    frames
}

/// Splits Annex B into access units: a new one starts at the first
/// delimiter, parameter set or prefix SEI after a slice.
fn access_units(data: &[u8], codec: Codec) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut index = 0;
    while index + 3 <= data.len() {
        if data[index..index + 3] == [0, 0, 1] {
            let begin = if index > 0 && data[index - 1] == 0 {
                index - 1
            } else {
                index
            };
            starts.push((begin, data.get(index + 3).copied().unwrap_or(0)));
            index += 3;
        } else {
            index += 1;
        }
    }
    let mut units = Vec::new();
    let mut unit_start = 0;
    let mut seen_slice = false;
    for (begin, header) in starts {
        let (slice, starter) = if codec == Codec::H264 {
            let kind = header & 0x1f;
            ((1..=5).contains(&kind), matches!(kind, 6..=9))
        } else {
            let kind = (header >> 1) & 0x3f;
            (kind <= 31, matches!(kind, 32..=35 | 39))
        };
        if starter && seen_slice {
            units.push(&data[unit_start..begin]);
            unit_start = begin;
            seen_slice = false;
        }
        seen_slice |= slice;
    }
    units.push(&data[unit_start..]);
    units
}

/// What a decode run produced, per picture in display order.
#[derive(Debug, Default)]
struct Decoded {
    hashes: Vec<u64>,
    infos: Vec<FrameInfo>,
    mastering: Vec<Option<MasteringDisplay>>,
    light: Vec<Option<ContentLightLevel>>,
}

impl Decoded {
    fn timestamps(&self) -> Vec<i64> {
        self.infos.iter().map(|info| info.timestamp).collect()
    }
}

fn decode_host(context: &Arc<Context>, clip: &Clip, config: DecoderConfig) -> Decoded {
    let mut decoder = Decoder::<Host>::new(context, config).unwrap();
    let mut out = Decoded::default();
    let mut drain = |decoder: &mut Decoder<Host>| {
        while let Some(frame) = decoder.next_frame() {
            out.hashes.push(fnv1a64(frame.data()));
            out.infos.push(*frame.info());
            out.mastering.push(frame.sei().mastering_display());
            out.light.push(frame.sei().content_light_level());
        }
    };
    for (data, timestamp) in clip.packets() {
        decoder
            .decode(Packet::new(&data, timestamp))
            .unwrap_or_else(|error| panic!("{}: {error}", clip.file));
        drain(&mut decoder);
    }
    decoder.flush().unwrap();
    drain(&mut decoder);
    out
}

/// Decodes `clip` in host mode and checks every picture, its size and its
/// timestamp against FFmpeg and the input.
fn check_clip(clip: &Clip) -> Option<Decoded> {
    let context = context()?;
    if !clip.supported(&context) {
        return None;
    }
    let decoded = decode_host(&context, clip, DecoderConfig::new(clip.codec));
    let expected = clip.expected();
    assert_eq!(
        decoded.hashes.len(),
        expected.len(),
        "{}: picture count",
        clip.file
    );
    for (index, (got, want)) in decoded.hashes.iter().zip(&expected).enumerate() {
        assert_eq!(
            got, want,
            "{}: picture {index} differs from FFmpeg's ({:?})",
            clip.file, decoded.infos[index]
        );
    }
    let sizes: Vec<(u32, u32)> = clip
        .sequences
        .iter()
        .flat_map(|&(width, height, count)| std::iter::repeat_n((width, height), count))
        .collect();
    let got_sizes: Vec<(u32, u32)> = decoded
        .infos
        .iter()
        .map(|info| (info.width, info.height))
        .collect();
    assert_eq!(got_sizes, sizes, "{}: display sizes", clip.file);
    for info in &decoded.infos {
        assert_eq!(info.bit_depth, clip.bit_depth(), "{}", clip.file);
        assert_ne!(info.status, DecodeStatus::Corrupted, "{}", clip.file);
    }
    // Pictures come back in presentation order with the timestamps of the
    // packets that carried them.
    let mut input: Vec<i64> = clip.packets().into_iter().map(|(_, pts)| pts).collect();
    input.sort_unstable();
    assert_eq!(decoded.timestamps(), input, "{}: timestamps", clip.file);
    eprintln!(
        "{}: {} pictures match FFmpeg, first {:?}",
        clip.file,
        decoded.hashes.len(),
        decoded.infos.first()
    );
    Some(decoded)
}

#[test]
fn h264_with_b_frames_matches_ffmpeg_in_presentation_order() {
    let Some(_) = check_clip(&H264_BFRAMES) else {
        return;
    };
    let input: Vec<i64> = H264_BFRAMES
        .packets()
        .into_iter()
        .map(|(_, pts)| pts)
        .collect();
    assert!(
        input.windows(2).any(|pair| pair[1] < pair[0]),
        "the clip must reorder for this test to mean anything"
    );
}

#[test]
fn h264_crop_honours_every_edge_of_the_display_area() {
    let Some(decoded) = check_clip(&H264_CROP) else {
        return;
    };
    let info = decoded.infos[0];
    assert_eq!((info.coded_width, info.coded_height), (400, 304));
    assert_eq!(
        (
            info.display_area.left,
            info.display_area.top,
            info.display_area.right,
            info.display_area.bottom
        ),
        (8, 6, 400, 300)
    );
    assert_eq!(info.format, PixelFormat::Nv12);
}

#[test]
fn resolution_changes_mid_stream() {
    check_clip(&RES_CHANGE);
}

#[test]
fn hevc_main8_matches_ffmpeg() {
    check_clip(&HEVC_MAIN8);
}

#[test]
fn hevc_main10_hdr10_matches_ffmpeg_with_colour_and_sei() {
    let Some(decoded) = check_clip(&HEVC_MAIN10_HDR10) else {
        return;
    };
    let info = decoded.infos[0];
    assert_eq!(info.format, PixelFormat::P016);
    assert_eq!(
        (
            info.colour.primaries,
            info.colour.transfer,
            info.colour.matrix,
            info.colour.full_range
        ),
        (9, 16, 9, false),
        "BT.2020 primaries, PQ, BT.2020 non-constant luminance, limited range"
    );
    let mastering = decoded.mastering[0].expect("the IDR picture carries the mastering display");
    assert_eq!(
        mastering,
        MasteringDisplay {
            primaries: [(13250, 34500), (7500, 3000), (34000, 16000)],
            white_point: (15635, 16450),
            max_luminance: 10_000_000,
            min_luminance: 1,
        }
    );
    assert_eq!(
        decoded.light[0],
        Some(ContentLightLevel {
            max_content: 1000,
            max_frame_average: 400
        })
    );
    // x265 sends the HDR SEI with keyframes only: as many pictures carry it
    // as access units hold a prefix SEI NAL starting with payload type 137,
    // so it does not leak onto pictures that reuse a surface.
    let sent = HEVC_MAIN10_HDR10
        .packets()
        .iter()
        .filter(|(data, _)| data.windows(6).any(|w| w == [0, 0, 1, 0x4e, 0x01, 0x89]))
        .count();
    let with_sei = decoded.mastering.iter().filter(|sei| sei.is_some()).count();
    eprintln!("pictures with mastering display SEI: {with_sei} of {sent} sent");
    assert!(sent > 0);
    assert_eq!(with_sei, sent);
    for (mastering, light) in decoded
        .mastering
        .iter()
        .zip(&decoded.light)
        .filter(|(m, _)| m.is_some())
    {
        assert_eq!(*mastering, decoded.mastering[0]);
        assert_eq!(*light, decoded.light[0]);
    }
}

#[test]
fn hevc_444_matches_ffmpeg() {
    let Some(decoded) = check_clip(&HEVC_444) else {
        return;
    };
    assert_eq!(decoded.infos[0].format, PixelFormat::Yuv444);
}

#[test]
fn vp8_matches_ffmpeg() {
    check_clip(&VP8);
}

#[test]
fn vp9_profile0_matches_ffmpeg() {
    check_clip(&VP9_PROFILE0);
}

#[test]
fn vp9_profile2_matches_ffmpeg() {
    let Some(decoded) = check_clip(&VP9_PROFILE2) else {
        return;
    };
    assert_eq!(decoded.infos[0].format, PixelFormat::P016);
}

#[test]
fn av1_8bit_matches_ffmpeg() {
    check_clip(&AV1_8BIT);
}

#[test]
fn av1_10bit_matches_ffmpeg() {
    let Some(decoded) = check_clip(&AV1_10BIT) else {
        return;
    };
    let colour = decoded.infos[0].colour;
    eprintln!("AV1 10-bit colour: {colour:?}");
    assert_eq!(
        (colour.primaries, colour.transfer, colour.matrix),
        (9, 16, 9)
    );
}

#[test]
fn device_frames_equal_host_frames() {
    let Some(context) = context() else { return };
    let consumer = Stream::new(&context).unwrap();
    for clip in [
        &H264_CROP,
        &HEVC_MAIN10_HDR10,
        &HEVC_444,
        &VP9_PROFILE2,
        &RES_CHANGE,
    ] {
        if !clip.supported(&context) {
            continue;
        }
        let mut decoder =
            Decoder::<OnDevice>::new(&context, DecoderConfig::new(clip.codec)).unwrap();
        let mut hashes = Vec::new();
        let mut host = Vec::new();
        let mut drain = |decoder: &mut Decoder<OnDevice>| {
            while let Some(frame) = decoder.next_frame() {
                let layout = frame.download(&mut host).unwrap();
                assert_eq!(layout.total_bytes(), host.len());
                hashes.push(fnv1a64(&host));
                // The plane layout NVENC takes: chroma right after luma, at
                // the buffer's pitch.
                let planes = frame.layout().planes();
                assert_eq!(planes[1].offset, frame.buffer().pitch() * planes[0].rows);
                assert!(frame.buffer().pitch() >= planes[0].width_bytes);
                // An asynchronous consumer: a device copy on its own stream,
                // ordered after the decode copy and before buffer reuse.
                let mut copy = DeviceBuffer::pitched(
                    &context,
                    frame.buffer().width_bytes(),
                    frame.buffer().rows(),
                )
                .unwrap();
                frame.wait(&consumer).unwrap();
                copy.copy_from_async(
                    Point::default(),
                    frame.buffer(),
                    Region::new(frame.buffer().width_bytes(), frame.buffer().rows()),
                    &consumer,
                )
                .unwrap();
                frame.release_after(&consumer).unwrap();
            }
        };
        for (data, timestamp) in clip.packets() {
            decoder.decode(Packet::new(&data, timestamp)).unwrap();
            drain(&mut decoder);
        }
        decoder.flush().unwrap();
        drain(&mut decoder);
        consumer.synchronize().unwrap();
        assert_eq!(hashes, clip.expected(), "{}: device frames", clip.file);
        eprintln!("{}: {} device frames match", clip.file, hashes.len());
    }
}

/// `data` without its access unit delimiters, as MP4 and Matroska carry
/// H.264 / HEVC.
fn without_delimiters(data: &[u8], codec: Codec) -> Vec<u8> {
    let mut starts: Vec<usize> = (0..data.len().saturating_sub(2))
        .filter(|&at| data[at..at + 3] == [0, 0, 1])
        .collect();
    starts.push(data.len());
    let mut out = Vec::new();
    for pair in starts.windows(2) {
        let unit = &data[pair[0]..pair[1]];
        let header = unit.get(3).copied().unwrap_or(0);
        let delimiter = match codec {
            Codec::H264 => header & 0x1f == 9,
            _ => (header >> 1) & 0x3f == 35,
        };
        if !delimiter {
            out.extend_from_slice(unit);
        }
    }
    out
}

/// Hashes of the pictures decoded from `packets`, each with its
/// [`Packet::whole_picture`].
fn decode_framed(
    context: &Arc<Context>,
    codec: Codec,
    packets: &[(Vec<u8>, Option<bool>)],
) -> Vec<u64> {
    let mut decoder = Decoder::<Host>::new(context, DecoderConfig::new(codec)).unwrap();
    let mut hashes = Vec::new();
    for (index, (data, whole)) in packets.iter().enumerate() {
        let mut packet = Packet::new(data, i64::try_from(index).unwrap());
        if let Some(whole) = whole {
            packet = packet.with_whole_picture(*whole);
        }
        decoder.decode(packet).unwrap();
        while let Some(frame) = decoder.next_frame() {
            hashes.push(fnv1a64(frame.data()));
        }
    }
    decoder.flush().unwrap();
    while let Some(frame) = decoder.next_frame() {
        hashes.push(fnv1a64(frame.data()));
    }
    hashes
}

/// Packets that are not one picture each, sent without
/// `Packet::whole_picture`, and whole ones mixed with them, decode every
/// picture as FFmpeg does, also without access unit delimiters (H.264
/// needs one between an open picture and a whole packet: the decoder sends
/// it).
#[test]
fn packets_of_several_or_part_of_a_picture_decode_bit_exact() {
    let Some(context) = context() else {
        return;
    };
    for clip in [&H264_BFRAMES, &HEVC_MAIN8] {
        if !clip.supported(&context) {
            continue;
        }
        let expected = clip.expected();
        let data = std::fs::read(fixture(clip.file)).unwrap();
        for delimited in [true, false] {
            let units: Vec<Vec<u8>> = access_units(&data, clip.codec)
                .into_iter()
                .map(|unit| {
                    if delimited {
                        unit.to_vec()
                    } else {
                        without_delimiters(unit, clip.codec)
                    }
                })
                .collect();
            let mixed: Vec<_> = units
                .iter()
                .enumerate()
                .map(|(index, unit)| (unit.clone(), Some(index % 3 != 1)))
                .collect();
            let pairs: Vec<_> = units
                .chunks(2)
                .map(|pair| (pair.concat(), Some(false)))
                .collect();
            let halves: Vec<_> = units
                .iter()
                .flat_map(|unit| {
                    let (first, second) = unit.split_at(unit.len() / 2);
                    [
                        (first.to_vec(), Some(false)),
                        (second.to_vec(), Some(false)),
                    ]
                })
                .collect();
            for (case, packets) in [("mixed", mixed), ("pairs", pairs), ("halves", halves)] {
                assert_eq!(
                    decode_framed(&context, clip.codec, &packets),
                    expected,
                    "{} {case}, delimiters {delimited}",
                    clip.file
                );
            }
        }
    }
}

#[test]
fn display_delay_keeps_order_and_pictures() {
    let Some(context) = context() else { return };
    for delay in [1, 4] {
        let config = DecoderConfig::new(Codec::H264).with_max_display_delay(delay);
        let decoded = decode_host(&context, &H264_BFRAMES, config);
        assert_eq!(decoded.hashes, H264_BFRAMES.expected(), "delay {delay}");
    }
    // Packets split anywhere, without the one-picture-per-packet promise.
    let config = DecoderConfig::new(Codec::Hevc).with_whole_pictures(false);
    let mut decoder = Decoder::<Host>::new(&context, config).unwrap();
    let data = std::fs::read(fixture(HEVC_MAIN8.file)).unwrap();
    let mut hashes = Vec::new();
    for chunk in data.chunks(1000) {
        decoder.decode(Packet::untimed(chunk)).unwrap();
        while let Some(frame) = decoder.next_frame() {
            hashes.push(fnv1a64(frame.data()));
        }
    }
    decoder.flush().unwrap();
    while let Some(frame) = decoder.next_frame() {
        hashes.push(fnv1a64(frame.data()));
    }
    assert_eq!(hashes, HEVC_MAIN8.expected());
}

#[test]
fn held_frames_are_bounded_and_returned() {
    let Some(context) = context() else { return };
    let config = DecoderConfig::new(Codec::H264).with_max_frames(4);
    let mut decoder = Decoder::<Host>::new(&context, config).unwrap();
    let mut held = Vec::new();
    let mut refused = false;
    for (data, timestamp) in H264_BFRAMES.packets() {
        match decoder.decode(Packet::new(&data, timestamp)) {
            Ok(()) => {}
            Err(Error::FramesInFlight { capacity: 4 }) => {
                refused = true;
                break;
            }
            Err(error) => panic!("{error}"),
        }
        while let Some(frame) = decoder.next_frame() {
            held.push(frame);
        }
    }
    assert!(refused, "the fifth held frame must be refused");
    assert_eq!(held.len() + decoder.pending(), 4);
    held.clear();
    while decoder.next_frame().is_some() {}
    // With the frames back, a discontinuity outputs what the parser still
    // held (like a flush), and decoding goes on from the next keyframe.
    decoder
        .decode(Packet::untimed(&[]).with_discontinuity())
        .unwrap();
    while decoder.next_frame().is_some() {}
    let mut hashes = Vec::new();
    for (data, timestamp) in H264_BFRAMES.packets() {
        decoder.decode(Packet::new(&data, timestamp)).unwrap();
        while let Some(frame) = decoder.next_frame() {
            hashes.push(fnv1a64(frame.data()));
        }
    }
    decoder.flush().unwrap();
    while let Some(frame) = decoder.next_frame() {
        hashes.push(fnv1a64(frame.data()));
    }
    assert_eq!(hashes, H264_BFRAMES.expected());
}

#[test]
fn flush_then_decode_again() {
    let Some(context) = context() else { return };
    let mut decoder = Decoder::<Host>::new(&context, DecoderConfig::new(Codec::Hevc)).unwrap();
    for round in 0..2 {
        let mut hashes = Vec::new();
        for (data, timestamp) in HEVC_MAIN8.packets() {
            decoder.decode(Packet::new(&data, timestamp)).unwrap();
            while let Some(frame) = decoder.next_frame() {
                hashes.push(fnv1a64(frame.data()));
            }
        }
        decoder.flush().unwrap();
        while let Some(frame) = decoder.next_frame() {
            hashes.push(fnv1a64(frame.data()));
        }
        assert_eq!(hashes, HEVC_MAIN8.expected(), "round {round}");
    }
}

/// A small deterministic generator (xorshift64*), so failures reproduce.
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(bound).unwrap()).unwrap()
    }
}

/// Feeds damaged input, then the clean clip after a discontinuity, which
/// must still decode exactly.
fn survives(context: &Arc<Context>, clip: &Clip, damage: impl Fn(&mut Noise, &mut Vec<u8>)) {
    let mut decoder = Decoder::<Host>::new(context, DecoderConfig::new(clip.codec)).unwrap();
    let mut noise = Noise(0x9e37_79b9_7f4a_7c15);
    let mut errors = BTreeMap::<String, usize>::new();
    for (mut data, timestamp) in clip.packets() {
        damage(&mut noise, &mut data);
        if let Err(error) = decoder.decode(Packet::new(&data, timestamp)) {
            *errors.entry(error.to_string()).or_default() += 1;
        }
        while let Some(frame) = decoder.next_frame() {
            let _ = frame.info().status;
        }
    }
    if let Err(error) = decoder.flush() {
        *errors.entry(error.to_string()).or_default() += 1;
    }
    while decoder.next_frame().is_some() {}
    eprintln!("{}: errors {errors:?}", clip.file);
    let mut hashes = Vec::new();
    for (index, (data, timestamp)) in clip.packets().into_iter().enumerate() {
        let packet = Packet::new(&data, timestamp);
        let packet = if index == 0 {
            packet.with_discontinuity()
        } else {
            packet
        };
        decoder.decode(packet).unwrap();
        while let Some(frame) = decoder.next_frame() {
            hashes.push(fnv1a64(frame.data()));
        }
    }
    decoder.flush().unwrap();
    while let Some(frame) = decoder.next_frame() {
        hashes.push(fnv1a64(frame.data()));
    }
    assert_eq!(
        hashes,
        clip.expected(),
        "{}: clean decode after damage",
        clip.file
    );
}

#[test]
fn damaged_input_fails_cleanly() {
    let Some(context) = context() else { return };
    for clip in [
        &H264_BFRAMES,
        &HEVC_MAIN10_HDR10,
        &VP9_PROFILE0,
        &AV1_8BIT,
        &VP8,
    ] {
        if !clip.supported(&context) {
            continue;
        }
        // Truncated packets.
        survives(&context, clip, |noise, data| {
            let keep = noise.below(data.len() + 1);
            data.truncate(keep);
        });
        // Flipped bits, past the first bytes so most packets still parse.
        survives(&context, clip, |noise, data| {
            for _ in 0..8 {
                if data.len() > 8 {
                    let index = 8 + noise.below(data.len() - 8);
                    data[index] ^= 1 << noise.below(8);
                }
            }
        });
        // Random bytes of random length.
        survives(&context, clip, |noise, data| {
            let length = noise.below(4096);
            *data = (0..length).map(|_| noise.next().to_le_bytes()[0]).collect();
        });
    }
}

#[test]
fn caps_describe_the_gpu() {
    let Some(context) = context() else { return };
    for codec in [Codec::H264, Codec::Hevc, Codec::Vp8, Codec::Vp9, Codec::Av1] {
        for depth in [8, 10] {
            let caps = DecoderCaps::query(&context, codec, ChromaFormat::Yuv420, depth).unwrap();
            eprintln!(
                "{codec} {depth}-bit: supported {} engines {} {}x{}..{}x{} {} MBs formats {:#06x}",
                caps.supported,
                caps.engines,
                caps.min_width,
                caps.min_height,
                caps.max_width,
                caps.max_height,
                caps.max_macroblocks,
                caps.output_formats.bits()
            );
        }
    }
    let h264 = DecoderCaps::query(&context, Codec::H264, ChromaFormat::Yuv420, 8).unwrap();
    assert!(h264.supported, "every NVDEC GPU decodes 8-bit H.264");
    assert!(h264.max_width >= 1920 && h264.min_width > 0);
    assert!(h264.output_formats.contains(PixelFormat::Nv12));
    assert!(h264.check(1920, 1080, PixelFormat::Nv12).is_ok());
    assert!(matches!(
        h264.check(h264.max_width + 16, 1088, PixelFormat::Nv12),
        Err(Error::Unsupported { .. })
    ));
}

#[test]
fn decoders_cross_threads() {
    fn send<T: Send>() {}
    fn send_sync<T: Send + Sync>() {}
    send::<Decoder<Host>>();
    send::<Decoder<OnDevice>>();
    send_sync::<ferricast_nvcodec::decode::HostFrame>();
    send_sync::<ferricast_nvcodec::decode::DeviceFrame>();
}

/// Writes `<clip>.fnv` from FFmpeg's decode of each clip. Run by
/// `scripts/decode_fixtures.sh`, with FFmpeg on PATH or in
/// `FERRICAST_FFMPEG_DIR`.
#[test]
#[ignore = "regenerates the reference hashes with FFmpeg"]
fn reference_hashes() {
    assert!(
        std::env::var_os("FERRICAST_NVDEC_WRITE_REFERENCES").is_some(),
        "set FERRICAST_NVDEC_WRITE_REFERENCES=1 to overwrite the reference hashes"
    );
    let ffmpeg = std::env::var_os("FERRICAST_FFMPEG_DIR").map_or_else(
        || PathBuf::from("ffmpeg"),
        |dir| PathBuf::from(dir).join("ffmpeg"),
    );
    for clip in CLIPS {
        let output = Command::new(&ffmpeg)
            .env_remove("LD_LIBRARY_PATH")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-flags",
                "unaligned",
                "-i",
            ])
            .arg(fixture(clip.file))
            .args(["-noautoscale", "-fps_mode", "passthrough", "-f", "rawvideo"])
            .args(["-pix_fmt", clip.pix_fmt, "-"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}: {}",
            clip.file,
            String::from_utf8_lossy(&output.stderr)
        );
        let mut rest = output.stdout.as_slice();
        let mut lines = String::new();
        for &(width, height, count) in clip.sequences {
            for _ in 0..count {
                let (picture, tail) =
                    rest.split_at(clip.picture_bytes(width as usize, height as usize));
                writeln!(lines, "{:016x}", fnv1a64(picture)).unwrap();
                rest = tail;
            }
        }
        assert!(
            rest.is_empty(),
            "{}: {} bytes left over",
            clip.file,
            rest.len()
        );
        std::fs::write(fixture(&format!("{}.fnv", clip.file)), lines).unwrap();
    }
}
