use std::path::PathBuf;

use ferricast_core::{Codec, EncodedFrame};
use ferricast_demuxer::Demuxer;

const FRAMES: usize = 24;
const FRAME_US: u64 = 33_333;

const FIXTURES: &[(&str, Codec)] = &[
    ("h264.mp4", Codec::H264),
    ("h264.mov", Codec::H264),
    ("hevc.mp4", Codec::H265),
    ("vp9.mp4", Codec::Vp9),
    ("av1.mp4", Codec::Av1),
    ("vp8.webm", Codec::Vp8),
    ("vp9.webm", Codec::Vp9),
    ("av1.webm", Codec::Av1),
    ("h264.mkv", Codec::H264),
    ("hevc.mkv", Codec::H265),
];

fn open(name: &str) -> Demuxer {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    Demuxer::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn frames(demuxer: &mut Demuxer) -> Vec<EncodedFrame> {
    std::iter::from_fn(|| demuxer.next_frame().unwrap()).collect()
}

#[test]
fn every_fixture_reports_its_codec_and_size() {
    for &(name, codec) in FIXTURES {
        let demuxer = open(name);
        let stream = demuxer.video();
        assert_eq!(stream.codec, codec, "{name}");
        assert_eq!((stream.width, stream.height), (128, 72), "{name}");
    }
}

#[test]
fn every_fixture_yields_all_frames_and_two_keyframes() {
    for &(name, _) in FIXTURES {
        let frames = frames(&mut open(name));
        assert_eq!(frames.len(), FRAMES, "{name}");
        assert!(frames[0].is_keyframe, "{name}: first frame");
        let keyframes = frames.iter().filter(|f| f.is_keyframe).count();
        assert_eq!(keyframes, 2, "{name}");
    }
}

#[test]
fn presentation_times_cover_the_clip_at_thirty_frames_a_second() {
    for &(name, _) in FIXTURES {
        let mut times: Vec<u64> = frames(&mut open(name))
            .iter()
            .map(|f| f.timestamp_us)
            .collect();
        times.sort_unstable();
        for (i, time) in times.iter().enumerate() {
            let expected = i as u64 * FRAME_US;
            assert!(
                time.abs_diff(expected) <= 1_000,
                "{name}: frame {i} at {time}"
            );
        }
    }
}

#[test]
fn mp4_decode_times_never_run_backwards() {
    for name in ["h264.mp4", "h264.mov", "hevc.mp4", "vp9.mp4", "av1.mp4"] {
        let dts: Vec<u64> = frames(&mut open(name))
            .iter()
            .map(|f| f.pts_dts.1)
            .collect();
        assert!(dts.windows(2).all(|w| w[0] <= w[1]), "{name}");
    }
}

#[test]
fn h264_and_hevc_reorder_frames_in_decode_order() {
    for name in ["h264.mp4", "hevc.mp4", "h264.mkv", "hevc.mkv"] {
        let times: Vec<u64> = frames(&mut open(name))
            .iter()
            .map(|f| f.timestamp_us)
            .collect();
        assert!(times.windows(2).any(|w| w[0] > w[1]), "{name}");
    }
}

#[test]
fn h264_and_hevc_keyframes_start_with_their_parameter_sets() {
    for (name, first_nal) in [
        ("h264.mp4", 0x67),
        ("h264.mkv", 0x67),
        ("hevc.mp4", 0x40),
        ("hevc.mkv", 0x40),
    ] {
        let frames = frames(&mut open(name));
        assert_eq!(&frames[0].data[..4], &[0, 0, 0, 1], "{name}");
        assert_eq!(frames[0].data[4], first_nal, "{name}");
    }
}

#[test]
fn rewinding_replays_the_same_frames() {
    for &(name, _) in FIXTURES {
        let mut demuxer = open(name);
        let first: Vec<_> = frames(&mut demuxer)
            .iter()
            .map(|f| (f.timestamp_us, f.data.clone()))
            .collect();
        demuxer.rewind().unwrap();
        let again: Vec<_> = frames(&mut demuxer)
            .iter()
            .map(|f| (f.timestamp_us, f.data.clone()))
            .collect();
        assert_eq!(first, again, "{name}");
    }
}

#[test]
fn a_text_file_is_not_a_container() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    assert!(Demuxer::open(&path).is_err());
}
