use std::path::PathBuf;

use ferricast_core::{CapturedFrame, PixelFormat, RawFrame};
use ferricast_player::Player;

const FRAME_US: u64 = 33_333;
const WIDTH: usize = 128;
const HEIGHT: usize = 72;
const SIGNATURE_LEN: usize = 216;
// openh264 is not bit-exact on some B slices: https://github.com/cisco/openh264/issues/3269
const B_SLICE_TOLERANCE: u8 = 8;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../demuxer/tests/fixtures")
        .join(name)
}

fn cell_means(plane: &[u8], width: usize, height: usize, cell: (usize, usize)) -> Vec<u8> {
    let mut means = Vec::new();
    for top in (0..height).step_by(cell.1) {
        for left in (0..width).step_by(cell.0) {
            let (mut sum, mut count) = (0usize, 0usize);
            for row in top..(top + cell.1).min(height) {
                let line = &plane[row * width + left..row * width + (left + cell.0).min(width)];
                sum += line.iter().map(|&p| usize::from(p)).sum::<usize>();
                count += line.len();
            }
            means.push(u8::try_from((sum + count / 2) / count).unwrap());
        }
    }
    means
}

fn signature(raw: &RawFrame) -> Vec<u8> {
    let (luma, chroma) = raw.data.split_at(WIDTH * HEIGHT);
    let u: Vec<u8> = chroma.iter().step_by(2).copied().collect();
    let v: Vec<u8> = chroma.iter().skip(1).step_by(2).copied().collect();
    let mut signature = cell_means(luma, WIDTH, HEIGHT, (16, 8));
    signature.extend(cell_means(&u, WIDTH / 2, HEIGHT / 2, (8, 4)));
    signature.extend(cell_means(&v, WIDTH / 2, HEIGHT / 2, (8, 4)));
    signature
}

fn play(player: &mut Player) -> Vec<RawFrame> {
    std::iter::from_fn(|| player.next_frame().unwrap())
        .map(|frame| {
            let CapturedFrame::Cpu(raw) = frame else {
                panic!("expected a CPU frame");
            };
            assert_eq!(raw.format, PixelFormat::Nv12);
            assert_eq!(raw.stride as usize, WIDTH);
            raw
        })
        .collect()
}

fn distance(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .zip(b)
        .map(|(a, b)| usize::from(a.abs_diff(*b)))
        .sum()
}

fn assert_matches_reference(name: &str, tolerance: u8) {
    let frames = play(&mut Player::open(&fixture(name)).unwrap());
    let reference = std::fs::read(fixture(&format!("{name}.sig"))).unwrap();
    let (chunks, _) = reference.as_chunks::<SIGNATURE_LEN>();
    let references: Vec<&[u8]> = chunks.iter().map(<[u8; 216]>::as_slice).collect();
    assert_eq!(frames.len(), references.len(), "{name}");
    for (i, frame) in frames.iter().enumerate() {
        let decoded = signature(frame);
        let nearest = (0..references.len())
            .min_by_key(|&r| distance(&decoded, references[r]))
            .unwrap();
        assert_eq!(nearest, i, "{name}: frame {i} looks like frame {nearest}");
        let worst = decoded
            .iter()
            .zip(references[i])
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(worst <= tolerance, "{name}: frame {i} is off by {worst}");
        assert!(
            frame.timestamp_us.abs_diff(i as u64 * FRAME_US) <= 1_000,
            "{name}: frame {i} at {}",
            frame.timestamp_us
        );
    }
}

#[test]
fn h264_clips_decode_to_the_reference_frames_in_presentation_order() {
    for name in ["h264.mp4", "h264.mov", "h264.mkv"] {
        assert_matches_reference(name, B_SLICE_TOLERANCE);
    }
}

#[test]
fn hevc_clips_decode_to_the_reference_frames_in_presentation_order() {
    for name in ["hevc.mp4", "hevc.mkv"] {
        assert_matches_reference(name, 0);
    }
}

#[test]
fn av1_clips_decode_to_the_reference_frames() {
    for name in ["av1.mp4", "av1.webm"] {
        assert_matches_reference(name, 0);
    }
}

#[test]
fn vp8_and_vp9_clips_decode_to_the_reference_frames() {
    for name in ["vp8.webm", "vp9.webm", "vp9.mp4"] {
        assert_matches_reference(name, 0);
    }
}

#[test]
fn rewinding_after_the_end_replays_the_clip() {
    let mut player = Player::open(&fixture("h264.mp4")).unwrap();
    let first = play(&mut player);
    player.rewind().unwrap();
    let again = play(&mut player);
    assert_eq!(again.len(), first.len());
    for (a, b) in first.iter().zip(&again) {
        assert_eq!((a.timestamp_us, &a.data), (b.timestamp_us, &b.data));
    }
}

#[test]
fn rewinding_midway_restarts_from_the_first_frame() {
    let mut player = Player::open(&fixture("h264.mp4")).unwrap();
    for _ in 0..5 {
        player.next_frame().unwrap().unwrap();
    }
    player.rewind().unwrap();
    let frames = play(&mut player);
    assert_eq!(frames.len(), 24);
    assert_eq!(frames[0].timestamp_us, 0);
}

#[test]
fn the_stream_description_matches_the_clip() {
    let player = Player::open(&fixture("h264.mkv")).unwrap();
    let video = player.video();
    assert_eq!(
        (video.width, video.height),
        (
            u32::try_from(WIDTH).unwrap(),
            u32::try_from(HEIGHT).unwrap()
        )
    );
}
