//! Audio tracks, interleaving and seeking, on files generated with
//! ffmpeg (`FERRICAST_FFMPEG`, else `PATH`); skipped without it.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

use ferricast_core::{AudioCodec, Codec};
use ferricast_demuxer::{Demuxer, Packet};

const SECONDS: u32 = 4;
/// Keyframe every second at 30 fps.
const GOP: u32 = 30;

fn ffmpeg() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("FERRICAST_FFMPEG") {
        return Some(PathBuf::from(p));
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|d| d.join("ffmpeg"))
            .find(|p| p.is_file())
    })
}

/// Encodes a test clip into `dir/name`; `None` without ffmpeg.
fn generate(
    dir: &Path,
    name: &str,
    video: &[&str],
    audio: &[&str],
    extra: &[&str],
) -> Option<Vec<u8>> {
    let Some(ffmpeg) = ffmpeg() else {
        eprintln!("ffmpeg not found (set FERRICAST_FFMPEG) — skipping");
        return None;
    };
    let gop = GOP.to_string();
    let status = Command::new(ffmpeg)
        .env_remove("LD_LIBRARY_PATH")
        .current_dir(dir)
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30"])
        .args(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"])
        .args(["-t", &SECONDS.to_string()])
        .args(video)
        .args(["-g", &gop, "-keyint_min", &gop, "-pix_fmt", "yuv420p"])
        .args(audio)
        .args(["-ac", "2"])
        .args(extra)
        .arg(name)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "ffmpeg failed for {name}: {status}");
    Some(std::fs::read(dir.join(name)).unwrap())
}

const H264: &[&str] = &["-c:v", "libx264", "-profile:v", "high", "-bf", "2"];
const VP9: &[&str] = &[
    "-c:v",
    "libvpx-vp9",
    "-deadline",
    "realtime",
    "-b:v",
    "200k",
];
const AAC: &[&str] = &["-c:a", "aac", "-b:a", "96k"];
const OPUS: &[&str] = &["-c:a", "libopus", "-b:a", "64k"];

struct Case {
    name: &'static str,
    video: &'static [&'static str],
    audio: &'static [&'static str],
    extra: &'static [&'static str],
    codecs: (Codec, AudioCodec),
}

const CASES: &[Case] = &[
    Case {
        name: "faststart.mp4",
        video: H264,
        audio: AAC,
        extra: &["-movflags", "+faststart"],
        codecs: (Codec::H264, AudioCodec::Aac),
    },
    Case {
        name: "moov_at_end.mp4",
        video: H264,
        audio: AAC,
        extra: &[],
        codecs: (Codec::H264, AudioCodec::Aac),
    },
    Case {
        name: "h264_aac.mkv",
        video: H264,
        audio: AAC,
        extra: &[],
        codecs: (Codec::H264, AudioCodec::Aac),
    },
    Case {
        name: "vp9_opus.webm",
        video: VP9,
        audio: OPUS,
        extra: &[],
        codecs: (Codec::Vp9, AudioCodec::Opus),
    },
    Case {
        name: "h264_opus.mp4",
        video: H264,
        audio: OPUS,
        extra: &[],
        codecs: (Codec::H264, AudioCodec::Opus),
    },
];

fn packets(demuxer: &mut Demuxer) -> Vec<Packet> {
    std::iter::from_fn(|| demuxer.next_packet().unwrap()).collect()
}

fn timestamp(packet: &Packet) -> u64 {
    match packet {
        Packet::Video(f) => f.pts_dts.1,
        Packet::Audio(a) => a.timestamp_us,
    }
}

#[test]
fn audio_and_video_come_interleaved_from_any_reader() {
    let dir = tempfile::tempdir().unwrap();
    for case in CASES {
        let Some(bytes) = generate(dir.path(), case.name, case.video, case.audio, case.extra)
        else {
            return;
        };
        let mut demuxer = Demuxer::from_reader(Cursor::new(bytes))
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let video = demuxer.video().expect("video track").clone();
        let audio = demuxer.audio().expect("audio track").clone();
        assert_eq!((video.codec, audio.codec), case.codecs, "{}", case.name);
        assert_eq!((video.width, video.height), (160, 120), "{}", case.name);
        assert_eq!(
            (audio.sample_rate, audio.channels),
            (48_000, 2),
            "{}",
            case.name
        );
        let duration = demuxer.duration_us().expect("duration");
        assert!(
            duration.abs_diff(u64::from(SECONDS) * 1_000_000) < 100_000,
            "{}: duration {duration}",
            case.name
        );

        let all = packets(&mut demuxer);
        let video_frames = all.iter().filter(|p| matches!(p, Packet::Video(_))).count();
        let audio_packets: Vec<_> = all
            .iter()
            .filter_map(|p| match p {
                Packet::Audio(a) => Some(a),
                Packet::Video(_) => None,
            })
            .collect();
        assert_eq!(video_frames, 120, "{}", case.name);
        let audio_span = audio_packets.last().unwrap().timestamp_us;
        assert!(
            audio_span > 3_800_000,
            "{}: audio ends at {audio_span}",
            case.name
        );
        assert!(
            audio_packets
                .windows(2)
                .all(|w| w[0].timestamp_us < w[1].timestamp_us),
            "{}: audio timestamps not increasing",
            case.name
        );
        if audio.codec == AudioCodec::Aac {
            assert!(
                audio_packets.iter().all(|a| a.data[..2] == [0xFF, 0xF1]),
                "{}: AAC not ADTS-framed",
                case.name
            );
        }
        // Interleaved: no stretch of more than a second of one track
        // before the other catches up.
        let mut last_video = 0;
        let mut last_audio = 0;
        for packet in &all {
            match packet {
                Packet::Video(_) => last_video = timestamp(packet),
                Packet::Audio(_) => last_audio = timestamp(packet),
            }
            assert!(
                last_video.abs_diff(last_audio) < 1_100_000 || last_video == 0 || last_audio == 0,
                "{}: tracks drift apart ({last_video} vs {last_audio})",
                case.name
            );
        }
    }
}

#[test]
fn seek_lands_on_the_keyframe_before_the_target_and_audio_follows() {
    let dir = tempfile::tempdir().unwrap();
    for case in CASES {
        let Some(bytes) = generate(dir.path(), case.name, case.video, case.audio, case.extra)
        else {
            return;
        };
        let mut demuxer = Demuxer::from_reader(Cursor::new(bytes)).unwrap();
        for target in [2_500_000u64, 1_000_000, 3_900_000, 0] {
            let landed = demuxer.seek(target).unwrap();
            assert!(
                landed <= target,
                "{}: landed {landed} after {target}",
                case.name
            );
            assert!(
                target - landed <= 1_050_000,
                "{}: landed {landed} far before {target}",
                case.name
            );
            let after = packets(&mut demuxer);
            let first_video = after
                .iter()
                .find_map(|p| match p {
                    Packet::Video(f) => Some(f),
                    Packet::Audio(_) => None,
                })
                .unwrap_or_else(|| {
                    panic!(
                        "{}: no video after seeking to {target} (landed {landed}, {} packets)",
                        case.name,
                        after.len()
                    )
                });
            assert!(first_video.is_keyframe, "{}: seek to {target}", case.name);
            assert_eq!(first_video.timestamp_us, landed, "{}", case.name);
            let first_audio = after
                .iter()
                .find_map(|p| match p {
                    Packet::Audio(a) => Some(a.timestamp_us),
                    Packet::Video(_) => None,
                })
                .unwrap();
            assert!(
                first_audio.abs_diff(landed) < 100_000,
                "{}: audio resumes at {first_audio}, video at {landed}",
                case.name
            );
            demuxer.seek(0).unwrap();
        }
    }
}

#[test]
fn audio_only_files_open() {
    let dir = tempfile::tempdir().unwrap();
    let Some(bytes) = generate(dir.path(), "audio.m4a", &["-vn"], AAC, &[]) else {
        return;
    };
    let mut demuxer = Demuxer::from_reader(Cursor::new(bytes)).unwrap();
    assert!(demuxer.video().is_none());
    assert_eq!(demuxer.audio().unwrap().codec, AudioCodec::Aac);
    assert_eq!(demuxer.seek(2_000_000).unwrap(), 2_000_000);
    let Some(Packet::Audio(first)) = demuxer.next_packet().unwrap() else {
        panic!("expected audio");
    };
    assert!(
        first.timestamp_us.abs_diff(2_000_000) < 30_000,
        "{}",
        first.timestamp_us
    );
}

#[test]
fn fragmented_mp4_is_rejected_clearly() {
    let dir = tempfile::tempdir().unwrap();
    let Some(bytes) = generate(
        dir.path(),
        "fragmented.mp4",
        H264,
        AAC,
        &["-movflags", "frag_keyframe+empty_moov"],
    ) else {
        return;
    };
    let error = Demuxer::from_reader(Cursor::new(bytes))
        .err()
        .expect("must fail");
    assert!(error.to_string().contains("fragmented"), "{error}");
}

#[test]
fn opening_a_file_by_path_ignores_its_audio() {
    let dir = tempfile::tempdir().unwrap();
    let Some(_) = generate(dir.path(), "clip.mkv", H264, AAC, &[]) else {
        return;
    };
    let mut demuxer = Demuxer::open(&dir.path().join("clip.mkv")).unwrap();
    assert!(demuxer.audio().is_none());
    assert_eq!(packets(&mut demuxer).len(), 120);
}
