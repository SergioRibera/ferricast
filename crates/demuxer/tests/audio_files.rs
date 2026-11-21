//! Audio-only files (MP3, FLAC, Ogg Vorbis/Opus, WAV, ADTS), generated
//! with ffmpeg (`FERRICAST_FFMPEG`, else `PATH`); skipped without it.

#![cfg(feature = "audio-files")]

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

use ferricast_core::AudioCodec;
use ferricast_core::frame::SourceAudioFormat;
use ferricast_demuxer::{Demuxer, Packet};

const SECONDS: f64 = 3.0;

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

fn generate(dir: &Path, name: &str, args: &[&str]) -> Option<Vec<u8>> {
    let Some(ffmpeg) = ffmpeg() else {
        eprintln!("ffmpeg not found (set FERRICAST_FFMPEG) — skipping");
        return None;
    };
    let status = Command::new(ffmpeg)
        .env_remove("LD_LIBRARY_PATH")
        .current_dir(dir)
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100"])
        .args(["-t", &SECONDS.to_string(), "-ac", "2"])
        .args(args)
        .arg(name)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "ffmpeg failed for {name}: {status}");
    Some(std::fs::read(dir.join(name)).unwrap())
}

/// `(name, ffmpeg codec args, expected codec)`.
const CASES: &[(&str, &[&str], AudioCodec)] = &[
    (
        "tagged.mp3",
        &[
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            "-metadata",
            "title=ferricast",
            "-id3v2_version",
            "3",
        ],
        AudioCodec::Pcm,
    ),
    (
        "vbr.mp3",
        &["-c:a", "libmp3lame", "-q:a", "4"],
        AudioCodec::Pcm,
    ),
    ("lossless.flac", &["-c:a", "flac"], AudioCodec::Pcm),
    (
        "vorbis.ogg",
        &["-c:a", "libvorbis", "-q:a", "3"],
        AudioCodec::Pcm,
    ),
    (
        "opus.ogg",
        &["-c:a", "libopus", "-b:a", "64k"],
        AudioCodec::Opus,
    ),
    ("s16.wav", &["-c:a", "pcm_s16le"], AudioCodec::Pcm),
    ("s24.wav", &["-c:a", "pcm_s24le"], AudioCodec::Pcm),
    ("f32.wav", &["-c:a", "pcm_f32le"], AudioCodec::Pcm),
    (
        "adts.aac",
        &["-c:a", "aac", "-b:a", "96k", "-f", "adts"],
        AudioCodec::Aac,
    ),
];

/// Seconds of audio in a packet: PCM from its size, otherwise from the
/// gap to the next packet.
fn pcm_seconds(packet: &ferricast_core::AudioFrame) -> f64 {
    let frames = packet.data.len() / 2 / usize::from(packet.channels);
    frames as f64 / f64::from(packet.sample_rate)
}

#[test]
fn audio_files_report_their_duration_and_seek_accurately() {
    let dir = tempfile::tempdir().unwrap();
    for &(name, args, codec) in CASES {
        let Some(bytes) = generate(dir.path(), name, args) else {
            return;
        };
        let mut demuxer = Demuxer::from_reader(Cursor::new(bytes)).unwrap();
        assert!(demuxer.video().is_none(), "{name}");
        let audio = demuxer.audio().unwrap().clone();
        assert_eq!(audio.codec, codec, "{name}");
        let source = match name.rsplit_once('.').map(|(_, ext)| ext) {
            Some("mp3") => Some(SourceAudioFormat::Mp3),
            Some("flac") => Some(SourceAudioFormat::Flac),
            Some("wav") => Some(SourceAudioFormat::Wav),
            _ if name.starts_with("vorbis") => Some(SourceAudioFormat::Vorbis),
            _ => None,
        };
        assert_eq!(audio.source, source, "{name}");
        assert_eq!(audio.channels, 2, "{name}");
        let duration = demuxer.duration_us().expect("duration") as f64 / 1e6;

        let mut timestamps = Vec::new();
        let mut pcm = 0.0;
        while let Some(packet) = demuxer.next_packet().unwrap() {
            let Packet::Audio(a) = packet else {
                panic!("{name}: video packet")
            };
            if a.codec == AudioCodec::Pcm {
                pcm += pcm_seconds(&a);
            }
            timestamps.push(a.timestamp_us);
        }
        let span = (timestamps.last().unwrap() - timestamps[0]) as f64 / 1e6;
        eprintln!(
            "{name}: {codec:?} {} Hz, duration {duration:.3}s, {} packets spanning {span:.3}s, \
             {pcm:.3}s of PCM",
            audio.sample_rate,
            timestamps.len()
        );
        assert!(
            (duration - SECONDS).abs() < 0.1,
            "{name}: duration {duration}"
        );
        assert!(
            timestamps.windows(2).all(|w| w[0] < w[1]),
            "{name}: timestamps"
        );
        assert!(
            span > SECONDS - 0.2 && span < SECONDS + 0.1,
            "{name}: span {span}"
        );
        if codec == AudioCodec::Pcm {
            assert!((pcm - SECONDS).abs() < 0.06, "{name}: {pcm}s of PCM");
        }

        let landed = demuxer.seek(2_000_000).unwrap();
        let Some(Packet::Audio(first)) = demuxer.next_packet().unwrap() else {
            panic!("{name}: nothing after the seek")
        };
        eprintln!(
            "{name}: seek to 2.000s landed {:.3}s, first packet {:.3}s",
            landed as f64 / 1e6,
            first.timestamp_us as f64 / 1e6
        );
        assert!(
            first.timestamp_us <= 2_000_000 && first.timestamp_us > 1_800_000,
            "{name}: first packet after seek at {}",
            first.timestamp_us
        );
    }
}
