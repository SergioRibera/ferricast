//! 5.1 AAC from ffmpeg's native encoder, in ADTS and MP4, demuxed and
//! decoded to six channels. Skipped without ffmpeg
//! (`FERRICAST_FFMPEG`, else `PATH`).

#![cfg(feature = "fdk-aac-decode")]

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

use ferricast_core::{AudioDecoder, AudioDecoderConfig};
use ferricast_decoder::AacDecoder;
use ferricast_demuxer::{Demuxer, Packet};

const SECONDS: u32 = 2;
const RATE: u32 = 48_000;
/// A different tone per channel, in 5.1 (WAV) order: FL, FR, FC, LFE,
/// BL, BR. The LFE tone stays under the encoder's low-pass.
const TONES: [u32; 6] = [440, 660, 880, 60, 1_100, 1_320];

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

/// Encodes the six tones as 5.1 AAC into `dir/name`; `None` without ffmpeg.
fn generate(dir: &Path, name: &str) -> Option<Vec<u8>> {
    let Some(ffmpeg) = ffmpeg() else {
        eprintln!("ffmpeg not found (set FERRICAST_FFMPEG) — skipping");
        return None;
    };
    let expr = TONES
        .iter()
        .map(|f| format!("0.3*sin({f}*2*PI*t)"))
        .collect::<Vec<_>>()
        .join("|");
    let source = format!("aevalsrc={expr}:c=5.1:s={RATE}");
    let status = Command::new(ffmpeg)
        .env_remove("LD_LIBRARY_PATH")
        .current_dir(dir)
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(["-f", "lavfi", "-i", &source])
        .args(["-t", &SECONDS.to_string()])
        .args(["-c:a", "aac", "-b:a", "384k"])
        .arg(name)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "ffmpeg failed for {name}: {status}");
    Some(std::fs::read(dir.join(name)).unwrap())
}

/// Demuxes and decodes `bytes`: the stream's channel count, then the
/// decoded rate, channel count and interleaved samples.
fn decode(bytes: Vec<u8>) -> (u16, u32, u16, Vec<i16>) {
    let mut demuxer = Demuxer::from_reader(Cursor::new(bytes)).unwrap();
    let stream = demuxer.audio().expect("audio track").clone();
    let mut decoder = AacDecoder::default();
    decoder
        .configure(&AudioDecoderConfig {
            extra_data: None,
            codec: stream.codec,
            sample_rate: stream.sample_rate,
            channels: stream.channels,
            backend: ferricast_core::BackendSelection::default(),
        })
        .unwrap();
    let mut shape = None;
    let mut pcm = Vec::new();
    while let Some(packet) = demuxer.next_packet().unwrap() {
        let Packet::Audio(frame) = packet else {
            continue;
        };
        if let Some(audio) = decoder.decode(frame).unwrap() {
            shape = Some((audio.sample_rate, audio.channels));
            pcm.extend(audio.pcm);
        }
    }
    for audio in decoder.flush().unwrap() {
        pcm.extend(audio.pcm);
    }
    let (rate, channels) = shape.expect("decoded audio");
    (stream.channels, rate, channels, pcm)
}

/// Strength of `freq` in channel `ch` of interleaved `pcm` (one DFT bin).
fn tone_level(pcm: &[i16], channels: usize, ch: usize, freq: u32) -> f64 {
    let (mut re, mut im) = (0.0f64, 0.0f64);
    let samples = pcm.iter().skip(ch).step_by(channels);
    let mut n = 0usize;
    for (i, &s) in samples.enumerate() {
        let phase = std::f64::consts::TAU * f64::from(freq) * i as f64 / f64::from(RATE);
        re += f64::from(s) * phase.cos();
        im += f64::from(s) * phase.sin();
        n = i + 1;
    }
    (re * re + im * im).sqrt() / n.max(1) as f64
}

#[test]
fn surround_aac_decodes_to_six_channels_in_order() {
    let dir = tempfile::tempdir().unwrap();
    for name in ["surround.aac", "surround.m4a"] {
        let Some(bytes) = generate(dir.path(), name) else {
            return;
        };
        let (stream_channels, rate, channels, pcm) = decode(bytes);
        assert_eq!(stream_channels, 6, "{name}: demuxed channel count");
        assert_eq!((rate, channels), (RATE, 6), "{name}: decoded shape");
        let frames = pcm.len() / 6;
        assert!(
            frames.abs_diff((RATE * SECONDS) as usize) < 4_096,
            "{name}: {frames} frames"
        );
        // Each channel carries its own tone, well above the others.
        for (ch, &tone) in TONES.iter().enumerate() {
            let own = tone_level(&pcm, 6, ch, tone);
            for (other, &leak) in TONES.iter().enumerate().filter(|&(o, _)| o != ch) {
                let level = tone_level(&pcm, 6, ch, leak);
                assert!(
                    own > 20.0 * level,
                    "{name}: channel {ch} has tone {leak} of channel {other} at {level:.1} vs {own:.1}"
                );
            }
        }
    }
}
