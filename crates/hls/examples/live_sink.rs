//! Serve a pre-encoded H.264 Annex B file as a live HLS stream through
//! [`HlsFrameSink`], paced in real time with wall-clock timestamps, so
//! the server can be checked with real players (`ffmpeg`, `ffprobe`,
//! `curl`).
//!
//! The input must delimit access units with AUD NAL units and carry
//! SPS/PPS on every IDR, e.g.:
//!
//! ```text
//! ffmpeg -f lavfi -i testsrc2=size=640x360:rate=30 -t 30 -c:v libx264 \
//!   -profile:v baseline -g 60 -keyint_min 60 -sc_threshold 0 -bf 0 \
//!   -x264-params aud=1:repeat-headers=1 -f h264 test.h264
//! cargo run -p ferricast-hls --example live_sink -- test.h264 30 25
//! ```
//!
//! Arguments: input file, frame rate (default 30), run time in seconds
//! (default 25), optional LL-HLS part target in seconds. Prints the
//! playlist URL on the first line of stdout.

use std::io::Write as _;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use ferricast_core::{Codec, EncodedFrame};
use ferricast_hls::{HlsConfig, HlsFrameSink};
use ferricast_muxer::mpeg_ts::in_band_parameter_sets;
use tokio::sync::mpsc;

/// Split an Annex B stream into access units at each AUD NAL (type 9),
/// dropping the AUD itself (the muxer writes its own).
fn access_units(stream: &[u8]) -> Vec<(Vec<u8>, bool)> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 4 < stream.len() {
        if stream[i..i + 3] == [0, 0, 1] {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut units: Vec<(Vec<u8>, bool)> = Vec::new();
    for (n, &start) in starts.iter().enumerate() {
        let end = starts.get(n + 1).map_or(stream.len(), |&next| next - 3);
        let nal = &stream[start..end];
        let nal = &nal[..nal.iter().rposition(|&b| b != 0).map_or(0, |p| p + 1)];
        let Some(&header) = nal.first() else { continue };
        match header & 0x1F {
            9 => units.push((Vec::new(), false)),
            kind => {
                if units.is_empty() {
                    units.push((Vec::new(), false));
                }
                let unit = units.last_mut().expect("just ensured non-empty");
                unit.0.extend_from_slice(&[0, 0, 0, 1]);
                unit.0.extend_from_slice(nal);
                unit.1 |= kind == 5;
            }
        }
    }
    units.retain(|(data, _)| !data.is_empty());
    units
}

fn now_us() -> u64 {
    u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros(),
    )
    .unwrap()
}

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: live_sink <file.h264> [fps] [seconds]");
    let fps: u32 = args.next().map_or(30, |a| a.parse().unwrap());
    let seconds: u64 = args.next().map_or(25, |a| a.parse().unwrap());
    let part_target_secs: Option<f32> = args.next().map(|a| a.parse().unwrap());

    let stream = std::fs::read(&path).unwrap();
    let units = access_units(&stream);
    let first_idr = units.iter().find(|(_, key)| *key).expect("no IDR in input");
    let parameter_sets = in_band_parameter_sets(Codec::H264, &first_idr.0).expect("no SPS/PPS");

    let (tx, rx) = mpsc::channel(256);
    let sink = HlsFrameSink::start(
        "127.0.0.1:0",
        rx,
        None,
        parameter_sets,
        HlsConfig {
            segment_target_secs: 2.0,
            playlist_target_duration: 4,
            target_fps: fps,
            part_target_secs,
            ..HlsConfig::default()
        },
    )
    .await
    .unwrap();
    println!(
        "URL {}://{}{}",
        sink.scheme(),
        sink.local_addr(),
        sink.playlist_path()
    );
    std::io::stdout().flush().unwrap();

    let ready = sink.first_segment_ready();
    tokio::spawn(async move {
        let ok = ready.await;
        println!("READY {ok}");
        std::io::stdout().flush().unwrap();
    });

    let period = Duration::from_secs_f64(1.0 / f64::from(fps));
    let mut ticker = tokio::time::interval(period);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(seconds);
    // Start on an IDR and loop the clip.
    let start = units.iter().position(|(_, key)| *key).unwrap();
    for (data, key) in units[start..].iter().cycle() {
        ticker.tick().await;
        if tokio::time::Instant::now() >= deadline {
            break;
        }
        let frame = EncodedFrame {
            codec: Codec::H264,
            data: Bytes::from(data.clone()),
            // Capture-style wall-clock timestamp, with the jitter of the
            // pacing timer.
            timestamp_us: now_us(),
            is_keyframe: *key,
            duration_us: None,
            pts_dts: (0, 0),
        };
        if tx.send(frame).await.is_err() {
            break;
        }
    }
    println!("DONE");
}
