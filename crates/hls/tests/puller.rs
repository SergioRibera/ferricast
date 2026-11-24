//! HLS puller against real ffmpeg-generated streams served over a
//! local HTTP server. Skips when ffmpeg is unavailable.

#![expect(clippy::cast_possible_truncation, reason = "fixture counters are tiny")]

#[path = "../../../tests/support/mod.rs"]
mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use ferricast_core::{AudioCodec, Codec, MediaPacket, MediaPuller, PullSpec};
use ferricast_hls::HlsPuller;
use support::{Reply, TestServer, live_server};

fn spec(url: String) -> PullSpec {
    PullSpec {
        url,
        headers: std::collections::HashMap::default(),
    }
}

#[derive(Default, Debug)]
struct Collected {
    video_pts: Vec<u64>,
    video_dts: Vec<u64>,
    keyframes: usize,
    audio: usize,
    discontinuities: usize,
    present_from: Option<u64>,
}

async fn collect(puller: &mut HlsPuller, limit: Duration) -> Collected {
    let mut c = Collected::default();
    let deadline = Instant::now() + limit;
    loop {
        let packet = tokio::time::timeout_at(deadline.into(), puller.next())
            .await
            .expect("puller stalled")
            .expect("puller error");
        match packet {
            MediaPacket::Video(v) => {
                c.video_pts.push(v.pts_dts.0);
                c.video_dts.push(v.pts_dts.1);
                c.keyframes += usize::from(v.is_keyframe);
            }
            MediaPacket::Audio(_) => c.audio += 1,
            MediaPacket::Discontinuity => c.discontinuities += 1,
            MediaPacket::PresentFrom { timestamp_us } => c.present_from = Some(timestamp_us),
            MediaPacket::Eos => return c,
        }
    }
}

fn assert_monotonic(values: &[u64]) {
    assert!(
        values.windows(2).all(|w| w[0] <= w[1]),
        "not monotonic: {values:?}"
    );
}

#[tokio::test]
async fn vod_plays_to_eos_with_dts_and_keyframes() {
    let dir = tempfile::tempdir().unwrap();
    if !support::generate_hls(dir.path(), 6, 2, &[]) {
        return;
    }
    let server = TestServer::start(support::dir_handler(dir.path())).await;
    let mut puller = HlsPuller::new();
    let info = puller.open(spec(server.url("/index.m3u8"))).await.unwrap();
    assert!(!info.is_live);
    assert_eq!(info.video.as_ref().map(|v| v.codec), Some(Codec::H264));
    let audio = info.audio.as_ref().unwrap();
    assert_eq!(
        (audio.codec, audio.sample_rate, audio.channels),
        (AudioCodec::Aac, 48_000, 2)
    );
    let duration = info.duration_us.unwrap();
    assert!((5_900_000..=6_100_000).contains(&duration), "{duration}");

    let c = collect(&mut puller, Duration::from_secs(20)).await;
    assert_eq!(c.video_pts.len(), 180);
    assert_monotonic(&c.video_dts);
    // B-frames: PTS is not in decode order, and differs from DTS.
    assert!(c.video_pts.windows(2).any(|w| w[0] > w[1]));
    assert!(c.video_pts.iter().zip(&c.video_dts).any(|(p, d)| p != d));
    assert_eq!(c.keyframes, 3);
    // ffmpeg packs several ADTS frames into one PES.
    assert!(c.audio > 20, "audio packets {}", c.audio);
    assert_eq!(c.discontinuities, 0);
    puller.close().await.unwrap();
}

#[tokio::test]
async fn aes128_and_byterange_streams_play() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("enc.key"), [0x42u8; 16]).unwrap();
    std::fs::write(
        dir.path().join("keyinfo"),
        "enc.key\nenc.key\n000102030405060708090a0b0c0d0e0f\n",
    )
    .unwrap();
    if !support::generate_hls(
        dir.path(),
        4,
        2,
        &["-hls_key_info_file", "keyinfo", "-hls_flags", "single_file"],
    ) {
        return;
    }
    let playlist = std::fs::read_to_string(dir.path().join("index.m3u8")).unwrap();
    assert!(playlist.contains("EXT-X-KEY:METHOD=AES-128"), "{playlist}");
    assert!(playlist.contains("EXT-X-BYTERANGE"), "{playlist}");

    let server = TestServer::start(support::dir_handler(dir.path())).await;
    let mut puller = HlsPuller::new();
    let info = puller.open(spec(server.url("/index.m3u8"))).await.unwrap();
    assert!(info.video.is_some() && info.audio.is_some());
    let c = collect(&mut puller, Duration::from_secs(20)).await;
    assert_eq!(c.video_pts.len(), 120);
    assert_eq!(c.keyframes, 2);
}

#[tokio::test]
async fn relative_uris_resolve_against_the_redirected_url() {
    let dir = tempfile::tempdir().unwrap();
    if !support::generate_hls(dir.path(), 2, 2, &[]) {
        return;
    }
    let files = support::dir_handler(dir.path());
    let server = TestServer::start(Arc::new(move |path: &str| {
        if path == "/old/index.m3u8" {
            Reply::Redirect("/new/index.m3u8".into())
        } else if let Some(rest) = path.strip_prefix("/new") {
            files(rest)
        } else {
            Reply::Status(404)
        }
    }))
    .await;
    let mut puller = HlsPuller::new();
    puller
        .open(spec(server.url("/old/index.m3u8")))
        .await
        .unwrap();
    let c = collect(&mut puller, Duration::from_secs(10)).await;
    assert_eq!(c.video_pts.len(), 60);
}

#[tokio::test]
async fn transient_server_errors_are_retried() {
    let dir = tempfile::tempdir().unwrap();
    if !support::generate_hls(dir.path(), 2, 2, &[]) {
        return;
    }
    let files = support::dir_handler(dir.path());
    let failures = Arc::new(AtomicU32::new(0));
    let failures_srv = failures.clone();
    let server = TestServer::start(Arc::new(move |path: &str| {
        if is_ts(path) && failures_srv.fetch_add(1, Ordering::SeqCst) < 2 {
            Reply::Status(503)
        } else {
            files(path)
        }
    }))
    .await;
    let mut puller = HlsPuller::new();
    puller.open(spec(server.url("/index.m3u8"))).await.unwrap();
    let c = collect(&mut puller, Duration::from_secs(10)).await;
    assert_eq!(c.video_pts.len(), 60);
    assert!(failures.load(Ordering::SeqCst) >= 3);
}

#[tokio::test]
async fn fmp4_vod_plays_with_continuous_timestamps_and_seeks() {
    let dir = tempfile::tempdir().unwrap();
    if !support::generate_hls(dir.path(), 6, 2, &["-hls_segment_type", "fmp4"]) {
        return;
    }
    let playlist = std::fs::read_to_string(dir.path().join("index.m3u8")).unwrap();
    assert!(playlist.contains("EXT-X-MAP"), "{playlist}");
    let server = TestServer::start(support::dir_handler(dir.path())).await;
    let mut puller = HlsPuller::new();
    let info = puller.open(spec(server.url("/index.m3u8"))).await.unwrap();
    let video = info.video.as_ref().unwrap();
    assert_eq!(
        (video.codec, video.width, video.height),
        (Codec::H264, 320, 240)
    );
    let audio = info.audio.as_ref().unwrap();
    assert_eq!(
        (audio.codec, audio.sample_rate, audio.channels),
        (AudioCodec::Aac, 48_000, 2)
    );
    let c = collect(&mut puller, Duration::from_secs(20)).await;
    eprintln!(
        "fMP4 HLS: {} frames, {} keyframes, {} audio packets, DTS {}..{}",
        c.video_pts.len(),
        c.keyframes,
        c.audio,
        c.video_dts[0],
        c.video_dts.last().unwrap()
    );
    assert_eq!(c.video_pts.len(), 180);
    assert_eq!(c.keyframes, 3);
    assert!(c.audio > 250, "audio packets {}", c.audio);
    // One timeline across segments: DTS advances a frame at a time.
    assert_monotonic(&c.video_dts);
    assert!(c.video_dts.windows(2).all(|w| w[1] - w[0] < 40_000));
    let first_pts = *c.video_pts.iter().min().unwrap();

    puller.seek(3_000_000).await.unwrap();
    let c = collect(&mut puller, Duration::from_secs(20)).await;
    // The 2 s segment holds the target; the second before it is
    // decode-only.
    assert_eq!(c.video_pts.len(), 120);
    let from = c.present_from.expect("seek target marker");
    assert!(
        from.abs_diff(first_pts + 3_000_000) < 50_000,
        "{from} vs {first_pts}"
    );
    puller.close().await.unwrap();
}

/// AES-128-CBC with PKCS#7 padding, as an HLS packager writes it.
fn encrypt_aes128(key: &[u8; 16], iv: &[u8; 16], plain: &[u8]) -> Vec<u8> {
    use aes::Aes128;
    use aes::cipher::generic_array::GenericArray;
    use aes::cipher::{BlockEncrypt, KeyInit};

    let pad = 16 - plain.len() % 16;
    let mut padded = plain.to_vec();
    padded.extend(std::iter::repeat_n(pad as u8, pad));
    let cipher = Aes128::new(GenericArray::from_slice(key));
    let mut prev = *iv;
    let mut out = Vec::with_capacity(padded.len());
    for block in padded.chunks_exact(16) {
        let mut buf = GenericArray::clone_from_slice(block);
        buf.iter_mut().zip(prev).for_each(|(b, p)| *b ^= p);
        cipher.encrypt_block(&mut buf);
        prev.copy_from_slice(&buf);
        out.extend_from_slice(&buf);
    }
    out
}

#[tokio::test]
async fn aes128_fmp4_decrypts_the_initialization_segment() {
    let dir = tempfile::tempdir().unwrap();
    if !support::generate_hls(dir.path(), 4, 2, &["-hls_segment_type", "fmp4"]) {
        return;
    }
    // Encrypt the init segment and every media segment, then put the
    // key tag ahead of the map so it covers both.
    let key = [0x5au8; 16];
    let iv = [0x0fu8; 16];
    std::fs::write(dir.path().join("enc.key"), key).unwrap();
    let playlist = std::fs::read_to_string(dir.path().join("index.m3u8")).unwrap();
    let mut rewritten = String::new();
    let mut encrypted = 0;
    for line in playlist.lines() {
        let file = if let Some(attrs) = line.strip_prefix("#EXT-X-MAP:") {
            rewritten.push_str(
                "#EXT-X-KEY:METHOD=AES-128,URI=\"enc.key\",IV=0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f\n",
            );
            attrs
                .split('"')
                .nth(1)
                .map(str::to_owned)
                .expect("EXT-X-MAP URI")
        } else if !line.starts_with('#') && !line.is_empty() {
            line.to_owned()
        } else {
            String::new()
        };
        if !file.is_empty() {
            let path = dir.path().join(&file);
            let plain = std::fs::read(&path).unwrap();
            std::fs::write(&path, encrypt_aes128(&key, &iv, &plain)).unwrap();
            encrypted += 1;
        }
        rewritten.push_str(line);
        rewritten.push('\n');
    }
    assert_eq!(encrypted, 3, "{playlist}");
    std::fs::write(dir.path().join("index.m3u8"), rewritten).unwrap();

    let server = TestServer::start(support::dir_handler(dir.path())).await;
    let mut puller = HlsPuller::new();
    let info = puller.open(spec(server.url("/index.m3u8"))).await.unwrap();
    let video = info.video.as_ref().unwrap();
    assert_eq!((video.width, video.height), (320, 240));
    assert_eq!(info.audio.as_ref().map(|a| a.codec), Some(AudioCodec::Aac));
    let c = collect(&mut puller, Duration::from_secs(20)).await;
    assert_eq!(c.video_pts.len(), 120);
    assert_eq!(c.keyframes, 2);
    puller.close().await.unwrap();
}

#[tokio::test]
async fn close_does_not_hang_when_the_consumer_stopped_reading() {
    let dir = tempfile::tempdir().unwrap();
    if !support::generate_hls(dir.path(), 8, 2, &[]) {
        return;
    }
    let server = TestServer::start(support::dir_handler(dir.path())).await;
    let mut puller = HlsPuller::new();
    puller.open(spec(server.url("/index.m3u8"))).await.unwrap();
    // 8 s of 30 fps video plus audio overflows the packet buffer:
    // the worker ends up parked on a full channel.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let started = Instant::now();
    tokio::time::timeout(Duration::from_secs(5), puller.close())
        .await
        .expect("close hung")
        .unwrap();
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[tokio::test]
async fn seek_discards_queued_packets_and_restarts_the_timeline() {
    let dir = tempfile::tempdir().unwrap();
    if !support::generate_hls(dir.path(), 6, 2, &[]) {
        return;
    }
    let server = TestServer::start(support::dir_handler(dir.path())).await;
    let mut puller = HlsPuller::new();
    puller.open(spec(server.url("/index.m3u8"))).await.unwrap();
    let first_pts = loop {
        if let MediaPacket::Video(v) = puller.next().await.unwrap() {
            break v.pts_dts.0;
        }
    };
    tokio::time::sleep(Duration::from_millis(200)).await;
    puller.seek(4_500_000).await.unwrap();
    // The first packet after the seek is the discontinuity marker;
    // nothing queued before it leaks through.
    assert!(matches!(
        puller.next().await.unwrap(),
        MediaPacket::Discontinuity
    ));
    let c = collect(&mut puller, Duration::from_secs(10)).await;
    // Segment-accurate: playback restarts at the 4 s segment.
    assert_eq!(c.video_pts.len(), 60);
    let min = *c.video_pts.iter().min().unwrap();
    assert!(
        min.abs_diff(first_pts + 4_000_000) < 50_000,
        "{min} vs {first_pts}"
    );
    // The half second between the segment start and the target is
    // marked decode-only.
    let from = c.present_from.expect("seek target marker");
    assert!(
        from.abs_diff(first_pts + 4_500_000) < 50_000,
        "{from} vs {first_pts}"
    );
}

#[tokio::test]
async fn live_starts_near_the_edge_and_follows_reloads() {
    let Some((_dir, server)) = live_server(None).await else {
        return;
    };
    let mut puller = HlsPuller::new();
    let info = puller.open(spec(server.url("/live.m3u8"))).await.unwrap();
    assert!(info.is_live);
    assert_eq!(info.duration_us, None);

    let mut pts = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_secs(3), puller.next())
            .await
            .expect("live puller stalled")
            .unwrap()
        {
            MediaPacket::Video(v) => pts.push(v.pts_dts.1),
            MediaPacket::Audio(_) => {}
            other => panic!("unexpected {other:?}"),
        }
    }
    // Started 3 target durations from the end of a 6-segment window:
    // the first segment fetched is index 3 (seg 3.ts).
    let log = server.log.lock().unwrap().clone();
    let first_segment = log.iter().find(|p| is_ts(p)).unwrap();
    assert_eq!(first_segment, "/index3.ts", "{log:?}");
    // Kept following the live playlist without gaps: DTS advances by
    // one frame (33.3 ms) at a time.
    assert_monotonic(&pts);
    assert!(pts.windows(2).all(|w| w[1] - w[0] < 40_000));
    assert!(pts.len() > 100, "only {} frames in 5 s", pts.len());
    assert!(log.iter().filter(|p| *p == "/live.m3u8").count() >= 3);
    puller.close().await.unwrap();
}

#[tokio::test]
async fn restarted_live_source_is_followed_with_a_discontinuity() {
    let Some((_dir, server)) = live_server(Some(Duration::from_secs(4))).await else {
        return;
    };
    let mut puller = HlsPuller::new();
    puller.open(spec(server.url("/live.m3u8"))).await.unwrap();
    let mut discontinuities = 0;
    let mut frames_after = 0;
    let deadline = Instant::now() + Duration::from_secs(9);
    while Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_secs(3), puller.next())
            .await
            .expect("live puller stalled after restart")
            .unwrap()
        {
            MediaPacket::Discontinuity => discontinuities += 1,
            MediaPacket::Video(_) if discontinuities > 0 => frames_after += 1,
            MediaPacket::Eos => panic!("live stream ended"),
            _ => {}
        }
    }
    assert!(discontinuities >= 1);
    assert!(frames_after > 30, "{frames_after}");
}

/// Whether `path` names a segment (`.ts`, as the fixtures write it).
fn is_ts(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("ts"))
}
