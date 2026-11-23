//! Local loop for AirPlay video (URL playback): this file plays the
//! sender. It checks `/info` advertises video, reverses a connection for
//! the `/event`s, hands the receiver an ffmpeg-generated HLS stream (and
//! then a progressive MP4) served by a local HTTP server, and drives
//! playback with `/playback-info`, `/scrub`, `/rate` and `/stop` while the
//! `StreamManager` pipeline decodes into a headless sink.
//!
//! Skips when ffmpeg is unavailable (set `FERRICAST_FFMPEG`).

#![cfg(all(feature = "airplay", feature = "aac"))]

mod support;

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ferricast::airplay::{AirPlayReceiver, AirPlayReceiverConfig, CredentialStorage};
use ferricast::{
    AdvertiseInfo, Advertiser, CapturedFrame, Codec, DecodedAudio, FrameSink, ManagerEvent,
    ReceiverProtocol, Result, StreamManager,
};
use plist::{Dictionary, Value};
use support::TestServer;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::mpsc;

const SESSION: &str = "7A1E6F60-1D2B-4C8E-9C1F-AB12CD34EF56";
const CLIP_SECS: u32 = 10;

// ── Receiver under test, without mDNS ───────────────────────────

struct NoAdvertiser;

impl Advertiser for NoAdvertiser {
    const PROTOCOL: &'static str = "airplay";
    async fn start(&mut self, _info: AdvertiseInfo) -> Result<()> {
        Ok(())
    }
    async fn stop(&mut self) -> Result<()> {
        Ok(())
    }
    fn is_running(&self) -> bool {
        true
    }
}

struct TestReceiver(AirPlayReceiver);

impl ReceiverProtocol for TestReceiver {
    const PROTOCOL: &'static str = "airplay";
    const SUPPORTED_CODECS: &'static [Codec] = &[Codec::H264];
    type Advertiser = NoAdvertiser;
    type Control = <AirPlayReceiver as ReceiverProtocol>::Control;
    type Puller = <AirPlayReceiver as ReceiverProtocol>::Puller;

    fn create_advertiser(&self) -> NoAdvertiser {
        NoAdvertiser
    }
    fn create_control(&self) -> Result<Self::Control> {
        self.0.create_control()
    }
    fn create_puller(&self) -> Result<Self::Puller> {
        self.0.create_puller()
    }
    fn advertise_info(&self) -> AdvertiseInfo {
        self.0.advertise_info()
    }
}

// ── Headless sink ───────────────────────────────────────────────

#[derive(Default)]
struct SinkLog {
    video_frames: usize,
    audio_chunks: usize,
    created: usize,
    dropped: usize,
}

struct HeadlessSink(Arc<Mutex<SinkLog>>);

impl Drop for HeadlessSink {
    fn drop(&mut self) {
        self.0.lock().unwrap().dropped += 1;
    }
}

#[async_trait::async_trait]
impl FrameSink for HeadlessSink {
    async fn push_video(&mut self, _frame: CapturedFrame) -> Result<()> {
        self.0.lock().unwrap().video_frames += 1;
        Ok(())
    }
    async fn push_audio(&mut self, _audio: DecodedAudio) -> Result<()> {
        self.0.lock().unwrap().audio_chunks += 1;
        Ok(())
    }
}

// ── A minimal AirPlay video sender ──────────────────────────────

struct Reply {
    status: u16,
    content_type: Option<String>,
    body: Vec<u8>,
}

impl Reply {
    fn plist(&self) -> Dictionary {
        plist::from_bytes::<Value>(&self.body)
            .unwrap_or_else(|e| {
                panic!(
                    "not a plist ({e}): {:?}",
                    String::from_utf8_lossy(&self.body)
                )
            })
            .into_dictionary()
            .expect("plist dictionary")
    }
}

/// Reads one HTTP message head + body (by `Content-Length`) from `buf`
/// and `stream`; returns the head lines and the body.
async fn read_message(stream: &mut TcpStream, buf: &mut Vec<u8>) -> Option<(Vec<String>, Vec<u8>)> {
    loop {
        if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..end]).into_owned();
            let lines: Vec<String> = head.split("\r\n").map(str::to_owned).collect();
            let len = lines
                .iter()
                .filter_map(|l| l.split_once(':'))
                .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                .map_or(0, |(_, v)| v.trim().parse::<usize>().unwrap());
            if buf.len() >= end + 4 + len {
                let body = buf[end + 4..end + 4 + len].to_vec();
                buf.drain(..end + 4 + len);
                return Some((lines, body));
            }
        }
        let mut chunk = [0_u8; 8192];
        let n = stream.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
}

struct Sender {
    stream: TcpStream,
    buf: Vec<u8>,
}

impl Sender {
    async fn connect(addr: SocketAddr) -> Self {
        Self {
            stream: TcpStream::connect(addr).await.unwrap(),
            buf: Vec::new(),
        }
    }

    async fn request(
        &mut self,
        method: &str,
        path: &str,
        content_type: Option<&str>,
        body: &[u8],
    ) -> Reply {
        let mut head = format!(
            "{method} {path} HTTP/1.1\r\nUser-Agent: PigeonCastTest/1.0\r\nX-Apple-Session-ID: {SESSION}\r\nContent-Length: {}\r\n",
            body.len()
        );
        if let Some(ct) = content_type {
            head.push_str(&format!("Content-Type: {ct}\r\n"));
        }
        head.push_str("\r\n");
        self.stream.write_all(head.as_bytes()).await.unwrap();
        self.stream.write_all(body).await.unwrap();
        let (lines, body) = read_message(&mut self.stream, &mut self.buf)
            .await
            .unwrap_or_else(|| panic!("{method} {path}: connection closed"));
        let status = lines[0]
            .split(' ')
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("bad status line {:?}", lines[0]));
        let content_type = lines
            .iter()
            .filter_map(|l| l.split_once(':'))
            .find(|(k, _)| k.eq_ignore_ascii_case("content-type"))
            .map(|(_, v)| v.trim().to_owned());
        Reply {
            status,
            content_type,
            body,
        }
    }

    async fn get(&mut self, path: &str) -> Reply {
        self.request("GET", path, None, &[]).await
    }

    async fn post(&mut self, path: &str) -> Reply {
        let reply = self.request("POST", path, None, &[]).await;
        assert_eq!(reply.status, 200, "POST {path}");
        reply
    }

    /// `/playback-info`, `None` for the empty "nothing plays" answer.
    async fn playback_info(&mut self) -> Option<Dictionary> {
        let reply = self.get("/playback-info").await;
        assert_eq!(reply.status, 200);
        (!reply.body.is_empty()).then(|| {
            assert_eq!(
                reply.content_type.as_deref(),
                Some("text/x-apple-plist+xml")
            );
            reply.plist()
        })
    }
}

fn real(dict: &Dictionary, key: &str) -> f64 {
    dict.get(key)
        .and_then(Value::as_real)
        .unwrap_or_else(|| panic!("no real {key} in {dict:?}"))
}

/// Opens the reverse connection; the `state` of every `/event` goes to
/// the returned channel.
async fn reverse(addr: SocketAddr) -> mpsc::UnboundedReceiver<String> {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let head = format!(
        "POST /reverse HTTP/1.1\r\nUpgrade: PTTH/1.0\r\nConnection: Upgrade\r\nX-Apple-Purpose: event\r\nContent-Length: 0\r\nUser-Agent: PigeonCastTest/1.0\r\nX-Apple-Session-ID: {SESSION}\r\n\r\n"
    );
    stream.write_all(head.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    let (lines, _) = read_message(&mut stream, &mut buf).await.unwrap();
    assert!(lines[0].starts_with("HTTP/1.1 101"), "{lines:?}");
    assert!(lines.iter().any(|l| l == "Upgrade: PTTH/1.0"), "{lines:?}");
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Some((lines, body)) = read_message(&mut stream, &mut buf).await {
            assert!(lines[0].starts_with("POST /event HTTP/1.1"), "{lines:?}");
            assert!(
                lines
                    .iter()
                    .any(|l| l == &format!("X-Apple-Session-ID: {SESSION}")),
                "{lines:?}"
            );
            let dict = plist::from_bytes::<Value>(&body)
                .unwrap()
                .into_dictionary()
                .unwrap();
            assert_eq!(
                dict.get("category").and_then(Value::as_string),
                Some("video")
            );
            let state = dict
                .get("state")
                .and_then(Value::as_string)
                .unwrap()
                .to_owned();
            if tx.send(state).is_err() {
                return;
            }
            let ok = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
            if stream.write_all(ok).await.is_err() {
                return;
            }
        }
    });
    rx
}

async fn wait_state(events: &mut mpsc::UnboundedReceiver<String>, want: &str) {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let state = tokio::time::timeout_at(deadline.into(), events.recv())
            .await
            .unwrap_or_else(|_| panic!("timed out waiting for the {want} event"))
            .expect("reverse connection closed");
        if state == want {
            return;
        }
        assert_ne!(state, "stopped", "stopped while waiting for {want}");
    }
}

async fn wait_manager(
    events: &mut mpsc::Receiver<ManagerEvent>,
    what: &str,
    pick: impl Fn(&ManagerEvent) -> bool,
) {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let ev = tokio::time::timeout_at(deadline.into(), events.recv())
            .await
            .unwrap_or_else(|_| panic!("timed out waiting for {what}"))
            .expect("event channel closed");
        if let ManagerEvent::ReceiverError { message, .. } = &ev {
            panic!("receiver error while waiting for {what}: {message}");
        }
        if pick(&ev) {
            return;
        }
    }
}

fn bplist(entries: &[(&str, Value)]) -> Vec<u8> {
    let dict: Dictionary = entries
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect();
    let mut out = Vec::new();
    plist::to_writer_binary(&mut out, &Value::Dictionary(dict)).unwrap();
    out
}

/// How the test hands the URL over.
enum PlayBody {
    /// iOS-style binary plist, starting at the beginning.
    Plist,
    /// Android-style text parameters starting at `Start-Position-Seconds`.
    Text { start_secs: f64 },
}

struct Rig {
    addr: SocketAddr,
    log: Arc<Mutex<SinkLog>>,
    events: mpsc::Receiver<ManagerEvent>,
    _manager: ferricast::StreamManager,
}

async fn rig() -> Rig {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,ferricast_airplay=info,ferricast=info".into()),
        )
        .with_test_writer()
        .try_init();
    let receiver = AirPlayReceiver::new(
        AirPlayReceiverConfig {
            name: "ferricast video loop".into(),
            credentials: CredentialStorage::Ephemeral,
            ..Default::default()
        }
        .with_ephemeral_ports(),
    )
    .await;
    let log = Arc::new(Mutex::new(SinkLog::default()));
    let log_factory = log.clone();
    let (mut manager, events) = StreamManager::builder()
        .with_receiver(TestReceiver(receiver.clone()))
        .with_h264_decoder()
        .with_aac_decoder()
        .set_sink_factory(move |_remote, info| {
            assert!(info.video.is_some(), "media without video");
            log_factory.lock().unwrap().created += 1;
            let sink = HeadlessSink(log_factory.clone());
            async move { Ok(Box::new(sink) as Box<dyn FrameSink>) }
        })
        .build_with_events();
    manager.start_receivers().await.unwrap();
    let port = receiver.bound_rtsp_port().expect("control port bound");
    Rig {
        addr: SocketAddr::from(([127, 0, 0, 1], port)),
        log,
        events,
        _manager: manager,
    }
}

async fn play_and_drive(rig: &mut Rig, url: &str, body: PlayBody) {
    // ── Discovery: /info and /server-info ──
    let mut info_conn = Sender::connect(rig.addr).await;
    let info = info_conn.get("/info").await;
    assert_eq!(info.status, 200);
    let features = info
        .plist()
        .get("features")
        .and_then(Value::as_unsigned_integer)
        .expect("features");
    assert_eq!(features & 0x1, 0x1, "Video bit missing: {features:#x}");
    assert_eq!(
        features & 0x10,
        0x10,
        "VideoHTTPLiveStreams bit missing: {features:#x}"
    );
    assert_eq!(
        features & 0x80,
        0x80,
        "screen mirroring bit lost: {features:#x}"
    );
    let server_info = info_conn.get("/server-info").await.plist();
    assert!(server_info.contains_key("deviceid"), "{server_info:?}");

    // ── Reverse connection, then /play on another one ──
    let mut events = reverse(rig.addr).await;
    let mut sender = Sender::connect(rig.addr).await;
    assert!(
        sender.playback_info().await.is_none(),
        "something plays already"
    );
    let start_secs = match body {
        PlayBody::Plist => {
            let play = bplist(&[
                ("Content-Location", Value::String(url.into())),
                ("Start-Position", Value::Real(0.0)),
                (
                    "uuid",
                    Value::String("2B7D6A1E-0000-4000-8000-000000000001".into()),
                ),
                ("clientProcName", Value::String("PigeonCast".into())),
            ]);
            let reply = sender
                .request(
                    "POST",
                    "/play",
                    Some("application/x-apple-binary-plist"),
                    &play,
                )
                .await;
            assert_eq!(reply.status, 200);
            0.0
        }
        PlayBody::Text { start_secs } => {
            let play =
                format!("Content-Location: {url}\r\nStart-Position-Seconds: {start_secs}\r\n");
            let reply = sender
                .request("POST", "/play", Some("text/parameters"), play.as_bytes())
                .await;
            assert_eq!(reply.status, 200);
            start_secs
        }
    };
    wait_state(&mut events, "loading").await;
    wait_state(&mut events, "playing").await;

    // ── Ready, duration right, position follows wall time ──
    let info = sender
        .playback_info()
        .await
        .expect("playback info while playing");
    assert_eq!(
        info.get("readyToPlay").and_then(Value::as_boolean),
        Some(true)
    );
    assert_eq!(real(&info, "rate"), 1.0);
    let duration = real(&info, "duration");
    assert!(
        (duration - f64::from(CLIP_SECS)).abs() < 0.6,
        "duration {duration} for a {CLIP_SECS} s clip"
    );
    let p0 = real(&info, "position");
    assert!(
        (p0 - start_secs).abs() < 1.0,
        "start position {p0}, wanted {start_secs}"
    );
    let t0 = Instant::now();
    tokio::time::sleep(Duration::from_secs(2)).await;
    let p1 = real(&sender.playback_info().await.unwrap(), "position");
    let wall = t0.elapsed().as_secs_f64();
    assert!(
        ((p1 - p0) - wall).abs() < 0.5,
        "position moved {} s in {wall} s",
        p1 - p0
    );
    let frames = rig.log.lock().unwrap().video_frames;
    assert!(
        frames > 20,
        "only {frames} frames decoded in 2 s of playback"
    );

    // ── Scrub to 3 s ──
    sender.post("/scrub?position=3.000000").await;
    let scrub = sender.get("/scrub").await;
    assert_eq!(scrub.content_type.as_deref(), Some("text/parameters"));
    let text = String::from_utf8(scrub.body).unwrap();
    let position: f64 = text
        .lines()
        .find_map(|l| l.strip_prefix("position: "))
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no position in {text:?}"));
    assert!(
        (3.0..3.6).contains(&position),
        "position {position} after scrubbing to 3 s"
    );

    // ── Pause: the pipeline's own position, frozen ──
    sender.post("/rate?value=0.000000").await;
    wait_state(&mut events, "paused").await;
    // The paused report re-bases the position on the pump's.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let paused = sender.playback_info().await.unwrap();
    assert_eq!(real(&paused, "rate"), 0.0);
    let held = real(&paused, "position");
    assert!(
        (3.0..4.5).contains(&held),
        "paused at {held}, scrubbed to 3 s"
    );
    tokio::time::sleep(Duration::from_secs(1)).await;
    let still = real(&sender.playback_info().await.unwrap(), "position");
    assert!(
        (still - held).abs() < 0.05,
        "position moved while paused: {held} -> {still}"
    );

    // ── Resume ──
    sender.post("/rate?value=1.000000").await;
    wait_state(&mut events, "playing").await;
    tokio::time::sleep(Duration::from_secs(1)).await;
    let resumed = real(&sender.playback_info().await.unwrap(), "position");
    assert!(
        resumed > still + 0.5,
        "not playing after resume: {still} -> {resumed}"
    );

    // ── Properties senders set before playing ──
    let reply = sender
        .request(
            "PUT",
            "/setProperty?forwardEndTime",
            Some("application/x-apple-binary-plist"),
            &bplist(&[("value", Value::Real(0.0))]),
        )
        .await;
    assert_eq!(reply.status, 200);

    // ── Stop: the sink closes, the pipeline session ends ──
    sender.post("/stop").await;
    wait_state(&mut events, "stopped").await;
    wait_manager(&mut rig.events, "ReceiverStopped", |ev| {
        matches!(ev, ManagerEvent::ReceiverStopped { .. })
    })
    .await;
    {
        let log = rig.log.lock().unwrap();
        assert_eq!(log.created, 1, "one sink per /play");
        assert_eq!(log.dropped, 1, "the sink stays open after /stop");
        assert!(log.audio_chunks > 0, "no audio decoded");
    }
    assert!(
        sender.playback_info().await.is_none(),
        "still playing after /stop"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn plays_hls_by_url() {
    let dir = tempfile::tempdir().unwrap();
    if !support::generate_hls(dir.path(), CLIP_SECS, 2, &[]) {
        return;
    }
    let server = TestServer::start(support::dir_handler(dir.path())).await;
    let mut rig = rig().await;
    play_and_drive(&mut rig, &server.url("/index.m3u8"), PlayBody::Plist).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn plays_progressive_mp4_by_url() {
    let dir = tempfile::tempdir().unwrap();
    if !support::generate_progressive(dir.path(), "clip.mp4", CLIP_SECS, support::H264_AAC, &[]) {
        return;
    }
    let server = TestServer::start(support::dir_handler(dir.path())).await;
    let mut rig = rig().await;
    play_and_drive(
        &mut rig,
        &server.url("/clip.mp4"),
        PlayBody::Text { start_secs: 2.0 },
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn refuses_youtube_playlists_and_bad_bodies() {
    let mut rig = rig().await;
    let mut sender = Sender::connect(rig.addr).await;
    let play = bplist(&[
        (
            "Content-Location",
            Value::String("mlhls://localhost/master.m3u8".into()),
        ),
        ("Start-Position-Seconds", Value::Real(0.0)),
    ]);
    let reply = sender
        .request(
            "POST",
            "/play",
            Some("application/x-apple-binary-plist"),
            &play,
        )
        .await;
    assert_eq!(reply.status, 400);
    let reply = sender
        .request(
            "POST",
            "/play",
            Some("text/parameters"),
            b"Start-Position: 0\r\n",
        )
        .await;
    assert_eq!(reply.status, 400);
    assert!(sender.playback_info().await.is_none());
    assert!(
        rig.events.try_recv().is_err(),
        "a refused /play reached the pipeline"
    );
}
