//! End to end: a CASTV2 sender (minimal TLS client in this file)
//! drives the Chromecast receiver through the `StreamManager`, which
//! pulls an ffmpeg-generated HLS stream from a local HTTP server,
//! decodes it and paces it into a headless sink.
//!
//! Skips when ffmpeg is unavailable (set `FERRICAST_FFMPEG`).

#![cfg(feature = "chromecast")]

mod support;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::BytesMut;
use ferricast::{
    AdvertiseInfo, Advertiser, CapturedFrame, Codec, DecodedAudio, FrameSink, ManagerEvent,
    ReceiverProtocol, Result, StreamManager,
};
use ferricast_chromecast::ChromecastReceiver;
use ferricast_chromecast::castv2::{CastMessage, namespace};
use serde_json::{Value, json};
use support::TestServer;
use tokio::io::{AsyncReadExt, AsyncWriteExt, ReadHalf, WriteHalf};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio_rustls::client::TlsStream;

// ── Receiver under test, without mDNS ───────────────────────────

struct NoAdvertiser;

impl Advertiser for NoAdvertiser {
    const PROTOCOL: &'static str = "chromecast";
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

/// The real Chromecast receiver protocol with advertising disabled,
/// so the test doesn't announce itself on the LAN.
struct TestReceiver(ChromecastReceiver);

impl ReceiverProtocol for TestReceiver {
    const PROTOCOL: &'static str = "chromecast";
    const SUPPORTED_CODECS: &'static [Codec] = &[Codec::H264];
    type Advertiser = NoAdvertiser;
    type Control = <ChromecastReceiver as ReceiverProtocol>::Control;
    type Puller = <ChromecastReceiver as ReceiverProtocol>::Puller;

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
    /// (arrival, PTS) per video frame.
    video: Vec<(Instant, u64)>,
    audio_chunks: usize,
    /// (arrival, PTS, duration) per audio chunk.
    audio: Vec<(Instant, u64, Duration)>,
    /// Whether each audio chunk is pure silence (the pipeline's lead-in).
    audio_silent: Vec<bool>,
    /// Audio chunks received when each `flush_audio` arrived.
    flushes: Vec<usize>,
    /// (sample rate, channels) per audio chunk.
    audio_shapes: Vec<(u32, u16)>,
    gains: Vec<f32>,
    sinks_created: usize,
    sinks_dropped: usize,
}

struct HeadlessSink(Arc<Mutex<SinkLog>>);

impl Drop for HeadlessSink {
    fn drop(&mut self) {
        self.0.lock().unwrap().sinks_dropped += 1;
    }
}

#[async_trait::async_trait]
impl FrameSink for HeadlessSink {
    async fn push_video(&mut self, frame: CapturedFrame) -> Result<()> {
        let pts = frame.timestamp_us();
        self.0.lock().unwrap().video.push((Instant::now(), pts));
        Ok(())
    }
    async fn push_audio(&mut self, audio: DecodedAudio) -> Result<()> {
        let frames = audio.pcm.len() / usize::from(audio.channels.max(1));
        let duration = Duration::from_secs_f64(
            f64::from(u32::try_from(frames).unwrap()) / f64::from(audio.sample_rate.max(1)),
        );
        let mut log = self.0.lock().unwrap();
        log.audio_chunks += 1;
        log.audio
            .push((Instant::now(), audio.timestamp_us, duration));
        log.audio_silent.push(audio.pcm.iter().all(|&s| s == 0));
        log.audio_shapes.push((audio.sample_rate, audio.channels));
        Ok(())
    }
    async fn flush_audio(&mut self) -> Result<()> {
        let mut log = self.0.lock().unwrap();
        let at = log.audio.len();
        log.flushes.push(at);
        Ok(())
    }
    async fn set_volume(&mut self, gain: f32) -> Result<()> {
        self.0.lock().unwrap().gains.push(gain);
        Ok(())
    }
}

struct Receiver {
    manager: StreamManager,
    events: mpsc::Receiver<ManagerEvent>,
    log: Arc<Mutex<SinkLog>>,
    port: u16,
}

impl Receiver {
    async fn start() -> Self {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let log = Arc::new(Mutex::new(SinkLog::default()));
        let log_factory = log.clone();
        let builder = StreamManager::builder()
            .with_receiver(TestReceiver(ChromecastReceiver::with_port("e2e", port)))
            .with_h264_decoder()
            .with_aac_decoder();
        #[cfg(feature = "vp9-decode")]
        let builder = builder.register_video_decoder::<ferricast::decoder::Vp9Decoder>();
        #[cfg(feature = "opus-decode")]
        let builder = builder.register_audio_decoder::<ferricast::decoder::OpusDecoder>();
        let builder = builder.register_audio_decoder::<ferricast::decoder::PcmDecoder>();
        let (mut manager, events) = builder
            .set_sink_factory(move |_remote, _info| {
                log_factory.lock().unwrap().sinks_created += 1;
                let sink = HeadlessSink(log_factory.clone());
                async move { Ok(Box::new(sink) as Box<dyn FrameSink>) }
            })
            .build_with_events();
        manager.start_receivers().await.unwrap();
        Self {
            manager,
            events,
            log,
            port,
        }
    }

    async fn wait_event(&mut self, what: &str, pred: impl Fn(&ManagerEvent) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let ev = tokio::time::timeout_at(deadline.into(), self.events.recv())
                .await
                .unwrap_or_else(|_| panic!("timed out waiting for {what}"))
                .expect("event channel closed");
            if pred(&ev) {
                return;
            }
        }
    }
}

// ── Minimal CASTV2 sender ───────────────────────────────────────

#[derive(Debug)]
struct NoVerify;

impl rustls::client::danger::ServerCertVerifier for NoVerify {
    fn verify_server_cert(
        &self,
        _: &rustls::pki_types::CertificateDer<'_>,
        _: &[rustls::pki_types::CertificateDer<'_>],
        _: &rustls::pki_types::ServerName<'_>,
        _: &[u8],
        _: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        _: &[u8],
        _: &rustls::pki_types::CertificateDer<'_>,
        _: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }
    fn verify_tls13_signature(
        &self,
        _: &[u8],
        _: &rustls::pki_types::CertificateDer<'_>,
        _: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }
    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

struct Sender {
    reader: ReadHalf<TlsStream<TcpStream>>,
    writer: WriteHalf<TlsStream<TcpStream>>,
    buf: BytesMut,
    id: String,
    next_request: i64,
    transport_id: Option<String>,
}

impl Sender {
    async fn connect(port: u16, id: &str) -> Self {
        let config = rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(NoVerify))
            .with_no_client_auth();
        let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
        let tcp = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        let tls = connector
            .connect("localhost".try_into().unwrap(), tcp)
            .await
            .unwrap();
        let (reader, writer) = tokio::io::split(tls);
        let mut s = Self {
            reader,
            writer,
            buf: BytesMut::new(),
            id: id.to_string(),
            next_request: 1,
            transport_id: None,
        };
        s.send_raw(
            "receiver-0",
            namespace::CONNECTION,
            json!({"type": "CONNECT"}),
        )
        .await;
        s
    }

    async fn write(&mut self, msg: CastMessage) {
        let bytes = msg.encode_length_prefixed().unwrap();
        self.writer.write_all(&bytes).await.unwrap();
        self.writer.flush().await.unwrap();
    }

    async fn send_raw(&mut self, dest: &str, ns: &str, payload: Value) {
        let msg = CastMessage::new_json(&self.id, dest, ns, &payload).unwrap();
        self.write(msg).await;
    }

    /// Sends a request with a fresh `requestId` and returns the id.
    async fn request(&mut self, ns: &str, mut payload: Value) -> i64 {
        let id = self.next_request;
        self.next_request += 1;
        payload["requestId"] = json!(id);
        let dest = if ns == namespace::MEDIA {
            self.transport_id.clone().expect("LAUNCH first")
        } else {
            "receiver-0".to_string()
        };
        self.send_raw(&dest, ns, payload).await;
        id
    }

    async fn recv(&mut self, timeout: Duration) -> Option<CastMessage> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(msg) = CastMessage::decode_length_prefixed(&mut self.buf).unwrap() {
                return Some(msg);
            }
            let read =
                tokio::time::timeout_at(deadline.into(), self.reader.read_buf(&mut self.buf))
                    .await
                    .ok()?;
            if read.ok()? == 0 {
                return None;
            }
        }
    }

    /// Next JSON message on `ns` matching `pred`; panics on timeout.
    async fn expect(&mut self, ns: &str, what: &str, pred: impl Fn(&Value) -> bool) -> Value {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let msg = self
                .recv(left)
                .await
                .unwrap_or_else(|| panic!("no {what} before timeout / disconnect"));
            if msg.namespace != ns {
                continue;
            }
            let Some(text) = msg.payload_utf8.as_deref() else {
                continue;
            };
            let v: Value = serde_json::from_str(text).unwrap();
            if pred(&v) {
                return v;
            }
        }
    }

    async fn launch(&mut self) {
        let id = self
            .request(
                namespace::RECEIVER,
                json!({"type": "LAUNCH", "appId": "CC1AD845"}),
            )
            .await;
        let status = self
            .expect(namespace::RECEIVER, "RECEIVER_STATUS", |v| {
                v["requestId"] == id
            })
            .await;
        let transport = status["status"]["applications"][0]["transportId"]
            .as_str()
            .expect("transportId")
            .to_string();
        self.send_raw(
            &transport,
            namespace::CONNECTION,
            json!({"type": "CONNECT"}),
        )
        .await;
        self.transport_id = Some(transport);
    }

    async fn load(&mut self, url: &str, content_type: &str) -> Value {
        let id = self
            .request(
                namespace::MEDIA,
                json!({
                    "type": "LOAD",
                    "autoplay": true,
                    "media": {"contentId": url, "contentType": content_type, "streamType": "BUFFERED"},
                }),
            )
            .await;
        self.expect(namespace::MEDIA, "LOAD reply", |v| v["requestId"] == id)
            .await
    }
}

fn hls_fixture(seconds: u32) -> Option<(tempfile::TempDir, ())> {
    let dir = tempfile::tempdir().unwrap();
    support::generate_hls(dir.path(), seconds, 2, &[]).then_some((dir, ()))
}

// ── Tests ───────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn vod_plays_in_real_time_and_reports_finished() {
    let Some((dir, ())) = hls_fixture(6) else {
        return;
    };
    let server = TestServer::start(support::dir_handler(dir.path())).await;
    let mut rx = Receiver::start().await;
    let mut sender = Sender::connect(rx.port, "sender-e2e").await;
    sender.launch().await;

    let started = Instant::now();
    let reply = sender
        .load(&server.url("/index.m3u8"), "application/x-mpegurl")
        .await;
    assert_eq!(reply["type"], "MEDIA_STATUS", "{reply}");
    assert_eq!(reply["status"][0]["playerState"], "PLAYING", "{reply}");
    let session_id = reply["status"][0]["mediaSessionId"].clone();

    // Unsolicited IDLE/FINISHED when the pump reaches the end.
    let finished = sender
        .expect(namespace::MEDIA, "IDLE FINISHED", |v| {
            v["status"][0]["playerState"] == "IDLE"
        })
        .await;
    assert_eq!(
        finished["status"][0]["idleReason"], "FINISHED",
        "{finished}"
    );
    assert_eq!(finished["status"][0]["mediaSessionId"], session_id);
    let total = started.elapsed();

    let log = std::mem::take(&mut *rx.log.lock().unwrap());
    let frames = &log.video;
    let pts: Vec<u64> = frames.iter().map(|(_, p)| *p).collect();
    assert!(
        pts.windows(2).all(|w| w[0] < w[1]),
        "PTS not strictly increasing"
    );
    let pts_span = Duration::from_micros(pts[pts.len() - 1] - pts[0]);
    let wall_span = frames[frames.len() - 1].0 - frames[0].0;
    let drift = wall_span.as_secs_f64() - pts_span.as_secs_f64();
    eprintln!(
        "E2E VOD: {} frames, {} audio chunks, PTS span {:.3}s, wall span {:.3}s, \
         drift {:+.3}s, LOAD→FINISHED {:.3}s",
        frames.len(),
        log.audio_chunks,
        pts_span.as_secs_f64(),
        wall_span.as_secs_f64(),
        drift,
        total.as_secs_f64()
    );
    assert!(frames.len() >= 170, "only {} frames shown", frames.len());
    assert!(drift.abs() < 0.25, "paced {drift:+.3}s off real time");
    // Audio: the first chunk is the priming silence; the rest covers
    // the whole stream and arrives AUDIO_LEAD (200 ms) ahead of the
    // video frame with the same PTS.
    let audio_total: Duration = log.audio.iter().skip(1).map(|a| a.2).sum();
    let (t0, p0) = frames[0];
    let offsets_ms: Vec<f64> = log
        .audio
        .iter()
        .skip(1)
        .map(|(at, pts, _)| {
            let video_time = if *pts >= p0 {
                t0 + Duration::from_micros(pts - p0)
            } else {
                t0.checked_sub(Duration::from_micros(p0 - pts)).unwrap()
            };
            (video_time.saturating_duration_since(*at).as_secs_f64()
                - at.saturating_duration_since(video_time).as_secs_f64())
                * 1000.0
        })
        .collect();
    let silence = log.audio[0].2;
    eprintln!(
        "E2E audio: {} chunks, {:.3}s of audio after {:.0} ms priming silence, lead vs video \
         min {:.1} ms max {:.1} ms",
        log.audio.len(),
        audio_total.as_secs_f64(),
        silence.as_secs_f64() * 1000.0,
        offsets_ms.iter().copied().fold(f64::MAX, f64::min),
        offsets_ms.iter().copied().fold(f64::MIN, f64::max),
    );
    assert!(
        (audio_total.as_secs_f64() - 6.0).abs() < 0.2,
        "{audio_total:?}"
    );
    assert!((silence.as_secs_f64() - 0.2).abs() < 0.01);
    // First chunk may be pushed on anchoring; all others ~200 ms early.
    assert!(
        offsets_ms
            .iter()
            .skip(1)
            .all(|ms| (150.0..250.0).contains(ms)),
        "{offsets_ms:?}"
    );
    rx.manager.stop_receivers().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_load_keeps_the_connection_and_volume_reaches_the_sink() {
    let Some((dir, ())) = hls_fixture(8) else {
        return;
    };
    let server = TestServer::start(support::dir_handler(dir.path())).await;
    let mut rx = Receiver::start().await;
    let mut sender = Sender::connect(rx.port, "sender-e2e").await;
    sender.launch().await;

    // Missing playlist → LOAD_FAILED, connection stays usable.
    let reply = sender
        .load(&server.url("/missing.m3u8"), "application/x-mpegurl")
        .await;
    assert_eq!(reply["type"], "LOAD_FAILED", "{reply}");
    // DASH is refused up front.
    let reply = sender
        .load(&server.url("/manifest.mpd"), "application/dash+xml")
        .await;
    assert_eq!(reply["type"], "LOAD_FAILED", "{reply}");
    assert_eq!(
        rx.log.lock().unwrap().sinks_created,
        0,
        "no sink for failed loads"
    );
    let id = sender
        .request(namespace::RECEIVER, json!({"type": "GET_STATUS"}))
        .await;
    sender
        .expect(
            namespace::RECEIVER,
            "RECEIVER_STATUS after LOAD_FAILED",
            |v| v["requestId"] == id,
        )
        .await;

    // A good LOAD on the same connection plays.
    let reply = sender
        .load(&server.url("/index.m3u8"), "application/x-mpegurl")
        .await;
    assert_eq!(reply["status"][0]["playerState"], "PLAYING", "{reply}");

    // Receiver-level volume, then mute.
    let id = sender
        .request(
            namespace::RECEIVER,
            json!({"type": "SET_VOLUME", "volume": {"level": 0.3}}),
        )
        .await;
    let status = sender
        .expect(namespace::RECEIVER, "RECEIVER_STATUS for SET_VOLUME", |v| {
            v["requestId"] == id
        })
        .await;
    assert!((status["status"]["volume"]["level"].as_f64().unwrap() - 0.3).abs() < 1e-6);
    let id = sender
        .request(
            namespace::MEDIA,
            json!({"type": "SET_VOLUME", "volume": {"muted": true}}),
        )
        .await;
    let media = sender
        .expect(namespace::MEDIA, "MEDIA_STATUS for SET_VOLUME", |v| {
            v["requestId"] == id
        })
        .await;
    assert_eq!(media["status"][0]["volume"]["muted"], true, "{media}");
    tokio::time::sleep(Duration::from_millis(300)).await;
    let gains = rx.log.lock().unwrap().gains.clone();
    eprintln!("E2E volume: sink gains {gains:?}");
    assert_eq!(gains.first().copied(), Some(1.0));
    assert!(gains.iter().any(|g| (g - 0.3).abs() < 1e-6), "{gains:?}");
    assert_eq!(gains.last().copied(), Some(0.0), "{gains:?}");

    // Device-auth challenges get an error back, not a dropped
    // connection.
    let challenge = CastMessage::new_binary(
        "sender-e2e",
        "receiver-0",
        "urn:x-cast:com.google.cast.tp.deviceauth",
        vec![0x0a, 0x00],
    );
    sender.write(challenge).await;
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let msg = sender
            .recv(deadline.saturating_duration_since(Instant::now()))
            .await
            .expect("no device-auth reply");
        if msg.namespace.ends_with("deviceauth") {
            assert!(msg.payload_binary.is_some());
            break;
        }
    }
    sender
        .send_raw("receiver-0", namespace::HEARTBEAT, json!({"type": "PING"}))
        .await;
    sender
        .expect(namespace::HEARTBEAT, "PONG", |v| v["type"] == "PONG")
        .await;
    rx.manager.stop_receivers().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sender_disconnect_stops_the_pump() {
    let Some((dir, ())) = hls_fixture(20) else {
        return;
    };
    let server = TestServer::start(support::dir_handler(dir.path())).await;
    let mut rx = Receiver::start().await;
    let mut sender = Sender::connect(rx.port, "sender-e2e").await;
    sender.launch().await;
    let reply = sender
        .load(&server.url("/index.m3u8"), "application/x-mpegurl")
        .await;
    assert_eq!(reply["status"][0]["playerState"], "PLAYING", "{reply}");
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(!rx.log.lock().unwrap().video.is_empty());

    let disconnected = Instant::now();
    drop(sender);
    rx.wait_event("ReceiverStopped", |e| {
        matches!(e, ManagerEvent::ReceiverStopped { .. })
    })
    .await;
    let stop_latency = disconnected.elapsed();
    let (frames_then, dropped) = {
        let log = rx.log.lock().unwrap();
        (log.video.len(), log.sinks_dropped)
    };
    tokio::time::sleep(Duration::from_millis(500)).await;
    let frames_later = rx.log.lock().unwrap().video.len();
    eprintln!(
        "E2E disconnect: stopped after {:.3}s, sink dropped={dropped}, frames {frames_then}→{frames_later}",
        stop_latency.as_secs_f64()
    );
    assert_eq!(dropped, 1, "sink not released");
    assert_eq!(frames_then, frames_later, "pump kept running");
    rx.manager.stop_receivers().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn second_sender_is_served_and_its_load_interrupts_the_first() {
    let Some((dir, ())) = hls_fixture(20) else {
        return;
    };
    let server = TestServer::start(support::dir_handler(dir.path())).await;
    let mut rx = Receiver::start().await;
    let mut first = Sender::connect(rx.port, "sender-a").await;
    first.launch().await;
    let reply = first
        .load(&server.url("/index.m3u8"), "application/x-mpegurl")
        .await;
    assert_eq!(reply["status"][0]["playerState"], "PLAYING", "{reply}");

    // A second connection is answered while the first plays.
    let mut second = Sender::connect(rx.port, "sender-b").await;
    let id = second
        .request(
            namespace::RECEIVER,
            json!({"type": "GET_APP_AVAILABILITY", "appId": ["CC1AD845", "233637DE"]}),
        )
        .await;
    let avail = second
        .expect(namespace::RECEIVER, "GET_APP_AVAILABILITY", |v| {
            v["requestId"] == id
        })
        .await;
    assert_eq!(
        avail["availability"]["CC1AD845"], "APP_AVAILABLE",
        "{avail}"
    );
    assert_eq!(
        avail["availability"]["233637DE"], "APP_UNAVAILABLE",
        "{avail}"
    );

    // Its LOAD takes over; the first sender sees INTERRUPTED.
    second.launch().await;
    let reply = second
        .load(&server.url("/index.m3u8"), "application/x-mpegurl")
        .await;
    assert_eq!(reply["status"][0]["playerState"], "PLAYING", "{reply}");
    let interrupted = first
        .expect(namespace::MEDIA, "IDLE INTERRUPTED", |v| {
            v["status"][0]["playerState"] == "IDLE"
        })
        .await;
    assert_eq!(
        interrupted["status"][0]["idleReason"], "INTERRUPTED",
        "{interrupted}"
    );
    tokio::time::sleep(Duration::from_millis(300)).await;
    let counts = {
        let log = rx.log.lock().unwrap();
        (log.sinks_created, log.sinks_dropped)
    };
    assert_eq!(counts, (2, 1));
    rx.manager.stop_receivers().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn live_stream_plays_paced_from_near_the_live_edge() {
    let Some((_dir, server)) = support::live_server(None).await else {
        return;
    };
    let mut rx = Receiver::start().await;
    let mut sender = Sender::connect(rx.port, "sender-live").await;
    sender.launch().await;
    let reply = sender
        .load(&server.url("/live.m3u8"), "application/x-mpegurl")
        .await;
    assert_eq!(reply["status"][0]["playerState"], "PLAYING", "{reply}");
    tokio::time::sleep(Duration::from_secs(5)).await;

    let log = std::mem::take(&mut *rx.log.lock().unwrap());
    let frames = &log.video;
    assert!(frames.len() > 60, "only {} frames", frames.len());
    assert!(
        frames.windows(2).all(|w| w[0].1 < w[1].1),
        "PTS not increasing"
    );
    let pts_span = Duration::from_micros(frames[frames.len() - 1].1 - frames[0].1);
    let wall_span = frames[frames.len() - 1].0 - frames[0].0;
    let drift = wall_span.as_secs_f64() - pts_span.as_secs_f64();
    eprintln!(
        "E2E live: {} frames in {:.3}s wall for {:.3}s of PTS (drift {drift:+.3}s), {} audio chunks",
        frames.len(),
        wall_span.as_secs_f64(),
        pts_span.as_secs_f64(),
        log.audio.len()
    );
    assert!(drift.abs() < 0.25, "live paced {drift:+.3}s off real time");
    assert!(!log.audio.is_empty());
    rx.manager.stop_receivers().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn taken_port_is_reported_once_without_spinning() {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let squatter = std::net::TcpListener::bind("0.0.0.0:0").unwrap();
    let port = squatter.local_addr().unwrap().port();
    let (mut manager, mut events) = StreamManager::builder()
        .with_receiver(TestReceiver(ChromecastReceiver::with_port("e2e", port)))
        .set_sink_factory(|_remote, _info| async {
            Err::<Box<dyn FrameSink>, _>(ferricast::FerricastError::Other("unused".into()))
        })
        .build_with_events();
    manager.start_receivers().await.unwrap();
    let first = tokio::time::timeout(Duration::from_secs(2), events.recv())
        .await
        .expect("no event for the taken port")
        .unwrap();
    match first {
        ManagerEvent::ReceiverError { message, .. } => {
            eprintln!("E2E taken port: {message}");
            assert!(message.contains("bind"), "{message}");
        }
        other => panic!("unexpected {other:?}"),
    }
    // Nothing else follows: no retry loop flooding errors.
    assert!(
        tokio::time::timeout(Duration::from_millis(500), events.recv())
            .await
            .is_err()
    );
    manager.stop_receivers().await.unwrap();
}

// ── Progressive files ───────────────────────────────────────────

/// What a finished playback looked like in the sink.
struct Playback {
    frames: usize,
    pts_span: f64,
    drift: f64,
    audio_secs: f64,
    audio_chunks: usize,
}

impl Playback {
    fn of(log: &SinkLog) -> Self {
        let frames = &log.video;
        let (first, last) = (frames[0], frames[frames.len() - 1]);
        let pts_span = Duration::from_micros(last.1 - first.1).as_secs_f64();
        let wall_span = (last.0 - first.0).as_secs_f64();
        // The first chunk is the priming silence.
        let audio: Duration = log.audio.iter().skip(1).map(|a| a.2).sum();
        Self {
            frames: frames.len(),
            pts_span,
            drift: wall_span - pts_span,
            audio_secs: audio.as_secs_f64(),
            audio_chunks: log.audio.len(),
        }
    }
}

/// Loads `url` and plays it to IDLE/FINISHED; returns the LOAD reply.
async fn play_to_end(rx: &Receiver, sender: &mut Sender, url: &str, content_type: &str) -> Value {
    std::mem::take(&mut *rx.log.lock().unwrap());
    let reply = sender.load(url, content_type).await;
    assert_eq!(reply["type"], "MEDIA_STATUS", "{reply}");
    assert_eq!(reply["status"][0]["playerState"], "PLAYING", "{reply}");
    let finished = sender
        .expect(namespace::MEDIA, "IDLE FINISHED", |v| {
            v["status"][0]["playerState"] == "IDLE"
        })
        .await;
    assert_eq!(
        finished["status"][0]["idleReason"], "FINISHED",
        "{finished}"
    );
    reply
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn progressive_files_play_paced_with_audio_and_report_duration() {
    let dir = tempfile::tempdir().unwrap();
    let mut cases = vec![
        ("moov_at_end.mp4", support::H264_AAC, &[][..], "video/mp4"),
        (
            "faststart.mp4",
            support::H264_AAC,
            &["-movflags", "+faststart"][..],
            "video/mp4",
        ),
        (
            "h264_aac.mkv",
            support::H264_AAC,
            &[][..],
            "video/x-matroska",
        ),
        (
            "fragmented.mp4",
            support::H264_AAC,
            &["-movflags", "frag_keyframe+empty_moov+default_base_moof"][..],
            "video/mp4",
        ),
    ];
    if cfg!(all(feature = "vp9-decode", feature = "opus-decode")) {
        cases.push(("vp9_opus.webm", support::VP9_OPUS, &[][..], "video/webm"));
    }
    for (name, codecs, extra, _) in &cases {
        if !support::generate_progressive(dir.path(), name, 4, codecs, extra) {
            return;
        }
    }
    let server = TestServer::start(support::dir_handler(dir.path())).await;
    let mut rx = Receiver::start().await;
    let mut sender = Sender::connect(rx.port, "sender-progressive").await;
    sender.launch().await;
    for (name, _, _, content_type) in &cases {
        let reply = play_to_end(
            &rx,
            &mut sender,
            &server.url(&format!("/{name}")),
            content_type,
        )
        .await;
        let media = &reply["status"][0]["media"];
        let p = Playback::of(&rx.log.lock().unwrap());
        eprintln!(
            "E2E progressive {name}: duration {} streamType {}, {} frames, PTS span {:.3}s, \
             drift {:+.3}s, {} audio chunks = {:.3}s of audio",
            media["duration"],
            media["streamType"],
            p.frames,
            p.pts_span,
            p.drift,
            p.audio_chunks,
            p.audio_secs
        );
        assert_eq!(media["streamType"], "BUFFERED", "{name}");
        let duration = media["duration"]
            .as_f64()
            .unwrap_or_else(|| panic!("{name}: no duration"));
        assert!((duration - 4.0).abs() < 0.1, "{name}: duration {duration}");
        assert!(
            p.drift.abs() < 0.25,
            "{name}: paced {:+.3}s off real time",
            p.drift
        );
        assert!(
            (p.audio_secs - 4.0).abs() < 0.2,
            "{name}: {:.3}s of audio",
            p.audio_secs
        );
        assert!(p.pts_span > 3.8, "{name}: PTS span {:.3}s", p.pts_span);
        assert!(p.frames >= 100, "{name}: only {} frames shown", p.frames);
    }
    let requests = server.log.lock().unwrap().len();
    eprintln!("E2E progressive: {requests} HTTP requests in total");
    rx.manager.stop_receivers().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn seek_in_a_progressive_file_lands_near_the_target() {
    let dir = tempfile::tempdir().unwrap();
    if support::generate_progressive(dir.path(), "long.mp4", 12, support::H264_AAC, &[]) {
        seek_lands_on_the_target(dir.path(), "long.mp4", "video/mp4", Origin::Zero).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn seek_in_a_fragmented_mp4_lands_on_the_target() {
    let dir = tempfile::tempdir().unwrap();
    let extra = ["-movflags", "frag_keyframe+empty_moov+default_base_moof"];
    let name = "long_fragmented.mp4";
    if support::generate_progressive(dir.path(), name, 12, support::H264_AAC, &extra) {
        seek_lands_on_the_target(dir.path(), name, "video/mp4", Origin::Zero).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn seek_in_hls_lands_on_the_target() {
    let dir = tempfile::tempdir().unwrap();
    if support::generate_hls(dir.path(), 12, 2, &[]) {
        let mime = "application/vnd.apple.mpegurl";
        seek_lands_on_the_target(dir.path(), "index.m3u8", mime, Origin::FirstFrame).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn seek_in_fmp4_hls_lands_on_the_target() {
    let dir = tempfile::tempdir().unwrap();
    if support::generate_hls(dir.path(), 12, 2, &["-hls_segment_type", "fmp4"]) {
        let mime = "application/vnd.apple.mpegurl";
        seek_lands_on_the_target(dir.path(), "index.m3u8", mime, Origin::FirstFrame).await;
    }
}

/// Where media time zero is in the timestamps a sink sees.
#[derive(Clone, Copy)]
enum Origin {
    /// Timestamps are media time (progressive files).
    Zero,
    /// At the first frame shown (MPEG-TS and HLS start anywhere).
    FirstFrame,
}

/// Seeks to 7.5 s into a 12 s clip with a keyframe every second: the
/// first frame and audio shown are at the target, no stale audio is
/// pushed after the flush, and the status reports the right position.
#[expect(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    reason = "one scripted seek scenario; microsecond positions print as seconds"
)]
async fn seek_lands_on_the_target(dir: &std::path::Path, name: &str, mime: &str, origin: Origin) {
    let server = TestServer::start(support::dir_handler(dir)).await;
    let mut rx = Receiver::start().await;
    let mut sender = Sender::connect(rx.port, "sender-seek").await;
    sender.launch().await;
    let reply = sender.load(&server.url(&format!("/{name}")), mime).await;
    assert_eq!(reply["status"][0]["playerState"], "PLAYING", "{reply}");
    let session_id = reply["status"][0]["mediaSessionId"].clone();
    tokio::time::sleep(Duration::from_secs(1)).await;

    let (shown_before, heard_before, base) = {
        let log = rx.log.lock().unwrap();
        let base = match origin {
            Origin::Zero => 0,
            Origin::FirstFrame => log
                .video
                .iter()
                .map(|v| v.1)
                .min()
                .expect("frames before the seek"),
        };
        (log.video.len(), log.audio.len(), base)
    };
    let seek_sent = Instant::now();
    let id = sender
        .request(
            namespace::MEDIA,
            json!({"type": "SEEK", "mediaSessionId": session_id, "currentTime": 7.5}),
        )
        .await;
    sender
        .expect(namespace::MEDIA, "MEDIA_STATUS for SEEK", |v| {
            v["requestId"] == id
        })
        .await;
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let (first_after, last_after, at_first) = {
        let log = rx.log.lock().unwrap();
        let after = &log.video[shown_before..];
        let first = after
            .iter()
            .find(|(_, pts)| *pts >= base + 5_000_000)
            .copied()
            .expect("frames after the seek");
        (first.1 - base, after.last().unwrap().1 - base, first.0)
    };
    let (first_audio, flushed_at, stale) = {
        let log = rx.log.lock().unwrap();
        let flushed_at = log
            .flushes
            .iter()
            .copied()
            .find(|&at| at >= heard_before)
            .expect("the seek flushes the sink audio");
        let after = flushed_at..log.audio.len();
        let first = after
            .clone()
            .find(|&i| !log.audio_silent[i])
            .map(|i| log.audio[i].1.saturating_sub(base))
            .expect("audio after the seek");
        // Anything pushed after the flush from before the target is stale.
        let stale = after
            .filter(|&i| !log.audio_silent[i] && log.audio[i].1 < base + 7_400_000)
            .count();
        (first, flushed_at, stale)
    };
    let id = sender
        .request(
            namespace::MEDIA,
            json!({"type": "PAUSE", "mediaSessionId": session_id}),
        )
        .await;
    sender
        .expect(namespace::MEDIA, "MEDIA_STATUS for PAUSE", |v| {
            v["requestId"] == id
        })
        .await;
    let paused = sender
        .expect(namespace::MEDIA, "PAUSED status from the pipeline", |v| {
            v["requestId"] == 0 && v["status"][0]["playerState"] == "PAUSED"
        })
        .await;
    let reported = paused["status"][0]["currentTime"].as_f64().unwrap();
    eprintln!(
        "E2E seek {name}: target 7.500s, first frame after seek at {:.3}s ({:.0} ms after SEEK), \
         last shown {:.3}s, PAUSED currentTime {reported:.3}s",
        first_after as f64 / 1e6,
        (at_first - seek_sent).as_secs_f64() * 1000.0,
        last_after as f64 / 1e6,
    );
    eprintln!(
        "E2E seek {name}: first audio after seek at {:.3}s, flush after {flushed_at} chunks, {stale} stale chunks",
        first_audio as f64 / 1e6
    );
    // Keyframes every second: decoding restarts at 7.0 s, but nothing
    // before the target is shown or heard.
    assert!(
        first_after.abs_diff(7_500_000) <= 34_000,
        "first frame after the seek at {first_after}"
    );
    assert!(
        first_audio.abs_diff(7_500_000) <= 25_000,
        "first audio after the seek at {first_audio}"
    );
    assert_eq!(stale, 0, "stale audio after the flush");
    assert!(
        (reported - last_after as f64 / 1e6).abs() < 0.3,
        "status says {reported}, last frame {last_after}"
    );
    rx.manager.stop_receivers().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn progressive_file_plays_from_a_server_without_range_support() {
    let dir = tempfile::tempdir().unwrap();
    if !support::generate_progressive(dir.path(), "moov_at_end.mp4", 4, support::H264_AAC, &[]) {
        return;
    }
    let body = std::fs::read(dir.path().join("moov_at_end.mp4")).unwrap();
    let server = TestServer::start(Arc::new(move |path: &str| match path {
        "/moov_at_end.mp4" => support::Reply::OkIgnoringRange(body.clone()),
        _ => support::Reply::Status(404),
    }))
    .await;
    let mut rx = Receiver::start().await;
    let mut sender = Sender::connect(rx.port, "sender-norange").await;
    sender.launch().await;
    let reply = play_to_end(
        &rx,
        &mut sender,
        &server.url("/moov_at_end.mp4"),
        "video/mp4",
    )
    .await;
    let p = Playback::of(&rx.log.lock().unwrap());
    let requests = server.log.lock().unwrap().len();
    eprintln!(
        "E2E no-range: duration {}, {} frames, drift {:+.3}s, {:.3}s of audio, {requests} HTTP requests",
        reply["status"][0]["media"]["duration"], p.frames, p.drift, p.audio_secs
    );
    assert_eq!(requests, 1, "the file is downloaded once");
    assert!(p.frames >= 100 && p.drift.abs() < 0.25);
    assert!((p.audio_secs - 4.0).abs() < 0.2);

    // Something that is not a media file fails the LOAD cleanly.
    let reply = sender
        .load(&server.url("/missing.webm"), "video/webm")
        .await;
    assert_eq!(reply["type"], "LOAD_FAILED", "{reply}");
    rx.manager.stop_receivers().await.unwrap();
}

/// Pacing of the audible (non-silent) audio chunks of an audio-only
/// playback: `(PTS span, wall span, seconds of audio)`.
fn audio_pacing(log: &SinkLog) -> (f64, f64, f64) {
    let heard: Vec<_> = log
        .audio
        .iter()
        .zip(&log.audio_silent)
        .filter(|(_, silent)| !**silent)
        .map(|(a, _)| *a)
        .collect();
    let (first, last) = (heard[0], heard[heard.len() - 1]);
    let pts_span = Duration::from_micros(last.1 - first.1).as_secs_f64();
    let wall_span = (last.0 - first.0).as_secs_f64();
    let secs = heard.iter().map(|a| a.2).sum::<Duration>().as_secs_f64();
    (pts_span, wall_span, secs)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn audio_files_play_paced_and_report_duration() {
    let dir = tempfile::tempdir().unwrap();
    let mut cases: Vec<(&str, &[&str], &str)> = vec![
        (
            "tone.mp3",
            &["-c:a", "libmp3lame", "-q:a", "4"],
            "audio/mpeg",
        ),
        ("tone.flac", &["-c:a", "flac"], "audio/flac"),
        ("vorbis.ogg", &["-c:a", "libvorbis"], "audio/ogg"),
        ("tone.wav", &["-c:a", "pcm_s24le"], "audio/wav"),
        ("tone.aac", &["-c:a", "aac", "-f", "adts"], "audio/aac"),
    ];
    if cfg!(feature = "opus-decode") {
        cases.push(("opus.ogg", &["-c:a", "libopus"], "audio/ogg"));
    }
    for (name, args, _) in &cases {
        if !support::generate_audio(dir.path(), name, 4, args) {
            return;
        }
    }
    let mut expected_shape = vec![None; cases.len()];
    if cfg!(feature = "fdk-aac") {
        std::fs::write(dir.path().join("he1.aac"), he_aac_adts(4, false)).unwrap();
        std::fs::write(dir.path().join("he2.aac"), he_aac_adts(4, true)).unwrap();
        cases.push(("he1.aac", &[], "audio/aac"));
        cases.push(("he2.aac", &[], "audio/aac"));
        expected_shape.extend([Some((48_000, 2)), Some((48_000, 2))]);
    }
    let server = TestServer::start(support::dir_handler(dir.path())).await;
    let mut rx = Receiver::start().await;
    let mut sender = Sender::connect(rx.port, "sender-audio").await;
    sender.launch().await;
    for ((name, _, content_type), shape) in cases.iter().zip(expected_shape) {
        let reply = play_to_end(
            &rx,
            &mut sender,
            &server.url(&format!("/{name}")),
            content_type,
        )
        .await;
        let log = rx.log.lock().unwrap();
        let duration = reply["status"][0]["media"]["duration"]
            .as_f64()
            .unwrap_or_else(|| panic!("{name}: no duration"));
        let (pts_span, wall_span, secs) = audio_pacing(&log);
        let heard_shape = log.audio_shapes.last().copied().unwrap();
        eprintln!(
            "E2E audio {name}: duration {duration:.3}s, {} chunks, {secs:.3}s of audio, \
             PTS span {pts_span:.3}s over {wall_span:.3}s wall, {} Hz / {} ch, no video: {}",
            log.audio.len(),
            heard_shape.0,
            heard_shape.1,
            log.video.is_empty()
        );
        assert!(log.video.is_empty(), "{name}: video frames");
        assert!((duration - 4.0).abs() < 0.15, "{name}: duration {duration}");
        assert!((secs - 4.0).abs() < 0.2, "{name}: {secs:.3}s of audio");
        assert!(
            (wall_span - pts_span).abs() < 0.25,
            "{name}: PTS span {pts_span:.3}s over {wall_span:.3}s"
        );
        if let Some(shape) = shape {
            assert_eq!(heard_shape, shape, "{name}: decoded shape");
        }
    }
    rx.manager.stop_receivers().await.unwrap();
}

/// `seconds` of stereo tones at 48 kHz encoded with libfdk-aac as ADTS
/// HE-AAC (`v2`: with parametric stereo), the only HE-AAC encoder at
/// hand (ffmpeg builds rarely include it).
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "a few seconds of a 12000-peak test tone: every value fits"
)]
fn he_aac_adts(seconds: u32, v2: bool) -> Vec<u8> {
    use fdk_aac::enc::{AudioObjectType, BitRate, ChannelMode, Encoder, EncoderParams, Transport};
    let rate = 48_000u32;
    let encoder = Encoder::new(EncoderParams {
        bit_rate: BitRate::Cbr(48_000),
        sample_rate: rate,
        transport: Transport::Adts,
        channels: ChannelMode::Stereo,
        audio_object_type: if v2 {
            AudioObjectType::Mpeg4HeAacV2
        } else {
            AudioObjectType::Mpeg4HeAac
        },
    })
    .unwrap();
    let chunk = encoder.info().unwrap().frameLength as usize * 2;
    let pcm: Vec<i16> = (0..(rate * seconds) as usize)
        .flat_map(|n| {
            let t = n as f32 / rate as f32;
            let l = (t * 1_000.0 * std::f32::consts::TAU).sin();
            let r = (t * 3_000.0 * std::f32::consts::TAU).sin();
            [(l * 12_000.0) as i16, (r * 12_000.0) as i16]
        })
        .collect();
    let mut adts = Vec::new();
    let mut out = vec![0u8; 8192];
    for mut input in pcm.chunks(chunk) {
        while !input.is_empty() {
            let info = encoder.encode(input, &mut out).unwrap();
            adts.extend_from_slice(&out[..info.output_size]);
            if info.input_consumed == 0 {
                break;
            }
            input = &input[info.input_consumed..];
        }
    }
    adts
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fmp4_hls_plays_paced_with_audio() {
    let dir = tempfile::tempdir().unwrap();
    if !support::generate_hls(dir.path(), 6, 2, &["-hls_segment_type", "fmp4"]) {
        return;
    }
    let server = TestServer::start(support::dir_handler(dir.path())).await;
    let mut rx = Receiver::start().await;
    let mut sender = Sender::connect(rx.port, "sender-fmp4-hls").await;
    sender.launch().await;
    let reply = play_to_end(
        &rx,
        &mut sender,
        &server.url("/index.m3u8"),
        "application/vnd.apple.mpegurl",
    )
    .await;
    let p = Playback::of(&rx.log.lock().unwrap());
    eprintln!(
        "E2E fMP4 HLS: duration {}, {} frames, PTS span {:.3}s, drift {:+.3}s, {:.3}s of audio",
        reply["status"][0]["media"]["duration"], p.frames, p.pts_span, p.drift, p.audio_secs
    );
    assert!(p.frames >= 170, "only {} frames shown", p.frames);
    assert!(p.pts_span > 5.8, "PTS span {:.3}s", p.pts_span);
    assert!(p.drift.abs() < 0.25, "paced {:+.3}s off real time", p.drift);
    assert!(
        (p.audio_secs - 6.0).abs() < 0.25,
        "{:.3}s of audio",
        p.audio_secs
    );
    rx.manager.stop_receivers().await.unwrap();
}
