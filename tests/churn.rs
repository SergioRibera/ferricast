//! Connection churn against the receivers, shaped like what Android
//! casting apps do to a LAN receiver: a Cast connection every second or
//! so that does TLS and the device-auth challenge and then drops the
//! socket, TCP reachability probes that close before TLS, and AirPlay
//! `/info` probes, pairing and `SETUP` on connections that are then
//! dropped.
//!
//! The quick test checks that everything a churned connection creates
//! (tasks, sockets, receiver sessions) is gone once it closes. The soak
//! test (`--ignored`) runs for minutes and prints RSS, open fds,
//! threads, tasks, CPU and receiver session counts over time.

#![cfg(all(feature = "chromecast", feature = "airplay"))]

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use bytes::{Bytes, BytesMut};
use ferricast::airplay::{AirPlayReceiver, AirPlayReceiverConfig, CredentialStorage};
use ferricast::device::{AirplayConfig, Features, Flags};
use ferricast::{
    AdvertiseInfo, Advertiser, CapturedFrame, CastSession, Codec, DecodedAudio, Device,
    DeviceCapabilities, FrameSink, ManagerEvent, ReceiverProtocol, Result, StreamConfig,
    StreamManager,
};
use ferricast_airplay::AirPlaySession;
use ferricast_chromecast::ChromecastReceiver;
use ferricast_chromecast::castv2::{CastMessage, namespace};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

// ── Receivers under test, without mDNS ──────────────────────────

macro_rules! unadvertised {
    ($name:ident, $adv:ident, $inner:ty, $proto:literal) => {
        struct $adv;

        impl Advertiser for $adv {
            const PROTOCOL: &'static str = $proto;
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

        struct $name($inner);

        impl ReceiverProtocol for $name {
            const PROTOCOL: &'static str = $proto;
            const SUPPORTED_CODECS: &'static [Codec] = &[Codec::H264];
            type Advertiser = $adv;
            type Control = <$inner as ReceiverProtocol>::Control;
            type Puller = <$inner as ReceiverProtocol>::Puller;

            fn create_advertiser(&self) -> $adv {
                $adv
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
    };
}

unadvertised!(
    CastUnderTest,
    NoCastAdvertiser,
    ChromecastReceiver,
    "chromecast"
);
unadvertised!(
    AirPlayUnderTest,
    NoAirPlayAdvertiser,
    AirPlayReceiver,
    "airplay"
);

struct NullSink;

#[async_trait::async_trait]
impl FrameSink for NullSink {
    async fn push_video(&mut self, _frame: CapturedFrame) -> Result<()> {
        Ok(())
    }
    async fn push_audio(&mut self, _audio: DecodedAudio) -> Result<()> {
        Ok(())
    }
}

/// Counts of the manager events a GUI would turn into UI work.
#[derive(Default)]
struct EventCounts {
    incoming: AtomicU64,
    errors: AtomicU64,
    stopped: AtomicU64,
    sinks: AtomicU64,
}

/// Where the churn goes.
#[derive(Debug, Clone, Copy)]
struct Target {
    cast_port: u16,
    airplay: SocketAddr,
}

struct Harness {
    manager: StreamManager,
    cast_port: u16,
    airplay: SocketAddr,
    events: Arc<EventCounts>,
    drain: tokio::task::JoinHandle<()>,
}

impl Harness {
    fn target(&self) -> Target {
        Target {
            cast_port: self.cast_port,
            airplay: self.airplay,
        }
    }

    async fn start(with_discovery: bool) -> Self {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let cast_port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let airplay = AirPlayReceiver::new(
            AirPlayReceiverConfig {
                name: "churn".into(),
                credentials: CredentialStorage::Ephemeral,
                ..Default::default()
            }
            .with_ephemeral_ports(),
        )
        .await;
        let events = Arc::new(EventCounts::default());
        let sinks = events.clone();
        let mut builder = StreamManager::builder();
        if with_discovery {
            builder = builder.with_chromecast().with_airplay();
        }
        let (mut manager, mut rx) = builder
            .with_receiver(CastUnderTest(ChromecastReceiver::with_port(
                "churn", cast_port,
            )))
            .with_receiver(AirPlayUnderTest(airplay.clone()))
            .with_h264_decoder()
            .set_sink_factory(move |_remote, _info| {
                sinks.sinks.fetch_add(1, Ordering::Relaxed);
                async { Ok(Box::new(NullSink) as Box<dyn FrameSink>) }
            })
            .build_with_events();
        manager.start_receivers().await.unwrap();
        if with_discovery {
            manager.start_discovery().await.unwrap();
        }
        let port = airplay.bound_rtsp_port().expect("AirPlay port bound");
        let counts = events.clone();
        let drain = tokio::spawn(async move {
            while let Some(ev) = rx.recv().await {
                let counter = match ev {
                    ManagerEvent::ReceiverIncoming { .. } => &counts.incoming,
                    ManagerEvent::ReceiverError { .. } => &counts.errors,
                    ManagerEvent::ReceiverStopped { .. } => &counts.stopped,
                    _ => continue,
                };
                counter.fetch_add(1, Ordering::Relaxed);
            }
        });
        Self {
            manager,
            cast_port,
            airplay: SocketAddr::from(([127, 0, 0, 1], port)),
            events,
            drain,
        }
    }
}

// ── Churn clients ───────────────────────────────────────────────

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

fn tls_connector() -> tokio_rustls::TlsConnector {
    let config = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(NoVerify))
        .with_no_client_auth();
    tokio_rustls::TlsConnector::from(Arc::new(config))
}

/// TLS, device-auth challenge, wait for the reply, then drop the
/// socket without `close_notify`, like a sender that gave up on auth.
async fn cast_auth_probe(connector: &tokio_rustls::TlsConnector, port: u16) {
    let Ok(tcp) = TcpStream::connect(("127.0.0.1", port)).await else {
        return;
    };
    let Ok(mut tls) = connector
        .connect("localhost".try_into().unwrap(), tcp)
        .await
    else {
        return;
    };
    let challenge = CastMessage::new_binary(
        "sender-0",
        "receiver-0",
        "urn:x-cast:com.google.cast.tp.deviceauth",
        vec![0x0a, 0x00],
    );
    if tls
        .write_all(&challenge.encode_length_prefixed().unwrap())
        .await
        .is_err()
    {
        return;
    }
    let mut buf = BytesMut::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    while CastMessage::decode_length_prefixed(&mut buf)
        .ok()
        .flatten()
        .is_none()
    {
        match tokio::time::timeout_at(deadline.into(), tls.read_buf(&mut buf)).await {
            Ok(Ok(n)) if n > 0 => {}
            _ => break,
        }
    }
    // Dropping the TLS stream closes the socket without close_notify.
}

/// A sender that passes auth (one that skips it) and checks the
/// connection is served: CONNECT + PING must get a PONG.
async fn cast_served(connector: &tokio_rustls::TlsConnector, port: u16) -> bool {
    let tcp = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    let mut tls = connector
        .connect("localhost".try_into().unwrap(), tcp)
        .await
        .unwrap();
    for (ns, payload) in [
        (namespace::CONNECTION, json!({"type": "CONNECT"})),
        (namespace::HEARTBEAT, json!({"type": "PING"})),
    ] {
        let msg = CastMessage::new_json("sender-ok", "receiver-0", ns, &payload).unwrap();
        tls.write_all(&msg.encode_length_prefixed().unwrap())
            .await
            .unwrap();
    }
    let mut buf = BytesMut::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(msg) = CastMessage::decode_length_prefixed(&mut buf).unwrap() {
            if msg.namespace == namespace::HEARTBEAT {
                return true;
            }
            continue;
        }
        match tokio::time::timeout_at(deadline.into(), tls.read_buf(&mut buf)).await {
            Ok(Ok(n)) if n > 0 => {}
            _ => return false,
        }
    }
}

/// TCP connect, close before any TLS byte.
async fn tcp_probe(port: u16) {
    if let Ok(s) = TcpStream::connect(("127.0.0.1", port)).await {
        drop(s);
    }
}

async fn airplay_info_probe(addr: SocketAddr) {
    let Ok(mut s) = TcpStream::connect(addr).await else {
        return;
    };
    let req = b"GET /info RTSP/1.0\r\nCSeq: 1\r\nUser-Agent: AirPlay/550.10\r\n\r\n";
    if s.write_all(req).await.is_err() {
        return;
    }
    let mut buf = vec![0_u8; 4096];
    let _ = tokio::time::timeout(Duration::from_secs(5), s.read(&mut buf)).await;
}

fn airplay_device(addr: SocketAddr) -> Device {
    Device {
        id: uuid::Uuid::new_v4(),
        name: "churn".into(),
        protocol: "airplay",
        protocol_icon: Bytes::new(),
        addresses: addr.into(),
        model: None,
        metadata: HashMap::new(),
        capabilities: DeviceCapabilities {
            supports_screen_mirror: true,
            airplay_config: Some(AirplayConfig {
                features: Features::parse_txt("0x5A7FFEE6,0x0").unwrap(),
                flags: Flags::empty(),
                pk: None,
                device_id: Some("churn-receiver".into()),
            }),
            ..Default::default()
        },
    }
}

/// Pair-setup + pair-verify, then drop; with `setup`, also fp-setup and
/// the mirroring `SETUP` before dropping.
async fn airplay_session_probe(addr: SocketAddr, setup: bool) {
    let mut session = AirPlaySession::ephemeral();
    if session.connect(&airplay_device(addr)).await.is_err() {
        return;
    }
    if setup {
        let _ = session
            .setup_stream(&StreamConfig {
                width: 640,
                height: 360,
                fps: 30,
                bitrate_kbps: 1000,
                codec: Codec::H264,
                ..Default::default()
            })
            .await;
    }
    drop(session);
}

/// One round of churn, `i` picks the kind of connection.
async fn churn_one(
    i: u64,
    connector: &tokio_rustls::TlsConnector,
    cast_port: u16,
    airplay: SocketAddr,
) {
    match i % 20 {
        0..=7 => cast_auth_probe(connector, cast_port).await,
        8..=12 => tcp_probe(cast_port).await,
        13..=15 => airplay_info_probe(airplay).await,
        16 => tcp_probe(airplay.port()).await,
        17 | 18 => airplay_session_probe(airplay, false).await,
        _ => airplay_session_probe(airplay, true).await,
    }
}

/// Runs `total` churn rounds, `parallel` at a time, `gap` apart per lane.
async fn churn(target: Target, total: u64, parallel: u64, gap: Duration) {
    let next = Arc::new(AtomicU64::new(0));
    let mut lanes = tokio::task::JoinSet::new();
    for _ in 0..parallel {
        let next = next.clone();
        let Target { cast_port, airplay } = target;
        lanes.spawn(async move {
            let connector = tls_connector();
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= total {
                    break;
                }
                churn_one(i, &connector, cast_port, airplay).await;
                if !gap.is_zero() {
                    tokio::time::sleep(gap).await;
                }
            }
        });
    }
    while lanes.join_next().await.is_some() {}
}

// ── Process metrics ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct Sample {
    rss_kib: u64,
    fds: usize,
    threads: u64,
    tasks: usize,
    cpu_ms: u64,
    sessions: usize,
}

fn status_field(status: &str, key: &str) -> u64 {
    status
        .lines()
        .find_map(|l| l.strip_prefix(key))
        .and_then(|v| v.split_whitespace().next())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

fn cpu_ms(pid: &str) -> u64 {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
    // Fields after the parenthesised command name; utime and stime are
    // the 12th and 13th of those.
    let rest = stat.rsplit_once(')').map_or("", |(_, r)| r);
    let fields: Vec<u64> = rest
        .split_whitespace()
        .map(|f| f.parse().unwrap_or(0))
        .collect();
    let ticks = fields.get(11).copied().unwrap_or(0) + fields.get(12).copied().unwrap_or(0);
    // USER_HZ is 100 on every Linux configuration we run on.
    ticks * 10
}

fn sample(h: &Harness) -> Sample {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    Sample {
        rss_kib: status_field(&status, "VmRSS:"),
        fds: std::fs::read_dir("/proc/self/fd").map_or(0, Iterator::count),
        threads: status_field(&status, "Threads:"),
        tasks: tokio::runtime::Handle::current()
            .metrics()
            .num_alive_tasks(),
        cpu_ms: cpu_ms("self"),
        sessions: h.manager.receiver_session_count(),
    }
}

/// Waits until the churn's tasks and sockets are gone (or `limit`).
async fn settle(h: &Harness, baseline: Sample, limit: Duration) -> Sample {
    let deadline = Instant::now() + limit;
    loop {
        let s = sample(h);
        let settled = s.sessions == 0 && s.tasks <= baseline.tasks + 2 && s.fds <= baseline.fds + 2;
        if settled || Instant::now() > deadline {
            return s;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn print_sample(label: &str, s: Sample, h: &Harness) {
    eprintln!(
        "CHURN {label:>10}: rss={:>7} KiB fds={:>4} threads={:>3} tasks={:>4} sessions={:>3} cpu={:>6} ms \
         events(incoming={} errors={} stopped={} sinks={})",
        s.rss_kib,
        s.fds,
        s.threads,
        s.tasks,
        s.sessions,
        s.cpu_ms,
        h.events.incoming.load(Ordering::Relaxed),
        h.events.errors.load(Ordering::Relaxed),
        h.events.stopped.load(Ordering::Relaxed),
        h.events.sinks.load(Ordering::Relaxed),
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn churned_connections_leave_nothing_behind() {
    let h = Harness::start(false).await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    let baseline = sample(&h);
    print_sample("baseline", baseline, &h);

    churn(h.target(), 1000, 6, Duration::ZERO).await;
    let after = settle(&h, baseline, Duration::from_secs(20)).await;
    print_sample("after", after, &h);

    assert_eq!(after.sessions, 0, "receiver sessions left behind");
    assert!(
        after.tasks <= baseline.tasks + 2,
        "tasks left behind: {} -> {}",
        baseline.tasks,
        after.tasks
    );
    assert!(
        after.fds <= baseline.fds + 2,
        "sockets left behind: {} -> {}",
        baseline.fds,
        after.fds
    );
    // Connections that never got past device auth or a probe are not
    // senders: they must not reach the pipeline or the UI.
    let incoming = h.events.incoming.load(Ordering::Relaxed);
    let errors = h.events.errors.load(Ordering::Relaxed);
    assert!(
        incoming <= 60,
        "{incoming} probe connections surfaced as senders"
    );
    assert!(
        errors <= 60,
        "{errors} probe connections surfaced as errors"
    );

    // A sender arriving right after the churn is served straight away.
    let started = Instant::now();
    assert!(cast_served(&tls_connector(), h.cast_port).await);
    assert!(started.elapsed() < Duration::from_secs(1));

    let mut manager = h.manager;
    manager.stop_receivers().await.unwrap();
    h.drain.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn probes_do_not_delay_real_senders() {
    let h = Harness::start(false).await;
    // A stream of TCP probes that close before TLS, running alongside.
    let port = h.cast_port;
    let probes = tokio::spawn(async move {
        loop {
            tcp_probe(port).await;
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    });
    tokio::time::sleep(Duration::from_millis(500)).await;
    let connector = tls_connector();
    for _ in 0..5 {
        let started = Instant::now();
        assert!(cast_served(&connector, h.cast_port).await);
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "sender waited {:?} behind probes",
            started.elapsed()
        );
    }
    probes.abort();
    let mut manager = h.manager;
    manager.stop_receivers().await.unwrap();
    h.drain.abort();
}

/// Minutes of churn at the observed rate plus a fast burst, printing
/// the process metrics every 10 s. `FERRICAST_SOAK_SECS` sets the
/// length (default 300), `FERRICAST_SOAK_DISCOVERY=1` also runs mDNS
/// discovery like the GUI does.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "takes minutes; run with --ignored --nocapture"]
async fn soak_under_churn() {
    let secs: u64 = std::env::var("FERRICAST_SOAK_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let discovery = std::env::var("FERRICAST_SOAK_DISCOVERY").is_ok_and(|v| v == "1");
    let h = Arc::new(Harness::start(discovery).await);
    tokio::time::sleep(Duration::from_secs(1)).await;
    let baseline = sample(&h);
    print_sample("baseline", baseline, &h);

    // Fast burst first: 2000 connections, 32 at a time.
    churn(h.target(), 2000, 32, Duration::ZERO).await;
    let burst = settle(&h, baseline, Duration::from_secs(30)).await;
    print_sample("burst", burst, &h);

    // Then the observed rate, ~2 connections per second per lane.
    let steady = {
        let h = h.clone();
        tokio::spawn(async move {
            churn(h.target(), u64::MAX, 4, Duration::from_millis(500)).await;
        })
    };
    let started = Instant::now();
    let mut samples = Vec::new();
    while started.elapsed() < Duration::from_secs(secs) {
        tokio::time::sleep(Duration::from_secs(10)).await;
        let s = sample(&h);
        print_sample(&format!("t={}s", started.elapsed().as_secs()), s, &h);
        samples.push(s);
    }
    steady.abort();
    let end = settle(&h, baseline, Duration::from_secs(30)).await;
    print_sample("end", end, &h);

    assert_eq!(end.sessions, 0);
    assert!(
        end.fds <= baseline.fds + 4,
        "{} -> {}",
        baseline.fds,
        end.fds
    );
    assert!(end.tasks <= baseline.tasks + 4);
    assert!(end.threads <= burst.threads + 4);
    // RSS settles after warm-up: the second half of the run must not
    // keep climbing.
    if let (Some(mid), Some(last)) = (samples.get(samples.len() / 2), samples.last()) {
        assert!(
            last.rss_kib <= mid.rss_kib + 8 * 1024,
            "RSS kept growing: {} KiB at mid-run, {} KiB at the end",
            mid.rss_kib,
            last.rss_kib
        );
    }
}

/// The same churn against a running process (the GUI), measured from
/// outside through `/proc/<pid>`:
///
/// ```text
/// FERRICAST_CHURN_PID=$(pgrep -f ferricast-gui) \
///   cargo test --test churn external_process -- --ignored --nocapture
/// ```
///
/// `FERRICAST_CHURN_HOST` (default 127.0.0.1), `FERRICAST_CHURN_CAST_PORT`
/// (8009), `FERRICAST_CHURN_AIRPLAY_PORT` (7000) and `FERRICAST_SOAK_SECS`
/// (300) adjust it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a running receiver process; see the doc comment"]
async fn external_process() {
    let env = |key: &str, default: &str| std::env::var(key).unwrap_or_else(|_| default.into());
    let pid = std::env::var("FERRICAST_CHURN_PID").expect("set FERRICAST_CHURN_PID");
    let host: std::net::IpAddr = env("FERRICAST_CHURN_HOST", "127.0.0.1").parse().unwrap();
    let cast_port: u16 = env("FERRICAST_CHURN_CAST_PORT", "8009").parse().unwrap();
    let airplay_port: u16 = env("FERRICAST_CHURN_AIRPLAY_PORT", "7000").parse().unwrap();
    let secs: u64 = env("FERRICAST_SOAK_SECS", "300").parse().unwrap();
    assert!(host.is_loopback(), "the churn clients dial 127.0.0.1");
    let _ = rustls::crypto::ring::default_provider().install_default();
    let target = Target {
        cast_port,
        airplay: SocketAddr::new(host, airplay_port),
    };
    let report = |label: &str| {
        let status = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap_or_default();
        eprintln!(
            "CHURN {label:>10}: rss={:>7} KiB fds={:>4} threads={:>3} cpu={:>6} ms",
            status_field(&status, "VmRSS:"),
            std::fs::read_dir(format!("/proc/{pid}/fd")).map_or(0, Iterator::count),
            status_field(&status, "Threads:"),
            cpu_ms(&pid),
        );
    };
    report("baseline");
    churn(target, 2000, 6, Duration::ZERO).await;
    tokio::time::sleep(Duration::from_secs(5)).await;
    report("burst");
    let steady = tokio::spawn(churn(target, u64::MAX, 2, Duration::from_secs(1)));
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(secs) {
        tokio::time::sleep(Duration::from_secs(10)).await;
        report(&format!("t={}s", started.elapsed().as_secs()));
    }
    steady.abort();
    tokio::time::sleep(Duration::from_secs(10)).await;
    report("end");
}
