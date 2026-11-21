//! Local loop: ferricast's AirPlay sender mirrors a test pattern and a
//! 440 Hz tone to ferricast's AirPlay receiver on localhost, through the
//! `StreamManager` receiver pipeline into a headless sink.
//!
//! Covers legacy pairing, PIN pairing (the test types the PIN the receiver
//! reports), FairPlay `fp-setup` and `ekey` decryption, the mirroring and
//! audio streams, decoding, and `TEARDOWN`.

#![cfg(all(feature = "airplay", feature = "openh264", feature = "fdk-aac"))]

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use ferricast::airplay::{AirPlayReceiver, AirPlayReceiverConfig, CredentialStorage};
use ferricast::device::{AirplayConfig, Features, Flags};
use ferricast::encoder::aac::{ELD_SAMPLE_RATE, FdkEldEncoder};
use ferricast::encoder::h264::X264Encoder;
use ferricast::{
    AdvertiseInfo, Advertiser, AudioCodec, AudioEncoder, AudioEncoderConfig, AudioFrame,
    AudioStreamConfig, CapturedFrame, CastSession, Codec, ConnectOutcome, DecodedAudio, Device,
    DeviceCapabilities, EncoderConfig, FrameSink, ManagerEvent, PairingResponse, PixelFormat,
    RawFrame, ReceiverProtocol, Result, StreamConfig, StreamManager, VideoEncoder,
};
use ferricast_airplay::AirPlaySession;
use tokio::sync::mpsc;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const FPS: u32 = 30;
const SECONDS: u32 = 5;
const TONE_HZ: f64 = 440.0;

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
    /// (width, height, PTS) per video frame.
    video: Vec<(u32, u32, u64)>,
    /// Interleaved stereo PCM, silence included.
    pcm: Vec<i16>,
    audio_shape: Option<(u32, u16)>,
    sinks: usize,
}

struct HeadlessSink(Arc<Mutex<SinkLog>>);

#[async_trait::async_trait]
impl FrameSink for HeadlessSink {
    async fn push_video(&mut self, frame: CapturedFrame) -> Result<()> {
        self.0
            .lock()
            .unwrap()
            .video
            .push((frame.width(), frame.height(), frame.timestamp_us()));
        Ok(())
    }
    async fn push_audio(&mut self, audio: DecodedAudio) -> Result<()> {
        let mut log = self.0.lock().unwrap();
        log.audio_shape = Some((audio.sample_rate, audio.channels));
        log.pcm.extend_from_slice(&audio.pcm);
        Ok(())
    }
}

// ── Sender media ────────────────────────────────────────────────

fn pattern(frame: u32) -> Bytes {
    let mut data = vec![0_u8; (WIDTH * HEIGHT * 4) as usize];
    for (i, px) in data.chunks_exact_mut(4).enumerate() {
        let x = (i as u32) % WIDTH;
        let y = (i as u32) / WIDTH;
        px[0] = ((x + frame * 4) % 256) as u8;
        px[1] = ((y + frame * 2) % 256) as u8;
        px[2] = if (x / 80 + y / 80 + frame / 15) % 2 == 0 {
            220
        } else {
            30
        };
        px[3] = 255;
    }
    Bytes::from(data)
}

fn tone(start: u64, count: u64) -> Bytes {
    let mut out = Vec::with_capacity(count as usize * 4);
    for n in start..start + count {
        let t = n as f64 / f64::from(ELD_SAMPLE_RATE);
        let v = ((2.0 * std::f64::consts::PI * TONE_HZ * t).sin() * 8000.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
    }
    Bytes::from(out)
}

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

/// Strongest frequency (10 Hz steps, 100 Hz to 2 kHz) of a mono signal.
fn peak_frequency(mono: &[f64], rate: f64) -> f64 {
    let mut best = (0.0, 0.0);
    let mut f = 100.0;
    while f <= 2000.0 {
        let w = 2.0 * std::f64::consts::PI * f / rate;
        let coeff = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0, 0.0);
        for &x in mono {
            let s = x + coeff * s1 - s2;
            s2 = s1;
            s1 = s;
        }
        let power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
        if power > best.1 {
            best = (f, power);
        }
        f += 10.0;
    }
    best.0
}

fn device(addr: SocketAddr, pin: bool) -> Device {
    Device {
        id: uuid::Uuid::new_v4(),
        name: "loop".into(),
        protocol: "airplay",
        protocol_icon: Bytes::new(),
        addresses: addr.into(),
        model: None,
        metadata: HashMap::new(),
        capabilities: DeviceCapabilities {
            supports_screen_mirror: true,
            airplay_config: Some(AirplayConfig {
                features: Features::parse_txt("0x5A7FFEE6,0x0").unwrap(),
                // How discovery reads `pw=true`.
                flags: if pin {
                    Flags::PASSWORD_REQUIRED
                } else {
                    Flags::empty()
                },
                pk: None,
                device_id: Some("loop-receiver".into()),
            }),
            ..Default::default()
        },
    }
}

async fn wait_event(
    events: &mut mpsc::Receiver<ManagerEvent>,
    what: &str,
    mut pick: impl FnMut(&ManagerEvent) -> bool,
) {
    let deadline = Instant::now() + Duration::from_secs(10);
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

async fn mirror_loop(pin: bool) {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,ferricast_airplay=info".into()),
        )
        .with_test_writer()
        .try_init();

    let receiver = AirPlayReceiver::new(
        AirPlayReceiverConfig {
            name: "ferricast loop".into(),
            require_pin: pin,
            credentials: CredentialStorage::Ephemeral,
            ..Default::default()
        }
        .with_ephemeral_ports(),
    )
    .await;
    let log = Arc::new(Mutex::new(SinkLog::default()));
    let log_factory = log.clone();
    let (mut manager, mut events) = StreamManager::builder()
        .with_receiver(TestReceiver(receiver.clone()))
        .register_audio_decoder::<ferricast::decoder::AacEldDecoder>()
        .with_h264_decoder()
        .set_sink_factory(move |_remote, info| {
            assert!(info.video.is_some(), "stream announced without video");
            assert!(info.audio.is_some(), "stream announced without audio");
            log_factory.lock().unwrap().sinks += 1;
            let sink = HeadlessSink(log_factory.clone());
            async move { Ok(Box::new(sink) as Box<dyn FrameSink>) }
        })
        .build_with_events();
    manager.start_receivers().await.unwrap();
    let port = receiver.bound_rtsp_port().expect("control port bound");
    let addr = SocketAddr::from(([127, 0, 0, 1], port));

    // ── Pairing ──
    let mut session = AirPlaySession::ephemeral();
    match session.connect(&device(addr, pin)).await.unwrap() {
        ConnectOutcome::Ready => assert!(!pin, "PIN mode connected without a PIN"),
        ConnectOutcome::PairingRequired(_) => {
            assert!(pin, "pairing asked for without PIN mode");
            let mut shown = None;
            wait_event(&mut events, "the PIN", |ev| {
                if let ManagerEvent::ReceiverPin { pin: Some(p), .. } = ev {
                    shown = Some(p.clone());
                    true
                } else {
                    false
                }
            })
            .await;
            let shown = shown.unwrap();
            assert_eq!(shown.len(), 4);
            session
                .submit_pairing(PairingResponse::Pin(shown))
                .await
                .expect("PIN pairing");
        }
    }

    // ── fp-setup + SETUP ──
    session
        .setup_stream(&StreamConfig {
            width: WIDTH,
            height: HEIGHT,
            fps: FPS,
            bitrate_kbps: 3000,
            codec: Codec::H264,
            audio: Some(AudioStreamConfig {
                codec: AudioCodec::AacEld,
                sample_rate: ELD_SAMPLE_RATE,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect("setup_stream");

    // ── Stream ──
    let mut encoder = X264Encoder::default();
    encoder
        .configure(&EncoderConfig {
            width: WIDTH,
            height: HEIGHT,
            fps: FPS,
            bitrate_kbps: 3000,
            keyframe_interval_secs: 2.0,
            pixel_format: PixelFormat::Bgra,
            ..Default::default()
        })
        .unwrap();
    let mut audio_encoder = FdkEldEncoder::new();
    audio_encoder
        .configure(&AudioEncoderConfig {
            codec: AudioCodec::AacEld,
            sample_rate: ELD_SAMPLE_RATE,
            channels: 2,
            bitrate_kbps: 128,
        })
        .unwrap();
    let start_us = now_us();
    let mut samples_sent = 0_u64;
    let mut tick = tokio::time::interval(Duration::from_secs(1) / FPS);
    for n in 0..SECONDS * FPS {
        tick.tick().await;
        let now = now_us();
        let frame = CapturedFrame::Cpu(RawFrame {
            width: WIDTH,
            height: HEIGHT,
            stride: WIDTH * 4,
            format: PixelFormat::Bgra,
            data: pattern(n),
            timestamp_us: now,
        });
        let encoded = encoder.encode(frame).unwrap();
        session.send_frame(&encoded).await.unwrap();
        let due = (now - start_us) * u64::from(ELD_SAMPLE_RATE) / 1_000_000;
        audio_encoder
            .encode(&AudioFrame {
                codec: AudioCodec::Pcm,
                data: tone(samples_sent, due - samples_sent),
                timestamp_us: start_us + samples_sent * 1_000_000 / u64::from(ELD_SAMPLE_RATE),
                sample_rate: ELD_SAMPLE_RATE,
                channels: 2,
            })
            .unwrap();
        samples_sent = due;
        for aac in audio_encoder.take_output() {
            session.send_audio_frame(&aac).await.unwrap();
        }
        assert!(session.is_alive(), "receiver dropped the session");
    }
    // Let the pipeline present what is queued.
    tokio::time::sleep(Duration::from_millis(500)).await;

    // ── TEARDOWN ──
    session.stop().await.unwrap();
    wait_event(&mut events, "the session to stop", |ev| {
        matches!(ev, ManagerEvent::ReceiverStopped { .. })
    })
    .await;
    manager.stop_receivers().await.unwrap();

    // ── What came out ──
    let log = log.lock().unwrap();
    assert_eq!(log.sinks, 1, "one stream announced");
    let video = &log.video;
    println!(
        "pin={pin}: {} video frames, {} audio samples",
        video.len(),
        log.pcm.len() / 2
    );
    assert!(
        video.len() >= (SECONDS * FPS) as usize * 3 / 4,
        "only {} of {} frames shown",
        video.len(),
        SECONDS * FPS
    );
    assert!(video.iter().all(|&(w, h, _)| (w, h) == (WIDTH, HEIGHT)));
    assert!(
        video.windows(2).all(|w| w[0].2 < w[1].2),
        "PTS not monotonic"
    );
    // PTS are mapped to local wall-clock time.
    let first_pts = video[0].2;
    assert!(
        first_pts.abs_diff(start_us) < 2_000_000,
        "PTS {first_pts} vs start {start_us}"
    );

    assert_eq!(log.audio_shape, Some((44_100, 2)));
    let mono: Vec<f64> = log.pcm.chunks_exact(2).map(|s| f64::from(s[0])).collect();
    assert!(
        mono.len() >= 44_100 * (SECONDS as usize - 1),
        "only {} audio samples",
        mono.len()
    );
    // One second from the middle, past the lead-in silence.
    let mid = mono.len() / 2;
    let window = &mono[mid - 22_050..mid + 22_050];
    let rms = (window.iter().map(|x| x * x).sum::<f64>() / window.len() as f64).sqrt();
    assert!(rms > 1000.0, "audio is (nearly) silent: rms {rms}");
    let peak = peak_frequency(window, 44_100.0);
    println!("pin={pin}: audio peak at {peak} Hz, rms {rms:.0}");
    assert!((peak - TONE_HZ).abs() <= 10.0, "tone at {peak} Hz");
}

#[test]
fn fairplay_ekey_of_the_sender_decrypts_to_its_key() {
    let key = ferricast_fairplay::decrypt_ekey(
        &ferricast_fairplay::KEY_MESSAGE,
        &ferricast_fairplay::EKEY,
    )
    .unwrap();
    assert_eq!(key, ferricast_fairplay::AES_KEY);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mirrors_to_our_own_receiver() {
    mirror_loop(false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mirrors_to_our_own_receiver_with_a_pin() {
    mirror_loop(true).await;
}
