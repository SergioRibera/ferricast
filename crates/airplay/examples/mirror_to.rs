//! Mirror a synthetic test pattern and a 440 Hz tone to an AirPlay
//! receiver by address.
//!
//! ```text
//! cargo run -p ferricast-airplay --features fairplay --example mirror_to -- 192.168.1.20:7000 [seconds] [--pin] [--password[=<pw>[,<pw>...]]] [--no-audio] [--ports=<first>-<last>]
//! ```
//!
//! `--pin` treats the receiver as PIN-protected (UxPlay `-pin`); the PIN
//! shown by the receiver is then read from stdin. `--password` marks it as
//! password-protected (UxPlay `-pw`). Whenever the receiver asks for its
//! password, the next `=<pw>[,<pw>...]` entry answers, then stdin.
//! `--ports` binds the local UDP sockets (timing, audio) in that range.

use std::collections::{HashMap, VecDeque};
use std::io::BufRead as _;
use std::net::SocketAddr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use ferricast_airplay::AirPlaySession;
use ferricast_core::device::{AirplayConfig, Features, Flags};
use ferricast_core::{
    AudioCodec, AudioEncoder, AudioEncoderConfig, AudioFrame, AudioStreamConfig, CapturedFrame,
    CastSession, Codec, ConnectOutcome, Device, DeviceCapabilities, EncoderConfig, FerricastError,
    PairingChallenge, PairingResponse, PixelFormat, RawFrame, StreamConfig, VideoEncoder,
};
use ferricast_encoder::aac::{ELD_SAMPLE_RATE, FdkEldEncoder};
use ferricast_encoder::h264::X264Encoder;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const FPS: u32 = 30;
const TONE_HZ: f64 = 440.0;
/// Password prompts answered before giving up, like the manager.
const MAX_PASSWORDS: u32 = 3;

fn pattern(frame: u32) -> Bytes {
    let mut data = vec![0_u8; (WIDTH * HEIGHT * 4) as usize];
    for (i, px) in data.chunks_exact_mut(4).enumerate() {
        let x = (i as u32) % WIDTH;
        let y = (i as u32) / WIDTH;
        px[0] = ((x + frame * 4) % 256) as u8;
        px[1] = ((y + frame * 2) % 256) as u8;
        px[2] = if (x / 80 + y / 80 + frame / 15).is_multiple_of(2) {
            220
        } else {
            30
        };
        px[3] = 255;
    }
    Bytes::from(data)
}

/// `count` stereo S16LE samples of the test tone, starting at sample `start`.
fn tone(start: u64, count: u64) -> Bytes {
    let mut out = Vec::with_capacity(count as usize * 4);
    for n in start..start + count {
        let t = n as f64 / f64::from(ELD_SAMPLE_RATE);
        let v = ((2.0 * std::f64::consts::PI * TONE_HZ * t).sin() * 6000.0) as i16;
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

fn next_password(
    passwords: &mut VecDeque<String>,
    rejected: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    if rejected {
        println!("receiver rejected the password");
    }
    match passwords.pop_front() {
        Some(password) => Ok(password),
        None => read_line("receiver asks for its password; type it:"),
    }
}

fn read_line(prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
    println!("{prompt}");
    Ok(std::io::stdin().lock().lines().next().ok_or("no input")??)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ferricast_airplay=debug".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let addr: SocketAddr = args
        .first()
        .ok_or("usage: mirror_to <ip:port> [seconds] [--pin] [--password[=<pw>[,<pw>...]]] [--no-audio] [--ports=<first>-<last>]")?
        .parse()?;
    let seconds: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let pin = args.iter().any(|a| a == "--pin");
    let password = args
        .iter()
        .find_map(|a| a.strip_prefix("--password"))
        .map(|rest| rest.strip_prefix('=').map(str::to_owned));
    let audio = !args.iter().any(|a| a == "--no-audio");
    // `--ports=<first>-<last>`: bind the local UDP sockets in this range
    // (one a host firewall lets the receiver reach).
    let port_range = args
        .iter()
        .find_map(|a| a.strip_prefix("--ports="))
        .and_then(|r| r.split_once('-'))
        .and_then(|(a, b)| Some((a.parse::<u16>().ok()?, b.parse::<u16>().ok()?)));

    let device = Device {
        id: uuid::Uuid::new_v4(),
        name: "receiver".into(),
        protocol: "airplay",
        protocol_icon: Bytes::new(),
        addresses: addr.into(),
        model: None,
        metadata: HashMap::new(),
        capabilities: DeviceCapabilities {
            supports_screen_mirror: true,
            airplay_config: Some(AirplayConfig {
                // What UxPlay advertises.
                features: Features::parse_txt("0x5A7FFEE6,0x0").unwrap(),
                // UxPlay advertises `pw=true` for both `-pin` and `-pw`.
                flags: if pin || password.is_some() {
                    Flags::PASSWORD_REQUIRED
                } else {
                    Flags::empty()
                },
                pk: None,
                device_id: Some(format!("example-{addr}")),
            }),
            ..Default::default()
        },
    };

    let mut passwords: VecDeque<String> = password
        .flatten()
        .map(|list| list.split(',').map(str::to_owned).collect())
        .unwrap_or_default();

    let mut session = AirPlaySession::default();
    if let ConnectOutcome::PairingRequired(challenge) = session.connect(&device).await? {
        let response = match challenge {
            PairingChallenge::Credential { rejected } => {
                PairingResponse::Password(next_password(&mut passwords, rejected)?)
            }
            other => PairingResponse::Pin(read_line(&format!(
                "receiver asks for {other:?}; type the PIN it shows:"
            ))?),
        };
        session.submit_pairing(response).await?;
    }

    let config = StreamConfig {
        width: WIDTH,
        height: HEIGHT,
        fps: FPS,
        bitrate_kbps: 3000,
        codec: Codec::H264,
        audio: audio.then(|| AudioStreamConfig {
            codec: AudioCodec::AacEld,
            sample_rate: ELD_SAMPLE_RATE,
            ..Default::default()
        }),
        port_range,
        ..Default::default()
    };
    // Like the manager: a password-protected receiver challenges `SETUP`;
    // answer and retry on the same session.
    let mut asked = 0;
    loop {
        match session.setup_stream(&config).await {
            Ok(()) => break,
            Err(FerricastError::CredentialRequired { rejected }) if asked < MAX_PASSWORDS => {
                asked += 1;
                let password = next_password(&mut passwords, rejected)?;
                session
                    .submit_pairing(PairingResponse::Password(password))
                    .await?;
            }
            Err(e) => return Err(e.into()),
        }
    }

    let mut encoder = X264Encoder::default();
    encoder.configure(&EncoderConfig {
        width: WIDTH,
        height: HEIGHT,
        fps: FPS,
        bitrate_kbps: 3000,
        keyframe_interval_secs: 2.0,
        pixel_format: PixelFormat::Bgra,
        ..Default::default()
    })?;
    let mut audio_encoder = FdkEldEncoder::new();
    audio_encoder.configure(&AudioEncoderConfig {
        codec: AudioCodec::AacEld,
        sample_rate: ELD_SAMPLE_RATE,
        channels: 2,
        bitrate_kbps: 128,
        backend: ferricast_core::BackendSelection::default(),
    })?;

    let start_us = now_us();
    let mut samples_sent = 0_u64;
    let mut audio_frames = 0_u64;
    let mut tick = tokio::time::interval(Duration::from_secs(1) / FPS);
    for n in 0..seconds * FPS {
        tick.tick().await;
        let now = now_us();
        let frame = CapturedFrame::Cpu(RawFrame {
            colour: None,
            width: WIDTH,
            height: HEIGHT,
            stride: WIDTH * 4,
            format: PixelFormat::Bgra,
            data: pattern(n),
            timestamp_us: now,
        });
        let encoded = encoder.encode(frame)?;
        session.send_frame(&encoded).await?;

        if audio {
            // Feed the tone up to the current time, stamped like capture.
            let due = (now - start_us) * u64::from(ELD_SAMPLE_RATE) / 1_000_000;
            let count = due.saturating_sub(samples_sent);
            audio_encoder.encode(&AudioFrame {
                codec: AudioCodec::Pcm,
                data: tone(samples_sent, count),
                timestamp_us: start_us + samples_sent * 1_000_000 / u64::from(ELD_SAMPLE_RATE),
                sample_rate: ELD_SAMPLE_RATE,
                channels: 2,
            })?;
            samples_sent = due;
            for aac in audio_encoder.take_output() {
                session.send_audio_frame(&aac).await?;
                audio_frames += 1;
            }
        }
        if !session.is_alive() {
            return Err("receiver dropped the session".into());
        }
    }
    session.stop().await?;
    println!(
        "sent {} video frames, {audio_frames} audio frames",
        seconds * FPS
    );
    Ok(())
}
