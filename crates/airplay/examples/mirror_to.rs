//! Mirror a synthetic test pattern to an AirPlay receiver by address.
//!
//! ```text
//! cargo run -p ferricast-airplay --features fairplay --example mirror_to -- 192.168.1.20:7000 [seconds] [--pin]
//! ```
//!
//! `--pin` treats the receiver as PIN-protected (UxPlay `-pin`); the PIN
//! shown by the receiver is then read from stdin.

use std::collections::HashMap;
use std::io::BufRead as _;
use std::net::SocketAddr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use ferricast_airplay::AirPlaySession;
use ferricast_core::device::{AirplayConfig, Features, Flags};
use ferricast_core::{
    CapturedFrame, CastSession, Codec, ConnectOutcome, Device, DeviceCapabilities, EncoderConfig,
    PairingResponse, PixelFormat, RawFrame, StreamConfig, VideoEncoder,
};
use ferricast_encoder::h264::X264Encoder;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const FPS: u32 = 30;

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

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
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
        .ok_or("usage: mirror_to <ip:port> [seconds] [--pin]")?
        .parse()?;
    let seconds: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let pin = args.iter().any(|a| a == "--pin");

    let device = Device {
        id: uuid::Uuid::new_v4(),
        name: "receiver".into(),
        protocol: "airplay",
        protocol_icon: Bytes::new(),
        addr: addr.ip(),
        port: addr.port(),
        model: None,
        metadata: HashMap::new(),
        capabilities: DeviceCapabilities {
            supports_screen_mirror: true,
            airplay_config: Some(AirplayConfig {
                // What UxPlay advertises.
                features: Features::parse_txt("0x5A7FFEE6,0x0").unwrap(),
                flags: if pin {
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

    let mut session = AirPlaySession::default();
    if let ConnectOutcome::PairingRequired(challenge) = session.connect(&device).await? {
        println!("receiver asks for {challenge:?}; type the PIN it shows:");
        let pin = std::io::stdin().lock().lines().next().ok_or("no PIN")??;
        session.submit_pairing(PairingResponse::Pin(pin)).await?;
    }

    let config = StreamConfig {
        width: WIDTH,
        height: HEIGHT,
        fps: FPS,
        bitrate_kbps: 3000,
        codec: Codec::H264,
        ..Default::default()
    };
    session.setup_stream(&config).await?;

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

    let mut tick = tokio::time::interval(Duration::from_secs(1) / FPS);
    for n in 0..seconds * FPS {
        tick.tick().await;
        let frame = CapturedFrame::Cpu(RawFrame {
            width: WIDTH,
            height: HEIGHT,
            stride: WIDTH * 4,
            format: PixelFormat::Bgra,
            data: pattern(n),
            timestamp_us: now_us(),
        });
        let encoded = encoder.encode(frame)?;
        session.send_frame(&encoded).await?;
        if !session.is_alive() {
            return Err("receiver dropped the session".into());
        }
    }
    session.stop().await?;
    println!("sent {} frames", seconds * FPS);
    Ok(())
}
