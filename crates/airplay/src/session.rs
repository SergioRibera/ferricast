use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ed25519_dalek::SigningKey;
use hap_tlv8::Tlv8Writer;
use rand::Rng;
use rand::rngs::OsRng;
use tokio::io::BufReader;
use tokio::net::TcpStream;
use tracing::{info, warn};
use uuid::Uuid;

use ferricast_core::{
    CastSession, Codec, ConnectOutcome, Device, EncodedFrame, FerricastError,
    PairingChallenge, PairingResponse, Result, StreamConfig,
};

use crate::rtsp::{RtspManager, RtspResponse};

const TLV_TYPE_STATE: u8 = 6;
const TLV_TYPE_METHOD: u8 = 0;
const TLV_TYPE_FLAGS: u8 = 0x13;

const TLV_FLAGS_TRANSIENT: [u8; 4] = 0x00000010_u32.to_le_bytes();

/// Internal state of the AirPlay session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionState {
    Disconnected,
    /// TCP open, `/pair-pin-start` sent, waiting for PIN from user.
    AwaitingPin,
    Connected,
    Paired,
    Ready,
    Streaming,
    TearingDown,
}

#[derive(Debug, Clone, Copy)]
pub enum PairingMode {
    Legacy,
    Hap,
    Transient
}

impl std::fmt::Display for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disconnected => write!(f, "Disconnected"),
            Self::AwaitingPin => write!(f, "AwaitingPin"),
            Self::Connected => write!(f, "Connected"),
            Self::Paired => write!(f, "Paired"),
            Self::Ready => write!(f, "Ready"),
            Self::Streaming => write!(f, "Streaming"),
            Self::TearingDown => write!(f, "TearingDown"),
        }
    }
}

/// An AirPlay 2 screen mirroring session.
pub struct AirPlaySession {
    state: SessionState,
    session_id: String,
    client_device_id: String,
    alive: Arc<AtomicBool>,
    frame_counter: u64,
    pairing_mode: Option<PairingMode>,
    /// Held open between `connect()` and `submit_pairing()`.
    /// `connect()` sends `/pair-pin-start` and waits for the user to
    /// type the PIN shown on the TV screen; `submit_pairing()` takes
    /// this stream and continues the SRP-based pair-setup exchange.
    pending_conn: Option<TcpStream>,
}

impl Default for AirPlaySession {
    fn default() -> Self {
        let session_id = Uuid::new_v4().to_string();
        let client_device_id = generate_device_id();
        Self {
            state: SessionState::Disconnected,
            session_id,
            client_device_id,
            alive: Default::default(),
            frame_counter: Default::default(),
            pending_conn: None,
            pairing_mode: None,
        }
    }
}

impl AirPlaySession {
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

impl CastSession for AirPlaySession {
    async fn connect(&mut self, device: &Device) -> Result<ConnectOutcome> {
        if device.protocol != "airplay" {
            return Err(FerricastError::Protocol(format!(
                "Expected AirPlay device, got {:?}",
                device.protocol
            )));
        }

        let pair_method = match device.metadata.get("pair_method").expect("Ferricast bug in airplay metadata").as_str() {
            "Transient" => PairingMode::Transient,
            m => todo!("Invalid method {:?}", m),
        };
        
        if self.state != SessionState::Disconnected {
            return Err(FerricastError::SessionAlreadyActive(device.name.clone()));
        }

        info!(addr = %device.addr, port = device.port, "connecting to AirPlay device");

        let mut socket =
            TcpStream::connect((device.addr, device.port))
                .await
                .map_err(|e| {
                    FerricastError::Connection(format!("Cannot connect to AirPlay device: {e}"))
                })?;

        let manager = RtspManager::new(pair_method);



        {
            let (read_half, mut write_half) = socket.split();
            let mut buf_reader = BufReader::new(read_half);

            /*
            manager.builder()
                .path("*".to_string())
                .options()
                .header(("Client-Instance".to_string(), "56B29BB6CB904862".to_string()))
                .header(("DACP-ID".to_string(), "56B29BB6CB904862".to_string()))
                .header(("Active-Remote".to_string(), "1986535575".to_string()))
                .write(&mut write_half)
                .await?;

            let res = RtspResponse::read(&mut buf_reader).await?;

            let need_pin = !res.is_success();

            if need_pin {
                manager
                    .builder()
                    .path("/pair-pin-start".to_string())
                    .write(&mut write_half)
                    .await?;

                RtspResponse::read(&mut buf_reader).await?.is_ok()?;

                info!("Asking for Pin");

            }   
            */

            // Transient pairing

            let mut tlv_bytes = Vec::new();
            let mut w = Tlv8Writer::new(&mut tlv_bytes);
 
            w.push(TLV_TYPE_METHOD, &[0]);
            w.push(TLV_TYPE_STATE, &[1]);
            w.push(TLV_TYPE_FLAGS, &TLV_FLAGS_TRANSIENT);
            drop(w);

            manager
                .builder()
                .path("/pair-setup".to_string())
                .post()
                .content_type("application/octet-stream".to_string())
                .body(tlv_bytes)
                .write(&mut write_half)
                .await?;

            let mut csprng = OsRng;

            let signing_key = ed25519_dalek::SigningKey::generate(&mut csprng);

            let public_key = signing_key.verifying_key().to_bytes();

                   println!("{:?}",  RtspResponse::read(&mut buf_reader).await?);

            
             manager
                .builder()
                .path("/pair-setup".to_string())
                .post()
                .content_type("application/octet-stream".to_string())
                .body(public_key.to_vec())
                .write(&mut write_half)
                .await?;
            

                          println!("{:?}",  RtspResponse::read(&mut buf_reader).await?);


            
        
           
       
        }

        self.pairing_mode = Some(pair_method);
        self.pending_conn = Some(socket);
        self.state = SessionState::AwaitingPin;

        info!("AirPlay pair-pin-start sent; waiting for user PIN");
        Ok(ConnectOutcome::PairingRequired(PairingChallenge::Pin {
            digits: 4,
        }))
    }

    async fn submit_pairing(&mut self, response: PairingResponse) -> Result<()> {
        let pin = match response {
            PairingResponse::Pin(p) => p,
            PairingResponse::Cancelled => {
                self.state = SessionState::Disconnected;
                self.pending_conn = None;
                return Err(FerricastError::Protocol("AirPlay pairing cancelled".into()));
            }
            PairingResponse::Confirmed => {
                return Err(FerricastError::Protocol(
                    "AirPlay pairing expects a PIN, got Confirmed".into(),
                ))
            }
        };

        let mut socket = self.pending_conn.take().ok_or_else(|| {
            FerricastError::Protocol("submit_pairing called without a pending connection".into())
        })?;

        let manager = RtspManager::new(self.pairing_mode.ok_or_else(|| {
            FerricastError::Protocol("Invalid pairing mode".into())
        })?);

        {
            let (read_half, mut write_half) = socket.split();
            let mut buf_reader = BufReader::new(read_half);

            // TODO: complete SRP exchange using `pin`.
            // Steps (once SRP crate integration is worked out):
            //   1. derive verifier from PIN using srp::client::SrpClient<Sha512>
            //   2. POST /pair-setup with SRP M1 + PIN verifier
            //   3. read SRP M2 from device (server proof)
            //   4. verify server proof
            //   5. POST /pair-verify with our public key
            //   6. derive session keys
            let _ = &pin;
            let _ = &mut buf_reader;
            let _ = &mut write_half;
        }

        self.pending_conn = Some(socket);
        self.state = SessionState::Connected;
        info!("AirPlay pairing complete");
        Ok(())
    }

    async fn setup_stream(&mut self, config: &StreamConfig) -> Result<()> {
        if self.state != SessionState::Connected && self.state != SessionState::Paired {
            return Err(FerricastError::Protocol(format!(
                "Cannot setup stream in state: {}",
                self.state
            )));
        }
        if config.codec != Codec::H264 {
            return Err(FerricastError::UnsupportedCodec {
                codec: config.codec,
                protocol: "airplay",
            });
        }

        info!(
            width = config.width,
            height = config.height,
            fps = config.fps,
            bitrate_kbps = config.bitrate_kbps,
            "setting up AirPlay stream"
        );
        self.state = SessionState::Streaming;
        Ok(())
    }

    async fn send_frame(&mut self, frame: &EncodedFrame) -> Result<()> {
        if self.state != SessionState::Streaming {
            return Err(FerricastError::Streaming(format!(
                "Cannot send frame in state: {}",
                self.state
            )));
        }
        if frame.codec != Codec::H264 {
            return Err(FerricastError::UnsupportedCodec {
                codec: frame.codec,
                protocol: "airplay",
            });
        }
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        if self.state == SessionState::Disconnected {
            return Ok(());
        }
        info!(
            session_id = %self.session_id,
            frames_sent = self.frame_counter,
            "stopping AirPlay session"
        );
        self.state = SessionState::TearingDown;
        self.pending_conn = None;
        Ok(())
    }

    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
            && matches!(
                self.state,
                SessionState::Connected
                    | SessionState::Paired
                    | SessionState::Ready
                    | SessionState::Streaming
            )
    }
}

impl Drop for AirPlaySession {
    fn drop(&mut self) {
        if self.state != SessionState::Disconnected {
            warn!(
                session_id = %self.session_id,
                state = %self.state,
                "AirPlaySession dropped while still active"
            );
            self.alive.store(false, Ordering::Relaxed);
        }
    }
}

fn generate_device_id() -> String {
    let bytes: [u8; 6] = rand::random();
    format!(
        "{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5]
    )
}

fn generate_auth_token() -> (String, String) {
    let signing_key = SigningKey::generate(&mut OsRng);
    let private_key = signing_key.to_bytes();
    let hex = hex::encode(private_key);
    (random_string(16), hex)
}

fn random_string(len: usize) -> String {
    let mut string = String::with_capacity(len);
    let mut rng = rand::thread_rng();
    for _ in 0..len {
        string.push(rng.gen_range(48u8..=90) as char);
    }
    string
}
