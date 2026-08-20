use std::collections::HashMap;
use std::io::{Read, Write};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ed25519_dalek::SigningKey;
use hap_tlv8::Tlv8Writer;
use rand::Rng;
use rand::rngs::OsRng;
use sha2::Sha512;
use srp::client::SrpClient;
use srp::groups::{G_2048, G_3072};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tracing::{info, warn};
use uuid::Uuid;

use ferricast_core::{
    CastSession, Codec, Device, EncodedFrame, FerricastError, Result, StreamConfig,
};

use crate::rtsp::RtspManager;

const TLV_TYPE_STATE: u8 = 6;
const TLV_TYPE_METHOD: u8 = 0;




/// Internal state of the AirPlay session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionState {
    /// Initial state, not connected.
    Disconnected,
    /// TCP connection established but not yet paired/negotiated.
    Connected,
    /// Pair-Verify completed, encryption keys established.
    Paired,
    /// RTSP SETUP completed, data channel ready.
    Ready,
    /// RTSP RECORD sent, actively streaming.
    Streaming,
    /// Session is being torn down.
    TearingDown,
}

impl std::fmt::Display for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disconnected => write!(f, "Disconnected"),
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
        }
    }
}

impl AirPlaySession {
    /// Get the session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

impl CastSession for AirPlaySession {
    async fn connect(&mut self, device: &Device) -> Result<()> {
        if device.protocol != "airplay" {
            return Err(FerricastError::Protocol(format!(
                "Expected AirPlay device, got {:?}",
                device.protocol
            )));
        }

        if self.state != SessionState::Disconnected {
            return Err(FerricastError::SessionAlreadyActive(device.name.clone()));
        }



      println!("{}", format!("{:?}:{}", device.addr, device.port));

      let mut socket = TcpStream::connect(format!("{:?}:{}", device.addr, device.port))
          .await
          .map_err(|e| FerricastError::Connection(format!("Cannot connect to AirPlay device {e}")))?;

        info!("Connecting to Airplay device"); 

        let manager = RtspManager::new();

        manager.builder()
            .path("/pair-pin-start".to_string())
            .write(&mut socket)
            .await?;

        let mut buffer = vec![0_u8; 4096];

        let n = socket.read(&mut buffer).await?;


        println!("{:?}", String::from_utf8(buffer[..n].to_vec()));


        let srp = SrpClient::<Sha512>::new(&G_3072);
    
        let mut bytes = Vec::new();

        let mut w = Tlv8Writer::new(&mut bytes);

        let mut data = [0_u8; 32];

        data[0] = 1;

        w.push(TLV_TYPE_STATE, &data);

        w.push(TLV_TYPE_METHOD, &[0]);

        
        manager.builder()
            .path("/pair-setup".to_string())
            .content_type("application/octet-stream".to_string())
            .body(bytes)
            .write(&mut socket)
            .await?;

                let n = socket.read(&mut buffer).await?;


        println!("{:?}", String::from_utf8(buffer[..n].to_vec()));



        
    


        /*
        manager.builder()
            .path("/pair-setup".to_string())
            .header(("User-Agent".to_string(), "AirPlay/381.13".to_string()))
            .header(("X-Apple-HKP".to_string(), "3".to_string()))
            .header(("X-Apple-Client-Name".to_string(), "Ferricast Airplay".to_string()));
        */


        // IMPROVE PAIRING!
        //
        loop {}


    

        /*
    
        socket.write(b"POST /pair-pin-start HTTP/1.0\r\nUser-Agent: Airplay/320.20\r\nConnection: keep-alive\r\n\r\n")
            .await
            .map_err(|e| FerricastError::Connection(format!("Cannot write to Airplay device {e}")))?;

        let (client_id, _) = generate_auth_token();

        println!("nice!");

        let mut buf = vec![0_u8; 8096];

        socket.read(&mut buf).await.unwrap();



        let mut pair_setup_data = HashMap::new();
        pair_setup_data.insert("method", "pin".to_string());
        pair_setup_data.insert("user", generate_device_id());

        let mut pair_setup_bin = Vec::new();

        // I hate apple
        plist::to_writer_binary(&mut pair_setup_bin, &pair_setup_data)
            .map_err(|e| FerricastError::Connection(format!("Cannot encode plist {e}")))?;

        socket.write(format!("POST /pair-setup-pin HTTP/1.0\r\nUser-Agent: AirPlay/320.20\r\nConnection: keep-alive\r\nContent-Length: {}\r\nContent-Type: application/x-apple-binary-plist\r\n\r\n", pair_setup_bin.len()).as_bytes())
            .await
            .map_err(|e| FerricastError::Connection(format!("Cannot write to Airplay device {e}")))?;

        socket.write(&pair_setup_bin)
            .await
            .map_err(|e| FerricastError::Connection(format!("Cannot write to Airplay device {e}")))?;


        println!("pin:");

        //let mut buffer = vec![0u8; 4];

        //std::io::stdin().read_exact(&mut buffer).unwrap();

        //let pin = String::from_utf8(buffer).unwrap();

        //let srp_client = SrpClient::<sha1::Sha1>::new(&G_2048);

        

             let n = socket.read(&mut buf).await.unwrap();

             
             let n = socket.read(&mut buf).await.unwrap();



        println!("{:?}", String::from_utf8_lossy(&buf[..n]));
        */

        Ok(())
    }

    async fn setup_stream(&mut self, config: &StreamConfig) -> Result<()> {
        if self.state != SessionState::Connected && self.state != SessionState::Paired {
            return Err(FerricastError::Protocol(format!(
                "Cannot setup stream in state: {}",
                self.state
            )));
        }

        // Validate codec
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
            "Setting up AirPlay stream"
        );

        self.state = SessionState::Streaming;
        info!("AirPlay stream is now active");

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
            "Stopping AirPlay session"
        );

        self.state = SessionState::TearingDown;

        info!("AirPlay session stopped");
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

/// Generate a random device ID in MAC address format.
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

    for i in 0..=len {
        string.push(rng.gen_range(48..=90) as u8 as char);
    }

    string
}
