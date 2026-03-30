use std::sync::Arc;

use rust_cast::{channels::{heartbeat::HeartbeatResponse, media::Media, receiver::CastDeviceApp}, CastDevice, ChannelMessage};
use rustls::pki_types::ServerName;
use serde::de;
use tokio::sync::Mutex;
use tracing::debug;

use ferricast_core::{CastSession, Device, EncodedFrame, Result, StreamConfig};

type TlsStream = tokio_rustls::client::TlsStream<tokio::net::TcpStream>;

const DEFAULT_DESTINATION_ID: &str = "receiver-0";


/// A shared, split TLS writer half protected by a mutex so multiple tasks
/// (heartbeat, frame sender) can write concurrently.
type SharedWriter = Arc<Mutex<tokio::io::WriteHalf<TlsStream>>>;

const URL: &'static str = "<video url in https>";

#[derive(Default)]
pub struct ChromecastSession {
    device: Option<CastDevice<'static>>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct DeviceInfo {
    name: String,
    addr: std::net::IpAddr,
    port: u16,
}

impl CastSession for ChromecastSession {
    async fn connect(&mut self, device: &Device) -> Result<()> {
        let device = CastDevice::connect_without_host_verification(device.addr.to_string(), device.port).unwrap();

        println!("connecting");
        device.connection.connect(DEFAULT_DESTINATION_ID.to_string()).unwrap();
        device.heartbeat.ping().unwrap();
    

    
        println!("connected");

        let device_app = CastDeviceApp::DefaultMediaReceiver;
        println!("{:?}", device_app);

        let app = device.receiver.launch_app(&device_app).unwrap();
        println!("launched");

        device.connection.connect(app.transport_id.as_str()).unwrap();

        device.media.load(app.transport_id.as_str(), app.session_id.as_str(), &Media {
            content_id: URL.to_string(),
            content_type: "video/mp4".to_string(),
            metadata: None,
            stream_type: rust_cast::channels::media::StreamType::Live,
            duration: None,
        }).unwrap();
        println!("sending media");

        loop {
            match device.receive() {
                Ok(ChannelMessage::Heartbeat(response)) => {
                    println!("[Heartbeat] {:?}", response);

                    if let HeartbeatResponse::Ping = response {
                        device.heartbeat.pong().unwrap();
                    }
                }

                Ok(ChannelMessage::Connection(response)) => println!("[Connection] {:?}", response),
                Ok(ChannelMessage::Media(response)) => println!("[Media] {:?}", response),
                Ok(ChannelMessage::Receiver(response)) => println!("[Receiver] {:?}", response),
                Ok(ChannelMessage::Raw(response)) => println!(
                    "Support for the following message type is not yet supported: {:?}",
                    response
                ),

                Err(error) => panic!("Error occurred while receiving message {}", error),
            }
        }
    
    


        

        self.device = Some(device);
        
        Ok(())
    }

    async fn setup_stream(&mut self, config: &StreamConfig) -> Result<()> {
         
        Ok(())
    }

    async fn send_frame(&mut self, frame: &EncodedFrame) -> Result<()> {
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        Ok(())
    }

    fn is_alive(&self) -> bool {
        false
    }
}

#[derive(Debug)]
struct NoCertVerifier;

impl rustls::client::danger::ServerCertVerifier for NoCertVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        debug!("accepting chromecast self-signed certificate");
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::RSA_PKCS1_SHA384,
            rustls::SignatureScheme::RSA_PKCS1_SHA512,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::ECDSA_NISTP521_SHA512,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
            rustls::SignatureScheme::ED25519,
            rustls::SignatureScheme::ED448,
        ]
    }
}
