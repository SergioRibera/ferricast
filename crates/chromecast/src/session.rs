use std::sync::Arc;

use rustls::pki_types::ServerName;
use tokio::sync::Mutex;
use tracing::debug;

use ferricast_core::{CastSession, Device, EncodedFrame, Result, StreamConfig};

type TlsStream = tokio_rustls::client::TlsStream<tokio::net::TcpStream>;

/// A shared, split TLS writer half protected by a mutex so multiple tasks
/// (heartbeat, frame sender) can write concurrently.
type SharedWriter = Arc<Mutex<tokio::io::WriteHalf<TlsStream>>>;

#[derive(Default)]
pub struct ChromecastSession;

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct DeviceInfo {
    name: String,
    addr: std::net::IpAddr,
    port: u16,
}

impl CastSession for ChromecastSession {
    async fn connect(&mut self, device: &Device) -> Result<()> {
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
