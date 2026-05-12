//! Async CASTv2 framing on top of a `tokio-rustls` TLS stream.
//!
//! Chromecast speaks length-prefixed protobuf over TLS on TCP/8009.
//! This module owns the TLS handshake (with a permissive
//! `ServerCertVerifier` because Chromecasts ship self-signed certs),
//! splits the stream into read / write halves, and exposes a
//! `BufWrite` that's `Arc<Mutex<…>>`-shareable so concurrent senders
//! (heartbeat, control, media) never interleave bytes.

use std::sync::Arc;

use bytes::BytesMut;
use rustls::ClientConfig;
use rustls::client::danger::ServerCertVerifier;
use rustls::pki_types::ServerName;
use tokio::io::{AsyncReadExt, AsyncWriteExt, ReadHalf, WriteHalf};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio_rustls::TlsConnector;
use tokio_rustls::client::TlsStream;
use tracing::debug;

use ferricast_core::{FerricastError, Result};

use crate::castv2::{CastMessage, MAX_MESSAGE_SIZE};

pub type SharedWriter = Arc<Mutex<WriteHalf<TlsStream<TcpStream>>>>;

/// Connect to a Chromecast at `addr:port`, doing the rustls handshake
/// with a verifier that accepts self-signed certs. Returns split
/// read/write halves and the local TCP address — the latter is the IP
/// the chromecast can route back to, which we use to advertise our
/// embedded HLS URL.
pub async fn connect(
    addr: std::net::IpAddr,
    port: u16,
) -> Result<(
    ReadHalf<TlsStream<TcpStream>>,
    SharedWriter,
    std::net::IpAddr,
)> {
    // TCP connect with a 10 s ceiling. Healthy LAN connect is sub-
    // millisecond; 10 s gives us margin for slow Wi-Fi while still
    // surfacing a wedged target as an error rather than hanging
    // forever.
    let tcp = match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        TcpStream::connect((addr, port)),
    )
    .await
    {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            return Err(FerricastError::Connection(format!(
                "tcp connect to {addr}:{port}: {e}"
            )));
        }
        Err(_) => {
            return Err(FerricastError::Timeout(format!(
                "tcp connect to {addr}:{port} did not complete in 10s"
            )));
        }
    };

    // The local end of the TLS socket is the address the chromecast
    // can route back to — kernel already picked the right
    // interface to reach `addr`, so this is authoritative. Fall
    // back to the shared `local_addr_for` UDP probe in the
    // (unlikely) case `local_addr()` returns something unroutable
    // (0.0.0.0 / ::) — that's the same trick but keeps us using
    // the shared helper for the diagnostic, which is what the
    // other receiver protocols will do when they don't have a TCP
    // connection to derive from.
    let local_ip = {
        let from_socket = tcp.local_addr().map_err(FerricastError::from)?.ip();
        if from_socket.is_unspecified() {
            ferricast_core::local_addr_for(addr)?
        } else {
            from_socket
        }
    };

    // Force TLS 1.2 only. Chromecast firmware uniformly speaks 1.2;
    // some 1st/2nd-gen models reject ClientHellos that *also*
    // advertise TLS 1.3 (which rustls 0.23 does by default). The
    // safer floor is to negotiate exactly what the receiver wants.
    let config = ClientConfig::builder_with_protocol_versions(&[&rustls::version::TLS12])
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(NoCertVerifier))
        .with_no_client_auth();

    let connector = TlsConnector::from(Arc::new(config));
    let server_name = ServerName::IpAddress(addr.into());

    // 5 s ceiling on the TLS handshake. A healthy chromecast acks
    // the ClientHello in <50 ms; if we've waited 5 s the receiver
    // is in a wedged state (mid-firmware-update, locked-up app,
    // or stuck post-crash) and needs a power cycle. Without this
    // timeout, `connect()` would hang the whole streaming task
    // indefinitely — observed in the field as a 30+ s "nothing
    // happens after `connecting to chromecast` log" gap.
    let tls = match tokio::time::timeout(
        std::time::Duration::from_secs(5),
        connector.connect(server_name, tcp),
    )
    .await
    {
        Ok(Ok(t)) => t,
        Ok(Err(e)) => {
            return Err(FerricastError::Connection(format!("tls handshake: {e}")));
        }
        Err(_) => {
            return Err(FerricastError::Timeout(format!(
                "TLS handshake to {addr}:{port} did not complete in 5s — \
                 receiver is wedged; physically power-cycle the chromecast"
            )));
        }
    };
    tracing::info!(%addr, port, %local_ip, "TLS handshake to chromecast complete");

    let (read_half, write_half) = tokio::io::split(tls);
    Ok((read_half, Arc::new(Mutex::new(write_half)), local_ip))
}

/// Send a length-prefixed `CastMessage` on the shared writer.
///
/// Hard 5 s ceiling on the whole send (lock + write + flush). Tiny
/// CASTv2 messages (≤ a few hundred bytes) on a healthy LAN take
/// sub-millisecond; anything past 5 s means the receiver wedged
/// the TLS stream (we've seen this when a chromecast firmware
/// bug or a third-party receiver app stops draining the TCP
/// socket). Surface it as an explicit error so connect() doesn't
/// hang forever.
pub async fn send(writer: &SharedWriter, msg: &CastMessage) -> Result<()> {
    let bytes = msg
        .encode_length_prefixed()
        .map_err(|e| FerricastError::Protocol(format!("encode cast message: {e}")))?;
    tracing::info!(
        ns = %msg.namespace,
        src = %msg.source_id,
        dst = %msg.destination_id,
        ty = msg.message_type().as_deref().unwrap_or("(binary)"),
        size = bytes.len(),
        "cast→ sending"
    );
    let send_inner = async {
        let mut w = writer.lock().await;
        w.write_all(&bytes)
            .await
            .map_err(|e| FerricastError::Connection(format!("write cast message: {e}")))?;
        w.flush()
            .await
            .map_err(|e| FerricastError::Connection(format!("flush: {e}")))?;
        Ok::<_, FerricastError>(())
    };
    match tokio::time::timeout(std::time::Duration::from_secs(5), send_inner).await {
        Ok(r) => r,
        Err(_) => Err(FerricastError::Timeout(format!(
            "wire::send (ns={}, ty={:?}) blocked >5s — receiver is wedged; \
             try power-cycling the chromecast",
            msg.namespace,
            msg.message_type()
        ))),
    }
}

/// Read one length-prefixed `CastMessage` from the stream, blocking
/// until a full frame is available. Returns `Ok(None)` only on a
/// clean EOF.
pub async fn recv<R: tokio::io::AsyncRead + Unpin>(
    reader: &mut R,
    buf: &mut BytesMut,
) -> Result<Option<CastMessage>> {
    loop {
        match CastMessage::decode_length_prefixed(buf) {
            Ok(Some(msg)) => {
                tracing::info!(
                    ns = %msg.namespace,
                    src = %msg.source_id,
                    dst = %msg.destination_id,
                    ty = msg.message_type().as_deref().unwrap_or("(binary)"),
                    "cast← received"
                );
                return Ok(Some(msg));
            }
            Ok(None) => {}
            Err(e) => {
                return Err(FerricastError::Protocol(format!(
                    "decode cast message: {e}"
                )));
            }
        }

        // Need more bytes — make sure the buffer has headroom and
        // refuse anything that's blatantly past the spec maximum so a
        // malicious / bug-ridden peer can't OOM us.
        if buf.len() > MAX_MESSAGE_SIZE + 8 {
            return Err(FerricastError::Protocol(
                "incoming cast frame exceeds MAX_MESSAGE_SIZE".into(),
            ));
        }
        let n = reader
            .read_buf(buf)
            .await
            .map_err(|e| FerricastError::Connection(format!("read: {e}")))?;
        if n == 0 {
            debug!("cast peer closed connection");
            return Ok(None);
        }
    }
}

/// Permissive verifier — Chromecasts present a self-signed cert with
/// a CN that doesn't match the IP we dialled. Stock rustls would
/// reject; we don't need authentication of the receiver itself, only
/// confidentiality of the LAN traffic.
#[derive(Debug)]
struct NoCertVerifier;

impl ServerCertVerifier for NoCertVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
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
