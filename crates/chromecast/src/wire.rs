//! Async CASTv2 framing on top of a `tokio-rustls` TLS stream.
//!
//! Chromecast speaks length-prefixed protobuf over TLS on TCP/8009.
//! This module owns the TLS handshake (with a permissive
//! `ServerCertVerifier` because Chromecasts ship self-signed certs),
//! splits the stream into read / write halves, and exposes a
//! [`CastWriter`] that's `Arc`-shareable so concurrent senders
//! (heartbeat, control, media) never interleave bytes.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use bytes::BytesMut;
use rustls::ClientConfig;
use rustls::client::danger::ServerCertVerifier;
use rustls::pki_types::ServerName;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadHalf};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio_rustls::TlsConnector;
use tokio_rustls::client::TlsStream;
use tracing::{debug, warn};

use ferricast_core::{FerricastError, Result};

use crate::castv2::{CastMessage, CastV2Error, LENGTH_PREFIX_SIZE, MAX_MESSAGE_SIZE};

/// Hard ceiling on one [`CastWriter::send`] (lock + write + flush).
const SEND_TIMEOUT: Duration = Duration::from_secs(5);

/// Kernel-side dead-peer detection for the control socket. The
/// application heartbeat catches a silent receiver within ~15 s; these
/// options make the kernel agree, so a write into a vanished peer fails
/// in seconds instead of after `tcp_retries2` (~15 min) of
/// retransmissions.
const TCP_KEEPALIVE_IDLE: Duration = Duration::from_secs(5);
const TCP_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(2);
const TCP_KEEPALIVE_RETRIES: u32 = 3;
#[cfg(any(target_os = "linux", target_os = "android"))]
const TCP_USER_TIMEOUT: Duration = Duration::from_secs(15);

/// Write half of a CASTv2 channel, shared by every sender of one
/// connection so frames never interleave.
///
/// A failed or timed-out send may have written part of a frame, after
/// which the length-prefixed framing is unrecoverable. The writer then
/// marks itself broken and refuses every later send, so nothing is
/// appended to a corrupted stream; owners poll [`Self::is_broken`].
pub struct CastWriter {
    inner: Mutex<Box<dyn AsyncWrite + Send + Unpin>>,
    broken: AtomicBool,
}

pub type SharedWriter = Arc<CastWriter>;

impl CastWriter {
    /// Wrap any byte sink. Production uses the TLS write half; tests
    /// use an in-memory duplex.
    #[must_use]
    pub fn shared(writer: impl AsyncWrite + Send + Unpin + 'static) -> SharedWriter {
        Arc::new(Self {
            inner: Mutex::new(Box::new(writer)),
            broken: AtomicBool::new(false),
        })
    }

    /// `true` once a send failed; the channel must be torn down.
    pub fn is_broken(&self) -> bool {
        self.broken.load(Ordering::Acquire)
    }

    /// Send one length-prefixed `CastMessage`, bounded by
    /// [`SEND_TIMEOUT`]. Tiny CASTv2 messages on a healthy LAN take
    /// sub-millisecond; past 5 s the receiver stopped draining its
    /// socket (firmware bug, wedged receiver app).
    ///
    /// # Errors
    ///
    /// `Protocol` if the message cannot be encoded (nothing written),
    /// `Connection` if the channel is already broken or the write
    /// fails, `Timeout` if the send blocks past [`SEND_TIMEOUT`]. The
    /// last two break the channel for good.
    pub async fn send(&self, msg: &CastMessage) -> Result<()> {
        if self.is_broken() {
            return Err(FerricastError::Connection(
                "cast channel is broken by an earlier failed send".into(),
            ));
        }
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
            let mut w = self.inner.lock().await;
            w.write_all(&bytes)
                .await
                .map_err(|e| FerricastError::Connection(format!("write cast message: {e}")))?;
            w.flush()
                .await
                .map_err(|e| FerricastError::Connection(format!("flush: {e}")))
        };
        let result = match tokio::time::timeout(SEND_TIMEOUT, send_inner).await {
            Ok(r) => r,
            Err(_) => Err(FerricastError::Timeout(format!(
                "cast send (ns={}, ty={:?}) blocked >{}s — receiver is wedged; \
                 try power-cycling the chromecast",
                msg.namespace,
                msg.message_type(),
                SEND_TIMEOUT.as_secs()
            ))),
        };
        if let Err(e) = &result {
            // The frame may be half on the wire: poison the channel.
            if !self.broken.swap(true, Ordering::AcqRel) {
                warn!(%e, "cast send failed; channel marked broken");
            }
        }
        result
    }
}

/// Connect to a Chromecast at `addr:port`, doing the rustls handshake
/// with a verifier that accepts self-signed certs. Returns split
/// read/write halves and the local TCP address — the latter is the IP
/// the chromecast can route back to, which we use to advertise our
/// embedded HLS URL.
///
/// # Errors
///
/// `Connection` / `Timeout` when the TCP connect or TLS handshake fails.
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
    let tcp = match tokio::time::timeout(Duration::from_secs(10), TcpStream::connect((addr, port)))
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
    configure_dead_peer_detection(&tcp);

    // The local end of the TLS socket is the address the chromecast
    // can route back to — kernel already picked the right
    // interface to reach `addr`, so this is authoritative. Fall
    // back to the shared `local_addr_for` UDP probe in the
    // (unlikely) case `local_addr()` returns something unroutable
    // (0.0.0.0 / ::).
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
    //
    // The provider is pinned: the workspace enables both ring and
    // aws-lc-rs, so the process-level default is ambiguous unless the
    // binary installed one, and rustls panics in that case.
    let config =
        ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_protocol_versions(&[&rustls::version::TLS12])
            .map_err(|e| FerricastError::Connection(format!("tls config: {e}")))?
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(NoCertVerifier))
            .with_no_client_auth();

    let connector = TlsConnector::from(Arc::new(config));
    let server_name = ServerName::IpAddress(addr.into());

    // 5 s ceiling on the TLS handshake. A healthy chromecast acks
    // the ClientHello in <50 ms; if we've waited 5 s the receiver
    // is in a wedged state (mid-firmware-update, locked-up app,
    // or stuck post-crash) and needs a power cycle.
    let tls =
        match tokio::time::timeout(Duration::from_secs(5), connector.connect(server_name, tcp))
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
    Ok((read_half, CastWriter::shared(write_half), local_ip))
}

/// Enable TCP keepalive and `TCP_USER_TIMEOUT` so a receiver that
/// lost power or Wi-Fi fails our socket within seconds. Best effort:
/// the application-level heartbeat timeout is the primary detector.
fn configure_dead_peer_detection(tcp: &TcpStream) {
    let sock = socket2::SockRef::from(tcp);
    let keepalive = socket2::TcpKeepalive::new()
        .with_time(TCP_KEEPALIVE_IDLE)
        .with_interval(TCP_KEEPALIVE_INTERVAL)
        .with_retries(TCP_KEEPALIVE_RETRIES);
    if let Err(e) = sock.set_tcp_keepalive(&keepalive) {
        warn!(%e, "could not enable TCP keepalive on the cast socket");
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    if let Err(e) = sock.set_tcp_user_timeout(Some(TCP_USER_TIMEOUT)) {
        warn!(%e, "could not set TCP_USER_TIMEOUT on the cast socket");
    }
}

/// Send a length-prefixed `CastMessage` on the shared writer.
///
/// # Errors
///
/// See [`CastWriter::send`].
pub async fn send(writer: &SharedWriter, msg: &CastMessage) -> Result<()> {
    writer.send(msg).await
}

/// Read one length-prefixed `CastMessage` from the stream, blocking
/// until a full frame is available. Returns `Ok(None)` only on a
/// clean EOF.
///
/// Cancellation-safe: every byte read is kept in `buf`, so dropping
/// the future (e.g. under `tokio::time::timeout`) and calling again
/// with the same `buf` loses nothing.
///
/// A frame whose body does not decode, or whose payload fields
/// contradict its `payload_type`, is skipped: the length prefix
/// already delimited it, so the framing stays intact.
///
/// # Errors
///
/// `Protocol` for a frame larger than [`MAX_MESSAGE_SIZE`] (framing
/// lost), `Connection` for a read error.
pub async fn recv<R: AsyncRead + Unpin>(
    reader: &mut R,
    buf: &mut BytesMut,
) -> Result<Option<CastMessage>> {
    loop {
        match CastMessage::decode_length_prefixed(buf) {
            Ok(Some(msg)) => {
                if let Err(e) = msg.validate() {
                    warn!(%e, ns = %msg.namespace, "dropping malformed cast message");
                    continue;
                }
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
            Err(e @ CastV2Error::Decode(_)) => {
                warn!(%e, "dropping undecodable cast frame");
                continue;
            }
            Err(e) => {
                return Err(FerricastError::Protocol(format!(
                    "decode cast message: {e}"
                )));
            }
        }

        // Need more bytes. Reserve room for the whole pending frame so
        // a large message is read in few syscalls; its size was already
        // bounded by `decode_length_prefixed`.
        let wanted = pending_frame_len(buf).unwrap_or(LENGTH_PREFIX_SIZE);
        buf.reserve(wanted.saturating_sub(buf.len()).max(1024));
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

/// Total size (prefix + body) of the frame at the front of `buf`, once
/// its length prefix is complete.
fn pending_frame_len(buf: &BytesMut) -> Option<usize> {
    let prefix: [u8; LENGTH_PREFIX_SIZE] = buf.get(..LENGTH_PREFIX_SIZE)?.try_into().ok()?;
    let body = usize::try_from(u32::from_be_bytes(prefix)).ok()?;
    Some(LENGTH_PREFIX_SIZE + body.min(MAX_MESSAGE_SIZE))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::castv2::{ping_message, pong_message};
    use std::pin::Pin;
    use std::task::{Context, Poll};

    fn frame(msg: &CastMessage) -> Vec<u8> {
        msg.encode_length_prefixed().unwrap()
    }

    #[tokio::test]
    async fn recv_handles_split_and_coalesced_frames() {
        let (mut tx, mut rx) = tokio::io::duplex(64);
        let ping = ping_message().unwrap();
        let pong = pong_message().unwrap();
        let mut bytes = frame(&ping);
        bytes.extend(frame(&pong));
        let writer = tokio::spawn(async move {
            // Dribble 3 bytes at a time across both frames.
            for chunk in bytes.chunks(3) {
                tx.write_all(chunk).await.unwrap();
                tokio::task::yield_now().await;
            }
        });
        let mut buf = BytesMut::new();
        assert_eq!(recv(&mut rx, &mut buf).await.unwrap(), Some(ping));
        assert_eq!(recv(&mut rx, &mut buf).await.unwrap(), Some(pong));
        writer.await.unwrap();
        assert_eq!(recv(&mut rx, &mut buf).await.unwrap(), None);
    }

    #[tokio::test]
    async fn recv_reads_frames_larger_than_initial_buffer() {
        let big = CastMessage::new_binary("a", "b", "ns", vec![7; 200_000]);
        // Bypass the 64 KiB send limit: the inbound side is tolerant.
        let body = prost::Message::encode_to_vec(&big);
        let mut bytes = u32::try_from(body.len()).unwrap().to_be_bytes().to_vec();
        bytes.extend(body);
        let (mut tx, mut rx) = tokio::io::duplex(4096);
        tokio::spawn(async move { tx.write_all(&bytes).await.unwrap() });
        let mut buf = BytesMut::with_capacity(16);
        assert_eq!(recv(&mut rx, &mut buf).await.unwrap(), Some(big));
    }

    #[tokio::test]
    async fn recv_rejects_oversized_frame() {
        let len = u32::try_from(MAX_MESSAGE_SIZE + 1).unwrap();
        let mut rx: &[u8] = &len.to_be_bytes();
        let mut buf = BytesMut::new();
        assert!(matches!(
            recv(&mut rx, &mut buf).await,
            Err(FerricastError::Protocol(_))
        ));
    }

    #[tokio::test]
    async fn recv_skips_malformed_frame_and_keeps_framing() {
        let mut bad = ping_message().unwrap();
        bad.payload_type = Some(1); // BINARY, yet carries a utf8 payload
        let mut bytes = vec![0u8, 0, 0, 2, 0xff, 0xff]; // undecodable body
        bytes.extend(frame(&bad));
        let pong = pong_message().unwrap();
        bytes.extend(frame(&pong));
        let mut rx: &[u8] = &bytes;
        let mut buf = BytesMut::new();
        assert_eq!(recv(&mut rx, &mut buf).await.unwrap(), Some(pong));
    }

    /// Sink that accepts a few bytes and then blocks forever, like a
    /// receiver that stopped draining its socket mid-frame.
    struct Stalling {
        accepted: usize,
    }

    impl AsyncWrite for Stalling {
        fn poll_write(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            if self.accepted >= 3 {
                return Poll::Pending;
            }
            let n = buf.len().min(3 - self.accepted);
            self.accepted += n;
            Poll::Ready(Ok(n))
        }
        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    #[tokio::test(start_paused = true)]
    async fn send_timeout_breaks_the_channel() {
        let writer = CastWriter::shared(Stalling { accepted: 0 });
        let ping = ping_message().unwrap();
        assert!(matches!(
            writer.send(&ping).await,
            Err(FerricastError::Timeout(_))
        ));
        assert!(writer.is_broken());
        // No further bytes may follow the partial frame.
        assert!(matches!(
            writer.send(&ping).await,
            Err(FerricastError::Connection(_))
        ));
    }

    #[tokio::test]
    async fn send_error_breaks_the_channel() {
        let (tx, rx) = tokio::io::duplex(8);
        drop(rx);
        let writer = CastWriter::shared(tx);
        assert!(writer.send(&ping_message().unwrap()).await.is_err());
        assert!(writer.is_broken());
    }
}
