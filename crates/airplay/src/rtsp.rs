//! Minimal RTSP/1.0 client for the AirPlay control connection.
//!
//! Every request carries a `CSeq` and the sender's identity headers. After a
//! HomeKit pair-verify the connection can be switched to the encrypted
//! framing with [`RtspClient::enable_encryption`].

use std::fmt::Write as _;
use std::net::SocketAddr;
use std::time::Duration;

use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use ferricast_core::{FerricastError, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;

/// Upper bound for a response body. AirPlay control responses are small
/// plists / TLVs; anything larger is a broken or hostile peer.
const MAX_BODY_LEN: usize = 1024 * 1024;
/// Upper bound for the status line + headers.
const MAX_HEAD_LEN: usize = 64 * 1024;
/// Plaintext bytes per encrypted control frame (HAP limit).
const MAX_FRAME_LEN: usize = 1024;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

pub(crate) const CONTENT_OCTET: &str = "application/octet-stream";
pub(crate) const CONTENT_BPLIST: &str = "application/x-apple-binary-plist";
const USER_AGENT: &str = "AirPlay/550.10";

/// Identity headers sent with every request.
#[derive(Debug, Clone)]
pub(crate) struct ClientIdentity {
    /// 64-bit DACP id rendered as 16 upper-case hex digits.
    pub dacp_id: String,
    /// Random `Active-Remote` token.
    pub active_remote: u32,
}

impl Default for ClientIdentity {
    fn default() -> Self {
        Self {
            dacp_id: format!("{:016X}", rand::random::<u64>()),
            active_remote: rand::random(),
        }
    }
}

/// A parsed RTSP response.
#[derive(Debug, Clone)]
pub(crate) struct RtspResponse {
    pub status: u16,
    pub reason: String,
    headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl RtspResponse {
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// Case-insensitive header lookup.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Fails with the status line when the response is not 2xx.
    pub fn ensure_success(self, what: &str) -> Result<Self> {
        if self.is_success() {
            Ok(self)
        } else {
            Err(FerricastError::Rtsp(format!(
                "{what} failed: {} {}",
                self.status, self.reason
            )))
        }
    }

    /// Parse a complete head (status line + headers, without the blank
    /// line). Returns the response and its announced body length.
    fn parse_head(head: &str) -> Result<(Self, usize)> {
        let mut lines = head.split("\r\n");
        let status_line = lines
            .next()
            .ok_or_else(|| FerricastError::Rtsp("empty RTSP response".into()))?;
        let mut parts = status_line.splitn(3, ' ');
        let version = parts.next().unwrap_or_default();
        if version != "RTSP/1.0" && version != "HTTP/1.1" {
            return Err(FerricastError::Rtsp(format!(
                "unexpected response protocol {version:?}"
            )));
        }
        let status = parts
            .next()
            .unwrap_or_default()
            .parse::<u16>()
            .map_err(|e| FerricastError::Rtsp(format!("invalid status code: {e}")))?;
        let reason = parts.next().unwrap_or_default().to_owned();
        let mut headers = Vec::new();
        for line in lines.filter(|l| !l.is_empty()) {
            let (name, value) = line
                .split_once(':')
                .ok_or_else(|| FerricastError::Rtsp(format!("malformed header {line:?}")))?;
            headers.push((name.trim().to_owned(), value.trim().to_owned()));
        }
        let response = Self {
            status,
            reason,
            headers,
            body: Vec::new(),
        };
        let content_length = match response.header("Content-Length") {
            Some(v) => v
                .parse::<usize>()
                .map_err(|e| FerricastError::Rtsp(format!("invalid Content-Length {v:?}: {e}")))?,
            None => 0,
        };
        if content_length > MAX_BODY_LEN {
            return Err(FerricastError::Rtsp(format!(
                "response body of {content_length} bytes exceeds {MAX_BODY_LEN}"
            )));
        }
        Ok((response, content_length))
    }
}

/// ChaCha20-Poly1305 framing used on the control connection after HomeKit
/// pair-verify: `len: u16 LE ‖ ciphertext ‖ tag`, the length as AAD and a
/// per-direction 64-bit little-endian counter as nonce.
struct SecureChannel {
    write: ChaCha20Poly1305,
    read: ChaCha20Poly1305,
    write_counter: u64,
    read_counter: u64,
}

fn counter_nonce(counter: u64) -> Nonce {
    let mut nonce = [0_u8; 12];
    nonce[4..].copy_from_slice(&counter.to_le_bytes());
    nonce.into()
}

impl SecureChannel {
    fn new(write_key: &[u8; 32], read_key: &[u8; 32]) -> Self {
        Self {
            write: ChaCha20Poly1305::new(Key::from_slice(write_key)),
            read: ChaCha20Poly1305::new(Key::from_slice(read_key)),
            write_counter: 0,
            read_counter: 0,
        }
    }

    fn seal(&mut self, plaintext: &[u8]) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(plaintext.len() + 32);
        for chunk in plaintext.chunks(MAX_FRAME_LEN) {
            let len = u16::try_from(chunk.len())
                .map_err(|_| FerricastError::Rtsp("control frame too long".into()))?
                .to_le_bytes();
            let sealed = self
                .write
                .encrypt(
                    &counter_nonce(self.write_counter),
                    Payload {
                        msg: chunk,
                        aad: &len,
                    },
                )
                .map_err(|_| FerricastError::Rtsp("control channel encryption failed".into()))?;
            self.write_counter += 1;
            out.extend_from_slice(&len);
            out.extend_from_slice(&sealed);
        }
        Ok(out)
    }

    fn open(&mut self, len: [u8; 2], sealed: &[u8]) -> Result<Vec<u8>> {
        let plain = self
            .read
            .decrypt(
                &counter_nonce(self.read_counter),
                Payload {
                    msg: sealed,
                    aad: &len,
                },
            )
            .map_err(|_| FerricastError::Rtsp("control channel authentication failed".into()))?;
        self.read_counter += 1;
        Ok(plain)
    }
}

/// RTSP client bound to one TCP connection.
pub(crate) struct RtspClient {
    stream: TcpStream,
    /// Bytes received (and decrypted) but not yet consumed.
    rx: Vec<u8>,
    cseq: u32,
    identity: ClientIdentity,
    hkp: Option<&'static str>,
    secure: Option<SecureChannel>,
    local_addr: SocketAddr,
    peer_addr: SocketAddr,
}

impl RtspClient {
    /// Connect to the first reachable address, in order.
    pub async fn connect(
        addrs: &[SocketAddr],
        identity: ClientIdentity,
        hkp: Option<&'static str>,
    ) -> Result<Self> {
        let (stream, peer_addr) = ferricast_core::connect_tcp_any(addrs, REQUEST_TIMEOUT)
            .await
            .map_err(|e| {
                FerricastError::Connection(format!("cannot connect to AirPlay receiver: {e}"))
            })?;
        // Small request/response exchanges; don't let Nagle add
        // latency. Failure only costs latency.
        let _ = stream.set_nodelay(true);
        let local_addr = stream.local_addr()?;
        Ok(Self {
            stream,
            rx: Vec::new(),
            cseq: 0,
            identity,
            hkp,
            secure: None,
            local_addr,
            peer_addr,
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub fn peer_addr(&self) -> SocketAddr {
        self.peer_addr
    }

    /// The receiver's address on another port, keeping the scope id of
    /// a link-local IPv6 peer.
    pub fn peer_at(&self, port: u16) -> SocketAddr {
        let mut addr = self.peer_addr;
        addr.set_port(port);
        addr
    }

    /// Switch to the encrypted control framing. Keys come from the HomeKit
    /// pair-verify shared secret.
    pub fn enable_encryption(&mut self, write_key: &[u8; 32], read_key: &[u8; 32]) {
        self.secure = Some(SecureChannel::new(write_key, read_key));
    }

    /// Send one request and wait for its response.
    pub async fn request(
        &mut self,
        method: &str,
        uri: &str,
        content_type: Option<&str>,
        body: &[u8],
    ) -> Result<RtspResponse> {
        self.request_with(method, uri, content_type, body, &[])
            .await
    }

    /// Like [`Self::request`] with extra headers.
    pub async fn request_with(
        &mut self,
        method: &str,
        uri: &str,
        content_type: Option<&str>,
        body: &[u8],
        extra: &[(&str, &str)],
    ) -> Result<RtspResponse> {
        self.cseq += 1;
        let mut head = format!("{method} {uri} RTSP/1.0\r\nCSeq: {}\r\n", self.cseq);
        // `write!` into a String cannot fail.
        let _ = write!(
            head,
            "User-Agent: {USER_AGENT}\r\nDACP-ID: {id}\r\nActive-Remote: {ar}\r\nClient-Instance: {id}\r\n",
            id = self.identity.dacp_id,
            ar = self.identity.active_remote,
        );
        if let Some(hkp) = self.hkp {
            let _ = write!(head, "X-Apple-HKP: {hkp}\r\n");
        }
        for (name, value) in extra {
            let _ = write!(head, "{name}: {value}\r\n");
        }
        if let Some(ct) = content_type {
            let _ = write!(head, "Content-Type: {ct}\r\n");
        }
        let _ = write!(head, "Content-Length: {}\r\n\r\n", body.len());

        let mut wire = head.into_bytes();
        wire.extend_from_slice(body);
        if let Some(secure) = self.secure.as_mut() {
            wire = secure.seal(&wire)?;
        }

        timeout(REQUEST_TIMEOUT, async {
            self.stream.write_all(&wire).await?;
            self.read_response().await
        })
        .await
        .map_err(|_| FerricastError::Timeout(format!("RTSP {method} {uri}")))?
    }

    /// Read more bytes from the socket into `rx`, decrypting if needed.
    async fn fill(&mut self) -> Result<()> {
        if let Some(secure) = self.secure.as_mut() {
            let mut len = [0_u8; 2];
            self.stream.read_exact(&mut len).await?;
            let mut sealed = vec![0_u8; usize::from(u16::from_le_bytes(len)) + 16];
            self.stream.read_exact(&mut sealed).await?;
            let plain = secure.open(len, &sealed)?;
            self.rx.extend_from_slice(&plain);
        } else {
            let mut buf = [0_u8; 4096];
            let n = self.stream.read(&mut buf).await?;
            if n == 0 {
                return Err(FerricastError::Connection(
                    "AirPlay receiver closed the control connection".into(),
                ));
            }
            self.rx.extend_from_slice(&buf[..n]);
        }
        Ok(())
    }

    async fn read_response(&mut self) -> Result<RtspResponse> {
        let head_end = loop {
            if let Some(pos) = self.rx.windows(4).position(|w| w == b"\r\n\r\n") {
                break pos;
            }
            if self.rx.len() > MAX_HEAD_LEN {
                return Err(FerricastError::Rtsp("RTSP response head too large".into()));
            }
            self.fill().await?;
        };
        let head = std::str::from_utf8(&self.rx[..head_end])
            .map_err(|_| FerricastError::Rtsp("RTSP response head is not UTF-8".into()))?;
        let (mut response, body_len) = RtspResponse::parse_head(head)?;
        let body_start = head_end + 4;
        while self.rx.len() < body_start + body_len {
            self.fill().await?;
        }
        response.body = self.rx[body_start..body_start + body_len].to_vec();
        self.rx.drain(..body_start + body_len);
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_head_and_caps_body() {
        let (r, len) = RtspResponse::parse_head(
            "RTSP/1.0 200 OK\r\nCSeq: 3\r\ncontent-length: 32\r\nServer: AirTunes/220.68",
        )
        .unwrap();
        assert_eq!(r.status, 200);
        assert_eq!(len, 32);
        assert_eq!(r.header("Server"), Some("AirTunes/220.68"));
        assert!(RtspResponse::parse_head("RTSP/1.0 200 OK\r\nContent-Length: 999999999").is_err());
        assert!(RtspResponse::parse_head("SIP/2.0 200 OK").is_err());
    }

    #[test]
    fn secure_channel_round_trip() {
        let mut a = SecureChannel::new(&[1; 32], &[2; 32]);
        let mut b = SecureChannel::new(&[2; 32], &[1; 32]);
        let msg = vec![7_u8; 2500];
        let sealed = a.seal(&msg).unwrap();
        let mut plain = Vec::new();
        let mut rest = sealed.as_slice();
        while !rest.is_empty() {
            let len = [rest[0], rest[1]];
            let n = usize::from(u16::from_le_bytes(len)) + 16;
            plain.extend(b.open(len, &rest[2..2 + n]).unwrap());
            rest = &rest[2 + n..];
        }
        assert_eq!(plain, msg);
        // A replayed frame must not authenticate.
        let len = [sealed[0], sealed[1]];
        assert!(b.open(len, &sealed[2..2 + 1024 + 16]).is_err());
    }
}
