//! Server half of the RTSP control connection: reads requests, writes
//! responses. Legacy pairing leaves the connection in clear text, so no
//! framing other than RTSP/1.0 is needed.

use std::fmt::Write as _;

use ferricast_core::{FerricastError, Result};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Upper bound for a request body (pairing messages, plists, cover art).
const MAX_BODY_LEN: usize = 4 * 1024 * 1024;
/// Upper bound for the request line + headers.
const MAX_HEAD_LEN: usize = 64 * 1024;
/// `Server` header every response carries, as AirPlay receivers do.
const SERVER: &str = "AirTunes/220.68";

/// One parsed request.
#[derive(Debug, Clone)]
pub(super) struct Request {
    pub method: String,
    pub uri: String,
    headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    /// Case-insensitive header lookup.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// The path of the URI, without scheme, host or query.
    pub fn path(&self) -> &str {
        let uri = self.uri.as_str();
        let path = match uri.find("://") {
            Some(scheme_end) => {
                let rest = &uri[scheme_end + 3..];
                rest.find('/').map_or("/", |p| &rest[p..])
            }
            None => uri,
        };
        path.split('?').next().unwrap_or(path)
    }

    fn parse_head(head: &str) -> Result<(Self, usize)> {
        let mut lines = head.split("\r\n");
        let request_line = lines
            .next()
            .ok_or_else(|| FerricastError::Rtsp("empty RTSP request".into()))?;
        let mut parts = request_line.split(' ');
        let (Some(method), Some(uri), Some(version)) = (parts.next(), parts.next(), parts.next())
        else {
            return Err(FerricastError::Rtsp(format!(
                "malformed request line {request_line:?}"
            )));
        };
        if !matches!(version, "RTSP/1.0" | "HTTP/1.1" | "HTTP/1.0") {
            return Err(FerricastError::Rtsp(format!(
                "unexpected request protocol {version:?}"
            )));
        }
        let mut headers = Vec::new();
        for line in lines.filter(|l| !l.is_empty()) {
            let (name, value) = line
                .split_once(':')
                .ok_or_else(|| FerricastError::Rtsp(format!("malformed header {line:?}")))?;
            headers.push((name.trim().to_owned(), value.trim().to_owned()));
        }
        let request = Self {
            method: method.to_owned(),
            uri: uri.to_owned(),
            headers,
            body: Vec::new(),
        };
        let len = match request.header("Content-Length") {
            Some(v) => v
                .parse::<usize>()
                .map_err(|e| FerricastError::Rtsp(format!("invalid Content-Length {v:?}: {e}")))?,
            None => 0,
        };
        if len > MAX_BODY_LEN {
            return Err(FerricastError::Rtsp(format!(
                "request body of {len} bytes exceeds {MAX_BODY_LEN}"
            )));
        }
        Ok((request, len))
    }
}

/// A response under construction.
#[derive(Debug, Clone)]
pub(super) struct Response {
    status: u16,
    reason: &'static str,
    headers: Vec<(&'static str, String)>,
    body: Vec<u8>,
    /// Close the connection once this response is written.
    pub close: bool,
}

impl Response {
    pub fn ok() -> Self {
        Self::status(200, "OK")
    }

    pub fn status(status: u16, reason: &'static str) -> Self {
        Self {
            status,
            reason,
            headers: Vec::new(),
            body: Vec::new(),
            close: false,
        }
    }

    pub fn header(mut self, name: &'static str, value: impl Into<String>) -> Self {
        self.headers.push((name, value.into()));
        self
    }

    pub fn body(mut self, content_type: &'static str, body: Vec<u8>) -> Self {
        self.headers.push(("Content-Type", content_type.to_owned()));
        self.body = body;
        self
    }

    pub fn closing(mut self) -> Self {
        self.close = true;
        self
    }

    pub fn status_code(&self) -> u16 {
        self.status
    }

    /// Wire form, echoing the request's `CSeq`.
    fn encode(&self, cseq: Option<&str>, protocol: &str) -> Vec<u8> {
        let mut head = format!("{protocol} {} {}\r\n", self.status, self.reason);
        // `write!` into a String cannot fail.
        if let Some(cseq) = cseq {
            let _ = write!(head, "CSeq: {cseq}\r\n");
        }
        let _ = write!(head, "Server: {SERVER}\r\n");
        for (name, value) in &self.headers {
            let _ = write!(head, "{name}: {value}\r\n");
        }
        if self.close {
            head.push_str("Connection: close\r\n");
        }
        let _ = write!(head, "Content-Length: {}\r\n\r\n", self.body.len());
        let mut wire = head.into_bytes();
        wire.extend_from_slice(&self.body);
        wire
    }
}

/// Reads requests from and writes responses to one connection.
pub(super) struct RtspServerConn<S> {
    stream: S,
    rx: Vec<u8>,
}

impl<S: AsyncRead + AsyncWrite + Unpin> RtspServerConn<S> {
    pub fn new(stream: S) -> Self {
        Self {
            stream,
            rx: Vec::new(),
        }
    }

    /// The next complete request; `None` when the peer closed the
    /// connection between requests.
    ///
    /// Cancel safe: bytes already read stay buffered for the next call.
    pub async fn read_request(&mut self) -> Result<Option<Request>> {
        loop {
            if let Some(head_end) = self.rx.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = std::str::from_utf8(&self.rx[..head_end])
                    .map_err(|_| FerricastError::Rtsp("request head is not UTF-8".into()))?;
                let (mut request, body_len) = Request::parse_head(head)?;
                let body_start = head_end + 4;
                if self.rx.len() >= body_start + body_len {
                    request.body = self.rx[body_start..body_start + body_len].to_vec();
                    self.rx.drain(..body_start + body_len);
                    return Ok(Some(request));
                }
            } else if self.rx.len() > MAX_HEAD_LEN {
                return Err(FerricastError::Rtsp("request head too large".into()));
            }
            let mut buf = [0_u8; 8192];
            let n = self.stream.read(&mut buf).await?;
            if n == 0 {
                return if self.rx.is_empty() {
                    Ok(None)
                } else {
                    Err(FerricastError::Rtsp(
                        "connection closed in the middle of a request".into(),
                    ))
                };
            }
            self.rx.extend_from_slice(&buf[..n]);
        }
    }

    pub async fn write_response(&mut self, request: &Request, response: &Response) -> Result<()> {
        let protocol = if request.uri.starts_with("rtsp") || request.header("CSeq").is_some() {
            "RTSP/1.0"
        } else {
            "HTTP/1.1"
        };
        let wire = response.encode(request.header("CSeq"), protocol);
        self.stream.write_all(&wire).await?;
        self.stream.flush().await?;
        Ok(())
    }

    pub async fn shutdown(&mut self) {
        let _ = self.stream.shutdown().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn reads_pipelined_requests_and_echoes_cseq() {
        let (client, server) = tokio::io::duplex(4096);
        let mut conn = RtspServerConn::new(server);
        let (mut rd, mut wr) = tokio::io::split(client);
        wr.write_all(
            b"POST /pair-setup RTSP/1.0\r\nCSeq: 1\r\nContent-Length: 3\r\n\r\nabcGET /info RTSP/1.0\r\nCSeq: 2\r\n\r\n",
        )
        .await
        .unwrap();
        let first = conn.read_request().await.unwrap().unwrap();
        assert_eq!(
            (first.method.as_str(), first.path(), first.body.as_slice()),
            ("POST", "/pair-setup", &b"abc"[..])
        );
        let second = conn.read_request().await.unwrap().unwrap();
        assert_eq!(second.path(), "/info");
        conn.write_response(&second, &Response::ok().body("text/plain", b"hi".to_vec()))
            .await
            .unwrap();
        let mut buf = vec![0_u8; 256];
        let n = rd.read(&mut buf).await.unwrap();
        let text = std::str::from_utf8(&buf[..n]).unwrap();
        assert!(text.starts_with("RTSP/1.0 200 OK\r\nCSeq: 2\r\n"), "{text}");
        assert!(text.ends_with("Content-Length: 2\r\n\r\nhi"), "{text}");
        wr.shutdown().await.unwrap();
        assert!(conn.read_request().await.unwrap().is_none());
    }

    #[test]
    fn path_strips_scheme_host_and_query() {
        let (r, _) = Request::parse_head("SETUP rtsp://10.0.0.2/1234?x=1 RTSP/1.0").unwrap();
        assert_eq!(r.path(), "/1234");
        let (r, _) = Request::parse_head("GET /info?txtAirPlay HTTP/1.1").unwrap();
        assert_eq!(r.path(), "/info");
    }
}
