//! Minimal RTSP/1.0 message layer with the Wi-Fi Display extensions.
//!
//! This module only knows the wire shape — the M1–M7 state machine is
//! layered on top in [`session`](crate::session) (source) and
//! [`receiver`](crate::receiver) (sink). Source and sink each keep
//! independent monotonic `CSeq` counters; sharing one is the #2 cause
//! of interop breakage after missing `Require: org.wfa.wfd1.0`.
//!
//! Bodies carrying `wfd_*` parameters are `Content-Type: text/parameters`
//! with CRLF line endings. Decoding tolerates LF-only for defensive
//! interop; encoding always writes CRLF.

use std::collections::BTreeMap;
use std::fmt;

use bytes::{Bytes, BytesMut};
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

/// The `Require` tag every WFD 1.0 message carries.
pub const WFD_REQUIRE_TAG: &str = "org.wfa.wfd1.0";
/// `Content-Type` header value for WFD parameter bodies.
pub const PARAM_CONTENT_TYPE: &str = "text/parameters";
/// Default Session timeout in seconds. Clients send a keepalive at ~30 s.
pub const SESSION_TIMEOUT_SECS: u32 = 60;

#[derive(Debug, Error)]
pub enum RtspError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("malformed RTSP message: {0}")]
    Malformed(&'static str),
    #[error("expected response, got request")]
    UnexpectedRequest,
    #[error("expected request, got response")]
    UnexpectedResponse,
    #[error("connection closed mid-message")]
    Closed,
    #[error("bad status line: `{0}`")]
    BadStatusLine(String),
    #[error("bad request line: `{0}`")]
    BadRequestLine(String),
    #[error("unsupported RTSP version: `{0}`")]
    BadVersion(String),
}

pub type Result<T> = std::result::Result<T, RtspError>;

/// RTSP methods we care about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Options,
    GetParameter,
    SetParameter,
    Setup,
    Play,
    Pause,
    Teardown,
    /// Any other method the peer sends. We respond `501 Not Implemented`.
    Other,
}

impl Method {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Options => "OPTIONS",
            Self::GetParameter => "GET_PARAMETER",
            Self::SetParameter => "SET_PARAMETER",
            Self::Setup => "SETUP",
            Self::Play => "PLAY",
            Self::Pause => "PAUSE",
            Self::Teardown => "TEARDOWN",
            Self::Other => "OTHER",
        }
    }

    fn parse(s: &str) -> Self {
        match s {
            "OPTIONS" => Self::Options,
            "GET_PARAMETER" => Self::GetParameter,
            "SET_PARAMETER" => Self::SetParameter,
            "SETUP" => Self::Setup,
            "PLAY" => Self::Play,
            "PAUSE" => Self::Pause,
            "TEARDOWN" => Self::Teardown,
            _ => Self::Other,
        }
    }
}

/// Case-insensitive header map that preserves the canonical casing it
/// was given. Many sinks pattern-match on exact casing (`CSeq`,
/// `Content-Length`) even though RFC 2326 says case-insensitive; keep
/// the caller's spelling on the wire, lower-case only for lookup.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Headers {
    /// `(lowercased key → (original_name, value))`.
    entries: BTreeMap<String, (String, String)>,
}

impl Headers {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, value: impl Into<String>) {
        let name = canonicalize_header_name(&name.into());
        let key = name.to_ascii_lowercase();
        self.entries.insert(key, (name, value.into()));
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<&str> {
        self.entries
            .get(&name.to_ascii_lowercase())
            .map(|(_, v)| v.as_str())
    }

    #[must_use]
    pub fn cseq(&self) -> Option<u32> {
        self.get("cseq").and_then(|v| v.trim().parse().ok())
    }

    #[must_use]
    pub fn session(&self) -> Option<&str> {
        self.get("session")
            .map(|v| v.split(';').next().unwrap_or(v).trim())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.values().map(|(n, v)| (n.as_str(), v.as_str()))
    }
}

/// Convert `content-length` → `Content-Length`, `cseq` → `CSeq`, etc.
/// Preserves user capitalisation when it already looks canonical.
fn canonicalize_header_name(raw: &str) -> String {
    let lower = raw.to_ascii_lowercase();
    match lower.as_str() {
        "cseq" => "CSeq".to_string(),
        _ => {
            let mut out = String::with_capacity(raw.len());
            let mut capitalize = true;
            for c in lower.chars() {
                if c == '-' {
                    out.push('-');
                    capitalize = true;
                } else if capitalize {
                    out.push(c.to_ascii_uppercase());
                    capitalize = false;
                } else {
                    out.push(c);
                }
            }
            out
        }
    }
}

#[derive(Debug, Clone)]
pub struct Request {
    pub method: Method,
    pub raw_method: String,
    pub uri: String,
    pub headers: Headers,
    pub body: Bytes,
}

impl Request {
    #[must_use]
    pub fn new(method: Method, uri: impl Into<String>) -> Self {
        Self {
            method,
            raw_method: method.as_str().to_string(),
            uri: uri.into(),
            headers: Headers::new(),
            body: Bytes::new(),
        }
    }

    pub fn with_header(mut self, name: &str, value: impl Into<String>) -> Self {
        self.headers.insert(name, value);
        self
    }

    pub fn with_body_params(mut self, body: impl Into<String>) -> Self {
        let bytes = body.into().into_bytes();
        self.headers.insert("content-type", PARAM_CONTENT_TYPE);
        self.headers.insert("content-length", bytes.len().to_string());
        self.body = Bytes::from(bytes);
        self
    }
}

impl fmt::Display for Request {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} RTSP/1.0\r\n", self.raw_method, self.uri)?;
        for (name, value) in self.headers.iter() {
            write!(f, "{name}: {value}\r\n")?;
        }
        f.write_str("\r\n")?;
        if !self.body.is_empty() {
            f.write_str(std::str::from_utf8(&self.body).unwrap_or(""))?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Response {
    pub status: u16,
    pub reason: String,
    pub headers: Headers,
    pub body: Bytes,
}

impl Response {
    #[must_use]
    pub fn new(status: u16, reason: impl Into<String>) -> Self {
        Self {
            status,
            reason: reason.into(),
            headers: Headers::new(),
            body: Bytes::new(),
        }
    }

    #[must_use]
    pub fn ok() -> Self {
        Self::new(200, "OK")
    }

    #[must_use]
    pub fn not_implemented() -> Self {
        Self::new(501, "Not Implemented")
    }

    pub fn with_header(mut self, name: &str, value: impl Into<String>) -> Self {
        self.headers.insert(name, value);
        self
    }

    pub fn with_body_params(mut self, body: impl Into<String>) -> Self {
        let bytes = body.into().into_bytes();
        self.headers.insert("content-type", PARAM_CONTENT_TYPE);
        self.headers.insert("content-length", bytes.len().to_string());
        self.body = Bytes::from(bytes);
        self
    }
}

impl fmt::Display for Response {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RTSP/1.0 {} {}\r\n", self.status, self.reason)?;
        for (name, value) in self.headers.iter() {
            write!(f, "{name}: {value}\r\n")?;
        }
        f.write_str("\r\n")?;
        if !self.body.is_empty() {
            f.write_str(std::str::from_utf8(&self.body).unwrap_or(""))?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    Request(Request),
    Response(Response),
}

/// Framed RTSP peer. Each direction (source and sink) owns its own
/// `CSeq` counter; the frame struct tracks the one *we* allocate.
pub struct RtspFramed {
    reader: BufReader<tokio::io::ReadHalf<TcpStream>>,
    writer: tokio::io::WriteHalf<TcpStream>,
    outgoing_cseq: u32,
}

impl RtspFramed {
    #[must_use]
    pub fn new(stream: TcpStream) -> Self {
        let (r, w) = tokio::io::split(stream);
        Self {
            reader: BufReader::new(r),
            writer: w,
            outgoing_cseq: 0,
        }
    }

    /// Next CSeq to allocate for a message we initiate.
    pub fn next_cseq(&mut self) -> u32 {
        self.outgoing_cseq = self.outgoing_cseq.saturating_add(1);
        self.outgoing_cseq
    }

    pub async fn send_request(&mut self, mut req: Request) -> Result<u32> {
        let cseq = self.next_cseq();
        req.headers.insert("cseq", cseq.to_string());
        self.writer.write_all(req.to_string().as_bytes()).await?;
        self.writer.flush().await?;
        Ok(cseq)
    }

    pub async fn send_response(&mut self, resp: Response) -> Result<()> {
        self.writer.write_all(resp.to_string().as_bytes()).await?;
        self.writer.flush().await?;
        Ok(())
    }

    pub async fn read_message(&mut self) -> Result<Message> {
        let mut head: Vec<u8> = Vec::new();
        loop {
            let read = self.reader.read_until(b'\n', &mut head).await?;
            if read == 0 {
                return Err(RtspError::Closed);
            }
            if head.len() >= 4 && head.ends_with(b"\r\n\r\n") {
                break;
            }
            if head.len() >= 2 && head.ends_with(b"\n\n") {
                break;
            }
            if head.len() > 64 * 1024 {
                return Err(RtspError::Malformed("header block > 64 KiB"));
            }
        }
        let head_str =
            std::str::from_utf8(&head).map_err(|_| RtspError::Malformed("non-UTF8 header"))?;
        let mut lines = head_str.split('\n');
        let start_line = lines
            .next()
            .ok_or(RtspError::Malformed("empty message"))?
            .trim_end_matches('\r')
            .trim();
        let mut headers = Headers::new();
        for line in lines {
            let trimmed = line.trim_end_matches('\r').trim();
            if trimmed.is_empty() {
                continue;
            }
            let (name, value) = trimmed
                .split_once(':')
                .ok_or(RtspError::Malformed("header without colon"))?;
            headers.insert(name.trim(), value.trim());
        }
        let content_length: usize = headers
            .get("content-length")
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0);
        let mut body = BytesMut::with_capacity(content_length);
        body.resize(content_length, 0);
        if content_length > 0 {
            self.reader.read_exact(&mut body).await?;
        }
        let body = body.freeze();

        if start_line.starts_with("RTSP/") {
            let mut parts = start_line.splitn(3, ' ');
            let version = parts
                .next()
                .ok_or_else(|| RtspError::BadStatusLine(start_line.to_string()))?;
            if version != "RTSP/1.0" {
                return Err(RtspError::BadVersion(version.to_string()));
            }
            let status: u16 = parts
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| RtspError::BadStatusLine(start_line.to_string()))?;
            let reason = parts.next().unwrap_or("").to_string();
            Ok(Message::Response(Response {
                status,
                reason,
                headers,
                body,
            }))
        } else {
            let mut parts = start_line.splitn(3, ' ');
            let method_tok = parts
                .next()
                .ok_or_else(|| RtspError::BadRequestLine(start_line.to_string()))?;
            let uri = parts
                .next()
                .ok_or_else(|| RtspError::BadRequestLine(start_line.to_string()))?
                .to_string();
            let version = parts
                .next()
                .ok_or_else(|| RtspError::BadRequestLine(start_line.to_string()))?;
            if version != "RTSP/1.0" {
                return Err(RtspError::BadVersion(version.to_string()));
            }
            Ok(Message::Request(Request {
                method: Method::parse(method_tok),
                raw_method: method_tok.to_string(),
                uri,
                headers,
                body,
            }))
        }
    }
}

/// Convenience to build M1-style OPTIONS.
#[must_use]
pub fn build_options(user_agent: &str) -> Request {
    Request::new(Method::Options, "*")
        .with_header("require", WFD_REQUIRE_TAG)
        .with_header("user-agent", user_agent)
}

/// Convenience to build an OPTIONS 200 response advertising the public
/// methods we accept.
#[must_use]
pub fn options_response(public_methods: &[&str]) -> Response {
    Response::ok().with_header("public", public_methods.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_renders_with_crlf() {
        let req = Request::new(Method::Options, "*")
            .with_header("cseq", "1")
            .with_header("require", WFD_REQUIRE_TAG)
            .with_header("user-agent", "FerriCast/0.1");
        let rendered = req.to_string();
        assert!(rendered.starts_with("OPTIONS * RTSP/1.0\r\n"));
        assert!(rendered.ends_with("\r\n\r\n"));
        assert!(rendered.contains("Require"));
        assert!(rendered.contains("org.wfa.wfd1.0"));
    }

    #[test]
    fn response_renders_with_body() {
        let resp = Response::ok()
            .with_header("cseq", "2")
            .with_body_params("wfd_video_formats: 00 00 01 01 00000020 00000000 00000000 00 0000 0000 00 none none\r\n");
        let rendered = resp.to_string();
        assert!(rendered.contains("Content-Length"));
        assert!(rendered.contains("wfd_video_formats"));
    }

    #[test]
    fn header_cseq_parses() {
        let mut h = Headers::new();
        h.insert("CSeq", "42");
        assert_eq!(h.cseq(), Some(42));
    }

    #[test]
    fn header_session_strips_timeout() {
        let mut h = Headers::new();
        h.insert("Session", "0F93F363;timeout=60");
        assert_eq!(h.session(), Some("0F93F363"));
    }
}
