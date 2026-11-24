//! Where the receiver fetches our HLS playlist, and when it is worth
//! pointing it there.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

use ferricast_core::{FerricastError, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{error, info, warn};

/// Upper bound on a playlist response read over loopback. A live
/// playlist holds a handful of segments; anything this large is broken.
const MAX_PLAYLIST_BYTES: u64 = 256 * 1024;
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const POLL_INTERVAL: Duration = Duration::from_millis(250);

/// The HLS media playlist as seen from the receiver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlaylistLocation {
    scheme: &'static str,
    addr: SocketAddr,
    path: String,
}

impl PlaylistLocation {
    /// `ip` must be the address the receiver can route to (the local
    /// end of the control socket), `path` the playlist path served by
    /// the HLS sink.
    pub(crate) fn new(scheme: &'static str, ip: IpAddr, port: u16, path: &str) -> Self {
        let path = if path.starts_with('/') {
            path.to_owned()
        } else {
            format!("/{path}")
        };
        Self {
            scheme,
            addr: SocketAddr::new(ip, port),
            path,
        }
    }

    /// URL handed to the receiver in LOAD. `SocketAddr`'s `Display`
    /// brackets IPv6 hosts (`http://[fe80::1]:8080/…`), which a bare
    /// `IpAddr` would not.
    pub(crate) fn url(&self) -> String {
        format!("{}://{}{}", self.scheme, self.addr, self.path)
    }

    pub(crate) fn scheme(&self) -> &'static str {
        self.scheme
    }

    /// Fetch the playlist from our own server over loopback. The HLS
    /// sink listens on every interface, so 127.0.0.1 reaches it.
    async fn fetch_over_loopback(&self) -> Result<String> {
        let target = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), self.addr.port());
        let request = format!(
            "GET {} HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            self.path
        );
        let probe = async move {
            let mut s = tokio::net::TcpStream::connect(target)
                .await
                .map_err(|e| FerricastError::Streaming(format!("self-test connect: {e}")))?;
            s.write_all(request.as_bytes())
                .await
                .map_err(|e| FerricastError::Streaming(format!("self-test write: {e}")))?;
            let mut raw = Vec::new();
            (&mut s)
                .take(MAX_PLAYLIST_BYTES)
                .read_to_end(&mut raw)
                .await
                .map_err(|e| FerricastError::Streaming(format!("self-test read: {e}")))?;
            parse_http_body(&raw)
        };
        tokio::time::timeout(PROBE_TIMEOUT, probe)
            .await
            .map_err(|_| FerricastError::Timeout("self-test fetch exceeded 5s".into()))?
    }
}

fn parse_http_body(raw: &[u8]) -> Result<String> {
    let text = String::from_utf8_lossy(raw);
    let (head, body) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| FerricastError::Streaming("self-test: truncated HTTP response".into()))?;
    let status = head.lines().next().unwrap_or_default();
    if status.split_whitespace().nth(1) != Some("200") {
        return Err(FerricastError::Streaming(format!(
            "self-test: playlist answered {status:?}"
        )));
    }
    Ok(body.to_owned())
}

/// Number of complete media segments listed in a playlist. LL-HLS
/// partial segments (`#EXT-X-PART`) don't count.
pub(crate) fn count_media_segments(playlist: &str) -> usize {
    playlist
        .lines()
        .filter(|l| l.trim_start().starts_with("#EXTINF:"))
        .count()
}

/// Hold LOAD until the playlist lists `min_segments` segments.
///
/// RFC 8216 §6.3.3: a client should not start a live playlist less than
/// three target durations from its end. With a single segment the
/// receiver starts at the live edge with no buffer and stalls into
/// BUFFERING almost immediately.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LoadGate {
    pub min_segments: usize,
    /// Give up waiting (and send LOAD anyway) after this long.
    pub deadline: Duration,
}

impl LoadGate {
    pub(crate) const DEFAULT_MIN_SEGMENTS: usize = 3;

    /// Gate sized for segments of `segment_secs`: room for twice the
    /// expected production time, plus slack for the first keyframe.
    pub(crate) fn new(min_segments: usize, segment_secs: f32) -> Self {
        let expected = Duration::from_secs_f32(segment_secs.max(0.5))
            .saturating_mul(u32::try_from(min_segments).unwrap_or(u32::MAX));
        Self {
            min_segments,
            deadline: expected.saturating_mul(2) + Duration::from_secs(4),
        }
    }

    /// Poll the playlist over loopback until it is deep enough.
    /// Returns the segment count seen last. Never fails: a gate that
    /// cannot observe the playlist (TLS endpoint, loopback refused)
    /// must not keep the receiver on the splash screen, so it logs and
    /// lets LOAD go out after [`Self::deadline`].
    pub(crate) async fn wait(&self, location: &PlaylistLocation) -> usize {
        if location.scheme() != "http" {
            info!(
                scheme = location.scheme(),
                "HLS self-test skipped under TLS (raw-TCP probe doesn't speak TLS)"
            );
            return 0;
        }
        let started = tokio::time::Instant::now();
        let mut reported_ok = false;
        let mut reported_err = false;
        let mut segments = 0;
        loop {
            match location.fetch_over_loopback().await {
                Ok(body) => {
                    segments = count_media_segments(&body);
                    if !reported_ok {
                        reported_ok = true;
                        let head: String = body.chars().take(256).collect();
                        info!(bytes = body.len(), body_head = %head, "HLS self-test OK");
                    }
                    if segments >= self.min_segments {
                        return segments;
                    }
                }
                Err(e) if !reported_err => {
                    reported_err = true;
                    error!(
                        error = %e,
                        "HLS self-test FAILED — chromecast almost certainly can't reach the URL either"
                    );
                }
                Err(_) => {}
            }
            if started.elapsed() >= self.deadline {
                warn!(
                    segments,
                    wanted = self.min_segments,
                    "playlist still shallow at the deadline; sending LOAD anyway"
                );
                return segments;
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as _;
    use std::net::Ipv6Addr;

    #[test]
    fn ipv4_url() {
        let loc =
            PlaylistLocation::new("http", "192.168.1.20".parse().unwrap(), 8080, "/index.m3u8");
        assert_eq!(loc.url(), "http://192.168.1.20:8080/index.m3u8");
    }

    #[test]
    fn ipv6_url_is_bracketed() {
        let ip = IpAddr::V6("fe80::1".parse::<Ipv6Addr>().unwrap());
        let loc = PlaylistLocation::new("http", ip, 8080, "tok3n/index.m3u8");
        assert_eq!(loc.url(), "http://[fe80::1]:8080/tok3n/index.m3u8");
    }

    #[test]
    fn counts_full_segments_only() {
        let playlist = "#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXT-X-PART:DURATION=0.5,URI=\"p0.ts\"\n\
                        #EXTINF:2.000,\nseg0.ts\n#EXTINF:2.000,\nseg1.ts\n";
        assert_eq!(count_media_segments(playlist), 2);
        assert_eq!(count_media_segments(""), 0);
    }

    #[test]
    fn http_body_requires_200() {
        assert_eq!(
            parse_http_body(b"HTTP/1.1 200 OK\r\nX: y\r\n\r\n#EXTM3U\n").unwrap(),
            "#EXTM3U\n"
        );
        assert!(parse_http_body(b"HTTP/1.1 403 Forbidden\r\n\r\n").is_err());
        assert!(parse_http_body(b"HTTP/1.1 200 OK\r\n").is_err());
    }

    #[test]
    fn gate_deadline_scales_with_segments() {
        let gate = LoadGate::new(3, 2.0);
        assert_eq!(gate.min_segments, 3);
        assert_eq!(gate.deadline, Duration::from_secs(16));
    }

    #[tokio::test]
    async fn gate_waits_for_enough_segments() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            for n in 1..=5 {
                let (mut s, _) = listener.accept().await.unwrap();
                let mut req = [0u8; 512];
                let _ = s.read(&mut req).await.unwrap();
                let mut body = String::from("#EXTM3U\n");
                for i in 0..n {
                    write!(body, "#EXTINF:2.0,\nseg{i}.ts\n").unwrap();
                }
                let resp = format!(
                    "HTTP/1.0 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                );
                s.write_all(resp.as_bytes()).await.unwrap();
            }
        });
        let loc =
            PlaylistLocation::new("http", IpAddr::V4(Ipv4Addr::LOCALHOST), port, "/index.m3u8");
        let gate = LoadGate {
            min_segments: 3,
            deadline: Duration::from_secs(10),
        };
        assert_eq!(gate.wait(&loc).await, 3);
    }

    #[tokio::test]
    async fn gate_gives_up_at_deadline() {
        // Nothing listens on this port: every probe fails.
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let loc =
            PlaylistLocation::new("http", IpAddr::V4(Ipv4Addr::LOCALHOST), port, "/index.m3u8");
        let gate = LoadGate {
            min_segments: 3,
            deadline: Duration::from_millis(300),
        };
        assert_eq!(gate.wait(&loc).await, 0);
    }
}
