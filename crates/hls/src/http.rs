//! Minimal HTTP/1.x server for the HLS endpoints.
//!
//! Routes:
//! * `GET /`, `GET /playlist.m3u8`, `GET /index.m3u8` → media playlist
//! * `GET /segment-{seq}.ts` → MPEG-TS segment from the ring
//!
//! Also handles `HEAD` and `OPTIONS` (the latter for CORS preflight
//! from browser players). Connections are closed after a single
//! request — no keep-alive — which keeps the per-connection state
//! trivially small.
//!
//! No third-party HTTP parser; we only need the request line and we
//! drain the rest of the headers before responding.

use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::tcp::OwnedWriteHalf;
use tokio::net::TcpStream;
use tokio::sync::RwLock;

use ferricast_core::{FerricastError, Result};

use crate::ring::SegmentRing;

const PLAYLIST_TARGET_DURATION_HEADROOM: u8 = 4;
const MAX_REQUEST_BYTES: usize = 8 * 1024;

/// Per-connection entry point. Reads one request, writes one
/// response, returns. Errors only when the socket itself misbehaves
/// — protocol-level failures are converted to HTTP error responses.
pub async fn handle(socket: TcpStream, ring: Arc<RwLock<SegmentRing>>) -> Result<()> {
    let (read_half, mut write_half) = socket.into_split();
    let mut reader = BufReader::new(read_half);

    let request = match read_request(&mut reader).await {
        Ok(r) => r,
        Err(e) => {
            // Best-effort error response; ignore write failures since
            // the peer may already have hung up.
            let _ = write_status(&mut write_half, 400, "Bad Request", b"").await;
            return Err(e);
        }
    };

    match (request.method.as_str(), request.path.as_str()) {
        ("OPTIONS", _) => {
            // CORS preflight. No body, but include the same access
            // headers the actual responses do.
            write_options(&mut write_half).await?;
        }
        ("GET" | "HEAD", "/" | "/playlist.m3u8" | "/index.m3u8") => {
            let body = build_playlist(&ring).await?;
            let send_body = request.method == "GET";
            write_ok(
                &mut write_half,
                "application/vnd.apple.mpegurl",
                body.as_bytes(),
                send_body,
                /* cacheable */ false,
            )
            .await?;
        }
        ("GET" | "HEAD", path) if is_segment_path(path) => {
            let Some(seq) = parse_segment_seq(path) else {
                write_status(&mut write_half, 400, "Bad Request", b"").await?;
                return Ok(());
            };
            let segment = {
                let g = ring.read().await;
                g.get(seq).cloned()
            };
            match segment {
                Some(s) => {
                    let send_body = request.method == "GET";
                    write_ok(
                        &mut write_half,
                        "video/mp2t",
                        &s.data,
                        send_body,
                        /* cacheable */ true,
                    )
                    .await?;
                }
                None => {
                    write_status(&mut write_half, 404, "Not Found", b"").await?;
                }
            }
        }
        _ => {
            write_status(&mut write_half, 404, "Not Found", b"").await?;
        }
    }

    finalize(write_half).await;
    Ok(())
}

/// Flush and half-close the write side. Without this, fast clients
/// (ffplay's HLS demuxer reloads several times a second) sometimes
/// see EOF before the kernel has drained the response body — the
/// drop-on-scope-exit close races the loopback FIN.
async fn finalize(mut w: OwnedWriteHalf) {
    let _ = w.flush().await;
    let _ = w.shutdown().await;
}

struct Request {
    method: String,
    path: String,
}

async fn read_request<R>(reader: &mut BufReader<R>) -> Result<Request>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut first_line = String::new();
    let mut total = 0usize;
    let n = reader.read_line(&mut first_line).await?;
    if n == 0 {
        return Err(FerricastError::Hls("client closed before request".into()));
    }
    total += n;
    if total > MAX_REQUEST_BYTES {
        return Err(FerricastError::Hls("request line too long".into()));
    }
    let line = first_line.trim_end_matches(|c| c == '\r' || c == '\n');
    let mut parts = line.splitn(3, ' ');
    let method = parts
        .next()
        .ok_or_else(|| FerricastError::Hls("missing method".into()))?
        .to_string();
    let path = parts
        .next()
        .ok_or_else(|| FerricastError::Hls("missing path".into()))?
        .to_string();
    let _version = parts.next();

    // Drain remaining headers up to the empty line. We don't need any
    // of them, but RFC 9112 requires reading the full message before
    // the response so we don't strand bytes in the buffer.
    loop {
        let mut hdr = String::new();
        let n = reader.read_line(&mut hdr).await?;
        if n == 0 {
            break;
        }
        total += n;
        if total > MAX_REQUEST_BYTES {
            return Err(FerricastError::Hls("request headers too long".into()));
        }
        if hdr == "\r\n" || hdr == "\n" {
            break;
        }
    }

    Ok(Request { method, path })
}

fn is_segment_path(path: &str) -> bool {
    path.starts_with("/segment-") && path.ends_with(".ts")
}

fn parse_segment_seq(path: &str) -> Option<u64> {
    let body = path.strip_prefix("/segment-")?;
    let num = body.strip_suffix(".ts")?;
    num.parse().ok()
}

async fn build_playlist(ring: &Arc<RwLock<SegmentRing>>) -> Result<String> {
    let g = ring.read().await;
    g.build_playlist(PLAYLIST_TARGET_DURATION_HEADROOM)
}

// ---------------------------------------------------------------------
// Response writers
// ---------------------------------------------------------------------

async fn write_ok<W>(
    w: &mut W,
    content_type: &str,
    body: &[u8],
    send_body: bool,
    cacheable: bool,
) -> Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    let mut head = String::with_capacity(256);
    head.push_str("HTTP/1.1 200 OK\r\n");
    head.push_str(&format!("Content-Type: {}\r\n", content_type));
    head.push_str(&format!("Content-Length: {}\r\n", body.len()));
    if cacheable {
        // Segments are immutable once produced; the player can cache
        // forever (within its retention window).
        head.push_str("Cache-Control: public, max-age=31536000, immutable\r\n");
    } else {
        head.push_str("Cache-Control: no-cache, no-store, must-revalidate\r\n");
    }
    head.push_str("Connection: close\r\n");
    write_cors(&mut head);
    head.push_str("\r\n");

    w.write_all(head.as_bytes()).await?;
    if send_body {
        w.write_all(body).await?;
    }
    Ok(())
}

async fn write_status<W>(w: &mut W, code: u16, reason: &str, body: &[u8]) -> Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    let mut head = String::with_capacity(128);
    head.push_str(&format!("HTTP/1.1 {} {}\r\n", code, reason));
    head.push_str(&format!("Content-Length: {}\r\n", body.len()));
    head.push_str("Connection: close\r\n");
    write_cors(&mut head);
    head.push_str("\r\n");
    w.write_all(head.as_bytes()).await?;
    if !body.is_empty() {
        w.write_all(body).await?;
    }
    Ok(())
}

async fn write_options<W>(w: &mut W) -> Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    let mut head = String::with_capacity(192);
    head.push_str("HTTP/1.1 204 No Content\r\n");
    head.push_str("Content-Length: 0\r\n");
    head.push_str("Connection: close\r\n");
    write_cors(&mut head);
    head.push_str("\r\n");
    w.write_all(head.as_bytes()).await?;
    Ok(())
}

/// Permissive CORS — HLS playback in browsers (hls.js, video.js,
/// shaka) only ever does plain GET, so wildcarding origin is safe.
fn write_cors(head: &mut String) {
    head.push_str("Access-Control-Allow-Origin: *\r\n");
    head.push_str("Access-Control-Allow-Methods: GET, HEAD, OPTIONS\r\n");
    head.push_str("Access-Control-Allow-Headers: Range, Origin, Accept\r\n");
    head.push_str("Access-Control-Expose-Headers: Content-Length\r\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_paths_round_trip() {
        assert!(is_segment_path("/segment-0.ts"));
        assert!(is_segment_path("/segment-12345.ts"));
        assert!(!is_segment_path("/segment.ts"));
        assert!(!is_segment_path("/playlist.m3u8"));
        assert_eq!(parse_segment_seq("/segment-0.ts"), Some(0));
        assert_eq!(parse_segment_seq("/segment-42.ts"), Some(42));
        assert_eq!(parse_segment_seq("/segment-x.ts"), None);
        assert_eq!(parse_segment_seq("/segment-.ts"), None);
    }
}
