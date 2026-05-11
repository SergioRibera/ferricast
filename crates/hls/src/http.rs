//! Minimal HTTP/1.x server for the HLS endpoints.
//!
//! Routes:
//! * `GET /`, `GET /playlist.m3u8`, `GET /index.m3u8` → media playlist
//! * `GET /segment-{seq}.ts` → MPEG-TS segment from the ring
//!
//! HTTP/1.1 with keep-alive, byte-range support on segment GETs, and
//! the headers the Cast Application Framework's HLS player likes to
//! see (`Date`, `Accept-Ranges`, conditional `Connection`). The
//! keep-alive loop lets a single chromecast TCP connection serve
//! many playlist polls + segment fetches; without it every fetch
//! is a fresh TCP handshake, which on 1st/2nd-gen receivers
//! amplifies their firmware's RST-cascade failure mode.
//!
//! No third-party HTTP parser; we only need the request line and we
//! parse the few headers we care about (`Connection`, `Range`)
//! manually.

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::tcp::OwnedWriteHalf;
use tokio::net::TcpStream;
use tokio::sync::RwLock;
use tokio::time::timeout;

use ferricast_core::{AdaptiveBitrateState, FerricastError, Result};

use crate::ring::SegmentRing;
use crate::stats::SessionStats;

const PLAYLIST_TARGET_DURATION_HEADROOM: u8 = 4;
const MAX_REQUEST_BYTES: usize = 8 * 1024;
/// Drop a keep-alive connection that's been idle longer than this.
/// HLS players poll the playlist every `target_duration / 2` seconds
/// (so 2 s in our config); 30 s of silence means the client moved on.
const KEEPALIVE_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
/// Upper bound on how long we block a `?_HLS_msn=X&_HLS_part=Y`
/// playlist request waiting for the ring to reach that point. RFC
/// 8216bis §6.2.5.2 doesn't fix a number; in practice Apple's
/// reference player gives up around 6 × PART-TARGET, so we pick a
/// generous 10 s — that covers a couple of segment-build cycles
/// even at the larger 4 s segment_target.
const BLOCKING_RELOAD_TIMEOUT: Duration = Duration::from_secs(10);

/// Per-connection entry point. Loops as long as the client wants to
/// keep the TCP connection alive (HTTP/1.1 default; opt-out via
/// `Connection: close`).
pub async fn handle(
    socket: TcpStream,
    ring: Arc<RwLock<SegmentRing>>,
    adaptive: Option<Arc<AdaptiveBitrateState>>,
    stats: Arc<SessionStats>,
) -> Result<()> {
    let peer = socket
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| "?".into());
    // TCP_NODELAY: pre-fragment a request/response on its own TCP
    // segment. Helps small responses (playlists, 304s) avoid waiting
    // for the next batch of pacing data on the keep-alive path.
    let _ = socket.set_nodelay(true);
    let (read_half, mut write_half) = socket.into_split();
    let mut reader = BufReader::new(read_half);

    loop {
        let request = match timeout(KEEPALIVE_IDLE_TIMEOUT, read_request(&mut reader)).await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                // Distinguish "client closed quietly" (FIN/zero read,
                // very common with chromecast's speculative-prefetch
                // RST behaviour) from real parse errors.
                if matches!(&e, FerricastError::Hls(msg) if msg.contains("client closed before request"))
                {
                    return Ok(());
                }
                tracing::warn!(%peer, %e, "HLS bad request");
                let _ = write_status(&mut write_half, 400, "Bad Request", b"", false).await;
                return Err(e);
            }
            Err(_) => {
                // Idle timeout. Drop the connection.
                tracing::trace!(%peer, "HLS keep-alive idle timeout");
                break;
            }
        };

        let client_wants_close = request.connection_close;
        let send_start = Instant::now();
        let mut delivery_warning: Option<(u64, u64, f64, f32, u32)> = None;
        let mut suppress_generic_log = false;

        // Split path and query — we use the query on the playlist
        // endpoint to implement LL-HLS blocking reload, but the
        // route match below should still ignore it.
        let (path_only, query) = match request.path.split_once('?') {
            Some((p, q)) => (p, q),
            None => (request.path.as_str(), ""),
        };
        let (status, body_len) = match (request.method.as_str(), path_only) {
            ("OPTIONS", _) => {
                write_options(&mut write_half, !client_wants_close).await?;
                (204, 0usize)
            }
            ("GET" | "HEAD", "/" | "/playlist.m3u8" | "/index.m3u8") => {
                // LL-HLS blocking reload: if the client asked for a
                // specific (msn, part) point, wait until the ring
                // has reached it (or we hit the timeout) before
                // serving. Without this, LL-HLS clients fall back
                // to the spec's busy-poll behaviour, which works
                // but loses the latency benefit.
                if let Some((msn, part)) = parse_blocking_reload(query) {
                    let waited = wait_for_msn(&ring, msn, part, BLOCKING_RELOAD_TIMEOUT).await;
                    tracing::debug!(
                        %peer,
                        msn,
                        ?part,
                        served = waited,
                        "LL-HLS blocking reload"
                    );
                }
                let body = build_playlist(&ring).await?;
                let send_body = request.method == "GET";
                let len = body.len();
                write_ok(
                    &mut write_half,
                    "application/vnd.apple.mpegurl",
                    body.as_bytes(),
                    send_body,
                    /* cacheable */ false,
                    /* range_supported */ false,
                    /* keep_alive */ !client_wants_close,
                )
                .await?;
                (200, len)
            }
            ("GET" | "HEAD", path) if is_part_path(path) => {
                let Some((seg_seq, part_idx)) = parse_part_seq_and_idx(path) else {
                    write_status(&mut write_half, 400, "Bad Request", b"", !client_wants_close)
                        .await?;
                    if client_wants_close {
                        break;
                    }
                    continue;
                };
                let part = {
                    let g = ring.read().await;
                    g.get_part(seg_seq, part_idx).cloned()
                };
                match part {
                    Some(p) => {
                        let send_body = request.method == "GET";
                        let len = p.data.len();
                        write_ok(
                            &mut write_half,
                            "video/mp2t",
                            &p.data,
                            send_body,
                            /* cacheable */ true,
                            /* range_supported */ false,
                            /* keep_alive */ !client_wants_close,
                        )
                        .await?;
                        if send_body && len > 0 {
                            tracing::debug!(
                                %peer,
                                seg_seq,
                                part_idx,
                                bytes = len,
                                independent = p.independent,
                                "HLS part delivered"
                            );
                        }
                        (200, len)
                    }
                    None => {
                        write_status(&mut write_half, 404, "Not Found", b"", !client_wants_close)
                            .await?;
                        (404, 0)
                    }
                }
            }
            ("GET" | "HEAD", path) if is_segment_path(path) => {
                let Some(seq) = parse_segment_seq(path) else {
                    write_status(&mut write_half, 400, "Bad Request", b"", !client_wants_close)
                        .await?;
                    tracing::info!(
                        %peer,
                        method = %request.method,
                        path = %request.path,
                        status = 400,
                        "HLS"
                    );
                    if client_wants_close {
                        break;
                    }
                    continue;
                };
                let segment = {
                    let g = ring.read().await;
                    g.get(seq).cloned()
                };
                match segment {
                    Some(s) => {
                        let send_body = request.method == "GET";
                        let total_len = s.data.len();
                        let seg_dur = s.duration_secs;
                        let produced_at = s.produced_at;
                        // Resolve any Range request. Clients that don't
                        // ask just get the full body.
                        let (start, end) =
                            resolve_range(request.range, total_len).unwrap_or((0, total_len));
                        let slice = &s.data[start..end];
                        let len = slice.len();
                        if let Some(_r) = request.range {
                            write_partial_content(
                                &mut write_half,
                                "video/mp2t",
                                slice,
                                send_body,
                                start,
                                end,
                                total_len,
                                !client_wants_close,
                            )
                            .await?;
                        } else {
                            write_ok(
                                &mut write_half,
                                "video/mp2t",
                                slice,
                                send_body,
                                /* cacheable */ true,
                                /* range_supported */ true,
                                /* keep_alive */ !client_wants_close,
                            )
                            .await?;
                        }
                        if send_body && len > 0 {
                            suppress_generic_log = true;
                            let elapsed = send_start.elapsed();
                            let elapsed_s = elapsed.as_secs_f64().max(0.001);
                            let mbps = (len as f64 * 8.0) / 1_000_000.0 / elapsed_s;
                            let staleness = send_start
                                .checked_duration_since(produced_at)
                                .unwrap_or_default();
                            let budget_ratio = elapsed.as_secs_f32() / seg_dur.max(0.001);
                            let pct = (budget_ratio * 100.0) as u32;
                            let encoded_kbps = ((total_len as f64 * 8.0)
                                / 1000.0
                                / seg_dur.max(0.001) as f64)
                                as u32;

                            let tele = stats.record_segment_get(seq, mbps, Instant::now());
                            let gap_ms = tele.inter_request_gap.map(|d| d.as_millis() as u64);
                            let seq_jump = tele
                                .prev_seq
                                .and_then(|p| if seq > p { Some(seq - p) } else { None });

                            tracing::info!(
                                %peer,
                                seq,
                                bytes = len,
                                encoded_kbps,
                                elapsed_ms = elapsed.as_millis() as u64,
                                staleness_ms = staleness.as_millis() as u64,
                                mbps = format_args!("{mbps:.2}"),
                                seg_duration_s = seg_dur,
                                budget_used_pct = pct,
                                inter_request_gap_ms = ?gap_ms,
                                seq_jump = ?seq_jump,
                                count = tele.count_so_far,
                                "HLS segment delivered"
                            );

                            if let (Some(avg_mbps), Some(state)) =
                                (tele.probe_avg_mbps, adaptive.as_deref())
                            {
                                let avg_kbps = (avg_mbps * 1000.0) as u32;
                                match state.observe_link_capacity_kbps(avg_kbps) {
                                    Some(new_target) => tracing::warn!(
                                        measured_avg_mbps = format_args!("{avg_mbps:.2}"),
                                        measured_avg_kbps = avg_kbps,
                                        new_target_kbps = new_target,
                                        samples = SessionStats::PROBE_SAMPLES,
                                        "bandwidth probe complete — adjusted target"
                                    ),
                                    None => tracing::info!(
                                        measured_avg_mbps = format_args!("{avg_mbps:.2}"),
                                        measured_avg_kbps = avg_kbps,
                                        samples = SessionStats::PROBE_SAMPLES,
                                        "bandwidth probe complete — link comfortably above target, no adjustment"
                                    ),
                                }
                            }

                            if let Some(state) = adaptive.as_deref() {
                                if let Some(new_kbps) = state.record_pressure(pct) {
                                    tracing::info!(
                                        new_target_kbps = new_kbps,
                                        ema_pct = state.pressure_pct(),
                                        sample_pct = pct,
                                        "adaptive bitrate adjustment requested"
                                    );
                                }
                            }

                            if budget_ratio >= 0.7 {
                                delivery_warning = Some((
                                    seq,
                                    elapsed.as_millis() as u64,
                                    mbps,
                                    seg_dur,
                                    pct,
                                ));
                            }
                            let stale_ratio = staleness.as_secs_f32() / seg_dur.max(0.001);
                            if stale_ratio >= 2.0 {
                                tracing::warn!(
                                    %peer,
                                    seq,
                                    staleness_ms = staleness.as_millis() as u64,
                                    seg_duration_s = seg_dur,
                                    stale_ratio = format_args!("{stale_ratio:.1}x"),
                                    "HLS segment was stale on fetch — receiver paused, will likely trip 301 soon"
                                );
                            }
                            if let Some(gap) = tele.inter_request_gap {
                                let gap_ratio = gap.as_secs_f32() / seg_dur.max(0.001);
                                if gap_ratio >= 2.0 {
                                    tracing::warn!(
                                        %peer,
                                        seq,
                                        gap_ms = gap.as_millis() as u64,
                                        seg_duration_s = seg_dur,
                                        gap_ratio = format_args!("{gap_ratio:.1}x"),
                                        "HLS receiver fetch cadence slowed past 2× EXTINF — buffer is draining"
                                    );
                                }
                            }
                        }
                        (200, len)
                    }
                    None => {
                        write_status(&mut write_half, 404, "Not Found", b"", !client_wants_close)
                            .await?;
                        (404, 0)
                    }
                }
            }
            _ => {
                write_status(&mut write_half, 404, "Not Found", b"", !client_wants_close).await?;
                (404, 0)
            }
        };

        if !suppress_generic_log {
            tracing::info!(
                %peer,
                method = %request.method,
                path = %request.path,
                status,
                body_len,
                "HLS"
            );
        }

        if let Some((seq, elapsed_ms, mbps, seg_dur, pct)) = delivery_warning {
            tracing::warn!(
                %peer,
                seq,
                elapsed_ms,
                mbps = format_args!("{mbps:.2}"),
                seg_duration_s = seg_dur,
                budget_used_pct = pct,
                "HLS segment delivery used most of its playback budget — receiver bandwidth is the bottleneck (lower bitrate or move closer to AP)"
            );
        }

        write_half.flush().await.ok();

        if client_wants_close {
            break;
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
    /// `true` if the client sent `Connection: close`. HTTP/1.1
    /// defaults to keep-alive, so the absence of this header (or its
    /// presence with the value `keep-alive`) means "keep the
    /// connection open".
    connection_close: bool,
    /// Parsed `Range: bytes=START-END?` (single-range form only —
    /// multi-range / suffix-range fall back to a full-body response).
    range: Option<(u64, Option<u64>)>,
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
    let version = parts.next().unwrap_or("HTTP/1.0");
    // HTTP/1.0 is opt-out for keep-alive; default close. 1.1 is opt-in.
    let mut connection_close = version.eq_ignore_ascii_case("HTTP/1.0");
    let mut range: Option<(u64, Option<u64>)> = None;

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
        // Parse the two headers we care about. Everything else gets
        // drained (per RFC 9112 we must consume the full message
        // before responding).
        if let Some((name, value)) = hdr.split_once(':') {
            let name = name.trim();
            let value = value.trim().trim_end_matches(|c| c == '\r' || c == '\n');
            if name.eq_ignore_ascii_case("Connection") {
                if value.eq_ignore_ascii_case("close") {
                    connection_close = true;
                } else if value.eq_ignore_ascii_case("keep-alive") {
                    connection_close = false;
                }
            } else if name.eq_ignore_ascii_case("Range") {
                range = parse_range(value);
            }
        }
    }

    Ok(Request {
        method,
        path,
        connection_close,
        range,
    })
}

/// Parse `bytes=START-END?`. Returns `None` for multi-range, suffix-
/// range, or any malformed form — callers fall back to a full-body
/// response, which is always a valid HTTP answer to a Range request
/// (the server simply chose not to honour it).
fn parse_range(value: &str) -> Option<(u64, Option<u64>)> {
    let rest = value.strip_prefix("bytes=")?;
    // Multi-range form (`bytes=0-99,200-299`) — punt.
    if rest.contains(',') {
        return None;
    }
    let (start, end) = rest.split_once('-')?;
    let start = start.trim().parse::<u64>().ok()?;
    let end = end.trim();
    let end = if end.is_empty() {
        None
    } else {
        Some(end.parse::<u64>().ok()?)
    };
    Some((start, end))
}

/// Convert a parsed Range header into a half-open byte interval `[a, b)`
/// over the segment body, clamping to its actual length. Returns
/// `None` if the range is entirely outside the body (caller should
/// then fall back to a 200 full-body response — strict HTTP would
/// say 416, but Chromecast HLS players don't rely on Range strictness
/// and falling back is the more forgiving choice).
fn resolve_range(range: Option<(u64, Option<u64>)>, total: usize) -> Option<(usize, usize)> {
    let (start, end) = range?;
    let total64 = total as u64;
    if start >= total64 {
        return None;
    }
    let last = match end {
        Some(e) => e.min(total64.saturating_sub(1)),
        None => total64.saturating_sub(1),
    };
    if last < start {
        return None;
    }
    Some((start as usize, (last + 1) as usize))
}

fn is_segment_path(path: &str) -> bool {
    path.starts_with("/segment-") && path.ends_with(".ts")
}

fn parse_segment_seq(path: &str) -> Option<u64> {
    let body = path.strip_prefix("/segment-")?;
    let num = body.strip_suffix(".ts")?;
    num.parse().ok()
}

fn is_part_path(path: &str) -> bool {
    path.starts_with("/part-") && path.ends_with(".ts")
}

/// `/part-{seg}.{idx}.ts` → `(seg, idx)`.
fn parse_part_seq_and_idx(path: &str) -> Option<(u64, u32)> {
    let body = path.strip_prefix("/part-")?;
    let body = body.strip_suffix(".ts")?;
    let (seg, idx) = body.split_once('.')?;
    Some((seg.parse().ok()?, idx.parse().ok()?))
}

/// Parse the `?_HLS_msn=X&_HLS_part=Y` (or just `_HLS_msn=X`) query
/// the LL-HLS spec uses to ask for a blocking playlist reload. Any
/// other query keys are ignored. Returns `None` when the client
/// isn't asking for blocking semantics — caller serves the current
/// playlist immediately in that case.
fn parse_blocking_reload(query: &str) -> Option<(u64, Option<u32>)> {
    if query.is_empty() {
        return None;
    }
    let mut msn: Option<u64> = None;
    let mut part: Option<u32> = None;
    for pair in query.split('&') {
        let (k, v) = pair.split_once('=')?;
        match k {
            "_HLS_msn" => msn = v.parse().ok(),
            "_HLS_part" => part = v.parse().ok(),
            _ => {}
        }
    }
    msn.map(|m| (m, part))
}

/// Wait until the ring has published at least up to `(msn, part)`,
/// or `deadline` elapses, whichever comes first. The implementation
/// captures the ring's `Notify` once and waits on a fresh future per
/// iteration so notifications fired between two checks don't get
/// lost.
async fn wait_for_msn(
    ring: &Arc<RwLock<SegmentRing>>,
    msn: u64,
    part: Option<u32>,
    deadline: Duration,
) -> bool {
    let notify = {
        let g = ring.read().await;
        if g.has_reached(msn, part) {
            return true;
        }
        g.notify.clone()
    };
    let start = Instant::now();
    loop {
        // The future captures "right now"; if a notification fires
        // before we await, the await completes immediately. So we
        // must create it BEFORE re-checking.
        let notified = notify.notified();
        if ring.read().await.has_reached(msn, part) {
            return true;
        }
        let remaining = match deadline.checked_sub(start.elapsed()) {
            Some(r) if !r.is_zero() => r,
            _ => return false,
        };
        match timeout(remaining, notified).await {
            Ok(()) => continue,
            Err(_) => return false,
        }
    }
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
    range_supported: bool,
    keep_alive: bool,
) -> Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    let mut head = String::with_capacity(384);
    head.push_str("HTTP/1.1 200 OK\r\n");
    write_common(&mut head, content_type, body.len(), cacheable, range_supported, keep_alive);
    head.push_str("\r\n");

    w.write_all(head.as_bytes()).await?;
    if send_body {
        w.write_all(body).await?;
    }
    Ok(())
}

async fn write_partial_content<W>(
    w: &mut W,
    content_type: &str,
    body: &[u8],
    send_body: bool,
    start: usize,
    end: usize,
    total: usize,
    keep_alive: bool,
) -> Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    let mut head = String::with_capacity(384);
    head.push_str("HTTP/1.1 206 Partial Content\r\n");
    write_common(&mut head, content_type, body.len(), true, true, keep_alive);
    head.push_str(&format!(
        "Content-Range: bytes {}-{}/{}\r\n",
        start,
        end.saturating_sub(1),
        total
    ));
    head.push_str("\r\n");
    w.write_all(head.as_bytes()).await?;
    if send_body {
        w.write_all(body).await?;
    }
    Ok(())
}

async fn write_status<W>(
    w: &mut W,
    code: u16,
    reason: &str,
    body: &[u8],
    keep_alive: bool,
) -> Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    let mut head = String::with_capacity(192);
    head.push_str(&format!("HTTP/1.1 {} {}\r\n", code, reason));
    write_common(&mut head, "text/plain", body.len(), false, false, keep_alive);
    head.push_str("\r\n");
    w.write_all(head.as_bytes()).await?;
    if !body.is_empty() {
        w.write_all(body).await?;
    }
    Ok(())
}

async fn write_options<W>(w: &mut W, keep_alive: bool) -> Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    let mut head = String::with_capacity(256);
    head.push_str("HTTP/1.1 204 No Content\r\n");
    write_common(&mut head, "text/plain", 0, false, false, keep_alive);
    head.push_str("\r\n");
    w.write_all(head.as_bytes()).await?;
    Ok(())
}

/// Headers we put on every response: Date (required by RFC 7231),
/// Content-Type, Content-Length, Cache-Control, optional
/// Accept-Ranges, Connection, CORS. Splitting this out keeps the
/// individual response writers small.
fn write_common(
    head: &mut String,
    content_type: &str,
    content_length: usize,
    cacheable: bool,
    range_supported: bool,
    keep_alive: bool,
) {
    // Date header — required for any origin response per RFC 7231
    // §7.1.1.2. Some CAF receivers correlate it with the playlist's
    // EXT-X-PROGRAM-DATE-TIME for live-edge tracking; missing Date
    // is at best a missed signal, at worst (rare) a 500.
    head.push_str(&format!("Date: {}\r\n", http_date(SystemTime::now())));
    head.push_str(&format!("Content-Type: {}\r\n", content_type));
    head.push_str(&format!("Content-Length: {}\r\n", content_length));
    if cacheable {
        head.push_str("Cache-Control: public, max-age=31536000, immutable\r\n");
    } else {
        head.push_str("Cache-Control: no-cache, no-store, must-revalidate\r\n");
    }
    if range_supported {
        // Advertise byte-range support. We honour single-range
        // requests; multi-range falls back to 200 + full body.
        // Telling the receiver up front avoids it probing with a
        // HEAD or a `Range: bytes=0-0` test fetch.
        head.push_str("Accept-Ranges: bytes\r\n");
    } else {
        head.push_str("Accept-Ranges: none\r\n");
    }
    head.push_str(if keep_alive {
        "Connection: keep-alive\r\nKeep-Alive: timeout=30\r\n"
    } else {
        "Connection: close\r\n"
    });
    write_cors(head);
}

/// Permissive CORS — HLS playback in browsers (hls.js, video.js,
/// shaka) only ever does plain GET, so wildcarding origin is safe.
fn write_cors(head: &mut String) {
    head.push_str("Access-Control-Allow-Origin: *\r\n");
    head.push_str("Access-Control-Allow-Methods: GET, HEAD, OPTIONS\r\n");
    head.push_str("Access-Control-Allow-Headers: Range, Origin, Accept\r\n");
    head.push_str("Access-Control-Expose-Headers: Content-Length, Content-Range, Accept-Ranges\r\n");
}

// ---------------------------------------------------------------------
// IMF-fixdate (RFC 7231 §7.1.1.1) formatter — no dependency
// ---------------------------------------------------------------------

/// Format a `SystemTime` as an HTTP `Date` header value, e.g.
/// `Sun, 06 Nov 1994 08:49:37 GMT`. The math uses Howard Hinnant's
/// civil-from-days algorithm so it's correct across the entire
/// proleptic Gregorian range — no special-casing of leap years or
/// century boundaries.
fn http_date(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let secs_per_day: u64 = 86_400;
    let days = (secs / secs_per_day) as i64;
    let secs_today = (secs % secs_per_day) as u32;
    let hour = secs_today / 3600;
    let minute = (secs_today / 60) % 60;
    let second = secs_today % 60;
    // 1970-01-01 was a Thursday; days since then.
    let weekday = ((days + 4).rem_euclid(7)) as usize;
    let (year, month, day) = civil_from_days(days);

    const WD: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const MO: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    format!(
        "{}, {:02} {} {:04} {:02}:{:02}:{:02} GMT",
        WD[weekday],
        day,
        MO[(month - 1) as usize],
        year,
        hour,
        minute,
        second
    )
}

/// Days since 1970-01-01 → (year, month [1-12], day [1-31]).
/// Howard Hinnant's `civil_from_days`. Year is always positive in
/// practical use (we feed it `SystemTime::now()`).
fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i32 + (era as i32) * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
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

    #[test]
    fn range_header_single_open() {
        assert_eq!(parse_range("bytes=0-99"), Some((0, Some(99))));
        assert_eq!(parse_range("bytes=100-"), Some((100, None)));
    }

    #[test]
    fn range_header_invalid_falls_back() {
        assert_eq!(parse_range("foo=0-99"), None);
        assert_eq!(parse_range("bytes=abc-99"), None);
        // Multi-range — we deliberately decline.
        assert_eq!(parse_range("bytes=0-99,200-299"), None);
    }

    #[test]
    fn resolve_range_clamps_to_body() {
        assert_eq!(resolve_range(Some((0, Some(9))), 1000), Some((0, 10)));
        assert_eq!(resolve_range(Some((0, Some(9999))), 1000), Some((0, 1000)));
        assert_eq!(resolve_range(Some((100, None)), 1000), Some((100, 1000)));
        // Start past EOF → None (caller falls back to 200 full body).
        assert_eq!(resolve_range(Some((10_000, None)), 1000), None);
    }

    #[test]
    fn http_date_format() {
        // 2024-01-15 12:34:56 UTC is a Monday.
        let t = UNIX_EPOCH + Duration::from_secs(1_705_322_096);
        assert_eq!(http_date(t), "Mon, 15 Jan 2024 12:34:56 GMT");
    }

    #[test]
    fn http_date_leap_year() {
        // 2024-02-29 is valid (leap year). Seconds since epoch:
        // 2024-02-29 00:00:00 UTC = 1_709_164_800.
        let t = UNIX_EPOCH + Duration::from_secs(1_709_164_800);
        assert_eq!(http_date(t), "Thu, 29 Feb 2024 00:00:00 GMT");
    }

    #[test]
    fn part_path_parses() {
        assert!(is_part_path("/part-42.3.ts"));
        assert!(!is_part_path("/segment-42.ts"));
        assert_eq!(parse_part_seq_and_idx("/part-42.3.ts"), Some((42, 3)));
        assert_eq!(parse_part_seq_and_idx("/part-0.0.ts"), Some((0, 0)));
        assert_eq!(parse_part_seq_and_idx("/part-x.3.ts"), None);
        assert_eq!(parse_part_seq_and_idx("/part-42.ts"), None);
    }

    #[test]
    fn blocking_reload_query_parses() {
        assert_eq!(parse_blocking_reload(""), None);
        assert_eq!(parse_blocking_reload("foo=bar"), None);
        assert_eq!(parse_blocking_reload("_HLS_msn=12"), Some((12, None)));
        assert_eq!(
            parse_blocking_reload("_HLS_msn=12&_HLS_part=3"),
            Some((12, Some(3)))
        );
        // Order doesn't matter.
        assert_eq!(
            parse_blocking_reload("_HLS_part=3&_HLS_msn=12"),
            Some((12, Some(3)))
        );
    }
}
