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

use std::fmt::Write as _;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::RwLock;
use tokio::time::timeout;

use ferricast_core::{AdaptiveBitrateState, FerricastError, Result};

use crate::ring::SegmentRing;
use crate::stats::SessionStats;

const MAX_REQUEST_BYTES: usize = 8 * 1024;
/// Drop a keep-alive connection that's been idle longer than this.
/// HLS players poll the playlist every `target_duration / 2` seconds
/// (so 2 s in our config); 30 s of silence means the client moved on.
const KEEPALIVE_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
/// A blocking playlist reload (`?_HLS_msn=X&_HLS_part=Y`) waits at
/// most this many target durations before answering 503
/// (rfc8216bis §6.2.5.2).
const BLOCKING_RELOAD_TARGET_DURATIONS: u32 = 3;

/// Segments are only cached briefly: URLs are unique per server (see
/// [`UrlToken`]), but a long `immutable` lifetime buys nothing for a
/// live window that is gone within a minute.
const SEGMENT_MAX_AGE_SECS: u32 = 60;

/// Random path prefix (`/<token>/…`) every URL of one server carries.
///
/// Segment names restart at `segment-0.ts` for every new server and
/// the port is often reused, so without a per-server prefix a client
/// (or an HTTP cache) could be handed bytes cached from an earlier
/// stream. It also keeps other LAN hosts from guessing the stream URL.
/// 128 bits from the OS CSPRNG, hex encoded.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct UrlToken(String);

impl UrlToken {
    /// # Errors
    ///
    /// The system random number generator failed.
    pub(crate) fn generate() -> Result<Self> {
        use ring::rand::SecureRandom as _;
        let mut bytes = [0u8; 16];
        ring::rand::SystemRandom::new()
            .fill(&mut bytes)
            .map_err(|_| FerricastError::Hls("system RNG failed to produce a URL token".into()))?;
        Ok(Self(bytes.iter().fold(
            String::with_capacity(32),
            |mut hex, b| {
                let _ = write!(hex, "{b:02x}");
                hex
            },
        )))
    }

    /// Path of the media playlist, e.g. `/<token>/index.m3u8`.
    pub(crate) fn playlist_path(&self) -> String {
        format!("/{}/index.m3u8", self.0)
    }

    /// The route part of `path` (`/index.m3u8`, `/segment-3.ts`, …)
    /// when it starts with this token, `None` otherwise. The token is
    /// compared in constant time.
    fn strip<'a>(&self, path: &'a str) -> Option<&'a str> {
        let rest = path.strip_prefix('/')?;
        let token = rest.get(..self.0.len())?;
        let route = rest.get(self.0.len()..)?;
        let same = token
            .bytes()
            .zip(self.0.bytes())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0;
        (same && route.starts_with('/')).then_some(route)
    }
}

impl std::fmt::Debug for UrlToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Keep the secret out of logs.
        f.write_str("UrlToken(..)")
    }
}

/// State one HLS server shares with all of its connection handlers.
pub(crate) struct ServerState {
    pub(crate) ring: Arc<RwLock<SegmentRing>>,
    pub(crate) adaptive: Option<Arc<AdaptiveBitrateState>>,
    pub(crate) stats: SessionStats,
    pub(crate) token: UrlToken,
    /// Peers allowed to fetch besides loopback; empty = everyone.
    /// Canonical form (IPv4-mapped IPv6 folded to IPv4).
    allowed_peers: std::sync::RwLock<Vec<IpAddr>>,
}

impl ServerState {
    /// # Errors
    ///
    /// The URL token could not be generated.
    pub(crate) fn new(
        ring: Arc<RwLock<SegmentRing>>,
        adaptive: Option<Arc<AdaptiveBitrateState>>,
    ) -> Result<Self> {
        Ok(Self {
            ring,
            adaptive,
            stats: SessionStats::new(),
            token: UrlToken::generate()?,
            allowed_peers: std::sync::RwLock::new(Vec::new()),
        })
    }

    /// Restrict fetching to the given peer (plus loopback and any
    /// other allowed peer). Until the first call every peer may fetch.
    pub(crate) fn allow_peer(&self, ip: IpAddr) {
        let ip = ip.to_canonical();
        let mut allowed = self
            .allowed_peers
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        if !allowed.contains(&ip) {
            allowed.push(ip);
        }
    }

    /// Whether `ip` may fetch: always for loopback, otherwise when no
    /// allowlist is set or `ip` is on it.
    fn peer_allowed(&self, ip: IpAddr) -> bool {
        let ip = ip.to_canonical();
        if ip.is_loopback() {
            return true;
        }
        let allowed = self
            .allowed_peers
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        allowed.is_empty() || allowed.contains(&ip)
    }

    /// Whether segment deliveries to `ip` feed the adaptive bitrate
    /// controller: the allowed (cast) peers when an allowlist is set,
    /// otherwise the first peer that fetched a segment.
    fn drives_adaptive(&self, ip: IpAddr) -> bool {
        let ip = ip.to_canonical();
        let allowed = self
            .allowed_peers
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        if allowed.is_empty() {
            self.stats.is_first_peer(ip)
        } else {
            allowed.contains(&ip)
        }
    }
}

/// Per-connection entry point. Loops as long as the client wants to
/// keep the TCP connection alive (HTTP/1.1 default; opt-out via
/// `Connection: close`).
///
/// `io` is generic so the same handler runs on plain TCP and on
/// `TlsStream<TcpStream>` (HTTPS). The TCP-specific `set_nodelay`
/// and `peer_addr` work happens in the caller (the accept loop in
/// [`crate::HlsFrameSink`] / [`crate::HlsServer`]) so this layer
/// stays IO-agnostic.
#[expect(
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "one request loop; the float and integer casts only feed log and telemetry fields, saturating"
)]
pub(crate) async fn handle<I>(peer: SocketAddr, io: I, state: Arc<ServerState>) -> Result<()>
where
    I: AsyncRead + AsyncWrite + Unpin + Send,
{
    let ring = &state.ring;
    let adaptive = state.adaptive.as_ref();
    let stats = &state.stats;
    let peer_ip = peer.ip();
    let peer = peer.to_string();
    let (read_half, write_half) = tokio::io::split(io);
    let mut reader = BufReader::new(read_half);
    // A peer that stops reading (powered-off receiver) must not pin
    // this task until the kernel gives up on the socket.
    let write_timeout = {
        let target = u64::from(ring.read().await.target_duration());
        Duration::from_secs(target.saturating_mul(2)).max(MIN_WRITE_STALL_TIMEOUT)
    };
    let mut write_half = StallTimeout::new(write_half, write_timeout);

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

        // Checked per request: the allowlist may be set after this
        // connection was accepted.
        if !state.peer_allowed(peer_ip) {
            tracing::warn!(%peer, path = %request.path, "HLS request from a peer that is not allowed");
            let _ = write_status(&mut write_half, 403, "Forbidden", b"", false).await;
            break;
        }

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
        // Every route lives under the server's token; anything else
        // falls through to 404 (OPTIONS preflights are answered for
        // any path, they expose nothing).
        let route = state.token.strip(path_only).unwrap_or("");
        let (status, body_len) = match (request.method.as_str(), route) {
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
                if let Some((code, reason)) = blocking_reload(ring, query, &peer).await {
                    write_status(&mut write_half, code, reason, b"", !client_wants_close).await?;
                    tracing::info!(%peer, path = %request.path, status = code, "HLS");
                    if client_wants_close {
                        break;
                    }
                    continue;
                }
                let body = build_playlist(ring).await?;
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
                    write_status(
                        &mut write_half,
                        400,
                        "Bad Request",
                        b"",
                        !client_wants_close,
                    )
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
                if let Some(p) = part {
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
                } else {
                    write_status(&mut write_half, 404, "Not Found", b"", !client_wants_close)
                        .await?;
                    (404, 0)
                }
            }
            ("GET" | "HEAD", path) if is_segment_path(path) => {
                let Some(seq) = parse_segment_seq(path) else {
                    write_status(
                        &mut write_half,
                        400,
                        "Bad Request",
                        b"",
                        !client_wants_close,
                    )
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
                if let Some(s) = segment {
                    let send_body = request.method == "GET";
                    let total_len = s.data.len();
                    let seg_dur = s.duration_secs;
                    let produced_at = s.produced_at;
                    // Resolve any Range request. Clients that don't
                    // ask just get the full body.
                    let reply = resolve_range(request.range, total_len);
                    let (start, end) = match reply {
                        RangeReply::Partial { start, end } => (start, end),
                        RangeReply::Full | RangeReply::Unsatisfiable => (0, total_len),
                    };
                    let slice = s.data.get(start..end).unwrap_or_default();
                    let len = slice.len();
                    if reply == RangeReply::Unsatisfiable {
                        write_range_not_satisfiable(
                            &mut write_half,
                            total_len,
                            !client_wants_close,
                        )
                        .await?;
                        tracing::info!(%peer, path = %request.path, status = 416, "HLS");
                        if client_wants_close {
                            break;
                        }
                        continue;
                    } else if let RangeReply::Partial { .. } = reply {
                        write_partial_content(
                            &mut write_half,
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
                        let encoded_kbps =
                            ((total_len as f64 * 8.0) / 1000.0 / f64::from(seg_dur.max(0.001)))
                                as u32;

                        let tele = stats.record_segment_get(peer_ip, seq, mbps, Instant::now());
                        // Only the cast receiver's deliveries steer
                        // the encoder bitrate; other viewers are
                        // just logged.
                        let adaptive_for_peer = adaptive.filter(|_| state.drives_adaptive(peer_ip));
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
                            (tele.probe_avg_mbps, adaptive_for_peer.map(Arc::as_ref))
                        {
                            let avg_kbps = (avg_mbps * 1000.0) as u32;
                            if let Some(new_target) = state.observe_link_capacity_kbps(avg_kbps) {
                                tracing::warn!(
                                    measured_avg_mbps = format_args!("{avg_mbps:.2}"),
                                    measured_avg_kbps = avg_kbps,
                                    new_target_kbps = new_target,
                                    samples = SessionStats::PROBE_SAMPLES,
                                    "bandwidth probe complete — adjusted target"
                                );
                            } else {
                                tracing::info!(
                                    measured_avg_mbps = format_args!("{avg_mbps:.2}"),
                                    measured_avg_kbps = avg_kbps,
                                    samples = SessionStats::PROBE_SAMPLES,
                                    "bandwidth probe complete — link comfortably above target, no adjustment"
                                );
                            }
                        }

                        if let Some(controller) = adaptive_for_peer
                            && let Some(new_kbps) = controller.record_pressure(pct)
                        {
                            tracing::info!(
                                new_target_kbps = new_kbps,
                                ema_pct = controller.pressure_pct(),
                                sample_pct = pct,
                                "adaptive bitrate adjustment requested"
                            );
                        }

                        if budget_ratio >= 0.7 {
                            delivery_warning =
                                Some((seq, elapsed.as_millis() as u64, mbps, seg_dur, pct));
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
                } else {
                    write_status(&mut write_half, 404, "Not Found", b"", !client_wants_close)
                        .await?;
                    (404, 0)
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
async fn finalize<W>(mut w: W)
where
    W: AsyncWrite + Unpin,
{
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

/// Read one `\n`-terminated line, consuming at most `budget` bytes
/// (decremented by what was read). The cap applies *while* reading, so
/// a client that never sends a newline cannot grow the buffer past
/// the request limit. `Ok(None)` on a clean EOF before any byte.
async fn read_line_bounded<R>(
    reader: &mut BufReader<R>,
    budget: &mut usize,
) -> Result<Option<String>>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut buf = Vec::new();
    // `take` stops at the budget; one extra byte distinguishes
    // "exactly at the limit" from "over it".
    let limit = u64::try_from(*budget).unwrap_or(u64::MAX).saturating_add(1);
    let n = (&mut *reader)
        .take(limit)
        .read_until(b'\n', &mut buf)
        .await?;
    if n == 0 {
        return Ok(None);
    }
    if n > *budget {
        return Err(FerricastError::Hls("request headers too long".into()));
    }
    *budget -= n;
    String::from_utf8(buf)
        .map(Some)
        .map_err(|_| FerricastError::Hls("request is not valid UTF-8".into()))
}

async fn read_request<R>(reader: &mut BufReader<R>) -> Result<Request>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut budget = MAX_REQUEST_BYTES;
    let Some(first_line) = read_line_bounded(reader, &mut budget).await? else {
        return Err(FerricastError::Hls("client closed before request".into()));
    };
    let line = first_line.trim_end_matches(['\r', '\n']);
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
        let Some(hdr) = read_line_bounded(reader, &mut budget).await? else {
            break;
        };
        if hdr == "\r\n" || hdr == "\n" {
            break;
        }
        // Parse the two headers we care about. Everything else gets
        // drained (per RFC 9112 we must consume the full message
        // before responding).
        if let Some((name, value)) = hdr.split_once(':') {
            let name = name.trim();
            let value = value.trim().trim_end_matches(['\r', '\n']);
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

/// How a segment GET answers its (optional) `Range` header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RangeReply {
    /// No usable range: 200 with the whole body. Covers both "no
    /// header" and a syntactically invalid one (`last < first`), which
    /// RFC 9110 §14.2 says to ignore.
    Full,
    /// 206 with the half-open byte interval `[start, end)`.
    Partial { start: usize, end: usize },
    /// 416: the first byte is past the end of the body.
    Unsatisfiable,
}

/// Resolve a parsed Range header against a body of `total` bytes,
/// clamping the last byte to the body length.
fn resolve_range(range: Option<(u64, Option<u64>)>, total: usize) -> RangeReply {
    let Some((start, end)) = range else {
        return RangeReply::Full;
    };
    if let Some(e) = end
        && e < start
    {
        return RangeReply::Full;
    }
    let total64 = u64::try_from(total).unwrap_or(u64::MAX);
    if start >= total64 {
        return RangeReply::Unsatisfiable;
    }
    let last = end.map_or(total64 - 1, |e| e.min(total64 - 1));
    // Both bounds are < total, which is a usize.
    match (usize::try_from(start), usize::try_from(last)) {
        (Ok(start), Ok(last)) => RangeReply::Partial {
            start,
            end: last + 1,
        },
        _ => RangeReply::Full,
    }
}

/// Whether `path` names a `.ts` file. Our URLs are lowercase; the check
/// is case-insensitive like file systems that serve the same names.
fn has_ts_extension(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("ts"))
}

fn is_segment_path(path: &str) -> bool {
    path.starts_with("/segment-") && has_ts_extension(path)
}

fn parse_segment_seq(path: &str) -> Option<u64> {
    let body = path.strip_prefix("/segment-")?;
    let num = body.strip_suffix(".ts")?;
    num.parse().ok()
}

fn is_part_path(path: &str) -> bool {
    path.starts_with("/part-") && has_ts_extension(path)
}

/// `/part-{seg}.{idx}.ts` → `(seg, idx)`.
fn parse_part_seq_and_idx(path: &str) -> Option<(u64, u32)> {
    let body = path.strip_prefix("/part-")?;
    let body = body.strip_suffix(".ts")?;
    let (seg, idx) = body.split_once('.')?;
    Some((seg.parse().ok()?, idx.parse().ok()?))
}

/// A malformed blocking-reload query; answered with 400.
#[derive(Debug, PartialEq, Eq)]
struct BadReloadQuery;

/// Parse the `?_HLS_msn=X&_HLS_part=Y` (or just `_HLS_msn=X`) query
/// the LL-HLS spec uses to ask for a blocking playlist reload. Other
/// query keys are ignored. `Ok(None)` when the client isn't asking for
/// blocking semantics; `Err` for `_HLS_part` without `_HLS_msn` or a
/// non-numeric value (rfc8216bis §6.2.5.2).
fn parse_blocking_reload(
    query: &str,
) -> std::result::Result<Option<(u64, Option<u32>)>, BadReloadQuery> {
    let mut msn: Option<u64> = None;
    let mut part: Option<u32> = None;
    for pair in query.split('&').filter(|p| !p.is_empty()) {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        match k {
            "_HLS_msn" => msn = Some(v.parse().map_err(|_| BadReloadQuery)?),
            "_HLS_part" => part = Some(v.parse().map_err(|_| BadReloadQuery)?),
            _ => {}
        }
    }
    match (msn, part) {
        (None, Some(_)) => Err(BadReloadQuery),
        (None, None) => Ok(None),
        (Some(m), p) => Ok(Some((m, p))),
    }
}

/// Handle the LL-HLS blocking-reload part of a playlist request.
/// `None` means "serve the playlist now"; `Some((code, reason))` is
/// an error status to answer with instead: 400 for a malformed query,
/// an `_HLS_msn` more than two past the last segment or an `_HLS_part`
/// beyond the Advance Part Limit, 503 when the
/// requested point is not reached within three target durations
/// (rfc8216bis §6.2.5.2).
async fn blocking_reload(
    ring: &Arc<RwLock<SegmentRing>>,
    query: &str,
    peer: &str,
) -> Option<(u16, &'static str)> {
    let (msn, part) = match parse_blocking_reload(query) {
        Ok(Some(point)) => point,
        Ok(None) => return None,
        Err(BadReloadQuery) => return Some((400, "Bad Request")),
    };
    let (too_far, target_duration) = {
        let g = ring.read().await;
        // The segment under construction is `next_seq`; the last
        // complete one is the one before it.
        let msn_too_far = match g.next_seq().checked_sub(1) {
            Some(last) => msn > last.saturating_add(2),
            // Nothing published yet: allow the first two segments.
            None => msn > 1,
        };
        let part_too_far = part.is_some_and(|p| g.exceeds_advance_part_limit(msn, p));
        (msn_too_far || part_too_far, g.target_duration())
    };
    if too_far {
        return Some((400, "Bad Request"));
    }
    let deadline = Duration::from_secs(u64::from(target_duration))
        .saturating_mul(BLOCKING_RELOAD_TARGET_DURATIONS);
    let reached = wait_for_msn(ring, msn, part, deadline).await;
    tracing::debug!(%peer, msn, ?part, reached, "LL-HLS blocking reload");
    (!reached).then_some((503, "Service Unavailable"))
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
        if timeout(remaining, notified).await.is_err() {
            return false;
        }
    }
}

async fn build_playlist(ring: &Arc<RwLock<SegmentRing>>) -> Result<String> {
    let g = ring.read().await;
    g.build_playlist()
}

// ---------------------------------------------------------------------
// Response writers
// ---------------------------------------------------------------------

#[expect(
    clippy::fn_params_excessive_bools,
    reason = "each flag toggles one independent header"
)]
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
    write_common(
        &mut head,
        content_type,
        body.len(),
        cacheable,
        range_supported,
        keep_alive,
    );
    head.push_str("\r\n");

    w.write_all(head.as_bytes()).await?;
    if send_body {
        w.write_all(body).await?;
    }
    Ok(())
}

/// 206 response carrying `body`, the `[start, end)` slice of a
/// `total`-byte MPEG-TS segment.
async fn write_partial_content<W>(
    w: &mut W,
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
    write_common(&mut head, "video/mp2t", body.len(), true, true, keep_alive);
    let _ = write!(
        head,
        "Content-Range: bytes {start}-{}/{total}\r\n",
        end.saturating_sub(1)
    );
    head.push_str("\r\n");
    w.write_all(head.as_bytes()).await?;
    if send_body {
        w.write_all(body).await?;
    }
    Ok(())
}

/// 416 for a range starting past the end of a `total`-byte body, with
/// the `Content-Range: bytes */total` header RFC 9110 §15.5.17 asks for.
async fn write_range_not_satisfiable<W>(w: &mut W, total: usize, keep_alive: bool) -> Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    let mut head = String::with_capacity(256);
    head.push_str("HTTP/1.1 416 Range Not Satisfiable\r\n");
    write_common(&mut head, "text/plain", 0, false, true, keep_alive);
    let _ = write!(head, "Content-Range: bytes */{total}\r\n\r\n");
    w.write_all(head.as_bytes()).await?;
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
    let _ = write!(head, "HTTP/1.1 {code} {reason}\r\n");
    write_common(
        &mut head,
        "text/plain",
        body.len(),
        false,
        false,
        keep_alive,
    );
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
    let _ = write!(head, "Date: {}\r\n", http_date(SystemTime::now()));
    let _ = write!(head, "Content-Type: {content_type}\r\n");
    let _ = write!(head, "Content-Length: {content_length}\r\n");
    if cacheable {
        let _ = write!(
            head,
            "Cache-Control: public, max-age={SEGMENT_MAX_AGE_SECS}\r\n"
        );
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
    head.push_str(
        "Access-Control-Expose-Headers: Content-Length, Content-Range, Accept-Ranges\r\n",
    );
}

// ---------------------------------------------------------------------
// IMF-fixdate (RFC 7231 §7.1.1.1) formatter — no dependency
// ---------------------------------------------------------------------

/// Format a `SystemTime` as an HTTP `Date` header value, e.g.
/// `Sun, 06 Nov 1994 08:49:37 GMT`. The math uses Howard Hinnant's
/// civil-from-days algorithm so it's correct across the entire
/// proleptic Gregorian range — no special-casing of leap years or
/// century boundaries.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "seconds since 1970 fit i64; seconds of a day, weekday and month are small"
)]
fn http_date(t: SystemTime) -> String {
    const WD: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const MO: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
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
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "Hinnant's algorithm keeps day-of-era and year-of-era in range"
)]
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
        let partial = |start, end| RangeReply::Partial { start, end };
        assert_eq!(resolve_range(Some((0, Some(9))), 1000), partial(0, 10));
        assert_eq!(resolve_range(Some((0, Some(9999))), 1000), partial(0, 1000));
        assert_eq!(resolve_range(Some((100, None)), 1000), partial(100, 1000));
        assert_eq!(resolve_range(None, 1000), RangeReply::Full);
        // last < first is syntactically invalid: ignored.
        assert_eq!(resolve_range(Some((10, Some(5))), 1000), RangeReply::Full);
        // Start past EOF → 416.
        assert_eq!(
            resolve_range(Some((10_000, None)), 1000),
            RangeReply::Unsatisfiable
        );
        assert_eq!(
            resolve_range(Some((1000, Some(1999))), 1000),
            RangeReply::Unsatisfiable
        );
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
        let t = UNIX_EPOCH + Duration::from_hours(474_768);
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
        assert_eq!(parse_blocking_reload(""), Ok(None));
        assert_eq!(parse_blocking_reload("foo=bar"), Ok(None));
        assert_eq!(parse_blocking_reload("_HLS_msn=12"), Ok(Some((12, None))));
        assert_eq!(
            parse_blocking_reload("_HLS_msn=12&_HLS_part=3"),
            Ok(Some((12, Some(3))))
        );
        // Order doesn't matter.
        assert_eq!(
            parse_blocking_reload("_HLS_part=3&_HLS_msn=12"),
            Ok(Some((12, Some(3))))
        );
        // _HLS_part without _HLS_msn, or garbage values, are rejected.
        assert_eq!(parse_blocking_reload("_HLS_part=3"), Err(BadReloadQuery));
        assert_eq!(parse_blocking_reload("_HLS_msn=x"), Err(BadReloadQuery));
    }

    fn ring_with_parts() -> Arc<RwLock<SegmentRing>> {
        let mut ring = SegmentRing::new(6, 1);
        ring.enable_low_latency(0.5);
        for _ in 0..2 {
            ring.push_pending_part(0.5, true, bytes::Bytes::from_static(&[0x47; 188]));
            ring.push_pending_part(0.5, false, bytes::Bytes::from_static(&[0x47; 188]));
            ring.complete_pending_segment(1.0, false);
        }
        Arc::new(RwLock::new(ring))
    }

    #[tokio::test(start_paused = true)]
    async fn blocking_reload_rejects_far_future_and_times_out_with_503() {
        // Segments 0 and 1 are complete; 2 is under construction.
        let ring = ring_with_parts();
        assert_eq!(blocking_reload(&ring, "_HLS_msn=1", "t").await, None);
        assert_eq!(
            blocking_reload(&ring, "_HLS_part=0", "t").await,
            Some((400, "Bad Request"))
        );
        // last = 1 → up to msn 3 may block; 4 is too far.
        assert_eq!(
            blocking_reload(&ring, "_HLS_msn=4", "t").await,
            Some((400, "Bad Request"))
        );
        let started = tokio::time::Instant::now();
        assert_eq!(
            blocking_reload(&ring, "_HLS_msn=3", "t").await,
            Some((503, "Service Unavailable"))
        );
        // Target duration 1 s → gives up after 3 s.
        assert_eq!(started.elapsed(), Duration::from_secs(3));
    }

    #[tokio::test(start_paused = true)]
    async fn blocking_reload_rejects_parts_beyond_the_advance_part_limit() {
        // Last published part is (1, 1); PART-TARGET 0.5 s gives a
        // limit of 6 parts and two parts per segment.
        let ring = ring_with_parts();
        let started = tokio::time::Instant::now();
        assert_eq!(
            blocking_reload(&ring, "_HLS_msn=2&_HLS_part=6", "t").await,
            Some((400, "Bad Request"))
        );
        assert_eq!(
            blocking_reload(&ring, "_HLS_msn=3&_HLS_part=5", "t").await,
            Some((400, "Bad Request"))
        );
        // Rejections are immediate, not after the blocking deadline.
        assert_eq!(started.elapsed(), Duration::ZERO);
        // Exactly at the limit still blocks and then times out.
        assert_eq!(
            blocking_reload(&ring, "_HLS_msn=2&_HLS_part=5", "t").await,
            Some((503, "Service Unavailable"))
        );
        assert_eq!(started.elapsed(), Duration::from_secs(3));
    }

    #[test]
    fn part_past_the_end_of_a_completed_segment_maps_to_the_next_one() {
        let ring = ring_with_parts();
        let mut g = ring.try_write().unwrap();
        // Segment 0 has parts 0 and 1; part 2 means (1, 0), which exists.
        assert!(g.has_reached(0, Some(2)));
        // Segment 1 has parts 0 and 1; part 2 means (2, 0): not yet.
        assert!(!g.has_reached(1, Some(2)));
        g.push_pending_part(0.5, true, bytes::Bytes::from_static(&[0x47; 188]));
        assert!(g.has_reached(1, Some(2)));
    }
}

#[cfg(test)]
mod request_limit_tests {
    use super::*;

    #[tokio::test]
    async fn request_without_newline_is_rejected_at_the_limit() {
        // An endless header line: must error out instead of buffering.
        let mut input = b"GET /index.m3u8 HTTP/1.1\r\nX: ".to_vec();
        input.extend(std::iter::repeat_n(b'a', MAX_REQUEST_BYTES * 4));
        let mut reader = BufReader::new(&input[..]);
        let err = read_request(&mut reader).await.err().expect("must fail");
        assert!(err.to_string().contains("too long"), "{err}");
    }

    #[tokio::test]
    async fn normal_request_parses() {
        let input = b"GET /a/index.m3u8 HTTP/1.1\r\nRange: bytes=0-9\r\nConnection: close\r\n\r\n";
        let mut reader = BufReader::new(&input[..]);
        let req = read_request(&mut reader).await.unwrap();
        assert_eq!(req.method, "GET");
        assert_eq!(req.path, "/a/index.m3u8");
        assert!(req.connection_close);
        assert_eq!(req.range, Some((0, Some(9))));
    }
}

#[cfg(test)]
mod range_reply_tests {
    use super::*;

    #[tokio::test]
    async fn unsatisfiable_range_reply_carries_total_length() {
        let mut out = Vec::new();
        write_range_not_satisfiable(&mut out, 1234, true)
            .await
            .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("HTTP/1.1 416 Range Not Satisfiable\r\n"));
        assert!(text.contains("Content-Range: bytes */1234\r\n"));
        assert!(text.contains("Content-Length: 0\r\n"));
        assert!(text.ends_with("\r\n\r\n"));
    }
}

#[cfg(test)]
mod allowlist_tests {
    use super::*;

    fn state() -> Arc<ServerState> {
        let ring = Arc::new(RwLock::new(SegmentRing::new(6, 4)));
        Arc::new(ServerState::new(ring, None).unwrap())
    }

    async fn request_from(state: &Arc<ServerState>, peer: &str) -> String {
        let (mut client, server) = tokio::io::duplex(64 * 1024);
        let path = state.token.playlist_path();
        let task = tokio::spawn(handle(peer.parse().unwrap(), server, state.clone()));
        client
            .write_all(format!("GET {path} HTTP/1.0\r\n\r\n").as_bytes())
            .await
            .unwrap();
        let mut out = String::new();
        client.read_to_string(&mut out).await.unwrap();
        let _ = task.await;
        out
    }

    #[tokio::test]
    async fn allowlist_admits_only_allowed_peers_and_loopback() {
        let state = state();
        // No allowlist: everyone may fetch.
        assert!(
            request_from(&state, "192.168.1.50:4000")
                .await
                .starts_with("HTTP/1.1 200")
        );
        state.allow_peer("192.168.1.30".parse().unwrap());
        let denied = request_from(&state, "192.168.1.50:4000").await;
        assert!(denied.starts_with("HTTP/1.1 403"), "{denied}");
        assert!(
            denied.contains("Access-Control-Allow-Origin: *"),
            "{denied}"
        );
        assert!(
            request_from(&state, "192.168.1.30:4000")
                .await
                .starts_with("HTTP/1.1 200")
        );
        // IPv4-mapped IPv6 form of the allowed peer.
        assert!(
            request_from(&state, "[::ffff:192.168.1.30]:4000")
                .await
                .starts_with("HTTP/1.1 200")
        );
        assert!(
            request_from(&state, "127.0.0.1:4000")
                .await
                .starts_with("HTTP/1.1 200")
        );
        assert!(
            request_from(&state, "[::1]:4000")
                .await
                .starts_with("HTTP/1.1 200")
        );
    }
}

/// Lower bound for the write stall timeout (normally two target
/// durations).
const MIN_WRITE_STALL_TIMEOUT: Duration = Duration::from_secs(4);

/// `AsyncWrite` adapter that fails with `TimedOut` when the inner
/// writer makes no progress for `limit`. The timer runs only while a
/// write/flush/shutdown is pending and restarts on every bit of
/// progress, so slow-but-moving transfers are unaffected; only a peer
/// that stopped reading entirely trips it.
struct StallTimeout<W> {
    inner: W,
    limit: Duration,
    deadline: std::pin::Pin<Box<tokio::time::Sleep>>,
    armed: bool,
}

impl<W> StallTimeout<W> {
    fn new(inner: W, limit: Duration) -> Self {
        Self {
            inner,
            limit,
            deadline: Box::pin(tokio::time::sleep(limit)),
            armed: false,
        }
    }

    /// Map a poll of the inner writer through the stall timer.
    fn guard<T>(
        &mut self,
        cx: &mut std::task::Context<'_>,
        poll: std::task::Poll<std::io::Result<T>>,
    ) -> std::task::Poll<std::io::Result<T>> {
        use std::future::Future as _;
        use std::task::Poll;
        match poll {
            Poll::Ready(result) => {
                self.armed = false;
                Poll::Ready(result)
            }
            Poll::Pending => {
                if !self.armed {
                    self.armed = true;
                    let at = tokio::time::Instant::now() + self.limit;
                    self.deadline.as_mut().reset(at);
                }
                match self.deadline.as_mut().poll(cx) {
                    Poll::Ready(()) => Poll::Ready(Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "HLS client stopped reading",
                    ))),
                    Poll::Pending => Poll::Pending,
                }
            }
        }
    }
}

impl<W: AsyncWrite + Unpin> AsyncWrite for StallTimeout<W> {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        let poll = std::pin::Pin::new(&mut self.inner).poll_write(cx, buf);
        self.guard(cx, poll)
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let poll = std::pin::Pin::new(&mut self.inner).poll_flush(cx);
        self.guard(cx, poll)
    }

    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let poll = std::pin::Pin::new(&mut self.inner).poll_shutdown(cx);
        self.guard(cx, poll)
    }
}

#[cfg(test)]
mod stall_tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn writer_that_never_drains_times_out() {
        // A 1 KiB pipe nobody reads from: the second write stalls.
        let (_reader, writer) = tokio::io::duplex(1024);
        let mut w = StallTimeout::new(writer, Duration::from_secs(8));
        let started = tokio::time::Instant::now();
        let err = w.write_all(&[0u8; 4096]).await.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::TimedOut);
        assert_eq!(started.elapsed(), Duration::from_secs(8));
    }

    #[tokio::test]
    async fn draining_writer_completes() {
        let (mut reader, writer) = tokio::io::duplex(1024);
        let drain = tokio::spawn(async move {
            let mut sink = Vec::new();
            reader.read_to_end(&mut sink).await.unwrap();
            sink.len()
        });
        let mut w = StallTimeout::new(writer, Duration::from_secs(1));
        w.write_all(&vec![0u8; 64 * 1024]).await.unwrap();
        w.shutdown().await.unwrap();
        drop(w);
        assert_eq!(drain.await.unwrap(), 64 * 1024);
    }
}

#[cfg(test)]
mod adaptive_peer_tests {
    use super::*;

    #[test]
    fn adaptive_follows_the_allowed_peer_or_else_the_first_one() {
        let ring = Arc::new(RwLock::new(SegmentRing::new(6, 4)));
        let state = ServerState::new(ring, None).unwrap();
        let viewer: IpAddr = "192.168.1.50".parse().unwrap();
        let cast: IpAddr = "192.168.1.30".parse().unwrap();
        let now = Instant::now();
        state.stats.record_segment_get(viewer, 0, 1.0, now);
        state.stats.record_segment_get(cast, 0, 1.0, now);
        assert!(state.drives_adaptive(viewer));
        assert!(!state.drives_adaptive(cast));
        state.allow_peer(cast);
        assert!(state.drives_adaptive(cast));
        assert!(!state.drives_adaptive(viewer));
        assert!(!state.drives_adaptive("127.0.0.1".parse().unwrap()));
    }
}
