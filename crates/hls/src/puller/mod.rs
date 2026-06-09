//! HLS media puller. Implements [`MediaPuller`] for the receiver
//! pipeline: opens an HLS URL, demuxes the MPEG-TS segments, and
//! emits a unified [`MediaPacket`] stream the rest of the pipeline
//! decodes and plays.
//!
//! Supports the two HLS playlist shapes the wild throws at us:
//!
//! - **Master** playlist (`#EXT-X-STREAM-INF`) — picks the highest-
//!   bandwidth variant and recurses into it. No quality switching
//!   on the fly; receivers we ship at typically have one display so
//!   the top variant is the right answer, and pulling halfway
//!   through a session to renegotiate adds latency without buying
//!   the user anything.
//! - **Media** playlist (`#EXTINF` directly) — streams those
//!   segments in order. Live playlists are re-fetched every
//!   `target_duration / 2`; VOD playlists run to `#EXT-X-ENDLIST`
//!   then emit [`MediaPacket::Eos`].
//!
//! Demux happens segment-by-segment on the worker task; packets
//! cross into the [`HlsPuller`] via a tokio mpsc, so `next()` is
//! a non-blocking pop from that channel.

mod ts;

use std::collections::HashMap;
use std::time::Duration;

use bytes::Bytes;
use ferricast_core::{
    AudioStreamInfo, FerricastError, MediaInfo, MediaPacket, MediaPuller, PullSpec, Result,
    VideoStreamInfo,
};
use futures_util::StreamExt;
use m3u8_rs::Playlist;
use reqwest::Client;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use url::Url;

use ts::{DemuxedPacket, TsDemuxer};

/// Worker → puller control messages from the puller side. Seek
/// reissues into the playlist; close cancels the worker task.
enum Cmd {
    Seek(u64),
    Close,
}

pub struct HlsPuller {
    rx: Option<mpsc::Receiver<Result<MediaPacket>>>,
    cmd_tx: Option<mpsc::Sender<Cmd>>,
    task: Option<JoinHandle<()>>,
}

impl Default for HlsPuller {
    fn default() -> Self {
        Self {
            rx: None,
            cmd_tx: None,
            task: None,
        }
    }
}

impl HlsPuller {
    pub fn new() -> Self {
        Self::default()
    }
}

impl MediaPuller for HlsPuller {
    fn open(&mut self, spec: PullSpec) -> impl std::future::Future<Output = Result<MediaInfo>> + Send {
        async move {
            // Already open? Replace the running worker. Real callers
            // re-open() only after close() but we don't want a leak
            // on misuse either.
            self.close().await?;

            // 256-slot media-packet buffer (was 64). The downstream
            // consumer is the manager's pump, which decodes + pushes
            // to a sink — each slow per-frame decode used to back-
            // pressure into here and stall the segment-fetch loop,
            // letting the sender's HLS ring rotate past us. With
            // skip-to-live (below) the puller can now bail out of
            // a stale playlist instead of plodding through 404s,
            // but a larger soft buffer keeps short hiccups silent.
            let (packet_tx, packet_rx) = mpsc::channel::<Result<MediaPacket>>(256);
            let (cmd_tx, cmd_rx) = mpsc::channel::<Cmd>(4);
            let (info_tx, info_rx) = oneshot::channel::<Result<MediaInfo>>();

            let task = tokio::spawn(run_worker(spec, packet_tx, cmd_rx, info_tx));
            self.rx = Some(packet_rx);
            self.cmd_tx = Some(cmd_tx);
            self.task = Some(task);

            match info_rx.await {
                Ok(r) => r,
                Err(_) => Err(FerricastError::Pull(
                    "HLS worker exited before reporting MediaInfo".into(),
                )),
            }
        }
    }

    fn next(&mut self) -> impl std::future::Future<Output = Result<MediaPacket>> + Send {
        async move {
            let Some(rx) = self.rx.as_mut() else {
                return Err(FerricastError::Pull("HlsPuller::next before open".into()));
            };
            match rx.recv().await {
                Some(r) => r,
                None => Ok(MediaPacket::Eos),
            }
        }
    }

    fn seek(&mut self, position_us: u64) -> impl std::future::Future<Output = Result<()>> + Send {
        async move {
            let Some(tx) = self.cmd_tx.as_ref() else {
                return Err(FerricastError::Pull("HlsPuller::seek before open".into()));
            };
            tx.send(Cmd::Seek(position_us))
                .await
                .map_err(|_| FerricastError::Pull("HLS worker closed".into()))
        }
    }

    fn close(&mut self) -> impl std::future::Future<Output = Result<()>> + Send {
        async move {
            if let Some(tx) = self.cmd_tx.take() {
                let _ = tx.send(Cmd::Close).await;
            }
            if let Some(task) = self.task.take() {
                let _ = task.await;
            }
            self.rx = None;
            Ok(())
        }
    }
}

/// Top-level worker entry. Resolves master → media, sends `MediaInfo`
/// back to the caller via `info_tx`, then streams segments through
/// `packet_tx` until cancelled or EOS.
async fn run_worker(
    spec: PullSpec,
    packet_tx: mpsc::Sender<Result<MediaPacket>>,
    mut cmd_rx: mpsc::Receiver<Cmd>,
    info_tx: oneshot::Sender<Result<MediaInfo>>,
) {
    let client = match build_client(&spec.headers) {
        Ok(c) => c,
        Err(e) => {
            let _ = info_tx.send(Err(e));
            return;
        }
    };

    let media_url = match resolve_to_media_playlist(&client, &spec.url).await {
        Ok(u) => u,
        Err(e) => {
            let _ = info_tx.send(Err(e));
            return;
        }
    };

    // Fetch media playlist once up front so we can probe codecs.
    let media_pl_bytes = match http_get(&client, media_url.as_str()).await {
        Ok(b) => b,
        Err(e) => {
            let _ = info_tx.send(Err(e));
            return;
        }
    };
    let pl = match m3u8_rs::parse_media_playlist_res(&media_pl_bytes) {
        Ok(p) => p,
        Err(e) => {
            let _ = info_tx.send(Err(FerricastError::Pull(format!(
                "media playlist parse: {e}"
            ))));
            return;
        }
    };

    // Probe the first segment to discover codecs + audio params. We
    // need this before reporting `MediaInfo` so the manager can pick
    // decoders. The probed bytes are also pushed downstream — no
    // double fetch.
    let first_segment_url = match pl.segments.first() {
        Some(s) => match media_url.join(&s.uri) {
            Ok(u) => u,
            Err(e) => {
                let _ = info_tx.send(Err(FerricastError::Pull(format!(
                    "segment URL join: {e}"
                ))));
                return;
            }
        },
        None => {
            let _ = info_tx.send(Err(FerricastError::Pull(
                "media playlist had no segments".into(),
            )));
            return;
        }
    };

    let first_bytes = match http_get(&client, first_segment_url.as_str()).await {
        Ok(b) => b,
        Err(e) => {
            let _ = info_tx.send(Err(e));
            return;
        }
    };
    let mut demuxer = TsDemuxer::new();
    let probe_packets = std::mem::take(demuxer.push(&first_bytes));
    // First-segment flush — same reasoning as the segment-loop flush
    // below: independently decodable segments don't trail their last
    // PES with another PUSI.
    let probe_trailing = std::mem::take(demuxer.flush());
    let info = demuxer.info();
    let media_info = MediaInfo {
        video: info.video_codec.map(|c| VideoStreamInfo {
            codec: c,
            // Width/height aren't carried in PMT; the puller leaves
            // them at zero and the decoder discovers real
            // dimensions from the bitstream's SPS. That's fine for
            // every consumer in the workspace — they call
            // `frame.width()` post-decode rather than trusting the
            // pre-decoded hint.
            width: 0,
            height: 0,
            fps: 0,
        }),
        audio: info.audio_codec.map(|c| AudioStreamInfo {
            codec: c,
            sample_rate: info.audio_sample_rate,
            channels: info.audio_channels,
        }),
        duration_us: total_duration_us(&pl),
        is_live: pl.end_list == false,
    };
    if info_tx.send(Ok(media_info)).is_err() {
        // Caller cancelled before getting info; nothing to do.
        return;
    }

    // Hand over the first-segment packets we already demuxed.
    for p in probe_packets.into_iter().chain(probe_trailing) {
        let packet = match p {
            DemuxedPacket::Video(v) => MediaPacket::Video(v),
            DemuxedPacket::Audio(a) => MediaPacket::Audio(a),
        };
        if packet_tx.send(Ok(packet)).await.is_err() {
            return;
        }
    }

    // Steady-state segment loop. The cursor (`next_index`) advances
    // after each segment; on live we re-fetch the playlist when we
    // run off the end of the current snapshot, on VOD we send Eos.
    let mut current_pl = pl;
    let current_base = media_url.clone();
    let mut next_index = 1usize;
    /// If we fall this many segments behind the live edge, skip
    /// forward instead of plodding through expired segments. The
    /// sender's `keep_segments` is typically 12 (see
    /// `ferricast_hls::HlsConfig`); 3 leaves enough margin that
    /// we're still inside the ring but never wading through history.
    const LIVE_EDGE_LAG_THRESHOLD: usize = 3;

    loop {
        // Drain any cmd that arrived between iterations.
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                Cmd::Close => return,
                Cmd::Seek(pos_us) => {
                    if let Some(idx) = segment_index_for_position(&current_pl, pos_us) {
                        next_index = idx;
                        demuxer = TsDemuxer::new();
                    }
                }
            }
        }

        let Some(seg) = current_pl.segments.get(next_index) else {
            // Out of segments. Live → refresh playlist; VOD → EOS.
            if current_pl.end_list {
                let _ = packet_tx.send(Ok(MediaPacket::Eos)).await;
                return;
            }
            // Live refresh. Wait target_duration / 2 to align with
            // the standard polling cadence (RFC 8216 §6.3.4).
            let pause = Duration::from_secs_f32((current_pl.target_duration as f32 / 2.0).max(0.5));
            tokio::select! {
                _ = tokio::time::sleep(pause) => {}
                Some(cmd) = cmd_rx.recv() => match cmd {
                    Cmd::Close => return,
                    Cmd::Seek(_) => {}, // handled on next loop tick
                }
            }
            match http_get(&client, current_base.as_str()).await {
                Ok(bytes) => match m3u8_rs::parse_media_playlist_res(&bytes) {
                    Ok(new_pl) => {
                        // Advance `next_index` past whatever we
                        // already played. media_sequence on the new
                        // playlist gives us the absolute index of
                        // its first segment; our `next_index` was
                        // relative to the old snapshot's first
                        // segment, so translate.
                        let old_first =
                            current_pl.media_sequence as i64 + next_index as i64;
                        let new_first = new_pl.media_sequence as i64;
                        next_index = (old_first - new_first).max(0) as usize;
                        // Skip-to-live: if a decoder hiccup pushed us
                        // multiple segments behind, jump to the live
                        // edge instead of plodding through everything
                        // we missed. Without this, a single slow
                        // window-startup spike snowballs into the
                        // sender's ring rotating past us and us
                        // chasing 404s until reconnect.
                        let pending = new_pl.segments.len().saturating_sub(next_index);
                        if pending > LIVE_EDGE_LAG_THRESHOLD {
                            let skip_to =
                                new_pl.segments.len().saturating_sub(LIVE_EDGE_LAG_THRESHOLD);
                            tracing::warn!(
                                from_index = next_index,
                                to_index = skip_to,
                                pending,
                                "HLS puller fell behind live edge — skipping forward"
                            );
                            next_index = skip_to;
                            // Drop demuxer state because we're about
                            // to discontinuity-jump; the demuxer
                            // resyncs on the next PAT/PMT.
                            demuxer = TsDemuxer::new();
                        }
                        current_pl = new_pl;
                    }
                    Err(e) => {
                        let _ = packet_tx
                            .send(Err(FerricastError::Pull(format!(
                                "playlist refresh parse: {e}"
                            ))))
                            .await;
                        return;
                    }
                },
                Err(e) => {
                    let _ = packet_tx.send(Err(e)).await;
                    return;
                }
            }
            continue;
        };

        let seg_url = match current_base.join(&seg.uri) {
            Ok(u) => u,
            Err(e) => {
                let _ = packet_tx
                    .send(Err(FerricastError::Pull(format!("segment URL join: {e}"))))
                    .await;
                return;
            }
        };

        let bytes = match http_get_segment(&client, seg_url.as_str()).await {
            SegmentFetch::Ok(b) => b,
            SegmentFetch::NotFound => {
                // Segment evicted from the sender's ring before we
                // asked for it — almost certainly because the decode
                // pump fell behind. Refresh the playlist, skip to
                // live, and resync the demuxer. Non-fatal: live HLS
                // is allowed to lose history; we just rejoin at the
                // current edge.
                tracing::warn!(
                    seq = next_index,
                    url = %seg_url,
                    "HLS segment 404 — segment evicted; refreshing playlist + skipping to live"
                );
                match http_get(&client, current_base.as_str()).await {
                    Ok(refreshed) => match m3u8_rs::parse_media_playlist_res(&refreshed) {
                        Ok(new_pl) => {
                            let skip_to = new_pl
                                .segments
                                .len()
                                .saturating_sub(LIVE_EDGE_LAG_THRESHOLD);
                            next_index = skip_to;
                            current_pl = new_pl;
                            demuxer = TsDemuxer::new();
                            continue;
                        }
                        Err(e) => {
                            let _ = packet_tx
                                .send(Err(FerricastError::Pull(format!(
                                    "playlist refresh after 404: {e}"
                                ))))
                                .await;
                            return;
                        }
                    },
                    Err(e) => {
                        let _ = packet_tx.send(Err(e)).await;
                        return;
                    }
                }
            }
            SegmentFetch::Err(e) => {
                let _ = packet_tx.send(Err(e)).await;
                return;
            }
        };

        // Push everything the segment produced, then flush any
        // trailing PES the demuxer was still assembling. HLS
        // segments are guaranteed to be independently decodable
        // (RFC 8216 §3.3) so the last PES in each segment ends
        // without a follower PUSI to trigger the natural emit —
        // explicit flush at segment boundary is the right place.
        for p in std::mem::take(demuxer.push(&bytes)) {
            let packet = match p {
                DemuxedPacket::Video(v) => MediaPacket::Video(v),
                DemuxedPacket::Audio(a) => MediaPacket::Audio(a),
            };
            if packet_tx.send(Ok(packet)).await.is_err() {
                return;
            }
        }
        for p in std::mem::take(demuxer.flush()) {
            let packet = match p {
                DemuxedPacket::Video(v) => MediaPacket::Video(v),
                DemuxedPacket::Audio(a) => MediaPacket::Audio(a),
            };
            if packet_tx.send(Ok(packet)).await.is_err() {
                return;
            }
        }

        next_index += 1;
    }
}

fn build_client(headers: &HashMap<String, String>) -> Result<Client> {
    let mut header_map = reqwest::header::HeaderMap::new();
    for (k, v) in headers {
        let name = reqwest::header::HeaderName::from_bytes(k.as_bytes())
            .map_err(|e| FerricastError::Pull(format!("invalid header name {k}: {e}")))?;
        let value = reqwest::header::HeaderValue::from_str(v)
            .map_err(|e| FerricastError::Pull(format!("invalid header value: {e}")))?;
        header_map.insert(name, value);
    }
    Client::builder()
        .default_headers(header_map)
        // HLS senders typically time out playlist reads at 10s; we
        // give a little more headroom because rustls handshake adds
        // a few hundred ms on first connect.
        .connect_timeout(Duration::from_secs(5))
        // Whole-request timeout. Long enough for a single 10s
        // segment to download even on a 1 Mbps link.
        .timeout(Duration::from_secs(30))
        // Accept self-signed TLS certs. The HLS sender in this
        // workspace (`ferricast_hls::HlsFrameSink` with
        // `HlsConfig::tls`) uses a per-session `rcgen` cert that
        // is intentionally unsigned by any CA — the receiver isn't
        // expected to validate identity over an ad-hoc cast link.
        // Public HLS origins are usually CA-signed so this only
        // takes effect when we'd otherwise reject ourselves; we
        // keep the rustls TLS handshake itself (so the data is
        // still encrypted on the wire).
        .danger_accept_invalid_certs(true)
        .build()
        .map_err(|e| FerricastError::Pull(format!("reqwest client: {e}")))
}

/// Result of fetching a media segment. The 404 case is split out
/// because it's recoverable — segments expire from the sender's ring
/// in steady-state operation and the puller can skip-to-live instead
/// of failing the session.
enum SegmentFetch {
    Ok(Bytes),
    NotFound,
    Err(FerricastError),
}

async fn http_get_segment(client: &Client, url: &str) -> SegmentFetch {
    let resp = match client.get(url).send().await {
        Ok(r) => r,
        Err(e) => return SegmentFetch::Err(FerricastError::Pull(format!("GET {url}: {e}"))),
    };
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return SegmentFetch::NotFound;
    }
    if !resp.status().is_success() {
        return SegmentFetch::Err(FerricastError::Pull(format!(
            "GET {url} -> {}",
            resp.status()
        )));
    }
    let mut stream = resp.bytes_stream();
    let mut buf = bytes::BytesMut::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(c) => buf.extend_from_slice(&c),
            Err(e) => {
                return SegmentFetch::Err(FerricastError::Pull(format!("body chunk: {e}")))
            }
        }
    }
    SegmentFetch::Ok(buf.freeze())
}

async fn http_get(client: &Client, url: &str) -> Result<Bytes> {
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| FerricastError::Pull(format!("GET {url}: {e}")))?;
    if !resp.status().is_success() {
        return Err(FerricastError::Pull(format!(
            "GET {url} -> {}",
            resp.status()
        )));
    }
    // `bytes_stream` chunks the body so a slow connection doesn't
    // block the worker — we still concatenate into one Bytes for
    // the demuxer (segments are bounded and small).
    let mut stream = resp.bytes_stream();
    let mut buf = bytes::BytesMut::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| FerricastError::Pull(format!("body chunk: {e}")))?;
        buf.extend_from_slice(&chunk);
    }
    Ok(buf.freeze())
}

/// Fetch + classify a URL. If it's a master playlist, pick the
/// highest-bandwidth variant and return its absolute URL; if it's
/// already a media playlist, return the input URL unchanged.
async fn resolve_to_media_playlist(client: &Client, url: &str) -> Result<Url> {
    let parsed = Url::parse(url).map_err(|e| FerricastError::Pull(format!("URL parse: {e}")))?;
    let bytes = http_get(client, url).await?;
    match m3u8_rs::parse_playlist_res(&bytes) {
        Ok(Playlist::MasterPlaylist(master)) => {
            // Highest bandwidth wins. Ties broken by stream order.
            let pick = master
                .variants
                .iter()
                .max_by_key(|v| v.bandwidth)
                .ok_or_else(|| FerricastError::Pull("master playlist had no variants".into()))?;
            let variant_url = parsed
                .join(&pick.uri)
                .map_err(|e| FerricastError::Pull(format!("variant URL join: {e}")))?;
            tracing::info!(
                variant_uri = %pick.uri,
                bandwidth = pick.bandwidth,
                "HLS master playlist resolved to variant"
            );
            Ok(variant_url)
        }
        Ok(Playlist::MediaPlaylist(_)) => Ok(parsed),
        Err(e) => Err(FerricastError::Pull(format!("playlist parse: {e}"))),
    }
}

fn total_duration_us(pl: &m3u8_rs::MediaPlaylist) -> Option<u64> {
    if !pl.end_list {
        return None;
    }
    let secs: f64 = pl.segments.iter().map(|s| s.duration as f64).sum();
    Some((secs * 1_000_000.0) as u64)
}

fn segment_index_for_position(pl: &m3u8_rs::MediaPlaylist, pos_us: u64) -> Option<usize> {
    let mut acc_us: u64 = 0;
    for (idx, seg) in pl.segments.iter().enumerate() {
        let dur_us = (seg.duration as f64 * 1_000_000.0) as u64;
        if pos_us < acc_us + dur_us {
            return Some(idx);
        }
        acc_us += dur_us;
    }
    None
}

