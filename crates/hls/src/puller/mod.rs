//! HLS media puller. Implements [`MediaPuller`] for the receiver
//! pipeline: opens an HLS URL, demuxes the MPEG-TS segments, and
//! emits a unified [`MediaPacket`] stream the rest of the pipeline
//! decodes and plays.
//!
//! Supports the two HLS playlist shapes the wild throws at us:
//!
//! - **Master** playlist (`#EXT-X-STREAM-INF`) — picks one variant
//!   (decodable codecs first, then muxed audio, then the highest
//!   bandwidth) and follows it. No quality switching on the fly.
//! - **Media** playlist (`#EXTINF` directly) — streams those
//!   segments in order. Live playlists start three target durations
//!   from the end (RFC 8216 §6.3.3) and are reloaded at the RFC
//!   cadence; VOD playlists run to `#EXT-X-ENDLIST` then emit
//!   [`MediaPacket::Eos`].
//!
//! Handled: `EXT-X-BYTERANGE` (HTTP range requests), `EXT-X-KEY`
//! `METHOD=AES-128` (media and initialization segments),
//! `EXT-X-DISCONTINUITY`, live sources that restart
//! (media sequence going backwards), transient HTTP failures (retried
//! with backoff) and expired live segments (skip to the live edge).
//! Fragmented MP4 segments (`EXT-X-MAP` initialization segment plus
//! CMAF media segments) go through the MP4 demuxer. Rejected with an
//! explicit error: `SAMPLE-AES` and DRM key formats. Alternate audio renditions in a
//! separate playlist are not fetched; such variants play video only.
//!
//! Demux happens segment-by-segment on the worker task; packets
//! cross into the [`HlsPuller`] via a tokio mpsc, so `next()` is a
//! pop from that channel.

mod fetch;
mod playlist;
mod ts;

use std::time::{Duration, Instant};

use ferricast_core::{
    AudioStreamInfo, FerricastError, MediaInfo, MediaPacket, MediaPuller, PullSpec, Result,
    VideoStreamInfo,
};
use m3u8_rs::{MediaPlaylist, Playlist};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use url::Url;

use bytes::Bytes;
use fetch::{
    ByteRange, FetchError, Fetched, Fetcher, INIT_LIMIT, KEY_LIMIT, PLAYLIST_LIMIT, SEGMENT_LIMIT,
};
use playlist::{InitSegment, MapKeyOrder, SegmentKey, SegmentRef};
use ts::{DemuxedPacket, TsDemuxer};

/// Puller → worker control messages.
enum Cmd {
    Seek { position_us: u64, generation: u64 },
    Close,
}

/// A packet tagged with the seek generation it belongs to, so packets
/// queued before a seek can be discarded by [`HlsPuller::next`].
type Tagged = (u64, Result<MediaPacket>);

/// How long [`HlsPuller::close`] waits for the worker before aborting
/// it. Past that point the worker can only be blocked on network I/O,
/// which the abort cancels cleanly.
const WORKER_JOIN_TIMEOUT: Duration = Duration::from_secs(2);
/// Media-packet buffer between worker and consumer. Absorbs short
/// consumer stalls (window startup) without stalling segment fetches.
const PACKET_CHANNEL_DEPTH: usize = 256;
/// Consecutive missing segments tolerated before giving up.
const MAX_CONSECUTIVE_NOT_FOUND: u32 = 5;
/// A live client this many segments behind its start window skips
/// forward to the live edge instead of downloading expired history.
const LIVE_LAG_SEGMENTS: u64 = 3;
/// Retry window for requests made before the target duration is
/// known (master / first media playlist, first segment).
const OPEN_RETRY_BUDGET: Duration = Duration::from_secs(10);
const RETRY_BACKOFF_START: Duration = Duration::from_millis(250);
const RETRY_BACKOFF_MAX: Duration = Duration::from_secs(4);

#[derive(Default)]
pub struct HlsPuller {
    rx: Option<mpsc::Receiver<Tagged>>,
    cmd_tx: Option<mpsc::Sender<Cmd>>,
    task: Option<JoinHandle<()>>,
    /// Bumped on every seek; packets of older generations are stale.
    generation: u64,
    is_live: bool,
}

impl HlsPuller {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl MediaPuller for HlsPuller {
    async fn open(&mut self, spec: PullSpec) -> Result<MediaInfo> {
        // Already open? Replace the running worker.
        self.close().await?;

        let (packet_tx, packet_rx) = mpsc::channel::<Tagged>(PACKET_CHANNEL_DEPTH);
        let (cmd_tx, cmd_rx) = mpsc::channel::<Cmd>(4);
        let (info_tx, info_rx) = oneshot::channel::<Result<MediaInfo>>();

        self.generation = 0;
        self.task = Some(tokio::spawn(run_worker(spec, packet_tx, cmd_rx, info_tx)));
        self.rx = Some(packet_rx);
        self.cmd_tx = Some(cmd_tx);

        let info = info_rx.await.map_err(|_| {
            FerricastError::Pull("HLS worker exited before reporting MediaInfo".into())
        })??;
        self.is_live = info.is_live;
        Ok(info)
    }

    async fn next(&mut self) -> Result<MediaPacket> {
        let Some(rx) = self.rx.as_mut() else {
            return Err(FerricastError::Pull("HlsPuller::next before open".into()));
        };
        loop {
            match rx.recv().await {
                Some((generation, packet)) if generation == self.generation => return packet,
                // Queued before the latest seek.
                Some(_) => {}
                None => return Ok(MediaPacket::Eos),
            }
        }
    }

    async fn seek(&mut self, position_us: u64) -> Result<()> {
        let Some(tx) = self.cmd_tx.as_ref() else {
            return Err(FerricastError::Pull("HlsPuller::seek before open".into()));
        };
        if self.is_live {
            return Err(FerricastError::Pull(
                "cannot seek in a live HLS stream".into(),
            ));
        }
        let generation = self.generation + 1;
        tx.send(Cmd::Seek {
            position_us,
            generation,
        })
        .await
        .map_err(|_| FerricastError::Pull("HLS worker closed".into()))?;
        self.generation = generation;
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        // Drop the packet receiver first: a worker parked in
        // `packet_tx.send().await` on a full channel then wakes with a
        // send error and exits instead of waiting forever for a
        // consumer that is busy joining it.
        self.rx = None;
        if let Some(tx) = self.cmd_tx.take() {
            // `try_send`: the command channel may be full too, and the
            // dropped receiver already guarantees an exit.
            let _ = tx.try_send(Cmd::Close);
        }
        if let Some(mut task) = self.task.take()
            && tokio::time::timeout(WORKER_JOIN_TIMEOUT, &mut task)
                .await
                .is_err()
        {
            tracing::warn!("HLS worker did not stop in time; aborting it");
            task.abort();
        }
        Ok(())
    }
}

/// Why the worker stopped what it was doing.
enum Stop {
    /// The puller closed (or dropped its channels).
    Closed,
    Seek {
        position_us: u64,
        generation: u64,
    },
    /// VOD playlist fully played.
    Ended,
    Failed(FerricastError),
}

type Step<T> = std::result::Result<T, Stop>;

/// The worker's channels plus the HTTP client: every potentially
/// long wait goes through [`Io::race`] so a close or seek interrupts
/// it immediately.
struct Io {
    fetcher: Fetcher,
    cmd_rx: mpsc::Receiver<Cmd>,
    packet_tx: mpsc::Sender<Tagged>,
    generation: u64,
}

impl Io {
    async fn race<T>(&mut self, fut: impl Future<Output = T>) -> Step<T> {
        tokio::select! {
            biased;
            cmd = self.cmd_rx.recv() => Err(match cmd {
                Some(Cmd::Seek { position_us, generation }) => Stop::Seek { position_us, generation },
                Some(Cmd::Close) | None => Stop::Closed,
            }),
            out = fut => Ok(out),
        }
    }

    async fn emit(&mut self, packet: Result<MediaPacket>) -> Step<()> {
        let tx = self.packet_tx.clone();
        let tagged = (self.generation, packet);
        self.race(tx.send(tagged)).await?.map_err(|_| Stop::Closed)
    }

    /// GET with retries on transient failures (backoff, bounded by
    /// `budget`). 404 and permanent errors return immediately.
    async fn fetch(
        &mut self,
        url: &Url,
        range: Option<ByteRange>,
        limit: usize,
        budget: Duration,
    ) -> Step<std::result::Result<Fetched, FetchError>> {
        let deadline = Instant::now() + budget;
        let mut backoff = RETRY_BACKOFF_START;
        loop {
            let fetcher = &self.fetcher;
            let attempt = tokio::select! {
                biased;
                cmd = self.cmd_rx.recv() => return Err(match cmd {
                    Some(Cmd::Seek { position_us, generation }) => Stop::Seek { position_us, generation },
                    Some(Cmd::Close) | None => Stop::Closed,
                }),
                r = fetcher.get(url, range, limit) => r,
            };
            match attempt {
                Err(FetchError::Transient(msg)) if Instant::now() + backoff < deadline => {
                    tracing::warn!(%url, %msg, retry_in = ?backoff, "HLS fetch failed; retrying");
                    self.race(tokio::time::sleep(backoff)).await?;
                    backoff = (backoff * 2).min(RETRY_BACKOFF_MAX);
                }
                other => return Ok(other),
            }
        }
    }

    /// [`Self::fetch`] where any failure ends the session.
    async fn fetch_required(&mut self, url: &Url, limit: usize, budget: Duration) -> Step<Fetched> {
        self.fetch(url, None, limit, budget)
            .await?
            .map_err(|e| Stop::Failed(e.into_pull_error(url)))
    }
}

/// Parses a media playlist, with the `EXT-X-KEY` / `EXT-X-MAP` order
/// the parser does not keep.
fn parse_media(body: &[u8]) -> Step<(MediaPlaylist, MapKeyOrder)> {
    let playlist = m3u8_rs::parse_media_playlist_res(body)
        .map_err(|e| Stop::Failed(FerricastError::Pull(format!("media playlist parse: {e}"))))?;
    Ok((playlist, MapKeyOrder::scan(body)))
}

/// Playback state of one HLS media playlist.
struct Worker {
    io: Io,
    /// Re-requested on every live reload.
    playlist_url: Url,
    /// Final URL of the last playlist response; segment and key URIs
    /// resolve against it.
    playlist_base: Url,
    playlist: MediaPlaylist,
    /// Which key encrypts each `EXT-X-MAP` of `playlist`.
    map_key_order: MapKeyOrder,
    /// Absolute media sequence number of the next segment to fetch.
    next_seq: u64,
    demuxer: TsDemuxer,
    /// Last AES-128 key fetched, by URI.
    key_cache: Option<(Url, [u8; 16])>,
    not_found_streak: u32,
    /// Whether the last live reload brought new segments; picks the
    /// reload interval (RFC 8216 §6.3.4).
    last_reload_changed: bool,
    /// The next segment starts a new timeline (seek, skip to live,
    /// restarted source).
    reset_pending: bool,
    /// Something was emitted already; a timeline reset must then be
    /// signalled downstream.
    timeline_started: bool,
    /// After a seek: how far into the next segment the target lies.
    /// That much of the segment is decode-only.
    seek_offset_us: Option<u64>,
    /// Last `EXT-X-MAP` initialization segment fetched.
    init_cache: Option<(InitSegment, Bytes)>,
    /// Streams of the last fragmented MP4 segment (MPEG-TS streams are
    /// tracked by the TS demuxer).
    fmp4_streams: Option<(Option<VideoStreamInfo>, Option<AudioStreamInfo>)>,
}

/// Top-level worker entry: resolves the playlist, probes the first
/// segment, reports `MediaInfo` through `info_tx`, then streams
/// segments until closed, ended or failed.
async fn run_worker(
    spec: PullSpec,
    packet_tx: mpsc::Sender<Tagged>,
    cmd_rx: mpsc::Receiver<Cmd>,
    info_tx: oneshot::Sender<Result<MediaInfo>>,
) {
    let fetcher = match Fetcher::new(&spec.headers) {
        Ok(f) => f,
        Err(e) => {
            let _ = info_tx.send(Err(e));
            return;
        }
    };
    let io = Io {
        fetcher,
        cmd_rx,
        packet_tx,
        generation: 0,
    };
    let (worker, info, probe) = match Worker::open(io, &spec.url).await {
        Ok(opened) => opened,
        Err(Stop::Failed(e)) => {
            let _ = info_tx.send(Err(e));
            return;
        }
        // No seek can arrive before `open` returns; a close ends here.
        Err(Stop::Closed | Stop::Seek { .. } | Stop::Ended) => return,
    };
    if info_tx.send(Ok(info)).is_err() {
        // Caller gave up before getting the info.
        return;
    }
    worker.run(probe).await;
}

impl Worker {
    /// Resolves `url` to a media playlist and demuxes the first
    /// segment to learn the codecs.
    async fn open(mut io: Io, url: &str) -> Step<(Self, MediaInfo, Vec<MediaPacket>)> {
        let request_url = Url::parse(url)
            .map_err(|e| Stop::Failed(FerricastError::Pull(format!("URL parse {url}: {e}"))))?;
        let fetched = io
            .fetch_required(&request_url, PLAYLIST_LIMIT, OPEN_RETRY_BUDGET)
            .await?;
        let (playlist_url, fetched) = match m3u8_rs::parse_playlist_res(&fetched.body) {
            Ok(Playlist::MasterPlaylist(master)) => {
                let (variant, external_audio) =
                    playlist::choose_variant(&master).ok_or_else(|| {
                        Stop::Failed(FerricastError::Pull(
                            "master playlist had no variants".into(),
                        ))
                    })?;
                let variant_url = fetched.url.join(&variant.uri).map_err(|e| {
                    Stop::Failed(FerricastError::Pull(format!("variant URL join: {e}")))
                })?;
                tracing::info!(
                    variant_uri = %variant.uri,
                    bandwidth = variant.bandwidth,
                    codecs = ?variant.codecs,
                    "HLS master playlist resolved to variant"
                );
                if external_audio {
                    tracing::warn!(
                        "HLS variant carries audio in a separate rendition playlist, which is \
                         not supported — playing video only"
                    );
                }
                let media = io
                    .fetch_required(&variant_url, PLAYLIST_LIMIT, OPEN_RETRY_BUDGET)
                    .await?;
                (variant_url, media)
            }
            Ok(Playlist::MediaPlaylist(_)) => (request_url, fetched),
            Err(e) => {
                return Err(Stop::Failed(FerricastError::Pull(format!(
                    "playlist parse: {e}"
                ))));
            }
        };
        let (playlist, map_key_order) = parse_media(&fetched.body)?;
        playlist::check_supported(&playlist).map_err(Stop::Failed)?;
        if playlist.segments.is_empty() {
            return Err(Stop::Failed(FerricastError::Pull(
                "media playlist had no segments".into(),
            )));
        }
        let start = if playlist.end_list {
            0
        } else {
            playlist::live_start_index(&playlist)
        };

        let mut worker = Self {
            io,
            playlist_url,
            playlist_base: fetched.url,
            next_seq: playlist.media_sequence + start as u64,
            playlist,
            map_key_order,
            demuxer: TsDemuxer::new(),
            key_cache: None,
            not_found_streak: 0,
            last_reload_changed: false,
            reset_pending: false,
            timeline_started: false,
            seek_offset_us: None,
            init_cache: None,
            fmp4_streams: None,
        };

        // Probe the first segment: codecs and audio parameters are
        // only known once its PMT and first ADTS header are parsed.
        // The probed packets are emitted afterwards, not re-fetched.
        let probe = worker.next_packets().await.map_err(|stop| match stop {
            Stop::Ended => Stop::Failed(FerricastError::Pull("HLS playlist has no media".into())),
            other => other,
        })?;
        let (video, audio) = match worker.fmp4_streams.clone() {
            Some(streams) => streams,
            None => {
                let ts_info = worker.demuxer.info();
                (
                    ts_info.video_codec.map(|codec| VideoStreamInfo {
                        codec,
                        // Not carried in the PMT; decoders read the real
                        // dimensions from the SPS.
                        width: 0,
                        height: 0,
                        fps: 0,
                    }),
                    ts_info.audio_codec.map(|codec| AudioStreamInfo {
                        codec,
                        sample_rate: ts_info.audio_sample_rate,
                        channels: ts_info.audio_channels,
                        source: None,
                    }),
                )
            }
        };
        if video.is_none() && audio.is_none() {
            return Err(Stop::Failed(FerricastError::Pull(
                "no supported elementary stream in the first HLS segment".into(),
            )));
        }
        let info = MediaInfo {
            video,
            audio,
            duration_us: playlist::total_duration_us(&worker.playlist),
            is_live: !worker.playlist.end_list,
        };
        Ok((worker, info, probe))
    }

    async fn run(mut self, probe: Vec<MediaPacket>) {
        let mut pending = probe;
        loop {
            let step = match self.emit_all(std::mem::take(&mut pending)).await {
                Ok(()) => self.next_packets().await.map(|p| pending = p),
                Err(stop) => Err(stop),
            };
            let Err(mut stop) = step else { continue };
            loop {
                match stop {
                    Stop::Seek {
                        position_us,
                        generation,
                    } => {
                        self.io.generation = generation;
                        self.seek(position_us);
                        break;
                    }
                    Stop::Closed => return,
                    Stop::Ended => {
                        // Stay alive after EOS: the consumer may still
                        // seek back into the VOD before it reads the end.
                        stop = match self.io.emit(Ok(MediaPacket::Eos)).await {
                            Ok(()) => match self.io.race(std::future::pending::<()>()).await {
                                Err(next) => next,
                                Ok(()) => Stop::Closed,
                            },
                            Err(next) => next,
                        };
                    }
                    Stop::Failed(e) => {
                        tracing::warn!(%e, "HLS puller failed");
                        let _ = self.io.emit(Err(e)).await;
                        return;
                    }
                }
            }
        }
    }

    async fn emit_all(&mut self, packets: Vec<MediaPacket>) -> Step<()> {
        for packet in packets {
            self.io.emit(Ok(packet)).await?;
        }
        Ok(())
    }

    fn seek(&mut self, position_us: u64) {
        match playlist::segment_for_position(&self.playlist, position_us) {
            Some((idx, segment_start_us)) => {
                tracing::info!(position_us, segment_start_us, "HLS seek");
                self.next_seq = self.playlist.media_sequence + idx as u64;
                self.seek_offset_us = Some(position_us.saturating_sub(segment_start_us));
            }
            // Past the end: play nothing more.
            None => {
                self.next_seq = self.playlist.media_sequence + self.playlist.segments.len() as u64;
            }
        }
        self.reset_pending = true;
    }

    fn retry_budget(&self) -> Duration {
        Duration::from_secs(3 * self.playlist.target_duration)
            .clamp(Duration::from_secs(6), Duration::from_secs(30))
    }

    fn jump_to_live_edge(&mut self) {
        self.next_seq =
            self.playlist.media_sequence + playlist::live_start_index(&self.playlist) as u64;
        self.reset_pending = true;
    }

    /// Fetches and demuxes the next segment, reloading the playlist
    /// as needed, and returns the packets it produced.
    async fn next_packets(&mut self) -> Step<Vec<MediaPacket>> {
        loop {
            let first = self.playlist.media_sequence;
            let Some(idx) = self.next_seq.checked_sub(first) else {
                // The segment we wanted expired from the live window.
                tracing::warn!(
                    wanted = self.next_seq,
                    first,
                    "HLS segment left the live window — skipping to the live edge"
                );
                self.jump_to_live_edge();
                continue;
            };
            let idx = usize::try_from(idx).unwrap_or(usize::MAX);
            if idx >= self.playlist.segments.len() {
                if self.playlist.end_list {
                    return Err(Stop::Ended);
                }
                self.wait_and_reload().await?;
                continue;
            }

            let seg = playlist::resolve_segment(
                &self.playlist,
                &self.map_key_order,
                &self.playlist_base,
                idx,
            )
            .map_err(Stop::Failed)?;
            let budget = self.retry_budget();
            match self
                .io
                .fetch(&seg.url, seg.range, SEGMENT_LIMIT, budget)
                .await?
            {
                Ok(fetched) => {
                    self.not_found_streak = 0;
                    let out = self.demux_segment(&seg, &fetched).await?;
                    self.next_seq += 1;
                    if !out.is_empty() {
                        self.timeline_started = true;
                    }
                    return Ok(out);
                }
                Err(FetchError::NotFound) => {
                    self.not_found_streak += 1;
                    if self.not_found_streak > MAX_CONSECUTIVE_NOT_FOUND {
                        return Err(Stop::Failed(FerricastError::Pull(format!(
                            "{} consecutive HLS segments not found (last {})",
                            self.not_found_streak, seg.url
                        ))));
                    }
                    if self.playlist.end_list {
                        tracing::warn!(url = %seg.url, "HLS VOD segment missing — skipping it");
                        self.next_seq += 1;
                        self.reset_pending = true;
                    } else {
                        // Evicted before we asked for it. Reload; the
                        // window check above then moves us forward.
                        tracing::warn!(url = %seg.url, "HLS live segment 404 — reloading playlist");
                        self.reload().await?;
                    }
                }
                Err(e) => return Err(Stop::Failed(e.into_pull_error(&seg.url))),
            }
        }
    }

    async fn demux_segment(
        &mut self,
        seg: &SegmentRef,
        fetched: &Fetched,
    ) -> Step<Vec<MediaPacket>> {
        let decrypted;
        let body: &[u8] = match &seg.key {
            Some(key) => {
                let key_bytes = self.key_bytes(key).await?;
                decrypted = playlist::decrypt_aes128(&key_bytes, &key.iv, &fetched.body)
                    .map_err(Stop::Failed)?;
                &decrypted
            }
            None => &fetched.body,
        };

        let mut out = Vec::new();
        if self.reset_pending || seg.discontinuity {
            self.reset_pending = false;
            // A fresh demuxer drops the timestamp unwrap state and any
            // partial PES; every HLS segment starts with PAT/PMT.
            self.demuxer = TsDemuxer::new();
            if self.timeline_started {
                out.push(MediaPacket::Discontinuity);
            }
        }
        if let Some(map) = &seg.map {
            let init = self.init_segment(map).await?;
            out.extend(self.demux_fragmented(&init, body).map_err(Stop::Failed)?);
        } else {
            let demuxed = std::mem::take(self.demuxer.push(body))
                .into_iter()
                .chain(std::mem::take(self.demuxer.flush()));
            out.extend(demuxed.map(|p| match p {
                DemuxedPacket::Video(v) => MediaPacket::Video(v),
                DemuxedPacket::Audio(a) => MediaPacket::Audio(a),
            }));
        }
        if let Some(offset_us) = self.seek_offset_us.take() {
            mark_seek_target(&mut out, offset_us);
        }
        Ok(out)
    }

    /// The `EXT-X-MAP` initialization segment `map`, fetched once.
    async fn init_segment(&mut self, map: &InitSegment) -> Step<Bytes> {
        if let Some((cached, bytes)) = &self.init_cache
            && cached == map
        {
            return Ok(bytes.clone());
        }
        let budget = self.retry_budget();
        let fetched = self
            .io
            .fetch(&map.url, map.range, INIT_LIMIT, budget)
            .await?
            .map_err(|e| Stop::Failed(e.into_pull_error(&map.url)))?;
        let body = match &map.key {
            Some(key) => {
                let key_bytes = self.key_bytes(key).await?;
                let plain =
                    playlist::decrypt_aes128(&key_bytes, &key.iv, &fetched.body).map_err(|e| {
                        Stop::Failed(FerricastError::Pull(format!("EXT-X-MAP {}: {e}", map.url)))
                    })?;
                Bytes::from(plain)
            }
            None => fetched.body,
        };
        self.init_cache = Some((map.clone(), body.clone()));
        Ok(body)
    }

    /// Demuxes a fragmented MP4 media segment against its
    /// initialization segment.
    fn demux_fragmented(&mut self, init: &[u8], body: &[u8]) -> Result<Vec<MediaPacket>> {
        let segment = ferricast_demuxer::demux_segment(init, body)
            .map_err(|e| FerricastError::Pull(format!("fragmented MP4 segment: {e}")))?;
        self.fmp4_streams = Some((
            segment.video.as_ref().map(|v| VideoStreamInfo {
                codec: v.codec,
                width: v.width,
                height: v.height,
                fps: 0,
            }),
            segment.audio.as_ref().map(|a| AudioStreamInfo {
                codec: a.codec,
                sample_rate: a.sample_rate,
                channels: a.channels,
                source: a.source,
            }),
        ));
        Ok(segment
            .packets
            .into_iter()
            .map(|p| match p {
                ferricast_demuxer::Packet::Video(v) => MediaPacket::Video(v),
                ferricast_demuxer::Packet::Audio(a) => MediaPacket::Audio(a),
            })
            .collect())
    }

    async fn key_bytes(&mut self, key: &SegmentKey) -> Step<[u8; 16]> {
        if let Some((uri, bytes)) = &self.key_cache
            && *uri == key.uri
        {
            return Ok(*bytes);
        }
        let budget = self.retry_budget();
        let fetched = self.io.fetch_required(&key.uri, KEY_LIMIT, budget).await?;
        let bytes: [u8; 16] = fetched.body.as_ref().try_into().map_err(|_| {
            Stop::Failed(FerricastError::Pull(format!(
                "AES-128 key at {} is {} bytes, expected 16",
                key.uri,
                fetched.body.len()
            )))
        })?;
        self.key_cache = Some((key.uri.clone(), bytes));
        Ok(bytes)
    }

    /// Sleeps the RFC 8216 §6.3.4 reload interval, then reloads.
    async fn wait_and_reload(&mut self) -> Step<()> {
        let target = self.playlist.target_duration as f64;
        let secs = if self.last_reload_changed {
            target
        } else {
            target / 2.0
        };
        let pause = Duration::from_secs_f64(secs.max(0.5));
        self.io.race(tokio::time::sleep(pause)).await?;
        self.reload().await
    }

    async fn reload(&mut self) -> Step<()> {
        let url = self.playlist_url.clone();
        let budget = self.retry_budget();
        let fetched = self.io.fetch_required(&url, PLAYLIST_LIMIT, budget).await?;
        let (new, map_key_order) = parse_media(&fetched.body)?;
        playlist::check_supported(&new).map_err(Stop::Failed)?;

        let old = &self.playlist;
        let old_end = old.media_sequence + old.segments.len() as u64;
        let new_end = new.media_sequence + new.segments.len() as u64;
        let restarted = new.media_sequence < old.media_sequence
            || new.discontinuity_sequence < old.discontinuity_sequence;
        self.last_reload_changed = new_end != old_end;
        self.playlist = new;
        self.map_key_order = map_key_order;
        self.playlist_base = fetched.url;

        if restarted {
            tracing::warn!("HLS media sequence went backwards — source restarted, resyncing");
            self.jump_to_live_edge();
        } else if !self.playlist.end_list {
            let live_start =
                self.playlist.media_sequence + playlist::live_start_index(&self.playlist) as u64;
            if self.next_seq + LIVE_LAG_SEGMENTS < live_start {
                tracing::warn!(
                    from = self.next_seq,
                    to = live_start,
                    "HLS puller fell behind the live edge — skipping forward"
                );
                self.jump_to_live_edge();
            }
        }
        Ok(())
    }
}

/// Marks the first `offset_us` of a segment's packets as decode-only:
/// a seek target that falls inside a segment is reached by decoding from
/// the segment start. The segment start is its first video frame (HLS
/// segments open on a keyframe), or its first audio packet without video.
fn mark_seek_target(packets: &mut Vec<MediaPacket>, offset_us: u64) {
    let first_video = packets.iter().find_map(|p| match p {
        MediaPacket::Video(v) => Some(v.timestamp_us),
        _ => None,
    });
    let first_audio = packets.iter().find_map(|p| match p {
        MediaPacket::Audio(a) => Some(a.timestamp_us),
        _ => None,
    });
    let Some(start_us) = first_video.or(first_audio) else {
        return;
    };
    let at = usize::from(matches!(packets.first(), Some(MediaPacket::Discontinuity)));
    packets.insert(
        at,
        MediaPacket::PresentFrom {
            timestamp_us: start_us.saturating_add(offset_us),
        },
    );
}
