//! Chromecast (CASTv2) session.
//!
//! Lifecycle, in the order [`StreamManager`] drives:
//!
//! 1. [`ChromecastSession::connect`] — open a TLS channel to the
//!    receiver, do the connection / heartbeat handshake, launch the
//!    Default Media Receiver app, latch its `transport_id` /
//!    `session_id`, spawn a background read loop (PONG + status
//!    ingestion) and a heartbeat ticker.
//! 2. [`ChromecastSession::setup_stream`] — store config. We don't
//!    bind HLS or send `LOAD` yet because the segmenter needs SPS /
//!    PPS, which only show up in the bitstream at the first IDR.
//! 3. [`ChromecastSession::send_frame`] — forward each encoded frame
//!    to the HLS segmenter. On the very first keyframe we extract
//!    SPS/PPS, spin up the segmenter, wait until the playlist holds a
//!    few segments, then send `LOAD` so the cast device begins
//!    pulling from us.
//! 4. [`ChromecastSession::stop`] — best-effort `STOP` on the
//!    receiver, drop the writer (closes TLS), abort background
//!    tasks, drop the HLS sink (closes its listener and segmenter).
//!
//! The session ends on its own when the receiver goes silent, the
//! channel breaks, playback fails, or the user / another sender takes
//! the receiver over. [`ChromecastSession::is_alive`] then turns false
//! and [`ChromecastSession::end_reason`] says why; a
//! [`FerricastError::RemoteStopped`] reason means "do not reconnect".
//!
//! [`StreamManager`]: ferricast_core::CastSession

mod monitor;
mod playlist;
#[cfg(test)]
mod tests;

use std::net::IpAddr;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use bytes::BytesMut;
use ferricast_core::{
    AudioCodec, AudioFrame, CastSession, Codec, ConnectOutcome, Device, EncodedFrame,
    FerricastError, Result, StreamConfig, bind_in_range,
};
use ferricast_hls::{HlsConfig, HlsFrameSink};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tracing::{debug, error, info, trace, warn};

use crate::castv2::{
    DEFAULT_MEDIA_RECEIVER_APP_ID, DEFAULT_RECEIVER_ID, DEFAULT_SENDER_ID, MediaInfo,
    close_message, connect_message, get_status_message, launch_message, load_media_message,
    ping_message, stop_app_message,
};
use crate::wire::{self, SharedWriter};
use monitor::{Action, ReceiverMonitor, SessionEnd, Timing};
use playlist::{LoadGate, PlaylistLocation};

/// Channel depth between the manager-driven `send_frame` and the HLS
/// segmenter. Generous enough that a brief stall in the segmenter
/// (file IO, lock contention) doesn't backpressure the encoder, but
/// small enough to drop frames quickly if the player can't keep up.
const FRAME_QUEUE_DEPTH: usize = 64;

/// Channel depth for the audio frame side. Each AAC-LC frame is
/// ~21 ms of audio so a queue of 64 absorbs ~1.36 s of segmenter
/// backlog before frames start getting dropped — paired with the
/// `MAX_AUDIO_STALENESS_US` check inside the segmenter, that's
/// enough headroom to ride out brief HTTP / lock-contention stalls
/// without bloating the queue to the point where buffered audio
/// arrives at the muxer with PTS values seconds in the past
/// (which the chromecast then plays as choppy / wrong-order audio).
const AUDIO_QUEUE_DEPTH: usize = 64;

/// Receivers that don't ack LAUNCH in this window are typically wedged
/// and need a power cycle.
const LAUNCH_TIMEOUT: Duration = Duration::from_secs(15);

/// HLS segment length handed to the segmenter. 2 s segments + 4 s
/// playlist target keep glass-to-glass around 6 s at the cost of ~2×
/// the IDR rate (~10-15 % more bitrate at fixed quality).
const SEGMENT_TARGET_SECS: f32 = 2.0;
const PLAYLIST_TARGET_DURATION: u8 = 4;

/// Terminal state shared by the session and its background tasks.
/// The first reason recorded wins; later ones are consequences.
#[derive(Debug, Default)]
struct SessionHealth {
    end: OnceLock<SessionEnd>,
}

impl SessionHealth {
    fn end(&self, reason: SessionEnd) {
        if self.end.set(reason.clone()).is_ok() {
            match &reason {
                SessionEnd::Remote(why) => info!(%why, "chromecast session ended by the receiver"),
                SessionEnd::Failure(why) => warn!(%why, "chromecast session failed"),
            }
        }
    }

    fn ended(&self) -> Option<&SessionEnd> {
        self.end.get()
    }

    fn is_alive(&self) -> bool {
        self.end.get().is_none()
    }
}

/// Aborts the task when dropped, so an early return (a failed
/// `connect`, a dropped session) never leaves a task behind holding the
/// TLS connection open.
struct AbortOnDrop(JoinHandle<()>);

impl AbortOnDrop {
    fn spawn(fut: impl Future<Output = ()> + Send + 'static) -> Self {
        Self(tokio::spawn(fut))
    }

    async fn cancel(mut self) {
        self.0.abort();
        // A cancelled task resolves with `JoinError::Cancelled`; that is
        // the expected outcome, nothing to report.
        let _ = (&mut self.0).await;
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

pub struct ChromecastSession {
    /// All TLS / app-launch state, populated by [`Self::connect`].
    /// `None` before connect or after `stop`.
    state: Option<ConnectedState>,
    /// HLS / streaming state, populated lazily on the first keyframe.
    stream: Option<StreamState>,
    /// Encoder config carried from `setup_stream` so the HLS sink is
    /// sized correctly when the first frame arrives.
    cfg: Option<StreamConfig>,
    /// Why the last connection ended, kept across `stop` so the caller
    /// can still ask after tearing down.
    last_end: Option<SessionEnd>,
    /// Segments the playlist must list before LOAD goes out.
    load_min_segments: usize,
    timing: Timing,
}

impl Default for ChromecastSession {
    fn default() -> Self {
        Self {
            state: None,
            stream: None,
            cfg: None,
            last_end: None,
            load_min_segments: LoadGate::DEFAULT_MIN_SEGMENTS,
            timing: Timing::default(),
        }
    }
}

struct ConnectedState {
    writer: SharedWriter,
    /// Application transport ID, used as the destination for all
    /// media-namespace messages once the receiver app is launched.
    transport_id: String,
    /// Session ID returned by LAUNCH. Required by STOP; otherwise
    /// the receiver refuses to tear the app down.
    session_id: String,
    /// Local IP from the receiver's perspective (the LAN side of the
    /// TCP socket). Used to build the HLS URL we hand back.
    local_ip: IpAddr,
    /// The receiver itself: the only remote peer allowed to fetch the
    /// stream.
    device_ip: IpAddr,
    /// Mirror of `device.capabilities.requires_audio` so the HLS
    /// bootstrap path knows whether to ask the MPEG-TS muxer to
    /// inject a silent AAC track. Old generic Chromecasts need
    /// this; Ultra / Google TV don't.
    requires_audio: bool,
    /// Mirror of `device.capabilities.supports_low_latency_hls`.
    /// Old Chromecasts choke on the LL-HLS (EXT-X-VERSION:6) playlist
    /// and end up LOADING forever, so they get classic HLS.
    supports_low_latency_hls: bool,
    /// Monotonic CASTv2 request-id counter, shared with the read
    /// loop so requests never reuse an id.
    request_id: Arc<AtomicI64>,
    health: Arc<SessionHealth>,
    read_task: AbortOnDrop,
    heartbeat_task: AbortOnDrop,
}

impl ConnectedState {
    fn is_alive(&self) -> bool {
        self.health.is_alive() && !self.writer.is_broken()
    }

    fn end(&self) -> Option<SessionEnd> {
        self.health.ended().cloned().or_else(|| {
            self.writer
                .is_broken()
                .then(|| SessionEnd::Failure("cast channel broken by a failed send".into()))
        })
    }
}

struct StreamState {
    /// Sender into the HLS segmenter task. Dropping it (via `stop`)
    /// makes the segmenter drain and exit cleanly.
    frame_tx: mpsc::Sender<EncodedFrame>,
    /// Audio sender, when `StreamConfig::audio` was set in
    /// `setup_stream`. `None` keeps the segmenter in video-only
    /// mode (falls back to silent-AAC injection if the receiver
    /// is a 1st/2nd-gen Chromecast).
    audio_tx: Option<mpsc::Sender<AudioFrame>>,
    /// HLS endpoint. Owns the TCP listener + segmenter task; held
    /// solely for its Drop side-effect (cleanup).
    _sink: HlsFrameSink,
    /// Waits for the playlist to fill and then sends `LOAD`. Runs in
    /// the background because the segmenter only emits segments as
    /// more frames arrive through `send_frame`; blocking there would
    /// starve it.
    _load_task: AbortOnDrop,
}

impl ChromecastSession {
    /// Number of HLS segments the playlist must hold before `LOAD` is
    /// sent (default 3). Lower values start faster but leave the
    /// receiver no buffer at the live edge.
    #[must_use]
    pub fn with_min_segments_before_load(mut self, segments: usize) -> Self {
        self.load_min_segments = segments.max(1);
        self
    }

    /// Why the session stopped being alive, if it did. Stays available
    /// after [`CastSession::stop`] until the next `connect`.
    ///
    /// [`FerricastError::RemoteStopped`] means the user or another
    /// sender ended the session on purpose (another app took the
    /// screen, "Stop casting", media interrupted): do not reconnect.
    /// Anything else is a failure that a reconnect may fix.
    #[must_use]
    pub fn end_reason(&self) -> Option<FerricastError> {
        self.state
            .as_ref()
            .and_then(ConnectedState::end)
            .or_else(|| self.last_end.clone())
            .map(|end| end.to_error())
    }

    fn remote_stop(&self) -> Option<FerricastError> {
        match self.state.as_ref()?.health.ended()? {
            end @ SessionEnd::Remote(_) => Some(end.to_error()),
            SessionEnd::Failure(_) => None,
        }
    }
}

impl CastSession for ChromecastSession {
    async fn connect(&mut self, device: &Device) -> Result<ConnectOutcome> {
        if device.protocol != "chromecast" {
            return Err(FerricastError::Protocol(format!(
                "ChromecastSession cannot connect to a {:?} device",
                device.protocol
            )));
        }
        if self.state.is_some() {
            return Err(FerricastError::SessionAlreadyActive(device.name.clone()));
        }
        self.last_end = None;

        info!(
            name = %device.name,
            addr = %device.addr,
            port = device.port,
            "connecting to chromecast"
        );
        let (reader, writer, local_ip) = wire::connect(device.addr, device.port).await?;

        // Spawn the read loop *before* sending anything. Some firmwares
        // answer CONNECT with status / heartbeat traffic that must be
        // drained or the receive buffers fill and backpressure our
        // writes, and PINGs must be PONG'd during the launch handshake
        // too. The loop reports the launched app's (transport_id,
        // session_id) through a one-shot. Every `?` below drops
        // `read_task`, which aborts the loop and closes the connection.
        let health = Arc::new(SessionHealth::default());
        let request_id = Arc::new(AtomicI64::new(1));
        let launch_id = request_id.fetch_add(1, Ordering::Relaxed);
        let (app_tx, app_rx) = oneshot::channel::<Result<(String, String)>>();
        let monitor = ReceiverMonitor::launching(
            DEFAULT_MEDIA_RECEIVER_APP_ID,
            launch_id,
            self.timing,
            Instant::now(),
        );
        let read_task = AbortOnDrop::spawn(read_loop(
            reader,
            ReadLoopCtx {
                writer: writer.clone(),
                health: health.clone(),
                request_id: request_id.clone(),
                launch_tx: Some(app_tx),
            },
            monitor,
        ));

        // Virtual-connect to receiver-0 so subsequent control messages
        // (LAUNCH, GET_STATUS, …) are accepted. The payload stays the
        // bare `{"type":"CONNECT"}`: some firmwares close the channel on
        // userAgent / origin / senderInfo from a sender they don't know.
        wire::send(
            &writer,
            &connect_message(DEFAULT_SENDER_ID, DEFAULT_RECEIVER_ID)
                .map_err(|e| FerricastError::Protocol(format!("build CONNECT: {e}")))?,
        )
        .await?;

        // Some firmwares (1st-gen / Audio devices, others in odd
        // states) close the channel when LAUNCH arrives without a prior
        // heartbeat. Fire-and-forget; no need to wait for the PONG.
        wire::send(
            &writer,
            &ping_message().map_err(|e| FerricastError::Protocol(format!("build PING: {e}")))?,
        )
        .await?;

        wire::send(
            &writer,
            &launch_message(launch_id, DEFAULT_MEDIA_RECEIVER_APP_ID)
                .map_err(|e| FerricastError::Protocol(format!("build LAUNCH: {e}")))?,
        )
        .await?;

        // Some firmwares treat LAUNCH of an already-running app as a
        // no-op and never re-broadcast RECEIVER_STATUS. GET_STATUS
        // forces the current state out.
        let status_id = request_id.fetch_add(1, Ordering::Relaxed);
        wire::send(
            &writer,
            &get_status_message(status_id)
                .map_err(|e| FerricastError::Protocol(format!("build GET_STATUS: {e}")))?,
        )
        .await?;

        let (transport_id, session_id) = match tokio::time::timeout(LAUNCH_TIMEOUT, app_rx).await {
            Ok(Ok(Ok(ids))) => ids,
            Ok(Ok(Err(e))) => return Err(e),
            Ok(Err(_)) => {
                return Err(FerricastError::Connection(
                    "read loop exited before launch ack".into(),
                ));
            }
            Err(_) => {
                return Err(FerricastError::Timeout(format!(
                    "waiting for receiver to launch app ({}s)",
                    LAUNCH_TIMEOUT.as_secs()
                )));
            }
        };
        debug!(transport_id, session_id, "DefaultMediaReceiver launched");

        // Virtual-connect to the launched app's transport so we can
        // address it from the media namespace.
        wire::send(
            &writer,
            &connect_message(DEFAULT_SENDER_ID, &transport_id)
                .map_err(|e| FerricastError::Protocol(format!("build app CONNECT: {e}")))?,
        )
        .await?;

        let heartbeat_task = AbortOnDrop::spawn(heartbeat_loop(
            writer.clone(),
            health.clone(),
            self.timing.heartbeat,
        ));

        self.state = Some(ConnectedState {
            writer,
            transport_id,
            session_id,
            local_ip,
            device_ip: device.addr,
            requires_audio: device.capabilities.requires_audio,
            supports_low_latency_hls: device.capabilities.supports_low_latency_hls,
            request_id,
            health,
            read_task,
            heartbeat_task,
        });
        Ok(ConnectOutcome::Ready)
    }

    async fn setup_stream(&mut self, config: &StreamConfig) -> Result<()> {
        if self.state.is_none() {
            return Err(FerricastError::NoActiveSession);
        }

        if config.codec != Codec::H264 {
            return Err(FerricastError::UnsupportedCodec {
                codec: config.codec,
                protocol: "chromecast",
            });
        }

        info!(
            width = config.width,
            height = config.height,
            fps = config.fps,
            bitrate_kbps = config.bitrate_kbps,
            "chromecast stream configured"
        );
        self.cfg = Some(config.clone());
        Ok(())
    }

    async fn send_frame(&mut self, frame: &EncodedFrame) -> Result<()> {
        if let Some(stopped) = self.remote_stop() {
            return Err(stopped);
        }
        if frame.codec != Codec::H264 {
            return Err(FerricastError::UnsupportedCodec {
                codec: frame.codec,
                protocol: "chromecast",
            });
        }

        // Lazy init on the first keyframe: the segmenter needs SPS +
        // PPS, which only appear at IDR access units. Pre-IDR frames
        // are dropped (the segmenter would drop them anyway).
        if self.stream.is_none() {
            if !frame.is_keyframe {
                trace!("dropping pre-keyframe before HLS init");
                return Ok(());
            }
            self.bootstrap_hls(frame).await?;
        }

        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| FerricastError::Streaming("HLS stream not initialised".into()))?;

        // try_send avoids stalling the upstream encoder when the
        // segmenter falls behind. Recovery happens at the next IDR.
        match stream.frame_tx.try_send(frame.clone()) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                debug!("HLS segmenter backlogged, dropping frame");
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                return Err(FerricastError::Streaming(
                    "HLS segmenter channel closed".into(),
                ));
            }
        }

        Ok(())
    }

    async fn send_audio_frame(&mut self, frame: &AudioFrame) -> Result<()> {
        if frame.codec != AudioCodec::Aac {
            return Err(FerricastError::Encoder(format!(
                "chromecast: expected AAC audio, got {:?}",
                frame.codec
            )));
        }
        // Audio that arrives before the HLS bootstrap (first video
        // keyframe) is dropped: the segmenter has no PTS anchor yet
        // and pushing audio without it would mis-align A/V.
        let Some(stream) = self.stream.as_mut() else {
            trace!("dropping pre-bootstrap audio frame");
            return Ok(());
        };
        let Some(audio_tx) = stream.audio_tx.as_ref() else {
            // Stream was bootstrapped without audio (e.g. the user
            // didn't enable PipeWire audio capture). Treat as a
            // silent drop so the manager's audio task can keep
            // pushing without spamming errors.
            return Ok(());
        };
        match audio_tx.try_send(frame.clone()) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                // Only fires when the segmenter is wedged for longer
                // than the queue depth (HTTP starvation, ring lock
                // contention). `warn!` so it shows without trace logs.
                warn!("HLS audio segmenter backlogged, dropping AAC frame");
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                return Err(FerricastError::Streaming(
                    "HLS audio segmenter channel closed".into(),
                ));
            }
        }
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        // Drop HLS first so the segmenter sees EOF on its channel
        // before we kill the TLS link.
        self.stream.take();

        let Some(state) = self.state.take() else {
            return Ok(());
        };
        self.last_end = state.end();

        // Best-effort STOP + CLOSE so a failed stream doesn't leave the
        // receiver app on screen. Skipped when the channel is broken, or
        // when the user or another app ended our session: our app is
        // gone then and a STOP could only hit somebody else's.
        let remote_ended = matches!(self.last_end, Some(SessionEnd::Remote(_)));
        if !remote_ended && !state.writer.is_broken() {
            let stop_id = state.request_id.fetch_add(1, Ordering::Relaxed);
            if let Ok(stop_msg) = stop_app_message(stop_id, &state.session_id)
                && let Err(e) = wire::send(&state.writer, &stop_msg).await
            {
                warn!(%e, "STOP message failed");
            }
            if let Ok(close) = close_message(DEFAULT_SENDER_ID, DEFAULT_RECEIVER_ID)
                && let Err(e) = wire::send(&state.writer, &close).await
            {
                debug!(%e, "CLOSE message failed");
            }
        }

        state.read_task.cancel().await;
        state.heartbeat_task.cancel().await;

        info!("chromecast session stopped");
        Ok(())
    }

    fn is_alive(&self) -> bool {
        self.state.as_ref().is_some_and(ConnectedState::is_alive)
    }

    /// The HLS server reports segment-fetch throughput into the shared
    /// adaptive-bitrate state (`HlsConfig::adaptive`).
    fn feeds_adaptive_bitrate(&self) -> bool {
        true
    }

    fn end_reason(&self) -> Option<FerricastError> {
        ChromecastSession::end_reason(self)
    }
}

impl ChromecastSession {
    /// Build the HLS frame sink and stash it in `self.stream`.
    /// Called exactly once, on the first keyframe; subsequent
    /// `send_frame` calls become straight forwards into the channel.
    async fn bootstrap_hls(&mut self, first_keyframe: &EncodedFrame) -> Result<()> {
        let state = self.state.as_ref().ok_or(FerricastError::NoActiveSession)?;
        let cfg = self
            .cfg
            .as_ref()
            .ok_or_else(|| FerricastError::Streaming("setup_stream not called".into()))?;

        let parameter_sets = ferricast_encoder::h264::utils::extract_sps_pps(&first_keyframe.data);
        if parameter_sets.is_empty() {
            return Err(FerricastError::Encoder(
                "first keyframe carries no SPS/PPS; cannot bootstrap HLS".into(),
            ));
        }
        info!(
            param_set_bytes = parameter_sets.len(),
            "extracted SPS/PPS from first keyframe"
        );

        let (frame_tx, frame_rx) = mpsc::channel::<EncodedFrame>(FRAME_QUEUE_DEPTH);

        // Without an `audio` block in `setup_stream` the HLS sink stays
        // video-only (which preserves the silent-AAC fallback for old
        // Chromecasts).
        let (audio_tx, audio_rx) = if cfg.audio.is_some() {
            let (tx, rx) = mpsc::channel::<AudioFrame>(AUDIO_QUEUE_DEPTH);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        // LL-HLS only for device classes where it's been verified: the
        // 1st/2nd-gen firmware stalls in LOADING forever on a
        // `#EXT-X-VERSION:6` playlist.
        let part_target_secs = state.supports_low_latency_hls.then_some(0.5);
        let hls_config = HlsConfig {
            // Silent-AAC injection stays gated on the device flag; real
            // audio overrides it inside the segmenter so the audio PID
            // never mixes silence with real samples. Old Chromecast
            // 1st/2nd gen reject a video-only stream with LOAD_FAILED.
            inject_silent_audio: state.requires_audio,
            adaptive: cfg.adaptive.clone(),
            part_target_secs,
            // Plain HTTP on purpose: the Default Media Receiver's
            // onboard Chrome rejects self-signed certificates before
            // reading the playlist (`LOAD_FAILED`, no detailed code).
            tls: None,
            // Match the segmenter's PTS clock to the real frame rate;
            // the default (60) lags video behind wall time for a whole
            // segment on 30 fps streams and pushes audio ahead of video.
            target_fps: cfg.fps,
            segment_target_secs: SEGMENT_TARGET_SECS,
            playlist_target_duration: PLAYLIST_TARGET_DURATION,
            ..Default::default()
        };
        let sink = match cfg.port_range {
            Some((start, end)) => {
                let listener = bind_in_range(start, end).await?;
                HlsFrameSink::start_on_listener(
                    listener,
                    frame_rx,
                    audio_rx,
                    parameter_sets,
                    hls_config,
                )
                .await?
            }
            None => {
                HlsFrameSink::start("0.0.0.0:0", frame_rx, audio_rx, parameter_sets, hls_config)
                    .await?
            }
        };
        // Bound on every interface, so only the receiver (and loopback)
        // may fetch; advertise the LAN-side IP observed on the control
        // socket (`0.0.0.0` is unroutable).
        sink.allow_peer(state.device_ip);
        let location = PlaylistLocation::new(
            sink.scheme(),
            state.local_ip,
            sink.local_addr().port(),
            &sink.playlist_path(),
        );
        info!(
            url = %location.url(),
            "chromecast HLS endpoint ready (open this URL in a player to verify)"
        );

        let load_task = AbortOnDrop::spawn(load_when_ready(LoadCtx {
            first_segment: Box::pin(sink.first_segment_ready()),
            gate: LoadGate::new(self.load_min_segments, SEGMENT_TARGET_SECS),
            location,
            writer: state.writer.clone(),
            transport_id: state.transport_id.clone(),
            request_id: state.request_id.clone(),
            health: state.health.clone(),
        }));

        // Seed the keyframe ourselves so the segmenter doesn't have
        // to wait for the next IDR (which is `keyframe_interval`
        // frames away). The channel was just created, so it's empty.
        if let Err(e) = frame_tx.try_send(first_keyframe.clone()) {
            warn!(?e, "could not seed first keyframe (channel full at init)");
        }

        self.stream = Some(StreamState {
            frame_tx,
            audio_tx,
            _sink: sink,
            _load_task: load_task,
        });
        Ok(())
    }
}

struct LoadCtx {
    first_segment: std::pin::Pin<Box<dyn Future<Output = bool> + Send>>,
    gate: LoadGate,
    location: PlaylistLocation,
    writer: SharedWriter,
    transport_id: String,
    request_id: Arc<AtomicI64>,
    health: Arc<SessionHealth>,
}

/// Wait for the playlist to be playable, then send LOAD. Any failure
/// ends the session: without LOAD the receiver sits on its splash
/// screen forever.
async fn load_when_ready(ctx: LoadCtx) {
    if !ctx.first_segment.await {
        ctx.health.end(SessionEnd::Failure(
            "HLS segmenter produced no segment — encoder failed to produce output".into(),
        ));
        return;
    }
    let segments = ctx.gate.wait(&ctx.location).await;
    if !ctx.health.is_alive() {
        return;
    }

    let url = ctx.location.url();
    let media = MediaInfo {
        content_id: url.clone(),
        // Default Media Receiver accepts the lowercase form (the spec
        // is case-insensitive but some receiver versions are picky).
        content_type: "application/x-mpegurl".to_string(),
        stream_type: Some("LIVE".to_string()),
        duration: None,
        // Audio and video share one MPEG-TS stream.
        hls_segment_format: Some("ts".to_string()),
        hls_video_segment_format: Some("mpeg2_ts".to_string()),
    };
    let request_id = ctx.request_id.fetch_add(1, Ordering::Relaxed);
    let msg = match load_media_message(request_id, &ctx.transport_id, media) {
        Ok(m) => m,
        Err(e) => {
            ctx.health
                .end(SessionEnd::Failure(format!("build LOAD message: {e}")));
            return;
        }
    };
    match wire::send(&ctx.writer, &msg).await {
        Ok(()) => info!(%url, segments, "sent LOAD to chromecast"),
        Err(e) => ctx
            .health
            .end(SessionEnd::Failure(format!("LOAD send failed: {e}"))),
    }
}

struct ReadLoopCtx {
    writer: SharedWriter,
    health: Arc<SessionHealth>,
    request_id: Arc<AtomicI64>,
    /// Resolved once with the launched app's ids, or with the error
    /// that ended the launch.
    launch_tx: Option<oneshot::Sender<Result<(String, String)>>>,
}

/// Background read loop: feeds every inbound message and timer expiry
/// to the [`ReceiverMonitor`] and carries out what it asks for. Exits,
/// recording the reason in [`SessionHealth`], when the session ends.
async fn read_loop<R>(mut reader: R, mut ctx: ReadLoopCtx, mut monitor: ReceiverMonitor)
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut buf = BytesMut::with_capacity(8 * 1024);
    let end = loop {
        // `wire::recv` keeps partial frames in `buf`, so cancelling it
        // on a timer and calling again loses nothing.
        let received =
            tokio::time::timeout_at(monitor.next_wakeup(), wire::recv(&mut reader, &mut buf)).await;
        let now = Instant::now();
        let action = match received {
            Err(_elapsed) => monitor.on_tick(now),
            Ok(Ok(Some(msg))) => monitor.on_message(&msg, now),
            Ok(Ok(None)) => {
                Action::End(SessionEnd::Failure("receiver closed the connection".into()))
            }
            Ok(Err(e)) => Action::End(SessionEnd::Failure(format!("cast read failed: {e}"))),
        };
        let sent = match action {
            Action::Nothing => Ok(()),
            Action::Reply(reply) => wire::send(&ctx.writer, &reply).await,
            Action::Nudge => {
                let id = ctx.request_id.fetch_add(1, Ordering::Relaxed);
                match get_status_message(id) {
                    Ok(m) => wire::send(&ctx.writer, &m).await,
                    Err(e) => Err(FerricastError::Protocol(format!("build GET_STATUS: {e}"))),
                }
            }
            Action::Launched {
                transport_id,
                session_id,
            } => {
                if let Some(tx) = ctx.launch_tx.take() {
                    // The receiver only drops if `connect` gave up.
                    let _ = tx.send(Ok((transport_id, session_id)));
                }
                Ok(())
            }
            Action::End(end) => break end,
        };
        if let Err(e) = sent {
            break SessionEnd::Failure(format!("cast send failed: {e}"));
        }
        if let Some(end) = ctx.health.ended() {
            break end.clone();
        }
    };
    if let Some(tx) = ctx.launch_tx.take() {
        let _ = tx.send(Err(end.to_error()));
    }
    if let SessionEnd::Failure(why) = &end {
        error!(%why, "chromecast read loop terminating");
    }
    ctx.health.end(end);
}

/// Background heartbeat ticker. A failed PING ends the session: the
/// channel is broken at that point.
async fn heartbeat_loop(writer: SharedWriter, health: Arc<SessionHealth>, period: Duration) {
    let mut tick = tokio::time::interval(period);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // The first tick fires immediately; launch just pinged.
    tick.tick().await;
    loop {
        tick.tick().await;
        if !health.is_alive() {
            return;
        }
        let ping = match ping_message() {
            Ok(p) => p,
            Err(e) => {
                health.end(SessionEnd::Failure(format!("build PING: {e}")));
                return;
            }
        };
        if let Err(e) = wire::send(&writer, &ping).await {
            health.end(SessionEnd::Failure(format!("heartbeat send failed: {e}")));
            return;
        }
    }
}
