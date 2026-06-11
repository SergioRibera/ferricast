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
//!    SPS/PPS, spin up the segmenter, wait for the first segment to
//!    materialise, then send `LOAD` so the cast device begins
//!    pulling from us.
//! 4. [`ChromecastSession::stop`] — best-effort `STOP` on the
//!    receiver, drop the writer (closes TLS), abort background
//!    tasks, drop the HLS sink (closes its listener and segmenter).

use std::net::IpAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::time::Duration;

use bytes::BytesMut;
use ferricast_core::{
    AudioCodec, AudioFrame, CastSession, Codec, Device, EncodedFrame, FerricastError, Result,
    StreamConfig,
};
use ferricast_hls::{HlsConfig, HlsFrameSink};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, info, trace, warn};

use crate::castv2::{
    CastMessage, DEFAULT_MEDIA_RECEIVER_APP_ID, DEFAULT_RECEIVER_ID, DEFAULT_SENDER_ID,
    MediaDetailedErrorCode, MediaInfo, MediaStatusPayload, ReceiverStatusPayload, close_message,
    connect_message, get_status_message, launch_message, load_media_message, namespace,
    ping_message, pong_message, stop_app_message,
};
use crate::wire::{self, SharedWriter};

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

/// How often we ping the receiver. The Chromecast disconnects
/// senders that go silent for ~10 s.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Default)]
pub struct ChromecastSession {
    /// All TLS / app-launch state, populated by [`Self::connect`].
    /// `None` before connect or after `stop`.
    state: Option<ConnectedState>,
    /// HLS / streaming state, populated lazily on the first keyframe.
    stream: Option<StreamState>,
    /// Encoder config carried from `setup_stream` so the HLS sink is
    /// sized correctly when the first frame arrives.
    cfg: Option<StreamConfig>,
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
    /// Mirror of `device.capabilities.requires_audio` so the HLS
    /// bootstrap path knows whether to ask the MPEG-TS muxer to
    /// inject a silent AAC track. Old generic Chromecasts need
    /// this; Ultra / Google TV don't.
    requires_audio: bool,
    /// Mirror of `device.capabilities.supports_low_latency_hls`.
    /// Controls whether `bootstrap_hls` enables LL-HLS (parts,
    /// blocking reload, EXT-X-VERSION:6) on the HLS server. Old
    /// Chromecasts choke on the v6 playlist and end up in a
    /// LOADING-forever state; their capabilities row carries
    /// `false` here so the HLS endpoint stays in classic HLS
    /// mode (no parts, EXT-X-VERSION:3) for them.
    supports_low_latency_hls: bool,
    /// Monotonic CASTv2 request-id counter, shared with the read
    /// loop so requests and responses don't collide.
    request_id: Arc<AtomicI64>,
    /// Live flag flipped to `false` by the read loop on EOF / fatal
    /// error, and read by [`Self::is_alive`].
    alive: Arc<AtomicBool>,
    /// Background tasks: read loop + heartbeat ticker. Aborted in
    /// `stop` so we don't leak them on stream restart.
    read_handle: JoinHandle<()>,
    heartbeat_handle: JoinHandle<()>,
}

struct StreamState {
    /// Sender into the HLS segmenter task. Dropping it (via `stop`)
    /// makes the segmenter drain and exit cleanly.
    frame_tx: mpsc::Sender<EncodedFrame>,
    /// Audio sender, when `StreamConfig::audio` was set in
    /// `setup_stream`. `None` keeps the segmenter in video-only
    /// mode (falls back to silent-AAC injection if the receiver
    /// is a 1st/2nd-gen Chromecast). Dropping it cleanly signals
    /// EOF to the audio side of the segmenter.
    audio_tx: Option<mpsc::Sender<AudioFrame>>,
    /// HLS endpoint. Owns the TCP listener + segmenter task; held
    /// solely for its Drop side-effect (cleanup) — the wait+LOAD
    /// task gets its own clone of the readiness future.
    _sink: HlsFrameSink,
    /// Background task that waits for the first segment to land and
    /// then sends `LOAD` to the receiver. Spawned because doing it
    /// inline in `send_frame` deadlocks the manager loop — the
    /// segmenter only emits a segment after seeing 1+ keyframe past
    /// `target_duration`, and those keyframes can only arrive
    /// *through* `send_frame` calls. Blocking inside one starves the
    /// rest. Aborted in `Drop` so it doesn't leak past the session.
    load_task: JoinHandle<()>,
}

impl Drop for StreamState {
    fn drop(&mut self) {
        self.load_task.abort();
    }
}

impl CastSession for ChromecastSession {
    async fn connect(&mut self, device: &Device) -> Result<()> {
        if device.protocol != "chromecast" {
            return Err(FerricastError::Protocol(format!(
                "ChromecastSession cannot connect to a {:?} device",
                device.protocol
            )));
        }
        if self.state.is_some() {
            return Err(FerricastError::SessionAlreadyActive(device.name.clone()));
        }

        info!(
            name = %device.name,
            addr = %device.addr,
            port = device.port,
            "connecting to chromecast"
        );
        let (reader, writer, local_ip) = wire::connect(device.addr, device.port).await?;

        // Step 1: spawn the read loop *before* sending anything on
        // the wire. Some chromecast firmwares immediately respond
        // after CONNECT with status / heartbeat traffic that has
        // to be drained or the TLS / TCP receive buffers fill and
        // backpressure our writes. Putting the read loop in charge
        // from the start also means PINGs from the receiver get
        // PONG'd unconditionally — including during the launch
        // handshake — without any "during launch / after launch"
        // branching like the old code had.
        //
        // The read loop publishes the launched app's
        // (transport_id, session_id) through a one-shot. We block
        // on it after sending LAUNCH.
        let alive = Arc::new(AtomicBool::new(true));
        let (app_tx, app_rx) = oneshot::channel::<Result<(String, String)>>();
        let read_alive = alive.clone();
        let read_writer = writer.clone();
        let read_handle = tokio::spawn(read_loop(
            reader,
            read_writer,
            read_alive,
            Some((DEFAULT_MEDIA_RECEIVER_APP_ID.to_string(), app_tx)),
        ));

        // Step 2: virtual-connect to receiver-0 so subsequent control
        // messages (LAUNCH, GET_STATUS, …) are accepted.
        wire::send(
            &writer,
            &connect_message(DEFAULT_SENDER_ID, DEFAULT_RECEIVER_ID)
                .map_err(|e| FerricastError::Protocol(format!("build CONNECT: {e}")))?,
        )
        .await?;

        // Step 3: ping the heartbeat namespace. Some Chromecast
        // firmwares (notably 1st-gen / older Audio devices, but also
        // some newer ones in odd states) close the TLS connection
        // mid-handshake when LAUNCH arrives without a prior
        // heartbeat — they treat the absence as "sender doesn't
        // implement the protocol" and bail. rust_cast and
        // pychromecast both ping right here. The ping is fire-and-
        // forget; we don't wait for a PONG before continuing.
        wire::send(
            &writer,
            &ping_message().map_err(|e| FerricastError::Protocol(format!("build PING: {e}")))?,
        )
        .await?;

        // Step 4: launch the Default Media Receiver. The read loop
        // will catch the resulting RECEIVER_STATUS and publish the
        // app's transport_id / session_id through the oneshot.
        let request_id = Arc::new(AtomicI64::new(1));
        let launch_id = request_id.fetch_add(1, Ordering::Relaxed);
        wire::send(
            &writer,
            &launch_message(launch_id, DEFAULT_MEDIA_RECEIVER_APP_ID)
                .map_err(|e| FerricastError::Protocol(format!("build LAUNCH: {e}")))?,
        )
        .await?;

        // Step 4b: force a RECEIVER_STATUS dump. Some firmwares
        // treat LAUNCH for an already-running app as idempotent
        // and *don't* re-broadcast STATUS — observed in practice
        // as PONG arriving after PING but no RECEIVER_STATUS ever
        // following LAUNCH on a chromecast that's been recently
        // talked to. GET_STATUS forces the current state out
        // unconditionally; the read loop catches the result the
        // same way it would a launch-triggered STATUS. Cheap
        // belt-and-suspenders.
        let status_id = request_id.fetch_add(1, Ordering::Relaxed);
        wire::send(
            &writer,
            &get_status_message(status_id)
                .map_err(|e| FerricastError::Protocol(format!("build GET_STATUS: {e}")))?,
        )
        .await?;

        // Step 5: wait for the launch ack. Cap at 15 s — receivers
        // that don't ack in that window are typically wedged and
        // need a power cycle.
        let (transport_id, session_id) =
            match tokio::time::timeout(Duration::from_secs(15), app_rx).await {
                Ok(Ok(Ok(ids))) => ids,
                Ok(Ok(Err(e))) => return Err(e),
                Ok(Err(_)) => {
                    return Err(FerricastError::Connection(
                        "read loop exited before launch ack".into(),
                    ));
                }
                Err(_) => {
                    return Err(FerricastError::Timeout(
                        "waiting for receiver to launch app (15s)".into(),
                    ));
                }
            };
        debug!(transport_id, session_id, "DefaultMediaReceiver launched");

        // Step 6: virtual-connect to the launched app's transport so
        // we can address it from the media namespace.
        wire::send(
            &writer,
            &connect_message(DEFAULT_SENDER_ID, &transport_id)
                .map_err(|e| FerricastError::Protocol(format!("build app CONNECT: {e}")))?,
        )
        .await?;

        // Step 7: spawn the heartbeat ticker (read loop is already
        // running and PONGs the receiver's pings — this side keeps
        // our half warm with periodic outgoing pings).
        let hb_writer = writer.clone();
        let hb_alive = alive.clone();
        let heartbeat_handle = tokio::spawn(heartbeat_loop(hb_writer, hb_alive));

        self.state = Some(ConnectedState {
            writer,
            transport_id,
            session_id,
            local_ip,
            requires_audio: device.capabilities.requires_audio,
            supports_low_latency_hls: device.capabilities.supports_low_latency_hls,
            request_id,
            alive,
            read_handle,
            heartbeat_handle,
        });
        Ok(())
    }

    async fn setup_stream(&mut self, config: &StreamConfig) -> Result<()> {
        let _ = self.state.as_ref().ok_or(FerricastError::NoActiveSession)?;

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
                // With AUDIO_QUEUE_DEPTH=256 (~5.4 s), this only
                // fires when the segmenter is wedged for longer
                // than one segment-target window — i.e. something
                // is genuinely wrong (HTTP starvation, ring
                // RwLock pathologically contended). Surface at
                // `warn!` so it's visible without `RUST_LOG=trace`.
                warn!("HLS audio segmenter backlogged 5+ s, dropping AAC frame");
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

        // Best-effort STOP — receiver will time us out anyway.
        let stop_id = state.request_id.fetch_add(1, Ordering::Relaxed);
        if let Ok(stop_msg) = stop_app_message(stop_id, &state.session_id) {
            if let Err(e) = wire::send(&state.writer, &stop_msg).await {
                warn!(%e, "STOP message failed");
            }
        }
        if let Ok(close) = close_message(DEFAULT_SENDER_ID, DEFAULT_RECEIVER_ID) {
            let _ = wire::send(&state.writer, &close).await;
        }

        state.alive.store(false, Ordering::Relaxed);
        state.read_handle.abort();
        state.heartbeat_handle.abort();
        let _ = state.read_handle.await;
        let _ = state.heartbeat_handle.await;

        info!("chromecast session stopped");
        Ok(())
    }

    fn is_alive(&self) -> bool {
        self.state
            .as_ref()
            .map(|s| s.alive.load(Ordering::Relaxed))
            .unwrap_or(false)
    }
}

impl Drop for ChromecastSession {
    /// Defensive cleanup: if the session is dropped without `stop()`
    /// being called (e.g. process aborted, panic in the streaming
    /// loop) abort the read + heartbeat tasks so their handles don't
    /// outlive the runtime they were spawned on.
    ///
    /// `StreamState`'s drop handles HLS shutdown automatically
    /// (HlsFrameSink and the load_task JoinHandle abort on drop), so
    /// we only need to deal with `ConnectedState` here.
    fn drop(&mut self) {
        if let Some(state) = self.state.as_ref() {
            state.alive.store(false, Ordering::Relaxed);
            state.read_handle.abort();
            state.heartbeat_handle.abort();
        }
    }
}

impl ChromecastSession {
    /// Build the HLS frame sink and stash it in `self.stream`.
    /// Called exactly once, on the first keyframe; subsequent
    /// `send_frame` calls become straight forwards into the channel.
    ///
    /// Spawns a background task that waits for the segmenter's first
    /// segment and then sends `LOAD` to the receiver. It can't run
    /// inline because the segmenter needs more frames to produce a
    /// segment, and those frames only flow through subsequent
    /// `send_frame` calls — blocking here would prevent that.
    async fn bootstrap_hls(&mut self, first_keyframe: &EncodedFrame) -> Result<()> {
        let state = self.state.as_ref().ok_or(FerricastError::NoActiveSession)?;
        let local_ip = state.local_ip;
        let writer = state.writer.clone();
        let transport_id = state.transport_id.clone();
        let request_id_counter = state.request_id.clone();
        let requires_audio = state.requires_audio;

        let _ = self
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

        // Audio channel — created lazily when `setup_stream` was
        // handed an `audio` block. Without that, the HLS sink stays
        // in classic video-only mode (which preserves the silent-
        // AAC fallback for old Chromecasts).
        let audio_enabled = self
            .cfg
            .as_ref()
            .map(|c| c.audio.is_some())
            .unwrap_or(false);
        let (audio_tx, audio_rx) = if audio_enabled {
            let (tx, rx) = mpsc::channel::<AudioFrame>(AUDIO_QUEUE_DEPTH);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        // Bind on every interface but advertise the LAN-side IP we
        // observed during TLS connect — `0.0.0.0` is unroutable.
        // `inject_silent_audio` comes from the device's
        // `DeviceCapabilities::requires_audio` — old Chromecast 1st/
        // 2nd gen need a silent audio track to not reject the HLS
        // stream with LOAD_FAILED.
        let adaptive_for_hls = self.cfg.as_ref().and_then(|c| c.adaptive.clone());
        // LL-HLS opt-in, gated on `DeviceCapabilities::
        // supports_low_latency_hls`. The 1st/2nd-gen Chromecast
        // firmware demonstrably stalls in the LOADING-forever
        // state when handed a `#EXT-X-VERSION:6` playlist (field-
        // tested: receiver enters IDLE/extendedStatus=LOADING and
        // never fetches any segment or part). Conservative default
        // is `false` in `capabilities_for_model`; only flip it on
        // for device classes where it's been verified to work.
        // Classic-HLS receivers get the legacy code path
        // (EXT-X-VERSION:3, whole-segment fetches), identical to
        // what we were shipping before the LL-HLS commit landed.
        let part_target_secs = if state.supports_low_latency_hls {
            Some(0.5)
        } else {
            None
        };
        // HLS over plain HTTP. We *tried* self-signed HTTPS for a
        // while (commit 7d20a22) on the theory that CAF 1.56+ refused
        // `http://` URLs even on the LAN, but the Default Media
        // Receiver runs inside the Chromecast's onboard Chrome and
        // Chrome's HLS fetch path strictly validates TLS — self-signed
        // certs get rejected before the playlist is even read, which
        // surfaces as `LOAD_FAILED` with `detailedErrorCode=None`
        // (no MEDIA_NETWORK, no decoder error — the load just dies in
        // the cert-check stage). Field-tested 2026-06: Dormitorio
        // (md=Chromecast) consistently 0/N over HTTPS, 100% over HTTP.
        // Newer cast firmwares that genuinely require HTTPS will need
        // a proper trust path (system root or pre-shared CA the
        // receiver trusts) — out of scope for ad-hoc local cast.
        //
        // Mirror the manager-negotiated `fps` into the HLS
        // segmenter so its synthetic PTS counter advances at the
        // same rate the encoder + capture pipeline is actually
        // producing frames. Without this, the segmenter's
        // `target_fps` stays at the `HlsConfig::default` value (60)
        // even when the chromecast caps to 30, the segmenter's
        // EMA needs a full segment to re-converge, and during that
        // window video PTS lags wall — which makes the audio PID
        // (whose PTS tracks wall directly) drift ahead of video on
        // the receiver and forces it to stash audio in its jitter
        // buffer for a few hundred ms. Pinning `target_fps` to the
        // real value eliminates that startup misalignment entirely.
        let stream_fps = self
            .cfg
            .as_ref()
            .map(|c| c.fps)
            .unwrap_or(HlsConfig::default().target_fps);
        let hls_config = HlsConfig {
            // Silent-AAC injection stays gated on the device flag,
            // but real audio (`audio_enabled`) overrides it inside
            // the segmenter so we never end up with both silence
            // and real samples on the same audio PID.
            inject_silent_audio: requires_audio,
            adaptive: adaptive_for_hls,
            part_target_secs,
            tls: None,
            target_fps: stream_fps,
            ..Default::default()
        };
        let sink = HlsFrameSink::start(
            "0.0.0.0:0",
            frame_rx,
            audio_rx,
            parameter_sets,
            hls_config,
        )
        .await?;
        let local_addr = sink.local_addr();
        let scheme = sink.scheme();
        let media_url = format!("{scheme}://{}:{}/index.m3u8", local_ip, local_addr.port());
        info!(
            url = %media_url,
            "chromecast HLS endpoint ready (open this URL in a player to verify)"
        );

        let ready = sink.first_segment_ready();
        let load_url = media_url.clone();
        let probe_port = local_addr.port();
        let probe_scheme = scheme;
        let load_task = tokio::spawn(async move {
            ready.await;

            // Self-test: try fetching the playlist over loopback
            // before pointing the receiver at it. If we can't reach
            // our own server, neither can the chromecast — and the
            // log makes the failure mode obvious instead of "TV
            // shows black, no errors anywhere".
            //
            // Skipped under HTTPS: the raw-TCP probe doesn't speak
            // TLS, and adding a TLS client just for this diagnostic
            // would pull rustls/webpki into the chromecast crate.
            // The HLS server itself logs every accept, so a wedged
            // listener still shows up downstream.
            if probe_scheme == "http" {
                match self_test_playlist(probe_port).await {
                    Ok(snippet) => info!(
                        bytes = snippet.len(),
                        body_head = %snippet,
                        "HLS self-test OK; sending LOAD"
                    ),
                    Err(e) => tracing::error!(
                        error = %e,
                        "HLS self-test FAILED — chromecast almost certainly can't reach the URL either"
                    ),
                }
            } else {
                info!(
                    scheme = probe_scheme,
                    "HLS self-test skipped under TLS (raw-TCP probe doesn't speak TLS)"
                );
            }

            let request_id = request_id_counter.fetch_add(1, Ordering::Relaxed);
            let media = MediaInfo {
                content_id: load_url.clone(),
                // Default Media Receiver accepts the lowercase form
                // (the spec is case-insensitive but some receiver
                // versions are picky and the lowercase variant is
                // what every reference HLS sample uses).
                content_type: "application/x-mpegurl".to_string(),
                stream_type: Some("LIVE".to_string()),
                duration: None,
            };
            let msg = match load_media_message(request_id, &transport_id, media) {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!(%e, "failed to build LOAD message");
                    return;
                }
            };
            match wire::send(&writer, &msg).await {
                Ok(()) => info!(url = %load_url, "sent LOAD to chromecast"),
                Err(e) => tracing::error!(%e, "LOAD send failed"),
            }
        });

        // Seed the keyframe ourselves so the segmenter doesn't have
        // to wait for the next IDR (which is `keyframe_interval`
        // frames away). Use try_send so we never block here either —
        // the channel was just created so it's empty and try_send
        // will succeed.
        if let Err(e) = frame_tx.try_send(first_keyframe.clone()) {
            warn!(?e, "could not seed first keyframe (channel full at init)");
        }

        self.stream = Some(StreamState {
            frame_tx,
            audio_tx,
            _sink: sink,
            load_task,
        });
        Ok(())
    }
}

/// Background read loop. Drains every message the receiver sends
/// us, dispatches them by namespace, and (optionally) signals when
/// the receiver launches the app we asked for.
///
/// `launch_watch` is `Some(app_id, tx)` during the connect handshake
/// — the loop scans every RECEIVER_STATUS and resolves the oneshot
/// once it sees an entry with our app_id and fully-populated
/// transport_id + session_id. Once resolved, the watch is dropped
/// and steady-state reads no longer touch it.
async fn read_loop<R>(
    mut reader: R,
    writer: SharedWriter,
    alive: Arc<AtomicBool>,
    mut launch_watch: Option<(String, oneshot::Sender<Result<(String, String)>>)>,
) where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut buf = BytesMut::with_capacity(8 * 1024);
    // Per-session diagnostic state. None of this is hot-path; it's
    // here to make the log a coherent timeline of receiver
    // behaviour, not a stream of disconnected events.
    let mut last_player_state: Option<String> = None;
    let mut state_entered_at = std::time::Instant::now();
    let mut last_current_time: Option<f64> = None;
    let mut last_seekable_end: Option<f64> = None;
    let mut last_status_at = std::time::Instant::now();

    loop {
        match wire::recv(&mut reader, &mut buf).await {
            Ok(Some(msg)) => {
                // PING from the receiver → must PONG or it'll close
                // us for being unresponsive. Cheap, always do it.
                if msg.namespace == namespace::HEARTBEAT
                    && msg.message_type().as_deref() == Some("PING")
                {
                    if let Ok(pong) = pong_message() {
                        if let Err(e) = wire::send(&writer, &pong).await {
                            warn!(%e, "pong send failed");
                            break;
                        }
                    }
                    continue;
                }

                // CLOSE on the connection namespace = receiver tore
                // us down. Two flavours we care about:
                //
                // - src == DEFAULT_RECEIVER_ID ("receiver-0"): the
                //   whole platform connection is gone (rare; usually
                //   means the chromecast rebooted or its TCP stack
                //   gave up).
                // - src == app transport_id (the UUID-looking one):
                //   the Default Media Receiver app exited on the
                //   receiver — observed when the receiver decides
                //   playback is unrecoverable after sustained
                //   BUFFERING. Continuing to push frames at this
                //   point is pointless; nothing on the receiver
                //   will pull them.
                //
                // Either way it's fatal for the streaming session:
                // break out, the loop's exit path flips `alive` to
                // false, and the stream manager's watchdog notices
                // and tears the rest of the pipeline down.
                if msg.namespace == namespace::CONNECTION
                    && msg.message_type().as_deref() == Some("CLOSE")
                {
                    let err = FerricastError::Connection(format!(
                        "receiver sent CLOSE on connection namespace from src={:?}; payload: {:?}",
                        msg.source_id, msg.payload_utf8
                    ));
                    if let Some((_, tx)) = launch_watch.take() {
                        let _ = tx.send(Err(err));
                    } else if msg.source_id == DEFAULT_RECEIVER_ID {
                        tracing::error!(
                            src = %msg.source_id,
                            payload = ?msg.payload_utf8,
                            "platform receiver-0 sent CLOSE — chromecast disconnected the sender entirely"
                        );
                    } else {
                        tracing::error!(
                            src = %msg.source_id,
                            payload = ?msg.payload_utf8,
                            "receiver app sent CLOSE — Default Media Receiver exited (likely after sustained BUFFERING); session is dead"
                        );
                    }
                    break;
                }

                if msg.namespace == namespace::RECEIVER {
                    info!(
                        payload = msg.payload_utf8.as_deref().unwrap_or("(none)"),
                        "RECEIVER_STATUS"
                    );
                    if let Some((app_id, _)) = launch_watch.as_ref() {
                        match try_extract_app(&msg, app_id) {
                            ExtractApp::Found(transport_id, session_id) => {
                                info!(
                                    transport_id = %transport_id,
                                    session_id = %session_id,
                                    "receiver app fully launched"
                                );
                                if let Some((_, tx)) = launch_watch.take() {
                                    let _ = tx.send(Ok((transport_id, session_id)));
                                }
                            }
                            ExtractApp::Pending(reason) => {
                                info!(%reason, "launch ack pending, sending GET_STATUS to nudge");
                                // Some firmwares emit only ONE
                                // bootstrap RECEIVER_STATUS without
                                // populated IDs and then go quiet.
                                // Poke them to re-emit the full
                                // record. Fire-and-forget; the
                                // response comes back here.
                                let req_id = msg
                                    .payload_utf8
                                    .as_deref()
                                    .and_then(|p| {
                                        serde_json::from_str::<crate::castv2::GenericPayload>(p)
                                            .ok()
                                    })
                                    .and_then(|g| g.request_id)
                                    .map(|x| x.wrapping_add(1))
                                    .unwrap_or(1);
                                if let Ok(get_status) = crate::castv2::get_status_message(req_id) {
                                    let _ = wire::send(&writer, &get_status).await;
                                }
                            }
                            ExtractApp::Skip => {}
                        }
                    }
                } else if msg.namespace == namespace::MEDIA {
                    let mtype = msg.message_type();
                    if matches!(
                        mtype.as_deref(),
                        Some("LOAD_FAILED")
                            | Some("INVALID_REQUEST")
                            | Some("INVALID_PLAYER_STATE")
                            | Some("ERROR")
                    ) {
                        // Parse `detailedErrorCode` (when present)
                        // into the named enum ported from rust_cast.
                        // The enum is exhaustive over what Google
                        // documents, so the log line goes from a
                        // bare integer to a self-describing variant
                        // (`SegmentNetwork(301)` instead of `301`).
                        let detailed_raw = msg
                            .payload_utf8
                            .as_deref()
                            .and_then(|p| serde_json::from_str::<serde_json::Value>(p).ok())
                            .and_then(|v| v.get("detailedErrorCode").and_then(|c| c.as_i64()));
                        let detailed = detailed_raw.map(MediaDetailedErrorCode::from_code);
                        tracing::error!(
                            ty = ?mtype,
                            detailed_code = ?detailed_raw,
                            detailed = ?detailed,
                            retryable = detailed.map(|d| d.is_retryable()),
                            payload = ?msg.payload_utf8,
                            "chromecast rejected our media (session is terminal)"
                        );
                        // Per-code hints aimed at the user reading
                        // the log to figure out what to do next.
                        match detailed {
                            Some(MediaDetailedErrorCode::SegmentNetwork) => tracing::error!(
                                "SegmentNetwork (301): receiver gave up fetching a segment. \
                                 On 1st/2nd-gen Chromecast this is firmware-internal — the link \
                                 is usually fine. Auto-reconnect supervisor will rebuild the \
                                 session; if it loops, lower StreamConfig::bitrate_kbps or use \
                                 newer Cast hardware."
                            ),
                            Some(MediaDetailedErrorCode::MediaSrcNotSupported) => tracing::error!(
                                "MediaSrcNotSupported (104): the receiver's decoder rejected the \
                                 bitstream. Check `max_h264_profile` in capabilities_for_model() \
                                 — older Chromecasts choke on High profile."
                            ),
                            Some(MediaDetailedErrorCode::LoadFailed) => tracing::error!(
                                "LoadFailed (905): receiver couldn't start the load. Check that \
                                 the HLS URL we advertised is reachable from the device's network \
                                 (the `HLS self-test OK` line on startup should confirm)."
                            ),
                            _ => {}
                        }
                        break;
                    } else if mtype.as_deref() == Some("MEDIA_STATUS") {
                        // Receiver pushes MEDIA_STATUS every couple
                        // of seconds. We surface state transitions
                        // at info!/warn! (loud) and steady-state
                        // repeats at debug! (quiet by default but
                        // available for deep dives via
                        // `RUST_LOG=ferricast_chromecast=debug`).
                        // Either way every line carries the full
                        // diagnostic bundle so the log is a coherent
                        // timeline.
                        let parsed: Option<MediaStatusPayload> = msg
                            .payload_utf8
                            .as_deref()
                            .and_then(|p| serde_json::from_str(p).ok());
                        let s0 = parsed.as_ref().and_then(|p| p.status.first());
                        let new_state = s0.and_then(|s| s.player_state.clone());
                        let current_time = s0.and_then(|s| s.current_time);
                        let idle_reason = s0.and_then(|s| s.idle_reason.clone());
                        let extended_status = s0.and_then(|s| s.extended_status.clone());
                        let seekable = s0.and_then(|s| s.live_seekable_range.clone());
                        let playback_rate = s0.and_then(|s| s.playback_rate);

                        // Derived diagnostics:
                        //   * advance_s: how much currentTime moved
                        //     between two consecutive MEDIA_STATUS
                        //     samples. Equals wall delta when the
                        //     player is healthy (1× playback). When
                        //     the receiver buffers, this collapses
                        //     to ~0 even though wall time keeps
                        //     advancing — exposes BUFFERING the
                        //     receiver doesn't always announce.
                        //   * lag_s: seekable.end - currentTime,
                        //     i.e. distance from the live edge.
                        //     Should hover around 1-2 segments;
                        //     growing unboundedly = player falling
                        //     behind, 301 imminent.
                        //   * window_s: seekable.end - seekable.start,
                        //     the receiver's view of how much
                        //     playable history we have. If this
                        //     shrinks to ~0 the receiver is about to
                        //     fall off the back of the playlist.
                        let now = std::time::Instant::now();
                        let wall_delta = now.duration_since(last_status_at).as_secs_f64();
                        last_status_at = now;
                        let advance_s = match (current_time, last_current_time) {
                            (Some(c), Some(p)) => Some(c - p),
                            _ => None,
                        };
                        let lag_s = match (current_time, seekable.as_ref()) {
                            (Some(c), Some(r)) => Some(r.end - c),
                            _ => None,
                        };
                        let window_s = seekable.as_ref().map(|r| r.end - r.start);
                        if let Some(c) = current_time {
                            last_current_time = Some(c);
                        }
                        if let Some(r) = seekable.as_ref() {
                            last_seekable_end = Some(r.end);
                        }
                        let _ = last_seekable_end; // referenced for future starvation watchdog

                        let state_changed = new_state != last_player_state;
                        let prev_state_dur = if state_changed {
                            let d = state_entered_at.elapsed();
                            state_entered_at = now;
                            Some(d.as_millis() as u64)
                        } else {
                            None
                        };

                        // Pick log level based on what changed.
                        // BUFFERING entry / IDLE entry get warn,
                        // other entries get info, no-change repeats
                        // get debug.
                        let level_warn =
                            matches!(new_state.as_deref(), Some("BUFFERING") | Some("IDLE"))
                                && state_changed;

                        macro_rules! emit {
                            ($macro:ident, $msg:literal) => {
                                tracing::$macro!(
                                    state = ?new_state,
                                    prev_state = ?last_player_state,
                                    prev_state_ms = ?prev_state_dur,
                                    ?current_time,
                                    ?advance_s,
                                    wall_delta_s = format_args!("{wall_delta:.2}"),
                                    ?lag_s,
                                    ?window_s,
                                    ?idle_reason,
                                    ?extended_status,
                                    ?playback_rate,
                                    $msg
                                )
                            };
                        }

                        if level_warn {
                            match new_state.as_deref() {
                                Some("BUFFERING") => {
                                    emit!(warn, "MEDIA_STATUS: chromecast entered BUFFERING")
                                }
                                Some("IDLE") => emit!(
                                    warn,
                                    "MEDIA_STATUS: chromecast entered IDLE (receiver may have stopped)"
                                ),
                                _ => {}
                            }
                        } else if state_changed {
                            emit!(info, "MEDIA_STATUS: state transition");
                        } else {
                            emit!(debug, "MEDIA_STATUS: steady");
                        }

                        if state_changed {
                            last_player_state = new_state;
                        }
                    } else {
                        debug!(
                            ty = ?mtype,
                            payload = ?msg.payload_utf8,
                            "chromecast media status"
                        );
                    }
                }
            }
            Ok(None) => {
                info!("chromecast read loop: peer closed");
                if let Some((_, tx)) = launch_watch.take() {
                    let _ = tx.send(Err(FerricastError::Connection(
                        "receiver closed before launch ack \
                         (try rebooting the chromecast — its session \
                         state may be stuck from a prior run)"
                            .into(),
                    )));
                }
                break;
            }
            Err(e) => {
                warn!(%e, "chromecast read loop terminating");
                if let Some((_, tx)) = launch_watch.take() {
                    let _ = tx.send(Err(e));
                }
                break;
            }
        }
    }
    alive.store(false, Ordering::Relaxed);
}

enum ExtractApp {
    /// Successful: the launched app is fully populated.
    Found(String, String),
    /// We saw a STATUS but our app isn't usable yet (missing,
    /// half-populated). Caller should keep waiting.
    Pending(String),
    /// Not relevant (couldn't parse, or no status block).
    Skip,
}

fn try_extract_app(msg: &CastMessage, app_id: &str) -> ExtractApp {
    let payload: ReceiverStatusPayload = match msg.parse_payload() {
        Ok(p) => p,
        Err(e) => {
            warn!(%e, payload = ?msg.payload_utf8, "RECEIVER_STATUS parse failed");
            return ExtractApp::Skip;
        }
    };
    let Some(status) = payload.status else {
        return ExtractApp::Pending("status block absent".into());
    };
    let Some(app) = status
        .applications
        .iter()
        .find(|a| a.app_id.eq_ignore_ascii_case(app_id))
    else {
        return ExtractApp::Pending(format!(
            "our app not in applications list ({} entries)",
            status.applications.len()
        ));
    };
    if app.transport_id.is_empty() || app.session_id.is_empty() {
        return ExtractApp::Pending(format!(
            "app entry incomplete (transport_id='{}', session_id='{}')",
            app.transport_id, app.session_id
        ));
    }
    ExtractApp::Found(app.transport_id.clone(), app.session_id.clone())
}

/// Background heartbeat ticker. Stops as soon as the read loop
/// flips `alive` to `false` or a ping write fails.
async fn heartbeat_loop(writer: SharedWriter, alive: Arc<AtomicBool>) {
    
    let mut tick = tokio::time::interval(HEARTBEAT_INTERVAL);
    // First tick fires immediately; skip it so we don't ping before
    // launch settles.
    tick.tick().await;
    while alive.load(Ordering::Relaxed) {
        tick.tick().await;
        let Ok(ping) = ping_message() else { break };
        if let Err(e) = wire::send(&writer, &ping).await {
            debug!(%e, "heartbeat ping failed, exiting heartbeat loop");
            break;
        }
    }
}

/// Loopback HTTP/1.0 GET against our own HLS endpoint. Returns the
/// first ~512 bytes of the response body so the caller can sanity-
/// check it actually looks like a `#EXTM3U` playlist before pointing
/// the receiver at it.
///
/// 5 s ceiling so a wedged listener doesn't keep `LOAD` from ever
/// firing — if the local server can't answer in 5 s on localhost
/// something is profoundly wrong and the receiver wouldn't have
/// fared any better.
async fn self_test_playlist(port: u16) -> Result<String> {
    let probe = async move {
        let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .map_err(|e| FerricastError::Streaming(format!("self-test connect: {e}")))?;
        tokio::io::AsyncWriteExt::write_all(
            &mut s,
            b"GET /index.m3u8 HTTP/1.0\r\nHost: localhost\r\n\r\n",
        )
        .await
        .map_err(|e| FerricastError::Streaming(format!("self-test write: {e}")))?;
        let mut buf = [0u8; 512];
        let n = tokio::io::AsyncReadExt::read(&mut s, &mut buf)
            .await
            .map_err(|e| FerricastError::Streaming(format!("self-test read: {e}")))?;
        Ok::<String, FerricastError>(String::from_utf8_lossy(&buf[..n]).into_owned())
    };
    match tokio::time::timeout(std::time::Duration::from_secs(5), probe).await {
        Ok(r) => r,
        Err(_) => Err(FerricastError::Timeout(
            "self-test fetch exceeded 5s".into(),
        )),
    }
}
