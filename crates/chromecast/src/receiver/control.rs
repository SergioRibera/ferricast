//! Receiver-side CASTV2 control channel.
//!
//! One [`ChromecastReceiverControl`] serves one TCP connection; the
//! TLS listener itself is shared by every control created from the
//! same [`super::ChromecastReceiver`], so several senders (a phone
//! plus a laptop, or a status probe next to an active cast) can be
//! connected at once.
//!
//! Internal namespaces handled inline (no [`MediaCommand`] forwarded
//! to the manager):
//!
//! - `tp.connection` — `CONNECT` opens a virtual channel; `CLOSE`
//!   tears it down. Neither gets a reply.
//! - `tp.heartbeat` — every `PING` gets an immediate `PONG`.
//! - `tp.deviceauth` — answered with an `AuthError`: the challenge
//!   must be signed by a Google-issued device certificate, which
//!   third-party receivers cannot have. Senders that enforce device
//!   authentication (Chrome, Android Cast SDK) therefore give up;
//!   senders that skip it keep going.
//! - `receiver` — `GET_STATUS`, `GET_APP_AVAILABILITY`, `LAUNCH`,
//!   `STOP`, `SET_VOLUME`, answered with `RECEIVER_STATUS` /
//!   `LAUNCH_ERROR`. Only the Default Media Receiver (`CC1AD845`) can
//!   be launched: other apps need their own web receiver and custom
//!   namespaces.
//!
//! Media-namespace commands (`LOAD`, `PLAY`, `PAUSE`, `STOP`,
//! `SEEK`, `SET_VOLUME`, …) translate into [`MediaCommand`]s that
//! ride out through [`ControlSession::next_command`]. `LOAD` is
//! answered once the manager reports the outcome through
//! [`ControlSession::report_state`]: `MEDIA_STATUS` on success,
//! `LOAD_FAILED` on [`PlaybackState::Error`].

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use bytes::BytesMut;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt, ReadHalf, WriteHalf};
use tokio::net::TcpStream;
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinHandle;
use uuid::Uuid;

use ferricast_core::{
    ControlSession, FerricastError, MediaCommand, PlaybackState, PlaybackStatus, RemoteSender,
    Result,
};

use crate::castv2::{CastMessage, DEFAULT_MEDIA_RECEIVER_APP_ID, MAX_MESSAGE_SIZE, namespace};

use super::Shared;
use super::listener::{Admitted, Listener};

/// Device-authentication namespace. Senders issue an `AuthChallenge`
/// here right after connecting.
pub(super) const DEVICE_AUTH_NAMESPACE: &str = "urn:x-cast:com.google.cast.tp.deviceauth";
/// `DeviceAuthMessage { error: AuthError { error_type: INTERNAL_ERROR } }`
/// in protobuf wire format (field 3, length 2, { field 1 = 0 }).
const DEVICE_AUTH_ERROR: [u8; 4] = [0x1a, 0x02, 0x08, 0x00];
/// The id this receiver answers platform messages from.
const PLATFORM_ID: &str = "receiver-0";

// Cast `supportedMediaCommands` bits.
const CMD_PAUSE: u32 = 1;
const CMD_SEEK: u32 = 2;
const CMD_STREAM_VOLUME: u32 = 4;
const CMD_STREAM_MUTE: u32 = 8;

pub(super) type ServerStream = tokio_rustls::server::TlsStream<TcpStream>;
pub(super) type ServerWriter = Arc<Mutex<WriteHalf<ServerStream>>>;

async fn server_send(writer: &ServerWriter, msg: &CastMessage) -> Result<()> {
    let bytes = msg
        .encode_length_prefixed()
        .map_err(|e| FerricastError::Receiver(format!("encode cast message: {e}")))?;
    let mut w = writer.lock().await;
    w.write_all(&bytes)
        .await
        .map_err(|e| FerricastError::Receiver(format!("write cast message: {e}")))?;
    w.flush()
        .await
        .map_err(|e| FerricastError::Receiver(format!("flush: {e}")))?;
    Ok(())
}

/// Answers a device-auth challenge with an auth error: the challenge
/// must be signed by a Google-issued device certificate, which
/// third-party receivers cannot have.
pub(super) async fn answer_device_auth(
    writer: &ServerWriter,
    challenge: &CastMessage,
) -> Result<()> {
    tracing::debug!(
        source = %challenge.source_id,
        "Chromecast receiver: device-auth challenge, answering with AuthError"
    );
    let reply = CastMessage::new_binary(
        &challenge.destination_id,
        &challenge.source_id,
        DEVICE_AUTH_NAMESPACE,
        DEVICE_AUTH_ERROR.to_vec(),
    );
    server_send(writer, &reply).await
}

/// Next message, or `None` once the peer closed the connection, with
/// or without a TLS `close_notify` (phones drop sockets without one).
pub(super) async fn server_recv<R: tokio::io::AsyncRead + Unpin>(
    reader: &mut R,
    buf: &mut BytesMut,
) -> Result<Option<CastMessage>> {
    loop {
        match CastMessage::decode_length_prefixed(buf) {
            Ok(Some(msg)) => return Ok(Some(msg)),
            Ok(None) => {}
            Err(e) => {
                return Err(FerricastError::Receiver(format!(
                    "decode cast message: {e}"
                )));
            }
        }
        if buf.len() > MAX_MESSAGE_SIZE + 8 {
            return Err(FerricastError::Receiver(
                "incoming cast frame exceeds MAX_MESSAGE_SIZE".into(),
            ));
        }
        let n = match reader.read_buf(buf).await {
            Ok(n) => n,
            Err(e) if is_peer_gone(&e) => return Ok(None),
            Err(e) => return Err(FerricastError::Receiver(format!("read: {e}"))),
        };
        if n == 0 {
            return Ok(None);
        }
    }
}

/// Whether a read error just means the peer went away.
fn is_peer_gone(e: &std::io::Error) -> bool {
    use std::io::ErrorKind;
    matches!(
        e.kind(),
        ErrorKind::UnexpectedEof
            | ErrorKind::ConnectionReset
            | ErrorKind::ConnectionAborted
            | ErrorKind::BrokenPipe
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlayerState {
    Idle,
    Buffering,
    Playing,
    Paused,
}

impl PlayerState {
    fn as_str(self) -> &'static str {
        match self {
            PlayerState::Idle => "IDLE",
            PlayerState::Buffering => "BUFFERING",
            PlayerState::Playing => "PLAYING",
            PlayerState::Paused => "PAUSED",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IdleReason {
    Cancelled,
    Interrupted,
    Finished,
    Error,
}

impl IdleReason {
    fn as_str(self) -> &'static str {
        match self {
            IdleReason::Cancelled => "CANCELLED",
            IdleReason::Interrupted => "INTERRUPTED",
            IdleReason::Finished => "FINISHED",
            IdleReason::Error => "ERROR",
        }
    }
}

/// The media session created by the latest LOAD.
#[derive(Debug, Clone)]
struct MediaState {
    media_session_id: i64,
    content_id: String,
    content_type: String,
    is_live: bool,
    /// Media duration in seconds, once the receiver reported it.
    duration: Option<f64>,
    player_state: PlayerState,
    idle_reason: Option<IdleReason>,
    /// Position at `playing_since` (or the frozen position when not
    /// playing), seconds.
    position: f64,
    playing_since: Option<Instant>,
    /// A STOP came from the sender; the next IDLE is `CANCELLED`.
    stop_requested: bool,
}

impl MediaState {
    fn current_time(&self) -> f64 {
        self.position
            + self
                .playing_since
                .map_or(0.0, |since| since.elapsed().as_secs_f64())
    }

    fn freeze_clock(&mut self) {
        self.position = self.current_time();
        self.playing_since = None;
    }

    fn set_state(&mut self, state: PlayerState) {
        match state {
            PlayerState::Playing => {
                if self.playing_since.is_none() {
                    self.playing_since = Some(Instant::now());
                }
            }
            PlayerState::Idle | PlayerState::Buffering | PlayerState::Paused => {
                self.freeze_clock();
            }
        }
        if state != PlayerState::Idle {
            self.idle_reason = None;
        }
        self.player_state = state;
    }

    /// Takes position, duration and liveness from the pipeline: its
    /// position replaces the locally extrapolated one (re-basing the
    /// clock when playing), and what the media really is overrides the
    /// sender's LOAD `streamType`.
    #[expect(
        clippy::cast_precision_loss,
        reason = "media positions in microseconds stay far below 2^52"
    )]
    fn apply(&mut self, status: &PlaybackStatus) {
        if let Some(position_us) = status.position_us {
            self.position = position_us as f64 / 1_000_000.0;
            if self.playing_since.is_some() {
                self.playing_since = Some(Instant::now());
            }
        }
        self.is_live = status.is_live;
        self.duration = status
            .duration_us
            .filter(|_| !status.is_live)
            .map(|d| d as f64 / 1_000_000.0);
    }

    fn supported_commands(&self) -> u32 {
        let base = CMD_PAUSE | CMD_STREAM_VOLUME | CMD_STREAM_MUTE;
        if self.is_live { base } else { base | CMD_SEEK }
    }
}

#[derive(Debug)]
struct SessionState {
    /// Sender id of the latest CONNECT; destination of status
    /// updates not tied to a request.
    sender_id: String,
    /// Transport id of the launched app (destination of app
    /// namespace messages from the sender).
    transport_id: String,
    session_id: String,
    app_launched: bool,
    media: Option<MediaState>,
    next_media_session_id: i64,
    /// `requestId` of a LOAD awaiting its outcome.
    pending_load: Option<i64>,
}

impl SessionState {
    fn new(sender_id: String) -> Self {
        Self {
            sender_id,
            transport_id: format!("transport-{}", Uuid::new_v4().simple()),
            session_id: Uuid::new_v4().to_string(),
            app_launched: false,
            media: None,
            next_media_session_id: 1,
            pending_load: None,
        }
    }
}

type StateRef = Arc<Mutex<Option<SessionState>>>;

pub struct ChromecastReceiverControl {
    listener: Arc<Listener>,
    shared: Arc<Shared>,
    reader_task: Option<JoinHandle<()>>,
    commands_rx: Option<mpsc::Receiver<Result<MediaCommand>>>,
    writer: Option<ServerWriter>,
    state: StateRef,
}

impl ChromecastReceiverControl {
    pub(super) fn new(listener: Arc<Listener>, shared: Arc<Shared>) -> Self {
        Self {
            listener,
            shared,
            reader_task: None,
            commands_rx: None,
            writer: None,
            state: Arc::new(Mutex::new(None)),
        }
    }

    /// Address the shared TLS listener is bound to.
    ///
    /// # Errors
    /// When the socket can no longer report its address.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.listener.local_addr())
    }
}

impl Drop for ChromecastReceiverControl {
    fn drop(&mut self) {
        if let Some(task) = self.reader_task.take() {
            task.abort();
        }
    }
}

impl ControlSession for ChromecastReceiverControl {
    async fn accept(&mut self) -> Result<RemoteSender> {
        let Admitted {
            peer,
            reader,
            writer,
            buf,
            first,
        } = self.listener.next().await?;
        self.writer = Some(writer.clone());

        let (cmd_tx, cmd_rx) = mpsc::channel::<Result<MediaCommand>>(32);
        self.commands_rx = Some(cmd_rx);
        let ctx = Ctx {
            writer,
            cmd_tx,
            state: self.state.clone(),
            shared: self.shared.clone(),
        };
        self.reader_task = Some(tokio::spawn(run_reader(reader, buf, first, ctx)));

        Ok(RemoteSender {
            id: Uuid::new_v4(),
            addr: peer.ip(),
            name: None,
        })
    }

    async fn next_command(&mut self) -> Result<MediaCommand> {
        let rx = self
            .commands_rx
            .as_mut()
            .ok_or_else(|| FerricastError::Receiver("next_command before accept".into()))?;
        match rx.recv().await {
            Some(r) => r,
            // The reader ends on EOF: the sender hung up.
            None => Err(FerricastError::RemoteStopped(
                "sender closed the connection".into(),
            )),
        }
    }

    async fn report_state(&mut self, state: PlaybackState) -> Result<()> {
        self.publish(state, None).await
    }

    async fn report_status(&mut self, status: PlaybackStatus) -> Result<()> {
        let state = status.state.clone();
        self.publish(state, Some(status)).await
    }

    async fn close(&mut self) -> Result<()> {
        if let Some(task) = self.reader_task.take() {
            task.abort();
        }
        if let Some(w) = self.writer.take() {
            let _ = w.lock().await.shutdown().await;
        }
        self.commands_rx.take();
        *self.state.lock().await = None;
        Ok(())
    }

    fn is_alive(&self) -> bool {
        self.reader_task.as_ref().is_some_and(|t| !t.is_finished())
    }
}

impl ChromecastReceiverControl {
    /// Applies a state change (and, from `report_status`, the position,
    /// duration and liveness) to the media session and sends the
    /// resulting `MEDIA_STATUS`, preceded by `LOAD_FAILED` when a
    /// pending LOAD failed.
    async fn publish(
        &mut self,
        state: PlaybackState,
        status: Option<PlaybackStatus>,
    ) -> Result<()> {
        let Some(writer) = self.writer.clone() else {
            return Ok(());
        };
        let volume = self.shared.volume();
        let mut guard = self.state.lock().await;
        let Some(s) = guard.as_mut() else {
            return Ok(());
        };
        let Some(media) = s.media.as_mut() else {
            return Ok(());
        };
        let mut load_failed = None;
        match &state {
            PlaybackState::Playing => media.set_state(PlayerState::Playing),
            PlaybackState::Paused => media.set_state(PlayerState::Paused),
            PlaybackState::Buffering => media.set_state(PlayerState::Buffering),
            PlaybackState::Idle => {
                media.set_state(PlayerState::Idle);
                media.idle_reason = Some(if media.stop_requested {
                    IdleReason::Cancelled
                } else {
                    IdleReason::Interrupted
                });
            }
            PlaybackState::Ended => {
                media.set_state(PlayerState::Idle);
                media.idle_reason = Some(IdleReason::Finished);
            }
            PlaybackState::Error(reason) => {
                media.set_state(PlayerState::Idle);
                media.idle_reason = Some(IdleReason::Error);
                load_failed = s
                    .pending_load
                    .take()
                    .map(|request_id| (request_id, reason.clone()));
            }
        }
        if let Some(status) = status {
            media.apply(&status);
        }
        let request_id = s.pending_load.take();
        let mut messages = Vec::new();
        if let Some((request_id, reason)) = load_failed {
            tracing::warn!(%reason, request_id, "Chromecast receiver: LOAD_FAILED");
            messages.push(media_message(
                s,
                &json!({ "type": "LOAD_FAILED", "requestId": request_id, "reason": "LOAD_FAILED" }),
            )?);
        }
        messages.push(build_media_status(s, request_id, volume)?);
        drop(guard);
        for msg in &messages {
            server_send(&writer, msg).await?;
        }
        Ok(())
    }
}

/// What the reader task needs to answer and forward messages.
struct Ctx {
    writer: ServerWriter,
    cmd_tx: mpsc::Sender<Result<MediaCommand>>,
    state: StateRef,
    shared: Arc<Shared>,
}

impl Ctx {
    async fn send(&self, msg: &CastMessage) -> Result<()> {
        server_send(&self.writer, msg).await
    }

    async fn forward(&self, cmd: MediaCommand) {
        let _ = self.cmd_tx.send(Ok(cmd)).await;
    }
}

/// Reader loop: answers platform namespaces inline and forwards
/// media commands to the manager.
async fn run_reader(
    mut reader: ReadHalf<ServerStream>,
    mut buf: BytesMut,
    first: CastMessage,
    ctx: Ctx,
) {
    if let Err(e) = dispatch(&first, &ctx).await {
        tracing::info!(%e, "Chromecast receiver: connection lost");
        return;
    }
    loop {
        match server_recv(&mut reader, &mut buf).await {
            Ok(Some(msg)) => {
                if let Err(e) = dispatch(&msg, &ctx).await {
                    // Only write failures land here: the connection
                    // is gone.
                    tracing::info!(%e, "Chromecast receiver: connection lost");
                    break;
                }
            }
            Ok(None) => {
                tracing::info!("Chromecast receiver: sender closed connection");
                break;
            }
            Err(e) => {
                let _ = ctx.cmd_tx.send(Err(e)).await;
                break;
            }
        }
    }
}

/// Handles one message. Malformed payloads are logged and skipped;
/// only failures to write a reply are returned.
async fn dispatch(msg: &CastMessage, ctx: &Ctx) -> Result<()> {
    let ns = msg.namespace.as_str();

    if ns == DEVICE_AUTH_NAMESPACE {
        return answer_device_auth(&ctx.writer, msg).await;
    }

    let Some(payload) = msg.payload_utf8.as_deref() else {
        tracing::debug!(ns, "Chromecast receiver: ignoring binary message");
        return Ok(());
    };
    let value: Value = match serde_json::from_str(payload) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(ns, %e, "Chromecast receiver: unparseable payload ignored");
            return Ok(());
        }
    };
    let msg_type = value.get("type").and_then(Value::as_str).unwrap_or("");
    let request_id = value.get("requestId").and_then(Value::as_i64);

    tracing::debug!(
        ns,
        msg_type,
        ?request_id,
        source = %msg.source_id,
        dest = %msg.destination_id,
        "cast← receiver"
    );

    match ns {
        namespace::CONNECTION => {
            if msg_type == "CONNECT" {
                let mut guard = ctx.state.lock().await;
                match guard.as_mut() {
                    Some(s) => s.sender_id.clone_from(&msg.source_id),
                    None => *guard = Some(SessionState::new(msg.source_id.clone())),
                }
            }
            // CLOSE: the sender tears the virtual channel down; the
            // TCP connection follows.
            Ok(())
        }
        namespace::HEARTBEAT => {
            if msg_type == "PING" {
                let pong = CastMessage::new_json(
                    &msg.destination_id,
                    &msg.source_id,
                    namespace::HEARTBEAT,
                    &json!({ "type": "PONG" }),
                )
                .map_err(|e| FerricastError::Receiver(format!("encode PONG: {e}")))?;
                ctx.send(&pong).await?;
            }
            Ok(())
        }
        namespace::RECEIVER => dispatch_receiver(msg, &value, msg_type, request_id, ctx).await,
        namespace::MEDIA => dispatch_media(msg, &value, msg_type, request_id, ctx).await,
        _ => {
            tracing::info!(
                ns,
                msg_type,
                source = %msg.source_id,
                payload_head = %payload.chars().take(160).collect::<String>(),
                "Chromecast receiver: namespace not implemented — ignoring"
            );
            Ok(())
        }
    }
}

async fn dispatch_receiver(
    msg: &CastMessage,
    value: &Value,
    msg_type: &str,
    request_id: Option<i64>,
    ctx: &Ctx,
) -> Result<()> {
    let reply_to = |payload: &Value| {
        CastMessage::new_json(
            &msg.destination_id,
            &msg.source_id,
            namespace::RECEIVER,
            payload,
        )
        .map_err(|e| FerricastError::Receiver(format!("encode receiver reply: {e}")))
    };
    match msg_type {
        "GET_STATUS" => {}
        "GET_APP_AVAILABILITY" => {
            let ids: Vec<&str> = value
                .get("appId")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            let availability: serde_json::Map<String, Value> = ids
                .into_iter()
                .map(|id| {
                    let state = if id == DEFAULT_MEDIA_RECEIVER_APP_ID {
                        "APP_AVAILABLE"
                    } else {
                        "APP_UNAVAILABLE"
                    };
                    (id.to_string(), Value::from(state))
                })
                .collect();
            let reply = reply_to(&json!({
                "type": "GET_APP_AVAILABILITY",
                "requestId": request_id.unwrap_or(0),
                "responseType": "GET_APP_AVAILABILITY",
                "availability": availability,
            }))?;
            return ctx.send(&reply).await;
        }
        "LAUNCH" => {
            let app_id = value
                .get("appId")
                .and_then(Value::as_str)
                .unwrap_or(DEFAULT_MEDIA_RECEIVER_APP_ID);
            if app_id != DEFAULT_MEDIA_RECEIVER_APP_ID {
                tracing::info!(%app_id, "Chromecast receiver: LAUNCH of unsupported app refused");
                let reply = reply_to(&json!({
                    "type": "LAUNCH_ERROR",
                    "requestId": request_id.unwrap_or(0),
                    "reason": "NOT_FOUND",
                }))?;
                return ctx.send(&reply).await;
            }
            tracing::info!(sender = %msg.source_id, "Chromecast receiver: LAUNCH");
            ctx.state
                .lock()
                .await
                .get_or_insert_with(|| SessionState::new(msg.source_id.clone()))
                .app_launched = true;
            ctx.forward(MediaCommand::LaunchApp {
                app_id: app_id.to_string(),
            })
            .await;
        }
        "STOP" => {
            if let Some(s) = ctx.state.lock().await.as_mut() {
                s.app_launched = false;
                if let Some(media) = s.media.as_mut() {
                    media.stop_requested = true;
                }
            }
            ctx.forward(MediaCommand::StopApp).await;
        }
        "SET_VOLUME" => {
            if let Some(vol) = value.get("volume") {
                if let Some(level) = vol.get("level").and_then(Value::as_f64) {
                    let level = level.clamp(0.0, 1.0);
                    ctx.shared.set_volume_level(level);
                    ctx.forward(MediaCommand::SetSystemVolume(narrow(level)))
                        .await;
                }
                if let Some(muted) = vol.get("muted").and_then(Value::as_bool) {
                    ctx.shared.set_muted(muted);
                    ctx.forward(MediaCommand::SetSystemMute(muted)).await;
                }
            }
        }
        _ => {
            tracing::debug!(msg_type, "Chromecast receiver: receiver message ignored");
            return Ok(());
        }
    }
    let reply = {
        let mut guard = ctx.state.lock().await;
        let s = guard.get_or_insert_with(|| SessionState::new(msg.source_id.clone()));
        build_receiver_status(s, request_id, &msg.source_id, ctx.shared.volume())?
    };
    ctx.send(&reply).await
}

#[expect(
    clippy::too_many_lines,
    reason = "one match over the media namespace's message types"
)]
async fn dispatch_media(
    msg: &CastMessage,
    value: &Value,
    msg_type: &str,
    request_id: Option<i64>,
    ctx: &Ctx,
) -> Result<()> {
    match msg_type {
        "LOAD" => return handle_load(msg, value, request_id, ctx).await,
        "QUEUE_LOAD" => {
            // Play the start item; queues are not kept.
            let start = value.get("startIndex").and_then(Value::as_u64).unwrap_or(0);
            let item = value
                .get("items")
                .and_then(Value::as_array)
                .and_then(|items| items.get(usize::try_from(start).unwrap_or(0)))
                .cloned()
                .unwrap_or(Value::Null);
            let mut load = item;
            if let Some(obj) = load.as_object_mut() {
                obj.insert("autoplay".into(), Value::Bool(true));
            }
            return handle_load(msg, &load, request_id, ctx).await;
        }
        "PLAY" => {
            update_media(ctx, |m| m.set_state(PlayerState::Playing)).await;
            ctx.forward(MediaCommand::Play).await;
        }
        "PAUSE" => {
            update_media(ctx, |m| m.set_state(PlayerState::Paused)).await;
            ctx.forward(MediaCommand::Pause).await;
        }
        "STOP" => {
            update_media(ctx, |m| {
                m.stop_requested = true;
                m.set_state(PlayerState::Idle);
                m.idle_reason = Some(IdleReason::Cancelled);
            })
            .await;
            ctx.forward(MediaCommand::Stop).await;
        }
        "SEEK" => {
            if let Some(pos) = value.get("currentTime").and_then(Value::as_f64) {
                let pos = pos.max(0.0);
                update_media(ctx, |m| {
                    m.position = pos;
                    if m.playing_since.is_some() {
                        m.playing_since = Some(Instant::now());
                    }
                })
                .await;
                ctx.forward(MediaCommand::Seek {
                    position_us: secs_to_us(pos),
                })
                .await;
            }
            if let Some(resume) = value.get("resumeState").and_then(Value::as_str) {
                match resume {
                    "PLAYBACK_START" => {
                        update_media(ctx, |m| m.set_state(PlayerState::Playing)).await;
                        ctx.forward(MediaCommand::Play).await;
                    }
                    "PLAYBACK_PAUSE" => {
                        update_media(ctx, |m| m.set_state(PlayerState::Paused)).await;
                        ctx.forward(MediaCommand::Pause).await;
                    }
                    _ => {}
                }
            }
        }
        "SET_VOLUME" => {
            if let Some(vol) = value.get("volume") {
                if let Some(level) = vol.get("level").and_then(Value::as_f64) {
                    let level = level.clamp(0.0, 1.0);
                    ctx.shared.set_volume_level(level);
                    ctx.forward(MediaCommand::SetVolume(narrow(level))).await;
                }
                if let Some(muted) = vol.get("muted").and_then(Value::as_bool) {
                    ctx.shared.set_muted(muted);
                    ctx.forward(MediaCommand::SetMute(muted)).await;
                }
            }
        }
        "GET_STATUS" => {}
        "SET_PLAYBACK_RATE" => {
            if let Some(rate) = value.get("playbackRate").and_then(Value::as_f64) {
                ctx.forward(MediaCommand::SetPlaybackRate(narrow(rate)))
                    .await;
            }
        }
        "EDIT_TRACKS_INFO" => {
            let active = value
                .get("activeTrackIds")
                .and_then(Value::as_array)
                .map(|arr| arr.iter().filter_map(Value::as_i64).collect::<Vec<_>>())
                .unwrap_or_default();
            ctx.forward(MediaCommand::EditTracks(ferricast_core::TrackSelection {
                active_track_ids: active,
            }))
            .await;
        }
        _ => {
            tracing::debug!(msg_type, "Chromecast receiver: media message ignored");
            return Ok(());
        }
    }
    send_media_status(ctx, request_id, &msg.source_id).await
}

async fn update_media(ctx: &Ctx, f: impl FnOnce(&mut MediaState)) {
    if let Some(media) = ctx
        .state
        .lock()
        .await
        .as_mut()
        .and_then(|s| s.media.as_mut())
    {
        f(media);
    }
}

async fn handle_load(
    msg: &CastMessage,
    value: &Value,
    request_id: Option<i64>,
    ctx: &Ctx,
) -> Result<()> {
    let media = value.get("media");
    let field = |name: &str| {
        media
            .and_then(|m| m.get(name))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
    };
    // Newer senders put the URL in `contentUrl` and an opaque id in
    // `contentId`.
    let Some(url) = field("contentUrl").or_else(|| field("contentId")) else {
        tracing::warn!("LOAD without media.contentUrl / contentId");
        let reply = CastMessage::new_json(
            &msg.destination_id,
            &msg.source_id,
            namespace::MEDIA,
            &json!({
                "type": "LOAD_FAILED",
                "requestId": request_id.unwrap_or(0),
                "reason": "INVALID_PARAMS",
            }),
        )
        .map_err(|e| FerricastError::Receiver(format!("encode LOAD_FAILED: {e}")))?;
        return ctx.send(&reply).await;
    };
    let url = url.to_string();
    let content_type = field("contentType").map(str::to_string);
    let is_live = field("streamType") == Some("LIVE");
    let autoplay = value
        .get("autoplay")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let current_time = value
        .get("currentTime")
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
        .max(0.0);
    let metadata = media
        .and_then(|m| m.get("metadata"))
        .map(extract_metadata)
        .unwrap_or_default();
    tracing::info!(%url, ?content_type, sender = %msg.source_id, "Chromecast receiver: LOAD");

    let superseded = {
        let mut guard = ctx.state.lock().await;
        let s = guard.get_or_insert_with(|| SessionState::new(msg.source_id.clone()));
        let media_session_id = s.next_media_session_id;
        s.next_media_session_id += 1;
        s.media = Some(MediaState {
            media_session_id,
            content_id: url.clone(),
            content_type: content_type
                .clone()
                .unwrap_or_else(|| "application/x-mpegurl".into()),
            is_live,
            duration: None,
            player_state: PlayerState::Buffering,
            idle_reason: None,
            position: current_time,
            playing_since: None,
            stop_requested: false,
        });
        std::mem::replace(&mut s.pending_load, request_id)
            .map(|old| media_message(s, &json!({ "type": "LOAD_CANCELLED", "requestId": old })))
    };
    if let Some(cancelled) = superseded {
        ctx.send(&cancelled?).await?;
    }

    // The reply (MEDIA_STATUS or LOAD_FAILED) goes out once the
    // manager reports the outcome through `report_state`.
    ctx.forward(MediaCommand::Load {
        url,
        content_type,
        start_time_us: (current_time > 0.0).then(|| secs_to_us(current_time)),
        autoplay,
        metadata,
    })
    .await;
    Ok(())
}

async fn send_media_status(ctx: &Ctx, request_id: Option<i64>, destination: &str) -> Result<()> {
    let reply = {
        let guard = ctx.state.lock().await;
        let Some(s) = guard.as_ref() else {
            return Ok(());
        };
        let mut msg = build_media_status(s, request_id, ctx.shared.volume())?;
        msg.destination_id = destination.to_string();
        msg
    };
    ctx.send(&reply).await
}

/// A sender's volume level or playback rate as the `f32` the media
/// commands carry.
#[expect(
    clippy::cast_possible_truncation,
    reason = "volume levels (0..=1) and playback rates fit f32"
)]
fn narrow(value: f64) -> f32 {
    value as f32
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "clamped non-negative, and far below u64::MAX µs for any real position"
)]
fn secs_to_us(secs: f64) -> u64 {
    // Non-negative and far below u64::MAX µs for any real position.
    (secs.max(0.0) * 1_000_000.0) as u64
}

fn media_message(s: &SessionState, payload: &Value) -> Result<CastMessage> {
    CastMessage::new_json(&s.transport_id, &s.sender_id, namespace::MEDIA, payload)
        .map_err(|e| FerricastError::Receiver(format!("encode media message: {e}")))
}

fn volume_json(volume: super::Volume) -> Value {
    json!({ "level": volume.level, "muted": volume.muted })
}

fn build_receiver_status(
    s: &SessionState,
    request_id: Option<i64>,
    destination: &str,
    volume: super::Volume,
) -> Result<CastMessage> {
    let mut applications = Vec::new();
    if s.app_launched {
        applications.push(json!({
            "appId": DEFAULT_MEDIA_RECEIVER_APP_ID,
            "universalAppId": DEFAULT_MEDIA_RECEIVER_APP_ID,
            "displayName": "Default Media Receiver",
            "isIdleScreen": false,
            "launchedFromCloud": false,
            "namespaces": [
                { "name": namespace::MEDIA },
                { "name": namespace::CONNECTION },
                { "name": namespace::HEARTBEAT },
            ],
            "sessionId": s.session_id,
            "statusText": "Ferricast",
            "transportId": s.transport_id,
        }));
    }
    let mut volume = volume_json(volume);
    if let Some(obj) = volume.as_object_mut() {
        obj.insert("controlType".into(), Value::from("attenuation"));
        obj.insert("stepInterval".into(), Value::from(0.05));
    }
    let payload = json!({
        "type": "RECEIVER_STATUS",
        "requestId": request_id.unwrap_or(0),
        "status": {
            "applications": applications,
            "isActiveInput": true,
            "isStandBy": false,
            "volume": volume,
        }
    });
    CastMessage::new_json(PLATFORM_ID, destination, namespace::RECEIVER, &payload)
        .map_err(|e| FerricastError::Receiver(format!("encode RECEIVER_STATUS: {e}")))
}

fn build_media_status(
    s: &SessionState,
    request_id: Option<i64>,
    volume: super::Volume,
) -> Result<CastMessage> {
    let status: Vec<Value> = s
        .media
        .iter()
        .map(|m| {
            let mut entry = json!({
                "mediaSessionId": m.media_session_id,
                "playbackRate": 1,
                "playerState": m.player_state.as_str(),
                "currentTime": m.current_time(),
                "supportedMediaCommands": m.supported_commands(),
                "volume": volume_json(volume),
                "media": {
                    "contentId": m.content_id,
                    "contentType": m.content_type,
                    "streamType": if m.is_live { "LIVE" } else { "BUFFERED" },
                },
            });
            if let (Some(duration), Some(media)) = (m.duration, entry.get_mut("media")) {
                media["duration"] = Value::from(duration);
            }
            if let (Some(reason), Some(obj)) = (m.idle_reason, entry.as_object_mut()) {
                obj.insert("idleReason".into(), Value::from(reason.as_str()));
            }
            entry
        })
        .collect();
    media_message(
        s,
        &json!({
            "type": "MEDIA_STATUS",
            "requestId": request_id.unwrap_or(0),
            "status": status,
        }),
    )
}

fn extract_metadata(v: &Value) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    if let Value::Object(map) = v {
        for (k, val) in map {
            let s = match val {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            out.insert(k.clone(), s);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn media(is_live: bool) -> MediaState {
        MediaState {
            media_session_id: 1,
            content_id: "http://x/a.m3u8".into(),
            content_type: "application/x-mpegurl".into(),
            is_live,
            duration: None,
            player_state: PlayerState::Buffering,
            idle_reason: None,
            position: 3.0,
            playing_since: None,
            stop_requested: false,
        }
    }

    #[test]
    fn clock_runs_only_while_playing() {
        let mut m = media(false);
        assert!((m.current_time() - 3.0).abs() < 1e-9);
        m.set_state(PlayerState::Playing);
        std::thread::sleep(std::time::Duration::from_millis(50));
        m.set_state(PlayerState::Paused);
        let frozen = m.current_time();
        assert!((3.05..3.5).contains(&frozen), "{frozen}");
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!((m.current_time() - frozen).abs() < 1e-9);
    }

    #[test]
    fn live_media_cannot_seek() {
        assert_eq!(media(true).supported_commands() & CMD_SEEK, 0);
        assert_ne!(media(false).supported_commands() & CMD_SEEK, 0);
    }

    #[test]
    fn media_status_reports_idle_reason_and_volume() {
        let mut s = SessionState::new("sender-1".into());
        let mut m = media(false);
        m.set_state(PlayerState::Idle);
        m.idle_reason = Some(IdleReason::Finished);
        s.media = Some(m);
        let msg = build_media_status(
            &s,
            Some(7),
            super::super::Volume {
                level: 0.25,
                muted: true,
            },
        )
        .unwrap();
        let v: Value = serde_json::from_str(msg.payload_utf8.as_deref().unwrap()).unwrap();
        assert_eq!(v["requestId"], 7);
        let st = &v["status"][0];
        assert_eq!(st["playerState"], "IDLE");
        assert_eq!(st["idleReason"], "FINISHED");
        assert_eq!(st["volume"]["level"], 0.25);
        assert_eq!(st["volume"]["muted"], true);
    }

    #[test]
    fn reported_status_sets_position_duration_and_stream_type() {
        let mut s = SessionState::new("sender-1".into());
        let mut m = media(true);
        m.set_state(PlayerState::Playing);
        m.apply(&PlaybackStatus {
            state: PlaybackState::Playing,
            position_us: Some(12_500_000),
            duration_us: Some(60_000_000),
            is_live: false,
        });
        s.media = Some(m);
        let msg = build_media_status(&s, None, super::super::Volume::default()).unwrap();
        let v: Value = serde_json::from_str(msg.payload_utf8.as_deref().unwrap()).unwrap();
        let st = &v["status"][0];
        assert_eq!(st["media"]["streamType"], "BUFFERED");
        assert_eq!(st["media"]["duration"], 60.0);
        let at = st["currentTime"].as_f64().unwrap();
        assert!((12.5..12.6).contains(&at), "{at}");
        assert_ne!(
            st["supportedMediaCommands"].as_u64().unwrap() & u64::from(CMD_SEEK),
            0
        );

        let mut live = media(false);
        live.apply(&PlaybackStatus {
            state: PlaybackState::Playing,
            position_us: None,
            duration_us: Some(30_000_000),
            is_live: true,
        });
        assert!(live.is_live);
        assert_eq!(live.duration, None);
        assert!((live.position - 3.0).abs() < 1e-9);
    }

    #[test]
    fn media_status_without_media_is_empty() {
        let s = SessionState::new("sender-1".into());
        let msg = build_media_status(&s, None, super::super::Volume::default()).unwrap();
        let v: Value = serde_json::from_str(msg.payload_utf8.as_deref().unwrap()).unwrap();
        assert_eq!(v["status"], json!([]));
    }
}
