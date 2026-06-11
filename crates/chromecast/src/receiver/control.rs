//! Receiver-side CASTV2 control channel.
//!
//! Owns the TLS server listening on the advertised port. One sender
//! at a time: real-world senders (YouTube, Spotify, VLC's cast UI)
//! only ever open one virtual control session per receiver — they
//! disconnect first if the user picks a new one. The
//! [`ControlSession::accept`] loop reflects this; concurrent senders
//! would require splitting state per `sender_id`, which is
//! follow-up work.
//!
//! Internal namespaces handled inline (no [`MediaCommand`] forwarded
//! to the manager):
//!
//! - `connection` — `CONNECT` opens a virtual channel; `CLOSE`
//!   tears it down. We don't reply to either.
//! - `heartbeat` — every `PING` gets an immediate `PONG`. Stock
//!   senders drop the connection after ~10 s of unacked pings.
//! - `receiver` — `GET_STATUS` returns the cached
//!   `RECEIVER_STATUS`; `LAUNCH` registers the app id, assigns
//!   `sessionId` + `transportId`, replies with a populated status.
//!
//! Media-namespace commands (`LOAD`, `PLAY`, `PAUSE`, `STOP`,
//! `SEEK`, queue ops, …) translate into [`MediaCommand`] variants
//! and ride out via the `commands_tx` mpsc; the manager's pipeline
//! consumes them through [`ControlSession::next_command`].

use std::net::SocketAddr;
use std::sync::Arc;

use bytes::BytesMut;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt, ReadHalf, WriteHalf, split};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinHandle;
use uuid::Uuid;

use ferricast_core::{
    ControlSession, FerricastError, MediaCommand, PlaybackState, RemoteSender, Result,
};

use crate::castv2::{CastMessage, MAX_MESSAGE_SIZE, namespace, DEFAULT_MEDIA_RECEIVER_APP_ID};

use super::tls::build_server_config;

type ServerStream = tokio_rustls::server::TlsStream<TcpStream>;
type ServerWriter = Arc<Mutex<WriteHalf<ServerStream>>>;

/// Receiver-side wire mechanics, mirror of `wire.rs` but for the
/// server TLS type. Stays inside this module because the sender
/// side has no use for it and exposing it would force generics
/// everywhere upstream.
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

async fn server_recv<R: tokio::io::AsyncRead + Unpin>(
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
        let n = reader
            .read_buf(buf)
            .await
            .map_err(|e| FerricastError::Receiver(format!("read: {e}")))?;
        if n == 0 {
            return Ok(None);
        }
    }
}

#[derive(Clone, Debug)]
struct SessionState {
    /// Sender id the most recent CONNECT came from. Used as the
    /// `destination_id` of every reply we issue.
    sender_id: String,
    /// Transport id we assigned to the launched app (mirrors the
    /// session id in the official receiver).
    transport_id: String,
    /// Session id for the launched app.
    session_id: String,
    /// `appId` of the currently launched app, if any.
    app_id: Option<String>,
    /// Media session id we hand out on the first LOAD.
    media_session_id: i64,
    /// Last LOAD's contentId — we echo it back inside MEDIA_STATUS so
    /// the sender can confirm its load was acknowledged.
    last_content_id: Option<String>,
    /// Last LOAD's contentType.
    last_content_type: Option<String>,
    /// Last known playback state, echoed back to the sender via
    /// `report_state` and on unsolicited GET_STATUS replies.
    player_state: String,
    current_time: f64,
}

impl SessionState {
    fn new(sender_id: String) -> Self {
        Self {
            sender_id,
            transport_id: format!("transport-{}", Uuid::new_v4()),
            session_id: Uuid::new_v4().to_string(),
            app_id: None,
            media_session_id: 1,
            last_content_id: None,
            last_content_type: None,
            player_state: "IDLE".into(),
            current_time: 0.0,
        }
    }
}

pub struct ChromecastReceiverControl {
    port: u16,
    tls_config: Arc<rustls::server::ServerConfig>,
    listener: Option<TcpListener>,
    reader_task: Option<JoinHandle<()>>,
    commands_rx: Option<mpsc::Receiver<Result<MediaCommand>>>,
    writer: Option<ServerWriter>,
    state: Arc<Mutex<Option<SessionState>>>,
}

impl ChromecastReceiverControl {
    pub fn new(port: u16, advertised_ips: Vec<std::net::IpAddr>) -> Result<Self> {
        let tls_config = build_server_config(&advertised_ips)?;
        Ok(Self {
            port,
            tls_config,
            listener: None,
            reader_task: None,
            commands_rx: None,
            writer: None,
            state: Arc::new(Mutex::new(None)),
        })
    }
}

impl ControlSession for ChromecastReceiverControl {
    fn accept(&mut self) -> impl std::future::Future<Output = Result<RemoteSender>> + Send {
        async move {
            // Bind lazily on first accept so the constructor doesn't
            // need an async context. Idempotent: subsequent accepts
            // re-use the same listener.
            if self.listener.is_none() {
                let addr: SocketAddr = ([0, 0, 0, 0], self.port).into();
                let listener = TcpListener::bind(addr).await.map_err(|e| {
                    FerricastError::Receiver(format!("bind tcp {addr}: {e}"))
                })?;
                tracing::info!(%addr, "Chromecast receiver TLS listener up");
                self.listener = Some(listener);
            }
            let listener = self.listener.as_mut().unwrap();
            let (tcp, peer) = listener
                .accept()
                .await
                .map_err(|e| FerricastError::Receiver(format!("tcp accept: {e}")))?;
            tracing::info!(%peer, "Chromecast receiver: TCP connect");

            let acceptor = tokio_rustls::TlsAcceptor::from(self.tls_config.clone());
            let tls = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                acceptor.accept(tcp),
            )
            .await
            .map_err(|_| {
                FerricastError::Receiver(format!(
                    "TLS handshake with {peer} did not complete in 5s"
                ))
            })?
            .map_err(|e| FerricastError::Receiver(format!("TLS handshake: {e}")))?;
            tracing::info!(%peer, "Chromecast receiver: TLS handshake complete");

            let (read_half, write_half) = split(tls);
            let writer = Arc::new(Mutex::new(write_half));
            self.writer = Some(writer.clone());

            let (cmd_tx, cmd_rx) = mpsc::channel::<Result<MediaCommand>>(32);
            self.commands_rx = Some(cmd_rx);
            let state = self.state.clone();

            let task = tokio::spawn(run_reader(read_half, writer, cmd_tx, state.clone()));
            self.reader_task = Some(task);

            // The sender's id isn't known until the CONNECT lands;
            // populate state from the peer IP for now. The reader
            // task updates `sender_id` once CONNECT arrives — but
            // for the manager's `ReceiverIncoming` event we hand
            // back what we know now (peer addr + generated UUID).
            Ok(RemoteSender {
                id: Uuid::new_v4(),
                addr: peer.ip(),
                name: None,
            })
        }
    }

    fn next_command(&mut self) -> impl std::future::Future<Output = Result<MediaCommand>> + Send {
        async move {
            let rx = self.commands_rx.as_mut().ok_or_else(|| {
                FerricastError::Receiver("next_command before accept".into())
            })?;
            match rx.recv().await {
                Some(r) => r,
                None => Err(FerricastError::Receiver(
                    "control reader closed".into(),
                )),
            }
        }
    }

    fn report_state(
        &mut self,
        state: PlaybackState,
    ) -> impl std::future::Future<Output = Result<()>> + Send {
        async move {
            let writer = match self.writer.as_ref() {
                Some(w) => w.clone(),
                None => return Ok(()),
            };
            let mut guard = self.state.lock().await;
            let s = match guard.as_mut() {
                Some(s) => s,
                None => return Ok(()),
            };
            s.player_state = match state {
                PlaybackState::Idle => "IDLE",
                PlaybackState::Buffering => "BUFFERING",
                PlaybackState::Playing => "PLAYING",
                PlaybackState::Paused => "PAUSED",
                PlaybackState::Ended => "IDLE",
                PlaybackState::Error(_) => "IDLE",
            }
            .to_string();
            let msg = build_media_status(s, None);
            drop(guard);
            server_send(&writer, &msg).await
        }
    }

    fn close(&mut self) -> impl std::future::Future<Output = Result<()>> + Send {
        async move {
            if let Some(task) = self.reader_task.take() {
                task.abort();
            }
            if let Some(w) = self.writer.take() {
                let _ = w.lock().await.shutdown().await;
            }
            self.listener.take();
            self.commands_rx.take();
            *self.state.lock().await = None;
            Ok(())
        }
    }

    fn is_alive(&self) -> bool {
        self.reader_task
            .as_ref()
            .map(|t| !t.is_finished())
            .unwrap_or(false)
    }
}

/// Reader loop. Owns the read half + a shared writer (for replies to
/// internal namespaces). Forwards media-namespace commands to
/// `cmd_tx` for the manager's pipeline to consume.
async fn run_reader(
    read: ReadHalf<ServerStream>,
    writer: ServerWriter,
    cmd_tx: mpsc::Sender<Result<MediaCommand>>,
    state: Arc<Mutex<Option<SessionState>>>,
) {
    let mut reader = read;
    let mut buf = BytesMut::with_capacity(8 * 1024);
    loop {
        match server_recv(&mut reader, &mut buf).await {
            Ok(Some(msg)) => {
                if let Err(e) = dispatch(&msg, &writer, &cmd_tx, &state).await {
                    let _ = cmd_tx.send(Err(e)).await;
                    break;
                }
            }
            Ok(None) => {
                tracing::info!("Chromecast receiver: sender closed connection");
                break;
            }
            Err(e) => {
                let _ = cmd_tx.send(Err(e)).await;
                break;
            }
        }
    }
}

async fn dispatch(
    msg: &CastMessage,
    writer: &ServerWriter,
    cmd_tx: &mpsc::Sender<Result<MediaCommand>>,
    state: &Arc<Mutex<Option<SessionState>>>,
) -> Result<()> {
    let ns = msg.namespace.as_str();
    let payload = msg.payload_utf8.as_deref().unwrap_or("{}");
    let value: Value = serde_json::from_str(payload)
        .map_err(|e| FerricastError::Receiver(format!("payload parse: {e}")))?;
    let msg_type = value
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let request_id = value.get("requestId").and_then(|v| v.as_i64());

    tracing::info!(
        ns,
        msg_type,
        ?request_id,
        source = %msg.source_id,
        dest = %msg.destination_id,
        "cast← receiver"
    );

    match ns {
        namespace::CONNECTION => {
            // CONNECT establishes the virtual channel. We don't reply
            // — Cast senders take silence as success.
            if msg_type == "CONNECT" {
                let mut guard = state.lock().await;
                let sender_id = msg.source_id.clone();
                tracing::info!(
                    %sender_id,
                    dest = %msg.destination_id,
                    "Chromecast receiver: virtual CONNECT opened"
                );
                if guard.is_none() {
                    *guard = Some(SessionState::new(sender_id.clone()));
                } else if let Some(s) = guard.as_mut() {
                    s.sender_id = sender_id;
                }
            }
            // CLOSE = sender tearing the channel down; reader will
            // see EOF on the next loop. Nothing else to do.
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
                server_send(writer, &pong).await?;
            }
            Ok(())
        }
        namespace::RECEIVER => match msg_type {
            "GET_STATUS" => {
                let mut guard = state.lock().await;
                if guard.is_none() {
                    *guard = Some(SessionState::new(msg.source_id.clone()));
                }
                let s = guard.as_ref().unwrap();
                let reply =
                    build_receiver_status(s, request_id, &msg.destination_id, &msg.source_id)?;
                drop(guard);
                server_send(writer, &reply).await
            }
            "LAUNCH" => {
                let app_id = value
                    .get("appId")
                    .and_then(|v| v.as_str())
                    .unwrap_or(DEFAULT_MEDIA_RECEIVER_APP_ID)
                    .to_string();
                tracing::info!(
                    %app_id,
                    sender = %msg.source_id,
                    "Chromecast receiver: LAUNCH"
                );
                let mut guard = state.lock().await;
                if guard.is_none() {
                    *guard = Some(SessionState::new(msg.source_id.clone()));
                }
                let s = guard.as_mut().unwrap();
                s.app_id = Some(app_id.clone());
                let reply =
                    build_receiver_status(s, request_id, &msg.destination_id, &msg.source_id)?;
                tracing::info!(
                    transport_id = %s.transport_id,
                    session_id = %s.session_id,
                    "Chromecast receiver: RECEIVER_STATUS reply for LAUNCH built"
                );
                let _ = cmd_tx.send(Ok(MediaCommand::LaunchApp { app_id })).await;
                drop(guard);
                server_send(writer, &reply).await
            }
            "STOP" => {
                let mut guard = state.lock().await;
                if let Some(s) = guard.as_mut() {
                    s.app_id = None;
                }
                let _ = cmd_tx.send(Ok(MediaCommand::StopApp)).await;
                if let Some(s) = guard.as_ref() {
                    let reply =
                        build_receiver_status(s, request_id, &msg.destination_id, &msg.source_id)?;
                    drop(guard);
                    server_send(writer, &reply).await?;
                }
                Ok(())
            }
            "SET_VOLUME" => {
                if let Some(vol) = value.get("volume") {
                    if let Some(level) = vol.get("level").and_then(|v| v.as_f64()) {
                        let _ = cmd_tx
                            .send(Ok(MediaCommand::SetSystemVolume(level as f32)))
                            .await;
                    }
                    if let Some(muted) = vol.get("muted").and_then(|v| v.as_bool()) {
                        let _ = cmd_tx.send(Ok(MediaCommand::SetSystemMute(muted))).await;
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        },
        namespace::MEDIA => match msg_type {
            "LOAD" => {
                let media = value.get("media");
                let url = media
                    .and_then(|m| m.get("contentId"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if url.is_empty() {
                    tracing::warn!(
                        full_payload = %payload,
                        "LOAD without media.contentId — sender sent malformed request"
                    );
                    return Err(FerricastError::Receiver(
                        "LOAD without media.contentId".into(),
                    ));
                }
                let content_type = media
                    .and_then(|m| m.get("contentType"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                tracing::info!(
                    %url,
                    content_type = ?content_type,
                    sender = %msg.source_id,
                    "Chromecast receiver: LOAD"
                );
                let autoplay = value
                    .get("autoplay")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let current_time = value
                    .get("currentTime")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let metadata = media
                    .and_then(|m| m.get("metadata"))
                    .map(extract_metadata)
                    .unwrap_or_default();

                {
                    let mut guard = state.lock().await;
                    let s = guard
                        .get_or_insert_with(|| SessionState::new(msg.source_id.clone()));
                    s.last_content_id = Some(url.clone());
                    s.last_content_type = content_type.clone();
                    s.player_state = if autoplay {
                        "BUFFERING"
                    } else {
                        "PAUSED"
                    }
                    .to_string();
                    s.current_time = current_time;
                }

                let _ = cmd_tx
                    .send(Ok(MediaCommand::Load {
                        url,
                        content_type,
                        start_time_us: Some((current_time * 1_000_000.0) as u64),
                        autoplay,
                        metadata,
                    }))
                    .await;

                let guard = state.lock().await;
                if let Some(s) = guard.as_ref() {
                    let reply = build_media_status(s, request_id);
                    drop(guard);
                    server_send(writer, &reply).await?;
                }
                Ok(())
            }
            "PLAY" => {
                let _ = cmd_tx.send(Ok(MediaCommand::Play)).await;
                send_media_status_ack(writer, request_id, state).await
            }
            "PAUSE" => {
                let _ = cmd_tx.send(Ok(MediaCommand::Pause)).await;
                send_media_status_ack(writer, request_id, state).await
            }
            "STOP" => {
                let _ = cmd_tx.send(Ok(MediaCommand::Stop)).await;
                send_media_status_ack(writer, request_id, state).await
            }
            "SEEK" => {
                if let Some(pos) = value.get("currentTime").and_then(|v| v.as_f64()) {
                    let _ = cmd_tx
                        .send(Ok(MediaCommand::Seek {
                            position_us: (pos * 1_000_000.0) as u64,
                        }))
                        .await;
                }
                send_media_status_ack(writer, request_id, state).await
            }
            "GET_STATUS" => send_media_status_ack(writer, request_id, state).await,
            "SET_PLAYBACK_RATE" => {
                if let Some(rate) = value.get("playbackRate").and_then(|v| v.as_f64()) {
                    let _ = cmd_tx
                        .send(Ok(MediaCommand::SetPlaybackRate(rate as f32)))
                        .await;
                }
                Ok(())
            }
            "EDIT_TRACKS_INFO" => {
                let active = value
                    .get("activeTrackIds")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<_>>())
                    .unwrap_or_default();
                let _ = cmd_tx
                    .send(Ok(MediaCommand::EditTracks(
                        ferricast_core::TrackSelection {
                            active_track_ids: active,
                        },
                    )))
                    .await;
                Ok(())
            }
            _ => Ok(()),
        },
        _ => {
            // Bumped from debug → info because the most common
            // sender-rejects-us failure mode is a namespace the
            // sender insists on that we don't implement (notably
            // `urn:x-cast:com.google.cast.tp.deviceauth` for senders
            // that enforce CA signing). Surfacing this in the log
            // makes those failures instantly diagnosable instead of
            // silent.
            tracing::info!(
                ns,
                msg_type,
                source = %msg.source_id,
                dest = %msg.destination_id,
                payload_head = %payload.chars().take(160).collect::<String>(),
                "Chromecast receiver: namespace not implemented — ignoring"
            );
            Ok(())
        }
    }
}

async fn send_media_status_ack(
    writer: &ServerWriter,
    request_id: Option<i64>,
    state: &Arc<Mutex<Option<SessionState>>>,
) -> Result<()> {
    let guard = state.lock().await;
    if let Some(s) = guard.as_ref() {
        let reply = build_media_status(s, request_id);
        drop(guard);
        server_send(writer, &reply).await?;
    }
    Ok(())
}

fn build_receiver_status(
    s: &SessionState,
    request_id: Option<i64>,
    self_id: &str,
    sender_id: &str,
) -> Result<CastMessage> {
    let mut applications = Vec::new();
    if let Some(app_id) = &s.app_id {
        applications.push(json!({
            "appId": app_id,
            "displayName": "Ferricast",
            "namespaces": [
                { "name": namespace::MEDIA },
                { "name": namespace::CONNECTION },
                { "name": namespace::HEARTBEAT },
            ],
            "sessionId": s.session_id,
            "statusText": "Ready",
            "transportId": s.transport_id,
        }));
    }
    let payload = json!({
        "type": "RECEIVER_STATUS",
        "requestId": request_id.unwrap_or(0),
        "status": {
            "applications": applications,
            "volume": { "level": 1.0, "muted": false }
        }
    });
    CastMessage::new_json(self_id, sender_id, namespace::RECEIVER, &payload)
        .map_err(|e| FerricastError::Receiver(format!("encode RECEIVER_STATUS: {e}")))
}

fn build_media_status(s: &SessionState, request_id: Option<i64>) -> CastMessage {
    let payload = json!({
        "type": "MEDIA_STATUS",
        "requestId": request_id.unwrap_or(0),
        "status": [{
            "mediaSessionId": s.media_session_id,
            "playerState": s.player_state,
            "currentTime": s.current_time,
            "supportedMediaCommands": 0x0F,
            "media": s.last_content_id.as_ref().map(|cid| json!({
                "contentId": cid,
                "contentType": s.last_content_type.as_deref().unwrap_or("application/vnd.apple.mpegurl"),
                "streamType": "BUFFERED"
            }))
        }]
    });
    // unwrap: payload is always valid JSON we built ourselves
    CastMessage::new_json(
        &s.transport_id,
        &s.sender_id,
        namespace::MEDIA,
        &payload,
    )
    .expect("MEDIA_STATUS payload encode")
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
