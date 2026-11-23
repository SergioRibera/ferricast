//! One RTSP control connection: answers the sender's requests, sets up
//! the media streams it asks for and feeds the pipeline.

mod video;

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use ferricast_core::{
    AudioCodec, FerricastError, MediaCommand, PushedStream, RemoteSender, Result,
};
use plist::{Dictionary, Value};
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::assembler::{Assembler, MediaEvent};
use super::audio::{AudioFormat, AudioReceiver, PayloadDecipher};
use super::clock::{RemoteClock, TimingClient};
use super::pairing::{Pairing, random_pin};
use super::rtsp::{Request, Response, RtspServerConn};
use super::server::{Accepted, Promoted, Shared, closed};
use super::video::VideoHandle;
use super::{MODEL, SOURCE_VERSION, txt_bytes};
use crate::crypto::sha512_16;
use crate::store::PairedSender;

const BPLIST: &str = "application/x-apple-binary-plist";
const OCTET: &str = "application/octet-stream";
const STREAM_MIRROR: u64 = 110;
const STREAM_AUDIO: u64 = 96;
/// How long a fixed port may stay taken by the session being replaced.
const BIND_RETRY: Duration = Duration::from_secs(2);
const BIND_RETRY_STEP: Duration = Duration::from_millis(50);
/// How long the sender has to open the mirroring connection.
const MIRROR_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// How long a connection without a session may go without a request.
const IDLE_TIMEOUT: Duration = Duration::from_secs(60);
/// Media events queued from the stream receivers.
const MEDIA_QUEUE: usize = 256;
/// UUID UxPlay reports for its one display.
const DISPLAY_UUID: &str = "e0ff8a27-6738-3d56-8a16-cc53aacee925";

/// Media streams of the connection, torn down when dropped.
struct Media {
    aes_key: [u8; 16],
    eiv: [u8; 16],
    clock: RemoteClock,
    _timing: Option<TimingClient>,
    mirror: Option<JoinHandle<()>>,
    audio: Option<(JoinHandle<()>, mpsc::Sender<Option<u16>>)>,
    /// Fires when another connection takes the media over.
    preempted: mpsc::Receiver<()>,
}

impl Media {
    fn stop_mirror(&mut self) {
        if let Some(task) = self.mirror.take() {
            task.abort();
        }
    }

    fn stop_audio(&mut self) {
        if let Some((task, _)) = self.audio.take() {
            task.abort();
        }
    }
}

impl Drop for Media {
    fn drop(&mut self) {
        self.stop_mirror();
        self.stop_audio();
    }
}

pub(super) struct Connection {
    shared: Arc<Shared>,
    id: u64,
    peer: SocketAddr,
    ready: mpsc::Sender<Accepted>,
    promoted: Option<Promoted>,
    pairing: Pairing,
    /// FairPlay key message (the second `fp-setup` message).
    keymsg: Option<[u8; 164]>,
    /// The sender's name, from the session `SETUP`.
    name: Option<String>,
    pin_shown: bool,
    media: Option<Media>,
    assembler: Assembler,
    media_tx: mpsc::Sender<MediaEvent>,
    media_rx: mpsc::Receiver<MediaEvent>,
    /// The video session of the sender, once it sent a video request.
    video: Option<VideoHandle>,
}

/// Whether the loop goes on after a request.
enum Next {
    Continue,
    Close,
    /// The sender reversed the connection (`POST /reverse`) for the
    /// events of this session key.
    Reverse(String),
}

fn plist_body(dict: Dictionary) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    plist::to_writer_binary(&mut out, &Value::Dictionary(dict))
        .map_err(|e| FerricastError::Protocol(format!("encode plist: {e}")))?;
    Ok(out)
}

fn plist_dict(body: &[u8]) -> Option<Dictionary> {
    plist::from_bytes::<Value>(body)
        .ok()
        .and_then(Value::into_dictionary)
}

fn uint(dict: &Dictionary, key: &str) -> Option<u64> {
    let value = dict.get(key)?;
    value.as_unsigned_integer().or_else(|| {
        value
            .as_signed_integer()
            .map(|v| u64::from_ne_bytes(v.to_ne_bytes()))
    })
}

fn bplist_response(dict: Dictionary) -> Response {
    match plist_body(dict) {
        Ok(body) => Response::ok().body(BPLIST, body),
        Err(e) => {
            warn!(error = %e, "cannot encode a reply");
            Response::status(500, "Internal Server Error")
        }
    }
}

/// Retry `bind` while a fixed port is still held by the session being
/// replaced.
async fn bind_retrying<T, F, Fut>(port: u16, what: &str, mut bind: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = ferricast_core::Result<T>>,
{
    let deadline = tokio::time::Instant::now() + BIND_RETRY;
    loop {
        match bind().await {
            Ok(v) => return Ok(v),
            Err(e) if port != 0 && tokio::time::Instant::now() < deadline => {
                debug!(port, what, error = %e, "port busy; retrying");
                tokio::time::sleep(BIND_RETRY_STEP).await;
            }
            Err(e) => {
                return Err(FerricastError::Receiver(format!(
                    "cannot bind the {what} port {port}: {e}"
                )));
            }
        }
    }
}

fn fixed(port: u16) -> Option<(u16, u16)> {
    (port != 0).then_some((port, port))
}

async fn bind_udp(peer: IpAddr, port: u16, what: &str) -> Result<UdpSocket> {
    bind_retrying(port, what, || ferricast_core::bind_udp(peer, fixed(port))).await
}

async fn bind_tcp(peer: IpAddr, port: u16, what: &str) -> Result<TcpListener> {
    bind_retrying(port, what, || ferricast_core::bind_tcp(peer, fixed(port))).await
}

/// Linear gain of a volume in the sender's dB scale.
pub(super) fn gain_from_db(db: f32) -> f32 {
    if db <= -30.0 {
        0.0
    } else {
        10_f32.powf(db.min(0.0) / 20.0)
    }
}

impl Connection {
    pub fn new(shared: Arc<Shared>, peer: SocketAddr, ready: mpsc::Sender<Accepted>) -> Self {
        let (media_tx, media_rx) = mpsc::channel(MEDIA_QUEUE);
        Self {
            id: shared.next_connection_id(),
            shared,
            peer,
            ready,
            promoted: None,
            pairing: Pairing::default(),
            keymsg: None,
            name: None,
            pin_shown: false,
            media: None,
            assembler: Assembler::default(),
            media_tx,
            media_rx,
            video: None,
        }
    }

    pub async fn run(mut self, stream: TcpStream) {
        let mut rtsp = RtspServerConn::new(stream);
        let mut idle_deadline = tokio::time::Instant::now() + IDLE_TIMEOUT;
        let mut reversed = None;
        loop {
            let deadline = self.assembler.deadline();
            // Only a connection without a session can idle out: one
            // with media keeps streaming while the sender sends nothing, and
            // a paused video may go unpolled.
            let can_idle = self.promoted.is_none() && self.media.is_none() && self.video.is_none();
            tokio::select! {
                () = tokio::time::sleep_until(idle_deadline), if can_idle => {
                    debug!(peer = %self.peer, "AirPlay control connection idle; closing");
                    break;
                }
                request = rtsp.read_request() => {
                    idle_deadline = tokio::time::Instant::now() + IDLE_TIMEOUT;
                    let request = match request {
                        Ok(Some(r)) => r,
                        Ok(None) => break,
                        Err(e) => {
                            debug!(peer = %self.peer, error = %e, "AirPlay control connection failed");
                            break;
                        }
                    };
                    let (response, next) = self.handle(&request).await;
                    debug!(
                        peer = %self.peer,
                        method = %request.method,
                        uri = %request.uri,
                        status = response.status_code(),
                        "AirPlay request"
                    );
                    if let Err(e) = rtsp.write_response(&request, &response).await {
                        debug!(error = %e, "AirPlay response failed");
                        break;
                    }
                    match next {
                        _ if response.close => break,
                        Next::Close => break,
                        Next::Reverse(key) => {
                            reversed = Some(key);
                            break;
                        }
                        Next::Continue => {}
                    }
                }
                Some(event) = self.media_rx.recv() => {
                    if let Some(stream) = self.assembler.on_event(event) {
                        self.announce(stream).await;
                    }
                }
                () = sleep_until(deadline) => {
                    if let Some(stream) = self.assembler.on_deadline() {
                        self.announce(stream).await;
                    }
                }
                _ = closed(&mut self.promoted) => {
                    debug!(peer = %self.peer, "pipeline closed the AirPlay session");
                    break;
                }
                _ = preempted(&mut self.media) => {
                    info!(peer = %self.peer, "AirPlay session replaced by another sender");
                    break;
                }
            }
        }
        self.media = None;
        self.shared.release_media(self.id);
        if self.pin_shown {
            self.command(MediaCommand::PairingPin(None)).await;
        }
        match reversed {
            Some(key) => {
                let shared = self.shared.clone();
                drop(self);
                shared.reverse.serve(&key, rtsp.into_inner()).await;
            }
            None => {
                rtsp.shutdown().await;
                debug!(peer = %self.peer, "AirPlay control connection closed");
            }
        }
    }

    /// Hand the connection to the pipeline, once.
    async fn promote(&mut self) {
        if self.promoted.is_some() {
            return;
        }
        let remote = RemoteSender {
            id: Uuid::new_v4(),
            addr: self.peer.ip(),
            name: self.name.clone(),
        };
        self.promoted = Some(Promoted::open(&self.ready, remote).await);
    }

    async fn command(&mut self, cmd: MediaCommand) {
        self.promote().await;
        if let Some(p) = &self.promoted {
            p.command(cmd).await;
        }
    }

    async fn announce(&mut self, stream: PushedStream) {
        self.command(MediaCommand::Push(stream)).await;
    }

    async fn handle(&mut self, request: &Request) -> (Response, Next) {
        if let Some(handled) = self.handle_video(request).await {
            return handled;
        }
        let result = match (request.method.as_str(), request.path()) {
            ("GET", "/info") => Ok(self.info(request)),
            ("POST", "/pair-setup") => self.pair_setup(request).await,
            ("POST", "/pair-verify") => self.pair_verify(request).await,
            ("POST", "/pair-pin-start") => self.pair_pin_start().await,
            ("POST", "/pair-setup-pin") => self.pair_setup_pin(request).await,
            ("POST", "/fp-setup") => self.fp_setup(request).await,
            ("POST", "/feedback" | "/audioMode") => Ok(Response::ok()),
            ("OPTIONS", _) => Ok(Response::ok().header(
                "Public",
                "SETUP, RECORD, FLUSH, TEARDOWN, OPTIONS, GET_PARAMETER, SET_PARAMETER, POST, GET",
            )),
            ("SETUP", _) => self.setup(request).await,
            ("RECORD", _) => Ok(Response::ok()
                .header("Audio-Latency", "0")
                .header("Audio-Jack-Status", "connected; type=analog")),
            ("GET_PARAMETER", _) => Ok(self.get_parameter(request)),
            ("SET_PARAMETER", _) => Ok(self.set_parameter(request).await),
            ("FLUSH", _) => Ok(self.flush(request).await),
            ("TEARDOWN", _) => return self.teardown(request).await,
            (method, path) => {
                debug!(method, path, "unhandled AirPlay request");
                Ok(Response::status(404, "Not Found"))
            }
        };
        match result {
            Ok(response) => (response, Next::Continue),
            Err(e) => {
                warn!(peer = %self.peer, uri = %request.uri, error = %e, "AirPlay request failed");
                (
                    Response::status(470, "Connection Authorization Required"),
                    Next::Continue,
                )
            }
        }
    }

    fn info(&self, request: &Request) -> Response {
        let settings = &self.shared.settings;
        let mut qualifiers = Vec::new();
        let is_plist = request
            .header("Content-Type")
            .is_some_and(|ct| ct.contains("apple-binary-plist"));
        if is_plist
            && let Some(dict) = plist_dict(&request.body)
            && let Some(list) = dict.get("qualifier").and_then(Value::as_array)
        {
            qualifiers.extend(list.iter().filter_map(Value::as_string).map(str::to_owned));
        }
        if request.header("CSeq").is_none() {
            for q in ["txtAirPlay", "txtRAOP"] {
                if request.uri.contains(q) {
                    qualifiers.push(q.to_owned());
                }
            }
        }
        let mut dict = Dictionary::new();
        for q in &qualifiers {
            let txt = match q.as_str() {
                "txtAirPlay" => settings.airplay_txt(),
                "txtRAOP" => settings.raop_txt(),
                _ => continue,
            };
            dict.insert(q.clone(), Value::Data(txt_bytes(&txt)));
        }
        if is_plist {
            return bplist_response(dict);
        }
        let id = &settings.identity;
        let config = &settings.config;
        let (low, high) = settings.features();
        dict.insert("deviceID".into(), Value::String(id.device_id_colons()));
        dict.insert("macAddress".into(), Value::String(id.device_id_colons()));
        dict.insert(
            "pk".into(),
            Value::Data(id.key.verifying_key().as_bytes().to_vec()),
        );
        dict.insert(
            "features".into(),
            Value::Integer(((u64::from(high) << 32) | u64::from(low)).into()),
        );
        dict.insert("name".into(), Value::String(config.name.clone()));
        dict.insert("pi".into(), Value::String(id.pi.clone()));
        dict.insert("vv".into(), Value::Integer(2.into()));
        dict.insert("statusFlags".into(), Value::Integer(68.into()));
        dict.insert("keepAliveLowPower".into(), Value::Integer(1.into()));
        dict.insert("sourceVersion".into(), Value::String(SOURCE_VERSION.into()));
        dict.insert("keepAliveSendStatsAsBody".into(), Value::Boolean(true));
        dict.insert("model".into(), Value::String(MODEL.into()));
        dict.insert(
            "initialVolume".into(),
            Value::Real(f64::from(self.shared.volume_db())),
        );
        let latency = |kind: u64| {
            let mut d = Dictionary::new();
            d.insert("type".into(), Value::Integer(kind.into()));
            d.insert("audioType".into(), Value::String("default".into()));
            d.insert("inputLatencyMicros".into(), Value::Integer(0.into()));
            d.insert("outputLatencyMicros".into(), Value::Boolean(false));
            Value::Dictionary(d)
        };
        dict.insert(
            "audioLatencies".into(),
            Value::Array(vec![latency(100), latency(101)]),
        );
        let format = |kind: u64| {
            let mut d = Dictionary::new();
            d.insert("type".into(), Value::Integer(kind.into()));
            d.insert(
                "audioInputFormats".into(),
                Value::Integer(0x3ff_fffc.into()),
            );
            d.insert(
                "audioOutputFormats".into(),
                Value::Integer(0x3ff_fffc.into()),
            );
            Value::Dictionary(d)
        };
        dict.insert(
            "audioFormats".into(),
            Value::Array(vec![format(100), format(101)]),
        );
        let mut display = Dictionary::new();
        display.insert("uuid".into(), Value::String(DISPLAY_UUID.into()));
        display.insert("widthPhysical".into(), Value::Integer(0.into()));
        display.insert("heightPhysical".into(), Value::Integer(0.into()));
        for key in ["width", "widthPixels"] {
            display.insert(key.into(), Value::Integer(u64::from(config.width).into()));
        }
        for key in ["height", "heightPixels"] {
            display.insert(key.into(), Value::Integer(u64::from(config.height).into()));
        }
        display.insert("rotation".into(), Value::Boolean(false));
        display.insert(
            "refreshRate".into(),
            Value::Real(1.0 / f64::from(config.max_fps.max(1))),
        );
        display.insert(
            "maxFPS".into(),
            Value::Integer(u64::from(config.max_fps).into()),
        );
        display.insert("overscanned".into(), Value::Boolean(false));
        display.insert("features".into(), Value::Integer(14.into()));
        dict.insert(
            "displays".into(),
            Value::Array(vec![Value::Dictionary(display)]),
        );
        bplist_response(dict)
    }

    async fn pair_setup(&mut self, request: &Request) -> Result<Response> {
        if request.body.len() != 32 {
            return Err(FerricastError::Protocol(
                "pair-setup: expected 32 bytes".into(),
            ));
        }
        let key = self.shared.settings.identity.key.verifying_key();
        Ok(Response::ok().body(OCTET, key.as_bytes().to_vec()))
    }

    /// Whether a sender pair-verifying with `key` may connect in PIN mode.
    fn sender_paired(&self, key: &[u8; 32]) -> bool {
        self.shared.is_paired(&hex::encode(key))
    }

    async fn pair_verify(&mut self, request: &Request) -> Result<Response> {
        let identity = &self.shared.settings.identity.key;
        match request.body.first() {
            Some(1) => {
                let reply = self.pairing.verify_start(identity, &request.body)?;
                if self.shared.settings.config.require_pin {
                    let known = self
                        .pairing
                        .verifying_client()
                        .is_some_and(|k| self.sender_paired(&k));
                    if !known {
                        // What UxPlay answers an unknown sender: no
                        // handshake, so it falls back to the PIN.
                        debug!(peer = %self.peer, "pair-verify from a sender not paired with the PIN");
                        return Ok(Response::ok());
                    }
                }
                Ok(Response::ok().body(OCTET, reply))
            }
            Some(0) => match self.pairing.verify_finish(&request.body) {
                Ok(()) => {
                    debug!(peer = %self.peer, "pair-verify complete");
                    Ok(Response::ok())
                }
                Err(e) => {
                    warn!(peer = %self.peer, error = %e, "pair-verify failed");
                    Ok(Response::status(470, "Connection Authorization Required").closing())
                }
            },
            _ => Err(FerricastError::Protocol("pair-verify: unknown step".into())),
        }
    }

    async fn pair_pin_start(&mut self) -> Result<Response> {
        let pin = random_pin();
        info!(peer = %self.peer, "AirPlay sender asks to pair: showing a PIN");
        self.pairing.pin_start(pin.clone());
        self.pin_shown = true;
        self.command(MediaCommand::PairingPin(Some(pin))).await;
        Ok(Response::ok())
    }

    async fn pair_setup_pin(&mut self, request: &Request) -> Result<Response> {
        let dict = plist_dict(&request.body)
            .ok_or_else(|| FerricastError::Protocol("pair-setup-pin: not a plist".into()))?;
        let identity = self.shared.settings.identity.key.clone();
        match self.pairing.pin_setup(&identity, &dict) {
            Ok((reply, paired)) => {
                if let Some(paired) = paired {
                    info!(device_id = %paired.device_id, "AirPlay sender paired with the PIN");
                    self.shared.add_paired(hex::encode(paired.public_key));
                    let saved = self.shared.settings.store.save_paired_sender(PairedSender {
                        device_id: paired.device_id,
                        public_key: hex::encode(paired.public_key),
                    });
                    if let Err(e) = saved {
                        warn!(error = %e, "cannot remember the paired sender");
                    }
                    self.pin_shown = false;
                    self.command(MediaCommand::PairingPin(None)).await;
                }
                Ok(bplist_response(reply))
            }
            Err(e) => {
                warn!(peer = %self.peer, error = %e, "PIN pairing failed");
                Ok(Response::status(470, "Client Authentication Failure"))
            }
        }
    }

    #[cfg(feature = "fairplay")]
    async fn fp_setup(&mut self, request: &Request) -> Result<Response> {
        use ferricast_fairplay as fp;
        let fp_error = |e: fp::FairPlayError| FerricastError::Protocol(format!("fp-setup: {e}"));
        match request.body.len() {
            fp::FP_SETUP1_LEN => {
                let reply = fp::setup1_reply(&request.body).map_err(fp_error)?;
                Ok(Response::ok().body(OCTET, reply.to_vec()))
            }
            fp::FP_SETUP2_LEN => {
                let reply = fp::setup2_reply(&request.body).map_err(fp_error)?;
                let mut keymsg = [0_u8; 164];
                keymsg.copy_from_slice(&request.body);
                self.keymsg = Some(keymsg);
                Ok(Response::ok().body(OCTET, reply.to_vec()))
            }
            n => Err(FerricastError::Protocol(format!("fp-setup: {n} bytes"))),
        }
    }

    #[cfg(not(feature = "fairplay"))]
    #[expect(clippy::unused_async, reason = "same signature as the FairPlay build")]
    async fn fp_setup(&mut self, _request: &Request) -> Result<Response> {
        warn!("fp-setup refused: built without FairPlay support (the `fairplay` feature)");
        Ok(Response::status(501, "Not Implemented"))
    }

    /// The AES key of `ekey`: FairPlay-decrypted with the stored key
    /// message, then hashed with the pairing secret when pairing ran.
    fn session_key(&self, ekey: &[u8]) -> Result<[u8; 16]> {
        let base = match self.keymsg {
            #[cfg(feature = "fairplay")]
            Some(keymsg) => {
                let ekey: &[u8; 72] =
                    ekey.get(..72)
                        .and_then(|e| e.try_into().ok())
                        .ok_or_else(|| {
                            FerricastError::Protocol(format!("SETUP: ekey of {} bytes", ekey.len()))
                        })?;
                ferricast_fairplay::decrypt_ekey(&keymsg, ekey)
                    .map_err(|e| FerricastError::Protocol(format!("ekey: {e}")))?
            }
            #[cfg(not(feature = "fairplay"))]
            Some(_) => [0; 16],
            None => {
                // No fp-setup: senders that skip FairPlay (ferricast built
                // without it) use an all-zero base key.
                debug!(
                    ekey_len = ekey.len(),
                    "SETUP without fp-setup; all-zero base key"
                );
                [0; 16]
            }
        };
        Ok(match self.pairing.ecdh() {
            Some(ecdh) => sha512_16(&[&base, &ecdh]),
            None => base,
        })
    }

    async fn setup(&mut self, request: &Request) -> Result<Response> {
        let dict = plist_dict(&request.body)
            .ok_or_else(|| FerricastError::Protocol("SETUP: not a plist".into()))?;
        let mut reply = Dictionary::new();
        if let (Some(ekey), Some(eiv)) = (
            dict.get("ekey").and_then(Value::as_data),
            dict.get("eiv").and_then(Value::as_data),
        ) {
            let eiv: [u8; 16] = eiv
                .get(..16)
                .and_then(|v| v.try_into().ok())
                .ok_or_else(|| FerricastError::Protocol("SETUP: eiv too short".into()))?;
            let aes_key = self.session_key(ekey)?;
            if let Some(name) = dict.get("name").and_then(Value::as_string) {
                self.name = Some(name.to_owned());
            }
            let model = dict.get("model").and_then(Value::as_string);
            info!(
                peer = %self.peer,
                name = ?self.name,
                model = ?model,
                "AirPlay session setup"
            );
            // Stop our own streams before the new ones take the ports.
            self.media = None;
            let preempted = self.shared.claim_media(self.id);
            let clock = RemoteClock::default();
            let timing = match uint(&dict, "timingPort").and_then(|p| u16::try_from(p).ok()) {
                Some(port) if port != 0 => {
                    let socket = bind_udp(
                        self.peer.ip(),
                        self.shared.settings.config.timing_port,
                        "timing",
                    )
                    .await?;
                    let remote = SocketAddr::new(self.peer.ip(), port);
                    Some(TimingClient::start(socket, remote, clock.clone()))
                }
                _ => {
                    warn!("SETUP without a timingPort; timestamps follow arrival");
                    None
                }
            };
            let timing_port = timing.as_ref().map_or(0, TimingClient::port);
            self.media = Some(Media {
                aes_key,
                eiv,
                clock,
                _timing: timing,
                mirror: None,
                audio: None,
                preempted,
            });
            self.promote().await;
            reply.insert("timingPort".into(), Value::Integer(timing_port.into()));
            reply.insert("eventPort".into(), Value::Integer(0.into()));
        }
        if let Some(streams) = dict.get("streams").and_then(Value::as_array) {
            let mut out = Vec::new();
            for stream in streams.iter().filter_map(Value::as_dictionary) {
                match uint(stream, "type") {
                    Some(STREAM_MIRROR) => out.push(self.setup_mirror(stream).await?),
                    Some(STREAM_AUDIO) => {
                        let (entry, announced) = self.setup_audio(stream).await?;
                        out.push(entry);
                        if let Some(stream) = announced {
                            self.announce(stream).await;
                        }
                    }
                    other => {
                        return Err(FerricastError::Protocol(format!(
                            "SETUP: unsupported stream type {other:?}"
                        )));
                    }
                }
            }
            reply.insert("streams".into(), Value::Array(out));
        }
        Ok(bplist_response(reply))
    }

    async fn setup_mirror(&mut self, stream: &Dictionary) -> Result<Value> {
        let peer = self.peer.ip();
        let port = self.shared.settings.config.mirror_port;
        let media = self.media.as_mut().ok_or_else(|| {
            FerricastError::Protocol("mirroring SETUP before the session SETUP".into())
        })?;
        let stream_id = uint(stream, "streamConnectionID").unwrap_or_default();
        let cipher = crate::mirror::stream_cipher(&media.aes_key, stream_id);
        media.stop_mirror();
        let listener = bind_tcp(peer, port, "mirroring").await?;
        let data_port = listener.local_addr()?.port();
        let clock = media.clock.clone();
        let events = self.media_tx.clone();
        media.mirror = Some(tokio::spawn(async move {
            let accepted = tokio::time::timeout(MIRROR_CONNECT_TIMEOUT, async {
                loop {
                    match listener.accept().await {
                        Ok((tcp, from)) if from.ip() == peer => return Ok(tcp),
                        Ok((_, from)) => debug!(%from, "mirroring connection from another host"),
                        Err(e) => return Err(e),
                    }
                }
            })
            .await;
            drop(listener);
            let tcp = match accepted {
                Ok(Ok(tcp)) => tcp,
                Ok(Err(e)) => {
                    warn!(error = %e, "mirroring accept failed");
                    return;
                }
                Err(_) => {
                    warn!("the sender never opened the mirroring connection");
                    return;
                }
            };
            let _ = tcp.set_nodelay(true);
            if let Err(e) = super::mirror::receive(tcp, cipher, clock, events).await {
                warn!(error = %e, "mirroring stream failed");
            }
        }));
        self.assembler.set_video_expected(true);
        info!(data_port, stream_id, "AirPlay mirroring stream set up");
        let mut entry = Dictionary::new();
        entry.insert("type".into(), Value::Integer(STREAM_MIRROR.into()));
        entry.insert("dataPort".into(), Value::Integer(data_port.into()));
        Ok(Value::Dictionary(entry))
    }

    async fn setup_audio(&mut self, stream: &Dictionary) -> Result<(Value, Option<PushedStream>)> {
        let peer = self.peer.ip();
        let config = &self.shared.settings.config;
        let (data_port, control_port) = (config.audio_data_port, config.audio_control_port);
        let codec = match uint(stream, "ct") {
            Some(2) => AudioCodec::Alac,
            Some(8) => AudioCodec::AacEld,
            other => {
                return Err(FerricastError::Protocol(format!(
                    "audio SETUP: unsupported compression type {other:?}"
                )));
            }
        };
        let sample_rate = uint(stream, "sr")
            .and_then(|v| u32::try_from(v).ok())
            .filter(|v| *v > 0)
            .unwrap_or(44_100);
        let format = AudioFormat {
            codec,
            sample_rate,
            channels: 2,
        };
        let remote_control = uint(stream, "controlPort")
            .and_then(|p| u16::try_from(p).ok())
            .filter(|p| *p != 0)
            .map(|p| SocketAddr::new(peer, p));
        let media = self.media.as_mut().ok_or_else(|| {
            FerricastError::Protocol("audio SETUP before the session SETUP".into())
        })?;
        media.stop_audio();
        let data = bind_udp(peer, data_port, "audio data").await?;
        let control = bind_udp(peer, control_port, "audio control").await?;
        let (local_data, local_control) = (data.local_addr()?.port(), control.local_addr()?.port());
        let (flush_tx, flush_rx) = mpsc::channel(4);
        let receiver = AudioReceiver {
            data,
            control,
            decipher: PayloadDecipher::new(&media.aes_key, media.eiv),
            format,
            clock: media.clock.clone(),
            events: self.media_tx.clone(),
            remote_control,
            flushes: flush_rx,
        };
        media.audio = Some((tokio::spawn(receiver.run()), flush_tx));
        info!(
            ?codec,
            local_data, local_control, "AirPlay audio stream set up"
        );
        let announced = self.assembler.set_audio(Some(format));
        let mut entry = Dictionary::new();
        entry.insert("type".into(), Value::Integer(STREAM_AUDIO.into()));
        entry.insert("dataPort".into(), Value::Integer(local_data.into()));
        entry.insert("controlPort".into(), Value::Integer(local_control.into()));
        Ok((Value::Dictionary(entry), announced))
    }

    fn get_parameter(&self, request: &Request) -> Response {
        let asks_volume = request.header("Content-Type") == Some("text/parameters")
            && String::from_utf8_lossy(&request.body)
                .lines()
                .any(|l| l.trim() == "volume");
        if asks_volume {
            let body = format!("volume: {:.6}\r\n", self.shared.volume_db());
            return Response::ok().body("text/parameters", body.into_bytes());
        }
        Response::ok()
    }

    async fn set_parameter(&mut self, request: &Request) -> Response {
        if request.header("Content-Type") != Some("text/parameters") {
            // Cover art, DMAP metadata: accepted and ignored.
            return Response::ok();
        }
        let body = String::from_utf8_lossy(&request.body).into_owned();
        for line in body.lines() {
            if let Some(value) = line.strip_prefix("volume:")
                && let Ok(db) = value.trim().parse::<f32>()
            {
                self.shared.set_volume_db(db);
                let gain = gain_from_db(db);
                debug!(db, gain, "AirPlay volume");
                self.command(MediaCommand::SetVolume(gain)).await;
                // The sender's video playback, if any, follows the same volume.
                self.shared
                    .videos
                    .broadcast(|| super::video::VideoRequest::Volume(gain));
            }
        }
        Response::ok()
    }

    async fn flush(&mut self, request: &Request) -> Response {
        let next_seq = request
            .header("RTP-Info")
            .and_then(|info| info.split(';').find_map(|p| p.trim().strip_prefix("seq=")))
            .and_then(|s| s.parse::<u16>().ok());
        if let Some((_, flush)) = self.media.as_ref().and_then(|m| m.audio.as_ref()) {
            let _ = flush.send(next_seq).await;
        }
        Response::ok()
    }

    async fn teardown(&mut self, request: &Request) -> (Response, Next) {
        let types: Vec<u64> = plist_dict(&request.body)
            .and_then(|d| d.get("streams").and_then(Value::as_array).cloned())
            .map(|streams| {
                streams
                    .iter()
                    .filter_map(Value::as_dictionary)
                    .filter_map(|s| uint(s, "type"))
                    .collect()
            })
            .unwrap_or_default();
        if types.is_empty() {
            info!(peer = %self.peer, "AirPlay session torn down");
            self.media = None;
            self.shared.release_media(self.id);
            self.command(MediaCommand::Stop).await;
            return (Response::ok().closing(), Next::Close);
        }
        if let Some(media) = self.media.as_mut() {
            for kind in &types {
                match *kind {
                    STREAM_MIRROR => {
                        media.stop_mirror();
                        self.assembler.set_video_expected(false);
                    }
                    STREAM_AUDIO => {
                        media.stop_audio();
                        let _ = self.assembler.set_audio(None);
                    }
                    other => debug!(other, "TEARDOWN of an unknown stream type"),
                }
            }
            if media.mirror.is_none() && media.audio.is_none() {
                info!(peer = %self.peer, "AirPlay streams torn down");
                self.command(MediaCommand::Stop).await;
            }
        }
        (Response::ok(), Next::Continue)
    }
}

async fn sleep_until(deadline: Option<tokio::time::Instant>) {
    match deadline {
        Some(d) => tokio::time::sleep_until(d).await,
        None => std::future::pending().await,
    }
}

/// Resolves when another connection takes the media over.
async fn preempted(media: &mut Option<Media>) {
    match media {
        Some(m) => {
            if m.preempted.recv().await.is_none() {
                std::future::pending::<()>().await;
            }
        }
        None => std::future::pending().await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volume_maps_to_gain() {
        assert_eq!(gain_from_db(-144.0), 0.0);
        assert_eq!(gain_from_db(-30.0), 0.0);
        assert!((gain_from_db(0.0) - 1.0).abs() < 1e-6);
        assert!((gain_from_db(-20.0) - 0.1).abs() < 1e-6);
    }
}
