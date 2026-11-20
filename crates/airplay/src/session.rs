//! AirPlay screen-mirroring sender session.
//!
//! `connect` pairs with the receiver (asking for a PIN through
//! [`ConnectOutcome::PairingRequired`] when needed), `setup_stream` runs
//! FairPlay setup, both `SETUP` requests and `RECORD`, then opens the
//! mirroring data connection; `send_frame` pushes encrypted H.264 access
//! units and a background task posts `/feedback` every two seconds.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ferricast_core::device::{AirPlayPairing, AirplayConfig};
use ferricast_core::{
    CastSession, Codec, ConnectOutcome, Device, EncodedFrame, FerricastError, PairingChallenge,
    PairingResponse, Result, StreamConfig,
};
use plist::{Dictionary, Value};
use tokio::net::{TcpStream, UdpSocket};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::crypto::sha512_16;
use crate::fairplay;
use crate::mirror::{MirrorStream, stream_cipher};
use crate::ntp::TimingServer;
use crate::pair::{PairedSession, hap, legacy, legacy_pin};
use crate::rtsp::{CONTENT_BPLIST, ClientIdentity, RtspClient, RtspResponse};
use crate::store::{CredentialStore, Identity, ReceiverKey};

const FEEDBACK_INTERVAL: Duration = Duration::from_secs(2);
const DATA_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// `SETUP` stream type for screen mirroring.
const STREAM_TYPE_MIRROR: u64 = 110;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Disconnected,
    /// Pair-setup needs the PIN the receiver is showing.
    AwaitingPin,
    /// Paired and verified; ready for `setup_stream`.
    Paired,
    Streaming,
}

/// Aborts the wrapped task when dropped.
struct TaskGuard(JoinHandle<()>);

impl Drop for TaskGuard {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Everything that only exists while streaming.
struct Stream {
    mirror: MirrorStream,
    _timing: TimingServer,
    _feedback: TaskGuard,
}

/// Connection to the receiver that survives from `connect` to `stop`.
struct Link {
    rtsp: Arc<Mutex<RtspClient>>,
    /// Receiver device id (key for stored credentials).
    device_id: String,
    mode: AirPlayPairing,
    /// `rtsp://<local ip>/<session id>` request URI.
    session_uri: String,
}

/// An AirPlay screen-mirroring session.
pub struct AirPlaySession {
    store: CredentialStore,
    identity: Option<Identity>,
    state: State,
    link: Option<Link>,
    paired: PairedSession,
    stream: Option<Stream>,
    alive: Arc<AtomicBool>,
}

impl Default for AirPlaySession {
    fn default() -> Self {
        Self {
            store: CredentialStore::default(),
            identity: None,
            state: State::Disconnected,
            link: None,
            paired: PairedSession::default(),
            stream: None,
            alive: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// Addresses to try for `device`: the resolved address first, then any
/// extra addresses discovery recorded.
fn candidate_addrs(device: &Device) -> Vec<SocketAddr> {
    let mut addrs = vec![SocketAddr::new(device.addr, device.port)];
    if let Some(extra) = device.metadata.get(crate::discovery::META_ADDRESSES) {
        addrs.extend(
            extra
                .split(',')
                .filter_map(|a| a.trim().parse::<IpAddr>().ok())
                .map(|ip| SocketAddr::new(ip, device.port))
                .filter(|a| a.ip() != device.addr),
        );
    }
    addrs
}

fn bplist(dict: Dictionary) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    plist::to_writer_binary(&mut out, &Value::Dictionary(dict))
        .map_err(|e| FerricastError::Protocol(format!("encode SETUP plist: {e}")))?;
    Ok(out)
}

fn reply_dict(response: &RtspResponse, what: &str) -> Result<Dictionary> {
    plist::from_bytes::<Value>(&response.body)
        .ok()
        .and_then(Value::into_dictionary)
        .ok_or_else(|| FerricastError::Protocol(format!("{what}: reply is not a plist dictionary")))
}

impl AirPlaySession {
    fn identity(&mut self) -> Identity {
        self.identity
            .get_or_insert_with(|| self.store.identity())
            .clone()
    }

    fn link(&self) -> Result<&Link> {
        self.link
            .as_ref()
            .ok_or_else(|| FerricastError::Protocol("AirPlay session is not connected".into()))
    }

    fn reset(&mut self) {
        self.stream = None;
        self.link = None;
        self.paired = PairedSession::default();
        self.state = State::Disconnected;
        self.alive.store(false, Ordering::Relaxed);
    }

    /// Try pair-verify with stored credentials. `Ok(false)` means there are
    /// none or the receiver no longer accepts them.
    async fn verify_stored(
        &mut self,
        rtsp: &mut RtspClient,
        device_id: &str,
        mode: AirPlayPairing,
    ) -> Result<bool> {
        let Some(stored) = self.store.receiver(device_id) else {
            return Ok(false);
        };
        let identity = self.identity();
        let result = match mode {
            AirPlayPairing::LegacyPin => match stored.public_key_bytes() {
                Some(pk) => legacy::pair_verify(rtsp, &identity.key, &pk)
                    .await
                    .map(|ecdh| self.paired.ecdh_secret = Some(ecdh)),
                None => Err(FerricastError::Protocol(
                    "stored receiver key is corrupt".into(),
                )),
            },
            AirPlayPairing::Hap => {
                hap::pair_verify(rtsp, &identity, &stored)
                    .await
                    .and_then(|secret| {
                        let (write, read) = hap::control_keys(&secret)?;
                        rtsp.enable_encryption(&write, &read);
                        self.paired.hap_secret = Some(secret);
                        Ok(())
                    })
            }
            AirPlayPairing::Legacy | AirPlayPairing::HapTransient => return Ok(false),
        };
        match result {
            Ok(()) => Ok(true),
            Err(e) => {
                warn!(error = %e, device_id, "stored AirPlay pairing rejected; pairing again");
                self.store.forget_receiver(device_id);
                Ok(false)
            }
        }
    }

    async fn pair(
        &mut self,
        rtsp: &mut RtspClient,
        device_id: &str,
        mode: AirPlayPairing,
    ) -> Result<ConnectOutcome> {
        let identity = self.identity();
        match mode {
            AirPlayPairing::Legacy => {
                let server_key = legacy::pair_setup(rtsp, &identity.key).await?;
                self.paired.ecdh_secret =
                    Some(legacy::pair_verify(rtsp, &identity.key, &server_key).await?);
                Ok(ConnectOutcome::Ready)
            }
            AirPlayPairing::HapTransient => {
                let (secret, _) = hap::pair_setup(rtsp, &identity, None).await?;
                let (write, read) = hap::control_keys(&secret)?;
                rtsp.enable_encryption(&write, &read);
                self.paired.hap_secret = Some(secret);
                Ok(ConnectOutcome::Ready)
            }
            AirPlayPairing::LegacyPin | AirPlayPairing::Hap => {
                if self.verify_stored(rtsp, device_id, mode).await? {
                    return Ok(ConnectOutcome::Ready);
                }
                legacy_pin::pin_start(rtsp).await?;
                Ok(ConnectOutcome::PairingRequired(PairingChallenge::Pin {
                    digits: 4,
                }))
            }
        }
    }

    async fn finish_pin_pairing(&mut self, pin: &str) -> Result<()> {
        let identity = self.identity();
        let link = self.link()?;
        let (rtsp, device_id, mode) = (link.rtsp.clone(), link.device_id.clone(), link.mode);
        let mut rtsp = rtsp.lock().await;
        match mode {
            AirPlayPairing::LegacyPin => {
                let server_key =
                    legacy_pin::pin_setup(&mut rtsp, &identity.device_id, pin, &identity.key)
                        .await?;
                let key = ReceiverKey {
                    id: String::new(),
                    public_key: hex::encode(server_key),
                };
                if let Err(e) = self.store.save_receiver(&device_id, &key) {
                    warn!(error = %e, "cannot store AirPlay pairing; the PIN will be asked again");
                }
                self.paired.ecdh_secret =
                    Some(legacy::pair_verify(&mut rtsp, &identity.key, &server_key).await?);
            }
            AirPlayPairing::Hap => {
                let (_, key) = hap::pair_setup(&mut rtsp, &identity, Some(pin)).await?;
                let key = key.ok_or_else(|| {
                    FerricastError::Protocol("pair-setup returned no receiver key".into())
                })?;
                if let Err(e) = self.store.save_receiver(&device_id, &key) {
                    warn!(error = %e, "cannot store AirPlay pairing; the PIN will be asked again");
                }
                let secret = hap::pair_verify(&mut rtsp, &identity, &key).await?;
                let (write, read) = hap::control_keys(&secret)?;
                rtsp.enable_encryption(&write, &read);
                self.paired.hap_secret = Some(secret);
            }
            AirPlayPairing::Legacy | AirPlayPairing::HapTransient => {
                return Err(FerricastError::Protocol(
                    "this receiver does not use PIN pairing".into(),
                ));
            }
        }
        Ok(())
    }

    async fn start_stream(&mut self, config: &StreamConfig) -> Result<Stream> {
        let identity = self.identity();
        let link = self.link()?;
        let (rtsp, uri) = (link.rtsp.clone(), link.session_uri.clone());
        let mut guard = rtsp.lock().await;

        let fp = fairplay::setup(&mut guard).await?;
        let base_key = match self.paired.ecdh_secret {
            Some(ecdh) => sha512_16(&[&fp.aes_key, &ecdh]),
            None => fp.aes_key,
        };

        let timing_socket = match config.port_range {
            Some((start, end)) => ferricast_core::bind_udp_in_range(start, end).await?,
            None => UdpSocket::bind(("0.0.0.0", 0)).await?,
        };
        let timing = TimingServer::start(timing_socket)?;

        let mut setup = Dictionary::new();
        setup.insert("deviceID".into(), Value::String(identity.device_id.clone()));
        setup.insert(
            "macAddress".into(),
            Value::String(identity.device_id.clone()),
        );
        setup.insert(
            "sessionUUID".into(),
            Value::String(uuid::Uuid::new_v4().to_string().to_uppercase()),
        );
        setup.insert("name".into(), Value::String("ferricast".into()));
        setup.insert("model".into(), Value::String("Ferricast1,1".into()));
        setup.insert("osName".into(), Value::String("Linux".into()));
        setup.insert("sourceVersion".into(), Value::String("550.10".into()));
        setup.insert("isScreenMirroringSession".into(), Value::Boolean(true));
        setup.insert("timingProtocol".into(), Value::String("NTP".into()));
        setup.insert("timingPort".into(), Value::Integer(timing.port().into()));
        setup.insert("et".into(), Value::Integer(32.into()));
        setup.insert("ekey".into(), Value::Data(fp.ekey));
        setup.insert(
            "eiv".into(),
            Value::Data(rand::random::<[u8; 16]>().to_vec()),
        );
        let reply = guard
            .request("SETUP", &uri, Some(CONTENT_BPLIST), &bplist(setup)?)
            .await?
            .ensure_success("SETUP")?;
        debug!(reply = ?reply_dict(&reply, "SETUP").ok(), "SETUP session reply");

        // Keep the id below 2^63: receivers parse it into signed integers.
        let stream_connection_id = rand::random::<u64>() >> 1;
        let mut stream = Dictionary::new();
        stream.insert("type".into(), Value::Integer(STREAM_TYPE_MIRROR.into()));
        stream.insert(
            "streamConnectionID".into(),
            Value::Integer(stream_connection_id.into()),
        );
        stream.insert("timestampInfo".into(), Value::Array(Vec::new()));
        let mut setup = Dictionary::new();
        setup.insert(
            "streams".into(),
            Value::Array(vec![Value::Dictionary(stream)]),
        );
        let reply = guard
            .request("SETUP", &uri, Some(CONTENT_BPLIST), &bplist(setup)?)
            .await?
            .ensure_success("SETUP mirroring stream")?;
        let data_port = reply_dict(&reply, "SETUP mirroring stream")?
            .get("streams")
            .and_then(Value::as_array)
            .and_then(|s| {
                s.iter()
                    .find_map(|s| s.as_dictionary()?.get("dataPort")?.as_unsigned_integer())
            })
            .and_then(|p| u16::try_from(p).ok())
            .filter(|p| *p != 0)
            .ok_or_else(|| {
                FerricastError::Protocol("SETUP reply has no mirroring dataPort".into())
            })?;

        let data_addr = SocketAddr::new(guard.peer_addr().ip(), data_port);
        let tcp = tokio::time::timeout(DATA_CONNECT_TIMEOUT, TcpStream::connect(data_addr))
            .await
            .map_err(|_| {
                FerricastError::Timeout(format!("mirroring data connection to {data_addr}"))
            })?
            .map_err(|e| {
                FerricastError::Connection(format!("mirroring data connection to {data_addr}: {e}"))
            })?;

        guard
            .request("RECORD", &uri, None, &[])
            .await?
            .ensure_success("RECORD")?;
        drop(guard);

        info!(%data_addr, timing_port = timing.port(), "AirPlay mirroring stream established");
        let feedback = spawn_feedback(rtsp, self.alive.clone());
        Ok(Stream {
            mirror: MirrorStream::new(
                tcp,
                stream_cipher(&base_key, stream_connection_id),
                config.width,
                config.height,
            ),
            _timing: timing,
            _feedback: TaskGuard(feedback),
        })
    }
}

/// Post `/feedback` periodically; the receiver resets the session after a
/// few seconds without it.
fn spawn_feedback(rtsp: Arc<Mutex<RtspClient>>, alive: Arc<AtomicBool>) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(FEEDBACK_INTERVAL);
        tick.tick().await;
        loop {
            tick.tick().await;
            let result = rtsp
                .lock()
                .await
                .request("POST", "/feedback", None, &[])
                .await;
            match result {
                Ok(r) if r.is_success() => {}
                Ok(r) => {
                    warn!(status = r.status, "AirPlay feedback rejected; session lost");
                    alive.store(false, Ordering::Relaxed);
                    break;
                }
                Err(e) => {
                    warn!(error = %e, "AirPlay feedback failed; session lost");
                    alive.store(false, Ordering::Relaxed);
                    break;
                }
            }
        }
    })
}

impl CastSession for AirPlaySession {
    async fn connect(&mut self, device: &Device) -> Result<ConnectOutcome> {
        if device.protocol != "airplay" {
            return Err(FerricastError::Protocol(format!(
                "expected an AirPlay device, got {:?}",
                device.protocol
            )));
        }
        if self.state != State::Disconnected {
            return Err(FerricastError::SessionAlreadyActive(device.name.clone()));
        }
        let config: AirplayConfig =
            device.capabilities.airplay_config.clone().ok_or_else(|| {
                FerricastError::Protocol(format!(
                    "device {:?} has no AirPlay discovery data",
                    device.name
                ))
            })?;
        let mode = config.pairing_mode();
        let device_id = config
            .device_id
            .clone()
            .unwrap_or_else(|| device.addr.to_string());
        info!(addr = %device.addr, port = device.port, ?mode, "connecting to AirPlay receiver");

        let mut rtsp = RtspClient::connect(
            &candidate_addrs(device),
            ClientIdentity::default(),
            mode.hkp_header(),
        )
        .await?;
        let outcome = match self.pair(&mut rtsp, &device_id, mode).await {
            Ok(o) => o,
            Err(e) => {
                self.reset();
                return Err(e);
            }
        };
        let session_uri = format!(
            "rtsp://{}/{}",
            rtsp.local_addr().ip(),
            rand::random::<u32>()
        );
        self.link = Some(Link {
            rtsp: Arc::new(Mutex::new(rtsp)),
            device_id,
            mode,
            session_uri,
        });
        self.state = match outcome {
            ConnectOutcome::Ready => State::Paired,
            ConnectOutcome::PairingRequired(_) => State::AwaitingPin,
        };
        self.alive.store(true, Ordering::Relaxed);
        Ok(outcome)
    }

    async fn submit_pairing(&mut self, response: PairingResponse) -> Result<()> {
        if self.state != State::AwaitingPin {
            return Err(FerricastError::Protocol(
                "AirPlay session is not waiting for a PIN".into(),
            ));
        }
        let pin = match response {
            PairingResponse::Pin(pin) => pin,
            PairingResponse::Cancelled => {
                self.reset();
                return Err(FerricastError::Protocol("AirPlay pairing cancelled".into()));
            }
            PairingResponse::Confirmed => {
                return Err(FerricastError::Protocol(
                    "AirPlay pairing needs a PIN".into(),
                ));
            }
        };
        let pin = pin.trim();
        if pin.len() != 4 || !pin.bytes().all(|b| b.is_ascii_digit()) {
            return Err(FerricastError::Protocol(
                "the AirPlay PIN has 4 digits".into(),
            ));
        }
        match self.finish_pin_pairing(pin).await {
            Ok(()) => {
                self.state = State::Paired;
                info!("AirPlay pairing complete");
                Ok(())
            }
            Err(e) => {
                // Receivers invalidate the PIN after one attempt; a retry
                // starts over with `connect`.
                self.reset();
                Err(e)
            }
        }
    }

    async fn setup_stream(&mut self, config: &StreamConfig) -> Result<()> {
        if self.state != State::Paired {
            return Err(FerricastError::Protocol(format!(
                "cannot set up an AirPlay stream in state {:?}",
                self.state
            )));
        }
        if config.codec != Codec::H264 {
            return Err(FerricastError::UnsupportedCodec {
                codec: config.codec,
                protocol: "airplay",
            });
        }
        let stream = match self.start_stream(config).await {
            Ok(s) => s,
            Err(e) => {
                self.alive.store(false, Ordering::Relaxed);
                return Err(e);
            }
        };
        self.stream = Some(stream);
        self.state = State::Streaming;
        Ok(())
    }

    async fn send_frame(&mut self, frame: &EncodedFrame) -> Result<()> {
        if frame.codec != Codec::H264 {
            return Err(FerricastError::UnsupportedCodec {
                codec: frame.codec,
                protocol: "airplay",
            });
        }
        let Some(stream) = self.stream.as_mut() else {
            return Err(FerricastError::Streaming(format!(
                "cannot send a frame in state {:?}",
                self.state
            )));
        };
        if let Err(e) = stream.mirror.send(frame).await {
            self.alive.store(false, Ordering::Relaxed);
            return Err(FerricastError::Streaming(format!(
                "mirroring connection lost: {e}"
            )));
        }
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        if let Some(mut stream) = self.stream.take() {
            info!(
                frames = stream.mirror.frames_sent(),
                "stopping AirPlay mirroring"
            );
            drop(stream._feedback);
            stream.mirror.shutdown().await;
        }
        if let Some(link) = self.link.take() {
            let mut rtsp = link.rtsp.lock().await;
            let mut stream = Dictionary::new();
            stream.insert("type".into(), Value::Integer(STREAM_TYPE_MIRROR.into()));
            let mut body = Dictionary::new();
            body.insert(
                "streams".into(),
                Value::Array(vec![Value::Dictionary(stream)]),
            );
            // The receiver may close the connection after the first
            // TEARDOWN; neither failure matters on the way out.
            if let Ok(body) = bplist(body) {
                let _ = rtsp
                    .request("TEARDOWN", &link.session_uri, Some(CONTENT_BPLIST), &body)
                    .await;
            }
            let _ = rtsp.request("TEARDOWN", &link.session_uri, None, &[]).await;
        }
        self.reset();
        Ok(())
    }

    fn poll_keyframe_request(&mut self) -> bool {
        self.stream
            .as_mut()
            .is_some_and(|s| s.mirror.take_keyframe_request())
    }

    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed) && matches!(self.state, State::Paired | State::Streaming)
    }
}

impl Drop for AirPlaySession {
    fn drop(&mut self) {
        if self.state != State::Disconnected {
            debug!(state = ?self.state, "AirPlay session dropped without stop()");
        }
    }
}
