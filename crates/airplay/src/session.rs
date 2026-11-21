//! AirPlay screen-mirroring sender session.
//!
//! `connect` pairs with the receiver (asking for a PIN through
//! [`ConnectOutcome::PairingRequired`] when needed), `setup_stream` runs
//! FairPlay setup, both `SETUP` requests and `RECORD` (failing with
//! [`FerricastError::CredentialRequired`] when the receiver wants a
//! password, to be retried after `submit_pairing`), then opens the
//! mirroring data connection; `send_frame` pushes encrypted H.264 access
//! units and a background task posts `/feedback` every two seconds.

use std::net::IpAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ferricast_core::device::{AirPlayPairing, AirplayConfig, Flags};
use ferricast_core::{
    AudioCodec, AudioFrame, CastSession, Codec, ConnectOutcome, Device, EncodedFrame,
    FerricastError, PairingChallenge, PairingResponse, Result, StreamConfig,
};
use plist::{Dictionary, Value};
use tokio::net::{TcpStream, UdpSocket};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::audio::{self, AudioStream, PayloadCipher, STREAM_TYPE_AUDIO};
use crate::crypto::sha512_16;
use crate::digest::DigestChallenge;
use crate::fairplay;
use crate::mirror::{MirrorStream, stream_cipher};
use crate::ntp::TimingServer;
use crate::pair::{PairedSession, hap, legacy, legacy_pin};
use crate::rtsp::{CONTENT_BPLIST, ClientIdentity, RtspClient, RtspResponse};
use crate::store::{CredentialStore, Identity, PasswordState, ReceiverKey};

const FEEDBACK_INTERVAL: Duration = Duration::from_secs(2);
const DATA_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// `SETUP` stream type for screen mirroring.
const STREAM_TYPE_MIRROR: u64 = 110;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Disconnected,
    /// Pair-setup needs the PIN the receiver is showing.
    AwaitingPin,
    /// Paired, but `SETUP` was challenged: the receiver wants its access
    /// password before `setup_stream` is tried again.
    AwaitingPassword,
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
    /// `None` when audio is off or the receiver refused the audio stream.
    audio: Option<AudioStream>,
    _timing: TimingServer,
    /// Connection to the receiver's `eventPort`, held open like iOS does;
    /// `None` when the receiver announces none (UxPlay) or refuses it.
    _event: Option<TcpStream>,
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
    /// Access password for `SETUP` (password-protected receivers).
    password: Option<Password>,
    /// The credential store marks this receiver as wanting a password.
    password_known: bool,
}

/// Access password and where it came from.
#[derive(Clone)]
struct Password {
    value: String,
    /// Read from the credential store rather than typed by the user.
    stored: bool,
}

/// How the receiver controls access, from the `sf` status flags of its
/// RAOP TXT record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Access {
    /// PIN shown on the receiver, entered in pair-setup.
    Pin,
    /// Password checked with HTTP Digest authentication on `SETUP`.
    Password,
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
    async fn identity(&mut self) -> Identity {
        if let Some(identity) = &self.identity {
            return identity.clone();
        }
        let identity = self.store.identity().await;
        self.identity = Some(identity.clone());
        identity
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
        let identity = self.identity().await;
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
        let identity = self.identity().await;
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
        let identity = self.identity().await;
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

    /// Send the first `SETUP`, answering a Digest challenge with the link's
    /// password. A password the receiver accepts is remembered. A challenge
    /// without a password, or a rejected password, fails with
    /// [`FerricastError::CredentialRequired`] and marks the receiver as
    /// wanting a password.
    async fn authorized_setup(
        &self,
        rtsp: &mut RtspClient,
        link: &Link,
        body: &[u8],
    ) -> Result<RtspResponse> {
        let Link {
            session_uri: uri,
            device_id,
            mode,
            password,
            password_known,
            ..
        } = link;
        let reply = rtsp
            .request("SETUP", uri, Some(CONTENT_BPLIST), body)
            .await?;
        if reply.status != 401 {
            let reply = reply.ensure_success("SETUP")?;
            if *password_known {
                info!("AirPlay receiver no longer asks for a password");
                self.store.clear_password(device_id).await;
            }
            return Ok(reply);
        }
        let challenge = reply
            .header("WWW-Authenticate")
            .and_then(DigestChallenge::parse)
            .ok_or_else(|| {
                FerricastError::Protocol("SETUP: 401 without a Digest challenge".into())
            })?;
        self.store.mark_password_required(device_id);
        if *mode == AirPlayPairing::LegacyPin {
            // Password-protected receivers are paired without a PIN from now
            // on, so the key from this connection's PIN pairing is dead
            // weight.
            self.store.forget_receiver(device_id);
        }
        let Some(password) = password else {
            info!("AirPlay receiver requires a password");
            return Err(FerricastError::CredentialRequired { rejected: false });
        };
        let authorization = challenge.authorization("SETUP", uri, &password.value);
        let reply = rtsp
            .request_with(
                "SETUP",
                uri,
                Some(CONTENT_BPLIST),
                body,
                &[("Authorization", &authorization)],
            )
            .await?;
        if reply.status == 401 {
            warn!(
                stored = password.stored,
                "AirPlay receiver rejected the password"
            );
            if password.stored {
                self.store.forget_password(device_id).await;
            }
            return Err(FerricastError::CredentialRequired { rejected: true });
        }
        let reply = reply.ensure_success("SETUP")?;
        if !password.stored {
            self.store.save_password(device_id, &password.value).await;
        }
        info!("AirPlay receiver accepted the password");
        Ok(reply)
    }

    async fn start_stream(&mut self, config: &StreamConfig) -> Result<Stream> {
        let identity = self.identity().await;
        let link = self.link()?;
        let (rtsp, uri) = (link.rtsp.clone(), link.session_uri.clone());
        let mut guard = rtsp.lock().await;

        let fp = fairplay::setup(&mut guard).await?;
        let base_key = match self.paired.ecdh_secret {
            Some(ecdh) => sha512_16(&[&fp.aes_key, &ecdh]),
            None => fp.aes_key,
        };

        let timing = TimingServer::start(bind_udp(config, guard.peer_addr().ip()).await?)?;

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
        let eiv: [u8; 16] = rand::random();
        setup.insert("eiv".into(), Value::Data(eiv.to_vec()));
        let reply = self
            .authorized_setup(&mut guard, link, &bplist(setup)?)
            .await?;
        let reply = reply_dict(&reply, "SETUP").unwrap_or_default();
        debug!(?reply, "SETUP session reply");
        let event = connect_event(&guard, &reply).await;

        // iOS opens the event connection and sends RECORD before it sets up
        // the mirroring stream. Receivers written against iOS (AirScreen)
        // can drop the connection when the mirroring SETUP arrives right
        // behind the session SETUP instead.
        guard
            .request("RECORD", &uri, None, &[])
            .await?
            .ensure_success("RECORD")?;

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
        let reply = reply_dict(&reply, "SETUP mirroring stream")?;
        let data_port =
            reply_stream_port(&reply, STREAM_TYPE_MIRROR, "dataPort").ok_or_else(|| {
                FerricastError::Protocol("SETUP reply has no mirroring dataPort".into())
            })?;

        let data_addr = guard.peer_at(data_port);
        let tcp = tokio::time::timeout(DATA_CONNECT_TIMEOUT, TcpStream::connect(data_addr))
            .await
            .map_err(|_| {
                FerricastError::Timeout(format!("mirroring data connection to {data_addr}"))
            })?
            .map_err(|e| {
                FerricastError::Connection(format!("mirroring data connection to {data_addr}: {e}"))
            })?;

        let audio = match &config.audio {
            Some(_) => match setup_audio(&mut guard, &uri, config, &base_key, eiv).await {
                Ok(audio) => Some(audio),
                Err(e) => {
                    warn!(error = %e, "AirPlay audio stream refused; mirroring without audio");
                    None
                }
            },
            None => None,
        };
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
            audio,
            _timing: timing,
            _event: event,
            _feedback: TaskGuard(feedback),
        })
    }
}

/// Value of `key` in a DNS-SD TXT record (length-prefixed `key=value`
/// strings).
fn txt_value<'a>(txt: &'a [u8], key: &str) -> Option<&'a str> {
    let mut rest = txt;
    while let Some((&len, tail)) = rest.split_first() {
        let (entry, next) = tail.split_at_checked(usize::from(len))?;
        rest = next;
        if let Ok(entry) = std::str::from_utf8(entry)
            && let Some((k, v)) = entry.split_once('=')
            && k.eq_ignore_ascii_case(key)
        {
            return Some(v);
        }
    }
    None
}

/// Access control announced by the `sf` status flags of a RAOP TXT record.
fn access_from_raop_txt(txt: &[u8]) -> Option<Access> {
    let sf = txt_value(txt, "sf")?;
    let bits =
        u32::from_str_radix(sf.trim_start_matches("0x").trim_start_matches("0X"), 16).ok()?;
    let flags = Flags::from_bits_retain(bits);
    if flags.contains(Flags::PIN_REQUIRED) {
        Some(Access::Pin)
    } else if flags.contains(Flags::PASSWORD_REQUIRED) {
        Some(Access::Password)
    } else {
        None
    }
}

/// Ask the receiver for its RAOP TXT record (`GET /info` with the
/// `txtRAOP` qualifier, as UxPlay answers it) and read its access control.
/// `None` when the receiver does not answer or announces none.
async fn probe_access(rtsp: &mut RtspClient) -> Option<Access> {
    let mut body = Dictionary::new();
    body.insert(
        "qualifier".into(),
        Value::Array(vec![Value::String("txtRAOP".into())]),
    );
    let reply = rtsp
        .request("GET", "/info", Some(CONTENT_BPLIST), &bplist(body).ok()?)
        .await
        .inspect_err(|e| debug!(error = %e, "GET /info failed"))
        .ok()?;
    if !reply.is_success() {
        debug!(status = reply.status, "GET /info refused");
        return None;
    }
    let access = reply_dict(&reply, "GET /info")
        .ok()?
        .get("txtRAOP")
        .and_then(Value::as_data)
        .and_then(access_from_raop_txt);
    debug!(?access, "receiver access control");
    access
}

/// UDP socket of `peer`'s address family in `config.port_range`, or on
/// an ephemeral port.
async fn bind_udp(config: &StreamConfig, peer: IpAddr) -> Result<UdpSocket> {
    ferricast_core::bind_udp(peer, config.port_range).await
}

/// Port `key` of the reply's stream entry of type `stream_type`.
fn reply_stream_port(reply: &Dictionary, stream_type: u64, key: &str) -> Option<u16> {
    reply
        .get("streams")?
        .as_array()?
        .iter()
        .filter_map(Value::as_dictionary)
        .find(|s| {
            s.get("type")
                .and_then(Value::as_unsigned_integer)
                .is_none_or(|t| t == stream_type)
        })?
        .get(key)?
        .as_unsigned_integer()
        .and_then(|p| u16::try_from(p).ok())
        .filter(|p| *p != 0)
}

/// Open the event connection to the `eventPort` of the session `SETUP`
/// reply. Receivers that announce 0 or no port get none; a refused or slow
/// connection is not fatal either, mirroring works without it.
async fn connect_event(rtsp: &RtspClient, reply: &Dictionary) -> Option<TcpStream> {
    let port = reply
        .get("eventPort")
        .and_then(Value::as_unsigned_integer)
        .and_then(|p| u16::try_from(p).ok())
        .filter(|p| *p != 0)?;
    let addr = rtsp.peer_at(port);
    match tokio::time::timeout(DATA_CONNECT_TIMEOUT, TcpStream::connect(addr)).await {
        Ok(Ok(tcp)) => {
            debug!(%addr, "AirPlay event connection open");
            Some(tcp)
        }
        Ok(Err(e)) => {
            debug!(%addr, error = %e, "AirPlay event connection refused");
            None
        }
        Err(_) => {
            debug!(%addr, "AirPlay event connection timed out");
            None
        }
    }
}

/// `SETUP` the AAC-ELD audio stream and open its sockets.
async fn setup_audio(
    rtsp: &mut RtspClient,
    uri: &str,
    config: &StreamConfig,
    base_key: &[u8; 16],
    eiv: [u8; 16],
) -> Result<AudioStream> {
    let peer = rtsp.peer_addr().ip();
    let data = bind_udp(config, peer).await?;
    let control = bind_udp(config, peer).await?;
    let control_port = control
        .local_addr()
        .map_err(|e| FerricastError::Connection(format!("audio control socket: {e}")))?
        .port();
    let mut setup = Dictionary::new();
    setup.insert(
        "streams".into(),
        Value::Array(vec![Value::Dictionary(audio::setup_stream_dict(
            control_port,
        ))]),
    );
    let reply = rtsp
        .request("SETUP", uri, Some(CONTENT_BPLIST), &bplist(setup)?)
        .await?
        .ensure_success("SETUP audio stream")?;
    let reply = reply_dict(&reply, "SETUP audio stream")?;
    let port = |key| {
        reply_stream_port(&reply, STREAM_TYPE_AUDIO, key).ok_or_else(|| {
            FerricastError::Protocol(format!("SETUP audio stream reply has no {key}"))
        })
    };
    let data_addr = rtsp.peer_at(port("dataPort")?);
    let control_addr = rtsp.peer_at(port("controlPort")?);
    info!(%data_addr, %control_addr, "AirPlay audio stream established");
    AudioStream::start(
        data,
        control,
        data_addr,
        control_addr,
        PayloadCipher::new(base_key, eiv),
    )
    .await
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
            .unwrap_or_else(|| device.addresses.primary().ip().to_string());
        info!(addresses = ?device.addresses.as_slice(), ?mode, "connecting to AirPlay receiver");

        let mut rtsp = RtspClient::connect(
            device.addresses.as_slice(),
            ClientIdentity::default(),
            mode.hkp_header(),
        )
        .await?;
        // Receivers advertise `pw=true` for both a PIN and a password. The
        // `sf` flags of their RAOP record tell the two apart when they are
        // set (UxPlay 1.73 reports 0x4 for both); otherwise a receiver is
        // known to want a password once a `SETUP` was challenged.
        // Password-protected receivers pair without a PIN and check the
        // password on `SETUP`, which asks for it through
        // `FerricastError::CredentialRequired` when none is stored.
        let known_password = self.store.password(&device_id).await;
        let password_known = known_password != PasswordState::NotRequired;
        let password_protected = password_known
            || (mode == AirPlayPairing::LegacyPin
                && probe_access(&mut rtsp).await == Some(Access::Password));
        let mode = if password_protected && mode == AirPlayPairing::LegacyPin {
            AirPlayPairing::Legacy
        } else {
            mode
        };
        let password = match known_password {
            PasswordState::Required(Some(value)) => Some(Password {
                value,
                stored: true,
            }),
            PasswordState::Required(None) | PasswordState::NotRequired => None,
        };
        if password_protected {
            info!(
                stored = password.is_some(),
                "AirPlay receiver requires a password"
            );
        }
        let outcome = match self.pair(&mut rtsp, &device_id, mode).await {
            Ok(o) => o,
            Err(e) => {
                self.reset();
                return Err(e);
            }
        };
        // An IPv6 host is bracketed and carries no zone: a URI has no
        // usable syntax for one, and receivers only echo it back.
        let host = match rtsp.local_addr().ip() {
            IpAddr::V6(v6) => format!("[{v6}]"),
            ip @ IpAddr::V4(_) => ip.to_string(),
        };
        let session_uri = format!("rtsp://{host}/{}", rand::random::<u32>());
        self.link = Some(Link {
            rtsp: Arc::new(Mutex::new(rtsp)),
            device_id,
            mode,
            session_uri,
            password,
            password_known,
        });
        self.state = match &outcome {
            ConnectOutcome::Ready => State::Paired,
            ConnectOutcome::PairingRequired(_) => State::AwaitingPin,
        };
        self.alive.store(true, Ordering::Relaxed);
        Ok(outcome)
    }

    async fn submit_pairing(&mut self, response: PairingResponse) -> Result<()> {
        if !matches!(self.state, State::AwaitingPin | State::AwaitingPassword) {
            return Err(FerricastError::Protocol(
                "AirPlay session is not waiting for a PIN or password".into(),
            ));
        }
        let pin = match (self.state, response) {
            (_, PairingResponse::Cancelled) => {
                self.reset();
                return Err(FerricastError::Protocol("AirPlay pairing cancelled".into()));
            }
            (State::AwaitingPassword, PairingResponse::Password(value)) => {
                if value.is_empty() {
                    return Err(FerricastError::Protocol(
                        "the AirPlay password is empty".into(),
                    ));
                }
                // The receiver checks it on the next `SETUP`.
                if let Some(link) = self.link.as_mut() {
                    link.password = Some(Password {
                        value,
                        stored: false,
                    });
                }
                self.state = State::Paired;
                return Ok(());
            }
            (State::AwaitingPassword, _) => {
                return Err(FerricastError::Protocol(
                    "this AirPlay receiver needs its password".into(),
                ));
            }
            (_, PairingResponse::Pin(pin)) => pin,
            (_, PairingResponse::Confirmed | PairingResponse::Password(_)) => {
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
            Err(e @ FerricastError::CredentialRequired { .. }) => {
                // The connection stays up: the receiver checks a password
                // on the next `SETUP` without pairing again.
                if let Some(link) = self.link.as_mut() {
                    link.password = None;
                }
                self.state = State::AwaitingPassword;
                return Err(e);
            }
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

    async fn send_audio_frame(&mut self, frame: &AudioFrame) -> Result<()> {
        let Some(audio) = self.stream.as_mut().and_then(|s| s.audio.as_mut()) else {
            // Audio off or refused by the receiver: mirroring goes on silent.
            return Ok(());
        };
        if frame.codec != AudioCodec::AacEld {
            return Err(FerricastError::Streaming(format!(
                "AirPlay mirroring audio is AAC-ELD, got {:?}",
                frame.codec
            )));
        }
        audio
            .send(frame)
            .await
            .map_err(|e| FerricastError::Streaming(format!("AirPlay audio send failed: {e}")))
    }

    fn preferred_audio_codec(&self) -> AudioCodec {
        AudioCodec::AacEld
    }

    async fn stop(&mut self) -> Result<()> {
        let mut stream_types = vec![STREAM_TYPE_MIRROR];
        if let Some(mut stream) = self.stream.take() {
            info!(
                frames = stream.mirror.frames_sent(),
                audio_frames = stream.audio.as_ref().map(AudioStream::frames_sent),
                "stopping AirPlay mirroring"
            );
            if stream.audio.is_some() {
                stream_types.push(STREAM_TYPE_AUDIO);
            }
            drop(stream._feedback);
            stream.mirror.shutdown().await;
        }
        if let Some(link) = self.link.take() {
            let mut rtsp = link.rtsp.lock().await;
            let streams = stream_types
                .into_iter()
                .map(|t| {
                    let mut stream = Dictionary::new();
                    stream.insert("type".into(), Value::Integer(t.into()));
                    Value::Dictionary(stream)
                })
                .collect();
            let mut body = Dictionary::new();
            body.insert("streams".into(), Value::Array(streams));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn txt(entries: &[&str]) -> Vec<u8> {
        entries
            .iter()
            .flat_map(|e| {
                let mut v = vec![u8::try_from(e.len()).unwrap()];
                v.extend_from_slice(e.as_bytes());
                v
            })
            .collect()
    }

    #[test]
    fn raop_status_flags_tell_pin_from_password() {
        // UxPlay 1.73: `-pin` → sf=0x8c, `-pw` → sf=0x84, neither → sf=0x4.
        let base = ["ch=2", "cn=0,1,2,3", "pw=true"];
        let with_sf = |sf: &str| {
            let mut e = base.to_vec();
            e.push(sf);
            txt(&e)
        };
        assert_eq!(access_from_raop_txt(&with_sf("sf=0x8c")), Some(Access::Pin));
        assert_eq!(
            access_from_raop_txt(&with_sf("sf=0x84")),
            Some(Access::Password)
        );
        assert_eq!(access_from_raop_txt(&with_sf("sf=0x4")), None);
        assert_eq!(access_from_raop_txt(&txt(&base)), None);
        // A truncated record yields what was readable before the cut.
        let mut cut = with_sf("sf=0x84");
        cut.truncate(cut.len() - 2);
        assert_eq!(access_from_raop_txt(&cut), None);
    }

    #[test]
    fn reply_stream_port_picks_the_matching_stream() {
        let stream = |t: u64, port: u64| {
            let mut d = Dictionary::new();
            d.insert("type".into(), Value::Integer(t.into()));
            d.insert("dataPort".into(), Value::Integer(port.into()));
            Value::Dictionary(d)
        };
        let mut reply = Dictionary::new();
        reply.insert(
            "streams".into(),
            Value::Array(vec![stream(110, 7100), stream(96, 6000)]),
        );
        assert_eq!(
            reply_stream_port(&reply, STREAM_TYPE_AUDIO, "dataPort"),
            Some(6000)
        );
        assert_eq!(
            reply_stream_port(&reply, STREAM_TYPE_MIRROR, "dataPort"),
            Some(7100)
        );
        assert_eq!(reply_stream_port(&reply, 96, "controlPort"), None);
    }
}
