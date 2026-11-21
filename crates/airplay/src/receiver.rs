//! AirPlay receiver: make this process an AirPlay screen-mirroring and
//! audio target for iPhones, iPads, Macs and ferricast's own sender.
//!
//! It speaks the legacy AirPlay mirroring protocol of the open-source
//! receivers (UxPlay, RPiPlay), which every Apple sender still uses with
//! them:
//!
//! 1. [`AirPlayReceiverAdvertiser`] publishes `_airplay._tcp` and
//!    `_raop._tcp` with an AppleTV3,2-shaped TXT record.
//! 2. [`AirPlayReceiverControl`] serves the RTSP control connection:
//!    `GET /info`, legacy pairing (optionally with a PIN shown to the
//!    user), FairPlay `fp-setup`, `SETUP` of the mirroring (type 110) and
//!    audio (type 96) streams, volume, `TEARDOWN`.
//! 3. The media connections the sender opens are received, decrypted and
//!    handed to the pipeline as a [`ferricast_core::PushedStream`]
//!    through [`ferricast_core::MediaCommand::Push`]; the
//!    [`ferricast_core::ReceiverProtocol::Puller`] is never used.
//!
//! Every port is fixed (see [`AirPlayReceiverConfig`]) so a firewall rule
//! can let senders in; `0` picks an ephemeral port instead.

mod advertiser;
mod assembler;
mod audio;
mod clock;
mod connection;
mod mirror;
mod pairing;
mod rtsp;
mod server;

use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError, Weak};

use ferricast_core::{
    AdvertiseInfo, Codec, DeviceCapabilities, FerricastError, H264Profile, ReceiverProtocol, Result,
};

pub use advertiser::AirPlayReceiverAdvertiser;
pub use server::AirPlayReceiverControl;

use crate::store::{CredentialStore, Identity};
use server::Server;

/// Model the receiver claims to be: the newest model whose mirroring
/// protocol open-source receivers implement.
const MODEL: &str = "AppleTV3,2";
/// AirPlay server version announced (`srcvers`, `Server: AirTunes/…`).
const SOURCE_VERSION: &str = "220.68";
/// Feature bits 0-31 (UxPlay's set): mirroring, audio, FairPlay, legacy
/// pairing, no AirPlay video (HLS).
const FEATURES_LOW: u32 = 0x5A7F_FEE6;
/// Feature bit 42 (`SupportsScreenMultiCodec`, in the high word) lets
/// senders mirror in HEVC.
const FEATURE_SCREEN_MULTI_CODEC: u32 = 1 << (42 - 32);

/// Where the receiver keeps its identity and the senders paired with a PIN.
#[derive(Debug, Clone, Default)]
pub enum CredentialStorage {
    /// `$XDG_DATA_HOME/ferricast/airplay`, the Ed25519 seed in the system
    /// keyring (or a 0600 file without one).
    #[default]
    Default,
    /// Another directory, same layout.
    Dir(PathBuf),
    /// Nothing is kept: a new identity on every start, PINs asked every
    /// time. For tests.
    Ephemeral,
}

impl CredentialStorage {
    fn store(&self) -> CredentialStore {
        match self {
            Self::Default => CredentialStore::default(),
            Self::Dir(dir) => CredentialStore::in_dir(dir),
            Self::Ephemeral => CredentialStore::ephemeral(),
        }
    }
}

/// Configuration of the AirPlay receiver.
#[derive(Debug, Clone)]
pub struct AirPlayReceiverConfig {
    /// Name shown in the senders' AirPlay menu.
    pub name: String,
    /// RTSP control port (TCP), the advertised one.
    pub rtsp_port: u16,
    /// Mirroring data port (TCP).
    pub mirror_port: u16,
    /// Audio data port (UDP).
    pub audio_data_port: u16,
    /// Audio control port (UDP): sync packets, retransmissions.
    pub audio_control_port: u16,
    /// Local port the timing (NTP) queries to the sender go out from
    /// (UDP; the replies come back to it).
    pub timing_port: u16,
    /// Senders must type a 4-digit PIN shown on this side, once per
    /// sender (`pair-pin-start`).
    pub require_pin: bool,
    /// Accept HEVC mirroring (advertises feature bit 42).
    pub hevc: bool,
    /// Display size and refresh rate offered to senders.
    pub width: u32,
    pub height: u32,
    pub max_fps: u32,
    pub credentials: CredentialStorage,
}

impl Default for AirPlayReceiverConfig {
    /// The port set of `uxplay -p`: 7000/7001/7100 TCP, 6000/6001/7011 UDP.
    fn default() -> Self {
        Self {
            name: "Ferricast".into(),
            rtsp_port: 7000,
            mirror_port: 7100,
            audio_data_port: 6000,
            audio_control_port: 6001,
            timing_port: 7011,
            require_pin: false,
            hevc: false,
            width: 1920,
            height: 1080,
            max_fps: 60,
            credentials: CredentialStorage::Default,
        }
    }
}

impl AirPlayReceiverConfig {
    /// All ports 0: the OS picks them. For tests.
    #[must_use]
    pub fn with_ephemeral_ports(mut self) -> Self {
        self.rtsp_port = 0;
        self.mirror_port = 0;
        self.audio_data_port = 0;
        self.audio_control_port = 0;
        self.timing_port = 0;
        self
    }
}

/// The receiver's long-term identity: what `deviceid` and `pk` announce.
#[derive(Clone)]
struct ReceiverIdentity {
    /// Six bytes shown as a MAC address.
    device_id: [u8; 6],
    key: ed25519_dalek::SigningKey,
    /// Stable per-identity UUID for the `pi` TXT key.
    pi: String,
}

impl ReceiverIdentity {
    fn from_store(identity: Identity) -> Self {
        let mut device_id = [0_u8; 6];
        for (byte, part) in device_id.iter_mut().zip(identity.device_id.split(':')) {
            *byte = u8::from_str_radix(part, 16).unwrap_or_default();
        }
        Self {
            device_id,
            key: identity.key,
            pi: identity.pairing_id.to_lowercase(),
        }
    }

    /// `aa:bb:cc:dd:ee:ff`, the `_airplay._tcp` form.
    fn device_id_colons(&self) -> String {
        self.device_id
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join(":")
    }

    /// `AABBCCDDEEFF`, the `_raop._tcp` instance-name prefix.
    fn device_id_hex(&self) -> String {
        self.device_id.iter().map(|b| format!("{b:02X}")).collect()
    }

    fn public_key_hex(&self) -> String {
        hex::encode(self.key.verifying_key().as_bytes())
    }
}

/// Everything fixed for the receiver's lifetime.
struct Settings {
    config: AirPlayReceiverConfig,
    identity: ReceiverIdentity,
    store: CredentialStore,
}

impl Settings {
    fn features(&self) -> (u32, u32) {
        let high = if self.config.hevc {
            FEATURE_SCREEN_MULTI_CODEC
        } else {
            0
        };
        (FEATURES_LOW, high)
    }

    fn features_txt(&self) -> String {
        let (low, high) = self.features();
        format!("0x{low:X},0x{high:X}")
    }

    /// TXT record of `_airplay._tcp`.
    fn airplay_txt(&self) -> Vec<(&'static str, String)> {
        let id = &self.identity;
        vec![
            ("deviceid", id.device_id_colons()),
            ("features", self.features_txt()),
            ("pw", self.config.require_pin.to_string()),
            ("flags", "0x4".into()),
            ("model", MODEL.into()),
            ("pk", id.public_key_hex()),
            ("pi", id.pi.clone()),
            ("srcvers", SOURCE_VERSION.into()),
            ("vv", "2".into()),
        ]
    }

    /// TXT record of `_raop._tcp`. `sf` 0x8c tells senders the password is
    /// a PIN shown on screen.
    fn raop_txt(&self) -> Vec<(&'static str, String)> {
        let pin = self.config.require_pin;
        vec![
            ("ch", "2".into()),
            ("cn", "0,1,2,3".into()),
            ("da", "true".into()),
            ("et", "0,3,5".into()),
            ("vv", "2".into()),
            ("ft", self.features_txt()),
            ("am", MODEL.into()),
            ("md", "0,1,2".into()),
            ("rhd", "5.6.0.0".into()),
            ("pw", pin.to_string()),
            ("sf", if pin { "0x8c" } else { "0x4" }.into()),
            ("sr", "44100".into()),
            ("ss", "16".into()),
            ("sv", "false".into()),
            ("tp", "UDP".into()),
            ("txtvers", "1".into()),
            ("vs", SOURCE_VERSION.into()),
            ("vn", "65537".into()),
            ("pk", self.identity.public_key_hex()),
        ]
    }

    /// `_raop._tcp` instance name: `AABBCCDDEEFF@name`.
    fn raop_name(&self) -> String {
        format!("{}@{}", self.identity.device_id_hex(), self.config.name)
    }
}

/// DNS-SD wire form of a TXT record (length-prefixed `key=value`).
fn txt_bytes(txt: &[(&'static str, String)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (k, v) in txt {
        let entry = format!("{k}={v}");
        let Ok(len) = u8::try_from(entry.len()) else {
            continue;
        };
        out.push(len);
        out.extend_from_slice(entry.as_bytes());
    }
    out
}

/// Receiver-side protocol handler for AirPlay. Cheap to clone.
#[derive(Clone)]
pub struct AirPlayReceiver {
    settings: Arc<Settings>,
    /// The control server, shared by every live control and released
    /// when the last one is dropped (receivers stopped).
    server: Arc<Mutex<Weak<Server>>>,
}

impl AirPlayReceiver {
    /// Load (or create) the receiver identity and build the handler.
    pub async fn new(config: AirPlayReceiverConfig) -> Self {
        let store = config.credentials.store();
        let identity = ReceiverIdentity::from_store(store.receiver_identity().await);
        Self {
            settings: Arc::new(Settings {
                config,
                identity,
                store,
            }),
            server: Arc::new(Mutex::new(Weak::new())),
        }
    }

    /// The configuration in use.
    #[must_use]
    pub fn config(&self) -> &AirPlayReceiverConfig {
        &self.settings.config
    }

    /// The RTSP port actually bound, once a control exists (useful with
    /// ephemeral ports).
    #[must_use]
    pub fn bound_rtsp_port(&self) -> Option<u16> {
        self.server
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .upgrade()
            .map(|s| s.port())
    }

    fn server(&self) -> Result<Arc<Server>> {
        let mut slot = self.server.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(server) = slot.upgrade() {
            return Ok(server);
        }
        tokio::runtime::Handle::try_current().map_err(|e| {
            FerricastError::Receiver(format!("AirPlay receiver needs a tokio runtime: {e}"))
        })?;
        let server = Server::bind(self.settings.clone())?;
        *slot = Arc::downgrade(&server);
        Ok(server)
    }
}

/// Never used: AirPlay media is pushed through
/// [`ferricast_core::MediaCommand::Push`]. Fails if asked to open a URL.
pub struct NoPuller;

impl ferricast_core::MediaPuller for NoPuller {
    async fn open(&mut self, _spec: ferricast_core::PullSpec) -> Result<ferricast_core::MediaInfo> {
        Err(FerricastError::Pull(
            "the AirPlay receiver does not pull URLs".into(),
        ))
    }
    async fn next(&mut self) -> Result<ferricast_core::MediaPacket> {
        Err(FerricastError::Pull("nothing to pull".into()))
    }
    async fn seek(&mut self, _position_us: u64) -> Result<()> {
        Err(FerricastError::Pull("nothing to seek".into()))
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

impl ReceiverProtocol for AirPlayReceiver {
    const PROTOCOL: &'static str = "airplay";
    const SUPPORTED_CODECS: &'static [Codec] = &[Codec::H264, Codec::H265];

    type Advertiser = AirPlayReceiverAdvertiser;
    type Control = AirPlayReceiverControl;
    type Puller = NoPuller;

    fn create_advertiser(&self) -> Self::Advertiser {
        AirPlayReceiverAdvertiser::new(self.settings.clone(), self.server.clone())
    }

    /// Binds the control port on first use, so a port conflict surfaces
    /// before anything is advertised.
    fn create_control(&self) -> Result<Self::Control> {
        Ok(AirPlayReceiverControl::new(self.server()?))
    }

    fn create_puller(&self) -> Result<Self::Puller> {
        Ok(NoPuller)
    }

    fn advertise_info(&self) -> AdvertiseInfo {
        let s = &self.settings;
        AdvertiseInfo {
            friendly_name: s.config.name.clone(),
            port: s.config.rtsp_port,
            capabilities: DeviceCapabilities {
                supports_audio: true,
                supports_video: true,
                supports_screen_mirror: true,
                max_width: Some(s.config.width),
                max_height: Some(s.config.height),
                max_fps: Some(s.config.max_fps),
                max_h264_profile: Some(H264Profile::High),
                supported_codecs: if s.config.hevc {
                    vec![Codec::H264, Codec::H265]
                } else {
                    vec![Codec::H264]
                },
                ..Default::default()
            },
            device_id: s.identity.device_id_colons(),
            icon: bytes::Bytes::new(),
            txt: s
                .airplay_txt()
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn txt_records_follow_the_pin_setting() {
        let mut config = AirPlayReceiverConfig {
            credentials: CredentialStorage::Ephemeral,
            ..Default::default()
        };
        let r = AirPlayReceiver::new(config.clone()).await;
        let txt = r.settings.airplay_txt();
        let get = |txt: &[(&str, String)], k: &str| {
            txt.iter()
                .find(|(key, _)| *key == k)
                .map(|(_, v)| v.clone())
        };
        assert_eq!(get(&txt, "features").as_deref(), Some("0x5A7FFEE6,0x0"));
        assert_eq!(get(&txt, "pw").as_deref(), Some("false"));
        assert_eq!(get(&txt, "deviceid").unwrap().len(), 17);
        config.require_pin = true;
        config.hevc = true;
        let r = AirPlayReceiver::new(config).await;
        let raop = r.settings.raop_txt();
        assert_eq!(get(&raop, "sf").as_deref(), Some("0x8c"));
        assert_eq!(get(&raop, "ft").as_deref(), Some("0x5A7FFEE6,0x400"));
        assert!(r.settings.raop_name().contains('@'));
        let bytes = txt_bytes(&raop);
        assert_eq!(usize::from(bytes[0]), "ch=2".len());
    }
}
