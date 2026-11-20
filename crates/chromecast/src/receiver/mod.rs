//! Chromecast receiver-side surface — make this process discoverable
//! as a Cast target and accept LOAD / PLAY / PAUSE / SEEK from
//! senders (YouTube, Spotify, VLC's Cast button, generic Cast SDK
//! apps).
//!
//! Composition:
//! 1. [`ChromecastReceiverAdvertiser`] — mDNS `_googlecast._tcp`
//!    publish with Chromecast-Ultra-shaped TXT records.
//! 2. [`ChromecastReceiverControl`] — TLS server on the advertised
//!    port that speaks CASTV2, handling `connection`/`heartbeat`/
//!    `receiver` namespaces inline and forwarding media-namespace
//!    commands to the manager's pipeline.
//! 3. [`ferricast_hls::HlsPuller`] — every Cast LOAD points at an
//!    HTTP(S) URL (typically an HLS variant); the puller fetches
//!    the playlist + segments and demuxes them into
//!    [`ferricast_core::MediaPacket`]s.
//!
//! Device-auth caveat: Chrome and the Android / iOS Cast SDKs send a
//! `deviceauth` challenge right after connecting and only continue
//! when it is signed by a device certificate chaining to Google's
//! Cast root CA. Those certificates are issued to licensed hardware
//! only, so this receiver answers with an auth error and those
//! senders drop it. Senders that skip device authentication drive it
//! fine: ferricast's own Chromecast sender, pychromecast-based tools
//! (catt, mkchromecast), go-chromecast, and anything else that speaks
//! plain CASTV2 to the Default Media Receiver (`CC1AD845`).

mod advertiser;
mod control;
mod tls;
mod util;

use std::net::SocketAddr;
use std::sync::{Arc, Mutex, PoisonError, Weak};

use ferricast_core::{
    AdvertiseInfo, Codec, DeviceCapabilities, FerricastError, H264Profile, H265Profile,
    ReceiverProtocol, Result,
};
use ferricast_hls::HlsPuller;
use tokio::net::{TcpListener, TcpSocket};
use uuid::Uuid;

pub use advertiser::ChromecastReceiverAdvertiser;
pub use control::ChromecastReceiverControl;

/// Receiver-side protocol handler. Construct via [`Self::new`] or
/// `Default` — both pick port 8009 (the canonical Cast TLS port)
/// and a randomly-generated device id that persists for the
/// process's lifetime.
#[derive(Clone)]
pub struct ChromecastReceiver {
    inner: Arc<Inner>,
}

struct Inner {
    port: u16,
    device_id: String,
    friendly_name: String,
    /// The bound TLS listener, shared by every live control. Held
    /// weakly so the port is released once all controls are gone
    /// (receivers stopped) and re-bound by the next `create_control`.
    listener: Mutex<Weak<TcpListener>>,
    /// TLS config and volume, created on first use.
    shared: Mutex<Weak<Shared>>,
}

/// Receiver volume (Cast `volume` object).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Volume {
    pub(crate) level: f64,
    pub(crate) muted: bool,
}

impl Default for Volume {
    fn default() -> Self {
        Self {
            level: 1.0,
            muted: false,
        }
    }
}

/// State shared by every control of one receiver: one TLS identity
/// and one volume, whichever sender is asking.
pub(crate) struct Shared {
    tls_config: Arc<rustls::server::ServerConfig>,
    volume: Mutex<Volume>,
}

impl Shared {
    pub(crate) fn volume(&self) -> Volume {
        *self.volume.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn set_volume_level(&self, level: f64) {
        self.volume
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .level = level;
    }

    pub(crate) fn set_muted(&self, muted: bool) {
        self.volume
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .muted = muted;
    }
}

impl ChromecastReceiver {
    pub fn new(friendly_name: impl Into<String>) -> Self {
        Self::with_port(friendly_name, 8009)
    }

    pub fn with_port(friendly_name: impl Into<String>, port: u16) -> Self {
        Self {
            inner: Arc::new(Inner {
                port,
                device_id: Uuid::new_v4().simple().to_string(),
                friendly_name: friendly_name.into(),
                listener: Mutex::new(Weak::new()),
                shared: Mutex::new(Weak::new()),
            }),
        }
    }

    /// The TLS listener, bound on first use and shared until every
    /// control holding it is dropped.
    fn listener(&self) -> Result<Arc<TcpListener>> {
        let mut slot = self
            .inner
            .listener
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(listener) = slot.upgrade() {
            return Ok(listener);
        }
        // Binding needs a runtime with I/O enabled; fail instead of
        // panicking when called outside one.
        tokio::runtime::Handle::try_current().map_err(|e| {
            FerricastError::Receiver(format!("receiver control needs a tokio runtime: {e}"))
        })?;
        let addr = SocketAddr::from(([0, 0, 0, 0], self.inner.port));
        let bind = || -> std::io::Result<TcpListener> {
            let socket = TcpSocket::new_v4()?;
            // Rebinding right after a restart must not fail on
            // connections lingering in TIME_WAIT.
            socket.set_reuseaddr(true)?;
            socket.bind(addr)?;
            socket.listen(128)
        };
        let listener = Arc::new(
            bind().map_err(|e| FerricastError::Receiver(format!("bind tcp {addr}: {e}")))?,
        );
        tracing::info!(
            addr = %listener.local_addr().unwrap_or(addr),
            "Chromecast receiver TLS listener up"
        );
        *slot = Arc::downgrade(&listener);
        Ok(listener)
    }

    fn shared(&self) -> Result<Arc<Shared>> {
        let mut slot = self
            .inner
            .shared
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(shared) = slot.upgrade() {
            return Ok(shared);
        }
        // The TLS cert SANs must include every host the sender will
        // resolve us as. Most senders dial the IP they got from mDNS
        // directly (`192.168.x.x`), so the crucial SAN is the
        // LAN-facing IPv4. Without it, strict-validating TLS clients
        // (VLC mobile's GnuTLS in particular) abort the handshake.
        // `FERRICAST_CAST_INTERFACE` pins the IP like the advertiser.
        let ips = util::detect_advertise_ips();
        let shared = Arc::new(Shared {
            tls_config: tls::build_server_config(&ips)?,
            volume: Mutex::new(Volume::default()),
        });
        *slot = Arc::downgrade(&shared);
        Ok(shared)
    }
}

impl Default for ChromecastReceiver {
    fn default() -> Self {
        Self::new("Ferricast")
    }
}

impl ReceiverProtocol for ChromecastReceiver {
    const PROTOCOL: &'static str = "chromecast";
    const SUPPORTED_CODECS: &'static [Codec] = &[Codec::H264, Codec::H265];

    type Advertiser = ChromecastReceiverAdvertiser;
    type Control = ChromecastReceiverControl;
    type Puller = HlsPuller;

    fn create_advertiser(&self) -> Self::Advertiser {
        ChromecastReceiverAdvertiser::new()
    }

    /// Binds the shared listener on first use, so a port conflict
    /// surfaces here, before anything is advertised.
    fn create_control(&self) -> Result<Self::Control> {
        Ok(ChromecastReceiverControl::new(
            self.listener()?,
            self.shared()?,
        ))
    }

    fn create_puller(&self) -> Result<Self::Puller> {
        Ok(HlsPuller::new())
    }

    fn advertise_info(&self) -> AdvertiseInfo {
        let mut txt = std::collections::HashMap::new();
        // Belt-and-braces: the advertiser's defaults already cover
        // these, but pinning here keeps the handler's `advertise_info`
        // self-describing.
        txt.insert("ve".to_string(), "05".to_string());
        AdvertiseInfo {
            friendly_name: self.inner.friendly_name.clone(),
            port: self.inner.port,
            capabilities: DeviceCapabilities {
                supports_audio: true,
                supports_video: true,
                supports_screen_mirror: false,
                max_width: Some(3840),
                max_height: Some(2160),
                max_fps: Some(60),
                max_bitrate_kbps: Some(35_000),
                max_audio_bitrate_kbps: Some(320),
                requires_audio: false,
                max_h264_profile: Some(H264Profile::High),
                max_h265_profile: Some(H265Profile::Main10),
                supported_codecs: vec![Codec::H264, Codec::H265],
                supports_low_latency_hls: false,
                supports_cast_streaming: false,
                airplay_config: None,
            },
            device_id: self.inner.device_id.clone(),
            icon: bytes::Bytes::new(),
            txt,
        }
    }
}
