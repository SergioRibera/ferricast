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
//! Device-auth caveat: modern Cast senders (YouTube, Spotify) issue
//! a `deviceauth` challenge before LOADing and refuse to talk to a
//! receiver that can't sign with a Google Cast CA cert. Ferricast
//! advertises and accepts the wire correctly but can't satisfy
//! that signature (no third-party software can — the certs are
//! issued only to licensed manufacturers). So:
//!
//! - Senders that bypass auth in dev mode: works.
//! - VLC, BubbleUPnP receivers, casttube, Stream2Chromecast: works.
//! - YouTube official: connects but won't LOAD.
//! - Spotify official: same.
//!
//! Tracking the YouTube-TV-Android emulation path as future work.

mod advertiser;
mod control;
mod tls;
mod util;

use std::sync::Arc;

use ferricast_core::{
    AdvertiseInfo, Codec, DeviceCapabilities, H264Profile, ReceiverProtocol, Result,
};
use ferricast_hls::HlsPuller;
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
            }),
        }
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

    fn create_control(&self) -> Result<Self::Control> {
        // The TLS cert SANs must include every host the sender will
        // resolve us as. Most senders dial the IP they got from mDNS
        // directly (`192.168.x.x`), so the *crucial* SAN is the
        // LAN-facing IPv4. Without it, strict-validating TLS clients
        // (VLC mobile's GnuTLS in particular) abort the handshake
        // with a generic "Interrupted system call" error before any
        // Cast traffic flows.
        //
        // The same detection logic the mDNS advertiser uses runs
        // here; if the user pinned a specific NIC via
        // `FERRICAST_CAST_INTERFACE`, we honour the pinned IP.
        let ips = util::detect_advertise_ips();
        ChromecastReceiverControl::new(self.inner.port, ips)
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
                requires_audio: false,
                max_h264_profile: Some(H264Profile::High),
                supported_codecs: vec![Codec::H264, Codec::H265],
                supports_low_latency_hls: false,
            },
            device_id: self.inner.device_id.clone(),
            icon: bytes::Bytes::new(),
            txt,
        }
    }
}

