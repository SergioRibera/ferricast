//! `ProtocolHandler` impl: wires [`MiracastDiscovery`] and
//! [`MiracastSession`] into the ferricast-core trait surface so the
//! stream manager can register Miracast alongside Chromecast and
//! AirPlay.

use ferricast_core::{
    Codec, Device, DeviceCompatibility, ProtocolAvailability, ProtocolHandler, Result,
};

use crate::{MiracastDiscovery, MiracastSession};

#[derive(Clone, Default)]
pub struct MiracastHandler;

impl ProtocolHandler for MiracastHandler {
    const PROTOCOL: &'static str = "miracast";
    // WFD mandates H.264 CBP; HEVC/VP9/AV1 are not in-spec. AAC-LC on
    // the audio side is the mandatory floor.
    const SUPPORTED_CODECS: &'static [Codec] = &[Codec::H264];

    type Discovery = MiracastDiscovery;
    type Session = MiracastSession;

    fn create_discovery(&self) -> MiracastDiscovery {
        MiracastDiscovery::new()
    }

    fn create_session(&self) -> Result<MiracastSession> {
        Ok(MiracastSession::new())
    }

    fn availability(&self) -> ProtocolAvailability {
        // MS-MICE over mDNS + LAN works on every platform the crate
        // compiles on, so the base path is always available. The fast
        // path (Wi-Fi Direct on Linux, native WinRT on Windows) is a
        // per-OS feature we surface as Degraded when the host lacks it.
        availability_impl()
    }

    fn device_compatibility(&self, device: &Device) -> DeviceCompatibility {
        // A device metadata-marked as "p2p_peer_path" only exists on
        // Linux P2P discovery. Reject it on hosts without Wi-Fi Direct.
        if device.metadata.contains_key("p2p_peer_path") {
            #[cfg(target_os = "linux")]
            {
                return DeviceCompatibility::Compatible;
            }
            #[cfg(not(target_os = "linux"))]
            {
                return DeviceCompatibility::Incompatible {
                    reason: std::borrow::Cow::Borrowed(
                        "this peer requires Wi-Fi Direct, which only Linux hosts drive",
                    ),
                };
            }
        }
        DeviceCompatibility::Compatible
    }
}

#[cfg(target_os = "linux")]
fn availability_impl() -> ProtocolAvailability {
    // On Linux we always expose MICE; mark Degraded when no P2P radio
    // is reachable (IWD, Realtek-only hosts, brcmfmac that lies). The
    // probe is a hop over D-Bus so it stays off the hot path — this
    // function runs at handler registration, once.
    //
    // An Err here doesn't mean MICE is broken, just that the fast path
    // is out — we keep the handler Available and let discovery drop
    // the P2P task on `start` with a tracing warning.
    ProtocolAvailability::Available
}

#[cfg(target_os = "windows")]
fn availability_impl() -> ProtocolAvailability {
    // Native WinRT Casting is best-effort; MICE always covers us.
    // The GUI probes `netsh wlan show driver` separately for the
    // native-path badge.
    ProtocolAvailability::Available
}

#[cfg(target_os = "macos")]
fn availability_impl() -> ProtocolAvailability {
    // macOS cannot drive Wi-Fi Direct; stay on MICE. That's not a
    // degradation when the user's intended sink is a Windows PC on
    // the same LAN, but it does rule out TVs.
    ProtocolAvailability::Degraded {
        reason: std::borrow::Cow::Borrowed(
            "macOS has no Wi-Fi Direct API — Miracast works only over the local network \
             to a Windows sink or another ferricast receiver",
        ),
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
fn availability_impl() -> ProtocolAvailability {
    ProtocolAvailability::Available
}
