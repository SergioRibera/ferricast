//! Platform-specific transport bring-up for Miracast's control and
//! media sockets.
//!
//! The shared protocol stack ([`crate::rtsp`], [`crate::ts`],
//! [`crate::rtp`]) doesn't know how the peer became reachable — it
//! just wants a `TcpStream` for RTSP and the right UDP socket pair for
//! RTP. Each sub-module here produces exactly that:
//!
//! - [`mice`]: portable path over mDNS + TCP on the normal LAN
//!   ([MS-MICE]). Only option on macOS; used as Linux/Windows fallback.
//! - [`linux_p2p`]: wpa_supplicant D-Bus driving Wi-Fi Direct. Only on
//!   Linux, only when the Wi-Fi adapter supports P2P-Device iftype and
//!   NetworkManager hasn't exclusively claimed the radio.
//! - [`windows_native`]: WinRT `Windows.Media.Casting` sender shim.
//!
//! [MS-MICE]: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-mice/

pub mod mice;

#[cfg(target_os = "linux")]
pub mod dhcp;
#[cfg(target_os = "linux")]
pub mod linux_p2p;

#[cfg(target_os = "windows")]
pub mod windows_native;

use std::net::SocketAddr;

use thiserror::Error;

/// How a discovered peer can be reached.
#[derive(Debug, Clone)]
pub enum PeerAddress {
    /// MS-MICE sink reachable over the existing LAN.
    Mice {
        mdns_name: String,
        rtsp_addr: SocketAddr,
        mice_addr: SocketAddr,
    },
    /// Wi-Fi Direct P2P peer discovered via wpa_supplicant.
    #[cfg(target_os = "linux")]
    LinuxP2p {
        /// supplicant's `Peer` D-Bus object path.
        peer_path: String,
        /// Device name advertised in P2P discovery (shown to the user).
        name: String,
        /// WFD IE bytes parsed out of the peer's `IEs` property.
        wfd_ie: bytes::Bytes,
    },
    /// Windows native sender — the WinRT stack handles negotiation
    /// internally. We only carry the picker's device ID.
    #[cfg(target_os = "windows")]
    WindowsNative { device_id: String, name: String },
}

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("mDNS/MICE: {0}")]
    Mice(String),
    #[cfg(target_os = "linux")]
    #[error("wpa_supplicant D-Bus: {0}")]
    LinuxP2p(String),
    #[cfg(target_os = "windows")]
    #[error("Windows Media Casting: {0}")]
    WindowsNative(String),
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("no transport available for this platform")]
    Unavailable,
}
