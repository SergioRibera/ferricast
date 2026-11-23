//! Wi-Fi Direct via wpa_supplicant D-Bus — Linux only.
//!
//! This module is a thin typed shell over the supplicant's
//! `fi.w1.wpa_supplicant1`, `.Interface.P2PDevice`, and `.Peer`
//! interfaces. The protocol details (WFDIEs byte layout, Primary
//! Device Type, GroupStarted role pivot) live in
//! [`crate::ie`] and the per-platform plan in memory.
//!
//! Compiled only when `target_os = "linux"` AND the `linux-p2p`
//! feature is enabled. Even then we don't load wpa_supplicant at
//! build time — the whole crate stays usable on hosts without P2P
//! radios through the MS-MICE fallback.
//!
//! Status: scaffolding only. The zbus proxies and the GroupStarted
//! → RTSP pivot are tracked as follow-up work; this module today
//! documents the surface so the handler can build with the feature
//! turned on.

use super::TransportError;

/// Probe whether the host has a wpa_supplicant we can talk to.
///
/// Returns `Ok(())` when the bus is reachable; a user-friendly error
/// otherwise so the GUI can surface "install wpa_supplicant" /
/// "NetworkManager has exclusive control" / "adapter lacks P2P-Device
/// iftype" without the caller touching D-Bus at all.
pub async fn probe_supplicant() -> Result<(), TransportError> {
    let conn = zbus::Connection::system()
        .await
        .map_err(|e| TransportError::LinuxP2p(format!("system bus: {e}")))?;
    let _proxy = zbus::Proxy::new(
        &conn,
        "fi.w1.wpa_supplicant1",
        "/fi/w1/wpa_supplicant1",
        "fi.w1.wpa_supplicant1",
    )
    .await
    .map_err(|e| TransportError::LinuxP2p(format!("supplicant proxy: {e}")))?;
    Ok(())
}
