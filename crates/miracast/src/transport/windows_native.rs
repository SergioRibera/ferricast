//! Windows-native Miracast sender via WinRT `Windows.Media.Casting`.
//!
//! Compiled only when `target_os = "windows"` AND the
//! `windows-native` feature is enabled. The receiver path
//! (`Windows.Media.Miracast.MiracastReceiver`) requires an MSIX-packaged
//! app with the `wiFiControl` + `userAccountInformation` capabilities
//! and is intentionally out of scope for the library crate.
//!
//! Status: scaffolding only. Exact `CastingDevicePicker` →
//! `CastingConnection` → `RequestStartCastingAsync` orchestration lives
//! in follow-up work; this module today documents the surface so the
//! handler can build with the feature on.

use super::TransportError;

/// Probe whether the host's Wi-Fi driver advertises Miracast.
///
/// `netsh wlan show driver` reports `Wireless Display Supported: Yes`
/// when the WDI driver is present. We spawn the process and parse the
/// line. On a desktop without Wi-Fi we return `Unavailable`.
pub async fn probe_wireless_display() -> Result<bool, TransportError> {
    let output = tokio::process::Command::new("netsh")
        .args(["wlan", "show", "driver"])
        .output()
        .await
        .map_err(|e| TransportError::WindowsNative(format!("netsh spawn: {e}")))?;
    let text = String::from_utf8_lossy(&output.stdout);
    let supported = text
        .lines()
        .any(|l| l.contains("Wireless Display Supported") && l.contains("Yes"));
    Ok(supported)
}
