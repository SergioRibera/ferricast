//! Windows-native Miracast sender shim for the `Windows.Media.Casting`
//! stack.
//!
//! Scope: enumerate every `CastingDevice` the OS has discovered
//! (Miracast + DLNA + Bluetooth), filter to video-capable Miracast
//! peers, and expose their id + friendly name so the GUI can hand the
//! user a list without needing the built-in `CastingDevicePicker`
//! flyout (which requires a WinRT-registered window handle).
//!
//! The `CastingConnection` + `RequestStartCastingAsync` path (i.e.
//! actually casting *content*) is kept out of the library: it needs a
//! `MediaPlayer` / `MediaElement` the GUI owns. The GUI builds that
//! and calls into the `windows` crate directly; this module is here
//! so callers have a one-call enumerator + a `netsh` capability probe.
//!
//! Compiled only when `target_os = "windows"` AND the `windows-native`
//! feature is enabled.

use super::TransportError;

/// A Miracast / Cast device the host's `Windows.Media.Casting` stack
/// is aware of.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowsCastDevice {
    /// `DeviceInformation.Id` — opaque, stable per-device. Hand it to
    /// `CastingDevice.FromIdAsync` from the GUI to re-materialise it.
    pub id: String,
    /// Friendly name the user sees in Win+K.
    pub name: String,
}

/// Enumerate every video-capable casting target the OS has indexed.
///
/// Returns an empty vector on hosts without a Miracast-capable Wi-Fi
/// driver — not an error, because non-Wi-Fi laptops still ship as
/// valid ferricast installs.
pub async fn list_cast_devices() -> Result<Vec<WindowsCastDevice>, TransportError> {
    use windows::Devices::Enumeration::DeviceInformation;
    use windows::Media::Casting::{CastingDevice, CastingPlaybackTypes};

    let selector = CastingDevice::GetDeviceSelector(CastingPlaybackTypes::Video)
        .map_err(|e| TransportError::WindowsNative(format!("GetDeviceSelector: {e}")))?;

    let op = DeviceInformation::FindAllAsyncAqsFilter(&selector)
        .map_err(|e| TransportError::WindowsNative(format!("FindAllAsync: {e}")))?;
    let collection = op
        .get()
        .map_err(|e| TransportError::WindowsNative(format!("FindAllAsync get: {e}")))?;

    let mut out = Vec::new();
    for dev_info in collection {
        let id = dev_info
            .Id()
            .map_err(|e| TransportError::WindowsNative(format!("Id: {e}")))?
            .to_string();
        let name = dev_info
            .Name()
            .map_err(|e| TransportError::WindowsNative(format!("Name: {e}")))?
            .to_string();
        out.push(WindowsCastDevice { id, name });
    }
    Ok(out)
}

/// Probe whether the host's Wi-Fi driver reports Miracast capability.
///
/// `netsh wlan show driver` prints `Wireless Display Supported: Yes/No`.
/// We shell out because the WinRT surface that exposes the same bit
/// (`WiFiAdapter.WiFiDirectSupported`) is MSIX-gated. On a host
/// without `netsh` this returns `Ok(false)`: let the caller fall back
/// to the portable MICE path.
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
