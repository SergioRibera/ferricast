// Pick at least one capture backend — `ferricast-capture` exists
// solely to wire one to `ferricast_core::ScreenCapture`, so building
// without either is always a configuration error rather than a
// "headless" build. Fail loudly here instead of letting the user hit
// a baffling "no variants" diagnostic on `NativeCapture` later.
#[cfg(not(any(feature = "pipewire", feature = "x11")))]
compile_error!(
    "ferricast-capture requires at least one capture backend feature to be enabled: \
     `pipewire` (Wayland / xdg-desktop-portal) and/or `x11`. \
     Enable one in your Cargo.toml, e.g. `ferricast-capture = { ..., features = [\"pipewire\"] }`."
);

#[cfg(any(feature = "pipewire", feature = "x11"))]
mod native;

#[cfg(feature = "pipewire")]
mod pipewire;
#[cfg(feature = "x11")]
mod x11;
#[cfg(feature = "x11")]
mod x11_enum;
#[cfg(feature = "wlroots")]
mod wlroots_enum;

#[cfg(any(feature = "pipewire", feature = "x11"))]
pub use native::NativeCapture;

#[cfg(feature = "pipewire")]
pub use pipewire::PipeWireCapture;
#[cfg(feature = "x11")]
pub use x11::X11Capture;
#[cfg(feature = "x11")]
pub use x11_enum::X11SourceEnumerator;
#[cfg(feature = "wlroots")]
pub use wlroots_enum::WlrootsSourceEnumerator;

use std::sync::Arc;

use ferricast_core::SourceEnumerator;

/// Best-effort source enumerator for the running session.
///
/// Resolution order:
///
/// 1. `XDG_SESSION_TYPE=wayland` (or `WAYLAND_DISPLAY` set) →
///    [`WlrootsSourceEnumerator`]. Falls through if the compositor
///    doesn't advertise `zwlr_foreign_toplevel_manager_v1`.
/// 2. `DISPLAY` set → [`X11SourceEnumerator`]. Falls through on
///    connection failure.
/// 3. Otherwise → `ferricast_core::StubEnumerator` (reports no
///    capabilities; pickers should defer to the OS portal).
///
/// Always returns `Arc<dyn SourceEnumerator>` so callers don't have
/// to feature-gate downstream code: the trait surface stays uniform
/// even when the concrete backend changes at runtime.
pub fn auto_enumerator() -> Arc<dyn SourceEnumerator> {
    #[cfg(feature = "wlroots")]
    {
        let is_wayland = std::env::var_os("WAYLAND_DISPLAY").is_some()
            || std::env::var("XDG_SESSION_TYPE")
                .map(|v| v.eq_ignore_ascii_case("wayland"))
                .unwrap_or(false);
        if is_wayland {
            match WlrootsSourceEnumerator::try_new() {
                Ok(e) => return Arc::new(e),
                Err(e) => tracing::info!(
                    %e,
                    "wlroots enumerator unavailable, falling through (compositor probably doesn't expose wlr-foreign-toplevel-management)"
                ),
            }
        }
    }
    #[cfg(feature = "x11")]
    {
        if std::env::var_os("DISPLAY").is_some() {
            match X11SourceEnumerator::try_new() {
                Ok(e) => return Arc::new(e),
                Err(e) => tracing::info!(%e, "x11 enumerator unavailable"),
            }
        }
    }
    Arc::new(ferricast_core::StubEnumerator::new())
}
