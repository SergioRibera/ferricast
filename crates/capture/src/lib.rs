// Pick at least one capture backend — `ferricast-capture` exists
// solely to wire one to `ferricast_core::ScreenCapture`, so building
// without either is always a configuration error rather than a
// "headless" build. Fail loudly here instead of letting the user hit
// a baffling "no variants" diagnostic on `NativeCapture` later.
#[cfg(not(any(feature = "pipewire", feature = "wayland-direct")))]
compile_error!(
    "ferricast-capture requires at least one capture backend feature to be enabled: \
     `pipewire` (Wayland / xdg-desktop-portal) \
     Enable one in your Cargo.toml, e.g. `ferricast-capture = { ..., features = [\"pipewire\"] }`."
);

#[cfg(feature = "pipewire")]
mod native;

#[cfg(feature = "pipewire")]
mod pipewire;
#[cfg(feature = "pipewire")]
mod pipewire_audio;
#[cfg(feature = "wayland-direct")]
mod wayland_direct;
#[cfg(feature = "wlroots")]
mod wayland_thumb;
#[cfg(feature = "wlroots")]
mod wlroots_enum;


/// Largest `(w, h)` that fits inside `(max_w, max_h)` while keeping
/// the aspect ratio of `(src_w, src_h)`. Both dimensions are
/// clamped to at least 1 so degenerate inputs don't produce a
/// 0-sized image buffer downstream. Shared by every thumbnail
/// backend so picker output stays size-consistent across protocols.
#[cfg(feature = "wlroots")]
pub(crate) fn fit_box(src_w: u32, src_h: u32, max_w: u32, max_h: u32) -> (u32, u32) {
    if src_w == 0 || src_h == 0 || max_w == 0 || max_h == 0 {
        return (1, 1);
    }
    let scale = (max_w as f32 / src_w as f32).min(max_h as f32 / src_h as f32);
    if scale >= 1.0 {
        return (src_w, src_h);
    }
    let w = ((src_w as f32 * scale).round() as u32).max(1);
    let h = ((src_h as f32 * scale).round() as u32).max(1);
    (w, h)
}

#[cfg(feature = "pipewire")]
pub use native::NativeCapture;

#[cfg(feature = "pipewire")]
pub use pipewire::PipeWireCapture;
#[cfg(feature = "pipewire")]
pub use pipewire_audio::PipeWireAudioCapture;
#[cfg(feature = "wayland-direct")]
pub use wayland_direct::WaylandDirectCapture;
#[cfg(feature = "wlroots")]
pub use wlroots_enum::WaylandSourceEnumerator;


use std::sync::Arc;

use ferricast_core::SourceEnumerator;

/// Best-effort source enumerator for the running session.
///
/// Resolution order:
///
/// 1. `XDG_SESSION_TYPE=wayland` (or `WAYLAND_DISPLAY` set) →
///    [`WaylandSourceEnumerator`]. Falls through if the compositor
///    advertises neither `zwlr_foreign_toplevel_management_v1` nor
///    `ext_foreign_toplevel_list_v1`.
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
            match WaylandSourceEnumerator::try_new() {
                Ok(e) => return Arc::new(e),
                Err(e) => tracing::info!(
                    %e,
                    "wayland enumerator unavailable, falling through (compositor exposes neither zwlr_foreign_toplevel_management_v1 nor ext_foreign_toplevel_list_v1)"
                ),
            }
        }
    }

    {
        if std::env::var_os("DISPLAY").is_some() {
            panic!("X11 is not supported");
        }
    }
    Arc::new(ferricast_core::StubEnumerator::new())
}
