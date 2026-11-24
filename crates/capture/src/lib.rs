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
#[cfg(feature = "vulkan")]
mod vulkan;
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
    if src_w <= max_w && src_h <= max_h {
        return (src_w, src_h);
    }
    // Integer form of `scale = min(max_w / src_w, max_h / src_h)`, rounded
    // to nearest; the scaled side is at most `max_*`, so it fits `u32`.
    let (sw, sh, mw, mh) = (
        u64::from(src_w),
        u64::from(src_h),
        u64::from(max_w),
        u64::from(max_h),
    );
    let (w, h) = if mw * sh <= mh * sw {
        (mw, (sh * mw + sw / 2) / sw)
    } else {
        ((sw * mh + sh / 2) / sh, mh)
    };
    let fit = |v: u64| u32::try_from(v).unwrap_or(u32::MAX).max(1);
    (fit(w), fit(h))
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
/// 2. Otherwise → `ferricast_core::StubEnumerator` (reports no
///    capabilities; pickers should defer to the OS portal). This
///    includes X11 sessions, which have no enumerator backend.
///
/// Always returns `Arc<dyn SourceEnumerator>` so callers don't have
/// to feature-gate downstream code: the trait surface stays uniform
/// even when the concrete backend changes at runtime.
pub fn auto_enumerator() -> Arc<dyn SourceEnumerator> {
    #[cfg(feature = "wlroots")]
    {
        let is_wayland = std::env::var_os("WAYLAND_DISPLAY").is_some()
            || std::env::var("XDG_SESSION_TYPE").is_ok_and(|v| v.eq_ignore_ascii_case("wayland"));
        if is_wayland {
            match WaylandSourceEnumerator::try_new() {
                Ok(e) => return Arc::new(e),
                Err(e) => tracing::warn!(
                    %e,
                    "wayland enumerator unavailable, falling through to stub"
                ),
            }
        }
    }

    if std::env::var_os("DISPLAY").is_some() {
        tracing::warn!("X11 source enumeration is not supported, using the stub enumerator");
    }
    Arc::new(ferricast_core::StubEnumerator::new())
}
