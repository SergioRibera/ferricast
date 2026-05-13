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

#[cfg(any(feature = "pipewire", feature = "x11"))]
pub use native::NativeCapture;

#[cfg(feature = "pipewire")]
pub use pipewire::PipeWireCapture;
#[cfg(feature = "x11")]
pub use x11::X11Capture;
