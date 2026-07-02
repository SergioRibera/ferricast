use ferricast_core::ScreenCapture;

#[cfg(feature = "pipewire")]
use crate::PipeWireCapture;
#[cfg(feature = "wayland-direct")]
use crate::WaylandDirectCapture;


/// Auto-selecting capture backend.
///
/// `NativeCapture::new` picks a concrete backend at runtime based on
/// `FERRICAST_CAPTURE` (explicit override) or `XDG_SESSION_TYPE`
/// (session-aware default). Each variant is gated by the matching
/// Cargo feature so this enum only carries backends that were
/// actually compiled in.
///
/// On Wayland the resolution order is:
///
///   1. `WaylandDirectCapture` — talks `wlr-screencopy` directly to
///      the compositor, allocates DMA-BUFs via gbm, emits
///      `CapturedFrame::Gpu`. No portal dialog. Available on every
///      Wayland compositor except GNOME / Mutter.
///   2. `PipeWireCapture` — xdg-desktop-portal flow. Pops a picker
///      dialog (the portal owns source selection), but covers
///      every compositor including GNOME.
///
/// The chain is evaluated at `start()` time: if `WaylandDirectCapture`
/// can't bind the protocols or the source can't be resolved, the
/// outer wrapper transparently retries on PipeWire (when its feature
/// is enabled).
pub enum NativeCapture {
    #[cfg(feature = "wayland-direct")]
    WaylandDirect(WaylandDirectCapture),
    #[cfg(feature = "pipewire")]
    Pipewire(PipeWireCapture),
}

impl NativeCapture {
    pub fn new() -> Self {
        if let Ok(v) = std::env::var("FERRICAST_CAPTURE") {
            return match v.as_str() {
                #[cfg(feature = "pipewire")]
                "pipewire" => Self::Pipewire(PipeWireCapture::new()),
                #[cfg(feature = "wayland-direct")]
                "wayland-direct" => Self::WaylandDirect(WaylandDirectCapture::new()),
                "x11" => panic!("X11 is not supported"),
                other => panic!(
                    "FERRICAST_CAPTURE={other:?} but the matching backend was \
                     not compiled in (check the `pipewire` / \
                     `wayland-direct` features)"
                ),
            };
        }

        if let Ok(v) = std::env::var("XDG_SESSION_TYPE") {
            match v.as_str() {
                #[cfg(feature = "wayland-direct")]
                "wayland" => return Self::WaylandDirect(WaylandDirectCapture::new()),
                #[cfg(all(not(feature = "wayland-direct"), feature = "pipewire"))]
                "wayland" => return Self::Pipewire(PipeWireCapture::new()),
                "x11" => panic!("X11 is not supported"),
                _ => {}
            }
        }

        Self::fallback()
    }

    /// Whichever backend is compiled in, in order of preference:
    /// wayland-direct (GPU, no portal) → pipewire (portal) → x11.
    fn fallback() -> Self {
        #[cfg(feature = "wayland-direct")]
        {
            return Self::WaylandDirect(WaylandDirectCapture::new());
        }
        #[cfg(all(not(feature = "wayland-direct"), feature = "pipewire"))]
        {
            return Self::Pipewire(PipeWireCapture::new());
        }
        #[cfg(all(
            not(feature = "wayland-direct"),
            not(feature = "pipewire"),
        ))]
        {
            panic!("X11 is not supported") 
        }
    }
}

impl ScreenCapture for NativeCapture {
    async fn start(
        &mut self,
        source: ferricast_core::CaptureSource,
        config: ferricast_core::CaptureConfig,
    ) -> ferricast_core::Result<()> {
        // On the Wayland-direct path, fall through to PipeWire (when
        // available) on any `start` failure. `WaylandDirectCapture`
        // is the modern preferred path but can fail for legitimate
        // reasons — compositor without wlr-screencopy (GNOME), no
        // DRM render node, dmabuf negotiation rejected. PipeWire
        // works around all of those at the cost of a portal dialog.
        #[cfg(all(feature = "wayland-direct", feature = "pipewire"))]
        if let Self::WaylandDirect(direct) = self {
            match direct.start(source.clone(), config.clone()).await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    tracing::warn!(%e, "wayland-direct failed, falling back to PipeWire/portal");
                    *self = Self::Pipewire(PipeWireCapture::new());
                    // Re-dispatch through the regular match below.
                }
            }
        }
        match self {
            #[cfg(feature = "wayland-direct")]
            Self::WaylandDirect(d) => d.start(source, config).await,
            #[cfg(feature = "pipewire")]
            Self::Pipewire(pipe) => pipe.start(source, config).await,
        }
    }
    fn get_pixel_format(&self) -> ferricast_core::PixelFormat {
        match self {
            #[cfg(feature = "wayland-direct")]
            Self::WaylandDirect(d) => d.get_pixel_format(),
            #[cfg(feature = "pipewire")]
            Self::Pipewire(pw) => pw.get_pixel_format(),
        }
    }
    fn get_screen_size(&self) -> (usize, usize) {
        match self {
            #[cfg(feature = "wayland-direct")]
            Self::WaylandDirect(d) => d.get_screen_size(),
            #[cfg(feature = "pipewire")]
            Self::Pipewire(pw) => pw.get_screen_size(),
        }
    }

    fn get_framerate(&self) -> u32 {
        match self {
            #[cfg(feature = "wayland-direct")]
            Self::WaylandDirect(d) => d.get_framerate(),
            #[cfg(feature = "pipewire")]
            Self::Pipewire(pw) => pw.get_framerate(),
        }
    }

    async fn next_frame(&mut self) -> ferricast_core::Result<ferricast_core::CapturedFrame> {
        match self {
            #[cfg(feature = "wayland-direct")]
            Self::WaylandDirect(d) => d.next_frame().await,
            #[cfg(feature = "pipewire")]
            Self::Pipewire(pipe) => pipe.next_frame().await,
        }
    }

    async fn stop(&mut self) -> ferricast_core::Result<()> {
        match self {
            #[cfg(feature = "wayland-direct")]
            Self::WaylandDirect(d) => d.stop().await,
            #[cfg(feature = "pipewire")]
            Self::Pipewire(pipe) => pipe.stop().await,
        }
    }

    fn is_running(&self) -> bool {
        match self {
            #[cfg(feature = "wayland-direct")]
            Self::WaylandDirect(d) => d.is_running(),
            #[cfg(feature = "pipewire")]
            Self::Pipewire(pipe) => pipe.is_running(),
        }
    }
}
