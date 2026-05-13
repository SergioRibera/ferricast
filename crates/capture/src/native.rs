use ferricast_core::ScreenCapture;

#[cfg(feature = "pipewire")]
use crate::PipeWireCapture;
#[cfg(feature = "x11")]
use crate::X11Capture;

/// Auto-selecting capture backend.
///
/// `NativeCapture::new` picks a concrete backend at runtime based on
/// `FERRICAST_CAPTURE` (explicit override) or `XDG_SESSION_TYPE`
/// (session-aware default). Each variant is gated by the matching
/// Cargo feature so this enum only carries backends that were
/// actually compiled in.
pub enum NativeCapture {
    #[cfg(feature = "x11")]
    X11(X11Capture),
    #[cfg(feature = "pipewire")]
    Pipewire(PipeWireCapture),
}

impl NativeCapture {
    pub fn new() -> Self {
        if let Ok(v) = std::env::var("FERRICAST_CAPTURE") {
            return match v.as_str() {
                #[cfg(feature = "pipewire")]
                "pipewire" => Self::Pipewire(PipeWireCapture::new()),
                #[cfg(feature = "x11")]
                "x11" => Self::X11(X11Capture::new()),
                other => panic!(
                    "FERRICAST_CAPTURE={other:?} but the matching backend was \
                     not compiled in (check the `pipewire` / `x11` features)"
                ),
            };
        }

        if let Ok(v) = std::env::var("XDG_SESSION_TYPE") {
            match v.as_str() {
                #[cfg(feature = "pipewire")]
                "wayland" => return Self::Pipewire(PipeWireCapture::new()),
                #[cfg(feature = "x11")]
                "x11" => return Self::X11(X11Capture::new()),
                _ => {}
            }
        }

        Self::fallback()
    }

    /// Picks whichever backend is compiled in. If both are compiled,
    /// prefers PipeWire because it covers Wayland (the modern default)
    /// and also works as the X11 portal backend on many setups.
    fn fallback() -> Self {
        #[cfg(feature = "pipewire")]
        {
            Self::Pipewire(PipeWireCapture::new())
        }
        #[cfg(all(not(feature = "pipewire"), feature = "x11"))]
        {
            Self::X11(X11Capture::new())
        }
    }
}

impl ScreenCapture for NativeCapture {
    async fn start(
        &mut self,
        source: ferricast_core::CaptureSource,
        config: ferricast_core::CaptureConfig,
    ) -> ferricast_core::Result<()> {
        match self {
            #[cfg(feature = "x11")]
            Self::X11(x) => x.start(source, config).await,
            #[cfg(feature = "pipewire")]
            Self::Pipewire(pipe) => pipe.start(source, config).await,
        }
    }
    fn get_pixel_format(&self) -> ferricast_core::PixelFormat {
        match self {
            #[cfg(feature = "x11")]
            Self::X11(x) => x.get_pixel_format(),
            #[cfg(feature = "pipewire")]
            Self::Pipewire(pw) => pw.get_pixel_format(),
        }
    }
    fn get_screen_size(&self) -> (usize, usize) {
        match self {
            #[cfg(feature = "x11")]
            Self::X11(x) => x.get_screen_size(),
            #[cfg(feature = "pipewire")]
            Self::Pipewire(pw) => pw.get_screen_size(),
        }
    }

    fn get_framerate(&self) -> u32 {
        match self {
            #[cfg(feature = "x11")]
            Self::X11(x) => x.get_framerate(),
            #[cfg(feature = "pipewire")]
            Self::Pipewire(pw) => pw.get_framerate(),
        }
    }

    async fn next_frame(&mut self) -> ferricast_core::Result<ferricast_core::CapturedFrame> {
        match self {
            #[cfg(feature = "x11")]
            Self::X11(x) => x.next_frame().await,
            #[cfg(feature = "pipewire")]
            Self::Pipewire(pipe) => pipe.next_frame().await,
        }
    }

    async fn stop(&mut self) -> ferricast_core::Result<()> {
        match self {
            #[cfg(feature = "x11")]
            Self::X11(x) => x.stop().await,
            #[cfg(feature = "pipewire")]
            Self::Pipewire(pipe) => pipe.stop().await,
        }
    }

    fn is_running(&self) -> bool {
        match self {
            #[cfg(feature = "x11")]
            Self::X11(x) => x.is_running(),
            #[cfg(feature = "pipewire")]
            Self::Pipewire(pipe) => pipe.is_running(),
        }
    }
}
