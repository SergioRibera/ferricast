use ferricast_core::ScreenCapture;

use crate::{PipeWireCapture, X11Capture};

pub enum NativeCapture {
    X11(X11Capture),
    Pipewire(PipeWireCapture),
}

impl NativeCapture {
    pub fn new() -> Self {
        if let Ok(v) = std::env::var("FERRICAST_CAPTURE") {  
            return match v.as_str() {
                "pipewire" => Self::Pipewire(PipeWireCapture::new()),
                "x11" => Self::X11(X11Capture::new()),
                _ => panic!("Invalid capture"),
            };
        }

        if let Ok(v) = std::env::var("XDG_SESSION_TYPE") {
            return match v.as_str() {
                "wayland" => Self::Pipewire(PipeWireCapture::new()),
                "x11" => Self::X11(X11Capture::new()),
                _ => panic!("Invalid window system"),
            };

        }
    
        panic!("Cannot create capture")
    }
}

impl ScreenCapture for NativeCapture {
    async fn start(
            &mut self,
            source: ferricast_core::CaptureSource,
            config: ferricast_core::CaptureConfig,
        ) -> ferricast_core::Result<()> {
        match self {
            Self::X11(x) => x.start(source, config).await,
            Self::Pipewire(pipe) => pipe.start(source, config).await,
        }
    }
    fn get_pixel_format(&self) -> ferricast_core::PixelFormat {
        match self {
            Self::X11(x) => x.get_pixel_format(),
            Self::Pipewire(pw) => pw.get_pixel_format(), 
        }
    }
    fn get_screen_size(&self) -> (usize, usize) {
        match self {
            Self::X11(x) => x.get_screen_size(),
            Self::Pipewire(pw) => pw.get_screen_size(),
        }
    }

    fn get_framerate(&self) -> u32 {
        match self {
            Self::X11(x) => x.get_framerate(),
            Self::Pipewire(pw) => pw.get_framerate(),
        }
    }

    async fn next_frame(&mut self) -> ferricast_core::Result<ferricast_core::CapturedFrame> {
        match self {
            Self::X11(x) => x.next_frame().await,
            Self::Pipewire(pipe) => pipe.next_frame().await,
        }
    }

    async fn stop(&mut self) -> ferricast_core::Result<()> {
        match self {
            Self::X11(x) => x.stop().await,
            Self::Pipewire(pipe) => pipe.stop().await,
        }
    }

    fn is_running(&self) -> bool {
        match self {
            Self::X11(x) => x.is_running(),
            Self::Pipewire(pipe) => pipe.is_running()
        }
    }
}
