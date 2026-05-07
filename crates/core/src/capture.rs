use crate::error::Result;
use crate::frame::CapturedFrame;
use crate::PixelFormat;

#[derive(Debug, Clone)]
pub enum CaptureSource {
    FullScreen { monitor: Option<String> },
    Window { identifier: WindowIdentifier },
}

#[derive(Debug, Clone)]
pub enum WindowIdentifier {
    Title(String),
    Id(u64),
}

#[derive(Debug, Clone)]
pub struct CaptureConfig {
    pub fps: u32,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub show_cursor: bool,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            fps: 30,
            width: None,
            height: None,
            show_cursor: true,
        }
    }
}

pub trait ScreenCapture: Send {
    fn start(
        &mut self,
        source: CaptureSource,
        config: CaptureConfig,
    ) -> impl Future<Output = Result<()>> + Send;

    /// Pull the next captured frame.
    ///
    /// Implementations may return either a CPU-resident frame
    /// (`CapturedFrame::Cpu`) or a GPU-resident DMA-BUF
    /// (`CapturedFrame::Gpu`). Encoders that need CPU bytes call
    /// `CapturedFrame::into_cpu()` to trigger a readback on demand.
    fn next_frame(&mut self) -> impl Future<Output = Result<CapturedFrame>> + Send;
    fn stop(&mut self) -> impl Future<Output = Result<()>> + Send;
    fn is_running(&self) -> bool;
    fn get_pixel_format(&self) -> PixelFormat;
    fn get_screen_size(&self) -> (usize, usize);
}
