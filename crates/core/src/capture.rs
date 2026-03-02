use crate::error::Result;
use crate::frame::RawFrame;

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

    fn next_frame(&mut self) -> impl Future<Output = Result<RawFrame>> + Send;
    fn stop(&mut self) -> impl Future<Output = Result<()>> + Send;
    fn is_running(&self) -> bool;
}
