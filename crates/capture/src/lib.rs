mod native;
mod pipewire;
mod x11;

pub use native::NativeCapture;
pub use pipewire::PipeWireCapture;
pub use x11::X11Capture;
