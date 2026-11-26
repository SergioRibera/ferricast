//! Screen and audio capture for ferricast, over the istmo
//! screen-capture plugin (`istmo.screen_capture`).
//!
//! The plugin owns every capture backend: wlr-screencopy and
//! ext-image-copy on Wayland (GBM DMA-BUFs, no dialog), X11, the
//! xdg-desktop-portal with PipeWire, and PipeWire audio. This crate maps
//! its frames onto ferricast's types without copying pixels:
//!
//! - [`NativeCapture`]: [`ferricast_core::ScreenCapture`], DMA-BUFs as
//!   `GpuFrame`s with a Vulkan readback importer attached, shared memory
//!   as [`ferricast_core::RawFrame`]s.
//! - [`NativeAudioCapture`]: [`ferricast_core::AudioCapture`], S16 PCM.
//! - [`auto_enumerator`]: [`ferricast_core::SourceEnumerator`] for
//!   in-process pickers, with thumbnails.
//!
//! Timestamps are the plugin's `wall_us`: UNIX-epoch microseconds, the
//! clock the HLS segmenter measures staleness on. On platforms the
//! plugin has no backend for yet, opening a capture fails with its
//! `NoBackendCompiled`.

mod audio;
mod plugin;
mod sources;
mod video;
#[cfg(target_os = "linux")]
mod vulkan;

use std::sync::Arc;

use ferricast_core::SourceEnumerator;

pub use audio::NativeAudioCapture;
pub use sources::NativeSourceEnumerator;
pub use video::{BACKEND_ENV_VAR, NativeCapture};

/// The source enumerator for this session: [`NativeSourceEnumerator`]
/// where the plugin lists monitors or windows without a dialog,
/// otherwise `ferricast_core::StubEnumerator` (no capabilities; pickers
/// defer to the portal, as on GNOME / Mutter).
///
/// Always returns `Arc<dyn SourceEnumerator>`, so callers treat
/// enumeration as always present and capability-gated.
pub fn auto_enumerator() -> Arc<dyn SourceEnumerator> {
    match NativeSourceEnumerator::try_new() {
        Ok(enumerator) => Arc::new(enumerator),
        Err(e) => {
            tracing::warn!(%e, "no source enumeration here, using the stub enumerator");
            Arc::new(ferricast_core::StubEnumerator::new())
        }
    }
}
