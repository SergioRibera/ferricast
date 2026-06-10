use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::PixelFormat;
use crate::error::Result;
use crate::frame::{AudioFrame, CapturedFrame};

#[derive(Debug, Clone)]
pub enum CaptureSource {
    FullScreen {
        monitor: Option<String>,
    },
    Window {
        identifier: Option<WindowIdentifier>,
    },
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

    /// Effective framerate the source is currently delivering at.
    ///
    /// For backends that negotiate (PipeWire / portal) this returns
    /// the value the compositor agreed to — which is what the encoder
    /// must be configured with so PTS spacing matches real frame
    /// arrival cadence. Returns `0` before negotiation completes;
    /// callers should treat that as "use the configured fps fallback".
    ///
    /// Default impl returns `0`; backends that don't negotiate (X11
    /// pull, native polling) override only if they have a real value.
    fn get_framerate(&self) -> u32 {
        0
    }
}

/// Where to grab the live audio stream from.
///
/// `Default` is the PipeWire default output sink's monitor — i.e.
/// "everything coming out of the speakers right now", which is what
/// users expect when they tick "share computer audio" during a
/// screencast. `Node` is an explicit PipeWire object id used when
/// the UI lets the user pick a specific source/sink-monitor (e.g.
/// the captured app's own output via pavucontrol-style routing).
#[derive(Debug, Clone)]
pub enum AudioSource {
    /// Monitor the default sink (system-wide output capture). The
    /// backend resolves this to whatever sink PipeWire reports as
    /// default at start time; subsequent default-sink changes do
    /// not migrate the active stream — restart capture to pick up
    /// a new default.
    DefaultMonitor,
    /// Bind directly to a PipeWire node id (use `pw-dump` /
    /// `pw-cli ls Node` to discover). Useful when the user wants
    /// "just this app's audio" via a manual route in pavucontrol.
    Node(u32),
}

impl Default for AudioSource {
    fn default() -> Self {
        AudioSource::DefaultMonitor
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AudioCaptureConfig {
    /// Target sample rate (Hz). PipeWire/WirePlumber will resample
    /// transparently if the actual sink uses something else. 48000
    /// is the chromecast-friendly value (AAC LC + ADTS at 48 kHz is
    /// the universal HLS audio config).
    pub sample_rate: u32,
    /// Channel count (1 = mono, 2 = stereo). Chromecast and every
    /// other HLS receiver expects stereo.
    pub channels: u16,
}

impl Default for AudioCaptureConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            channels: 2,
        }
    }
}

/// Runtime knob for "mute the live audio stream without tearing the
/// capture down". Held as `Arc<AtomicBool>` so the UI thread can
/// flip it while the PipeWire worker thread reads it on every
/// buffer. When set to `true` the capture backend MUST keep
/// producing frames at the negotiated cadence with PCM samples
/// zeroed — dropping frames instead would stall the audio elementary
/// stream on the receiver side (HLS players block waiting for the
/// audio PID to advance and report BUFFERING).
#[derive(Debug, Clone, Default)]
pub struct AudioMuteHandle {
    inner: Arc<AtomicBool>,
}

impl AudioMuteHandle {
    pub fn new(initial: bool) -> Self {
        Self {
            inner: Arc::new(AtomicBool::new(initial)),
        }
    }

    pub fn is_muted(&self) -> bool {
        self.inner.load(Ordering::Relaxed)
    }

    pub fn set_muted(&self, muted: bool) {
        self.inner.store(muted, Ordering::Relaxed);
    }
}

/// Trait for any source of PCM audio frames. Symmetric with
/// [`ScreenCapture`]: the runtime selects a concrete backend
/// ([`PipeWireAudioCapture`] today), the manager pulls frames in a
/// dedicated tokio task, and pushes them through an
/// [`crate::AudioEncoder`].
///
/// The mute handle is taken at `start()` so the backend can read it
/// inline on the buffer hot path without needing a separate channel.
pub trait AudioCapture: Send {
    fn start(
        &mut self,
        source: AudioSource,
        config: AudioCaptureConfig,
        mute: AudioMuteHandle,
    ) -> impl Future<Output = Result<()>> + Send;

    /// Pull the next chunk of PCM samples. Returns `AudioCodec::Pcm`
    /// in `AudioFrame::codec` with interleaved S16LE samples in
    /// `data`. The `timestamp_us` field is the same monotonic
    /// `CLOCK_MONOTONIC`-equivalent the screen-capture path uses, so
    /// downstream PTS computation can share one origin.
    fn next_frame(&mut self) -> impl Future<Output = Result<AudioFrame>> + Send;

    fn stop(&mut self) -> impl Future<Output = Result<()>> + Send;

    fn is_running(&self) -> bool;

    /// Negotiated sample rate (Hz). Returns 0 before negotiation
    /// completes — callers should default to the configured value.
    fn sample_rate(&self) -> u32 {
        0
    }

    /// Negotiated channel count. Returns 0 before negotiation
    /// completes.
    fn channels(&self) -> u16 {
        0
    }
}
