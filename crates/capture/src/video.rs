//! [`NatieCapture`]: ferricast's screen capture over the plugin.

mod formats;
mod frames;
mod target;

#[cfg(target_os = "linux")]
use std::sync::Arc;

use ferricast_core::{
    CaptureConfig, CaptureSource, CapturedFrame, FerricastError, PixelFormat, Result, ScreenCapture,
};
use istmo_core::{SinkPolicy, StreamItem};
use istmo_screen_capture::{
    CaptureEvent, CursorMode, CursorRequest, Exhaustion, FrameRate, LinuxOptions, NegotiatedStream,
    OrientationHandling, PipeWireConnection, ScreenCaptureClientExt as _, SessionConfig,
    SessionHandle, SourceFilter, SourceKind, Strength,
};
use tracing::{debug, info, warn};

use self::formats::{Formats, Readable};
use self::frames::FrameMapper;
pub use self::target::BACKEND_ENV_VAR;
use self::target::{Target, pinned_backend};
use crate::plugin::{Plugin, capture_error};
#[cfg(target_os = "linux")]
use crate::vulkan::VulkanImporter;

/// Frames queued between the plugin and [`ScreenCapture::next_frame`].
/// A consumer that falls behind gets the newest ones: older frames are
/// evicted, which also hands their buffers back to the producer.
const SINK_DEPTH: usize = 2;

/// Screen capture through the istmo screen-capture plugin.
///
/// The plugin picks the backend (`FERRICAST_CAPTURE` pins one, see
/// [`BACKEND_ENV_VAR`]): on Wayland wlr-screencopy or ext-image-copy
/// into GBM DMA-BUFs with no portal dialog, on X11 shared memory. When
/// that cannot open the source (GNOME / Mutter, a window no backend
/// captures by id), the xdg-desktop-portal dialog asks the user, as the
/// portal is the one path left.
///
/// DMA-BUF frames go out as [`CapturedFrame::Gpu`] with the Vulkan
/// importer attached, so CPU encoders can still read them back; CPU
/// frames are the plugin's shared memory itself, not a copy.
#[derive(Default)]
pub struct NativeCapture {
    running: Option<Running>,
}

struct Running {
    session: SessionHandle,
    mapper: FrameMapper,
    fps: u32,
    size: (usize, usize),
    format: Option<PixelFormat>,
}

impl NativeCapture {
    /// A capture that opens on [`ScreenCapture::start`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn running(&self) -> Result<&Running> {
        self.running
            .as_ref()
            .ok_or_else(|| FerricastError::Capture("capture: not started".into()))
    }
}

/// The session options `config` asks for, the source and backend aside.
fn session_config(config: &CaptureConfig, formats: Formats<'_>, target: &Target) -> SessionConfig {
    let (source, backend) = target.selection();
    let cursor = CursorRequest {
        mode: if config.show_cursor {
            CursorMode::Embedded
        } else {
            CursorMode::Hidden
        },
        strength: Strength::Prefer,
    };
    let mut session = SessionConfig::of(source)
        .backend(backend)
        .formats(formats.request())
        .fps(FrameRate::Max(config.fps.max(1)))
        .cursor(cursor)
        // Damage rectangles where the backend has them. wlr-screencopy is
        // damage-driven either way (an idle output costs nothing and the
        // stream's pacer repeats the last frame);
        // `ISTMO_SCREEN_CAPTURE_FULL_COPY=1` asks for every frame instead.
        .damage(Strength::Prefer)
        // Frames keep flowing while the consumer holds every buffer it
        // may: the compositor copies into a CPU frame instead.
        .exhaustion(Exhaustion::CopyToCpu)
        // The orientation travels with the frame; this crate turns what
        // the consumer does not (one copy either way, and none at all for
        // a consumer that turns GPU frames).
        .orientation(OrientationHandling::InMeta);
    session.linux = LinuxOptions {
        pipewire: PipeWireConnection::SessionDaemon,
    };
    if target.is_picker() {
        session.consent.allow_picker = true;
        // A portal DMA-BUF stream has no readback for `CopyToCpu`.
        session.video.exhaustion = Exhaustion::Stall;
    }
    session
}

/// The Vulkan importer for buffers of the GPU the plugin's backends
/// produce on, and the layouts it reads back. `None` (logged) without a
/// usable Vulkan driver: capture then asks for CPU frames.
#[cfg(target_os = "linux")]
async fn importer(plugin: &Plugin) -> Option<(Arc<VulkanImporter>, Readable)> {
    let producer = plugin
        .client()
        .capabilities(istmo_screen_capture::CapabilityQuery {
            kind: Some(SourceKind::Monitor),
            ..istmo_screen_capture::CapabilityQuery::default()
        })
        .await
        .ok()
        .and_then(|report| {
            report
                .backends
                .into_iter()
                .find_map(|caps| match caps.producer {
                    Some(istmo_screen_capture::GpuIdentity::Drm(dev)) => Some(dev),
                    _ => None,
                })
        });
    let built = tokio::task::spawn_blocking(move || VulkanImporter::new(producer)).await;
    match built {
        Ok(Ok(importer)) => {
            let readable = Readable {
                bgra: importer.supported_modifiers(PixelFormat::Bgra),
                rgba: importer.supported_modifiers(PixelFormat::Rgba),
            };
            debug!(?producer, ?readable, "capture: Vulkan importer ready");
            Some((Arc::new(importer), readable))
        }
        Ok(Err(e)) => {
            warn!(%e, "capture: no Vulkan importer, asking for CPU frames");
            None
        }
        Err(e) => {
            warn!(%e, "capture: Vulkan importer start panicked, asking for CPU frames");
            None
        }
    }
}

/// Logs the session's events (downgrades, exhaustion, renegotiation)
/// until it ends.
fn log_events(session: &SessionHandle) {
    let Ok(events) = session.events() else {
        return;
    };
    let spawned = std::thread::Builder::new()
        .name("ferricast-capture-events".into())
        .spawn(move || {
            while let Ok(StreamItem::Event(event)) = events.recv() {
                match event {
                    CaptureEvent::Started(_) => {}
                    CaptureEvent::Renegotiated(n) => info!(
                        backend = %n.backend,
                        format = ?n.format,
                        memory = ?n.memory,
                        modifier = n.modifier.map(|m| format!("0x{m:016x}")),
                        producer = ?n.producer,
                        "capture renegotiated"
                    ),
                    CaptureEvent::Downgraded(d) => warn!(?d, "capture downgraded"),
                    CaptureEvent::Exhausted(e) => {
                        debug!(?e, "capture pool exhausted (consumer holds every buffer)");
                    }
                    CaptureEvent::Failed(e) => warn!(%e, "capture failed"),
                    CaptureEvent::Stopped(reason) => {
                        info!(?reason, "capture stopped");
                        break;
                    }
                    other => debug!(?other, "capture event"),
                }
            }
        });
    if let Err(e) = spawned {
        warn!(%e, "capture: no thread for session events");
    }
}

fn oriented_size(negotiated: &NegotiatedStream) -> (usize, usize) {
    let size = negotiated.source.orientation.applied_size(negotiated.size);
    (size.width as usize, size.height as usize)
}

impl ScreenCapture for NativeCapture {
    async fn start(&mut self, source: CaptureSource, config: CaptureConfig) -> Result<()> {
        self.stop().await?;
        let plugin = Plugin::get()?;
        #[cfg(target_os = "linux")]
        let importer = importer(plugin).await;
        #[cfg(target_os = "linux")]
        let readable = importer.as_ref().map(|(_, readable)| readable);
        #[cfg(not(target_os = "linux"))]
        let readable: Option<&Readable> = None;
        let formats = Formats {
            consumer: &config.consumer_modifiers,
            readable,
        };
        let kind = match source {
            CaptureSource::FullScreen { .. } => SourceKind::Monitor,
            CaptureSource::Window { .. } => SourceKind::Window,
        };
        let listed = plugin
            .client()
            .sources(SourceFilter {
                kinds: vec![kind],
                backend: None,
            })
            .await
            .map_or_else(
                |e| {
                    debug!(%e, "capture: no source listing");
                    Vec::new()
                },
                |list| list.sources,
            );
        let target = Target::resolve(&source, &listed, pinned_backend());
        let opened = match &target {
            Ok(target) => self::open(plugin, &config, formats, target).await,
            Err(e) => Err(FerricastError::Capture(e.to_string())),
        };
        let session = match (opened, &target) {
            (Ok(session), _) => session,
            // Nothing captures this source directly here: the portal is
            // the path that is left, at the cost of its dialog.
            (Err(e), target) if !target.as_ref().is_ok_and(Target::is_picker) => {
                warn!(%e, "capture: direct capture failed, asking through the portal");
                self::open(plugin, &config, formats, &Target::portal(kind)).await?
            }
            (Err(e), _) => return Err(e),
        };
        let negotiated = session.negotiated();
        info!(
            backend = %negotiated.backend,
            source = %negotiated.source.name,
            width = negotiated.size.width,
            height = negotiated.size.height,
            format = ?negotiated.format,
            memory = ?negotiated.memory,
            copies = ?negotiated.copies,
            fps = ?negotiated.fps,
            "capture started"
        );
        log_events(&session);
        self.running = Some(Running {
            size: oriented_size(negotiated),
            #[cfg(target_os = "linux")]
            mapper: FrameMapper::new(
                importer.map(|(importer, _)| importer),
                config.consumer_orients,
            ),
            #[cfg(not(target_os = "linux"))]
            mapper: FrameMapper::new(),
            session,
            fps: config.fps.max(1),
            format: None,
        });
        Ok(())
    }

    async fn next_frame(&mut self) -> Result<CapturedFrame> {
        let running = self
            .running
            .as_mut()
            .ok_or_else(|| FerricastError::Capture("capture: not started".into()))?;
        loop {
            let Some(frame) = running.session.video().recv_async().await else {
                let reason = running.session.video().end_reason();
                return Err(FerricastError::Capture(format!(
                    "capture ended: {reason:?}"
                )));
            };
            let (meta, buffer) = frame.into_parts();
            let converted = match running.mapper.convert(&meta, buffer) {
                Ok(converted) => converted,
                Err(e) => {
                    warn!(%e, "capture: dropping a frame");
                    continue;
                }
            };
            let frame = if converted.pending.is_some() {
                // A readback and a full-frame turn: off the async workers.
                tokio::task::spawn_blocking(move || converted.finish())
                    .await
                    .map_err(|e| FerricastError::Capture(format!("capture readback: {e}")))?
            } else {
                converted.finish()
            };
            match frame {
                Ok(frame) => {
                    running.format = Some(frame.pixel_format());
                    running.size = (frame.width() as usize, frame.height() as usize);
                    return Ok(frame);
                }
                Err(e) => warn!(%e, "capture: dropping a frame"),
            }
        }
    }

    async fn stop(&mut self) -> Result<()> {
        if let Some(running) = self.running.take() {
            running
                .session
                .close()
                .await
                .map_err(capture_error("closing the capture session"))?;
        }
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running.is_some()
    }

    fn get_pixel_format(&self) -> PixelFormat {
        self.running()
            .ok()
            .and_then(|running| running.format)
            .unwrap_or(PixelFormat::Bgra)
    }

    fn get_screen_size(&self) -> (usize, usize) {
        self.running().map_or((0, 0), |running| running.size)
    }

    fn get_framerate(&self) -> u32 {
        self.running().map_or(0, |running| running.fps)
    }
}

/// Opens `target` with `config`.
async fn open(
    plugin: &Plugin,
    config: &CaptureConfig,
    formats: Formats<'_>,
    target: &Target,
) -> Result<SessionHandle> {
    plugin
        .client()
        .open_session(
            plugin.runtime(),
            session_config(config, formats, target),
            SinkPolicy::queue(SINK_DEPTH),
        )
        .await
        .map_err(capture_error("opening the capture session"))
}
