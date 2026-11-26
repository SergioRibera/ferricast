//! [`NativeSourceEnumerator`]: monitors, windows and thumbnails for
//! in-process pickers, from the plugin's source listing.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use async_trait::async_trait;
use ferricast_core::{
    EnumerationCapability, Geometry, MonitorInfo, SourceChange, SourceEnumerator, SourceError,
    WindowInfo,
};
use istmo_screen_capture::{
    CapabilityQuery, CaptureError, FormatRequest, PixelLayout, Rect, ScreenCaptureClientExt as _,
    ScreenshotRequest, Source, SourceCapture, SourceChange as PluginChange, SourceFilter, SourceId,
    SourceKind, SourceSelector,
};
use tokio::sync::broadcast;
use tracing::{debug, warn};

use crate::plugin::Plugin;

/// Change notifications kept for slow subscribers (they re-list anyway).
const CHANGE_CAPACITY: usize = 16;

/// Lists what the plugin lists here, without a dialog: on Wayland the
/// outputs and the compositor's toplevel list, on X11 `RandR` and the
/// window tree. Windows only a picker can capture are listed too (their
/// capture goes through the portal).
#[derive(Debug)]
pub struct NativeSourceEnumerator {
    capabilities: Vec<EnumerationCapability>,
    changes: Arc<broadcast::Sender<SourceChange>>,
}

fn backend_error(error: impl std::fmt::Display) -> SourceError {
    SourceError::Backend(error.to_string())
}

fn capability(kind: SourceKind) -> Option<EnumerationCapability> {
    match kind {
        SourceKind::Monitor => Some(EnumerationCapability::Monitors),
        SourceKind::Window => Some(EnumerationCapability::Windows),
        SourceKind::Application => None,
    }
}

impl NativeSourceEnumerator {
    /// Probe what this session lists and start following its changes.
    ///
    /// # Errors
    /// [`SourceError::Unsupported`] when nothing here lists sources
    /// (GNOME / Mutter, where only the portal picks; a platform without a
    /// backend).
    pub fn try_new() -> Result<Self, SourceError> {
        let plugin = Plugin::get().map_err(backend_error)?;
        let client = plugin.client();
        let report = pollster::block_on(client.capabilities(CapabilityQuery::default()))
            .map_err(backend_error)?;
        let listed = pollster::block_on(client.sources(SourceFilter::default()))
            .map(|list| list.sources)
            .unwrap_or_default();
        let mut capabilities: Vec<EnumerationCapability> = report
            .backends
            .iter()
            .filter(|caps| caps.listing == istmo_screen_capture::SourceListing::Enumerable)
            .flat_map(|caps| caps.source_kinds.iter().copied())
            .chain(listed.iter().map(|source| source.kind))
            .filter_map(capability)
            .collect();
        capabilities.sort_by_key(|c| c.as_str());
        capabilities.dedup();
        if capabilities.is_empty() {
            return Err(SourceError::Unsupported(EnumerationCapability::Monitors));
        }
        let (changes, _) = broadcast::channel(CHANGE_CAPACITY);
        let changes = Arc::new(changes);
        Self::follow(Arc::downgrade(&changes));
        Ok(Self {
            capabilities,
            changes,
        })
    }

    /// Forwards the plugin's source watch as coarse changes, until the
    /// enumerator is gone (checked on each change) or the watch ends.
    fn follow(changes: Weak<broadcast::Sender<SourceChange>>) {
        let spawned = std::thread::Builder::new()
            .name("ferricast-source-watch".into())
            .spawn(move || {
                let Ok(plugin) = Plugin::get() else {
                    return;
                };
                let mut watch =
                    match pollster::block_on(plugin.client().source_watch(SourceFilter::default()))
                    {
                        Ok(watch) => watch,
                        Err(e) => {
                            debug!(%e, "source watch unavailable; listings are snapshots only");
                            return;
                        }
                    };
                loop {
                    let kinds: HashMap<SourceId, SourceKind> = watch
                        .sources()
                        .iter()
                        .map(|source| (source.id.clone(), source.kind))
                        .collect();
                    let change = match watch.recv() {
                        Some(Ok(change)) => change,
                        Some(Err(e)) => {
                            warn!(%e, "source watch ended");
                            return;
                        }
                        None => return,
                    };
                    let Some(changes) = changes.upgrade() else {
                        return;
                    };
                    let kind = match &change {
                        PluginChange::Listed(_) => continue,
                        PluginChange::Added(source) | PluginChange::Changed(source) => {
                            Some(source.kind)
                        }
                        PluginChange::Removed(id) => kinds.get(id).copied(),
                    };
                    let coarse = match kind.and_then(capability) {
                        Some(EnumerationCapability::Monitors) => SourceChange::Monitors,
                        Some(EnumerationCapability::Windows) => SourceChange::Windows,
                        None => continue,
                    };
                    // No subscriber is fine: listings stay correct.
                    let _ = changes.send(coarse);
                }
            });
        if let Err(e) = spawned {
            warn!(%e, "no thread for the source watch; listings are snapshots only");
        }
    }

    fn supports(&self, capability: EnumerationCapability) -> Result<(), SourceError> {
        if self.capabilities.contains(&capability) {
            Ok(())
        } else {
            Err(SourceError::Unsupported(capability))
        }
    }

    async fn listed(kinds: Vec<SourceKind>) -> Result<Vec<Source>, SourceError> {
        let plugin = Plugin::get().map_err(backend_error)?;
        plugin
            .client()
            .sources(SourceFilter {
                kinds,
                backend: None,
            })
            .await
            .map(|list| list.sources)
            .map_err(backend_error)
    }

    /// A PNG of `source`, fitted in `max_width` × `max_height`, upright.
    async fn thumbnail(
        source: &Source,
        max_width: u32,
        max_height: u32,
    ) -> Result<Vec<u8>, SourceError> {
        let plugin = Plugin::get().map_err(backend_error)?;
        let request = ScreenshotRequest::of(SourceSelector::Id(source.id.clone()))
            .formats(FormatRequest::cpu());
        let image = plugin
            .client()
            .screenshot_cpu(request, PixelLayout::default())
            .await
            .map_err(|e| match e {
                CaptureError::SourceNotFound(id) => SourceError::NotFound(id.value),
                other => backend_error(other),
            })?;
        encode_png(
            image.width,
            image.height,
            image.bytes,
            max_width,
            max_height,
        )
    }
}

/// Tight RGBA rows fitted in the box and encoded as PNG.
fn encode_png(
    width: u32,
    height: u32,
    rgba: Vec<u8>,
    max_width: u32,
    max_height: u32,
) -> Result<Vec<u8>, SourceError> {
    let image = image::RgbaImage::from_raw(width, height, rgba)
        .ok_or_else(|| SourceError::Backend(format!("screenshot shorter than {width}x{height}")))?;
    let (w, h) = fit_box(width, height, max_width, max_height);
    let thumb = image::imageops::thumbnail(&image, w, h);
    let mut out = Vec::with_capacity(8 * 1024);
    thumb
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .map_err(|e| SourceError::Backend(format!("png encode: {e}")))?;
    Ok(out)
}

/// Largest `(w, h)` that fits inside `(max_w, max_h)` while keeping
/// the aspect ratio of `(src_w, src_h)`. Both dimensions are
/// clamped to at least 1 so degenerate inputs don't produce a
/// 0-sized image buffer downstream.
fn fit_box(src_w: u32, src_h: u32, max_w: u32, max_h: u32) -> (u32, u32) {
    if src_w == 0 || src_h == 0 || max_w == 0 || max_h == 0 {
        return (1, 1);
    }
    if src_w <= max_w && src_h <= max_h {
        return (src_w, src_h);
    }
    // Integer form of `scale = min(max_w / src_w, max_h / src_h)`, rounded
    // to nearest; the scaled side is at most `max_*`, so it fits `u32`.
    let (sw, sh, mw, mh) = (
        u64::from(src_w),
        u64::from(src_h),
        u64::from(max_w),
        u64::from(max_h),
    );
    let (w, h) = if mw * sh <= mh * sw {
        (mw, (sh * mw + sw / 2) / sw)
    } else {
        ((sw * mh + sh / 2) / sh, mh)
    };
    let fit = |v: u64| u32::try_from(v).unwrap_or(u32::MAX).max(1);
    (fit(w), fit(h))
}

const fn geometry(rect: Rect) -> Geometry {
    Geometry {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
    }
}

/// The ferricast id of a monitor: its connector (`DP-3`), the name
/// ferricast has always used, else the plugin's id.
fn monitor_id(source: &Source) -> String {
    source
        .connector
        .clone()
        .unwrap_or_else(|| source.id.value.clone())
}

fn monitor_info(source: &Source) -> MonitorInfo {
    #[expect(
        clippy::cast_precision_loss,
        reason = "a scale in permille is far below 2^23"
    )]
    let scale = source
        .scale_permille
        .map_or(1.0, |permille| permille as f32 / 1000.0);
    MonitorInfo {
        id: monitor_id(source),
        name: source
            .description
            .clone()
            .unwrap_or_else(|| source.name.clone()),
        make: source.make.clone(),
        model: source.model.clone(),
        geometry: source.geometry.map_or(
            Geometry {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            },
            geometry,
        ),
        scale,
        refresh_mhz: source.refresh_mhz,
        primary: source.flags.primary,
    }
}

/// The monitor holding most of `window`, when both have a geometry.
fn dominant_monitor<'a>(window: &Source, monitors: &'a [Source]) -> Option<&'a Source> {
    let area = |rect: Rect| u64::from(rect.width) * u64::from(rect.height);
    let geometry = window.geometry?;
    monitors
        .iter()
        .filter_map(|monitor| {
            let overlap = geometry.intersect(monitor.geometry?)?;
            Some((area(overlap), monitor))
        })
        .filter(|(overlap, _)| *overlap > 0)
        .max_by_key(|(overlap, _)| *overlap)
        .map(|(_, monitor)| monitor)
}

fn window_info(source: &Source, monitors: &[Source]) -> WindowInfo {
    WindowInfo {
        id: source.id.value.clone(),
        title: source.name.clone(),
        app_id: source.app_id.clone(),
        pid: source.pid,
        geometry: source.geometry.map(geometry),
        on_monitor: dominant_monitor(source, monitors).map(monitor_id),
    }
}

#[async_trait]
impl SourceEnumerator for NativeSourceEnumerator {
    fn backend_name(&self) -> &'static str {
        "istmo"
    }

    fn capabilities(&self) -> Vec<EnumerationCapability> {
        self.capabilities.clone()
    }

    async fn list_monitors(&self) -> Result<Vec<MonitorInfo>, SourceError> {
        self.supports(EnumerationCapability::Monitors)?;
        let monitors = Self::listed(vec![SourceKind::Monitor]).await?;
        Ok(monitors.iter().map(monitor_info).collect())
    }

    async fn list_windows(&self) -> Result<Vec<WindowInfo>, SourceError> {
        self.supports(EnumerationCapability::Windows)?;
        let sources = Self::listed(vec![SourceKind::Monitor, SourceKind::Window]).await?;
        let (monitors, windows): (Vec<Source>, Vec<Source>) = sources
            .into_iter()
            .partition(|source| source.kind == SourceKind::Monitor);
        Ok(windows
            .iter()
            .map(|window| window_info(window, &monitors))
            .collect())
    }

    async fn monitor_thumbnail(
        &self,
        id: &str,
        max_width: u32,
        max_height: u32,
    ) -> Result<Vec<u8>, SourceError> {
        self.supports(EnumerationCapability::Monitors)?;
        let monitors = Self::listed(vec![SourceKind::Monitor]).await?;
        let monitor = monitors
            .iter()
            .find(|m| monitor_id(m) == id || m.id.value == id)
            .ok_or_else(|| SourceError::NotFound(id.to_owned()))?;
        Self::thumbnail(monitor, max_width, max_height).await
    }

    async fn window_thumbnail(
        &self,
        id: &str,
        max_width: u32,
        max_height: u32,
    ) -> Result<Vec<u8>, SourceError> {
        self.supports(EnumerationCapability::Windows)?;
        let windows = Self::listed(vec![SourceKind::Window]).await?;
        let window = windows
            .iter()
            .find(|w| w.id.value == id)
            .ok_or_else(|| SourceError::NotFound(id.to_owned()))?;
        match window.capture {
            SourceCapture::ById => Self::thumbnail(window, max_width, max_height).await,
            // Only a picker captures it: no screenshot without a dialog.
            SourceCapture::PickerOnly(_) => {
                Err(SourceError::Unsupported(EnumerationCapability::Windows))
            }
        }
    }

    fn subscribe(&self) -> broadcast::Receiver<SourceChange> {
        self.changes.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use istmo_screen_capture::BackendId;

    use super::*;

    fn source(kind: SourceKind, value: &str, rect: Option<Rect>) -> Source {
        let mut source = Source::new(
            SourceId::new(BackendId::LinuxWlrScreencopy, value),
            kind,
            value,
        );
        source.geometry = rect;
        source
    }

    #[test]
    fn fit_box_keeps_the_aspect_inside_the_box() {
        assert_eq!(fit_box(1920, 1080, 320, 320), (320, 180));
        assert_eq!(fit_box(1080, 1920, 320, 320), (180, 320));
        assert_eq!(fit_box(100, 50, 320, 320), (100, 50));
        assert_eq!(fit_box(0, 50, 320, 320), (1, 1));
        assert_eq!(fit_box(u32::MAX, 1, 2, 2), (2, 1));
    }

    #[test]
    fn a_window_is_on_the_monitor_holding_most_of_it() {
        let mut left = source(SourceKind::Monitor, "17", Some(Rect::new(0, 0, 1920, 1080)));
        left.connector = Some("DP-3".into());
        let mut right = source(
            SourceKind::Monitor,
            "18",
            Some(Rect::new(1920, 0, 1920, 1080)),
        );
        right.connector = Some("HDMI-A-1".into());
        let monitors = [left, right];
        let window = source(SourceKind::Window, "w", Some(Rect::new(1800, 10, 400, 300)));
        assert_eq!(
            window_info(&window, &monitors).on_monitor.as_deref(),
            Some("HDMI-A-1")
        );
        let unplaced = source(SourceKind::Window, "u", None);
        assert_eq!(window_info(&unplaced, &monitors).on_monitor, None);
    }

    #[test]
    fn monitors_keep_their_connector_as_id() {
        let mut monitor = source(SourceKind::Monitor, "17", Some(Rect::new(0, 0, 2560, 1440)));
        monitor.connector = Some("DP-3".into());
        monitor.description = Some("Dell U2720Q (DP-3)".into());
        monitor.scale_permille = Some(1500);
        let info = monitor_info(&monitor);
        assert_eq!(info.id, "DP-3");
        assert_eq!(info.name, "Dell U2720Q (DP-3)");
        assert!((info.scale - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn thumbnails_are_pngs_inside_the_box() {
        let png = encode_png(4, 2, vec![200u8; 4 * 2 * 4], 2, 2).unwrap();
        let decoded = image::load_from_memory(&png).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (2, 1));
    }
}
