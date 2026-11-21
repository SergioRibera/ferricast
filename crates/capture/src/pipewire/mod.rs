//! PipeWire screen capture backend.
//!
//! Talks to `xdg-desktop-portal.ScreenCast` (via `ashpd`) to obtain a
//! PipeWire socket and node id, then runs a dedicated PipeWire main loop
//! on its own OS thread to pull video buffers and ferry them to the
//! async [`ferricast_core::ScreenCapture`] trait.
//!
//! Module layout:
//! * [`portal`]    — D-Bus portal handshake.
//! * [`stream`]    — PipeWire main-loop worker.
//! * [`format`]    — SPA pod construction & format conversion.
//! * [`lease`]     — keeps DMA-BUF buffers dequeued while consumers read them.

mod format;
mod lease;
mod portal;
mod stream;

use std::sync::{Arc, RwLock};

use ferricast_core::{
    CaptureConfig, CaptureSource, CapturedFrame, FerricastError, PixelFormat, Result, ScreenCapture,
};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{info, warn};

use self::portal::PortalSession;
use self::stream::{SharedFormat, WorkerHandle};

/// Screen capture backend using PipeWire via `xdg-desktop-portal`.
///
/// ```ignore
/// let mut cap = PipeWireCapture::new();
/// cap.start(CaptureSource::FullScreen { monitor: None }, CaptureConfig::default()).await?;
/// while let Ok(frame) = cap.next_frame().await {
///     // feed the encoder...
/// }
/// cap.stop().await?;
/// ```
pub struct PipeWireCapture {
    /// `None` until `start()` succeeds, then `Some` until `stop()` is called.
    worker: Option<WorkerHandle>,
    /// Portal session backing the running capture. `Some` exactly
    /// when `worker` is; closed in `stop()`.
    portal: Option<ActivePortal>,
    /// Filled by the PW thread once the format negotiation completes.
    /// Read by the synchronous getters (`get_screen_size`, etc.).
    shared_format: SharedFormat,
}

/// The portal session plus a task watching its `Closed` signal.
struct ActivePortal {
    session: PortalSession,
    /// Flips to `true` when the portal reports the session closed
    /// ("Stop sharing", compositor revoked the cast).
    closed: watch::Receiver<bool>,
    /// `None` when `start()` ran outside a Tokio runtime, in which
    /// case only the PipeWire stream state reports the end of the cast.
    watcher: Option<JoinHandle<()>>,
}

impl ActivePortal {
    fn new(session: PortalSession) -> Self {
        let (closed_tx, closed) = watch::channel(false);
        let watcher = match tokio::runtime::Handle::try_current() {
            Ok(rt) => {
                let session = session.clone();
                Some(rt.spawn(async move {
                    match session.closed().await {
                        Ok(()) => {
                            info!("portal session closed by the compositor or user");
                            closed_tx.send_replace(true);
                        }
                        Err(e) => warn!(error = %e, "can't watch portal session Closed signal"),
                    }
                }))
            }
            Err(_) => {
                warn!("no Tokio runtime; portal Closed signal not watched");
                None
            }
        };
        Self {
            session,
            closed,
            watcher,
        }
    }

    async fn close(self) {
        if let Some(watcher) = self.watcher {
            watcher.abort();
        }
        self.session.close().await;
    }
}

impl PipeWireCapture {
    pub fn new() -> Self {
        Self {
            worker: None,
            portal: None,
            shared_format: Arc::new(RwLock::new(None)),
        }
    }

    /// Take a snapshot of the negotiated format, if any. Cheap; takes a
    /// read lock on the small `RwLock` populated by the worker.
    fn snapshot(&self) -> Option<format::NegotiatedFormat> {
        self.shared_format.read().ok().and_then(|g| *g)
    }
}

impl Drop for PipeWireCapture {
    /// Best-effort close for captures dropped without `stop()`: the
    /// portal session would otherwise keep the cast alive.
    fn drop(&mut self) {
        let Some(portal) = self.portal.take() else {
            return;
        };
        match tokio::runtime::Handle::try_current() {
            Ok(rt) => {
                rt.spawn(portal.close());
            }
            Err(_) => {
                warn!("PipeWireCapture dropped outside a Tokio runtime; portal session left open")
            }
        }
    }
}

impl Default for PipeWireCapture {
    fn default() -> Self {
        Self::new()
    }
}

impl ScreenCapture for PipeWireCapture {
    async fn start(&mut self, source: CaptureSource, config: CaptureConfig) -> Result<()> {
        if self.worker.is_some() {
            return Err(FerricastError::Capture(
                "PipeWire capture already running".into(),
            ));
        }

        info!(?source, ?config, "starting PipeWire capture");

        // Reset any leftover state from a previous run before kicking off
        // the new one.
        if let Ok(mut g) = self.shared_format.write() {
            *g = None;
        }

        let (portal_stream, session) = portal::open_session(&source, &config).await?;
        let worker = match stream::spawn(portal_stream, config, Arc::clone(&self.shared_format)) {
            Ok(worker) => worker,
            Err(e) => {
                session.close().await;
                return Err(e);
            }
        };
        self.worker = Some(worker);
        self.portal = Some(ActivePortal::new(session));

        info!("PipeWire capture started");
        Ok(())
    }

    async fn next_frame(&mut self) -> Result<CapturedFrame> {
        let worker = self
            .worker
            .as_mut()
            .ok_or_else(|| FerricastError::Capture("capture not started".into()))?;
        let portal_closed = async {
            match self.portal.as_mut() {
                // `Err` = watcher gone without seeing `Closed`; that
                // must not end the capture, so wait forever instead.
                Some(p) => {
                    if p.closed.wait_for(|closed| *closed).await.is_err() {
                        std::future::pending::<()>().await;
                    }
                }
                None => std::future::pending::<()>().await,
            }
        };

        tokio::select! {
            // Errors take precedence so the user sees the real cause.
            biased;
            err = worker.errors.recv() => {
                let msg = err.unwrap_or_else(|| "PipeWire worker exited".into());
                Err(FerricastError::Capture(format!("PipeWire: {msg}")))
            }
            () = portal_closed => {
                Err(FerricastError::Capture("PipeWire: screencast portal session closed".into()))
            }
            frame = worker.frames.recv() => {
                let mut latest = frame
                    .ok_or_else(|| FerricastError::Capture("PipeWire stream ended".into()))?;
                // Drain anything else queued in the channel: when the PW
                // worker temporarily falls behind the segmenter (or vice
                // versa) the channel can hold several frames, and after a
                // capture stall the burst that PW catches up with would
                // otherwise be encoded back-to-back with their original
                // wall-clock timestamps — the player perceives that as
                // "stutter then jump". We always keep only the freshest
                // frame so the encoder produces one fresh frame after a
                // stall and then continues at the natural pace, not a
                // catch-up burst.
                while let Ok(newer) = worker.frames.try_recv() {
                    latest = newer;
                }
                Ok(latest)
            }
        }
    }

    async fn stop(&mut self) -> Result<()> {
        info!("stopping PipeWire capture");
        if let Some(worker) = self.worker.take() {
            // Return only after the worker thread has actually joined.
            worker.shutdown().await;
        }
        // Close the portal session last: the compositor tears the
        // cast down (and clears its sharing indicator) only on Close.
        if let Some(portal) = self.portal.take() {
            portal.close().await;
        }
        if let Ok(mut g) = self.shared_format.write() {
            *g = None;
        }
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.worker.is_some()
    }

    fn get_pixel_format(&self) -> PixelFormat {
        self.snapshot()
            .map(|n| n.pixel_format)
            // Reasonable default before negotiation finishes; matches what
            // most Wayland compositors produce.
            .unwrap_or(PixelFormat::Bgra)
    }

    fn get_screen_size(&self) -> (usize, usize) {
        self.snapshot()
            .map(|n| (n.width as usize, n.height as usize))
            .unwrap_or((0, 0))
    }

    /// Returns the framerate the compositor agreed on. The negotiated
    /// fraction is a `num/denom` pair (e.g. 60/1, 24000/1001 for
    /// NTSC) — we round to the nearest whole fps because every
    /// downstream encoder we wrap takes an integer.
    fn get_framerate(&self) -> u32 {
        let Some(n) = self.snapshot() else { return 0 };
        let f = n.framerate;
        if f.denom == 0 {
            return 0;
        }
        // Round-to-nearest so 24000/1001 surfaces as 24, not 23.
        ((f.num as u64 + (f.denom as u64 / 2)) / f.denom as u64) as u32
    }
}
