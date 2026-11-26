//! The process's one istmo runtime with the screen-capture host.

use std::sync::{Arc, OnceLock};

use ferricast_core::{FerricastError, Result};
use istmo_core::{Envelope, Frame, Runtime, RuntimeInit};
use istmo_screen_capture::{
    CaptureError, CapturePolicy, DesktopCapture, ScreenCaptureClient, ScreenCaptureHost,
};
use tracing::warn;

/// The screen-capture plugin, hosted in this process.
///
/// One runtime serves every capture, audio session and source listing,
/// so the plugin's probe cache and worker pool are shared. It is private
/// to ferricast (not the process-global runtime an app may install with
/// `Runtime::init`): the plugin is only reached from Rust here, and the
/// policy keeps every method local.
pub(crate) struct Plugin {
    runtime: Arc<Runtime>,
    client: Arc<ScreenCaptureClient>,
}

impl Plugin {
    /// The shared instance, started on first use.
    ///
    /// # Errors
    /// The runtime refused the client (never expected: the host is
    /// registered just before), or no thread could drain its outbound
    /// channel.
    pub(crate) fn get() -> Result<&'static Self> {
        static PLUGIN: OnceLock<std::result::Result<Plugin, String>> = OnceLock::new();
        PLUGIN
            .get_or_init(Self::start)
            .as_ref()
            .map_err(|reason| FerricastError::Capture(reason.clone()))
    }

    fn start() -> std::result::Result<Self, String> {
        // `mock` is `init` without installing the runtime as the process
        // global, so nothing else in the process can reach this one.
        let RuntimeInit { runtime, outbound } = Runtime::mock();
        // Every owned handle dropped (each capture and audio session)
        // announces its release on the outbound channel, for a peer that
        // could hold it. There is no peer here: drain the channel, or the
        // releases fill it and then pile up behind it for the life of the
        // process. Dropping the receiver instead would turn each release
        // into a `ChannelClosed` error.
        std::thread::Builder::new()
            .name("ferricast-istmo-outbound".into())
            .spawn(move || drain(&outbound))
            .map_err(|e| format!("screen-capture outbound drain thread: {e}"))?;
        runtime.register_host(ScreenCaptureHost::new(DesktopCapture::with_policy(
            CapturePolicy::default(),
        )));
        let client = ScreenCaptureClient::from_runtime(&runtime)
            .map_err(|e| format!("screen-capture client: {e}"))?;
        Ok(Self {
            runtime,
            client: Arc::new(client),
        })
    }

    pub(crate) const fn runtime(&self) -> &Arc<Runtime> {
        &self.runtime
    }

    pub(crate) const fn client(&self) -> &Arc<ScreenCaptureClient> {
        &self.client
    }
}

/// Discards the runtime's outbound envelopes until the runtime is gone.
/// Only releases are expected; anything else is a frame for a peer that
/// does not exist, worth a line in the log.
fn drain(outbound: &flume::Receiver<Envelope>) {
    while let Ok(envelope) = outbound.recv() {
        if !matches!(envelope.frame, Frame::ReleaseNativeHandle { .. }) {
            warn!(
                ?envelope,
                "screen-capture: an outbound frame with no peer to take it"
            );
        }
    }
}

/// A plugin failure as a ferricast capture error, saying what was tried.
pub(crate) fn capture_error(what: &'static str) -> impl Fn(CaptureError) -> FerricastError {
    move |error| FerricastError::Capture(format!("{what}: {error}"))
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    /// Owned handles released over and over (sessions opened and closed
    /// in a long-lived daemon) leave nothing queued behind the outbound
    /// channel.
    #[test]
    fn released_handles_do_not_pile_up() {
        let runtime = Plugin::get().unwrap().runtime();
        for _ in 0..1000 {
            let id = runtime.alloc_native_handle_id();
            runtime.release_native_handle(id).unwrap();
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while runtime.outbound_spill().queued != 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        let spill = runtime.outbound_spill();
        assert_eq!(spill.queued, 0, "{spill:?}");
    }
}
