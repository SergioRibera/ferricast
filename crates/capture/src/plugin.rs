//! The process's one istmo runtime with the screen-capture host.

use std::sync::{Arc, OnceLock};

use ferricast_core::{FerricastError, Result};
use istmo_core::{Runtime, RuntimeInit};
use istmo_screen_capture::{
    CaptureError, CapturePolicy, DesktopCapture, ScreenCaptureClient, ScreenCaptureHost,
};

/// The screen-capture plugin, hosted in this process.
///
/// One runtime serves every capture, audio session and source listing,
/// so the plugin's probe cache and worker pool are shared. It is private
/// to ferricast (not the process-global runtime an app may install with
/// `Runtime::init`): the plugin is only reached from Rust here, and the
/// policy keeps every method local.
pub(crate) struct Plugin {
    init: RuntimeInit,
    client: Arc<ScreenCaptureClient>,
}

impl Plugin {
    /// The shared instance, started on first use.
    ///
    /// # Errors
    /// The runtime refused the client (never expected: the host is
    /// registered just before).
    pub(crate) fn get() -> Result<&'static Self> {
        static PLUGIN: OnceLock<std::result::Result<Plugin, String>> = OnceLock::new();
        PLUGIN
            .get_or_init(Self::start)
            .as_ref()
            .map_err(|reason| FerricastError::Capture(reason.clone()))
    }

    fn start() -> std::result::Result<Self, String> {
        // `mock` builds a runtime without installing it as the process
        // global; its outbound channel stays unused (no remote peers),
        // but is kept alive with the runtime.
        let init = Runtime::mock();
        init.runtime
            .register_host(ScreenCaptureHost::new(DesktopCapture::with_policy(
                CapturePolicy::default(),
            )));
        let client = ScreenCaptureClient::from_runtime(&init.runtime)
            .map_err(|e| format!("screen-capture client: {e}"))?;
        Ok(Self {
            init,
            client: Arc::new(client),
        })
    }

    pub(crate) const fn runtime(&self) -> &Arc<Runtime> {
        &self.init.runtime
    }

    pub(crate) const fn client(&self) -> &Arc<ScreenCaptureClient> {
        &self.client
    }
}

/// A plugin failure as a ferricast capture error, saying what was tried.
pub(crate) fn capture_error(what: &'static str) -> impl Fn(CaptureError) -> FerricastError {
    move |error| FerricastError::Capture(format!("{what}: {error}"))
}
