//! A video encoder that picks its backend.

use std::sync::Arc;

use ferricast_core::{
    Acceleration, BackendId, CapturedFrame, CodecError, CodecResult, CodecTarget, EncodedFrame,
    EncoderConfig, Operation, RuntimeFallback,
};
use tracing::warn;

use crate::cache::SelectionCache;
use crate::events::{EventReceiver, SessionEvent, SessionEvents};
use crate::registry::{Plan, Registry};
use crate::session::VideoEncoderSession;

/// The registry of one video encode job.
pub type VideoEncoderRegistry = Registry<EncoderConfig, dyn VideoEncoderSession>;

struct Active {
    session: Box<dyn VideoEncoderSession>,
    backend: BackendId,
    acceleration: Acceleration,
    position: usize,
    plan: Plan,
    config: EncoderConfig,
}

/// A [`ferricast_core::VideoEncoder`] body that selects its backend from
/// a [`VideoEncoderRegistry`] according to [`EncoderConfig::backend`].
///
/// The backend is chosen on the first `configure` and again when a
/// later `configure` asks for a different selection. When the open
/// backend refuses a new configuration (a resolution its hardware
/// session cannot take), the next backend of the selection opens with
/// it instead, unless the selection disables runtime fallback.
pub struct SelectingVideoEncoder {
    registry: VideoEncoderRegistry,
    active: Option<Active>,
    events: SessionEvents,
}

impl std::fmt::Debug for SelectingVideoEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectingVideoEncoder")
            .field("target", &self.registry.target())
            .field("backend", &self.backend())
            .finish_non_exhaustive()
    }
}

impl SelectingVideoEncoder {
    #[must_use]
    /// An encoder over `registry`.
    pub fn new(registry: VideoEncoderRegistry) -> Self {
        Self {
            registry,
            active: None,
            events: SessionEvents::default(),
        }
    }

    /// Remembers automatic selection's winner in `cache` instead of the
    /// process-wide one (see [`crate::cache`]).
    #[must_use]
    pub fn with_cache(mut self, cache: Arc<dyn SelectionCache>) -> Self {
        self.registry.set_cache(cache);
        self
    }

    /// The job this encoder does.
    #[must_use]
    pub fn target(&self) -> CodecTarget {
        self.registry.target()
    }

    /// The registry, for capability reports.
    #[must_use]
    pub fn registry(&self) -> &VideoEncoderRegistry {
        &self.registry
    }

    /// The backend encoding now, once configured.
    #[must_use]
    pub fn backend(&self) -> Option<(BackendId, Acceleration)> {
        self.active
            .as_ref()
            .map(|active| (active.backend, active.acceleration))
    }

    /// The session's event hub.
    #[must_use]
    pub fn events(&self) -> &SessionEvents {
        &self.events
    }

    /// Subscribes to the session's events; see
    /// [`SessionEvents::subscribe_at`].
    #[must_use]
    pub fn subscribe(&self, capacity: usize) -> EventReceiver {
        self.events.subscribe_at(capacity, self.backend())
    }

    /// Selects (or re-selects) the backend, or reconfigures the open one.
    ///
    /// # Errors
    /// The selection's errors (see [`Registry::plan`] /
    /// [`Registry::open`]); the open backend's `configure` error when no
    /// fallback applies.
    pub fn configure(&mut self, config: &EncoderConfig) -> CodecResult<()> {
        let result = self.reconfigure(config);
        self.events.report(result)
    }

    fn reconfigure(&mut self, config: &EncoderConfig) -> CodecResult<()> {
        let Some(active) = &mut self.active else {
            return self.select(config);
        };
        if active.config.backend != config.backend {
            return self.select(config);
        }
        let Err(error) = active.session.configure(config) else {
            active.config = config.clone();
            return Ok(());
        };
        if matches!(active.plan.runtime_fallback(), RuntimeFallback::Disabled)
            || !active.plan.has_after(active.position)
        {
            return Err(error);
        }
        warn!(
            target_job = %self.registry.target(),
            backend = %active.backend,
            %error,
            "encoder backend refused the new configuration; moving to the next backend"
        );
        let opened = self
            .registry
            .open(&active.plan, active.position + 1, config)
            .map_err(|_| error.clone())?;
        self.events.emit(&SessionEvent::Switched {
            from: active.backend,
            to: opened.backend,
            acceleration: opened.acceleration,
            reason: error,
        });
        active.session = opened.session;
        active.backend = opened.backend;
        active.acceleration = opened.acceleration;
        active.position = opened.position;
        active.config = config.clone();
        Ok(())
    }

    fn select(&mut self, config: &EncoderConfig) -> CodecResult<()> {
        let plan = self.registry.plan(&config.backend)?;
        let opened = self.registry.open(&plan, 0, config)?;
        self.events.emit(&SessionEvent::Opened {
            backend: opened.backend,
            acceleration: opened.acceleration,
        });
        self.active = Some(Active {
            session: opened.session,
            backend: opened.backend,
            acceleration: opened.acceleration,
            position: opened.position,
            plan,
            config: config.clone(),
        });
        Ok(())
    }

    fn active(&mut self, operation: Operation) -> CodecResult<&mut Active> {
        let target = self.registry.target();
        self.active
            .as_mut()
            .ok_or(CodecError::NotConfigured { target, operation })
    }

    /// # Errors
    /// [`CodecError::NotConfigured`] before `configure`, or the backend's.
    pub fn encode(&mut self, frame: CapturedFrame) -> CodecResult<EncodedFrame> {
        let result = self
            .active(Operation::Encode)
            .and_then(|active| active.session.encode(frame));
        self.events.report(result)
    }

    /// # Errors
    /// The backend's; an encoder never configured flushes nothing.
    pub fn flush(mut self) -> CodecResult<Vec<EncodedFrame>> {
        let result = self
            .active
            .take()
            .map_or_else(|| Ok(Vec::new()), |active| active.session.flush());
        self.events.report(result)
    }

    /// # Errors
    /// [`CodecError::NotConfigured`] before `configure`, or the backend's.
    pub fn headers(&mut self) -> CodecResult<Vec<u8>> {
        let result = self
            .active(Operation::Headers)
            .and_then(|active| active.session.headers());
        self.events.report(result)
    }

    /// Asks for a keyframe next; a no-op before `configure`.
    pub fn request_keyframe(&mut self) {
        if let Some(active) = &mut self.active {
            active.session.request_keyframe();
            self.events.emit(&SessionEvent::KeyframeRequested);
        }
    }

    /// # Errors
    /// [`CodecError::NotConfigured`] before `configure`, or the backend's.
    pub fn set_bitrate_kbps(&mut self, kbps: u32) -> CodecResult<()> {
        let result = self
            .active(Operation::SetBitrate)
            .and_then(|active| active.session.set_bitrate_kbps(kbps));
        if result.is_ok() {
            self.events.emit(&SessionEvent::BitrateChanged { kbps });
        }
        self.events.report(result)
    }
}

impl Drop for SelectingVideoEncoder {
    fn drop(&mut self) {
        self.events.emit(&SessionEvent::Closed);
    }
}
