//! Audio encoders and decoders that pick their backend.
//!
//! Audio backends do not fail on a stream they accepted the way video
//! hardware does, so these only select on `configure` (and select again
//! when the selection changes, or the open backend refuses a new
//! configuration and the selection allows falling back).

use ferricast_core::{
    Acceleration, AudioDecoderConfig, AudioEncoderConfig, AudioFrame, BackendId, BackendSelection,
    CodecError, CodecResult, CodecTarget, DecodedAudio, Operation, RuntimeFallback,
};
use tracing::warn;

use crate::registry::{Plan, Registry};
use crate::session::{AudioDecoderSession, AudioEncoderSession};

/// The registry of one audio encode job.
pub type AudioEncoderRegistry = Registry<AudioEncoderConfig, dyn AudioEncoderSession>;
/// The registry of one audio decode job.
pub type AudioDecoderRegistry = Registry<AudioDecoderConfig, dyn AudioDecoderSession>;

/// What the selecting audio codecs share: a session, where it sits in
/// its plan, and the selection it was opened for.
struct Active<S: ?Sized> {
    session: Box<S>,
    backend: BackendId,
    acceleration: Acceleration,
    position: usize,
    plan: Plan,
    selection: BackendSelection,
}

/// Opens or re-opens `active` for `config`: a new selection re-plans,
/// a refused reconfiguration moves on to the next candidate.
fn configure<C, S: ?Sized>(
    registry: &Registry<C, S>,
    active: &mut Option<Active<S>>,
    config: &C,
    selection: &BackendSelection,
    reconfigure: impl FnOnce(&mut S, &C) -> CodecResult<()>,
) -> CodecResult<()> {
    if let Some(open) = active.as_mut()
        && open.selection == *selection
    {
        let Err(error) = reconfigure(&mut open.session, config) else {
            return Ok(());
        };
        if matches!(open.plan.runtime_fallback(), RuntimeFallback::Disabled)
            || !open.plan.has_after(open.position)
        {
            return Err(error);
        }
        warn!(target_job = %registry.target(), backend = %open.backend, %error, "audio backend refused the new configuration; moving to the next backend");
        let opened = registry
            .open(&open.plan, open.position + 1, config)
            .map_err(|_| error)?;
        open.session = opened.session;
        open.backend = opened.backend;
        open.acceleration = opened.acceleration;
        open.position = opened.position;
        return Ok(());
    }
    let plan = registry.plan(selection)?;
    let opened = registry.open(&plan, 0, config)?;
    *active = Some(Active {
        session: opened.session,
        backend: opened.backend,
        acceleration: opened.acceleration,
        position: opened.position,
        plan,
        selection: selection.clone(),
    });
    Ok(())
}

/// A [`ferricast_core::AudioEncoder`] body that selects its backend
/// according to [`AudioEncoderConfig::backend`].
pub struct SelectingAudioEncoder {
    registry: AudioEncoderRegistry,
    active: Option<Active<dyn AudioEncoderSession>>,
}

impl std::fmt::Debug for SelectingAudioEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectingAudioEncoder")
            .field("target", &self.registry.target())
            .field("backend", &self.backend())
            .finish_non_exhaustive()
    }
}

impl SelectingAudioEncoder {
    #[must_use]
    /// An encoder over `registry`.
    pub fn new(registry: AudioEncoderRegistry) -> Self {
        Self {
            registry,
            active: None,
        }
    }

    #[must_use]
    /// The job this encoder does.
    pub fn target(&self) -> CodecTarget {
        self.registry.target()
    }

    #[must_use]
    /// The registry, for capability reports.
    pub fn registry(&self) -> &AudioEncoderRegistry {
        &self.registry
    }

    /// The backend encoding now, once configured.
    #[must_use]
    pub fn backend(&self) -> Option<(BackendId, Acceleration)> {
        self.active.as_ref().map(|a| (a.backend, a.acceleration))
    }

    /// # Errors
    /// The selection's errors or the backend's `configure` error.
    pub fn configure(&mut self, config: &AudioEncoderConfig) -> CodecResult<()> {
        configure(
            &self.registry,
            &mut self.active,
            config,
            &config.backend,
            AudioEncoderSession::configure,
        )
    }

    /// # Errors
    /// [`CodecError::NotConfigured`] before `configure`, or the backend's.
    pub fn encode(&mut self, frame: &AudioFrame) -> CodecResult<()> {
        let target = self.registry.target();
        self.active
            .as_mut()
            .ok_or(CodecError::NotConfigured {
                target,
                operation: Operation::Encode,
            })?
            .session
            .encode(frame)
    }

    /// Output produced so far; empty before `configure`.
    pub fn take_output(&mut self) -> Vec<AudioFrame> {
        self.active
            .as_mut()
            .map(|active| active.session.take_output())
            .unwrap_or_default()
    }

    /// The open backend's codec configuration; empty before `configure`.
    #[must_use]
    pub fn codec_config(&self) -> Vec<u8> {
        self.active
            .as_ref()
            .map(|active| active.session.codec_config())
            .unwrap_or_default()
    }

    /// # Errors
    /// The backend's; an encoder never configured flushes nothing.
    pub fn flush(self) -> CodecResult<Vec<AudioFrame>> {
        self.active
            .map_or_else(|| Ok(Vec::new()), |active| active.session.flush())
    }
}

/// A [`ferricast_core::AudioDecoder`] body that selects its backend
/// according to [`AudioDecoderConfig::backend`].
pub struct SelectingAudioDecoder {
    registry: AudioDecoderRegistry,
    active: Option<Active<dyn AudioDecoderSession>>,
}

impl std::fmt::Debug for SelectingAudioDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectingAudioDecoder")
            .field("target", &self.registry.target())
            .field("backend", &self.backend())
            .finish_non_exhaustive()
    }
}

impl SelectingAudioDecoder {
    #[must_use]
    /// A decoder over `registry`.
    pub fn new(registry: AudioDecoderRegistry) -> Self {
        Self {
            registry,
            active: None,
        }
    }

    #[must_use]
    /// The job this decoder does.
    pub fn target(&self) -> CodecTarget {
        self.registry.target()
    }

    #[must_use]
    /// The registry, for capability reports.
    pub fn registry(&self) -> &AudioDecoderRegistry {
        &self.registry
    }

    /// The backend decoding now, once configured.
    #[must_use]
    pub fn backend(&self) -> Option<(BackendId, Acceleration)> {
        self.active.as_ref().map(|a| (a.backend, a.acceleration))
    }

    /// # Errors
    /// The selection's errors or the backend's `configure` error.
    pub fn configure(&mut self, config: &AudioDecoderConfig) -> CodecResult<()> {
        configure(
            &self.registry,
            &mut self.active,
            config,
            &config.backend,
            AudioDecoderSession::configure,
        )
    }

    /// # Errors
    /// [`CodecError::NotConfigured`] before `configure`, or the backend's.
    pub fn decode(&mut self, frame: AudioFrame) -> CodecResult<Option<DecodedAudio>> {
        let target = self.registry.target();
        self.active
            .as_mut()
            .ok_or(CodecError::NotConfigured {
                target,
                operation: Operation::Decode,
            })?
            .session
            .decode(frame)
    }

    /// # Errors
    /// The backend's; a decoder never configured flushes nothing.
    pub fn flush(&mut self) -> CodecResult<Vec<DecodedAudio>> {
        self.active
            .as_mut()
            .map_or_else(|| Ok(Vec::new()), |active| active.session.flush())
    }
}
