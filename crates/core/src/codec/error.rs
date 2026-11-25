//! How codecs fail.
//!
//! Every variant names the job ([`CodecTarget`]) or the backend it is
//! about, so a message read by an end user says which encoder or
//! decoder broke and why, and code can match on the failure mode. All
//! fields are owned data so the error crosses the istmo wire intact.

use std::fmt;

use super::backend::{BackendId, CodecTarget};
use crate::frame::PixelFormat;

/// Result of a codec operation.
pub type CodecResult<T> = Result<T, CodecError>;

/// The step of a codec's life an error happened in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub enum Operation {
    /// Checking whether the backend can run on this machine.
    Probe,
    /// Opening a session with the first configuration.
    Open,
    /// Applying a new configuration to an open session.
    Configure,
    Encode,
    Decode,
    Flush,
    /// Producing the out-of-band stream headers (SPS / PPS, ASC).
    Headers,
    /// Changing the target bitrate of a running encoder.
    SetBitrate,
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Probe => "probe",
            Self::Open => "open",
            Self::Configure => "configure",
            Self::Encode => "encode",
            Self::Decode => "decode",
            Self::Flush => "flush",
            Self::Headers => "produce stream headers",
            Self::SetBitrate => "change the bitrate",
        })
    }
}

/// One backend automatic selection tried, and why it was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub struct BackendAttempt {
    pub backend: BackendId,
    /// The rejection, rendered: an attempt holds the message of the
    /// [`CodecError`] the backend returned.
    pub reason: String,
    /// How long the attempt took, probe and open together, in microseconds.
    pub elapsed_us: u64,
}

impl fmt::Display for BackendAttempt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.backend, self.reason)
    }
}

/// A codec failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
#[non_exhaustive]
pub enum CodecError {
    /// The build has no backend for the job at all.
    #[error(
        "this build has no backend that can {target}; rebuild with one of the cargo features: \
         {features}"
    )]
    NoBackendCompiled {
        target: CodecTarget,
        features: String,
    },

    /// A selection named a backend this build does not include.
    #[error(
        "the {backend} backend is not compiled into this build; rebuild with the `{feature}` \
         cargo feature to {target} with it"
    )]
    BackendNotCompiled {
        backend: BackendId,
        target: CodecTarget,
        feature: String,
    },

    /// A selection named a backend that never handles the job (e.g.
    /// `x264` for a decoder).
    #[error("the {backend} backend cannot {target}")]
    BackendCannotHandle {
        backend: BackendId,
        target: CodecTarget,
    },

    /// The backend is compiled in but cannot run here: missing driver,
    /// library or device, wrong platform.
    #[error("the {backend} backend is not available on this system: {reason}")]
    BackendUnavailable { backend: BackendId, reason: String },

    /// Every candidate of a selection was tried and rejected.
    #[error(
        "no backend could {target}; tried {}",
        attempts.iter().map(ToString::to_string).collect::<Vec<_>>().join("; ")
    )]
    NoBackendOpened {
        target: CodecTarget,
        attempts: Vec<BackendAttempt>,
    },

    /// The automatic selection policy filtered out every compiled backend.
    #[error("no compiled backend that can {target} matches the selection policy: {policy}")]
    NoBackendMatchesPolicy { target: CodecTarget, policy: String },

    /// A backend selection read from text (an environment variable, a
    /// setting) does not parse.
    #[error("{origin} is set to {value:?}, which is not a backend selection: {reason}")]
    InvalidSelection {
        origin: String,
        value: String,
        reason: String,
    },

    /// The configuration asks for something the backend cannot do.
    #[error("the {backend} backend does not support {what}")]
    UnsupportedConfig { backend: BackendId, what: String },

    /// The bitstream uses a feature the backend does not implement;
    /// another backend may decode it.
    #[error("the {backend} backend cannot decode this stream: {reason}")]
    UnsupportedStream { backend: BackendId, reason: String },

    /// A frame arrived in a pixel format the backend does not accept.
    #[error("the {backend} backend needs {expected} input, got {actual:?}")]
    UnsupportedInput {
        backend: BackendId,
        expected: String,
        actual: PixelFormat,
    },

    /// A codec was used before its first `configure`.
    #[error("cannot {operation} with the {target} codec before it is configured")]
    NotConfigured {
        target: CodecTarget,
        operation: Operation,
    },

    /// A session handle names no open session of the kind asked for.
    #[error("no open {kind} session has the handle {session}; it was closed or is another kind")]
    NoSuchSession { session: u64, kind: String },

    /// The codec service cannot open sessions (its host runtime is gone).
    #[error("the codec service cannot host sessions: {reason}")]
    SessionsUnavailable { reason: String },

    /// The backend failed while doing its job.
    #[error("the {backend} backend failed to {operation}: {reason}")]
    Backend {
        backend: BackendId,
        operation: Operation,
        reason: String,
    },

    /// The backend runs here but cannot take a session right now (a GPU at
    /// its session limit, a driver that failed to start this time, a GPU
    /// lost and not back yet). Selection moves on, and a later selection
    /// tries the backend again. (Last, so the wire encoding of the other
    /// variants is unchanged.)
    #[error("the {backend} backend cannot take a session right now: {reason}")]
    BackendBusy { backend: BackendId, reason: String },
}

impl CodecError {
    /// A [`Self::Backend`] failure.
    #[must_use]
    pub fn backend(backend: BackendId, operation: Operation, reason: impl fmt::Display) -> Self {
        Self::Backend {
            backend,
            operation,
            reason: reason.to_string(),
        }
    }

    /// Maps the crate-wide error a backend's internals produce into a
    /// codec error naming `backend` and `operation`; an error that is
    /// already a codec error is kept as is.
    pub fn from_backend(
        backend: BackendId,
        operation: Operation,
    ) -> impl FnOnce(crate::FerricastError) -> Self {
        move |error| match error {
            crate::FerricastError::Codec(error) => error,
            other => Self::backend(backend, operation, other),
        }
    }

    /// The backend the error is about, when it is about one.
    #[must_use]
    pub fn backend_id(&self) -> Option<BackendId> {
        match self {
            Self::BackendNotCompiled { backend, .. }
            | Self::BackendCannotHandle { backend, .. }
            | Self::BackendUnavailable { backend, .. }
            | Self::BackendBusy { backend, .. }
            | Self::UnsupportedConfig { backend, .. }
            | Self::UnsupportedStream { backend, .. }
            | Self::UnsupportedInput { backend, .. }
            | Self::Backend { backend, .. } => Some(*backend),
            Self::NoBackendCompiled { .. }
            | Self::NoBackendOpened { .. }
            | Self::NoBackendMatchesPolicy { .. }
            | Self::InvalidSelection { .. }
            | Self::NoSuchSession { .. }
            | Self::SessionsUnavailable { .. }
            | Self::NotConfigured { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Codec;

    #[test]
    fn messages_name_the_backend_the_job_and_the_cause() {
        let tried = CodecError::NoBackendOpened {
            target: CodecTarget::encode(Codec::H264),
            attempts: vec![
                BackendAttempt {
                    backend: BackendId::Nvenc,
                    reason: "libcuda.so.1 not found".into(),
                    elapsed_us: 120,
                },
                BackendAttempt {
                    backend: BackendId::Vaapi,
                    reason: "no render node".into(),
                    elapsed_us: 3_400,
                },
            ],
        };
        assert_eq!(
            tried.to_string(),
            "no backend could H.264 encode; tried nvenc: libcuda.so.1 not found; vaapi: no \
             render node"
        );
        let missing = CodecError::BackendNotCompiled {
            backend: BackendId::X264,
            target: CodecTarget::encode(Codec::H264),
            feature: "ferricast-encoder/x264".into(),
        };
        assert!(missing.to_string().contains("`ferricast-encoder/x264`"));
        assert_eq!(missing.backend_id(), Some(BackendId::X264));
    }

    #[test]
    fn crate_errors_from_backend_internals_keep_their_backend() {
        let mapped = CodecError::from_backend(BackendId::X264, Operation::Encode)(
            crate::FerricastError::Other("bad".into()),
        );
        assert_eq!(
            mapped,
            CodecError::backend(BackendId::X264, Operation::Encode, "bad")
        );
        let typed = CodecError::UnsupportedConfig {
            backend: BackendId::Vaapi,
            what: "4:4:4".into(),
        };
        let kept = CodecError::from_backend(BackendId::X264, Operation::Encode)(
            crate::FerricastError::Codec(typed.clone()),
        );
        assert_eq!(kept, typed);
    }
}
