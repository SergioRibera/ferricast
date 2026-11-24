//! NVENC sessions.
//!
//! [`Session`] owns one encoder session opened on a CUDA context through the
//! function list of one API version; it is what the encoder is built on.

use std::ffi::c_void;
use std::fmt;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::cuda::Context;
use crate::error::Result;
use crate::version::{Api, DeviceType, NvencApi};

/// An NVENC encoder session on a CUDA context, destroyed on drop.
///
/// Holds an `Arc` of its context, so the context outlives the session, and a
/// copy of the function list it was opened with, so every call on it uses the
/// same API version.
pub struct Session<A: NvencApi> {
    encoder: NonNull<c_void>,
    api: Api<A>,
    context: Arc<Context>,
}

// SAFETY: an NVENC session has no thread affinity: the driver accepts calls on
// it from any thread, one at a time. Its CUDA context is pushed around every
// call that needs it, and `Session` is not `Sync`, so calls never overlap.
unsafe impl<A: NvencApi> Send for Session<A> {}

impl<A: NvencApi> Session<A> {
    /// Opens a session on `context` (`nvEncOpenEncodeSessionEx`).
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`](crate::Error::Cuda) if the context cannot be made current,
    /// [`Error::Nvenc`](crate::Error::Nvenc) if the driver refuses the session (no NVENC on this
    /// GPU, session limit reached, version mismatch).
    pub fn open(api: Api<A>, context: &Arc<Context>) -> Result<Self> {
        let device = context.as_non_null();
        let _current = context.guard()?;
        // SAFETY: `device` is the context's handle and the session holds an
        // `Arc` of the context, so it outlives the encoder; `api` holds the
        // list for `A`.
        let encoder = unsafe { A::open_encode_session(api.functions(), device, DeviceType::Cuda)? };
        Ok(Self {
            encoder,
            api,
            context: Arc::clone(context),
        })
    }

    /// The encoder handle every NVENC call takes.
    #[must_use]
    pub fn as_ptr(&self) -> *mut c_void {
        self.encoder.as_ptr()
    }

    /// The function list the session was opened with.
    #[must_use]
    pub fn api(&self) -> &Api<A> {
        &self.api
    }

    /// The CUDA context the session encodes on.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// The driver's explanation of the session's last failure
    /// (`nvEncGetLastErrorString`), to attach to an [`Error::Nvenc`](crate::Error::Nvenc).
    #[must_use]
    pub fn last_error(&self) -> Option<String> {
        // SAFETY: the session is live and opened with this list.
        unsafe { A::last_error(self.api.functions(), self.encoder) }
    }
}

impl<A: NvencApi> Drop for Session<A> {
    fn drop(&mut self) {
        let destroyed = self.context.guard().and_then(|_current| {
            // SAFETY: the session is live, opened with this list, and not
            // used again.
            unsafe { A::destroy_encoder(self.api.functions(), self.encoder) }
        });
        if let Err(error) = destroyed {
            tracing::warn!(%error, api = %A::VERSION, "destroying an NVENC session");
        }
    }
}

impl<A: NvencApi> fmt::Debug for Session<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session")
            .field("encoder", &self.encoder)
            .field("api", &A::VERSION)
            .field("context", &self.context)
            .finish()
    }
}
