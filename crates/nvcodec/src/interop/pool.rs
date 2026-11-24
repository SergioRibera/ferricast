//! Reused pitched device buffers for consumers that cannot take a mapped
//! DMA-BUF directly (they need device memory rather than a CUDA array, or
//! must keep the picture after the producer reuses the buffer).

use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, PoisonError, Weak};

use crate::cuda::{Context, DeviceBuffer};
use crate::error::Result;

/// Idle buffers kept for reuse; more are freed when they come back.
const MAX_IDLE: usize = 4;

type FreeList = Mutex<Vec<DeviceBuffer>>;

/// A pool of pitched device buffers of one size (`width_bytes` × `rows`).
///
/// [`FramePool::take`] reuses an idle buffer or allocates one; the
/// [`PooledBuffer`] goes back to the pool when dropped, so the steady state
/// allocates nothing.
pub struct FramePool {
    context: Arc<Context>,
    width_bytes: usize,
    rows: usize,
    free: Arc<FreeList>,
}

impl FramePool {
    /// A pool of `width_bytes` × `rows` buffers on `context`.
    #[must_use]
    pub fn new(context: &Arc<Context>, width_bytes: usize, rows: usize) -> Self {
        Self {
            context: Arc::clone(context),
            width_bytes,
            rows,
            free: Arc::default(),
        }
    }

    /// Bytes per row of the buffers.
    #[must_use]
    pub fn width_bytes(&self) -> usize {
        self.width_bytes
    }

    /// Rows of the buffers.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// An idle buffer, or a new one.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`](crate::Error::Cuda) when allocating fails.
    pub fn take(&self) -> Result<PooledBuffer> {
        let reused = self
            .free
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        let buffer = match reused {
            Some(buffer) => buffer,
            None => DeviceBuffer::pitched(&self.context, self.width_bytes, self.rows)?,
        };
        Ok(PooledBuffer {
            buffer: Some(buffer),
            home: Arc::downgrade(&self.free),
        })
    }

    /// How many idle buffers wait for reuse.
    #[must_use]
    pub fn idle(&self) -> usize {
        self.free
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}

impl fmt::Debug for FramePool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FramePool")
            .field("width_bytes", &self.width_bytes)
            .field("rows", &self.rows)
            .field("idle", &self.idle())
            .finish_non_exhaustive()
    }
}

/// A buffer taken from a [`FramePool`], returned to it on drop (freed if
/// the pool is gone or has enough idle buffers).
///
/// Drop it only once the GPU work reading it has completed or is queued on
/// the stream the next copy into it will use: the pool hands it out again
/// right away.
pub struct PooledBuffer {
    buffer: Option<DeviceBuffer>,
    home: Weak<FreeList>,
}

impl Deref for PooledBuffer {
    type Target = DeviceBuffer;

    fn deref(&self) -> &DeviceBuffer {
        self.buffer
            .as_ref()
            .expect("invariant: the buffer is only taken out in drop")
    }
}

impl DerefMut for PooledBuffer {
    fn deref_mut(&mut self) -> &mut DeviceBuffer {
        self.buffer
            .as_mut()
            .expect("invariant: the buffer is only taken out in drop")
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        let (Some(buffer), Some(home)) = (self.buffer.take(), self.home.upgrade()) else {
            return;
        };
        let mut free = home.lock().unwrap_or_else(PoisonError::into_inner);
        if free.len() < MAX_IDLE {
            free.push(buffer);
        }
    }
}

impl fmt::Debug for PooledBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PooledBuffer").field(&self.buffer).finish()
    }
}
