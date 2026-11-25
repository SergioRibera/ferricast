//! `PipeWire` explicit sync (`SPA_META_SyncTimeline`).
//!
//! A producer that supports it (Mutter, `KWin` with explicit sync) adds a
//! `spa_meta_sync_timeline` to every buffer plus two extra data blocks
//! of type `SPA_DATA_SyncObj`: the acquire and the release DRM timeline
//! syncobjs. The consumer waits for the acquire point before reading
//! the buffer and signals the release point once nothing reads it any
//! more; only then may the producer render into it again.
//!
//! This is the only synchronisation that holds on NVIDIA: its driver
//! attaches no implicit fence to the DMA-BUF, so polling the buffer
//! returns while the producer's GPU work is still pending. The wait
//! here runs on the CPU (`DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT`), bounded by
//! [`ACQUIRE_TIMEOUT`], so the frame leaves the capture complete
//! ([`FrameSync::Ready`](ferricast_core::FrameSync::Ready)) and no
//! consumer has to know about syncobjs.

use std::os::fd::BorrowedFd;
use std::time::Duration;

use ferricast_core::{DrmDevice, SyncWait, Syncobj, SyncobjDevice};
use pipewire as pw;
use tracing::{info, warn};

/// `SPA_META_SyncTimeline` (`spa/buffer/meta.h`, `PipeWire` 1.2). Kept
/// here rather than taken from the bindings, which follow whatever
/// headers the build machine has.
pub(super) const SPA_META_SYNC_TIMELINE: u32 = 9;
/// `SPA_DATA_SyncObj` (`spa/buffer/buffer.h`, `PipeWire` 1.2).
pub(super) const SPA_DATA_SYNC_OBJ: u32 = 5;

/// How long the producer's acquire point is waited for before the
/// frame is dropped. A compositor finishes a frame well within this.
const ACQUIRE_TIMEOUT: Duration = Duration::from_millis(100);

/// `struct spa_meta_sync_timeline`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct MetaSyncTimeline {
    flags: u32,
    padding: u32,
    acquire_point: u64,
    release_point: u64,
}

/// A buffer's timeline: where to wait and what to signal.
#[derive(Debug, Clone, Copy)]
struct Timeline<'b> {
    acquire: BorrowedFd<'b>,
    acquire_point: u64,
    release: BorrowedFd<'b>,
    release_point: u64,
}

impl Timeline<'_> {
    /// The timeline of `buffer`, or `None` when it carries none (the
    /// producer did not negotiate explicit sync).
    ///
    /// # Safety
    ///
    /// `buffer` is a live dequeued `spa_buffer` that `'b` does not
    /// outlive.
    unsafe fn of(buffer: *const pw::spa::sys::spa_buffer) -> Option<Self> {
        if buffer.is_null() {
            return None;
        }
        // SAFETY: the caller guarantees `buffer` is live; `metas` and
        // `datas` hold `n_metas` / `n_datas` entries.
        let (metas, datas) = unsafe {
            let buffer = &*buffer;
            if buffer.metas.is_null() || buffer.datas.is_null() {
                return None;
            }
            (
                std::slice::from_raw_parts(buffer.metas, usize::try_from(buffer.n_metas).ok()?),
                std::slice::from_raw_parts(buffer.datas, usize::try_from(buffer.n_datas).ok()?),
            )
        };
        let meta = metas.iter().find(|meta| {
            meta.type_ == SPA_META_SYNC_TIMELINE
                && !meta.data.is_null()
                && meta.size as usize >= size_of::<MetaSyncTimeline>()
        })?;
        // SAFETY: the meta is at least as large as the struct; it may be
        // unaligned in the shared memory.
        let meta = unsafe { meta.data.cast::<MetaSyncTimeline>().read_unaligned() };
        let mut syncobjs = datas
            .iter()
            .filter(|data| data.type_ == SPA_DATA_SYNC_OBJ)
            .filter_map(|data| i32::try_from(data.fd).ok().filter(|fd| *fd >= 0));
        let (acquire, release) = (syncobjs.next()?, syncobjs.next()?);
        // SAFETY: the descriptors belong to the dequeued buffer, which
        // outlives `'b` (caller's contract).
        let (acquire, release) = unsafe {
            (
                BorrowedFd::borrow_raw(acquire),
                BorrowedFd::borrow_raw(release),
            )
        };
        Some(Self {
            acquire,
            acquire_point: meta.acquire_point,
            release,
            release_point: meta.release_point,
        })
    }
}

/// The release point of one dequeued buffer, signalled when the buffer
/// goes back to the producer.
#[derive(Debug)]
pub(super) struct ReleasePoint {
    syncobj: Syncobj,
    point: u64,
}

impl ReleasePoint {
    /// Tells the producer the buffer is free again.
    pub(super) fn signal(self) {
        if let Err(error) = self.syncobj.signal(self.point) {
            warn!(%error, point = self.point, "signalling a PipeWire release point");
        }
    }
}

/// Why an explicitly synchronised buffer cannot be read.
#[derive(Debug, thiserror::Error)]
pub(super) enum AcquireError {
    #[error("no DRM node to wait on PipeWire's syncobjs: {0}")]
    Device(#[source] std::io::Error),
    #[error("importing a PipeWire syncobj: {0}")]
    Import(#[source] std::io::Error),
    #[error("waiting for the acquire point: {0}")]
    Wait(#[source] std::io::Error),
    #[error("the producer's acquire point did not signal within {ACQUIRE_TIMEOUT:?}")]
    TimedOut,
}

/// The worker's explicit-sync state: the DRM node syncobjs are waited
/// on, opened when the first synchronised buffer arrives.
#[derive(Debug)]
pub(super) struct ExplicitSync {
    node: Option<DrmDevice>,
    device: Option<SyncobjDevice>,
    announced: bool,
}

impl ExplicitSync {
    /// Waits on syncobjs through `node` (the importer's GPU) when given.
    pub(super) fn new(node: Option<DrmDevice>) -> Self {
        Self {
            node,
            device: None,
            announced: false,
        }
    }

    fn device(&mut self) -> Result<&SyncobjDevice, AcquireError> {
        if self.device.is_none() {
            let device =
                SyncobjDevice::open_render_node(self.node).map_err(AcquireError::Device)?;
            self.device = Some(device);
        }
        self.device
            .as_ref()
            .ok_or_else(|| AcquireError::Device(std::io::ErrorKind::NotFound.into()))
    }

    /// The release point of `buffer`, or `None` without explicit sync.
    ///
    /// # Safety
    ///
    /// As [`Timeline::of`].
    pub(super) unsafe fn release_point(
        &mut self,
        buffer: *const pw::spa::sys::spa_buffer,
    ) -> Result<Option<ReleasePoint>, AcquireError> {
        // SAFETY: forwarded contract.
        let Some(timeline) = (unsafe { Timeline::of(buffer) }) else {
            return Ok(None);
        };
        let syncobj = self
            .device()?
            .import(timeline.release)
            .map_err(AcquireError::Import)?;
        Ok(Some(ReleasePoint {
            syncobj,
            point: timeline.release_point,
        }))
    }

    /// Waits until the producer finished writing `buffer`. `Ok(false)`
    /// when the buffer carries no timeline (implicit sync).
    ///
    /// # Safety
    ///
    /// As [`Timeline::of`].
    pub(super) unsafe fn acquire(
        &mut self,
        buffer: *const pw::spa::sys::spa_buffer,
    ) -> Result<bool, AcquireError> {
        // SAFETY: forwarded contract.
        let Some(timeline) = (unsafe { Timeline::of(buffer) }) else {
            return Ok(false);
        };
        if !std::mem::replace(&mut self.announced, true) {
            info!("PipeWire producer uses explicit sync (SyncTimeline): waiting on acquire points");
        }
        let syncobj = self
            .device()?
            .import(timeline.acquire)
            .map_err(AcquireError::Import)?;
        match syncobj
            .wait(timeline.acquire_point, ACQUIRE_TIMEOUT)
            .map_err(AcquireError::Wait)?
        {
            SyncWait::Signalled => Ok(true),
            SyncWait::TimedOut => Err(AcquireError::TimedOut),
        }
    }
}
