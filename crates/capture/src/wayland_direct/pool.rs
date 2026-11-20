//! Pool of compositor-writable DMA-BUFs with downstream leases.
//!
//! Every screencopy round needs a buffer the compositor renders into.
//! Allocating a gbm bo per frame is slow, so buffers are reused — but a
//! `GpuFrame` handed downstream shares the DMA-BUF with the encoder
//! (VA-API / NVENC import, Vulkan readback). Rendering the next frame
//! into a buffer an encoder still reads tears its picture. So a buffer
//! backing a live `GpuFrame` carries a lease, and leased buffers are
//! never handed to the next screencopy.
//!
//! The pool lives on the capture worker thread; only the lease flag
//! crosses threads. A lease's guard drops wherever the last clone of
//! the frame dies and just clears an atomic flag, which the worker
//! reads on its next [`BufferPool::pick`]. Buffers are created and
//! destroyed on the worker thread only.
//!
//! Generic over the buffer type so the bookkeeping is unit-testable
//! without a GPU or a compositor.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ferricast_core::BufferLease;

/// Most buffers alive at once, retired ones included. Downstream can
/// hold up to four frames at a time — two queued in the frame
/// channel, one being encoded and the HLS segmenter's copy of the
/// last frame — and the compositor needs one more to render into.
pub(super) const MAX_POOL_BUFFERS: usize = 5;

/// What the pool's buffers must match: a change invalidates them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PoolKey {
    pub(super) width: u32,
    pub(super) height: u32,
    /// DRM fourcc of the compositor's dmabuf offer.
    pub(super) fourcc: u32,
}

/// Position of a slot, valid until the pool is next mutated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SlotIdx(usize);

/// Outcome of [`BufferPool::pick`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Pick {
    /// An unleased buffer to render into.
    Free(SlotIdx),
    /// Every buffer is leased, but one more fits under the cap:
    /// allocate it and [`BufferPool::insert`] it.
    Grow,
    /// Every buffer is leased and the cap is reached. The caller
    /// captures this frame some other way (CPU copy) rather than
    /// dropping it.
    Exhausted,
}

/// `true` while a frame backed by the buffer is alive downstream.
type LeaseFlag = Arc<AtomicBool>;

/// The guard inside a [`BufferLease`]: clears the flag when the last
/// clone of the frame is dropped.
struct ReleaseOnDrop(LeaseFlag);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        // Release pairs with the worker's Acquire load: downstream
        // reads of the buffer happen-before the compositor is allowed
        // to render into it again.
        self.0.store(false, Ordering::Release);
    }
}

#[derive(Debug)]
struct Slot<B> {
    buffer: B,
    leased: LeaseFlag,
}

impl<B> Slot<B> {
    fn new(buffer: B) -> Self {
        Self {
            buffer,
            leased: Arc::new(AtomicBool::new(false)),
        }
    }

    fn is_leased(&self) -> bool {
        self.leased.load(Ordering::Acquire)
    }
}

/// Reusable buffers for one [`PoolKey`], plus buffers retired by a
/// key change that are still leased. Retired buffers stay alive
/// (downstream still reads them) until their lease ends, then drop on
/// the next [`BufferPool::pick`] / [`BufferPool::reconfigure`].
#[derive(Debug)]
pub(super) struct BufferPool<B> {
    key: Option<PoolKey>,
    slots: Vec<Slot<B>>,
    retired: Vec<Slot<B>>,
    cap: usize,
}

impl<B> Default for BufferPool<B> {
    fn default() -> Self {
        Self::with_cap(MAX_POOL_BUFFERS)
    }
}

impl<B> BufferPool<B> {
    #[must_use]
    pub(super) fn with_cap(cap: usize) -> Self {
        Self {
            key: None,
            slots: Vec::new(),
            retired: Vec::new(),
            cap,
        }
    }

    /// Make the pool serve buffers matching `key`. On a change, free
    /// buffers drop and leased ones are retired. Returns whether
    /// buffers of a previous key were invalidated.
    pub(super) fn reconfigure(&mut self, key: PoolKey) -> bool {
        if self.key == Some(key) {
            return false;
        }
        let invalidated = self.key.is_some();
        self.key = Some(key);
        self.retired
            .extend(self.slots.drain(..).filter(Slot::is_leased));
        self.collect_retired();
        invalidated
    }

    /// Choose the buffer for the next screencopy.
    pub(super) fn pick(&mut self) -> Pick {
        self.collect_retired();
        if let Some(i) = self.slots.iter().position(|s| !s.is_leased()) {
            return Pick::Free(SlotIdx(i));
        }
        if self.alive() < self.cap {
            Pick::Grow
        } else {
            Pick::Exhausted
        }
    }

    /// Add a freshly allocated buffer, unleased.
    pub(super) fn insert(&mut self, buffer: B) -> SlotIdx {
        self.slots.push(Slot::new(buffer));
        SlotIdx(self.slots.len() - 1)
    }

    pub(super) fn get(&self, idx: SlotIdx) -> Option<&B> {
        self.slots.get(idx.0).map(|s| &s.buffer)
    }

    /// Lease the buffer at `idx` to a downstream frame: it won't be
    /// picked again until every clone of the returned lease is gone.
    /// `None` when `idx` is stale or the buffer is already leased.
    pub(super) fn lease(&self, idx: SlotIdx) -> Option<BufferLease> {
        let slot = self.slots.get(idx.0)?;
        // Only the worker sets the flag, so check-then-set can't race
        // with another lease; a concurrent release only clears it.
        if slot.is_leased() {
            return None;
        }
        slot.leased.store(true, Ordering::Release);
        Some(BufferLease::new(ReleaseOnDrop(Arc::clone(&slot.leased))))
    }

    /// Remove an unleased buffer from the pool (e.g. the compositor
    /// rejected it), returning it for disposal.
    pub(super) fn discard(&mut self, idx: SlotIdx) -> Option<B> {
        if self.slots.get(idx.0)?.is_leased() {
            return None;
        }
        Some(self.slots.remove(idx.0).buffer)
    }

    /// Buffers alive: pooled plus retired-but-leased.
    pub(super) fn alive(&self) -> usize {
        self.slots.len() + self.retired.len()
    }

    fn collect_retired(&mut self) {
        self.retired.retain(Slot::is_leased);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    const KEY: PoolKey = PoolKey {
        width: 1920,
        height: 1080,
        fourcc: 0x3432_5258,
    };

    /// Counts drops so tests can see when a buffer is freed.
    #[derive(Debug)]
    struct Bo(Arc<AtomicUsize>);

    impl Drop for Bo {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn pool(cap: usize) -> (BufferPool<Bo>, Arc<AtomicUsize>) {
        let mut pool = BufferPool::with_cap(cap);
        assert!(!pool.reconfigure(KEY), "first key invalidates nothing");
        (pool, Arc::new(AtomicUsize::new(0)))
    }

    fn grow(pool: &mut BufferPool<Bo>, drops: &Arc<AtomicUsize>) -> SlotIdx {
        assert_eq!(pool.pick(), Pick::Grow);
        pool.insert(Bo(Arc::clone(drops)))
    }

    #[test]
    fn unleased_buffer_is_reused() {
        let (mut pool, drops) = pool(3);
        let idx = grow(&mut pool, &drops);
        assert_eq!(pool.pick(), Pick::Free(idx));
        assert_eq!(pool.pick(), Pick::Free(idx));
        assert_eq!(pool.alive(), 1);
    }

    #[test]
    fn leased_buffer_is_skipped_until_released() {
        let (mut pool, drops) = pool(3);
        let first = grow(&mut pool, &drops);
        let lease = pool.lease(first).expect("unleased");
        let second = grow(&mut pool, &drops);
        assert_ne!(first, second);
        assert_eq!(pool.pick(), Pick::Free(second));

        drop(lease);
        assert_eq!(pool.pick(), Pick::Free(first));
    }

    #[test]
    fn lease_is_held_by_every_frame_clone() {
        let (mut pool, drops) = pool(1);
        let idx = grow(&mut pool, &drops);
        let lease = pool.lease(idx).expect("unleased");
        let clone = lease.clone();
        assert!(pool.lease(idx).is_none(), "already leased");

        drop(lease);
        assert_eq!(pool.pick(), Pick::Exhausted);
        drop(clone);
        assert_eq!(pool.pick(), Pick::Free(idx));
    }

    #[test]
    fn exhausted_at_cap_when_everything_is_leased() {
        let (mut pool, drops) = pool(2);
        let a = grow(&mut pool, &drops);
        let _la = pool.lease(a).expect("unleased");
        let b = grow(&mut pool, &drops);
        let lb = pool.lease(b).expect("unleased");
        assert_eq!(pool.pick(), Pick::Exhausted);

        drop(lb);
        assert_eq!(pool.pick(), Pick::Free(b));
    }

    #[test]
    fn release_from_another_thread_frees_the_slot() {
        let (mut pool, drops) = pool(1);
        let idx = grow(&mut pool, &drops);
        let lease = pool.lease(idx).expect("unleased");
        std::thread::spawn(move || drop(lease))
            .join()
            .expect("release thread");
        assert_eq!(pool.pick(), Pick::Free(idx));
    }

    #[test]
    fn key_change_drops_free_and_retires_leased_buffers() {
        let (mut pool, drops) = pool(3);
        let leased = grow(&mut pool, &drops);
        let lease = pool.lease(leased).expect("unleased");
        grow(&mut pool, &drops);

        let resized = PoolKey {
            width: 1280,
            height: 720,
            ..KEY
        };
        assert!(pool.reconfigure(resized));
        assert_eq!(drops.load(Ordering::SeqCst), 1, "free buffer freed now");
        assert_eq!(pool.alive(), 1, "leased buffer kept alive");
        assert!(!pool.reconfigure(resized), "same key is a no-op");

        // The retired buffer still counts toward the cap...
        let fresh = grow(&mut pool, &drops);
        let _fresh_lease = pool.lease(fresh).expect("unleased");
        grow(&mut pool, &drops);
        assert_eq!(pool.alive(), 3);
        assert_eq!(pool.pick(), Pick::Free(SlotIdx(1)));
        // ...is never handed out again, and is freed once released.
        drop(lease);
        assert!(matches!(pool.pick(), Pick::Free(i) if i != fresh));
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert_eq!(pool.alive(), 2);
    }

    #[test]
    fn discard_removes_only_unleased_buffers() {
        let (mut pool, drops) = pool(2);
        let a = grow(&mut pool, &drops);
        let _lease = pool.lease(a).expect("unleased");
        assert!(pool.discard(a).is_none(), "leased buffer stays");

        let b = grow(&mut pool, &drops);
        drop(pool.discard(b).expect("unleased"));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(pool.alive(), 1);
    }

    #[test]
    fn stale_index_is_rejected() {
        let (pool, _drops) = pool(1);
        assert!(pool.get(SlotIdx(0)).is_none());
        assert!(pool.lease(SlotIdx(0)).is_none());
    }
}
