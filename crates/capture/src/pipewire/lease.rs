//! Keeps DMA-BUF `pw_buffer`s dequeued while downstream reads them.
//!
//! PipeWire re-queues a buffer to the compositor as soon as it is
//! queued back, after which the compositor may render the next frame
//! into the same DMA-BUF. A `GpuFrame` handed to an encoder therefore
//! carries a [`BufferLease`] whose guard, on drop, posts the buffer's
//! [`LeaseId`] back to the PipeWire loop thread, which re-queues it.
//! Raw buffer pointers never leave that thread.
//!
//! Lease ids are never reused. When PipeWire removes a buffer
//! (renegotiation, disconnect) its entry is forgotten, so a release
//! arriving later matches nothing and never touches freed memory.

use std::ptr::NonNull;

use ferricast_core::BufferLease;
use pipewire as pw;
use pw::spa::buffer::Data;
use pw::stream::StreamRef;
use tracing::{debug, trace};

/// Identity of one hold on a dequeued buffer. Unique for the life of
/// the worker, so a stale release can't alias a newer hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LeaseId(u64);

/// Posted from a lease guard's `Drop` to the PipeWire loop.
pub(super) struct Release(LeaseId);

/// The guard inside a [`BufferLease`]; carries no pointer, only the id.
struct ReleaseOnDrop {
    id: LeaseId,
    tx: pw::channel::Sender<Release>,
}

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        // Fails only once the worker is gone, and with it the buffers.
        if self.tx.send(Release(self.id)).is_err() {
            trace!(
                id = self.id.0,
                "lease released after the PipeWire worker stopped"
            );
        }
    }
}

/// Bookkeeping of held buffers, independent of PipeWire so it can be
/// unit-tested. `B` is the buffer handle (`NonNull<pw_buffer>`).
#[derive(Debug)]
pub(super) struct HeldBuffers<B> {
    /// Buffers currently allocated on the stream, tracked through the
    /// `add_buffer` / `remove_buffer` events.
    pool_size: usize,
    held: Vec<(LeaseId, B)>,
    next_id: u64,
}

impl<B> Default for HeldBuffers<B> {
    fn default() -> Self {
        Self {
            pool_size: 0,
            held: Vec::new(),
            next_id: 0,
        }
    }
}

impl<B: Copy + PartialEq> HeldBuffers<B> {
    pub(super) fn buffer_added(&mut self) {
        self.pool_size += 1;
    }

    /// PipeWire is about to free `buffer`. Returns whether it was held;
    /// its lease, if any, becomes a no-op.
    pub(super) fn buffer_removed(&mut self, buffer: B) -> bool {
        self.pool_size = self.pool_size.saturating_sub(1);
        let before = self.held.len();
        self.held.retain(|&(_, b)| b != buffer);
        self.held.len() != before
    }

    /// Most buffers we may hold at once: all but one, so the
    /// compositor always has a buffer to render the next frame into.
    pub(super) fn limit(&self) -> usize {
        self.pool_size.saturating_sub(1)
    }

    pub(super) fn held(&self) -> usize {
        self.held.len()
    }

    /// Record `buffer` as held, or `None` at the limit.
    pub(super) fn hold(&mut self, buffer: B) -> Option<LeaseId> {
        if self.held.len() >= self.limit() {
            return None;
        }
        let id = LeaseId(self.next_id);
        self.next_id += 1;
        self.held.push((id, buffer));
        Some(id)
    }

    /// End the hold `id`, returning the buffer to re-queue. `None` for
    /// a hold whose buffer was removed meanwhile.
    pub(super) fn release(&mut self, id: LeaseId) -> Option<B> {
        let pos = self.held.iter().position(|&(held, _)| held == id)?;
        Some(self.held.swap_remove(pos).1)
    }
}

/// Lease side of the PipeWire worker. Lives on the loop thread only.
pub(super) struct BufferLeases {
    book: HeldBuffers<NonNull<pw::sys::pw_buffer>>,
    tx: pw::channel::Sender<Release>,
}

impl BufferLeases {
    pub(super) fn new(tx: pw::channel::Sender<Release>) -> Self {
        Self {
            book: HeldBuffers::default(),
            tx,
        }
    }

    pub(super) fn buffer_added(&mut self) {
        self.book.buffer_added();
        trace!(pool = self.book.pool_size, "PipeWire buffer added");
    }

    pub(super) fn buffer_removed(&mut self, buffer: *mut pw::sys::pw_buffer) {
        let Some(buffer) = NonNull::new(buffer) else {
            return;
        };
        if self.book.buffer_removed(buffer) {
            debug!("PipeWire removed a buffer still leased downstream; its lease is now inert");
        }
    }

    /// Keep `buffer` dequeued until the returned lease's last clone
    /// drops. At the hold limit the buffer is handed back unchanged.
    pub(super) fn hold<'s>(
        &mut self,
        buffer: DequeuedBuffer<'s>,
    ) -> std::result::Result<BufferLease, DequeuedBuffer<'s>> {
        let Some(id) = self.book.hold(buffer.raw) else {
            return Err(buffer);
        };
        // The book now owns the re-queue; the guard must not do it too.
        std::mem::forget(buffer);
        Ok(BufferLease::new(ReleaseOnDrop {
            id,
            tx: self.tx.clone(),
        }))
    }

    pub(super) fn limit(&self) -> usize {
        self.book.limit()
    }

    pub(super) fn held(&self) -> usize {
        self.book.held()
    }

    /// Re-queue the buffer behind `release`, unless it is gone.
    pub(super) fn release(&mut self, stream: &StreamRef, release: Release) {
        let Some(buffer) = self.book.release(release.0) else {
            trace!(id = release.0.0, "late release of a removed buffer ignored");
            return;
        };
        // SAFETY: `buffer` came from `dequeue_raw_buffer` on this stream
        // and has not been queued since (`HeldBuffers::release` returns
        // each hold once). It is still allocated: `remove_buffer` drops
        // the entry before PipeWire frees it, and releases are only
        // processed on the loop thread while the stream exists (the
        // release receiver is detached before the stream is destroyed).
        unsafe { stream.queue_raw_buffer(buffer.as_ptr()) };
    }
}

/// A buffer dequeued from the stream; re-queued on drop unless handed
/// to [`BufferLeases::hold`]. Like `pw::buffer::Buffer`, but its raw
/// pointer can be kept past the `process` callback.
pub(super) struct DequeuedBuffer<'s> {
    stream: &'s StreamRef,
    raw: NonNull<pw::sys::pw_buffer>,
}

impl<'s> DequeuedBuffer<'s> {
    /// Dequeue everything ready, re-queue all but the newest. Returns
    /// the newest buffer and how many were dequeued.
    pub(super) fn newest(stream: &'s StreamRef) -> (Option<Self>, usize) {
        let mut drained = 0;
        let mut newest = None;
        loop {
            // SAFETY: the returned pointer is either NULL or a buffer
            // owned by `stream`, which `Self` re-queues exactly once.
            let raw = unsafe { stream.dequeue_raw_buffer() };
            let Some(raw) = NonNull::new(raw) else {
                break;
            };
            drained += 1;
            // Dropping the previous newest re-queues it.
            newest = Some(Self { stream, raw });
        }
        (newest, drained)
    }

    pub(super) fn datas_mut(&mut self) -> &mut [Data] {
        // SAFETY: `raw` is a live dequeued `pw_buffer`; while we hold it
        // PipeWire doesn't touch its `spa_buffer`. `Data` is a
        // `repr(transparent)` wrapper of `spa_data`, the same cast
        // `pw::buffer::Buffer::datas_mut` performs.
        unsafe {
            let buffer = self.raw.as_ref().buffer;
            if buffer.is_null() || (*buffer).datas.is_null() {
                return &mut [];
            }
            let Ok(n) = usize::try_from((*buffer).n_datas) else {
                return &mut [];
            };
            std::slice::from_raw_parts_mut((*buffer).datas.cast::<Data>(), n)
        }
    }
}

impl Drop for DequeuedBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: `raw` was dequeued from `stream` and not queued since.
        unsafe { self.stream.queue_raw_buffer(self.raw.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::HeldBuffers;

    fn pool(n: usize) -> HeldBuffers<u32> {
        let mut book = HeldBuffers::default();
        (0..n).for_each(|_| book.buffer_added());
        book
    }

    #[test]
    fn holds_all_but_one_buffer() {
        let mut book = pool(4);
        assert_eq!(book.limit(), 3);
        let ids: Vec<_> = (0..3).map(|b| book.hold(b).expect("under limit")).collect();
        assert_eq!(book.hold(3), None, "the compositor keeps one buffer");
        assert_eq!(book.release(ids[1]), Some(1));
        assert!(book.hold(3).is_some());
        assert_eq!(book.held(), 3);
    }

    #[test]
    fn tiny_or_unknown_pool_holds_nothing() {
        assert_eq!(pool(0).hold(0), None);
        assert_eq!(pool(1).hold(0), None);
    }

    #[test]
    fn release_returns_each_hold_once() {
        let mut book = pool(3);
        let id = book.hold(7).expect("under limit");
        assert_eq!(book.release(id), Some(7));
        assert_eq!(book.release(id), None);
        // Same buffer held again gets a fresh id; the old one stays dead.
        let again = book.hold(7).expect("under limit");
        assert_ne!(again, id);
        assert_eq!(book.release(id), None);
        assert_eq!(book.release(again), Some(7));
    }

    #[test]
    fn removed_buffers_make_late_releases_inert() {
        let mut book = pool(3);
        let held = book.hold(1).expect("under limit");
        let kept = book.hold(2).expect("under limit");
        // Renegotiation: PipeWire frees the whole pool.
        assert!(book.buffer_removed(1));
        assert!(book.buffer_removed(2));
        assert!(!book.buffer_removed(3));
        assert_eq!(book.held(), 0);
        assert_eq!(book.limit(), 0);
        // New pool, possibly reusing the same addresses.
        (0..3).for_each(|_| book.buffer_added());
        let fresh = book.hold(1).expect("under limit");
        assert_eq!(book.release(held), None);
        assert_eq!(book.release(kept), None);
        assert_eq!(book.release(fresh), Some(1));
    }

    #[test]
    fn shrinking_pool_lowers_the_limit_below_current_holds() {
        let mut book = pool(4);
        let a = book.hold(0).expect("under limit");
        book.hold(1).expect("under limit");
        assert!(!book.buffer_removed(3));
        assert!(!book.buffer_removed(2));
        assert_eq!(book.limit(), 1);
        assert_eq!(book.hold(2), None);
        assert_eq!(book.release(a), Some(0));
        assert_eq!(book.hold(2), None, "still at the limit with one held");
    }
}
