//! A bounded pool of reusable frame buffers.
//!
//! Frames own a [`Pooled`] item that goes back to the pool when they drop, on
//! whatever thread that happens. The pool counts items handed out and refuses
//! more than its capacity, which bounds what a caller that never drops frames
//! can make the decoder allocate. When the frame geometry changes the pool
//! starts a new generation: idle items are freed and items of an older
//! generation are freed instead of pooled when they come back.

use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::error::{Error, Result};

#[derive(Debug)]
struct Inner<T> {
    generation: u64,
    idle: Vec<T>,
    outstanding: usize,
    capacity: usize,
}

#[derive(Debug)]
struct Shared<T> {
    inner: Mutex<Inner<T>>,
}

impl<T> Shared<T> {
    /// Locks the pool. The state is consistent between statements, so a
    /// panic elsewhere while holding the lock leaves nothing to repair.
    fn lock(&self) -> MutexGuard<'_, Inner<T>> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The decoder's side of the pool.
#[derive(Debug)]
pub(super) struct Pool<T> {
    shared: Arc<Shared<T>>,
}

impl<T> Pool<T> {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            shared: Arc::new(Shared {
                inner: Mutex::new(Inner {
                    generation: 0,
                    idle: Vec::with_capacity(capacity),
                    outstanding: 0,
                    capacity,
                }),
            }),
        }
    }

    /// An idle item, or a new one from `make` while fewer than the capacity
    /// are handed out.
    pub(super) fn take(&self, make: impl FnOnce() -> Result<T>) -> Result<Pooled<T>> {
        let (reused, generation) = {
            let mut inner = self.shared.lock();
            if inner.outstanding >= inner.capacity {
                return Err(Error::FramesInFlight {
                    capacity: inner.capacity,
                });
            }
            inner.outstanding += 1;
            (inner.idle.pop(), inner.generation)
        };
        let item = match reused {
            Some(item) => item,
            None => match make() {
                Ok(item) => item,
                Err(error) => {
                    self.shared.lock().outstanding -= 1;
                    return Err(error);
                }
            },
        };
        Ok(Pooled {
            item: ManuallyDrop::new(item),
            generation,
            shared: Arc::clone(&self.shared),
        })
    }

    /// Starts a new generation: idle items are freed now, handed-out ones
    /// when they come back.
    pub(super) fn retire(&self) {
        let idle = {
            let mut inner = self.shared.lock();
            inner.generation += 1;
            std::mem::take(&mut inner.idle)
        };
        drop(idle);
    }

    #[cfg(test)]
    fn idle(&self) -> usize {
        self.shared.lock().idle.len()
    }
}

/// An item lent by a [`Pool`], returned on drop.
#[derive(Debug)]
pub(super) struct Pooled<T> {
    item: ManuallyDrop<T>,
    generation: u64,
    shared: Arc<Shared<T>>,
}

impl<T> Deref for Pooled<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.item
    }
}

impl<T> DerefMut for Pooled<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.item
    }
}

impl<T> Drop for Pooled<T> {
    fn drop(&mut self) {
        // SAFETY: `item` is taken exactly once, here, and never used again.
        let item = unsafe { ManuallyDrop::take(&mut self.item) };
        let stale = {
            let mut inner = self.shared.lock();
            inner.outstanding -= 1;
            if inner.generation == self.generation {
                inner.idle.push(item);
                None
            } else {
                Some(item)
            }
        };
        // Freed outside the lock: freeing device memory waits for the GPU.
        drop(stale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn items_are_reused_and_bounded() {
        let pool = Pool::new(2);
        let mut allocations = 0;
        let mut allocate = || {
            allocations += 1;
            Ok(vec![0_u8; 4])
        };
        let first = pool.take(&mut allocate).unwrap();
        let second = pool.take(&mut allocate).unwrap();
        assert!(matches!(
            pool.take(&mut allocate),
            Err(Error::FramesInFlight { capacity: 2 })
        ));
        drop(first);
        let third = pool.take(&mut allocate).unwrap();
        drop((second, third));
        assert_eq!(allocations, 2);
        assert_eq!(pool.idle(), 2);
    }

    #[test]
    fn retired_items_are_freed() {
        let pool = Pool::new(4);
        let held = pool.take(|| Ok(1)).unwrap();
        let idle = pool.take(|| Ok(2)).unwrap();
        drop(idle);
        assert_eq!(pool.idle(), 1);
        pool.retire();
        assert_eq!(pool.idle(), 0);
        drop(held);
        assert_eq!(pool.idle(), 0);
        assert_eq!(*pool.take(|| Ok(3)).unwrap(), 3);
    }

    #[test]
    fn a_failed_make_frees_the_slot() {
        let pool = Pool::<u8>::new(1);
        assert!(
            pool.take(|| Err(Error::SizeOverflow { what: "test" }))
                .is_err()
        );
        assert!(pool.take(|| Ok(1)).is_ok());
    }
}
