//! What happens to a codec session, for whoever watches it: which
//! backend opened, a move to another backend and why, keyframe and
//! bitrate requests, failures, the end of the session.
//!
//! Every `Selecting*` codec owns a [`SessionEvents`] hub. Subscribers
//! get a bounded queue each; emitting never blocks the media path: an
//! event that does not fit a full queue is dropped for that subscriber,
//! which then receives a [`SessionEvent::Lagged`] with the number it
//! missed before its next event. A subscriber whose receiver is gone is
//! dropped.

use std::sync::mpsc;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use ferricast_core::{Acceleration, BackendId, CodecError, CodecResult};

/// Queue length of a subscription unless the subscriber picks one.
pub const DEFAULT_CAPACITY: usize = 32;

/// One thing that happened to a codec session.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "istmo", istmo_macros::message(crate = "::istmo_core"))]
pub enum SessionEvent {
    /// A backend opened for the session's selection (on the first
    /// `configure`, on a new selection, and to every new subscriber for
    /// the backend open at that moment).
    Opened {
        /// The backend now open.
        backend: BackendId,
        /// Whether it runs on media hardware.
        acceleration: Acceleration,
    },
    /// The session moved from `from` to `to` because `from` failed with
    /// `reason` (a decoder rejecting the stream, an encoder refusing a
    /// new configuration) and the selection allows falling back.
    Switched {
        /// The backend that failed.
        from: BackendId,
        /// The backend now open.
        to: BackendId,
        /// Whether `to` runs on media hardware.
        acceleration: Acceleration,
        /// Why `from` was left.
        reason: CodecError,
    },
    /// A keyframe was asked for.
    KeyframeRequested,
    /// The target bitrate was changed.
    BitrateChanged {
        /// The new target, kbit/s.
        kbps: u32,
    },
    /// A call failed with `error`, and no fallback applied.
    Failed {
        /// The error the call returned.
        error: CodecError,
    },
    /// This subscriber's queue was full: `missed` events were dropped
    /// before the next one.
    Lagged {
        /// Events dropped.
        missed: u64,
    },
    /// The session ended; nothing follows.
    Closed,
}

/// Whether an event reached a subscriber.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Delivery {
    Sent,
    Full,
    Gone,
}

/// Where a subscriber's events go.
pub(crate) trait Sink: Send {
    /// Queues `event` without blocking.
    fn offer(&self, event: SessionEvent) -> Delivery;
}

impl Sink for mpsc::SyncSender<SessionEvent> {
    fn offer(&self, event: SessionEvent) -> Delivery {
        match self.try_send(event) {
            Ok(()) => Delivery::Sent,
            Err(mpsc::TrySendError::Full(_)) => Delivery::Full,
            Err(mpsc::TrySendError::Disconnected(_)) => Delivery::Gone,
        }
    }
}

#[cfg(feature = "istmo")]
impl Sink for flume::Sender<SessionEvent> {
    fn offer(&self, event: SessionEvent) -> Delivery {
        match self.try_send(event) {
            Ok(()) => Delivery::Sent,
            Err(flume::TrySendError::Full(_)) => Delivery::Full,
            Err(flume::TrySendError::Disconnected(_)) => Delivery::Gone,
        }
    }
}

struct Subscriber {
    sink: Box<dyn Sink>,
    missed: u64,
}

impl Subscriber {
    /// Delivers `event`, after the pending [`SessionEvent::Lagged`]
    /// marker; `false` once the receiver is gone.
    fn deliver(&mut self, event: SessionEvent) -> bool {
        if self.missed > 0 {
            match self.sink.offer(SessionEvent::Lagged {
                missed: self.missed,
            }) {
                Delivery::Sent => self.missed = 0,
                Delivery::Full => {
                    self.missed += 1;
                    return true;
                }
                Delivery::Gone => return false,
            }
        }
        match self.sink.offer(event) {
            Delivery::Sent => true,
            Delivery::Full => {
                self.missed += 1;
                true
            }
            Delivery::Gone => false,
        }
    }
}

/// The subscribers of one codec session. Cloning shares them.
#[derive(Clone, Default)]
pub struct SessionEvents(Arc<Mutex<Vec<Subscriber>>>);

impl std::fmt::Debug for SessionEvents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionEvents")
            .field("subscribers", &self.subscribers())
            .finish()
    }
}

impl SessionEvents {
    /// Subscribes with a queue of `capacity` events (at least one).
    #[must_use]
    pub fn subscribe(&self, capacity: usize) -> EventReceiver {
        self.subscribe_with(capacity, None)
    }

    /// [`Self::subscribe`], with `first` queued for this subscriber only.
    #[must_use]
    pub fn subscribe_with(&self, capacity: usize, first: Option<SessionEvent>) -> EventReceiver {
        let (sender, receiver) = mpsc::sync_channel(capacity.max(1));
        self.attach(Box::new(sender), first);
        EventReceiver(receiver)
    }

    /// Adds a subscriber that receives through `sink`.
    pub(crate) fn attach(&self, sink: Box<dyn Sink>, first: Option<SessionEvent>) {
        let mut subscriber = Subscriber { sink, missed: 0 };
        if first.is_none_or(|event| subscriber.deliver(event)) {
            self.lock().push(subscriber);
        }
    }

    /// Live subscribers.
    #[must_use]
    pub fn subscribers(&self) -> usize {
        self.lock().len()
    }

    /// Subscribes with a queue of `capacity`; the first event is
    /// [`SessionEvent::Opened`] for `current`, the backend open now.
    #[must_use]
    pub fn subscribe_at(
        &self,
        capacity: usize,
        current: Option<(BackendId, Acceleration)>,
    ) -> EventReceiver {
        let first = current.map(|(backend, acceleration)| SessionEvent::Opened {
            backend,
            acceleration,
        });
        self.subscribe_with(capacity, first)
    }

    /// Tells subscribers about `result`'s error, and hands it back.
    pub(crate) fn report<T>(&self, result: CodecResult<T>) -> CodecResult<T> {
        result.inspect_err(|error| {
            self.emit(&SessionEvent::Failed {
                error: error.clone(),
            });
        })
    }

    /// Sends `event` to every subscriber, dropping those that are gone.
    pub fn emit(&self, event: &SessionEvent) {
        let mut subscribers = self.lock();
        subscribers.retain_mut(|subscriber| subscriber.deliver(event.clone()));
    }

    /// The subscriber list holds plain queues a panic cannot corrupt.
    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Subscriber>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// A subscription to a session's events. Dropping it unsubscribes.
#[derive(Debug)]
pub struct EventReceiver(mpsc::Receiver<SessionEvent>);

impl EventReceiver {
    /// The next event, waiting for it; `None` once the session is gone
    /// and every event was received.
    #[must_use]
    pub fn recv(&self) -> Option<SessionEvent> {
        self.0.recv().ok()
    }

    /// The next event if one is queued.
    #[must_use]
    pub fn try_recv(&self) -> Option<SessionEvent> {
        self.0.try_recv().ok()
    }

    /// The next event, waiting at most `timeout`.
    #[must_use]
    pub fn recv_timeout(&self, timeout: Duration) -> Option<SessionEvent> {
        self.0.recv_timeout(timeout).ok()
    }

    /// Every event queued now, without waiting.
    pub fn drain(&self) -> impl Iterator<Item = SessionEvent> + '_ {
        self.0.try_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_queue_drops_events_and_reports_how_many() {
        let events = SessionEvents::default();
        let receiver = events.subscribe(1);
        for kbps in 1..=4 {
            events.emit(&SessionEvent::BitrateChanged { kbps });
        }
        assert_eq!(
            receiver.try_recv(),
            Some(SessionEvent::BitrateChanged { kbps: 1 })
        );
        events.emit(&SessionEvent::KeyframeRequested);
        assert_eq!(
            receiver.try_recv(),
            Some(SessionEvent::Lagged { missed: 3 })
        );
        events.emit(&SessionEvent::Closed);
        assert_eq!(
            receiver.try_recv(),
            Some(SessionEvent::Lagged { missed: 1 })
        );
        assert_eq!(receiver.try_recv(), None);
    }

    #[test]
    fn a_dropped_receiver_unsubscribes_and_others_still_hear() {
        let events = SessionEvents::default();
        let kept = events.subscribe_with(
            4,
            Some(SessionEvent::Opened {
                backend: BackendId::X264,
                acceleration: Acceleration::Software,
            }),
        );
        drop(events.subscribe(4));
        events.emit(&SessionEvent::KeyframeRequested);
        assert_eq!(events.subscribers(), 1);
        assert_eq!(kept.drain().count(), 2);
    }
}
