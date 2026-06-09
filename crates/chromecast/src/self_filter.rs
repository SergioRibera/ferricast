//! Cross-process self-discovery suppression.
//!
//! When the same binary runs as both a Chromecast sender (with the
//! discovery loop) and a Chromecast receiver (with the mDNS publisher),
//! the discovery's `_googlecast._tcp` browse picks up the
//! advertisement the same process just published. That's a real bug
//! in practice — the picker UI lists ourselves as a target, and casting
//! to it just lights up the loopback path, which collides with the
//! sender's own HLS server and triggers the receiver-paused / segment-
//! 404 cascade we keep tripping when testing same-host.
//!
//! The fix is small: the receiver advertiser registers its
//! `device_id` here on start (the same id it ships in the `id` TXT
//! record), the discovery resolver checks the registry on every
//! `ServiceResolved` event and drops matches. Cross-process casting
//! still works — IDs are random per-process so two ferricast binaries
//! on the same LAN see each other normally.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

fn registry() -> &'static Mutex<HashSet<String>> {
    static REG: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Mark `id` as belonging to the running process. The mDNS discovery
/// will skip any resolved service whose `id` TXT record equals this.
pub fn register(id: &str) {
    if let Ok(mut g) = registry().lock() {
        g.insert(id.to_string());
    }
}

/// Remove `id` from the self-set (called on receiver stop so a later
/// restart with a different id doesn't trip on a stale entry).
pub fn unregister(id: &str) {
    if let Ok(mut g) = registry().lock() {
        g.remove(id);
    }
}

/// Returns true if `id` is currently registered as a self-published
/// receiver. Cheap — backed by a `Mutex<HashSet<String>>` with at most
/// a handful of entries (one per active receiver advertiser).
pub fn contains(id: &str) -> bool {
    registry()
        .lock()
        .map(|g| g.contains(id))
        .unwrap_or(false)
}
