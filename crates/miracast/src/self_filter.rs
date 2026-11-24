//! Cross-process self-discovery suppression for Miracast.
//!
//! When the same binary runs as both a Miracast sender (which browses
//! `_display._tcp` mDNS) and a Miracast receiver (which publishes
//! `_display._tcp`), the browse picks up the just-published record.
//! Without filtering, the device picker lists this very process as a
//! target and casting into it collides with our own RTSP listener.
//!
//! The advertiser registers its `device_id` on start; the discovery
//! loop (mDNS + Wi-Fi Direct P2P peer scan) skips resolved services /
//! peers whose `id` TXT or WFD peer MAC maps to a registered entry.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

fn registry() -> &'static Mutex<HashSet<String>> {
    static REG: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashSet::new()))
}

pub fn register(id: &str) {
    if let Ok(mut g) = registry().lock() {
        g.insert(id.to_string());
    }
}

pub fn unregister(id: &str) {
    if let Ok(mut g) = registry().lock() {
        g.remove(id);
    }
}

#[must_use]
pub fn contains(id: &str) -> bool {
    registry().lock().map(|g| g.contains(id)).unwrap_or(false)
}
