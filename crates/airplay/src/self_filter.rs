//! Device ids this process advertises as an AirPlay receiver, so its own
//! discovery does not list itself as a target.

use std::collections::HashSet;
use std::sync::{Mutex, PoisonError};

static OWN: Mutex<Option<HashSet<String>>> = Mutex::new(None);

fn normalise(device_id: &str) -> String {
    device_id.to_ascii_lowercase()
}

pub(crate) fn register(device_id: &str) {
    OWN.lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get_or_insert_with(HashSet::new)
        .insert(normalise(device_id));
}

pub(crate) fn unregister(device_id: &str) {
    if let Some(own) = OWN.lock().unwrap_or_else(PoisonError::into_inner).as_mut() {
        own.remove(&normalise(device_id));
    }
}

pub(crate) fn is_own(device_id: &str) -> bool {
    OWN.lock()
        .unwrap_or_else(PoisonError::into_inner)
        .as_ref()
        .is_some_and(|own| own.contains(&normalise(device_id)))
}

#[cfg(test)]
mod tests {
    #[test]
    fn registered_ids_match_case_insensitively() {
        super::register("AA:BB:CC:00:11:22");
        assert!(super::is_own("aa:bb:cc:00:11:22"));
        super::unregister("aa:bb:cc:00:11:22");
        assert!(!super::is_own("AA:BB:CC:00:11:22"));
    }
}
