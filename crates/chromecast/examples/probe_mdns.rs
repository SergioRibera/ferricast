//! Pure-Rust mDNS probe — same library (`mdns-sd`) we use to publish,
//! used here to verify what we publish actually shows up on the wire.
//!
//! Avahi is the usual debugger but it needs a system daemon running;
//! `nix-shell -p avahi` won't help because `avahi-browse` talks to
//! that daemon over D-Bus. This example bypasses that entirely.
//!
//! Run:
//!     cargo run --example probe_mdns -p ferricast-chromecast
//!
//! It browses `_googlecast._tcp` for ~10 s, prints every resolved
//! service plus its TXT records, then exits. Use it after starting
//! `ferricast-gui` to confirm the TXT records reach the network
//! (look for `Ferricast` in the listing, and verify it lists `id`,
//! `fn`, `md`, `ca`, `ic`, `ve`, `nf`, `bs`, `st`).

use std::time::Duration;

use mdns_sd::{ServiceDaemon, ServiceEvent};

fn main() {
    let daemon = ServiceDaemon::new().expect("ServiceDaemon");
    let rx = daemon.browse("_googlecast._tcp.local.").expect("browse");
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    println!("probing _googlecast._tcp.local. for 10 s ...");
    while std::time::Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        match rx.recv_timeout(remaining) {
            Ok(ServiceEvent::ServiceResolved(info)) => {
                let txt: Vec<String> = info
                    .get_properties()
                    .iter()
                    .map(|p| {
                        let v = p
                            .val()
                            .map(|b| String::from_utf8_lossy(b).into_owned())
                            .unwrap_or_default();
                        format!("{}={}", p.key(), v)
                    })
                    .collect();
                println!("\n=== {} ===", info.get_fullname());
                println!("  hostname: {}", info.get_hostname());
                println!("  addrs:    {:?}", info.get_addresses_v4());
                println!("  port:     {}", info.get_port());
                println!("  txt ({} keys):", txt.len());
                for t in txt {
                    println!("    {}", t);
                }
            }
            Ok(ServiceEvent::ServiceFound(_ty, name)) => {
                println!("  found: {} (resolving …)", name);
            }
            Ok(ServiceEvent::ServiceRemoved(_, name)) => {
                println!("  removed: {}", name);
            }
            Ok(ServiceEvent::SearchStarted(ty)) => {
                println!("  search started for {}", ty);
            }
            Ok(ServiceEvent::SearchStopped(_)) => {}
            Err(_) => break,
        }
    }
    let _ = daemon.shutdown();
    println!("\nprobe done");
}
