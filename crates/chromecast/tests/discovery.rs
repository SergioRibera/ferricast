//! Browses for a Chromecast advertised in-process with mdns-sd.
//!
//! Ignored by default: it needs multicast on a real interface.
//! Run with `cargo test -p ferricast-chromecast --test discovery -- --ignored`.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Duration;

use ferricast_chromecast::ChromecastDiscovery;
use ferricast_core::{Device, Discovery, DiscoveryEvent};
use mdns_sd::{ServiceDaemon, ServiceInfo};

const CAST_ID: &str = "0123456789abcdef0123456789abcdef";

/// Whether a non-loopback interface has an IPv6 link-local address.
fn host_has_link_local_v6() -> bool {
    std::fs::read_to_string("/proc/net/if_inet6").is_ok_and(|table| {
        table
            .lines()
            .any(|l| l.starts_with("fe80") && !l.trim_end().ends_with(" lo"))
    })
}

fn is_link_local_v6(addr: &SocketAddr) -> bool {
    matches!(addr, SocketAddr::V6(v6) if v6.ip().is_unicast_link_local())
}

#[tokio::test]
#[ignore = "needs multicast networking"]
async fn finds_advertised_chromecast_on_ipv4_and_ipv6() {
    let responder = ServiceDaemon::new().unwrap();
    let txt: HashMap<String, String> = [
        ("id", CAST_ID),
        ("fn", "Ferricast Test TV"),
        ("md", "Chromecast Ultra"),
        ("ca", "201221"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v.to_owned()))
    .collect();
    let info = ServiceInfo::new(
        "_googlecast._tcp.local.",
        "Chromecast-Ultra-ferricast-test",
        "ferricast-cast-test.local.",
        "",
        8009,
        txt,
    )
    .unwrap()
    .enable_addr_auto();
    responder.register(info).unwrap();

    let want_v6 = host_has_link_local_v6();
    let complete = |d: &Device| {
        let has_v4 = d
            .addresses
            .iter()
            .any(|a| a.is_ipv4() && !a.ip().is_loopback());
        has_v4 && (!want_v6 || d.addresses.iter().any(|a| is_link_local_v6(&a)))
    };

    let (tx, mut rx) = tokio::sync::mpsc::channel(16);
    let mut discovery = ChromecastDiscovery::default();
    discovery.start(tx).await.unwrap();
    // mdns-sd re-resolves as A and AAAA answers arrive, so wait for an
    // announcement carrying both families.
    let mut latest = None;
    let _ = tokio::time::timeout(Duration::from_secs(15), async {
        while let Some(event) = rx.recv().await {
            if let DiscoveryEvent::DeviceFound(device) = event
                && device.name == "Ferricast Test TV"
            {
                let done = complete(&device);
                latest = Some(device);
                if done {
                    return;
                }
            }
        }
    })
    .await;
    let found = latest.expect("chromecast not discovered");
    let addrs: Vec<SocketAddr> = found.addresses.iter().collect();
    eprintln!("discovered addresses: {addrs:?}");

    assert!(complete(&found), "{addrs:?}");
    assert_eq!(found.id.simple().to_string(), CAST_ID);
    assert!(addrs.iter().all(|a| a.port() == 8009), "{addrs:?}");
    assert!(found.addresses.primary().is_ipv4(), "{addrs:?}");
    assert!(
        addrs
            .iter()
            .filter(|a| is_link_local_v6(a))
            .all(|a| matches!(a, SocketAddr::V6(v6) if v6.scope_id() != 0)),
        "unscoped link-local address in {addrs:?}"
    );

    discovery.stop().await.unwrap();
    let _ = responder.shutdown();
}
