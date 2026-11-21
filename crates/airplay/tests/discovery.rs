//! Browses for a UxPlay-like receiver advertised in-process with mdns-sd.
//!
//! Ignored by default: it needs multicast on a real interface.
//! Run with `cargo test -p ferricast-airplay --test discovery -- --ignored`.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use ferricast_airplay::AirPlayDiscovery;
use ferricast_core::{AirPlayPairing, Device, Discovery, DiscoveryEvent};
use mdns_sd::{ServiceDaemon, ServiceInfo};

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

/// The first announcement of `name` that `done` accepts, or the
/// last one seen when the deadline passes. mdns-sd re-resolves a
/// service as A and AAAA answers trickle in, so early announcements
/// may carry IPv4 only. With the responder on this host, answers on
/// each interface also flush the addresses learned on the others, so
/// the set changes between announcements.
async fn discover(
    discovery: &mut AirPlayDiscovery,
    name: &str,
    done: impl Fn(&Device) -> bool,
) -> Device {
    let (tx, mut rx) = tokio::sync::mpsc::channel(16);
    discovery.start(tx).await.unwrap();
    let mut latest = None;
    let _ = tokio::time::timeout(Duration::from_secs(15), async {
        while let Some(event) = rx.recv().await {
            if let DiscoveryEvent::DeviceFound(device) = event
                && device.name == name
            {
                let finished = done(&device);
                latest = Some(device);
                if finished {
                    return;
                }
            }
        }
    })
    .await;
    latest.expect("receiver not discovered")
}

#[tokio::test]
#[ignore = "needs multicast networking"]
async fn finds_advertised_uxplay_receiver_on_every_address() {
    // The receiver's port: a dual-stack listener, so whichever address
    // discovery reports can be connected to.
    let listener = ferricast_core::bind_tcp(IpAddr::V6(Ipv6Addr::UNSPECIFIED), None)
        .await
        .unwrap();
    let port = listener.local_addr().unwrap().port();
    // Keep accepting: discovery probes `GET /info` on every announcement,
    // and unaccepted connections would fill the backlog.
    let acceptor = tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            drop(stream);
        }
    });

    let responder = ServiceDaemon::new().unwrap();
    let txt: HashMap<String, String> = [
        ("deviceid", "02:11:22:33:44:55"),
        ("features", "0x5A7FFEE6,0x0"),
        ("pw", "false"),
        ("flags", "0x4"),
        ("model", "AppleTV3,2"),
        ("srcvers", "220.68"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v.to_owned()))
    .collect();
    // Publish the addresses of every interface, IPv4 and IPv6, the way
    // a receiver with several links does.
    let info = ServiceInfo::new(
        "_airplay._tcp.local.",
        "ferricast-discovery-test",
        "ferricast-discovery-test.local.",
        "",
        port,
        txt,
    )
    .unwrap()
    .enable_addr_auto();
    responder.register(info).unwrap();

    let want_v6 = host_has_link_local_v6();
    let mut discovery = AirPlayDiscovery::default();
    let found = discover(&mut discovery, "ferricast-discovery-test", |d| {
        let has_v4 = d
            .addresses
            .iter()
            .any(|a| a.is_ipv4() && !a.ip().is_loopback());
        has_v4 && (!want_v6 || d.addresses.iter().any(|a| is_link_local_v6(&a)))
    })
    .await;
    let addrs: Vec<SocketAddr> = found.addresses.iter().collect();
    eprintln!("discovered addresses: {addrs:?}");

    assert!(addrs.iter().all(|a| a.port() == port), "{addrs:?}");
    assert!(found.addresses.primary().is_ipv4(), "{addrs:?}");
    // Routable IPv4 first, then IPv6; loopback (this host answers on lo
    // too) goes last.
    let routable: Vec<&SocketAddr> = addrs.iter().filter(|a| !a.ip().is_loopback()).collect();
    if let Some(i) = routable.iter().position(|a| a.is_ipv6()) {
        assert!(routable[i..].iter().all(|a| a.is_ipv6()), "{addrs:?}");
    }
    let config = found.capabilities.airplay_config.unwrap();
    assert_eq!(config.pairing_mode(), AirPlayPairing::Legacy);
    assert_eq!(config.device_id.as_deref(), Some("02:11:22:33:44:55"));

    if want_v6 {
        let link_local: Vec<SocketAddr> = addrs.iter().copied().filter(is_link_local_v6).collect();
        assert!(!link_local.is_empty(), "no link-local IPv6 in {addrs:?}");
        assert!(
            link_local
                .iter()
                .all(|a| matches!(a, SocketAddr::V6(v6) if v6.scope_id() != 0)),
            "unscoped link-local address in {link_local:?}"
        );
        // The scope ids are usable: the link-local addresses alone reach
        // the listener.
        let (_stream, connected) =
            ferricast_core::connect_tcp_any(&link_local, Duration::from_secs(3))
                .await
                .unwrap();
        assert!(link_local.contains(&connected));
    }

    discovery.stop().await.unwrap();
    assert!(!discovery.is_running());
    acceptor.abort();
    let _ = responder.shutdown();
}
