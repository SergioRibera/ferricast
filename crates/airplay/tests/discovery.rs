//! Browses for a UxPlay-like receiver advertised in-process with mdns-sd.
//!
//! Ignored by default: it needs multicast on a real interface.
//! Run with `cargo test -p ferricast-airplay --test discovery -- --ignored`.

use std::collections::HashMap;
use std::time::Duration;

use ferricast_airplay::AirPlayDiscovery;
use ferricast_core::{AirPlayPairing, Discovery, DiscoveryEvent};
use mdns_sd::{ServiceDaemon, ServiceInfo};

#[tokio::test]
#[ignore = "needs multicast networking"]
async fn finds_advertised_uxplay_receiver() {
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
    let host_ip = ferricast_core::local_addr_for("224.0.0.251".parse().unwrap()).unwrap();
    let info = ServiceInfo::new(
        "_airplay._tcp.local.",
        "ferricast-discovery-test",
        "ferricast-discovery-test.local.",
        host_ip,
        47999,
        txt,
    )
    .unwrap();
    responder.register(info).unwrap();

    let (tx, mut rx) = tokio::sync::mpsc::channel(16);
    let mut discovery = AirPlayDiscovery::default();
    discovery.start(tx).await.unwrap();
    let found = tokio::time::timeout(Duration::from_secs(15), async {
        while let Some(event) = rx.recv().await {
            if let DiscoveryEvent::DeviceFound(device) = event
                && device.name == "ferricast-discovery-test"
            {
                return device;
            }
        }
        panic!("discovery channel closed");
    })
    .await
    .expect("receiver not discovered");
    assert_eq!(found.port, 47999);
    let config = found.capabilities.airplay_config.unwrap();
    assert_eq!(config.pairing_mode(), AirPlayPairing::Legacy);
    assert_eq!(config.device_id.as_deref(), Some("02:11:22:33:44:55"));

    discovery.stop().await.unwrap();
    assert!(!discovery.is_running());
    let _ = responder.shutdown();
}
