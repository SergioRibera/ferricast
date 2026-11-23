//! Miracast sink discovery.
//!
//! Browses mDNS for MS-MICE sinks (`_display._tcp.local.`). On Linux
//! the P2P peer-scan path is layered on top when the `linux-p2p`
//! feature is enabled; results merge into the same
//! [`DiscoveryEvent`] channel so the GUI's device picker is protocol
//! agnostic.

use std::collections::HashSet;
use std::sync::Arc;

use tokio::sync::{mpsc, Mutex};
use uuid::Uuid;

use ferricast_core::{
    Codec, Device, DeviceAddresses, DeviceCapabilities, Discovery, DiscoveryEvent,
    FerricastError, MdnsDiscovery, Result,
};

use crate::transport::mice::MICE_SERVICE_TYPE;

#[cfg(all(target_os = "linux", feature = "linux-p2p"))]
use crate::ie::{DeviceInformation, WfdIe};
#[cfg(all(target_os = "linux", feature = "linux-p2p"))]
use crate::transport::linux_p2p::{DiscoveredPeer, LinuxP2pClient};

#[derive(Default)]
pub struct MiracastDiscovery {
    running: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    #[cfg(all(target_os = "linux", feature = "linux-p2p"))]
    p2p_task: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl MiracastDiscovery {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl MdnsDiscovery for MiracastDiscovery {
    const SERVICE_TYPE: &'static str = MICE_SERVICE_TYPE;
}

impl Discovery for MiracastDiscovery {
    const PROTOCOL: &'static str = "miracast";

    async fn start(&mut self, tx: mpsc::Sender<DiscoveryEvent>) -> Result<()> {
        let daemon = mdns_sd::ServiceDaemon::new()
            .map_err(|e| FerricastError::Discovery(format!("mdns-sd: {e}")))?;
        let receiver = daemon
            .browse(Self::SERVICE_TYPE)
            .map_err(|e| FerricastError::Discovery(format!("browse: {e}")))?;
        let running = Arc::clone(&self.running);
        let tx_mdns = tx.clone();
        let handle = tokio::spawn(async move {
            let tx = tx_mdns;
            let mut seen: HashSet<String> = HashSet::new();
            while let Ok(event) = receiver.recv_async().await {
                match event {
                    mdns_sd::ServiceEvent::ServiceResolved(service) => {
                        if !is_mice(&service) {
                            continue;
                        }
                        let fullname = service.get_fullname().to_string();
                        if !seen.insert(fullname.clone()) {
                            continue;
                        }
                        let Some(addresses) = DeviceAddresses::from_mdns(&service) else {
                            tracing::debug!(name = %fullname, "miracast: no usable address");
                            continue;
                        };
                        let name = service
                            .get_properties()
                            .get_property_val_str("fn")
                            .unwrap_or_else(|| service.get_hostname())
                            .trim_end_matches('.')
                            .to_string();
                        let mut metadata = std::collections::HashMap::new();
                        if let Some(port) = service
                            .get_properties()
                            .get_property_val_str("rtsp_port")
                        {
                            metadata.insert("rtsp_port".to_string(), port.to_string());
                        }
                        let device = Device {
                            id: Uuid::new_v4(),
                            name,
                            protocol: "miracast",
                            protocol_icon: bytes::Bytes::new(),
                            addresses,
                            model: None,
                            capabilities: DeviceCapabilities {
                                supported_codecs: vec![Codec::H264],
                                max_width: Some(1920),
                                max_height: Some(1080),
                                max_fps: Some(30),
                                ..DeviceCapabilities::default()
                            },
                            metadata,
                        };
                        if tx
                            .send(DiscoveryEvent::DeviceFound(device))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    mdns_sd::ServiceEvent::ServiceRemoved(_, fullname) => {
                        seen.remove(&fullname);
                        // We don't track IDs by fullname yet — issue
                        // a device-lost by name on the receiver side.
                    }
                    _ => {}
                }
            }
            let mut slot = running.lock().await;
            *slot = None;
        });
        let mut slot = self.running.lock().await;
        *slot = Some(handle);
        drop(slot);

        #[cfg(all(target_os = "linux", feature = "linux-p2p"))]
        {
            if let Err(e) = self.start_linux_p2p(tx.clone()).await {
                tracing::warn!(error = %e, "miracast p2p: Wi-Fi Direct discovery disabled");
            }
        }
        #[cfg(not(all(target_os = "linux", feature = "linux-p2p")))]
        let _ = &tx;
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        let mut slot = self.running.lock().await;
        if let Some(handle) = slot.take() {
            handle.abort();
        }
        #[cfg(all(target_os = "linux", feature = "linux-p2p"))]
        {
            let mut p2p = self.p2p_task.lock().await;
            if let Some(h) = p2p.take() {
                h.abort();
            }
        }
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running
            .try_lock()
            .map(|slot| slot.is_some())
            .unwrap_or(true)
    }
}

fn is_mice(service: &mdns_sd::ResolvedService) -> bool {
    service
        .get_properties()
        .get_property_val_str("mice")
        .map(|v| v.trim() == "1")
        .unwrap_or(false)
}

#[cfg(all(target_os = "linux", feature = "linux-p2p"))]
impl MiracastDiscovery {
    async fn start_linux_p2p(&self, tx: mpsc::Sender<DiscoveryEvent>) -> Result<()> {
        let wfd_ie = WfdIe::new(DeviceInformation::linux_dual_role());
        let client = LinuxP2pClient::connect(&wfd_ie)
            .await
            .map_err(|e| FerricastError::Discovery(format!("wpa_supplicant: {e}")))?;
        let mut rx = client
            .start_discovery()
            .await
            .map_err(|e| FerricastError::Discovery(format!("wpa_supplicant Find: {e}")))?;
        let tx_out = tx;
        let handle = tokio::spawn(async move {
            let _client = client; // keep alive
            while let Some(peer) = rx.recv().await {
                if !peer.is_sink() {
                    continue;
                }
                if let Some(device) = p2p_peer_to_device(&peer) {
                    if tx_out.send(DiscoveryEvent::DeviceFound(device)).await.is_err() {
                        break;
                    }
                }
            }
        });
        *self.p2p_task.lock().await = Some(handle);
        Ok(())
    }
}

#[cfg(all(target_os = "linux", feature = "linux-p2p"))]
fn p2p_peer_to_device(peer: &DiscoveredPeer) -> Option<Device> {
    let addr: std::net::SocketAddr = "0.0.0.0:7236".parse().expect("literal");
    let addresses = DeviceAddresses::ranked(
        std::iter::once(addr),
        &ferricast_core::net::local_ipv4_subnets(),
    )?;
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("p2p_peer_path".to_string(), peer.path.clone());
    metadata.insert(
        "mac".to_string(),
        peer.address
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join(":"),
    );
    Some(Device {
        id: Uuid::new_v4(),
        name: peer.name.clone(),
        protocol: "miracast",
        protocol_icon: bytes::Bytes::new(),
        addresses,
        model: None,
        capabilities: DeviceCapabilities {
            supported_codecs: vec![Codec::H264],
            max_width: Some(1920),
            max_height: Some(1080),
            max_fps: Some(30),
            ..DeviceCapabilities::default()
        },
        metadata,
    })
}
