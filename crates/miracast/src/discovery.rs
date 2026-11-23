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

#[derive(Default)]
pub struct MiracastDiscovery {
    running: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
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
        let handle = tokio::spawn(async move {
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
                            metadata: std::collections::HashMap::new(),
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
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        let mut slot = self.running.lock().await;
        if let Some(handle) = slot.take() {
            handle.abort();
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
