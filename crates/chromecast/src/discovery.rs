use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use mdns_sd::{ServiceDaemon, ServiceEvent};
use tokio::sync::mpsc;
use tracing::{debug, error, info, trace, warn};
use uuid::Uuid;

use ferricast_core::{
    Codec, Device, DeviceCapabilities, Discovery, DiscoveryEvent, FerricastError, MdnsDiscovery,
    Result,
};

const CHROMECAST_SERVICE_TYPE: &str = "_googlecast._tcp.local.";

const CA_VIDEO_OUT: u32 = 1;
#[allow(dead_code)]
const CA_VIDEO_IN: u32 = 2;
const CA_AUDIO_OUT: u32 = 4;
#[allow(dead_code)]
const CA_AUDIO_IN: u32 = 8;
#[allow(dead_code)]
const CA_MULTIZONE_GROUP: u32 = 32;

pub struct ChromecastDiscovery {
    running: Arc<AtomicBool>,
    daemon: Option<ServiceDaemon>,
    task_handle: Option<tokio::task::JoinHandle<()>>,
}

impl Default for ChromecastDiscovery {
    fn default() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            daemon: None,
            task_handle: None,
        }
    }
}

impl MdnsDiscovery for ChromecastDiscovery {
    const SERVICE_TYPE: &'static str = CHROMECAST_SERVICE_TYPE;
}

impl Discovery for ChromecastDiscovery {
    const PROTOCOL: &'static str = "chromecast";

    async fn start(&mut self, tx: mpsc::Sender<DiscoveryEvent>) -> Result<()> {
        if self.running.load(Ordering::SeqCst) {
            debug!("chromecast discovery already running");
            return Ok(());
        }

        info!("starting chromecast mDNS discovery");

        let daemon = ServiceDaemon::new().map_err(|e| {
            FerricastError::Discovery(format!("failed to create mDNS daemon: {e}"))
        })?;

        let receiver = daemon.browse(CHROMECAST_SERVICE_TYPE).map_err(|e| {
            FerricastError::Discovery(format!("failed to browse for chromecast services: {e}"))
        })?;

        self.daemon = Some(daemon);
        self.running.store(true, Ordering::SeqCst);

        let running = self.running.clone();

        let handle = tokio::task::spawn(async move {
            let mut known: HashMap<String, Uuid> = HashMap::new();

            while running.load(Ordering::SeqCst) {
                let event = {
                    let receiver_ref = receiver.clone();
                    match tokio::task::spawn_blocking(move || {
                        receiver_ref.recv_timeout(std::time::Duration::from_secs(2))
                    })
                    .await
                    {
                        Ok(Ok(event)) => event,
                        Ok(Err(_timeout)) => continue,
                        Err(join_err) => {
                            error!("mDNS browse task panicked: {join_err}");
                            break;
                        }
                    }
                };

                match event {
                    ServiceEvent::ServiceResolved(info) => {
                        debug!(
                            name = info.get_fullname(),
                            "chromecast service resolved"
                        );

                        let properties = info.get_properties();
                        let txt: HashMap<String, String> = properties
                            .iter()
                            .map(|p| (p.key().to_string(), p.val_str().to_string()))
                            .collect();

                        trace!(?txt, "TXT records");

                        let device_uuid = txt
                            .get("id")
                            .and_then(|id| Uuid::try_parse(id).ok())
                            .unwrap_or_else(|| Uuid::new_v4());

                        let friendly_name = txt
                            .get("fn")
                            .cloned()
                            .unwrap_or_else(|| info.get_fullname().to_string());

                        let model = txt.get("md").cloned();

                        let ca = txt
                            .get("ca")
                            .and_then(|v| v.parse::<u32>().ok())
                            .unwrap_or(0);

                        let capabilities = DeviceCapabilities {
                            supports_audio: ca & CA_AUDIO_OUT != 0 || ca == 0,
                            supports_video: ca & CA_VIDEO_OUT != 0 || ca == 0,
                            supports_screen_mirror: ca & CA_VIDEO_OUT != 0 || ca == 0,
                            max_width: Some(1920),
                            max_height: Some(1080),
                            supported_codecs: vec![Codec::H264, Codec::Vp8],
                        };

                    
                        let addr: std::net::IpAddr = match info.get_addresses_v4().iter().next() {
                            Some(addr) => (*(*addr)).into(),
                            None => {
                                warn!(
                                    name = info.get_fullname(),
                                    "resolved service has no addresses, skipping"
                                );
                                continue;
                            }
                        };

                        let port = info.get_port();

                        let device = Device {
                            id: device_uuid,
                            name: friendly_name,
                            protocol: "chromecast",
                            addr,
                            port,
                            model,
                            capabilities,
                            metadata: txt,
                        };

                        known.insert(info.get_fullname().to_string(), device_uuid);

                        info!(
                            id = %device.id,
                            name = %device.name,
                            addr = %device.addr,
                            port = device.port,
                            "discovered chromecast device"
                        );

                        if tx.send(DiscoveryEvent::DeviceFound(device)).await.is_err()
                        {
                            debug!("discovery event channel closed, stopping");
                            break;
                        }
                    }

                    ServiceEvent::ServiceRemoved(_, fullname) => {
                        info!(name = %fullname, "chromecast service removed");

                        if let Some(uuid) = known.remove(fullname.as_str()) {
                            if tx.send(DiscoveryEvent::DeviceLost(uuid)).await.is_err() {
                                debug!("discovery event channel closed, stopping");
                                break;
                            }
                        }
                    }

                    ServiceEvent::SearchStarted(stype) => {
                        debug!(service_type = %stype, "mDNS search started");
                    }

                    ServiceEvent::ServiceFound(_, _) => {
                        trace!("service found (waiting for resolve)");
                    }

                    ServiceEvent::SearchStopped(stype) => {
                        debug!(service_type = %stype, "mDNS search stopped");
                    }
                }
            }

            debug!("chromecast discovery task exiting");
        });

        self.task_handle = Some(handle);
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        if !self.running.load(Ordering::SeqCst) {
            return Ok(());
        }

        info!("stopping chromecast mDNS discovery");
        self.running.store(false, Ordering::SeqCst);

        if let Some(daemon) = self.daemon.take() {
            if let Err(e) = daemon.shutdown() {
                warn!("error shutting down mDNS daemon: {e}");
            }
        }

        if let Some(handle) = self.task_handle.take() {
            handle.abort();
            let _ = handle.await;
        }

        debug!("chromecast discovery stopped");
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}
