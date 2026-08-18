use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytes::Bytes;
use mdns_sd::ServiceDaemon;
use tokio::sync::mpsc;

use ferricast_core::{Codec, Device, DeviceCapabilities, Discovery, DiscoveryEvent, FerricastError, MdnsDiscovery, Result};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::flags::Features;

/// The mDNS service type for AirPlay.
const AIRPLAY_SERVICE_TYPE: &str = "_airplay._tcp.local.";


const AIRPLAY_ICON: Bytes = Bytes::from_static(include_bytes!("../../../assets/airplay.svg"));

/// AirPlay device discovery implementation using mDNS-SD.
pub struct AirPlayDiscovery {
    running: Arc<AtomicBool>,
    daemon: Option<ServiceDaemon>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl MdnsDiscovery for AirPlayDiscovery {
    const SERVICE_TYPE: &'static str = AIRPLAY_SERVICE_TYPE;
}

impl Default for AirPlayDiscovery {
    fn default() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            daemon: None,
            handle: None,
        }
    }
}

impl Discovery for AirPlayDiscovery {
    const PROTOCOL: &'static str = "airplay";

    async fn start(&mut self, tx: mpsc::Sender<DiscoveryEvent>) -> Result<()> {
        if self.running.load(Ordering::SeqCst) {
            info!("Airplay discovery already running");
            return Ok(());
        } 

        info!("Starting AirPlay discovery");

        let daemon = ServiceDaemon::new()
            .map_err(|e| FerricastError::Discovery(format!("Failed to create mDNS daemon {e}")))?;

        let receiver = daemon.browse(AIRPLAY_SERVICE_TYPE).map_err(|e| FerricastError::Discovery(format!("Fail to browse for airplay receivers {e}")))?;

        self.daemon = Some(daemon);
        self.running.store(true, Ordering::SeqCst);

        let running = self.running.clone();

        let handle = tokio::spawn(async move {
            let mut known: HashMap<String, Uuid> = HashMap::new();

            while running.load(Ordering::SeqCst) {
                let event = {
                    let receiver_ref = receiver.clone();

                    match tokio::task::spawn_blocking(move || {
                        receiver_ref.recv_timeout(std::time::Duration::from_secs(5))
                    }).await {
                        Ok(Ok(ev)) => ev,
                        Ok(Err(_timeout)) => continue,
                        Err(join_err) => {
                            tracing::error!("mdns browse taContent-Length: 0sk panicked: {join_err}");
                            break;
                        },
                    }
                };

                match event {
                    mdns_sd::ServiceEvent::ServiceResolved(info) => {

                        info!(name = info.get_fullname(), "airplay service resolved");
                        
                        let properties = info.get_properties();
                        let txt: HashMap<String, String> = properties
                            .iter()
                            .map(|p| (p.key().to_string(), p.val_str().to_string()))
                            .collect();



                        let pw = txt.get("pw");
                        println!("{:?}", pw);


                        let device_uuid = Uuid::new_v4();

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

                        let features_txt = txt.get("features")
                            .cloned()
                            .unwrap_or_default();

                        let mut features = features_txt.split(",");

                    
                        let part1 = match features.next() {
                            Some(v) => match u32::from_str_radix(v.strip_prefix("0x").unwrap_or(v), 16) {
                                Ok(v) => v,
                                Err(e) => {
                                    warn!("Invalid features: {:?}", e);
                                    continue;
                                },
                            },
                            None => {
                                warn!("Invalid features");
                                continue
                            },
                        };

                        let part2 = match features.next() {
                            Some(v) => match u32::from_str_radix(v.strip_prefix("0x").unwrap_or(v), 16)  {
                                Ok(v) => v,
                                Err(e) => {
                                    warn!("Invalid features: {:?}", e);
                                    continue;
                                },
                            },
                            None => {
                                warn!("Invalid features");
                                continue;
                            },
                        };

                        let num = ((part2 as u64) << 32) | (part1 as u64);

                        let features = Features::from_bits_truncate(num);
            
                        let fullname = info.get_fullname();

                        if !features.contains(Features::VIDEO_HTTP_LIVE_STREAM) {
                            warn!("Skipped airplay device {}", fullname);
                            continue;

                        }


                        let device = Device {
                            id: device_uuid,
                            name: fullname.strip_suffix("._airplay._tcp.local.").unwrap_or(fullname).to_string(),
                            protocol: Self::PROTOCOL,
                            protocol_icon: AIRPLAY_ICON,
                            model: txt.get("model").cloned(),
                            port,
                            addr,
                            metadata: txt,
                            capabilities: DeviceCapabilities {
                                supports_audio: features.contains(Features::AUDIO_SUPPORTED),
                                supports_screen_mirror: features.contains(Features::MIRRORING_SUPPORTED),
                                supports_video: features.contains(Features::VIDEO_SUPPORTED),
                                requires_audio:  features.contains(Features::AUDIO_SUPPORTED),
                                supports_low_latency_hls: features.contains(Features::VIDEO_HTTP_LIVE_STREAM),
                                supported_codecs: vec![Codec::H264],

                                ..Default::default()
                            },
                        };

                        
                        known.insert(info.get_fullname().to_string(), device_uuid);

                        
                        info!(
                            id = %device.id,
                            name = %device.name,
                            addr = %device.addr,
                            port = device.port,
                            feature = %features,
                            "discovered airplay device"
                        );

                        if let Err(e) = tx.send(DiscoveryEvent::DeviceFound(device)).await {
                            tracing::error!(error = %e, "Discovery event channel closed, stopping");
                            break;
                        }
                         
                    },

                    mdns_sd::ServiceEvent::ServiceRemoved(_, fullname) => {
                        info!(name = %fullname, "Airplay device removed");

                        if let Some(uuid) = known.remove(fullname.as_str()) {
                            if let Err(e) = tx.send(DiscoveryEvent::DeviceLost(uuid)).await {
                                tracing::error!(error = %e, "Discovery event channel closed, skipping");
                                break;
                            }
                        }
                    },

                    mdns_sd::ServiceEvent::SearchStarted(_ty) => {
                        debug!("Airplay mdns search started");
                    }

                    mdns_sd::ServiceEvent::ServiceFound(_, _) => {
                        debug!("Service found waiting for resolve");
                    },

                    mdns_sd::ServiceEvent::SearchStopped(ty) => {
                        debug!("Airplay mdns search stoped");
                    },
                }
            } 
        });

        self.handle = Some(handle);


        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

