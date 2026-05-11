use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use mdns_sd::{ServiceDaemon, ServiceEvent};
use tokio::sync::mpsc;
use tracing::{debug, error, info, trace, warn};
use uuid::Uuid;

use ferricast_core::{
    Codec, Device, DeviceCapabilities, Discovery, DiscoveryEvent, FerricastError, H264Profile,
    MdnsDiscovery, Result,
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

                        let capabilities =
                            capabilities_for_model(model.as_deref().unwrap_or(""), ca);

                    
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

/// Build a [`DeviceCapabilities`] profile keyed on the chromecast
/// `md` mDNS field (the model name advertised by the device) plus
/// the `ca` capability bitmask. This is the single source of truth
/// for "what will this receiver actually play" — the manager reads
/// the resulting caps before configuring capture / encoder so we
/// never blast a 4K HDR HEVC stream at a 1st-gen Chromecast.
///
/// Identification is heuristic based on the model string Google
/// hands out in mDNS. There's no documented programmatic capability
/// API on chromecast, so we work backwards from the known specs of
/// each product line. Conservative on unknowns — the default
/// branch matches `md == "Chromecast"` which is the oldest
/// (1st/2nd/3rd gen) hardware and has the strictest constraints.
fn capabilities_for_model(md: &str, ca: u32) -> DeviceCapabilities {
    let supports_video = ca & CA_VIDEO_OUT != 0 || ca == 0;
    let supports_audio = ca & CA_AUDIO_OUT != 0 || ca == 0;

    // Floor: oldest generic Chromecasts (md == "Chromecast"),
    // ve=05, ca=201221 in the wild. Decoder rated for 1080p @ 30
    // fps Main profile and rejects HLS without audio. Bitrate
    // ceiling from Cast spec for that hardware class.
    let mut caps = DeviceCapabilities {
        supports_audio,
        supports_video,
        supports_screen_mirror: supports_video,
        max_width: Some(1920),
        max_height: Some(1080),
        max_fps: Some(30),
        // 2.5 Mbps, not 5. The Cast HW datasheet rates the 1st/2nd
        // gen decoder at 5 Mbps and we tried 3.5 Mbps as a safety
        // margin, but field-tested both: even 3.5 Mbps in CBR mode
        // (with NVENC's default VBV buffer) spikes to ~6 Mbps on
        // scene changes, and 2.4 GHz Wi-Fi to a single-band Cast
        // antenna can't sustain that. The receiver aborts with
        // `detailedErrorCode=301` (MEDIA_NETWORK) after ~1 min.
        // 2.5 Mbps average keeps peaks under ~4 Mbps and runs
        // reliably end-to-end. 1080p@30 at 2.5 Mbps is acceptable
        // for screen sharing (mostly low motion). Newer hardware
        // (Ultra / Google TV / Android TV) overrides this in the
        // branches below.
        max_bitrate_kbps: Some(2_500),
        requires_audio: true,
        max_h264_profile: Some(H264Profile::Main),
        supported_codecs: vec![Codec::H264, Codec::Vp8],
        // 1st/2nd-gen Chromecast firmware's HLS parser locks up on
        // EXT-X-VERSION:6 / EXT-X-PART-INF (field-tested). Newer
        // model branches below override to `true` only where it's
        // been verified to actually work.
        supports_low_latency_hls: false,
    };

    // Lowercased once so each branch can use cheap `contains`.
    let md_lc = md.to_lowercase();

    if md_lc.contains("ultra") {
        // Chromecast Ultra: 4K @ 30 fps with HDR, or 1080p @ 60 fps.
        // H.264 High up to L5.1, HEVC Main10, VP9. Newer firmware
        // accepts video-only HLS.
        caps.max_width = Some(3840);
        caps.max_height = Some(2160);
        caps.max_fps = Some(30);
        caps.max_bitrate_kbps = Some(30_000);
        caps.requires_audio = false;
        caps.max_h264_profile = Some(H264Profile::High);
        caps.supported_codecs.extend([Codec::H265, Codec::Vp9]);
    } else if md_lc.contains("google tv") || md_lc.contains("android tv") {
        // Cast-built-in on Google / Android TV. Decoder is the
        // TV's hardware codec — uniformly capable of 1080p @ 60
        // High @ L4.2 minimum. Doesn't insist on audio.
        caps.max_fps = Some(60);
        caps.max_bitrate_kbps = Some(15_000);
        caps.requires_audio = false;
        caps.max_h264_profile = Some(H264Profile::High);
    } else if md_lc == "chromecast audio" {
        // No display.
        caps.supports_video = false;
        caps.supports_screen_mirror = false;
        caps.max_width = None;
        caps.max_height = None;
        caps.max_fps = None;
        caps.max_h264_profile = None;
        caps.supported_codecs.clear();
    } else if md_lc == "chromecast" {
        // Matches the conservative defaults above explicitly so
        // this branch documents the floor.
    } else {
        // Unknown model. Most third-party Cast-built-in devices
        // (smart speakers with screens, soundbars, TVs that
        // identify by SKU like "TV-BD5") have decoders at least
        // on par with Google TV. Bias slightly above floor but
        // below Ultra to be safe.
        caps.max_fps = Some(60);
        caps.max_bitrate_kbps = Some(10_000);
        caps.requires_audio = false;
        caps.max_h264_profile = Some(H264Profile::High);
    }

    caps
}
