use std::collections::HashMap;
use std::net::Ipv4Addr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytes::Bytes;
use mdns_sd::{ServiceDaemon, ServiceEvent};
use tokio::sync::mpsc;
use tracing::{debug, error, info, trace, warn};
use uuid::Uuid;

use ferricast_core::{
    Codec, Device, DeviceCapabilities, Discovery, DiscoveryEvent, FerricastError, H264Profile,
    H265Profile, MdnsDiscovery, Result,
};

use crate::self_filter;

const CHROMECAST_SERVICE_TYPE: &str = "_googlecast._tcp.local.";
const CHROMECAST_ICON: Bytes = Bytes::from_static(include_bytes!("../../../assets/chromecast.svg"));

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

        let daemon = ServiceDaemon::new()
            .map_err(|e| FerricastError::Discovery(format!("failed to create mDNS daemon: {e}")))?;

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
                        debug!(name = info.get_fullname(), "chromecast service resolved");

                        let properties = info.get_properties();
                        let txt: HashMap<String, String> = properties
                            .iter()
                            .map(|p| (p.key().to_string(), p.val_str().to_string()))
                            .collect();

                        trace!(?txt, "TXT records");

                        // Self-filter: if this resolved service is
                        // our own in-process receiver advertisement,
                        // skip it. Same-machine sender+receiver in
                        // one binary otherwise lists itself in the
                        // picker and (worse) lets the user pick
                        // itself, which deadlocks the HLS pipeline
                        // through the loopback path.
                        if let Some(id) = txt.get("id")
                            && self_filter::contains(id)
                        {
                            debug!(
                                name = info.get_fullname(),
                                device_id = %id,
                                "skipping self-advertised chromecast receiver"
                            );
                            continue;
                        }

                        let device_uuid =
                            device_uuid(txt.get("id").map(String::as_str), info.get_fullname());

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

                        let candidates: Vec<Ipv4Addr> =
                            info.get_addresses_v4().into_iter().copied().collect();
                        let addr: std::net::IpAddr =
                            match pick_address(&candidates, &local_ipv4_networks()) {
                                Some(addr) => addr.into(),
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
                            protocol_icon: CHROMECAST_ICON,
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

                        if tx.send(DiscoveryEvent::DeviceFound(device)).await.is_err() {
                            debug!("discovery event channel closed, stopping");
                            break;
                        }
                    }

                    ServiceEvent::ServiceRemoved(_, fullname) => {
                        info!(name = %fullname, "chromecast service removed");

                        if let Some(uuid) = known.remove(fullname.as_str())
                            && tx.send(DiscoveryEvent::DeviceLost(uuid)).await.is_err()
                        {
                            debug!("discovery event channel closed, stopping");
                            break;
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

        if let Some(daemon) = self.daemon.take()
            && let Err(e) = daemon.shutdown()
        {
            warn!("error shutting down mDNS daemon: {e}");
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
        // 128 kbps AAC-LC is the Cast spec floor for 1st/2nd-gen
        // hardware. Modern model branches below bump this.
        max_audio_bitrate_kbps: Some(128),
        requires_audio: true,
        max_h264_profile: Some(H264Profile::Main),
        // 1st/2nd-gen Chromecasts have no HEVC decoder in hardware —
        // the model branches below override to Main / Main10 only on
        // devices known to support it.
        max_h265_profile: None,
        supported_codecs: vec![Codec::H264],
        // 1st/2nd-gen Chromecast firmware's HLS parser locks up on
        // EXT-X-VERSION:6 / EXT-X-PART-INF (field-tested). Newer
        // model branches below override to `true` only where it's
        // been verified to actually work.
        supports_low_latency_hls: false,
        // 1st/2nd-gen firmware may reject OFFER from non-Chrome senders.
        supports_cast_streaming: false,

        airplay_config: None,
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
        // Ultra accepts 320 kbps AAC-LC stereo without trouble.
        caps.max_audio_bitrate_kbps = Some(320);
        caps.requires_audio = false;
        caps.max_h264_profile = Some(H264Profile::High);
        caps.max_h265_profile = Some(H265Profile::Main10);
        caps.supported_codecs.extend([Codec::H265, Codec::Vp9]);
        // Ultra firmware (CAF receiver) handles EXT-X-VERSION:6 +
        // EXT-X-PART-INF. Cuts glass-to-glass from ~6 s to ~1-2 s.
        caps.supports_low_latency_hls = true;
        caps.supports_cast_streaming = true;
    } else if md_lc.contains("google tv") || md_lc.contains("android tv") {
        // Cast-built-in on Google / Android TV. Decoder is the
        // TV's hardware codec — uniformly capable of 1080p @ 60
        // High @ L4.2 minimum. Doesn't insist on audio.
        caps.max_fps = Some(60);
        caps.max_bitrate_kbps = Some(15_000);
        caps.max_audio_bitrate_kbps = Some(256);
        caps.requires_audio = false;
        caps.max_h264_profile = Some(H264Profile::High);
        // Cast-built-in TVs ship hardware HEVC Main10 decoders;
        // taking advantage cuts bitrate ~40% vs H.264 at equal
        // perceptual quality.
        caps.max_h265_profile = Some(H265Profile::Main10);
        // Modern CAF receiver ships LL-HLS + Cast Streaming support.
        caps.supports_low_latency_hls = true;
        caps.supports_cast_streaming = true;
    } else if md_lc == "chromecast audio" {
        // No display.
        caps.supports_video = false;
        caps.supports_screen_mirror = false;
        caps.max_width = None;
        caps.max_height = None;
        caps.max_fps = None;
        caps.max_h264_profile = None;
        caps.max_h265_profile = None;
        caps.supported_codecs.clear();
        // Audio-only device — push fidelity, this is the entire
        // point of the unit.
        caps.max_audio_bitrate_kbps = Some(320);
    } else if md_lc == "chromecast" {
        // Generic "Chromecast" — covers 1st, 2nd, and 3rd gen.  3rd gen
        // firmware (1.36+) supports Cast Streaming; older hardware times
        // out on OFFER and the manager falls back to HLS automatically,
        // so enabling it here is safe for all three generations.
        caps.supports_cast_streaming = true;
    } else if md_lc.contains("chromecast hd") {
        // Chromecast with Google TV HD (2022).  The mDNS md field is
        // "Chromecast HD", which does not match the "google tv" branch
        // above but is fully capable hardware.
        caps.max_fps = Some(60);
        caps.max_bitrate_kbps = Some(15_000);
        caps.max_audio_bitrate_kbps = Some(256);
        caps.requires_audio = false;
        caps.max_h264_profile = Some(H264Profile::High);
        caps.max_h265_profile = Some(H265Profile::Main10);
        caps.supports_low_latency_hls = true;
        caps.supports_cast_streaming = true;
    } else {
        // Unknown model — third-party Cast-built-in devices (smart TVs,
        // soundbars, etc.).  Bias above the floor but below Ultra.
        // Cast Streaming is enabled: if the device rejects the OFFER the
        // manager falls back to HLS within 10 s.
        // Main profile: unknown devices vary widely; High profile caused
        // black screens on some receivers in testing. Conservative here.
        caps.max_fps = Some(60);
        caps.max_bitrate_kbps = Some(10_000);
        caps.max_audio_bitrate_kbps = Some(192);
        caps.requires_audio = false;
        caps.max_h264_profile = Some(H264Profile::Main);
        caps.supports_low_latency_hls = true;
        caps.supports_cast_streaming = true;
    }

    caps
}

/// Namespace for device ids derived from mDNS data, so a device keeps the
/// same id across rediscoveries and restarts.
const DEVICE_ID_NAMESPACE: Uuid = Uuid::from_u128(0x6c1f_4a2e_93d0_4b5f_8e7a_0c3d_2b19_f4e1);

/// Stable device id: the `id` TXT record when it is a UUID (Chromecasts
/// publish 32 hex digits), otherwise a name-based UUID of that record or,
/// failing that, of the mDNS full name.
fn device_uuid(id_txt: Option<&str>, fullname: &str) -> Uuid {
    let id_txt = id_txt.filter(|id| !id.is_empty());
    id_txt
        .and_then(|id| Uuid::try_parse(id).ok())
        .unwrap_or_else(|| {
            Uuid::new_v5(&DEVICE_ID_NAMESPACE, id_txt.unwrap_or(fullname).as_bytes())
        })
}

/// IPv4 networks (address, netmask) of the non-loopback local interfaces.
fn local_ipv4_networks() -> Vec<(Ipv4Addr, Ipv4Addr)> {
    if_addrs::get_if_addrs()
        .map(|ifaces| {
            ifaces
                .into_iter()
                .filter(|i| !i.is_loopback())
                .filter_map(|i| match i.addr {
                    if_addrs::IfAddr::V4(v4) => Some((v4.ip, v4.netmask)),
                    if_addrs::IfAddr::V6(_) => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Picks the advertised address to connect to: one on a directly attached
/// subnet when available (devices often also advertise addresses of other
/// interfaces, e.g. a docker bridge on Android TV), otherwise the lowest
/// address so the choice is deterministic.
fn pick_address(candidates: &[Ipv4Addr], local: &[(Ipv4Addr, Ipv4Addr)]) -> Option<Ipv4Addr> {
    let on_local_subnet = |addr: &Ipv4Addr| {
        local.iter().any(|(ip, mask)| {
            let mask = u32::from(*mask);
            mask != 0 && u32::from(*ip) & mask == u32::from(*addr) & mask
        })
    };
    let mut sorted = candidates.to_vec();
    sorted.sort_unstable();
    sorted
        .iter()
        .copied()
        .find(on_local_subnet)
        .or_else(|| sorted.first().copied())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_id_is_stable() {
        let hex = "0123456789abcdef0123456789abcdef";
        assert_eq!(device_uuid(Some(hex), "x"), Uuid::try_parse(hex).unwrap());
        let a = device_uuid(Some("not-a-uuid"), "TV._googlecast._tcp.local.");
        assert_eq!(a, device_uuid(Some("not-a-uuid"), "other"));
        let b = device_uuid(None, "TV._googlecast._tcp.local.");
        assert_eq!(b, device_uuid(Some(""), "TV._googlecast._tcp.local."));
        assert_ne!(a, b);
    }

    #[test]
    fn prefers_address_on_local_subnet() {
        let lan = Ipv4Addr::new(192, 168, 1, 40);
        let docker = Ipv4Addr::new(172, 17, 0, 1);
        let local = [(
            Ipv4Addr::new(192, 168, 1, 10),
            Ipv4Addr::new(255, 255, 255, 0),
        )];
        assert_eq!(pick_address(&[docker, lan], &local), Some(lan));
        assert_eq!(pick_address(&[lan, docker], &[]), Some(docker));
        assert_eq!(pick_address(&[], &local), None);
    }

    #[test]
    fn vp8_is_not_advertised() {
        assert!(
            !capabilities_for_model("Chromecast", 0)
                .supported_codecs
                .contains(&Codec::Vp8)
        );
    }
}
