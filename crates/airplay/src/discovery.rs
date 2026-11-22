//! AirPlay receiver discovery over mDNS (`_airplay._tcp`).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use bytes::Bytes;
use ferricast_core::device::{AirplayConfig, Features, Flags};
use ferricast_core::{
    Codec, Device, DeviceAddresses, DeviceCapabilities, Discovery, DiscoveryEvent, FerricastError,
    MdnsDiscovery, Result,
};
use mdns_sd::{ServiceDaemon, ServiceEvent};
use plist::Value;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::crypto::sha512_16;
use crate::rtsp::{ClientIdentity, RtspClient};

const AIRPLAY_SERVICE_TYPE: &str = "_airplay._tcp.local.";
const AIRPLAY_ICON: Bytes = Bytes::from_static(include_bytes!("../../../assets/airplay.svg"));
const INFO_TIMEOUT: Duration = Duration::from_secs(3);

/// AirPlay device discovery using mDNS-SD.
#[derive(Default)]
pub struct AirPlayDiscovery {
    running: Arc<AtomicBool>,
    daemon: Option<ServiceDaemon>,
    handle: Option<JoinHandle<()>>,
}

impl MdnsDiscovery for AirPlayDiscovery {
    const SERVICE_TYPE: &'static str = AIRPLAY_SERVICE_TYPE;
}

/// Stable device id: the same receiver keeps its UUID across restarts.
fn stable_id(device_id: &str) -> Uuid {
    uuid::Builder::from_random_bytes(sha512_16(&[b"ferricast-airplay:", device_id.as_bytes()]))
        .into_uuid()
}

/// Build a [`Device`] from a resolved service, or `None` if it cannot
/// mirror the screen.
fn device_from_service(
    fullname: &str,
    addresses: DeviceAddresses,
    txt: &HashMap<String, String>,
) -> Option<Device> {
    let Some(features) = txt.get("features").and_then(|f| Features::parse_txt(f)) else {
        warn!(
            name = fullname,
            "AirPlay service without valid features, skipping"
        );
        return None;
    };
    if !features.contains(Features::MIRRORING_SUPPORTED) {
        debug!(name = fullname, %features, "AirPlay receiver without screen mirroring, skipping");
        return None;
    }
    let mut flags = txt
        .get("flags")
        .and_then(|v| {
            u32::from_str_radix(v.trim_start_matches("0x").trim_start_matches("0X"), 16).ok()
        })
        .map(Flags::from_bits_retain)
        .unwrap_or_else(Flags::empty);
    // UxPlay / RPiPlay signal "PIN or password required" only via `pw`.
    if txt
        .get("pw")
        .is_some_and(|v| v.eq_ignore_ascii_case("true"))
    {
        flags |= Flags::PASSWORD_REQUIRED;
    }
    let device_id = txt.get("deviceid").cloned();
    if device_id.as_deref().is_some_and(crate::self_filter::is_own) {
        debug!(name = fullname, "skipping our own AirPlay receiver");
        return None;
    }
    let id = stable_id(device_id.as_deref().unwrap_or(fullname));
    Some(Device {
        id,
        // mdns-sd escapes dots inside the instance name.
        name: mdns_sd::instance_name_unescaped(fullname),
        protocol: "airplay",
        protocol_icon: AIRPLAY_ICON,
        addresses,
        model: txt.get("model").cloned(),
        metadata: HashMap::new(),
        capabilities: DeviceCapabilities {
            supports_audio: features.contains(Features::AUDIO_SUPPORTED),
            supports_video: features.contains(Features::VIDEO_SUPPORTED),
            supports_screen_mirror: true,
            supported_codecs: vec![Codec::H264],
            max_width: Some(1920),
            max_height: Some(1080),
            max_fps: Some(30),
            airplay_config: Some(AirplayConfig {
                features,
                flags,
                pk: txt.get("pk").cloned(),
                device_id,
            }),
            ..Default::default()
        },
    })
}

/// Display limits from `GET /info`: `(width, height, fps)`.
async fn probe_info(device: &Device) -> Result<(u32, u32, u32)> {
    let mut rtsp =
        RtspClient::connect(device.addresses.as_slice(), ClientIdentity::default(), None).await?;
    let reply = rtsp
        .request("GET", "/info", None, &[])
        .await?
        .ensure_success("GET /info")?;
    let info = plist::from_bytes::<Value>(&reply.body)
        .ok()
        .and_then(Value::into_dictionary)
        .ok_or_else(|| FerricastError::Protocol("GET /info: not a plist dictionary".into()))?;
    let display = info
        .get("displays")
        .and_then(Value::as_array)
        .and_then(|d| d.first())
        .and_then(Value::as_dictionary)
        .ok_or_else(|| FerricastError::Protocol("GET /info: no displays".into()))?;
    let get = |key: &str, default: u32| {
        display
            .get(key)
            .and_then(Value::as_unsigned_integer)
            .and_then(|v| u32::try_from(v).ok())
            .filter(|v| *v > 0)
            .unwrap_or(default)
    };
    Ok((
        get("widthPixels", 1920),
        get("heightPixels", 1080),
        get("maxFPS", 30),
    ))
}

impl Discovery for AirPlayDiscovery {
    const PROTOCOL: &'static str = "airplay";

    async fn start(&mut self, tx: mpsc::Sender<DiscoveryEvent>) -> Result<()> {
        if self.running.load(Ordering::SeqCst) {
            return Ok(());
        }
        let daemon = ServiceDaemon::new()
            .map_err(|e| FerricastError::Discovery(format!("cannot start mDNS daemon: {e}")))?;
        let events = daemon.browse(AIRPLAY_SERVICE_TYPE).map_err(|e| {
            FerricastError::Discovery(format!("cannot browse for AirPlay receivers: {e}"))
        })?;
        self.daemon = Some(daemon);
        self.running.store(true, Ordering::SeqCst);
        info!("AirPlay discovery started");

        let running = self.running.clone();
        self.handle = Some(tokio::spawn(async move {
            // Last announcement per service, before the `/info` probe.
            let mut known: HashMap<String, Device> = HashMap::new();
            while running.load(Ordering::SeqCst) {
                let Ok(event) = events.recv_async().await else {
                    break;
                };
                match event {
                    ServiceEvent::ServiceResolved(service) => {
                        let txt: HashMap<String, String> = service
                            .get_properties()
                            .iter()
                            .map(|p| (p.key().to_owned(), p.val_str().to_owned()))
                            .collect();
                        let fullname = service.get_fullname().to_owned();
                        let Some(addresses) = DeviceAddresses::from_mdns(&service) else {
                            warn!(
                                name = fullname,
                                "AirPlay service without usable addresses, skipping"
                            );
                            continue;
                        };
                        let Some(mut device) = device_from_service(&fullname, addresses, &txt)
                        else {
                            continue;
                        };
                        // mDNS re-resolves a service on every announcement;
                        // an unchanged one needs no new probe or event.
                        if known.get(&fullname) == Some(&device) {
                            debug!(name = %device.name, "AirPlay receiver re-announced unchanged");
                            continue;
                        }
                        known.insert(fullname, device.clone());
                        let tx = tx.clone();
                        // Probe display limits off the event loop so one slow
                        // receiver doesn't delay the others.
                        tokio::spawn(async move {
                            match tokio::time::timeout(INFO_TIMEOUT, probe_info(&device)).await {
                                Ok(Ok((w, h, fps))) => {
                                    device.capabilities.max_width = Some(w);
                                    device.capabilities.max_height = Some(h);
                                    device.capabilities.max_fps = Some(fps);
                                }
                                Ok(Err(e)) => {
                                    debug!(name = %device.name, error = %e, "AirPlay /info probe failed")
                                }
                                Err(_) => {
                                    debug!(name = %device.name, "AirPlay /info probe timed out")
                                }
                            }
                            info!(name = %device.name, addresses = ?device.addresses.as_slice(), "discovered AirPlay receiver");
                            let _ = tx.send(DiscoveryEvent::DeviceFound(device)).await;
                        });
                    }
                    ServiceEvent::ServiceRemoved(_, fullname) => {
                        if let Some(id) = known.remove(&fullname).map(|d| d.id)
                            && tx.send(DiscoveryEvent::DeviceLost(id)).await.is_err()
                        {
                            break;
                        }
                    }
                    ServiceEvent::SearchStarted(_)
                    | ServiceEvent::ServiceFound(_, _)
                    | ServiceEvent::SearchStopped(_) => {}
                    other => debug!(?other, "unhandled mDNS event"),
                }
            }
            running.store(false, Ordering::SeqCst);
        }));
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
        if let Some(daemon) = self.daemon.take() {
            if let Err(e) = daemon.stop_browse(AIRPLAY_SERVICE_TYPE) {
                debug!(error = %e, "stop AirPlay browse");
            }
            if let Err(e) = daemon.shutdown() {
                debug!(error = %e, "shut down mDNS daemon");
            }
        }
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addrs(list: &[&str]) -> DeviceAddresses {
        DeviceAddresses::ranked(list.iter().map(|a| a.parse().unwrap()), &[]).unwrap()
    }

    /// TXT record UxPlay 1.73.6 registers for `_airplay._tcp`.
    fn uxplay_txt(pw: bool) -> HashMap<String, String> {
        [
            ("deviceid", "60:cf:84:ad:6f:97"),
            ("features", "0x527FFEE6,0x0"),
            ("pw", if pw { "true" } else { "false" }),
            ("flags", "0x4"),
            ("model", "AppleTV3,2"),
            (
                "pk",
                "19a1e2ab4f4c48d78716415be798d18ae5fbab408033af2d82df3f639a9dea54",
            ),
            ("srcvers", "220.68"),
            ("vv", "2"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect()
    }

    #[test]
    fn uxplay_record_becomes_a_legacy_mirroring_device() {
        let d = device_from_service(
            "ferricast-test._airplay._tcp.local.",
            addrs(&["[fe80::1%2]:7000", "192.168.1.20:7000", "10.0.0.5:7000"]),
            &uxplay_txt(false),
        )
        .unwrap();
        assert_eq!(d.name, "ferricast-test");
        let all: Vec<String> = d.addresses.iter().map(|a| a.to_string()).collect();
        assert_eq!(
            all,
            ["10.0.0.5:7000", "192.168.1.20:7000", "[fe80::1%2]:7000"]
        );
        assert!(d.metadata.is_empty());
        let cfg = d.capabilities.airplay_config.unwrap();
        assert_eq!(cfg.pairing_mode(), ferricast_core::AirPlayPairing::Legacy);
        assert_eq!(cfg.device_id.as_deref(), Some("60:cf:84:ad:6f:97"));
        // Same deviceid → same UUID on every resolve.
        let again = device_from_service(
            "x._airplay._tcp.local.",
            addrs(&["10.0.0.5:7000"]),
            &uxplay_txt(false),
        )
        .unwrap();
        assert_eq!(again.id, d.id);
    }

    #[test]
    fn pw_true_requires_pin_pairing() {
        let d = device_from_service(
            "u._airplay._tcp.local.",
            addrs(&["10.0.0.5:7000"]),
            &uxplay_txt(true),
        )
        .unwrap();
        let cfg = d.capabilities.airplay_config.unwrap();
        assert_eq!(
            cfg.pairing_mode(),
            ferricast_core::AirPlayPairing::LegacyPin
        );
    }

    #[test]
    fn receivers_without_screen_mirroring_are_skipped() {
        let mut txt = uxplay_txt(false);
        txt.insert("features".into(), "0x10".into());
        assert!(
            device_from_service("a._airplay._tcp.local.", addrs(&["10.0.0.5:7000"]), &txt)
                .is_none()
        );
        txt.insert("features".into(), "nope".into());
        assert!(
            device_from_service("a._airplay._tcp.local.", addrs(&["10.0.0.5:7000"]), &txt)
                .is_none()
        );
    }
}
