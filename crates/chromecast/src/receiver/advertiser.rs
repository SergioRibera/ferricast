//! mDNS publish for the receiver-side Chromecast endpoint.
//!
//! Publishes `_googlecast._tcp` with the TXT keys senders look up
//! when populating the picker UI:
//!
//! - `id` — opaque device UUID (we surface the one from `AdvertiseInfo::device_id`)
//! - `fn` — friendly name
//! - `md` — model "Chromecast Ultra" by default so YouTube treats us
//!   as a video-capable receiver; senders use this to grey out
//!   "Cast to" for incompatible models
//! - `ca` — capabilities bitfield (`5` = video + audio, the value
//!   real Chromecasts advertise)
//! - `ic` — icon path (`/setup/icon.png`, mostly cosmetic)
//! - `ve` — protocol version (`05`)
//! - `rs` — current status text (we ship a blank value)
//!
//! All TXT keys can be overridden via `AdvertiseInfo::txt` —
//! supplying a key there wins over our defaults.

use ferricast_core::{AdvertiseInfo, Advertiser, FerricastError, Result};
use mdns_sd::{IfKind, ServiceDaemon, ServiceInfo};

use crate::self_filter;

pub const SERVICE_TYPE: &str = "_googlecast._tcp.local.";

/// Env-var override for the LAN-facing interface. Setting it to either
/// an interface name (`eth0`) or an IPv4 (`192.168.1.10`) pins mDNS
/// announces to that interface and skips runtime detection. Useful on
/// hosts where the default-route heuristic picks the wrong NIC —
/// e.g. machines where a VPN holds the default route but the LAN
/// receiver is on a separate subnet, or hosts with multiple LAN
/// adapters.
pub const ADVERTISE_INTERFACE_VAR: &str = "FERRICAST_CAST_INTERFACE";

#[derive(Default)]
pub struct ChromecastReceiverAdvertiser {
    daemon: Option<ServiceDaemon>,
    fullname: Option<String>,
    /// Cached for `stop()` so we can unregister from the self-filter
    /// without holding onto the full [`AdvertiseInfo`].
    device_id: Option<String>,
}

impl ChromecastReceiverAdvertiser {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Advertiser for ChromecastReceiverAdvertiser {
    const PROTOCOL: &'static str = "chromecast";

    fn start(&mut self, info: AdvertiseInfo) -> impl std::future::Future<Output = Result<()>> + Send {
        async move {
            let daemon = ServiceDaemon::new()
                .map_err(|e| FerricastError::Receiver(format!("mDNS daemon: {e}")))?;

            // Restrict announces to a single physical LAN interface.
            // Without this, on hosts with many interfaces (docker
            // bridges, virbr0, tailscale0, libvirt, WSL bridges, VPN
            // tunnels, multi-NIC servers — typical NixOS dev or any
            // cloud-ish box) mdns-sd fans out the announce across
            // every IP it finds. Cast Discovery on Android reads the
            // first response packet it sees per service; if that
            // response came from a virtual interface that only got
            // the SRV/A record in time (the TXT goes in a separate
            // packet whose ordering across 15+ multicast sockets is
            // racy), Android marks the service incomplete and
            // refuses to connect — the "texts: ,  network: 997"
            // failure mode in MdnsDeviceScannerEntry's log.
            //
            // IPv6 is always disabled: Cast Discovery on every sender
            // we care about (YouTube, Spotify, VLC mobile, Cast SDK
            // apps) only uses IPv4 multicast (224.0.0.251), so IPv6
            // announces would only ever add wire noise without ever
            // being read.
            let _ = daemon.disable_interface(IfKind::IPv6);
            let pinned_ip = pin_lan_interface(&daemon);

            // Default TXT keys real Chromecast senders look for.
            // `AdvertiseInfo::txt` overrides anything we set here.
            //
            // The order is the order real Chromecast firmware emits.
            // We use an ordered slice (not a HashMap) so the encoded
            // TXT record bytes match the field order Android's Cast
            // Discovery Service parses against — empirically, when
            // ordered by HashMap iteration (random), Android's mDNS
            // marks the service `Device status is missing` even
            // though every required key is present. With an ordered
            // construction the same data parses successfully.
            //
            // Bonus: dropping the empty `rs=` value — a few strict
            // parsers (older CAF) treat `key=` (length-prefixed
            // 3-byte "rs=") differently from `key` (2-byte "rs")
            // and prefer the key-only form for empty values. We
            // emit `key` only.
            let mut props: Vec<(String, String)> = Vec::new();
            props.push(("id".into(), info.device_id.clone()));
            props.push(("fn".into(), info.friendly_name.clone()));
            props.push(("md".into(), "Chromecast Ultra".into()));
            props.push(("ca".into(), "5".into()));
            props.push(("ic".into(), "/setup/icon.png".into()));
            props.push(("ve".into(), "05".into()));
            // Also emit the keys real Chromecast firmware advertises
            // alongside the basic set. Some Cast Discovery
            // implementations on Android refuse to mark a service
            // complete without them.
            //
            // - `nf=1` — network protocol Wi-Fi.
            // - `bs` — Bluetooth MAC; we fake one stable per
            //   device_id so re-advertises don't churn the entry in
            //   sender caches.
            // - `st=0` — Cast status flags (idle).
            // - `rm` — multizone group leader (empty when standalone).
            props.push(("nf".into(), "1".into()));
            props.push((
                "bs".into(),
                info.device_id
                    .chars()
                    .filter(|c| c.is_ascii_hexdigit())
                    .take(12)
                    .collect::<String>()
                    .to_uppercase(),
            ));
            props.push(("st".into(), "0".into()));
            // Note: we skip emitting `rs` and `rm` entirely when
            // their values would be empty — see comment above.
            for (k, v) in &info.txt {
                // Caller-supplied overrides: replace existing entry
                // with the same key, else append.
                if let Some(pos) = props.iter().position(|(pk, _)| pk == k) {
                    props[pos].1 = v.clone();
                } else {
                    props.push((k.clone(), v.clone()));
                }
            }
            let props_ref: Vec<(&str, &str)> = props
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();

            // Build the announce. When `pin_lan_interface` succeeded
            // we already know the IP we'll be announcing on — embed
            // it directly so the initial response packet carries the
            // A record. Falling back to `enable_addr_auto()` (used
            // only when detection fails completely) is correct but
            // produces a slightly later A record that can race the
            // TXT across consumers; we prefer the explicit path.
            let host_ip_str = match pinned_ip {
                Some(ip) => ip.to_string(),
                None => String::new(),
            };
            let hostname = format!("{}-ferricast.local.", info.device_id);
            let mut service = ServiceInfo::new(
                SERVICE_TYPE,
                &info.friendly_name,
                &hostname,
                host_ip_str.as_str(),
                info.port,
                &props_ref[..],
            )
            .map_err(|e| FerricastError::Receiver(format!("mDNS ServiceInfo: {e}")))?;
            if pinned_ip.is_none() {
                service = service.enable_addr_auto();
            }

            let fullname = service.get_fullname().to_string();
            daemon
                .register(service)
                .map_err(|e| FerricastError::Receiver(format!("mDNS register: {e}")))?;
            // Tell the in-process Chromecast discovery to skip this
            // device's id when it resolves. Otherwise same-process
            // sender+receiver lists ourselves in the picker.
            self_filter::register(&info.device_id);
            tracing::info!(
                fullname,
                port = info.port,
                device_id = %info.device_id,
                lan_ip = ?pinned_ip,
                txt_keys = %props.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>().join(","),
                "Chromecast receiver advertised via mDNS"
            );
            self.daemon = Some(daemon);
            self.fullname = Some(fullname);
            self.device_id = Some(info.device_id);
            Ok(())
        }
    }

    fn stop(&mut self) -> impl std::future::Future<Output = Result<()>> + Send {
        async move {
            if let (Some(daemon), Some(fullname)) = (self.daemon.take(), self.fullname.take()) {
                let _ = daemon.unregister(&fullname);
                let _ = daemon.shutdown();
            }
            if let Some(id) = self.device_id.take() {
                self_filter::unregister(&id);
            }
            Ok(())
        }
    }

    fn is_running(&self) -> bool {
        self.daemon.is_some()
    }
}

/// Detect the LAN-facing IPv4 and configure `daemon` so mDNS
/// announces only go out on that interface. Two paths:
///
/// 1. **Env override** (`FERRICAST_CAST_INTERFACE`) — if set, the
///    value is interpreted as either an interface name (`eth0`,
///    `wlan0`) or a literal IPv4 (`192.168.1.10`). We always
///    accept whatever the user supplied; if the host doesn't have
///    that interface mdns-sd will just emit nothing, which is the
///    desired failure mode for a manual override.
/// 2. **UDP + `connect`** heuristic — open a non-routed UDP socket,
///    `connect()` to a routable IPv4 sink (`8.8.8.8:53`). The
///    kernel runs source-address selection and binds the socket to
///    the IP it would use for that destination; we read it back
///    with `local_addr()`. No packets actually leave the host
///    because we never call `send_to` — purely a kernel-side
///    address lookup. Works on every Linux/macOS/Windows host with
///    a default IPv4 route, including offline LAN-only hosts (the
///    kernel still picks a source based on routing table even when
///    there's no path to 8.8.8.8).
///
/// Returns the pinned IP so the caller can embed it directly in the
/// `ServiceInfo`. When both paths fail (no routes at all, very rare),
/// returns `None` and the caller falls back to `enable_addr_auto`.
fn pin_lan_interface(daemon: &ServiceDaemon) -> Option<std::net::IpAddr> {
    if let Ok(val) = std::env::var(ADVERTISE_INTERFACE_VAR) {
        let val = val.trim();
        if !val.is_empty() {
            // Try IP first; if parsing succeeds it's an Addr, else
            // treat as interface name.
            let kind = match val.parse::<std::net::IpAddr>() {
                Ok(ip) => IfKind::Addr(ip),
                Err(_) => IfKind::Name(val.to_string()),
            };
            let _ = daemon.disable_interface(IfKind::All);
            let _ = daemon.enable_interface(kind.clone());
            tracing::info!(
                override_value = val,
                ?kind,
                "mDNS interface pinned via {} env var",
                ADVERTISE_INTERFACE_VAR
            );
            // We don't know the resulting IP up front when only the
            // name was given. Returning None makes the caller fall
            // back to `enable_addr_auto` for the ServiceInfo, which
            // is fine — mdns-sd's auto-detect runs against the
            // already-filtered interface set, so it resolves to the
            // pinned IP anyway.
            if let IfKind::Addr(ip) = kind {
                return Some(ip);
            }
            return None;
        }
    }

    let detected = detect_default_route_ipv4()?;
    let _ = daemon.disable_interface(IfKind::All);
    let _ = daemon.enable_interface(IfKind::Addr(detected));
    Some(detected)
}

use super::util::detect_default_route_ipv4;
