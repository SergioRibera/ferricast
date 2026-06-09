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
use mdns_sd::{ServiceDaemon, ServiceInfo};

use crate::self_filter;

pub const SERVICE_TYPE: &str = "_googlecast._tcp.local.";

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

            // Detect the LAN-facing IPv4 explicitly instead of relying
            // on mdns-sd's `enable_addr_auto`. With `enable_addr_auto`
            // the initial announce can land before mdns-sd has finished
            // discovering host interfaces, which on some setups
            // (notably WSL + multiple network namespaces) produces a
            // response with SRV but no TXT or A — exactly the
            // "Device status is missing" failure mode reported in
            // Android's MdnsDeviceScannerEntry log.
            let lan_ip = detect_lan_ipv4();
            let host_ip_str = match lan_ip {
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
            // Fall back to auto-detect only if explicit detection
            // returned nothing — better than crashing the receiver.
            if lan_ip.is_none() {
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
                lan_ip = ?lan_ip,
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

/// Discover the host's outgoing-route IPv4. Connects a UDP socket to
/// a public sink (no packets sent — `connect` on UDP just primes the
/// kernel's source-address selection); the bound local address is
/// the IP the kernel would actually use for off-link traffic, i.e.
/// the LAN-facing interface. Returns `None` on hosts with no IPv4
/// route at all (rare; the fallback is `enable_addr_auto`).
fn detect_lan_ipv4() -> Option<std::net::IpAddr> {
    let sock = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    // 8.8.8.8 is just a routable peer to coax the kernel into
    // picking a source address; no packets get sent because we
    // don't actually call `send_to`. The dual-stack 1.1.1.1 would
    // work equally well — the choice doesn't matter as long as it's
    // outside the host's loopback / link-local range.
    sock.connect("8.8.8.8:53").ok()?;
    let addr = sock.local_addr().ok()?.ip();
    if addr.is_unspecified() || addr.is_loopback() {
        None
    } else {
        Some(addr)
    }
}
