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

pub const SERVICE_TYPE: &str = "_googlecast._tcp.local.";

#[derive(Default)]
pub struct ChromecastReceiverAdvertiser {
    daemon: Option<ServiceDaemon>,
    fullname: Option<String>,
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
            let mut txt = std::collections::HashMap::new();
            txt.insert("id".to_string(), info.device_id.clone());
            txt.insert("fn".to_string(), info.friendly_name.clone());
            txt.insert("md".to_string(), "Chromecast Ultra".to_string());
            txt.insert("ca".to_string(), "5".to_string());
            txt.insert("ic".to_string(), "/setup/icon.png".to_string());
            txt.insert("ve".to_string(), "05".to_string());
            txt.insert("rs".to_string(), String::new());
            for (k, v) in &info.txt {
                txt.insert(k.clone(), v.clone());
            }

            // mdns-sd resolves the host's reachable IP itself when
            // we pass `&[]`. The `instance_name` is the user-facing
            // string in the picker; the `hostname` is what mdns-sd
            // uses in PTR/SRV records.
            let hostname = format!("{}-ferricast.local.", info.device_id);
            let service = ServiceInfo::new(
                SERVICE_TYPE,
                &info.friendly_name,
                &hostname,
                "",
                info.port,
                Some(txt),
            )
            .map_err(|e| FerricastError::Receiver(format!("mDNS ServiceInfo: {e}")))?
            .enable_addr_auto();

            let fullname = service.get_fullname().to_string();
            daemon
                .register(service)
                .map_err(|e| FerricastError::Receiver(format!("mDNS register: {e}")))?;
            tracing::info!(
                fullname,
                port = info.port,
                "Chromecast receiver advertised via mDNS"
            );
            self.daemon = Some(daemon);
            self.fullname = Some(fullname);
            Ok(())
        }
    }

    fn stop(&mut self) -> impl std::future::Future<Output = Result<()>> + Send {
        async move {
            if let (Some(daemon), Some(fullname)) = (self.daemon.take(), self.fullname.take()) {
                let _ = daemon.unregister(&fullname);
                let _ = daemon.shutdown();
            }
            Ok(())
        }
    }

    fn is_running(&self) -> bool {
        self.daemon.is_some()
    }
}
