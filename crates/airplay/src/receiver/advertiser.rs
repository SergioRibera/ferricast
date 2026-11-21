//! mDNS publication of the receiver: `_airplay._tcp` (screen mirroring)
//! and `_raop._tcp` (audio), both pointing at the RTSP control port.

use std::sync::{Arc, Mutex, PoisonError, Weak};

use ferricast_core::{AdvertiseInfo, Advertiser, FerricastError, Result};
use mdns_sd::{IfKind, ServiceDaemon, ServiceInfo};
use tracing::info;

use super::Settings;
use super::server::Server;

const AIRPLAY_SERVICE: &str = "_airplay._tcp.local.";
const RAOP_SERVICE: &str = "_raop._tcp.local.";

/// Pins announcements to one interface (name or IP), like the Chromecast
/// receiver's `FERRICAST_CAST_INTERFACE`.
const INTERFACE_VAR: &str = "FERRICAST_AIRPLAY_INTERFACE";

/// Publishes the receiver over mDNS.
pub struct AirPlayReceiverAdvertiser {
    settings: Arc<Settings>,
    server: Arc<Mutex<Weak<Server>>>,
    daemon: Option<ServiceDaemon>,
    fullnames: Vec<String>,
}

impl AirPlayReceiverAdvertiser {
    pub(super) fn new(settings: Arc<Settings>, server: Arc<Mutex<Weak<Server>>>) -> Self {
        Self {
            settings,
            server,
            daemon: None,
            fullnames: Vec::new(),
        }
    }
}

fn mdns(e: impl std::fmt::Display) -> FerricastError {
    FerricastError::Receiver(format!("mDNS: {e}"))
}

impl Advertiser for AirPlayReceiverAdvertiser {
    const PROTOCOL: &'static str = "airplay";

    async fn start(&mut self, info: AdvertiseInfo) -> Result<()> {
        if self.daemon.is_some() {
            return Ok(());
        }
        let port = match info.port {
            0 => self
                .server
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .upgrade()
                .map(|s| s.port())
                .ok_or_else(|| mdns("the control port is not bound"))?,
            port => port,
        };
        let daemon = ServiceDaemon::new().map_err(mdns)?;
        let _ = daemon.disable_interface(IfKind::IPv6);
        if let Ok(value) = std::env::var(INTERFACE_VAR)
            && !value.trim().is_empty()
        {
            let value = value.trim();
            let kind = value
                .parse::<std::net::IpAddr>()
                .map_or_else(|_| IfKind::Name(value.to_owned()), IfKind::Addr);
            let _ = daemon.disable_interface(IfKind::All);
            let _ = daemon.enable_interface(kind);
        }
        let s = &self.settings;
        let host = format!("{}-ferricast-airplay.local.", s.identity.device_id_hex());
        let services = [
            (AIRPLAY_SERVICE, s.config.name.clone(), s.airplay_txt()),
            (RAOP_SERVICE, s.raop_name(), s.raop_txt()),
        ];
        for (service_type, instance, txt) in services {
            let props: Vec<(&str, &str)> = txt.iter().map(|(k, v)| (*k, v.as_str())).collect();
            let service = ServiceInfo::new(service_type, &instance, &host, "", port, &props[..])
                .map_err(mdns)?
                .enable_addr_auto();
            let fullname = service.get_fullname().to_owned();
            daemon.register(service).map_err(mdns)?;
            info!(fullname, port, "AirPlay receiver advertised via mDNS");
            self.fullnames.push(fullname);
        }
        crate::self_filter::register(&s.identity.device_id_colons());
        self.daemon = Some(daemon);
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        if let Some(daemon) = self.daemon.take() {
            for fullname in self.fullnames.drain(..) {
                let _ = daemon.unregister(&fullname);
            }
            let _ = daemon.shutdown();
            crate::self_filter::unregister(&self.settings.identity.device_id_colons());
        }
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.daemon.is_some()
    }
}
