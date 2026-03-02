use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::mpsc;

use ferricast_core::{Discovery, DiscoveryEvent, MdnsDiscovery, Result};

/// The mDNS service type for AirPlay.
const AIRPLAY_SERVICE_TYPE: &str = "_airplay._tcp.local.";

/// AirPlay device discovery implementation using mDNS-SD.
pub struct AirPlayDiscovery {
    running: Arc<AtomicBool>,
}

impl Default for AirPlayDiscovery {
    fn default() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl Discovery for AirPlayDiscovery {
    const PROTOCOL: &'static str = "airplay";

    async fn start(&mut self, tx: mpsc::Sender<DiscoveryEvent>) -> Result<()> {
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

impl MdnsDiscovery for AirPlayDiscovery {
    const SERVICE_TYPE: &'static str = AIRPLAY_SERVICE_TYPE;
}
