use tokio::sync::mpsc;

use ferricast_core::{Discovery, DiscoveryEvent, Result};

#[derive(Default)]
pub struct MiracastDiscovery;

impl Discovery for MiracastDiscovery {
    const PROTOCOL: &'static str = "miracast";

    async fn start(&mut self, tx: mpsc::Sender<DiscoveryEvent>) -> Result<()> {
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        Ok(())
    }

    fn is_running(&self) -> bool {
        false
    }
}
