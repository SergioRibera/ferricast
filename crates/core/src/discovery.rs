use tokio::sync::mpsc;

use crate::device::DiscoveryEvent;
use crate::error::Result;

pub trait Discovery: Send + Sync {
    const PROTOCOL: &'static str;

    fn start(
        &mut self,
        tx: mpsc::Sender<DiscoveryEvent>,
    ) -> impl Future<Output = Result<()>> + Send;

    fn stop(&mut self) -> impl Future<Output = Result<()>> + Send;

    fn is_running(&self) -> bool;
}

pub trait MdnsDiscovery: Discovery {
    const SERVICE_TYPE: &'static str;
}
