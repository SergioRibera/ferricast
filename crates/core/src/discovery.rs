use std::net::{SocketAddr, SocketAddrV6};

use mdns_sd::{ResolvedService, ScopedIp};
use tokio::sync::mpsc;

use crate::device::{DeviceAddresses, DiscoveryEvent};
use crate::error::Result;
use crate::net::local_ipv4_subnets;

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

impl DeviceAddresses {
    /// Every address `service` resolved to, ranked against this host's
    /// interfaces. Link-local IPv6 addresses carry the index of the
    /// interface the answer arrived on as their scope id.
    ///
    /// Returns `None` when no usable address was resolved.
    #[must_use]
    pub fn from_mdns(service: &ResolvedService) -> Option<Self> {
        let port = service.get_port();
        let addrs = service
            .get_addresses()
            .iter()
            .filter_map(|ip| scoped_socket_addr(ip, port));
        Self::ranked(addrs, &local_ipv4_subnets())
    }
}

/// `ip:port`, scoped to its interface when `ip` is link-local IPv6.
fn scoped_socket_addr(ip: &ScopedIp, port: u16) -> Option<SocketAddr> {
    match ip {
        ScopedIp::V4(v4) => Some(SocketAddr::new((*v4.addr()).into(), port)),
        ScopedIp::V6(v6) => {
            let addr = *v6.addr();
            // A scope id only means something on a link-local address.
            let scope = if addr.is_unicast_link_local() {
                v6.scope_id().index
            } else {
                0
            };
            Some(SocketAddr::V6(SocketAddrV6::new(addr, port, 0, scope)))
        }
        other => {
            tracing::debug!(?other, "ignoring mDNS address of an unknown kind");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::IpAddr;

    #[test]
    fn only_link_local_addresses_keep_a_scope() {
        let v4 = ScopedIp::from(IpAddr::from([10, 0, 0, 5]));
        assert_eq!(
            scoped_socket_addr(&v4, 8009),
            Some("10.0.0.5:8009".parse().unwrap())
        );
        // `From<IpAddr>` leaves the interface unset (index 0), which is
        // what an unscoped answer looks like.
        let global = ScopedIp::from("2001:db8::5".parse::<IpAddr>().unwrap());
        assert_eq!(
            scoped_socket_addr(&global, 8009),
            Some("[2001:db8::5]:8009".parse().unwrap())
        );
    }
}
