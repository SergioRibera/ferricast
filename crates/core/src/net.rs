//! Networking helpers shared across receiver protocol crates.
//!
//! Receiver protocols (Chromecast HLS, AirPlay HTTP, DIAL, Miracast)
//! all need to advertise a URL that the device on the LAN can pull
//! from. That URL needs the local IP **the device can route back
//! to**, not just any random interface address — Docker bridges,
//! Tailscale, libvirt, podman and friends litter the host with
//! addresses that won't resolve from an external device.

use std::net::{IpAddr, SocketAddr, UdpSocket};

use crate::error::{FerricastError, Result};

/// The local IP address the kernel would use to reach `target`.
///
/// Doesn't actually transmit any bytes — opens a UDP socket and
/// "connects" it (a pure kernel routing operation that picks the
/// outbound interface and source IP without sending anything), then
/// reads `local_addr`. This is more reliable than enumerating
/// interfaces and guessing because the kernel already knows the
/// answer from its routing table, and that's the only authority
/// that matters: any IP we'd hand to the receiver has to be one the
/// receiver's reply traffic will actually arrive on.
///
/// The `target` doesn't have to be reachable — UDP `connect()` is
/// purely local. If the routing table has a default route to the
/// general direction of `target`, that route's source IP is what
/// you get back.
///
/// Used by:
/// * `ferricast-chromecast` for the HLS playlist URL passed to
///   the Default Media Receiver via `LOAD`.
/// * (future) `ferricast-airplay` for the RTSP `Transport:` URL.
/// * (future) `ferricast-dial` for the app-specific `Origin:` header.
pub fn local_addr_for(target: IpAddr) -> Result<IpAddr> {
    let bind: SocketAddr = match target {
        IpAddr::V4(_) => SocketAddr::from(([0, 0, 0, 0], 0)),
        IpAddr::V6(_) => SocketAddr::from(([0u16; 8], 0)),
    };
    let s = UdpSocket::bind(bind).map_err(|e| {
        FerricastError::Other(format!("local_addr_for: bind probe socket: {e}"))
    })?;
    // Port doesn't matter — `connect` on UDP is purely a routing
    // hint, no SYN goes out. Use 1 (root-only on most systems but
    // we never actually send so we don't need to bind it).
    s.connect(SocketAddr::new(target, 1)).map_err(|e| {
        FerricastError::Other(format!("local_addr_for: connect probe socket: {e}"))
    })?;
    s.local_addr()
        .map(|a| a.ip())
        .map_err(|e| FerricastError::Other(format!("local_addr_for: local_addr: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn returns_a_v4_for_a_v4_target() {
        // 8.8.8.8 has a route on every network-connected machine
        // (default gateway) — the test just checks we get *some*
        // IPv4 back, not what it is.
        let ip = local_addr_for(IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8))).unwrap();
        assert!(matches!(ip, IpAddr::V4(_)), "expected v4, got {ip}");
        assert!(!ip.is_unspecified(), "shouldn't be 0.0.0.0");
    }

    #[test]
    fn rfc1918_target_returns_routable_local() {
        // Asking for the source-IP toward 192.168.x.x should give
        // the LAN-side address (or fall back to default route's
        // source if no LAN). Either way: not 0.0.0.0, not loopback
        // unless the only route is loopback.
        let ip = local_addr_for(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1))).unwrap();
        assert!(!ip.is_unspecified(), "shouldn't be 0.0.0.0, got {ip}");
    }
}
