//! Networking helpers shared across receiver protocol crates.
//!
//! Receiver protocols (Chromecast HLS, AirPlay HTTP, DIAL, Miracast)
//! all need to advertise a URL that the device on the LAN can pull
//! from. That URL needs the local IP **the device can route back
//! to**, not just any random interface address — Docker bridges,
//! Tailscale, libvirt, podman and friends litter the host with
//! addresses that won't resolve from an external device.
//!
//! Listeners and sockets follow the address family of the device they
//! serve: an IPv4 device gets a socket on `0.0.0.0`, an IPv6 one a
//! dual-stack socket on `[::]` (so loopback self-tests over
//! `127.0.0.1` keep working).

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV6, UdpSocket};
use std::time::Duration;

use socket2::{Domain, Protocol, Socket, Type};
use tokio::net::{TcpListener, TcpStream};

use crate::error::{FerricastError, Result};

/// Backlog for listeners bound through [`bind_tcp`]; matches what
/// `std`/tokio use for `TcpListener::bind`.
const LISTEN_BACKLOG: i32 = 1024;

/// The local IP address the kernel would use to reach `target`.
///
/// Convenience over [`local_socket_addr_for`] for callers that only
/// have an `IpAddr`. IPv6 link-local targets need a zone, which an
/// `IpAddr` cannot carry, so they fail here; use
/// [`local_socket_addr_for`] with a scoped `SocketAddrV6` instead.
///
/// # Errors
///
/// See [`local_socket_addr_for`].
pub fn local_addr_for(target: IpAddr) -> Result<IpAddr> {
    local_socket_addr_for(SocketAddr::new(target, 0)).map(|a| a.ip())
}

/// The local address the kernel would use to reach `target`, with the
/// IPv6 scope id preserved.
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
/// purely local. Only its IP (and, for IPv6, scope id) is used; the
/// port is ignored and the returned port is meaningless.
///
/// For a link-local IPv6 target the returned address is link-local
/// too and carries the target's scope id. Such an address can't go
/// into an HTTP URL for a receiver (browsers reject zone ids), so
/// URL builders should prefer a routable address when one exists.
///
/// # Errors
///
/// [`FerricastError::Connection`] when `target` is an IPv6 link-local
/// address without a scope id (the kernel can't pick an interface),
/// or [`FerricastError::Other`] when the kernel has no route to it.
pub fn local_socket_addr_for(target: SocketAddr) -> Result<SocketAddr> {
    if let SocketAddr::V6(v6) = target
        && needs_zone(v6)
    {
        return Err(FerricastError::Connection(format!(
            "IPv6 link-local address {} has no interface scope id",
            v6.ip()
        )));
    }
    let s = UdpSocket::bind(SocketAddr::new(unspecified_for(target.ip()), 0))
        .map_err(|e| FerricastError::Other(format!("local_addr_for: bind probe socket: {e}")))?;
    // Port doesn't matter — `connect` on UDP is purely a routing
    // hint, no packet goes out.
    let mut probe = target;
    probe.set_port(1);
    s.connect(probe)
        .map_err(|e| FerricastError::Other(format!("local_addr_for: connect probe socket: {e}")))?;
    s.local_addr()
        .map_err(|e| FerricastError::Other(format!("local_addr_for: local_addr: {e}")))
}

/// The unspecified address (`0.0.0.0` or `::`) of `peer`'s family.
#[must_use]
pub fn unspecified_for(peer: IpAddr) -> IpAddr {
    match peer {
        IpAddr::V4(_) => IpAddr::V4(Ipv4Addr::UNSPECIFIED),
        IpAddr::V6(_) => IpAddr::V6(Ipv6Addr::UNSPECIFIED),
    }
}

/// Whether `addr` is an IPv6 link-local address with no scope id.
fn needs_zone(addr: SocketAddrV6) -> bool {
    addr.ip().is_unicast_link_local() && addr.scope_id() == 0
}

/// Bind a `TcpListener` that `peer` can reach, on the first free port
/// in `port_range` or on an ephemeral port when it is `None`.
///
/// The listener is bound on every interface of `peer`'s family; for
/// IPv6 it is dual-stack (`IPV6_V6ONLY` off) so IPv4 clients and
/// loopback still reach it. Returns the bound listener so the caller
/// hands it directly to the server without a TOCTOU re-bind race.
///
/// # Errors
///
/// [`FerricastError::Other`] when no port in the range is free, or the
/// socket error for an ephemeral bind.
pub async fn bind_tcp(peer: IpAddr, port_range: Option<(u16, u16)>) -> Result<TcpListener> {
    let ip = unspecified_for(peer);
    let Some((start, end)) = port_range else {
        return Ok(tcp_listener(SocketAddr::new(ip, 0))?);
    };
    for port in start..=end {
        if let Ok(l) = tcp_listener(SocketAddr::new(ip, port)) {
            return Ok(l);
        }
    }
    Err(FerricastError::Other(format!(
        "no free TCP port in range {start}–{end}"
    )))
}

/// Bind a tokio `UdpSocket` that can talk to `peer`, on the first free
/// port in `port_range` or on an ephemeral port when it is `None`.
/// UDP counterpart of [`bind_tcp`] (Cast Streaming RTP/RTCP, AirPlay
/// timing/control, …).
///
/// # Errors
///
/// [`FerricastError::Other`] when no port in the range is free, or the
/// socket error for an ephemeral bind.
pub async fn bind_udp(
    peer: IpAddr,
    port_range: Option<(u16, u16)>,
) -> Result<tokio::net::UdpSocket> {
    let ip = unspecified_for(peer);
    let Some((start, end)) = port_range else {
        return Ok(udp_socket(SocketAddr::new(ip, 0))?);
    };
    for port in start..=end {
        if let Ok(s) = udp_socket(SocketAddr::new(ip, port)) {
            return Ok(s);
        }
    }
    Err(FerricastError::Other(format!(
        "no free UDP port in range {start}–{end}"
    )))
}

/// Check that `target` accepts TCP connections, giving up after
/// `timeout`. The connection is closed straight away; nothing is sent.
///
/// Used before a stream starts so an offline or firewalled receiver is
/// reported in about a second instead of after the full handshake
/// timeout.
///
/// # Errors
///
/// [`FerricastError::Connection`] (`device unreachable: <addr>:<port>`)
/// when the connection is refused, has no route or does not complete
/// within `timeout`.
pub async fn probe_tcp(target: SocketAddr, timeout: Duration) -> Result<()> {
    let reason = match tokio::time::timeout(timeout, TcpStream::connect(target)).await {
        Ok(Ok(_stream)) => return Ok(()),
        Ok(Err(e)) => e.to_string(),
        Err(_) => format!("no answer within {} ms", timeout.as_millis()),
    };
    Err(FerricastError::Connection(format!(
        "device unreachable: {target} ({reason})"
    )))
}

/// A non-blocking socket of `addr`'s family, dual-stack for IPv6.
fn socket_for(addr: SocketAddr, ty: Type, proto: Protocol) -> std::io::Result<Socket> {
    let socket = Socket::new(Domain::for_address(addr), ty, Some(proto))?;
    if addr.is_ipv6() {
        socket.set_only_v6(false)?;
    }
    socket.set_nonblocking(true)?;
    Ok(socket)
}

fn tcp_listener(addr: SocketAddr) -> std::io::Result<TcpListener> {
    let socket = socket_for(addr, Type::STREAM, Protocol::TCP)?;
    // Same as std's listener: lets a restarted session rebind a port
    // still in TIME_WAIT from the previous one.
    socket.set_reuse_address(true)?;
    socket.bind(&addr.into())?;
    socket.listen(LISTEN_BACKLOG)?;
    TcpListener::from_std(socket.into())
}

fn udp_socket(addr: SocketAddr) -> std::io::Result<tokio::net::UdpSocket> {
    let socket = socket_for(addr, Type::DGRAM, Protocol::UDP)?;
    socket.bind(&addr.into())?;
    tokio::net::UdpSocket::from_std(socket.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    const V4_LOOPBACK: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
    const V6_LOOPBACK: IpAddr = IpAddr::V6(Ipv6Addr::LOCALHOST);

    /// Whether the host has IPv6 at all (sandboxes sometimes don't).
    fn ipv6_available() -> bool {
        std::net::UdpSocket::bind((Ipv6Addr::LOCALHOST, 0)).is_ok()
    }

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
        let ip = local_addr_for(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1))).unwrap();
        assert!(!ip.is_unspecified(), "shouldn't be 0.0.0.0, got {ip}");
    }

    #[test]
    fn loopback_targets_resolve_to_loopback_of_the_same_family() {
        assert_eq!(local_addr_for(V4_LOOPBACK).unwrap(), V4_LOOPBACK);
        if ipv6_available() {
            assert_eq!(local_addr_for(V6_LOOPBACK).unwrap(), V6_LOOPBACK);
        }
    }

    #[test]
    fn link_local_without_scope_is_rejected_with_a_clear_error() {
        let target: SocketAddr = "[fe80::1]:8009".parse().unwrap();
        let err = local_socket_addr_for(target).unwrap_err().to_string();
        assert!(err.contains("scope id"), "{err}");
        assert!(local_addr_for(target.ip()).is_err());
    }

    #[test]
    fn scope_id_is_preserved_for_link_local_targets() {
        if !ipv6_available() {
            return;
        }
        // Loopback is interface 1 on Linux but carries no link-local
        // address, so the routing probe may fail there; the check is
        // that a scoped target is accepted and the zone survives.
        let target = SocketAddr::V6(SocketAddrV6::new("fe80::1".parse().unwrap(), 9, 0, 1));
        if let Ok(SocketAddr::V6(local)) = local_socket_addr_for(target) {
            assert_eq!(local.scope_id(), 1);
        }
    }

    #[test]
    fn unspecified_follows_the_peer_family() {
        assert_eq!(
            unspecified_for(V4_LOOPBACK),
            IpAddr::V4(Ipv4Addr::UNSPECIFIED)
        );
        assert_eq!(
            unspecified_for(V6_LOOPBACK),
            IpAddr::V6(Ipv6Addr::UNSPECIFIED)
        );
    }

    #[tokio::test]
    async fn v4_peer_gets_a_v4_listener() {
        let l = bind_tcp(V4_LOOPBACK, None).await.unwrap();
        assert!(l.local_addr().unwrap().is_ipv4());
        let u = bind_udp(V4_LOOPBACK, None).await.unwrap();
        assert!(u.local_addr().unwrap().is_ipv4());
    }

    #[tokio::test]
    async fn v6_peer_gets_a_dual_stack_listener() {
        if !ipv6_available() {
            return;
        }
        let l = bind_tcp(V6_LOOPBACK, None).await.unwrap();
        let port = l.local_addr().unwrap().port();
        assert!(l.local_addr().unwrap().is_ipv6());
        // Dual-stack: both loopbacks reach the same listener.
        for ip in [V6_LOOPBACK, V4_LOOPBACK] {
            let connect = tokio::net::TcpStream::connect((ip, port));
            let (client, accepted) = tokio::join!(connect, l.accept());
            client.unwrap();
            accepted.unwrap();
        }
        let u = bind_udp(V6_LOOPBACK, None).await.unwrap();
        assert!(u.local_addr().unwrap().is_ipv6());
    }

    #[tokio::test]
    async fn ranged_bind_skips_taken_ports() {
        let taken = bind_tcp(V4_LOOPBACK, None).await.unwrap();
        let port = taken.local_addr().unwrap().port();
        let Some(next) = port.checked_add(1) else {
            return;
        };
        match bind_tcp(V4_LOOPBACK, Some((port, next))).await {
            Ok(l) => assert_eq!(l.local_addr().unwrap().port(), next),
            // `next` happened to be in use by something else.
            Err(e) => assert!(e.to_string().contains("no free TCP port"), "{e}"),
        }
    }

    #[tokio::test]
    async fn probe_passes_for_an_open_listener() {
        let l = bind_tcp(V4_LOOPBACK, None).await.unwrap();
        let target = SocketAddr::new(V4_LOOPBACK, l.local_addr().unwrap().port());
        probe_tcp(target, Duration::from_secs(3)).await.unwrap();
    }

    #[tokio::test]
    async fn probe_fails_fast_for_a_closed_port() {
        // Bind then drop to get a port nothing listens on.
        let port = bind_tcp(V4_LOOPBACK, None)
            .await
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let target = SocketAddr::new(V4_LOOPBACK, port);
        let started = std::time::Instant::now();
        let err = probe_tcp(target, Duration::from_secs(3))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "refusal took {:?}",
            started.elapsed()
        );
        assert!(
            err.contains(&format!("device unreachable: 127.0.0.1:{port}")),
            "{err}"
        );
    }
}
