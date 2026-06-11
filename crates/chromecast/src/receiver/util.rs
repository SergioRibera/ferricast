//! Cross-module helpers for the receiver side.
//!
//! Right now this just holds the LAN-IPv4 detection logic — it's
//! shared between the mDNS advertiser (pins announces to one
//! interface) and the TLS server (embeds the same IP in the cert
//! SAN list so strict-validating clients accept the connection).

use std::net::IpAddr;

/// Same env-var the advertiser checks. Duplicated here so callers
/// don't need to depend on the advertiser module just to read it.
pub const ADVERTISE_INTERFACE_VAR: &str = "FERRICAST_CAST_INTERFACE";

/// Resolve the IPs we'll be reachable on, in priority order.
///
/// The list goes into the TLS cert's Subject Alternative Name field,
/// so every host a sender might dial us as needs to be in it.
/// Strict TLS clients (VLC mobile's GnuTLS) reject the connection
/// when the dial host (`192.168.x.x` from mDNS) doesn't match any
/// SAN, with a generic "Interrupted system call" log that gives no
/// hint about hostname validation being the cause.
///
/// Resolution strategy:
/// 1. `FERRICAST_CAST_INTERFACE=<ip>` — if set and parses as IPv4,
///    use it verbatim. This handles the operator-override case where
///    `pin_lan_interface` would normally pick wrong (VPN holds the
///    default route, multi-NIC, etc.).
/// 2. UDP `connect("8.8.8.8:53")` + `local_addr()` — kernel picks the
///    source IP it would use for off-link traffic. No packets get
///    sent. Same trick the advertiser uses.
/// 3. Loopback (`127.0.0.1`) — always included so localhost-side
///    tooling (the HLS server's own self-test, browser players
///    pointed at the URL on the same host) keeps working.
pub fn detect_advertise_ips() -> Vec<IpAddr> {
    let mut ips: Vec<IpAddr> = Vec::new();

    if let Ok(val) = std::env::var(ADVERTISE_INTERFACE_VAR) {
        let trimmed = val.trim();
        if let Ok(ip) = trimmed.parse::<IpAddr>() {
            ips.push(ip);
        }
    }

    if let Some(ip) = detect_default_route_ipv4() {
        if !ips.contains(&ip) {
            ips.push(ip);
        }
    }

    // Loopback gets appended last as a fallback — strict TLS clients
    // dialling `127.0.0.1` for the HLS self-test still need to match
    // a SAN.
    let loopback: IpAddr = "127.0.0.1".parse().unwrap();
    if !ips.contains(&loopback) {
        ips.push(loopback);
    }

    ips
}

pub(super) fn detect_default_route_ipv4() -> Option<IpAddr> {
    let sock = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect("8.8.8.8:53").ok()?;
    let addr = sock.local_addr().ok()?.ip();
    if addr.is_unspecified() || addr.is_loopback() || !addr.is_ipv4() {
        None
    } else {
        Some(addr)
    }
}
