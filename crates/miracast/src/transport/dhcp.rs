//! In-process DHCPv4 client for the P2P group interface.
//!
//! When `wpa_supplicant` fires `GroupStarted` with `role=Client`, the
//! kernel iface (`wlp*`, `p2p-wlp*-N`) comes up with no IP. Without
//! an address the phone's DHCP OFFER goes nowhere and RTSP:7236 is
//! unreachable → Miracast session never starts.
//!
//! This module speaks the DHCPv4 DISCOVER/OFFER/REQUEST/ACK exchange
//! on a raw UDP socket bound to the group iface via `SO_BINDTODEVICE`
//! and installs the returned IP on the iface through `rtnetlink`. No
//! shell-out, no external daemon. Mirrors what MiracleCast's
//! `miracle-dhcp --client` does (`src/dhcp/client.c`) but inside our
//! Tokio runtime.

use std::io;
use std::net::{Ipv4Addr, SocketAddrV4};
use std::time::Duration;

use dhcproto::v4::{DhcpOption, Flags, Message, MessageType, Opcode, OptionCode};
use dhcproto::{Decodable, Decoder, Encodable, Encoder};
use socket2::{Domain, Protocol, Socket, Type};
use thiserror::Error;

/// Result of a successful DHCPv4 lease.
#[derive(Debug, Clone)]
pub struct Lease {
    pub addr: Ipv4Addr,
    pub mask: Ipv4Addr,
    pub router: Option<Ipv4Addr>,
}

#[derive(Debug, Error)]
pub enum DhcpError {
    #[error("I/O: {0}")]
    Io(#[from] io::Error),
    #[error("encode: {0}")]
    Encode(String),
    #[error("decode: {0}")]
    Decode(String),
    #[error("timeout waiting for {phase}")]
    Timeout { phase: &'static str },
    #[error("peer NAK'd our REQUEST")]
    Nak,
    #[error("rtnetlink: {0}")]
    Netlink(String),
    #[error("could not read MAC for {iface}: {source}")]
    NoMac {
        iface: String,
        #[source]
        source: io::Error,
    },
}

/// Read the 6-byte hardware address of `iface` from sysfs. We avoid
/// pulling a whole netlink round-trip just for this — the sysfs path
/// is stable across every Linux distro.
fn read_mac(iface: &str) -> Result<[u8; 6], DhcpError> {
    let path = format!("/sys/class/net/{iface}/address");
    let raw = std::fs::read_to_string(&path).map_err(|source| DhcpError::NoMac {
        iface: iface.to_string(),
        source,
    })?;
    let mut mac = [0u8; 6];
    for (i, part) in raw.trim().split(':').enumerate() {
        if i >= 6 {
            break;
        }
        mac[i] = u8::from_str_radix(part, 16).map_err(|_| DhcpError::NoMac {
            iface: iface.to_string(),
            source: io::Error::new(io::ErrorKind::InvalidData, "bad MAC"),
        })?;
    }
    Ok(mac)
}

/// Build an outbound DHCP message with the common options we always
/// want (vendor id, parameter request list).
fn base_message(mac: [u8; 6], xid: u32, msg_type: MessageType) -> Message {
    let mut m = Message::default();
    m.set_flags(Flags::default().set_broadcast())
        .set_chaddr(&mac)
        .set_opcode(Opcode::BootRequest)
        .set_xid(xid);
    m.opts_mut().insert(DhcpOption::MessageType(msg_type));
    m.opts_mut()
        .insert(DhcpOption::ClientIdentifier({
            let mut cid = Vec::with_capacity(7);
            cid.push(1); // hardware type: ethernet
            cid.extend_from_slice(&mac);
            cid
        }));
    m.opts_mut()
        .insert(DhcpOption::ParameterRequestList(vec![
            OptionCode::SubnetMask,
            OptionCode::Router,
            OptionCode::DomainNameServer,
            OptionCode::DomainName,
        ]));
    m
}

fn encode(msg: &Message) -> Result<Vec<u8>, DhcpError> {
    let mut buf = Vec::with_capacity(576);
    msg.encode(&mut Encoder::new(&mut buf))
        .map_err(|e| DhcpError::Encode(e.to_string()))?;
    Ok(buf)
}

fn decode(buf: &[u8]) -> Result<Message, DhcpError> {
    Message::decode(&mut Decoder::new(buf)).map_err(|e| DhcpError::Decode(e.to_string()))
}

/// Bring the iface up + run DHCP + install the lease on it. Short
/// timeout — the phone's DHCP server is a few milliseconds away.
pub async fn acquire_on(iface: &str) -> Result<Lease, DhcpError> {
    bring_iface_up(iface).await?;
    let mac = read_mac(iface)?;

    let sock = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
    sock.set_reuse_address(true)?;
    sock.set_broadcast(true)?;
    sock.bind_device(Some(iface.as_bytes()))?;
    let bind: SocketAddrV4 = SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 68);
    sock.bind(&bind.into())?;
    sock.set_nonblocking(true)?;

    let std_sock = std::net::UdpSocket::from(sock);
    let udp = tokio::net::UdpSocket::from_std(std_sock)?;

    let xid: u32 = rand::random();

    // DISCOVER → OFFER
    let discover = base_message(mac, xid, MessageType::Discover);
    let discover_bytes = encode(&discover)?;
    let bcast: SocketAddrV4 = SocketAddrV4::new(Ipv4Addr::BROADCAST, 67);
    udp.send_to(&discover_bytes, bcast).await?;

    let offer = recv_matching(&udp, xid, MessageType::Offer, "OFFER").await?;
    let offered: Ipv4Addr = offer.yiaddr();
    let server_id = offer
        .opts()
        .iter()
        .find_map(|(_, o)| match o {
            DhcpOption::ServerIdentifier(ip) => Some(*ip),
            _ => None,
        })
        .unwrap_or(Ipv4Addr::UNSPECIFIED);
    tracing::info!(
        iface,
        offered_ip = %offered,
        server = %server_id,
        "miracast dhcp: OFFER received"
    );

    // REQUEST → ACK
    let mut request = base_message(mac, xid, MessageType::Request);
    request
        .opts_mut()
        .insert(DhcpOption::RequestedIpAddress(offered));
    request
        .opts_mut()
        .insert(DhcpOption::ServerIdentifier(server_id));
    let request_bytes = encode(&request)?;
    udp.send_to(&request_bytes, bcast).await?;

    let ack = recv_matching(&udp, xid, MessageType::Ack, "ACK").await?;
    let mask = ack
        .opts()
        .iter()
        .find_map(|(_, o)| match o {
            DhcpOption::SubnetMask(m) => Some(*m),
            _ => None,
        })
        .unwrap_or(Ipv4Addr::new(255, 255, 255, 0));
    let router = ack.opts().iter().find_map(|(_, o)| match o {
        DhcpOption::Router(rs) => rs.first().copied(),
        _ => None,
    });
    tracing::info!(
        iface,
        addr = %offered,
        %mask,
        router = router.map(|r| r.to_string()).unwrap_or_else(|| "-".into()),
        "miracast dhcp: ACK — installing lease"
    );

    install_lease(iface, offered, mask).await?;
    Ok(Lease {
        addr: offered,
        mask,
        router,
    })
}

async fn recv_matching(
    udp: &tokio::net::UdpSocket,
    xid: u32,
    want: MessageType,
    phase: &'static str,
) -> Result<Message, DhcpError> {
    let timeout = Duration::from_secs(10);
    let fut = async {
        let mut buf = vec![0u8; 1500];
        loop {
            let (n, _from) = udp.recv_from(&mut buf).await?;
            let Ok(msg) = decode(&buf[..n]) else {
                continue;
            };
            if msg.xid() != xid {
                continue;
            }
            let got = msg.opts().iter().find_map(|(_, o)| match o {
                DhcpOption::MessageType(t) => Some(*t),
                _ => None,
            });
            match got {
                Some(t) if t == want => return Ok::<_, DhcpError>(msg),
                Some(MessageType::Nak) => return Err(DhcpError::Nak),
                _ => continue,
            }
        }
    };
    tokio::time::timeout(timeout, fut)
        .await
        .map_err(|_| DhcpError::Timeout { phase })?
}

/// `ip link set <iface> up` equivalent via rtnetlink.
async fn bring_iface_up(iface: &str) -> Result<(), DhcpError> {
    let (connection, handle, _) =
        rtnetlink::new_connection().map_err(|e| DhcpError::Netlink(e.to_string()))?;
    tokio::spawn(connection);
    use futures_util::TryStreamExt;
    let mut links = handle.link().get().match_name(iface.to_string()).execute();
    let link = links
        .try_next()
        .await
        .map_err(|e| DhcpError::Netlink(e.to_string()))?
        .ok_or_else(|| DhcpError::Netlink(format!("iface {iface} not found")))?;
    let idx = link.header.index;
    handle
        .link()
        .set(idx)
        .up()
        .execute()
        .await
        .map_err(|e| DhcpError::Netlink(e.to_string()))?;
    Ok(())
}

/// `ip addr add <addr>/<prefix> dev <iface>` equivalent via rtnetlink.
async fn install_lease(iface: &str, addr: Ipv4Addr, mask: Ipv4Addr) -> Result<(), DhcpError> {
    let prefix = u32::from(mask).count_ones() as u8;
    let (connection, handle, _) =
        rtnetlink::new_connection().map_err(|e| DhcpError::Netlink(e.to_string()))?;
    tokio::spawn(connection);
    use futures_util::TryStreamExt;
    let mut links = handle.link().get().match_name(iface.to_string()).execute();
    let link = links
        .try_next()
        .await
        .map_err(|e| DhcpError::Netlink(e.to_string()))?
        .ok_or_else(|| DhcpError::Netlink(format!("iface {iface} not found")))?;
    let idx = link.header.index;
    handle
        .address()
        .add(idx, std::net::IpAddr::V4(addr), prefix)
        .execute()
        .await
        .map_err(|e| DhcpError::Netlink(e.to_string()))?;
    Ok(())
}
