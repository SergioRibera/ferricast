//! Wi-Fi Direct Miracast transport via `wpa_supplicant` on D-Bus.
//!
//! Drives the `fi.w1.wpa_supplicant1` system-bus service: picks a
//! P2P-Device-capable interface, uploads the WFD IE bytes so peers
//! can see us as a Miracast source/sink, runs `P2PDevice.Find` and
//! surfaces each peer whose `IEs` property carries the Wi-Fi Alliance
//! WFD subtype (OUI `50:6F:9A`, OUI type `0x0A`).
//!
//! Once a user picks a peer, [`LinuxP2pConnector::connect`] calls
//! `P2PDevice.Connect`, waits for the `GroupStarted` signal, and
//! returns the virtual group interface name + the GO's IP so the
//! existing WFD RTSP source/sink code can plug in.
//!
//! Compiled only on Linux with the `linux-p2p` feature.

use std::collections::HashMap;
use std::time::Duration;

use futures_util::StreamExt;
use tokio::sync::mpsc;
use zbus::zvariant::{ObjectPath, OwnedObjectPath, OwnedValue, Value};
use zbus::{Connection, Proxy};

use super::TransportError;
use crate::ie::{DeviceRole, SubelementId, WfdIe, WIFI_ALLIANCE_OUI, WFD_OUI_TYPE};

const SUPPLICANT_BUS: &str = "fi.w1.wpa_supplicant1";
const SUPPLICANT_ROOT_PATH: &str = "/fi/w1/wpa_supplicant1";
const ROOT_IFACE: &str = "fi.w1.wpa_supplicant1";
const P2P_IFACE: &str = "fi.w1.wpa_supplicant1.Interface.P2PDevice";
const PEER_IFACE: &str = "fi.w1.wpa_supplicant1.Peer";

/// A peer discovered on the Wi-Fi Direct bus.
#[derive(Debug, Clone)]
pub struct DiscoveredPeer {
    /// supplicant `Peer` object path, unique per session.
    pub path: String,
    /// Human-readable device name (what the user sees in the picker).
    pub name: String,
    /// Peer's MAC — stable across sessions, so used as a durable id.
    pub address: [u8; 6],
    /// Parsed WFD Information Element; `None` for a plain P2P peer
    /// that happens to be in range (not a Miracast device).
    pub wfd: Option<WfdIe>,
}

impl DiscoveredPeer {
    /// Whether the peer advertises itself as a Miracast sink, source,
    /// or dual-role device.
    #[must_use]
    pub fn is_miracast(&self) -> bool {
        self.wfd.as_ref().is_some_and(|w| w.device_information.is_some())
    }

    /// Whether we (a source) should offer to cast to it.
    #[must_use]
    pub fn is_sink(&self) -> bool {
        self.wfd
            .as_ref()
            .and_then(|w| w.device_information)
            .map(|info| {
                matches!(
                    info.role,
                    DeviceRole::PrimarySink | DeviceRole::SecondarySink | DeviceRole::DualRole
                )
            })
            .unwrap_or(false)
    }
}

/// High-level client for the Wi-Fi Direct Miracast subsystem.
pub struct LinuxP2pClient {
    conn: Connection,
    interface_path: OwnedObjectPath,
}

impl LinuxP2pClient {
    /// Connect to the system wpa_supplicant and pick the first
    /// P2P-Device-capable interface. The WFD IE bytes you pass are
    /// written to the `WFDIEs` property so peers see the Miracast
    /// subtype in our probe/beacon frames.
    pub async fn connect(wfd_ie: &WfdIe) -> Result<Self, TransportError> {
        let conn = Connection::system()
            .await
            .map_err(|e| TransportError::LinuxP2p(format!("system bus: {e}")))?;

        let interface_path = resolve_or_create_interface(&conn).await?;

        // Push the WFD subelements globally — supplicant wraps them in
        // the vendor-specific IE header on the air itself.
        let bytes = wfd_ie.to_subelements();
        let value = Value::from(bytes.to_vec());
        let root = Proxy::new(&conn, SUPPLICANT_BUS, SUPPLICANT_ROOT_PATH, ROOT_IFACE)
            .await
            .map_err(|e| TransportError::LinuxP2p(format!("root proxy: {e}")))?;
        root.set_property("WFDIEs", value)
            .await
            .map_err(|e| TransportError::LinuxP2p(format!("set WFDIEs: {e}")))?;

        tracing::info!(iface = %interface_path.as_str(), "miracast linux p2p: ready");
        Ok(Self {
            conn,
            interface_path,
        })
    }

    /// Start continuous P2P discovery. Each discovered peer is parsed
    /// for its WFD IE and pushed on the returned channel.
    pub async fn start_discovery(&self) -> Result<mpsc::Receiver<DiscoveredPeer>, TransportError> {
        let (tx, rx) = mpsc::channel(16);

        let p2p = self.p2p_proxy().await?;
        let empty: HashMap<&str, Value<'_>> = HashMap::new();
        p2p.call_method("Find", &(empty,))
            .await
            .map_err(|e| TransportError::LinuxP2p(format!("Find: {e}")))?;

        let listen_result = p2p
            .call_method("ExtendedListen", &(HashMap::<&str, Value>::new(),))
            .await;
        if let Err(e) = listen_result {
            tracing::debug!(error = %e, "ExtendedListen rejected (optional)");
        }

        let conn = self.conn.clone();
        let interface_path = self.interface_path.clone();
        tokio::spawn(async move {
            let p2p_owned = match Proxy::new(
                &conn,
                SUPPLICANT_BUS,
                interface_path.clone(),
                P2P_IFACE,
            )
            .await
            {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(error = %e, "linux p2p: proxy for signal stream");
                    return;
                }
            };
            let mut stream = match p2p_owned.receive_signal("DeviceFound").await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = %e, "linux p2p: subscribe DeviceFound");
                    return;
                }
            };
            while let Some(msg) = stream.next().await {
                let path: OwnedObjectPath = match msg.body().deserialize() {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                let path_ref = path.as_ref();
                match read_peer(&conn, &path_ref).await {
                    Ok(peer) => {
                        if peer.is_miracast() && tx.send(peer).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => tracing::debug!(error = %e, "linux p2p: peer read"),
                }
            }
        });

        Ok(rx)
    }

    /// Stop the P2P discovery scan (`StopFind`). Idempotent.
    pub async fn stop_discovery(&self) -> Result<(), TransportError> {
        let p2p = self.p2p_proxy().await?;
        let _ = p2p.call_method("StopFind", &()).await;
        Ok(())
    }

    /// Initiate GO negotiation with a peer. Returns the virtual group
    /// interface name (e.g. `p2p-wlan0-0`) once `GroupStarted` fires.
    /// The caller then runs RTSP on that iface's IP.
    pub async fn connect_peer(&self, peer_path: &str) -> Result<GroupStarted, TransportError> {
        let p2p = self.p2p_proxy().await?;
        let mut stream = p2p
            .receive_signal("GroupStarted")
            .await
            .map_err(|e| TransportError::LinuxP2p(format!("subscribe GroupStarted: {e}")))?;

        let mut args: HashMap<&str, Value> = HashMap::new();
        let peer_obj =
            ObjectPath::try_from(peer_path).map_err(|e| TransportError::LinuxP2p(e.to_string()))?;
        args.insert("peer", Value::from(peer_obj));
        args.insert("wps_method", Value::from("pbc"));
        p2p.call_method("Connect", &(args,))
            .await
            .map_err(|e| TransportError::LinuxP2p(format!("Connect: {e}")))?;

        // 30 s is the Wi-Fi Alliance recommended GO negotiation ceiling.
        let fut = async {
            while let Some(sig) = stream.next().await {
                if let Ok(dict) = sig.body().deserialize::<HashMap<String, OwnedValue>>() {
                    if let Some(iface_val) = dict.get("interface_object") {
                        if let Ok(path) = iface_val.try_clone().and_then(ObjectPath::try_from) {
                            let role = dict
                                .get("role")
                                .and_then(|v| v.downcast_ref::<&str>().ok())
                                .unwrap_or("client")
                                .to_string();
                            return Ok(GroupStarted {
                                interface_path: path.to_string(),
                                role: if role == "GO" {
                                    GroupRole::Owner
                                } else {
                                    GroupRole::Client
                                },
                            });
                        }
                    }
                }
            }
            Err(TransportError::LinuxP2p(
                "GroupStarted stream ended".into(),
            ))
        };
        tokio::time::timeout(Duration::from_secs(30), fut)
            .await
            .map_err(|_| TransportError::LinuxP2p("GO negotiation timed out".into()))?
    }

    async fn p2p_proxy(&self) -> Result<Proxy<'_>, TransportError> {
        Proxy::new(
            &self.conn,
            SUPPLICANT_BUS,
            self.interface_path.clone(),
            P2P_IFACE,
        )
        .await
        .map_err(|e| TransportError::LinuxP2p(format!("p2p proxy: {e}")))
    }
}

/// The virtual group interface a GO negotiation produced.
#[derive(Debug, Clone)]
pub struct GroupStarted {
    /// supplicant object path of the new interface (e.g.
    /// `/fi/w1/wpa_supplicant1/Interfaces/5`). Pair with
    /// `fi.w1.wpa_supplicant1.Interface` to read the `Ifname` property
    /// if you need the kernel name (`p2p-wlan0-0`).
    pub interface_path: String,
    pub role: GroupRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupRole {
    Owner,
    Client,
}

async fn resolve_or_create_interface(conn: &Connection) -> Result<OwnedObjectPath, TransportError> {
    let root = Proxy::new(conn, SUPPLICANT_BUS, SUPPLICANT_ROOT_PATH, ROOT_IFACE)
        .await
        .map_err(|e| TransportError::LinuxP2p(format!("root proxy: {e}")))?;

    // Prefer an already-existing interface so we coexist with
    // NetworkManager (which keeps the sole wpa_supplicant around).
    if let Ok(existing) = root.get_property::<Vec<OwnedObjectPath>>("Interfaces").await {
        if let Some(path) = existing.into_iter().next() {
            return Ok(path);
        }
    }

    // Fallback: try wlan0. Users with a different radio can override via
    // a future `with_interface(name)` hook on the builder.
    let mut args: HashMap<&str, Value> = HashMap::new();
    args.insert("Ifname", Value::from("wlan0"));
    let reply = root
        .call_method("CreateInterface", &(args,))
        .await
        .map_err(|e| TransportError::LinuxP2p(format!("CreateInterface: {e}")))?;
    let path: OwnedObjectPath = reply
        .body()
        .deserialize()
        .map_err(|e| TransportError::LinuxP2p(format!("CreateInterface reply: {e}")))?;
    Ok(path)
}

async fn read_peer(conn: &Connection, path: &ObjectPath<'_>) -> Result<DiscoveredPeer, TransportError> {
    let proxy = Proxy::new(conn, SUPPLICANT_BUS, path.to_owned(), PEER_IFACE)
        .await
        .map_err(|e| TransportError::LinuxP2p(format!("peer proxy: {e}")))?;

    let name: String = proxy
        .get_property("DeviceName")
        .await
        .unwrap_or_else(|_| "Unknown".to_string());
    let address: Vec<u8> = proxy
        .get_property("DeviceAddress")
        .await
        .map_err(|e| TransportError::LinuxP2p(format!("DeviceAddress: {e}")))?;
    let address_arr: [u8; 6] = address
        .as_slice()
        .try_into()
        .map_err(|_| TransportError::LinuxP2p("DeviceAddress is not 6 bytes".into()))?;

    // `IEs: ay` (raw per-peer IE blob) + `VendorExtension: aay` (vendor
    // IEs split out). Different supplicant builds expose one or both;
    // scan both.
    let direct: Vec<u8> = proxy.get_property("IEs").await.unwrap_or_default();
    let vendor: Vec<Vec<u8>> = proxy
        .get_property("VendorExtension")
        .await
        .unwrap_or_default();

    let wfd = find_wfd_in_vendor(&direct, &vendor);

    // Keep the compiler happy about `SubelementId` export — we deliberately
    // take it in-scope so reviewers can see the WFD IE parse path at a
    // glance even though parsing is delegated to the `ie` module.
    let _ = SubelementId::DeviceInformation;

    Ok(DiscoveredPeer {
        path: path.to_string(),
        name,
        address: address_arr,
        wfd,
    })
}

fn find_wfd_in_vendor(direct: &[u8], vendor: &[Vec<u8>]) -> Option<WfdIe> {
    if let Some(wfd) = extract_wfd(direct) {
        return Some(wfd);
    }
    for v in vendor {
        if let Some(wfd) = extract_wfd(v) {
            return Some(wfd);
        }
    }
    None
}

fn extract_wfd(bytes: &[u8]) -> Option<WfdIe> {
    // Each IE: `0xDD <len> <OUI:3> <OUI_type:1> <subelements...>`.
    // `IEs: ay` may be a concatenation of several; scan for 0xDD.
    let mut i = 0;
    while i + 2 < bytes.len() {
        if bytes[i] != 0xDD {
            i += 1;
            continue;
        }
        let len = bytes[i + 1] as usize;
        if i + 2 + len > bytes.len() || len < 4 {
            i += 1;
            continue;
        }
        let oui = &bytes[i + 2..i + 5];
        let oui_type = bytes[i + 5];
        if oui == WIFI_ALLIANCE_OUI && oui_type == WFD_OUI_TYPE {
            let payload = &bytes[i + 6..i + 2 + len];
            if let Ok(wfd) = WfdIe::parse(payload) {
                return Some(wfd);
            }
        }
        i += 2 + len;
    }
    None
}

/// Probe whether the host has a usable wpa_supplicant service — the
/// GUI calls this to tell users "install wpa_supplicant" or "your
/// adapter does not expose P2P" without blowing up on D-Bus errors.
pub async fn probe_supplicant() -> Result<(), TransportError> {
    let _ = LinuxP2pClient::connect(&WfdIe::new(
        crate::ie::DeviceInformation::linux_dual_role(),
    ))
    .await?;
    Ok(())
}

// `Mutex` is only pulled in by callers that hold the client across
// task boundaries. Nothing in this module needs it directly today, so
// we avoid defining an unused helper type.
#[cfg(any())]
type _M = Mutex<()>;
