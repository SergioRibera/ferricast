use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr};
use std::time::Duration;

use bytes::Bytes;
use futures_lite::StreamExt;
use tokio::{sync::mpsc, task::JoinHandle};
use zbus::{Connection, zvariant::OwnedObjectPath};

use ferricast_core::{Device, DeviceCapabilities, Discovery, DiscoveryEvent, FerricastError, Result};
use uuid::Uuid;

use crate::dbus::{
    DeviceProxy, IWD_P2P_DEVICE_IFACE, IWD_P2P_PEER_IFACE,
    IwdObjectManagerProxy, IwdP2pDeviceProxy, NM_DEVICE_TYPE_WIFI_P2P, NetworkManagerProxy,
    P2pPeerProxy, WifiP2pProxy, WpaInterfaceProxy, WpaSupplicantProxy,
};
use crate::session::wfd_source_ies;

const MIRACAST_ICON: Bytes = Bytes::from_static(include_bytes!("../../../assets/miracast.svg"));

// ── backend discriminator ─────────────────────────────────────────────────────

/// Which D-Bus backend is backing the current discovery session.
/// Stored so `stop()` can call the right cleanup method.
enum ActiveBackend {
    Nm {
        conn: Connection,
        device_path: OwnedObjectPath,
    },
    Iwd {
        conn: Connection,
        device_path: OwnedObjectPath,
    },
}

// ── public type ───────────────────────────────────────────────────────────────

pub struct MiracastDiscovery {
    handle: Option<JoinHandle<()>>,
    backend: Option<ActiveBackend>,
    running: bool,
}

impl Default for MiracastDiscovery {
    fn default() -> Self {
        Self {
            handle: None,
            backend: None,
            running: false,
        }
    }
}

// ── Discovery impl ────────────────────────────────────────────────────────────

impl Discovery for MiracastDiscovery {
    const PROTOCOL: &'static str = "miracast";

    async fn start(&mut self, tx: mpsc::Sender<DiscoveryEvent>) -> Result<()> {
        let conn = Connection::system().await.map_err(|e| {
            FerricastError::Discovery(format!("D-Bus system bus unavailable: {e}"))
        })?;

        // Try iwd first (the user's backend), then fall back to NM/wpa_supplicant.
        if let Some(device_path) = find_iwd_p2p_device(&conn).await {
            self.start_iwd(conn, device_path, tx).await
        } else if let Some(device_path) = find_nm_p2p_device(&conn).await? {
            self.start_nm(conn, device_path, tx).await
        } else {
            Err(FerricastError::Discovery(
                "No Wi-Fi P2P device found via iwd or NetworkManager. \
                 Check your adapter, drivers, and that iwd or NetworkManager is running."
                    .into(),
            ))
        }
    }

    async fn stop(&mut self) -> Result<()> {
        if !self.running {
            return Ok(());
        }

        match self.backend.take() {
            Some(ActiveBackend::Nm { conn, device_path }) => {
                let p2p = WifiP2pProxy::new(&conn, device_path).await.map_err(|e| {
                    FerricastError::Discovery(format!("WifiP2P proxy for stop: {e}"))
                })?;
                p2p.stop_find().await.map_err(|e| {
                    FerricastError::Discovery(format!("NM WifiP2P.StopFind: {e}"))
                })?;
            }
            Some(ActiveBackend::Iwd { conn, device_path }) => {
                let dev = IwdP2pDeviceProxy::new(&conn, device_path)
                    .await
                    .map_err(|e| {
                        FerricastError::Discovery(format!("iwd P2P device proxy for stop: {e}"))
                    })?;
                dev.release_discovery().await.map_err(|e| {
                    FerricastError::Discovery(format!("iwd ReleaseDiscovery: {e}"))
                })?;
            }
            None => {}
        }

        if let Some(h) = self.handle.take() {
            h.abort();
        }

        self.running = false;
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running
    }
}

// ── backend startup ───────────────────────────────────────────────────────────

impl MiracastDiscovery {
    async fn start_nm(
        &mut self,
        conn: Connection,
        device_path: OwnedObjectPath,
        tx: mpsc::Sender<DiscoveryEvent>,
    ) -> Result<()> {
        let p2p = WifiP2pProxy::new(&conn, device_path.clone())
            .await
            .map_err(|e| FerricastError::Discovery(format!("WifiP2P proxy: {e}")))?;

        // Set WFD source IEs so P2P probe responses identify our device as a
        // WFD source.  TVs in screen-mirror mode filter on these before
        // accepting GO negotiation — without them the TV ignores our connect
        // attempt and we time out at 45-46 s.
        //
        // Two paths depending on what's actually driving the Wi-Fi radio:
        //   wpa_supplicant backend: set WFDIEs on the wpa_supplicant interface
        //   iwd backend under NM:   set WFDElements on the iwd P2P device
        if let Ok(dev) = DeviceProxy::new(&conn, device_path.clone()).await {
            if let Ok(ifname) = dev.interface().await {
                set_wpa_supplicant_wfd_ies(&conn, &ifname).await;
            }
        }
        if let Some(iwd_path) = find_iwd_p2p_device(&conn).await {
            if let Ok(iwd_dev) = IwdP2pDeviceProxy::new(&conn, iwd_path).await {
                if let Err(e) = iwd_dev.set_WFDElements(wfd_source_ies()).await {
                    tracing::debug!("iwd WFDElements via NM path not settable (non-fatal): {e}");
                } else {
                    tracing::debug!("WFD source IEs set on iwd device (NM discovery path)");
                }
            }
        }

        p2p.start_find(HashMap::new()).await.map_err(|e| {
            FerricastError::Discovery(format!("NM WifiP2P.StartFind: {e}"))
        })?;

        let mut peer_added = p2p.receive_peer_added().await.map_err(|e| {
            FerricastError::Discovery(format!("Subscribe to PeerAdded: {e}"))
        })?;
        let mut peer_removed = p2p.receive_peer_removed().await.map_err(|e| {
            FerricastError::Discovery(format!("Subscribe to PeerRemoved: {e}"))
        })?;

        let conn_task = conn.clone();
        let device_path_str = device_path.to_string();

        // wpa_supplicant's P2P find expires after ~120 s.  Restart it
        // periodically so new sinks remain discoverable during long sessions.
        let mut restart_interval = tokio::time::interval(Duration::from_secs(120));
        restart_interval.tick().await; // skip the immediate first tick

        let handle = tokio::spawn(async move {
            tracing::info!("Miracast discovery started (NM backend)");
            // peer_path_str → device UUID, for DeviceLost lookup.
            let mut known_peers: HashMap<String, uuid::Uuid> = HashMap::new();
            loop {
                tokio::select! {
                    sig = peer_added.next() => {
                        let Some(sig) = sig else {
                            tracing::warn!("NM PeerAdded stream ended");
                            break;
                        };

                        let result: Result<()> = async {
                            let peer_path = sig
                                .args()
                                .map_err(|e| FerricastError::Discovery(format!("PeerAdded args: {e}")))?
                                .path
                                .clone();

                            let proxy = P2pPeerProxy::new(&conn_task, peer_path.clone())
                                .await
                                .map_err(|e| FerricastError::Discovery(format!("P2pPeer proxy: {e}")))?;

                            let name = proxy.name().await.map_err(|e| {
                                FerricastError::Discovery(format!("P2pPeer.Name: {e}"))
                            })?;
                            let hw_address = proxy.hw_address().await.map_err(|e| {
                                FerricastError::Discovery(format!("P2pPeer.HwAddress: {e}"))
                            })?;
                            let wfd_ies = proxy.WfdIEs().await.unwrap_or_default();

                            if !is_miracast_sink(&wfd_ies) {
                                return Ok(());
                            }

                            let model = proxy.model().await.ok();
                            let mut metadata = HashMap::new();
                            metadata.insert("path".into(), peer_path.to_string());
                            metadata.insert("device_path".into(), device_path_str.clone());
                            metadata.insert("hw_address".into(), hw_address);
                            metadata.insert("backend".into(), "nm".into());

                            let device = make_device(
                                name,
                                model,
                                parse_wfd_rtsp_port(&wfd_ies),
                                metadata,
                            );
                            known_peers.insert(peer_path.to_string(), device.id);
                            tx.send(DiscoveryEvent::DeviceFound(device))
                                .await
                                .map_err(|_| {
                                    FerricastError::Discovery("DiscoveryEvent channel closed".into())
                                })
                        }
                        .await;

                        if let Err(e) = result {
                            tracing::error!(%e, "NM peer handling error");
                        }
                    }

                    sig = peer_removed.next() => {
                        let Some(sig) = sig else {
                            tracing::warn!("NM PeerRemoved stream ended");
                            break;
                        };
                        let result: Result<()> = async {
                            let peer_path = sig
                                .args()
                                .map_err(|e| FerricastError::Discovery(format!("PeerRemoved args: {e}")))?
                                .path
                                .clone();
                            if let Some(uuid) = known_peers.remove(&peer_path.to_string()) {
                                tx.send(DiscoveryEvent::DeviceLost(uuid))
                                    .await
                                    .map_err(|_| FerricastError::Discovery("DiscoveryEvent channel closed".into()))?;
                            }
                            Ok(())
                        }
                        .await;
                        if let Err(e) = result {
                            tracing::error!(%e, "NM peer removal error");
                        }
                    }

                    _ = restart_interval.tick() => {
                        tracing::debug!("Restarting NM P2P discovery (periodic refresh)");
                        if let Err(e) = p2p.start_find(HashMap::new()).await {
                            tracing::warn!("NM WifiP2P.StartFind restart failed: {e}");
                        }
                    }
                }
            }
        });

        self.handle = Some(handle);
        self.backend = Some(ActiveBackend::Nm { conn, device_path });
        self.running = true;
        Ok(())
    }

    async fn start_iwd(
        &mut self,
        conn: Connection,
        device_path: OwnedObjectPath,
        tx: mpsc::Sender<DiscoveryEvent>,
    ) -> Result<()> {
        let dev = IwdP2pDeviceProxy::new(&conn, device_path.clone())
            .await
            .map_err(|e| FerricastError::Discovery(format!("iwd P2P device proxy: {e}")))?;

        // Advertise ourselves as a WFD Source in P2P probe responses so TVs
        // in screen-mirror mode can identify us during their scan.
        // Non-fatal: iwd only supports this when compiled with WFD support.
        if let Err(e) = dev.set_WFDElements(wfd_source_ies()).await {
            tracing::debug!("iwd WFDElements not settable (non-fatal): {e}");
        } else {
            tracing::debug!("WFD source IEs set on iwd P2P device");
        }

        dev.request_discovery().await.map_err(|e| {
            FerricastError::Discovery(format!("iwd RequestDiscovery: {e}"))
        })?;

        let om = IwdObjectManagerProxy::new(&conn)
            .await
            .map_err(|e| FerricastError::Discovery(format!("iwd ObjectManager proxy: {e}")))?;

        // Emit any peers already visible at startup.
        let existing = om.get_managed_objects().await.map_err(|e| {
            FerricastError::Discovery(format!("iwd GetManagedObjects: {e}"))
        })?;
        let mut initial_known: HashMap<String, uuid::Uuid> = HashMap::new();
        for (path, ifaces) in &existing {
            if let Some(props) = ifaces.get(IWD_P2P_PEER_IFACE) {
                if let Some(ev) = peer_event_from_iwd_props(path, props) {
                    if let DiscoveryEvent::DeviceFound(ref d) = ev {
                        initial_known.insert(path.to_string(), d.id);
                    }
                    let _ = tx.send(ev).await;
                }
            }
        }

        let mut ifaces_added = om.receive_interfaces_added().await.map_err(|e| {
            FerricastError::Discovery(format!("Subscribe to InterfacesAdded: {e}"))
        })?;
        let mut ifaces_removed = om.receive_interfaces_removed().await.map_err(|e| {
            FerricastError::Discovery(format!("Subscribe to InterfacesRemoved: {e}"))
        })?;

        let handle = tokio::spawn(async move {
            tracing::info!("Miracast discovery started (iwd backend)");
            let mut known_peers = initial_known;
            loop {
                tokio::select! {
                    sig = ifaces_added.next() => {
                        let Some(sig) = sig else {
                            tracing::warn!("iwd InterfacesAdded stream ended");
                            break;
                        };
                        let result: Result<()> = async {
                            let args = sig.args().map_err(|e| {
                                FerricastError::Discovery(format!("InterfacesAdded args: {e}"))
                            })?;
                            let props = match args.interfaces.get(IWD_P2P_PEER_IFACE) {
                                Some(p) => p,
                                None => return Ok(()),
                            };
                            if let Some(ev) = peer_event_from_iwd_props(&args.path, props) {
                                if let DiscoveryEvent::DeviceFound(ref d) = ev {
                                    known_peers.insert(args.path.to_string(), d.id);
                                }
                                tx.send(ev).await.map_err(|_| {
                                    FerricastError::Discovery("DiscoveryEvent channel closed".into())
                                })?;
                            }
                            Ok(())
                        }
                        .await;
                        if let Err(e) = result {
                            tracing::error!(%e, "iwd peer handling error");
                        }
                    }

                    sig = ifaces_removed.next() => {
                        let Some(sig) = sig else {
                            tracing::warn!("iwd InterfacesRemoved stream ended");
                            break;
                        };
                        let result: Result<()> = async {
                            let args = sig.args().map_err(|e| {
                                FerricastError::Discovery(format!("InterfacesRemoved args: {e}"))
                            })?;
                            if args.interfaces.contains(&IWD_P2P_PEER_IFACE.to_string()) {
                                if let Some(uuid) = known_peers.remove(&args.path.to_string()) {
                                    tx.send(DiscoveryEvent::DeviceLost(uuid))
                                        .await
                                        .map_err(|_| FerricastError::Discovery("DiscoveryEvent channel closed".into()))?;
                                }
                            }
                            Ok(())
                        }
                        .await;
                        if let Err(e) = result {
                            tracing::error!(%e, "iwd peer removal error");
                        }
                    }
                }
            }
        });

        self.handle = Some(handle);
        self.backend = Some(ActiveBackend::Iwd { conn, device_path });
        self.running = true;
        Ok(())
    }
}

// ── wpa_supplicant WFD IE helper ─────────────────────────────────────────────

/// Finds the wpa_supplicant interface matching `ifname` and sets our WFD source
/// IEs on it.  These IEs end up in P2P probe responses so WFD sinks see us as
/// a WFD source during their scan.
///
/// Non-fatal: wpa_supplicant may not be running (iwd system), or may be an
/// older version without the `WFDIEs` property.  All failures are logged at
/// DEBUG and silently ignored.
async fn set_wpa_supplicant_wfd_ies(conn: &Connection, ifname: &str) {
    let Ok(wpa) = WpaSupplicantProxy::new(conn).await else {
        tracing::debug!("wpa_supplicant not on D-Bus — skipping WFD IE setup for NM path");
        return;
    };
    let Ok(iface_paths) = wpa.interfaces().await else { return; };
    for path in iface_paths {
        let Ok(iface) = WpaInterfaceProxy::new(conn, path).await else { continue; };
        if iface.ifname().await.ok().as_deref() != Some(ifname) {
            continue;
        }
        match iface.set_WFDIEs(wfd_source_ies()).await {
            Ok(()) => tracing::info!(ifname, "WFD source IEs set on wpa_supplicant interface"),
            Err(e) => tracing::debug!(
                ifname,
                "wpa_supplicant WFDIEs property not settable (non-fatal): {e}"
            ),
        }
        break;
    }
}

// ── backend detection ─────────────────────────────────────────────────────────

/// Returns the iwd adapter path that exposes `net.connman.iwd.p2p.Device`,
/// or `None` if iwd is not running or has no P2P-capable adapter.
async fn find_iwd_p2p_device(conn: &Connection) -> Option<OwnedObjectPath> {
    // Quick check: is iwd even on the bus?
    let om = IwdObjectManagerProxy::new(conn).await.ok()?;
    let objects = om.get_managed_objects().await.ok()?;
    objects
        .into_iter()
        .find(|(_, ifaces)| ifaces.contains_key(IWD_P2P_DEVICE_IFACE))
        .map(|(path, _)| path)
}

/// Returns the NM device object path with type `NM_DEVICE_TYPE_WIFI_P2P`,
/// or `None` if NM is not reachable or has no P2P device.
async fn find_nm_p2p_device(conn: &Connection) -> Result<Option<OwnedObjectPath>> {
    let nm = match NetworkManagerProxy::new(conn).await {
        Ok(n) => n,
        Err(_) => return Ok(None),
    };
    let devices = match nm.get_devices().await {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };

    for path in devices {
        let dev = DeviceProxy::new(conn, path.clone()).await.map_err(|e| {
            FerricastError::Discovery(format!("NM Device proxy: {e}"))
        })?;
        if dev.device_type().await.unwrap_or(0) == NM_DEVICE_TYPE_WIFI_P2P {
            let iface = dev.interface().await.unwrap_or_default();
            tracing::info!(iface, %path, "found NM Wi-Fi P2P device");
            return Ok(Some(path));
        }
    }
    Ok(None)
}

// ── iwd peer parsing ──────────────────────────────────────────────────────────

/// Extracts a `DiscoveryEvent::DeviceFound` from an iwd peer's property dict
/// (from `InterfacesAdded` or `GetManagedObjects`).
///
/// Returns `None` when the peer is not a Miracast sink.
fn peer_event_from_iwd_props(
    path: &OwnedObjectPath,
    props: &HashMap<String, zbus::zvariant::OwnedValue>,
) -> Option<DiscoveryEvent> {
    let wfd_ies: Vec<u8> = props
        .get("WFDElements")
        .and_then(|v| {
            if let zbus::zvariant::Value::Array(arr) = &**v {
                Some(
                    arr.iter()
                        .filter_map(|e| {
                            if let zbus::zvariant::Value::U8(b) = e {
                                Some(*b)
                            } else {
                                None
                            }
                        })
                        .collect(),
                )
            } else {
                None
            }
        })
        .unwrap_or_default();

    if !is_miracast_sink(&wfd_ies) {
        return None;
    }

    let name: String = props
        .get("Name")
        .and_then(|v| {
            if let zbus::zvariant::Value::Str(s) = &**v {
                Some(s.as_str().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "Unknown".to_string());

    let hw_address: String = props
        .get("DeviceAddress")
        .and_then(|v| {
            if let zbus::zvariant::Value::Str(s) = &**v {
                Some(s.as_str().to_string())
            } else {
                None
            }
        })
        .unwrap_or_default();

    tracing::debug!(name, hw_address, wfd_ie_len = wfd_ies.len(), "iwd Miracast sink found");

    let wfd_port = parse_wfd_rtsp_port(&wfd_ies);

    let mut metadata = HashMap::new();
    metadata.insert("path".into(), path.to_string());
    metadata.insert("hw_address".into(), hw_address);
    metadata.insert("backend".into(), "iwd".into());

    Some(DiscoveryEvent::DeviceFound(make_device(
        name, None, wfd_port, metadata,
    )))
}

// ── shared helpers ────────────────────────────────────────────────────────────

fn make_device(
    name: String,
    model: Option<String>,
    wfd_port: u16,
    metadata: HashMap<String, String>,
) -> Device {
    Device {
        id: Uuid::new_v4(),
        name,
        protocol: "miracast",
        protocol_icon: MIRACAST_ICON,
        addr: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
        port: wfd_port,
        model,
        capabilities: DeviceCapabilities {
            supports_video: true,
            supports_screen_mirror: true,
            ..Default::default()
        },
        metadata,
    }
}

/// Parses the WFD RTSP port from WFD IE bytes (WFD subelement 0x00).
pub fn parse_wfd_rtsp_port(wfd_ies: &[u8]) -> u16 {
    if wfd_ies.len() >= 9 && wfd_ies[0] == 0x00 {
        let subelement_len = u16::from_be_bytes([wfd_ies[1], wfd_ies[2]]);
        if subelement_len >= 6 {
            return u16::from_be_bytes([wfd_ies[5], wfd_ies[6]]);
        }
    }
    if wfd_ies.len() >= 6 && wfd_ies[0] == 0x00 {
        return u16::from_be_bytes([wfd_ies[4], wfd_ies[5]]);
    }
    if wfd_ies.len() >= 3 {
        return u16::from_be_bytes([wfd_ies[1], wfd_ies[2]]);
    }
    7236
}

/// Returns `true` when WFD IEs indicate a Miracast sink (device type ≠ Source).
fn is_miracast_sink(wfd_ies: &[u8]) -> bool {
    if wfd_ies.is_empty() {
        return false;
    }
    // Vendor-Specific IE wrapper (0xDD OUI 50:6F:9A:0A)
    if wfd_ies[0] == 0xdd && wfd_ies.len() >= 7 {
        return is_miracast_sink(&wfd_ies[6..]);
    }
    // WFD Device Information subelement (ID=0x00): bits [1:0] = device type
    // 0=Source, 1=Primary Sink, 2=Secondary Sink, 3=Dual Role
    if wfd_ies[0] == 0x00 && wfd_ies.len() >= 5 {
        let device_info = u16::from_be_bytes([wfd_ies[3], wfd_ies[4]]);
        let device_type = device_info & 0x0003;
        tracing::debug!(device_type, device_info, "WFD Device Information");
        return device_type != 0;
    }
    // Permissive fallback for unknown IE formats
    matches!(wfd_ies[0], 0x00 | 0x01 | 0x06 | 0x07)
}
