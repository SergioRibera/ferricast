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
    DeviceProxy, IWD_P2P_DEVICE_IFACE, IWD_P2P_DISPLAY_IFACE, IWD_P2P_PEER_IFACE,
    IWD_SIMPLE_CONFIG_IFACE, IwdObjectManagerProxy, IwdP2pDeviceProxy, IwdP2pServiceManagerProxy,
    NM_DEVICE_TYPE_WIFI, NM_DEVICE_TYPE_WIFI_P2P, NetworkManagerProxy,
    NmAccessPointProxy, NmWirelessProxy,
    P2pPeerProxy, WifiP2pProxy, WpaInterfaceProxy, WpaSupplicantProxy,
};
use crate::session::{WFD_DEFAULT_PORT, wfd_source_ies};

const MIRACAST_ICON: Bytes = Bytes::from_static(include_bytes!("../../../assets/miracast.svg"));

// ── backend discriminator ─────────────────────────────────────────────────────

/// Which D-Bus backend is backing a discovery source.
/// Multiple backends can run simultaneously (e.g. iwd P2P + NM Wi-Fi scan).
enum ActiveBackend {
    Nm {
        conn: Connection,
        device_path: OwnedObjectPath,
    },
    Iwd {
        conn: Connection,
        device_path: OwnedObjectPath,
    },
    /// NM regular Wi-Fi scan for DIRECT- Group Owners (Windows Connect on 5 GHz).
    /// No explicit D-Bus stop call needed — task abort is sufficient.
    NmWifi,
}

// ── public type ───────────────────────────────────────────────────────────────

pub struct MiracastDiscovery {
    handles: Vec<JoinHandle<()>>,
    backends: Vec<ActiveBackend>,
    running: bool,
}

impl Default for MiracastDiscovery {
    fn default() -> Self {
        Self {
            handles: Vec::new(),
            backends: Vec::new(),
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

        // iwd P2P: finds 2.4 GHz WFD peers in P2P device discovery mode.
        if let Some(device_path) = find_iwd_p2p_device(&conn).await {
            self.start_iwd(conn.clone(), device_path, tx.clone()).await?;
        }

        // NM Wi-Fi station scan: finds DIRECT- Group Owners on 5 GHz (e.g.
        // Windows Connect app).  Complements iwd P2P — different bands.
        if let Some(device_path) = find_nm_wifi_device(&conn).await? {
            self.start_nm_wifi(conn.clone(), device_path, tx.clone()).await?;
        }

        // NM P2P: last resort when iwd is absent and NM manages P2P natively.
        if self.backends.is_empty() {
            if let Some(device_path) = find_nm_p2p_device(&conn).await? {
                self.start_nm(conn, device_path, tx).await?;
            }
        }

        if self.backends.is_empty() {
            return Err(FerricastError::Discovery(
                "No Wi-Fi P2P device found via iwd or NetworkManager. \
                 Check your adapter, drivers, and that iwd or NetworkManager is running."
                    .into(),
            ));
        }

        self.running = true;
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        if !self.running {
            return Ok(());
        }

        // Abort tasks first so D-Bus subscriptions are released.
        for h in self.handles.drain(..) {
            h.abort();
        }

        for backend in self.backends.drain(..) {
            match backend {
                ActiveBackend::Nm { conn, device_path } => {
                    let p2p = WifiP2pProxy::new(&conn, device_path).await.map_err(|e| {
                        FerricastError::Discovery(format!("WifiP2P proxy for stop: {e}"))
                    })?;
                    p2p.stop_find().await.map_err(|e| {
                        FerricastError::Discovery(format!("NM WifiP2P.StopFind: {e}"))
                    })?;
                }
                ActiveBackend::Iwd { conn, device_path } => {
                    let dev = IwdP2pDeviceProxy::new(&conn, device_path)
                        .await
                        .map_err(|e| {
                            FerricastError::Discovery(format!("iwd P2P device proxy for stop: {e}"))
                        })?;
                    dev.release_discovery().await.map_err(|e| {
                        FerricastError::Discovery(format!("iwd ReleaseDiscovery: {e}"))
                    })?;
                }
                ActiveBackend::NmWifi => {}
            }
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
        if find_iwd_p2p_device(&conn).await.is_some() {
            register_iwd_wfd_source(&conn).await;
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
                            // WfdIEs is empty when NM's iwd backend lacks WFD support.
                            // Only filter by device type when NM actually provided IEs.
                            let wfd_ies = proxy.WfdIEs().await.unwrap_or_default();
                            if !wfd_ies.is_empty() && !is_miracast_sink(&wfd_ies) {
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

        self.handles.push(handle);
        self.backends.push(ActiveBackend::Nm { conn, device_path });
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

        // Enable P2P if the adapter has it disabled (e.g. fresh boot / user never
        // toggled it).  Non-fatal: if it fails we still attempt discovery.
        if !dev.enabled().await.unwrap_or(true) {
            match dev.set_enabled(true).await {
                Ok(()) => tracing::info!("iwd P2P device enabled"),
                Err(e) => tracing::warn!("Could not enable iwd P2P device: {e}"),
            }
        }

        // Register as a WFD source via iwd's ServiceManager so iwd includes
        // our subelements in P2P probe responses.  Non-fatal: only present
        // when iwd is compiled with WFD support (--enable-wfd).
        register_iwd_wfd_source(&conn).await;

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
            if ifaces.contains_key(IWD_P2P_PEER_IFACE) {
                if let Some(ev) = peer_event_from_iwd_props(path, ifaces) {
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
                            if !args.interfaces.contains_key(IWD_P2P_PEER_IFACE) {
                                return Ok(());
                            }
                            if let Some(ev) = peer_event_from_iwd_props(&args.path, &args.interfaces) {
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

        self.handles.push(handle);
        self.backends.push(ActiveBackend::Iwd { conn, device_path });
        Ok(())
    }

    async fn start_nm_wifi(
        &mut self,
        conn: Connection,
        device_path: OwnedObjectPath,
        tx: mpsc::Sender<DiscoveryEvent>,
    ) -> Result<()> {
        let wifi = NmWirelessProxy::new(&conn, device_path.clone())
            .await
            .map_err(|e| FerricastError::Discovery(format!("NM Wireless proxy: {e}")))?;

        if let Err(e) = wifi.request_scan(HashMap::new()).await {
            tracing::warn!("NM Wi-Fi initial scan failed (non-fatal): {e}");
        }

        // Emit DIRECT- APs already visible at startup.
        let existing = wifi.access_points().await.unwrap_or_default();
        let mut known_aps: HashMap<String, Uuid> = HashMap::new();
        for ap_path in &existing {
            if let Some(device) = nm_ap_as_miracast_device(&conn, ap_path, &device_path).await {
                known_aps.insert(ap_path.to_string(), device.id);
                let _ = tx.send(DiscoveryEvent::DeviceFound(device)).await;
            }
        }

        let mut ap_added = wifi.receive_access_point_added().await.map_err(|e| {
            FerricastError::Discovery(format!("Subscribe to AccessPointAdded: {e}"))
        })?;
        let mut ap_removed = wifi.receive_access_point_removed().await.map_err(|e| {
            FerricastError::Discovery(format!("Subscribe to AccessPointRemoved: {e}"))
        })?;

        let conn_task = conn.clone();
        let device_path_task = device_path.clone();
        let mut rescan_interval = tokio::time::interval(Duration::from_secs(30));
        rescan_interval.tick().await;

        let handle = tokio::spawn(async move {
            tracing::info!("Miracast discovery started (NM Wi-Fi scan backend)");
            let mut known_aps = known_aps;
            loop {
                tokio::select! {
                    sig = ap_added.next() => {
                        let Some(sig) = sig else {
                            tracing::warn!("NM AccessPointAdded stream ended");
                            break;
                        };
                        let result: Result<()> = async {
                            let ap_path = sig
                                .args()
                                .map_err(|e| FerricastError::Discovery(format!("AccessPointAdded args: {e}")))?
                                .access_point
                                .clone();
                            if let Some(device) = nm_ap_as_miracast_device(&conn_task, &ap_path, &device_path_task).await {
                                known_aps.insert(ap_path.to_string(), device.id);
                                tx.send(DiscoveryEvent::DeviceFound(device))
                                    .await
                                    .map_err(|_| FerricastError::Discovery("channel closed".into()))?;
                            }
                            Ok(())
                        }.await;
                        if let Err(e) = result {
                            tracing::error!(%e, "NM Wi-Fi AP handling error");
                        }
                    }

                    sig = ap_removed.next() => {
                        let Some(sig) = sig else {
                            tracing::warn!("NM AccessPointRemoved stream ended");
                            break;
                        };
                        let result: Result<()> = async {
                            let ap_path = sig
                                .args()
                                .map_err(|e| FerricastError::Discovery(format!("AccessPointRemoved args: {e}")))?
                                .access_point
                                .clone();
                            if let Some(uuid) = known_aps.remove(&ap_path.to_string()) {
                                tx.send(DiscoveryEvent::DeviceLost(uuid))
                                    .await
                                    .map_err(|_| FerricastError::Discovery("channel closed".into()))?;
                            }
                            Ok(())
                        }.await;
                        if let Err(e) = result {
                            tracing::error!(%e, "NM Wi-Fi AP removal error");
                        }
                    }

                    _ = rescan_interval.tick() => {
                        tracing::debug!("Periodic NM Wi-Fi rescan for DIRECT- networks");
                        if let Err(e) = wifi.request_scan(HashMap::new()).await {
                            tracing::warn!("NM Wi-Fi rescan failed: {e}");
                        }
                    }
                }
            }
        });

        self.handles.push(handle);
        self.backends.push(ActiveBackend::NmWifi);
        Ok(())
    }
}

// ── iwd WFD registration ──────────────────────────────────────────────────────

/// Calls `net.connman.iwd.p2p.ServiceManager.RegisterDisplayService` so iwd
/// advertises us as a WFD source in P2P probe responses.
///
/// iwd's ServiceManager takes structured properties (`Source: bool`, `Port: q`)
/// and builds the WFD IEs internally — it does NOT accept raw subelement bytes.
///
/// Non-fatal: the ServiceManager interface only exists when iwd is compiled
/// with `--enable-wfd`.  Failure is logged at WARN with actionable guidance.
async fn register_iwd_wfd_source(conn: &Connection) {
    let sm = match IwdP2pServiceManagerProxy::new(conn).await {
        Ok(sm) => sm,
        Err(e) => {
            tracing::warn!(
                "iwd WFD ServiceManager not reachable — Miracast sinks may not discover this \
                 device. Rebuild iwd with --enable-wfd. Error: {e}"
            );
            return;
        }
    };

    let source_val = match zbus::zvariant::OwnedValue::try_from(zbus::zvariant::Value::from(true)) {
        Ok(v) => v,
        Err(e) => { tracing::warn!("WFD Source value encoding failed: {e}"); return; }
    };
    let port_val = match zbus::zvariant::OwnedValue::try_from(
        zbus::zvariant::Value::from(WFD_DEFAULT_PORT),
    ) {
        Ok(v) => v,
        Err(e) => { tracing::warn!("WFD Port value encoding failed: {e}"); return; }
    };

    let mut props = std::collections::HashMap::new();
    props.insert("Source".to_string(), source_val);
    props.insert("Port".to_string(), port_val);

    match sm.register_display_service(props).await {
        Ok(()) => tracing::info!("WFD source registered with iwd ServiceManager (port {WFD_DEFAULT_PORT})"),
        Err(e) => tracing::warn!(
            "iwd WFD support not available — Miracast sinks may not discover this device. \
             Rebuild iwd with --enable-wfd. Error: {e}"
        ),
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

/// Returns the NM device object path with type `NM_DEVICE_TYPE_WIFI` (regular
/// Wi-Fi station), or `None` if NM is not reachable or has no Wi-Fi device.
/// Used for scanning DIRECT- Group Owners (e.g. Windows Connect on 5 GHz).
async fn find_nm_wifi_device(conn: &Connection) -> Result<Option<OwnedObjectPath>> {
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
        if dev.device_type().await.unwrap_or(0) == NM_DEVICE_TYPE_WIFI {
            let iface = dev.interface().await.unwrap_or_default();
            tracing::info!(iface, %path, "found NM Wi-Fi device for DIRECT- scan");
            return Ok(Some(path));
        }
    }
    Ok(None)
}

// ── NM Wi-Fi AP filtering ─────────────────────────────────────────────────────

/// Returns a `Device` if `ap_path` is a DIRECT- network (potential Miracast
/// sink acting as a Wi-Fi Direct Group Owner), `None` otherwise.
async fn nm_ap_as_miracast_device(
    conn: &Connection,
    ap_path: &OwnedObjectPath,
    nm_device_path: &OwnedObjectPath,
) -> Option<Device> {
    let ap = NmAccessPointProxy::new(conn, ap_path.clone()).await.ok()?;
    let ssid_bytes = ap.ssid().await.ok()?;
    if !ssid_bytes.starts_with(b"DIRECT-") {
        return None;
    }
    let ssid = String::from_utf8_lossy(&ssid_bytes).into_owned();
    let bssid = ap.hw_address().await.unwrap_or_default();
    let freq = ap.frequency().await.unwrap_or(0);

    tracing::debug!(ssid, bssid, freq, "found DIRECT- network (potential Miracast sink)");

    let mut metadata = HashMap::new();
    metadata.insert("ssid".into(), ssid.clone());
    metadata.insert("bssid".into(), bssid);
    metadata.insert("ap_path".into(), ap_path.to_string());
    metadata.insert("nm_device_path".into(), nm_device_path.to_string());
    metadata.insert("backend".into(), "nm_wifi".into());

    Some(make_device(ssid, None, WFD_DEFAULT_PORT, metadata))
}

// ── iwd peer parsing ──────────────────────────────────────────────────────────

/// Extracts a `DiscoveryEvent::DeviceFound` from an iwd peer's property dict
/// Examines the full interface map for a single D-Bus object (from
/// `InterfacesAdded` or `GetManagedObjects`) and returns a `DeviceFound` event
/// when the object is a connectable WFD sink, or `None` otherwise.
///
/// Conditions (matching iwd's own `wfd-source` reference implementation):
///   1. Must expose `net.connman.iwd.p2p.Peer` (basic P2P peer)
///   2. Must expose `net.connman.iwd.p2p.Display` with `Sink == true`
///   3. Must expose `net.connman.iwd.SimpleConfiguration` (WPS, for PushButton connect)
fn peer_event_from_iwd_props(
    path: &OwnedObjectPath,
    ifaces: &HashMap<String, HashMap<String, zbus::zvariant::OwnedValue>>,
) -> Option<DiscoveryEvent> {
    let iface_names: Vec<&str> = ifaces.keys().map(String::as_str).collect();

    // Extract name early so every skip message identifies the peer.
    let peer_name: String = ifaces
        .get(IWD_P2P_PEER_IFACE)
        .and_then(|p| p.get("Name"))
        .and_then(|v| {
            if let zbus::zvariant::Value::Str(s) = &**v {
                Some(s.as_str().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "<unknown>".to_string());

    tracing::debug!(?path, name = peer_name, ?iface_names, "iwd peer interfaces");

    // Require WPS interface — PushButton() is the correct connect method.
    if !ifaces.contains_key(IWD_SIMPLE_CONFIG_IFACE) {
        tracing::debug!(?path, name = peer_name, "iwd peer skipped: no SimpleConfiguration (WSC) interface");
        return None;
    }

    // Require WFD Display interface AND Sink == true.
    let display_props = match ifaces.get(IWD_P2P_DISPLAY_IFACE) {
        Some(p) => p,
        None => {
            tracing::debug!(?path, name = peer_name, "iwd peer skipped: no p2p.Display interface (iwd may not have parsed WFD IEs)");
            return None;
        }
    };
    let is_sink = display_props
        .get("Sink")
        .and_then(|v| {
            if let zbus::zvariant::Value::Bool(b) = &**v {
                Some(*b)
            } else {
                None
            }
        })
        .unwrap_or(false);
    if !is_sink {
        tracing::debug!(?path, name = peer_name, "iwd peer skipped: Display.Sink == false (peer is WFD source, not sink)");
        return None;
    }

    let peer_props = ifaces.get(IWD_P2P_PEER_IFACE)?;

    let name: String = peer_props
        .get("Name")
        .and_then(|v| {
            if let zbus::zvariant::Value::Str(s) = &**v {
                Some(s.as_str().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "Unknown".to_string());

    let hw_address: String = peer_props
        .get("DeviceAddress")
        .and_then(|v| {
            if let zbus::zvariant::Value::Str(s) = &**v {
                Some(s.as_str().to_string())
            } else {
                None
            }
        })
        .unwrap_or_default();

    tracing::debug!(name, hw_address, "iwd WFD sink found");

    let mut metadata = HashMap::new();
    metadata.insert("path".into(), path.to_string());
    metadata.insert("hw_address".into(), hw_address);
    metadata.insert("backend".into(), "iwd".into());

    Some(DiscoveryEvent::DeviceFound(make_device(
        name, None, WFD_DEFAULT_PORT, metadata,
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
