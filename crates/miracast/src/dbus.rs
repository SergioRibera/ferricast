use std::collections::HashMap;

use zbus::zvariant::{OwnedObjectPath, OwnedValue};

// ── NetworkManager constants ──────────────────────────────────────────────────

pub const NM_DEVICE_TYPE_WIFI: u32 = 2;
pub const NM_DEVICE_TYPE_WIFI_P2P: u32 = 30;
pub const NM_ACTIVE_CONNECTION_STATE_ACTIVATED: u32 = 2;
pub const NM_ACTIVE_CONNECTION_STATE_DEACTIVATING: u32 = 3;
pub const NM_ACTIVE_CONNECTION_STATE_DEACTIVATED: u32 = 4;

// ── iwd constants ─────────────────────────────────────────────────────────────

pub const IWD_P2P_PEER_IFACE: &str = "net.connman.iwd.p2p.Peer";
pub const IWD_P2P_DEVICE_IFACE: &str = "net.connman.iwd.p2p.Device";
pub const IWD_P2P_DISPLAY_IFACE: &str = "net.connman.iwd.p2p.Display";
pub const IWD_SIMPLE_CONFIG_IFACE: &str = "net.connman.iwd.SimpleConfiguration";

// ── NetworkManager proxies ────────────────────────────────────────────────────

#[zbus::proxy(
    interface = "org.freedesktop.NetworkManager",
    default_service = "org.freedesktop.NetworkManager",
    default_path = "/org/freedesktop/NetworkManager"
)]
pub trait NetworkManager {
    async fn get_devices(&self) -> zbus::Result<Vec<OwnedObjectPath>>;

    /// Returns `(settings_path, active_connection_path, result)`.
    async fn add_and_activate_connection2(
        &self,
        connection: HashMap<String, HashMap<String, OwnedValue>>,
        device: OwnedObjectPath,
        specific_object: OwnedObjectPath,
        options: HashMap<String, zvariant::Value<'_>>,
    ) -> zbus::Result<(OwnedObjectPath, OwnedObjectPath, HashMap<String, OwnedValue>)>;

    async fn deactivate_connection(&self, active_connection: OwnedObjectPath)
        -> zbus::Result<()>;
}

#[zbus::proxy(
    interface = "org.freedesktop.NetworkManager.WifiP2PPeer",
    default_service = "org.freedesktop.NetworkManager"
)]
pub trait P2pPeer {
    #[zbus(property)]
    fn flags(&self) -> zbus::Result<u32>;
    #[zbus(property)]
    fn hw_address(&self) -> zbus::Result<String>;
    #[zbus(property)]
    fn manufacturer(&self) -> zbus::Result<String>;
    #[zbus(property)]
    fn model(&self) -> zbus::Result<String>;
    #[zbus(property)]
    fn model_number(&self) -> zbus::Result<String>;
    #[zbus(property)]
    fn serial(&self) -> zbus::Result<String>;
    #[zbus(property)]
    #[allow(non_snake_case)]
    fn WfdIEs(&self) -> zbus::Result<Vec<u8>>;
    #[zbus(property)]
    fn name(&self) -> zbus::Result<String>;
    #[zbus(property)]
    fn last_seen(&self) -> zbus::Result<i64>;
}

#[zbus::proxy(
    interface = "org.freedesktop.NetworkManager.Device",
    default_service = "org.freedesktop.NetworkManager"
)]
pub trait Device {
    #[zbus(property)]
    fn device_type(&self) -> zbus::Result<u32>;
    #[zbus(property)]
    fn interface(&self) -> zbus::Result<String>;
}

#[zbus::proxy(
    interface = "org.freedesktop.NetworkManager.Device.WifiP2P",
    default_service = "org.freedesktop.NetworkManager"
)]
pub trait WifiP2p {
    async fn start_find(&self, options: HashMap<String, OwnedValue>) -> zbus::Result<()>;
    async fn stop_find(&self) -> zbus::Result<()>;

    #[zbus(property)]
    fn peers(&self) -> zbus::Result<Vec<OwnedObjectPath>>;

    #[zbus(signal)]
    fn peer_added(&self, path: OwnedObjectPath) -> zbus::Result<()>;
    #[zbus(signal)]
    fn peer_removed(&self, path: OwnedObjectPath) -> zbus::Result<()>;
}

#[zbus::proxy(
    interface = "org.freedesktop.NetworkManager.Connection.Active",
    default_service = "org.freedesktop.NetworkManager"
)]
pub trait ActiveConnection {
    /// 0=Unknown, 1=Activating, 2=Activated, 3=Deactivating, 4=Deactivated
    #[zbus(property)]
    fn state(&self) -> zbus::Result<u32>;
    #[zbus(property)]
    fn ip4_config(&self) -> zbus::Result<OwnedObjectPath>;
    #[zbus(property)]
    fn devices(&self) -> zbus::Result<Vec<OwnedObjectPath>>;
}

#[zbus::proxy(
    interface = "org.freedesktop.NetworkManager.IP4Config",
    default_service = "org.freedesktop.NetworkManager"
)]
pub trait Ip4Config {
    #[zbus(property)]
    fn address_data(&self) -> zbus::Result<Vec<HashMap<String, OwnedValue>>>;
    #[zbus(property)]
    fn gateway(&self) -> zbus::Result<String>;
}

// ── NetworkManager Wi-Fi proxies ──────────────────────────────────────────────

/// NM wireless device — used for regular AP scanning to find DIRECT- Group
/// Owners (e.g. Windows Connect app) that operate on 5 GHz and don't
/// appear in iwd P2P device discovery (which only scans 2.4 GHz social
/// channels per Wi-Fi Direct spec).
#[zbus::proxy(
    interface = "org.freedesktop.NetworkManager.Device.Wireless",
    default_service = "org.freedesktop.NetworkManager"
)]
pub trait NmWireless {
    async fn request_scan(&self, options: HashMap<String, OwnedValue>) -> zbus::Result<()>;

    #[zbus(property)]
    fn access_points(&self) -> zbus::Result<Vec<OwnedObjectPath>>;

    #[zbus(signal)]
    fn access_point_added(&self, access_point: OwnedObjectPath) -> zbus::Result<()>;
    #[zbus(signal)]
    fn access_point_removed(&self, access_point: OwnedObjectPath) -> zbus::Result<()>;
}

/// Individual Wi-Fi access point seen by NM.
#[zbus::proxy(
    interface = "org.freedesktop.NetworkManager.AccessPoint",
    default_service = "org.freedesktop.NetworkManager"
)]
pub trait NmAccessPoint {
    #[zbus(property)]
    fn ssid(&self) -> zbus::Result<Vec<u8>>;
    #[zbus(property)]
    fn hw_address(&self) -> zbus::Result<String>;
    #[zbus(property)]
    fn frequency(&self) -> zbus::Result<u32>;
    #[zbus(property)]
    fn strength(&self) -> zbus::Result<u8>;
}

// ── wpa_supplicant proxies ────────────────────────────────────────────────────

/// Root wpa_supplicant object — used to enumerate managed network interfaces.
#[zbus::proxy(
    interface = "fi.w1.wpa_supplicant1",
    default_service = "fi.w1.wpa_supplicant1",
    default_path = "/fi/w1/wpa_supplicant1"
)]
pub trait WpaSupplicant {
    /// D-Bus object paths of all managed network interfaces.
    #[zbus(property)]
    fn interfaces(&self) -> zbus::Result<Vec<OwnedObjectPath>>;
}

/// Per-interface wpa_supplicant object.
///
/// `WFDIEs` (available in wpa_supplicant ≥ 2.10) is a byte array of raw
/// WFD subelements.  When set, wpa_supplicant includes these IEs in P2P
/// probe responses so WFD sinks can identify our device as a source during
/// their scan.
#[zbus::proxy(
    interface = "fi.w1.wpa_supplicant1.Interface",
    default_service = "fi.w1.wpa_supplicant1"
)]
pub trait WpaInterface {
    #[zbus(property)]
    fn ifname(&self) -> zbus::Result<String>;

    #[zbus(property, name = "WFDIEs")]
    #[allow(non_snake_case)]
    fn WFDIEs(&self) -> zbus::Result<Vec<u8>>;
    #[zbus(property, name = "WFDIEs")]
    #[allow(non_snake_case)]
    fn set_WFDIEs(&self, value: Vec<u8>) -> zbus::Result<()>;
}


// ── iwd proxies ───────────────────────────────────────────────────────────────

/// Standard D-Bus ObjectManager on iwd's service root.
///
/// Used to enumerate all current objects and watch for peers appearing
/// (`InterfacesAdded`) and disappearing (`InterfacesRemoved`).
#[zbus::proxy(
    interface = "org.freedesktop.DBus.ObjectManager",
    default_service = "net.connman.iwd",
    default_path = "/"
)]
pub trait IwdObjectManager {
    async fn get_managed_objects(
        &self,
    ) -> zbus::Result<
        HashMap<OwnedObjectPath, HashMap<String, HashMap<String, OwnedValue>>>,
    >;

    #[zbus(signal)]
    fn interfaces_added(
        &self,
        path: OwnedObjectPath,
        interfaces: HashMap<String, HashMap<String, OwnedValue>>,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    fn interfaces_removed(
        &self,
        path: OwnedObjectPath,
        interfaces: Vec<String>,
    ) -> zbus::Result<()>;
}

/// iwd P2P device — lives on the same object path as the Wi-Fi adapter
/// (e.g. `/net/connman/iwd/0`).
#[zbus::proxy(
    interface = "net.connman.iwd.p2p.Device",
    default_service = "net.connman.iwd"
)]
pub trait IwdP2pDevice {
    /// Start P2P peer scanning. Call `ReleaseDiscovery` when done.
    async fn request_discovery(&self) -> zbus::Result<()>;
    async fn release_discovery(&self) -> zbus::Result<()>;

    /// Returns `(peer_path, signal_level_dBm)` pairs for currently visible peers.
    async fn get_peers(&self) -> zbus::Result<Vec<(OwnedObjectPath, i16)>>;

    #[zbus(property)]
    fn enabled(&self) -> zbus::Result<bool>;
    #[zbus(property)]
    fn set_enabled(&self, value: bool) -> zbus::Result<()>;
    #[zbus(property)]
    fn name(&self) -> zbus::Result<String>;

    /// WFD source IEs on peers (read-only, only present when iwd has WFD support).
    #[zbus(property)]
    #[allow(non_snake_case)]
    fn WFDElements(&self) -> zbus::Result<Vec<u8>>;
}

/// iwd P2P peer — appears as a D-Bus object when a peer is discovered.
#[zbus::proxy(
    interface = "net.connman.iwd.p2p.Peer",
    default_service = "net.connman.iwd"
)]
pub trait IwdP2pPeer {
    /// Connects to this peer (P2P group formation + IP assignment).
    /// Blocks until the connection is fully established or fails.
    async fn connect(&self) -> zbus::Result<()>;
    async fn disconnect(&self) -> zbus::Result<()>;

    #[zbus(property)]
    fn name(&self) -> zbus::Result<String>;
    #[zbus(property)]
    fn device(&self) -> zbus::Result<OwnedObjectPath>;
    #[zbus(property)]
    fn device_address(&self) -> zbus::Result<String>;
    #[zbus(property)]
    fn connected(&self) -> zbus::Result<bool>;

    /// Wi-Fi Display Information Elements. Only present on WFD-capable peers.
    #[zbus(property)]
    #[allow(non_snake_case)]
    fn WFDElements(&self) -> zbus::Result<Vec<u8>>;
}

/// iwd P2P Display interface — present on WFD-capable peers.
/// The `Sink` property is `true` when the peer is a WFD sink (display),
/// `false` when it is a WFD source.  Only sinks are connectable from our
/// perspective (we are a source).
#[zbus::proxy(
    interface = "net.connman.iwd.p2p.Display",
    default_service = "net.connman.iwd"
)]
pub trait IwdP2pDisplay {
    /// `true` = peer is a WFD sink (receiver); `false` = peer is a WFD source.
    #[zbus(property)]
    fn sink(&self) -> zbus::Result<bool>;
}

/// iwd Wi-Fi Simple Configuration (WPS) interface on a P2P peer.
///
/// `PushButton` initiates WPS-PBC P2P group formation.  This is the correct
/// way to connect to a WFD sink under iwd — `net.connman.iwd.p2p.Peer.Connect`
/// does not work for WFD peers.  The call blocks until group formation and
/// DHCP complete (up to ~120 s).
#[zbus::proxy(
    interface = "net.connman.iwd.SimpleConfiguration",
    default_service = "net.connman.iwd"
)]
pub trait IwdSimpleConfiguration {
    async fn push_button(&self) -> zbus::Result<()>;
    async fn start_enrollee(&self) -> zbus::Result<()>;
    async fn cancel(&self) -> zbus::Result<()>;
}

/// iwd P2P Service Manager — registers WFD display services so iwd includes
/// our WFD source subelements in P2P probe responses.
///
/// Lives at the root iwd object path (`/net/connman/iwd`), not on the
/// per-adapter path.  `RegisterDisplayService` must be called at startup;
/// iwd automatically unregisters when our D-Bus client connection drops.
#[zbus::proxy(
    interface = "net.connman.iwd.p2p.ServiceManager",
    default_service = "net.connman.iwd",
    default_path = "/net/connman/iwd"
)]
pub trait IwdP2pServiceManager {
    /// Register this process as a WFD source.
    ///
    /// `properties` accepts:
    /// - `"Source"` → `b` — `true` to register as a WFD source
    /// - `"Port"` → `q` — RTSP TCP port the source listens on (typically 7236)
    ///
    /// iwd constructs the WFD IEs from these properties internally.
    async fn register_display_service(
        &self,
        properties: HashMap<String, OwnedValue>,
    ) -> zbus::Result<()>;

    /// Explicit unregister (iwd also auto-unregisters on D-Bus disconnect).
    async fn unregister_display_service(&self) -> zbus::Result<()>;
}
