use std::collections::HashMap;

use zbus::zvariant::{OwnedObjectPath, OwnedValue};

// NM device type 30 = NM_DEVICE_TYPE_WIFI_P2P
pub const NM_DEVICE_TYPE_WIFI_P2P: u32 = 30;

// org.freedesktop.NetworkManager.Connection.Active.State
pub const NM_ACTIVE_CONNECTION_STATE_ACTIVATED: u32 = 2;
pub const NM_ACTIVE_CONNECTION_STATE_DEACTIVATED: u32 = 4;

#[zbus::proxy(
    interface = "org.freedesktop.NetworkManager",
    default_service = "org.freedesktop.NetworkManager",
    default_path = "/org/freedesktop/NetworkManager"
)]
pub trait NetworkManager {
    async fn get_devices(&self) -> zbus::Result<Vec<OwnedObjectPath>>;

    /// Adds and immediately activates a Wi-Fi P2P (or any) connection.
    /// Returns `(settings_path, active_connection_path, result)`.
    async fn add_and_activate_connection2(
        &self,
        connection: HashMap<String, HashMap<String, OwnedValue>>,
        device: OwnedObjectPath,
        specific_object: OwnedObjectPath,
        options: HashMap<String, OwnedValue>,
    ) -> zbus::Result<(OwnedObjectPath, OwnedObjectPath, HashMap<String, OwnedValue>)>;

    /// Deactivates an active connection by its object path.
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
    async fn start_find(
        &self,
        options: HashMap<String, OwnedValue>,
    ) -> zbus::Result<()>;

    async fn stop_find(&self) -> zbus::Result<()>;

    #[zbus(property)]
    fn peers(&self) -> zbus::Result<Vec<OwnedObjectPath>>;

    #[zbus(signal)]
    fn peer_added(&self, path: OwnedObjectPath) -> zbus::Result<()>;

    #[zbus(signal)]
    fn peer_removed(&self, path: OwnedObjectPath) -> zbus::Result<()>;
}

/// `org.freedesktop.NetworkManager.Connection.Active`
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

/// `org.freedesktop.NetworkManager.IP4Config`
#[zbus::proxy(
    interface = "org.freedesktop.NetworkManager.IP4Config",
    default_service = "org.freedesktop.NetworkManager"
)]
pub trait Ip4Config {
    /// Each entry: `{"address": s, "prefix": u}`.
    #[zbus(property)]
    fn address_data(&self) -> zbus::Result<Vec<HashMap<String, OwnedValue>>>;

    #[zbus(property)]
    fn gateway(&self) -> zbus::Result<String>;
}
