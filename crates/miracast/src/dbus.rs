use std::collections::HashMap;


#[zbus::proxy(
    interface = "org.freedesktop.NetworkManager",
    default_service = "org.freedesktop.NetworkManager",
    default_path = "/org/freedesktop/NetworkManager"    
)]
pub trait NetworkManager {
    async fn get_devices(&self) -> zbus::Result<Vec<zvariant::OwnedObjectPath>>;

    async fn add_and_activate_connection2(
        &self,
        connection: HashMap<&str, HashMap<&str, zvariant::Value<'_>>>,
        device: zvariant::ObjectPath<'_>,
        specific_object: zvariant::ObjectPath<'_>,
        options: HashMap<&str, zvariant::Value<'_>>,
    ) -> zbus::Result<(
        zvariant::OwnedObjectPath,
        zvariant::OwnedObjectPath,
        HashMap<String, zvariant::OwnedValue>,
    )>;
}

#[zbus::proxy(
    interface = "fi.w1.wpa_supplicant1",
    default_service = "fi.w1.wpa_supplicant1",
    default_path = "/f1/w1/wpa_supplicant1"
)]
pub trait WpaSupplicant {
    #[zbus(property, name="Interface")]
    fn interfaces(&self) -> zbus::Result<Vec<zvariant::OwnedObjectPath>>;
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
    default_service = "org.freedesktop.NetworkManager",
)]
pub trait WifiP2p {
    async fn start_find(&self, args: HashMap<&str, zbus::zvariant::Value<'_>>) -> zbus::Result<()>;

    async fn stop_find(&self) -> zbus::Result<()>;

    #[zbus(property)]
    fn peers(&self) -> zbus::Result<Vec<zvariant::OwnedObjectPath>>;

    #[zbus(signal)]
    async fn peer_added(&self, path: zvariant::OwnedObjectPath);

}
