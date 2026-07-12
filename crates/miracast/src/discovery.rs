use std::{collections::HashMap, net::{IpAddr, Ipv6Addr}, str::FromStr, sync::atomic::{AtomicBool, Ordering}};

use bytes::Bytes;
use futures_lite::StreamExt;
use tokio::{sync::mpsc, task::JoinHandle};

use ferricast_core::{Device, DeviceCapabilities, Discovery, DiscoveryEvent, FerricastError, Result};
use uuid::Uuid;
use zbus::Connection;
use zvariant::OwnedObjectPath;

const MIRACAST_ICON: Bytes = Bytes::from_static(include_bytes!("../../../assets/miracast.svg"));

#[derive(Default)]
pub struct MiracastDiscovery {
    handle: Option<JoinHandle<()>>,
    wifi_p2p: Option<WifiP2pProxy<'static>>,
    is_running: AtomicBool,
}


impl Discovery for MiracastDiscovery {
    const PROTOCOL: &'static str = "miracast";

    async fn start(&mut self, tx: mpsc::Sender<DiscoveryEvent>) -> Result<()> {
        let connection = Connection::system().await.map_err(|_| {
            FerricastError::Discovery("Cannot connect to dbus".to_string())
        })?;

        let nm = NetworkManagerProxy::new(&connection).await.map_err(|_| {
            FerricastError::Discovery("Cannot connect to network manager".to_string())
        })?;

        let devices = nm.get_devices().await.map_err(|_| {
            FerricastError::Discovery("Cannot get network manager devices".to_string())
        })?;

        let mut p2p_proxy: Option<WifiP2pProxy> = None;

        for device_path in devices {
            let device = DeviceProxy::new(&connection, device_path.clone()).await.map_err(|_| {
        FerricastError::Discovery("Cannot create device proxy".to_string())
            })?;

            let device_type = device.device_type().await.map_err(|_| {
                FerricastError::Discovery("Cannot get device type".to_string())
            })?;

            let device_interface = device.interface().await.unwrap_or_default();

            tracing::debug!("Device {}, type {}, interface {}", device_path, device_type, device_interface);

            if device_type != 30 {
                continue;
            }

            p2p_proxy = Some(WifiP2pProxy::new(&connection, device_path).await.map_err(|_| {
                FerricastError::Discovery("Cannot create wifi p2p proxy".to_string())
            })?);

            tracing::info!("Using device {}", device_interface);

            break;        
        }

        if p2p_proxy.is_none() {
            return Err(FerricastError::Discovery("No P2P-compatible device was found. Check your drivers or network adapter".to_string()));
        }

        let proxy = p2p_proxy.take().unwrap();
        

        proxy.start_find(HashMap::new()).await.map_err(|_| {
            FerricastError::Discovery("Cannot start P2P-find".to_string())
        })?;

        let mut peer_found = proxy.receive_peer_added().await.map_err(|_| {
            FerricastError::Discovery("Cannot listen to signal peer added".to_string())
        })?;
               
        let handle = tokio::task::spawn(async move {
            tracing::info!("Miracast discovery started");

            while let Some(peer) = peer_found.next().await {
                let r: Result<()> = async {
                    let args = peer.args().map_err(|_| {
                        FerricastError::Discovery("Invalid signal".to_string())
                    })?;

                    let peer = P2pPeerProxy::new(&connection, args.path).await.map_err(|_| {
                        FerricastError::Discovery("Cannot create P2P Peer".to_string())
                    })?; 

                    let name = peer.name().await.map_err(|_| {
                        FerricastError::Discovery("Invalid miracast device, no name".to_string())
                    })?;

                    let wfd_ies = peer.WfdIEs().await.unwrap_or_default();


                    let hw_address =  peer.hw_address().await.map_err(|_| {
                        FerricastError::Discovery("Invalid miracast device, no hw address".to_string())
                    })?;

                    tracing::debug!(
                        "Peer {}, Hw Address {}, WFD IEs length: {}, data: {:02x?}",
                        name,
                        hw_address,
                        wfd_ies.len(),
                        wfd_ies
                    );

                    if !is_miracast_sink(&wfd_ies) {
                        return Ok(());
                    }



                    // TODO: PARSE WFD IES
                

                    tx.send(DiscoveryEvent::DeviceFound(Device {
                        id: Uuid::new_v4(),
                        name,
                        protocol: "miracast",
                        protocol_icon: MIRACAST_ICON,
                        addr: IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0)),
                        port: parse_wfd_rtsp_port(&wfd_ies),
                        model: None,
                        capabilities: DeviceCapabilities {
                            ..Default::default()
                        },
                        metadata: HashMap::new(),
                    })).await.map_err(|_| {
                        FerricastError::Discovery("Cannot send miracast device".to_string())
                    })?; 
                    
                    
                    Ok(())
                }.await;


                if let Err(err) = r {
                    tracing::error!("{}", err);
                }
                
            }
    
            tracing::error!("Miracast discovery thread terminated");
        });

        self.handle = Some(handle);
        self.wifi_p2p = Some(proxy);
        self.is_running.store(true, Ordering::SeqCst);

            Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        if !self.is_running.load(Ordering::SeqCst) {
            return Ok(());
        }

        let wifi_p2p = self.wifi_p2p.take().expect("Ferricast miracast discovery bug");

        
        wifi_p2p.stop_find().await.map_err(|_| {
            FerricastError::Discovery("Cannot stop wifi p2p proxy".to_string())
        })?;

        self.handle.take();

        self.is_running.store(false, Ordering::SeqCst);

        Ok(())
    }

    fn is_running(&self) -> bool {
       self.is_running.load(Ordering::SeqCst) 
    }
}

pub fn parse_wfd_rtsp_port(wfd_ies: &[u8]) -> u16 {
    if wfd_ies.len() >= 9 && wfd_ies[0] == 0x00 {
        let declared_len = ((wfd_ies[1] as u16) << 8) | (wfd_ies[2] as u16);
        if declared_len >= 6 {
            return ((wfd_ies[5] as u16) << 8) | (wfd_ies[6] as u16);
        }
    }
    if wfd_ies.len() >= 6 && wfd_ies[0] == 0x00 {
        return ((wfd_ies[4] as u16) << 8) | (wfd_ies[5] as u16);
    }

    if wfd_ies.len() >= 3 {
        return ((wfd_ies[1] as u16) << 8) | (wfd_ies[2] as u16);
    }

    7236
}

#[zbus::proxy(
    interface = "org.freedesktop.NetworkManager",
    default_service = "org.freedesktop.NetworkManager",
    default_path = "/org/freedesktop/NetworkManager"    
)]
trait NetworkManager {
    async fn get_devices(&self) -> zbus::Result<Vec<zvariant::OwnedObjectPath>>;
}

#[zbus::proxy(
    interface = "org.freedesktop.NetworkManager.WifiP2PPeer",
    default_service = "org.freedesktop.NetworkManager"
)]
trait P2pPeer {
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
trait Device {
    #[zbus(property)]
    fn device_type(&self) -> zbus::Result<u32>;

    #[zbus(property)]
    fn interface(&self) -> zbus::Result<String>;
}

#[zbus::proxy(
    interface = "org.freedesktop.NetworkManager.Device.WifiP2P",
    default_service = "org.freedesktop.NetworkManager",
)]
trait WifiP2p {
    async fn start_find(&self, args: HashMap<&str, zbus::zvariant::Value<'_>>) -> zbus::Result<()>;

    async fn stop_find(&self) -> zbus::Result<()>;

    #[zbus(property)]
    fn peers(&self) -> zbus::Result<Vec<zvariant::OwnedObjectPath>>;

    #[zbus(signal)]
    async fn peer_added(&self, path: zvariant::OwnedObjectPath);

}

fn is_miracast_sink(wfd_ies: &[u8]) -> bool {
    if wfd_ies.is_empty() {
        return false;
    }
    let first_byte = wfd_ies[0];
    tracing::debug!(
        "WFD IEs: {:02x?}, first_byte: 0x{:02x}",
        wfd_ies,
        first_byte
    );
    if first_byte == 0xdd && wfd_ies.len() >= 4 && wfd_ies[3] == 0x0a {
        return true;
    }
    first_byte == 0x00 || first_byte == 0x01 || first_byte == 0x06 || first_byte == 0x07
}
