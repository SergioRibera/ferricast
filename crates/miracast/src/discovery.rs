use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr};

use bytes::Bytes;
use futures_lite::StreamExt;
use tokio::{sync::mpsc, task::JoinHandle};
use zbus::{Connection, zvariant::OwnedObjectPath};

use ferricast_core::{Device, DeviceCapabilities, Discovery, DiscoveryEvent, FerricastError, Result};
use uuid::Uuid;

use crate::dbus::{DeviceProxy, NM_DEVICE_TYPE_WIFI_P2P, NetworkManagerProxy, P2pPeerProxy, WifiP2pProxy};

const MIRACAST_ICON: Bytes = Bytes::from_static(include_bytes!("../../../assets/miracast.svg"));

pub struct MiracastDiscovery {
    handle: Option<JoinHandle<()>>,
    /// Stored so `stop()` can call `StopFind` without re-querying NM.
    p2p_device: Option<(Connection, OwnedObjectPath)>,
    running: bool,
}

impl Default for MiracastDiscovery {
    fn default() -> Self {
        Self {
            handle: None,
            p2p_device: None,
            running: false,
        }
    }
}

impl Discovery for MiracastDiscovery {
    const PROTOCOL: &'static str = "miracast";

    async fn start(&mut self, tx: mpsc::Sender<DiscoveryEvent>) -> Result<()> {
        let connection = Connection::system().await.map_err(|e| {
            FerricastError::Discovery(format!("Cannot connect to D-Bus system bus: {e}"))
        })?;

        let nm = NetworkManagerProxy::new(&connection).await.map_err(|e| {
            FerricastError::Discovery(format!("Cannot reach NetworkManager on D-Bus: {e}"))
        })?;

        let devices = nm.get_devices().await.map_err(|e| {
            FerricastError::Discovery(format!("NetworkManager.GetDevices failed: {e}"))
        })?;

        let mut p2p_device_path: Option<OwnedObjectPath> = None;

        for device_path in devices {
            let device = DeviceProxy::new(&connection, device_path.clone())
                .await
                .map_err(|e| {
                    FerricastError::Discovery(format!("Cannot create Device proxy: {e}"))
                })?;

            let device_type = device.device_type().await.map_err(|e| {
                FerricastError::Discovery(format!("Cannot read Device.DeviceType: {e}"))
            })?;

            if device_type != NM_DEVICE_TYPE_WIFI_P2P {
                continue;
            }

            let iface = device.interface().await.unwrap_or_default();
            tracing::info!(iface, path = %device_path, "found Wi-Fi P2P device");
            p2p_device_path = Some(device_path);
            break;
        }

        let device_path = p2p_device_path.ok_or_else(|| {
            FerricastError::Discovery(
                "No Wi-Fi P2P device found. Check your adapter and drivers.".into(),
            )
        })?;

        let p2p = WifiP2pProxy::new(&connection, device_path.clone())
            .await
            .map_err(|e| {
                FerricastError::Discovery(format!("Cannot create WifiP2P proxy: {e}"))
            })?;

        p2p.start_find(HashMap::new()).await.map_err(|e| {
            FerricastError::Discovery(format!("WifiP2P.StartFind failed: {e}"))
        })?;

        let mut peer_added = p2p.receive_peer_added().await.map_err(|e| {
            FerricastError::Discovery(format!("Cannot subscribe to PeerAdded signal: {e}"))
        })?;

        // Clone so both the task and self.p2p_device can own one.
        let conn_for_task = connection.clone();
        let device_path_str = device_path.to_string();

        let handle = tokio::task::spawn(async move {
            tracing::info!("Miracast discovery started");

            while let Some(signal) = peer_added.next().await {
                let result: Result<()> = async {
                    let args = signal.args().map_err(|e| {
                        FerricastError::Discovery(format!("Invalid PeerAdded signal args: {e}"))
                    })?;

                    let peer_path = args.path.clone();

                    let peer = P2pPeerProxy::new(&conn_for_task, peer_path.clone())
                        .await
                        .map_err(|e| {
                            FerricastError::Discovery(format!("Cannot create P2pPeer proxy: {e}"))
                        })?;

                    let name = peer.name().await.map_err(|e| {
                        FerricastError::Discovery(format!("P2pPeer.Name failed: {e}"))
                    })?;

                    let hw_address = peer.hw_address().await.map_err(|e| {
                        FerricastError::Discovery(format!("P2pPeer.HwAddress failed: {e}"))
                    })?;

                    let wfd_ies = peer.WfdIEs().await.unwrap_or_default();

                    tracing::debug!(
                        name,
                        hw_address,
                        wfd_ie_len = wfd_ies.len(),
                        wfd_ies = ?wfd_ies.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>(),
                        "P2P peer discovered"
                    );

                    if !is_miracast_sink(&wfd_ies) {
                        tracing::debug!(name, "peer is not a Miracast sink, skipping");
                        return Ok(());
                    }

                    let wfd_port = parse_wfd_rtsp_port(&wfd_ies);
                    let model = peer.model().await.ok();

                    let mut metadata = HashMap::new();
                    metadata.insert("path".to_string(), peer_path.to_string());
                    metadata.insert("device_path".to_string(), device_path_str.clone());
                    metadata.insert("hw_address".to_string(), hw_address.clone());

                    tx.send(DiscoveryEvent::DeviceFound(Device {
                        id: Uuid::new_v4(),
                        name,
                        protocol: "miracast",
                        protocol_icon: MIRACAST_ICON,
                        // IP is unknown until P2P connection is established.
                        addr: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
                        port: wfd_port,
                        model,
                        capabilities: DeviceCapabilities {
                            supports_video: true,
                            supports_screen_mirror: true,
                            ..Default::default()
                        },
                        metadata,
                    }))
                    .await
                    .map_err(|_| {
                        FerricastError::Discovery("DiscoveryEvent channel closed".into())
                    })?;

                    Ok(())
                }
                .await;

                if let Err(err) = result {
                    tracing::error!(%err, "Miracast peer handling error");
                }
            }

            tracing::warn!("Miracast PeerAdded signal stream ended");
        });

        self.handle = Some(handle);
        self.p2p_device = Some((connection, device_path));
        self.running = true;

        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        if !self.running {
            return Ok(());
        }

        if let Some((conn, device_path)) = self.p2p_device.take() {
            let p2p = WifiP2pProxy::new(&conn, device_path).await.map_err(|e| {
                FerricastError::Discovery(format!("Cannot create WifiP2P proxy for stop: {e}"))
            })?;
            p2p.stop_find().await.map_err(|e| {
                FerricastError::Discovery(format!("WifiP2P.StopFind failed: {e}"))
            })?;
        }

        if let Some(handle) = self.handle.take() {
            handle.abort();
        }

        self.running = false;
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running
    }
}

/// Parses the WFD RTSP port from WFD IE bytes (WFD subelement 0x00 = WFD Device Information).
///
/// The WFD IE layout (Wi-Fi Display spec §5.1.2):
/// `[subelement_id(1)] [length(2)] [device_info(2)] [session_mgmt_ctrl_port(2)] [throughput(2)]`
pub fn parse_wfd_rtsp_port(wfd_ies: &[u8]) -> u16 {
    // Full subelement format: ID(1) + len(2) + device_info(2) + port(2) + throughput(2) = 9 bytes minimum
    if wfd_ies.len() >= 9 && wfd_ies[0] == 0x00 {
        let subelement_len = u16::from_be_bytes([wfd_ies[1], wfd_ies[2]]);
        if subelement_len >= 6 {
            return u16::from_be_bytes([wfd_ies[5], wfd_ies[6]]);
        }
    }
    // Abbreviated form without subelement wrapper (some devices omit the outer IE header)
    if wfd_ies.len() >= 6 && wfd_ies[0] == 0x00 {
        return u16::from_be_bytes([wfd_ies[4], wfd_ies[5]]);
    }
    // Bare 3-byte form
    if wfd_ies.len() >= 3 {
        return u16::from_be_bytes([wfd_ies[1], wfd_ies[2]]);
    }
    // WFD default port (Wi-Fi Display spec §6.5)
    7236
}

/// Returns `true` when the WFD IEs indicate a Miracast sink (not a source or dual-role).
///
/// Checks the `Device Type` bits in WFD Device Information subelement (bits [1:0]):
/// - 0 = Source
/// - 1 = Primary Sink  ← what we want
/// - 2 = Secondary Sink
/// - 3 = Dual Role
fn is_miracast_sink(wfd_ies: &[u8]) -> bool {
    if wfd_ies.is_empty() {
        return false;
    }

    // WFD Vendor-Specific IE wrapping (0xDD OUI 50:6F:9A:0A)
    if wfd_ies[0] == 0xdd && wfd_ies.len() >= 7 {
        // bytes: DD len 50 6F 9A 0A [subelements...]
        let payload = &wfd_ies[6..];
        return is_miracast_sink(payload);
    }

    // WFD Device Information subelement (ID=0x00)
    if wfd_ies[0] == 0x00 && wfd_ies.len() >= 5 {
        // device_info is bytes [3..5] (after ID + 2-byte length)
        let device_info = u16::from_be_bytes([wfd_ies[3], wfd_ies[4]]);
        let device_type = device_info & 0x0003;
        // 1 = Primary Sink, 2 = Secondary Sink, 3 = Dual Role
        tracing::debug!(device_type, device_info, "WFD Device Information");
        return device_type != 0;
    }

    // Fallback: accept any non-empty WFD IE (permissive for unknown formats)
    let first = wfd_ies[0];
    first == 0x00 || first == 0x01 || first == 0x06 || first == 0x07
}

