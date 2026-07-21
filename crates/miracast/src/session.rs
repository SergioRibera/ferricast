use std::{collections::HashMap, hash::Hash, process::Command};

use ferricast_core::{CastSession, Device, EncodedFrame, FerricastError, Result, StreamConfig};
use zbus::Connection;
use zvariant::OwnedObjectPath;

use crate::dbus::{DeviceProxy, NetworkManagerProxy, P2pPeerProxy, WifiP2pProxy, WpaSupplicantProxy};

/// A Miracast (Wi-Fi Display) streaming session.
#[derive(Default)]
pub struct MiracastSession;

impl CastSession for MiracastSession {
    async fn connect(&mut self, device: &Device) -> Result<()> {
        let connection = Connection::system().await.map_err(|_| {
            FerricastError::Connection("Cannot (re)connect to dbus".to_string()) 
        })?;

        let wpa = WpaSupplicantProxy::new(&connection)
            .await
            .map_err(|_| {
                FerricastError::Connection("Canot connect to wpa".to_string())
            })?;

        wpa.interfaces().await.unwrap();




        let nm = NetworkManagerProxy::new(&connection).await.map_err(|_| {
            FerricastError::Connection("Cannot (re)connect to nm manager".to_string())
        })?;

        let devices = nm.get_devices().await.map_err(|_| {
            FerricastError::Discovery("Cannot get network manager devices".to_string())
        })?;

        let mut p2p_proxy: Option<WifiP2pProxy> = None;
        let mut d_path: Option<OwnedObjectPath> = None;

        for device_path in devices {
            let device = DeviceProxy::new(&connection, device_path.clone()).await.map_err(|_| {
        FerricastError::Connection("Cannot create device proxy".to_string())
            })?;

            let device_type = device.device_type().await.map_err(|_| {
                FerricastError::Connection("Cannot get device type".to_string())
            })?;

            let device_interface = device.interface().await.unwrap_or_default();

            tracing::debug!("Device {}, type {}, interface {}", device_path, device_type, device_interface);

            if device_type != 30 {
                continue;
            }

            d_path = Some(device_path.clone());

            p2p_proxy = Some(WifiP2pProxy::new(&connection, device_path).await.map_err(|_| {
                FerricastError::Connection("Cannot create wifi p2p proxy".to_string())
            })?);
            

            tracing::info!("Using device {}", device_interface);

            break;        
        }

        if p2p_proxy.is_none() {
            return Err(FerricastError::Connection("No P2P-compatible device was found. Check your drivers or network adapter".to_string()));
        }

        let p2p = p2p_proxy.take().unwrap();

        /*
        p2p.stop_find().await.map_err(|_| {
            FerricastError::Connection("Cannot stop discovery".to_string())
        })?;
        */

        let device_path = d_path.take().unwrap();

        let path = device.metadata.get("path").expect("Ferricast miracast discovery bug").as_str();
        let path = zvariant::ObjectPath::try_from(path)
            .map_err(|_| FerricastError::Connection("Zvariant error".to_string()))?;


        let peer = P2pPeerProxy::new(&connection, path.clone()).await.map_err(|_| {
            FerricastError::Connection("Cannot (re)connect to device".to_string())
        })?;

        let hw_address = peer.hw_address().await.map_err(|_| {
            FerricastError::Connection("Invalid Peer, no hw_address".to_string())
        })?;


        let device_obj_path = zvariant::ObjectPath::try_from(device_path.as_str())
            .map_err(|_| FerricastError::Connection("Zvariant error".to_string()))?;


        let mut p2p_props: HashMap<&str, zvariant::Value<'_>> = HashMap::new();

        p2p_props.insert(
            "peer",
            zvariant::Value::Str(zvariant::Str::from(&hw_address))
        );

        p2p_props.insert(
            "wfd-ies",
            zvariant::Value::Array(zvariant::Array::from(&vec![
                0,
                0, 0x06,
                0, 0x90,
                0x1C, 0x44,
                0x0, 0xC8
            ]))
        );

        let mut connection_props: HashMap<&str, zvariant::Value<'_>> = HashMap::new();

        connection_props.insert(
            "type",
            zvariant::Value::Str(zvariant::Str::from("wifi-p2p"))
        );

        connection_props.insert(
            "id",
            zvariant::Value::Str(zvariant::Str::from("ferricast")),
        );

        connection_props.insert(
            "autoconnect",
            zvariant::Value::Bool(false)
        );

        let mut ipv4_props: HashMap<&str, zvariant::Value<'_>> = HashMap::new();

        ipv4_props.insert("method", zvariant::Value::Str(zvariant::Str::from("auto")));

        ipv4_props.insert("never-default", zvariant::Value::Bool(true));


        let mut ipv6_props: HashMap<&str, zvariant::Value<'_>> = HashMap::new();

        ipv6_props.insert("method", zvariant::Value::Str(zvariant::Str::from("auto")));

        ipv6_props.insert("never-default", zvariant::Value::Bool(true));
        ipv6_props.insert("may-fail", zvariant::Value::Bool(true));

        let conn_config: HashMap<&str, HashMap<&str, zvariant::Value<'_>>> = HashMap::from([
            ("connection", connection_props),
            ("wifi-p2p", p2p_props),
            ("ipv4", ipv4_props),
            ("ipv6", ipv6_props),
        ]);

        let activation_options = HashMap::from([
            (
                "bind-activation",
                zvariant::Value::Str(zvariant::Str::from("dbus-client"))
            ),
            (
                "persist",
                zvariant::Value::Str(zvariant::Str::from("volatile")),
            ),
        ]);


        let (conn_path, active_conn_path, _) = nm
            .add_and_activate_connection2(conn_config, device_obj_path, path, activation_options)
            .await
            .map_err(|_| FerricastError::Connection("Cannot connect to miracast device".to_string()))?;

    
        tracing::info!("Connected to miracast device, (conn path: {:?}, active conn path: {:?})", conn_path, active_conn_path);

//        let a = wait_for_group().await;

  //      panic!("{:?}", a);


        Ok(())
    }

    async fn setup_stream(&mut self, config: &StreamConfig) -> Result<()> {
        Ok(())
    }

    async fn send_frame(&mut self, frame: &EncodedFrame) -> Result<()> {
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        Ok(())
    }

    fn is_alive(&self) -> bool {
        false
    }
}

#[derive(Debug)]
struct GroupStarted {
    interface: String,
    ip_addr: String,
    go_ip_addr: Option<String>
}

async fn wait_for_group() -> Option<GroupStarted> {
    for _ in 0..350 {
        let output = Command::new("journalctl")
            .args([
                "-u",
                "wpa_supplicant",
                "--since",
                "5 seconds ago",
                "--no-pager",
                "-o",
                "cat"
            ])
            .output()
            .ok()?;

            println!("{} {} {}", output.status, String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));

            let stdout = String::from_utf8_lossy(&output.stdout);



            if let Some(info) = stdout
                .lines()
                .rev()
                .find_map(parse_group_started_line)
            {
                return Some(info);        
            }

    
        
    }

    None
}

fn parse_group_started_line(line: &str) -> Option<GroupStarted> {
        let marker = "P2P-GROUP-STARTED ";
        let start = line.find(marker)? + marker.len();
        let data = &line[start..];
        let interface = data.split_whitespace().next()?.to_string();
        let ip_addr = data
            .split("ip_addr=")
            .nth(1)?
            .split_whitespace()
            .next()?
            .to_string();
        let go_ip_address = data
            .split("go_ip_addr=")
            .nth(1)?
            .split_whitespace()
            .next()?
            .to_string();

        Some(GroupStarted {
            interface,
            ip_addr,
            go_ip_addr: Some(go_ip_address),
        })
    }
