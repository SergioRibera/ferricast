use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use serde::Deserialize;
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, trace, warn};
use uuid::Uuid;

use ferricast_core::{Device, DeviceCapabilities, DiscoveryEvent, FerricastError, Result};

const SSDP_MULTICAST_ADDR: Ipv4Addr = Ipv4Addr::new(239, 255, 255, 250);
const SSDP_PORT: u16 = 1900;

const DIAL_SEARCH_TARGET: &str = "urn:dial-multiscreen-org:service:dial:1";
const DIAL_ICON: Bytes = Bytes::from_static(include_bytes!("../../../assets/dial.svg"));

const SSDP_RESPONSE_TIMEOUT: Duration = Duration::from_secs(5);
const SSDP_SCAN_INTERVAL: Duration = Duration::from_secs(30);
const SSDP_MAX_RESPONSE_SIZE: usize = 4096;

#[derive(Debug, Deserialize)]
#[serde(rename = "root")]
struct DeviceDescriptionRoot {
    device: DeviceDescriptionDevice,
}

#[derive(Debug, Deserialize)]
struct DeviceDescriptionDevice {
    #[serde(rename = "friendlyName", default)]
    friendly_name: String,
    #[serde(rename = "modelName", default)]
    model_name: Option<String>,
    #[serde(rename = "manufacturer", default)]
    manufacturer: Option<String>,
    #[serde(rename = "UDN", default)]
    udn: Option<String>,
}

#[derive(Debug, Clone)]
struct SsdpResponse {
    location: String,
    usn: Option<String>,
}

impl SsdpResponse {
    fn parse(raw: &[u8]) -> Option<Self> {
        let text = std::str::from_utf8(raw).ok()?;
        let mut location: Option<String> = None;
        let mut usn: Option<String> = None;

        for line in text.lines() {
            let line = line.trim();
            if let Some(value) = header_value(line, "LOCATION") {
                location = Some(value.to_owned());
            } else if let Some(value) = header_value(line, "USN") {
                usn = Some(value.to_owned());
            }
        }

        let location = location?;
        Some(Self { location, usn })
    }
}

fn header_value<'a>(line: &'a str, header_name: &str) -> Option<&'a str> {
    let colon = line.find(':')?;
    let name = &line[..colon];
    if name.eq_ignore_ascii_case(header_name) {
        Some(line[colon + 1..].trim())
    } else {
        None
    }
}

pub struct DialDiscovery {
    shutdown_tx: Option<watch::Sender<bool>>,
    task_handle: Option<JoinHandle<()>>,
    running: bool,
    known_devices: Arc<tokio::sync::Mutex<HashMap<String, Uuid>>>,
    http_client: reqwest::Client,
}

impl DialDiscovery {
    pub fn new() -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build reqwest client");

        Self {
            shutdown_tx: None,
            task_handle: None,
            running: false,
            known_devices: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            http_client,
        }
    }

    fn build_msearch_request() -> Vec<u8> {
        let request = format!(
            "M-SEARCH * HTTP/1.1\r\n\
             HOST: {SSDP_MULTICAST_ADDR}:{SSDP_PORT}\r\n\
             MAN: \"ssdp:discover\"\r\n\
             MX: 3\r\n\
             ST: {DIAL_SEARCH_TARGET}\r\n\
             USER-AGENT: ferricast/0.1 UPnP/1.1\r\n\
             \r\n"
        );
        request.into_bytes()
    }

    async fn send_msearch(socket: &UdpSocket) -> Result<()> {
        let payload = Self::build_msearch_request();
        let dest = SocketAddrV4::new(SSDP_MULTICAST_ADDR, SSDP_PORT);
        socket
            .send_to(&payload, SocketAddr::V4(dest))
            .await
            .map_err(|e| FerricastError::Discovery(format!("failed to send M-SEARCH: {e}")))?;
        debug!("sent SSDP M-SEARCH for DIAL devices");
        Ok(())
    }

    async fn fetch_device_description(
        client: &reqwest::Client,
        location_url: &str,
    ) -> Result<(DeviceDescriptionRoot, String)> {
        let resp = client
            .get(location_url)
            .send()
            .await
            .map_err(|e| FerricastError::Discovery(format!("HTTP GET {location_url}: {e}")))?;

        let app_url = resp
            .headers()
            .get("Application-URL")
            .or_else(|| resp.headers().get("application-url"))
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim_end_matches('/').to_owned())
            .ok_or_else(|| {
                FerricastError::Discovery(format!(
                    "no Application-URL header in response from {location_url}"
                ))
            })?;

        let body = resp.text().await.map_err(|e| {
            FerricastError::Discovery(format!("reading body from {location_url}: {e}"))
        })?;

        let desc: DeviceDescriptionRoot = quick_xml::de::from_str(&body).map_err(|e| {
            FerricastError::Discovery(format!("parsing device XML from {location_url}: {e}"))
        })?;

        Ok((desc, app_url))
    }

    fn build_device(
        desc: &DeviceDescriptionRoot,
        app_url: &str,
        addr: std::net::IpAddr,
        port: u16,
        usn: Option<&str>,
    ) -> Device {
        let mut metadata = HashMap::new();
        metadata.insert("application_url".to_owned(), app_url.to_owned());
        if let Some(manufacturer) = &desc.device.manufacturer {
            metadata.insert("manufacturer".to_owned(), manufacturer.clone());
        }
        if let Some(usn) = usn {
            metadata.insert("usn".to_owned(), usn.to_owned());
        }

        let id = desc
            .device
            .udn
            .as_deref()
            .and_then(|udn| {
                let raw = udn.strip_prefix("uuid:").unwrap_or(udn);
                Uuid::parse_str(raw).ok()
            })
            .unwrap_or_else(|| {
                let seed = usn.unwrap_or(app_url);
                let hash = {
                    use std::hash::{Hash, Hasher};
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    seed.hash(&mut hasher);
                    hasher.finish()
                };
                let bytes = hash.to_le_bytes();
                let mut uuid_bytes = [0u8; 16];
                uuid_bytes[..8].copy_from_slice(&bytes);
                uuid_bytes[8..16].copy_from_slice(&bytes);
                uuid_bytes[6] = (uuid_bytes[6] & 0x0F) | 0x40;
                uuid_bytes[8] = (uuid_bytes[8] & 0x3F) | 0x80;
                Uuid::from_bytes(uuid_bytes)
            });

        Device {
            id,
            name: desc.device.friendly_name.clone(),
            protocol: "dial",
            protocol_icon: DIAL_ICON,
            addr,
            port,
            model: desc.device.model_name.clone(),
            capabilities: DeviceCapabilities {
                supports_audio: true,
                supports_video: true,
                supports_screen_mirror: false,
                max_width: None,
                max_height: None,
                supported_codecs: vec![ferricast_core::Codec::H264],
                ..Default::default()
            },
            metadata,
        }
    }
}

impl Default for DialDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl ferricast_core::Discovery for DialDiscovery {
    const PROTOCOL: &'static str = "dial";

    async fn start(&mut self, tx: mpsc::Sender<DiscoveryEvent>) -> Result<()> {
        if self.running {
            warn!("DIAL discovery is already running");
            return Ok(());
        }

        let socket = UdpSocket::bind(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0))
            .await
            .map_err(|e| FerricastError::Discovery(format!("bind UDP socket: {e}")))?;

        socket
            .join_multicast_v4(SSDP_MULTICAST_ADDR, Ipv4Addr::UNSPECIFIED)
            .map_err(|e| FerricastError::Discovery(format!("join multicast group: {e}")))?;

        let socket = Arc::new(socket);

        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        self.shutdown_tx = Some(shutdown_tx);

        let known = Arc::clone(&self.known_devices);
        let client = self.http_client.clone();

        let handle = tokio::spawn(async move {
            if let Err(e) = scan_loop(socket, tx, shutdown_rx, known, client).await {
                error!("DIAL scan loop terminated with error: {e}");
            }
        });

        self.task_handle = Some(handle);
        self.running = true;
        info!("DIAL discovery started");
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(true);
        }
        if let Some(handle) = self.task_handle.take() {
            let _ = handle.await;
        }
        self.running = false;
        info!("DIAL discovery stopped");
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running
    }
}

async fn scan_loop(
    socket: Arc<UdpSocket>,
    tx: mpsc::Sender<DiscoveryEvent>,
    mut shutdown: watch::Receiver<bool>,
    known_devices: Arc<tokio::sync::Mutex<HashMap<String, Uuid>>>,
    client: reqwest::Client,
) -> Result<()> {
    loop {
        if *shutdown.borrow() {
            debug!("DIAL scan loop: shutdown requested");
            break;
        }

        if let Err(e) = DialDiscovery::send_msearch(&socket).await {
            let _ = tx
                .send(DiscoveryEvent::Error {
                    protocol: "dial",
                    message: format!("M-SEARCH send failed: {e}"),
                })
                .await;
        }

        let mut seen_locations: HashMap<String, (SocketAddr, Option<String>)> = HashMap::new();
        let deadline = tokio::time::Instant::now() + SSDP_RESPONSE_TIMEOUT;

        loop {
            let mut buf = vec![0u8; SSDP_MAX_RESPONSE_SIZE];
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }

            tokio::select! {
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        debug!("DIAL scan loop: shutdown during recv");
                        return Ok(());
                    }
                }
                result = tokio::time::timeout(remaining, socket.recv_from(&mut buf)) => {
                    match result {
                        Ok(Ok((len, src))) => {
                            trace!("received {len} bytes from {src}");
                            if let Some(resp) = SsdpResponse::parse(&buf[..len]) {
                                seen_locations
                                    .entry(resp.location.clone())
                                    .or_insert((src, resp.usn));
                            }
                        }
                        Ok(Err(e)) => {
                            warn!("UDP recv error: {e}");
                        }
                        Err(_) => {
                            break;
                        }
                    }
                }
            }
        }

        let mut this_scan_ids: Vec<String> = Vec::new();

        for (location, (src_addr, usn)) in &seen_locations {
            this_scan_ids.push(location.clone());

            {
                let known = known_devices.lock().await;
                if known.contains_key(location) {
                    trace!("already known: {location}");
                    continue;
                }
            }

            match DialDiscovery::fetch_device_description(&client, location).await {
                Ok((desc, app_url)) => {
                    let addr_ip = src_addr.ip();
                    let port = url::Url::parse(&app_url)
                        .map(|u| u.port().unwrap_or(80))
                        .unwrap_or(80);

                    let device =
                        DialDiscovery::build_device(&desc, &app_url, addr_ip, port, usn.as_deref());
                    let dev_id = device.id;

                    info!(
                        name = %device.name,
                        addr = %device.addr,
                        "discovered DIAL device"
                    );

                    {
                        let mut known = known_devices.lock().await;
                        known.insert(location.clone(), dev_id);
                    }

                    if tx.send(DiscoveryEvent::DeviceFound(device)).await.is_err() {
                        debug!("event channel closed, stopping scan");
                        return Ok(());
                    }
                }
                Err(e) => {
                    warn!(location, "failed to fetch device description: {e}");
                }
            }
        }

        {
            let mut known = known_devices.lock().await;
            let lost: Vec<(String, Uuid)> = known
                .iter()
                .filter(|(loc, _)| !this_scan_ids.contains(loc))
                .map(|(loc, id)| (loc.clone(), *id))
                .collect();
            for (loc, id) in lost {
                info!(id = %id, "DIAL device lost");
                known.remove(&loc);
                if tx.send(DiscoveryEvent::DeviceLost(id)).await.is_err() {
                    return Ok(());
                }
            }
        }

        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    break;
                }
            }
            _ = tokio::time::sleep(SSDP_SCAN_INTERVAL) => {}
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ssdp_response() {
        let raw = b"HTTP/1.1 200 OK\r\n\
            CACHE-CONTROL: max-age=1800\r\n\
            LOCATION: http://192.168.1.50:8008/ssdp/device-desc.xml\r\n\
            ST: urn:dial-multiscreen-org:service:dial:1\r\n\
            USN: uuid:abcdefab-1234-5678-9abc-def012345678::urn:dial-multiscreen-org:service:dial:1\r\n\
            \r\n";
        let resp = SsdpResponse::parse(raw).expect("should parse");
        assert_eq!(
            resp.location,
            "http://192.168.1.50:8008/ssdp/device-desc.xml"
        );
        assert!(resp.usn.is_some());
    }

    #[test]
    fn msearch_packet_is_well_formed() {
        let pkt = DialDiscovery::build_msearch_request();
        let text = String::from_utf8(pkt).unwrap();
        assert!(text.starts_with("M-SEARCH * HTTP/1.1\r\n"));
        assert!(text.contains(DIAL_SEARCH_TARGET));
        assert!(text.contains("MAN: \"ssdp:discover\""));
        assert!(text.ends_with("\r\n\r\n"));
    }

    #[test]
    fn header_value_case_insensitive() {
        assert_eq!(
            header_value("Location: http://example.com", "LOCATION"),
            Some("http://example.com")
        );
        assert_eq!(
            header_value("LOCATION: http://example.com", "location"),
            Some("http://example.com")
        );
        assert_eq!(header_value("ST: something", "LOCATION"), None);
    }
}
