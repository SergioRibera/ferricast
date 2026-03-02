use std::collections::HashMap;
use std::net::IpAddr;

use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct Device {
    pub id: Uuid,
    pub name: String,
    pub protocol: &'static str,
    pub addr: IpAddr,
    pub port: u16,
    pub model: Option<String>,
    pub capabilities: DeviceCapabilities,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Default)]
pub struct DeviceCapabilities {
    pub supports_audio: bool,
    pub supports_video: bool,
    pub supports_screen_mirror: bool,
    pub max_width: Option<u32>,
    pub max_height: Option<u32>,
    pub supported_codecs: Vec<crate::Codec>,
}

#[derive(Debug, Clone)]
pub enum DiscoveryEvent {
    DeviceFound(Device),
    DeviceLost(Uuid),
    Error {
        protocol: &'static str,
        message: String,
    },
}
