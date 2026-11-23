//! Wi-Fi Display Information Element codec.
//!
//! Carried as a vendor-specific IE in P2P Probe Request/Response, Beacon,
//! GO Negotiation, Invitation, and Provision Discovery frames:
//!
//! ```text
//! 0xDD <len> 0x50 0x6F 0x9A 0x0A <subelements...>
//! ```
//!
//! OUI `50:6F:9A` = Wi-Fi Alliance, OUI type `0x0A` = Wi-Fi Display.
//! Each subelement is `<id:1> <len:2 BE> <payload>`.
//!
//! On Linux the serialised byte vector is written to
//! `fi.w1.wpa_supplicant1`'s `WFDIEs: ay` property; wpa_supplicant then
//! injects it into the beacons / probes / negotiation frames it
//! generates. Reference: `miraclecast/src/wifi/wifid-supplicant.c`.

use bytes::{BufMut, Bytes, BytesMut};
use thiserror::Error;

/// WFD IE vendor OUI: Wi-Fi Alliance.
pub const WIFI_ALLIANCE_OUI: [u8; 3] = [0x50, 0x6F, 0x9A];
/// OUI type for Wi-Fi Display.
pub const WFD_OUI_TYPE: u8 = 0x0A;
/// Element ID for vendor-specific IEs.
pub const VENDOR_SPECIFIC_ELEMENT_ID: u8 = 0xDD;

/// Subelement identifiers defined in WFD v1.1 §5.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubelementId {
    DeviceInformation = 0,
    AssociatedBssid = 1,
    CoupledSink = 2,
    ExtendedCapability = 3,
    LocalIpAddress = 4,
    SessionInformation = 5,
    AlternateMacAddress = 6,
}

#[derive(Debug, Error)]
pub enum IeError {
    #[error("truncated WFD IE (needed {needed} more bytes)")]
    Truncated { needed: usize },
    #[error("wrong vendor OUI: got {got:02x?}, expected {:02x?}", WIFI_ALLIANCE_OUI)]
    WrongOui { got: [u8; 3] },
    #[error("wrong OUI type: got {got:02x}, expected {:02x}", WFD_OUI_TYPE)]
    WrongOuiType { got: u8 },
    #[error("unknown subelement id {0}")]
    UnknownSubelement(u8),
}

/// Device role bits in the Device Information bitfield.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum DeviceRole {
    Source = 0b00,
    PrimarySink = 0b01,
    SecondarySink = 0b10,
    /// Dual-role: can act as source or sink in the same session negotiation.
    DualRole = 0b11,
}

/// WFD Device Information subelement (subelement 0, 6 bytes payload).
///
/// Bytes 0..1 bitfield (BE):
/// - bits[1:0] device role (see [`DeviceRole`])
/// - bits[3:2] coupled-sink support
/// - bit 4 Session Available
/// - bit 5 WSD (Service Discovery)
/// - bit 6 Preferred Connectivity (0 = P2P, 1 = TDLS)
/// - bit 7 Content Protection (HDCP2.x)
/// - bit 8 Time Sync
/// - bit 9 Primary Sink: audio-only
/// - bit 10 Source: no audio
///
/// Bytes 2..3 BE: session mgmt control port (usually RTSP 7236).
/// Bytes 4..5 BE: max avg throughput Mbps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceInformation {
    pub role: DeviceRole,
    pub coupled_sink_source: bool,
    pub coupled_sink_sink: bool,
    pub session_available: bool,
    pub service_discovery: bool,
    pub prefers_tdls: bool,
    pub content_protection: bool,
    pub time_sync: bool,
    pub audio_only_primary_sink: bool,
    pub source_without_audio: bool,
    pub session_mgmt_port: u16,
    pub max_throughput_mbps: u16,
}

impl DeviceInformation {
    /// Pragmatic defaults for a Linux source: dual-role (so the peer can
    /// pick either direction), session available, HDCP off (we ship
    /// `wfd_content_protection: none`), RTSP on 7236, 40 Mbps.
    #[must_use]
    pub const fn linux_dual_role() -> Self {
        Self {
            role: DeviceRole::DualRole,
            coupled_sink_source: false,
            coupled_sink_sink: false,
            session_available: true,
            service_discovery: false,
            prefers_tdls: false,
            content_protection: false,
            time_sync: false,
            audio_only_primary_sink: false,
            source_without_audio: false,
            session_mgmt_port: 7236,
            max_throughput_mbps: 40,
        }
    }

    /// Primary-sink defaults: receiver only, session available, HDCP off.
    #[must_use]
    pub const fn primary_sink() -> Self {
        Self {
            role: DeviceRole::PrimarySink,
            ..Self::linux_dual_role()
        }
    }

    #[must_use]
    pub const fn to_bits(self) -> u16 {
        let mut b: u16 = self.role as u16;
        let coupled = (self.coupled_sink_source as u16) | ((self.coupled_sink_sink as u16) << 1);
        b |= coupled << 2;
        b |= (self.session_available as u16) << 4;
        b |= (self.service_discovery as u16) << 5;
        b |= (self.prefers_tdls as u16) << 6;
        b |= (self.content_protection as u16) << 7;
        b |= (self.time_sync as u16) << 8;
        b |= (self.audio_only_primary_sink as u16) << 9;
        b |= (self.source_without_audio as u16) << 10;
        b
    }

    fn write(self, out: &mut BytesMut) {
        out.put_u8(SubelementId::DeviceInformation as u8);
        out.put_u16(6);
        out.put_u16(self.to_bits());
        out.put_u16(self.session_mgmt_port);
        out.put_u16(self.max_throughput_mbps);
    }
}

/// Fully-built WFD IE.
#[derive(Debug, Clone, Default)]
pub struct WfdIe {
    pub device_information: Option<DeviceInformation>,
    pub associated_bssid: Option<[u8; 6]>,
    pub alternate_mac: Option<[u8; 6]>,
    /// Caller-supplied subelements the typed fields don't cover.
    pub extra: Vec<(SubelementId, Bytes)>,
}

impl WfdIe {
    #[must_use]
    pub fn new(device_information: DeviceInformation) -> Self {
        Self {
            device_information: Some(device_information),
            ..Self::default()
        }
    }

    /// Serialize the subelement sequence (no outer IE header).
    ///
    /// This is what gets written to wpa_supplicant's `WFDIEs: ay`
    /// property — supplicant wraps it in the vendor-specific IE header
    /// itself when it emits the beacon / probe.
    #[must_use]
    pub fn to_subelements(&self) -> Bytes {
        let mut out = BytesMut::new();
        if let Some(info) = self.device_information {
            info.write(&mut out);
        }
        if let Some(bssid) = self.associated_bssid {
            out.put_u8(SubelementId::AssociatedBssid as u8);
            out.put_u16(6);
            out.put_slice(&bssid);
        }
        if let Some(mac) = self.alternate_mac {
            out.put_u8(SubelementId::AlternateMacAddress as u8);
            out.put_u16(6);
            out.put_slice(&mac);
        }
        for (id, payload) in &self.extra {
            out.put_u8(*id as u8);
            let len = u16::try_from(payload.len()).unwrap_or(u16::MAX);
            out.put_u16(len);
            out.put_slice(payload);
        }
        out.freeze()
    }

    /// Serialize the whole IE including the 0xDD header + OUI (what
    /// actually sits on the air inside a probe/beacon).
    #[must_use]
    pub fn to_ie(&self) -> Bytes {
        let sub = self.to_subelements();
        let mut out = BytesMut::with_capacity(2 + 4 + sub.len());
        out.put_u8(VENDOR_SPECIFIC_ELEMENT_ID);
        let len = u8::try_from(4 + sub.len()).unwrap_or(u8::MAX);
        out.put_u8(len);
        out.put_slice(&WIFI_ALLIANCE_OUI);
        out.put_u8(WFD_OUI_TYPE);
        out.put_slice(&sub);
        out.freeze()
    }

    /// Parse the subelement sequence out of either a raw subelement
    /// stream (what we write to supplicant) or a full outer IE (what we
    /// read back from a peer's `IEs` property).
    pub fn parse(bytes: &[u8]) -> Result<Self, IeError> {
        let mut rest = bytes;

        // Strip outer IE header if present.
        if rest.first() == Some(&VENDOR_SPECIFIC_ELEMENT_ID) {
            if rest.len() < 6 {
                return Err(IeError::Truncated { needed: 6 });
            }
            let total_len = rest[1] as usize;
            if rest.len() < 2 + total_len {
                return Err(IeError::Truncated {
                    needed: 2 + total_len - rest.len(),
                });
            }
            let oui: [u8; 3] = rest[2..5].try_into().expect("checked length above");
            if oui != WIFI_ALLIANCE_OUI {
                return Err(IeError::WrongOui { got: oui });
            }
            let oui_type = rest[5];
            if oui_type != WFD_OUI_TYPE {
                return Err(IeError::WrongOuiType { got: oui_type });
            }
            rest = &rest[6..2 + total_len];
        }

        let mut out = Self::default();
        while !rest.is_empty() {
            if rest.len() < 3 {
                return Err(IeError::Truncated { needed: 3 - rest.len() });
            }
            let id = rest[0];
            let len = u16::from_be_bytes([rest[1], rest[2]]) as usize;
            if rest.len() < 3 + len {
                return Err(IeError::Truncated {
                    needed: 3 + len - rest.len(),
                });
            }
            let payload = &rest[3..3 + len];
            match id {
                x if x == SubelementId::DeviceInformation as u8 => {
                    if payload.len() != 6 {
                        return Err(IeError::Truncated {
                            needed: 6 - payload.len(),
                        });
                    }
                    let bits = u16::from_be_bytes([payload[0], payload[1]]);
                    let role = match bits & 0x03 {
                        0 => DeviceRole::Source,
                        1 => DeviceRole::PrimarySink,
                        2 => DeviceRole::SecondarySink,
                        _ => DeviceRole::DualRole,
                    };
                    out.device_information = Some(DeviceInformation {
                        role,
                        coupled_sink_source: bits & (1 << 2) != 0,
                        coupled_sink_sink: bits & (1 << 3) != 0,
                        session_available: bits & (1 << 4) != 0,
                        service_discovery: bits & (1 << 5) != 0,
                        prefers_tdls: bits & (1 << 6) != 0,
                        content_protection: bits & (1 << 7) != 0,
                        time_sync: bits & (1 << 8) != 0,
                        audio_only_primary_sink: bits & (1 << 9) != 0,
                        source_without_audio: bits & (1 << 10) != 0,
                        session_mgmt_port: u16::from_be_bytes([payload[2], payload[3]]),
                        max_throughput_mbps: u16::from_be_bytes([payload[4], payload[5]]),
                    });
                }
                x if x == SubelementId::AssociatedBssid as u8 => {
                    if payload.len() == 6 {
                        out.associated_bssid = Some(payload.try_into().expect("len 6"));
                    }
                }
                x if x == SubelementId::AlternateMacAddress as u8 => {
                    if payload.len() == 6 {
                        out.alternate_mac = Some(payload.try_into().expect("len 6"));
                    }
                }
                other => {
                    let sid = match other {
                        2 => SubelementId::CoupledSink,
                        3 => SubelementId::ExtendedCapability,
                        4 => SubelementId::LocalIpAddress,
                        5 => SubelementId::SessionInformation,
                        _ => return Err(IeError::UnknownSubelement(other)),
                    };
                    out.extra.push((sid, Bytes::copy_from_slice(payload)));
                }
            }
            rest = &rest[3 + len..];
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_info_round_trip_outer_ie() {
        let info = DeviceInformation::primary_sink();
        let ie = WfdIe::new(info);
        let bytes = ie.to_ie();
        assert_eq!(bytes[0], VENDOR_SPECIFIC_ELEMENT_ID);
        let parsed = WfdIe::parse(&bytes).expect("parse");
        assert_eq!(parsed.device_information, Some(info));
    }

    #[test]
    fn subelements_only_round_trip() {
        let info = DeviceInformation::linux_dual_role();
        let ie = WfdIe::new(info);
        let bytes = ie.to_subelements();
        let parsed = WfdIe::parse(&bytes).expect("parse");
        assert_eq!(parsed.device_information, Some(info));
    }

    #[test]
    fn device_role_bits_match_spec() {
        let mut info = DeviceInformation::primary_sink();
        info.session_available = true;
        info.content_protection = false;
        // role = 01, session_available = bit 4
        assert_eq!(info.to_bits(), 0b0000_0001_0001);
    }
}
