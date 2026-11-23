//! `wfd_*` parameter codec.
//!
//! Each wire-level parameter the Wi-Fi Display RTSP messages carry is
//! modelled as a typed value with `Display` / `FromStr` round-trip. The
//! grammar is lowercase ASCII-hex, fixed-width, space-separated. `none`
//! is the sentinel "not supported" for most parameters.
//!
//! Reference: Wi-Fi Alliance *Wi-Fi Display Technical Specification*
//! v1.1 §6. Byte layouts are called out per field below.

use std::fmt;
use std::str::FromStr;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParamError {
    #[error("empty parameter value")]
    Empty,
    #[error("malformed `{param}`: {reason}")]
    Malformed {
        param: &'static str,
        reason: &'static str,
    },
    #[error("unknown parameter: {0}")]
    Unknown(String),
    #[error("bad hex in `{param}` at `{token}`")]
    Hex { param: &'static str, token: String },
    #[error("value {value} out of range for `{param}`")]
    OutOfRange { param: &'static str, value: u64 },
}

/// H.264 profile bitmap in `wfd_video_formats`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct H264Profiles {
    pub cbp: bool,
    pub chp: bool,
}

impl H264Profiles {
    #[must_use]
    pub const fn cbp_only() -> Self {
        Self {
            cbp: true,
            chp: false,
        }
    }

    #[must_use]
    pub const fn to_bits(self) -> u8 {
        (self.cbp as u8) | ((self.chp as u8) << 1)
    }

    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        Self {
            cbp: bits & 0x01 != 0,
            chp: bits & 0x02 != 0,
        }
    }
}

/// H.264 level bitmap in `wfd_video_formats`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct H264Levels {
    pub l3_1: bool,
    pub l3_2: bool,
    pub l4_0: bool,
    pub l4_1: bool,
    pub l4_2: bool,
}

impl H264Levels {
    #[must_use]
    pub const fn baseline() -> Self {
        Self {
            l3_1: true,
            l3_2: false,
            l4_0: false,
            l4_1: false,
            l4_2: false,
        }
    }

    #[must_use]
    pub const fn to_bits(self) -> u8 {
        (self.l3_1 as u8)
            | ((self.l3_2 as u8) << 1)
            | ((self.l4_0 as u8) << 2)
            | ((self.l4_1 as u8) << 3)
            | ((self.l4_2 as u8) << 4)
    }

    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        Self {
            l3_1: bits & 0x01 != 0,
            l3_2: bits & 0x02 != 0,
            l4_0: bits & 0x04 != 0,
            l4_1: bits & 0x08 != 0,
            l4_2: bits & 0x10 != 0,
        }
    }
}

/// Which table a native-resolution index refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionTable {
    Cea = 0,
    Vesa = 1,
    Hh = 2,
}

/// Native-resolution reference inside `wfd_video_formats`.
///
/// `bits[2:0]` = table, `bits[7:3]` = index in that table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeResolution {
    pub table: ResolutionTable,
    pub index: u8,
}

impl NativeResolution {
    #[must_use]
    pub const fn to_byte(self) -> u8 {
        (self.table as u8) | (self.index << 3)
    }

    #[must_use]
    pub const fn from_byte(b: u8) -> Self {
        let table = match b & 0x07 {
            0 => ResolutionTable::Cea,
            1 => ResolutionTable::Vesa,
            _ => ResolutionTable::Hh,
        };
        Self {
            table,
            index: b >> 3,
        }
    }
}

/// `wfd_video_formats` — one codec entry.
///
/// Byte order on the wire (ASCII hex):
/// `NAT:2 PDMS:2 PROF:2 LVL:2 CEA:8 VESA:8 HH:8 lat:2 min_slice:4 slice_enc:4 frame_skip:2 max_hres:4|none max_vres:4|none`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoFormat {
    pub native: NativeResolution,
    pub preferred_display_mode: bool,
    pub profiles: H264Profiles,
    pub levels: H264Levels,
    pub cea: u32,
    pub vesa: u32,
    pub hh: u32,
    /// Minimum latency in units of 5 ms (0 = unspecified).
    pub latency: u8,
    pub min_slice_size: u16,
    pub slice_encoding: u16,
    pub frame_skipping: u8,
    pub max_hres: Option<u16>,
    pub max_vres: Option<u16>,
}

impl VideoFormat {
    /// Minimum-compliant default: 1280x720p30 CBP level 3.1.
    #[must_use]
    pub const fn baseline_720p30() -> Self {
        Self {
            // CEA bit 5 = 1280x720p30 → index 5, table CEA.
            native: NativeResolution {
                table: ResolutionTable::Cea,
                index: 5,
            },
            preferred_display_mode: false,
            profiles: H264Profiles::cbp_only(),
            levels: H264Levels::baseline(),
            cea: 1 << 5,
            vesa: 0,
            hh: 0,
            latency: 0,
            min_slice_size: 0,
            slice_encoding: 0,
            frame_skipping: 0,
            max_hres: None,
            max_vres: None,
        }
    }

    /// Richer default for a modern PC source: 1080p30 CBP level 4.0,
    /// plus 720p30/720p60/1080p30/1080p60 bits in the CEA bitmap.
    #[must_use]
    pub const fn pc_source_1080p30() -> Self {
        Self {
            native: NativeResolution {
                table: ResolutionTable::Cea,
                index: 7,
            },
            preferred_display_mode: false,
            profiles: H264Profiles::cbp_only(),
            levels: H264Levels {
                l3_1: true,
                l3_2: true,
                l4_0: true,
                l4_1: false,
                l4_2: false,
            },
            // bits 5,6,7,8 of CEA: 720p30, 720p60, 1080p30, 1080p60.
            cea: (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8),
            vesa: 0,
            hh: 0,
            latency: 0,
            min_slice_size: 0,
            slice_encoding: 0,
            frame_skipping: 0,
            max_hres: None,
            max_vres: None,
        }
    }
}

impl fmt::Display for VideoFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let max_hres = self
            .max_hres
            .map_or_else(|| "none".to_string(), |v| format!("{v:04x}"));
        let max_vres = self
            .max_vres
            .map_or_else(|| "none".to_string(), |v| format!("{v:04x}"));
        write!(
            f,
            "{:02x} {:02x} {:02x} {:02x} {:08x} {:08x} {:08x} {:02x} {:04x} {:04x} {:02x} {} {}",
            self.native.to_byte(),
            self.preferred_display_mode as u8,
            self.profiles.to_bits(),
            self.levels.to_bits(),
            self.cea,
            self.vesa,
            self.hh,
            self.latency,
            self.min_slice_size,
            self.slice_encoding,
            self.frame_skipping,
            max_hres,
            max_vres,
        )
    }
}

fn parse_hex_u32(param: &'static str, token: &str) -> Result<u32, ParamError> {
    u32::from_str_radix(token, 16).map_err(|_| ParamError::Hex {
        param,
        token: token.to_string(),
    })
}

fn parse_hex_u16(param: &'static str, token: &str) -> Result<u16, ParamError> {
    u16::from_str_radix(token, 16).map_err(|_| ParamError::Hex {
        param,
        token: token.to_string(),
    })
}

fn parse_hex_u8(param: &'static str, token: &str) -> Result<u8, ParamError> {
    u8::from_str_radix(token, 16).map_err(|_| ParamError::Hex {
        param,
        token: token.to_string(),
    })
}

fn parse_hex_u16_opt(param: &'static str, token: &str) -> Result<Option<u16>, ParamError> {
    if token.eq_ignore_ascii_case("none") {
        Ok(None)
    } else {
        parse_hex_u16(param, token).map(Some)
    }
}

impl FromStr for VideoFormat {
    type Err = ParamError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut it = s.split_whitespace();
        let mut next = || {
            it.next().ok_or(ParamError::Malformed {
                param: "wfd_video_formats",
                reason: "truncated",
            })
        };
        let native = NativeResolution::from_byte(parse_hex_u8("wfd_video_formats", next()?)?);
        let pdms = parse_hex_u8("wfd_video_formats", next()?)?;
        let profiles = H264Profiles::from_bits(parse_hex_u8("wfd_video_formats", next()?)?);
        let levels = H264Levels::from_bits(parse_hex_u8("wfd_video_formats", next()?)?);
        let cea = parse_hex_u32("wfd_video_formats", next()?)?;
        let vesa = parse_hex_u32("wfd_video_formats", next()?)?;
        let hh = parse_hex_u32("wfd_video_formats", next()?)?;
        let latency = parse_hex_u8("wfd_video_formats", next()?)?;
        let min_slice_size = parse_hex_u16("wfd_video_formats", next()?)?;
        let slice_encoding = parse_hex_u16("wfd_video_formats", next()?)?;
        let frame_skipping = parse_hex_u8("wfd_video_formats", next()?)?;
        let max_hres = parse_hex_u16_opt("wfd_video_formats", next()?)?;
        let max_vres = parse_hex_u16_opt("wfd_video_formats", next()?)?;
        Ok(Self {
            native,
            preferred_display_mode: pdms != 0,
            profiles,
            levels,
            cea,
            vesa,
            hh,
            latency,
            min_slice_size,
            slice_encoding,
            frame_skipping,
            max_hres,
            max_vres,
        })
    }
}

/// `wfd_audio_codecs` entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    /// LPCM. Modes bit 0 = 44.1 kHz 2ch, bit 1 = 48 kHz 2ch.
    Lpcm { modes: u32, latency: u8 },
    /// AAC-LC. Modes bit 0 = 48 kHz 2ch (**mandatory**), bit 1 = 4ch,
    /// bit 2 = 6ch, bit 3 = 8ch.
    Aac { modes: u32, latency: u8 },
    /// AC3 48 kHz. Modes bit 0 = 2ch, bit 1 = 4ch, bit 2 = 6ch.
    Ac3 { modes: u32, latency: u8 },
}

impl AudioFormat {
    /// The mandatory baseline every source MUST offer.
    #[must_use]
    pub const fn aac_mandatory() -> Self {
        Self::Aac {
            modes: 0x0000_0001,
            latency: 0,
        }
    }
}

impl fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (name, modes, lat) = match self {
            Self::Lpcm { modes, latency } => ("LPCM", modes, latency),
            Self::Aac { modes, latency } => ("AAC", modes, latency),
            Self::Ac3 { modes, latency } => ("AC3", modes, latency),
        };
        write!(f, "{name} {modes:08x} {lat:02x}")
    }
}

impl FromStr for AudioFormat {
    type Err = ParamError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut it = s.split_whitespace();
        let name = it.next().ok_or(ParamError::Malformed {
            param: "wfd_audio_codecs",
            reason: "missing codec name",
        })?;
        let modes_tok = it.next().ok_or(ParamError::Malformed {
            param: "wfd_audio_codecs",
            reason: "missing modes",
        })?;
        let lat_tok = it.next().ok_or(ParamError::Malformed {
            param: "wfd_audio_codecs",
            reason: "missing latency",
        })?;
        let modes = parse_hex_u32("wfd_audio_codecs", modes_tok)?;
        let latency = parse_hex_u8("wfd_audio_codecs", lat_tok)?;
        match name.to_ascii_uppercase().as_str() {
            "LPCM" => Ok(Self::Lpcm { modes, latency }),
            "AAC" => Ok(Self::Aac { modes, latency }),
            "AC3" => Ok(Self::Ac3 { modes, latency }),
            _ => Err(ParamError::Malformed {
                param: "wfd_audio_codecs",
                reason: "unknown codec name",
            }),
        }
    }
}

/// Comma-separated `wfd_audio_codecs` list.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AudioFormats(pub Vec<AudioFormat>);

impl fmt::Display for AudioFormats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, c) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            fmt::Display::fmt(c, f)?;
        }
        Ok(())
    }
}

impl FromStr for AudioFormats {
    type Err = ParamError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut out = Vec::new();
        for entry in s.split(',') {
            let trimmed = entry.trim();
            if trimmed.is_empty() {
                continue;
            }
            out.push(AudioFormat::from_str(trimmed)?);
        }
        Ok(Self(out))
    }
}

/// `wfd_content_protection`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentProtection {
    None,
    Hdcp20 { port: u16 },
    Hdcp21 { port: u16 },
}

impl fmt::Display for ContentProtection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::Hdcp20 { port } => write!(f, "HDCP2.0 port={port}"),
            Self::Hdcp21 { port } => write!(f, "HDCP2.1 port={port}"),
        }
    }
}

impl FromStr for ContentProtection {
    type Err = ParamError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("none") {
            return Ok(Self::None);
        }
        let (version, rest) = s.split_once(' ').ok_or(ParamError::Malformed {
            param: "wfd_content_protection",
            reason: "missing port spec",
        })?;
        let port_str = rest
            .trim()
            .strip_prefix("port=")
            .ok_or(ParamError::Malformed {
                param: "wfd_content_protection",
                reason: "expected `port=<decimal>`",
            })?;
        let port: u16 = port_str.parse().map_err(|_| ParamError::Malformed {
            param: "wfd_content_protection",
            reason: "port not a u16",
        })?;
        match version {
            "HDCP2.0" => Ok(Self::Hdcp20 { port }),
            "HDCP2.1" => Ok(Self::Hdcp21 { port }),
            _ => Err(ParamError::Malformed {
                param: "wfd_content_protection",
                reason: "unknown HDCP version",
            }),
        }
    }
}

/// `wfd_client_rtp_ports`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClientRtpPorts {
    pub rtp_port0: u16,
    pub rtp_port1: u16,
}

impl ClientRtpPorts {
    #[must_use]
    pub const fn primary(port: u16) -> Self {
        Self {
            rtp_port0: port,
            rtp_port1: 0,
        }
    }
}

impl fmt::Display for ClientRtpPorts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RTP/AVP/UDP;unicast {} {} mode=play",
            self.rtp_port0, self.rtp_port1,
        )
    }
}

impl FromStr for ClientRtpPorts {
    type Err = ParamError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let rest = s
            .strip_prefix("RTP/AVP/UDP;unicast")
            .ok_or(ParamError::Malformed {
                param: "wfd_client_rtp_ports",
                reason: "expected `RTP/AVP/UDP;unicast` prefix",
            })?
            .trim();
        let (p0, rest) = rest.split_once(' ').ok_or(ParamError::Malformed {
            param: "wfd_client_rtp_ports",
            reason: "missing rtp_port0",
        })?;
        let (p1, _mode) = rest.split_once(' ').ok_or(ParamError::Malformed {
            param: "wfd_client_rtp_ports",
            reason: "missing rtp_port1",
        })?;
        let rtp_port0: u16 = p0.parse().map_err(|_| ParamError::Malformed {
            param: "wfd_client_rtp_ports",
            reason: "rtp_port0 not a u16",
        })?;
        let rtp_port1: u16 = p1.parse().map_err(|_| ParamError::Malformed {
            param: "wfd_client_rtp_ports",
            reason: "rtp_port1 not a u16",
        })?;
        Ok(Self {
            rtp_port0,
            rtp_port1,
        })
    }
}

/// `wfd_trigger_method` legal values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerMethod {
    Setup,
    Pause,
    Teardown,
    Play,
}

impl fmt::Display for TriggerMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Setup => "SETUP",
            Self::Pause => "PAUSE",
            Self::Teardown => "TEARDOWN",
            Self::Play => "PLAY",
        })
    }
}

impl FromStr for TriggerMethod {
    type Err = ParamError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "SETUP" => Ok(Self::Setup),
            "PAUSE" => Ok(Self::Pause),
            "TEARDOWN" => Ok(Self::Teardown),
            "PLAY" => Ok(Self::Play),
            _ => Err(ParamError::Malformed {
                param: "wfd_trigger_method",
                reason: "unknown trigger",
            }),
        }
    }
}

/// `wfd_connector_type` — the physical display connector the sink
/// exposes. `0x05` = HDMI is the overwhelmingly common value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ConnectorType {
    Vga = 0x00,
    SVideo = 0x01,
    Composite = 0x02,
    Component = 0x03,
    Dvi = 0x04,
    Hdmi = 0x05,
    Lvds = 0x06,
    Jaeger = 0x08,
    MipiDsi = 0x09,
    DisplayPort = 0x0A,
    Internal = 0xFF,
}

impl fmt::Display for ConnectorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:02x}", *self as u8)
    }
}

/// `wfd_presentation_URL`. The URLs are absolute rtsp:// URLs or `none`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresentationUrl {
    pub url0: String,
    pub url1: Option<String>,
}

impl PresentationUrl {
    #[must_use]
    pub fn primary(url: impl Into<String>) -> Self {
        Self {
            url0: url.into(),
            url1: None,
        }
    }
}

impl fmt::Display for PresentationUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}",
            self.url0,
            self.url1.as_deref().unwrap_or("none")
        )
    }
}

impl FromStr for PresentationUrl {
    type Err = ParamError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.split_whitespace();
        let url0 = parts
            .next()
            .ok_or(ParamError::Malformed {
                param: "wfd_presentation_URL",
                reason: "missing url0",
            })?
            .to_string();
        let url1 = parts.next().and_then(|u| {
            if u.eq_ignore_ascii_case("none") {
                None
            } else {
                Some(u.to_string())
            }
        });
        Ok(Self { url0, url1 })
    }
}

/// One line of a `text/parameters` RTSP body: `wfd_<name>: <value>`.
/// Both directions of a body (request listing names, response carrying
/// values) go through this one struct; `value` is `None` when the line
/// is just a probe (`wfd_video_formats\r\n` in M3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamLine {
    pub name: String,
    pub value: Option<String>,
}

/// Serialize a list of params to the RTSP body format. CRLF between
/// lines, trailing CRLF. If `value` is `None`, only the name goes out.
#[must_use]
pub fn encode_body(lines: &[ParamLine]) -> String {
    let mut out = String::new();
    for line in lines {
        if let Some(v) = &line.value {
            out.push_str(&format!("{}: {}\r\n", line.name, v));
        } else {
            out.push_str(&format!("{}\r\n", line.name));
        }
    }
    out
}

/// Parse an RTSP `text/parameters` body. Accepts CRLF or LF line endings
/// (defensive — the Wi-Fi Alliance spec demands CRLF but the interop
/// reference catches LG WebOS returning CRLF only, Samsung sometimes
/// LF only).
#[must_use]
pub fn decode_body(body: &str) -> Vec<ParamLine> {
    let mut out = Vec::new();
    for raw in body.split('\n') {
        let line = raw.trim_end_matches('\r').trim();
        if line.is_empty() {
            continue;
        }
        let (name, value) = line.split_once(':').map_or_else(
            || (line.to_string(), None),
            |(n, v)| (n.trim().to_string(), Some(v.trim().to_string())),
        );
        out.push(ParamLine { name, value });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_format_round_trip_matches_reference() {
        // Spec example: 1080p30 CBP level 4.0, no VESA, no HH.
        let v = VideoFormat::pc_source_1080p30();
        let s = v.to_string();
        let back: VideoFormat = s.parse().expect("parse");
        assert_eq!(v, back);
    }

    #[test]
    fn audio_formats_round_trip() {
        let f = AudioFormats(vec![AudioFormat::aac_mandatory()]);
        assert_eq!(f.to_string(), "AAC 00000001 00");
        let parsed: AudioFormats = "AAC 00000001 00".parse().unwrap();
        assert_eq!(parsed.0.len(), 1);
    }

    #[test]
    fn multi_audio_formats_parse() {
        let parsed: AudioFormats = "LPCM 00000003 00, AAC 00000007 00, AC3 00000001 00"
            .parse()
            .unwrap();
        assert_eq!(parsed.0.len(), 3);
    }

    #[test]
    fn content_protection_round_trip() {
        assert_eq!(ContentProtection::None.to_string(), "none");
        let parsed: ContentProtection = "HDCP2.1 port=1234".parse().unwrap();
        assert!(matches!(parsed, ContentProtection::Hdcp21 { port: 1234 }));
    }

    #[test]
    fn client_rtp_ports_round_trip() {
        let p = ClientRtpPorts::primary(19000);
        assert_eq!(p.to_string(), "RTP/AVP/UDP;unicast 19000 0 mode=play");
        let parsed: ClientRtpPorts = p.to_string().parse().unwrap();
        assert_eq!(parsed, p);
    }

    #[test]
    fn trigger_method_round_trip() {
        for v in [
            TriggerMethod::Setup,
            TriggerMethod::Pause,
            TriggerMethod::Teardown,
            TriggerMethod::Play,
        ] {
            let parsed: TriggerMethod = v.to_string().parse().unwrap();
            assert_eq!(parsed, v);
        }
    }

    #[test]
    fn body_tolerates_lf_only() {
        let body = "wfd_audio_codecs: AAC 00000001 00\nwfd_video_formats: 00 00 01 01 00000020 00000000 00000000 00 0000 0000 00 none none\n";
        let parsed = decode_body(body);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "wfd_audio_codecs");
    }

    #[test]
    fn body_probe_lines_round_trip() {
        let lines = [
            ParamLine {
                name: "wfd_video_formats".into(),
                value: None,
            },
            ParamLine {
                name: "wfd_audio_codecs".into(),
                value: Some("AAC 00000001 00".into()),
            },
        ];
        let body = encode_body(&lines);
        let parsed = decode_body(&body);
        assert_eq!(parsed, lines.to_vec());
    }
}
