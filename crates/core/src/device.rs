use bytes::Bytes;
use std::collections::HashMap;
use std::fmt::Display;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq)]
pub struct Device {
    pub id: Uuid,
    pub name: String,
    pub protocol: &'static str,
    pub protocol_icon: Bytes,
    /// Every address the receiver answers on, most preferred first.
    pub addresses: DeviceAddresses,
    pub model: Option<String>,
    pub capabilities: DeviceCapabilities,
    pub metadata: HashMap<String, String>,
}

/// The socket addresses a receiver answers on, ordered by preference.
///
/// Receivers advertise several addresses over mDNS (IPv4 on more than
/// one interface, global and link-local IPv6). Connecting walks them
/// in this order, so the first is the one most likely to work from
/// this host:
///
/// 1. IPv4 on a subnet of a local interface,
/// 2. other private IPv4,
/// 3. other routable IPv4,
/// 4. IPv4 link-local (`169.254.0.0/16`),
/// 5. global or unique-local IPv6,
/// 6. link-local IPv6, carrying the scope id of the interface it was
///    seen on,
/// 7. loopback.
///
/// Unspecified and multicast addresses, and link-local IPv6 without a
/// scope id (no interface to send it on), are dropped. Never empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceAddresses(Vec<SocketAddr>);

impl DeviceAddresses {
    /// Rank `addrs` by preference, dropping duplicates and addresses
    /// nothing can connect to. `local_subnets` are the IPv4 subnets of
    /// this host's interfaces (`ferricast_core::net::local_ipv4_subnets`).
    ///
    /// Returns `None` when no usable address remains.
    #[must_use]
    pub fn ranked(
        addrs: impl IntoIterator<Item = SocketAddr>,
        local_subnets: &[Ipv4Subnet],
    ) -> Option<Self> {
        let mut ranked: Vec<(AddressRank, SocketAddr)> = addrs
            .into_iter()
            .filter_map(|addr| Some((AddressRank::of(addr, local_subnets)?, addr)))
            .collect();
        // Address then scope id as tie-breakers keep the order stable
        // across mDNS answers that list the same set differently.
        ranked.sort_unstable_by_key(|&(rank, addr)| (rank, addr.ip(), scope_id(addr)));
        ranked.dedup_by_key(|&mut (_, addr)| addr);
        let addrs: Vec<SocketAddr> = ranked.into_iter().map(|(_, addr)| addr).collect();
        (!addrs.is_empty()).then_some(Self(addrs))
    }

    /// The address to try first.
    ///
    /// # Panics
    /// Never: every constructor refuses an empty address list.
    #[must_use]
    pub fn primary(&self) -> SocketAddr {
        *self
            .0
            .first()
            .expect("invariant: DeviceAddresses is never empty")
    }

    /// Every address, most preferred first.
    #[must_use]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = SocketAddr> + '_ {
        self.0.iter().copied()
    }

    /// Every address, most preferred first.
    #[must_use]
    pub fn as_slice(&self) -> &[SocketAddr] {
        &self.0
    }

    /// Whether `ip` is one of the receiver's addresses, ignoring port
    /// and scope id. An IPv4-mapped IPv6 address matches its IPv4 form.
    #[must_use]
    pub fn contains_ip(&self, ip: IpAddr) -> bool {
        let ip = ip.to_canonical();
        self.0.iter().any(|a| a.ip().to_canonical() == ip)
    }
}

impl From<SocketAddr> for DeviceAddresses {
    /// A receiver known by a single address (manual entry, tests).
    fn from(addr: SocketAddr) -> Self {
        Self(vec![addr])
    }
}

impl Display for DeviceAddresses {
    /// The primary address; IPv6 prints as `[addr%scope]:port`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.primary().fmt(f)
    }
}

fn scope_id(addr: SocketAddr) -> u32 {
    match addr {
        SocketAddr::V4(_) => 0,
        SocketAddr::V6(v6) => v6.scope_id(),
    }
}

/// Preference class of one address; lower is tried first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum AddressRank {
    V4LocalSubnet,
    V4Private,
    V4Routable,
    V4LinkLocal,
    V6Routable,
    V6LinkLocal,
    Loopback,
}

impl AddressRank {
    /// `None` for addresses nothing can connect to.
    fn of(addr: SocketAddr, local_subnets: &[Ipv4Subnet]) -> Option<Self> {
        match addr {
            SocketAddr::V4(v4) => Self::of_v4(*v4.ip(), local_subnets),
            SocketAddr::V6(v6) => {
                let ip = *v6.ip();
                if let Some(mapped) = ip.to_ipv4_mapped() {
                    Self::of_v4(mapped, local_subnets)
                } else if ip.is_unspecified() || ip.is_multicast() {
                    None
                } else if ip.is_loopback() {
                    Some(Self::Loopback)
                } else if ip.is_unicast_link_local() {
                    // Only usable on the interface it was seen on.
                    (v6.scope_id() != 0).then_some(Self::V6LinkLocal)
                } else {
                    Some(Self::V6Routable)
                }
            }
        }
    }

    fn of_v4(ip: Ipv4Addr, local_subnets: &[Ipv4Subnet]) -> Option<Self> {
        if ip.is_unspecified() || ip.is_multicast() || ip.is_broadcast() {
            return None;
        }
        Some(if ip.is_loopback() {
            Self::Loopback
        } else if local_subnets.iter().any(|net| net.contains(ip)) {
            Self::V4LocalSubnet
        } else if ip.is_private() {
            Self::V4Private
        } else if ip.is_link_local() {
            Self::V4LinkLocal
        } else {
            Self::V4Routable
        })
    }
}

/// An IPv4 subnet of a local interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ipv4Subnet {
    pub addr: Ipv4Addr,
    pub netmask: Ipv4Addr,
}

impl Ipv4Subnet {
    /// Whether `ip` lies in this subnet. A zero netmask matches
    /// nothing, so a default route doesn't claim every address.
    #[must_use]
    pub fn contains(&self, ip: Ipv4Addr) -> bool {
        let mask = u32::from(self.netmask);
        mask != 0 && u32::from(self.addr) & mask == u32::from(ip) & mask
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent capability flags, each read from a receiver's advertisement"
)]
pub struct DeviceCapabilities {
    pub supports_audio: bool,
    pub supports_video: bool,
    pub supports_screen_mirror: bool,
    pub max_width: Option<u32>,
    pub max_height: Option<u32>,
    /// Maximum framerate the receiver's hardware decoder can sustain
    /// at `max_width × max_height`. Manager caps the encoder fps to
    /// this before configuring NVENC / VA-API / x264. `None` =
    /// caller-defined.
    pub max_fps: Option<u32>,
    /// Bitrate ceiling the receiver's H.264 decoder is rated for —
    /// equivalent to the protocol's documented Level cap. Used so
    /// the encoder doesn't blast 30 Mbps at a device that tops out
    /// at 5 Mbps.
    pub max_bitrate_kbps: Option<u32>,
    /// Bitrate ceiling for the audio codec (AAC-LC today). 128 kbps
    /// stereo is transparent for most chromecast-class receivers and
    /// the conservative floor; modern receivers (Ultra, Google /
    /// Android TV, soundbars) handle 256+ kbps. `None` = caller
    /// picks its own default.
    pub max_audio_bitrate_kbps: Option<u32>,
    /// Some receivers (older Chromecast firmwares, certain
    /// `md == \"Chromecast\"` 1st/2nd gen) reject HLS streams whose
    /// MPEG-TS PMT only carries video — the demuxer expects video +
    /// audio. When `true`, downstream muxers MUST inject a silent
    /// audio track (e.g. AAC LC) for the receiver to accept the
    /// stream. Newer receivers ignore this and play video-only
    /// fine.
    pub requires_audio: bool,
    /// Tightest H.264 profile the device's hardware decoder is known
    /// to accept. Older Chromecasts choke on High-profile features
    /// (CABAC, 8x8 transform); modern Chromecast Ultra / Google TV /
    /// Android TV handle High without complaint. `None` = downstream
    /// encoder picks its own default.
    pub max_h264_profile: Option<H264Profile>,
    /// Tightest HEVC profile the device's hardware decoder advertises.
    /// `None` = downstream encoder picks Main (8-bit 4:2:0) as the
    /// conservative floor — every HEVC-capable receiver we've shipped
    /// against accepts Main; Main10 is opt-in per device class.
    pub max_h265_profile: Option<H265Profile>,
    pub supported_codecs: Vec<crate::Codec>,
    /// Whether the receiver's HLS player understands Low-Latency
    /// HLS (RFC 8216bis): `#EXT-X-VERSION:6`,
    /// `#EXT-X-SERVER-CONTROL`, `#EXT-X-PART-INF`, `#EXT-X-PART`,
    /// and `?_HLS_msn=…&_HLS_part=…` blocking playlist reload.
    ///
    /// Conservative default: `false`. The 1st/2nd-gen Chromecast
    /// firmware (`md=Chromecast`) demonstrably trips into a
    /// LOADING-forever state when handed a v6 playlist — the
    /// EXT-X-VERSION:6 line alone seems to be enough to brick the
    /// load. Newer hardware (Ultra, Google TV, Android TV) ships
    /// with CAF builds that support LL-HLS, but we still default
    /// to `false` because firmware versions vary; opt in per
    /// device class in `capabilities_for_model` only where it's
    /// been field-tested.
    pub supports_low_latency_hls: bool,
    /// Whether this device supports Cast Streaming (the RTP/UDP
    /// mirroring protocol on `urn:x-cast:com.google.cast.webrtc`).
    /// When `true`, the manager may select [`ChromecastMirrorSession`]
    /// for sub-200 ms latency instead of the HLS path.
    ///
    /// Conservative default: `false`. Enabled for 3rd-gen+ Chromecast,
    /// Ultra, Google TV, and unknown models (optimistic). Disabled for
    /// 1st/2nd-gen (`md=Chromecast`) — older firmware may refuse OFFER
    /// from non-Chrome senders; fall back to HLS for those.
    pub supports_cast_streaming: bool,
    /// AirPlay-specific discovery data. `None` for every other protocol.
    pub airplay_config: Option<AirplayConfig>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AirplayConfig {
    /// Bitfield of supported features. Advertised over mDNS as one or two
    /// 32-bit hex words separated by a comma; the second word is optional.
    /// See <https://openairplay.github.io/airplay-spec/features.html>.
    pub features: Features,
    /// Bitfield of status flags.
    /// See <https://openairplay.github.io/airplay-spec/status_flags.html>.
    pub flags: Flags,
    /// Receiver Ed25519 public key (hex), from the `pk` TXT record.
    pub pk: Option<String>,
    /// Receiver device id (`deviceid` TXT record, usually a MAC address).
    /// Stable across restarts, so it keys stored pairing credentials.
    pub device_id: Option<String>,
}

impl AirplayConfig {
    /// Pairing procedure to run against this receiver, decided from what it
    /// advertises (the same split pyatv makes in `get_protocol_version`).
    ///
    /// Receivers that advertise CoreUtils pairing (bit 38) or unified
    /// pair-setup (bit 51) speak HomeKit pairing; everything else (UxPlay,
    /// RPiPlay, shairplay derivatives, AirPlay 1 hardware) speaks the legacy
    /// `/pair-setup` + `/pair-verify` exchange.
    #[must_use]
    pub fn pairing_mode(&self) -> AirPlayPairing {
        let hap = self.features.intersects(
            Features::CORE_UTILS_PAIRING_AND_ENCRYPTION | Features::UNIFIED_PAIR_SETUP_MFI,
        );
        let pin = self
            .flags
            .intersects(Flags::PIN_REQUIRED | Flags::PASSWORD_REQUIRED);
        if hap {
            let transient = self
                .features
                .intersects(Features::TRANSIENT_PAIRING | Features::SYSTEM_PAIRING);
            if transient && !pin {
                AirPlayPairing::HapTransient
            } else {
                AirPlayPairing::Hap
            }
        } else if pin {
            AirPlayPairing::LegacyPin
        } else {
            AirPlayPairing::Legacy
        }
    }
}

/// Pairing procedure an AirPlay sender runs before streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AirPlayPairing {
    /// `/pair-setup` with a raw Ed25519 key followed by `/pair-verify`.
    /// No user interaction and no `X-Apple-HKP` header.
    Legacy,
    /// `/pair-pin-start` + SRP over `/pair-setup-pin` (binary plists), then
    /// `/pair-verify`. The receiver shows a 4-digit PIN.
    LegacyPin,
    /// HomeKit pair-setup (SRP-6a, PIN shown on the receiver) followed by
    /// HomeKit pair-verify. `X-Apple-HKP: 3`.
    Hap,
    /// HomeKit transient pair-setup with the fixed PIN `3939`; no stored
    /// credentials. `X-Apple-HKP: 4`.
    HapTransient,
}

impl AirPlayPairing {
    /// Value of the `X-Apple-HKP` request header for this mode, if any.
    #[must_use]
    pub fn hkp_header(self) -> Option<&'static str> {
        match self {
            Self::Legacy | Self::LegacyPin => None,
            Self::Hap => Some("3"),
            Self::HapTransient => Some("4"),
        }
    }
}

bitflags::bitflags! {
     /// AirPlay feature bits (`features` TXT record).
     #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
     pub struct Features: u64 {
        const VIDEO_SUPPORTED = 1 << 0;
        const PHOTO_SUPPORTED = 1 << 1;
        const VIDEO_FAIRPLAY = 1 << 2;
        const VOLUME_CONTROL = 1 << 3;
        const VIDEO_HTTP_LIVE_STREAM = 1 << 4;
        const SLIDESHOW_SUPPORTED = 1 << 5;
        /// Screen mirroring ("Screen").
        const MIRRORING_SUPPORTED = 1 << 7;
        const SCREEN_ROTATE = 1 << 8;
        const AUDIO_SUPPORTED = 1 << 9;
        const AUDIO_REDUNDANCY_SUPPORTED = 1 << 11;
        const FAIRPLAY_SECURE_AUTH_SUPPORTED = 1 << 12;
        const PHOTO_CACHING = 1 << 13;
        const AUTH4 = 1 << 14;
        /// MFi authentication ("Authentication_8").
        const AUTHENTICATION_8 = 1 << 26;
        const SUPPORT_LEGACY_PAIRING = 1 << 27;
        const HAS_UNIFIED_ADVERTISER_INFO = 1 << 30;
        const SUPPORTS_VOLUME = 1 << 32;
        const AIRPLAY_VIDEO_PLAY_QUEUE = 1 << 33;
        const AIRPLAY_FROM_CLOUD = 1 << 34;
        const CORE_UTILS_PAIRING_AND_ENCRYPTION = 1 << 38;
        const BUFFERED_AUDIO = 1 << 40;
        const PTP = 1 << 41;
        const SCREEN_MULTI_CODEC = 1 << 42;
        const SYSTEM_PAIRING = 1 << 43;
        const HK_PAIRING_AND_ACCESS_CONTROL = 1 << 46;
        const TRANSIENT_PAIRING = 1 << 48;
        const UNIFIED_PAIR_SETUP_MFI = 1 << 51;
     }
}

impl Features {
    /// Parse the `features` TXT value: `0xLOW` or `0xLOW,0xHIGH`.
    ///
    /// Returns `None` when a word is not valid hex. Unknown bits are kept.
    #[must_use]
    pub fn parse_txt(value: &str) -> Option<Self> {
        fn word(v: &str) -> Option<u64> {
            let v = v.trim();
            let v = v
                .strip_prefix("0x")
                .or_else(|| v.strip_prefix("0X"))
                .unwrap_or(v);
            u32::from_str_radix(v, 16).ok().map(u64::from)
        }
        let mut parts = value.split(',');
        let low = word(parts.next()?)?;
        let high = match parts.next() {
            Some(v) => word(v)?,
            None => 0,
        };
        Some(Self::from_bits_retain((high << 32) | low))
    }
}

impl Display for Features {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

bitflags::bitflags! {
     /// AirPlay status flags (`flags` TXT record).
     #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
     pub struct Flags: u32 {
        const PROBLEM_HAS_BEEN_DETECTED = 1 << 0;
        const DEVICE_IS_NOT_CONFIGURED = 1 << 1;
        const AUDIO_CABLE = 1 << 2;
        /// The receiver requires a PIN before a sender may stream.
        const PIN_REQUIRED = 1 << 3;
        const SUPPORTS_AIRPLAY_FROM_CLOUD = 1 << 6;
        const PASSWORD_REQUIRED = 1 << 7;
        const PIN_ONE_TIME_PAIRING = 1 << 9;
        const DEVICE_WAS_SETUP_FOR_HK_ACCESS_CONTROL = 1 << 10;
        const DEVICE_SUPPORT_RELAY = 1 << 11;
        const SILENT_PRIMARY = 1 << 12;
        const RECEIVER_SESSION_IS_ACTIVE = 1 << 17;
    }
}

/// H.264 profile constraint. Used by [`DeviceCapabilities`] and
/// [`crate::EncoderConfig`] to negotiate compatibility between what
/// a receiver decodes and what an encoder produces. Ordering is
/// "less featureful" → "more featureful"; an encoder asked for High
/// MAY fall back to Main or Baseline if the hardware doesn't
/// support it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub enum H264Profile {
    /// Baseline / Constrained Baseline. No CABAC, no B-frames.
    /// Required floor for the oldest Chromecasts and Miracast 1.0.
    Baseline,
    /// Main. CAVLC entropy coding, B-frames allowed. Universally
    /// supported on every Chromecast generation and the safe
    /// default for live screencast.
    Main,
    /// High. CABAC + 8x8 transform + weighted prediction. Best
    /// compression ratio. Supported by Chromecast Ultra, Google
    /// TV, Android TV, AirPlay 2, modern Miracast sinks.
    High,
}

/// HEVC (H.265) profile constraint. Same negotiation contract as
/// [`H264Profile`]: ordered "less featureful" → "more featureful";
/// an encoder asked for Main10 falls back to Main when the hardware
/// can't emit 10-bit. We only model the two production-relevant
/// profiles — Main (8-bit 4:2:0) and Main10 (10-bit 4:2:0). Range
/// Extensions / 4:4:4 are deliberately out of scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub enum H265Profile {
    /// Main, 8-bit 4:2:0. Universally supported floor — every HEVC
    /// decoder advertises Main.
    Main,
    /// Main10, 10-bit 4:2:0. HDR-capable receivers (Chromecast Ultra,
    /// Google TV, modern Android TV, AirPlay 2) decode it; older
    /// hardware does not.
    Main10,
}

#[derive(Debug, Clone)]
#[expect(
    clippy::large_enum_variant,
    reason = "found devices dominate the stream; boxing them buys nothing"
)]
pub enum DiscoveryEvent {
    DeviceFound(Device),
    DeviceLost(Uuid),
    Error {
        protocol: &'static str,
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(features: Features, flags: Flags) -> AirplayConfig {
        AirplayConfig {
            features,
            flags,
            pk: None,
            device_id: None,
        }
    }

    #[test]
    fn pairing_mode_decision_table() {
        let hap = Features::CORE_UTILS_PAIRING_AND_ENCRYPTION;
        let cases = [
            (
                Features::SUPPORT_LEGACY_PAIRING,
                Flags::empty(),
                AirPlayPairing::Legacy,
            ),
            (Features::empty(), Flags::empty(), AirPlayPairing::Legacy),
            (
                Features::empty(),
                Flags::PASSWORD_REQUIRED,
                AirPlayPairing::LegacyPin,
            ),
            (
                Features::empty(),
                Flags::PIN_REQUIRED,
                AirPlayPairing::LegacyPin,
            ),
            (hap, Flags::empty(), AirPlayPairing::Hap),
            (
                hap | Features::TRANSIENT_PAIRING,
                Flags::empty(),
                AirPlayPairing::HapTransient,
            ),
            (
                hap | Features::TRANSIENT_PAIRING,
                Flags::PIN_REQUIRED,
                AirPlayPairing::Hap,
            ),
            (
                Features::UNIFIED_PAIR_SETUP_MFI,
                Flags::empty(),
                AirPlayPairing::Hap,
            ),
        ];
        for (features, flags, expected) in cases {
            assert_eq!(
                config(features, flags).pairing_mode(),
                expected,
                "{features:?} {flags:?}"
            );
        }
    }

    #[test]
    fn features_txt_second_word_is_optional() {
        // UxPlay advertises only the low word as meaningful.
        let f = Features::parse_txt("0x5A7FFEE6,0x0").unwrap();
        assert!(f.contains(Features::MIRRORING_SUPPORTED | Features::SUPPORT_LEGACY_PAIRING));
        assert!(f.contains(Features::HAS_UNIFIED_ADVERTISER_INFO));
        assert_eq!(Features::parse_txt("0x5A7FFEE6"), Some(f));
        let apple_tv = Features::parse_txt("0x4A7FDFD5,0xBC157FDE").unwrap();
        assert!(apple_tv.contains(Features::CORE_UTILS_PAIRING_AND_ENCRYPTION));
        assert!(Features::parse_txt("zz").is_none());
        assert!(Features::parse_txt("0x1,zz").is_none());
    }

    #[test]
    fn hkp_header_values() {
        assert_eq!(AirPlayPairing::Hap.hkp_header(), Some("3"));
        assert_eq!(AirPlayPairing::HapTransient.hkp_header(), Some("4"));
        assert_eq!(AirPlayPairing::Legacy.hkp_header(), None);
        assert_eq!(AirPlayPairing::LegacyPin.hkp_header(), None);
    }

    fn addrs(list: &[&str], subnets: &[Ipv4Subnet]) -> Option<Vec<String>> {
        let parsed = list.iter().map(|a| a.parse::<SocketAddr>().unwrap());
        DeviceAddresses::ranked(parsed, subnets).map(|d| d.iter().map(|a| a.to_string()).collect())
    }

    #[test]
    fn addresses_are_ordered_by_reachability() {
        let lan = Ipv4Subnet {
            addr: Ipv4Addr::new(192, 168, 1, 10),
            netmask: Ipv4Addr::new(255, 255, 255, 0),
        };
        let ordered = addrs(
            &[
                "[fe80::2%3]:7000",
                "[2001:db8::5]:7000",
                "169.254.3.4:7000",
                "8.8.4.4:7000",
                "172.17.0.1:7000",
                "[fd00::5]:7000",
                "192.168.1.40:7000",
                "127.0.0.1:7000",
            ],
            &[lan],
        )
        .unwrap();
        assert_eq!(
            ordered,
            [
                "192.168.1.40:7000",
                "172.17.0.1:7000",
                "8.8.4.4:7000",
                "169.254.3.4:7000",
                "[2001:db8::5]:7000",
                "[fd00::5]:7000",
                "[fe80::2%3]:7000",
                "127.0.0.1:7000",
            ]
        );
        // Without a matching subnet, private IPv4 still leads.
        assert_eq!(
            addrs(&["[fd00::5]:7000", "10.0.0.5:7000"], &[]).unwrap(),
            ["10.0.0.5:7000", "[fd00::5]:7000"]
        );
    }

    #[test]
    fn unusable_addresses_are_dropped_and_duplicates_merged() {
        assert_eq!(
            addrs(
                &[
                    "[fe80::2]:7000",
                    "0.0.0.0:7000",
                    "[::]:7000",
                    "10.0.0.5:7000",
                    "10.0.0.5:7000",
                    "[::ffff:10.0.0.6]:7000",
                ],
                &[]
            )
            .unwrap(),
            ["10.0.0.5:7000", "[::ffff:10.0.0.6]:7000"]
        );
        assert_eq!(addrs(&["[fe80::2]:7000"], &[]), None);
        assert_eq!(addrs(&[], &[]), None);
    }

    #[test]
    fn primary_and_membership() {
        let d = DeviceAddresses::ranked(
            ["[fe80::2%3]:8009", "10.0.0.5:8009"].map(|a| a.parse().unwrap()),
            &[],
        )
        .unwrap();
        assert_eq!(d.primary(), "10.0.0.5:8009".parse().unwrap());
        assert_eq!(d.to_string(), "10.0.0.5:8009");
        assert!(d.contains_ip("fe80::2".parse().unwrap()));
        assert!(d.contains_ip("::ffff:10.0.0.5".parse().unwrap()));
        assert!(!d.contains_ip("10.0.0.6".parse().unwrap()));
        let single = DeviceAddresses::from("[fe80::2%3]:8009".parse::<SocketAddr>().unwrap());
        assert_eq!(single.to_string(), "[fe80::2%3]:8009");
    }

    #[test]
    fn zero_netmask_matches_nothing() {
        let any = Ipv4Subnet {
            addr: Ipv4Addr::new(10, 0, 0, 1),
            netmask: Ipv4Addr::UNSPECIFIED,
        };
        assert!(!any.contains(Ipv4Addr::new(10, 0, 0, 2)));
    }
}
