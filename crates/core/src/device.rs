use std::collections::HashMap;
use std::net::IpAddr;

use uuid::Uuid;

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, Default, PartialEq)]
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
    pub supported_codecs: Vec<crate::Codec>,
}

/// H.264 profile constraint. Used by [`DeviceCapabilities`] and
/// [`crate::EncoderConfig`] to negotiate compatibility between what
/// a receiver decodes and what an encoder produces. Ordering is
/// "less featureful" → "more featureful"; an encoder asked for High
/// MAY fall back to Main or Baseline if the hardware doesn't
/// support it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

#[derive(Debug, Clone)]
pub enum DiscoveryEvent {
    DeviceFound(Device),
    DeviceLost(Uuid),
    Error {
        protocol: &'static str,
        message: String,
    },
}
