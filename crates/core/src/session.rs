use std::sync::Arc;

use crate::adaptive::AdaptiveBitrateState;
use crate::capture::AudioMuteHandle;
use crate::device::Device;
use crate::error::Result;
use crate::frame::{AudioCodec, EncodedFrame};

#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub bitrate_kbps: u32,
    pub codec: crate::Codec,

    /// Optional adaptive bitrate controller. When provided, the
    /// receiver protocol (e.g. Chromecast) plumbs it into its HLS
    /// server, and the stream manager polls it on the hot path to
    /// live-reconfigure the encoder when the receiver's link is
    /// under sustained pressure. `None` (default) keeps the
    /// pre-adaptive behaviour: fixed bitrate from configure, no
    /// runtime feedback.
    pub adaptive: Option<Arc<AdaptiveBitrateState>>,

    /// When `Some`, the manager-driven pipeline also starts an
    /// audio capture + encoder and feeds AAC (or future codec)
    /// frames into the receiver session alongside the video stream.
    /// `None` keeps the stream video-only (matching pre-audio
    /// behaviour: the chromecast HLS path falls back to silent-AAC
    /// injection when the device requires audio).
    pub audio: Option<AudioStreamConfig>,
}

/// Audio-side configuration carried alongside [`StreamConfig`]. The
/// receiver protocol picks the codec it wants (today every protocol
/// the codebase supports lands on AAC for HLS-MPEG-TS) and the
/// manager spins up a matching encoder.
#[derive(Debug, Clone)]
pub struct AudioStreamConfig {
    /// Output codec emitted to the receiver. Chromecast HLS only
    /// accepts AAC-LC inside MPEG-TS; AirPlay-style protocols would
    /// pick ALAC, etc. Kept here so the session layer can negotiate
    /// against `ProtocolHandler::SUPPORTED_AUDIO_CODECS` rather than
    /// hard-coding AAC.
    pub codec: AudioCodec,
    pub sample_rate: u32,
    pub channels: u16,
    pub bitrate_kbps: u32,
    /// Mute toggle shared with the audio capture worker. Setting it
    /// to `true` swaps the live PCM for silence at the source so the
    /// receiver's audio elementary stream keeps progressing — see
    /// the type-level docs on [`AudioMuteHandle`].
    pub mute: AudioMuteHandle,
}

impl Default for AudioStreamConfig {
    fn default() -> Self {
        Self {
            codec: AudioCodec::Aac,
            sample_rate: 48_000,
            channels: 2,
            bitrate_kbps: 128,
            mute: AudioMuteHandle::default(),
        }
    }
}

impl AudioStreamConfig {
    /// Build a config sized to a target device's capabilities. Picks
    /// `bitrate_kbps` from
    /// [`crate::DeviceCapabilities::max_audio_bitrate_kbps`] so the
    /// call site doesn't need to know how to derive it. `mute` is
    /// caller-supplied because the handle has to be shared with the
    /// UI side.
    pub fn for_device(caps: &crate::DeviceCapabilities, mute: AudioMuteHandle) -> Self {
        let default = Self::default();
        Self {
            codec: default.codec,
            sample_rate: default.sample_rate,
            channels: default.channels,
            bitrate_kbps: caps.max_audio_bitrate_kbps.unwrap_or(default.bitrate_kbps),
            mute,
        }
    }
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            // 60 matches typical desktop refresh rates. Some
            // xdg-desktop-portal backends only complete the
            // EnumFormat negotiation when the default framerate the
            // sender advertises matches what the compositor wants to
            // produce — `default: 30, range: 0-1000` was reproducibly
            // failing with `Paused → Error("no more input formats")`
            // on Niri+pipewire-shm. The actual value the encoder
            // ends up using is overridden by `ScreenCapture::
            // get_framerate()` once the format settles.
            fps: 60,
            bitrate_kbps: 5000,
            codec: crate::Codec::H264,
            adaptive: None,
            audio: None,
        }
    }
}

pub trait CastSession: Send + Sync {
    fn connect(&mut self, device: &Device) -> impl Future<Output = Result<()>> + Send;
    fn setup_stream(&mut self, config: &StreamConfig) -> impl Future<Output = Result<()>> + Send;
    fn send_frame(&mut self, frame: &EncodedFrame) -> impl Future<Output = Result<()>> + Send;

    /// Forward one encoded audio frame to the receiver. The frame's
    /// `codec` must match what was negotiated via
    /// [`AudioStreamConfig::codec`]; calling this without prior
    /// `setup_stream(..audio: Some(_))` configuration returns an
    /// error. Default impl drops the frame so receiver protocols
    /// that don't yet support audio (Miracast / AirPlay TODO) keep
    /// working unchanged when the manager flushes a frame their way.
    fn send_audio_frame(
        &mut self,
        _frame: &crate::AudioFrame,
    ) -> impl Future<Output = Result<()>> + Send {
        async { Ok(()) }
    }

    fn stop(&mut self) -> impl Future<Output = Result<()>> + Send;
    fn is_alive(&self) -> bool;
}
