use crate::error::Result;
use crate::frame::{AudioCodec, AudioFrame, CapturedFrame, EncodedFrame};
use crate::{Codec, H264Profile, PixelFormat};

#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub codec: Codec,
    pub width: u32,
    pub height: u32,
    pub bitrate_kbps: u32,
    pub fps: u32,
    /// Target GOP length in seconds. Backends convert to frames
    /// internally using [`Self::fps`] so the call site doesn't need
    /// to keep frame counts in sync with framerate (a frame-count
    /// field would silently halve the GOP duration on a 60 fps device
    /// vs a 30 fps device). 2.0 by default to match the HLS
    /// segmenter's `segment_target_secs`.
    pub keyframe_interval_secs: f32,
    pub pixel_format: PixelFormat,
    /// Upper bound on the H.264 profile the encoder is allowed to
    /// emit. Set by the manager from the target device's
    /// `DeviceCapabilities::max_h264_profile` so we never produce a
    /// bitstream the receiver's hardware decoder can't handle.
    /// `None` = encoder picks its own default.
    pub max_h264_profile: Option<H264Profile>,
}

impl EncoderConfig {
    /// Resolved GOP length in frames at the configured framerate.
    /// Clamped to at least 1 so `idr_period`/`gop_length` parameters
    /// passed to the underlying codec are always valid.
    pub fn keyframe_interval_frames(&self) -> u32 {
        let frames = (self.keyframe_interval_secs * self.fps as f32).round();
        if frames.is_finite() && frames >= 1.0 {
            frames as u32
        } else {
            1
        }
    }
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            codec: Codec::H264,
            width: 1920,
            height: 1080,
            bitrate_kbps: 5000,
            fps: 60,
            keyframe_interval_secs: 2.0,
            pixel_format: PixelFormat::Bgra,
            max_h264_profile: None,
        }
    }
}

pub trait VideoEncoder: Send {
    const CODEC: Codec;

    fn configure(&mut self, config: &EncoderConfig) -> Result<()>;

    /// Encode the given captured frame.
    ///
    /// Implementations decide what to do with each variant:
    /// * x264 — always reads CPU bytes, calling
    ///   [`CapturedFrame::into_cpu`] to trigger a Vulkan readback if
    ///   the source produced a `Gpu` frame.
    /// * VA-API — consumes the `Gpu` variant directly (zero-copy)
    ///   and uploads to a `VASurface` for the `Cpu` variant.
    fn encode(&mut self, frame: CapturedFrame) -> Result<EncodedFrame>;

    fn flush(self) -> Result<Vec<EncodedFrame>>;
    fn get_headers(&mut self) -> Result<Vec<u8>>;

    /// Ask the encoder to emit an IDR (instantaneous decoder refresh
    /// keyframe) on the next [`encode`] call, in addition to whatever
    /// natural keyframes its internal interval would produce. The
    /// HLS segmenter uses this to anchor segment boundaries to wall
    /// clock when the upstream capture stalls (PipeWire on idle
    /// desktops can pause for hundreds of ms) or runs slower than
    /// the target framerate, which would otherwise let segments
    /// overshoot `segment_target_secs` until the next natural IDR.
    ///
    /// Default is a no-op so backends that can't influence keyframe
    /// placement (e.g. x264 via the safe `x264` crate, which doesn't
    /// expose `i_type`) keep working — they simply rely on their
    /// internal interval.
    fn request_keyframe(&mut self) {}

    /// Live-update the encoder's target average bitrate in kbps,
    /// without tearing down the encoding session.
    ///
    /// Used by the adaptive bitrate controller
    /// ([`crate::AdaptiveBitrateState`]): when the HLS server
    /// observes that the receiver's link is saturated, it drops the
    /// target so the encoder's next macroblock budget shrinks; when
    /// the link recovers it raises it back up.
    ///
    /// Returns `Ok(())` even on backends that can't honour the
    /// request — the caller should always make adjustments on a
    /// best-effort basis. NVENC reconfigures live (no GOP gap);
    /// x264 via the safe crate has no exposed knob and is a no-op
    /// (the natural fallback is "screen-share users on x264 get
    /// what they get"). Default is no-op so existing impls don't
    /// have to opt in.
    fn set_bitrate_kbps(&mut self, _kbps: u32) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AudioEncoderConfig {
    /// Output codec. Only [`AudioCodec::Aac`] is wired through the
    /// chromecast HLS pipeline today; the field stays so future
    /// protocols (AirPlay → ALAC, raw PCM; Miracast → AAC/AC-3) can
    /// pick at runtime without re-shaping the trait.
    pub codec: AudioCodec,
    /// Input PCM sample rate (Hz). Encoders accept whatever the
    /// upstream [`crate::AudioCapture`] backend negotiated — most
    /// HLS deployments use 48000.
    pub sample_rate: u32,
    /// Input channel count.
    pub channels: u16,
    /// Target average bitrate in kbps. AAC-LC at 128 kbps stereo is
    /// the chromecast-friendly default (transparent above ~96 kbps).
    pub bitrate_kbps: u32,
}

impl Default for AudioEncoderConfig {
    fn default() -> Self {
        Self {
            codec: AudioCodec::Aac,
            sample_rate: 48_000,
            channels: 2,
            bitrate_kbps: 128,
        }
    }
}

/// Encoder trait that mirrors [`VideoEncoder`]. The contract is
/// "consume PCM, emit codec-framed bytes with a 90 kHz-friendly PTS".
/// Returning `Ok(None)` is legal — block-coded encoders (AAC) need
/// to accumulate at least one frame of input (1024 samples for
/// AAC-LC) before they can emit, so the first N input chunks
/// typically return `None` while internal state warms up.
pub trait AudioEncoder: Send {
    fn configure(&mut self, config: &AudioEncoderConfig) -> Result<()>;

    /// Push one chunk of PCM. The encoder may produce zero, one, or
    /// more output frames per input; output frames are drained via
    /// [`Self::take_output`] after this call. `timestamp_us` is the
    /// upstream capture timestamp of the first sample in the chunk —
    /// the encoder advances its internal monotonic counter from it
    /// so subsequent output frames carry strictly-increasing PTS in
    /// the same timeline.
    fn encode(&mut self, frame: &AudioFrame) -> Result<()>;

    /// Drain any output frames produced by the latest `encode`
    /// (and any previously-buffered residue). Returns an empty
    /// vec when the encoder is still warming up.
    fn take_output(&mut self) -> Vec<AudioFrame>;

    /// Optional codec-specific configuration descriptor (e.g. the
    /// 2-byte AudioSpecificConfig that the MPEG-TS muxer can use
    /// for the AAC ADTS header). Returns empty when the codec doesn't
    /// need an out-of-band header (every output frame is self-
    /// describing — true for ADTS-framed AAC).
    fn codec_config(&self) -> Vec<u8> {
        Vec::new()
    }

    /// Flush any internal frames and return them. Called once at
    /// shutdown.
    fn flush(self) -> Result<Vec<AudioFrame>>;
}
