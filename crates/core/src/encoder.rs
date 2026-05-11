use crate::error::Result;
use crate::frame::{CapturedFrame, EncodedFrame};
use crate::{Codec, H264Profile, PixelFormat};

#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub codec: Codec,
    pub width: u32,
    pub height: u32,
    pub bitrate_kbps: u32,
    pub fps: u32,
    pub keyframe_interval: u32,
    pub pixel_format: PixelFormat,
    /// Upper bound on the H.264 profile the encoder is allowed to
    /// emit. Set by the manager from the target device's
    /// `DeviceCapabilities::max_h264_profile` so we never produce a
    /// bitstream the receiver's hardware decoder can't handle.
    /// `None` = encoder picks its own default.
    pub max_h264_profile: Option<H264Profile>,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            codec: Codec::H264,
            width: 1920,
            height: 1080,
            bitrate_kbps: 5000,
            fps: 60,
            keyframe_interval: 60,
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
