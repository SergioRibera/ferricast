//! Helpers for deciding whether a `configure()` call can reuse the
//! open hardware session.

use ferricast_core::EncoderConfig;

pub(crate) trait EncoderConfigExt {
    /// `true` when `other` can be served by a session opened for
    /// `self`: same codec, geometry, frame rate, GOP, input format and
    /// profile ceilings. Bitrate is deliberately excluded — backends
    /// that can retarget it live do so instead of rebuilding.
    fn same_session_shape(&self, other: &Self) -> bool;
}

impl EncoderConfigExt for EncoderConfig {
    fn same_session_shape(&self, other: &Self) -> bool {
        self.codec == other.codec
            && self.width == other.width
            && self.height == other.height
            && self.fps == other.fps
            && self.keyframe_interval_frames() == other.keyframe_interval_frames()
            && self.pixel_format == other.pixel_format
            && self.max_h264_profile == other.max_h264_profile
            && self.max_h265_profile == other.max_h265_profile
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitrate_alone_keeps_the_session() {
        let a = EncoderConfig::default();
        let b = EncoderConfig {
            bitrate_kbps: a.bitrate_kbps * 2,
            ..a.clone()
        };
        assert!(a.same_session_shape(&b));
        let c = EncoderConfig {
            width: 1280,
            ..a.clone()
        };
        assert!(!a.same_session_shape(&c));
    }
}
