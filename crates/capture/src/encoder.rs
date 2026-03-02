use ferricast_core::{
    Codec, EncodedFrame, EncoderConfig, RawFrame, Result, VideoEncoder,
};

/// Passthrough "encoder" that wraps raw frames as encoded frames.
///
/// Placeholder until a real software encoder (e.g. openh264) is integrated.
pub struct PassthroughEncoder {
    codec: Codec,
    frame_count: u64,
    keyframe_interval: u32,
}

impl PassthroughEncoder {
    pub fn new(codec: Codec) -> Self {
        Self {
            codec,
            frame_count: 0,
            keyframe_interval: 60,
        }
    }
}

impl VideoEncoder for PassthroughEncoder {
    fn codec(&self) -> Codec {
        self.codec
    }

    fn configure(&mut self, config: &EncoderConfig) -> Result<()> {
        self.codec = config.codec;
        self.keyframe_interval = config.keyframe_interval;
        Ok(())
    }

    fn encode(&mut self, frame: &RawFrame) -> Result<EncodedFrame> {
        let is_keyframe = self.frame_count % self.keyframe_interval as u64 == 0;
        self.frame_count += 1;

        Ok(EncodedFrame {
            codec: self.codec,
            data: frame.data.clone(),
            timestamp_us: frame.timestamp_us,
            is_keyframe,
            duration_us: None,
        })
    }

    fn flush(&mut self) -> Result<Vec<EncodedFrame>> {
        Ok(Vec::new())
    }
}
