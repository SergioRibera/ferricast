//! Where decoded frames go — render to a GUI surface, dump to a
//! file, re-encode for a downstream cast, drop on the floor.
//!
//! Kept deliberately small: the receiver pipeline doesn't need to
//! know whether the consumer is a Freya window or an mp4 writer.
//!
//! Timing contract: the pipeline paces delivery. Each video frame is
//! pushed at its presentation time (show it immediately; its
//! `timestamp_us` is informational). Audio is pushed a short, fixed
//! lead ahead of its presentation time, preceded by that much silence
//! whenever the timeline (re)starts, so a sink that plays audio
//! back-to-back as it arrives stays in sync with the video.
//!
//! Audio and volume default impls do nothing so video-only sinks
//! (e.g. screenshot tools) don't have to write boilerplate.
//!
//! Uses [`async_trait`] (same as [`crate::source::SourceEnumerator`])
//! so the manager can hold a `Box<dyn FrameSink>` — the receiver
//! pipeline is dyn-dispatch on purpose because the host application
//! supplies the concrete sink at runtime (per-window in the GUI
//! case) and the manager can't be generic over it.

use async_trait::async_trait;

use crate::decoder::DecodedAudio;
use crate::error::Result;
use crate::frame::CapturedFrame;

#[async_trait]
pub trait FrameSink: Send {
    async fn push_video(&mut self, frame: CapturedFrame) -> Result<()>;

    async fn push_audio(&mut self, _audio: DecodedAudio) -> Result<()> {
        Ok(())
    }

    /// Output gain for audio, `0.0` (silent, also used for mute) to
    /// `1.0` (unity). Applies to audio already queued as well.
    async fn set_volume(&mut self, _gain: f32) -> Result<()> {
        Ok(())
    }
}
