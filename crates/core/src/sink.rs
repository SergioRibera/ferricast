//! Where decoded frames go — render to a GUI surface, dump to a
//! file, re-encode for a downstream cast, drop on the floor.
//!
//! Kept deliberately small: the receiver pipeline doesn't need to
//! know whether the consumer is a Freya window or an mp4 writer. The
//! sink only owes the pipeline two things — "I took this video
//! frame" and "I took this audio chunk".
//!
//! Audio default impl drops silently so video-only sinks
//! (e.g. screenshot tools) don't have to write boilerplate.

use crate::decoder::DecodedAudio;
use crate::error::Result;
use crate::frame::CapturedFrame;

pub trait FrameSink: Send {
    fn push_video(&mut self, frame: CapturedFrame) -> impl Future<Output = Result<()>> + Send;

    fn push_audio(&mut self, _audio: DecodedAudio) -> impl Future<Output = Result<()>> + Send {
        async { Ok(()) }
    }
}
