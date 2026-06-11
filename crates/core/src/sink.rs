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
}
