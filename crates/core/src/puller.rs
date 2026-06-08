//! Receiver-side media pull — fetch + demux a remote stream into
//! [`EncodedFrame`] / [`AudioFrame`] packets the decoder can consume.
//!
//! Sender of this data is whoever owns the URL: an HLS origin server,
//! a Chromecast sender app that just LOADed an `.m3u8`, an HTTP MP4
//! progressive download, etc. Concrete impls live in the protocol
//! crates (`ferricast-hls` for HLS, future `ferricast-dash` for DASH).
//!
//! Unified packet stream by design: HLS / MPEG-TS / fMP4 all
//! interleave video and audio in the underlying demuxer state, and
//! exposing two separate `next_video` / `next_audio` would force
//! every impl to buffer one stream while the caller polls the other.
//! `next` returns whichever packet the demuxer produced next; the
//! caller dispatches by variant.

use std::collections::HashMap;

use crate::error::Result;
use crate::frame::{AudioCodec, AudioFrame, EncodedFrame};
use crate::protocol::Codec;

/// Where + how to fetch a stream. Headers cover the common
/// authenticated-URL case (Cast LOAD with `Authorization`, signed
/// CDN URLs that need a cookie, etc).
#[derive(Debug, Clone)]
pub struct PullSpec {
    pub url: String,
    pub headers: HashMap<String, String>,
}

/// What the puller learned about the stream after opening it.
/// Receiver pipeline uses this to configure decoders, the FrameSink,
/// and any UI ("Playing: 1920x1080 H.264, 5.1 AAC").
#[derive(Debug, Clone)]
pub struct MediaInfo {
    pub video: Option<VideoStreamInfo>,
    pub audio: Option<AudioStreamInfo>,
    /// Total duration. `None` for live streams (HLS without
    /// `#EXT-X-ENDLIST`).
    pub duration_us: Option<u64>,
    /// Whether the source is live (no fixed end). Senders use this
    /// to grey out the seek bar / hide remaining-time displays.
    pub is_live: bool,
}

#[derive(Debug, Clone)]
pub struct VideoStreamInfo {
    pub codec: Codec,
    pub width: u32,
    pub height: u32,
    /// Nominal framerate from the stream metadata. Real cadence
    /// may vary; decoder MUST use packet PTS for timing.
    pub fps: u32,
}

#[derive(Debug, Clone)]
pub struct AudioStreamInfo {
    pub codec: AudioCodec,
    pub sample_rate: u32,
    pub channels: u16,
}

/// One demuxed packet. Video and audio share a single stream so the
/// caller sees them in the same order the source produced them,
/// which keeps A/V sync trivial — both go through their decoder in
/// arrival order and the [`crate::sink::FrameSink`] handles
/// presentation timing.
#[derive(Debug, Clone)]
pub enum MediaPacket {
    Video(EncodedFrame),
    Audio(AudioFrame),
    /// Stream ended cleanly. After this `next` should not be called;
    /// doing so returns [`crate::error::FerricastError::Pull`].
    Eos,
}

/// Pull and demux a remote stream.
///
/// Lifetime:
/// 1. `open` — resolve URL, fetch manifests/headers, identify codecs.
///    Returns the discovered [`MediaInfo`].
/// 2. Loop on `next` until you see [`MediaPacket::Eos`].
/// 3. `seek` at any time to jump in-stream (only meaningful when
///    [`MediaInfo::is_live`] is false; live impls MAY return an error).
/// 4. `close` to tear down sockets.
pub trait MediaPuller: Send {
    fn open(&mut self, spec: PullSpec) -> impl Future<Output = Result<MediaInfo>> + Send;
    fn next(&mut self) -> impl Future<Output = Result<MediaPacket>> + Send;
    fn seek(&mut self, position_us: u64) -> impl Future<Output = Result<()>> + Send;
    fn close(&mut self) -> impl Future<Output = Result<()>> + Send;
}
