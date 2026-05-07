//! Bounded ring buffer of finished segments. Producer (segmenter)
//! pushes; consumers (HTTP handlers) read by sequence number.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use tokio::sync::RwLock;

use ferricast_core::FerricastError;
use ferricast_m3u8::{M3u8Version, M3u8Writer, Segment as M3uSegment};

/// One muxed segment in the ring. `seq` is monotonic across the
/// lifetime of the server; players use it via `#EXT-X-MEDIA-SEQUENCE`
/// to correlate their continuity counter with our buffer.
#[derive(Clone)]
pub struct Segment {
    pub seq: u64,
    /// Wallclock duration spent capturing the segment, in seconds.
    pub duration_secs: f32,
    /// Set on the first segment we push after a stream restart.
    pub discontinuity: bool,
    pub data: Bytes,
}

pub struct SegmentRing {
    capacity: usize,
    segments: Vec<Segment>,
    next_seq: u64,
    /// Increments every time the segmenter has to start over (encoder
    /// restart, capture re-opened, …). Players advertise the latest
    /// value via `#EXT-X-DISCONTINUITY-SEQUENCE`.
    discontinuity_seq: u32,
}

impl SegmentRing {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            segments: Vec::with_capacity(capacity),
            next_seq: 0,
            discontinuity_seq: 0,
        }
    }

    /// Push a freshly produced segment, evicting the oldest entry
    /// when the ring is full. Returns the assigned sequence number.
    pub fn push(&mut self, duration_secs: f32, discontinuity: bool, data: Bytes) -> u64 {
        if discontinuity && !self.segments.is_empty() {
            self.discontinuity_seq = self.discontinuity_seq.saturating_add(1);
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        self.segments.push(Segment {
            seq,
            duration_secs,
            discontinuity,
            data,
        });
        // `Vec` rather than `VecDeque` — the ring is tiny (≤ ~10
        // entries in practice) so `remove(0)` is fine and we keep
        // contiguous slice access for `iter()`/`iter_mut()`.
        while self.segments.len() > self.capacity {
            self.segments.remove(0);
        }
        seq
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn get(&self, seq: u64) -> Option<&Segment> {
        // Linear scan — `capacity` is tiny.
        self.segments.iter().find(|s| s.seq == seq)
    }

    /// Build the live media playlist from the current ring contents.
    ///
    /// `min_target_duration` is the floor advertised in
    /// `#EXT-X-TARGETDURATION`; the actual value emitted is the
    /// maximum of that floor and `ceil(longest_observed_segment)`.
    /// Adapting upwards is required because the encoder's keyframe
    /// cadence (and capture jitter) determines real segment length —
    /// an under-sized target_duration would make every playlist
    /// rebuild fail RFC 8216 §4.3.3.1 and starve the player.
    pub fn build_playlist(
        &self,
        min_target_duration: u8,
    ) -> Result<String, FerricastError> {
        let observed_max = self
            .segments
            .iter()
            .map(|s| s.duration_secs)
            .filter(|d| d.is_finite() && *d > 0.0)
            .fold(0.0_f32, f32::max);
        let observed_ceil = if observed_max > 0.0 {
            (observed_max.ceil() as u32).min(u8::MAX as u32) as u8
        } else {
            0
        };
        let target = min_target_duration.max(observed_ceil).max(1);

        let first_seq = self.segments.first().map(|s| s.seq).unwrap_or(0);

        let mut w = M3u8Writer::live(M3u8Version::V3)
            .with_target_duration(target)
            .with_media_sequence(first_seq)
            .with_discontinuity_sequence(self.discontinuity_seq)
            .with_independent_segments(true);

        for s in &self.segments {
            let mut seg = M3uSegment::new(s.duration_secs, format!("segment-{}.ts", s.seq));
            seg.discontinuity = s.discontinuity;
            w.add_segment(seg)?;
        }

        w.to_string()
    }
}

/// Poll the ring until the segmenter has pushed at least one segment.
/// Cheap short-poll (50 ms) — the segmenter emits within
/// `segment_target_secs + keyframe_lag` so the loop runs only a few
/// times before exiting.
pub async fn wait_for_first_segment(ring: &Arc<RwLock<SegmentRing>>) {
    loop {
        if !ring.read().await.is_empty() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
