//! Bounded ring buffer of finished segments and their constituent
//! parts (LL-HLS). Producer (segmenter) pushes; consumers (HTTP
//! handlers) read by sequence number / (segment, part) index.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::sync::{Notify, RwLock};

use ferricast_core::FerricastError;
use ferricast_m3u8::{M3u8Version, M3u8Writer, Part as M3uPart, Segment as M3uSegment};

/// One LL-HLS partial segment ("part") within a regular segment.
/// Parts share the parent segment's MPEG-TS continuity counters, so
/// every part is a self-contained byte slice but only the part
/// flagged `independent` (the one containing the IDR) can be
/// decoded from cold — the rest need their preceding parts to be
/// processed first.
#[derive(Clone)]
pub struct Part {
    /// 0-based index within the parent segment.
    pub idx: u32,
    pub duration_secs: f32,
    /// True for the first part of a segment (the one carrying the
    /// IDR keyframe). Required by the LL-HLS spec; clients pick a
    /// "start playing here" point from independent parts.
    pub independent: bool,
    pub produced_at: Instant,
    pub data: Bytes,
}

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
    /// Monotonic timestamp when this segment was finished and
    /// inserted into the ring. The HTTP handler reads it on each
    /// GET to compute "staleness" — how long the segment sat in
    /// the ring before the receiver came to fetch it. Big stale
    /// numbers (>EXTINF) mean the receiver is starving (and we'll
    /// soon see 301).
    pub produced_at: Instant,
    pub data: Bytes,
    /// LL-HLS sub-parts. Empty when the segmenter wasn't running in
    /// LL-HLS mode. Each part's bytes are a slice of `data` (the
    /// concatenation of all parts equals the full segment), but we
    /// store them separately so the HTTP `/part-N.M.ts` route can
    /// serve them without re-slicing.
    pub parts: Vec<Part>,
}

pub struct SegmentRing {
    capacity: usize,
    segments: Vec<Segment>,
    next_seq: u64,
    /// Increments every time the segmenter has to start over (encoder
    /// restart, capture re-opened, …). Players advertise the latest
    /// value via `#EXT-X-DISCONTINUITY-SEQUENCE`.
    discontinuity_seq: u32,
    /// LL-HLS: parts that have been published for the segment
    /// currently under construction (no `data` yet because the
    /// segment isn't closed). When the segment is finished,
    /// `complete_pending_segment` moves these into the next
    /// `Segment.parts` entry.
    pending_parts: Vec<Part>,
    /// LL-HLS PART-TARGET advertised in the playlist. `Some(_)`
    /// gates emission of LL-HLS tags. The segmenter sets this from
    /// `HlsConfig::part_target_secs`.
    part_target_secs: Option<f32>,
    /// Fires whenever a new part or segment lands. The HTTP handler
    /// uses it to implement `#EXT-X-SERVER-CONTROL: CAN-BLOCK-RELOAD
    /// =YES`: a client asking for `?_HLS_msn=N&_HLS_part=M` blocks
    /// here until the ring reaches that point.
    pub notify: Arc<Notify>,
}

impl SegmentRing {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            segments: Vec::with_capacity(capacity),
            next_seq: 0,
            discontinuity_seq: 0,
            pending_parts: Vec::new(),
            part_target_secs: None,
            notify: Arc::new(Notify::new()),
        }
    }

    /// Enable LL-HLS playlist output by setting the advertised
    /// `PART-TARGET`. Idempotent; call before pushing any segment.
    pub fn enable_low_latency(&mut self, part_target_secs: f32) {
        self.part_target_secs = Some(part_target_secs);
    }

    /// What MSN the next segment will be assigned. The HTTP handler
    /// uses this to evaluate blocking-reload requests against the
    /// "current build" segment that doesn't have its body yet.
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    pub fn pending_parts(&self) -> &[Part] {
        &self.pending_parts
    }

    /// LL-HLS: add a freshly-flushed part to the in-progress segment.
    /// Returns the (msn, part_idx) tuple the client would use to
    /// reach it. Wakes everyone blocked on `notify`.
    pub fn push_pending_part(
        &mut self,
        duration_secs: f32,
        independent: bool,
        data: Bytes,
    ) -> (u64, u32) {
        let idx = self.pending_parts.len() as u32;
        let seq = self.next_seq;
        self.pending_parts.push(Part {
            idx,
            duration_secs,
            independent,
            produced_at: Instant::now(),
            data,
        });
        self.notify.notify_waiters();
        (seq, idx)
    }

    /// Push a freshly produced segment, evicting the oldest entry
    /// when the ring is full. Returns the assigned sequence number.
    pub fn push(&mut self, duration_secs: f32, discontinuity: bool, data: Bytes) -> u64 {
        self.push_with_parts(duration_secs, discontinuity, data, Vec::new())
    }

    /// LL-HLS variant: along with the full segment body, store the
    /// parts that were flushed during its build. Parts share their
    /// produced_at with the time they were originally flushed so the
    /// HTTP handler's staleness metric stays meaningful per part.
    pub fn push_with_parts(
        &mut self,
        duration_secs: f32,
        discontinuity: bool,
        data: Bytes,
        parts: Vec<Part>,
    ) -> u64 {
        if discontinuity && !self.segments.is_empty() {
            self.discontinuity_seq = self.discontinuity_seq.saturating_add(1);
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        self.segments.push(Segment {
            seq,
            duration_secs,
            discontinuity,
            produced_at: Instant::now(),
            data,
            parts,
        });
        while self.segments.len() > self.capacity {
            self.segments.remove(0);
        }
        // Pending parts are now part of the just-finished segment.
        // Clear so the next one can accumulate.
        self.pending_parts.clear();
        self.notify.notify_waiters();
        seq
    }

    /// Close the in-progress segment by consuming its accumulated
    /// `pending_parts` as the new segment's `parts`. Convenience for
    /// the segmenter — equivalent to a manual `push_with_parts` with
    /// `std::mem::take(&mut pending_parts)`.
    pub fn complete_pending_segment(
        &mut self,
        duration_secs: f32,
        discontinuity: bool,
        data: Bytes,
    ) -> u64 {
        let parts = std::mem::take(&mut self.pending_parts);
        self.push_with_parts(duration_secs, discontinuity, data, parts)
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn get(&self, seq: u64) -> Option<&Segment> {
        self.segments.iter().find(|s| s.seq == seq)
    }

    /// Retrieve a part by (segment_msn, part_idx). Looks first in the
    /// completed segments, then in the in-progress one.
    pub fn get_part(&self, seq: u64, idx: u32) -> Option<&Part> {
        if let Some(seg) = self.get(seq) {
            return seg.parts.iter().find(|p| p.idx == idx);
        }
        if seq == self.next_seq {
            return self.pending_parts.iter().find(|p| p.idx == idx);
        }
        None
    }

    /// Whether the ring has reached at least `(msn, part)`. Used by
    /// the blocking-reload HTTP handler.
    pub fn has_reached(&self, msn: u64, part: Option<u32>) -> bool {
        // The msn parameter may name a finished segment (any seg in
        // self.segments with that seq), or the in-progress one (==
        // self.next_seq, possibly with a pending part).
        if let Some(seg) = self.get(msn) {
            return match part {
                None => true,
                Some(p) => seg.parts.iter().any(|x| x.idx >= p),
            };
        }
        if msn == self.next_seq {
            return match part {
                None => false, // in-progress segment isn't done
                Some(p) => self.pending_parts.iter().any(|x| x.idx >= p),
            };
        }
        // msn is past next_seq (client is ahead of us) — not reached.
        // msn is before first seg (already evicted) — treat as reached
        // so the client immediately gets the current playlist.
        if let Some(first) = self.segments.first() {
            msn < first.seq
        } else {
            false
        }
    }

    /// Build the live media playlist from the current ring contents.
    pub fn build_playlist(&self, min_target_duration: u8) -> Result<String, FerricastError> {
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

        // Bump playlist version to 6 when LL-HLS is in play — Apple's
        // spec requires >= 6 for EXT-X-PART. Classic HLS sticks to 3.
        let version = if self.part_target_secs.is_some() {
            M3u8Version::V6
        } else {
            M3u8Version::V3
        };

        let mut w = M3u8Writer::live(version)
            .with_target_duration(target)
            .with_media_sequence(first_seq)
            .with_discontinuity_sequence(self.discontinuity_seq)
            .with_independent_segments(true);

        if let Some(pt) = self.part_target_secs {
            w = w.with_part_target(pt);
        }

        // Cap published parts to the most recent few segments — that's
        // all clients need to do a part-level seek to the live edge,
        // and keeping the playlist short matters for the chromecast's
        // tiny HLS parser buffer.
        let parts_window_segments = 3;
        let parts_visible_from = self.segments.len().saturating_sub(parts_window_segments);

        for (i, s) in self.segments.iter().enumerate() {
            let mut seg = M3uSegment::new(s.duration_secs, format!("segment-{}.ts", s.seq));
            seg.discontinuity = s.discontinuity;
            w.add_segment(seg)?;
            if i >= parts_visible_from && !s.parts.is_empty() {
                let parts_for_writer: Vec<M3uPart> = s
                    .parts
                    .iter()
                    .map(|p| {
                        let mut mp =
                            M3uPart::new(p.duration_secs, format!("part-{}.{}.ts", s.seq, p.idx));
                        mp.independent = p.independent;
                        mp.last_of_segment = p.idx as usize + 1 == s.parts.len();
                        mp
                    })
                    .collect();
                w.attach_parts(parts_for_writer)?;
            }
        }

        if !self.pending_parts.is_empty() {
            let pending_for_writer: Vec<M3uPart> = self
                .pending_parts
                .iter()
                .map(|p| {
                    let mut mp = M3uPart::new(
                        p.duration_secs,
                        format!("part-{}.{}.ts", self.next_seq, p.idx),
                    );
                    mp.independent = p.independent;
                    mp
                })
                .collect();
            w.set_pending_parts(pending_for_writer);
        }

        w.to_string()
    }
}

/// Poll the ring until the segmenter has pushed at least one segment.
pub async fn wait_for_first_segment(ring: &Arc<RwLock<SegmentRing>>) {
    loop {
        if !ring.read().await.is_empty() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1000)).await;
    }
}
