//! Media playlist (`.m3u8`) writer for HLS.
//!
//! Implements the subset of RFC 8216 that the live segmenter uses:
//! `#EXTM3U`, `#EXT-X-VERSION`, `#EXT-X-TARGETDURATION`,
//! `#EXT-X-MEDIA-SEQUENCE`, `#EXT-X-INDEPENDENT-SEGMENTS`,
//! `#EXT-X-DISCONTINUITY-SEQUENCE`, `#EXT-X-DISCONTINUITY`,
//! `#EXT-X-PROGRAM-DATE-TIME`, `#EXTINF`, plus the Low-Latency HLS
//! additions (RFC 8216bis / Apple HLS 2nd edition):
//! `#EXT-X-PART-INF`, `#EXT-X-SERVER-CONTROL`, `#EXT-X-PART`,
//! `#EXT-X-PRELOAD-HINT`.
//!
//! Two playlist flavours:
//! * **Live** — no `#EXT-X-ENDLIST`, the player keeps polling.
//! * **VOD** — finalised with `#EXT-X-ENDLIST` once the source ends.
//!
//! No third-party dependencies; everything is plain `String`/`Write`.

use std::fmt::Write as _;

use ferricast_core::FerricastError;

/// HLS protocol version emitted in `#EXT-X-VERSION`. Version 3 is the
/// minimum that allows fractional `EXTINF` durations (RFC 8216 §7),
/// which we always need for live segments.
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum M3u8Version {
    V3 = 3,
    V6 = 6,
}

#[derive(Debug, Clone, Copy)]
pub enum PlaylistType {
    /// Sliding-window live playlist; never includes `#EXT-X-ENDLIST`.
    Live,
    /// Fixed VOD playlist; emits `#EXT-X-PLAYLIST-TYPE:VOD` and an
    /// `#EXT-X-ENDLIST` trailer.
    Vod,
}

#[derive(Debug, Clone)]
pub struct Segment {
    /// Wallclock duration, in seconds, with sub-second precision.
    pub duration: f32,
    /// Relative URL the player will fetch.
    pub uri: String,
    /// Set when this segment is the first after an encoder reset, a
    /// resolution change, or any other timeline break. Emits an
    /// `#EXT-X-DISCONTINUITY` tag immediately before the segment.
    pub discontinuity: bool,
    /// Optional wallclock anchor (`#EXT-X-PROGRAM-DATE-TIME`). Useful
    /// on the first segment so the player exposes a meaningful
    /// `Date.now()`-style time.
    pub program_date_time: Option<String>,
}

impl Segment {
    pub fn new(duration: f32, uri: impl Into<String>) -> Self {
        Self {
            duration,
            uri: uri.into(),
            discontinuity: false,
            program_date_time: None,
        }
    }
}

/// One partial segment ("part") inside a regular segment. Low-Latency
/// HLS publishes these as soon as they're ready — typically every
/// ~200-500 ms — so clients can pull data without waiting for the
/// full 4-6 s segment to close. The first part of a segment is the
/// one carrying the IDR keyframe and is therefore the only one with
/// `independent = true` (other parts are P-frames and need a prior
/// reference to decode).
#[derive(Debug, Clone)]
pub struct Part {
    pub duration: f32,
    pub uri: String,
    /// Set on the first part of a segment (the one containing the
    /// IDR). Required by the spec — clients use it to know which
    /// parts they can start playback from.
    pub independent: bool,
    /// Marks the part that *ends* its parent segment. Clients can
    /// use this together with `EXTINF` to validate boundary
    /// continuity, but it's optional and most players don't care.
    pub last_of_segment: bool,
}

impl Part {
    pub fn new(duration: f32, uri: impl Into<String>) -> Self {
        Self {
            duration,
            uri: uri.into(),
            independent: false,
            last_of_segment: false,
        }
    }
}

/// Builder for a media playlist. Field-by-field setters return `Self`
/// so a playlist can be assembled inline:
///
/// ```ignore
/// let mut w = M3u8Writer::live(M3u8Version::V3)
///     .with_target_duration(4)
///     .with_media_sequence(42);
/// w.add_segment(Segment::new(2.4, "video42.ts"))?;
/// let body = w.to_string()?;
/// ```
pub struct M3u8Writer {
    version: M3u8Version,
    target_duration: u8,
    media_sequence: u64,
    discontinuity_sequence: u32,
    independent_segments: bool,
    playlist_type: PlaylistType,
    segments: Vec<Segment>,
    /// Parts associated with each segment by index, plus a final
    /// trailing list for the currently-being-built segment that has
    /// no `EXTINF` yet. `parts_by_segment[i]` is the part list
    /// for `segments[i]`; `pending_parts` is for the in-progress one.
    parts_by_segment: Vec<Vec<Part>>,
    pending_parts: Vec<Part>,
    /// LL-HLS specific. Empty `None`s mean "regular HLS, omit the
    /// extension tags entirely" — a player that doesn't know LL-HLS
    /// just sees the classic playlist and works.
    part_target_secs: Option<f32>,
    can_block_reload: bool,
    /// How far back from the live edge clients should hold (in
    /// seconds). RFC 8216bis recommends 3 × PART-TARGET as a floor.
    part_hold_back_secs: Option<f32>,
    /// Same idea but for segment-grade clients (the ones that don't
    /// fetch parts). Recommended at least 3 × TARGETDURATION.
    hold_back_secs: Option<f32>,
}

impl Default for M3u8Writer {
    fn default() -> Self {
        Self {
            version: M3u8Version::V3,
            target_duration: 6,
            media_sequence: 0,
            discontinuity_sequence: 0,
            independent_segments: true,
            playlist_type: PlaylistType::Live,
            segments: Vec::new(),
            parts_by_segment: Vec::new(),
            pending_parts: Vec::new(),
            part_target_secs: None,
            can_block_reload: false,
            part_hold_back_secs: None,
            hold_back_secs: None,
        }
    }
}

impl M3u8Writer {
    pub fn live(version: M3u8Version) -> Self {
        Self {
            version,
            playlist_type: PlaylistType::Live,
            ..Self::default()
        }
    }

    pub fn vod(version: M3u8Version) -> Self {
        Self {
            version,
            playlist_type: PlaylistType::Vod,
            ..Self::default()
        }
    }

    pub fn with_target_duration(mut self, seconds: u8) -> Self {
        self.target_duration = seconds.max(1);
        self
    }

    pub fn with_media_sequence(mut self, seq: u64) -> Self {
        self.media_sequence = seq;
        self
    }

    pub fn with_discontinuity_sequence(mut self, seq: u32) -> Self {
        self.discontinuity_sequence = seq;
        self
    }

    pub fn with_independent_segments(mut self, on: bool) -> Self {
        self.independent_segments = on;
        self
    }

    /// Enable Low-Latency HLS. `part_target_secs` is the advertised
    /// `PART-TARGET` (Apple recommends 0.2–0.5 s; clients pace
    /// blocking-reload requests to ~3× this value). Setting this
    /// also implicitly turns on `CAN-BLOCK-RELOAD=YES` and sets
    /// reasonable defaults for the two hold-back values (3 × the
    /// target / part-target floors). Override afterwards if needed.
    pub fn with_part_target(mut self, secs: f32) -> Self {
        self.part_target_secs = Some(secs);
        self.can_block_reload = true;
        if self.part_hold_back_secs.is_none() {
            self.part_hold_back_secs = Some(secs * 3.0);
        }
        if self.hold_back_secs.is_none() {
            self.hold_back_secs = Some((self.target_duration as f32) * 3.0);
        }
        self
    }

    pub fn with_part_hold_back(mut self, secs: f32) -> Self {
        self.part_hold_back_secs = Some(secs);
        self
    }

    pub fn with_hold_back(mut self, secs: f32) -> Self {
        self.hold_back_secs = Some(secs);
        self
    }

    /// Attach a list of parts to the most-recently-added segment.
    /// Call after `add_segment`. Empty lists are a no-op.
    pub fn attach_parts(&mut self, parts: Vec<Part>) -> Result<(), FerricastError> {
        if self.segments.is_empty() {
            return Err(FerricastError::M3u8(
                "attach_parts called before any segment was added".into(),
            ));
        }
        while self.parts_by_segment.len() < self.segments.len() {
            self.parts_by_segment.push(Vec::new());
        }
        self.parts_by_segment[self.segments.len() - 1] = parts;
        Ok(())
    }

    /// Attach parts for the in-progress segment that hasn't closed
    /// yet (no `EXTINF` for it). These render at the tail of the
    /// playlist; clients use them to start playback before the
    /// segment's full body is published.
    pub fn set_pending_parts(&mut self, parts: Vec<Part>) {
        self.pending_parts = parts;
    }

    /// Append a segment. Validates the RFC 8216 §4.3.3.1 constraint
    /// that `EXTINF` rounded to the nearest integer never exceeds
    /// `EXT-X-TARGETDURATION`; rejecting it here is friendlier than
    /// letting the player error out later.
    pub fn add_segment(&mut self, segment: Segment) -> Result<(), FerricastError> {
        if !segment.duration.is_finite() || segment.duration < 0.0 {
            return Err(FerricastError::M3u8(format!(
                "invalid segment duration {:.3}",
                segment.duration
            )));
        }
        let rounded = segment.duration.round().max(1.0) as u32;
        if rounded > self.target_duration as u32 {
            return Err(FerricastError::M3u8(format!(
                "segment duration {:.3}s rounds to {}s, exceeds target-duration {}s",
                segment.duration, rounded, self.target_duration
            )));
        }
        self.segments.push(segment);
        Ok(())
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// Serialise to a UTF-8 string. This is the form sent over HTTP.
    pub fn to_string(&self) -> Result<String, FerricastError> {
        let mut s = String::with_capacity(128 + self.segments.len() * 64);
        self.write(&mut s)?;
        Ok(s)
    }

    fn write(&self, w: &mut String) -> Result<(), FerricastError> {
        // Header tags must appear in the order defined by RFC 8216
        // §4.3.1: file-wide tags before any segment tag.
        w.push_str("#EXTM3U\n");
        writeln!(w, "#EXT-X-VERSION:{}", self.version as u8).map_err(fmt_err)?;
        if self.independent_segments {
            w.push_str("#EXT-X-INDEPENDENT-SEGMENTS\n");
        }
        if let PlaylistType::Vod = self.playlist_type {
            w.push_str("#EXT-X-PLAYLIST-TYPE:VOD\n");
        }
        writeln!(w, "#EXT-X-TARGETDURATION:{}", self.target_duration).map_err(fmt_err)?;
        writeln!(w, "#EXT-X-MEDIA-SEQUENCE:{}", self.media_sequence).map_err(fmt_err)?;
        if self.discontinuity_sequence > 0 {
            writeln!(
                w,
                "#EXT-X-DISCONTINUITY-SEQUENCE:{}",
                self.discontinuity_sequence
            )
            .map_err(fmt_err)?;
        }

        // Low-Latency HLS tags, when enabled. PART-INF tells the
        // client what the typical part length is; SERVER-CONTROL
        // advertises CAN-BLOCK-RELOAD and the hold-back values so
        // the client knows how to pace its blocking playlist reload
        // requests.
        if let Some(pt) = self.part_target_secs {
            writeln!(w, "#EXT-X-PART-INF:PART-TARGET={:.3}", pt).map_err(fmt_err)?;
        }
        if self.can_block_reload
            || self.part_hold_back_secs.is_some()
            || self.hold_back_secs.is_some()
        {
            let mut control = String::from("#EXT-X-SERVER-CONTROL:");
            let mut first = true;
            if self.can_block_reload {
                control.push_str("CAN-BLOCK-RELOAD=YES");
                first = false;
            }
            if let Some(hb) = self.hold_back_secs {
                if !first {
                    control.push(',');
                }
                let _ = write!(&mut control, "HOLD-BACK={:.3}", hb);
                first = false;
            }
            if let Some(phb) = self.part_hold_back_secs {
                if !first {
                    control.push(',');
                }
                let _ = write!(&mut control, "PART-HOLD-BACK={:.3}", phb);
            }
            control.push('\n');
            w.push_str(&control);
        }

        for (i, seg) in self.segments.iter().enumerate() {
            if seg.discontinuity {
                w.push_str("#EXT-X-DISCONTINUITY\n");
            }
            if let Some(pdt) = &seg.program_date_time {
                writeln!(w, "#EXT-X-PROGRAM-DATE-TIME:{}", pdt).map_err(fmt_err)?;
            }
            // Per RFC 8216bis: when both EXT-X-PART entries and the
            // full segment line are present for the same media, the
            // EXT-X-PARTs come BEFORE the EXTINF. They name the
            // sub-pieces a client could have stitched together to
            // arrive at the same byte stream as the segment URI.
            if let Some(parts) = self.parts_by_segment.get(i) {
                for part in parts {
                    write_part_line(w, part)?;
                }
            }
            // `,` after EXTINF duration is mandatory; the title field
            // after the comma is intentionally empty (RFC 8216 §4.3.2.1).
            writeln!(w, "#EXTINF:{:.3},", seg.duration).map_err(fmt_err)?;
            writeln!(w, "{}", seg.uri).map_err(fmt_err)?;
        }

        // Pending (not-yet-closed) segment's parts, if any. These
        // have no EXTINF or URI following them — clients that
        // understand LL-HLS pick them up and start playback ahead of
        // the segment closure; classic clients ignore the tags and
        // wait for the next playlist reload to see the EXTINF.
        for part in &self.pending_parts {
            write_part_line(w, part)?;
        }

        if let PlaylistType::Vod = self.playlist_type {
            w.push_str("#EXT-X-ENDLIST\n");
        }

        Ok(())
    }
}

fn write_part_line(w: &mut String, part: &Part) -> Result<(), FerricastError> {
    let mut line = String::from("#EXT-X-PART:");
    let _ = write!(&mut line, "DURATION={:.3}", part.duration);
    let _ = write!(&mut line, ",URI=\"{}\"", part.uri);
    if part.independent {
        line.push_str(",INDEPENDENT=YES");
    }
    if part.last_of_segment {
        line.push_str(",GAP=NO");
    }
    line.push('\n');
    w.push_str(&line);
    Ok(())
}

fn fmt_err(_: std::fmt::Error) -> FerricastError {
    FerricastError::M3u8("playlist formatting failed".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_playlist_round_trip() {
        let mut w = M3u8Writer::live(M3u8Version::V3)
            .with_target_duration(4)
            .with_media_sequence(10);
        w.add_segment(Segment::new(2.5, "video10.ts")).unwrap();
        w.add_segment(Segment::new(3.0, "video11.ts")).unwrap();
        let s = w.to_string().unwrap();
        assert!(s.starts_with("#EXTM3U\n"));
        assert!(s.contains("#EXT-X-VERSION:3"));
        assert!(s.contains("#EXT-X-TARGETDURATION:4"));
        assert!(s.contains("#EXT-X-MEDIA-SEQUENCE:10"));
        assert!(s.contains("#EXTINF:2.500,"));
        assert!(s.contains("video10.ts"));
        // Live playlists must not be terminated.
        assert!(!s.contains("#EXT-X-ENDLIST"));
    }

    #[test]
    fn vod_has_endlist() {
        let mut w = M3u8Writer::vod(M3u8Version::V3).with_target_duration(4);
        w.add_segment(Segment::new(2.0, "0.ts")).unwrap();
        let s = w.to_string().unwrap();
        assert!(s.contains("#EXT-X-PLAYLIST-TYPE:VOD"));
        assert!(s.trim_end().ends_with("#EXT-X-ENDLIST"));
    }

    #[test]
    fn rejects_oversize_segment() {
        let mut w = M3u8Writer::live(M3u8Version::V3).with_target_duration(2);
        // 2.6 rounds to 3 > 2.
        let r = w.add_segment(Segment::new(2.6, "x.ts"));
        assert!(r.is_err());
    }

    #[test]
    fn discontinuity_renders_before_segment() {
        let mut w = M3u8Writer::live(M3u8Version::V3).with_target_duration(4);
        let mut s1 = Segment::new(2.0, "a.ts");
        s1.discontinuity = true;
        w.add_segment(s1).unwrap();
        let s = w.to_string().unwrap();
        let disc = s.find("#EXT-X-DISCONTINUITY").unwrap();
        let extinf = s.find("#EXTINF").unwrap();
        assert!(disc < extinf);
    }

    #[test]
    fn ll_hls_tags_appear_when_part_target_set() {
        let w = M3u8Writer::live(M3u8Version::V6)
            .with_target_duration(4)
            .with_part_target(0.5);
        let s = w.to_string().unwrap();
        assert!(s.contains("#EXT-X-PART-INF:PART-TARGET=0.500"));
        assert!(s.contains("#EXT-X-SERVER-CONTROL"));
        assert!(s.contains("CAN-BLOCK-RELOAD=YES"));
        assert!(s.contains("PART-HOLD-BACK=1.500")); // 3 × 0.5
        assert!(s.contains("HOLD-BACK=12.000")); // 3 × target_duration=4
    }

    #[test]
    fn ll_hls_omitted_when_not_enabled() {
        let mut w = M3u8Writer::live(M3u8Version::V3).with_target_duration(4);
        w.add_segment(Segment::new(2.0, "a.ts")).unwrap();
        let s = w.to_string().unwrap();
        assert!(!s.contains("#EXT-X-PART"));
        assert!(!s.contains("#EXT-X-SERVER-CONTROL"));
    }

    #[test]
    fn parts_render_before_their_segment() {
        let mut w = M3u8Writer::live(M3u8Version::V6)
            .with_target_duration(4)
            .with_part_target(0.5);
        w.add_segment(Segment::new(2.0, "a.ts")).unwrap();
        let mut p0 = Part::new(0.5, "a.0.ts");
        p0.independent = true;
        let p1 = Part::new(0.5, "a.1.ts");
        w.attach_parts(vec![p0, p1]).unwrap();
        let s = w.to_string().unwrap();
        let p_idx = s.find("#EXT-X-PART:DURATION=0.500,URI=\"a.0.ts\"").unwrap();
        let extinf = s.find("#EXTINF:2.000").unwrap();
        assert!(p_idx < extinf, "parts must precede their parent EXTINF");
        assert!(s.contains("INDEPENDENT=YES"));
    }

    #[test]
    fn pending_parts_render_at_tail() {
        let mut w = M3u8Writer::live(M3u8Version::V6)
            .with_target_duration(4)
            .with_part_target(0.5);
        w.add_segment(Segment::new(2.0, "a.ts")).unwrap();
        let mut p0 = Part::new(0.5, "b.0.ts");
        p0.independent = true;
        w.set_pending_parts(vec![p0]);
        let s = w.to_string().unwrap();
        let extinf = s.find("#EXTINF").unwrap();
        let pending = s.find("#EXT-X-PART:DURATION=0.500,URI=\"b.0.ts\"").unwrap();
        assert!(
            pending > extinf,
            "pending parts must come after closed segments"
        );
    }
}
