//! Media playlist (`.m3u8`) writer for HLS.
//!
//! Implements the subset of RFC 8216 that the live segmenter uses:
//! `#EXTM3U`, `#EXT-X-VERSION`, `#EXT-X-TARGETDURATION`,
//! `#EXT-X-MEDIA-SEQUENCE`, `#EXT-X-INDEPENDENT-SEGMENTS`,
//! `#EXT-X-DISCONTINUITY-SEQUENCE`, `#EXT-X-DISCONTINUITY`,
//! `#EXT-X-PROGRAM-DATE-TIME`, and `#EXTINF`.
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

        for seg in &self.segments {
            if seg.discontinuity {
                w.push_str("#EXT-X-DISCONTINUITY\n");
            }
            if let Some(pdt) = &seg.program_date_time {
                writeln!(w, "#EXT-X-PROGRAM-DATE-TIME:{}", pdt).map_err(fmt_err)?;
            }
            // `,` after EXTINF duration is mandatory; the title field
            // after the comma is intentionally empty (RFC 8216 §4.3.2.1).
            writeln!(w, "#EXTINF:{:.3},", seg.duration).map_err(fmt_err)?;
            writeln!(w, "{}", seg.uri).map_err(fmt_err)?;
        }

        if let PlaylistType::Vod = self.playlist_type {
            w.push_str("#EXT-X-ENDLIST\n");
        }

        Ok(())
    }
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
}
