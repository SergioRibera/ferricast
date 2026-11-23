//! AirPlay video: the sender hands the receiver a media URL (`POST /play`)
//! instead of mirroring its screen, then drives playback over HTTP on the
//! control port (`/rate`, `/scrub`, `/stop`) and polls `/playback-info`.
//! State changes go back as `POST /event` on the connection the sender
//! turned around with `POST /reverse` (see [`reverse`]).
//!
//! The media itself goes through the receiver pipeline as a
//! [`ferricast_core::MediaCommand::Load`], like a Cast LOAD: HLS through
//! the HLS puller, progressive files through the demuxer.

mod request;
pub(super) mod reverse;
mod session;

use std::time::Duration;

use ferricast_core::{PlaybackState, PlaybackStatus};
use plist::{Dictionary, Value};
use tokio::sync::mpsc;
use tokio::time::Instant;

pub(super) use request::{PlayRequest, StartPosition};
pub(super) use reverse::ReverseHub;
pub(super) use session::{VideoHandle, VideoHub, VideoRequest};

/// How far a reported position may be from a seek target and still count
/// as reached (HLS seeks land on segment boundaries).
const SEEK_TOLERANCE: Duration = Duration::from_secs(1);
/// How long reported positions are checked against the last seek target.
const SEEK_SETTLE: Duration = Duration::from_secs(5);

/// The `state` of an AirPlay video `/event`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EventState {
    Loading,
    Playing,
    Paused,
    Stopped,
}

impl EventState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Loading => "loading",
            Self::Playing => "playing",
            Self::Paused => "paused",
            Self::Stopped => "stopped",
        }
    }
}

/// Where the media is, as far as the sender is told.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// `/play` received, the pipeline is still opening the media.
    Loading,
    Playing,
    Paused,
    /// Ended, failed or stopped by the receiver.
    Stopped,
}

impl From<Phase> for EventState {
    fn from(phase: Phase) -> Self {
        match phase {
            Phase::Loading => Self::Loading,
            Phase::Playing => Self::Playing,
            Phase::Paused => Self::Paused,
            Phase::Stopped => Self::Stopped,
        }
    }
}

/// What a pipeline status report changes.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct StatusChange {
    /// The new state to tell the sender, when it changed.
    pub event: Option<EventState>,
    /// A start position given as a fraction, now resolvable: seek there.
    pub seek_us: Option<u64>,
}

/// One URL playback driven by a sender.
pub(super) struct VideoPlayback {
    pub request: PlayRequest,
    phase: Phase,
    /// Media position at `since`, or the frozen position when not playing.
    position_us: u64,
    /// When playback (re)started at `position_us`; `None` unless playing.
    since: Option<Instant>,
    duration_us: Option<u64>,
    is_live: bool,
    /// A fractional start position waiting for the duration.
    pending_start: Option<StartPosition>,
    /// The last seek target and when it was asked for, until a report
    /// confirms it.
    seeking: Option<(u64, Instant)>,
    /// Fires when another connection takes the receiver over.
    pub preempted: mpsc::Receiver<()>,
}

impl VideoPlayback {
    pub fn new(request: PlayRequest, preempted: mpsc::Receiver<()>) -> Self {
        let pending_start = match request.start {
            StartPosition::Fraction(_) if !request.start.is_beginning() => Some(request.start),
            StartPosition::Fraction(_) | StartPosition::Seconds(_) => None,
        };
        let position_us = match request.start {
            StartPosition::Seconds(_) => request.start.resolve_us(None).unwrap_or_default(),
            StartPosition::Fraction(_) => 0,
        };
        Self {
            request,
            phase: Phase::Loading,
            position_us,
            since: None,
            duration_us: None,
            is_live: false,
            pending_start,
            seeking: None,
            preempted,
        }
    }

    /// The start position to load the media at, when known up front.
    pub fn load_start_us(&self) -> Option<u64> {
        match self.request.start {
            StartPosition::Seconds(_) if !self.request.start.is_beginning() => {
                self.request.start.resolve_us(None)
            }
            StartPosition::Seconds(_) | StartPosition::Fraction(_) => None,
        }
    }

    pub fn event_state(&self) -> EventState {
        self.phase.into()
    }

    /// Current position, extrapolated while playing.
    pub fn position_us(&self) -> u64 {
        let elapsed = self.since.map_or(Duration::ZERO, |s| s.elapsed());
        let position = self
            .position_us
            .saturating_add(u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX));
        match self.duration_us {
            Some(duration) if !self.is_live => position.min(duration),
            _ => position,
        }
    }

    fn rebase(&mut self, position_us: u64) {
        self.position_us = position_us;
        self.since = (self.phase == Phase::Playing).then(Instant::now);
    }

    fn set_phase(&mut self, phase: Phase) -> Option<EventState> {
        let position = self.position_us();
        let changed = self.phase != phase;
        self.phase = phase;
        self.rebase(position);
        changed.then(|| phase.into())
    }

    /// The sender seeked: the pipeline does not report the new position,
    /// so the clock jumps here.
    pub fn seek(&mut self, position_us: u64) {
        let position_us = match self.duration_us {
            Some(duration) if !self.is_live => position_us.min(duration),
            _ => position_us,
        };
        self.rebase(position_us);
        self.seeking = Some((position_us, Instant::now()));
    }

    /// Whether a reported position can be trusted: the pipeline reports
    /// state changes with the position it had, which may predate a seek it
    /// has not carried out yet.
    fn plausible(&mut self, position_us: u64) -> bool {
        let Some((target, at)) = self.seeking else {
            return true;
        };
        let elapsed = at.elapsed();
        let tolerance = u64::try_from(SEEK_TOLERANCE.as_micros()).unwrap_or(u64::MAX);
        let reached = target
            .saturating_add(u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX))
            .saturating_add(tolerance);
        let plausible = (target.saturating_sub(tolerance)..=reached).contains(&position_us);
        if plausible || elapsed > SEEK_SETTLE {
            self.seeking = None;
            return true;
        }
        false
    }

    /// `/rate`: the sender paused (`false`) or resumed (`true`). Ignored
    /// until the media is loaded (the pipeline applies it on load).
    pub fn set_playing(&mut self, playing: bool) -> Option<EventState> {
        match self.phase {
            Phase::Playing | Phase::Paused => self.set_phase(if playing {
                Phase::Playing
            } else {
                Phase::Paused
            }),
            Phase::Loading | Phase::Stopped => None,
        }
    }

    /// Applies a status report of the pipeline.
    pub fn on_status(&mut self, status: &PlaybackStatus) -> StatusChange {
        if status.duration_us.is_some() {
            self.duration_us = status.duration_us;
        }
        self.is_live = status.is_live;
        let phase = match status.state {
            PlaybackState::Buffering => Phase::Loading,
            PlaybackState::Playing => Phase::Playing,
            PlaybackState::Paused => Phase::Paused,
            PlaybackState::Idle | PlaybackState::Ended | PlaybackState::Error(_) => Phase::Stopped,
        };
        let event = self.set_phase(phase);
        if let Some(position_us) = status.position_us
            && self.plausible(position_us)
        {
            self.rebase(position_us);
        }
        let seek_us = match phase {
            Phase::Playing | Phase::Paused => self
                .pending_start
                .and_then(|start| start.resolve_us(self.duration_us))
                .inspect(|_| self.pending_start = None),
            Phase::Loading | Phase::Stopped => None,
        };
        if let Some(position_us) = seek_us {
            self.seek(position_us);
        }
        StatusChange { event, seek_us }
    }

    fn ready(&self) -> bool {
        matches!(self.phase, Phase::Playing | Phase::Paused)
    }

    fn rate(&self) -> f64 {
        if self.phase == Phase::Playing {
            1.0
        } else {
            0.0
        }
    }

    /// Duration as the sender expects it: seconds, 0 while unknown.
    fn duration_secs(&self) -> f64 {
        if self.is_live {
            return 0.0;
        }
        secs(self.duration_us.unwrap_or_default())
    }

    /// Body of `GET /playback-info`.
    pub fn playback_info(&self) -> Dictionary {
        let position = secs(self.position_us());
        let duration = self.duration_secs();
        let ready = self.ready();
        let mut dict = Dictionary::new();
        dict.insert("duration".into(), Value::Real(duration));
        dict.insert("position".into(), Value::Real(position));
        dict.insert("rate".into(), Value::Real(self.rate()));
        dict.insert("readyToPlay".into(), Value::Boolean(ready));
        dict.insert("playbackBufferEmpty".into(), Value::Boolean(!ready));
        dict.insert("playbackBufferFull".into(), Value::Boolean(ready));
        dict.insert("playbackLikelyToKeepUp".into(), Value::Boolean(ready));
        let range = |start: f64, duration: f64| {
            let mut d = Dictionary::new();
            d.insert("start".into(), Value::Real(start));
            d.insert("duration".into(), Value::Real(duration));
            Value::Array(vec![Value::Dictionary(d)])
        };
        if ready {
            dict.insert(
                "loadedTimeRanges".into(),
                range(0.0, duration.max(position)),
            );
            dict.insert("seekableTimeRanges".into(), range(0.0, duration));
        }
        dict
    }

    /// Body of `GET /scrub`.
    pub fn scrub_text(&self) -> String {
        format!(
            "duration: {:.6}\nposition: {:.6}\n",
            self.duration_secs(),
            secs(self.position_us())
        )
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "media positions fit an f64 mantissa"
)]
fn secs(us: u64) -> f64 {
    us as f64 / 1_000_000.0
}

/// Parses `position=12.5` (seconds) from a `/scrub` query.
pub(super) fn scrub_position_us(uri: &str) -> Option<u64> {
    let secs = query_value(uri, "position")?.parse::<f64>().ok()?;
    if !secs.is_finite() {
        return None;
    }
    StartPosition::Seconds(secs).resolve_us(None)
}

/// Parses `value=0` / `value=1.000000` from a `/rate` query: whether to play.
pub(super) fn rate_playing(uri: &str) -> Option<bool> {
    let rate = query_value(uri, "value")?.parse::<f64>().ok()?;
    Some(rate > 0.0)
}

fn query_value<'a>(uri: &'a str, key: &str) -> Option<&'a str> {
    let (_, query) = uri.split_once('?')?;
    query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find(|(k, _)| *k == key)
        .map(|(_, v)| v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn play(start: StartPosition) -> VideoPlayback {
        let (_tx, rx) = mpsc::channel(1);
        VideoPlayback::new(
            PlayRequest {
                url: "http://h/a.m3u8".into(),
                start,
                playback_id: None,
                client: None,
            },
            rx,
        )
    }

    fn status(state: PlaybackState, position_us: u64) -> PlaybackStatus {
        PlaybackStatus {
            state,
            position_us: Some(position_us),
            duration_us: Some(20_000_000),
            is_live: false,
        }
    }

    #[test]
    fn queries_parse() {
        assert_eq!(scrub_position_us("/scrub?position=3.5"), Some(3_500_000));
        assert_eq!(scrub_position_us("/scrub?position=-1"), Some(0));
        assert_eq!(scrub_position_us("/scrub?position=nan"), None);
        assert_eq!(scrub_position_us("/scrub"), None);
        assert_eq!(rate_playing("/rate?value=0.000000"), Some(false));
        assert_eq!(rate_playing("/rate?value=1"), Some(true));
        assert_eq!(rate_playing("/rate?x=1&value=1.0"), Some(true));
    }

    #[tokio::test(start_paused = true)]
    async fn clock_follows_status_seeks_and_rate() {
        let mut v = play(StartPosition::Fraction(0.0));
        assert_eq!(v.event_state(), EventState::Loading);
        assert!(!v.ready());
        let change = v.on_status(&status(PlaybackState::Playing, 0));
        assert_eq!(change.event, Some(EventState::Playing));
        assert_eq!(change.seek_us, None);
        tokio::time::advance(Duration::from_secs(2)).await;
        assert_eq!(v.position_us(), 2_000_000);
        v.seek(10_000_000);
        tokio::time::advance(Duration::from_secs(1)).await;
        assert_eq!(v.position_us(), 11_000_000);
        assert_eq!(v.set_playing(false), Some(EventState::Paused));
        tokio::time::advance(Duration::from_secs(5)).await;
        assert_eq!(v.position_us(), 11_000_000);
        assert_eq!(v.set_playing(false), None);
        assert!(
            v.scrub_text()
                .starts_with("duration: 20.000000\nposition: 11.000000")
        );
        let info = v.playback_info();
        assert_eq!(info.get("rate").and_then(Value::as_real), Some(0.0));
        assert_eq!(
            info.get("readyToPlay").and_then(Value::as_boolean),
            Some(true)
        );
        // Never past the end.
        v.seek(30_000_000);
        assert_eq!(v.position_us(), 20_000_000);
        let change = v.on_status(&status(PlaybackState::Ended, 20_000_000));
        assert_eq!(change.event, Some(EventState::Stopped));
    }

    #[tokio::test(start_paused = true)]
    async fn reports_older_than_a_seek_are_ignored() {
        let mut v = play(StartPosition::Seconds(0.0));
        v.on_status(&status(PlaybackState::Playing, 0));
        tokio::time::advance(Duration::from_secs(2)).await;
        v.seek(10_000_000);
        // Paused before the pipeline carried the seek out.
        v.on_status(&status(PlaybackState::Paused, 1_800_000));
        assert_eq!(v.position_us(), 10_000_000);
        // A report at the target confirms it; later ones count again.
        v.on_status(&status(PlaybackState::Paused, 10_000_000));
        v.on_status(&status(PlaybackState::Paused, 1_000_000));
        assert_eq!(v.position_us(), 1_000_000);
    }

    #[test]
    fn fractional_start_seeks_once_the_duration_is_known() {
        let mut v = play(StartPosition::Fraction(0.5));
        assert_eq!(v.load_start_us(), None);
        let change = v.on_status(&status(PlaybackState::Playing, 0));
        assert_eq!(change.seek_us, Some(10_000_000));
        let change = v.on_status(&status(PlaybackState::Paused, 10_000_000));
        assert_eq!(change.seek_us, None);
        let v = play(StartPosition::Seconds(7.0));
        assert_eq!(v.load_start_us(), Some(7_000_000));
    }
}
