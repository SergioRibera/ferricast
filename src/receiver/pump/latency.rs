//! Capture-to-output latency of a live stream, stage by stage.
//!
//! Live senders stamp their media with its capture time, mapped by the
//! protocol onto local wall-clock time (µs since the UNIX epoch). The
//! lag of a stage is then simply the wall clock when the item gets
//! there minus its timestamp: when it was read from the network
//! (`arrival`), when the decoder handed it out (`decoded`) and when it
//! is seen or heard (`shown` / `heard`). The figures include the
//! sender's own capture and encode time, and they are only as good as
//! the protocol's clock mapping.

use std::fmt;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// How often the summary is logged.
const REPORT_EVERY: Duration = Duration::from_secs(5);

/// The two kinds of stream the pump presents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    Video,
    Audio,
}

/// Where an item is in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Stage {
    Arrival,
    Decoded,
    /// Seen (video) or heard (audio).
    Output,
}

/// Wall-clock time in µs since the UNIX epoch, the clock live
/// timestamps are on.
fn wall_now_us() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_micros()).unwrap_or(i64::MAX))
        .unwrap_or_default()
}

/// Lags seen at one stage since the last report, in µs.
#[derive(Debug, Default, Clone, Copy)]
struct Lags {
    count: u64,
    sum: i64,
    min: i64,
    max: i64,
}

impl Lags {
    fn add(&mut self, lag_us: i64) {
        if self.count == 0 {
            self.min = lag_us;
            self.max = lag_us;
        } else {
            self.min = self.min.min(lag_us);
            self.max = self.max.max(lag_us);
        }
        self.count += 1;
        self.sum = self.sum.saturating_add(lag_us);
    }
}

impl fmt::Display for Lags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.count == 0 {
            return f.write_str("-");
        }
        let ms = |us: i64| us as f64 / 1000.0;
        let count = i64::try_from(self.count).unwrap_or(i64::MAX);
        write!(
            f,
            "{:.0}/{:.0}/{:.0}",
            ms(self.min),
            ms(self.sum / count),
            ms(self.max)
        )
    }
}

/// Lags of every stage of one stream.
#[derive(Debug, Default, Clone, Copy)]
struct StreamLags {
    arrival: Lags,
    decoded: Lags,
    output: Lags,
    /// Items dropped because they were too late to present.
    stale: u64,
}

impl StreamLags {
    fn stage(&mut self, stage: Stage) -> &mut Lags {
        match stage {
            Stage::Arrival => &mut self.arrival,
            Stage::Decoded => &mut self.decoded,
            Stage::Output => &mut self.output,
        }
    }
}

/// Collects stage lags and logs a summary every [`REPORT_EVERY`].
#[derive(Debug)]
pub(super) struct LatencyMeter {
    since: Instant,
    video: StreamLags,
    audio: StreamLags,
    first_video: bool,
    first_audio: bool,
}

impl Default for LatencyMeter {
    fn default() -> Self {
        Self {
            since: Instant::now(),
            video: StreamLags::default(),
            audio: StreamLags::default(),
            first_video: false,
            first_audio: false,
        }
    }
}

impl LatencyMeter {
    fn stream(&mut self, kind: Kind) -> &mut StreamLags {
        match kind {
            Kind::Video => &mut self.video,
            Kind::Audio => &mut self.audio,
        }
    }

    /// The item stamped `pts_us` reached `stage` now; for output, it is
    /// seen or heard `ahead` from now.
    pub(super) fn record(&mut self, kind: Kind, stage: Stage, pts_us: u64, ahead: Duration) {
        let ahead_us = i64::try_from(ahead.as_micros()).unwrap_or(i64::MAX);
        let pts = i64::try_from(pts_us).unwrap_or(i64::MAX);
        let lag_us = wall_now_us().saturating_add(ahead_us).saturating_sub(pts);
        tracing::trace!(?kind, ?stage, lag_ms = lag_us / 1000, "live latency");
        if stage == Stage::Output {
            let first = match kind {
                Kind::Video => &mut self.first_video,
                Kind::Audio => &mut self.first_audio,
            };
            if !*first {
                *first = true;
                tracing::info!(
                    ?kind,
                    lag_ms = lag_us / 1000,
                    "live stream: first output, capture-to-output latency"
                );
            }
        }
        self.stream(kind).stage(stage).add(lag_us);
        self.maybe_report();
    }

    /// An item of `kind` was dropped as too late to present.
    pub(super) fn stale(&mut self, kind: Kind) {
        self.stream(kind).stale += 1;
    }

    fn maybe_report(&mut self) {
        if self.since.elapsed() < REPORT_EVERY {
            return;
        }
        let (v, a) = (self.video, self.audio);
        tracing::info!(
            video_arrival = %v.arrival,
            video_decoded = %v.decoded,
            video_shown = %v.output,
            video_stale = v.stale,
            audio_arrival = %a.arrival,
            audio_decoded = %a.decoded,
            audio_heard = %a.output,
            audio_stale = a.stale,
            "live latency, ms min/avg/max since the last report"
        );
        self.video = StreamLags::default();
        self.audio = StreamLags::default();
        self.since = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lags_summarise_min_average_and_max() {
        let mut lags = Lags::default();
        assert_eq!(lags.to_string(), "-");
        for us in [100_000, 300_000, 200_000] {
            lags.add(us);
        }
        assert_eq!(lags.to_string(), "100/200/300");
    }

    #[test]
    fn output_lag_counts_the_time_ahead() {
        let mut meter = LatencyMeter::default();
        let pts = u64::try_from(wall_now_us()).unwrap();
        meter.record(Kind::Audio, Stage::Output, pts, Duration::from_millis(150));
        let heard = meter.audio.output;
        assert_eq!(heard.count, 1);
        assert!((150_000..200_000).contains(&heard.max), "{}", heard.max);
    }
}
