//! Per-session HLS telemetry shared between the HTTP handler tasks.
//!
//! The HTTP handler is invoked once per TCP connection, so any
//! cross-request state — "how long since the last segment GET",
//! "have we collected enough samples to probe bandwidth yet" — has
//! to live behind an `Arc`. Construct one of these per
//! `HlsFrameSink` / `HlsServer` and clone the `Arc` into every
//! handler spawn.
//!
//! Everything here is lock-free or behind a `parking_lot`-style
//! short-critical-section `Mutex` (we use `tokio::sync::Mutex` to
//! match the rest of the crate, though contention is rare — at most
//! one segment GET per ~4 s on the steady path).

use std::sync::Mutex;
use std::time::Instant;

/// Telemetry the HTTP handler updates on every segment GET. Read by
/// nothing else today; the values exist for log lines that produce a
/// human-readable timeline of how the receiver is interacting with
/// our HLS endpoint.
pub struct SessionStats {
    inner: Mutex<Inner>,
}

struct Inner {
    /// When the last successful segment GET completed. `None` until
    /// the very first one. Subsequent GETs report `now - last` as
    /// the inter-request gap.
    pub last_segment_get_at: Option<Instant>,

    /// Highest seq the receiver has successfully fetched. Used to
    /// spot the receiver going backward (would indicate a fresh
    /// LOAD after a 301) or skipping forward (catching up).
    pub last_fetched_seq: Option<u64>,

    /// How many segment GETs we've completed end-to-end. Bandwidth
    /// probing uses the first few of these to seed the adaptive
    /// controller with a realistic target.
    pub segment_get_count: u32,

    /// Rolling buffer of per-segment effective Mbps for the
    /// bandwidth probe. Capped at PROBE_SAMPLES; once full we
    /// average and call the probe once, then leave it alone.
    pub probe_samples_mbps: Vec<f64>,

    /// Set once we've fed the bandwidth probe to the controller so
    /// we don't keep re-probing on every segment.
    pub probe_done: bool,
}

impl SessionStats {
    /// First N delivery measurements that feed the bandwidth probe.
    /// 3 is enough to filter out a single outlier (first segment is
    /// usually weird — short, only K-frame, etc.) while still
    /// converging fast (~12 s on a typical 4 s-segment stream).
    pub const PROBE_SAMPLES: usize = 3;

    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                last_segment_get_at: None,
                last_fetched_seq: None,
                segment_get_count: 0,
                probe_samples_mbps: Vec::with_capacity(Self::PROBE_SAMPLES),
                probe_done: false,
            }),
        }
    }

    /// Record one successful segment GET. Returns the previous
    /// `last_segment_get_at` (so the caller can compute the
    /// inter-request gap inline) and, when enough samples have
    /// accumulated, the average measured Mbps for one-shot
    /// bandwidth probing — the caller is then responsible for
    /// feeding it to the adaptive controller.
    pub fn record_segment_get(
        &self,
        seq: u64,
        mbps: f64,
        completed_at: Instant,
    ) -> SegmentGetTelemetry {
        let mut g = self.inner.lock().expect("stats mutex poisoned");
        let prev_get_at = g.last_segment_get_at.replace(completed_at);
        let prev_seq = g.last_fetched_seq.replace(seq);
        g.segment_get_count = g.segment_get_count.saturating_add(1);
        let probe_avg = if !g.probe_done && g.probe_samples_mbps.len() < Self::PROBE_SAMPLES {
            g.probe_samples_mbps.push(mbps);
            if g.probe_samples_mbps.len() == Self::PROBE_SAMPLES {
                let avg =
                    g.probe_samples_mbps.iter().sum::<f64>() / g.probe_samples_mbps.len() as f64;
                g.probe_done = true;
                Some(avg)
            } else {
                None
            }
        } else {
            None
        };
        SegmentGetTelemetry {
            inter_request_gap: prev_get_at.map(|t| completed_at.duration_since(t)),
            prev_seq,
            count_so_far: g.segment_get_count,
            probe_avg_mbps: probe_avg,
        }
    }
}

impl Default for SessionStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Returned by `record_segment_get`. Bundles every cross-request
/// value the handler wants on the same log line so the timeline is
/// readable without grep-correlating multiple lines.
#[derive(Debug)]
pub struct SegmentGetTelemetry {
    pub inter_request_gap: Option<std::time::Duration>,
    pub prev_seq: Option<u64>,
    pub count_so_far: u32,
    /// `Some` exactly once, on the request that completed the
    /// bandwidth-probe window. The caller passes this to
    /// `AdaptiveBitrateState::observe_link_capacity_kbps` (after
    /// converting Mbps → kbps).
    pub probe_avg_mbps: Option<f64>,
}
