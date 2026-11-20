//! Per-peer HLS telemetry shared between the HTTP handler tasks.
//!
//! The HTTP handler is invoked once per TCP connection, so any
//! cross-request state — "how long since the last segment GET",
//! "have we collected enough samples to probe bandwidth yet" — has
//! to live behind an `Arc`. One [`SessionStats`] exists per
//! `HlsFrameSink` / `HlsServer`; inside it every client IP gets its
//! own counters, so a second viewer (VLC on a laptop, a loopback
//! self-test) neither skews the cast receiver's numbers nor its
//! bandwidth probe.
//!
//! Critical sections are a few field updates, so a `std` mutex is
//! enough (contention is at most one segment GET per peer every
//! couple of seconds).

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

/// Upper bound on peers with their own counters. Further peers still
/// get served, their deliveries are just not tracked across requests.
const MAX_TRACKED_PEERS: usize = 32;

/// Telemetry the HTTP handler updates on every segment GET, keyed by
/// client IP.
#[derive(Default)]
pub struct SessionStats {
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    peers: HashMap<IpAddr, PeerStats>,
    /// The first peer that fetched a segment. Without an allowlist it
    /// is the one whose deliveries drive the adaptive bitrate.
    first_peer: Option<IpAddr>,
}

/// Counters for one client IP.
#[derive(Default)]
struct PeerStats {
    /// When the last successful segment GET completed. `None` until
    /// the very first one. Subsequent GETs report `now - last` as
    /// the inter-request gap.
    last_segment_get_at: Option<Instant>,

    /// Last seq this peer fetched. Used to spot the receiver going
    /// backward (would indicate a fresh LOAD after a 301) or skipping
    /// forward (catching up).
    last_fetched_seq: Option<u64>,

    /// How many segment GETs this peer completed end-to-end.
    segment_get_count: u32,

    /// Per-segment effective Mbps for the bandwidth probe. Capped at
    /// `PROBE_SAMPLES`; once full we average and report the probe
    /// once, then leave it alone.
    probe_samples_mbps: Vec<f64>,

    /// Set once the probe was reported so it doesn't re-fire.
    probe_done: bool,
}

impl PeerStats {
    fn record(&mut self, seq: u64, mbps: f64, completed_at: Instant) -> SegmentGetTelemetry {
        let prev_get_at = self.last_segment_get_at.replace(completed_at);
        let prev_seq = self.last_fetched_seq.replace(seq);
        self.segment_get_count = self.segment_get_count.saturating_add(1);
        let probe_avg_mbps = if self.probe_done {
            None
        } else {
            self.probe_samples_mbps.push(mbps);
            (self.probe_samples_mbps.len() >= SessionStats::PROBE_SAMPLES).then(|| {
                self.probe_done = true;
                self.probe_samples_mbps.iter().sum::<f64>() / self.probe_samples_mbps.len() as f64
            })
        };
        SegmentGetTelemetry {
            inter_request_gap: prev_get_at.map(|t| completed_at.duration_since(t)),
            prev_seq,
            count_so_far: self.segment_get_count,
            probe_avg_mbps,
        }
    }
}

impl SessionStats {
    /// First N delivery measurements that feed the bandwidth probe.
    /// 3 is enough to filter out a single outlier (first segment is
    /// usually weird — short, only K-frame, etc.) while still
    /// converging fast (~12 s on a typical 4 s-segment stream).
    pub const PROBE_SAMPLES: usize = 3;

    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one successful segment GET by `peer`. Returns the
    /// peer's previous GET time and seq (so the caller can log the
    /// inter-request gap) and, once that peer has accumulated enough
    /// samples, the average measured Mbps for one-shot bandwidth
    /// probing — the caller decides whether to feed it to the
    /// adaptive controller.
    pub fn record_segment_get(
        &self,
        peer: IpAddr,
        seq: u64,
        mbps: f64,
        completed_at: Instant,
    ) -> SegmentGetTelemetry {
        let peer = peer.to_canonical();
        let mut g = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        g.first_peer.get_or_insert(peer);
        if !g.peers.contains_key(&peer) && g.peers.len() >= MAX_TRACKED_PEERS {
            return PeerStats::default().record(seq, mbps, completed_at);
        }
        g.peers
            .entry(peer)
            .or_default()
            .record(seq, mbps, completed_at)
    }

    /// Whether `peer` was the first client to fetch a segment.
    pub fn is_first_peer(&self, peer: IpAddr) -> bool {
        let g = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        g.first_peer == Some(peer.to_canonical())
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
    /// `Some` exactly once per peer, on the request that completed
    /// its bandwidth-probe window. The caller passes this to
    /// `AdaptiveBitrateState::observe_link_capacity_kbps` (after
    /// converting Mbps → kbps).
    pub probe_avg_mbps: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peers_are_tracked_independently() {
        let stats = SessionStats::new();
        let cast: IpAddr = "192.168.1.30".parse().unwrap();
        let viewer: IpAddr = "192.168.1.50".parse().unwrap();
        let t = Instant::now();
        assert!(
            stats
                .record_segment_get(cast, 0, 5.0, t)
                .probe_avg_mbps
                .is_none()
        );
        // Another peer's GETs don't advance the cast peer's probe.
        for seq in 0..5 {
            stats.record_segment_get(viewer, seq, 100.0, t);
        }
        let tele = stats.record_segment_get(cast, 1, 5.0, t);
        assert_eq!(tele.prev_seq, Some(0));
        assert_eq!(tele.count_so_far, 2);
        assert!(tele.probe_avg_mbps.is_none());
        let tele = stats.record_segment_get(cast, 2, 8.0, t);
        assert_eq!(tele.probe_avg_mbps, Some(6.0));
        assert!(stats.is_first_peer(cast));
        assert!(!stats.is_first_peer(viewer));
        // IPv4-mapped IPv6 is the same peer.
        assert!(stats.is_first_peer("::ffff:192.168.1.30".parse().unwrap()));
    }
}
