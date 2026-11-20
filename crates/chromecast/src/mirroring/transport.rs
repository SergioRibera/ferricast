//! UDP transport task for Cast Streaming: owns the socket and the per-stream
//! senders, paces RTP in bursts, sends Sender Reports and processes the
//! receiver's RTCP (openscreen `sender_packet_router.cc`).

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use ferricast_core::AdaptiveBitrateState;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use super::rtcp::{first_ssrc, parse_compound};
use super::rtp::FrameId;
use super::sender::{EnqueueRejected, StreamKind, StreamSender};

/// Minimum time between bursts.
pub(crate) const BURST_INTERVAL: Duration = Duration::from_millis(10);
/// Packets per burst for a 24 Mbit/s burst rate with 1472-byte packets and
/// 10 ms bursts: 24 Mi / 8 / 1472 / 100.
pub(crate) const MAX_PACKETS_PER_BURST: usize = 21;
/// How often link health and adaptive bitrate are evaluated.
const HEALTH_INTERVAL: Duration = Duration::from_secs(1);
/// How long to wait between key frame requests while waiting for one.
const KEY_FRAME_REQUEST_INTERVAL: Duration = Duration::from_secs(1);

/// One encoded frame handed to the transport.
#[derive(Debug, Clone)]
pub(crate) struct MediaFrame {
    pub(crate) kind: StreamKind,
    pub(crate) data: Bytes,
    pub(crate) is_key_frame: bool,
    pub(crate) capture_us: u64,
}

/// Link state shared with the session. Written by the transport task.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct LinkStats {
    pub(crate) first_video_sent_at: Option<Instant>,
    pub(crate) first_audio_sent_at: Option<Instant>,
    pub(crate) first_rtcp_at: Option<Instant>,
    pub(crate) last_rtcp_at: Option<Instant>,
    /// Highest video checkpoint reported by the receiver.
    pub(crate) video_checkpoint: Option<FrameId>,
    pub(crate) audio_checkpoint: Option<FrameId>,
    pub(crate) round_trip_time: Duration,
    pub(crate) video_frames_dropped: u64,
    pub(crate) packets_retransmitted: u64,
    pub(crate) packets_sent: u64,
}

impl LinkStats {
    /// The receiver has acknowledged at least one frame of `kind`.
    #[must_use]
    pub(crate) fn has_progress(&self, kind: StreamKind) -> bool {
        let checkpoint = match kind {
            StreamKind::Video => self.video_checkpoint,
            StreamKind::Audio => self.audio_checkpoint,
        };
        checkpoint.is_some_and(|c| c >= FrameId::FIRST)
    }
}

/// State shared between the session and the transport task.
#[derive(Debug, Default)]
pub(crate) struct TransportShared {
    pub(crate) key_frame_requested: AtomicBool,
    /// A video frame was dropped before reaching the transport.
    upstream_drop: AtomicBool,
    stats: Mutex<LinkStats>,
}

impl TransportShared {
    #[must_use]
    pub(crate) fn stats(&self) -> LinkStats {
        *self.stats.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Records that a video frame never reached the transport, so the next
    /// delta frame must not be sent.
    pub(crate) fn mark_upstream_drop(&self) {
        self.upstream_drop.store(true, Ordering::Relaxed);
    }

    fn update(&self, f: impl FnOnce(&mut LinkStats)) {
        f(&mut self.stats.lock().unwrap_or_else(PoisonError::into_inner));
    }
}

/// Owns the UDP socket and both stream senders.
pub(crate) struct Transport {
    socket: Arc<UdpSocket>,
    /// Audio first: it is smaller and more latency-sensitive, so it gets the
    /// head of every burst.
    senders: Vec<StreamSender>,
    shared: Arc<TransportShared>,
    adaptive: Option<Arc<AdaptiveBitrateState>>,
    last_burst: Option<Instant>,
    awaiting_video_key_frame: bool,
    last_key_frame_request: Option<Instant>,
    dropped_since_health: u64,
    packet_buf: Vec<u8>,
}

impl Transport {
    #[must_use]
    pub(crate) fn new(
        socket: Arc<UdpSocket>,
        mut senders: Vec<StreamSender>,
        shared: Arc<TransportShared>,
        adaptive: Option<Arc<AdaptiveBitrateState>>,
    ) -> Self {
        senders.sort_by_key(|s| match s.kind() {
            StreamKind::Audio => 0,
            StreamKind::Video => 1,
        });
        Self {
            socket,
            senders,
            shared,
            adaptive,
            last_burst: None,
            awaiting_video_key_frame: true,
            last_key_frame_request: None,
            dropped_since_health: 0,
            packet_buf: Vec::with_capacity(super::rtp::MAX_RTP_PACKET_SIZE),
        }
    }

    /// Runs until the frame channel closes.
    pub(crate) async fn run(mut self, mut frames: mpsc::Receiver<MediaFrame>) {
        let mut recv_buf = vec![0u8; 2048];
        let mut health = tokio::time::interval(HEALTH_INTERVAL);
        health.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            let wake = self.next_burst_at(Instant::now());
            let sleep_target = wake.map_or_else(
                || tokio::time::Instant::now() + Duration::from_secs(3600),
                tokio::time::Instant::from_std,
            );
            tokio::select! {
                frame = frames.recv() => match frame {
                    Some(frame) => self.enqueue(frame, Instant::now()),
                    None => break,
                },
                received = self.socket.recv(&mut recv_buf) => match received {
                    Ok(n) => {
                        if let Some(packet) = recv_buf.get(..n) {
                            self.on_packet(packet, Instant::now());
                        }
                    }
                    // ICMP port-unreachable from an earlier send surfaces
                    // here on a connected socket; the receiver may simply not
                    // be listening yet.
                    Err(e) if e.kind() == io::ErrorKind::ConnectionRefused => {
                        debug!("cast streaming: ICMP port unreachable from receiver");
                    }
                    Err(e) => {
                        warn!(%e, "cast streaming: UDP receive failed");
                        tokio::time::sleep(BURST_INTERVAL).await;
                    }
                },
                () = tokio::time::sleep_until(sleep_target), if wake.is_some() => {
                    self.send_burst(Instant::now()).await;
                }
                _ = health.tick() => self.on_health_tick(),
            }
        }
    }

    fn enqueue(&mut self, frame: MediaFrame, now: Instant) {
        if frame.kind == StreamKind::Video {
            if self.shared.upstream_drop.swap(false, Ordering::Relaxed) {
                self.awaiting_video_key_frame = true;
            }
            if frame.is_key_frame {
                self.awaiting_video_key_frame = false;
            } else if self.awaiting_video_key_frame {
                // Without the key frame the receiver cannot decode anything
                // that predicts from it.
                self.request_key_frame(now);
                return;
            }
        }
        let Some(sender) = self.senders.iter_mut().find(|s| s.kind() == frame.kind) else {
            return;
        };
        match sender.enqueue(&frame.data, frame.is_key_frame, frame.capture_us, now) {
            Ok(id) => {
                if id == FrameId::FIRST {
                    info!(kind = ?frame.kind, "cast streaming: first frame enqueued");
                    self.shared.update(|s| match frame.kind {
                        StreamKind::Video => s.first_video_sent_at = Some(now),
                        StreamKind::Audio => s.first_audio_sent_at = Some(now),
                    });
                }
            }
            Err(reason) => {
                debug!(kind = ?frame.kind, ?reason, "cast streaming: frame dropped");
                if frame.kind == StreamKind::Video {
                    self.dropped_since_health += 1;
                    self.shared.update(|s| s.video_frames_dropped += 1);
                    // The next delta frame would reference the dropped one.
                    self.awaiting_video_key_frame = true;
                    if reason != EnqueueRejected::PayloadTooLarge {
                        self.request_key_frame(now);
                    }
                }
            }
        }
    }

    fn request_key_frame(&mut self, now: Instant) {
        if self
            .last_key_frame_request
            .is_some_and(|at| now.saturating_duration_since(at) < KEY_FRAME_REQUEST_INTERVAL)
        {
            return;
        }
        self.last_key_frame_request = Some(now);
        self.shared
            .key_frame_requested
            .store(true, Ordering::Relaxed);
    }

    fn on_packet(&mut self, packet: &[u8], now: Instant) {
        // Only RTCP is expected from the receiver.
        if !packet.get(1).is_some_and(|pt| (200..=207).contains(pt)) {
            return;
        }
        let Some(ssrc) = first_ssrc(packet) else {
            return;
        };
        let Some(sender) = self.senders.iter_mut().find(|s| s.receiver_ssrc() == ssrc) else {
            debug!(ssrc, "cast streaming: RTCP for unknown receiver SSRC");
            return;
        };
        let parsed = match parse_compound(
            packet,
            sender.receiver_ssrc(),
            sender.sender_ssrc(),
            sender.last_enqueued(),
        ) {
            Ok(p) => p,
            Err(e) => {
                debug!(%e, len = packet.len(), "cast streaming: malformed RTCP ignored");
                return;
            }
        };
        let outcome = sender.on_rtcp(&parsed, now);
        let kind = sender.kind();
        let checkpoint = sender.checkpoint();
        let rtt = sender.round_trip_time();
        if outcome.packets_to_resend > 0 {
            debug!(kind = ?kind, packets = outcome.packets_to_resend, "cast streaming: NACK");
        }
        self.shared.update(|s| {
            s.first_rtcp_at.get_or_insert(now);
            s.last_rtcp_at = Some(now);
            match kind {
                StreamKind::Video => {
                    s.video_checkpoint = Some(checkpoint);
                    s.round_trip_time = rtt;
                }
                StreamKind::Audio => s.audio_checkpoint = Some(checkpoint),
            }
        });
        if outcome.key_frame_needed && kind == StreamKind::Video {
            debug!("cast streaming: picture loss, requesting key frame");
            self.last_key_frame_request = None;
            self.request_key_frame(now);
        }
    }

    /// Earliest time a burst should run, honouring the burst interval.
    fn next_burst_at(&self, now: Instant) -> Option<Instant> {
        let wanted = self
            .senders
            .iter()
            .flat_map(|s| [s.rtp_resume_time(now), s.rtcp_due_at()])
            .flatten()
            .min()?;
        let earliest = self.last_burst.map_or(wanted, |last| last + BURST_INTERVAL);
        Some(wanted.max(earliest))
    }

    async fn send_burst(&mut self, now: Instant) {
        self.last_burst = Some(now);
        let now_unix_us = unix_now_us();
        let mut budget = MAX_PACKETS_PER_BURST;
        // RTCP first: Sender Reports carry the lip-sync state the receiver
        // needs before it can play the RTP that follows.
        for i in 0..self.senders.len() {
            self.packet_buf.clear();
            let built = self
                .senders
                .get_mut(i)
                .is_some_and(|s| s.sender_report(now, now_unix_us, &mut self.packet_buf));
            if built {
                self.send_packet().await;
                budget = budget.saturating_sub(1);
            }
        }
        for i in 0..self.senders.len() {
            while budget > 0 {
                let built = self
                    .senders
                    .get_mut(i)
                    .is_some_and(|s| s.next_rtp_packet(now, &mut self.packet_buf));
                if !built {
                    break;
                }
                self.send_packet().await;
                budget -= 1;
            }
        }
    }

    async fn send_packet(&self) {
        if let Err(e) = self.socket.send(&self.packet_buf).await {
            debug!(%e, "cast streaming: UDP send failed");
        }
    }

    fn on_health_tick(&mut self) {
        let Some(video) = self.senders.iter().find(|s| s.kind() == StreamKind::Video) else {
            return;
        };
        let counters = video.counters();
        let rtt = video.round_trip_time();
        let in_flight = video.in_flight_media_duration(unix_now_us());
        let budget = video.target_playout_delay() * 2 / 3;
        let dropped = std::mem::take(&mut self.dropped_since_health);
        let packets_sent = self.senders.iter().map(|s| s.counters().packets_sent).sum();
        let retransmitted = self
            .senders
            .iter()
            .map(|s| s.counters().packets_retransmitted)
            .sum();
        self.shared.update(|s| {
            s.packets_sent = packets_sent;
            s.packets_retransmitted = retransmitted;
        });
        let Some(adaptive) = self.adaptive.as_deref() else {
            return;
        };
        // No receiver feedback yet: nothing to judge the link by.
        if video.last_rtcp_at().is_none() {
            return;
        }
        let pressure = link_pressure_pct(in_flight, budget, counters.fraction_lost, dropped);
        if let Some(new_kbps) = adaptive.record_pressure(pressure) {
            info!(
                new_kbps,
                pressure,
                rtt_ms = rtt.as_millis(),
                in_flight_ms = in_flight.as_millis(),
                "cast streaming: adaptive bitrate adjustment"
            );
        }
    }
}

/// Link pressure on the 0..=100+ scale [`AdaptiveBitrateState`] uses.
///
/// Media waiting for ACKs relative to two thirds of the playout delay is the
/// primary signal (it grows with RTT, pacing backlog and retransmissions);
/// receiver-reported packet loss and dropped frames push it up directly.
#[must_use]
pub(crate) fn link_pressure_pct(
    in_flight: Duration,
    budget: Duration,
    fraction_lost: u8,
    dropped_frames: u64,
) -> u32 {
    if dropped_frames > 0 {
        return 150;
    }
    let budget_us = budget.as_micros().max(1);
    let in_flight_pct = u32::try_from(in_flight.as_micros() * 100 / budget_us).unwrap_or(u32::MAX);
    // 8/256 lost (~3 %) already counts as saturation.
    let loss_pct = u32::from(fraction_lost) * 100 / 8;
    in_flight_pct.max(loss_pct).min(200)
}

fn unix_now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_micros()).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_scale() {
        let budget = Duration::from_millis(266);
        assert_eq!(link_pressure_pct(Duration::ZERO, budget, 0, 0), 0);
        assert_eq!(
            link_pressure_pct(Duration::from_millis(133), budget, 0, 0),
            50
        );
        assert_eq!(link_pressure_pct(Duration::ZERO, budget, 8, 0), 100);
        assert_eq!(link_pressure_pct(Duration::ZERO, budget, 0, 1), 150);
        assert_eq!(link_pressure_pct(Duration::from_secs(9), budget, 0, 0), 200);
    }

    #[test]
    fn burst_size_matches_openscreen_default() {
        let bits_per_second: usize = 24 << 20;
        let packets_per_second = bits_per_second / 8 / super::super::rtp::MAX_RTP_PACKET_SIZE;
        let bursts_per_second = 1000 / usize::try_from(BURST_INTERVAL.as_millis()).unwrap();
        assert_eq!(
            packets_per_second / bursts_per_second,
            MAX_PACKETS_PER_BURST
        );
    }
}
