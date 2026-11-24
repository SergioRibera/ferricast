//! Per-stream Cast Streaming sender state (openscreen `sender_impl.cc`):
//! in-flight frame bookkeeping, ACK/NACK handling, retransmission, kickstart,
//! round-trip-time estimation and Sender Report contents.
//!
//! All methods take the current time as arguments so the state machine is
//! deterministic and testable.

use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

use bytes::Bytes;

use super::rtcp::{
    CompoundRtcp, DlrrReply, ReceiverFeedback, SenderReport, build_sender_report, ntp_middle_32,
    unix_us_to_ntp,
};
use super::rtp::{
    ALL_PACKETS_LOST, BASE_RTP_HEADER_SIZE, EncryptedFrame, FrameCrypto, FrameId, RtpPacketizer,
    RtpPayloadType, packet_count_for,
};

/// Largest span of frame ids that may be unacknowledged. Fits in the 8-bit
/// wire frame id with room to compare across wrap-around.
pub(crate) const MAX_UNACKED_FRAMES: i64 = 120;
/// Interval between Sender Reports once the round trip time is known.
pub(crate) const RTCP_REPORT_INTERVAL: Duration = Duration::from_millis(500);
/// Floor of the in-flight window: two 30 fps frames plus capture jitter.
const MIN_IN_FLIGHT: Duration = Duration::from_micros(81_667);
/// Cap on the media duration attributed to one inter-frame step, so capture
/// pauses don't count as in-flight media.
const MAX_UNPAUSED_FRAME_DURATION: Duration = Duration::from_millis(50);
/// RTT assumed when a measurement comes out at or below zero.
const NEAR_ZERO_RTT: Duration = Duration::from_micros(75);
/// Sender Reports remembered for matching receiver report blocks.
const RECENT_SENDER_REPORTS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamKind {
    Audio,
    Video,
}

/// Negotiated parameters of one stream.
#[derive(Clone)]
pub(crate) struct StreamParams {
    pub(crate) kind: StreamKind,
    pub(crate) payload_type: RtpPayloadType,
    pub(crate) sender_ssrc: u32,
    pub(crate) receiver_ssrc: u32,
    pub(crate) rtp_timebase: u32,
    pub(crate) target_playout_delay: Duration,
    pub(crate) crypto: FrameCrypto,
    pub(crate) first_sequence: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EnqueueRejected {
    /// More than [`MAX_UNACKED_FRAMES`] frames would be unacknowledged.
    IdSpanLimit,
    /// More media than the playout window allows is still in flight.
    InFlightDurationLimit,
    /// The frame needs more packets than 16-bit packet ids can address.
    PayloadTooLarge,
}

/// What a received RTCP packet changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RtcpOutcome {
    /// The receiver asked for a key frame and none is already in flight.
    pub(crate) key_frame_needed: bool,
    pub(crate) checkpoint_advanced: bool,
    pub(crate) packets_to_resend: usize,
}

struct PendingFrame {
    frame: EncryptedFrame,
    /// Packets that still need to be (re)sent.
    send_flags: Vec<bool>,
    sent_at: Vec<Option<Instant>>,
    /// Clamped media duration up to the next enqueued frame.
    media_duration: Duration,
}

impl PendingFrame {
    fn first_flagged(&self) -> Option<u16> {
        self.send_flags
            .iter()
            .position(|&f| f)
            .and_then(|i| u16::try_from(i).ok())
    }
}

/// Counters exposed for logging and bitrate adaptation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SenderCounters {
    pub(crate) packets_sent: u64,
    pub(crate) packets_retransmitted: u64,
    pub(crate) frames_acked: u64,
    pub(crate) frames_rejected: u64,
    pub(crate) fraction_lost: u8,
}

pub(crate) struct StreamSender {
    params: StreamParams,
    packetizer: RtpPacketizer,
    frames: BTreeMap<FrameId, PendingFrame>,
    checkpoint: FrameId,
    last_enqueued: FrameId,
    latest_expected: FrameId,
    last_key_frame: FrameId,
    round_trip_time: Duration,
    /// Lip-sync mapping of the most recent frame: (capture UNIX µs, RTP ts).
    pending_report: Option<(u64, u32)>,
    packet_count: u32,
    octet_count: u32,
    recent_reports: VecDeque<(u32, Instant)>,
    first_report_at: Option<Instant>,
    next_report_at: Option<Instant>,
    latest_receiver_ntp: Option<u64>,
    /// Receiver reference time to echo in the next DLRR, and when it arrived.
    last_receiver_reference: Option<(u32, Instant)>,
    last_rtcp_at: Option<Instant>,
    counters: SenderCounters,
}

impl StreamSender {
    #[must_use]
    pub(crate) fn new(params: StreamParams) -> Self {
        let packetizer = RtpPacketizer::new(
            params.payload_type,
            params.sender_ssrc,
            params.first_sequence,
        );
        Self {
            params,
            packetizer,
            frames: BTreeMap::new(),
            checkpoint: FrameId::LEADER,
            last_enqueued: FrameId::LEADER,
            latest_expected: FrameId::LEADER,
            last_key_frame: FrameId::LEADER,
            round_trip_time: Duration::ZERO,
            pending_report: None,
            packet_count: 0,
            octet_count: 0,
            recent_reports: VecDeque::with_capacity(RECENT_SENDER_REPORTS),
            first_report_at: None,
            next_report_at: None,
            latest_receiver_ntp: None,
            last_receiver_reference: None,
            last_rtcp_at: None,
            counters: SenderCounters::default(),
        }
    }

    #[must_use]
    pub(crate) fn kind(&self) -> StreamKind {
        self.params.kind
    }

    #[must_use]
    pub(crate) fn receiver_ssrc(&self) -> u32 {
        self.params.receiver_ssrc
    }

    #[must_use]
    pub(crate) fn sender_ssrc(&self) -> u32 {
        self.params.sender_ssrc
    }

    #[must_use]
    pub(crate) fn checkpoint(&self) -> FrameId {
        self.checkpoint
    }

    #[must_use]
    pub(crate) fn last_enqueued(&self) -> FrameId {
        self.last_enqueued
    }

    #[must_use]
    pub(crate) fn round_trip_time(&self) -> Duration {
        self.round_trip_time
    }

    #[must_use]
    pub(crate) fn last_rtcp_at(&self) -> Option<Instant> {
        self.last_rtcp_at
    }

    #[must_use]
    pub(crate) fn counters(&self) -> SenderCounters {
        self.counters
    }

    #[must_use]
    pub(crate) fn target_playout_delay(&self) -> Duration {
        self.params.target_playout_delay
    }

    /// RTP timestamp for a capture time in UNIX-epoch microseconds. The RTP
    /// clock is the capture clock scaled to the stream timebase.
    #[must_use]
    pub(crate) fn rtp_timestamp_for(&self, unix_us: u64) -> u32 {
        let ticks = u128::from(unix_us) * u128::from(self.params.rtp_timebase) / 1_000_000;
        // Truncation to 32 bits is the RTP wrap-around.
        (ticks & 0xFFFF_FFFF) as u32
    }

    /// Media duration currently waiting for an ACK, counting the step to a
    /// frame captured at `next_capture_us`.
    #[must_use]
    pub(crate) fn in_flight_media_duration(&self, next_capture_us: u64) -> Duration {
        let mut total: Duration = self.frames.values().map(|p| p.media_duration).sum();
        if let Some(newest) = self.frames.get(&self.last_enqueued) {
            let step = next_capture_us.saturating_sub(newest.frame.reference_time_us);
            total += Duration::from_micros(step).min(MAX_UNPAUSED_FRAME_DURATION);
        }
        total
    }

    /// Longest media duration allowed in flight when a new frame arrives.
    ///
    /// Audio follows openscreen (whole playout delay). Video frames here are
    /// already encoded and predict from their predecessor, so dropping one
    /// costs a key frame; the window is therefore the full playout delay
    /// (anything older would miss its deadline anyway) rather than
    /// openscreen's RTT-based window, which assumes the encoder can skip a
    /// capture instead.
    #[must_use]
    pub(crate) fn max_in_flight_media_duration(&self) -> Duration {
        self.params.target_playout_delay.max(MIN_IN_FLIGHT)
    }

    /// Encrypts and queues a frame for sending. Delta frames reference the
    /// previous frame: the encoders feeding this sender emit no B-frames.
    ///
    /// # Errors
    /// Returns why the frame was not accepted; the caller decides whether a
    /// key frame is needed to recover.
    pub(crate) fn enqueue(
        &mut self,
        data: &[u8],
        is_key_frame: bool,
        capture_us: u64,
        now: Instant,
    ) -> Result<FrameId, EnqueueRejected> {
        let frame_id = self.last_enqueued.next();
        if frame_id.distance_from(self.checkpoint) > MAX_UNACKED_FRAMES {
            self.counters.frames_rejected += 1;
            return Err(EnqueueRejected::IdSpanLimit);
        }
        if self.in_flight_media_duration(capture_us) > self.max_in_flight_media_duration() {
            self.counters.frames_rejected += 1;
            return Err(EnqueueRejected::InFlightDurationLimit);
        }
        let Some(packet_count) = packet_count_for(data.len()) else {
            self.counters.frames_rejected += 1;
            return Err(EnqueueRejected::PayloadTooLarge);
        };

        let rtp_timestamp = self.rtp_timestamp_for(capture_us);
        let frame = EncryptedFrame {
            frame_id,
            referenced_frame_id: if is_key_frame {
                frame_id
            } else {
                frame_id.offset(-1)
            },
            is_key_frame,
            rtp_timestamp,
            reference_time_us: capture_us,
            new_playout_delay_ms: 0,
            data: Bytes::from(self.params.crypto.crypt(frame_id, data)),
        };
        if let Some(prev) = self.frames.get_mut(&self.last_enqueued) {
            let step = capture_us.saturating_sub(prev.frame.reference_time_us);
            prev.media_duration = Duration::from_micros(step).min(MAX_UNPAUSED_FRAME_DURATION);
        }
        let count = usize::from(packet_count);
        self.frames.insert(
            frame_id,
            PendingFrame {
                frame,
                send_flags: vec![true; count],
                sent_at: vec![None; count],
                media_duration: Duration::ZERO,
            },
        );
        self.last_enqueued = frame_id;
        if is_key_frame {
            self.last_key_frame = frame_id;
        }
        // Keep the lip-sync reference time monotonic.
        let reference_us = self
            .pending_report
            .map_or(capture_us, |(prev, _)| prev.max(capture_us));
        self.pending_report = Some((reference_us, rtp_timestamp));
        // Until a receiver report proves the receiver has a Sender Report,
        // send one with every frame: the receiver cannot schedule playout
        // without it.
        if self.round_trip_time.is_zero() || self.next_report_at.is_none() {
            self.next_report_at = Some(now);
        }
        Ok(frame_id)
    }

    /// When this sender next wants to send RTP, if at all.
    #[must_use]
    pub(crate) fn rtp_resume_time(&self, now: Instant) -> Option<Instant> {
        if self.next_flagged().is_some() {
            return Some(now);
        }
        self.kickstart().map(|(_, _, when)| when)
    }

    /// When the next Sender Report is due, if any frame was ever enqueued.
    #[must_use]
    pub(crate) fn rtcp_due_at(&self) -> Option<Instant> {
        self.pending_report.and(self.next_report_at)
    }

    fn next_flagged(&self) -> Option<(FrameId, u16)> {
        self.frames
            .range(self.checkpoint.next()..)
            .find_map(|(&id, p)| p.first_flagged().map(|pid| (id, pid)))
    }

    /// The packet that tells the receiver about frames it may not know of,
    /// and when to send it.
    fn kickstart(&self) -> Option<(FrameId, u16, Instant)> {
        if self.latest_expected >= self.last_enqueued {
            return None;
        }
        let pending = self.frames.get(&self.last_enqueued)?;
        let last_pid = pending.sent_at.len().checked_sub(1)?;
        let sent = pending.sent_at.get(last_pid).copied().flatten()?;
        let interval = (self.params.target_playout_delay / 20).max(self.round_trip_time * 2);
        Some((
            self.last_enqueued,
            u16::try_from(last_pid).ok()?,
            sent + interval,
        ))
    }

    /// Writes the next RTP packet due at `now` into `out`. Returns `false`
    /// when nothing needs sending.
    pub(crate) fn next_rtp_packet(&mut self, now: Instant, out: &mut Vec<u8>) -> bool {
        let chosen = match self.next_flagged() {
            Some(c) => Some(c),
            None => self
                .kickstart()
                .filter(|&(_, _, when)| when <= now)
                .map(|(id, pid, _)| (id, pid)),
        };
        let Some((frame_id, packet_id)) = chosen else {
            return false;
        };
        let Some(pending) = self.frames.get_mut(&frame_id) else {
            return false;
        };
        if !self
            .packetizer
            .generate_packet(&pending.frame, packet_id, out)
        {
            return false;
        }
        let idx = usize::from(packet_id);
        let was_sent = pending.sent_at.get(idx).copied().flatten().is_some();
        if let Some(flag) = pending.send_flags.get_mut(idx) {
            *flag = false;
        }
        if let Some(at) = pending.sent_at.get_mut(idx) {
            *at = Some(now);
        }
        self.counters.packets_sent += 1;
        if was_sent {
            self.counters.packets_retransmitted += 1;
        }
        self.packet_count = self.packet_count.wrapping_add(1);
        let octets = u32::try_from(out.len().saturating_sub(BASE_RTP_HEADER_SIZE)).unwrap_or(0);
        self.octet_count = self.octet_count.wrapping_add(octets);
        true
    }

    /// Appends a Sender Report (plus DLRR when the receiver sent a reference
    /// time) to `out` if one is due. The report maps "now" — not the last
    /// frame's capture time — onto the RTP timeline.
    pub(crate) fn sender_report(
        &mut self,
        now: Instant,
        now_unix_us: u64,
        out: &mut Vec<u8>,
    ) -> bool {
        let Some((reference_us, rtp)) = self.pending_report else {
            return false;
        };
        if self.next_report_at.is_none_or(|due| due > now) {
            return false;
        }
        let rtp_now = if now_unix_us >= reference_us {
            rtp.wrapping_add(self.rtp_timestamp_for(now_unix_us - reference_us))
        } else {
            rtp.wrapping_sub(self.rtp_timestamp_for(reference_us - now_unix_us))
        };
        let ntp = unix_us_to_ntp(now_unix_us);
        let report = SenderReport {
            sender_ssrc: self.params.sender_ssrc,
            ntp,
            rtp_timestamp: rtp_now,
            packet_count: self.packet_count,
            octet_count: self.octet_count,
        };
        let dlrr = self.last_receiver_reference.map(|(last_rr, at)| DlrrReply {
            receiver_ssrc: self.params.receiver_ssrc,
            last_rr,
            delay_since_last_rr: to_q16(now.saturating_duration_since(at)),
        });
        build_sender_report(&report, dlrr.as_ref(), out);
        if self.recent_reports.len() == RECENT_SENDER_REPORTS {
            self.recent_reports.pop_front();
        }
        self.recent_reports.push_back((ntp_middle_32(ntp), now));
        self.first_report_at.get_or_insert(now);
        self.next_report_at = Some(now + RTCP_REPORT_INTERVAL);
        true
    }

    /// Applies one parsed compound RTCP packet from the receiver.
    pub(crate) fn on_rtcp(&mut self, rtcp: &CompoundRtcp, arrival: Instant) -> RtcpOutcome {
        let mut outcome = RtcpOutcome::default();
        if let Some(ntp) = rtcp.receiver_reference_ntp {
            // Drop stale, reordered packets: RTCP always carries full state.
            if self.latest_receiver_ntp.is_some_and(|latest| ntp < latest) {
                return outcome;
            }
            self.latest_receiver_ntp = Some(ntp);
            self.last_receiver_reference = Some((ntp_middle_32(ntp), arrival));
        }
        self.last_rtcp_at = Some(arrival);

        if let Some(block) = rtcp.report_block {
            self.counters.fraction_lost = block.fraction_lost;
            self.on_report_block(
                block.last_sender_report_id,
                block.delay_since_last_report,
                arrival,
            );
        }
        if let Some(feedback) = &rtcp.feedback {
            outcome.checkpoint_advanced = self.on_feedback(feedback);
            outcome.packets_to_resend = self.on_nacks(feedback, arrival);
        }
        if rtcp.picture_loss {
            outcome.key_frame_needed = self.on_picture_loss();
        }
        outcome
    }

    fn on_report_block(&mut self, report_id: u32, delay_q16: u32, arrival: Instant) {
        if report_id == 0 {
            return;
        }
        let Some(&(_, sent_at)) = self
            .recent_reports
            .iter()
            .rev()
            .find(|(id, _)| *id == report_id)
        else {
            return;
        };
        let total = arrival.saturating_duration_since(sent_at);
        let non_network = from_q16(delay_q16);
        let measurement = total
            .checked_sub(non_network)
            .unwrap_or(Duration::ZERO)
            .max(NEAR_ZERO_RTT)
            .min(self.params.target_playout_delay);
        self.round_trip_time = smooth_rtt(self.round_trip_time, measurement);
    }

    fn on_feedback(&mut self, feedback: &ReceiverFeedback) -> bool {
        let mut advanced = false;
        if feedback.checkpoint <= self.last_enqueued {
            while self.checkpoint < feedback.checkpoint {
                self.checkpoint = self.checkpoint.next();
                if self.frames.remove(&self.checkpoint).is_some() {
                    self.counters.frames_acked += 1;
                }
                advanced = true;
            }
            self.latest_expected = self.latest_expected.max(feedback.checkpoint);
        }
        if let Some(&newest) = feedback.acks.last()
            && newest <= self.last_enqueued
        {
            for id in &feedback.acks {
                if self.frames.remove(id).is_some() {
                    self.counters.frames_acked += 1;
                }
            }
            self.latest_expected = self.latest_expected.max(newest);
        }
        advanced
    }

    fn on_nacks(&mut self, feedback: &ReceiverFeedback, arrival: Instant) -> usize {
        // A NACK may have been issued while the packet was still on its way;
        // only resend packets sent at least one round trip ago.
        let too_recent = arrival.checked_sub(self.round_trip_time);
        let mut resend = 0;
        for nack in &feedback.nacks {
            if nack.frame_id > self.last_enqueued {
                continue;
            }
            let Some(pending) = self.frames.get_mut(&nack.frame_id) else {
                continue;
            };
            self.latest_expected = self.latest_expected.max(nack.frame_id);
            let range = if nack.packet_id == ALL_PACKETS_LOST {
                0..pending.sent_at.len()
            } else {
                let pid = usize::from(nack.packet_id);
                if pid >= pending.sent_at.len() {
                    continue;
                }
                pid..pid + 1
            };
            for pid in range {
                let eligible = match (pending.sent_at.get(pid).copied().flatten(), too_recent) {
                    (Some(sent), Some(limit)) => sent <= limit,
                    // Never sent: it's already flagged.
                    (Some(_), None) | (None, _) => false,
                };
                if eligible && let Some(flag) = pending.send_flags.get_mut(pid) {
                    *flag = true;
                    resend += 1;
                }
            }
        }
        resend
    }

    fn on_picture_loss(&mut self) -> bool {
        // The receiver repeats PLI until it gets a key frame; don't force
        // another one while a key frame is still in flight.
        self.checkpoint >= self.last_key_frame
    }
}

/// Round trip time smoothing: fast attack, slow decay.
fn smooth_rtt(estimate: Duration, measurement: Duration) -> Duration {
    if estimate.is_zero() {
        measurement
    } else if measurement > estimate {
        (estimate + measurement) / 2
    } else {
        (estimate * 7 + measurement) / 8
    }
}

/// Duration in 1/65536 s units, saturating.
fn to_q16(d: Duration) -> u32 {
    let units = d.as_micros() * 65_536 / 1_000_000;
    u32::try_from(units).unwrap_or(u32::MAX)
}

fn from_q16(units: u32) -> Duration {
    Duration::from_micros(u64::from(units) * 1_000_000 / 65_536)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mirroring::rtcp::{PacketNack, ReportBlock};
    use crate::mirroring::rtp::MAX_PAYLOAD_SIZE;

    fn params(kind: StreamKind) -> StreamParams {
        StreamParams {
            kind,
            payload_type: RtpPayloadType::H264,
            sender_ssrc: 60_000,
            receiver_ssrc: 1,
            rtp_timebase: 90_000,
            target_playout_delay: Duration::from_millis(400),
            crypto: FrameCrypto::new([1; 16], [2; 16]),
            first_sequence: 0,
        }
    }

    fn feedback(checkpoint: i64, nacks: Vec<PacketNack>, acks: Vec<FrameId>) -> CompoundRtcp {
        CompoundRtcp {
            feedback: Some(ReceiverFeedback {
                checkpoint: FrameId::new(checkpoint),
                playout_delay_ms: 400,
                nacks,
                acks,
            }),
            ..CompoundRtcp::default()
        }
    }

    fn drain(s: &mut StreamSender, now: Instant) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let mut buf = Vec::new();
        while s.next_rtp_packet(now, &mut buf) {
            out.push(buf.clone());
        }
        out
    }

    const T0_US: u64 = 1_700_000_000_000_000;

    #[test]
    fn sends_all_packets_once_then_resends_nacked_after_rtt() {
        let t0 = Instant::now();
        let mut s = StreamSender::new(params(StreamKind::Video));
        let data = vec![0u8; MAX_PAYLOAD_SIZE * 3];
        assert_eq!(s.enqueue(&data, true, T0_US, t0), Ok(FrameId::FIRST));
        let pkts = drain(&mut s, t0);
        assert_eq!(pkts.len(), 3);
        assert_eq!(s.counters().packets_sent, 3);

        // RTT unknown (zero): any packet sent before the NACK arrived is resent.
        let nack = vec![PacketNack {
            frame_id: FrameId::FIRST,
            packet_id: 1,
        }];
        let out = s.on_rtcp(
            &feedback(-1, nack.clone(), vec![]),
            t0 + Duration::from_millis(5),
        );
        assert_eq!(out.packets_to_resend, 1);
        let resent = drain(&mut s, t0 + Duration::from_millis(5));
        assert_eq!(resent.len(), 1);
        assert_eq!(u16::from_be_bytes([resent[0][14], resent[0][15]]), 1);
        assert_eq!(s.counters().packets_retransmitted, 1);

        // Whole-frame NACK.
        let all = vec![PacketNack {
            frame_id: FrameId::FIRST,
            packet_id: ALL_PACKETS_LOST,
        }];
        let out = s.on_rtcp(&feedback(-1, all, vec![]), t0 + Duration::from_millis(10));
        assert_eq!(out.packets_to_resend, 3);

        // Checkpoint ACK releases the frame and its pending resends.
        let out = s.on_rtcp(&feedback(0, vec![], vec![]), t0 + Duration::from_millis(11));
        assert!(out.checkpoint_advanced);
        assert_eq!(s.checkpoint(), FrameId::FIRST);
        assert!(drain(&mut s, t0 + Duration::from_millis(11)).is_empty());
    }

    #[test]
    fn nack_within_rtt_is_not_resent() {
        let t0 = Instant::now();
        let mut s = StreamSender::new(params(StreamKind::Video));
        s.round_trip_time = Duration::from_millis(20);
        s.enqueue(&[1, 2, 3], true, T0_US, t0).unwrap();
        drain(&mut s, t0);
        let nack = vec![PacketNack {
            frame_id: FrameId::FIRST,
            packet_id: 0,
        }];
        let out = s.on_rtcp(
            &feedback(-1, nack.clone(), vec![]),
            t0 + Duration::from_millis(10),
        );
        assert_eq!(out.packets_to_resend, 0);
        let out = s.on_rtcp(&feedback(-1, nack, vec![]), t0 + Duration::from_millis(30));
        assert_eq!(out.packets_to_resend, 1);
    }

    #[test]
    fn cst2_acks_release_frames_beyond_checkpoint() {
        let t0 = Instant::now();
        let mut s = StreamSender::new(params(StreamKind::Video));
        for i in 0..4u64 {
            s.enqueue(&[0; 10], i == 0, T0_US + i * 16_000, t0).unwrap();
        }
        drain(&mut s, t0);
        s.on_rtcp(
            &feedback(0, vec![], vec![FrameId::new(2), FrameId::new(3)]),
            t0,
        );
        assert_eq!(
            s.frames.keys().copied().collect::<Vec<_>>(),
            vec![FrameId::new(1)]
        );
        assert_eq!(s.counters().frames_acked, 3);
        // An ACK beyond what was sent is ignored.
        s.on_rtcp(&feedback(0, vec![], vec![FrameId::new(9)]), t0);
        assert_eq!(s.frames.len(), 1);
        // A checkpoint beyond what was sent is ignored too.
        let out = s.on_rtcp(&feedback(9, vec![], vec![]), t0);
        assert!(!out.checkpoint_advanced);
    }

    #[test]
    fn span_limit_rejects_and_ack_reopens() {
        let t0 = Instant::now();
        let mut p = params(StreamKind::Video);
        p.target_playout_delay = Duration::from_mins(1);
        let mut s = StreamSender::new(p);
        for i in 0..MAX_UNACKED_FRAMES {
            let id = s
                .enqueue(
                    &[0; 4],
                    i == 0,
                    T0_US + u64::try_from(i).unwrap() * 1_000,
                    t0,
                )
                .unwrap();
            assert_eq!(id, FrameId::new(i));
        }
        assert_eq!(
            s.enqueue(&[0; 4], false, T0_US + 200_000, t0),
            Err(EnqueueRejected::IdSpanLimit)
        );
        s.on_rtcp(&feedback(10, vec![], vec![]), t0);
        assert!(s.enqueue(&[0; 4], false, T0_US + 201_000, t0).is_ok());
    }

    #[test]
    fn in_flight_duration_limit() {
        let t0 = Instant::now();
        let mut s = StreamSender::new(params(StreamKind::Video));
        // 30 fps frames, each step 33 ms: 400 ms allows ~12 in flight.
        let mut accepted = 0;
        for i in 0..20u64 {
            if s.enqueue(&[0; 4], i == 0, T0_US + i * 33_333, t0).is_ok() {
                accepted += 1;
            } else {
                break;
            }
        }
        assert_eq!(accepted, 13);
        // A 2 s capture pause counts as at most 50 ms.
        let mut s = StreamSender::new(params(StreamKind::Video));
        s.enqueue(&[0; 4], true, T0_US, t0).unwrap();
        assert!(s.in_flight_media_duration(T0_US + 2_000_000) <= MAX_UNPAUSED_FRAME_DURATION);
    }

    #[test]
    fn kickstart_resends_last_packet_when_receiver_unaware() {
        let t0 = Instant::now();
        let mut s = StreamSender::new(params(StreamKind::Video));
        s.enqueue(&vec![0u8; MAX_PAYLOAD_SIZE + 5], true, T0_US, t0)
            .unwrap();
        drain(&mut s, t0);
        // target 400 ms / 20 = 20 ms
        assert_eq!(s.rtp_resume_time(t0), Some(t0 + Duration::from_millis(20)));
        assert!(drain(&mut s, t0 + Duration::from_millis(19)).is_empty());
        let k = drain(&mut s, t0 + Duration::from_millis(20));
        assert_eq!(k.len(), 1);
        assert_eq!(u16::from_be_bytes([k[0][14], k[0][15]]), 1, "last packet");
        // Once the receiver NACKs/acks the frame it knows about it.
        s.on_rtcp(&feedback(0, vec![], vec![]), t0 + Duration::from_millis(21));
        assert_eq!(s.rtp_resume_time(t0 + Duration::from_millis(30)), None);
    }

    #[test]
    fn sender_report_maps_now_and_rtt_from_report_block() {
        let t0 = Instant::now();
        let mut s = StreamSender::new(params(StreamKind::Video));
        let mut out = Vec::new();
        assert!(!s.sender_report(t0, T0_US, &mut out), "no frame yet");
        s.enqueue(&[0; 4], true, T0_US, t0).unwrap();
        let rtp0 = s.rtp_timestamp_for(T0_US);
        // 100 ms after capture: RTP must be advanced by 9000 ticks.
        let send_us = T0_US + 100_000;
        assert!(s.sender_report(t0, send_us, &mut out));
        assert_eq!(out.len(), 28);
        assert_eq!(&out[4..8], &60_000u32.to_be_bytes());
        let ntp = u64::from_be_bytes(out[8..16].try_into().unwrap());
        assert_eq!(ntp, unix_us_to_ntp(send_us));
        let rtp_ts = u32::from_be_bytes(out[16..20].try_into().unwrap());
        assert_eq!(rtp_ts, rtp0.wrapping_add(9_000));

        // Not due again until RTT is known (every frame) or interval elapses.
        out.clear();
        assert!(!s.sender_report(t0, send_us, &mut out));

        // Receiver reports 10 ms after, with 4 ms processing delay.
        let rtcp = CompoundRtcp {
            receiver_reference_ntp: Some(unix_us_to_ntp(send_us + 5_000)),
            report_block: Some(ReportBlock {
                fraction_lost: 3,
                cumulative_lost: 0,
                extended_high_sequence: 0,
                jitter: 0,
                last_sender_report_id: ntp_middle_32(ntp),
                delay_since_last_report: to_q16(Duration::from_millis(4)),
            }),
            ..CompoundRtcp::default()
        };
        s.on_rtcp(&rtcp, t0 + Duration::from_millis(10));
        let rtt = s.round_trip_time();
        assert!(
            rtt >= Duration::from_micros(5_900) && rtt <= Duration::from_micros(6_100),
            "{rtt:?}"
        );
        assert_eq!(s.counters().fraction_lost, 3);

        // Next report carries a DLRR echoing the receiver reference time.
        out.clear();
        assert!(s.sender_report(t0 + RTCP_REPORT_INTERVAL, send_us + 500_000, &mut out));
        assert_eq!(out.len(), 52);
        assert_eq!(out[29], 207);
        assert_eq!(
            &out[44..48],
            &ntp_middle_32(unix_us_to_ntp(send_us + 5_000)).to_be_bytes()
        );
    }

    #[test]
    fn stale_rtcp_is_ignored() {
        let t0 = Instant::now();
        let mut s = StreamSender::new(params(StreamKind::Video));
        s.enqueue(&[0; 4], true, T0_US, t0).unwrap();
        let mut newer = feedback(0, vec![], vec![]);
        newer.receiver_reference_ntp = Some(1_000);
        let mut older = feedback(0, vec![], vec![]);
        older.receiver_reference_ntp = Some(999);
        older.picture_loss = true;
        s.on_rtcp(&newer, t0);
        assert_eq!(s.on_rtcp(&older, t0), RtcpOutcome::default());
    }

    #[test]
    fn picture_loss_ignored_while_key_frame_in_flight() {
        let t0 = Instant::now();
        let mut s = StreamSender::new(params(StreamKind::Video));
        s.enqueue(&[0; 4], true, T0_US, t0).unwrap();
        let pli = CompoundRtcp {
            picture_loss: true,
            ..CompoundRtcp::default()
        };
        assert!(!s.on_rtcp(&pli, t0).key_frame_needed);
        s.on_rtcp(&feedback(0, vec![], vec![]), t0);
        assert!(s.on_rtcp(&pli, t0).key_frame_needed);
    }

    #[test]
    fn delta_frames_reference_previous_frame() {
        let t0 = Instant::now();
        let mut s = StreamSender::new(params(StreamKind::Video));
        s.enqueue(&[0; 4], true, T0_US, t0).unwrap();
        s.enqueue(&[0; 4], false, T0_US + 16_000, t0).unwrap();
        let pkts = drain(&mut s, t0);
        assert_eq!((pkts[0][12], pkts[0][13], pkts[0][18]), (0xC0, 0, 0));
        assert_eq!((pkts[1][12], pkts[1][13], pkts[1][18]), (0x40, 1, 0));
    }

    #[test]
    fn q16_roundtrip() {
        assert_eq!(to_q16(Duration::from_secs(1)), 65_536);
        assert_eq!(from_q16(32_768), Duration::from_millis(500));
    }
}
