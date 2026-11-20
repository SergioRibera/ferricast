//! Cast Streaming RTCP: compound receiver-packet parsing and Sender Report
//! building (openscreen `compound_rtcp_parser.cc`, `rtcp_common.cc`,
//! `sender_report_builder.cc`).

use std::fmt;

use super::rtp::{ALL_PACKETS_LOST, FrameId};

const RTCP_COMMON_HEADER_SIZE: usize = 4;
const RTCP_VERSION_BITS: u8 = 0b100;
const PT_SENDER_REPORT: u8 = 200;
const PT_RECEIVER_REPORT: u8 = 201;
const PT_SOURCE_DESCRIPTION: u8 = 202;
const PT_APPLICATION_DEFINED: u8 = 204;
const PT_PAYLOAD_SPECIFIC: u8 = 206;
const PT_EXTENDED_REPORTS: u8 = 207;
const SUBTYPE_PICTURE_LOSS: u8 = 1;
const SUBTYPE_FEEDBACK: u8 = 15;

const REPORT_BLOCK_SIZE: usize = 24;
const FEEDBACK_HEADER_SIZE: usize = 16;
const FEEDBACK_LOSS_FIELD_SIZE: usize = 4;
const FEEDBACK_ACK_HEADER_SIZE: usize = 6;
const CAST_IDENTIFIER: u32 = u32::from_be_bytes(*b"CAST");
const CST2_IDENTIFIER: u32 = u32::from_be_bytes(*b"CST2");
const XR_BLOCK_RECEIVER_REFERENCE_TIME: u8 = 4;
const XR_BLOCK_DLRR: u8 = 5;
const PICTURE_LOSS_SIZE: usize = 8;

/// Seconds between the NTP epoch (1900) and the UNIX epoch (1970).
const NTP_UNIX_OFFSET_S: u64 = 2_208_988_800;

/// Converts UNIX-epoch microseconds to a 64-bit NTP timestamp.
#[must_use]
pub(crate) fn unix_us_to_ntp(unix_us: u64) -> u64 {
    let secs = unix_us / 1_000_000 + NTP_UNIX_OFFSET_S;
    let frac = ((unix_us % 1_000_000) << 32) / 1_000_000;
    (secs << 32) | frac
}

/// Middle 32 bits of an NTP timestamp — the "LSR"/"LRR" id echoed back by
/// the peer.
#[must_use]
pub(crate) const fn ntp_middle_32(ntp: u64) -> u32 {
    ((ntp >> 16) & 0xFFFF_FFFF) as u32
}

/// RTCP report block (RFC 3550 §6.4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReportBlock {
    pub(crate) fraction_lost: u8,
    pub(crate) cumulative_lost: u32,
    pub(crate) extended_high_sequence: u32,
    pub(crate) jitter: u32,
    /// Middle 32 bits of the NTP time of the last Sender Report received.
    pub(crate) last_sender_report_id: u32,
    /// Delay since that Sender Report, in 1/65536 s.
    pub(crate) delay_since_last_report: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PacketNack {
    pub(crate) frame_id: FrameId,
    /// [`ALL_PACKETS_LOST`] means every packet of the frame.
    pub(crate) packet_id: u16,
}

/// Cast Feedback (PSFB FMT 15) contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReceiverFeedback {
    /// Every frame up to and including this one has been received (or
    /// abandoned) by the receiver.
    pub(crate) checkpoint: FrameId,
    pub(crate) playout_delay_ms: u16,
    /// Sorted, de-duplicated, with whole-frame NACKs superseding per-packet
    /// ones.
    pub(crate) nacks: Vec<PacketNack>,
    /// Frames after the checkpoint that are complete (CST2 ACK vector).
    pub(crate) acks: Vec<FrameId>,
}

/// Everything in one compound RTCP packet that concerns one sender stream.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct CompoundRtcp {
    pub(crate) receiver_reference_ntp: Option<u64>,
    pub(crate) report_block: Option<ReportBlock>,
    pub(crate) feedback: Option<ReceiverFeedback>,
    pub(crate) picture_loss: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RtcpParseError {
    InvalidHeader,
    Truncated,
    MissingCastIdentifier,
    InvalidReferenceTimeBlock,
}

impl fmt::Display for RtcpParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidHeader => "invalid RTCP common header",
            Self::Truncated => "truncated RTCP packet",
            Self::MissingCastIdentifier => "Cast feedback without 'CAST' identifier",
            Self::InvalidReferenceTimeBlock => "malformed XR receiver reference time block",
        })
    }
}

/// Byte cursor that never panics: every read reports truncation.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], RtcpParseError> {
        let (head, tail) = self
            .0
            .split_at_checked(n)
            .ok_or(RtcpParseError::Truncated)?;
        self.0 = tail;
        Ok(head)
    }

    fn u8(&mut self) -> Result<u8, RtcpParseError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, RtcpParseError> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, RtcpParseError> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, RtcpParseError> {
        Ok((u64::from(self.u32()?) << 32) | u64::from(self.u32()?))
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

struct CommonHeader {
    packet_type: u8,
    /// Report count (SR/RR) or subtype (APP/PSFB).
    count_or_subtype: u8,
    payload_size: usize,
}

fn parse_common_header(r: &mut Reader<'_>) -> Result<CommonHeader, RtcpParseError> {
    let byte0 = r.u8()?;
    if byte0 >> 5 != RTCP_VERSION_BITS {
        return Err(RtcpParseError::InvalidHeader);
    }
    let packet_type = r.u8()?;
    if !matches!(
        packet_type,
        PT_SENDER_REPORT
            | PT_RECEIVER_REPORT
            | PT_SOURCE_DESCRIPTION
            | PT_APPLICATION_DEFINED
            | PT_PAYLOAD_SPECIFIC
            | PT_EXTENDED_REPORTS
    ) {
        return Err(RtcpParseError::InvalidHeader);
    }
    let words = r.u16()?;
    Ok(CommonHeader {
        packet_type,
        count_or_subtype: byte0 & 0x1F,
        payload_size: usize::from(words) * 4,
    })
}

/// SSRC in the first sub-packet of a compound RTCP packet. Receivers lead
/// with their Receiver Report, so this identifies the stream the packet is
/// about.
#[must_use]
pub(crate) fn first_ssrc(packet: &[u8]) -> Option<u32> {
    let b = packet.get(RTCP_COMMON_HEADER_SIZE..RTCP_COMMON_HEADER_SIZE + 4)?;
    if packet.first()? >> 6 != 2 {
        return None;
    }
    Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

/// Parses a compound RTCP packet sent by the receiver of one stream.
///
/// `receiver_ssrc`/`sender_ssrc` select the sub-packets addressed to this
/// stream; `max_feedback_frame_id` (the last frame this sender enqueued)
/// anchors the expansion of the 8-bit checkpoint id.
///
/// # Errors
/// Returns an error when any sub-packet is malformed; nothing from a
/// malformed packet should be acted upon.
pub(crate) fn parse_compound(
    packet: &[u8],
    receiver_ssrc: u32,
    sender_ssrc: u32,
    max_feedback_frame_id: FrameId,
) -> Result<CompoundRtcp, RtcpParseError> {
    let mut out = CompoundRtcp::default();
    let mut r = Reader(packet);
    while !r.is_empty() {
        let header = parse_common_header(&mut r)?;
        let mut payload = Reader(r.take(header.payload_size)?);
        match header.packet_type {
            PT_RECEIVER_REPORT => {
                let reporter = payload.u32()?;
                if reporter == receiver_ssrc {
                    out.report_block =
                        parse_report_block(&mut payload, header.count_or_subtype, sender_ssrc)?;
                }
            }
            PT_APPLICATION_DEFINED => {
                // Receiver frame/packet event logs; validated for size only.
                payload.take(8)?;
            }
            PT_PAYLOAD_SPECIFIC => match header.count_or_subtype {
                SUBTYPE_PICTURE_LOSS => {
                    if payload.len() < PICTURE_LOSS_SIZE {
                        return Err(RtcpParseError::Truncated);
                    }
                    if payload.u32()? == receiver_ssrc && payload.u32()? == sender_ssrc {
                        out.picture_loss = true;
                    }
                }
                SUBTYPE_FEEDBACK => parse_feedback(
                    &mut payload,
                    receiver_ssrc,
                    sender_ssrc,
                    max_feedback_frame_id,
                    &mut out.feedback,
                )?,
                _ => {}
            },
            PT_EXTENDED_REPORTS => {
                if let Some(ntp) = parse_extended_reports(&mut payload, receiver_ssrc)? {
                    out.receiver_reference_ntp = Some(ntp);
                }
            }
            _ => {}
        }
    }
    if let Some(feedback) = out.feedback.as_mut() {
        canonicalize_nacks(&mut feedback.nacks);
    }
    Ok(out)
}

fn parse_report_block(
    r: &mut Reader<'_>,
    count: u8,
    sender_ssrc: u32,
) -> Result<Option<ReportBlock>, RtcpParseError> {
    if r.len() < REPORT_BLOCK_SIZE * usize::from(count) {
        return Err(RtcpParseError::Truncated);
    }
    let mut found = None;
    for _ in 0..count {
        if r.u32()? != sender_ssrc {
            r.take(REPORT_BLOCK_SIZE - 4)?;
            continue;
        }
        let second = r.u32()?;
        found = Some(ReportBlock {
            fraction_lost: (second >> 24) as u8,
            cumulative_lost: second & 0x00FF_FFFF,
            extended_high_sequence: r.u32()?,
            jitter: r.u32()?,
            last_sender_report_id: r.u32()?,
            delay_since_last_report: r.u32()?,
        });
    }
    Ok(found)
}

fn parse_feedback(
    r: &mut Reader<'_>,
    receiver_ssrc: u32,
    sender_ssrc: u32,
    max_feedback_frame_id: FrameId,
    slot: &mut Option<ReceiverFeedback>,
) -> Result<(), RtcpParseError> {
    if r.len() < FEEDBACK_HEADER_SIZE {
        return Err(RtcpParseError::Truncated);
    }
    if r.u32()? != receiver_ssrc || r.u32()? != sender_ssrc {
        return Ok(());
    }
    if r.u32()? != CAST_IDENTIFIER {
        return Err(RtcpParseError::MissingCastIdentifier);
    }
    let checkpoint = max_feedback_frame_id.expand_less_than_or_equal(r.u8()?);
    let loss_fields = r.u8()?;
    let playout_delay_ms = r.u16()?;
    // Never let a later feedback in the same compound packet move the
    // checkpoint backwards.
    if slot.as_ref().is_some_and(|f| f.checkpoint > checkpoint) {
        return Ok(());
    }
    let mut feedback = ReceiverFeedback {
        checkpoint,
        playout_delay_ms,
        nacks: Vec::new(),
        acks: Vec::new(),
    };
    if r.len() < FEEDBACK_LOSS_FIELD_SIZE * usize::from(loss_fields) {
        return Err(RtcpParseError::Truncated);
    }
    for _ in 0..loss_fields {
        let frame_id = checkpoint.expand_greater_than(r.u8()?);
        let mut packet_id = r.u16()?;
        let mut bits = r.u8()?;
        feedback.nacks.push(PacketNack {
            frame_id,
            packet_id,
        });
        if packet_id != ALL_PACKETS_LOST {
            while bits != 0 {
                packet_id = packet_id.wrapping_add(1);
                if bits & 1 != 0 {
                    feedback.nacks.push(PacketNack {
                        frame_id,
                        packet_id,
                    });
                }
                bits >>= 1;
            }
        }
    }
    // Optional CST2 frame-level ACK vector. Anything else trailing is ignored
    // for compatibility with older receivers.
    if r.len() >= FEEDBACK_ACK_HEADER_SIZE && r.u32()? == CST2_IDENTIFIER {
        let _feedback_count = r.u8()?;
        let octets = r.u8()?;
        let vector = r.take(usize::from(octets))?;
        let mut base = checkpoint.offset(2);
        for &octet in vector {
            let mut bits = octet;
            let mut frame_id = base;
            while bits != 0 {
                if bits & 1 != 0 {
                    feedback.acks.push(frame_id);
                }
                frame_id = frame_id.next();
                bits >>= 1;
            }
            base = base.offset(8);
        }
    }
    *slot = Some(feedback);
    Ok(())
}

fn parse_extended_reports(
    r: &mut Reader<'_>,
    receiver_ssrc: u32,
) -> Result<Option<u64>, RtcpParseError> {
    if r.u32()? != receiver_ssrc {
        return Ok(None);
    }
    let mut ntp = None;
    while !r.is_empty() {
        let block_type = r.u8()?;
        r.u8()?;
        let size = usize::from(r.u16()?) * 4;
        let mut block = Reader(r.take(size)?);
        if block_type == XR_BLOCK_RECEIVER_REFERENCE_TIME {
            if size != 8 {
                return Err(RtcpParseError::InvalidReferenceTimeBlock);
            }
            ntp = Some(block.u64()?);
        }
    }
    Ok(ntp)
}

/// Sorts NACKs, removes duplicates and drops per-packet NACKs of frames that
/// are NACKed whole.
fn canonicalize_nacks(nacks: &mut Vec<PacketNack>) {
    // Whole-frame NACKs sort first within their frame.
    nacks.sort_by_key(|n| (n.frame_id, n.packet_id.wrapping_add(1)));
    nacks.dedup_by(|later, kept| {
        later.frame_id == kept.frame_id
            && (kept.packet_id == ALL_PACKETS_LOST || later.packet_id == kept.packet_id)
    });
}

/// Contents of one RTCP Sender Report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SenderReport {
    pub(crate) sender_ssrc: u32,
    pub(crate) ntp: u64,
    pub(crate) rtp_timestamp: u32,
    pub(crate) packet_count: u32,
    pub(crate) octet_count: u32,
}

/// Reply to the receiver's XR Receiver Reference Time block (RFC 3611 §4.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DlrrReply {
    pub(crate) receiver_ssrc: u32,
    /// Middle 32 bits of the receiver's reference NTP time.
    pub(crate) last_rr: u32,
    /// Time since that reference time was received, 1/65536 s.
    pub(crate) delay_since_last_rr: u32,
}

/// Appends a Sender Report (no report blocks), followed by an XR DLRR block
/// when `dlrr` is given, to `out`.
pub(crate) fn build_sender_report(sr: &SenderReport, dlrr: Option<&DlrrReply>, out: &mut Vec<u8>) {
    out.extend_from_slice(&[0x80, PT_SENDER_REPORT, 0x00, 0x06]);
    out.extend_from_slice(&sr.sender_ssrc.to_be_bytes());
    out.extend_from_slice(&sr.ntp.to_be_bytes());
    out.extend_from_slice(&sr.rtp_timestamp.to_be_bytes());
    out.extend_from_slice(&sr.packet_count.to_be_bytes());
    out.extend_from_slice(&sr.octet_count.to_be_bytes());
    if let Some(d) = dlrr {
        out.extend_from_slice(&[0x80, PT_EXTENDED_REPORTS, 0x00, 0x05]);
        out.extend_from_slice(&sr.sender_ssrc.to_be_bytes());
        out.extend_from_slice(&[XR_BLOCK_DLRR, 0x00, 0x00, 0x03]);
        out.extend_from_slice(&d.receiver_ssrc.to_be_bytes());
        out.extend_from_slice(&d.last_rr.to_be_bytes());
        out.extend_from_slice(&d.delay_since_last_rr.to_be_bytes());
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    const RX: u32 = 0x0000_1111;
    const TX: u32 = 0x0002_2222;

    fn header(byte0: u8, pt: u8, payload: &[u8]) -> Vec<u8> {
        assert_eq!(payload.len() % 4, 0);
        let mut v = vec![byte0, pt];
        v.extend_from_slice(&u16::try_from(payload.len() / 4).unwrap().to_be_bytes());
        v.extend_from_slice(payload);
        v
    }

    pub(crate) fn rr(report: Option<(u32, u32)>) -> Vec<u8> {
        let mut p = RX.to_be_bytes().to_vec();
        let count = if let Some((lsr, dlsr)) = report {
            p.extend_from_slice(&TX.to_be_bytes());
            p.extend_from_slice(&[0x10, 0x00, 0x00, 0x05]); // 16/256 lost, 5 cumulative
            p.extend_from_slice(&1234u32.to_be_bytes());
            p.extend_from_slice(&77u32.to_be_bytes());
            p.extend_from_slice(&lsr.to_be_bytes());
            p.extend_from_slice(&dlsr.to_be_bytes());
            1
        } else {
            0
        };
        header(0x80 | count, 201, &p)
    }

    pub(crate) fn xr_rrt(ntp: u64) -> Vec<u8> {
        let mut p = RX.to_be_bytes().to_vec();
        p.extend_from_slice(&[4, 0, 0, 2]);
        p.extend_from_slice(&ntp.to_be_bytes());
        header(0x80, 207, &p)
    }

    pub(crate) fn pli() -> Vec<u8> {
        let mut p = RX.to_be_bytes().to_vec();
        p.extend_from_slice(&TX.to_be_bytes());
        header(0x81, 206, &p)
    }

    pub(crate) fn cast_feedback(
        checkpoint: u8,
        delay: u16,
        loss: &[(u8, u16, u8)],
        acks: Option<&[u8]>,
    ) -> Vec<u8> {
        let mut p = RX.to_be_bytes().to_vec();
        p.extend_from_slice(&TX.to_be_bytes());
        p.extend_from_slice(b"CAST");
        p.push(checkpoint);
        p.push(u8::try_from(loss.len()).unwrap());
        p.extend_from_slice(&delay.to_be_bytes());
        for &(fid, pid, bits) in loss {
            p.push(fid);
            p.extend_from_slice(&pid.to_be_bytes());
            p.push(bits);
        }
        if let Some(vector) = acks {
            p.extend_from_slice(b"CST2");
            p.push(9);
            p.push(u8::try_from(vector.len()).unwrap());
            p.extend_from_slice(vector);
            while !p.len().is_multiple_of(4) {
                p.push(0);
            }
        }
        header(0x80 | 15, 206, &p)
    }

    fn app_log() -> Vec<u8> {
        let mut p = RX.to_be_bytes().to_vec();
        p.extend_from_slice(b"CAST");
        p.extend_from_slice(&[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0x10, 0x00]);
        header(0x82, 204, &p)
    }

    #[test]
    fn parses_full_compound_receiver_packet() {
        let ntp = 0xE000_0001_8000_0000;
        let mut pkt = rr(Some((0xAABB_CCDD, 0x0001_0000)));
        pkt.extend(xr_rrt(ntp));
        pkt.extend(pli());
        // checkpoint 0x05; NACK frame 0x07 pid 3 + bits 0b101 -> pids 4, 6;
        // whole-frame NACK of 0x08; CST2: bit0 = checkpoint+2 = 0x107.
        pkt.extend(cast_feedback(
            0x05,
            400,
            &[(0x07, 3, 0b0000_0101), (0x08, ALL_PACKETS_LOST, 0xFF)],
            Some(&[0b0000_0011, 0b1000_0000]),
        ));
        pkt.extend(app_log());

        let parsed = parse_compound(&pkt, RX, TX, FrameId::new(0x109)).unwrap();
        assert_eq!(parsed.receiver_reference_ntp, Some(ntp));
        assert!(parsed.picture_loss);
        let rb = parsed.report_block.unwrap();
        assert_eq!(rb.fraction_lost, 0x10);
        assert_eq!(rb.cumulative_lost, 5);
        assert_eq!(rb.last_sender_report_id, 0xAABB_CCDD);
        assert_eq!(rb.delay_since_last_report, 0x0001_0000);
        let fb = parsed.feedback.unwrap();
        assert_eq!(fb.checkpoint, FrameId::new(0x105));
        assert_eq!(fb.playout_delay_ms, 400);
        let f107 = FrameId::new(0x107);
        let f108 = FrameId::new(0x108);
        assert_eq!(
            fb.nacks,
            vec![
                PacketNack {
                    frame_id: f107,
                    packet_id: 3
                },
                PacketNack {
                    frame_id: f107,
                    packet_id: 4
                },
                PacketNack {
                    frame_id: f107,
                    packet_id: 6
                },
                PacketNack {
                    frame_id: f108,
                    packet_id: ALL_PACKETS_LOST
                },
            ]
        );
        assert_eq!(
            fb.acks,
            vec![
                FrameId::new(0x107),
                FrameId::new(0x108),
                FrameId::new(0x116)
            ]
        );
    }

    #[test]
    fn leader_checkpoint_before_any_frame() {
        let pkt = [rr(None), xr_rrt(1), cast_feedback(0xFF, 400, &[], None)].concat();
        let parsed = parse_compound(&pkt, RX, TX, FrameId::LEADER).unwrap();
        assert_eq!(parsed.feedback.unwrap().checkpoint, FrameId::LEADER);
        assert!(parsed.report_block.is_none());
    }

    #[test]
    fn foreign_ssrcs_are_ignored() {
        let pkt = [rr(Some((1, 1))), pli()].concat();
        let parsed = parse_compound(&pkt, 0xDEAD, TX, FrameId::FIRST).unwrap();
        assert_eq!(parsed, CompoundRtcp::default());
    }

    #[test]
    fn malformed_packets_are_rejected() {
        let mut bad_version = rr(None);
        bad_version[0] = 0x40;
        assert_eq!(
            parse_compound(&bad_version, RX, TX, FrameId::FIRST),
            Err(RtcpParseError::InvalidHeader)
        );
        let mut short = cast_feedback(0, 400, &[(1, 0, 0)], None);
        short.truncate(short.len() - 4);
        assert_eq!(
            parse_compound(&short, RX, TX, FrameId::FIRST),
            Err(RtcpParseError::Truncated)
        );
        let mut bad_xr = rr(None);
        let mut p = RX.to_be_bytes().to_vec();
        p.extend_from_slice(&[4, 0, 0, 1, 0, 0, 0, 0]);
        bad_xr.extend(header(0x80, 207, &p));
        assert_eq!(
            parse_compound(&bad_xr, RX, TX, FrameId::FIRST),
            Err(RtcpParseError::InvalidReferenceTimeBlock)
        );
    }

    #[test]
    fn whole_frame_nack_supersedes_packets() {
        let pkt = cast_feedback(
            0x00,
            400,
            &[
                (0x02, 5, 0),
                (0x02, ALL_PACKETS_LOST, 0),
                (0x01, 1, 0),
                (0x01, 1, 0),
            ],
            None,
        );
        let fb = parse_compound(&pkt, RX, TX, FrameId::new(3))
            .unwrap()
            .feedback
            .unwrap();
        assert_eq!(
            fb.nacks,
            vec![
                PacketNack {
                    frame_id: FrameId::new(1),
                    packet_id: 1
                },
                PacketNack {
                    frame_id: FrameId::new(2),
                    packet_id: ALL_PACKETS_LOST
                },
            ]
        );
    }

    #[test]
    fn first_ssrc_routes_on_receiver_report() {
        assert_eq!(first_ssrc(&rr(None)), Some(RX));
        assert_eq!(first_ssrc(&[0x80, 201]), None);
    }

    #[test]
    fn sender_report_bytes() {
        let sr = SenderReport {
            sender_ssrc: 0x0102_0304,
            ntp: 0x1122_3344_5566_7788,
            rtp_timestamp: 0x99AA_BBCC,
            packet_count: 5,
            octet_count: 6,
        };
        let mut out = Vec::new();
        build_sender_report(&sr, None, &mut out);
        assert_eq!(
            out,
            [
                0x80, 200, 0, 6, 1, 2, 3, 4, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
                0xAA, 0xBB, 0xCC, 0, 0, 0, 5, 0, 0, 0, 6
            ]
        );
        out.clear();
        let dlrr = DlrrReply {
            receiver_ssrc: 9,
            last_rr: 0xA,
            delay_since_last_rr: 0xB,
        };
        build_sender_report(&sr, Some(&dlrr), &mut out);
        assert_eq!(out.len(), 28 + 24);
        assert_eq!(
            &out[28..],
            [
                0x80, 207, 0, 5, 1, 2, 3, 4, 5, 0, 0, 3, 0, 0, 0, 9, 0, 0, 0, 0xA, 0, 0, 0, 0xB
            ]
        );
    }

    #[test]
    fn ntp_conversion() {
        assert_eq!(unix_us_to_ntp(0), NTP_UNIX_OFFSET_S << 32);
        assert_eq!(
            unix_us_to_ntp(1_500_000),
            ((NTP_UNIX_OFFSET_S + 1) << 32) | 0x8000_0000
        );
        assert_eq!(ntp_middle_32(0x0192_a3b4_c5d6_e7f8), 0xa3b4_c5d6);
    }
}
