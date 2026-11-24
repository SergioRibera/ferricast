//! RTP MP2T packer/parser for Miracast.
//!
//! RFC 2250 payload type **33** (MP2T). Miracast mandates **exactly
//! 7 TS packets per RTP packet** → 7 × 188 = 1316 B payload + 12 B RTP
//! header = 1328 B UDP payload. Timestamp clock is 90 kHz, derived
//! from the first TS packet's PCR. Marker bit is always 0. SSRC is
//! random and stable for the whole session.
//!
//! No RTCP — Miracast disables it by default; don't advertise an RTCP
//! port and ignore stray RTCP from well-meaning stacks.

use bytes::{BufMut, Bytes, BytesMut};
use thiserror::Error;

use crate::ts::TS_PACKET_SIZE;

/// WFD-mandated TS packets per RTP packet.
pub const TS_PER_RTP: usize = 7;
/// RTP header size in bytes (no CSRC, no extension).
pub const RTP_HEADER_SIZE: usize = 12;
/// RTP payload size: 7 × 188 TS = 1316 B.
pub const RTP_PAYLOAD_SIZE: usize = TS_PER_RTP * TS_PACKET_SIZE;
/// Full RTP-over-UDP packet size.
pub const RTP_PACKET_SIZE: usize = RTP_HEADER_SIZE + RTP_PAYLOAD_SIZE;
/// RFC 2250 static payload type for MP2T.
pub const RTP_PAYLOAD_TYPE_MP2T: u8 = 33;
/// RTP timestamp clock rate: 90 kHz for MP2T.
pub const RTP_TIMESTAMP_HZ: u32 = 90_000;

#[derive(Debug, Error)]
pub enum RtpError {
    #[error("RTP packet too short ({got} B, need ≥ {RTP_HEADER_SIZE})")]
    Truncated { got: usize },
    #[error("unexpected RTP version {0}")]
    BadVersion(u8),
    #[error("expected payload type {RTP_PAYLOAD_TYPE_MP2T}, got {0}")]
    BadPayloadType(u8),
    #[error("MP2T payload size {0} is not a multiple of {TS_PACKET_SIZE}")]
    BadPayloadSize(usize),
}

/// Packs MPEG2-TS packets into RTP for Miracast.
#[derive(Debug)]
pub struct MpegTsRtpPacker {
    seq: u16,
    ssrc: u32,
    /// Pending TS packets that haven't yet filled a full RTP packet.
    pending: Vec<u8>,
}

impl MpegTsRtpPacker {
    /// Random initial sequence number and SSRC. Reusing the SSRC from a
    /// prior session is a known interop breaker; fresh per session.
    #[must_use]
    pub fn new() -> Self {
        Self::with_rng(&mut rand::thread_rng())
    }

    /// Deterministic variant for tests.
    #[must_use]
    pub fn with_rng<R: rand::Rng>(rng: &mut R) -> Self {
        Self {
            seq: rng.r#gen(),
            ssrc: rng.r#gen(),
            pending: Vec::with_capacity(RTP_PAYLOAD_SIZE),
        }
    }

    #[must_use]
    pub const fn ssrc(&self) -> u32 {
        self.ssrc
    }

    /// Feed one 188-byte TS packet into the packer. Returns any RTP
    /// packets this call completed. `timestamp_90khz` is the RTP
    /// timestamp for the *first* TS packet of the packet about to be
    /// emitted — the caller gets to derive it from the TS payload's
    /// PCR so it matches the mandated 90 kHz subset of the 27 MHz
    /// program clock.
    pub fn push_ts(&mut self, ts_packet: &[u8], timestamp_90khz: u32) -> Vec<Bytes> {
        assert!(
            ts_packet.len() == TS_PACKET_SIZE,
            "ts packet must be {TS_PACKET_SIZE} B, got {}",
            ts_packet.len()
        );
        self.pending.extend_from_slice(ts_packet);
        if self.pending.len() >= RTP_PAYLOAD_SIZE {
            vec![self.emit(timestamp_90khz)]
        } else {
            Vec::new()
        }
    }

    /// Flush any buffered TS packets even if we don't have a full 7.
    /// Used at end-of-stream only; mid-stream calls to this skip the
    /// WFD constraint that every RTP packet carry exactly 7 TS
    /// packets and some sinks will close the connection.
    pub fn flush(&mut self, timestamp_90khz: u32) -> Option<Bytes> {
        if self.pending.is_empty() {
            None
        } else {
            Some(self.emit(timestamp_90khz))
        }
    }

    fn emit(&mut self, timestamp_90khz: u32) -> Bytes {
        let payload_len = self.pending.len().min(RTP_PAYLOAD_SIZE);
        let mut buf = BytesMut::with_capacity(RTP_HEADER_SIZE + payload_len);

        // Byte 0: V=2, P=0, X=0, CC=0.
        buf.put_u8(0b1000_0000);
        // Byte 1: M=0, PT=33.
        buf.put_u8(RTP_PAYLOAD_TYPE_MP2T);
        buf.put_u16(self.seq);
        buf.put_u32(timestamp_90khz);
        buf.put_u32(self.ssrc);
        buf.put_slice(&self.pending[..payload_len]);

        // Rotate pending: drain the TS packets we just shipped.
        self.pending.drain(..payload_len);
        self.seq = self.seq.wrapping_add(1);
        buf.freeze()
    }
}

impl Default for MpegTsRtpPacker {
    fn default() -> Self {
        Self::new()
    }
}

/// Parsed RTP MP2T packet.
#[derive(Debug, Clone)]
pub struct RtpMp2tPacket<'a> {
    pub sequence: u16,
    pub timestamp_90khz: u32,
    pub ssrc: u32,
    pub payload: &'a [u8],
}

impl<'a> RtpMp2tPacket<'a> {
    /// Parse one RTP MP2T packet. Rejects anything that isn't a
    /// well-formed MP2T carrier (wrong version, wrong payload type,
    /// payload not a multiple of 188).
    pub fn parse(bytes: &'a [u8]) -> Result<Self, RtpError> {
        if bytes.len() < RTP_HEADER_SIZE {
            return Err(RtpError::Truncated { got: bytes.len() });
        }
        let b0 = bytes[0];
        let version = b0 >> 6;
        if version != 2 {
            return Err(RtpError::BadVersion(version));
        }
        let padding = b0 & 0b0010_0000 != 0;
        let extension = b0 & 0b0001_0000 != 0;
        let cc = (b0 & 0b0000_1111) as usize;

        let b1 = bytes[1];
        let payload_type = b1 & 0x7F;
        if payload_type != RTP_PAYLOAD_TYPE_MP2T {
            return Err(RtpError::BadPayloadType(payload_type));
        }

        let sequence = u16::from_be_bytes([bytes[2], bytes[3]]);
        let timestamp_90khz = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let ssrc = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);

        // CSRC + optional extension.
        let mut header_end = RTP_HEADER_SIZE + cc * 4;
        if extension {
            if bytes.len() < header_end + 4 {
                return Err(RtpError::Truncated { got: bytes.len() });
            }
            let ext_len_words =
                u16::from_be_bytes([bytes[header_end + 2], bytes[header_end + 3]]) as usize;
            header_end += 4 + ext_len_words * 4;
        }

        let mut payload_end = bytes.len();
        if padding {
            let pad = *bytes.last().ok_or(RtpError::Truncated { got: 0 })? as usize;
            payload_end = payload_end
                .checked_sub(pad)
                .ok_or(RtpError::Truncated { got: bytes.len() })?;
        }
        if payload_end < header_end {
            return Err(RtpError::Truncated { got: bytes.len() });
        }
        let payload = &bytes[header_end..payload_end];
        if !payload.len().is_multiple_of(TS_PACKET_SIZE) {
            return Err(RtpError::BadPayloadSize(payload.len()));
        }
        Ok(Self {
            sequence,
            timestamp_90khz,
            ssrc,
            payload,
        })
    }

    /// Iterator over the TS packets inside this RTP payload.
    pub fn ts_packets(&self) -> impl Iterator<Item = &'a [u8]> + '_ {
        self.payload.chunks_exact(TS_PACKET_SIZE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn dummy_ts(continuity_counter: u8) -> [u8; TS_PACKET_SIZE] {
        let mut p = [0x00u8; TS_PACKET_SIZE];
        p[0] = 0x47; // TS sync byte
        p[1] = 0x00;
        p[2] = 0x11;
        p[3] = 0x10 | (continuity_counter & 0x0F);
        p
    }

    #[test]
    fn exactly_seven_ts_emits_one_packet() {
        let mut packer = MpegTsRtpPacker::with_rng(&mut StdRng::seed_from_u64(42));
        for i in 0..6 {
            let out = packer.push_ts(&dummy_ts(i), 900_000);
            assert!(out.is_empty());
        }
        let out = packer.push_ts(&dummy_ts(6), 900_000);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), RTP_PACKET_SIZE);
    }

    #[test]
    fn round_trip_parses_back() {
        let mut packer = MpegTsRtpPacker::with_rng(&mut StdRng::seed_from_u64(7));
        let mut emitted = Vec::new();
        for i in 0..7 {
            emitted.extend(packer.push_ts(&dummy_ts(i), 180_000));
        }
        let rtp = &emitted[0];
        let parsed = RtpMp2tPacket::parse(rtp).expect("parse");
        assert_eq!(parsed.timestamp_90khz, 180_000);
        assert_eq!(parsed.ssrc, packer.ssrc());
        assert_eq!(parsed.ts_packets().count(), TS_PER_RTP);
    }

    #[test]
    fn rejects_non_mp2t_payload_type() {
        let mut bytes = [0u8; RTP_HEADER_SIZE + TS_PACKET_SIZE];
        bytes[0] = 0b1000_0000;
        bytes[1] = 96; // dynamic type, not 33
        let err = RtpMp2tPacket::parse(&bytes).unwrap_err();
        assert!(matches!(err, RtpError::BadPayloadType(96)));
    }

    #[test]
    fn rejects_payload_not_multiple_of_188() {
        let mut bytes = vec![0u8; RTP_HEADER_SIZE + 100];
        bytes[0] = 0b1000_0000;
        bytes[1] = RTP_PAYLOAD_TYPE_MP2T;
        let err = RtpMp2tPacket::parse(&bytes).unwrap_err();
        assert!(matches!(err, RtpError::BadPayloadSize(100)));
    }

    #[test]
    fn sequence_number_increments() {
        let mut packer = MpegTsRtpPacker::with_rng(&mut StdRng::seed_from_u64(1));
        let initial_seq = packer.seq;
        for _ in 0..TS_PER_RTP {
            packer.push_ts(&dummy_ts(0), 0);
        }
        for _ in 0..TS_PER_RTP {
            packer.push_ts(&dummy_ts(0), 90);
        }
        assert_eq!(packer.seq, initial_seq.wrapping_add(2));
    }
}
