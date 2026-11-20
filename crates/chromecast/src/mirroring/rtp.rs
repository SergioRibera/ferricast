//! Cast Streaming RTP: frame identifiers, frame encryption and packetization.
//!
//! The wire layout follows openscreen `cast/streaming/impl/rtp_defines.h` and
//! `rtp_packetizer.cc`:
//!
//! ```text
//!  byte 0      0x80 (V=2, no padding, no RTP extensions, no CSRCs)
//!  byte 1      marker (last packet of the frame) | 7-bit payload type
//!  bytes 2..4  sequence number (u16 BE, random start)
//!  bytes 4..8  RTP timestamp (u32 BE)
//!  bytes 8..12 sender SSRC (u32 BE)
//!  byte 12     0x80 key frame | 0x40 reference frame id present | extension count
//!  byte 13     frame id, lower 8 bits
//!  bytes 14..16 packet id (u16 BE)
//!  bytes 16..18 max packet id (u16 BE) = packet count - 1
//!  byte 18     referenced frame id, lower 8 bits
//!  [ext]       Adaptive Latency: u16 (type 1 << 10 | size 2), u16 playout delay ms
//!  payload     slice of the AES-128-CTR encrypted frame
//! ```

use aes::Aes128;
use aes::cipher::{KeyIvInit, StreamCipher};
use bytes::Bytes;

/// Largest RTP packet that fits an Ethernet MTU over IPv4/UDP (1500 - 20 - 8).
pub(crate) const MAX_RTP_PACKET_SIZE: usize = 1500 - 20 - 8;
/// RTP header (12) + Cast header (7, reference frame id always included).
pub(crate) const BASE_RTP_HEADER_SIZE: usize = 19;
/// Size of the optional Adaptive Latency extension.
pub(crate) const ADAPTIVE_LATENCY_HEADER_SIZE: usize = 4;
/// Payload bytes per packet. Space for the optional extension is always
/// reserved so every packet of a frame carries the same chunk size.
pub(crate) const MAX_PAYLOAD_SIZE: usize =
    MAX_RTP_PACKET_SIZE - BASE_RTP_HEADER_SIZE - ADAPTIVE_LATENCY_HEADER_SIZE;
/// Packet id reserved by RTCP feedback for "all packets of the frame lost".
pub(crate) const ALL_PACKETS_LOST: u16 = 0xFFFF;
/// Highest packet id a frame may use.
pub(crate) const MAX_ALLOWED_FRAME_PACKET_ID: u16 = ALL_PACKETS_LOST - 1;

const RTP_REQUIRED_FIRST_BYTE: u8 = 0x80;
const RTP_MARKER_BIT: u8 = 0x80;
const KEY_FRAME_BIT: u8 = 0x80;
const HAS_REFERENCE_FRAME_ID_BIT: u8 = 0x40;
const ADAPTIVE_LATENCY_EXTENSION_TYPE: u16 = 1;
const EXTENSION_SIZE_FIELD_BITS: u32 = 10;

/// Cast Streaming RTP payload types (openscreen `RtpPayloadType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RtpPayloadType {
    Opus,
    H264,
}

impl RtpPayloadType {
    #[must_use]
    pub(crate) const fn value(self) -> u8 {
        match self {
            Self::Opus => 96,
            Self::H264 => 101,
        }
    }
}

/// Monotonic frame identifier. The first frame of a stream is `0`; `-1`
/// ("leader") stands for "before the first frame". Only the lower 8 bits go
/// on the wire, so values read back from the wire are expanded relative to
/// a known nearby id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct FrameId(i64);

impl FrameId {
    pub(crate) const LEADER: Self = Self(-1);
    pub(crate) const FIRST: Self = Self(0);

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn new(value: i64) -> Self {
        Self(value)
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn value(self) -> i64 {
        self.0
    }

    #[must_use]
    pub(crate) const fn next(self) -> Self {
        Self(self.0 + 1)
    }

    #[must_use]
    pub(crate) const fn offset(self, delta: i64) -> Self {
        Self(self.0 + delta)
    }

    /// Number of frames from `other` to `self`.
    #[must_use]
    pub(crate) const fn distance_from(self, other: Self) -> i64 {
        self.0 - other.0
    }

    #[must_use]
    pub(crate) const fn lower_8_bits(self) -> u8 {
        (self.0 & 0xFF) as u8
    }

    #[must_use]
    pub(crate) const fn lower_32_bits(self) -> u32 {
        (self.0 & 0xFFFF_FFFF) as u32
    }

    /// Largest id `<= self` whose lower 8 bits equal `truncated`.
    #[must_use]
    pub(crate) const fn expand_less_than_or_equal(self, truncated: u8) -> Self {
        let candidate = (self.0 & !0xFF) | truncated as i64;
        if candidate > self.0 {
            Self(candidate - 0x100)
        } else {
            Self(candidate)
        }
    }

    /// Smallest id `> self` whose lower 8 bits equal `truncated`.
    #[must_use]
    pub(crate) const fn expand_greater_than(self, truncated: u8) -> Self {
        let candidate = (self.0 & !0xFF) | truncated as i64;
        if candidate <= self.0 {
            Self(candidate + 0x100)
        } else {
            Self(candidate)
        }
    }
}

/// One encoded frame, already encrypted, ready to be split into packets.
#[derive(Debug, Clone)]
pub(crate) struct EncryptedFrame {
    pub(crate) frame_id: FrameId,
    /// Key frames reference themselves; delta frames reference the frame
    /// they were predicted from.
    pub(crate) referenced_frame_id: FrameId,
    pub(crate) is_key_frame: bool,
    pub(crate) rtp_timestamp: u32,
    /// Capture time of the frame in UNIX-epoch microseconds; the Sender
    /// Report maps this clock onto the RTP timeline.
    pub(crate) reference_time_us: u64,
    /// Non-zero when this frame changes the receiver's playout delay.
    pub(crate) new_playout_delay_ms: u16,
    pub(crate) data: Bytes,
}

impl EncryptedFrame {
    /// Number of packets needed for this frame, or `None` when the frame is
    /// too large to be addressed by 16-bit packet ids.
    #[must_use]
    pub(crate) fn packet_count(&self) -> Option<u16> {
        packet_count_for(self.data.len())
    }
}

/// Packets needed for `len` payload bytes. Empty frames still take one packet
/// (some audio codecs encode silence as zero bytes).
#[must_use]
pub(crate) fn packet_count_for(len: usize) -> Option<u16> {
    let count = len.div_ceil(MAX_PAYLOAD_SIZE).max(1);
    u16::try_from(count)
        .ok()
        .filter(|&c| c <= MAX_ALLOWED_FRAME_PACKET_ID)
}

/// AES-128-CTR frame encryption (openscreen `frame_crypto.cc`).
///
/// The nonce is 16 zero bytes with the lower 32 bits of the frame id written
/// big-endian at offset 8, XORed with the per-stream IV mask. The whole frame
/// is encrypted as one keystream; packets then carry consecutive slices of the
/// ciphertext.
#[derive(Clone)]
pub(crate) struct FrameCrypto {
    key: [u8; 16],
    iv_mask: [u8; 16],
}

impl FrameCrypto {
    #[must_use]
    pub(crate) const fn new(key: [u8; 16], iv_mask: [u8; 16]) -> Self {
        Self { key, iv_mask }
    }

    #[must_use]
    pub(crate) fn nonce(&self, frame_id: FrameId) -> [u8; 16] {
        let mut nonce = [0u8; 16];
        nonce[8..12].copy_from_slice(&frame_id.lower_32_bits().to_be_bytes());
        for (byte, mask) in nonce.iter_mut().zip(self.iv_mask) {
            *byte ^= mask;
        }
        nonce
    }

    /// Encrypts (or decrypts — CTR is symmetric) a whole frame.
    #[must_use]
    pub(crate) fn crypt(&self, frame_id: FrameId, data: &[u8]) -> Vec<u8> {
        let mut out = data.to_vec();
        let nonce = self.nonce(frame_id);
        ctr::Ctr128BE::<Aes128>::new((&self.key).into(), (&nonce).into()).apply_keystream(&mut out);
        out
    }
}

/// Splits encrypted frames into Cast RTP packets for one stream.
#[derive(Debug)]
pub(crate) struct RtpPacketizer {
    payload_type: RtpPayloadType,
    ssrc: u32,
    sequence_number: u16,
}

impl RtpPacketizer {
    #[must_use]
    pub(crate) const fn new(payload_type: RtpPayloadType, ssrc: u32, first_sequence: u16) -> Self {
        Self {
            payload_type,
            ssrc,
            sequence_number: first_sequence,
        }
    }

    /// Writes packet `packet_id` of `frame` into `out` (cleared first).
    /// Every call consumes one sequence number, retransmissions included.
    ///
    /// Returns `false` (leaving `out` empty) when `packet_id` is out of
    /// range for the frame.
    pub(crate) fn generate_packet(
        &mut self,
        frame: &EncryptedFrame,
        packet_id: u16,
        out: &mut Vec<u8>,
    ) -> bool {
        out.clear();
        let Some(num_packets) = frame.packet_count() else {
            return false;
        };
        if packet_id >= num_packets {
            return false;
        }
        let is_last_packet = packet_id + 1 == num_packets;
        let include_latency_change = packet_id == 0 && frame.new_playout_delay_ms > 0;

        let chunk_start = MAX_PAYLOAD_SIZE * usize::from(packet_id);
        let chunk_end = if is_last_packet {
            frame.data.len()
        } else {
            chunk_start + MAX_PAYLOAD_SIZE
        };
        let Some(chunk) = frame.data.get(chunk_start..chunk_end) else {
            return false;
        };

        out.reserve(BASE_RTP_HEADER_SIZE + ADAPTIVE_LATENCY_HEADER_SIZE + chunk.len());
        out.push(RTP_REQUIRED_FIRST_BYTE);
        out.push(if is_last_packet { RTP_MARKER_BIT } else { 0 } | self.payload_type.value());
        out.extend_from_slice(&self.sequence_number.to_be_bytes());
        self.sequence_number = self.sequence_number.wrapping_add(1);
        out.extend_from_slice(&frame.rtp_timestamp.to_be_bytes());
        out.extend_from_slice(&self.ssrc.to_be_bytes());

        out.push(
            if frame.is_key_frame { KEY_FRAME_BIT } else { 0 }
                | HAS_REFERENCE_FRAME_ID_BIT
                | u8::from(include_latency_change),
        );
        out.push(frame.frame_id.lower_8_bits());
        out.extend_from_slice(&packet_id.to_be_bytes());
        out.extend_from_slice(&(num_packets - 1).to_be_bytes());
        out.push(frame.referenced_frame_id.lower_8_bits());
        if include_latency_change {
            let type_and_size = (ADAPTIVE_LATENCY_EXTENSION_TYPE << EXTENSION_SIZE_FIELD_BITS) | 2;
            out.extend_from_slice(&type_and_size.to_be_bytes());
            out.extend_from_slice(&frame.new_playout_delay_ms.to_be_bytes());
        }
        out.extend_from_slice(chunk);
        true
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use aes::cipher::{BlockEncrypt, KeyInit};

    /// Independent AES-128-CTR following OpenSSL `AES_ctr128_encrypt` as
    /// called by openscreen's `FrameCrypto::Crypt`: the keystream state
    /// (`ecount`, `block_offset`, counter) carries across chunks.
    pub(crate) struct OpenSslStyleCtr {
        cipher: Aes128,
        counter: [u8; 16],
        ecount: [u8; 16],
        block_offset: usize,
    }

    impl OpenSslStyleCtr {
        pub(crate) fn new(key: &[u8; 16], iv_mask: &[u8; 16], frame_id: u32) -> Self {
            let mut counter = [0u8; 16];
            counter[8..12].copy_from_slice(&frame_id.to_be_bytes());
            for (c, m) in counter.iter_mut().zip(iv_mask) {
                *c ^= m;
            }
            Self {
                cipher: Aes128::new(key.into()),
                counter,
                ecount: [0; 16],
                block_offset: 0,
            }
        }

        fn increment(&mut self) {
            for byte in self.counter.iter_mut().rev() {
                *byte = byte.wrapping_add(1);
                if *byte != 0 {
                    break;
                }
            }
        }

        pub(crate) fn apply(&mut self, chunk: &[u8]) -> Vec<u8> {
            chunk
                .iter()
                .map(|b| {
                    if self.block_offset == 0 {
                        let mut block = self.counter.into();
                        self.cipher.encrypt_block(&mut block);
                        self.ecount.copy_from_slice(&block);
                        self.increment();
                    }
                    let out = b ^ self.ecount[self.block_offset];
                    self.block_offset = (self.block_offset + 1) % 16;
                    out
                })
                .collect()
        }
    }

    fn frame(id: i64, key: bool, len: usize, delay: u16) -> EncryptedFrame {
        EncryptedFrame {
            frame_id: FrameId::new(id),
            referenced_frame_id: if key {
                FrameId::new(id)
            } else {
                FrameId::new(id - 1)
            },
            is_key_frame: key,
            rtp_timestamp: 0x1122_3344,
            reference_time_us: 0,
            new_playout_delay_ms: delay,
            data: Bytes::from((0..len).map(|i| (i % 251) as u8).collect::<Vec<_>>()),
        }
    }

    #[test]
    fn key_frame_single_packet_is_byte_exact() {
        let mut p = RtpPacketizer::new(RtpPayloadType::H264, 0xAABB_CCDD, 0xFFFF);
        let f = frame(0x1_05, true, 3, 0);
        let mut out = Vec::new();
        assert!(p.generate_packet(&f, 0, &mut out));
        let expected = [
            0x80,
            0x80 | 101,
            0xFF,
            0xFF,
            0x11,
            0x22,
            0x33,
            0x44,
            0xAA,
            0xBB,
            0xCC,
            0xDD, // RTP
            0xC0,
            0x05,
            0x00,
            0x00,
            0x00,
            0x00,
            0x05, // Cast header
            0,
            1,
            2, // payload
        ];
        assert_eq!(out, expected);
        // Sequence number wraps.
        assert!(p.generate_packet(&f, 0, &mut out));
        assert_eq!(&out[2..4], &[0x00, 0x00]);
    }

    #[test]
    fn delta_frame_multi_packet_layout() {
        let mut p = RtpPacketizer::new(RtpPayloadType::Opus, 7, 100);
        let len = MAX_PAYLOAD_SIZE * 2 + 10;
        let f = frame(300, false, len, 0);
        let mut out = Vec::new();
        for pid in 0..3u16 {
            assert!(p.generate_packet(&f, pid, &mut out));
            let last = pid == 2;
            assert_eq!(out[0], 0x80);
            assert_eq!(out[1], if last { 0x80 | 96 } else { 96 });
            assert_eq!(u16::from_be_bytes([out[2], out[3]]), 100 + pid);
            assert_eq!(out[12], 0x40, "delta frame: only the reference bit");
            assert_eq!(out[13], (300 & 0xFF) as u8);
            assert_eq!(u16::from_be_bytes([out[14], out[15]]), pid);
            assert_eq!(u16::from_be_bytes([out[16], out[17]]), 2);
            assert_eq!(out[18], (299 & 0xFF) as u8);
            let start = MAX_PAYLOAD_SIZE * usize::from(pid);
            let end = if last { len } else { start + MAX_PAYLOAD_SIZE };
            assert_eq!(&out[19..], &f.data[start..end]);
            assert!(out.len() <= MAX_RTP_PACKET_SIZE);
        }
        assert!(!p.generate_packet(&f, 3, &mut out));
        assert!(out.is_empty());
    }

    #[test]
    fn adaptive_latency_extension_only_on_first_packet() {
        let mut p = RtpPacketizer::new(RtpPayloadType::H264, 1, 0);
        let f = frame(1, true, MAX_PAYLOAD_SIZE + 1, 500);
        let mut out = Vec::new();
        assert!(p.generate_packet(&f, 0, &mut out));
        assert_eq!(out[12], 0x80 | 0x40 | 1);
        assert_eq!(&out[19..23], &[0x04, 0x02, 0x01, 0xF4]);
        assert_eq!(out.len(), MAX_RTP_PACKET_SIZE);
        assert!(p.generate_packet(&f, 1, &mut out));
        assert_eq!(out[12], 0x80 | 0x40);
        assert_eq!(out.len(), BASE_RTP_HEADER_SIZE + 1);
    }

    #[test]
    fn empty_frame_still_gets_one_packet() {
        let mut p = RtpPacketizer::new(RtpPayloadType::Opus, 1, 0);
        let f = frame(0, true, 0, 0);
        let mut out = Vec::new();
        assert!(p.generate_packet(&f, 0, &mut out));
        assert_eq!(out.len(), BASE_RTP_HEADER_SIZE);
        assert_eq!(out[1], 0x80 | 96);
    }

    #[test]
    fn oversize_frame_rejected() {
        assert_eq!(packet_count_for(MAX_PAYLOAD_SIZE * 0xFFFE), Some(0xFFFE));
        assert_eq!(packet_count_for(MAX_PAYLOAD_SIZE * 0xFFFE + 1), None);
    }

    #[test]
    fn nonce_matches_frame_crypto_layout() {
        let mask = [0xA5; 16];
        let c = FrameCrypto::new([0; 16], mask);
        let nonce = c.nonce(FrameId::new(0x1_0203_0405));
        // Only the lower 32 bits of the id (0x0203_0405) enter the nonce.
        let mut expected = [0xA5; 16];
        expected[8] = 0xA5 ^ 0x02;
        expected[9] = 0xA5 ^ 0x03;
        expected[10] = 0xA5 ^ 0x04;
        expected[11] = 0xA5 ^ 0x05;
        assert_eq!(nonce, expected);
    }

    #[test]
    fn aes_known_answer_first_block() {
        // FIPS-197 C.1: AES-128(000102..0f, 00112233..ff) = 69c4e0d86a7b0430d8cdb78070b4c55a.
        // With an all-zero frame id and an IV mask equal to that plaintext,
        // the first keystream block is exactly that ciphertext.
        let key: [u8; 16] = core::array::from_fn(|i| i as u8);
        let mask: [u8; 16] = core::array::from_fn(|i| (i as u8) * 0x11);
        let c = FrameCrypto::new(key, mask);
        let ks = c.crypt(FrameId::new(0), &[0u8; 16]);
        assert_eq!(
            ks,
            [
                0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4,
                0xc5, 0x5a
            ]
        );
    }

    #[test]
    fn packet_chunks_decrypt_with_continuous_keystream() {
        let key = [0x3C; 16];
        let mask: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(37));
        let crypto = FrameCrypto::new(key, mask);
        let plain: Vec<u8> = (0..MAX_PAYLOAD_SIZE * 3 + 77)
            .map(|i| (i * 7) as u8)
            .collect();
        let id = FrameId::new(0x0012_3456_789A);
        let cipher = crypto.crypt(id, &plain);
        assert_ne!(cipher, plain);
        let mut f = frame(id.value(), true, 0, 0);
        f.data = Bytes::from(cipher);

        let mut p = RtpPacketizer::new(RtpPayloadType::H264, 9, 0);
        let mut dec = OpenSslStyleCtr::new(&key, &mask, id.lower_32_bits());
        let mut recovered = Vec::new();
        let mut out = Vec::new();
        for pid in 0..f.packet_count().unwrap() {
            assert!(p.generate_packet(&f, pid, &mut out));
            recovered.extend(dec.apply(&out[BASE_RTP_HEADER_SIZE..]));
        }
        assert_eq!(recovered, plain);
    }

    #[test]
    fn frame_id_expansion() {
        let max = FrameId::new(0x305);
        assert_eq!(max.expand_less_than_or_equal(0x05), FrameId::new(0x305));
        assert_eq!(max.expand_less_than_or_equal(0x04), FrameId::new(0x304));
        assert_eq!(max.expand_less_than_or_equal(0x06), FrameId::new(0x206));
        assert_eq!(
            FrameId::LEADER.expand_less_than_or_equal(0xFF),
            FrameId::LEADER
        );
        let cp = FrameId::new(0x2FE);
        assert_eq!(cp.expand_greater_than(0xFF), FrameId::new(0x2FF));
        assert_eq!(cp.expand_greater_than(0x01), FrameId::new(0x301));
        assert_eq!(cp.expand_greater_than(0xFE), FrameId::new(0x3FE));
        assert_eq!(FrameId::LEADER.expand_greater_than(0x00), FrameId::FIRST);
    }
}
