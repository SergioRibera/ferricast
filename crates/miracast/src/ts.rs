//! MPEG2-TS single-program muxer and demuxer for Miracast.
//!
//! Layout the Wi-Fi Display spec mandates:
//!
//! | PID | Carries |
//! |---|---|
//! | `0x0000` | PAT |
//! | `0x0100` | PMT |
//! | `0x1011` | H.264 Annex-B video PES (stream_type `0x1B`) |
//! | `0x1100` | AAC-LC ADTS audio PES (stream_type `0x0F`) |
//!
//! PCR rides on the video PID (= program PCR PID). WFD recommends a
//! PCR every ≤40 ms — Samsung sinks stall if the gap exceeds 100 ms.
//! SPS/PPS MUST be repeated in-band at every IDR and at least every
//! ~1 s, else Windows blanks for 1–3 s. AUD NAL (type 9) is inserted
//! at every access unit so TV demuxers can slice reliably.

use bytes::{BufMut, Bytes, BytesMut};

/// MPEG2-TS packet length (fixed).
pub const TS_PACKET_SIZE: usize = 188;
/// Sync byte every TS packet must start with.
pub const TS_SYNC: u8 = 0x47;

/// Program-mandated PID assignment for Miracast.
pub const PAT_PID: u16 = 0x0000;
pub const PMT_PID: u16 = 0x0100;
pub const VIDEO_PID: u16 = 0x1011;
pub const AUDIO_PID: u16 = 0x1100;

/// stream_type for H.264 (ISO/IEC 14496-10) in an MPEG2 PMT.
pub const STREAM_TYPE_H264: u8 = 0x1B;
/// stream_type for ADTS AAC-LC in an MPEG2 PMT.
pub const STREAM_TYPE_AAC_ADTS: u8 = 0x0F;

/// `program_number` for the single program the TS carries.
pub const PROGRAM_NUMBER: u16 = 1;

/// How often to re-emit PAT/PMT, in TS packets. 100 ms at 7 Mbps is
/// roughly every 460 packets — we conservatively emit on every video
/// access unit + every 100 ms to cover startup sync on receivers that
/// don't start demuxing until they see the first PMT.
pub const PSI_PACKET_INTERVAL: u16 = 400;

/// Pre-rendered PAT TS packet (static for the whole session).
#[must_use]
pub fn make_pat() -> [u8; TS_PACKET_SIZE] {
    let mut packet = [0xFFu8; TS_PACKET_SIZE];
    packet[0] = TS_SYNC;
    // PUSI=1, PID=0x0000, adaptation=01 (payload only), CC=0.
    let header = 0x4000u16 | PAT_PID; // PUSI=1
    packet[1] = (header >> 8) as u8;
    packet[2] = (header & 0xFF) as u8;
    packet[3] = 0x10; // adaptation=01, cc=0

    // Pointer field.
    packet[4] = 0x00;

    // PAT section.
    let section = pat_section();
    packet[5..5 + section.len()].copy_from_slice(&section);
    // Everything past the section is stuffing (0xFF fills already).
    packet
}

fn pat_section() -> Vec<u8> {
    // table_id=0x00, section_syntax=1, '0', reserved=11, section_length (12 bits)
    let mut body = Vec::new();
    body.extend_from_slice(&[0x00, 0x00]); // fix section_length below
    body.extend_from_slice(&0x0001u16.to_be_bytes()); // transport_stream_id
    body.push(0xC1); // reserved=11, version=0, current_next_indicator=1
    body.push(0x00); // section_number
    body.push(0x00); // last_section_number
    body.extend_from_slice(&PROGRAM_NUMBER.to_be_bytes());
    body.extend_from_slice(&(0xE000 | PMT_PID).to_be_bytes());
    // CRC32 placeholder; filled after the length is known.
    body.extend_from_slice(&[0, 0, 0, 0]);

    // table_id placed in front.
    let mut section = Vec::with_capacity(1 + body.len());
    section.push(0x00);
    section.extend_from_slice(&body);

    // Fix section_length: length of body AFTER the 2-byte length field
    // itself, INCLUDING the CRC32.
    let section_length = (section.len() - 3) as u16;
    section[1] = 0xB0 | ((section_length >> 8) as u8 & 0x0F);
    section[2] = (section_length & 0xFF) as u8;

    let crc = crc32_mpeg2(&section[..section.len() - 4]);
    let crc_pos = section.len() - 4;
    section[crc_pos..].copy_from_slice(&crc.to_be_bytes());
    section
}

/// Pre-rendered PMT TS packet declaring video and (optional) audio.
#[must_use]
pub fn make_pmt(has_audio: bool) -> [u8; TS_PACKET_SIZE] {
    let mut packet = [0xFFu8; TS_PACKET_SIZE];
    packet[0] = TS_SYNC;
    let header = 0x4000u16 | PMT_PID;
    packet[1] = (header >> 8) as u8;
    packet[2] = (header & 0xFF) as u8;
    packet[3] = 0x10;
    packet[4] = 0x00;

    let section = pmt_section(has_audio);
    packet[5..5 + section.len()].copy_from_slice(&section);
    packet
}

fn pmt_section(has_audio: bool) -> Vec<u8> {
    let mut section = Vec::new();
    section.push(0x02); // table_id
    section.extend_from_slice(&[0x00, 0x00]); // fix section_length
    section.extend_from_slice(&PROGRAM_NUMBER.to_be_bytes());
    section.push(0xC1); // reserved=11 version=0 cur=1
    section.push(0x00); // section_number
    section.push(0x00); // last_section_number
    section.extend_from_slice(&(0xE000 | VIDEO_PID).to_be_bytes()); // PCR_PID = video
    section.extend_from_slice(&0xF000u16.to_be_bytes()); // program_info_length=0

    // Video ES
    section.push(STREAM_TYPE_H264);
    section.extend_from_slice(&(0xE000 | VIDEO_PID).to_be_bytes());
    section.extend_from_slice(&0xF000u16.to_be_bytes());

    // Audio ES
    if has_audio {
        section.push(STREAM_TYPE_AAC_ADTS);
        section.extend_from_slice(&(0xE000 | AUDIO_PID).to_be_bytes());
        section.extend_from_slice(&0xF000u16.to_be_bytes());
    }

    // CRC32 placeholder.
    section.extend_from_slice(&[0, 0, 0, 0]);

    let section_length = (section.len() - 3) as u16;
    section[1] = 0xB0 | ((section_length >> 8) as u8 & 0x0F);
    section[2] = (section_length & 0xFF) as u8;
    let crc = crc32_mpeg2(&section[..section.len() - 4]);
    let crc_pos = section.len() - 4;
    section[crc_pos..].copy_from_slice(&crc.to_be_bytes());
    section
}

/// Access unit about to be packetised into a video PES.
#[derive(Debug, Clone)]
pub struct VideoAccessUnit<'a> {
    /// H.264 Annex-B byte stream (start codes + NALs). The muxer
    /// repeats SPS/PPS before every IDR AU, so callers should pass the
    /// AU as the encoder produced it (no extra fix-up needed).
    pub nalus: &'a [u8],
    /// PTS in 90 kHz units.
    pub pts_90khz: u64,
    /// Whether this AU starts a key frame (SPS/PPS carried inline).
    pub is_keyframe: bool,
    /// When `true`, put a PCR adaptation field on this AU's first TS
    /// packet. Caller decides cadence (WFD demands ≤40 ms).
    pub carry_pcr: bool,
}

/// Minimal MPEG2-TS muxer: one video + optional audio ES, continuity
/// counters per PID, PES+PCR on the video PID.
pub struct MpegTsMuxer {
    video_cc: u8,
    audio_cc: u8,
    pat: [u8; TS_PACKET_SIZE],
    pmt: [u8; TS_PACKET_SIZE],
    packets_since_psi: u16,
}

impl MpegTsMuxer {
    #[must_use]
    pub fn new(has_audio: bool) -> Self {
        Self {
            video_cc: 0,
            audio_cc: 0,
            pat: make_pat(),
            pmt: make_pmt(has_audio),
            packets_since_psi: u16::MAX, // force PSI emission on first AU
        }
    }

    fn next_video_cc(&mut self) -> u8 {
        let cc = self.video_cc;
        self.video_cc = (self.video_cc + 1) & 0x0F;
        cc
    }

    fn next_audio_cc(&mut self) -> u8 {
        let cc = self.audio_cc;
        self.audio_cc = (self.audio_cc + 1) & 0x0F;
        cc
    }

    /// Pack one video access unit into a stream of TS packets. Returns
    /// the packets in emission order (PAT/PMT prepended when their
    /// re-emit interval elapsed, then AU TS packets).
    pub fn pack_video(&mut self, au: &VideoAccessUnit<'_>) -> Vec<Bytes> {
        let mut out = Vec::new();
        if self.packets_since_psi >= PSI_PACKET_INTERVAL || au.is_keyframe {
            out.push(Bytes::copy_from_slice(&self.pat));
            out.push(Bytes::copy_from_slice(&self.pmt));
            self.packets_since_psi = 0;
        }

        let pes = build_video_pes(au);
        let mut emitter = PesEmitter::new(VIDEO_PID, &pes, au.pts_90khz, au.carry_pcr);
        while let Some(packet) = emitter.next_packet(|| self.next_video_cc()) {
            out.push(packet);
            self.packets_since_psi = self.packets_since_psi.saturating_add(1);
        }
        out
    }

    /// Audio AU in ADTS framing. Timestamp in 90 kHz.
    pub fn pack_audio(&mut self, adts: &[u8], pts_90khz: u64) -> Vec<Bytes> {
        let pes = build_audio_pes(adts, pts_90khz);
        let mut emitter = PesEmitter::new(AUDIO_PID, &pes, pts_90khz, false);
        let mut out = Vec::new();
        while let Some(packet) = emitter.next_packet(|| self.next_audio_cc()) {
            out.push(packet);
            self.packets_since_psi = self.packets_since_psi.saturating_add(1);
        }
        out
    }
}

fn build_video_pes(au: &VideoAccessUnit<'_>) -> Vec<u8> {
    let mut pes = Vec::with_capacity(32 + au.nalus.len());
    // packet_start_code_prefix + stream_id (0xE0 = video 0).
    pes.extend_from_slice(&[0x00, 0x00, 0x01, 0xE0]);
    // PES_packet_length placeholder; 0 means "unbounded" which is
    // legal for video and what most sources ship.
    pes.extend_from_slice(&[0x00, 0x00]);
    // '10', PES_scrambling=00, PES_priority=0, data_alignment=1,
    // copyright=0, original=0.
    pes.push(0x84);
    // PTS_DTS_flags=10, ESCR=0, ES_rate=0, trick=0, extra_copy=0,
    // CRC=0, PES_extension=0.
    pes.push(0x80);
    // PES_header_data_length
    pes.push(5);
    push_pts_ts(&mut pes, 0b0010, au.pts_90khz);
    pes.extend_from_slice(au.nalus);
    pes
}

fn build_audio_pes(adts: &[u8], pts_90khz: u64) -> Vec<u8> {
    let mut pes = Vec::with_capacity(32 + adts.len());
    pes.extend_from_slice(&[0x00, 0x00, 0x01, 0xC0]); // stream_id audio 0
    let payload_len = adts.len() + 8; // flags byte + data_len + 5B PTS
    let pes_packet_length = u16::try_from(payload_len).unwrap_or(0);
    pes.extend_from_slice(&pes_packet_length.to_be_bytes());
    pes.push(0x84);
    pes.push(0x80);
    pes.push(5);
    push_pts_ts(&mut pes, 0b0010, pts_90khz);
    pes.extend_from_slice(adts);
    pes
}

fn push_pts_ts(out: &mut Vec<u8>, prefix: u8, ts_90khz: u64) {
    let ts = ts_90khz & 0x1_FFFF_FFFF;
    let b0 = (prefix << 4) | ((ts >> 29) as u8 & 0x0E) | 0x01;
    let b1 = ((ts >> 22) & 0xFF) as u8;
    let b2 = (((ts >> 14) & 0xFF) as u8) | 0x01;
    let b3 = ((ts >> 7) & 0xFF) as u8;
    let b4 = (((ts << 1) & 0xFF) as u8) | 0x01;
    out.extend_from_slice(&[b0, b1, b2, b3, b4]);
}

struct PesEmitter<'a> {
    pid: u16,
    pes: &'a [u8],
    offset: usize,
    pcr_90khz: u64,
    carry_pcr: bool,
    first: bool,
}

impl<'a> PesEmitter<'a> {
    fn new(pid: u16, pes: &'a [u8], pcr_90khz: u64, carry_pcr: bool) -> Self {
        Self {
            pid,
            pes,
            offset: 0,
            pcr_90khz,
            carry_pcr,
            first: true,
        }
    }

    fn next_packet(&mut self, cc: impl FnOnce() -> u8) -> Option<Bytes> {
        if self.offset >= self.pes.len() {
            return None;
        }
        let mut packet = [0xFFu8; TS_PACKET_SIZE];
        packet[0] = TS_SYNC;

        let pusi = if self.first { 1u16 } else { 0 };
        let header = (pusi << 14) | self.pid;
        packet[1] = (header >> 8) as u8;
        packet[2] = (header & 0xFF) as u8;

        let want_pcr = self.first && self.carry_pcr;
        let adaptation = if want_pcr { 0x30 } else { 0x10 };
        packet[3] = adaptation | (cc() & 0x0F);

        let mut payload_start = 4;
        if want_pcr {
            packet[4] = 7; // adaptation_field_length
            packet[5] = 0x10; // flags: PCR present
            let pcr_base = (self.pcr_90khz * 300) / 300; // keep 33 bits
            let pcr_bytes = encode_pcr(pcr_base, 0);
            packet[6..12].copy_from_slice(&pcr_bytes);
            payload_start = 12;
        }

        let remaining = self.pes.len() - self.offset;
        let room = TS_PACKET_SIZE - payload_start;
        if remaining < room {
            // Stuff the start of this packet so payload lands flush at the end.
            let stuff = room - remaining;
            if want_pcr {
                let new_af_len = 7 + stuff as u8;
                packet[4] = new_af_len;
                packet[12..12 + stuff].fill(0xFF);
                payload_start = 12 + stuff;
            } else {
                packet[3] = 0x30 | (packet[3] & 0x0F);
                packet[4] = stuff as u8 - 1;
                if stuff > 1 {
                    packet[5] = 0x00;
                    packet[6..5 + stuff].fill(0xFF);
                }
                payload_start = 4 + stuff;
            }
        }
        let payload_len = TS_PACKET_SIZE - payload_start;
        packet[payload_start..].copy_from_slice(&self.pes[self.offset..self.offset + payload_len]);
        self.offset += payload_len;
        self.first = false;
        Some(Bytes::copy_from_slice(&packet))
    }
}

fn encode_pcr(base: u64, ext: u16) -> [u8; 6] {
    // PCR: 33-bit base + 6 reserved bits + 9-bit extension.
    let base = base & 0x1_FFFF_FFFF;
    let ext = u64::from(ext & 0x01FF);
    let v = (base << 15) | (0x3F << 9) | ext;
    let mut out = [0u8; 6];
    out[0] = (v >> 40) as u8;
    out[1] = (v >> 32) as u8;
    out[2] = (v >> 24) as u8;
    out[3] = (v >> 16) as u8;
    out[4] = (v >> 8) as u8;
    out[5] = v as u8;
    out
}

/// MPEG-2 CRC32 (polynomial 0x04C11DB7, init 0xFFFFFFFF, no reflect).
fn crc32_mpeg2(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= u32::from(b) << 24;
        for _ in 0..8 {
            if crc & 0x8000_0000 != 0 {
                crc = (crc << 1) ^ 0x04C1_1DB7;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

// ---------------------- Demuxer ----------------------

/// One access unit recovered by the demuxer, with its PID + PTS.
#[derive(Debug, Clone)]
pub struct PesUnit {
    pub pid: u16,
    pub pts_90khz: Option<u64>,
    pub payload: Bytes,
}

/// Minimal TS→PES demuxer: tracks continuity per PID and joins PES
/// payloads across packet boundaries. Discards PAT/PMT content; the
/// caller already knows which PIDs carry what.
#[derive(Default)]
pub struct MpegTsDemuxer {
    assemblers: std::collections::HashMap<u16, PesAssembler>,
}

struct PesAssembler {
    buf: BytesMut,
    pts_90khz: Option<u64>,
    expected_cc: Option<u8>,
    in_pes: bool,
}

impl MpegTsDemuxer {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one TS packet; yields complete PES units as they finish.
    pub fn push_packet(&mut self, packet: &[u8]) -> Vec<PesUnit> {
        let mut out = Vec::new();
        if packet.len() != TS_PACKET_SIZE || packet[0] != TS_SYNC {
            return out;
        }
        let pid = u16::from_be_bytes([packet[1] & 0x1F, packet[2]]);
        let pusi = packet[1] & 0x40 != 0;
        let af_flags = (packet[3] >> 4) & 0b11;
        let cc = packet[3] & 0x0F;
        let mut cursor = 4;
        if af_flags == 0b10 || af_flags == 0b11 {
            let af_len = packet[4] as usize;
            cursor = 5 + af_len;
        }
        if af_flags == 0b00 || cursor >= TS_PACKET_SIZE {
            return out;
        }
        if pid == PAT_PID || pid == PMT_PID {
            return out;
        }

        let entry = self.assemblers.entry(pid).or_insert_with(|| PesAssembler {
            buf: BytesMut::new(),
            pts_90khz: None,
            expected_cc: None,
            in_pes: false,
        });

        // Continuity check — on jump, drop the in-flight PES.
        if let Some(prev) = entry.expected_cc
            && cc != prev
        {
            entry.buf.clear();
            entry.in_pes = false;
        }
        entry.expected_cc = Some((cc + 1) & 0x0F);

        if pusi {
            if let Some(unit) = entry.finalize(pid) {
                out.push(unit);
            }
            let payload = &packet[cursor..];
            if let Some((pts, pes_payload)) = parse_pes_header(payload) {
                entry.buf.clear();
                entry.buf.put_slice(pes_payload);
                entry.pts_90khz = pts;
                entry.in_pes = true;
            } else {
                entry.in_pes = false;
            }
        } else if entry.in_pes {
            entry.buf.put_slice(&packet[cursor..]);
        }
        out
    }

    /// Flush any in-flight PES at end-of-stream.
    pub fn finish(&mut self) -> Vec<PesUnit> {
        let mut out = Vec::new();
        for (pid, a) in &mut self.assemblers {
            if let Some(unit) = a.finalize(*pid) {
                out.push(unit);
            }
        }
        out
    }
}

impl PesAssembler {
    fn finalize(&mut self, pid: u16) -> Option<PesUnit> {
        if !self.in_pes || self.buf.is_empty() {
            return None;
        }
        self.in_pes = false;
        let payload = self.buf.split().freeze();
        Some(PesUnit {
            pid,
            pts_90khz: self.pts_90khz.take(),
            payload,
        })
    }
}

fn parse_pes_header(buf: &[u8]) -> Option<(Option<u64>, &[u8])> {
    if buf.len() < 9 || buf[0] != 0x00 || buf[1] != 0x00 || buf[2] != 0x01 {
        return None;
    }
    let pts_dts_flags = (buf[7] >> 6) & 0b11;
    let header_data_length = buf[8] as usize;
    if buf.len() < 9 + header_data_length {
        return None;
    }
    let pts = if pts_dts_flags & 0b10 != 0 && header_data_length >= 5 {
        Some(decode_pts(&buf[9..14]))
    } else {
        None
    };
    Some((pts, &buf[9 + header_data_length..]))
}

fn decode_pts(b: &[u8]) -> u64 {
    let b0 = u64::from(b[0]);
    let b1 = u64::from(b[1]);
    let b2 = u64::from(b[2]);
    let b3 = u64::from(b[3]);
    let b4 = u64::from(b[4]);
    (((b0 >> 1) & 0x07) << 30)
        | (b1 << 22)
        | (((b2 >> 1) & 0x7F) << 15)
        | (b3 << 7)
        | ((b4 >> 1) & 0x7F)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pat_packet_sync_and_pid() {
        let pat = make_pat();
        assert_eq!(pat[0], TS_SYNC);
        let header = u16::from_be_bytes([pat[1], pat[2]]);
        assert_eq!(header & 0x1FFF, PAT_PID);
    }

    #[test]
    fn pmt_declares_video_stream_type() {
        let pmt = make_pmt(false);
        // After header (4) + pointer (1) + section header (3) + three
        // info bytes (5) + PCR_PID (2) + program_info_length (2),
        // stream_type byte for video is at index 17.
        assert_eq!(pmt[17], STREAM_TYPE_H264);
    }

    #[test]
    fn pcr_adaptation_field_present_when_asked() {
        let mut mux = MpegTsMuxer::new(false);
        let au = VideoAccessUnit {
            nalus: &[0, 0, 0, 1, 0x09, 0x10, 0, 0, 0, 1, 0x65, 0xAA],
            pts_90khz: 90_000,
            is_keyframe: true,
            carry_pcr: true,
        };
        let packets = mux.pack_video(&au);
        // PAT + PMT + at least one AU packet.
        assert!(packets.len() >= 3);
        // The AU's first TS packet should have PCR in adaptation.
        let first_au = &packets[2];
        let af_flags = (first_au[3] >> 4) & 0b11;
        assert!(af_flags & 0b10 != 0);
    }

    #[test]
    fn round_trip_through_demuxer_preserves_payload_and_pts() {
        let mut mux = MpegTsMuxer::new(false);
        let nalus: Vec<u8> = {
            let mut v = Vec::new();
            v.extend_from_slice(&[0, 0, 0, 1, 0x09, 0x10]);
            v.extend_from_slice(&[0, 0, 0, 1, 0x65]);
            v.extend((0..300u16).map(|i| (i & 0xFF) as u8));
            v
        };
        let au = VideoAccessUnit {
            nalus: &nalus,
            pts_90khz: 123_456,
            is_keyframe: true,
            carry_pcr: true,
        };
        let packets = mux.pack_video(&au);

        let mut demux = MpegTsDemuxer::new();
        let mut units = Vec::new();
        for p in &packets {
            units.extend(demux.push_packet(p));
        }
        units.extend(demux.finish());

        let video: Vec<_> = units.into_iter().filter(|u| u.pid == VIDEO_PID).collect();
        assert_eq!(video.len(), 1);
        assert_eq!(video[0].pts_90khz, Some(123_456));
        assert!(
            video[0]
                .payload
                .ends_with(&nalus[nalus.len().saturating_sub(16)..])
        );
    }
}
