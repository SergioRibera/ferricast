//! Minimal MPEG-TS muxer for a single H.264 video stream.
//!
//! Self-contained — no third-party muxer dependency. Implements the
//! parts of ISO/IEC 13818-1 that an HLS segment actually needs:
//!
//! * 188-byte transport packets with sync byte, PUSI, AFC, continuity
//!   counters.
//! * One program with one elementary stream (H.264, stream_type
//!   `0x1B`). PAT and PMT are emitted at the start of each segment so
//!   every segment is independently decodable per RFC 8216 §3.
//! * PES packets per access unit, carrying PTS+DTS in the standard
//!   33-bit 90 kHz form.
//! * PCR carried in the adaptation field of the first TS packet of
//!   every access unit. PCR is written slightly behind PTS to satisfy
//!   `PCR <= first_PTS_in_packet` (ISO/IEC 13818-1 §2.7.1).
//!
//! Each access unit is prefixed with a synthetic Access Unit Delimiter
//! (NAL type 9) so decoders that strictly enforce
//! `access_unit_delimiter_rbsp` (e.g. some hardware DTV chips) accept
//! the stream. Random-access points additionally re-emit the SPS+PPS
//! supplied via [`MpegTs::config`], so seeking into a segment never
//! needs out-of-band data.
//!
//! The muxer is single-segment: instantiate, configure, push frames,
//! drain, drop. The HLS segmenter creates a fresh one for every
//! segment.

use ferricast_core::{EncodedFrame, FerricastError};

use crate::Muxer;

const TS_PACKET_SIZE: usize = 188;
const SYNC_BYTE: u8 = 0x47;

const PID_PAT: u16 = 0x0000;
const PID_PMT: u16 = 0x1000;
const PID_VIDEO: u16 = 0x0100;

const STREAM_TYPE_H264: u8 = 0x1B;
const PES_STREAM_ID_VIDEO: u8 = 0xE0;

/// Distance (in 90 kHz ticks) by which PCR is biased *behind* PTS so
/// that `PCR <= first_PTS_in_packet` holds even with rounding. ~2 ms.
const PCR_PTS_OFFSET: u64 = 200;

/// Bit field giving the H.264 codec parameter sets (SPS + PPS,
/// optionally with x264-style SEI) in Annex B form. The muxer
/// prepends these bytes to every keyframe access unit so each
/// segment is independently decodable per RFC 8216 §3.
#[derive(Default)]
pub struct MpegTs {
    parameter_sets: Vec<u8>,
    /// PAT+PMT must be the very first bytes of a segment. We emit
    /// them lazily on the first frame so `drain()` after no frames
    /// returns an empty buffer rather than a partial header.
    psi_emitted: bool,
    /// Continuity counters persist across `drain()` so an HLS
    /// concatenation of segments keeps `cc` strictly monotonic per
    /// PID — ffmpeg's MPEG-TS demuxer treats a CC jump as a
    /// discontinuity and drops the in-progress PES, which manifested
    /// in the wild as `Packet corrupt (stream = 0, dts = …)` errors
    /// at every segment boundary.
    cc_pat: u8,
    cc_pmt: u8,
    cc_video: u8,
    out: Vec<u8>,
}

impl Muxer for MpegTs {
    fn config(&mut self, parameter_sets: Vec<u8>) -> Result<(), FerricastError> {
        if parameter_sets.is_empty() {
            return Err(FerricastError::Encoder(
                "muxer.config received empty H.264 parameter sets (SPS/PPS missing)".into(),
            ));
        }
        self.parameter_sets = parameter_sets;
        Ok(())
    }

    fn add_frame(
        &mut self,
        frame: &EncodedFrame,
        pts_90k: u64,
        dts_90k: u64,
    ) -> Result<(), FerricastError> {
        if frame.data.is_empty() {
            // The encoder produced no NAL units for this frame
            // (shouldn't happen with zerolatency tune, but guard so we
            // don't emit a slice-less PES that the H.264 decoder
            // rejects with "missing picture in access unit").
            return Err(FerricastError::Encoder(
                "muxer.add_frame received empty encoded frame".into(),
            ));
        }
        if !self.psi_emitted {
            self.emit_pat();
            self.emit_pmt();
            self.psi_emitted = true;
        }

        let au = build_access_unit(&self.parameter_sets, frame.is_keyframe, &frame.data);
        let pes = build_pes(&au, pts_90k, dts_90k);

        let pcr = dts_90k.saturating_sub(PCR_PTS_OFFSET);
        write_pes_to_ts(
            &mut self.out,
            PID_VIDEO,
            &mut self.cc_video,
            &pes,
            frame.is_keyframe,
            pcr,
        );

        Ok(())
    }

    fn drain(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.out)
    }
}

impl MpegTs {
    /// Begin a new HLS segment on the existing muxer state. PAT+PMT
    /// will be re-emitted at the start of the next `add_frame`, but
    /// continuity counters and parameter sets carry over so the
    /// concatenated stream stays valid across the segment boundary.
    pub fn start_segment(&mut self) {
        self.psi_emitted = false;
        debug_assert!(self.out.is_empty(), "drain() must be called between segments");
    }

    fn emit_pat(&mut self) {
        let section = build_pat_section();
        write_psi_packet(&mut self.out, PID_PAT, &mut self.cc_pat, &section);
    }

    fn emit_pmt(&mut self) {
        let section = build_pmt_section();
        write_psi_packet(&mut self.out, PID_PMT, &mut self.cc_pmt, &section);
    }
}

// ---------------------------------------------------------------------
// Section construction (PAT / PMT)
// ---------------------------------------------------------------------

fn build_pat_section() -> Vec<u8> {
    // section_length covers everything after the section_length field
    // through CRC inclusive: TSID(2) + flags(1) + sec_num(1) +
    // last_sec_num(1) + program_number(2) + program_map_PID(2) +
    // CRC(4) = 13 bytes.
    let mut s = Vec::with_capacity(17);
    s.push(0x00); // table_id = PAT
    s.extend_from_slice(&(0xB000u16 | 13).to_be_bytes()); // ssi=1, '0', reserved=11, section_length=13
    s.extend_from_slice(&1u16.to_be_bytes()); // transport_stream_id
    s.push(0xC1); // reserved=11, version=0, current_next=1
    s.push(0x00); // section_number
    s.push(0x00); // last_section_number
    s.extend_from_slice(&1u16.to_be_bytes()); // program_number
    s.extend_from_slice(&(0xE000u16 | PID_PMT).to_be_bytes()); // reserved=111, program_map_PID
    let crc = crc32_mpeg2(&s);
    s.extend_from_slice(&crc.to_be_bytes());
    s
}

fn build_pmt_section() -> Vec<u8> {
    // section_length: prog_num(2) + flags(1) + sec_num(1) +
    // last_sec_num(1) + PCR_PID(2) + program_info_len(2) +
    // stream_loop(5) + CRC(4) = 18 bytes.
    let mut s = Vec::with_capacity(22);
    s.push(0x02); // table_id = PMT
    s.extend_from_slice(&(0xB000u16 | 18).to_be_bytes()); // ssi=1, section_length=18
    s.extend_from_slice(&1u16.to_be_bytes()); // program_number
    s.push(0xC1); // reserved=11, version=0, current_next=1
    s.push(0x00); // section_number
    s.push(0x00); // last_section_number
    s.extend_from_slice(&(0xE000u16 | PID_VIDEO).to_be_bytes()); // PCR_PID
    s.extend_from_slice(&0xF000u16.to_be_bytes()); // program_info_length=0
    // Stream loop: H.264 video.
    s.push(STREAM_TYPE_H264);
    s.extend_from_slice(&(0xE000u16 | PID_VIDEO).to_be_bytes()); // elementary_PID
    s.extend_from_slice(&0xF000u16.to_be_bytes()); // ES_info_length=0
    let crc = crc32_mpeg2(&s);
    s.extend_from_slice(&crc.to_be_bytes());
    s
}

/// MPEG-2 PSI CRC (poly 0x04C11DB7, MSB-first, init 0xFFFFFFFF, no
/// final XOR).
fn crc32_mpeg2(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= (b as u32) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ 0x04C1_1DB7
            } else {
                crc << 1
            };
        }
    }
    crc
}

// ---------------------------------------------------------------------
// PES construction
// ---------------------------------------------------------------------

fn build_pes(payload: &[u8], pts_90k: u64, dts_90k: u64) -> Vec<u8> {
    // PES header structure with PTS+DTS:
    //   00 00 01            packet_start_code_prefix
    //   E0                  stream_id (video)
    //   pp pp               PES_packet_length (0 for video = unbounded)
    //   '10' marker, scrambling=00, priority=0, alignment=1, copy=00 → 0x84
    //   PTS_DTS_flags=11, others=0 → 0xC0
    //   PES_header_data_length = 10 (5 PTS + 5 DTS)
    //   5 bytes PTS (prefix 0011)
    //   5 bytes DTS (prefix 0001)
    let mut pes = Vec::with_capacity(19 + payload.len());
    pes.extend_from_slice(&[0x00, 0x00, 0x01, PES_STREAM_ID_VIDEO]);
    // packet_length = 0: legal for video PES, lets the elementary
    // stream span until the next start code without size hint.
    pes.extend_from_slice(&[0x00, 0x00]);
    pes.push(0x84);
    pes.push(0xC0);
    pes.push(0x0A);
    pes.extend_from_slice(&encode_ts(0b0011, pts_90k));
    pes.extend_from_slice(&encode_ts(0b0001, dts_90k));
    pes.extend_from_slice(payload);
    pes
}

/// Encode a 33-bit timestamp into 5 bytes per ISO/IEC 13818-1
/// figure 2-19. `prefix4` is the 4-bit tag (`0010` for PTS-only,
/// `0011` for PTS-with-DTS, `0001` for DTS).
fn encode_ts(prefix4: u8, ts: u64) -> [u8; 5] {
    let p = ts & 0x1_FFFF_FFFF;
    [
        (prefix4 << 4) | ((((p >> 30) as u8) & 0x07) << 1) | 0x01,
        ((p >> 22) & 0xFF) as u8,
        ((((p >> 15) as u8) & 0x7F) << 1) | 0x01,
        ((p >> 7) & 0xFF) as u8,
        (((p as u8) & 0x7F) << 1) | 0x01,
    ]
}

// ---------------------------------------------------------------------
// Access unit construction
// ---------------------------------------------------------------------

const AUD_NAL: [u8; 6] = [0x00, 0x00, 0x00, 0x01, 0x09, 0xF0];

fn build_access_unit(parameter_sets: &[u8], keyframe: bool, frame_data: &[u8]) -> Vec<u8> {
    // Strict order: AUD → (SPS+PPS, only on RAP) → slice NALs. Some
    // decoders refuse a stream that doesn't start an access unit with
    // an AUD; always prepending one is cheap and harmless.
    let mut au = Vec::with_capacity(
        AUD_NAL.len() + if keyframe { parameter_sets.len() } else { 0 } + frame_data.len(),
    );
    au.extend_from_slice(&AUD_NAL);
    if keyframe {
        au.extend_from_slice(parameter_sets);
    }
    au.extend_from_slice(frame_data);
    au
}

// ---------------------------------------------------------------------
// TS packetisation
// ---------------------------------------------------------------------

/// Pack a complete PSI section (PAT or PMT) into a single TS packet.
/// PSI sections are tiny (<23 bytes here) so they always fit inside
/// one packet with stuffing.
fn write_psi_packet(out: &mut Vec<u8>, pid: u16, cc: &mut u8, section: &[u8]) {
    let start = out.len();
    out.push(SYNC_BYTE);
    // PUSI=1, TEI=0, priority=0
    out.push(0x40 | (((pid >> 8) as u8) & 0x1F));
    out.push((pid & 0xFF) as u8);
    out.push(0x10 | (*cc & 0x0F)); // AFC=01 (payload only)
    *cc = (*cc + 1) & 0x0F;
    out.push(0x00); // pointer_field — section starts immediately after
    out.extend_from_slice(section);
    while out.len() - start < TS_PACKET_SIZE {
        out.push(0xFF);
    }
}

/// Split a PES packet across as many 188-byte TS packets as needed.
///
/// Layout:
/// * **First packet**: PUSI=1, AFC=11, adaptation field carrying PCR
///   (and `random_access_indicator=1` for keyframes). Stuffing keeps
///   the packet at exactly 188 bytes when the PES is small.
/// * **Middle packets**: AFC=01, full 184-byte payload.
/// * **Last packet**: AFC=11 with a stuffing-only adaptation field if
///   the remaining payload doesn't fill 184 bytes; otherwise AFC=01
///   like a middle packet.
fn write_pes_to_ts(
    out: &mut Vec<u8>,
    pid: u16,
    cc: &mut u8,
    pes: &[u8],
    keyframe: bool,
    pcr_90k: u64,
) {
    let mut offset = 0usize;
    let mut first = true;

    while offset < pes.len() {
        let pkt_start = out.len();

        out.push(SYNC_BYTE);
        let mut hdr1 = ((pid >> 8) as u8) & 0x1F;
        if first {
            hdr1 |= 0x40; // PUSI
        }
        out.push(hdr1);
        out.push((pid & 0xFF) as u8);
        let afc_cc_pos = out.len();
        out.push(0); // patched once we know AFC

        let remaining = pes.len() - offset;
        let payload_len: usize;
        let af_present: bool;

        if first {
            // Adaptation field carries the PCR + RAI. Minimum AF body
            // = flags(1) + PCR(6) = 7 bytes.
            const AF_BODY_MIN: usize = 7;
            const MAX_PAYLOAD: usize = TS_PACKET_SIZE - 4 /*hdr*/ - 1 /*af_len*/ - AF_BODY_MIN;
            let want = remaining.min(MAX_PAYLOAD);
            let stuffing = MAX_PAYLOAD - want;
            let af_len = AF_BODY_MIN + stuffing;
            out.push(af_len as u8);
            let mut flags = 0u8;
            if keyframe {
                flags |= 0x40; // random_access_indicator
            }
            flags |= 0x10; // PCR_flag
            out.push(flags);
            out.extend_from_slice(&encode_pcr(pcr_90k));
            for _ in 0..stuffing {
                out.push(0xFF);
            }
            payload_len = want;
            af_present = true;
        } else if remaining < TS_PACKET_SIZE - 4 {
            // Last packet: stuff to fill 188 bytes.
            const SPACE: usize = TS_PACKET_SIZE - 4;
            let af_total = SPACE - remaining; // bytes including af_len byte
            let af_len = af_total - 1;
            out.push(af_len as u8);
            if af_len >= 1 {
                out.push(0); // flags = 0 (no PCR, no RAI)
                for _ in 0..(af_len - 1) {
                    out.push(0xFF);
                }
            }
            payload_len = remaining;
            af_present = true;
        } else {
            payload_len = TS_PACKET_SIZE - 4;
            af_present = false;
        }

        let afc: u8 = match (af_present, payload_len > 0) {
            (true, true) => 0b11,
            (true, false) => 0b10,
            (false, true) => 0b01,
            (false, false) => 0b00,
        };
        out[afc_cc_pos] = (afc << 4) | (*cc & 0x0F);
        *cc = (*cc + 1) & 0x0F;

        out.extend_from_slice(&pes[offset..offset + payload_len]);
        offset += payload_len;
        first = false;

        debug_assert_eq!(out.len() - pkt_start, TS_PACKET_SIZE);
    }
}

/// PCR encoded as 33-bit base + 6 reserved bits + 9-bit extension.
/// Extension is always 0; `pcr_90k` is the base directly.
fn encode_pcr(pcr_90k: u64) -> [u8; 6] {
    let base = pcr_90k & 0x1_FFFF_FFFF;
    [
        ((base >> 25) & 0xFF) as u8,
        ((base >> 17) & 0xFF) as u8,
        ((base >> 9) & 0xFF) as u8,
        ((base >> 1) & 0xFF) as u8,
        (((base & 0x1) << 7) as u8) | 0x7E, // bit0 of base, 6 reserved=1, ext bit8 = 0
        0x00,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psi_packet_is_188_bytes() {
        let mut out = Vec::new();
        let mut cc = 0;
        write_psi_packet(&mut out, PID_PAT, &mut cc, &build_pat_section());
        assert_eq!(out.len(), TS_PACKET_SIZE);
        assert_eq!(out[0], SYNC_BYTE);
    }

    #[test]
    fn pat_crc_round_trip() {
        let s = build_pat_section();
        // Recomputing CRC over (section minus its trailing 4 CRC
        // bytes) must yield the appended CRC.
        let body = &s[..s.len() - 4];
        let appended = u32::from_be_bytes(s[s.len() - 4..].try_into().unwrap());
        assert_eq!(crc32_mpeg2(body), appended);
    }

    #[test]
    fn pmt_crc_round_trip() {
        let s = build_pmt_section();
        let body = &s[..s.len() - 4];
        let appended = u32::from_be_bytes(s[s.len() - 4..].try_into().unwrap());
        assert_eq!(crc32_mpeg2(body), appended);
    }

    #[test]
    fn pts_round_trip() {
        for &t in &[0u64, 1, 90_000, 0x1_FFFF_FFFF] {
            let bytes = encode_ts(0b0010, t);
            // Marker bits must all be 1.
            assert_eq!(bytes[0] & 0x01, 0x01);
            assert_eq!(bytes[2] & 0x01, 0x01);
            assert_eq!(bytes[4] & 0x01, 0x01);
            // Decode back.
            let decoded = ((bytes[0] as u64 & 0x0E) << 29)
                | ((bytes[1] as u64) << 22)
                | ((bytes[2] as u64 & 0xFE) << 14)
                | ((bytes[3] as u64) << 7)
                | ((bytes[4] as u64 & 0xFE) >> 1);
            assert_eq!(decoded, t & 0x1_FFFF_FFFF);
        }
    }

    #[test]
    fn pes_split_aligns_to_188() {
        let mut out = Vec::new();
        let mut cc = 0;
        // 5000 bytes — forces multiple TS packets.
        let pes = build_pes(&vec![0xAB; 5000], 90_000, 90_000);
        write_pes_to_ts(&mut out, PID_VIDEO, &mut cc, &pes, true, 89_800);
        assert!(!out.is_empty());
        assert_eq!(out.len() % TS_PACKET_SIZE, 0);
        for i in (0..out.len()).step_by(TS_PACKET_SIZE) {
            assert_eq!(out[i], SYNC_BYTE, "missing sync at packet {}", i / 188);
        }
    }

    #[test]
    fn small_pes_padded_to_188() {
        let mut out = Vec::new();
        let mut cc = 0;
        let pes = build_pes(&[0x01, 0x02, 0x03], 0, 0);
        write_pes_to_ts(&mut out, PID_VIDEO, &mut cc, &pes, true, 0);
        assert_eq!(out.len(), TS_PACKET_SIZE);
    }
}
