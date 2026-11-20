//! Minimal MPEG-TS demuxer for HLS pulled segments.
//!
//! The full TS spec (ISO/IEC 13818-1) is huge; receivers only need
//! a small slice of it because HLS-muxed streams hit a narrow
//! happy path:
//!
//! - 188-byte packets, sync byte `0x47`.
//! - One PAT (PID 0) pointing to one PMT PID.
//! - One PMT listing a video ES (stream_type `0x1B` = H.264,
//!   `0x24` = H.265) and an audio ES (stream_type `0x0F` = AAC ADTS).
//! - Each ES carries PES packets; PES headers carry PTS (and
//!   sometimes DTS) at 90 kHz.
//!
//! PSI sections (PAT / PMT) may span several TS packets and are
//! reassembled before parsing; parsing stays inside `section_length`
//! so the CRC and stuffing bytes are never read as stream entries.
//!
//! Timestamps are 33-bit counters that wrap every ~26.5 h; the
//! demuxer unwraps them into a monotonic 64-bit timeline so a stream
//! crossing the wrap point keeps increasing timestamps.
//!
//! No support for: encrypted transport streams (segment-level
//! AES-128 is decrypted by the caller before the bytes get here),
//! multi-program transport streams, PSI versioning, non-AAC audio.

use ferricast_core::{AudioCodec, AudioFrame, Codec, EncodedFrame};

/// Result of feeding bytes through [`TsDemuxer::push`]. The demuxer
/// emits one `Video` / `Audio` per fully reassembled PES packet —
/// callers typically pass the whole packet straight through to the
/// codec, which handles internal fragmentation.
#[derive(Debug)]
pub enum DemuxedPacket {
    Video(EncodedFrame),
    Audio(AudioFrame),
}

#[derive(Debug, Default)]
pub struct TsStreamInfo {
    pub video_codec: Option<Codec>,
    pub audio_codec: Option<AudioCodec>,
    /// Audio sample rate parsed from the first ADTS header. Filled
    /// in lazily during demux, not at PMT time, because the PMT
    /// only carries stream_type and not codec parameters.
    pub audio_sample_rate: u32,
    pub audio_channels: u16,
}

const TS_PACKET_LEN: usize = 188;
const PAT_PID: u16 = 0;
/// 2^33: the wrap period of PES timestamps in 90 kHz ticks.
const TS_WRAP: i64 = 1 << 33;
/// PSI sections are at most 1024 bytes (ISO/IEC 13818-1 §2.4.4);
/// anything larger is garbage and gets dropped.
const MAX_SECTION_LEN: usize = 1024;

#[derive(Debug, Default)]
pub struct TsDemuxer {
    pmt_pid: Option<u16>,
    video_pid: Option<u16>,
    audio_pid: Option<u16>,
    info: TsStreamInfo,

    pat_section: SectionAssembly,
    pmt_section: SectionAssembly,
    video_pes: PesAssembly,
    audio_pes: PesAssembly,
    clock: TimestampUnwrapper,
    out: Vec<DemuxedPacket>,
}

#[derive(Debug, Default)]
struct PesAssembly {
    buf: Vec<u8>,
    /// Unwrapped PTS / DTS of the PES being assembled, in 90 kHz
    /// ticks. `None` until a PES header carrying them lands.
    pts: Option<i64>,
    dts: Option<i64>,
    /// True after a PES header was parsed and until the PES is
    /// emitted; continuation packets outside a PES are dropped.
    in_pes: bool,
}

/// Reassembles one PSI table that may span several TS packets.
#[derive(Debug, Default)]
struct SectionAssembly {
    buf: Vec<u8>,
    /// True between a PUSI and the end of the section it started.
    active: bool,
}

impl SectionAssembly {
    /// Feeds one TS payload; returns a complete section (table_id up
    /// to and including the CRC) once all of it has arrived.
    fn push(&mut self, payload: &[u8], pusi: bool) -> Option<Vec<u8>> {
        if pusi {
            let pointer = usize::from(*payload.first()?);
            let start = payload.get(1 + pointer..)?;
            self.buf.clear();
            self.buf.extend_from_slice(start);
            self.active = true;
        } else if self.active {
            self.buf.extend_from_slice(payload);
        } else {
            return None;
        }
        if self.buf.len() < 3 {
            return None;
        }
        let total = 3 + ((usize::from(self.buf[1] & 0x0f) << 8) | usize::from(self.buf[2]));
        if total > MAX_SECTION_LEN {
            self.active = false;
            self.buf.clear();
            return None;
        }
        if self.buf.len() < total {
            return None;
        }
        self.active = false;
        self.buf.truncate(total);
        Some(std::mem::take(&mut self.buf))
    }
}

/// Extends 33-bit PES timestamps into a monotonic 64-bit timeline.
/// Audio and video share one instance because they share the
/// program clock; each value is placed in the wrap period closest to
/// the previous one.
#[derive(Debug, Default)]
struct TimestampUnwrapper {
    last: Option<i64>,
}

impl TimestampUnwrapper {
    fn unwrap(&mut self, raw: i64) -> i64 {
        let value = match self.last {
            None => raw,
            Some(last) => {
                let base = last - last.rem_euclid(TS_WRAP);
                [base - TS_WRAP + raw, base + raw, base + TS_WRAP + raw]
                    .into_iter()
                    .min_by_key(|candidate| (candidate - last).abs())
                    .unwrap_or(raw)
            }
        };
        self.last = Some(value);
        value
    }
}

/// 90 kHz ticks → microseconds, clamped at zero (an unwrapped
/// timestamp can dip below zero right after a backwards wrap).
fn ticks_to_us(ticks: i64) -> u64 {
    u64::try_from(ticks.max(0)).unwrap_or(0) * 100 / 9
}

impl TsDemuxer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn info(&self) -> &TsStreamInfo {
        &self.info
    }

    /// Feed a chunk of TS bytes into the demuxer. Any number of
    /// 188-byte packets fits — the demuxer scans for the next sync
    /// byte and resyncs on garbage rather than rejecting.
    pub fn push(&mut self, chunk: &[u8]) -> &mut Vec<DemuxedPacket> {
        let mut i = 0;
        while let Some(pkt) = chunk.get(i..i + TS_PACKET_LEN) {
            if pkt[0] != 0x47 {
                // Resync: scan forward until the next sync byte.
                // Partial / corrupted transfers can trigger this.
                i += 1;
                continue;
            }
            self.process_packet(pkt);
            i += TS_PACKET_LEN;
        }
        &mut self.out
    }

    /// Flush any PES that hasn't been emitted yet — the last frame
    /// of a segment usually has no follower to trigger the emit, so
    /// we flush on segment boundary or EOS.
    pub fn flush(&mut self) -> &mut Vec<DemuxedPacket> {
        self.emit(true);
        self.emit(false);
        &mut self.out
    }

    fn process_packet(&mut self, pkt: &[u8]) {
        // TS header: sync(1) | tei(1) pusi(1) tpr(1) pid_hi(5) | pid_lo(8) | scrambling(2) afc(2) cc(4)
        let pusi = (pkt[1] & 0x40) != 0;
        let pid = (u16::from(pkt[1] & 0x1f) << 8) | u16::from(pkt[2]);
        let afc = (pkt[3] >> 4) & 0x03;
        let payload_start = match afc {
            // Payload only.
            1 => 4,
            // Adaptation field followed by payload.
            3 => 5 + usize::from(pkt[4]),
            // Adaptation field only, or reserved.
            _ => return,
        };
        let Some(payload) = pkt.get(payload_start..).filter(|p| !p.is_empty()) else {
            return;
        };

        if pid == PAT_PID {
            if let Some(section) = self.pat_section.push(payload, pusi) {
                self.parse_pat(&section);
            }
        } else if Some(pid) == self.pmt_pid {
            if let Some(section) = self.pmt_section.push(payload, pusi) {
                self.parse_pmt(&section);
            }
        } else if Some(pid) == self.video_pid {
            self.collect_pes(true, payload, pusi);
        } else if Some(pid) == self.audio_pid {
            self.collect_pes(false, payload, pusi);
        }
        // Other PIDs (NIT, SDT, ECM, etc.) are ignored.
    }

    /// `section` spans table_id … CRC_32 inclusive.
    fn parse_pat(&mut self, section: &[u8]) {
        // table_id(1) | flags + section_length(2) | transport_stream_id(2)
        // | version(1) | section_number(1) | last_section_number(1)
        // then 4-byte program entries, then CRC_32(4).
        let Some(entries) = section.get(8..section.len().saturating_sub(4)) else {
            return;
        };
        // Single-program streams: the first non-zero program_number
        // (zero is the network PID) names the PMT.
        if let Some(entry) = entries
            .chunks_exact(4)
            .find(|e| u16::from_be_bytes([e[0], e[1]]) != 0)
        {
            self.pmt_pid = Some((u16::from(entry[2] & 0x1f) << 8) | u16::from(entry[3]));
        }
    }

    /// `section` spans table_id … CRC_32 inclusive.
    fn parse_pmt(&mut self, section: &[u8]) {
        // table_id(1) | flags + section_length(2) | program_number(2)
        // | version(1) | section_number(1) | last_section_number(1)
        // | PCR_PID(2) | program_info_length(2) | descriptors
        // | ES loop | CRC_32(4)
        if section.len() < 16 {
            return;
        }
        let program_info_length = (usize::from(section[10] & 0x0f) << 8) | usize::from(section[11]);
        let es_end = section.len() - 4;
        let mut i = 12 + program_info_length;
        while let Some(entry) = section.get(i..i + 5).filter(|_| i + 5 <= es_end) {
            let stream_type = entry[0];
            let elementary_pid = (u16::from(entry[1] & 0x1f) << 8) | u16::from(entry[2]);
            let es_info_length = (usize::from(entry[3] & 0x0f) << 8) | usize::from(entry[4]);

            match stream_type {
                0x1B | 0x24 if self.video_pid.is_none() => {
                    self.video_pid = Some(elementary_pid);
                    self.info.video_codec = Some(if stream_type == 0x1B {
                        Codec::H264
                    } else {
                        Codec::H265
                    });
                }
                // AAC ADTS. Only the first AAC track is played; other
                // audio codecs (MP3, AC-3, LATM) have no decoder here.
                0x0F if self.audio_pid.is_none() => {
                    self.audio_pid = Some(elementary_pid);
                    self.info.audio_codec = Some(AudioCodec::Aac);
                }
                _ => {}
            }
            i += 5 + es_info_length;
        }
    }

    /// Emits the PES currently assembled for one stream, if any.
    fn emit(&mut self, is_video: bool) {
        let assembly = if is_video {
            &mut self.video_pes
        } else {
            &mut self.audio_pes
        };
        if assembly.buf.is_empty() {
            return;
        }
        let data = bytes::Bytes::from(std::mem::take(&mut assembly.buf));
        let pts = assembly.pts.take().map_or(0, ticks_to_us);
        let dts = assembly.dts.take().map_or(pts, ticks_to_us);
        assembly.in_pes = false;
        if is_video {
            if let Some(codec) = self.info.video_codec {
                let is_keyframe = contains_random_access_point(codec, &data);
                self.out.push(DemuxedPacket::Video(EncodedFrame {
                    codec,
                    data,
                    timestamp_us: pts,
                    is_keyframe,
                    duration_us: None,
                    pts_dts: (pts, dts),
                }));
            }
        } else if let Some(codec) = self.info.audio_codec {
            self.out.push(DemuxedPacket::Audio(AudioFrame {
                codec,
                data,
                timestamp_us: pts,
                sample_rate: self.info.audio_sample_rate,
                channels: self.info.audio_channels,
            }));
        }
    }

    fn collect_pes(&mut self, is_video: bool, payload: &[u8], pusi: bool) {
        let codec_is_known = if is_video {
            self.info.video_codec.is_some()
        } else {
            self.info.audio_codec.is_some()
        };
        if !codec_is_known {
            return;
        }

        if pusi {
            // PES start: emit any in-flight buffer first.
            self.emit(is_video);
            if let Some(header) = parse_pes_header(payload) {
                let pts = header.pts.map(|raw| self.clock.unwrap(raw));
                let dts = header.dts.map(|raw| self.clock.unwrap(raw));
                let assembly = if is_video {
                    &mut self.video_pes
                } else {
                    &mut self.audio_pes
                };
                assembly.pts = pts;
                assembly.dts = dts;
                assembly.in_pes = true;
                if let Some(body) = payload.get(header.payload_offset..) {
                    assembly.buf.extend_from_slice(body);
                }
            }
        } else {
            let assembly = if is_video {
                &mut self.video_pes
            } else {
                &mut self.audio_pes
            };
            // Continuation of a PES whose start we never saw (joined
            // mid-PES, corrupt header) carries no timestamp; drop it.
            if assembly.in_pes {
                assembly.buf.extend_from_slice(payload);
            }
        }

        // If the upstream PES is audio and we don't have codec
        // params yet, sniff them from the first ADTS header
        // available in the assembled buffer.
        if !is_video
            && self.info.audio_sample_rate == 0
            && let Some((rate, channels)) = parse_adts_header(&self.audio_pes.buf)
        {
            self.info.audio_sample_rate = rate;
            self.info.audio_channels = channels;
        }
    }
}

struct PesHeader {
    /// Raw 33-bit PTS / DTS, 90 kHz.
    pts: Option<i64>,
    dts: Option<i64>,
    payload_offset: usize,
}

fn parse_pes_header(payload: &[u8]) -> Option<PesHeader> {
    if payload.len() < 9 || payload[..3] != [0, 0, 1] {
        // Not a PES start code; corrupt packet.
        return None;
    }
    let flags = payload[7];
    let header_data_length = usize::from(payload[8]);
    let pts = if flags & 0x80 != 0 {
        payload.get(9..14).map(parse_timestamp)
    } else {
        None
    };
    // DTS is only present together with PTS (PTS_DTS_flags = '11').
    let dts = if flags & 0xC0 == 0xC0 {
        payload.get(14..19).map(parse_timestamp)
    } else {
        None
    };
    Some(PesHeader {
        pts,
        dts,
        payload_offset: 9 + header_data_length,
    })
}

/// Decodes the 5-byte marker-bit-interleaved 33-bit timestamp.
fn parse_timestamp(b: &[u8]) -> i64 {
    (i64::from(b[0] >> 1) & 0x7) << 30
        | i64::from(b[1]) << 22
        | (i64::from(b[2] >> 1) & 0x7f) << 15
        | i64::from(b[3]) << 7
        | (i64::from(b[4] >> 1) & 0x7f)
}

/// True when an Annex-B access unit contains an IDR picture (H.264)
/// or an IRAP picture (H.265) — a point the decoder can start from.
fn contains_random_access_point(codec: Codec, au: &[u8]) -> bool {
    nal_headers(au).any(|header| match codec {
        Codec::H264 => header & 0x1f == 5,
        Codec::H265 => (16..=21).contains(&((header >> 1) & 0x3f)),
        Codec::Vp8 | Codec::Vp9 | Codec::Av1 => false,
    })
}

/// First byte of every NAL unit in an Annex-B byte stream.
fn nal_headers(au: &[u8]) -> impl Iterator<Item = u8> + '_ {
    au.windows(4)
        .filter_map(|w| (w[..3] == [0, 0, 1]).then_some(w[3]))
}

/// Parse the first ADTS frame header in `buf`. Returns `(sample_rate,
/// channels)` if the sync word lines up and the sample rate index is
/// in range; `None` otherwise.
fn parse_adts_header(buf: &[u8]) -> Option<(u32, u16)> {
    const RATES: [u32; 16] = [
        96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350, 0,
        0, 0,
    ];
    if buf.len() < 7 {
        return None;
    }
    // Sync word: 0xFFF (12 bits).
    if buf[0] != 0xFF || (buf[1] & 0xF0) != 0xF0 {
        return None;
    }
    // Sample rate index — 4 bits straddling byte 2 (bits 2-5).
    let rate = RATES
        .get(usize::from((buf[2] >> 2) & 0x0F))
        .copied()
        .filter(|&r| r != 0)?;
    // Channel config — 3 bits straddling bytes 2/3.
    let ch = u16::from(((buf[2] & 0x01) << 2) | ((buf[3] & 0xC0) >> 6));
    if ch == 0 {
        return None;
    }
    Some((rate, ch))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds one 188-byte TS packet carrying `payload` (stuffed with
    /// an adaptation field when shorter than 184 bytes).
    fn ts_packet(pid: u16, pusi: bool, payload: &[u8]) -> Vec<u8> {
        assert!(payload.len() <= 184);
        let mut pkt = vec![
            0x47,
            (if pusi { 0x40 } else { 0 }) | ((pid >> 8) as u8 & 0x1f),
            pid as u8,
        ];
        let stuffing = 184 - payload.len();
        if stuffing == 0 {
            pkt.push(0x10);
        } else {
            pkt.push(0x30);
            pkt.push((stuffing - 1) as u8);
            if stuffing > 1 {
                pkt.push(0x00);
                pkt.extend(std::iter::repeat_n(0xff, stuffing - 2));
            }
        }
        pkt.extend_from_slice(payload);
        assert_eq!(pkt.len(), 188);
        pkt
    }

    fn psi(table_id: u8, body: &[u8]) -> Vec<u8> {
        // body = everything after section_length up to (not incl.) CRC.
        let section_length = body.len() + 4;
        let mut s = vec![
            0x00, // pointer_field
            table_id,
            0xB0 | ((section_length >> 8) as u8 & 0x0f),
            section_length as u8,
        ];
        s.extend_from_slice(body);
        s.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef]); // CRC (unchecked)
        s
    }

    fn pat(pmt_pid: u16) -> Vec<u8> {
        psi(
            0x00,
            &[
                0x00,
                0x01,
                0xC1,
                0x00,
                0x00,
                0x00,
                0x01,
                0xE0 | (pmt_pid >> 8) as u8,
                pmt_pid as u8,
            ],
        )
    }

    fn pmt(streams: &[(u8, u16)]) -> Vec<u8> {
        let mut body = vec![0x00, 0x01, 0xC1, 0x00, 0x00, 0xE1, 0x00, 0xF0, 0x00];
        for &(st, pid) in streams {
            body.extend_from_slice(&[st, 0xE0 | (pid >> 8) as u8, pid as u8, 0xF0, 0x00]);
        }
        psi(0x02, &body)
    }

    fn timestamp_bytes(prefix: u8, ts: i64) -> [u8; 5] {
        [
            (prefix << 4) | (((ts >> 30) & 0x7) as u8) << 1 | 1,
            (ts >> 22) as u8,
            (((ts >> 15) & 0x7f) as u8) << 1 | 1,
            (ts >> 7) as u8,
            ((ts & 0x7f) as u8) << 1 | 1,
        ]
    }

    fn pes(pts: i64, dts: Option<i64>, body: &[u8]) -> Vec<u8> {
        let mut p = vec![0, 0, 1, 0xE0, 0, 0, 0x80];
        match dts {
            Some(dts) => {
                p.extend_from_slice(&[0xC0, 10]);
                p.extend_from_slice(&timestamp_bytes(3, pts));
                p.extend_from_slice(&timestamp_bytes(1, dts));
            }
            None => {
                p.extend_from_slice(&[0x80, 5]);
                p.extend_from_slice(&timestamp_bytes(2, pts));
            }
        }
        p.extend_from_slice(body);
        p
    }

    #[test]
    fn pmt_loop_stops_before_crc() {
        let mut d = TsDemuxer::new();
        d.push(&ts_packet(0, true, &pat(0x100)));
        // A single H.264 stream; the CRC bytes `de ad be ef` must not
        // be read as a sixth "stream" entry.
        d.push(&ts_packet(0x100, true, &pmt(&[(0x1B, 0x101)])));
        assert_eq!(d.video_pid, Some(0x101));
        assert_eq!(d.audio_pid, None);
    }

    #[test]
    fn pmt_spanning_two_packets_is_reassembled() {
        let mut d = TsDemuxer::new();
        d.push(&ts_packet(0, true, &pat(0x100)));
        // Many dummy private-data streams push the AAC entry past
        // the first packet.
        let mut streams: Vec<(u8, u16)> = (0..40).map(|i| (0x06, 0x200 + i)).collect();
        streams.push((0x1B, 0x101));
        streams.push((0x0F, 0x102));
        let section = pmt(&streams);
        assert!(section.len() > 184);
        d.push(&ts_packet(0x100, true, &section[..184]));
        assert_eq!(d.video_pid, None);
        d.push(&ts_packet(0x100, false, &section[184..]));
        assert_eq!(d.video_pid, Some(0x101));
        assert_eq!(d.audio_pid, Some(0x102));
    }

    #[test]
    fn first_aac_track_wins() {
        let mut d = TsDemuxer::new();
        d.push(&ts_packet(0, true, &pat(0x100)));
        d.push(&ts_packet(
            0x100,
            true,
            &pmt(&[(0x03, 0x103), (0x0F, 0x102), (0x0F, 0x104)]),
        ));
        assert_eq!(d.audio_pid, Some(0x102));
    }

    #[test]
    fn dts_and_keyframes_are_reported() {
        let mut d = TsDemuxer::new();
        d.push(&ts_packet(0, true, &pat(0x100)));
        d.push(&ts_packet(0x100, true, &pmt(&[(0x1B, 0x101)])));
        let idr = [0, 0, 0, 1, 0x65, 0x88];
        let non_idr = [0, 0, 0, 1, 0x41, 0x9a];
        d.push(&ts_packet(0x101, true, &pes(9000, Some(6000), &idr)));
        d.push(&ts_packet(0x101, true, &pes(18000, None, &non_idr)));
        let out: Vec<_> = std::mem::take(d.flush());
        let frames: Vec<_> = out
            .into_iter()
            .map(|p| match p {
                DemuxedPacket::Video(v) => v,
                DemuxedPacket::Audio(_) => panic!("unexpected audio"),
            })
            .collect();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].pts_dts, (100_000, 66_666));
        assert!(frames[0].is_keyframe);
        assert_eq!(frames[1].pts_dts, (200_000, 200_000));
        assert!(!frames[1].is_keyframe);
    }

    #[test]
    fn timestamps_unwrap_across_33_bit_boundary() {
        let mut u = TimestampUnwrapper::default();
        let near_end = TS_WRAP - 3000;
        assert_eq!(u.unwrap(near_end), near_end);
        assert_eq!(u.unwrap(3000), TS_WRAP + 3000);
        // A B-frame slightly earlier than the previous value stays in
        // the same period.
        assert_eq!(u.unwrap(1000), TS_WRAP + 1000);
        assert_eq!(u.unwrap(TS_WRAP - 100), TS_WRAP - 100);
    }

    #[test]
    fn hevc_irap_is_a_keyframe() {
        // NAL type 19 (IDR_W_RADL) → header byte 0x26.
        assert!(contains_random_access_point(
            Codec::H265,
            &[0, 0, 1, 0x26, 0x01]
        ));
        // NAL type 1 (TRAIL_R) → 0x02.
        assert!(!contains_random_access_point(
            Codec::H265,
            &[0, 0, 1, 0x02, 0x01]
        ));
    }
}
