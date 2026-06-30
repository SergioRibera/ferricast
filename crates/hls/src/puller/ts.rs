//! Minimal MPEG-TS demuxer for HLS pulled segments.
//!
//! The full TS spec (ISO/IEC 13818-1) is huge; receivers only need
//! a small slice of it because HLS-muxed streams hit a narrow
//! happy path:
//!
//! - 188-byte packets, sync byte `0x47`.
//! - One PAT (PID 0) per segment, pointing to one PMT PID.
//! - One PMT listing a video ES (stream_type `0x1B` = H.264) and an
//!   audio ES (stream_type `0x0F` = AAC ADTS).
//! - Each ES carries PES packets; PES headers carry PTS (and
//!   sometimes DTS) at 90 kHz.
//!
//! No support for: encrypted segments, multi-program transport
//! streams, splice/segmentation, PSI versioning beyond v0. The Cast
//! sender app never emits anything outside this happy path — it
//! always hands us a single-program HLS variant.

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

#[derive(Debug, Default)]
pub struct TsDemuxer {
    pmt_pid: Option<u16>,
    video_pid: Option<u16>,
    audio_pid: Option<u16>,
    info: TsStreamInfo,

    video_pes: PesAssembly,
    audio_pes: PesAssembly,
    out: Vec<DemuxedPacket>,
}

#[derive(Debug, Default)]
struct PesAssembly {
    buf: Vec<u8>,
    /// Last decoded PTS in microseconds (already converted from the
    /// 90 kHz domain). `None` until the first PES with a PTS lands.
    pts_us: Option<u64>,
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
        while i + 188 <= chunk.len() {
            if chunk[i] != 0x47 {
                // Resync: scan forward until the next sync byte.
                // HLS segments should never trigger this in practice,
                // but partial / corrupted transfers can.
                i += 1;
                continue;
            }
            self.process_packet(&chunk[i..i + 188]);
            i += 188;
        }
        // Drain emitted packets into the caller's return slot.
        &mut self.out
    }

    /// Flush any PES that hasn't been emitted yet — the last frame
    /// of a segment usually has no follower to trigger the emit, so
    /// we flush on segment boundary or EOS.
    pub fn flush(&mut self) -> &mut Vec<DemuxedPacket> {
        if !self.video_pes.buf.is_empty() {
            if let Some(codec) = self.info.video_codec {
                let pts = self.video_pes.pts_us.unwrap_or(0);
                self.out.push(DemuxedPacket::Video(EncodedFrame {
                    codec,
                    data: bytes::Bytes::from(std::mem::take(&mut self.video_pes.buf)),
                    timestamp_us: pts,
                    is_keyframe: false,
                    duration_us: None,
                    pts_dts: (pts, pts),
                }));
            }
        }
        if !self.audio_pes.buf.is_empty() {
            if let Some(codec) = self.info.audio_codec {
                let pts = self.audio_pes.pts_us.unwrap_or(0);
                self.out.push(DemuxedPacket::Audio(AudioFrame {
                    codec,
                    data: bytes::Bytes::from(std::mem::take(&mut self.audio_pes.buf)),
                    timestamp_us: pts,
                    sample_rate: self.info.audio_sample_rate,
                    channels: self.info.audio_channels,
                }));
            }
        }
        &mut self.out
    }

    fn process_packet(&mut self, pkt: &[u8]) {
        // TS header: sync(1) | tei(1) pusi(1) tpr(1) pid_hi(5) | pid_lo(8) | scrambling(2) afc(2) cc(4)
        let pusi = (pkt[1] & 0x40) != 0; // payload_unit_start_indicator
        let pid = (((pkt[1] & 0x1f) as u16) << 8) | pkt[2] as u16;
        let afc = (pkt[3] >> 4) & 0x03;
        let mut payload_start = 4;
        if afc == 2 {
            // Adaptation field only, no payload.
            return;
        }
        if afc == 3 {
            // Adaptation field followed by payload — skip its length
            // byte + the field itself.
            let af_len = pkt[4] as usize;
            payload_start = 5 + af_len;
            if payload_start >= 188 {
                return;
            }
        }
        let payload = &pkt[payload_start..];

        if pid == 0 {
            self.parse_pat(payload, pusi);
        } else if Some(pid) == self.pmt_pid {
            self.parse_pmt(payload, pusi);
        } else if Some(pid) == self.video_pid {
            self.collect_pes(true, payload, pusi);
        } else if Some(pid) == self.audio_pid {
            self.collect_pes(false, payload, pusi);
        }
        // Other PIDs (NIT, SDT, ECM, etc.) ignored — HLS doesn't
        // use them in any sender we care about.
    }

    fn parse_pat(&mut self, payload: &[u8], pusi: bool) {
        if !pusi || payload.is_empty() {
            return;
        }
        // PAT starts with a pointer_field byte after the PUSI.
        let pointer = payload[0] as usize;
        if 1 + pointer + 12 > payload.len() {
            return;
        }
        let section = &payload[1 + pointer..];
        // table_id(1) | section_syntax_indicator(1)'0'(1) reserved(2)
        //   section_length(12) | transport_stream_id(16) | reserved(2)
        //   version(5) cni(1) | section_number(8) | last_section_number(8)
        // then program loops: program_number(16) | reserved(3) PID(13)
        // For HLS there's exactly one program; pick the first
        // non-zero `program_number`.
        let mut i = 8;
        while i + 4 <= section.len() {
            let program_num = ((section[i] as u16) << 8) | section[i + 1] as u16;
            let pid = (((section[i + 2] & 0x1f) as u16) << 8) | section[i + 3] as u16;
            if program_num != 0 {
                self.pmt_pid = Some(pid);
                return;
            }
            i += 4;
        }
    }

    fn parse_pmt(&mut self, payload: &[u8], pusi: bool) {
        if !pusi || payload.is_empty() {
            return;
        }
        let pointer = payload[0] as usize;
        if 1 + pointer + 12 > payload.len() {
            return;
        }
        let section = &payload[1 + pointer..];
        // table_id(1) | ssi(1)'0'(1) reserved(2) section_length(12)
        //   program_number(16) | reserved(2) version(5) cni(1)
        //   section_number(8) | last_section_number(8)
        //   reserved(3) PCR_PID(13) | reserved(4) program_info_length(12)
        if section.len() < 12 {
            return;
        }
        let program_info_length = (((section[10] & 0x0f) as usize) << 8) | section[11] as usize;
        let mut i = 12 + program_info_length;
        while i + 5 <= section.len() {
            let stream_type = section[i];
            let elementary_pid = (((section[i + 1] & 0x1f) as u16) << 8) | section[i + 2] as u16;
            let es_info_length =
                (((section[i + 3] & 0x0f) as usize) << 8) | section[i + 4] as usize;

            match stream_type {
                0x1B => {
                    // H.264 / AVC.
                    self.video_pid = Some(elementary_pid);
                    self.info.video_codec = Some(Codec::H264);
                }
                0x24 => {
                    // H.265 / HEVC.
                    self.video_pid = Some(elementary_pid);
                    self.info.video_codec = Some(Codec::H265);
                }
                0x0F => {
                    // AAC ADTS.
                    self.audio_pid = Some(elementary_pid);
                    self.info.audio_codec = Some(AudioCodec::Aac);
                }
                _ => {}
            }
            i += 5 + es_info_length;
        }
    }

    fn collect_pes(&mut self, is_video: bool, payload: &[u8], pusi: bool) {
        let (assembly, codec_is_known) = if is_video {
            (&mut self.video_pes, self.info.video_codec.is_some())
        } else {
            (&mut self.audio_pes, self.info.audio_codec.is_some())
        };
        if !codec_is_known {
            return;
        }

        if pusi {
            // PES start: emit any in-flight buffer first.
            let prev = std::mem::take(&mut assembly.buf);
            let prev_pts = assembly.pts_us;
            if !prev.is_empty() {
                let pts = prev_pts.unwrap_or(0);
                if is_video {
                    if let Some(codec) = self.info.video_codec {
                        self.out.push(DemuxedPacket::Video(EncodedFrame {
                            codec,
                            data: bytes::Bytes::from(prev),
                            timestamp_us: pts,
                            // We don't peek for NAL type here; the
                            // decoder treats every input as untyped
                            // and produces output when it has enough.
                            is_keyframe: false,
                            duration_us: None,
                            pts_dts: (pts, pts),
                        }));
                    }
                } else if let Some(codec) = self.info.audio_codec {
                    self.out.push(DemuxedPacket::Audio(AudioFrame {
                        codec,
                        data: bytes::Bytes::from(prev),
                        timestamp_us: pts,
                        sample_rate: self.info.audio_sample_rate,
                        channels: self.info.audio_channels,
                    }));
                }
            }

            // Parse the new PES header to extract PTS + payload offset.
            if payload.len() < 9 {
                return;
            }
            if payload[0] != 0 || payload[1] != 0 || payload[2] != 1 {
                // Not a PES start code; corrupt packet, skip.
                return;
            }
            let pes_header_data_length = payload[8] as usize;
            let flags = payload[7];
            let mut pts_us = None;
            if flags & 0x80 != 0 && payload.len() >= 14 {
                // PTS present.
                let p0 = payload[9] as u64;
                let p1 = payload[10] as u64;
                let p2 = payload[11] as u64;
                let p3 = payload[12] as u64;
                let p4 = payload[13] as u64;
                let pts_90khz = ((p0 >> 1) & 0x7) << 30
                    | (p1 << 22)
                    | ((p2 >> 1) & 0x7f) << 15
                    | (p3 << 7)
                    | ((p4 >> 1) & 0x7f);
                pts_us = Some(pts_90khz * 100 / 9);
            }
            assembly.pts_us = pts_us;

            // PES payload starts after the optional headers.
            let payload_offset = 9 + pes_header_data_length;
            if payload_offset < payload.len() {
                assembly.buf.extend_from_slice(&payload[payload_offset..]);
            }
        } else {
            // Continuation: append straight to current PES buffer.
            assembly.buf.extend_from_slice(payload);
        }

        // If the upstream PES is audio and we don't have codec
        // params yet, sniff them from the first ADTS header
        // available in the assembled buffer.
        if !is_video && self.info.audio_sample_rate == 0 && assembly.buf.len() >= 7 {
            if let Some((rate, channels)) = parse_adts_header(&assembly.buf) {
                self.info.audio_sample_rate = rate;
                self.info.audio_channels = channels;
            }
        }
    }
}

/// Parse the first ADTS frame header in `buf`. Returns `(sample_rate,
/// channels)` if the sync word lines up and the sample rate index is
/// in range; `None` otherwise.
fn parse_adts_header(buf: &[u8]) -> Option<(u32, u16)> {
    if buf.len() < 7 {
        return None;
    }
    // Sync word: 0xFFF (12 bits).
    if buf[0] != 0xFF || (buf[1] & 0xF0) != 0xF0 {
        return None;
    }
    // Sample rate index — 4 bits straddling byte 2 (bits 2-5).
    let sr_idx = ((buf[2] >> 2) & 0x0F) as usize;
    const RATES: [u32; 16] = [
        96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350, 0,
        0, 0,
    ];
    let rate = RATES.get(sr_idx).copied()?;
    if rate == 0 {
        return None;
    }
    // Channel config — 3 bits straddling bytes 2/3.
    let ch = (((buf[2] & 0x01) << 2) | ((buf[3] & 0xC0) >> 6)) as u16;
    if ch == 0 {
        return None;
    }
    Some((rate, ch))
}
