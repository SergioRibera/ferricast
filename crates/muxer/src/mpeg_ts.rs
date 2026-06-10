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
const PID_AUDIO: u16 = 0x0101;

const STREAM_TYPE_H264: u8 = 0x1B;
const STREAM_TYPE_AAC_ADTS: u8 = 0x0F;
const PES_STREAM_ID_VIDEO: u8 = 0xE0;
const PES_STREAM_ID_AUDIO: u8 = 0xC0;

/// One AAC LC ADTS frame at 48 kHz / stereo / silence.
///
/// Older Chromecast Default Media Receiver firmwares (1st/2nd gen,
/// `md=Chromecast` in mDNS) reject HLS streams that don't carry an
/// audio track — the LOAD lands, the receiver builds a media
/// session, then immediately responds with
/// `LOAD_FAILED, idleReason=ERROR` because the demuxer expects two
/// elementary streams. Other players (VLC, hls.js, ffmpeg) don't
/// care and will play video-only HLS happily, which is why our
/// stream tests fine elsewhere but fails on the cast device.
///
/// Inject one silent AAC frame per ~21 ms (1024 samples / 48 kHz)
/// to satisfy the receiver. Total per-stream overhead: ~600 B/s.
///
/// Bytes are a hand-crafted silent AAC LC frame for stereo 48 kHz:
/// * 7-byte ADTS header (no CRC, profile=LC, sfi=3 → 48k,
///   channel_config=2, frame_length=13)
/// * 6-byte raw_data_block carrying a CPE with zero scale factors
///   and zero spectrum (= digital silence).
const SILENT_AAC_FRAME: [u8; 13] = [
    0xFF, 0xF1, 0x4C, 0x80, 0x01, 0xBF, 0xFC, 0x21, 0x10, 0x04, 0x60, 0x8C, 0x1C,
];

/// 1024 samples at 48 kHz = 1024/48000 s = 21.333 ms.
/// In MPEG-TS 90 kHz ticks: 1024 × (90000 / 48000) = 1920.
const AAC_FRAME_TICKS_90K: u64 = 1920;

/// How far behind the current video PTS the audio elementary stream
/// is allowed to drift before the muxer starts plugging the gap
/// with silent AAC frames. 45 000 ticks = ~500 ms.
///
/// External-audio mode (real PipeWire-captured AAC pushed via
/// [`MpegTs::add_audio_frame`]) leaves the audio PID alive only as
/// long as the upstream actually delivers frames. When the system
/// audio source goes idle (paused media, suspended sink) PipeWire
/// stops emitting buffers — the encoder produces nothing, the
/// segmenter pushes nothing, and the PID's elementary stream goes
/// silent in the literal "no packets at all" sense. The chromecast
/// (and any receiver that pre-buffers audio before letting video
/// play) interprets that as "stream loading" and freezes on the
/// last-seen video frame. Filling the gap with silent ADTS frames
/// keeps the PID alive and lets the receiver progress.
///
/// 500 ms is generous enough that brief gaps in real audio (encoder
/// warmup, segment-close stalls, a system audio cue between two
/// captured sounds) don't cause silent insertions that would then
/// collide with the real audio about to arrive — but short enough
/// that the receiver never enters its starvation timeout.
const SILENT_INJECT_LAG_90K: u64 = 45_000;

/// Hard cap on silent frames emitted per `add_frame` (video) call.
/// Past this we're already 8 × 21 ms = 170 ms ahead of where we
/// started; further injection wouldn't unstick anything and could
/// flood the TS body with silence on a one-off video-pts jump.
const SILENT_INJECT_MAX_PER_FRAME: u32 = 8;

/// Max ticks an incoming real-audio PTS may be behind the watermark
/// before we treat it as stale. AAC-LC at 48 kHz / 1024 samples per
/// frame is "exactly" 1920 ticks of 90 kHz, but the encoder's
/// per-frame PTS comes out of `samples_emitted * 1e6 / sample_rate`
/// truncated to integer microseconds, then the segmenter re-scales
/// by `9 / 100` — two integer truncations that drift the per-frame
/// delta between 1919 / 1920 / 1921 ticks on a 1-of-3 cycle. The
/// muxer's watermark advances by a flat +1920 per push, so without
/// tolerance every other frame lands "1 tick behind" and gets
/// dropped — chromecast audio degenerates to a robotic stutter.
/// 4 ticks (~44 µs) covers the rounding jitter with margin to
/// spare; anything more genuinely behind is still treated as stale.
const AUDIO_PTS_TOLERANCE_90K: u64 = 4;

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
    cc_audio: u8,
    /// Whether to advertise an audio stream in the PMT and inject
    /// silent AAC frames inline with video. Off by default (some
    /// receivers reject the extra ES; bandwidth is wasted on
    /// receivers that accept video-only). The chromecast pipeline
    /// flips this on when the target device's
    /// `DeviceCapabilities::requires_audio` is true.
    inject_silent_audio: bool,
    /// When `true`, the muxer expects external audio frames to be
    /// pushed via [`Self::add_audio_frame`] and advertises the audio
    /// PID in the PMT. This takes precedence over
    /// `inject_silent_audio`: when both are set, the external audio
    /// path wins and silence injection is suppressed (the live
    /// stream is the source of truth).
    has_external_audio: bool,
    /// Shared watermark for the audio elementary stream's PTS.
    /// `add_audio_frame` (real audio) advances it past each pushed
    /// frame; the silent-fallback loop inside `add_frame` (video)
    /// advances it for each silent ADTS frame it injects to bridge
    /// a gap. Either path consults it: silent injection only fires
    /// when the watermark trails the current video PTS by more
    /// than `SILENT_INJECT_LAG_90K`, and real audio whose PTS is
    /// behind the watermark is dropped (a silent frame already
    /// occupies that slot). `None` until the first video frame
    /// initialises it.
    audio_pts_90k: Option<u64>,
    out: Vec<u8>,
}

impl MpegTs {
    /// Toggle silent-audio injection. Set by the chromecast HLS
    /// pipeline when the target device is an older Chromecast
    /// (`md == "Chromecast"`) whose firmware rejects video-only
    /// HLS streams with `LOAD_FAILED`. Newer receivers (Ultra,
    /// Google TV) accept video-only and we can save ~6 KB/s by
    /// leaving this off.
    pub fn with_silent_audio(mut self, inject: bool) -> Self {
        self.inject_silent_audio = inject;
        self
    }

    /// Switch the muxer into "external audio" mode: advertise the
    /// audio PID in the PMT and stop injecting silence. The caller
    /// is then expected to push every audio frame via
    /// [`Self::add_audio_frame`]. Used by the chromecast HLS
    /// pipeline when the user opted into real-audio capture
    /// (`StreamConfig::audio = Some(_)`).
    pub fn with_external_audio(mut self, enable: bool) -> Self {
        self.has_external_audio = enable;
        self
    }

    /// Append one already-encoded audio access unit (today: ADTS-
    /// framed AAC-LC) with its 90 kHz PTS. The muxer takes care of
    /// PES packetisation, audio-PID continuity-counter bookkeeping
    /// and 188-byte TS framing.
    ///
    /// Returns `Err` if the muxer wasn't put into external-audio
    /// mode via [`Self::with_external_audio`].
    ///
    /// Audio whose PTS is *behind* the muxer's silent-injection
    /// watermark (`audio_pts_90k`) is dropped silently: the silent
    /// fallback has already filled that slot, and pushing the real
    /// frame anyway would create out-of-order PTS on the audio PID
    /// that some receivers fail to splice. This only fires when
    /// the upstream lagged the lag-threshold; in normal flow real
    /// audio's PTS always stays ahead of the watermark.
    pub fn add_audio_frame(
        &mut self,
        payload: &[u8],
        pts_90k: u64,
    ) -> Result<(), FerricastError> {
        if !self.has_external_audio {
            return Err(FerricastError::Encoder(
                "MpegTs::add_audio_frame called without with_external_audio(true)".into(),
            ));
        }
        if payload.is_empty() {
            return Ok(());
        }
        // Drop real audio that's behind where the silent-fallback
        // has already filled, but tolerate `AUDIO_PTS_TOLERANCE_90K`
        // ticks of slack — the encoder's per-frame PTS jitters
        // ±1 tick around the nominal 1920-tick AAC-LC step (see the
        // constant's docs). Without tolerance ~2 out of every 3 real
        // frames get rejected here even on a perfectly healthy
        // stream, which on the receiver sounds like a robotic
        // ratcheting stutter.
        if let Some(watermark) = self.audio_pts_90k {
            if pts_90k.saturating_add(AUDIO_PTS_TOLERANCE_90K) < watermark {
                tracing::debug!(
                    pts_90k,
                    watermark,
                    behind_90k = watermark - pts_90k,
                    "muxer: dropping real AAC frame behind silent watermark"
                );
                return Ok(());
            }
        }
        // PSI must still appear first if no video has shown up yet
        // (rare for the live screencast pipeline: audio anchors at
        // about the same wall-clock instant as the first IDR, but
        // a few audio chunks may arrive *just* before it). Emitting
        // PAT/PMT here on demand keeps the segment self-contained.
        if !self.psi_emitted {
            self.emit_pat();
            self.emit_pmt();
            self.psi_emitted = true;
        }
        let pes = build_audio_pes(payload, pts_90k);
        // PCR stays exclusive to PID_VIDEO (the PMT's declared
        // `PCR_PID`). Emitting PCR on the audio PID confuses the
        // receiver's STC PLL and produces glitchy A/V on chromecast.
        write_pes_to_ts(
            &mut self.out,
            PID_AUDIO,
            &mut self.cc_audio,
            &pes,
            /* random_access_indicator */ false,
            None,
        );
        // Advance the watermark past this frame so any subsequent
        // silent-injection check correctly bypasses the slot we
        // just filled, and any later real-audio frame with the
        // same or lower PTS is recognised as stale. `max` with the
        // existing watermark guards against rollback when a real
        // frame was admitted via `AUDIO_PTS_TOLERANCE_90K` after
        // the silent-fill loop had already advanced past it.
        let advanced = pts_90k.saturating_add(AAC_FRAME_TICKS_90K);
        let prev = self.audio_pts_90k.unwrap_or(0);
        self.audio_pts_90k = Some(advanced.max(prev));
        Ok(())
    }

    /// True when the muxer is currently advertising an audio PID.
    /// Used by the HLS segmenter to decide whether silent-AAC
    /// injection should be suppressed.
    pub fn audio_advertised(&self) -> bool {
        self.inject_silent_audio || self.has_external_audio
    }
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
            tracing::warn!("Empty frame, skipping");
            return Ok(());
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
            Some(pcr),
        );

        // Silent-AAC fallback. Runs whenever the audio PID is
        // advertised (either old-Chromecast silent-only mode OR
        // external-audio mode) and the audio elementary stream has
        // drifted more than `SILENT_INJECT_LAG_90K` (~500 ms) behind
        // the video PTS we just emitted.
        //
        // Why coexist with external audio: when the user's system
        // audio source goes idle (paused media, suspended sink),
        // PipeWire stops delivering monitor buffers → encoder
        // produces no AAC → muxer's `add_audio_frame` never gets
        // called → the audio PID in the segment is *advertised but
        // empty*. The chromecast (and most receivers that prebuffer
        // audio before starting video playback) interpret that as
        // "loading" and freeze on the last-seen video frame. Filling
        // the gap with silent ADTS frames keeps the PID alive.
        //
        // The 500 ms lag threshold is intentional: real audio coming
        // through gets a comfortable window to land before the
        // muxer starts plugging slots silent, so we don't end up
        // emitting silent right next to real audio with adjacent
        // PTSes. `audio_pts_90k` is the shared watermark — both
        // `add_audio_frame` and this loop advance it, so the
        // two paths never collide.
        let _ = pcr; // pcr binding kept alive for video write above
        if self.inject_silent_audio || self.has_external_audio {
            let lag_threshold = pts_90k.saturating_sub(SILENT_INJECT_LAG_90K);
            let start_apts = self
                .audio_pts_90k
                .unwrap_or(lag_threshold);
            if start_apts < lag_threshold {
                let mut apts = start_apts;
                let mut frames_emitted = 0u32;
                while apts < lag_threshold
                    && frames_emitted < SILENT_INJECT_MAX_PER_FRAME
                {
                    let aac_pes = build_audio_pes(&SILENT_AAC_FRAME, apts);
                    write_pes_to_ts(
                        &mut self.out,
                        PID_AUDIO,
                        &mut self.cc_audio,
                        &aac_pes,
                        /* random_access_indicator */ false,
                        None,
                    );
                    apts = apts.saturating_add(AAC_FRAME_TICKS_90K);
                    frames_emitted += 1;
                }
                if frames_emitted > 0 {
                    tracing::debug!(
                        frames_emitted,
                        from_pts_90k = start_apts,
                        to_pts_90k = apts,
                        video_pts_90k = pts_90k,
                        "muxer: injected silent AAC to bridge audio gap"
                    );
                }
                self.audio_pts_90k = Some(apts);
            } else if self.audio_pts_90k.is_none() {
                // Anchor the watermark even when no silent fill
                // happens this frame — keeps the first real audio
                // arrival from accidentally fighting an unset
                // counter.
                self.audio_pts_90k = Some(start_apts);
            }
        }

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
        debug_assert!(
            self.out.is_empty(),
            "drain() must be called between segments"
        );
    }

    fn emit_pat(&mut self) {
        let section = build_pat_section();
        write_psi_packet(&mut self.out, PID_PAT, &mut self.cc_pat, &section);
    }

    fn emit_pmt(&mut self) {
        // PMT advertises audio whenever EITHER silent injection or
        // external-audio mode is on — both share the same PID and
        // both produce AAC ADTS bytes, so the receiver doesn't need
        // to distinguish.
        let with_audio = self.inject_silent_audio || self.has_external_audio;
        let section = build_pmt_section(with_audio);
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

fn build_pmt_section(with_audio: bool) -> Vec<u8> {
    // section_length covers prog_num(2) + flags(1) + sec_num(1) +
    // last_sec_num(1) + PCR_PID(2) + program_info_len(2) +
    // stream_loop + CRC(4).
    // stream_loop is 5 bytes per stream (type + EPID + ES_info_len).
    let stream_count = if with_audio { 2 } else { 1 };
    let section_length: u16 = (13 + 5 * stream_count) as u16;
    let mut s = Vec::with_capacity(4 + section_length as usize);
    s.push(0x02); // table_id = PMT
    s.extend_from_slice(&(0xB000u16 | section_length).to_be_bytes());
    s.extend_from_slice(&1u16.to_be_bytes()); // program_number
    s.push(0xC1); // reserved=11, version=0, current_next=1
    s.push(0x00); // section_number
    s.push(0x00); // last_section_number
    s.extend_from_slice(&(0xE000u16 | PID_VIDEO).to_be_bytes()); // PCR_PID
    s.extend_from_slice(&0xF000u16.to_be_bytes()); // program_info_length=0
    // Stream loop: H.264 always; AAC only when the caller asked
    // for silent-audio injection (older Chromecasts that reject
    // video-only HLS — see `MpegTs::with_silent_audio`).
    s.push(STREAM_TYPE_H264);
    s.extend_from_slice(&(0xE000u16 | PID_VIDEO).to_be_bytes()); // elementary_PID
    s.extend_from_slice(&0xF000u16.to_be_bytes()); // ES_info_length=0
    if with_audio {
        s.push(STREAM_TYPE_AAC_ADTS);
        s.extend_from_slice(&(0xE000u16 | PID_AUDIO).to_be_bytes()); // elementary_PID
        s.extend_from_slice(&0xF000u16.to_be_bytes()); // ES_info_length=0
    }
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

/// Audio PES carrying one (or more) ADTS-framed AAC access units.
/// Unlike video, audio PES MUST carry a non-zero `PES_packet_length`
/// — `0` is only legal for video. PTS is included; DTS is omitted
/// (audio has no reordering, so DTS == PTS implicitly).
fn build_audio_pes(payload: &[u8], pts_90k: u64) -> Vec<u8> {
    // header: prefix(3) + stream_id(1) + length(2) + flags(2) +
    //         header_data_length(1) + PTS(5) = 14 bytes header, then payload.
    let pes_body_len = 3 /* flags+hdr_len */ + 5 /* PTS */ + payload.len();
    let mut pes = Vec::with_capacity(6 + pes_body_len);
    pes.extend_from_slice(&[0x00, 0x00, 0x01, PES_STREAM_ID_AUDIO]);
    pes.extend_from_slice(&(pes_body_len as u16).to_be_bytes());
    pes.push(0x84); // '10' marker + alignment_indicator=1
    pes.push(0x80); // PTS_DTS_flags=10 (PTS only)
    pes.push(0x05); // PES_header_data_length
    pes.extend_from_slice(&encode_ts(0b0010, pts_90k));
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
/// `pcr_90k` controls whether the first packet carries a Program
/// Clock Reference in its adaptation field. The PMT declares
/// `PCR_PID = PID_VIDEO`, so only the video PID must call this with
/// `Some(pcr)`; audio (and any other elementary stream) passes
/// `None` to keep PCR exclusive to the video timeline. Without this
/// gate, both PIDs emit PCRs at different rates and the receiver's
/// STC PLL (which uses `PCR_PID` to lock its system-time clock) gets
/// confused by the extra PCRs on a non-PCR PID — manifesting as
/// glitchy audio + stuttering video on chromecast even though every
/// frame's PTS is correct in isolation.
///
/// Layout:
/// * **First packet**: PUSI=1. AF carries PCR (video) or RAI-only
///   (keyframe without PCR) or no AF / stuffing AF (audio, no PCR).
///   Stuffing keeps the packet at exactly 188 bytes when the PES is
///   small.
/// * **Middle packets**: AFC=01, full 184-byte payload.
/// * **Last packet**: stuffing-only adaptation field if the
///   remaining payload doesn't fill 184 bytes; otherwise full
///   payload like a middle packet.
fn write_pes_to_ts(
    out: &mut Vec<u8>,
    pid: u16,
    cc: &mut u8,
    pes: &[u8],
    keyframe: bool,
    pcr_90k: Option<u64>,
) {
    const PAYLOAD_SPACE_NO_AF: usize = TS_PACKET_SIZE - 4; // 184
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
        out.push(0); // afc + cc patched below

        let remaining = pes.len() - offset;
        let payload_len: usize;
        let af_present: bool;

        // Decide what the first packet's adaptation field needs:
        // * PCR (video PID only) → 6-byte PCR field, flags byte.
        //   Optional RAI flag piggybacks on the same flags byte.
        // * RAI without PCR (rare: audio keyframe-equivalent — we
        //   don't surface that for AAC today, so this branch is
        //   defensive) → 1-byte flags only.
        // * Neither → no AF unless we need stuffing to round the
        //   packet out to 188 bytes.
        let first_af_body = if first {
            if pcr_90k.is_some() {
                7 // flags(1) + PCR(6)
            } else if keyframe {
                1 // flags(1)
            } else {
                0
            }
        } else {
            0
        };

        if first && first_af_body > 0 {
            // First packet with PCR and/or RAI in the AF.
            let max_payload = PAYLOAD_SPACE_NO_AF - 1 /*af_len byte*/ - first_af_body;
            let want = remaining.min(max_payload);
            let stuffing = max_payload - want;
            let af_len = first_af_body + stuffing;
            out.push(af_len as u8);
            let mut flags = 0u8;
            if keyframe {
                flags |= 0x40; // random_access_indicator
            }
            if pcr_90k.is_some() {
                flags |= 0x10; // PCR_flag
            }
            out.push(flags);
            if let Some(pcr) = pcr_90k {
                out.extend_from_slice(&encode_pcr(pcr));
            }
            for _ in 0..stuffing {
                out.push(0xFF);
            }
            payload_len = want;
            af_present = true;
        } else if first && remaining >= PAYLOAD_SPACE_NO_AF {
            // First packet, no PCR / RAI, payload fills the slot —
            // ship it without an adaptation field.
            payload_len = PAYLOAD_SPACE_NO_AF;
            af_present = false;
        } else if first {
            // First-and-only packet of a small PES (≤ 183 bytes,
            // no PCR/RAI) — pad with a stuffing-only AF so the TS
            // packet still hits 188 bytes.
            let af_total = PAYLOAD_SPACE_NO_AF - remaining; // includes af_len byte
            let af_len = af_total - 1;
            out.push(af_len as u8);
            if af_len >= 1 {
                out.push(0); // flags = 0
                for _ in 0..(af_len - 1) {
                    out.push(0xFF);
                }
            }
            payload_len = remaining;
            af_present = true;
        } else if remaining < PAYLOAD_SPACE_NO_AF {
            // Last packet: stuff to fill 188 bytes.
            let af_total = PAYLOAD_SPACE_NO_AF - remaining;
            let af_len = af_total - 1;
            out.push(af_len as u8);
            if af_len >= 1 {
                out.push(0);
                for _ in 0..(af_len - 1) {
                    out.push(0xFF);
                }
            }
            payload_len = remaining;
            af_present = true;
        } else {
            // Middle packet, full payload, no AF.
            payload_len = PAYLOAD_SPACE_NO_AF;
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
        for with_audio in [false, true] {
            let s = build_pmt_section(with_audio);
            let body = &s[..s.len() - 4];
            let appended = u32::from_be_bytes(s[s.len() - 4..].try_into().unwrap());
            assert_eq!(crc32_mpeg2(body), appended, "with_audio={with_audio}");
        }
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
        write_pes_to_ts(&mut out, PID_VIDEO, &mut cc, &pes, true, Some(89_800));
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
        write_pes_to_ts(&mut out, PID_VIDEO, &mut cc, &pes, true, Some(0));
        assert_eq!(out.len(), TS_PACKET_SIZE);
    }

    #[test]
    fn audio_pes_carries_no_pcr() {
        // First TS packet of an audio PES (PCR=None) must NOT set
        // the PCR_flag bit in its adaptation field, regardless of
        // whether stuffing forces an AF to exist.
        let mut out = Vec::new();
        let mut cc = 0;
        let pes = build_audio_pes(&[0x55; 32], 90_000);
        write_pes_to_ts(&mut out, PID_AUDIO, &mut cc, &pes, false, None);
        assert_eq!(out.len(), TS_PACKET_SIZE);
        let afc = (out[3] >> 4) & 0b11;
        // AF expected (the PES is short, so stuffing AF runs).
        assert!(afc == 0b11 || afc == 0b10, "afc={afc}");
        if afc == 0b11 || afc == 0b10 {
            let af_len = out[4] as usize;
            if af_len >= 1 {
                let flags = out[5];
                assert_eq!(
                    flags & 0x10, 0,
                    "audio PES first packet must NOT set PCR_flag"
                );
            }
        }
    }

    #[test]
    fn audio_keyframe_emits_rai_without_pcr() {
        // Defensive: when audio is marked random_access with no
        // PCR, the AF should set RAI=1 but PCR_flag=0.
        let mut out = Vec::new();
        let mut cc = 0;
        let pes = build_audio_pes(&[0xAA; 64], 90_000);
        write_pes_to_ts(&mut out, PID_AUDIO, &mut cc, &pes, true, None);
        assert_eq!(out.len(), TS_PACKET_SIZE);
        let afc = (out[3] >> 4) & 0b11;
        assert!(afc == 0b11 || afc == 0b10);
        let flags = out[5];
        assert_eq!(flags & 0x40, 0x40, "RAI should be set");
        assert_eq!(flags & 0x10, 0, "PCR_flag must remain clear");
    }
}
