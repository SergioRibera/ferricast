//! Round-trip the muxer through a minimal TS demuxer to validate
//! every byte. If the structure here parses cleanly, ffmpeg should
//! also accept the stream — and any drift between this and ffmpeg's
//! demuxer surfaces immediately as a failing assertion.

use bytes::Bytes;
use ferricast_core::{Codec, EncodedFrame};
use ferricast_muxer::Muxer;
use ferricast_muxer::mpeg_ts::MpegTs;

fn synthetic_frame(idx: u64, keyframe: bool, payload_len: usize) -> EncodedFrame {
    let mut data = Vec::new();
    data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
    data.push(if keyframe { 0x65 } else { 0x41 });
    data.extend_from_slice(
        &(0..payload_len)
            .map(|n| (((n.wrapping_mul(31) ^ (idx as usize)) & 0xFF) | 0x10) as u8)
            .collect::<Vec<u8>>(),
    );
    EncodedFrame {
        codec: Codec::H264,
        data: Bytes::from(data),
        timestamp_us: idx * 33_333,
        is_keyframe: keyframe,
        duration_us: None,
        pts_dts: (0, 0),
    }
}

const TS: usize = 188;

#[derive(Default)]
struct PesStream {
    bytes: Vec<u8>,
    started: bool,
}

#[derive(Default)]
struct Demux {
    pat_seen: bool,
    pmt_seen: bool,
    pmt_pid: Option<u16>,
    video_pid: Option<u16>,
    expected_cc: std::collections::HashMap<u16, u8>,
    pes: PesStream,
    pes_units: Vec<Vec<u8>>,
}

fn demux(bytes: &[u8]) -> Demux {
    assert_eq!(bytes.len() % TS, 0, "stream length must be packet-aligned");
    let mut d = Demux::default();

    for off in (0..bytes.len()).step_by(TS) {
        let pkt = &bytes[off..off + TS];
        assert_eq!(pkt[0], 0x47, "missing sync at {off}");
        let pusi = pkt[1] & 0x40 != 0;
        let pid = u16::from_be_bytes([pkt[1] & 0x1F, pkt[2]]);
        let afc = (pkt[3] >> 4) & 0x03;
        let cc = pkt[3] & 0x0F;

        // Continuity counter: must increment by 1 (mod 16) for any
        // packet with payload (AFC = 01 or 11) on the same PID.
        if afc == 0b01 || afc == 0b11 {
            if let Some(prev) = d.expected_cc.get(&pid) {
                assert_eq!(
                    cc,
                    *prev,
                    "CC discontinuity on PID {pid:#x} at offset {off}"
                );
            }
            d.expected_cc.insert(pid, (cc + 1) & 0x0F);
        }

        let mut idx = 4usize;
        if afc == 0b10 || afc == 0b11 {
            let af_len = pkt[idx] as usize;
            idx += 1;
            assert!(idx + af_len <= TS, "AF overflows packet at {off}");
            idx += af_len;
        }
        if afc == 0b00 || afc == 0b10 {
            continue; // no payload
        }

        let payload = &pkt[idx..];

        if pid == 0 {
            // PAT
            if pusi {
                let pf = payload[0] as usize;
                let section = &payload[1 + pf..];
                assert_eq!(section[0], 0x00, "PAT table_id");
                let section_len = (((section[1] as usize) & 0x0F) << 8) | section[2] as usize;
                let body = &section[3..3 + section_len - 4];
                let crc = u32::from_be_bytes([
                    section[3 + section_len - 4],
                    section[3 + section_len - 3],
                    section[3 + section_len - 2],
                    section[3 + section_len - 1],
                ]);
                let header = &section[..3];
                let mut crc_input = Vec::new();
                crc_input.extend_from_slice(header);
                crc_input.extend_from_slice(body);
                assert_eq!(crc32_mpeg2(&crc_input), crc, "PAT CRC");
                // PAT body layout: TSID(2) flags(1) sec(1) last_sec(1)
                // program_number(2) reserved+program_map_PID(2).
                let pmt_pid = u16::from_be_bytes([body[7] & 0x1F, body[8]]);
                d.pmt_pid = Some(pmt_pid);
                d.pat_seen = true;
            }
        } else if Some(pid) == d.pmt_pid {
            if pusi {
                let pf = payload[0] as usize;
                let section = &payload[1 + pf..];
                assert_eq!(section[0], 0x02, "PMT table_id");
                let section_len = (((section[1] as usize) & 0x0F) << 8) | section[2] as usize;
                let body = &section[3..3 + section_len - 4];
                let crc = u32::from_be_bytes([
                    section[3 + section_len - 4],
                    section[3 + section_len - 3],
                    section[3 + section_len - 2],
                    section[3 + section_len - 1],
                ]);
                let header = &section[..3];
                let mut crc_input = Vec::new();
                crc_input.extend_from_slice(header);
                crc_input.extend_from_slice(body);
                assert_eq!(crc32_mpeg2(&crc_input), crc, "PMT CRC");
                // body: prog_num(2) + flags(1) + sec(1) + last_sec(1)
                //       + PCR_PID(2) + prog_info_len(2) + streams + CRC
                let prog_info_len =
                    (((body[7] as usize) & 0x0F) << 8) | body[8] as usize;
                let streams_start = 9 + prog_info_len;
                let stream_type = body[streams_start];
                assert_eq!(stream_type, 0x1B, "expected H.264 (0x1B)");
                let video_pid = u16::from_be_bytes([
                    body[streams_start + 1] & 0x1F,
                    body[streams_start + 2],
                ]);
                d.video_pid = Some(video_pid);
                d.pmt_seen = true;
            }
        } else if Some(pid) == d.video_pid {
            if pusi && d.pes.started {
                d.pes_units.push(std::mem::take(&mut d.pes.bytes));
            }
            if pusi {
                d.pes.started = true;
            }
            if d.pes.started {
                d.pes.bytes.extend_from_slice(payload);
            }
        }
    }

    if d.pes.started {
        d.pes_units.push(std::mem::take(&mut d.pes.bytes));
    }
    d
}

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

/// Decode 33-bit timestamp from a 5-byte PES PTS/DTS field.
fn decode_ts(b: &[u8]) -> u64 {
    assert!(b[0] & 0x01 == 0x01 && b[2] & 0x01 == 0x01 && b[4] & 0x01 == 0x01);
    ((b[0] as u64 & 0x0E) << 29)
        | ((b[1] as u64) << 22)
        | ((b[2] as u64 & 0xFE) << 14)
        | ((b[3] as u64) << 7)
        | ((b[4] as u64 & 0xFE) >> 1)
}

#[test]
fn muxer_output_demuxes_cleanly() {
    let mut muxer = MpegTs::default();
    let parameter_sets = vec![
        0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0xC0, 0x1E, 0x96, 0x54, 0x05, 0x01, 0x6C, 0x80, // SPS
        0x00, 0x00, 0x00, 0x01, 0x68, 0xCE, 0x3C, 0x80, // PPS
        0x00, 0x00, 0x00, 0x01, 0x06, 0x05, 0x10, // SEI (x264-style user data prefix)
        0xDC, 0x45, 0xE9, 0xBD, 0xE6, 0xD9, 0x48, 0xB7, 0x96, 0x2C, 0xD8, 0x20, 0xD9, 0x23, 0xEE,
        0xEF, 0x80,
    ];
    muxer.config(parameter_sets.clone()).unwrap();

    // Realistic GOP: 1 IDR (50 KB) + 9 P-frames (5–15 KB each) + 1 IDR.
    let mut frames = Vec::new();
    for i in 0..11u64 {
        let kf = i == 0 || i == 10;
        let mut data = Vec::new();
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        data.push(if kf { 0x65 } else { 0x41 });
        let payload_len = if kf { 50_000 } else { 5_000 + (i as usize) * 1_000 };
        // Pseudo-random byte pattern; the demuxer just round-trips it.
        // Avoid 0x00 0x00 0x00 0x01 sequences that would be parsed as
        // start codes — H.264 uses emulation-prevention bytes for that
        // but since we never decode the slice we just dodge them here.
        data.extend_from_slice(
            &(0..payload_len)
                .map(|n| (((n.wrapping_mul(31) ^ (i as usize)) & 0xFF) | 0x10) as u8)
                .collect::<Vec<u8>>(),
        );
        frames.push(EncodedFrame {
            codec: Codec::H264,
            data: Bytes::from(data),
            timestamp_us: i * 33_333,
            is_keyframe: kf,
            duration_us: None,
            pts_dts: (0, 0),
        });
    }
    for (i, f) in frames.iter().enumerate() {
        let pts = (i as u64) * 33_333 * 9 / 100;
        muxer.add_frame(f, pts, pts).unwrap();
    }
    let bytes = muxer.drain();

    let d = demux(&bytes);
    assert!(d.pat_seen, "PAT not parsed");
    assert!(d.pmt_seen, "PMT not parsed");
    assert_eq!(d.video_pid, Some(0x100));
    assert_eq!(
        d.pes_units.len(),
        frames.len(),
        "should recover exactly one PES per input frame"
    );

    for (i, pes) in d.pes_units.iter().enumerate() {
        // 00 00 01 E0 LL LL FLAGS_A FLAGS_B HLEN ... payload
        assert_eq!(&pes[..3], &[0x00, 0x00, 0x01], "PES start code (frame {i})");
        assert_eq!(pes[3], 0xE0, "stream_id=video (frame {i})");
        // PES_packet_length is 0 → unbounded for video. Skip 2 bytes.
        assert_eq!(pes[6] & 0xC0, 0x80, "PES marker bits (frame {i})");
        let pts_dts_flags = (pes[7] >> 6) & 0x03;
        assert_eq!(pts_dts_flags, 0b11, "PTS+DTS flags (frame {i})");
        let header_data_len = pes[8] as usize;
        let pts = decode_ts(&pes[9..14]);
        let dts = decode_ts(&pes[14..19]);
        let expected = (i as u64) * 33_333 * 9 / 100;
        assert_eq!(pts, expected, "PTS for frame {i}");
        assert_eq!(dts, expected, "DTS for frame {i}");
        let au = &pes[9 + header_data_len..];
        // AU must start with the AUD NAL we inject.
        assert_eq!(
            &au[..6],
            &[0x00, 0x00, 0x00, 0x01, 0x09, 0xF0],
            "AUD NAL (frame {i})"
        );
        // Keyframes must contain the parameter sets injected by the
        // muxer immediately after the AUD.
        if frames[i].is_keyframe {
            let after_aud = &au[6..];
            assert!(
                after_aud.starts_with(&parameter_sets),
                "parameter sets missing at start of keyframe AU (frame {i})"
            );
            // And after the parameter sets, the encoder's NAL must
            // be present. Find a slice NAL (type 5 for IDR).
            let slice_off = parameter_sets.len();
            assert_eq!(
                &after_aud[slice_off..slice_off + 5],
                &[0x00, 0x00, 0x00, 0x01, 0x65],
                "IDR slice NAL after parameter sets (frame {i})"
            );
        } else {
            // Non-keyframes: AUD then slice (no parameter sets).
            assert_eq!(
                &au[6..6 + 5],
                &[0x00, 0x00, 0x00, 0x01, 0x41],
                "slice NAL after AUD (frame {i})"
            );
        }
    }
}

#[test]
fn cc_continues_across_segment_boundaries() {
    // ffmpeg's mpegts demuxer flags any CC jump as a discontinuity
    // and can drop the in-progress PES, surfacing as
    // `Packet corrupt (stream = 0, dts = ...)`. The muxer therefore
    // reuses its continuity state across `drain()` and only resets
    // on a fresh `MpegTs::default()` (i.e. server restart). This
    // test concatenates two segments and checks that every PID's CC
    // increments smoothly across the boundary.
    let mut muxer = MpegTs::default();
    muxer
        .config(vec![
            0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0xC0, 0x1E, 0x96, 0x54, 0x05, 0x01, 0x6C, 0x80,
            0x00, 0x00, 0x00, 0x01, 0x68, 0xCE, 0x3C, 0x80,
        ])
        .unwrap();

    // Segment 1: 5 frames, IDR + 4 P-frames.
    muxer.start_segment();
    for i in 0..5u64 {
        let f = synthetic_frame(i, i == 0, if i == 0 { 8000 } else { 1500 });
        muxer.add_frame(&f, i * 3000, i * 3000).unwrap();
    }
    let seg1 = muxer.drain();

    // Segment 2: another IDR + 4 P-frames. PTS continues.
    muxer.start_segment();
    for i in 5..10u64 {
        let f = synthetic_frame(i, i == 5, if i == 5 { 8000 } else { 1500 });
        muxer.add_frame(&f, i * 3000, i * 3000).unwrap();
    }
    let seg2 = muxer.drain();

    // Concatenate as the HLS player would and verify CC monotonicity.
    let mut joined = Vec::new();
    joined.extend_from_slice(&seg1);
    joined.extend_from_slice(&seg2);

    let mut last_cc: std::collections::HashMap<u16, u8> = Default::default();
    for off in (0..joined.len()).step_by(TS) {
        let pkt = &joined[off..off + TS];
        assert_eq!(pkt[0], 0x47);
        let pid = u16::from_be_bytes([pkt[1] & 0x1F, pkt[2]]);
        let afc = (pkt[3] >> 4) & 0x03;
        let cc = pkt[3] & 0x0F;
        // Only packets carrying payload increment CC.
        if afc == 0b01 || afc == 0b11 {
            if let Some(prev) = last_cc.get(&pid) {
                assert_eq!(
                    cc,
                    (prev + 1) & 0x0F,
                    "CC discontinuity on PID {pid:#x} at offset {off} (prev={prev}, got={cc})"
                );
            }
            last_cc.insert(pid, cc);
        }
    }
}
