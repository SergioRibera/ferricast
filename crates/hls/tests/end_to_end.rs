//! End-to-end smoke test for the muxer → ring → playlist pipeline.
//!
//! Builds a few synthetic H.264 access units, runs them through the
//! muxer to produce real MPEG-TS bytes, pushes the bytes into the
//! [`SegmentRing`], and verifies the playlist references them
//! correctly. Doesn't need an actual encoder or capture device.

use bytes::Bytes;
use ferricast_core::{Codec, EncodedFrame};
use ferricast_hls::SegmentRing;
use ferricast_muxer::Muxer;
use ferricast_muxer::mpeg_ts::MpegTs;

fn fake_frame(pts_us: u64, keyframe: bool) -> EncodedFrame {
    // Minimal Annex-B: AUD + (slice_idr if keyframe else slice). The
    // muxer prepends its own AUD too — duplicates are harmless.
    let mut data = Vec::new();
    data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
    data.push(if keyframe { 0x65 } else { 0x41 }); // NAL header
    data.extend_from_slice(&[0xAA; 32]); // dummy payload
    EncodedFrame {
        codec: Codec::H264,
        data: Bytes::from(data),
        timestamp_us: pts_us,
        is_keyframe: keyframe,
        duration_us: None,
        pts_dts: (0, 0),
    }
}

#[test]
fn segment_round_trip() {
    let mut muxer = MpegTs::default();
    // Bogus SPS+PPS so the muxer has something to inject on keyframes.
    muxer
        .config(vec![
            0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1F, // SPS (truncated, fine for muxer)
            0x00, 0x00, 0x00, 0x01, 0x68, 0xCE, 0x06, 0xE2, // PPS
        ])
        .unwrap();

    // 10 frames at ~30fps, IDR every 5.
    for i in 0..10u64 {
        let kf = i % 5 == 0;
        let pts_us = i * 33_333;
        let pts_90k = pts_us * 9 / 100;
        muxer
            .add_frame(&fake_frame(pts_us, kf), pts_90k, pts_90k)
            .unwrap();
    }

    let bytes = muxer.drain();
    assert!(!bytes.is_empty());
    // MPEG-TS is strictly aligned to 188-byte packets.
    assert_eq!(bytes.len() % 188, 0);
    // First byte must be a sync byte (PAT).
    assert_eq!(bytes[0], 0x47);
    // Subsequent packets must also start with sync byte.
    for off in (0..bytes.len()).step_by(188) {
        assert_eq!(bytes[off], 0x47, "packet at offset {off} missing sync byte");
    }

    // Push into a ring and verify the playlist references it.
    let mut ring = SegmentRing::new(6, 4);
    let seq = ring.push(2.5, false, Bytes::from(bytes));
    assert_eq!(seq, 0);
    let playlist = ring.build_playlist().unwrap();
    assert!(playlist.contains("#EXTM3U"));
    assert!(playlist.contains("segment-0.ts"));
    assert!(playlist.contains("#EXTINF:2.500,"));
    assert!(playlist.contains("#EXT-X-MEDIA-SEQUENCE:0"));
}

#[test]
fn ring_evicts_oldest_when_full() {
    let mut ring = SegmentRing::new(3, 2);
    for _ in 0..5 {
        ring.push(1.0, false, Bytes::from_static(&[0x47; 188]));
    }
    let playlist = ring.build_playlist().unwrap();
    // Last 3 entries kept (seqs 2, 3, 4).
    assert!(playlist.contains("#EXT-X-MEDIA-SEQUENCE:2"));
    assert!(playlist.contains("segment-2.ts"));
    assert!(playlist.contains("segment-4.ts"));
    assert!(!playlist.contains("segment-0.ts"));
}

#[test]
fn playlist_target_duration_is_fixed_and_clamps_long_segments() {
    // RFC 8216 §6.2.1: EXT-X-TARGETDURATION MUST NOT change between
    // reloads. An overlong segment (encoder ignored the forced IDR)
    // is clamped to the target instead of bumping it.
    let mut ring = SegmentRing::new(6, 4);
    ring.push(2.0, false, Bytes::from_static(&[0x47; 188]));
    let before = ring.build_playlist().unwrap();
    ring.push(4.6, false, Bytes::from_static(&[0x47; 188]));
    let after = ring
        .build_playlist()
        .expect("oversize segment must not fail playlist build");
    assert!(before.contains("#EXT-X-TARGETDURATION:4"), "{before}");
    assert!(after.contains("#EXT-X-TARGETDURATION:4"), "{after}");
    assert!(after.contains("#EXTINF:4.000,"), "{after}");
}

#[test]
fn playlist_target_duration_allows_rounding_down() {
    // 4.4 rounds to 4, which is allowed under a target of 4.
    let mut ring = SegmentRing::new(6, 4);
    ring.push(4.4, false, Bytes::from_static(&[0x47; 188]));
    let playlist = ring.build_playlist().unwrap();
    assert!(playlist.contains("#EXT-X-TARGETDURATION:4"));
    assert!(playlist.contains("#EXTINF:4.400,"));
}
