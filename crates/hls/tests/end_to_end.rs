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
        assert_eq!(
            bytes[off],
            0x47,
            "packet at offset {off} missing sync byte"
        );
    }

    // Push into a ring and verify the playlist references it.
    let mut ring = SegmentRing::new(6);
    let seq = ring.push(2.5, false, Bytes::from(bytes));
    assert_eq!(seq, 0);
    let playlist = ring.build_playlist(4).unwrap();
    assert!(playlist.contains("#EXTM3U"));
    assert!(playlist.contains("segment-0.ts"));
    assert!(playlist.contains("#EXTINF:2.500,"));
    assert!(playlist.contains("#EXT-X-MEDIA-SEQUENCE:0"));
}

#[test]
fn ring_evicts_oldest_when_full() {
    let mut ring = SegmentRing::new(3);
    for _ in 0..5 {
        ring.push(1.0, false, Bytes::from_static(&[0x47; 188]));
    }
    let playlist = ring.build_playlist(2).unwrap();
    // Last 3 entries kept (seqs 2, 3, 4).
    assert!(playlist.contains("#EXT-X-MEDIA-SEQUENCE:2"));
    assert!(playlist.contains("segment-2.ts"));
    assert!(playlist.contains("segment-4.ts"));
    assert!(!playlist.contains("segment-0.ts"));
}

#[test]
fn playlist_target_duration_adapts_to_long_segments() {
    // Reproduces the symptom the user hit: x264 keyint=250 produces
    // ~4 s segments while the configured min target was 4. With
    // jitter a single 4.5 s segment used to make build_playlist
    // error out and the HTTP handler closed the socket without
    // writing — players saw "End of file 0KB". The ring now bumps
    // target_duration up to ceil(max) so the playlist is always
    // RFC 8216 §4.3.3.1-compliant.
    let mut ring = SegmentRing::new(6);
    ring.push(4.5, false, Bytes::from_static(&[0x47; 188]));
    let playlist = ring
        .build_playlist(/* min */ 4)
        .expect("oversize segment must not fail playlist build");
    assert!(
        playlist.contains("#EXT-X-TARGETDURATION:5"),
        "expected target=5 (ceil of 4.5) in:\n{playlist}"
    );
    assert!(playlist.contains("#EXTINF:4.500,"));
}

#[test]
fn playlist_target_duration_keeps_floor_when_segments_are_short() {
    let mut ring = SegmentRing::new(6);
    ring.push(1.2, false, Bytes::from_static(&[0x47; 188]));
    ring.push(1.5, false, Bytes::from_static(&[0x47; 188]));
    let playlist = ring.build_playlist(/* min */ 4).unwrap();
    // Floor wins because ceil(1.5)=2 < 4.
    assert!(playlist.contains("#EXT-X-TARGETDURATION:4"));
}
