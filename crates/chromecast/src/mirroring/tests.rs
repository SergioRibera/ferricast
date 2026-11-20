//! End-to-end checks of the transport against a simulated receiver that
//! validates packets the way openscreen's `rtp_packet_parser.cc` does.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

use super::rtcp::tests::{cast_feedback, pli, rr, xr_rrt};
use super::rtcp::unix_us_to_ntp;
use super::rtp::tests::OpenSslStyleCtr;
use super::rtp::{ALL_PACKETS_LOST, FrameCrypto, MAX_PAYLOAD_SIZE, RtpPayloadType};
use super::sender::{StreamKind, StreamParams, StreamSender};
use super::transport::{MAX_PACKETS_PER_BURST, MediaFrame, Transport, TransportShared};

// SSRCs used by the RTCP builders in `rtcp::tests`.
const RX_SSRC: u32 = 0x0000_1111;
const TX_SSRC: u32 = 0x0002_2222;
const AUDIO_TX_SSRC: u32 = 77;
const AUDIO_RX_SSRC: u32 = 78;
const KEY: [u8; 16] = [0x42; 16];
const MASK: [u8; 16] = [0x17; 16];

/// One packet accepted by the receiver-side parser.
#[derive(Debug, Clone)]
struct Parsed {
    marker: bool,
    payload_type: u8,
    sequence: u16,
    rtp_timestamp: u32,
    is_key_frame: bool,
    frame_id_8: u8,
    packet_id: u16,
    max_packet_id: u16,
    referenced_8: u8,
    payload: Vec<u8>,
}

/// Mirrors `RtpPacketParser::ParseV1` acceptance rules.
fn parse_like_openscreen(buf: &[u8], expected_ssrc: u32) -> Option<Parsed> {
    const MIN_VALID_SIZE: usize = 18;
    if buf.len() < MIN_VALID_SIZE || buf[0] != 0x80 {
        return None;
    }
    let payload_type = buf[1] & 0x7F;
    if !matches!(payload_type, 96..=104 | 127) {
        return None;
    }
    let ssrc = u32::from_be_bytes(buf[8..12].try_into().ok()?);
    if ssrc != expected_ssrc {
        return None;
    }
    let byte12 = buf[12];
    let has_reference = byte12 & 0x40 != 0;
    let extensions = usize::from(byte12 & 0x3F);
    let packet_id = u16::from_be_bytes([buf[14], buf[15]]);
    let max_packet_id = u16::from_be_bytes([buf[16], buf[17]]);
    if max_packet_id == ALL_PACKETS_LOST || packet_id > max_packet_id {
        return None;
    }
    let mut rest = &buf[18..];
    let referenced_8 = if has_reference {
        let (&r, tail) = rest.split_first()?;
        rest = tail;
        r
    } else if byte12 & 0x80 != 0 {
        buf[13]
    } else {
        buf[13].wrapping_sub(1)
    };
    for _ in 0..extensions {
        if rest.len() < 2 {
            return None;
        }
        let type_and_size = u16::from_be_bytes([rest[0], rest[1]]);
        rest = &rest[2..];
        let size = usize::from(type_and_size & 0x3FF);
        if rest.len() < size || (type_and_size >> 10 == 1 && size != 2) {
            return None;
        }
        rest = &rest[size..];
    }
    Some(Parsed {
        marker: buf[1] & 0x80 != 0,
        payload_type,
        sequence: u16::from_be_bytes([buf[2], buf[3]]),
        rtp_timestamp: u32::from_be_bytes(buf[4..8].try_into().ok()?),
        is_key_frame: byte12 & 0x80 != 0,
        frame_id_8: buf[13],
        packet_id,
        max_packet_id,
        referenced_8,
        payload: rest.to_vec(),
    })
}

fn unix_now_us() -> u64 {
    u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros(),
    )
    .unwrap()
}

fn sender(kind: StreamKind) -> StreamSender {
    let (payload_type, tx, rx, timebase) = match kind {
        StreamKind::Video => (RtpPayloadType::H264, TX_SSRC, RX_SSRC, 90_000),
        StreamKind::Audio => (RtpPayloadType::Opus, AUDIO_TX_SSRC, AUDIO_RX_SSRC, 48_000),
    };
    StreamSender::new(StreamParams {
        kind,
        payload_type,
        sender_ssrc: tx,
        receiver_ssrc: rx,
        rtp_timebase: timebase,
        target_playout_delay: Duration::from_millis(400),
        crypto: FrameCrypto::new(KEY, MASK),
        first_sequence: 65_530,
    })
}

struct Harness {
    receiver: UdpSocket,
    frames: mpsc::Sender<MediaFrame>,
    shared: Arc<TransportShared>,
    _task: super::AbortOnDrop,
}

async fn harness() -> Harness {
    let receiver = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    socket
        .connect(receiver.local_addr().unwrap())
        .await
        .unwrap();
    receiver
        .connect(socket.local_addr().unwrap())
        .await
        .unwrap();
    let shared = Arc::new(TransportShared::default());
    let transport = Transport::new(
        Arc::new(socket),
        vec![sender(StreamKind::Video), sender(StreamKind::Audio)],
        shared.clone(),
        None,
    );
    let (frames, rx) = mpsc::channel(16);
    let task = super::AbortOnDrop(tokio::spawn(transport.run(rx)));
    Harness {
        receiver,
        frames,
        shared,
        _task: task,
    }
}

async fn recv(h: &Harness) -> Option<(Vec<u8>, Instant)> {
    let mut buf = vec![0u8; 2048];
    let n = tokio::time::timeout(Duration::from_millis(500), h.receiver.recv(&mut buf))
        .await
        .ok()?
        .ok()?;
    buf.truncate(n);
    Some((buf, Instant::now()))
}

fn is_rtcp(p: &[u8]) -> bool {
    (200..=207).contains(&p[1])
}

async fn send(h: &Harness, kind: StreamKind, data: &[u8], key: bool, capture_us: u64) {
    h.frames
        .send(MediaFrame {
            kind,
            data: Bytes::from(data.to_vec()),
            is_key_frame: key,
            capture_us,
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn video_frames_survive_the_wire_and_are_paced() {
    let h = harness().await;
    let capture = unix_now_us();
    let key_frame: Vec<u8> = (0..MAX_PAYLOAD_SIZE * 100 + 123)
        .map(|i| (i * 13) as u8)
        .collect();
    let delta: Vec<u8> = (0..2_000).map(|i| (i * 3) as u8).collect();

    // A delta frame before any key frame is useless to the receiver.
    send(&h, StreamKind::Video, &delta, false, capture - 10_000).await;
    send(&h, StreamKind::Video, &key_frame, true, capture).await;
    send(&h, StreamKind::Video, &delta, false, capture + 33_333).await;

    let mut rtp = Vec::new();
    let mut first_rtp_at = None;
    let mut last_rtp_at = None;
    let mut sender_report = None;
    while let Some((p, at)) = recv(&h).await {
        if is_rtcp(&p) {
            if p[1] == 200 && sender_report.is_none() {
                assert!(
                    rtp.is_empty(),
                    "Sender Report precedes the first RTP packet"
                );
                sender_report = Some(p);
            }
            continue;
        }
        first_rtp_at.get_or_insert(at);
        last_rtp_at = Some(at);
        rtp.push(parse_like_openscreen(&p, TX_SSRC).expect("packet passes openscreen checks"));
        if rtp.len() == 103 {
            break;
        }
    }
    assert!(h.shared.key_frame_requested.load(Ordering::Relaxed));
    assert_eq!(rtp.len(), 103, "101 key frame packets + 2 delta packets");

    // Pacing: at most 21 packets per 10 ms burst.
    let elapsed = last_rtp_at.unwrap() - first_rtp_at.unwrap();
    let min_gaps = u32::try_from(rtp.len().div_ceil(MAX_PACKETS_PER_BURST) - 1).unwrap();
    assert!(
        elapsed + Duration::from_millis(2) >= Duration::from_millis(10) * min_gaps,
        "packets were not paced: {elapsed:?}"
    );

    // Sequence numbers are consecutive and wrap.
    assert_eq!(rtp[0].sequence, 65_530);
    for pair in rtp.windows(2) {
        assert_eq!(pair[1].sequence, pair[0].sequence.wrapping_add(1));
    }

    // Reassemble and decrypt with an independent CTR implementation.
    let mut frames: BTreeMap<u8, Vec<&Parsed>> = BTreeMap::new();
    for p in &rtp {
        assert_eq!(p.payload_type, 101);
        frames.entry(p.frame_id_8).or_default().push(p);
    }
    for (fid, data, key) in [(0u8, &key_frame, true), (1, &delta, false)] {
        let pkts = &frames[&fid];
        assert_eq!(usize::from(pkts[0].max_packet_id) + 1, pkts.len());
        let mut ctr = OpenSslStyleCtr::new(&KEY, &MASK, u32::from(fid));
        let mut plain = Vec::new();
        for (i, p) in pkts.iter().enumerate() {
            assert_eq!(usize::from(p.packet_id), i);
            assert_eq!(p.marker, i + 1 == pkts.len());
            assert_eq!(p.is_key_frame, key);
            // Key frame references itself, the delta frame its predecessor.
            assert_eq!(p.referenced_8, 0);
            plain.extend(ctr.apply(&p.payload));
        }
        assert_eq!(&plain, data);
    }
    // RTP timestamps derive from capture time at 90 kHz.
    let ts0 = frames[&0][0].rtp_timestamp;
    let ts1 = frames[&1][0].rtp_timestamp;
    assert_eq!(ts1.wrapping_sub(ts0), 3_000);

    // The Sender Report maps its send time onto the same RTP timeline.
    let sr = sender_report.expect("a Sender Report was sent");
    assert_eq!(u32::from_be_bytes(sr[4..8].try_into().unwrap()), TX_SSRC);
    let ntp = u64::from_be_bytes(sr[8..16].try_into().unwrap());
    let sr_rtp = u32::from_be_bytes(sr[16..20].try_into().unwrap());
    let since_capture_s = (ntp - unix_us_to_ntp(capture)) as f64 / 4_294_967_296.0;
    let expected = ts0.wrapping_add((since_capture_s * 90_000.0).round() as u32);
    let diff = sr_rtp
        .wrapping_sub(expected)
        .min(expected.wrapping_sub(sr_rtp));
    assert!(diff <= 2, "SR RTP {sr_rtp} vs expected {expected}");
}

#[tokio::test]
async fn nack_retransmits_and_checkpoint_reports_progress() {
    let h = harness().await;
    send(
        &h,
        StreamKind::Video,
        &vec![7; MAX_PAYLOAD_SIZE * 5],
        true,
        unix_now_us(),
    )
    .await;
    let mut packets = 0;
    while let Some((p, _)) = recv(&h).await {
        if !is_rtcp(&p) {
            packets += 1;
            if packets == 5 {
                break;
            }
        }
    }
    assert_eq!(packets, 5);
    assert!(!h.shared.stats().has_progress(StreamKind::Video));

    // Packet 2 lost.
    let nack = [
        rr(None),
        xr_rrt(unix_us_to_ntp(unix_now_us())),
        cast_feedback(0xFF, 400, &[(0, 2, 0)], None),
    ]
    .concat();
    h.receiver.send(&nack).await.unwrap();
    let resent = loop {
        let (p, _) = recv(&h).await.expect("retransmission");
        if !is_rtcp(&p) {
            break parse_like_openscreen(&p, TX_SSRC).unwrap();
        }
    };
    assert_eq!((resent.frame_id_8, resent.packet_id), (0, 2));

    // Frame 0 complete, then a picture loss.
    let ack = [
        rr(None),
        xr_rrt(unix_us_to_ntp(unix_now_us())),
        cast_feedback(0x00, 400, &[], None),
        pli(),
    ]
    .concat();
    h.receiver.send(&ack).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    let stats = h.shared.stats();
    assert!(stats.has_progress(StreamKind::Video));
    assert!(stats.last_rtcp_at.is_some());
    assert!(h.shared.key_frame_requested.load(Ordering::Relaxed));
}

#[tokio::test]
async fn audio_frames_are_single_packet_key_frames() {
    let h = harness().await;
    let capture = unix_now_us();
    for i in 0..3u8 {
        send(
            &h,
            StreamKind::Audio,
            &[i; 120],
            true,
            capture + u64::from(i) * 20_000,
        )
        .await;
    }
    let mut got = Vec::new();
    while let Some((p, _)) = recv(&h).await {
        if !is_rtcp(&p) {
            got.push(parse_like_openscreen(&p, AUDIO_TX_SSRC).unwrap());
            if got.len() == 3 {
                break;
            }
        }
    }
    assert_eq!(got.len(), 3);
    for (i, p) in (0u8..).zip(&got) {
        assert_eq!(p.payload_type, 96);
        assert!(p.marker && p.is_key_frame);
        assert_eq!(p.frame_id_8, i);
        assert_eq!(p.referenced_8, i);
        assert_eq!((p.packet_id, p.max_packet_id), (0, 0));
        let mut ctr = OpenSslStyleCtr::new(&KEY, &MASK, u32::from(i));
        assert_eq!(ctr.apply(&p.payload), vec![i; 120]);
    }
    // 20 ms at 48 kHz.
    assert_eq!(got[1].rtp_timestamp.wrapping_sub(got[0].rtp_timestamp), 960);
}

fn app(app_id: &str, name: &str, transport: &str, session: &str) -> crate::castv2::ApplicationInfo {
    serde_json::from_value(serde_json::json!({
        "appId": app_id,
        "displayName": name,
        "transportId": transport,
        "sessionId": session,
    }))
    .unwrap()
}

#[test]
fn receiver_status_classification() {
    use super::{MIRRORING_APP_ID, StatusVerdict, classify_status};

    let partial = app(MIRRORING_APP_ID, "", "t-1", "");
    assert_eq!(classify_status(None, &[partial]), StatusVerdict::NotReady);

    let ours = app(MIRRORING_APP_ID, "Chrome Mirroring", "t-1", "s-1");
    assert_eq!(
        classify_status(None, std::slice::from_ref(&ours)),
        StatusVerdict::Launched {
            transport_id: "t-1".into(),
            session_id: "s-1".into()
        }
    );
    assert_eq!(
        classify_status(Some("s-1"), &[ours]),
        StatusVerdict::Running
    );

    let other = app("CC1AD845", "Default Media Receiver", "t-2", "s-2");
    assert_eq!(
        classify_status(Some("s-1"), &[other]),
        StatusVerdict::Stopped("mirroring replaced by Default Media Receiver".into())
    );
    assert_eq!(
        classify_status(Some("s-1"), &[]),
        StatusVerdict::Stopped("mirroring app stopped on the receiver".into())
    );
}
