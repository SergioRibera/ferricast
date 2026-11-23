//! End-to-end source ↔ sink loopback over `127.0.0.1`.
//!
//! Spins the sink listener on an ephemeral port, drives a source
//! `MiracastSession` through the full WFD M1-M7, pumps a short MPEG2-TS
//! stream across the RTP link, and asserts the sink's
//! [`PushedStream`] receives the decoded frames.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use tokio::sync::Mutex;
use uuid::Uuid;

use ferricast_core::{
    CastSession, Codec, ControlSession, Device, DeviceAddresses, DeviceCapabilities, EncodedFrame,
    MediaCommand, MediaPacket, MediaPuller, ReceiverProtocol, StreamConfig,
};
use ferricast_miracast::{MiracastReceiver, MiracastSession};

fn fake_h264_idr_au() -> Bytes {
    // AUD (type 9), SPS (7), PPS (8), slice IDR (5). Payloads are
    // intentionally minimal — we are testing the TS/RTP/RTSP plumbing,
    // not an H.264 decoder.
    let nalus: &[&[u8]] = &[
        &[0x09, 0x10],
        &[0x67, 0x42, 0x00, 0x1E, 0x8D, 0x68],
        &[0x68, 0xCE, 0x3C, 0x80],
        &[0x65, 0x88, 0x82, 0x00, 0x1F, 0xFF],
    ];
    let mut out = Vec::new();
    for nal in nalus {
        out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        out.extend_from_slice(nal);
    }
    Bytes::from(out)
}

fn fake_h264_p_au(index: u32) -> Bytes {
    let mut slice = vec![0x41, 0x9A, 0x00, 0x04];
    slice.extend_from_slice(&index.to_be_bytes());
    let mut out = Vec::new();
    out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x09, 0x30]);
    out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
    out.extend_from_slice(&slice);
    Bytes::from(out)
}

fn device_for(addr: SocketAddr) -> Device {
    let addresses = DeviceAddresses::ranked(
        std::iter::once(addr),
        &ferricast_core::net::local_ipv4_subnets(),
    )
    .expect("loopback is routable");
    Device {
        id: Uuid::new_v4(),
        name: "test-sink".into(),
        protocol: "miracast",
        protocol_icon: Bytes::new(),
        addresses,
        model: None,
        capabilities: DeviceCapabilities {
            supported_codecs: vec![Codec::H264],
            max_width: Some(1280),
            max_height: Some(720),
            max_fps: Some(30),
            ..DeviceCapabilities::default()
        },
        metadata: {
            let mut m = HashMap::new();
            m.insert("rtsp_port".to_string(), addr.port().to_string());
            m
        },
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn source_to_sink_full_m1_m7_pumps_push_stream() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("ferricast_miracast=debug,warn")
        .try_init();

    let sink = MiracastReceiver::new("LoopbackSink", "deadbeef").with_bind_port(0);
    let mut control = sink.create_control().expect("control");
    let bound = control.local_addr().await.expect("bind");
    // The sink listens on 0.0.0.0 (INADDR_ANY); dial back through
    // 127.0.0.1 so the device address passes `AddressRank` (which
    // discards `0.0.0.0` as unspecified).
    let sink_addr = SocketAddr::new("127.0.0.1".parse().unwrap(), bound.port());

    // Drive the sink's accept in a task; wait for it to return the
    // RemoteSender (which happens after M1-M7 completes).
    let control = Arc::new(Mutex::new(control));
    let sink_arc = Arc::clone(&control);
    let sink_task = tokio::spawn(async move {
        let mut guard = sink_arc.lock().await;
        let remote = guard.accept().await.expect("sink accept");
        // Then grab the MediaCommand::Push with the live stream.
        let push = guard.next_command().await.expect("push command");
        (remote, push)
    });

    // Drive the source through the same handshake.
    let device = device_for(sink_addr);
    let mut session = MiracastSession::new();
    session.connect(&device).await.expect("source connect");
    session
        .setup_stream(&StreamConfig::default())
        .await
        .expect("setup_stream");

    // Pump: one IDR, five Ps.
    for i in 0..6u32 {
        let data = if i == 0 {
            fake_h264_idr_au()
        } else {
            fake_h264_p_au(i)
        };
        let frame = EncodedFrame {
            codec: Codec::H264,
            data,
            timestamp_us: u64::from(i) * 33_333,
            is_keyframe: i == 0,
            duration_us: Some(33_333),
            pts_dts: (u64::from(i) * 33_333, u64::from(i) * 33_333),
        };
        session.send_frame(&frame).await.expect("send_frame");
    }

    let (_remote, push_cmd) = tokio::time::timeout(Duration::from_secs(5), sink_task)
        .await
        .expect("sink task join timeout")
        .expect("sink task panic");

    let MediaCommand::Push(mut stream) = push_cmd else {
        panic!("expected MediaCommand::Push");
    };

    stream
        .open(ferricast_core::PullSpec {
            url: String::new(),
            headers: HashMap::new(),
        })
        .await
        .expect("push stream open");

    // Collect packets until we see a video one; the test passes as
    // long as at least one TS/RTP round-trip reconstructs a PES.
    let mut video_frames = 0usize;
    let deadline = Duration::from_secs(5);
    for _ in 0..100 {
        let next = tokio::time::timeout(deadline, stream.next()).await;
        match next {
            Ok(Ok(MediaPacket::Video(_))) => {
                video_frames += 1;
                if video_frames >= 1 {
                    break;
                }
            }
            Ok(Ok(MediaPacket::Eos)) => break,
            Ok(Ok(_other)) => {}
            Ok(Err(e)) => panic!("stream.next: {e}"),
            Err(_) => panic!("stream.next timed out after {deadline:?}"),
        }
    }

    session.stop().await.ok();
    {
        let mut g = control.lock().await;
        g.close().await.ok();
    }

    assert!(
        video_frames >= 1,
        "expected at least one reconstructed video frame"
    );
}
