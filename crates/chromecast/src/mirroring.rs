//! Cast Streaming (mirror) session — sub-200 ms RTP/UDP alternative to HLS.
//!
//! Protocol flow on top of the same CASTv2 TLS channel as [`crate::session`]:
//! 1. [`ChromecastMirrorSession::connect`] — TLS handshake + launch the
//!    Mirroring Receiver app (`0F5096E8`), latch `transport_id`/`session_id`.
//! 2. [`ChromecastMirrorSession::setup_stream`] — build and send OFFER
//!    (H.264 + optional Opus stream specs) on `urn:x-cast:com.google.cast.webrtc`;
//!    wait for ANSWER which carries the receiver's UDP port.
//! 3. [`ChromecastMirrorSession::send_frame`] / [`ChromecastMirrorSession::send_audio_frame`]
//!    — RTP packetization + AES-128-CTR + UDP send (Phase 2; stubbed here).
//! 4. [`ChromecastMirrorSession::stop`] — STOP + close TLS + abort tasks.
//!
//! # 2nd-gen Chromecast compatibility
//! The Mirroring Receiver app is built-in on every Chromecast, but older
//! firmware may only accept OFFER from a Google-signed Chrome sender. If a
//! device stays in LOADING after OFFER, fall back to [`crate::session::ChromecastSession`]
//! (HLS). `DeviceCapabilities::supports_cast_streaming` is `false` for
//! 1st/2nd-gen `md=Chromecast` devices so the manager can route automatically.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::time::Duration;

use bytes::BytesMut;
use ferricast_core::{
    AudioCodec, AudioFrame, CastSession, Codec, Device, EncodedFrame, FerricastError, H264Profile,
    Result, StreamConfig, bind_udp_in_range,
};
use serde::{Deserialize, Serialize};
use tokio::net::UdpSocket;
use tokio::sync::{Mutex, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::castv2::{
    CastMessage, DEFAULT_RECEIVER_ID, DEFAULT_SENDER_ID, MIRRORING_APP_ID, ReceiverStatusPayload,
    close_message, connect_message, get_status_message, launch_message, namespace, ping_message,
    pong_message, stop_app_message,
};
use crate::wire::{self, SharedWriter};

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
const VIDEO_RTP_PAYLOAD_TYPE: u8 = 96;
const AUDIO_RTP_PAYLOAD_TYPE: u8 = 97;
const VIDEO_CLOCK_RATE: u32 = 90_000;
const AUDIO_CLOCK_RATE: u32 = 48_000;
/// Sequence number used in the single OFFER/ANSWER exchange per session.
const OFFER_SEQ_NUM: u32 = 1;
/// Milliseconds of buffering headroom suggested to the receiver.
const TARGET_DELAY_MS: u32 = 400;
/// How often to send RTCP Sender Reports (RFC 3550 §6.4.1).
const RTCP_INTERVAL: Duration = Duration::from_secs(1);
/// RTCP payload type for Sender Report.
const RTCP_PT_SR: u8 = 200;
/// RTCP payload type for Payload-Specific Feedback (PSFB). PLI uses FMT=1.
const RTCP_PT_PSFB: u8 = 206;

// ── OFFER / ANSWER types ──────────────────────────────────────────────────────

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OfferEnvelope<'a> {
    #[serde(rename = "type")]
    msg_type: &'static str,
    seq_num: u32,
    offer: &'a Offer,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Offer {
    cast_mode: &'static str,
    supported_streams: Vec<StreamSpec>,
}

/// One entry in `offer.supportedStreams`. The `"type"` field acts as the
/// serde tag (internally-tagged enum) so it lands in the JSON object
/// alongside the other stream fields, as the Cast spec requires.
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
enum StreamSpec {
    #[serde(rename = "video_source")]
    Video {
        index: u32,
        #[serde(rename = "codecName")]
        codec_name: String,
        #[serde(rename = "rtpProfile")]
        rtp_profile: String,
        #[serde(rename = "rtpPayloadType")]
        rtp_payload_type: u8,
        ssrc: u32,
        #[serde(rename = "aesKey")]
        aes_key: String,
        #[serde(rename = "aesIvMask")]
        aes_iv_mask: String,
        #[serde(rename = "timeBase")]
        time_base: String,
        #[serde(rename = "targetDelay")]
        target_delay: u32,
        resolutions: Vec<Resolution>,
        #[serde(rename = "maxFrameRate")]
        max_frame_rate: String,
        #[serde(rename = "maxBitRate")]
        max_bit_rate: u32,
        #[serde(rename = "videoCodecParams")]
        video_codec_params: VideoCodecParams,
    },
    #[serde(rename = "audio_source")]
    Audio {
        index: u32,
        #[serde(rename = "codecName")]
        codec_name: String,
        #[serde(rename = "rtpProfile")]
        rtp_profile: String,
        #[serde(rename = "rtpPayloadType")]
        rtp_payload_type: u8,
        ssrc: u32,
        #[serde(rename = "aesKey")]
        aes_key: String,
        #[serde(rename = "aesIvMask")]
        aes_iv_mask: String,
        #[serde(rename = "timeBase")]
        time_base: String,
        #[serde(rename = "targetDelay")]
        target_delay: u32,
        channels: u32,
        #[serde(rename = "bitRate")]
        bit_rate: u32,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Resolution {
    width: u32,
    height: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct VideoCodecParams {
    profile: String,
    level: String,
    avc: AvcParams,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct AvcParams {
    #[serde(rename = "packetizationMode")]
    packetization_mode: u8,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct AnswerEnvelope {
    seq_num: u32,
    result: String,
    answer: Option<AnswerBody>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct AnswerBody {
    udp_port: u16,
    send_indexes: Vec<u32>,
    // JSON has no unsigned type; Chromecast sends SSRCs as signed 32-bit
    ssrcs: Vec<i64>,
}

// ── Session structs ───────────────────────────────────────────────────────────

#[derive(Default)]
pub struct ChromecastMirrorSession {
    state: Option<ConnectedState>,
    stream: Option<MirrorStreamState>,
    cfg: Option<StreamConfig>,
    /// H.264 profile the encoder will use, captured from device caps during
    /// `connect` so `setup_stream` can advertise the matching profile_idc in
    /// the OFFER. Mismatching profile (e.g. advertising Main while sending
    /// High) causes receivers to mis-configure their decoder → black video.
    h264_profile: Option<H264Profile>,
}

struct ConnectedState {
    writer: SharedWriter,
    transport_id: String,
    session_id: String,
    peer_ip: IpAddr,
    request_id: Arc<AtomicI64>,
    alive: Arc<AtomicBool>,
    /// `setup_stream` arms this with a tx before sending OFFER; the read
    /// loop takes the tx and fires it when ANSWER arrives.
    answer_slot: Arc<Mutex<Option<oneshot::Sender<Result<AnswerBody>>>>>,
    read_handle: JoinHandle<()>,
    heartbeat_handle: JoinHandle<()>,
}

struct MirrorStreamState {
    /// UDP socket (connected, so `send` needs no address). Wrapped in `Arc`
    /// so the RTCP recv task can hold a clone and call `recv_from` while
    /// this side calls `send` — both take `&self` in tokio so no data race.
    socket: Arc<UdpSocket>,
    /// Chromecast's RTP/RTCP receive endpoint (IP from discovery, port from ANSWER).
    peer_udp: SocketAddr,
    video_ssrc: u32,
    audio_ssrc: Option<u32>,
    video_aes_key: [u8; 16],
    video_aes_iv_mask: [u8; 16],
    audio_aes_key: Option<[u8; 16]>,
    audio_aes_iv_mask: Option<[u8; 16]>,
    video_seq: u16,
    video_frame_id: u32,
    audio_seq: u16,
    audio_frame_id: u32,
    // ── RTCP Sender Report stats ──────────────────────────────────────────────
    video_pkt_count: u32,
    video_byte_count: u32,
    audio_pkt_count: u32,
    audio_byte_count: u32,
    last_video_rtp_ts: u32,
    last_audio_rtp_ts: u32,
    last_rtcp_at: std::time::Instant,
    // ── PLI signaling ─────────────────────────────────────────────────────────
    /// Set to `true` by the RTCP receive task on incoming PLI. Cleared by
    /// [`CastSession::poll_keyframe_request`] so the manager can call
    /// `encoder.request_keyframe()` on the next video frame.
    keyframe_requested: Arc<AtomicBool>,
    /// Background task that listens on the UDP recv half for RTCP from the
    /// Chromecast (primarily PLI, but also RR). Aborted in `stop()`.
    rtcp_handle: JoinHandle<()>,
}

// ── Codec parameter helpers ───────────────────────────────────────────────────

/// Map `H264Profile` → profile_idc hex string used in the Cast OFFER.
fn h264_profile_hex(profile: Option<H264Profile>) -> &'static str {
    match profile {
        Some(H264Profile::High) => "64",
        Some(H264Profile::Baseline) => "42",
        Some(H264Profile::Main) | None => "4d",
    }
}

/// Derive the minimum H.264 level_idc that covers the given resolution and
/// frame rate.  Returns the decimal level_idc value (e.g. 42 = Level 4.2).
/// Based on the MaxMBsPerSec limits in the H.264 spec Table A-1.
fn h264_level_idc(width: u32, height: u32, fps: u32) -> u32 {
    // Macro-blocks per second (each MB = 16×16 luma samples).
    let mbs_per_sec = ((width + 15) / 16) * ((height + 15) / 16) * fps;
    match mbs_per_sec {
        ..=40_500 => 30,
        ..=108_000 => 31,
        ..=216_000 => 32,
        ..=245_760 => 40,
        ..=522_240 => 42,
        ..=589_824 => 50,
        ..=983_040 => 51,
        ..=2_073_600 => 52,
        _ => 52,
    }
}

// ── CastSession implementation ────────────────────────────────────────────────

impl CastSession for ChromecastMirrorSession {
    async fn connect(&mut self, device: &Device) -> Result<()> {
        if device.protocol != "chromecast" {
            return Err(FerricastError::Protocol(format!(
                "ChromecastMirrorSession cannot connect to {:?}",
                device.protocol
            )));
        }
        if self.state.is_some() {
            return Err(FerricastError::SessionAlreadyActive(device.name.clone()));
        }

        info!(
            name = %device.name,
            addr = %device.addr,
            "connecting (mirror) to chromecast"
        );
        let (reader, writer, _local_ip) = wire::connect(device.addr, device.port).await?;

        let alive = Arc::new(AtomicBool::new(true));
        let answer_slot: Arc<Mutex<Option<oneshot::Sender<Result<AnswerBody>>>>> =
            Arc::new(Mutex::new(None));

        let (app_tx, app_rx) = oneshot::channel::<Result<(String, String)>>();
        let read_handle = tokio::spawn(mirror_read_loop(
            reader,
            writer.clone(),
            alive.clone(),
            Some((MIRRORING_APP_ID.to_string(), app_tx)),
            answer_slot.clone(),
        ));

        wire::send(
            &writer,
            &connect_message(DEFAULT_SENDER_ID, DEFAULT_RECEIVER_ID)
                .map_err(|e| FerricastError::Protocol(format!("build CONNECT: {e}")))?,
        )
        .await?;

        wire::send(
            &writer,
            &ping_message().map_err(|e| FerricastError::Protocol(format!("build PING: {e}")))?,
        )
        .await?;

        let request_id = Arc::new(AtomicI64::new(1));
        let launch_id = request_id.fetch_add(1, Ordering::Relaxed);
        wire::send(
            &writer,
            &launch_message(launch_id, MIRRORING_APP_ID)
                .map_err(|e| FerricastError::Protocol(format!("build LAUNCH: {e}")))?,
        )
        .await?;

        let status_id = request_id.fetch_add(1, Ordering::Relaxed);
        wire::send(
            &writer,
            &get_status_message(status_id)
                .map_err(|e| FerricastError::Protocol(format!("build GET_STATUS: {e}")))?,
        )
        .await?;

        let (transport_id, session_id) =
            match tokio::time::timeout(Duration::from_secs(15), app_rx).await {
                Ok(Ok(Ok(ids))) => ids,
                Ok(Ok(Err(e))) => return Err(e),
                Ok(Err(_)) => {
                    return Err(FerricastError::Connection(
                        "read loop exited before mirror app launch ack".into(),
                    ));
                }
                Err(_) => {
                    return Err(FerricastError::Timeout(
                        "waiting for Mirroring Receiver app to launch (15s)".into(),
                    ));
                }
            };
        debug!(transport_id, session_id, "MirroringReceiver app launched");

        wire::send(
            &writer,
            &connect_message(DEFAULT_SENDER_ID, &transport_id)
                .map_err(|e| FerricastError::Protocol(format!("build app CONNECT: {e}")))?,
        )
        .await?;

        let heartbeat_handle =
            tokio::spawn(mirror_heartbeat_loop(writer.clone(), alive.clone()));

        self.h264_profile = device.capabilities.max_h264_profile;
        self.state = Some(ConnectedState {
            writer,
            transport_id,
            session_id,
            peer_ip: device.addr,
            request_id,
            alive,
            answer_slot,
            read_handle,
            heartbeat_handle,
        });
        Ok(())
    }

    async fn setup_stream(&mut self, config: &StreamConfig) -> Result<()> {
        let state = self.state.as_ref().ok_or(FerricastError::NoActiveSession)?;

        if config.codec != Codec::H264 {
            return Err(FerricastError::UnsupportedCodec {
                codec: config.codec,
                protocol: "chromecast-mirror",
            });
        }

        self.cfg = Some(config.clone());

        let video_ssrc = random_ssrc();
        let video_aes_key = random_16_bytes();
        let video_aes_iv_mask = random_16_bytes();

        let mut streams = vec![StreamSpec::Video {
            index: 0,
            codec_name: "h264".into(),
            rtp_profile: "cast".into(),
            rtp_payload_type: VIDEO_RTP_PAYLOAD_TYPE,
            ssrc: video_ssrc,
            aes_key: hex_encode_16(&video_aes_key),
            aes_iv_mask: hex_encode_16(&video_aes_iv_mask),
            time_base: format!("1/{VIDEO_CLOCK_RATE}"),
            target_delay: TARGET_DELAY_MS,
            resolutions: vec![Resolution {
                width: config.width,
                height: config.height,
            }],
            max_frame_rate: format!("{}/1", config.fps),
            max_bit_rate: config.bitrate_kbps * 1_000,
            video_codec_params: VideoCodecParams {
                profile: h264_profile_hex(self.h264_profile).into(),
                level: h264_level_idc(config.width, config.height, config.fps).to_string(),
                avc: AvcParams {
                    packetization_mode: 1,
                },
            },
        }];

        let mut audio_ssrc_out = None;
        let mut audio_aes_key_out = None;
        let mut audio_aes_iv_out = None;

        if let Some(audio) = &config.audio {
            if audio.codec != AudioCodec::Opus {
                return Err(FerricastError::Encoder(format!(
                    "chromecast mirror: audio must be Opus, got {:?}",
                    audio.codec
                )));
            }
            let ssrc = random_ssrc();
            let aes_key = random_16_bytes();
            let aes_iv_mask = random_16_bytes();
            audio_ssrc_out = Some(ssrc);
            audio_aes_key_out = Some(aes_key);
            audio_aes_iv_out = Some(aes_iv_mask);
            streams.push(StreamSpec::Audio {
                index: 1,
                codec_name: "opus".into(),
                rtp_profile: "cast".into(),
                rtp_payload_type: AUDIO_RTP_PAYLOAD_TYPE,
                ssrc,
                aes_key: hex_encode_16(&aes_key),
                aes_iv_mask: hex_encode_16(&aes_iv_mask),
                time_base: format!("1/{AUDIO_CLOCK_RATE}"),
                target_delay: TARGET_DELAY_MS,
                channels: audio.channels as u32,
                bit_rate: audio.bitrate_kbps * 1_000,
            });
        }

        let offer = Offer {
            cast_mode: "mirroring",
            supported_streams: streams,
        };
        let envelope = OfferEnvelope {
            msg_type: "OFFER",
            seq_num: OFFER_SEQ_NUM,
            offer: &offer,
        };

        let offer_msg = CastMessage::new_json(
            DEFAULT_SENDER_ID,
            &state.transport_id,
            namespace::WEBRTC,
            &envelope,
        )
        .map_err(|e| FerricastError::Protocol(format!("build OFFER: {e}")))?;

        // Arm the slot BEFORE sending so we can't miss a fast ANSWER.
        let (answer_tx, answer_rx) = oneshot::channel::<Result<AnswerBody>>();
        *state.answer_slot.lock().await = Some(answer_tx);

        {
            let json = serde_json::to_string(&envelope)
                .unwrap_or_else(|e| format!("<serialize error: {e}>"));
            debug!(offer_json = %json, "Cast Streaming OFFER body");
        }
        wire::send(&state.writer, &offer_msg).await?;
        info!(transport_id = %state.transport_id, "sent Cast Streaming OFFER");

        let answer = match tokio::time::timeout(Duration::from_secs(10), answer_rx).await {
            Ok(Ok(Ok(a))) => a,
            Ok(Ok(Err(e))) => return Err(e),
            Ok(Err(_)) => {
                return Err(FerricastError::Connection(
                    "read loop exited before Cast Streaming ANSWER arrived".into(),
                ));
            }
            Err(_) => {
                return Err(FerricastError::Timeout(
                    "waiting for Cast Streaming ANSWER (10s) — \
                     device may not support the mirroring namespace"
                        .into(),
                ));
            }
        };

        info!(
            udp_port = answer.udp_port,
            send_indexes = ?answer.send_indexes,
            ssrcs = ?answer.ssrcs,
            "received Cast Streaming ANSWER"
        );

        let peer_udp = SocketAddr::new(state.peer_ip, answer.udp_port);

        let socket = match self.cfg.as_ref().and_then(|c| c.port_range) {
            Some((start, end)) => bind_udp_in_range(start, end).await.map_err(|e| {
                FerricastError::Connection(format!("bind cast streaming UDP in range: {e}"))
            })?,
            None => UdpSocket::bind("0.0.0.0:0").await.map_err(|e| {
                FerricastError::Connection(format!("bind cast streaming UDP: {e}"))
            })?,
        };
        socket
            .connect(peer_udp)
            .await
            .map_err(|e| FerricastError::Connection(format!("connect cast streaming UDP: {e}")))?;

        let socket = Arc::new(socket);
        let keyframe_requested = Arc::new(AtomicBool::new(false));
        let rtcp_handle = tokio::spawn(rtcp_recv_task(
            socket.clone(),
            video_ssrc,
            keyframe_requested.clone(),
        ));

        self.stream = Some(MirrorStreamState {
            socket,
            peer_udp,
            video_ssrc,
            audio_ssrc: audio_ssrc_out,
            video_aes_key,
            video_aes_iv_mask,
            audio_aes_key: audio_aes_key_out,
            audio_aes_iv_mask: audio_aes_iv_out,
            video_seq: 0,
            video_frame_id: 0,
            audio_seq: 0,
            audio_frame_id: 0,
            video_pkt_count: 0,
            video_byte_count: 0,
            audio_pkt_count: 0,
            audio_byte_count: 0,
            last_video_rtp_ts: 0,
            last_audio_rtp_ts: 0,
            last_rtcp_at: std::time::Instant::now(),
            keyframe_requested,
            rtcp_handle,
        });
        info!(peer_udp = %peer_udp, video_ssrc, "Cast Streaming UDP socket ready");
        Ok(())
    }

    async fn send_frame(&mut self, frame: &EncodedFrame) -> Result<()> {
        if frame.codec != Codec::H264 {
            return Err(FerricastError::UnsupportedCodec {
                codec: frame.codec,
                protocol: "chromecast-mirror",
            });
        }
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| FerricastError::Streaming("mirror stream not initialised".into()))?;

        let ts = video_rtp_ts(frame.timestamp_us);
        let data = &frame.data;

        // Cast Streaming splits the raw Annex-B bitstream into MAX_RTP_PAYLOAD
        // chunks. Each chunk gets a 9-byte Cast payload header (unencrypted)
        // followed by the AES-128-CTR encrypted data. No RFC 6184 FU-A
        // wrapping — the Cast header's packet_id/max_packet_id carries the
        // reassembly state.
        let chunks: Vec<&[u8]> = data.chunks(MAX_RTP_PAYLOAD).collect();
        let n_pkts = chunks.len();
        let max_packet_id = (n_pkts.saturating_sub(1)) as u16;

        debug!(frame_id = stream.video_frame_id, n_pkts, data_len = data.len(), keyframe = frame.is_keyframe, "mirror: send_frame");
        if frame.is_keyframe && stream.video_frame_id == 0 {
            let preview: String = data.iter().take(32)
                .map(|b| format!("{b:02x}"))
                .collect::<Vec<_>>()
                .join(" ");
            debug!(first32 = %preview, "mirror: keyframe[0] raw bytes (pre-encrypt)");
        }

        for (packet_id_usize, chunk) in chunks.iter().enumerate() {
            let packet_id = packet_id_usize as u16;
            let is_last = packet_id_usize + 1 == n_pkts;

            let iv = cast_iv(
                &stream.video_aes_iv_mask,
                stream.video_ssrc,
                stream.video_frame_id,
                packet_id,
            );

            let reference_frame_id = (!frame.is_keyframe)
                .then(|| stream.video_frame_id.wrapping_sub(1));
            let pkt = cast_rtp_video_packet(
                stream.video_seq,
                ts,
                stream.video_ssrc,
                is_last,
                frame.is_keyframe,
                stream.video_frame_id,
                packet_id,
                max_packet_id,
                reference_frame_id,
                chunk,
                &stream.video_aes_key,
                &iv,
            );
            let pkt_len = pkt.len();
            if let Err(e) = stream.socket.send(&pkt).await {
                debug!(%e, "mirror: video UDP send dropped");
            }
            stream.video_pkt_count = stream.video_pkt_count.wrapping_add(1);
            stream.video_byte_count = stream.video_byte_count.wrapping_add(pkt_len as u32);
            stream.video_seq = stream.video_seq.wrapping_add(1);
        }
        stream.last_video_rtp_ts = ts;
        stream.video_frame_id = stream.video_frame_id.wrapping_add(1);
        maybe_send_rtcp_sr(stream).await;
        Ok(())
    }

    async fn send_audio_frame(&mut self, frame: &AudioFrame) -> Result<()> {
        if frame.codec != AudioCodec::Opus {
            return Err(FerricastError::Encoder(format!(
                "chromecast mirror: expected Opus audio, got {:?}",
                frame.codec
            )));
        }
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| FerricastError::Streaming("mirror stream not initialised".into()))?;

        let (Some(ssrc), Some(key), Some(iv_mask)) = (
            stream.audio_ssrc,
            stream.audio_aes_key.as_ref(),
            stream.audio_aes_iv_mask.as_ref(),
        ) else {
            // Session was set up without audio — silently drop.
            return Ok(());
        };

        let ts = audio_rtp_ts(frame.timestamp_us);
        let iv = cast_iv(iv_mask, ssrc, stream.audio_frame_id, 0);

        // One Opus frame per RTP packet — packet_id=0, max_packet_id=0.
        let pkt = cast_rtp_audio_packet(
            stream.audio_seq,
            ts,
            ssrc,
            stream.audio_frame_id,
            &frame.data,
            key,
            &iv,
        );

        let pkt_len = pkt.len();
        if let Err(e) = stream.socket.send(&pkt).await {
            debug!(%e, "mirror: audio UDP send dropped");
        }
        stream.audio_pkt_count = stream.audio_pkt_count.wrapping_add(1);
        stream.audio_byte_count = stream.audio_byte_count.wrapping_add(pkt_len as u32);
        stream.last_audio_rtp_ts = ts;
        stream.audio_seq = stream.audio_seq.wrapping_add(1);
        stream.audio_frame_id = stream.audio_frame_id.wrapping_add(1);
        maybe_send_rtcp_sr(stream).await;
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        if let Some(stream) = self.stream.take() {
            let MirrorStreamState { rtcp_handle, .. } = stream;
            rtcp_handle.abort();
            let _ = rtcp_handle.await;
        }
        let Some(state) = self.state.take() else {
            return Ok(());
        };

        let stop_id = state.request_id.fetch_add(1, Ordering::Relaxed);
        if let Ok(stop_msg) = stop_app_message(stop_id, &state.session_id) {
            if let Err(e) = wire::send(&state.writer, &stop_msg).await {
                warn!(%e, "mirror STOP failed");
            }
        }
        if let Ok(close) = close_message(DEFAULT_SENDER_ID, DEFAULT_RECEIVER_ID) {
            let _ = wire::send(&state.writer, &close).await;
        }

        state.alive.store(false, Ordering::Relaxed);
        state.read_handle.abort();
        state.heartbeat_handle.abort();
        let _ = state.read_handle.await;
        let _ = state.heartbeat_handle.await;

        info!("chromecast mirror session stopped");
        Ok(())
    }

    fn is_alive(&self) -> bool {
        self.state
            .as_ref()
            .map(|s| s.alive.load(Ordering::Relaxed))
            .unwrap_or(false)
    }

    fn poll_keyframe_request(&mut self) -> bool {
        self.stream
            .as_ref()
            .map(|s| s.keyframe_requested.swap(false, Ordering::Relaxed))
            .unwrap_or(false)
    }
}

impl Drop for ChromecastMirrorSession {
    fn drop(&mut self) {
        if let Some(state) = self.state.as_ref() {
            state.alive.store(false, Ordering::Relaxed);
            state.read_handle.abort();
            state.heartbeat_handle.abort();
        }
        if let Some(stream) = self.stream.as_ref() {
            stream.rtcp_handle.abort();
        }
    }
}

// ── Background tasks ──────────────────────────────────────────────────────────

async fn mirror_read_loop<R>(
    mut reader: R,
    writer: SharedWriter,
    alive: Arc<AtomicBool>,
    mut launch_watch: Option<(String, oneshot::Sender<Result<(String, String)>>)>,
    answer_slot: Arc<Mutex<Option<oneshot::Sender<Result<AnswerBody>>>>>,
) where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut buf = BytesMut::with_capacity(8 * 1024);
    loop {
        match wire::recv(&mut reader, &mut buf).await {
            Ok(Some(msg)) => {
                if msg.namespace == namespace::HEARTBEAT
                    && msg.message_type().as_deref() == Some("PING")
                {
                    if let Ok(pong) = pong_message() {
                        if wire::send(&writer, &pong).await.is_err() {
                            break;
                        }
                    }
                    continue;
                }

                if msg.namespace == namespace::CONNECTION
                    && msg.message_type().as_deref() == Some("CLOSE")
                {
                    let err = FerricastError::Connection(format!(
                        "mirror: receiver sent CLOSE from src={:?}",
                        msg.source_id
                    ));
                    if let Some((_, tx)) = launch_watch.take() {
                        let _ = tx.send(Err(err));
                    } else {
                        warn!(src = %msg.source_id, "mirror: receiver closed connection");
                    }
                    break;
                }

                if msg.namespace == namespace::RECEIVER {
                    if let Some((app_id, _)) = launch_watch.as_ref() {
                        match extract_app_ids(&msg, app_id) {
                            Some((tid, sid)) => {
                                info!(transport_id = %tid, "mirror app launched");
                                if let Some((_, tx)) = launch_watch.take() {
                                    let _ = tx.send(Ok((tid, sid)));
                                }
                            }
                            None => {
                                debug!(payload = ?msg.payload_utf8, "RECEIVER_STATUS: mirror app not ready");
                                if let Ok(gs) = get_status_message(2) {
                                    let _ = wire::send(&writer, &gs).await;
                                }
                            }
                        }
                    }
                } else if msg.namespace == namespace::WEBRTC {
                    if msg.message_type().as_deref() == Some("ANSWER") {
                        let result = parse_answer(&msg);
                        let mut slot = answer_slot.lock().await;
                        if let Some(tx) = slot.take() {
                            let _ = tx.send(result);
                        }
                    } else {
                        debug!(
                            ty = ?msg.message_type(),
                            payload = ?msg.payload_utf8,
                            "mirror: unhandled webrtc message"
                        );
                    }
                }
            }
            Ok(None) => {
                info!("mirror read loop: peer closed");
                if let Some((_, tx)) = launch_watch.take() {
                    let _ = tx.send(Err(FerricastError::Connection(
                        "mirror: receiver closed before launch ack".into(),
                    )));
                }
                break;
            }
            Err(e) => {
                warn!(%e, "mirror read loop terminating");
                if let Some((_, tx)) = launch_watch.take() {
                    let _ = tx.send(Err(e));
                }
                break;
            }
        }
    }
    alive.store(false, Ordering::Relaxed);
}

async fn mirror_heartbeat_loop(writer: SharedWriter, alive: Arc<AtomicBool>) {
    let mut tick = tokio::time::interval(HEARTBEAT_INTERVAL);
    tick.tick().await;
    while alive.load(Ordering::Relaxed) {
        tick.tick().await;
        let Ok(ping) = ping_message() else { break };
        if wire::send(&writer, &ping).await.is_err() {
            break;
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_answer(msg: &CastMessage) -> Result<AnswerBody> {
    let payload = msg
        .payload_utf8
        .as_deref()
        .ok_or_else(|| FerricastError::Protocol("ANSWER: no payload".into()))?;
    let env: AnswerEnvelope = serde_json::from_str(payload)
        .map_err(|e| FerricastError::Protocol(format!("ANSWER parse: {e}")))?;
    if env.result != "ok" {
        return Err(FerricastError::Protocol(format!(
            "Cast Streaming ANSWER result={:?}",
            env.result
        )));
    }
    env.answer
        .ok_or_else(|| FerricastError::Protocol("ANSWER result=ok but body absent".into()))
}

/// Pull `(transport_id, session_id)` for `app_id` from a RECEIVER_STATUS
/// message. Returns `None` if the app is missing or its IDs are unpopulated.
fn extract_app_ids(msg: &CastMessage, app_id: &str) -> Option<(String, String)> {
    let payload: ReceiverStatusPayload = msg.parse_payload().ok()?;
    let status = payload.status?;
    let app = status
        .applications
        .iter()
        .find(|a| a.app_id.eq_ignore_ascii_case(app_id))?;
    if app.transport_id.is_empty() || app.session_id.is_empty() {
        return None;
    }
    Some((app.transport_id.clone(), app.session_id.clone()))
}

fn random_ssrc() -> u32 {
    uuid::Uuid::new_v4().as_u128() as u32
}

fn random_16_bytes() -> [u8; 16] {
    uuid::Uuid::new_v4().into_bytes()
}

/// Encode 16 bytes as 32 lowercase hex chars (Cast's `aesKey`/`aesIvMask` format).
fn hex_encode_16(bytes: &[u8; 16]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ── RTP / crypto helpers ──────────────────────────────────────────────────────

/// Maximum RTP payload bytes per packet. Conservative: leaves room for IP+UDP
/// headers on a standard 1500-byte Ethernet MTU, and for the receiver's
/// reassembly window to stay comfortable.
const MAX_RTP_PAYLOAD: usize = 1_400;

/// Convert a capture timestamp (µs) to a 90 kHz RTP clock value (video).
fn video_rtp_ts(timestamp_us: u64) -> u32 {
    ((timestamp_us * 9) / 100) as u32
}

/// Convert a capture timestamp (µs) to a 48 kHz RTP clock value (Opus audio).
fn audio_rtp_ts(timestamp_us: u64) -> u32 {
    ((timestamp_us * 48) / 1_000) as u32
}

/// Build a 12-byte standard RTP header. `ext` sets the X bit (extension present).
fn rtp_header(pt: u8, seq: u16, ts: u32, ssrc: u32, marker: bool, ext: bool) -> [u8; 12] {
    let mut h = [0u8; 12];
    // V=2, P=0, X=ext, CC=0
    h[0] = 0x80 | (u8::from(ext) << 4);
    h[1] = pt | (u8::from(marker) << 7);
    h[2..4].copy_from_slice(&seq.to_be_bytes());
    h[4..8].copy_from_slice(&ts.to_be_bytes());
    h[8..12].copy_from_slice(&ssrc.to_be_bytes());
    h
}

/// Build the 8-byte RFC 5285 one-byte-header extension block for Cast Streaming.
///
/// Carries the absolute send time using extension ID 3, 3 data bytes.
///
/// Layout:
/// ```text
/// [0xBE, 0xDE]  profile (RFC 5285 one-byte header)
/// [0x00, 0x01]  length = 1 word (4 bytes of content)
/// [0x32]        element: ID=3, len-1=2 (3 bytes of data)
/// [b0, b1, b2]  abs_send_time_90khz (24-bit big-endian, lower 24 bits)
/// ```
///
/// The 24-bit absolute send time is the current system monotonic clock
/// expressed in 90 kHz ticks, truncated to 24 bits. The receiver uses it
/// to align interleaved audio and video packets by their wall-clock send
/// time, enabling accurate A/V sync independent of frame timestamps.
fn cast_rtp_extension(abs_send_time_90khz: u32) -> [u8; 8] {
    let mut ext = [0u8; 8];
    ext[0] = 0xBE;
    ext[1] = 0xDE;
    ext[2] = 0x00;
    ext[3] = 0x01; // 1 word = 4 bytes of content
    // RFC 5285 one-byte element header: ID=3, len-1=2 (3 data bytes)
    ext[4] = (3u8 << 4) | 2;
    let be = abs_send_time_90khz.to_be_bytes();
    ext[5] = be[1]; // bits 23..16
    ext[6] = be[2]; // bits 15..8
    ext[7] = be[3]; // bits 7..0
    ext
}

/// Returns the lower 24 bits of the current wall-clock time in 90 kHz ticks.
/// Used as the Cast abs-send-time extension value.
///
/// The 24-bit 90kHz counter wraps every ~186 s; the receiver correlates
/// audio and video packets via RTCP Sender Reports that carry the same
/// clock, so the wrap is handled transparently by the jitter buffer.
fn abs_send_time_90khz() -> u32 {
    let us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;
    // µs → 90 kHz ticks: µs * 90_000 / 1_000_000 = µs * 9 / 100
    let ticks = (us * 9) / 100;
    (ticks as u32) & 0x00FF_FFFF
}

/// Derive the AES-128-CTR IV for one Cast Streaming RTP packet.
///
/// Counter block layout (16 bytes, all big-endian), matching openscreen
/// `frame_crypto.cc` (`CreateNewIv`):
/// ```text
/// [0..4]   SSRC
/// [4..8]   frame_id
/// [8..10]  packet_id (fragment index within the frame, 2 bytes)
/// [10..16] zeros
/// ```
/// Final IV = `aes_iv_mask XOR counter_block`.
fn cast_iv(iv_mask: &[u8; 16], ssrc: u32, frame_id: u32, packet_id: u16) -> [u8; 16] {
    let mut counter = [0u8; 16];
    counter[0..4].copy_from_slice(&ssrc.to_be_bytes());
    counter[4..8].copy_from_slice(&frame_id.to_be_bytes());
    counter[8..10].copy_from_slice(&packet_id.to_be_bytes());
    // [10..16] zeros — already zero
    let mut iv = [0u8; 16];
    for (i, b) in iv.iter_mut().enumerate() {
        *b = iv_mask[i] ^ counter[i];
    }
    iv
}

/// Encrypt `data` in-place using AES-128-CTR with the given key and IV.
fn aes_ctr_encrypt(key: &[u8; 16], iv: &[u8; 16], data: &mut [u8]) {
    use aes::Aes128;
    use aes::cipher::{KeyIvInit, StreamCipher};
    use ctr::Ctr128BE;
    Ctr128BE::<Aes128>::new(key.into(), iv.into()).apply_keystream(data);
}

/// Build one Cast Streaming video RTP packet.
///
/// Cast payload header layout (openscreen `rtp_packet_builder.cc`):
/// ```text
/// Byte 0:      flags — 0x80 key frame | 0x40 has_reference_frame_id
/// Bytes 1–4:   frame_id (BE u32)
/// Bytes 5–6:   packet_id (BE u16, 0-based index within the frame)
/// Bytes 7–8:   max_packet_id (BE u16, total_packets − 1)
/// [Bytes 9-12: reference_frame_id (BE u32) — present iff has_reference_frame_id]
/// ```
///
/// Key frames → 9-byte header, byte0=0x80, no reference field.
/// P-frames → 13-byte header, byte0=0x40, 4-byte reference_frame_id appended.
/// The abs-send-time extension is included on every packet (matches openscreen).
#[allow(clippy::too_many_arguments)]
fn cast_rtp_video_packet(
    seq: u16,
    ts: u32,
    ssrc: u32,
    marker: bool,
    is_key_frame: bool,
    frame_id: u32,
    packet_id: u16,
    max_packet_id: u16,
    // None for key frames; Some(prev_frame_id) for P-frames.
    reference_frame_id: Option<u32>,
    chunk: &[u8],
    key: &[u8; 16],
    iv: &[u8; 16],
) -> Vec<u8> {
    // abs-send-time extension on every video packet (openscreen behaviour).
    let ext_block = cast_rtp_extension(abs_send_time_90khz());
    let hdr = rtp_header(VIDEO_RTP_PAYLOAD_TYPE, seq, ts, ssrc, marker, true);

    // Cast payload header: 9 bytes for key frames, 13 for P-frames.
    let cast_hdr_len = if is_key_frame { 9 } else { 13 };
    let mut cast_hdr = Vec::with_capacity(cast_hdr_len);
    if is_key_frame {
        cast_hdr.push(0x80u8);
    } else {
        cast_hdr.push(0x40u8); // has_reference_frame_id
    }
    cast_hdr.extend_from_slice(&frame_id.to_be_bytes());
    cast_hdr.extend_from_slice(&packet_id.to_be_bytes());
    cast_hdr.extend_from_slice(&max_packet_id.to_be_bytes());
    if !is_key_frame {
        let ref_id = reference_frame_id.unwrap_or_else(|| frame_id.wrapping_sub(1));
        cast_hdr.extend_from_slice(&ref_id.to_be_bytes());
    }

    let mut payload = chunk.to_vec();
    aes_ctr_encrypt(key, iv, &mut payload);

    let mut pkt = Vec::with_capacity(12 + 8 + cast_hdr_len + payload.len());
    pkt.extend_from_slice(&hdr);
    pkt.extend_from_slice(&ext_block);
    pkt.extend_from_slice(&cast_hdr);
    pkt.extend_from_slice(&payload);
    pkt
}

/// Build one Cast Streaming audio RTP packet (single Opus frame, no fragmentation).
fn cast_rtp_audio_packet(
    seq: u16,
    ts: u32,
    ssrc: u32,
    frame_id: u32,
    data: &[u8],
    key: &[u8; 16],
    iv: &[u8; 16],
) -> Vec<u8> {
    // abs-send-time extension on every audio packet.
    let hdr = rtp_header(AUDIO_RTP_PAYLOAD_TYPE, seq, ts, ssrc, true, true);
    let ext = cast_rtp_extension(abs_send_time_90khz());

    // Every Opus frame is independently decodable → always is_key_frame.
    // Cast receivers wait for an audio "keyframe" before starting A/V
    // render; without 0x80, audio never starts and video is also held.
    // packet_id = 0, max_packet_id = 0 (one Opus packet per Cast frame).
    let mut cast_hdr = [0u8; 9];
    cast_hdr[0] = 0x80; // is_key_frame
    cast_hdr[1..5].copy_from_slice(&frame_id.to_be_bytes());
    // bytes 5-8 remain zero: packet_id=0, max_packet_id=0

    let mut payload = data.to_vec();
    aes_ctr_encrypt(key, iv, &mut payload);

    let mut pkt = Vec::with_capacity(12 + 8 + 9 + payload.len());
    pkt.extend_from_slice(&hdr);
    pkt.extend_from_slice(&ext);
    pkt.extend_from_slice(&cast_hdr);
    pkt.extend_from_slice(&payload);
    pkt
}

// ── RTCP helpers ─────────────────────────────────────────────────────────────

/// Return the current wall-clock time as a 64-bit NTP timestamp.
///
/// NTP epoch is 1900-01-01; UNIX epoch is 1970-01-01 (offset = 70 years =
/// 2 208 988 800 s). The 32-bit fraction is `subsec_nanos * 2^32 / 10^9`.
fn system_time_ntp() -> (u32, u32) {
    const NTP_UNIX_OFFSET_S: u64 = 2_208_988_800;
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let ntp_sec = (dur.as_secs() + NTP_UNIX_OFFSET_S) as u32;
    let ntp_frac = ((dur.subsec_nanos() as u64) << 32) / 1_000_000_000;
    (ntp_sec, ntp_frac as u32)
}

/// Build a 28-byte RTCP Sender Report (RFC 3550 §6.4.1).
///
/// No reception-report blocks (RC=0) — we're the sender.
fn rtcp_sr(
    ssrc: u32,
    ntp_sec: u32,
    ntp_frac: u32,
    rtp_ts: u32,
    pkt_count: u32,
    byte_count: u32,
) -> [u8; 28] {
    let mut pkt = [0u8; 28];
    // V=2, P=0, RC=0 → 0x80
    pkt[0] = 0x80;
    pkt[1] = RTCP_PT_SR;
    // length = (28 / 4) - 1 = 6
    pkt[2] = 0x00;
    pkt[3] = 0x06;
    pkt[4..8].copy_from_slice(&ssrc.to_be_bytes());
    pkt[8..12].copy_from_slice(&ntp_sec.to_be_bytes());
    pkt[12..16].copy_from_slice(&ntp_frac.to_be_bytes());
    pkt[16..20].copy_from_slice(&rtp_ts.to_be_bytes());
    pkt[20..24].copy_from_slice(&pkt_count.to_be_bytes());
    pkt[24..28].copy_from_slice(&byte_count.to_be_bytes());
    pkt
}

/// Send RTCP SRs for active streams if [`RTCP_INTERVAL`] has elapsed.
///
/// Called at the end of `send_frame` and `send_audio_frame` so the interval
/// is driven by media throughput — avoids spawning a separate timer task.
async fn maybe_send_rtcp_sr(stream: &mut MirrorStreamState) {
    if stream.last_rtcp_at.elapsed() < RTCP_INTERVAL {
        return;
    }
    stream.last_rtcp_at = std::time::Instant::now();
    let (ntp_sec, ntp_frac) = system_time_ntp();

    let video_sr = rtcp_sr(
        stream.video_ssrc,
        ntp_sec,
        ntp_frac,
        stream.last_video_rtp_ts,
        stream.video_pkt_count,
        stream.video_byte_count,
    );
    if let Err(e) = stream.socket.send(&video_sr).await {
        debug!(%e, "mirror: RTCP video SR send dropped");
    }

    if let Some(ssrc) = stream.audio_ssrc {
        let audio_sr = rtcp_sr(
            ssrc,
            ntp_sec,
            ntp_frac,
            stream.last_audio_rtp_ts,
            stream.audio_pkt_count,
            stream.audio_byte_count,
        );
        if let Err(e) = stream.socket.send(&audio_sr).await {
            debug!(%e, "mirror: RTCP audio SR send dropped");
        }
    }
}

/// Returns `true` if `pkt` is an RTCP PLI for our video SSRC.
///
/// PLI: V=2 P=0 FMT=1 PT=206, 12 bytes, media SSRC in bytes 8..12.
fn is_rtcp_pli(pkt: &[u8], video_ssrc: u32) -> bool {
    pkt.len() >= 12
        && (pkt[0] & 0xE0) == 0x80   // V=2, P=0
        && (pkt[0] & 0x1F) == 1       // FMT=1
        && pkt[1] == RTCP_PT_PSFB     // PT=206
        && u32::from_be_bytes(pkt[8..12].try_into().unwrap()) == video_ssrc
}

/// Background task: receive RTCP packets from the Chromecast and set
/// `keyframe_requested` on PLI. Holds a clone of the `Arc<UdpSocket>`;
/// exits when the socket is dropped (Arc count reaches 0 on session stop)
/// or on a fatal receive error.
async fn rtcp_recv_task(
    socket: Arc<UdpSocket>,
    video_ssrc: u32,
    keyframe_requested: Arc<AtomicBool>,
) {
    let mut buf = [0u8; 1500];
    loop {
        match socket.recv_from(&mut buf).await {
            Ok((n, src)) => {
                let pkt = &buf[..n];
                let hex_preview: String = pkt.iter().take(16)
                    .map(|b| format!("{b:02x}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                debug!(n, %src, first_bytes = %hex_preview, "mirror: UDP recv from Chromecast");
                if is_rtcp_pli(pkt, video_ssrc) {
                    debug!("mirror: PLI received — flagging keyframe request");
                    keyframe_requested.store(true, Ordering::Relaxed);
                }
            }
            Err(e) => {
                debug!(%e, "mirror: RTCP recv loop exiting");
                break;
            }
        }
    }
}

