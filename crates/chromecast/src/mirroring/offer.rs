//! OFFER / ANSWER messages on `urn:x-cast:com.google.cast.webrtc`
//! (openscreen `offer_messages.cc`, `answer_messages.cc`).

use serde::{Deserialize, Serialize};

use super::rtp::RtpPayloadType;

/// Video RTP timebase mandated by the Cast Streaming spec.
pub(crate) const VIDEO_TIMEBASE: u32 = 90_000;
/// Playout delay requested from the receiver (openscreen default).
pub(crate) const TARGET_DELAY_MS: u32 = 400;
/// Lowest audio bit rate the mirroring spec allows.
const MIN_AUDIO_BIT_RATE: u32 = 32_000;

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub(crate) struct OfferEnvelope {
    #[serde(rename = "type")]
    pub(crate) msg_type: &'static str,
    pub(crate) seq_num: u32,
    pub(crate) offer: Offer,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Offer {
    pub(crate) cast_mode: &'static str,
    pub(crate) supported_streams: Vec<StreamOffer>,
}

/// Fields shared by audio and video stream offers.
#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CommonStream {
    pub(crate) index: u32,
    pub(crate) codec_name: &'static str,
    pub(crate) rtp_profile: &'static str,
    pub(crate) rtp_payload_type: u8,
    pub(crate) ssrc: u32,
    pub(crate) target_delay: u32,
    pub(crate) aes_key: String,
    pub(crate) aes_iv_mask: String,
    pub(crate) receiver_rtcp_event_log: bool,
    pub(crate) time_base: String,
    pub(crate) channels: u32,
}

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Resolution {
    pub(crate) width: u32,
    pub(crate) height: u32,
}

#[derive(Serialize, Debug, Clone)]
#[serde(tag = "type")]
pub(crate) enum StreamOffer {
    #[serde(rename = "audio_source")]
    Audio {
        #[serde(flatten)]
        common: CommonStream,
        #[serde(rename = "bitRate")]
        bit_rate: u32,
    },
    #[serde(rename = "video_source")]
    Video {
        #[serde(flatten)]
        common: CommonStream,
        #[serde(rename = "maxFrameRate")]
        max_frame_rate: String,
        #[serde(rename = "maxBitRate")]
        max_bit_rate: u32,
        resolutions: Vec<Resolution>,
    },
}

impl StreamOffer {
    #[must_use]
    pub(crate) fn index(&self) -> u32 {
        match self {
            Self::Audio { common, .. } | Self::Video { common, .. } => common.index,
        }
    }
}

/// Parameters of the Opus stream to offer.
#[derive(Debug, Clone)]
pub(crate) struct AudioOfferParams {
    pub(crate) ssrc: u32,
    pub(crate) aes_key: [u8; 16],
    pub(crate) aes_iv_mask: [u8; 16],
    pub(crate) sample_rate: u32,
    pub(crate) channels: u16,
    pub(crate) bit_rate: u32,
}

/// Parameters of the H.264 stream to offer.
#[derive(Debug, Clone)]
pub(crate) struct VideoOfferParams {
    pub(crate) ssrc: u32,
    pub(crate) aes_key: [u8; 16],
    pub(crate) aes_iv_mask: [u8; 16],
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) fps: u32,
    pub(crate) max_bit_rate: u32,
}

/// Builds the OFFER. Audio streams come first, as in openscreen, so audio
/// takes index 0 when present.
#[must_use]
pub(crate) fn build_offer(
    seq_num: u32,
    audio: Option<&AudioOfferParams>,
    video: &VideoOfferParams,
) -> OfferEnvelope {
    let mut streams = Vec::with_capacity(2);
    if let Some(a) = audio {
        streams.push(StreamOffer::Audio {
            common: CommonStream {
                index: 0,
                codec_name: "opus",
                rtp_profile: "cast",
                rtp_payload_type: RtpPayloadType::Opus.value(),
                ssrc: a.ssrc,
                target_delay: TARGET_DELAY_MS,
                aes_key: hex16(&a.aes_key),
                aes_iv_mask: hex16(&a.aes_iv_mask),
                receiver_rtcp_event_log: true,
                time_base: format!("1/{}", a.sample_rate),
                channels: u32::from(a.channels),
            },
            bit_rate: a.bit_rate.max(MIN_AUDIO_BIT_RATE),
        });
    }
    let video_index = u32::try_from(streams.len()).unwrap_or(0);
    streams.push(StreamOffer::Video {
        common: CommonStream {
            index: video_index,
            codec_name: "h264",
            rtp_profile: "cast",
            rtp_payload_type: RtpPayloadType::H264.value(),
            ssrc: video.ssrc,
            target_delay: TARGET_DELAY_MS,
            aes_key: hex16(&video.aes_key),
            aes_iv_mask: hex16(&video.aes_iv_mask),
            receiver_rtcp_event_log: true,
            time_base: format!("1/{VIDEO_TIMEBASE}"),
            channels: 1,
        },
        max_frame_rate: format!("{}/1", video.fps.max(1)),
        max_bit_rate: video.max_bit_rate,
        resolutions: vec![Resolution {
            width: video.width,
            height: video.height,
        }],
    });
    OfferEnvelope {
        msg_type: "OFFER",
        seq_num,
        offer: Offer {
            cast_mode: "mirroring",
            supported_streams: streams,
        },
    }
}

/// Lowercase hex, 32 characters, as `aesKey`/`aesIvMask` require.
#[must_use]
pub(crate) fn hex16(bytes: &[u8; 16]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::with_capacity(32), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct AnswerEnvelope {
    seq_num: Option<u32>,
    result: Option<String>,
    answer: Option<AnswerBody>,
    error: Option<AnswerError>,
}

#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AnswerBody {
    pub(crate) udp_port: u16,
    pub(crate) send_indexes: Vec<u32>,
    pub(crate) ssrcs: Vec<u32>,
}

#[derive(Deserialize, Debug)]
struct AnswerError {
    code: Option<i64>,
    description: Option<String>,
}

/// Why an ANSWER could not be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AnswerProblem {
    Malformed(String),
    SequenceMismatch {
        expected: u32,
        got: Option<u32>,
    },
    Rejected(String),
    /// `sendIndexes` and `ssrcs` differ in length, or name no offered stream.
    InvalidStreams,
}

impl std::fmt::Display for AnswerProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(e) => write!(f, "malformed ANSWER: {e}"),
            Self::SequenceMismatch { expected, got } => {
                write!(f, "ANSWER seqNum {got:?} does not match OFFER {expected}")
            }
            Self::Rejected(why) => write!(f, "ANSWER rejected the OFFER: {why}"),
            Self::InvalidStreams => f.write_str("ANSWER selects no valid offered stream"),
        }
    }
}

/// Stream selection accepted by the receiver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AcceptedAnswer {
    pub(crate) udp_port: u16,
    /// Receiver SSRC per accepted offered stream index.
    pub(crate) streams: Vec<(u32, u32)>,
}

impl AcceptedAnswer {
    #[must_use]
    pub(crate) fn receiver_ssrc_for(&self, index: u32) -> Option<u32> {
        self.streams
            .iter()
            .find_map(|&(i, ssrc)| (i == index).then_some(ssrc))
    }
}

/// Parses and validates an ANSWER payload against the OFFER that was sent.
///
/// # Errors
/// Returns why the ANSWER is unusable.
pub(crate) fn parse_answer(
    payload: &str,
    expected_seq: u32,
    offered: &[u32],
) -> Result<AcceptedAnswer, AnswerProblem> {
    let env: AnswerEnvelope =
        serde_json::from_str(payload).map_err(|e| AnswerProblem::Malformed(e.to_string()))?;
    if env.seq_num != Some(expected_seq) {
        return Err(AnswerProblem::SequenceMismatch {
            expected: expected_seq,
            got: env.seq_num,
        });
    }
    if env.result.as_deref() != Some("ok") {
        let why = env.error.map_or_else(
            || format!("result={:?}", env.result),
            |e| format!("code={:?} {}", e.code, e.description.unwrap_or_default()),
        );
        return Err(AnswerProblem::Rejected(why));
    }
    let body = env
        .answer
        .ok_or_else(|| AnswerProblem::Malformed("result=ok without answer".into()))?;
    if body.send_indexes.len() != body.ssrcs.len() {
        return Err(AnswerProblem::InvalidStreams);
    }
    let streams: Vec<(u32, u32)> = body
        .send_indexes
        .iter()
        .copied()
        .zip(body.ssrcs.iter().copied())
        .filter(|(i, _)| offered.contains(i))
        .collect();
    if streams.is_empty() {
        return Err(AnswerProblem::InvalidStreams);
    }
    Ok(AcceptedAnswer {
        udp_port: body.udp_port,
        streams,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn video() -> VideoOfferParams {
        VideoOfferParams {
            ssrc: 70_000,
            aes_key: [0xAB; 16],
            aes_iv_mask: [0x01; 16],
            width: 1280,
            height: 720,
            fps: 30,
            max_bit_rate: 5_000_000,
        }
    }

    #[test]
    fn offer_json_matches_openscreen_fields() {
        let audio = AudioOfferParams {
            ssrc: 42,
            aes_key: [0; 16],
            aes_iv_mask: [0xFF; 16],
            sample_rate: 48_000,
            channels: 2,
            bit_rate: 128_000,
        };
        let env = build_offer(7, Some(&audio), &video());
        let v = serde_json::to_value(&env).unwrap();
        assert_eq!(v["type"], "OFFER");
        assert_eq!(v["seqNum"], 7);
        assert_eq!(v["offer"]["castMode"], "mirroring");
        let a = &v["offer"]["supportedStreams"][0];
        assert_eq!(a["type"], "audio_source");
        assert_eq!(a["index"], 0);
        assert_eq!(a["codecName"], "opus");
        assert_eq!(a["rtpProfile"], "cast");
        assert_eq!(a["rtpPayloadType"], 96);
        assert_eq!(a["ssrc"], 42);
        assert_eq!(a["timeBase"], "1/48000");
        assert_eq!(a["channels"], 2);
        assert_eq!(a["bitRate"], 128_000);
        assert_eq!(a["targetDelay"], 400);
        assert_eq!(a["aesKey"], "00000000000000000000000000000000");
        assert_eq!(a["aesIvMask"], "ffffffffffffffffffffffffffffffff");
        let vid = &v["offer"]["supportedStreams"][1];
        assert_eq!(vid["type"], "video_source");
        assert_eq!(vid["index"], 1);
        assert_eq!(vid["codecName"], "h264");
        assert_eq!(vid["rtpPayloadType"], 101);
        assert_eq!(vid["timeBase"], "1/90000");
        assert_eq!(vid["maxFrameRate"], "30/1");
        assert_eq!(vid["maxBitRate"], 5_000_000);
        assert_eq!(vid["resolutions"][0]["width"], 1280);
        assert_eq!(vid["resolutions"][0]["height"], 720);
        assert_eq!(vid["aesKey"].as_str().unwrap().len(), 32);
        assert!(vid.get("videoCodecParams").is_none());
    }

    #[test]
    fn video_only_offer_uses_index_zero() {
        let v = serde_json::to_value(build_offer(1, None, &video())).unwrap();
        let streams = v["offer"]["supportedStreams"].as_array().unwrap();
        assert_eq!(streams.len(), 1);
        assert_eq!(streams[0]["index"], 0);
    }

    #[test]
    fn answer_validation() {
        let ok = r#"{"type":"ANSWER","seqNum":3,"result":"ok",
            "answer":{"udpPort":2344,"sendIndexes":[1],"ssrcs":[4000000000]}}"#;
        let a = parse_answer(ok, 3, &[0, 1]).unwrap();
        assert_eq!(a.udp_port, 2344);
        assert_eq!(a.receiver_ssrc_for(1), Some(4_000_000_000));
        assert_eq!(a.receiver_ssrc_for(0), None, "audio not accepted");

        assert_eq!(
            parse_answer(ok, 4, &[0, 1]),
            Err(AnswerProblem::SequenceMismatch {
                expected: 4,
                got: Some(3)
            })
        );
        assert_eq!(
            parse_answer(ok, 3, &[0]),
            Err(AnswerProblem::InvalidStreams)
        );

        let err = r#"{"type":"ANSWER","seqNum":3,"result":"error",
            "error":{"code":2,"description":"no codec"}}"#;
        assert!(
            matches!(parse_answer(err, 3, &[0]), Err(AnswerProblem::Rejected(m)) if m.contains("no codec"))
        );

        let mismatch = r#"{"seqNum":3,"result":"ok",
            "answer":{"udpPort":1,"sendIndexes":[0,1],"ssrcs":[5]}}"#;
        assert_eq!(
            parse_answer(mismatch, 3, &[0, 1]),
            Err(AnswerProblem::InvalidStreams)
        );
    }
}
