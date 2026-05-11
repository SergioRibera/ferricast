//! CASTv2 protocol message types and wire-format serialization.
//!
//! The Chromecast protocol uses Protocol Buffers for the outer message envelope
//! (`CastMessage`) and JSON payloads inside the `payload_utf8` field for
//! application-level messaging.
//!
//! Wire format: each message is length-prefixed with a 4-byte big-endian
//! unsigned integer followed by the protobuf-encoded `CastMessage`.

use bytes::{Buf, BufMut, BytesMut};
use prost::Message;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, trace};

#[derive(Debug, Error)]
pub enum CastV2Error {
    #[error("protobuf encode error: {0}")]
    Encode(#[from] prost::EncodeError),

    #[error("protobuf decode error: {0}")]
    Decode(#[from] prost::DecodeError),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("incomplete frame: need {needed} bytes, have {have}")]
    Incomplete { needed: usize, have: usize },

    #[error("frame too large: {size} bytes (max {max})")]
    FrameTooLarge { size: usize, max: usize },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, CastV2Error>;

// ---------------------------------------------------------------------------
// Maximum frame size (Chromecast uses 64 KiB default, but can go up to 1 MiB
// for media payloads).
// ---------------------------------------------------------------------------

pub const MAX_MESSAGE_SIZE: usize = 1024 * 1024; // 1 MiB
pub const LENGTH_PREFIX_SIZE: usize = 4;

/// Payload type enumeration matching the CASTv2 `.proto` definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, prost::Enumeration)]
#[repr(i32)]
pub enum PayloadType {
    String = 0,
    Binary = 1,
}

/// The outer envelope for every CASTv2 message on the wire.
///
/// Field numbers match the official `cast_channel.proto`:
///   1 = protocol_version
///   2 = source_id
///   3 = destination_id
///   4 = namespace
///   5 = payload_type
///   6 = payload_utf8
///   7 = payload_binary
///
/// `protocol_version` and `payload_type` are wrapped in `Option`
/// even though they're conceptually required — that forces prost to
/// emit the tag on the wire even when the value is the proto3
/// default (0). The official schema is proto2 with `required`
/// semantics; some chromecast firmwares reject CastMessages that
/// lack tag=1 / tag=5 entirely (which is what plain `i32` fields do
/// when they happen to be zero — `CASTV2_1_0 = 0` and `STRING = 0`
/// are the only values we ever send for these). Without this, the
/// receiver silently FINs the TLS channel right after our LAUNCH
/// message lands.
#[derive(Clone, PartialEq, Message)]
pub struct CastMessage {
    #[prost(enumeration = "ProtocolVersion", optional, tag = "1")]
    pub protocol_version: Option<i32>,

    #[prost(string, tag = "2")]
    pub source_id: String,

    #[prost(string, tag = "3")]
    pub destination_id: String,

    #[prost(string, tag = "4")]
    pub namespace: String,

    #[prost(enumeration = "PayloadType", optional, tag = "5")]
    pub payload_type: Option<i32>,

    #[prost(string, optional, tag = "6")]
    pub payload_utf8: Option<String>,

    #[prost(bytes = "vec", optional, tag = "7")]
    pub payload_binary: Option<Vec<u8>>,
}


/// Protocol version enum (only CASTV2_1_0 is used in practice).
#[derive(Debug, Clone, Copy, PartialEq, Eq, prost::Enumeration)]
#[repr(i32)]
pub enum ProtocolVersion {
    Castv21_0 = 0,
}

// ---------------------------------------------------------------------------
// Well-known namespaces
// ---------------------------------------------------------------------------

pub mod namespace {
    pub const CONNECTION: &str = "urn:x-cast:com.google.cast.tp.connection";
    pub const HEARTBEAT: &str = "urn:x-cast:com.google.cast.tp.heartbeat";
    pub const RECEIVER: &str = "urn:x-cast:com.google.cast.receiver";
    pub const MEDIA: &str = "urn:x-cast:com.google.cast.media";
}

// ---------------------------------------------------------------------------
// Well-known sender/receiver IDs
// ---------------------------------------------------------------------------

pub const DEFAULT_SENDER_ID: &str = "sender-0";
pub const DEFAULT_RECEIVER_ID: &str = "receiver-0";
pub const TRANSPORT_ID_PREFIX: &str = "sender-";

/// A generic JSON payload wrapper -- every Cast JSON message has a `type`
/// field plus an optional `requestId`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericPayload {
    #[serde(rename = "type")]
    pub msg_type: String,

    #[serde(rename = "requestId", skip_serializing_if = "Option::is_none")]
    pub request_id: Option<i64>,

    /// Capture all additional fields so we can forward them transparently.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

// -- Connection namespace payloads ------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectPayload {
    #[serde(rename = "type")]
    pub msg_type: String,

    #[serde(rename = "userAgent", skip_serializing_if = "Option::is_none")]
    pub user_agent: Option<String>,

    #[serde(rename = "connType", skip_serializing_if = "Option::is_none")]
    pub conn_type: Option<i32>,

    #[serde(rename = "origin", skip_serializing_if = "Option::is_none")]
    pub origin: Option<serde_json::Value>,
}

impl ConnectPayload {
    /// Minimal CONNECT — just `{"type":"CONNECT"}`. We initially
    /// shipped userAgent / connType / origin because the official
    /// proto schema documents them, but several chromecast firmwares
    /// (notably 1st-gen audio devices and certain Google TV
    /// versions) silently close the TLS channel when those fields
    /// are present from a sender they don't recognise. Both
    /// pychromecast and rust_cast send plain `{type:CONNECT}` and
    /// it's the lowest-common-denominator form.
    pub fn connect() -> Self {
        Self {
            msg_type: "CONNECT".into(),
            user_agent: None,
            conn_type: None,
            origin: None,
        }
    }

    pub fn close() -> Self {
        Self {
            msg_type: "CLOSE".into(),
            user_agent: None,
            conn_type: None,
            origin: None,
        }
    }
}

// -- Heartbeat namespace payloads -------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatPayload {
    #[serde(rename = "type")]
    pub msg_type: String,
}

impl HeartbeatPayload {
    pub fn ping() -> Self {
        Self {
            msg_type: "PING".into(),
        }
    }

    pub fn pong() -> Self {
        Self {
            msg_type: "PONG".into(),
        }
    }
}

// -- Receiver namespace payloads --------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiverStatusPayload {
    #[serde(rename = "type")]
    pub msg_type: String,

    #[serde(rename = "requestId", skip_serializing_if = "Option::is_none")]
    pub request_id: Option<i64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<ReceiverStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiverStatus {
    #[serde(default)]
    pub applications: Vec<ApplicationInfo>,

    #[serde(rename = "isActiveInput", skip_serializing_if = "Option::is_none")]
    pub is_active_input: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume: Option<VolumeInfo>,
}

/// One application entry inside a `RECEIVER_STATUS` payload. Every
/// field except `app_id` is optional because real receivers send
/// progressively-populated entries during the launch sequence —
/// e.g. an early status may have `appId` + `transportId` but lack
/// `displayName` and `sessionId`. With strict types serde would
/// reject the whole payload and `wait_for_app` would silently
/// continue, never finding the app it was looking for.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplicationInfo {
    #[serde(rename = "appId")]
    pub app_id: String,

    #[serde(rename = "displayName", default)]
    pub display_name: String,

    #[serde(rename = "transportId", default)]
    pub transport_id: String,

    #[serde(rename = "sessionId", default)]
    pub session_id: String,

    #[serde(default)]
    pub namespaces: Vec<NamespaceEntry>,

    #[serde(rename = "statusText", skip_serializing_if = "Option::is_none")]
    pub status_text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceEntry {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub muted: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaunchPayload {
    #[serde(rename = "type")]
    pub msg_type: String,

    #[serde(rename = "requestId")]
    pub request_id: i64,

    #[serde(rename = "appId")]
    pub app_id: String,
}

impl LaunchPayload {
    pub fn new(request_id: i64, app_id: &str) -> Self {
        Self {
            msg_type: "LAUNCH".into(),
            request_id,
            app_id: app_id.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetStatusPayload {
    #[serde(rename = "type")]
    pub msg_type: String,

    #[serde(rename = "requestId")]
    pub request_id: i64,
}

impl GetStatusPayload {
    pub fn new(request_id: i64) -> Self {
        Self {
            msg_type: "GET_STATUS".into(),
            request_id,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopPayload {
    #[serde(rename = "type")]
    pub msg_type: String,

    #[serde(rename = "requestId")]
    pub request_id: i64,

    #[serde(rename = "sessionId")]
    pub session_id: String,
}

impl StopPayload {
    pub fn new(request_id: i64, session_id: &str) -> Self {
        Self {
            msg_type: "STOP".into(),
            request_id,
            session_id: session_id.into(),
        }
    }
}

// -- Media namespace payloads -----------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaStatusPayload {
    #[serde(rename = "type")]
    pub msg_type: String,

    #[serde(rename = "requestId", skip_serializing_if = "Option::is_none")]
    pub request_id: Option<i64>,

    #[serde(default)]
    pub status: Vec<MediaStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaStatus {
    #[serde(rename = "mediaSessionId")]
    pub media_session_id: i64,

    #[serde(rename = "playerState", skip_serializing_if = "Option::is_none")]
    pub player_state: Option<String>,

    #[serde(rename = "currentTime", skip_serializing_if = "Option::is_none")]
    pub current_time: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub media: Option<MediaInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaInfo {
    #[serde(rename = "contentId")]
    pub content_id: String,

    #[serde(rename = "contentType")]
    pub content_type: String,

    #[serde(rename = "streamType", skip_serializing_if = "Option::is_none")]
    pub stream_type: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadPayload {
    #[serde(rename = "type")]
    pub msg_type: String,

    #[serde(rename = "requestId")]
    pub request_id: i64,

    pub media: MediaInfo,

    #[serde(rename = "autoplay")]
    pub autoplay: bool,

    #[serde(rename = "currentTime")]
    pub current_time: f64,
}

impl LoadPayload {
    pub fn new(request_id: i64, media: MediaInfo) -> Self {
        Self {
            msg_type: "LOAD".into(),
            request_id,
            media,
            autoplay: true,
            current_time: 0.0,
        }
    }
}

pub const DEFAULT_MEDIA_RECEIVER_APP_ID: &str = "CC1AD845";
pub const MIRRORING_APP_ID: &str = "0F5096E8";

impl CastMessage {
    /// Create a new string-payload message.
    pub fn new_json(
        source_id: &str,
        destination_id: &str,
        namespace: &str,
        payload: &impl Serialize,
    ) -> Result<Self> {
        let json = serde_json::to_string(payload)?;
        trace!(
            ns = namespace,
            src = source_id,
            dst = destination_id,
            "cast message payload: {}",
            json
        );
        Ok(Self {
            protocol_version: Some(ProtocolVersion::Castv21_0 as i32),
            source_id: source_id.into(),
            destination_id: destination_id.into(),
            namespace: namespace.into(),
            payload_type: Some(PayloadType::String as i32),
            payload_utf8: Some(json),
            payload_binary: None,
        })
    }

    /// Create a binary-payload message.
    pub fn new_binary(
        source_id: &str,
        destination_id: &str,
        namespace: &str,
        data: Vec<u8>,
    ) -> Self {
        Self {
            protocol_version: Some(ProtocolVersion::Castv21_0 as i32),
            source_id: source_id.into(),
            destination_id: destination_id.into(),
            namespace: namespace.into(),
            payload_type: Some(PayloadType::Binary as i32),
            payload_utf8: None,
            payload_binary: Some(data),
        }
    }

    /// Encode this message into a length-prefixed byte buffer ready for the wire.
    pub fn encode_length_prefixed(&self) -> Result<Vec<u8>> {
        let proto_len = self.encoded_len();
        if proto_len > MAX_MESSAGE_SIZE {
            return Err(CastV2Error::FrameTooLarge {
                size: proto_len,
                max: MAX_MESSAGE_SIZE,
            });
        }

        let mut buf = Vec::with_capacity(LENGTH_PREFIX_SIZE + proto_len);
        buf.put_u32(proto_len as u32);
        self.encode(&mut buf)?;

        debug!(
            ns = %self.namespace,
            size = proto_len,
            "encoded cast message"
        );
        Ok(buf)
    }

    /// Attempt to decode a length-prefixed `CastMessage` from the front of
    /// `buf`.  On success the consumed bytes are drained from `buf`.
    ///
    /// Returns `Ok(None)` if there are not enough bytes yet (the caller should
    /// read more data and retry).
    pub fn decode_length_prefixed(buf: &mut BytesMut) -> Result<Option<Self>> {
        if buf.len() < LENGTH_PREFIX_SIZE {
            return Ok(None);
        }

        let msg_len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;

        if msg_len > MAX_MESSAGE_SIZE {
            return Err(CastV2Error::FrameTooLarge {
                size: msg_len,
                max: MAX_MESSAGE_SIZE,
            });
        }

        let total = LENGTH_PREFIX_SIZE + msg_len;
        if buf.len() < total {
            return Ok(None);
        }

        buf.advance(LENGTH_PREFIX_SIZE);

        let data = buf.split_to(msg_len);
        let msg = CastMessage::decode(data)?;

        debug!(
            ns = %msg.namespace,
            src = %msg.source_id,
            dst = %msg.destination_id,
            "decoded cast message"
        );
        Ok(Some(msg))
    }

    pub fn parse_payload<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        let payload = self
            .payload_utf8
            .as_deref()
            .unwrap_or("{}");
        Ok(serde_json::from_str(payload)?)
    }

    pub fn message_type(&self) -> Option<String> {
        let payload = self.payload_utf8.as_deref()?;
        let parsed: GenericPayload = serde_json::from_str(payload).ok()?;
        Some(parsed.msg_type)
    }
}

/// Build a CONNECT message for the connection namespace.
pub fn connect_message(source_id: &str, destination_id: &str) -> Result<CastMessage> {
    CastMessage::new_json(
        source_id,
        destination_id,
        namespace::CONNECTION,
        &ConnectPayload::connect(),
    )
}

/// Build a CLOSE message for the connection namespace.
pub fn close_message(source_id: &str, destination_id: &str) -> Result<CastMessage> {
    CastMessage::new_json(
        source_id,
        destination_id,
        namespace::CONNECTION,
        &ConnectPayload::close(),
    )
}

/// Build a PING message.
pub fn ping_message() -> Result<CastMessage> {
    CastMessage::new_json(
        DEFAULT_SENDER_ID,
        DEFAULT_RECEIVER_ID,
        namespace::HEARTBEAT,
        &HeartbeatPayload::ping(),
    )
}

/// Build a PONG message.
pub fn pong_message() -> Result<CastMessage> {
    CastMessage::new_json(
        DEFAULT_SENDER_ID,
        DEFAULT_RECEIVER_ID,
        namespace::HEARTBEAT,
        &HeartbeatPayload::pong(),
    )
}

/// Build a GET_STATUS message for the receiver namespace.
pub fn get_status_message(request_id: i64) -> Result<CastMessage> {
    CastMessage::new_json(
        DEFAULT_SENDER_ID,
        DEFAULT_RECEIVER_ID,
        namespace::RECEIVER,
        &GetStatusPayload::new(request_id),
    )
}

/// Build a LAUNCH message for the receiver namespace.
pub fn launch_message(request_id: i64, app_id: &str) -> Result<CastMessage> {
    CastMessage::new_json(
        DEFAULT_SENDER_ID,
        DEFAULT_RECEIVER_ID,
        namespace::RECEIVER,
        &LaunchPayload::new(request_id, app_id),
    )
}

/// Build a STOP message for the receiver namespace.
pub fn stop_app_message(
    request_id: i64,
    session_id: &str,
) -> Result<CastMessage> {
    CastMessage::new_json(
        DEFAULT_SENDER_ID,
        DEFAULT_RECEIVER_ID,
        namespace::RECEIVER,
        &StopPayload::new(request_id, session_id),
    )
}

/// Build a LOAD message for the media namespace.
pub fn load_media_message(
    request_id: i64,
    transport_id: &str,
    media: MediaInfo,
) -> Result<CastMessage> {
    CastMessage::new_json(
        DEFAULT_SENDER_ID,
        transport_id,
        namespace::MEDIA,
        &LoadPayload::new(request_id, media),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::BytesMut;

    #[test]
    fn roundtrip_length_prefixed() {
        let msg = CastMessage::new_json(
            "sender-0",
            "receiver-0",
            namespace::HEARTBEAT,
            &HeartbeatPayload::ping(),
        )
        .unwrap();

        let encoded = msg.encode_length_prefixed().unwrap();
        let mut buf = BytesMut::from(&encoded[..]);
        let decoded = CastMessage::decode_length_prefixed(&mut buf)
            .unwrap()
            .expect("should decode complete message");

        assert_eq!(msg, decoded);
        assert!(buf.is_empty());
    }

    #[test]
    fn incomplete_frame_returns_none() {
        let msg = ping_message().unwrap();
        let encoded = msg.encode_length_prefixed().unwrap();

        let half = encoded.len() / 2;
        let mut buf = BytesMut::from(&encoded[..half]);
        let result = CastMessage::decode_length_prefixed(&mut buf).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn message_type_extraction() {
        let msg = ping_message().unwrap();
        assert_eq!(msg.message_type().as_deref(), Some("PING"));
    }
}
