use thiserror::Error;

#[derive(Debug, Error)]
pub enum FerricastError {
    #[error("Discovery error: {0}")]
    Discovery(String),

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Streaming error: {0}")]
    Streaming(String),

    #[error("Encoding error: {0}")]
    Encoding(String),

    #[error("Rtsp error: {0}")]
    Rtsp(String),

    #[error("Tlv error: {0}")]
    Tlv(String),

    #[error("Capture error: {0}")]
    Capture(String),

    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("Encoder error: {0}")]
    Encoder(String),

    #[error("Device not found: {0}")]
    DeviceNotFound(String),

    /// The receiver or its user ended the session on purpose (another app
    /// took over, "Stop casting"). Not a failure: the manager must not
    /// reconnect.
    #[error("Stopped by the receiver: {0}")]
    RemoteStopped(String),

    #[error("Hls error: {0}")]
    Hls(String),

    #[error("M3u8 Error: {0}")]
    M3u8(String),

    #[error("Receiver error: {0}")]
    Receiver(String),

    #[error("Decode error: {0}")]
    Decode(String),

    #[error("Demux error: {0}")]
    Demux(String),

    #[error("Pull error: {0}")]
    Pull(String),

    #[error("Unsupported codec: {codec:?} for protocol {protocol}")]
    UnsupportedCodec {
        codec: crate::Codec,
        protocol: &'static str,
    },

    /// The receiver wants an access password before it streams.
    /// `setup_stream` fails with this when the receiver challenged the
    /// request (`rejected` when a password was sent and refused); the
    /// session then waits for `submit_pairing(PairingResponse::Password)`
    /// and `setup_stream` can be called again on the same session.
    #[error("{}", if *rejected { "the receiver rejected the password" } else { "the receiver requires a password" })]
    CredentialRequired { rejected: bool },

    #[error("Session already active for device {0}")]
    SessionAlreadyActive(String),

    #[error("No active session")]
    NoActiveSession,

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error(transparent)]
    Frame(#[from] crate::frame::FrameError),

    #[error(transparent)]
    Codec(#[from] crate::codec::CodecError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, FerricastError>;
