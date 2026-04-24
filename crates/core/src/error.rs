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

    #[error("Capture error: {0}")]
    Capture(String),

    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("Encoder error: {0}")]
    Encoder(String),

    #[error("Device not found: {0}")]
    DeviceNotFound(String),

    #[error("Hls error: {0}")]
    Hls(String),

    #[error("M3u8 Error: {0}")]
    M3u8(String),
    
    #[error("Unsupported codec: {codec:?} for protocol {protocol}")]
    UnsupportedCodec {
        codec: crate::Codec,
        protocol: &'static str,
    },

    #[error("Session already active for device {0}")]
    SessionAlreadyActive(String),

    #[error("No active session")]
    NoActiveSession,

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, FerricastError>;
