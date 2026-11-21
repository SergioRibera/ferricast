pub mod adaptive;
pub mod advertiser;
pub mod capture;
pub mod codec;
pub mod control;
pub mod decoder;
pub mod device;
#[cfg(feature = "net")]
pub mod discovery;
pub mod encoder;
pub mod error;
pub mod frame;
#[cfg(feature = "net")]
pub mod net;
pub mod pairing;
#[cfg(feature = "net")]
pub mod protocol;
pub mod puller;
pub mod session;
pub mod sink;
#[cfg(feature = "net")]
pub mod source;

pub use adaptive::AdaptiveBitrateState;
pub use advertiser::{AdvertiseInfo, Advertiser, MdnsAdvertiser};
pub use capture::{
    AudioCapture, AudioCaptureConfig, AudioMuteHandle, AudioSource, CaptureConfig, CaptureSource,
    ScreenCapture, WindowIdentifier,
};
pub use codec::Codec;
pub use control::{
    ControlSession, MediaCommand, PlaybackState, PlaybackStatus, QueueItem, RemoteSender,
    RepeatMode, TrackSelection,
};
pub use decoder::{AudioDecoder, AudioDecoderConfig, DecodedAudio, DecoderConfig, VideoDecoder};
pub use device::{
    AirPlayPairing, Device, DeviceAddresses, DeviceCapabilities, DiscoveryEvent, H264Profile,
    H265Profile, Ipv4Subnet,
};
#[cfg(feature = "net")]
pub use discovery::{Discovery, MdnsDiscovery};
pub use encoder::{AudioEncoder, AudioEncoderConfig, EncoderConfig, VideoEncoder};
pub use error::{FerricastError, Result};
pub use frame::{
    AacChannelConfig, AudioCodec, AudioFrame, BufferLease, CapturedFrame, DmaBufFd, DmaBufId,
    DmaBufImporter, DmaBufPlane, EncodedFrame, GpuFrame, PixelFormat, RawFrame, SourceAudioFormat,
};
#[cfg(feature = "net")]
pub use net::{
    bind_tcp, bind_udp, connect_tcp_any, local_addr_for, local_ipv4_subnets, local_socket_addr_for,
    probe_tcp, unspecified_for,
};
pub use pairing::{PairingChallenge, PairingResponse};
#[cfg(feature = "net")]
pub use protocol::{ProtocolHandler, ReceiverProtocol};
pub use puller::{AudioStreamInfo, MediaInfo, MediaPacket, MediaPuller, PullSpec, VideoStreamInfo};
pub use session::{AudioStreamConfig, CastSession, ConnectOutcome, StreamConfig};
pub use sink::FrameSink;
#[cfg(feature = "net")]
pub use source::{
    EnumerationCapability, Geometry, MonitorInfo, SourceChange, SourceEnumerator, SourceError,
    StubEnumerator, WindowInfo,
};
