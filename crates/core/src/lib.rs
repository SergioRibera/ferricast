pub mod adaptive;
pub mod capture;
pub mod device;
pub mod discovery;
pub mod encoder;
pub mod error;
pub mod frame;
pub mod net;
pub mod protocol;
pub mod session;

pub use adaptive::AdaptiveBitrateState;
pub use capture::{CaptureConfig, CaptureSource, ScreenCapture, WindowIdentifier};
pub use device::{Device, DeviceCapabilities, DiscoveryEvent, H264Profile};
pub use discovery::{Discovery, MdnsDiscovery};
pub use encoder::{EncoderConfig, VideoEncoder};
pub use error::{FerricastError, Result};
pub use frame::{
    AudioCodec, AudioFrame, CapturedFrame, DmaBufImporter, DmaBufPlane, EncodedFrame, GpuFrame,
    PixelFormat, RawFrame,
};
pub use net::local_addr_for;
pub use protocol::{Codec, ProtocolHandler};
pub use session::{CastSession, StreamConfig};
