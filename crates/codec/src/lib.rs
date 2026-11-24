//! Backend selection for the ferricast encoders and decoders.
//!
//! Each codec job (H.264 encode, AAC decode, ...) has a [`Registry`]:
//! the backends compiled into this build that can do it, in the order
//! automatic selection prefers them, plus the backends this build
//! leaves out (so a selection naming one fails saying which cargo
//! feature brings it in). The `Selecting*` types are the bodies of the
//! codec facades (`ferricast_encoder::H264Encoder`, ...): they read
//! [`ferricast_core::BackendSelection`] from the codec configuration,
//! open the backend it resolves to, and fall back to the next backend
//! when the selection allows it.

pub mod audio;
pub mod bitstream;
pub mod facade;
pub mod registry;
pub mod session;
pub mod video_decoder;
pub mod video_encoder;

pub use audio::{
    AudioDecoderRegistry, AudioEncoderRegistry, SelectingAudioDecoder, SelectingAudioEncoder,
};
pub use registry::{
    Availability, BackendReport, Candidate, Missing, OpenFn, Plan, ProbeFn, Registry,
};
pub use session::{
    AudioDecoderSession, AudioEncoderSession, VideoDecoderSession, VideoEncoderSession,
    open_audio_decoder, open_audio_encoder, open_video_decoder, open_video_encoder,
};
pub use video_decoder::{SelectingVideoDecoder, VideoDecoderRegistry};
pub use video_encoder::{SelectingVideoEncoder, VideoEncoderRegistry};
