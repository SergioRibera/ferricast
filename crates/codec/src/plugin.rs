//! The codecs as the istmo plugin `ferricast.codec`.
//!
//! The plugin is the control plane: what this build can encode and
//! decode ([`Codecs::capabilities`]), what a selection resolves to on
//! this device ([`Codecs::resolve`]), and codec sessions
//! ([`Codecs::open`], keyframe and bitrate requests, [`Codecs::close`]).
//! Any istmo client can drive it: Rust, and the Kotlin / Swift clients
//! istmo generates.
//!
//! Frames never cross the istmo wire, which copies every payload: a
//! session's handle opens the data plane on the Rust side, through the
//! same [`CodecService`] the host runs on ([`CodecService::video_encoder`]
//! and friends), at the cost of one uncontended mutex per frame.
//!
//! ```ignore
//! let service = CodecService::new(catalog);         // see `Catalog`
//! runtime.register_host(CodecsHost::new(service.clone()));
//!
//! let client = CodecsClient::from_runtime(&runtime);
//! let session = client.open_owned(OpenRequest::VideoEncoder(config)).await?;
//! let encoder = service.video_encoder(session.session.id())?;
//! let packet = encoder.lock().expect("not poisoned").encode(frame)?;
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak};

use ferricast_core::{
    Acceleration, AudioCodec, AudioDecoderConfig, AudioEncoderConfig, BackendId, Codec, CodecError,
    CodecTarget, DecoderConfig, Direction, EncoderConfig,
};
use istmo_core::{IstmoError, NativeHandleId, Runtime};
use istmo_macros::{message, owned, plugin};

use crate::audio::{
    AudioDecoderRegistry, AudioEncoderRegistry, SelectingAudioDecoder, SelectingAudioEncoder,
};
use crate::bitstream::NalSyntax;
use crate::registry::BackendReport;
use crate::video_decoder::{SelectingVideoDecoder, VideoDecoderRegistry};
use crate::video_encoder::{SelectingVideoEncoder, VideoEncoderRegistry};

/// The plugin id clients and hosts register under.
pub const PLUGIN_ID: &str = "ferricast.codec";

/// Phantom tag of a codec session handle.
#[derive(Debug, Clone, Copy)]
pub struct CodecSessionTag(std::convert::Infallible);

/// A codec job with the standing of every backend that can do it.
#[message(crate = "::istmo_core")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobReport {
    /// The job.
    pub target: CodecTarget,
    /// Its backends, compiled ones first in automatic order.
    pub backends: Vec<BackendReport>,
}

/// A codec to open, with its configuration (which carries the
/// [`ferricast_core::BackendSelection`]).
#[message(crate = "::istmo_core")]
#[derive(Debug, Clone)]
pub enum OpenRequest {
    /// A video encoder.
    VideoEncoder(EncoderConfig),
    /// A video decoder.
    VideoDecoder(DecoderConfig),
    /// An audio encoder.
    AudioEncoder(AudioEncoderConfig),
    /// An audio decoder.
    AudioDecoder(AudioDecoderConfig),
}

impl OpenRequest {
    /// The job the request is for.
    #[must_use]
    pub fn target(&self) -> CodecTarget {
        match self {
            Self::VideoEncoder(config) => CodecTarget::encode(config.codec),
            Self::VideoDecoder(config) => CodecTarget::decode(config.codec),
            Self::AudioEncoder(config) => CodecTarget::encode(config.codec),
            Self::AudioDecoder(config) => CodecTarget::decode(config.codec),
        }
    }
}

/// The backend a selection resolved to.
#[message(crate = "::istmo_core")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolution {
    /// The job.
    pub target: CodecTarget,
    /// The backend that opened.
    pub backend: BackendId,
    /// Whether it runs on media hardware.
    pub acceleration: Acceleration,
}

/// An open codec session. Drop the owned handle from
/// [`CodecsClient::open_owned`] (or call [`Codecs::close`]) to end it.
#[message(crate = "::istmo_core")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecSession {
    /// The session; pass it to [`CodecService`] for the data plane.
    #[handle(CodecSessionTag)]
    pub session: NativeHandleId,
    /// The backend it runs on.
    pub resolution: Resolution,
}

/// The codec control plane.
#[plugin(name = "ferricast.codec", crate = "::istmo_core")]
pub trait Codecs {
    /// Every job this build can do and every backend for it, probed
    /// for availability on this device.
    async fn capabilities(&self) -> Result<Vec<JobReport>, CodecError>;

    /// Opens the codec `request` describes, reports which backend its
    /// selection picked, and closes it again.
    async fn resolve(&self, request: OpenRequest) -> Result<Resolution, CodecError>;

    /// Opens a codec session.
    #[owned]
    async fn open(&self, request: OpenRequest) -> Result<CodecSession, CodecError>;

    /// Ends a session (dropping its owned handle does the same).
    async fn close(&self, session: NativeHandleId) -> Result<(), CodecError>;

    /// Asks a video encoder session for a keyframe next.
    async fn request_keyframe(&self, session: NativeHandleId) -> Result<(), CodecError>;

    /// Retargets a video encoder session's bitrate.
    async fn set_bitrate(&self, session: NativeHandleId, kbps: u32) -> Result<(), CodecError>;
}

/// Reads the typed [`CodecError`] out of a failed plugin call.
pub trait IstmoCodecError {
    /// The codec error the host returned, or the transport failure
    /// unchanged when the call never reached a codec.
    ///
    /// # Errors
    /// Hands back the [`IstmoError`] when it is not a codec error.
    fn into_codec_error(self) -> Result<CodecError, IstmoError>;
}

impl IstmoCodecError for IstmoError {
    fn into_codec_error(self) -> Result<CodecError, IstmoError> {
        match self {
            IstmoError::PluginError { bytes } => {
                match istmo_core::codec::decode::<CodecError>(&bytes) {
                    Ok((error, _)) => Ok(error),
                    Err(_) => Err(IstmoError::PluginError { bytes }),
                }
            }
            other => Err(other),
        }
    }
}

/// A video decode job: its codec, registry and hand-over syntax.
type VideoDecodeJob = (Codec, fn() -> VideoDecoderRegistry, Option<NalSyntax>);

/// The codec jobs a [`CodecService`] serves: one registry function per
/// job, contributed by the encoder and decoder crates.
#[derive(Default)]
pub struct Catalog {
    video_encoders: Vec<(Codec, fn() -> VideoEncoderRegistry)>,
    video_decoders: Vec<VideoDecodeJob>,
    audio_encoders: Vec<(AudioCodec, fn() -> AudioEncoderRegistry)>,
    audio_decoders: Vec<(AudioCodec, fn() -> AudioDecoderRegistry)>,
}

impl fmt::Debug for Catalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Catalog")
            .field("jobs", &self.jobs())
            .finish()
    }
}

impl Catalog {
    /// Adds a video encode job.
    #[must_use]
    pub fn video_encoder(mut self, codec: Codec, registry: fn() -> VideoEncoderRegistry) -> Self {
        self.video_encoders.push((codec, registry));
        self
    }

    /// Adds a video decode job; `syntax` as for
    /// [`SelectingVideoDecoder::new`].
    #[must_use]
    pub fn video_decoder(
        mut self,
        codec: Codec,
        registry: fn() -> VideoDecoderRegistry,
        syntax: Option<NalSyntax>,
    ) -> Self {
        self.video_decoders.push((codec, registry, syntax));
        self
    }

    /// Adds an audio encode job.
    #[must_use]
    pub fn audio_encoder(
        mut self,
        codec: AudioCodec,
        registry: fn() -> AudioEncoderRegistry,
    ) -> Self {
        self.audio_encoders.push((codec, registry));
        self
    }

    /// Adds an audio decode job.
    #[must_use]
    pub fn audio_decoder(
        mut self,
        codec: AudioCodec,
        registry: fn() -> AudioDecoderRegistry,
    ) -> Self {
        self.audio_decoders.push((codec, registry));
        self
    }

    /// Every job in the catalog.
    #[must_use]
    pub fn jobs(&self) -> Vec<CodecTarget> {
        let video_encode = self
            .video_encoders
            .iter()
            .map(|(c, _)| CodecTarget::encode(*c));
        let video_decode = self
            .video_decoders
            .iter()
            .map(|(c, ..)| CodecTarget::decode(*c));
        let audio_encode = self
            .audio_encoders
            .iter()
            .map(|(c, _)| CodecTarget::encode(*c));
        let audio_decode = self
            .audio_decoders
            .iter()
            .map(|(c, _)| CodecTarget::decode(*c));
        video_encode
            .chain(video_decode)
            .chain(audio_encode)
            .chain(audio_decode)
            .collect()
    }

    fn reports(&self) -> Vec<JobReport> {
        let video_encode = self.video_encoders.iter().map(|(_, r)| r().report());
        let video_decode = self.video_decoders.iter().map(|(_, r, _)| r().report());
        let audio_encode = self.audio_encoders.iter().map(|(_, r)| r().report());
        let audio_decode = self.audio_decoders.iter().map(|(_, r)| r().report());
        self.jobs()
            .into_iter()
            .zip(
                video_encode
                    .chain(video_decode)
                    .chain(audio_encode)
                    .chain(audio_decode),
            )
            .map(|(target, backends)| JobReport { target, backends })
            .collect()
    }

    fn not_in_catalog(target: CodecTarget) -> CodecError {
        CodecError::NoBackendCompiled {
            target,
            features: format!(
                "the codec service was built without a {} {} job",
                target.codec, target.direction
            ),
        }
    }

    /// Opens `request` as a session.
    fn open(&self, request: OpenRequest) -> Result<Session, CodecError> {
        let target = request.target();
        match request {
            OpenRequest::VideoEncoder(config) => {
                let (_, registry) = self
                    .video_encoders
                    .iter()
                    .find(|(codec, _)| *codec == config.codec)
                    .ok_or_else(|| Self::not_in_catalog(target))?;
                let mut encoder = SelectingVideoEncoder::new(registry());
                encoder.configure(&config)?;
                Ok(Session::VideoEncoder(Arc::new(Mutex::new(encoder))))
            }
            OpenRequest::VideoDecoder(config) => {
                let (_, registry, syntax) = self
                    .video_decoders
                    .iter()
                    .find(|(codec, ..)| *codec == config.codec)
                    .ok_or_else(|| Self::not_in_catalog(target))?;
                let mut decoder = SelectingVideoDecoder::new(registry(), *syntax);
                decoder.configure(&config)?;
                Ok(Session::VideoDecoder(Arc::new(Mutex::new(decoder))))
            }
            OpenRequest::AudioEncoder(config) => {
                let (_, registry) = self
                    .audio_encoders
                    .iter()
                    .find(|(codec, _)| *codec == config.codec)
                    .ok_or_else(|| Self::not_in_catalog(target))?;
                let mut encoder = SelectingAudioEncoder::new(registry());
                encoder.configure(&config)?;
                Ok(Session::AudioEncoder(Arc::new(Mutex::new(encoder))))
            }
            OpenRequest::AudioDecoder(config) => {
                let (_, registry) = self
                    .audio_decoders
                    .iter()
                    .find(|(codec, _)| *codec == config.codec)
                    .ok_or_else(|| Self::not_in_catalog(target))?;
                let mut decoder = SelectingAudioDecoder::new(registry());
                decoder.configure(&config)?;
                Ok(Session::AudioDecoder(Arc::new(Mutex::new(decoder))))
            }
        }
    }
}

/// A shared codec session; lock it for each call.
pub type Shared<T> = Arc<Mutex<T>>;

enum Session {
    VideoEncoder(Shared<SelectingVideoEncoder>),
    VideoDecoder(Shared<SelectingVideoDecoder>),
    AudioEncoder(Shared<SelectingAudioEncoder>),
    AudioDecoder(Shared<SelectingAudioDecoder>),
}

impl Session {
    fn resolution(&self) -> Resolution {
        let (target, backend) = match self {
            Self::VideoEncoder(s) => {
                let s = lock(s);
                (s.target(), s.backend())
            }
            Self::VideoDecoder(s) => {
                let s = lock(s);
                (s.target(), s.backend())
            }
            Self::AudioEncoder(s) => {
                let s = lock(s);
                (s.target(), s.backend())
            }
            Self::AudioDecoder(s) => {
                let s = lock(s);
                (s.target(), s.backend())
            }
        };
        let (backend, acceleration) =
            backend.expect("invariant: sessions are only stored once configure opened a backend");
        Resolution {
            target,
            backend,
            acceleration,
        }
    }
}

/// A codec session's state is only touched inside one call at a time; a
/// panic in a backend poisons nothing the next call cannot use.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

struct Inner {
    catalog: Catalog,
    sessions: Mutex<HashMap<NativeHandleId, Session>>,
    runtime: OnceLock<Weak<Runtime>>,
}

/// The `ferricast.codec` host and the Rust data plane of its sessions.
/// Cloning shares the same sessions.
#[derive(Clone)]
pub struct CodecService(Arc<Inner>);

impl fmt::Debug for CodecService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CodecService")
            .field("catalog", &self.0.catalog)
            .field("sessions", &lock(&self.0.sessions).len())
            .finish()
    }
}

/// Why a session id does not name a usable session.
fn no_session(session: NativeHandleId, kind: &str) -> CodecError {
    CodecError::NoSuchSession {
        session: session.get(),
        kind: kind.to_owned(),
    }
}

impl CodecService {
    /// A service over `catalog` that allocates session handles from the
    /// process-global istmo runtime once one runs.
    #[must_use]
    pub fn new(catalog: Catalog) -> Self {
        Self(Arc::new(Inner {
            catalog,
            sessions: Mutex::new(HashMap::new()),
            runtime: OnceLock::new(),
        }))
    }

    /// A service over `catalog` bound to `runtime` (a non-global one,
    /// e.g. [`Runtime::mock`] in tests).
    #[must_use]
    pub fn with_runtime(catalog: Catalog, runtime: &Arc<Runtime>) -> Self {
        let service = Self::new(catalog);
        let bound = service.0.runtime.set(service.attach(runtime));
        debug_assert!(bound.is_ok(), "a new service has no runtime yet");
        service
    }

    /// Installs on `runtime` the hook that closes a session when its
    /// owned handle is dropped.
    fn attach(&self, runtime: &Arc<Runtime>) -> Weak<Runtime> {
        let inner = Arc::downgrade(&self.0);
        runtime.install_native_handle_release_hook(Arc::new(move |id: NativeHandleId| {
            // Every release reaches every hook; ids this service never
            // handed out are not in the map.
            if let Some(inner) = inner.upgrade() {
                let closed = lock(&inner.sessions).remove(&id);
                drop(closed);
            }
        }));
        Arc::downgrade(runtime)
    }

    /// The catalog this service serves.
    #[must_use]
    pub fn catalog(&self) -> &Catalog {
        &self.0.catalog
    }

    /// The runtime sessions are allocated from, installing the hook that
    /// closes a session when its owned handle is dropped.
    fn runtime(&self) -> Result<Arc<Runtime>, CodecError> {
        if let Some(bound) = self.0.runtime.get() {
            return bound
                .upgrade()
                .ok_or_else(|| CodecError::SessionsUnavailable {
                    reason: "the istmo runtime it was bound to has shut down".to_owned(),
                });
        }
        // Not bound yet: adopt the global runtime once one runs. Two
        // racing first calls both attach a hook; the spare one only
        // repeats removals the first already did.
        let runtime = Runtime::global().map_err(|error| CodecError::SessionsUnavailable {
            reason: format!("no istmo runtime is running ({error})"),
        })?;
        let weak = self.attach(&runtime);
        self.0.runtime.set(weak).ok();
        Ok(runtime)
    }

    fn with_session<T>(
        &self,
        session: NativeHandleId,
        kind: &str,
        pick: impl FnOnce(&Session) -> Option<T>,
    ) -> Result<T, CodecError> {
        lock(&self.0.sessions)
            .get(&session)
            .and_then(pick)
            .ok_or_else(|| no_session(session, kind))
    }

    /// The video encoder of `session`, for the data plane.
    ///
    /// # Errors
    /// [`CodecError::InvalidSelection`] when `session` is not an open
    /// video encoder session.
    pub fn video_encoder(
        &self,
        session: NativeHandleId,
    ) -> Result<Shared<SelectingVideoEncoder>, CodecError> {
        self.with_session(session, "video encoder", |s| match s {
            Session::VideoEncoder(e) => Some(Arc::clone(e)),
            _ => None,
        })
    }

    /// The video decoder of `session`, for the data plane.
    ///
    /// # Errors
    /// [`CodecError::InvalidSelection`] when `session` is not an open
    /// video decoder session.
    pub fn video_decoder(
        &self,
        session: NativeHandleId,
    ) -> Result<Shared<SelectingVideoDecoder>, CodecError> {
        self.with_session(session, "video decoder", |s| match s {
            Session::VideoDecoder(d) => Some(Arc::clone(d)),
            _ => None,
        })
    }

    /// The audio encoder of `session`, for the data plane.
    ///
    /// # Errors
    /// [`CodecError::InvalidSelection`] when `session` is not an open
    /// audio encoder session.
    pub fn audio_encoder(
        &self,
        session: NativeHandleId,
    ) -> Result<Shared<SelectingAudioEncoder>, CodecError> {
        self.with_session(session, "audio encoder", |s| match s {
            Session::AudioEncoder(e) => Some(Arc::clone(e)),
            _ => None,
        })
    }

    /// The audio decoder of `session`, for the data plane.
    ///
    /// # Errors
    /// [`CodecError::InvalidSelection`] when `session` is not an open
    /// audio decoder session.
    pub fn audio_decoder(
        &self,
        session: NativeHandleId,
    ) -> Result<Shared<SelectingAudioDecoder>, CodecError> {
        self.with_session(session, "audio decoder", |s| match s {
            Session::AudioDecoder(d) => Some(Arc::clone(d)),
            _ => None,
        })
    }
}

impl Codecs for CodecService {
    async fn capabilities(&self) -> Result<Vec<JobReport>, CodecError> {
        Ok(self.0.catalog.reports())
    }

    async fn resolve(&self, request: OpenRequest) -> Result<Resolution, CodecError> {
        Ok(self.0.catalog.open(request)?.resolution())
    }

    async fn open(&self, request: OpenRequest) -> Result<CodecSession, CodecError> {
        let runtime = self.runtime()?;
        let session = self.0.catalog.open(request)?;
        let resolution = session.resolution();
        let id = runtime.alloc_native_handle_id();
        lock(&self.0.sessions).insert(id, session);
        Ok(CodecSession {
            session: id,
            resolution,
        })
    }

    async fn close(&self, session: NativeHandleId) -> Result<(), CodecError> {
        lock(&self.0.sessions)
            .remove(&session)
            .map(drop)
            .ok_or_else(|| no_session(session, "codec"))
    }

    async fn request_keyframe(&self, session: NativeHandleId) -> Result<(), CodecError> {
        lock(&*self.video_encoder(session)?).request_keyframe();
        Ok(())
    }

    async fn set_bitrate(&self, session: NativeHandleId, kbps: u32) -> Result<(), CodecError> {
        lock(&*self.video_encoder(session)?).set_bitrate_kbps(kbps)
    }
}

impl fmt::Display for JobReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let direction = match self.target.direction {
            Direction::Encode => "encoders",
            Direction::Decode => "decoders",
        };
        write!(f, "{} {direction}:", self.target.codec)?;
        for report in &self.backends {
            write!(f, " {} ({:?})", report.backend, report.availability)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ferricast_core::{
        AutoSelect, BackendSelection, CapturedFrame, CodecResult, EncodedFrame, PixelFormat,
        RawFrame, VideoEncoder,
    };
    use istmo_core::RuntimeInit;

    use super::*;
    use crate::registry::{Availability, Candidate};

    /// Echoes each frame's timestamp as its packet.
    #[derive(Default)]
    struct Echo;

    impl VideoEncoder for Echo {
        const CODEC: Codec = Codec::H264;
        fn configure(&mut self, _: &EncoderConfig) -> CodecResult<()> {
            Ok(())
        }
        fn encode(&mut self, frame: CapturedFrame) -> CodecResult<EncodedFrame> {
            Ok(EncodedFrame {
                codec: Codec::H264,
                data: bytes::Bytes::new(),
                timestamp_us: frame.timestamp_us(),
                is_keyframe: true,
                duration_us: None,
                pts_dts: (0, 0),
            })
        }
        fn flush(self) -> CodecResult<Vec<EncodedFrame>> {
            Ok(Vec::new())
        }
        fn get_headers(&mut self) -> CodecResult<Vec<u8>> {
            Ok(Vec::new())
        }
    }

    fn registry() -> VideoEncoderRegistry {
        VideoEncoderRegistry::new(CodecTarget::encode(Codec::H264))
            .candidate(Candidate::new(
                BackendId::X264,
                Acceleration::Software,
                crate::open_video_encoder::<Echo>,
            ))
            .missing(BackendId::Nvenc, "ferricast-encoder/nvenc")
    }

    fn service() -> (CodecService, CodecsClient) {
        let RuntimeInit { runtime, outbound } = Runtime::mock();
        // Nothing reads the outbound frames of an in-process test.
        std::thread::spawn(move || while outbound.recv().is_ok() {});
        let service = CodecService::with_runtime(
            Catalog::default().video_encoder(Codec::H264, registry),
            &runtime,
        );
        runtime.register_host(CodecsHost::new(service.clone()));
        let client = CodecsClient::from_runtime(&runtime).unwrap();
        (service, client)
    }

    fn frame(timestamp_us: u64) -> CapturedFrame {
        CapturedFrame::Cpu(RawFrame {
            width: 2,
            height: 2,
            stride: 8,
            format: PixelFormat::Bgra,
            data: bytes::Bytes::from_static(&[0; 16]),
            timestamp_us,
            colour: None,
        })
    }

    #[test]
    fn capabilities_list_compiled_and_missing_backends() {
        let (_, client) = service();
        let jobs = pollster::block_on(client.capabilities()).unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].target, CodecTarget::encode(Codec::H264));
        assert_eq!(jobs[0].backends[0].availability, Availability::Unprobed);
        assert!(matches!(
            jobs[0].backends[1].availability,
            Availability::NotCompiled { .. }
        ));
    }

    #[test]
    fn a_session_opened_over_istmo_encodes_on_the_rust_data_plane() {
        let (service, client) = service();
        let session = pollster::block_on(
            client.open_owned(OpenRequest::VideoEncoder(EncoderConfig::default())),
        )
        .unwrap();
        assert_eq!(session.resolution.backend, BackendId::X264);
        let id = session.session.id();
        let encoder = service.video_encoder(id).unwrap();
        let packet = lock(&encoder).encode(frame(42)).unwrap();
        assert_eq!(packet.timestamp_us, 42);
        drop(encoder);
        drop(session);
        assert_eq!(
            service.video_encoder(id).unwrap_err(),
            CodecError::NoSuchSession {
                session: id.get(),
                kind: "video encoder".into(),
            }
        );
    }

    #[test]
    fn codec_errors_cross_the_wire_typed() {
        let (_, client) = service();
        let hardware = EncoderConfig {
            backend: BackendSelection::Fixed(BackendId::Nvenc),
            ..EncoderConfig::default()
        };
        let error = pollster::block_on(client.resolve(OpenRequest::VideoEncoder(hardware)))
            .unwrap_err()
            .into_codec_error()
            .unwrap();
        assert_eq!(
            error,
            CodecError::BackendNotCompiled {
                backend: BackendId::Nvenc,
                target: CodecTarget::encode(Codec::H264),
                feature: "ferricast-encoder/nvenc".into(),
            }
        );
        let software = EncoderConfig {
            backend: AutoSelect::default().into(),
            ..EncoderConfig::default()
        };
        let missing_job =
            pollster::block_on(client.resolve(OpenRequest::VideoEncoder(EncoderConfig {
                codec: Codec::Av1,
                ..software
            })))
            .unwrap_err()
            .into_codec_error()
            .unwrap();
        assert!(matches!(missing_job, CodecError::NoBackendCompiled { .. }));
    }
}
