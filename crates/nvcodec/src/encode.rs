//! NVENC encoding.
//!
//! [`Encoder`] is the encoder: opened from a [`Config`] on a CUDA
//! [`Context`], it takes pictures from host memory ([`Encoder::encode_host`]),
//! from device buffers such as NVDEC output ([`Encoder::encode_device`]) or
//! from CUDA arrays such as EGL-imported DMA-BUFs ([`Encoder::encode_array`])
//! and hands out [`Packet`]s in submission order ([`Encoder::next_packet`]).
//!
//! It talks to the driver through the NVENC API version negotiated at open
//! ([`crate::version`]): the session logic is written once and instantiated
//! for every version this crate carries (`raw.rs`), and [`Encoder`] is an
//! enum over those instantiations. A feature the negotiated version lacks
//! ([`ApiFeature`]) is refused with [`Error::Unsupported`]
//! naming the version that adds it; what the GPU lacks is refused the same
//! way, from [`Capabilities`].
//!
//! Every input is registered with NVENC once and cached: host pictures are
//! copied on the encoder's CUDA stream into pooled device buffers registered
//! when first used, caller device buffers are cached by `Arc` identity and
//! CUDA arrays by a caller key. Nothing is allocated or registered per
//! picture in steady state ([`Encoder::registrations`] counts registrations).
//!
//! ```no_run
//! use std::num::NonZeroU32;
//! use ferricast_nvcodec::cuda::{Context, Device};
//! use ferricast_nvcodec::encode::{Codec, Config, Encoder, FrameOptions, FrameRate, HostFrame, InputFormat};
//!
//! # fn main() -> ferricast_nvcodec::Result<()> {
//! let context = Context::primary(Device::get(0)?)?;
//! let rate = FrameRate::per_second(NonZeroU32::new(30).unwrap());
//! let config = Config::low_latency(Codec::H264, 1280, 720, InputFormat::Nv12, rate, 4_000_000);
//! let mut encoder = Encoder::open(&context, &config)?;
//! let picture = vec![16u8; 1280 * 720 * 3 / 2];
//! encoder.encode_host(
//!     HostFrame::tight(&picture, InputFormat::Nv12, 1280, 720),
//!     FrameOptions { timestamp: 0, force_idr: false },
//! )?;
//! while let Some(packet) = encoder.next_packet() {
//!     println!("{} bytes, {:?}", packet.data.len(), packet.picture_type);
//!     encoder.recycle(packet);
//! }
//! # Ok(())
//! # }
//! ```

mod caps;
mod config;
mod engine;
mod raw;
mod sei;

use std::ffi::c_void;
use std::fmt;
use std::ptr::NonNull;
use std::sync::Arc;

pub use self::caps::{Capabilities, CodecCapabilities, CodecSupport};
pub use self::config::{
    Chromaticity, Codec, ColorDescription, ColorPrimaries, Config, ContentLightLevel, FrameRate,
    Gop, Hdr10, InputFormat, MasteringDisplay, MatrixCoefficients, Preset, Profile, QpSet,
    RateControl, TransferCharacteristics, Tuning, Vbv,
};
use self::engine::Engine;
pub use self::raw::ApiFeature;
use self::raw::EncodeApi;
use crate::cuda::{ArrayView, Context, DeviceBuffer, Event, HostImage, PitchedView};
use crate::error::{Error, Result};
use crate::version::{AnyApi, Api, ApiVersion, DeviceType, NvencApi, api};

/// An NVENC encoder session on a CUDA context, destroyed on drop.
///
/// Holds an `Arc` of its context, so the context outlives the session, and a
/// copy of the function list it was opened with, so every call on it uses the
/// same API version.
pub struct Session<A: NvencApi> {
    encoder: NonNull<c_void>,
    api: Api<A>,
    context: Arc<Context>,
}

// SAFETY: an NVENC session has no thread affinity: the driver accepts calls on
// it from any thread, one at a time. Its CUDA context is pushed around every
// call that needs it, and `Session` is not `Sync`, so calls never overlap.
unsafe impl<A: NvencApi> Send for Session<A> {}

impl<A: NvencApi> Session<A> {
    /// Opens a session on `context` (`nvEncOpenEncodeSessionEx`).
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`] if the context cannot be made current,
    /// [`Error::Nvenc`] if the driver refuses the session (no NVENC on this
    /// GPU, session limit reached, version mismatch).
    pub fn open(api: Api<A>, context: &Arc<Context>) -> Result<Self> {
        // Opened on the session thread (see `crate::lifecycle`).
        crate::lifecycle::run(context, || {
            let device = context.as_non_null();
            let _current = context.guard()?;
            // SAFETY: `device` is the context's handle and the session holds
            // an `Arc` of the context, so it outlives the encoder; `api`
            // holds the list for `A`.
            let encoder =
                unsafe { A::open_encode_session(api.functions(), device, DeviceType::Cuda)? };
            Ok(Self {
                encoder,
                api,
                context: Arc::clone(context),
            })
        })
    }

    /// The encoder handle every NVENC call takes.
    #[must_use]
    pub fn as_ptr(&self) -> *mut c_void {
        self.encoder.as_ptr()
    }

    /// The function list the session was opened with.
    #[must_use]
    pub fn api(&self) -> &Api<A> {
        &self.api
    }

    /// The CUDA context the session encodes on.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// The driver's explanation of the session's last failure
    /// (`nvEncGetLastErrorString`), to attach to an [`Error::Nvenc`].
    #[must_use]
    pub fn last_error(&self) -> Option<String> {
        // SAFETY: the session is live and opened with this list.
        unsafe { A::last_error(self.api.functions(), self.encoder) }
    }
}

/// A session handle on its way to the session thread to be destroyed.
struct Doomed(NonNull<c_void>);

// SAFETY: an NVENC session has no thread affinity, and the handle is used by
// one thread only (the session thread destroying it).
unsafe impl Send for Doomed {}

impl<A: NvencApi> Drop for Session<A> {
    fn drop(&mut self) {
        let (api, doomed, context) = (self.api, Doomed(self.encoder), &self.context);
        // Destroyed on the session thread (see `crate::lifecycle`).
        let destroyed = crate::lifecycle::run(context, move || {
            let doomed = doomed;
            context.guard().and_then(|_current| {
                // SAFETY: the session is live, opened with this list, and
                // not used again.
                unsafe { A::destroy_encoder(api.functions(), doomed.0) }
            })
        });
        if let Err(error) = destroyed {
            tracing::warn!(%error, api = %A::VERSION, "destroying an NVENC session");
        }
    }
}

impl<A: NvencApi> fmt::Debug for Session<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session")
            .field("encoder", &self.encoder)
            .field("api", &A::VERSION)
            .field("context", &self.context)
            .finish()
    }
}

/// `NV_ENC_ERR_OUT_OF_MEMORY` and `NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY`: what
/// `nvEncOpenEncodeSessionEx` answers when the GPU's concurrent session limit
/// is reached (driver 595 answers the latter after 12 sessions on an RTX
/// 3060; older drivers the former).
const SESSION_LIMIT: [u32; 2] = [10, 21];

/// Opens a session, telling the session limit apart from other failures.
fn open_session<A: NvencApi>(api: Api<A>, context: &Arc<Context>) -> Result<Session<A>> {
    Session::open(api, context).map_err(|error| match error {
        Error::Nvenc {
            call: call @ "nvEncOpenEncodeSessionEx",
            status,
            detail,
        } if SESSION_LIMIT.contains(&status.0) => Error::NvencSessionLimit {
            call,
            status,
            detail,
        },
        other => other,
    })
}

/// Per-picture options.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FrameOptions {
    /// Passed through to [`Packet::timestamp`] (any unit; NVENC reorders it
    /// with the pictures when B frames are on).
    pub timestamp: u64,
    /// Make this picture an IDR (with parameter sets when
    /// [`Config::repeat_parameter_sets`] is on).
    pub force_idr: bool,
}

/// A picture in host memory.
#[derive(Clone, Copy, Debug)]
pub enum HostFrame<'a> {
    /// One plane: [`InputFormat::Argb`], [`InputFormat::Abgr`].
    Packed(HostImage<'a>),
    /// Luma, then half-height interleaved chroma: [`InputFormat::Nv12`],
    /// [`InputFormat::P010`].
    SemiPlanar {
        /// The luma plane.
        luma: HostImage<'a>,
        /// The interleaved chroma plane.
        chroma: HostImage<'a>,
    },
}

impl<'a> HostFrame<'a> {
    /// A picture whose rows are tightly packed in `data`, the chroma plane
    /// (if any) right after the luma rows. A too-short `data` is reported
    /// when the picture is encoded.
    #[must_use]
    pub fn tight(data: &'a [u8], format: InputFormat, width: u32, height: u32) -> Self {
        let pitch = format.row_bytes(width);
        if format.is_semi_planar() {
            let split = (pitch * height as usize).min(data.len());
            let (luma, chroma) = data.split_at(split);
            Self::SemiPlanar {
                luma: HostImage { data: luma, pitch },
                chroma: HostImage {
                    data: chroma,
                    pitch,
                },
            }
        } else {
            Self::Packed(HostImage { data, pitch })
        }
    }
}

/// A picture in a device buffer of the encoder's context, such as an NVDEC
/// output frame: luma rows from row 0, chroma rows (semi-planar formats)
/// from [`DeviceFrame::chroma_row`], all `pitch` bytes apart.
#[derive(Clone, Copy, Debug)]
pub struct DeviceFrame<'a> {
    /// The buffer. Registrations are cached by its `Arc` and dropped once the
    /// last `Arc` is gone (or [`Encoder::forget_device_buffer`]).
    pub buffer: &'a Arc<DeviceBuffer>,
    /// First chroma row; `None` is right after the picture's luma rows.
    /// Anything else needs [`ApiFeature::ChromaPlaneOffset`].
    pub chroma_row: Option<usize>,
    /// Signalled when whoever wrote the buffer is done with it; the encoder
    /// stream waits for it. `None`: the writes are already complete.
    pub ready: Option<&'a Event>,
}

impl<'a> DeviceFrame<'a> {
    /// A picture filling `buffer` from row 0, written already.
    #[must_use]
    pub fn new(buffer: &'a Arc<DeviceBuffer>) -> Self {
        Self {
            buffer,
            chroma_row: None,
            ready: None,
        }
    }
}

/// The coding type of an encoded picture (`NV_ENC_PIC_TYPE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PictureType {
    /// Forward predicted.
    P,
    /// Bidirectionally predicted.
    B,
    /// Intra, not a random access point.
    I,
    /// Instantaneous decoder refresh (AV1: key frame): a random access point.
    Idr,
    /// Bi-predicted with only intra references.
    Bi,
    /// Skipped.
    Skipped,
    /// Intra refresh.
    IntraRefresh,
    /// P picture no other picture references.
    NonReferenceP,
    /// AV1 switch frame.
    Switch,
    /// A value this crate does not know.
    Unknown(u32),
}

impl PictureType {
    fn from_raw(raw: u32) -> Self {
        match raw {
            0 => Self::P,
            1 => Self::B,
            2 => Self::I,
            3 => Self::Idr,
            4 => Self::Bi,
            5 => Self::Skipped,
            6 => Self::IntraRefresh,
            7 => Self::NonReferenceP,
            8 => Self::Switch,
            other => Self::Unknown(other),
        }
    }
}

/// One encoded picture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packet {
    /// The bitstream: Annex B for H.264 / HEVC, low-overhead OBUs for AV1.
    /// Give the packet back with [`Encoder::recycle`] to reuse the
    /// allocation.
    pub data: Vec<u8>,
    /// [`FrameOptions::timestamp`] of the picture.
    pub timestamp: u64,
    /// Its coding type.
    pub picture_type: PictureType,
    /// Its position in submission order, from 0.
    pub index: u64,
}

impl Packet {
    /// Whether a decoder can start here (an IDR / AV1 key frame).
    #[must_use]
    pub fn is_keyframe(&self) -> bool {
        self.picture_type == PictureType::Idr
    }
}

/// The encoder instantiated for each API version.
enum Versioned {
    V11_1(Engine<api::V11_1>),
    V12_0(Engine<api::V12_0>),
    V12_1(Engine<api::V12_1>),
    V12_2(Engine<api::V12_2>),
    V13_0(Engine<api::V13_0>),
    V13_1(Engine<api::V13_1>),
}

/// Runs `$body` with `$engine` bound to the engine of whichever version.
macro_rules! with_engine {
    ($versioned:expr, $engine:ident => $body:expr) => {
        match $versioned {
            Versioned::V11_1($engine) => $body,
            Versioned::V12_0($engine) => $body,
            Versioned::V12_1($engine) => $body,
            Versioned::V12_2($engine) => $body,
            Versioned::V13_0($engine) => $body,
            Versioned::V13_1($engine) => $body,
        }
    };
}

/// Runs `$body` with `$engine` bound to the engine of an open encoder, or
/// `$fallback` with `$closed` bound to what a closed one remembers.
macro_rules! on_engine {
    ($state:expr, $engine:ident => $body:expr, $closed:ident => $fallback:expr) => {
        match $state {
            State::Open(versioned) => with_engine!(versioned, $engine => $body),
            State::Closed($closed) => $fallback,
        }
    };
}

/// Whether the encoder has a session.
#[expect(
    clippy::large_enum_variant,
    reason = "an open encoder is the normal state; boxing it would add an indirection to every call"
)]
enum State {
    Open(Versioned),
    /// A reopen failed twice ([`Encoder::reopen`]): no session until the
    /// next successful reopen.
    Closed(Box<Closed>),
}

/// What a closed encoder remembers of its last session.
struct Closed {
    config: Config,
    caps: CodecCapabilities,
    version: ApiVersion,
    reason: String,
}

impl Closed {
    fn error(&self) -> Error {
        Error::EncoderClosed {
            reason: self.reason.clone(),
        }
    }
}

/// An NVENC encoder. See the [module docs](self).
///
/// `Send`, not `Sync`: one thread at a time drives it.
pub struct Encoder {
    state: State,
    api: AnyApi,
    context: Arc<Context>,
}

impl Encoder {
    /// Opens an encoder through the negotiated API version
    /// ([`AnyApi::negotiate`], honouring `FERRICAST_NVENC_API`).
    ///
    /// # Errors
    ///
    /// Loading and negotiation errors; [`Error::InvalidEncodeConfig`] for a
    /// configuration no GPU takes; [`Error::Unsupported`] for what the API
    /// version or the GPU lacks; [`Error::NvencSessionLimit`] when the GPU
    /// takes no more sessions; [`Error::Nvenc`] / [`Error::Cuda`] for driver
    /// failures.
    pub fn open(context: &Arc<Context>, config: &Config) -> Result<Self> {
        Self::open_with(AnyApi::negotiate()?, context, config)
    }

    /// As [`Encoder::open`] through `api`.
    ///
    /// # Errors
    ///
    /// As [`Encoder::open`].
    pub fn open_with(api: AnyApi, context: &Arc<Context>, config: &Config) -> Result<Self> {
        let versioned = Self::open_versioned(&api, context, config)
            .map_err(|error| context.note_error(&error).unwrap_or(error))?;
        Ok(Self {
            state: State::Open(versioned),
            api,
            context: Arc::clone(context),
        })
    }

    fn open_versioned(api: &AnyApi, context: &Arc<Context>, config: &Config) -> Result<Versioned> {
        Ok(match *api {
            AnyApi::V11_1(api) => Versioned::V11_1(Engine::open(api, context, config)?),
            AnyApi::V12_0(api) => Versioned::V12_0(Engine::open(api, context, config)?),
            AnyApi::V12_1(api) => Versioned::V12_1(Engine::open(api, context, config)?),
            AnyApi::V12_2(api) => Versioned::V12_2(Engine::open(api, context, config)?),
            AnyApi::V13_0(api) => Versioned::V13_0(Engine::open(api, context, config)?),
            AnyApi::V13_1(api) => Versioned::V13_1(Engine::open(api, context, config)?),
        })
    }

    /// Marks the context lost when `result` says the device is gone, so the
    /// next user of the context can tell ([`Context::is_lost`]).
    fn noted<T>(&self, result: Result<T>) -> Result<T> {
        result.map_err(|error| self.context.note_error(&error).unwrap_or(error))
    }

    /// Closes the session and opens a new one for `config`, on the same
    /// context and API version: the old session is closed first, so a GPU
    /// at its session limit takes the new one. When `config` cannot be
    /// opened, a session for the previous configuration is opened again.
    /// Pending pictures are dropped (call [`Encoder::flush`] first to keep
    /// their packets); packets not taken yet are lost.
    ///
    /// # Errors
    ///
    /// As [`Encoder::open`]. When the previous configuration cannot be
    /// reopened either, the encoder stays closed: every call fails with
    /// [`Error::EncoderClosed`] until a reopen succeeds.
    pub fn reopen(&mut self, config: &Config) -> Result<()> {
        let context = Arc::clone(&self.context);
        self.reopen_on(&context, config)
    }

    /// As [`Encoder::reopen`], on `context`: the way back from a lost
    /// context ([`Context::is_lost`]).
    ///
    /// # Errors
    ///
    /// As [`Encoder::reopen`].
    pub fn reopen_on(&mut self, context: &Arc<Context>, config: &Config) -> Result<()> {
        let previous = Closed {
            config: self.config().clone(),
            caps: self.capabilities().clone(),
            version: self.api_version(),
            reason: String::new(),
        };
        // Closes the old session before the new one opens.
        let old = std::mem::replace(
            &mut self.state,
            State::Closed(Box::new(Closed {
                reason: "reopening".into(),
                ..previous
            })),
        );
        Self::close(&self.context, old);
        self.context = Arc::clone(context);
        let error = match Self::open_versioned(&self.api, &self.context, config) {
            Ok(versioned) => {
                self.state = State::Open(versioned);
                return Ok(());
            }
            Err(error) => error,
        };
        let error = self.context.note_error(&error).unwrap_or(error);
        let State::Closed(previous) = &mut self.state else {
            unreachable!("the state was set to closed above");
        };
        match Self::open_versioned(&self.api, &self.context, &previous.config) {
            Ok(versioned) => self.state = State::Open(versioned),
            Err(again) => {
                tracing::warn!(%again, "reopening NVENC with the previous configuration");
                previous.reason = format!("reopening failed: {error}; then: {again}");
            }
        }
        Err(error)
    }

    /// Switches to `config`: in place when only the size (up to the size
    /// the session was opened with), frame rate, GOP, rate control or
    /// colour change ([`nvEncReconfigureEncoder`], pending pictures flushed
    /// and their packets kept), otherwise through [`Encoder::reopen`].
    ///
    /// # Errors
    ///
    /// As [`Encoder::reopen`]; driver errors of the in-place change.
    pub fn reconfigure(&mut self, config: &Config) -> Result<()> {
        let in_place = on_engine!(&mut self.state,
            engine => engine.reconfigure(config),
            closed => Err(closed.error()));
        match in_place {
            Ok(()) => Ok(()),
            Err(Error::Unsupported { .. } | Error::EncoderClosed { .. }) => self.reopen(config),
            Err(error) => self.noted(Err(error)),
        }
    }

    /// Whether the encoder has a session (see [`Encoder::reopen`]).
    #[must_use]
    pub fn is_open(&self) -> bool {
        matches!(self.state, State::Open(_))
    }

    /// The CUDA context the encoder runs on.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// The NVENC API version in use.
    #[must_use]
    pub fn api_version(&self) -> ApiVersion {
        fn version<A: EncodeApi>(_: &Engine<A>) -> ApiVersion {
            A::VERSION
        }
        on_engine!(&self.state, engine => version(engine), closed => closed.version)
    }

    /// The configuration in effect (rate control updated by
    /// [`Encoder::set_rate_control`]).
    #[must_use]
    pub fn config(&self) -> &Config {
        on_engine!(&self.state, engine => engine.config(), closed => &closed.config)
    }

    /// What the GPU can do for this encoder's codec.
    #[must_use]
    pub fn capabilities(&self) -> &CodecCapabilities {
        on_engine!(&self.state, engine => engine.capabilities(), closed => &closed.caps)
    }

    /// Encodes a picture from host memory: copied on the encoder's stream
    /// into a pooled device buffer (registered the first time its slot is
    /// used), then submitted. The host memory is free again when this
    /// returns.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidEncodeInput`] for a layout other than the
    /// configuration's, [`Error::CopyOutOfBounds`] / [`Error::PitchTooSmall`]
    /// for planes too small for the picture, driver errors.
    pub fn encode_host(&mut self, frame: HostFrame<'_>, options: FrameOptions) -> Result<()> {
        let result = on_engine!(&mut self.state,
            engine => engine.encode_host(frame, options),
            closed => Err(closed.error()));
        self.noted(result)
    }

    /// Encodes a picture from a device buffer without copying it. The
    /// buffer's registration is cached and reused while it lives; the
    /// encoder keeps an `Arc` of it until the picture's packet is out, and
    /// the buffer must not be written in that time. Dropping the buffer
    /// afterwards is fine: its memory is freed once the registration is
    /// undone.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidEncodeInput`] for a buffer of another context, too
    /// small for the picture, or still being encoded;
    /// [`Error::Unsupported`] for a chroma row offset the API version cannot
    /// express ([`ApiFeature::ChromaPlaneOffset`]); driver errors.
    pub fn encode_device(&mut self, frame: DeviceFrame<'_>, options: FrameOptions) -> Result<()> {
        let result = on_engine!(&mut self.state,
            engine => engine.encode_device(frame, options),
            closed => Err(closed.error()));
        self.noted(result)
    }

    /// Encodes a picture from a CUDA array (packed RGB formats only), such
    /// as an EGL-imported DMA-BUF. The registration is cached under `key`
    /// until [`Encoder::invalidate_array`] / [`Encoder::clear_input_caches`]
    /// or until the same key comes with another array or size.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidEncodeInput`] for a semi-planar configuration, an
    /// array of another context or too small; driver errors.
    ///
    /// # Safety
    ///
    /// - The array behind `key` must stay allocated (and, for interop
    ///   arrays, mapped) until its registration is invalidated or the encoder
    ///   is dropped: invalidate `key` before destroying or unmapping it
    ///   ([`Encoder::invalidate_array`] returns once NVENC is done with it).
    ///   A new array that reuses an old handle under the same key without an
    ///   invalidation in between would be read through the stale
    ///   registration.
    /// - The array must not be written until this picture's packet is out
    ///   ([`Encoder::next_packet`]); with `frames_in_flight` 0 and no B
    ///   frames or lookahead that is when this call returns.
    pub unsafe fn encode_array(
        &mut self,
        key: u64,
        array: ArrayView<'_>,
        options: FrameOptions,
    ) -> Result<()> {
        let result = on_engine!(&mut self.state,
            // SAFETY: the caller upholds the contract above, which is the
            // engine's.
            engine => unsafe { engine.encode_array(key, array, options) },
            closed => Err(closed.error()));
        self.noted(result)
    }

    /// Encodes a picture from pitched device memory someone else owns
    /// (packed RGB formats only), such as a LINEAR DMA-BUF imported through
    /// EGL ([`crate::cuda::PitchedView`]), without copying it. Cached under
    /// `key` like [`Encoder::encode_array`], sharing its keys: the same key
    /// with other memory or another pitch is registered again.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidEncodeInput`] for a semi-planar configuration, memory
    /// of another context, too small, or with a pitch NVENC cannot take;
    /// driver errors.
    ///
    /// # Safety
    ///
    /// As [`Encoder::encode_array`], for the memory behind `view`.
    pub unsafe fn encode_pitched(
        &mut self,
        key: u64,
        view: PitchedView<'_>,
        options: FrameOptions,
    ) -> Result<()> {
        let result = on_engine!(&mut self.state,
            // SAFETY: the caller upholds the contract above, which is the
            // engine's.
            engine => unsafe { engine.encode_pitched(key, view, options) },
            closed => Err(closed.error()));
        self.noted(result)
    }

    /// The next packet, in submission order (decode order with B frames).
    pub fn next_packet(&mut self) -> Option<Packet> {
        on_engine!(&mut self.state, engine => engine.next_packet(), _closed => None)
    }

    /// Hands a packet's buffer back for reuse.
    pub fn recycle(&mut self, packet: Packet) {
        on_engine!(&mut self.state, engine => engine.recycle(packet.data), _closed => ());
    }

    /// Hands a buffer back for reuse as a packet's data: a
    /// [`Packet::data`] the caller took out of its packet and is done with.
    pub fn recycle_buffer(&mut self, buffer: Vec<u8>) {
        on_engine!(&mut self.state, engine => engine.recycle(buffer), _closed => ());
    }

    /// Makes the next picture an IDR.
    pub fn request_idr(&mut self) {
        on_engine!(&mut self.state, engine => engine.request_idr(), _closed => ());
    }

    /// Changes the rate control without a new session (keeping `max ==
    /// average` with [`RateControl::Cbr`]).
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] when the GPU cannot change bitrate or mode on
    /// the fly, [`Error::InvalidEncodeConfig`], driver errors.
    pub fn set_rate_control(&mut self, rate_control: RateControl) -> Result<()> {
        let result = on_engine!(&mut self.state,
            engine => engine.set_rate_control(rate_control),
            closed => Err(closed.error()));
        self.noted(result)
    }

    /// The parameter sets (H.264 SPS + PPS, HEVC VPS + SPS + PPS in Annex B;
    /// the AV1 sequence header OBU), for out-of-band signalling.
    ///
    /// # Errors
    ///
    /// Driver errors.
    pub fn sequence_params(&mut self) -> Result<Vec<u8>> {
        let result = on_engine!(&mut self.state,
            engine => engine.sequence_params(),
            closed => Err(closed.error()));
        self.noted(result)
    }

    /// Ends the stream: every picture still held (B frames, lookahead,
    /// frames in flight) is encoded and its packet queued. The encoder takes
    /// new pictures afterwards.
    ///
    /// # Errors
    ///
    /// Driver errors.
    pub fn flush(&mut self) -> Result<()> {
        let result = on_engine!(&mut self.state, engine => engine.flush(), _closed => Ok(()));
        self.noted(result)
    }

    /// Drops the cached registration of `buffer`, e.g. before reusing its
    /// memory for another layout. Optional: registrations of dropped buffers
    /// go on their own.
    pub fn forget_device_buffer(&mut self, buffer: &Arc<DeviceBuffer>) {
        on_engine!(&mut self.state, engine => engine.forget_device_buffer(buffer), _closed => ());
    }

    /// Drops the cached registration of the array (or pitched memory, see
    /// [`Encoder::encode_pitched`]) under `key`. When this returns NVENC no
    /// longer reads it: a pending picture that does is encoded first (its
    /// packet queued), so the memory may be freed or unmapped right after.
    pub fn invalidate_array(&mut self, key: u64) {
        on_engine!(&mut self.state, engine => engine.invalidate_array(key), _closed => ());
    }

    /// Drops every cached device-buffer and array registration (the host
    /// input pool keeps its own), as [`Encoder::invalidate_array`] for each.
    pub fn clear_input_caches(&mut self) {
        on_engine!(&mut self.state, engine => engine.clear_input_caches(), _closed => ());
    }

    /// How many input resources this encoder has registered with NVENC so
    /// far: constant in steady state.
    #[must_use]
    pub fn registrations(&self) -> u64 {
        on_engine!(&self.state, engine => engine.registrations(), _closed => 0)
    }
}

impl Encoder {
    /// Drops `state`'s session on the session thread ([`crate::lifecycle`]),
    /// where it was opened.
    fn close(context: &Arc<Context>, state: State) {
        if matches!(state, State::Open(_)) {
            crate::lifecycle::run(context, move || drop(state));
        }
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        let State::Open(_) = &self.state else {
            return;
        };
        let closed = State::Closed(Box::new(Closed {
            config: self.config().clone(),
            caps: self.capabilities().clone(),
            version: self.api_version(),
            reason: "dropped".into(),
        }));
        let state = std::mem::replace(&mut self.state, closed);
        Self::close(&self.context, state);
    }
}

impl fmt::Debug for Encoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Encoder")
            .field("api", &self.api_version())
            .field("config", self.config())
            .finish_non_exhaustive()
    }
}
