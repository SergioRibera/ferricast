//! NVDEC: hardware decoding of H.264, HEVC, VP8, VP9 and AV1 through
//! NVIDIA's bitstream parser (`cuvidCreateVideoParser`).
//!
//! # Input
//!
//! [`Decoder::decode`] takes one [`Packet`] at a time, in decode order:
//!
//! - H.264 / HEVC: one access unit in Annex B form (start codes), with the
//!   parameter sets (SPS / PPS, and VPS for HEVC) in band before the first
//!   picture that uses them. Streams from MP4 / Matroska (length-prefixed,
//!   parameter sets in `avcC` / `hvcC`) must be converted first.
//! - VP8 / VP9: one frame as stored in IVF / WebM (a VP9 superframe is one
//!   packet).
//! - AV1: one temporal unit of low-overhead OBUs (Section 5 of the AV1
//!   specification), as stored in IVF / MP4 / WebM; not Annex B.
//!
//! Each packet carries an optional timestamp, in any unit (microseconds
//! suggested; tell the parser the rate with [`DecoderConfig::clock_rate`]).
//! The parser hands it back with the picture that packet started, in display
//! order: nothing re-parses the bitstream to time pictures. A packet with
//! [`Packet::discontinuity`] resets the parser first (a seek), outputting
//! what it still held;
//! [`Decoder::flush`] signals the end of the stream and outputs every
//! picture still held for reordering.
//!
//! # Output
//!
//! Each decoded picture is copied out of NVDEC's surface right away (the
//! surface is never lent to the caller) into one of two kinds of frame,
//! chosen by the decoder's type:
//!
//! - [`Decoder<Host>`] gives [`HostFrame`]s: packed planes in a pooled host
//!   buffer, byte-for-byte what FFmpeg's `rawvideo` output holds for the
//!   same pixel format.
//! - [`Decoder<Device>`] gives [`DeviceFrame`]s: planes in a pooled pitched
//!   [`DeviceBuffer`](crate::cuda::DeviceBuffer), the zero-copy path to NVENC
//!   and to GPU renderers.
//!
//! Frames come out of [`Decoder::next_frame`] in display order and return
//! their buffer to the decoder's pool when dropped, so steady-state decoding
//! allocates nothing. At most [`DecoderConfig::max_frames`] can be held at
//! once; past that, decoding fails with
//! [`Error::FramesInFlight`] until some are
//! dropped.
//!
//! Every frame describes itself in a [`FrameInfo`]: the display size (the
//! stream's display area, all four edges of it, not only its size), the
//! [`PixelFormat`] (NV12 at 8 bits, P016 above, the 4:2:2 / 4:4:4 variants
//! where the GPU decodes them), the bit depth, the stream's [`Colour`] as
//! H.273 code points, field information, and whether NVDEC reported the
//! picture as damaged. Its SEI messages (with the HDR mastering display and
//! content light level decoded) come with it when the driver has the SEI
//! callback (NVDEC API 12.0, R520 and newer).
//!
//! ```no_run
//! use ferricast_nvcodec::cuda::{Context, Device};
//! use ferricast_nvcodec::decode::{Codec, Decoder, DecoderConfig, Host, Packet};
//!
//! # fn main() -> ferricast_nvcodec::Result<()> {
//! # let access_units: Vec<(Vec<u8>, i64)> = Vec::new();
//! let context = Context::primary(Device::get(0)?)?;
//! let mut decoder = Decoder::<Host>::new(&context, DecoderConfig::new(Codec::Hevc))?;
//! for (data, pts) in &access_units {
//!     decoder.decode(Packet::new(data, *pts))?;
//!     while let Some(frame) = decoder.next_frame() {
//!         let info = frame.info();
//!         println!("{} {}x{} {:?}", info.timestamp, info.width, info.height, info.format);
//!     }
//! }
//! decoder.flush()?;
//! # Ok(())
//! # }
//! ```

mod caps;
mod crop;
mod format;
mod frame;
mod hevc;
mod nal;
mod obu;
mod parser;
mod pool;
mod sei;
mod vp8;
mod vp9;

use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

pub use caps::{ChromaFormat, Codec, DecoderCaps, OutputFormats};
pub use format::{
    Colour, DecodeStatus, Fields, FrameInfo, Layout, PixelFormat, Plane, Rect, SequenceInfo,
};
pub use frame::{DeviceFrame, HostFrame};
pub use sei::{
    CONTENT_LIGHT_LEVEL_INFO, ContentLightLevel, MASTERING_DISPLAY_COLOUR_VOLUME, MasteringDisplay,
    SeiMessage, SeiMessages,
};

use self::parser::{CtxLock, Output, Parser, State, StateBox};
use crate::cuda::{Context, Stream};
use crate::error::{CudaStatus, Error, Result};
use crate::loader::{CudaLibrary, CuvidLibrary};
use crate::sys::cuda as cu;
use crate::sys::cuvid as sys;

/// NVDEC runs on a CUDA context: both libraries are needed.
fn cuvid_library() -> Result<&'static CuvidLibrary> {
    CudaLibrary::get()?;
    CuvidLibrary::get()
}

/// A `u32` as the header's `unsigned long` (64 bits on Linux, 32 on Windows).
#[allow(
    clippy::useless_conversion,
    reason = "the identity conversion on Windows, where `unsigned long` is 32 bits"
)]
fn ulong(value: u32) -> sys::tcu_ulong {
    value.into()
}

/// Turns a cuvid status into a `Result` naming the call.
trait CuvidCheck {
    fn check_cuvid(self, call: &'static str) -> Result<()>;
}

impl CuvidCheck for cu::CUresult {
    fn check_cuvid(self, call: &'static str) -> Result<()> {
        if self == Self::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(Error::Cuvid {
                call,
                status: CudaStatus(self),
            })
        }
    }
}

/// How a [`Decoder`] is set up.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DecoderConfig {
    /// The stream's codec.
    pub codec: Codec,
    /// How many pictures the parser may hold back before display
    /// (`ulMaxDisplayDelay`). 0, the default, outputs each picture as soon
    /// as the reorder rules allow: the lowest latency. NVIDIA suggests 2-4
    /// for throughput; each adds a decode surface.
    pub max_display_delay: u32,
    /// NVDEC output surfaces (`ulNumOutputSurfaces`). Pictures are copied
    /// out and unmapped one at a time, so 1 suffices; the default is 2.
    pub output_surfaces: u32,
    /// Most frames handed out and not yet dropped (default 32). Bounds the
    /// memory a caller that holds frames can make the decoder allocate.
    pub max_frames: usize,
    /// The largest coded size later sequences may switch to without
    /// recreating the decoder (`ulMaxWidth` / `ulMaxHeight`). `None`, the
    /// default, reserves the first sequence's size: smaller sequences are
    /// resized in place (`cuvidReconfigureDecoder`), larger ones recreate the
    /// decoder.
    pub max_coded_size: Option<(u32, u32)>,
    /// Timestamp ticks per second (`ulClockRate`), for the timestamps the
    /// parser interpolates for packets without one. Default 1 000 000
    /// (microseconds).
    pub clock_rate: u32,
    /// Every packet holds exactly one whole picture (or AV1 temporal unit,
    /// or VP9 superframe), so the parser may decode it without waiting for
    /// the next packet (`CUVID_PKT_ENDOFPICTURE`). Default `true`, which
    /// saves a frame of latency; set `false` when packets split or join
    /// pictures arbitrarily.
    pub whole_pictures: bool,
    /// The AV1 operating point to decode, for scalable streams (default 0;
    /// out of range falls back to 0).
    pub av1_operating_point: u16,
}

impl DecoderConfig {
    /// The defaults for `codec`.
    #[must_use]
    pub const fn new(codec: Codec) -> Self {
        Self {
            codec,
            max_display_delay: 0,
            output_surfaces: 2,
            max_frames: 32,
            max_coded_size: None,
            clock_rate: 1_000_000,
            whole_pictures: true,
            av1_operating_point: 0,
        }
    }

    /// With [`DecoderConfig::max_display_delay`].
    #[must_use]
    pub const fn with_max_display_delay(self, max_display_delay: u32) -> Self {
        Self {
            max_display_delay,
            ..self
        }
    }

    /// With [`DecoderConfig::max_frames`].
    #[must_use]
    pub const fn with_max_frames(self, max_frames: usize) -> Self {
        Self { max_frames, ..self }
    }

    /// With [`DecoderConfig::max_coded_size`].
    #[must_use]
    pub const fn with_max_coded_size(self, width: u32, height: u32) -> Self {
        Self {
            max_coded_size: Some((width, height)),
            ..self
        }
    }

    /// With [`DecoderConfig::whole_pictures`].
    #[must_use]
    pub const fn with_whole_pictures(self, whole_pictures: bool) -> Self {
        Self {
            whole_pictures,
            ..self
        }
    }
}

/// One unit of compressed input: see the [module docs](self) for what a
/// packet holds per codec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Packet<'a> {
    /// The compressed bytes.
    pub data: &'a [u8],
    /// Presentation timestamp, returned with the picture.
    pub timestamp: Option<i64>,
    /// The stream restarts here (after a seek or a gap). Pictures the parser
    /// still holds are output first, as by [`Decoder::flush`]; then the
    /// packet is parsed from a clean state, so it should start at a keyframe.
    pub discontinuity: bool,
    /// Whether this packet holds exactly one whole picture (AV1 temporal
    /// unit, VP9 superframe), so the parser may decode it at once
    /// (`CUVID_PKT_ENDOFPICTURE`). `None`, the default, follows
    /// [`DecoderConfig::whole_pictures`]. Set `Some(false)` for a packet
    /// that holds several pictures or part of one: with the flag the parser
    /// would decode only the first of several, or a part as a whole.
    pub whole_picture: Option<bool>,
}

impl<'a> Packet<'a> {
    /// A packet with a timestamp.
    #[must_use]
    pub const fn new(data: &'a [u8], timestamp: i64) -> Self {
        Self {
            data,
            timestamp: Some(timestamp),
            discontinuity: false,
            whole_picture: None,
        }
    }

    /// A packet without a timestamp: the parser interpolates one.
    #[must_use]
    pub const fn untimed(data: &'a [u8]) -> Self {
        Self {
            data,
            timestamp: None,
            discontinuity: false,
            whole_picture: None,
        }
    }

    /// The same packet, marked as following a discontinuity.
    #[must_use]
    pub const fn with_discontinuity(self) -> Self {
        Self {
            discontinuity: true,
            ..self
        }
    }

    /// With [`Packet::whole_picture`].
    #[must_use]
    pub const fn with_whole_picture(self, whole: bool) -> Self {
        Self {
            whole_picture: Some(whole),
            ..self
        }
    }
}

mod sealed {
    pub trait Sealed {
        /// Frames go to device memory rather than host memory.
        const DEVICE: bool;
    }
}

/// Where a [`Decoder`] puts pictures: [`Host`] or [`Device`].
pub trait OutputMode: sealed::Sealed {}

/// Decode into host memory ([`HostFrame`]).
#[derive(Debug)]
pub enum Host {}

/// Decode into pooled CUDA buffers ([`DeviceFrame`]).
#[derive(Debug)]
pub enum Device {}

impl sealed::Sealed for Host {
    const DEVICE: bool = false;
}
impl OutputMode for Host {}
impl sealed::Sealed for Device {
    const DEVICE: bool = true;
}
impl OutputMode for Device {}

/// An NVDEC decoder with its bitstream parser. See the [module docs](self).
///
/// Dropping it destroys, in order: the NVDEC decoder (no surface is ever
/// left mapped), the parser, the context lock, and its hold on the CUDA
/// context. Frames already handed out stay valid: each keeps its own buffer
/// and context alive.
pub struct Decoder<M: OutputMode> {
    // Field order is drop order.
    state: StateBox,
    parser: Parser,
    _lock: CtxLock,
    context: Arc<Context>,
    /// The end of a non-whole HEVC packet that may continue in the next.
    held: nal::HeldBack,
    /// The last H.264 / HEVC packet went to the parser without
    /// `CUVID_PKT_ENDOFPICTURE`: its last picture is still open.
    open_picture: bool,
    _mode: PhantomData<fn() -> M>,
}

// SAFETY: nothing in a decoder has thread affinity: the parser and the NVDEC
// decoder are plain driver handles, the CUDA context is pushed around every
// call on the calling thread, and the parser calls back only on the thread
// running `decode`, which holds `&mut self`. `Decoder` is not `Sync`.
unsafe impl<M: OutputMode> Send for Decoder<M> {}

impl<M: OutputMode> Decoder<M> {
    /// A decoder for `config.codec` on `context`. The NVDEC decoder itself
    /// is created when the first sequence header arrives, sized for it.
    ///
    /// # Errors
    ///
    /// Loading errors ([`Error::LibraryMissing`] without an NVIDIA driver),
    /// [`Error::Cuda`] / [`Error::Cuvid`].
    pub fn new(context: &Arc<Context>, config: DecoderConfig) -> Result<Self> {
        let lock = CtxLock::new(context)?;
        let state = StateBox::new(State::new(
            config,
            context,
            lock.raw,
            <M as sealed::Sealed>::DEVICE,
        )?);
        let parser = Parser::new(&state)?;
        Ok(Self {
            state,
            parser,
            _lock: lock,
            context: Arc::clone(context),
            held: nal::HeldBack::default(),
            open_picture: false,
            _mode: PhantomData,
        })
    }

    /// Feeds one packet. Pictures it completes become available from
    /// [`Decoder::next_frame`].
    ///
    /// Before the parser sees it, a packet is checked where NVIDIA's parser
    /// would fault on malformed data instead of failing (see
    /// `design/NVCODEC.md`): every HEVC VPS is parsed in full, AV1 OBU
    /// sizes must fit, VP8 partitions must fit and inter frames need a key
    /// frame first, and AV1 inter frames need references NVDEC decoded.
    ///
    /// # Errors
    ///
    /// [`Error::Cuvid`] when the parser or NVDEC rejects the data,
    /// [`Error::Unsupported`] for a stream the GPU cannot decode (with the
    /// limit it exceeds), [`Error::FramesInFlight`] when every output frame
    /// is still held, [`Error::InvalidPacket`] for a packet the checks
    /// refuse, [`Error::MissingReference`] for a picture whose references
    /// were lost (it is dropped), [`Error::Cuda`] for copy failures. The
    /// decoder stays usable: later packets (ideally from a keyframe, after
    /// a [`Packet::discontinuity`]) decode normally.
    pub fn decode(&mut self, packet: Packet<'_>) -> Result<()> {
        if packet.data.is_empty() && !packet.discontinuity {
            return Ok(());
        }
        if packet.discontinuity {
            self.held.clear();
            self.open_picture = false;
            // On its own: the parser drops the data of a packet that carries the
            // flag (an H.264 IDR with its parameter sets, losing the GOP).
            self.parse(sys::CUvideopacketflags::CUVID_PKT_DISCONTINUITY.0, &[], 0)?;
            if packet.data.is_empty() {
                return Ok(());
            }
        }
        let timestamp = packet.timestamp.unwrap_or(0);
        let mut flags = 0;
        if packet.timestamp.is_some() {
            flags |= sys::CUvideopacketflags::CUVID_PKT_TIMESTAMP.0;
        }
        let codec = self.state.get().config.codec;
        let whole = packet
            .whole_picture
            .unwrap_or(self.state.get().config.whole_pictures);
        if whole {
            flags |= sys::CUvideopacketflags::CUVID_PKT_ENDOFPICTURE.0;
        }
        match codec {
            Codec::H264 | Codec::Hevc => {
                self.close_picture(codec, whole)?;
                let mut held = std::mem::take(&mut self.held);
                let crop = &mut self.state.get_mut().crop;
                let parsed = held
                    .split(packet.data, |data| nal::screen(codec, data, whole, crop))
                    .and_then(|ready| {
                        if ready.is_empty() {
                            Ok(())
                        } else {
                            self.parse(flags, ready, timestamp)
                        }
                    });
                self.held = held;
                parsed
            }
            Codec::Av1 => {
                obu::check_temporal_unit(packet.data)?;
                self.parse(flags, packet.data, timestamp)
            }
            Codec::Vp8 => {
                let frame = vp8::check_frame(packet.data)?;
                if !frame.key_frame && !self.state.get().vp8_keyed {
                    return Err(Error::MissingReference {
                        reason: "VP8 inter frame before any key frame".into(),
                    });
                }
                let parsed = self.parse(flags, packet.data, timestamp);
                // A key frame the decoder accepted set up the references.
                let state = self.state.get_mut();
                state.vp8_keyed |= frame.key_frame && state.sequence.is_some();
                parsed
            }
            Codec::Vp9 => {
                vp9::check_packet(packet.data)?;
                self.parse(flags, packet.data, timestamp)
            }
        }
    }

    /// Ends the stream: every picture the parser still holds is decoded and
    /// output. Decoding can go on afterwards, from a keyframe.
    ///
    /// # Errors
    ///
    /// As [`Decoder::decode`].
    pub fn flush(&mut self) -> Result<()> {
        self.held.clear();
        self.open_picture = false;
        self.parse(sys::CUvideopacketflags::CUVID_PKT_ENDOFSTREAM.0, &[], 0)
    }

    /// Before a whole packet that follows a packet sent without
    /// `CUVID_PKT_ENDOFPICTURE`: NVIDIA's parser then reads the open
    /// picture and the whole packet as one picture (losing a picture and
    /// corrupting the ones that reference it, seen on driver 595) unless an
    /// access unit delimiter separates them, which streams from MP4 or
    /// Matroska lack. So the decoder sends one on its own.
    fn close_picture(&mut self, codec: Codec, whole: bool) -> Result<()> {
        /// `access_unit_delimiter_rbsp()`, `primary_pic_type` 7 (any).
        const H264_DELIMITER: [u8; 6] = [0, 0, 0, 1, 0x09, 0xf0];
        /// `access_unit_delimiter_rbsp()`, `pic_type` 2 (any), layer 0.
        const HEVC_DELIMITER: [u8; 7] = [0, 0, 0, 1, 0x46, 0x01, 0x50];
        let open = std::mem::replace(&mut self.open_picture, !whole);
        if open && whole {
            let delimiter: &[u8] = if codec == Codec::H264 {
                &H264_DELIMITER
            } else {
                &HEVC_DELIMITER
            };
            self.parse(0, delimiter, 0)?;
        }
        Ok(())
    }

    fn parse(&mut self, flags: u32, data: &[u8], timestamp: i64) -> Result<()> {
        let cuvid = cuvid_library()?;
        let mut packet = sys::CUVIDSOURCEDATAPACKET {
            flags: ulong(flags),
            payload_size: sys::tcu_ulong::try_from(data.len())
                .map_err(|_| Error::SizeOverflow { what: "a packet" })?,
            payload: if data.is_empty() {
                std::ptr::null()
            } else {
                data.as_ptr()
            },
            timestamp,
        };
        self.state.get_mut().fed.parse(data.len());
        let parsed = {
            let _current = self.context.guard()?;
            // SAFETY: a live parser and a packet whose payload is borrowed
            // for the call. The callbacks it runs reach the state through the
            // parser's user-data pointer only; no reference into the state is
            // held here while it runs.
            unsafe { (cuvid.parse_video_data)(self.parser.raw, &raw mut packet) }
        };
        let result = match self.state.get_mut().error.take() {
            Some(error) => Err(error),
            None => parsed.check_cuvid("cuvidParseVideoData"),
        };
        // A sticky CUDA error marks the shared context lost, so the next
        // decoder or encoder starts on a fresh one.
        result.map_err(|error| self.context.note_error(&error).unwrap_or(error))
    }

    /// The current sequence's format, once a sequence header has been
    /// accepted.
    #[must_use]
    pub fn sequence(&self) -> Option<SequenceInfo> {
        self.state.get().sequence
    }

    /// Decoded frames waiting in [`Decoder::next_frame`].
    #[must_use]
    pub fn pending(&self) -> usize {
        self.state.get().output.len()
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &DecoderConfig {
        &self.state.get().config
    }

    /// The CUDA context frames live on.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// The stream every copy out of NVDEC runs on. Device frames are ready
    /// once the work queued here before their [`DeviceFrame::ready`] event
    /// has run.
    #[must_use]
    pub fn stream(&self) -> &Stream {
        &self.state.get().stream
    }
}

impl Decoder<Host> {
    /// The next decoded picture in display order, if any.
    ///
    /// The copy out of NVDEC into a page-locked host buffer runs on while
    /// the parser goes on; this waits for it. A picture whose copy failed
    /// is dropped (logged) and the next one returned.
    pub fn next_frame(&mut self) -> Option<HostFrame> {
        let Output::Host { queue, .. } = &mut self.state.get_mut().output else {
            return None;
        };
        while let Some(mut frame) = queue.pop_front() {
            match frame.slot.wait() {
                Ok(()) => return Some(frame),
                Err(error) => tracing::warn!(%error, "dropping a picture whose copy failed"),
            }
        }
        None
    }
}

impl Decoder<Device> {
    /// The next decoded picture in display order, if any.
    pub fn next_frame(&mut self) -> Option<DeviceFrame> {
        match &mut self.state.get_mut().output {
            Output::Device { queue, .. } => queue.pop_front(),
            Output::Host { .. } => None,
        }
    }
}

impl<M: OutputMode> fmt::Debug for Decoder<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Decoder")
            .field("state", self.state.get())
            .finish_non_exhaustive()
    }
}
