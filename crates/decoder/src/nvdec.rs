//! NVDEC hardware decode (H.264, HEVC, VP8, VP9, AV1) through
//! `ferricast-nvcodec`.
//!
//! One generic [`NvdecDecoder`] serves every codec; a sealed
//! [`NvdecCodec`] marker fixes the codec at compile time.
//!
//! - Timestamps travel through NVIDIA's parser with each packet and come
//!   back with the picture it started, in display order.
//! - Colour is the stream's own (H.264 / HEVC VUI, VP9 colour space, AV1
//!   colour config, as NVDEC reports it), each unspecified field filled
//!   from the container's signalling.
//! - HDR metadata: the SEI messages NVDEC hands over with each picture,
//!   else ferricast's own reading of the stream's SEI (H.264 / HEVC) or
//!   metadata OBUs (AV1), else the container's.
//! - Output: [`FrameMemory::Cpu`] decodes into pooled host buffers and
//!   hands them out as [`RawFrame`]s without a copy (NV12, or P010 for
//!   deeper streams when [`DecoderConfig::pixel_format`] asks for it);
//!   [`FrameMemory::Native`] keeps the pictures in CUDA device memory
//!   ([`CudaSurface`]), which the NVENC encoder takes without a copy.

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::Arc;

use bytes::Bytes;
use ferricast_codec::bitstream::NalSyntax;
use ferricast_codec::bitstream::hdr::HdrTracker;
use ferricast_codec::bitstream::vui::ColourTracker;
use ferricast_codec::cuda::nvcodec::decode::{
    self as nv, ChromaFormat, Decoder, DecoderCaps, Device, FrameInfo, Host, HostFrame, Packet,
    SeiMessages,
};
use ferricast_codec::cuda::{self, CudaSurface, map_error};
use ferricast_codec::{Candidate, VideoDecoderRegistry, VideoDecoderSession};
use ferricast_core::{
    Acceleration, BackendId, CapturedFrame, Codec, CodecError, CodecResult, CodecTarget,
    ColourDescription, ColourPrimaries, ContentLightLevel, DecoderConfig, EncodedFrame,
    FrameMemory, HdrMetadata, MasteringDisplay, MatrixCoefficients, NativeFrame, Operation,
    PixelFormat, RawFrame, TransferCharacteristics, VideoDecoder,
};
use tracing::{debug, info};

use crate::nv12::{self, Container, OutputFormat};

/// Host pictures NVDEC may have handed out at once (each holds a pooled
/// buffer until its last `Bytes` drops).
const MAX_HOST_FRAMES: usize = 64;
/// Past this many host pictures still held, new ones are copied out so the
/// pool never runs dry under a consumer that keeps pictures (a jitter
/// buffer).
const LEND_LIMIT: usize = 48;
/// Device pictures handed out at once.
const MAX_DEVICE_FRAMES: usize = 32;
/// `video_format` of a stream without a video signal type (H.273 5,
/// "unspecified").
const VIDEO_FORMAT_UNSPECIFIED: u8 = 5;

/// Where a codec's bitstream carries the HDR metadata ferricast reads
/// itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bitstream {
    /// SEI messages in Annex B NAL units (H.264, HEVC).
    Nal(NalSyntax),
    /// Metadata OBUs (AV1).
    Av1,
    /// Nowhere (VP8, VP9: only the container has it).
    Opaque,
}

/// Sealed marker pinning a [`NvdecDecoder`] to one codec.
pub trait NvdecCodec: sealed::Sealed + Send + 'static {
    /// The ferricast codec ([`VideoDecoder::CODEC`]).
    const FERRICAST_CODEC: Codec;
    /// The NVDEC codec.
    const NV_CODEC: nv::Codec;
    /// Where the bitstream carries HDR metadata (and, for H.264 / HEVC,
    /// the chroma sample location NVDEC does not report).
    const BITSTREAM: Bitstream;
}

mod sealed {
    pub trait Sealed {}
}

macro_rules! marker {
    ($(#[$meta:meta])* $name:ident, $codec:expr, $nv:expr, $bitstream:expr) => {
        $(#[$meta])*
        #[derive(Debug)]
        pub enum $name {}
        impl sealed::Sealed for $name {}
        impl NvdecCodec for $name {
            const FERRICAST_CODEC: Codec = $codec;
            const NV_CODEC: nv::Codec = $nv;
            const BITSTREAM: Bitstream = $bitstream;
        }
    };
}

marker!(
    /// H.264.
    H264,
    Codec::H264,
    nv::Codec::H264,
    Bitstream::Nal(NalSyntax::H264)
);
#[cfg(feature = "nvdec-hevc-decode")]
marker!(
    /// HEVC.
    Hevc,
    Codec::H265,
    nv::Codec::Hevc,
    Bitstream::Nal(NalSyntax::H265)
);
marker!(
    /// VP8.
    Vp8,
    Codec::Vp8,
    nv::Codec::Vp8,
    Bitstream::Opaque
);
marker!(
    /// VP9.
    Vp9,
    Codec::Vp9,
    nv::Codec::Vp9,
    Bitstream::Opaque
);
marker!(
    /// AV1.
    Av1,
    Codec::Av1,
    nv::Codec::Av1,
    Bitstream::Av1
);

/// The H.264 NVDEC decoder.
pub type NvdecH264Decoder = NvdecDecoder<H264>;
/// The HEVC NVDEC decoder.
#[cfg(feature = "nvdec-hevc-decode")]
pub type NvdecH265Decoder = NvdecDecoder<Hevc>;
/// The VP8 NVDEC decoder.
pub type NvdecVp8Decoder = NvdecDecoder<Vp8>;
/// The VP9 NVDEC decoder.
pub type NvdecVp9Decoder = NvdecDecoder<Vp9>;
/// The AV1 NVDEC decoder.
pub type NvdecAv1Decoder = NvdecDecoder<Av1>;

/// Adds NVDEC for `codec` to `registry`, as the first (hardware)
/// candidate of automatic selection.
pub(crate) fn register(registry: VideoDecoderRegistry, codec: Codec) -> VideoDecoderRegistry {
    match codec {
        Codec::H264 => registry.candidate(candidate::<H264>()),
        #[cfg(feature = "nvdec-hevc-decode")]
        Codec::H265 => registry.candidate(candidate::<Hevc>()),
        #[cfg(not(feature = "nvdec-hevc-decode"))]
        Codec::H265 => registry.missing(BackendId::Nvdec, "ferricast-decoder/nvdec-hevc-decode"),
        Codec::Vp8 => registry.candidate(candidate::<Vp8>()),
        Codec::Vp9 => registry.candidate(candidate::<Vp9>()),
        Codec::Av1 => registry.candidate(candidate::<Av1>()),
    }
}

fn candidate<C: NvdecCodec>() -> Candidate<DecoderConfig, dyn VideoDecoderSession> {
    Candidate::new(BackendId::Nvdec, Acceleration::Hardware, |config| {
        // Capability query first: cheap, and it names a missing driver or
        // codec instead of failing on the first packet.
        NvdecDecoder::<C>::probe()?;
        ferricast_codec::open_video_decoder::<NvdecDecoder<C>>(config)
    })
    .with_probe(NvdecDecoder::<C>::probe)
}

/// The decoder of one stream, in host or device mode.
enum Session {
    Host {
        decoder: Decoder<Host>,
        /// Cloned into every picture lent out without a copy: its count is
        /// how many are still held.
        lent: Arc<()>,
    },
    Device(Decoder<Device>),
}

/// NVDEC decoder for the codec `C`.
pub struct NvdecDecoder<C: NvdecCodec> {
    session: Option<Session>,
    /// One packet can release several pictures (a B-frame run ending);
    /// they wait here.
    ready: VecDeque<CapturedFrame>,
    format: OutputFormat,
    signalling: Signalling,
    _codec: PhantomData<fn() -> C>,
}

impl<C: NvdecCodec> Default for NvdecDecoder<C> {
    fn default() -> Self {
        Self {
            session: None,
            ready: VecDeque::new(),
            format: OutputFormat::default(),
            signalling: Signalling::new(C::BITSTREAM, Container::default()),
            _codec: PhantomData,
        }
    }
}

impl<C: NvdecCodec> NvdecDecoder<C> {
    /// Whether this machine's GPU decodes `C` (8-bit 4:2:0): loads the
    /// driver and asks it, without creating a decoder.
    ///
    /// # Errors
    ///
    /// [`CodecError::BackendUnavailable`] without an NVIDIA driver or GPU,
    /// or when the GPU has no decoder for the codec.
    pub fn probe() -> CodecResult<()> {
        let context = cuda::context().map_err(map_error(BackendId::Nvdec, Operation::Probe))?;
        let caps = DecoderCaps::query(&context, C::NV_CODEC, ChromaFormat::Yuv420, 8)
            .map_err(map_error(BackendId::Nvdec, Operation::Probe))?;
        if !caps.supported {
            return Err(CodecError::BackendUnavailable {
                backend: BackendId::Nvdec,
                reason: format!("this GPU does not decode {}", C::NV_CODEC),
            });
        }
        debug!(
            codec = %C::NV_CODEC,
            max_width = caps.max_width,
            max_height = caps.max_height,
            engines = caps.engines,
            "NVDEC caps probe OK"
        );
        Ok(())
    }

    fn open(config: &DecoderConfig) -> CodecResult<Session> {
        let context = cuda::context().map_err(map_error(BackendId::Nvdec, Operation::Open))?;
        let nv_config = nv::DecoderConfig::new(C::NV_CODEC);
        let session = match config.memory {
            FrameMemory::Cpu => Session::Host {
                decoder: Decoder::new(&context, nv_config.with_max_frames(MAX_HOST_FRAMES))
                    .map_err(map_error(BackendId::Nvdec, Operation::Open))?,
                lent: Arc::new(()),
            },
            FrameMemory::Native => Session::Device(
                Decoder::new(&context, nv_config.with_max_frames(MAX_DEVICE_FRAMES))
                    .map_err(map_error(BackendId::Nvdec, Operation::Open))?,
            ),
        };
        info!(codec = %C::NV_CODEC, memory = ?config.memory, "NVDEC decoder ready");
        Ok(session)
    }

    /// Moves every picture NVDEC has finished into `ready`.
    fn drain(&mut self) -> CodecResult<()> {
        let Self {
            session,
            ready,
            format,
            signalling,
            ..
        } = self;
        match session {
            None => {}
            Some(Session::Host { decoder, lent }) => {
                while let Some(frame) = decoder.next_frame() {
                    let raw = host_picture(frame, lent, format.0, signalling)?;
                    ready.push_back(CapturedFrame::Cpu(raw));
                }
            }
            Some(Session::Device(decoder)) => {
                while let Some(frame) = decoder.next_frame() {
                    let info = *frame.info();
                    let (colour, hdr) = signalling.picture(&info, frame.sei());
                    let surface = CudaSurface::new(frame).ok_or_else(|| no_pixel_format(&info))?;
                    ready.push_back(CapturedFrame::Native(NativeFrame {
                        width: info.width,
                        height: info.height,
                        timestamp_us: timestamp(&info),
                        colour: Some(colour),
                        hdr,
                        surface: Arc::new(surface),
                    }));
                }
            }
        }
        Ok(())
    }
}

/// The picture's timestamp: the packet's, as given to the parser.
fn timestamp(info: &FrameInfo) -> u64 {
    u64::try_from(info.timestamp).unwrap_or(0)
}

/// NVDEC decoded the stream to a layout ferricast has no pixel format for
/// (4:2:2, 4:4:4): another backend takes it over.
fn no_pixel_format(info: &FrameInfo) -> CodecError {
    CodecError::UnsupportedStream {
        backend: BackendId::Nvdec,
        reason: format!(
            "NVDEC decodes this stream to {:?}, which ferricast has no pixel format for",
            info.format
        ),
    }
}

/// A host picture as a [`RawFrame`] in the requested `format`. NV12 and
/// P010 pictures are lent without a copy while few are held.
fn host_picture(
    frame: HostFrame,
    lent: &Arc<()>,
    format: PixelFormat,
    signalling: &mut Signalling,
) -> CodecResult<RawFrame> {
    let info = *frame.info();
    let (colour, hdr) = signalling.picture(&info, frame.sei());
    let deep = match info.format {
        nv::PixelFormat::Nv12 => false,
        nv::PixelFormat::P016 => true,
        nv::PixelFormat::Nv16
        | nv::PixelFormat::P216
        | nv::PixelFormat::Yuv444
        | nv::PixelFormat::Yuv444P16 => return Err(no_pixel_format(&info)),
    };
    let size = (info.width as usize, info.height as usize);
    let (data, stride) = HostPicture::bytes(frame, lent);
    let deliver = if deep {
        nv12::deliver_p010
    } else {
        nv12::deliver
    };
    let mut raw =
        deliver(data, stride, size, timestamp(&info), format, colour).ok_or_else(|| {
            CodecError::backend(
                BackendId::Nvdec,
                Operation::Decode,
                "NVDEC returned short planes",
            )
        })?;
    raw.hdr = hdr;
    Ok(raw)
}

/// A host frame lent to a `Bytes`: its pooled buffer goes back to the
/// decoder when the last clone of the bytes drops.
struct HostPicture {
    frame: HostFrame,
    _lent: Arc<()>,
}

impl AsRef<[u8]> for HostPicture {
    fn as_ref(&self) -> &[u8] {
        self.frame.data()
    }
}

impl HostPicture {
    /// The planes of a two-plane picture as one buffer with one stride:
    /// the frame itself when both planes have the same pitch (even widths)
    /// and few frames are lent, a packed copy otherwise.
    fn bytes(frame: HostFrame, lent: &Arc<()>) -> (Bytes, usize) {
        let layout = frame.layout();
        if let [luma, chroma] = layout.planes()
            && luma.pitch == chroma.pitch
        {
            let stride = luma.pitch;
            if Arc::strong_count(lent) <= LEND_LIMIT {
                let picture = Self {
                    frame,
                    _lent: Arc::clone(lent),
                };
                return (Bytes::from_owner(picture), stride);
            }
            return (Bytes::copy_from_slice(frame.data()), stride);
        }
        // Odd widths: the chroma row holds one sample more than the luma
        // row; ferricast's NV12 / P010 frames use the chroma pitch for both.
        let planes = layout.planes();
        let stride = planes.iter().map(|plane| plane.pitch).max().unwrap_or(0);
        let rows: usize = planes.iter().map(|plane| plane.rows).sum();
        let mut out = vec![0; stride * rows];
        let mut row = 0;
        for (index, plane) in planes.iter().enumerate() {
            let Some(data) = frame.plane(index) else {
                break;
            };
            for line in data.chunks_exact(plane.pitch) {
                if let Some(target) = out.get_mut(row * stride..row * stride + plane.width_bytes) {
                    target.copy_from_slice(&line[..plane.width_bytes]);
                }
                row += 1;
            }
        }
        (Bytes::from(out), stride)
    }
}

/// What a stream signals about its colour and HDR metadata: the
/// container's description, and what ferricast reads from the bitstream
/// where NVDEC does not report it.
#[derive(Debug, Clone)]
struct Signalling {
    bitstream: Bitstream,
    container: Container,
    /// The chroma sample location of H.264 / HEVC VUIs.
    vui: ColourTracker,
    /// HDR metadata ferricast read from the stream itself.
    read: HdrTracker,
    /// HDR metadata from the SEI messages NVDEC handed over; once it has
    /// any, ferricast stops reading SEI itself.
    nvdec: HdrMetadata,
}

impl Signalling {
    fn new(bitstream: Bitstream, container: Container) -> Self {
        Self {
            bitstream,
            container,
            vui: ColourTracker::default(),
            read: HdrTracker::default(),
            nvdec: HdrMetadata::default(),
        }
    }

    /// Reads one packet before NVDEC decodes it.
    fn observe(&mut self, packet: &[u8]) {
        match self.bitstream {
            Bitstream::Nal(syntax) => {
                self.vui.observe_annex_b(packet, syntax);
                if self.nvdec.is_empty() {
                    self.read.observe_annex_b(packet, syntax);
                }
            }
            Bitstream::Av1 => self.read.observe_av1(packet),
            Bitstream::Opaque => {}
        }
    }

    /// The colour and HDR metadata of a decoded picture with `sei`.
    fn picture(
        &mut self,
        info: &FrameInfo,
        sei: SeiMessages<'_>,
    ) -> (ColourDescription, Option<HdrMetadata>) {
        if let Some(display) = sei.clone().mastering_display() {
            self.nvdec.mastering_display = Some(mastering_display(display));
        }
        if let Some(light) = sei.content_light_level() {
            self.nvdec.content_light = Some(ContentLightLevel {
                max_cll: light.max_content,
                max_fall: light.max_frame_average,
            });
        }
        (self.colour(info.colour), self.hdr())
    }

    /// The stream's HDR metadata (NVDEC's SEI, else what ferricast read),
    /// per half; the container's when the stream has none.
    fn hdr(&self) -> Option<HdrMetadata> {
        let read = self.read.current_or(None).unwrap_or_default();
        let stream = HdrMetadata {
            mastering_display: self.nvdec.mastering_display.or(read.mastering_display),
            content_light: self.nvdec.content_light.or(read.content_light),
        };
        if stream.is_empty() {
            self.container.hdr
        } else {
            Some(stream)
        }
    }

    /// NVDEC's colour, each unspecified field taken from the container.
    fn colour(&self, nvdec: nv::Colour) -> ColourDescription {
        let stream = ColourDescription::from_codes(
            nvdec.primaries,
            nvdec.transfer,
            nvdec.matrix,
            nvdec.full_range,
        );
        let location = self.vui.chroma_location();
        let Some(container) = self.container.colour else {
            return match location {
                Some(location) => stream.with_chroma_location(location),
                None => stream,
            };
        };
        // A stream without a video signal type reports all of it
        // unspecified (and limited range): the container's range applies.
        let signalled = nvdec.full_range
            || nvdec.video_format != VIDEO_FORMAT_UNSPECIFIED
            || stream.primaries != ColourPrimaries::Unspecified
            || stream.transfer != TransferCharacteristics::Unspecified
            || stream.matrix != MatrixCoefficients::Unspecified;
        ColourDescription {
            primaries: if stream.primaries == ColourPrimaries::Unspecified {
                container.primaries
            } else {
                stream.primaries
            },
            transfer: if stream.transfer == TransferCharacteristics::Unspecified {
                container.transfer
            } else {
                stream.transfer
            },
            matrix: if stream.matrix == MatrixCoefficients::Unspecified {
                container.matrix
            } else {
                stream.matrix
            },
            range: if signalled {
                stream.range
            } else {
                container.range
            },
            chroma_location: location.or(container.chroma_location),
        }
    }
}

/// NVDEC's decoded mastering display SEI as ferricast's, through the SEI
/// payload (primaries in bitstream order: green, blue, red).
fn mastering_display(display: nv::MasteringDisplay) -> MasteringDisplay {
    let mut payload = [0_u8; MasteringDisplay::SEI_LEN];
    let points = display.primaries.into_iter().chain([display.white_point]);
    for (chunk, (x, y)) in payload.chunks_exact_mut(4).zip(points) {
        chunk[..2].copy_from_slice(&x.to_be_bytes());
        chunk[2..].copy_from_slice(&y.to_be_bytes());
    }
    payload[16..20].copy_from_slice(&display.max_luminance.to_be_bytes());
    payload[20..24].copy_from_slice(&display.min_luminance.to_be_bytes());
    MasteringDisplay::from_sei(&payload)
        .expect("invariant: the payload is exactly MasteringDisplay::SEI_LEN bytes")
}

impl<C: NvdecCodec> VideoDecoder for NvdecDecoder<C> {
    const CODEC: Codec = C::FERRICAST_CODEC;

    fn configure(&mut self, config: &DecoderConfig) -> CodecResult<()> {
        if config.codec != C::FERRICAST_CODEC {
            return Err(CodecError::UnsupportedConfig {
                backend: BackendId::Nvdec,
                what: format!("{:?} with the {} decoder", config.codec, C::NV_CODEC),
            });
        }
        self.session = Some(Self::open(config)?);
        self.ready.clear();
        self.format = OutputFormat(config.pixel_format);
        self.signalling = Signalling::new(C::BITSTREAM, Container::of(config));
        Ok(())
    }

    fn decode(&mut self, frame: EncodedFrame) -> CodecResult<Option<CapturedFrame>> {
        let session = self.session.as_mut().ok_or(CodecError::NotConfigured {
            target: CodecTarget::decode(C::FERRICAST_CODEC),
            operation: Operation::Decode,
        })?;
        // Read before NVDEC sees the packet, so its parameter sets count
        // whatever happens to the decode.
        self.signalling.observe(&frame.data);
        let packet = Packet::new(
            &frame.data,
            i64::try_from(frame.timestamp_us).unwrap_or(i64::MAX),
        );
        match session {
            Session::Host { decoder, .. } => decoder.decode(packet),
            Session::Device(decoder) => decoder.decode(packet),
        }
        .map_err(map_error(BackendId::Nvdec, Operation::Decode))?;
        self.drain()?;
        Ok(self.ready.pop_front())
    }

    fn flush(&mut self) -> CodecResult<Vec<CapturedFrame>> {
        match self.session.as_mut() {
            Some(Session::Host { decoder, .. }) => decoder.flush(),
            Some(Session::Device(decoder)) => decoder.flush(),
            None => Ok(()),
        }
        .map_err(map_error(BackendId::Nvdec, Operation::Flush))?;
        self.drain()?;
        Ok(self.ready.drain(..).collect())
    }
}

#[cfg(test)]
mod tests {
    use ferricast_core::ColourRange;

    use super::*;

    fn nvdec_colour(primaries: u8, transfer: u8, matrix: u8, video_format: u8) -> nv::Colour {
        nv::Colour {
            primaries,
            transfer,
            matrix,
            full_range: false,
            video_format,
        }
    }

    fn signalling(colour: Option<ColourDescription>, hdr: Option<HdrMetadata>) -> Signalling {
        Signalling::new(Bitstream::Opaque, Container { colour, hdr })
    }

    #[test]
    fn the_stream_colour_wins_and_the_container_fills_what_it_leaves_unspecified() {
        let container = ColourDescription::from_codes(9, 16, 9, true);
        // VP9 profile 2: BT.2020 colour space in the bitstream, no transfer.
        let colour = signalling(Some(container), None).colour(nvdec_colour(9, 2, 9, 0));
        assert_eq!(colour.primaries, ColourPrimaries::Bt2020);
        assert_eq!(colour.transfer, TransferCharacteristics::Pq);
        // The stream has a signal type: its (limited) range stands.
        assert_eq!(colour.range, ColourRange::Limited);
    }

    #[test]
    fn a_stream_without_a_signal_type_takes_the_container_range() {
        let container = ColourDescription::from_codes(1, 1, 1, true);
        let colour = signalling(Some(container), None).colour(nvdec_colour(2, 2, 2, 5));
        assert_eq!(colour, container);
        let alone = signalling(None, None).colour(nvdec_colour(2, 2, 2, 5));
        assert_eq!(alone, ColourDescription::unspecified(ColourRange::Limited));
    }

    #[test]
    fn nvdec_sei_arrives_as_ferricast_metadata_and_wins_over_the_container() {
        let display = nv::MasteringDisplay {
            primaries: [(8500, 39850), (6550, 2300), (35400, 14600)],
            white_point: (15635, 16450),
            max_luminance: 10_000_000,
            min_luminance: 1,
        };
        let ours = mastering_display(display);
        assert_eq!((ours.green.x, ours.blue.x, ours.red.x), (8500, 6550, 35400));
        assert_eq!(ours.white_point.y, 16450);
        assert_eq!((ours.max_luminance, ours.min_luminance), (10_000_000, 1));

        let container = HdrMetadata {
            mastering_display: None,
            content_light: Some(ContentLightLevel {
                max_cll: 1,
                max_fall: 1,
            }),
        };
        let mut signalling = signalling(None, Some(container));
        assert_eq!(signalling.hdr(), Some(container));
        signalling.nvdec.mastering_display = Some(ours);
        // The stream's metadata replaces the container's as a whole.
        assert_eq!(
            signalling.hdr(),
            Some(HdrMetadata {
                mastering_display: Some(ours),
                content_light: None,
            })
        );
    }
}
