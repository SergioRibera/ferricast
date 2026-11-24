//! H.264 / H.265 encode through `AMediaCodec` with byte-buffer NV12
//! input, or P010 for HEVC Main10 ([`BitDepth::Ten`]).
//!
//! A [`HardwareBuffer`] picture (`CapturedFrame::Native`, e.g. from the
//! `MediaCodec` decoder) is locked and copied straight into the input
//! buffer, with no colour conversion and no intermediate frame. True
//! zero-copy input would go through `AMediaCodec_createInputSurface`,
//! which needs an EGL renderer to put buffers into it; that is not done.
//!
//! `MediaCodec` only produces the stream's parameter sets (one
//! `BUFFER_FLAG_CODEC_CONFIG` buffer) after its first input, and many
//! hardware encoders hold a picture or two before the first output.
//! `configure` therefore primes the session: it feeds black pictures
//! until the parameter sets and a first picture came out, discards that
//! output and asks for a sync frame. From then on the pipeline is full,
//! every `encode` call queues one picture and returns the oldest one
//! the codec finished (pictures leave in order: no B-frames), and
//! `get_headers` has the parameter sets from the start. Every keyframe
//! carries them in-band, as the x264 backend's do. An encoder that
//! needs more input before it releases a picture than priming filled
//! reports "no picture came out" for the first frames, as x264 does for
//! a buffered picture, and catches up one picture per frame after.

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use bytes::Bytes;
use ferricast_codec::android::HardwareBuffer;
use ferricast_codec::yuv420::{self, Crop, SemiPlanar};
use ferricast_core::{
    BackendId, BitDepth, CapturedFrame, Codec, CodecError, CodecResult, ColourDescription,
    ColourPrimaries, ColourRange, EncodedFrame, EncoderConfig, H264Profile, H265Profile,
    HdrMetadata, MatrixCoefficients, Operation, PixelFormat, TransferCharacteristics, VideoEncoder,
};
use ndk::media::media_codec::MediaFormat;
use tracing::{info, warn};

use super::codec::{
    self, COLOR_FORMAT_NV12, COLOR_FORMAT_P010, Codec as MediaCodec, FLAG_CODEC_CONFIG,
    FLAG_END_OF_STREAM, FLAG_KEY_FRAME, Output, Pick, fail, key, media,
};
use crate::frame_view::FrameView;
use crate::nv12::{self, Nv12Layout};
use crate::scale::{FrameScaler, FrameSize};

/// Black pictures fed at most while priming, enough for any encoder's
/// pipeline depth.
const MAX_PRIMING_FRAMES: usize = 8;

/// How long `encode` waits for the codec to finish a picture.
const OUTPUT_TIMEOUT: Duration = Duration::from_millis(250);

/// `MediaCodecInfo.EncoderCapabilities` bitrate modes.
const BITRATE_MODE_VBR: i32 = 1;
const BITRATE_MODE_CBR: i32 = 2;

/// The MIME type of `codec`, when `MediaCodec` encodes it.
pub(crate) fn mime(codec: Codec) -> Option<&'static str> {
    match codec {
        Codec::H264 => Some("video/avc"),
        Codec::H265 => Some("video/hevc"),
        Codec::Vp8 | Codec::Vp9 | Codec::Av1 => None,
    }
}

/// The Google software encoder of `codec`.
pub(crate) fn software_name(codec: Codec) -> Option<&'static str> {
    match codec {
        Codec::H264 => Some("c2.android.avc.encoder"),
        Codec::H265 => Some("c2.android.hevc.encoder"),
        Codec::Vp8 | Codec::Vp9 | Codec::Av1 => None,
    }
}

/// A real picture handed to the codec.
#[derive(Debug, Clone, Copy)]
struct Submitted {
    /// Its capture timestamp.
    capture_us: u64,
    /// Its input-order index.
    index: u64,
}

struct Session {
    codec: MediaCodec,
    kind: Codec,
    width: u32,
    height: u32,
    fps: u32,
    colour: ColourDescription,
    depth: BitDepth,
    hdr: Option<HdrMetadata>,
    /// The codec's input buffer layout, at `depth`.
    input: SemiPlanar,
    /// `input` for the 8-bit NV12 writer (equal strides).
    layout: Nv12Layout,
    /// NV12 staging for RGB input to a 10-bit session.
    narrow: Vec<u8>,
    /// Parameter sets, Annex B.
    headers: Vec<u8>,
    /// Presentation time handed to the codec → capture timestamp of the
    /// picture (`None` for a priming picture, whose output is dropped).
    in_flight: HashMap<u64, Option<Submitted>>,
    /// Input-order index of the next real picture: `pts_dts` units.
    next_index: u64,
    next_pts: u64,
    finished: VecDeque<EncodedFrame>,
    /// A priming picture came out: the pipeline is full.
    primed: bool,
    scratch: Vec<u8>,
}

/// A `MediaCodec` video encoder; `pick` chooses which codec of the type.
pub(crate) struct MediaCodecVideoEncoder {
    pick: Pick,
    session: Option<Session>,
    scaler: FrameScaler,
    pending_keyframe: bool,
}

impl MediaCodecVideoEncoder {
    pub(crate) fn new(pick: Pick) -> Self {
        Self {
            pick,
            session: None,
            scaler: FrameScaler::default(),
            pending_keyframe: false,
        }
    }

    fn session(&mut self, operation: Operation) -> CodecResult<&mut Session> {
        self.session
            .as_mut()
            .ok_or_else(|| fail(operation, "the encoder is not configured"))
    }
}

/// Opens a session for `config`; the candidate's open function.
pub(crate) fn open(pick: Pick, config: &EncoderConfig) -> CodecResult<MediaCodecVideoEncoder> {
    let mut encoder = MediaCodecVideoEncoder::new(pick);
    VideoEncoder::configure(&mut encoder, config)?;
    Ok(encoder)
}

/// `MediaFormat` colour keys (`MediaFormat.COLOR_STANDARD_*`, ...) for
/// `colour`; `None` for a value `MediaCodec` has no key for.
fn colour_keys(colour: ColourDescription) -> (Option<i32>, i32, Option<i32>) {
    let standard = match (colour.primaries, colour.matrix) {
        (ColourPrimaries::Bt709, _) | (_, MatrixCoefficients::Bt709) => Some(1),
        (ColourPrimaries::Bt470Bg, _) | (_, MatrixCoefficients::Bt470Bg) => Some(2),
        (ColourPrimaries::Smpte170M, _) | (_, MatrixCoefficients::Smpte170M) => Some(4),
        (ColourPrimaries::Bt2020, _) | (_, MatrixCoefficients::Bt2020Ncl) => Some(6),
        _ => None,
    };
    let range = match colour.range {
        ColourRange::Full => 1,
        ColourRange::Limited => 2,
    };
    let transfer = match colour.transfer {
        TransferCharacteristics::Linear => Some(1),
        TransferCharacteristics::Bt709
        | TransferCharacteristics::Smpte170M
        | TransferCharacteristics::Bt2020Ten
        | TransferCharacteristics::Bt2020Twelve => Some(3),
        TransferCharacteristics::Pq => Some(6),
        TransferCharacteristics::Hlg => Some(7),
        _ => None,
    };
    (standard, range, transfer)
}

/// `MediaCodecInfo.CodecProfileLevel.HEVCProfileMain10`.
const HEVC_MAIN10: i32 = 2;
/// `MediaCodecInfo.CodecProfileLevel.HEVCProfileMain10HDR10`.
const HEVC_MAIN10_HDR10: i32 = 0x1000;

/// `MediaCodecInfo.CodecProfileLevel` of the configured profile cap and
/// bit depth.
///
/// # Errors
/// [`CodecError::UnsupportedConfig`] for 10-bit H.264, or 10-bit HEVC
/// with the profile capped at Main.
fn profile(config: &EncoderConfig) -> CodecResult<Option<i32>> {
    let unsupported = |what: String| CodecError::UnsupportedConfig {
        backend: BackendId::MediaCodec,
        what,
    };
    Ok(match (config.codec, config.bit_depth) {
        (Codec::H265, BitDepth::Ten) => {
            if config.max_h265_profile == Some(H265Profile::Main) {
                return Err(unsupported(
                    "10-bit HEVC with the profile capped at Main (8-bit)".into(),
                ));
            }
            // HDR10 is Main10 with static metadata in the stream.
            Some(if config.hdr.is_some() {
                HEVC_MAIN10_HDR10
            } else {
                HEVC_MAIN10
            })
        }
        (codec @ (Codec::H264 | Codec::Vp8 | Codec::Vp9 | Codec::Av1), BitDepth::Ten) => {
            return Err(unsupported(format!(
                "10-bit {codec} encode (MediaCodec encodes 10 bits as HEVC Main10 only)"
            )));
        }
        (Codec::H264, BitDepth::Eight) => config.max_h264_profile.map(|profile| match profile {
            H264Profile::Baseline => 1,
            H264Profile::Main => 2,
            H264Profile::High => 8,
        }),
        (Codec::H265, BitDepth::Eight) => config.max_h265_profile.map(|profile| match profile {
            H265Profile::Main => 1,
            H265Profile::Main10 => HEVC_MAIN10,
        }),
        (Codec::Vp8 | Codec::Vp9 | Codec::Av1, BitDepth::Eight) => None,
    })
}

/// The `MediaCodec` input colour format of `depth`.
fn colour_format(depth: BitDepth) -> i32 {
    match depth {
        BitDepth::Eight => COLOR_FORMAT_NV12,
        BitDepth::Ten => COLOR_FORMAT_P010,
    }
}

fn format(config: &EncoderConfig, mime: &str, bitrate_mode: i32) -> CodecResult<MediaFormat> {
    let dimension = |value: u32, what: &str| {
        i32::try_from(value).map_err(|_| CodecError::UnsupportedConfig {
            backend: ferricast_core::BackendId::MediaCodec,
            what: format!("a {what} of {value}"),
        })
    };
    let mut format = MediaFormat::new();
    format.set_str(key::MIME, mime);
    format.set_i32(key::WIDTH, dimension(config.width, "width")?);
    format.set_i32(key::HEIGHT, dimension(config.height, "height")?);
    format.set_i32(key::COLOR_FORMAT, colour_format(config.bit_depth));
    format.set_i32(
        key::BIT_RATE,
        dimension(config.bitrate_kbps.saturating_mul(1000), "bitrate")?,
    );
    format.set_i32(key::BITRATE_MODE, bitrate_mode);
    format.set_i32(key::FRAME_RATE, dimension(config.fps.max(1), "frame rate")?);
    // Whole seconds: an int32 is what every API level reads (a float is
    // only honoured from API 25, and not by every codec).
    // Clamped to 1..=3600 first, so the cast cannot truncate.
    let keyframe_interval = config.keyframe_interval_secs.ceil().clamp(1.0, 3600.0) as i32;
    format.set_i32(key::I_FRAME_INTERVAL, keyframe_interval);
    // Real-time priority, one picture of latency, no B-frames: the
    // keys a codec does not know are ignored.
    format.set_i32(key::PRIORITY, 0);
    format.set_i32(key::LATENCY, 1);
    format.set_i32(key::MAX_B_FRAMES, 0);
    format.set_i32(key::PREPEND_HEADERS_TO_SYNC_FRAMES, 1);
    if let Some(profile) = profile(config)? {
        format.set_i32(key::PROFILE, profile);
    }
    if let Some(hdr) = config.hdr.filter(|hdr| !hdr.is_empty()) {
        format.set_buffer(key::HDR_STATIC_INFO, &hdr.to_cta861());
    }
    let (standard, range, transfer) = colour_keys(config.colour);
    if let Some(standard) = standard {
        format.set_i32(key::COLOR_STANDARD, standard);
    }
    format.set_i32(key::COLOR_RANGE, range);
    if let Some(transfer) = transfer {
        format.set_i32(key::COLOR_TRANSFER, transfer);
    }
    Ok(format)
}

/// Whether an Annex B access unit starts with a parameter set.
/// NAL kinds of an access unit that matter for in-band parameter sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Nal {
    ParameterSet,
    AccessUnitDelimiter,
    Sei,
    Other,
}

fn nal_kind(kind: Codec, header: u8) -> Nal {
    match kind {
        Codec::H265 => match (header >> 1) & 0x3f {
            32..=34 => Nal::ParameterSet,
            35 => Nal::AccessUnitDelimiter,
            39 | 40 => Nal::Sei,
            _ => Nal::Other,
        },
        Codec::H264 | Codec::Vp8 | Codec::Vp9 | Codec::Av1 => match header & 0x1f {
            7 | 8 => Nal::ParameterSet,
            9 => Nal::AccessUnitDelimiter,
            6 => Nal::Sei,
            _ => Nal::Other,
        },
    }
}

/// `data` (an Annex B keyframe) with `headers` in front of its first
/// slice unless it already carries parameter sets before it; a leading
/// access unit delimiter stays first, as both standards require.
fn with_parameter_sets(kind: Codec, data: Vec<u8>, headers: &[u8]) -> Vec<u8> {
    let units = ferricast_codec::bitstream::split_annex_b(&data);
    let mut leading = units
        .iter()
        .map(|nal| {
            nal.first()
                .map_or(Nal::Other, |&header| nal_kind(kind, header))
        })
        .take_while(|nal| *nal != Nal::Other);
    if leading.any(|nal| nal == Nal::ParameterSet) {
        return data;
    }
    // After a leading delimiter: the byte offset where the unit after it
    // (with its start code) begins.
    let split = units
        .first()
        .filter(|nal| {
            nal.first()
                .is_some_and(|&h| nal_kind(kind, h) == Nal::AccessUnitDelimiter)
        })
        .map_or(0, |aud| {
            aud.as_ptr() as usize - data.as_ptr() as usize + aud.len()
        });
    let mut out = Vec::with_capacity(headers.len() + data.len());
    out.extend_from_slice(&data[..split]);
    out.extend_from_slice(headers);
    out.extend_from_slice(&data[split..]);
    out
}

impl Session {
    fn open(pick: Pick, config: &EncoderConfig) -> CodecResult<Self> {
        let mime = mime(config.codec).ok_or_else(|| CodecError::BackendCannotHandle {
            backend: ferricast_core::BackendId::MediaCodec,
            target: ferricast_core::CodecTarget::encode(config.codec),
        })?;
        // Rejecting a colour this crate cannot convert to now beats
        // failing on the first frame.
        nv12::conversion(config.colour, ferricast_core::BackendId::MediaCodec)?;
        profile(config)?;
        let codec = MediaCodec::create(mime, pick)?;
        // Constant bitrate suits a live link; a codec without it gets VBR.
        let started = codec
            .start_encoder(&format(config, mime, BITRATE_MODE_CBR)?)
            .or_else(|_| codec.start_encoder(&format(config, mime, BITRATE_MODE_VBR)?));
        if let Err(error) = started {
            return Err(CodecError::UnsupportedConfig {
                backend: ferricast_core::BackendId::MediaCodec,
                what: format!(
                    "{}x{} @ {} fps {}-bit on {} ({error})",
                    config.width,
                    config.height,
                    config.fps,
                    config.bit_depth.bits(),
                    codec.label()
                ),
            });
        }
        let input = codec.input_format();
        let width = config.width as usize;
        let stride = input
            .i32(key::STRIDE)
            .and_then(|s| usize::try_from(s).ok())
            .filter(|&s| s >= width)
            .unwrap_or(width);
        let slice_height = input
            .i32(key::SLICE_HEIGHT)
            .and_then(|h| usize::try_from(h).ok())
            .filter(|&h| h >= config.height as usize)
            .unwrap_or(config.height as usize);
        let depth = config.bit_depth;
        let wanted = colour_format(depth);
        if let Some(color_format) = input.i32(key::COLOR_FORMAT)
            && color_format != wanted
        {
            return Err(CodecError::UnsupportedConfig {
                backend: ferricast_core::BackendId::MediaCodec,
                what: format!(
                    "{} input colour format {color_format:#x} (only {wanted:#x} input is written)",
                    codec.label()
                ),
            });
        }
        // A P010 `stride` is a sample count by `KEY_STRIDE`'s definition;
        // one at least two rows of samples wide is already in bytes.
        let stride_bytes = match depth {
            BitDepth::Eight => stride,
            BitDepth::Ten if stride >= 2 * width => stride,
            BitDepth::Ten => stride * 2,
        };
        info!(codec = %codec.label(), width = config.width, height = config.height, bits = depth.bits(), "MediaCodec encoder opened");
        let mut session = Self {
            codec,
            kind: config.codec,
            width: config.width,
            height: config.height,
            fps: config.fps.max(1),
            colour: config.colour,
            depth,
            hdr: config.hdr,
            input: SemiPlanar {
                stride: stride_bytes,
                chroma_offset: stride_bytes * slice_height,
            },
            layout: Nv12Layout {
                width: config.width,
                height: config.height,
                stride,
                chroma_offset: stride * slice_height,
            },
            narrow: Vec::new(),
            headers: Vec::new(),
            in_flight: HashMap::new(),
            next_index: 0,
            next_pts: 0,
            finished: VecDeque::new(),
            primed: false,
            scratch: Vec::new(),
        };
        session.prime()?;
        Ok(session)
    }

    /// Feeds black pictures until the parameter sets and a first picture
    /// came out, then drops that output and asks for a sync frame.
    fn prime(&mut self) -> CodecResult<()> {
        let black = self.black();
        for _ in 0..MAX_PRIMING_FRAMES {
            let pts = self.alloc_pts(None);
            let codec = &self.codec;
            codec.queue(
                Operation::Open,
                pts,
                0,
                |buffer| codec::copy_into(buffer, &black, Operation::Open),
                |_| Ok(()),
            )?;
            while let Some(output) = self.codec.next_output(Operation::Open, codec::WAIT)? {
                self.accept(output);
            }
            if !self.headers.is_empty() && self.primed {
                break;
            }
        }
        if !self.primed {
            while let Some(output) = self.codec.next_output(Operation::Open, OUTPUT_TIMEOUT)? {
                self.accept(output);
                if self.primed {
                    break;
                }
            }
        }
        // Priming pictures still in flight keep the pipeline full; their
        // output is dropped when it comes.
        if self.headers.is_empty() || !self.primed {
            return Err(fail(
                Operation::Open,
                "the encoder produced no parameter sets or picture for its first pictures",
            ));
        }
        self.finished.clear();
        self.request_sync_frame();
        Ok(())
    }

    /// Bytes of one input picture.
    fn input_len(&self) -> usize {
        self.input.len(self.height as usize)
    }

    /// A black picture in the input layout: 8-bit 16 / 128 (limited
    /// range), the same values in the high byte of each P010 sample.
    fn black(&self) -> Vec<u8> {
        let mut black = vec![0; self.input_len()];
        let (luma, chroma) = black.split_at_mut(self.input.chroma_offset);
        match self.depth {
            BitDepth::Eight => {
                luma.fill(16);
                chroma.fill(128);
            }
            BitDepth::Ten => {
                for sample in luma.chunks_exact_mut(2) {
                    sample[1] = 16;
                }
                for sample in chroma.chunks_exact_mut(2) {
                    sample[1] = 128;
                }
            }
        }
        black
    }

    /// Writes `frame` into the scratch input picture: a matching
    /// [`HardwareBuffer`] is copied straight from its locked planes,
    /// anything else goes through CPU memory (and the scaler).
    fn fill(&mut self, frame: CapturedFrame, scaler: &mut FrameScaler) -> CodecResult<()> {
        self.scratch.resize(self.input_len(), 0);
        if let CapturedFrame::Native(native) = &frame
            && native.width == self.width
            && native.height == self.height
            && let Some(buffer) = native.surface::<HardwareBuffer>()
        {
            return buffer
                .write_to(self.depth, self.input, &mut self.scratch)
                .map_err(|error| fail(Operation::Encode, error));
        }
        let raw = frame
            .into_cpu()
            .map_err(|error| fail(Operation::Encode, error))?;
        let raw = scaler
            .fit(raw, FrameSize::new(self.width, self.height))
            .map_err(|error| fail(Operation::Encode, error))?;
        let view = FrameView::new(&raw, self.width, self.height)
            .map_err(|error| fail(Operation::Encode, error))?;
        let crop = Crop::full(self.width as usize, self.height as usize);
        let pack = |planes, from, dst: &mut [u8], to, layout| {
            yuv420::pack(planes, crop, from, to, layout, dst)
                .map_err(|error| fail(Operation::Encode, error))
        };
        let semi = |data, depth| {
            yuv420::semi_planar(data, view.stride(), view.chroma_offset(), depth)
                .map_err(|error| fail(Operation::Encode, error))
        };
        match (self.depth, view.format()) {
            (BitDepth::Eight, PixelFormat::P010) => pack(
                semi(view.data(), BitDepth::Ten)?,
                BitDepth::Ten,
                &mut self.scratch,
                BitDepth::Eight,
                self.input,
            ),
            (
                BitDepth::Eight,
                PixelFormat::Bgra | PixelFormat::Rgba | PixelFormat::Nv12 | PixelFormat::I420,
            ) => nv12::write(
                &view,
                self.colour,
                self.layout,
                &mut self.scratch,
                BackendId::MediaCodec,
            ),
            (BitDepth::Ten, PixelFormat::P010 | PixelFormat::Nv12) => {
                let from = view.format().bit_depth();
                pack(
                    semi(view.data(), from)?,
                    from,
                    &mut self.scratch,
                    BitDepth::Ten,
                    self.input,
                )
            }
            (BitDepth::Ten, PixelFormat::I420) => {
                let planes = yuv420::planar(view.data(), view.stride(), self.height as usize)
                    .map_err(|error| fail(Operation::Encode, error))?;
                pack(
                    planes,
                    BitDepth::Eight,
                    &mut self.scratch,
                    BitDepth::Ten,
                    self.input,
                )
            }
            (BitDepth::Ten, PixelFormat::Bgra | PixelFormat::Rgba) => {
                // RGB converts to 8-bit NV12 with the configured matrix,
                // then widens: 8-bit precision in a 10-bit stream.
                let narrow = Nv12Layout::packed(self.width, self.height);
                self.narrow.resize(narrow.len(), 0);
                nv12::write(
                    &view,
                    self.colour,
                    narrow,
                    &mut self.narrow,
                    BackendId::MediaCodec,
                )?;
                let planes = yuv420::semi_planar(
                    &self.narrow,
                    narrow.stride,
                    narrow.chroma_offset,
                    BitDepth::Eight,
                )
                .map_err(|error| fail(Operation::Encode, error))?;
                pack(
                    planes,
                    BitDepth::Eight,
                    &mut self.scratch,
                    BitDepth::Ten,
                    self.input,
                )
            }
        }
    }

    fn alloc_pts(&mut self, capture_us: Option<u64>) -> u64 {
        let pts = self.next_pts;
        self.next_pts += 1_000_000 / u64::from(self.fps);
        let submitted = capture_us.map(|capture_us| {
            let index = self.next_index;
            self.next_index += 1;
            Submitted { capture_us, index }
        });
        self.in_flight.insert(pts, submitted);
        pts
    }

    /// Files one output buffer: parameter sets become the headers, a
    /// priming picture is dropped, a real one is queued for `encode`.
    fn accept(&mut self, output: Output) {
        if output.is(FLAG_CODEC_CONFIG) {
            self.headers = output.data;
            return;
        }
        if output.data.is_empty() {
            return;
        }
        let Some(capture) = self.in_flight.remove(&output.pts_us) else {
            warn!(
                pts = output.pts_us,
                "MediaCodec output has no matching input"
            );
            return;
        };
        let Some(Submitted {
            capture_us: timestamp_us,
            index,
        }) = capture
        else {
            self.primed = true;
            return;
        };
        let is_keyframe = output.is(FLAG_KEY_FRAME);
        let data = if is_keyframe {
            with_parameter_sets(self.kind, output.data, &self.headers)
        } else {
            output.data
        };
        self.finished.push_back(EncodedFrame {
            codec: self.kind,
            data: Bytes::from(data),
            timestamp_us,
            is_keyframe,
            duration_us: Some(1_000_000 / u64::from(self.fps)),
            // Frame indices; with B-frames off the codec keeps input order,
            // so presentation and decode index are the same.
            pts_dts: (index, index),
        });
    }

    fn request_sync_frame(&self) {
        let mut params = MediaFormat::new();
        params.set_i32(key::REQUEST_SYNC_FRAME, 0);
        if let Err(error) = self.codec.set_parameters(params) {
            warn!(%error, "MediaCodec refused a sync-frame request");
        }
    }

    fn encode(
        &mut self,
        frame: CapturedFrame,
        scaler: &mut FrameScaler,
    ) -> CodecResult<EncodedFrame> {
        let timestamp_us = frame.timestamp_us();
        self.fill(frame, scaler)?;
        let pts = self.alloc_pts(Some(timestamp_us));
        let mut drained = Vec::new();
        let scratch = &self.scratch;
        self.codec.queue(
            Operation::Encode,
            pts,
            0,
            |buffer| codec::copy_into(buffer, scratch, Operation::Encode),
            |codec| {
                while let Some(output) = codec.next_output(Operation::Encode, Duration::ZERO)? {
                    drained.push(output);
                }
                Ok(())
            },
        )?;
        for output in drained {
            self.accept(output);
        }
        if self.finished.is_empty()
            && let Some(output) = self.codec.next_output(Operation::Encode, OUTPUT_TIMEOUT)?
        {
            self.accept(output);
        }
        while let Some(output) = self.codec.next_output(Operation::Encode, Duration::ZERO)? {
            self.accept(output);
        }
        self.finished.pop_front().ok_or_else(|| {
            fail(
                Operation::Encode,
                format!(
                    "no picture came out within {} ms",
                    OUTPUT_TIMEOUT.as_millis()
                ),
            )
        })
    }

    fn flush(mut self) -> CodecResult<Vec<EncodedFrame>> {
        let pts = self.next_pts;
        self.codec.queue(
            Operation::Flush,
            pts,
            FLAG_END_OF_STREAM,
            |_| Ok(0),
            |_| Ok(()),
        )?;
        while let Some(output) = self.codec.next_output(Operation::Flush, OUTPUT_TIMEOUT)? {
            let end = output.is(FLAG_END_OF_STREAM);
            self.accept(output);
            if end {
                break;
            }
        }
        if let Err(error) = self.codec.stop() {
            warn!(%error, "AMediaCodec_stop failed after the flush");
        }
        Ok(self.finished.into_iter().collect())
    }
}

impl VideoEncoder for MediaCodecVideoEncoder {
    // The facade reports the job's codec; a session serves the one its
    // configuration names.
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, config: &EncoderConfig) -> CodecResult<()> {
        if let Some(session) = &self.session
            && session.kind == config.codec
            && session.width == config.width
            && session.height == config.height
            && session.fps == config.fps.max(1)
            && session.colour == config.colour
            && session.depth == config.bit_depth
            && session.hdr == config.hdr
        {
            // Only the bitrate can change on a running session.
            return self.set_bitrate_kbps(config.bitrate_kbps);
        }
        // Stop the old session before opening the next: hardware
        // encoders have few instances.
        self.session = None;
        self.session = Some(Session::open(self.pick, config)?);
        self.pending_keyframe = false;
        Ok(())
    }

    fn encode(&mut self, frame: CapturedFrame) -> CodecResult<EncodedFrame> {
        if std::mem::take(&mut self.pending_keyframe) {
            self.session(Operation::Encode)?.request_sync_frame();
        }
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| fail(Operation::Encode, "the encoder is not configured"))?;
        session.encode(frame, &mut self.scaler)
    }

    fn flush(self) -> CodecResult<Vec<EncodedFrame>> {
        self.session.map_or_else(|| Ok(Vec::new()), Session::flush)
    }

    fn get_headers(&mut self) -> CodecResult<Vec<u8>> {
        Ok(self.session(Operation::Headers)?.headers.clone())
    }

    fn request_keyframe(&mut self) {
        self.pending_keyframe = true;
    }

    fn set_bitrate_kbps(&mut self, kbps: u32) -> CodecResult<()> {
        let session = self.session(Operation::SetBitrate)?;
        let bps = i32::try_from(kbps.saturating_mul(1000)).unwrap_or(i32::MAX);
        let mut params = MediaFormat::new();
        params.set_i32(key::VIDEO_BITRATE, bps);
        session
            .codec
            .set_parameters(params)
            .map_err(media(Operation::SetBitrate, "AMediaCodec_setParameters"))
    }
}
