//! H.264 / H.265 encoding through Media Foundation transforms.
//!
//! `encode` is one frame in, one packet out. Software MFTs and hardware
//! MFTs in low-latency mode (`CODECAPI_AVLowLatencyMode`) emit each
//! packet right after its frame. A hardware MFT that still asks for
//! more input before its first output buffers the frame: `encode` then
//! reports that as an error, the packet arrives with a later call and
//! every later call returns the oldest finished packet, so the stream
//! runs a constant number of frames behind instead of losing any.
//!
//! Parameter sets: `configure` encodes one black frame (discarded) so
//! they are known before the first real frame, which is then forced to
//! an IDR. Every keyframe carries them in-band (as x264 is set
//! up to do): when the MFT emits an IDR without them, the latest ones
//! (from `MF_MT_MPEG_SEQUENCE_HEADER` or an earlier keyframe) are put
//! in front.
//!
//! 10 bits: [`BitDepth::Ten`] encodes HEVC Main10 from P010 input (P010
//! frames are copied, 8-bit NV12 / I420 ones widened). The colour and
//! the static HDR metadata go on the media types (`MF_MT_VIDEO_PRIMARIES`,
//! `MF_MT_TRANSFER_FUNCTION`, `MF_MT_MAX_MASTERING_LUMINANCE`, ...);
//! whether they reach the bitstream (VUI, SEI) is up to the transform.
//!
//! Direct3D 11 input: a [`CapturedFrame::Native`] frame holding a
//! [`D3d11Texture`] of the encoder's size and input format is encoded
//! straight from the texture. The first one rebuilds the session on its
//! device (`MFT_MESSAGE_SET_D3D_MANAGER`, then a forced IDR); a
//! transform that is not Direct3D 11 aware, or a texture of another
//! size or format, is read back to system memory instead.

use std::collections::VecDeque;
use std::time::Duration;

use bytes::Bytes;
use ferricast_codec::bitstream::{self, NalSyntax, ParameterSets};
use ferricast_codec::{OpenFn, ProbeFn, VideoEncoderSession};
use ferricast_core::{
    BitDepth, CapturedFrame, Codec, CodecError, CodecResult, CodecTarget, ColourDescription,
    ColourRange, EncodedFrame, EncoderConfig, H264Profile, NativeFrame, NativeSurface, Operation,
    PixelFormat,
};
use windows::Win32::Media::MediaFoundation::{
    CODECAPI_AVEncCommonMeanBitRate, CODECAPI_AVEncCommonRateControlMode,
    CODECAPI_AVEncMPVDefaultBPictureCount, CODECAPI_AVEncMPVGOPSize,
    CODECAPI_AVEncVideoForceKeyFrame, CODECAPI_AVLowLatencyMode, ICodecAPI, IMFMediaType,
    IMFSample, MF_MT_AVG_BITRATE, MF_MT_DEFAULT_STRIDE, MF_MT_FRAME_RATE, MF_MT_FRAME_SIZE,
    MF_MT_INTERLACE_MODE, MF_MT_MAX_KEYFRAME_SPACING, MF_MT_MPEG_SEQUENCE_HEADER,
    MF_MT_MPEG2_PROFILE, MF_MT_PIXEL_ASPECT_RATIO, MF_MT_SUBTYPE, MFCreateSample,
    MFMediaType_Video, MFSampleExtension_CleanPoint, MFT_CATEGORY_VIDEO_ENCODER,
    MFT_INPUT_STREAM_INFO, MFVideoFormat_H264, MFVideoFormat_HEVC, MFVideoFormat_NV12,
    MFVideoFormat_P010, MFVideoInterlace_Progressive, eAVEncCommonRateControlMode_CBR,
    eAVEncH264VProfile_Base, eAVEncH264VProfile_High, eAVEncH264VProfile_Main,
    eAVEncH265VProfile_Main_420_8, eAVEncH265VProfile_Main_420_10,
};
use windows::core::GUID;

use super::mf::{
    self, BACKEND, D3d11Texture, Device, Kind, MfResult, Output, Search, Transform, Worker,
    colour_attributes, hns, media_type, micros, pack, set_codec_u32, set_hdr,
};
use crate::frame_view::FrameView;
use crate::nv12::{self, Nv12Layout, fill_u16};
use crate::scale::{FrameScaler, FrameSize};

/// How long `encode` lets a hardware MFT finish the frame it was given.
const OUTPUT_WAIT: Duration = Duration::from_millis(500);

/// The open and probe functions of one (codec, kind) candidate.
#[derive(Clone, Copy)]
pub(super) struct Opener {
    pub(super) open: OpenFn<EncoderConfig, dyn VideoEncoderSession>,
    pub(super) probe: ProbeFn,
}

pub(super) const H264_HARDWARE: Opener = Opener {
    open: open::<false, true>,
    probe: probe::<false, true>,
};
pub(super) const H264_SOFTWARE: Opener = Opener {
    open: open::<false, false>,
    probe: probe::<false, false>,
};
pub(super) const H265_HARDWARE: Opener = Opener {
    open: open::<true, true>,
    probe: probe::<true, true>,
};
pub(super) const H265_SOFTWARE: Opener = Opener {
    open: open::<true, false>,
    probe: probe::<true, false>,
};

/// The transforms for `hevc` / `hardware` taking `input` (NV12 for
/// 8-bit streams, P010 for 10-bit ones).
fn search(hevc: bool, hardware: bool, input: PixelFormat) -> Search {
    let (subtype, job) = match (hevc, input) {
        (true, PixelFormat::P010) => (MFVideoFormat_HEVC, "H.265 Main10 encoder"),
        (true, _) => (MFVideoFormat_HEVC, "H.265 encoder"),
        (false, _) => (MFVideoFormat_H264, "H.264 encoder"),
    };
    Search {
        category: MFT_CATEGORY_VIDEO_ENCODER,
        kind: if hardware {
            Kind::Hardware
        } else {
            Kind::Software
        },
        input: Some((MFMediaType_Video, input_subtype(input))),
        output: Some((MFMediaType_Video, subtype)),
        job,
    }
}

fn input_subtype(input: PixelFormat) -> GUID {
    if input == PixelFormat::P010 {
        MFVideoFormat_P010
    } else {
        MFVideoFormat_NV12
    }
}

fn probe<const HEVC: bool, const HARDWARE: bool>() -> CodecResult<()> {
    search(HEVC, HARDWARE, PixelFormat::Nv12).probe()
}

fn open<const HEVC: bool, const HARDWARE: bool>(
    config: &EncoderConfig,
) -> CodecResult<Box<dyn VideoEncoderSession>> {
    let codec = if HEVC { Codec::H265 } else { Codec::H264 };
    if config.codec != codec {
        return Err(CodecError::BackendCannotHandle {
            backend: BACKEND,
            target: CodecTarget::encode(config.codec),
        });
    }
    let input = match config.bit_depth {
        BitDepth::Eight => PixelFormat::Nv12,
        BitDepth::Ten if HEVC => PixelFormat::P010,
        BitDepth::Ten => {
            return Err(CodecError::UnsupportedConfig {
                backend: BACKEND,
                what: "10-bit H.264 (Media Foundation encodes 10 bits as HEVC Main10 only)".into(),
            });
        }
    };
    Ok(Box::new(MfVideoEncoder::open(
        search(HEVC, HARDWARE, input),
        config,
    )?))
}

/// A Media Foundation H.264 / H.265 encoder session.
pub(super) struct MfVideoEncoder {
    search: Search,
    worker: Worker<Session>,
}

impl MfVideoEncoder {
    fn open(search: Search, config: &EncoderConfig) -> CodecResult<Self> {
        let config = config.clone();
        let worker = Worker::spawn("ferricast-mf-video-encoder", move || {
            Session::open(search, &config, None)
        })?;
        Ok(Self { search, worker })
    }
}

impl VideoEncoderSession for MfVideoEncoder {
    fn configure(&mut self, config: &EncoderConfig) -> CodecResult<()> {
        // A new session is the only change every MFT accepts mid-stream.
        *self = Self::open(self.search, config)?;
        Ok(())
    }

    fn reset(&mut self, config: &EncoderConfig) -> CodecResult<()> {
        // `configure` already starts a new session.
        self.configure(config)
    }

    fn encode(&mut self, frame: CapturedFrame) -> CodecResult<EncodedFrame> {
        self.worker
            .run(Operation::Encode, move |session| session.encode(frame))
    }

    fn flush(self: Box<Self>) -> CodecResult<Vec<EncodedFrame>> {
        self.worker.run(Operation::Flush, Session::flush)
    }

    fn headers(&mut self) -> CodecResult<Vec<u8>> {
        self.worker.run(Operation::Headers, |session| {
            let sets = session.parameter_sets.annex_b();
            if sets.is_empty() {
                return Err(CodecError::backend(
                    BACKEND,
                    Operation::Headers,
                    "the encoder has not produced its parameter sets yet (they come with the \
                     first keyframe)",
                ));
            }
            Ok(sets)
        })
    }

    fn request_keyframe(&mut self) {
        let queued = self.worker.run(Operation::Encode, |session| {
            session.keyframe_requested = true;
            Ok(())
        });
        if let Err(error) = queued {
            tracing::warn!(%error, "keyframe request lost");
        }
    }

    fn set_bitrate_kbps(&mut self, kbps: u32) -> CodecResult<()> {
        self.worker.run(Operation::SetBitrate, move |session| {
            session.config.bitrate_kbps = kbps;
            let Some(api) = &session.codec_api else {
                tracing::debug!(transform = session.transform.name(), "no ICodecAPI: bitrate stays");
                return Ok(());
            };
            if let Err(error) = set_codec_u32(api, &CODECAPI_AVEncCommonMeanBitRate, kbps.saturating_mul(1000)) {
                // Best effort, as the trait asks: the stream goes on at
                // the old rate.
                tracing::warn!(transform = session.transform.name(), %error, "live bitrate change refused");
            }
            Ok(())
        })
    }
}

/// Whether the session encodes from a Direct3D 11 device.
enum Gpu {
    /// System-memory input only (no texture seen yet).
    None,
    /// Bound to this device: its textures go in as they are.
    Bound(Device),
    /// Rebinding to this device failed or its transform is not Direct3D
    /// aware: its textures are read back, without retrying.
    Refused(Device),
}

/// The encoder state on its worker thread.
struct Session {
    search: Search,
    config: EncoderConfig,
    transform: Transform,
    codec_api: Option<ICodecAPI>,
    codec: Codec,
    /// `Nv12`, or `P010` for a 10-bit stream.
    input: PixelFormat,
    size: FrameSize,
    fps: u32,
    colour: ColourDescription,
    layout: Nv12Layout,
    alignment: u32,
    gpu: Gpu,
    scaler: FrameScaler,
    keyframe_requested: bool,
    parameter_sets: ParameterSets,
    ready: VecDeque<EncodedFrame>,
    packets: u64,
    /// `(sample time set, capture timestamp)` of the frames fed and not
    /// yet out, oldest first: an MFT that rewrites or drops sample
    /// times still gets each packet the capture time of its frame.
    in_flight: VecDeque<(i64, u64)>,
}

impl Session {
    fn open(search: Search, config: &EncoderConfig, device: Option<Device>) -> CodecResult<Self> {
        if config.width == 0
            || config.height == 0
            || !config.width.is_multiple_of(2)
            || !config.height.is_multiple_of(2)
        {
            return Err(CodecError::UnsupportedConfig {
                backend: BACKEND,
                what: format!(
                    "a {}x{} picture (4:2:0 input needs even, non-zero dimensions)",
                    config.width, config.height
                ),
            });
        }
        let colour = colour_attributes(config.colour)?;
        let mut device = device;
        search.open(|transform| Self::setup(search, transform, config, &colour, &mut device))
    }

    /// Sets `transform` up; a `device` it takes moves into the session.
    #[expect(
        clippy::too_many_lines,
        reason = "one linear Media Foundation media-type negotiation"
    )]
    fn setup(
        search: Search,
        mut transform: Transform,
        config: &EncoderConfig,
        colour: &[(GUID, Option<u32>)],
        device: &mut Option<Device>,
    ) -> CodecResult<Self> {
        let op = Operation::Open;
        let hevc = config.codec == Codec::H265;
        let input_format = if config.bit_depth == BitDepth::Ten {
            PixelFormat::P010
        } else {
            PixelFormat::Nv12
        };
        let bytes_per_sample = if input_format == PixelFormat::P010 {
            2
        } else {
            1
        };
        let fps = config.fps.max(1);
        let gop = config.keyframe_interval_frames();
        let bitrate = config.bitrate_kbps.saturating_mul(1000);
        // The device manager goes in before any media type.
        let bound = match device.as_ref() {
            Some(gpu) => transform.attach_d3d11(gpu)?,
            None => false,
        };
        let codec_api = transform.codec_api();
        // Rate control has to be chosen before the media types are set.
        if let Some(api) = &codec_api {
            let settings = [
                (
                    CODECAPI_AVEncCommonRateControlMode,
                    mf::flag(eAVEncCommonRateControlMode_CBR.0),
                    "CBR rate control",
                ),
                (CODECAPI_AVEncCommonMeanBitRate, bitrate, "the mean bitrate"),
                (CODECAPI_AVLowLatencyMode, 1, "low-latency mode"),
                (CODECAPI_AVEncMPVDefaultBPictureCount, 0, "no B-frames"),
                (CODECAPI_AVEncMPVGOPSize, gop, "the GOP size"),
            ];
            for (key, value, what) in settings {
                if let Err(error) = set_codec_u32(api, &key, value) {
                    tracing::debug!(transform = transform.name(), %error, "{what} not accepted");
                }
            }
        }
        let subtype = if hevc {
            MFVideoFormat_HEVC
        } else {
            MFVideoFormat_H264
        };
        let profile = match (hevc, input_format) {
            (true, PixelFormat::P010) => mf::flag(eAVEncH265VProfile_Main_420_10.0),
            (true, _) => mf::flag(eAVEncH265VProfile_Main_420_8.0),
            (false, _) => match config.max_h264_profile {
                Some(H264Profile::Baseline) => mf::flag(eAVEncH264VProfile_Base.0),
                Some(H264Profile::Main) => mf::flag(eAVEncH264VProfile_Main.0),
                Some(H264Profile::High) | None => mf::flag(eAVEncH264VProfile_High.0),
            },
        };
        let output = media_type(&MFMediaType_Video, &subtype, op)?;
        let picture = |media: &IMFMediaType| -> CodecResult<()> {
            // SAFETY: plain attribute writes on a live media type.
            unsafe {
                media
                    .SetUINT64(&MF_MT_FRAME_SIZE, pack(config.width, config.height))
                    .mf(op, "setting MF_MT_FRAME_SIZE")?;
                media
                    .SetUINT64(&MF_MT_FRAME_RATE, pack(fps, 1))
                    .mf(op, "setting MF_MT_FRAME_RATE")?;
                media
                    .SetUINT64(&MF_MT_PIXEL_ASPECT_RATIO, pack(1, 1))
                    .mf(op, "setting MF_MT_PIXEL_ASPECT_RATIO")?;
                media
                    .SetUINT32(
                        &MF_MT_INTERLACE_MODE,
                        mf::flag(MFVideoInterlace_Progressive.0),
                    )
                    .mf(op, "setting MF_MT_INTERLACE_MODE")?;
                for (key, value) in colour {
                    if let Some(value) = value {
                        media
                            .SetUINT32(key, *value)
                            .mf(op, "setting a colour attribute")?;
                    }
                }
            }
            if let Some(hdr) = config.hdr {
                set_hdr(media, hdr, op)?;
            }
            Ok(())
        };
        picture(&output)?;
        // SAFETY: plain COM calls on the live media type and transform.
        unsafe {
            output
                .SetUINT32(&MF_MT_AVG_BITRATE, bitrate)
                .mf(op, "setting MF_MT_AVG_BITRATE")?;
            output
                .SetUINT32(&MF_MT_MPEG2_PROFILE, profile)
                .mf(op, "setting MF_MT_MPEG2_PROFILE")?;
            output
                .SetUINT32(&MF_MT_MAX_KEYFRAME_SPACING, gop)
                .mf(op, "setting MF_MT_MAX_KEYFRAME_SPACING")?;
            transform
                .mft()
                .SetOutputType(0, &output, 0)
                .mf(op, "IMFTransform::SetOutputType")?;
        }
        let input_subtype = input_subtype(input_format);
        let input = media_type(&MFMediaType_Video, &input_subtype, op)?;
        picture(&input)?;
        // SAFETY: plain COM call on the live transform.
        if let Err(error) = unsafe { transform.mft().SetInputType(0, &input, 0) } {
            // Some MFTs only take a type they offer themselves.
            let offered = Self::offered_input(&transform, input_subtype, &picture)
                .ok_or_else(|| mf::failure(op, "IMFTransform::SetInputType", &error))?;
            // SAFETY: as above.
            unsafe { transform.mft().SetInputType(0, &offered, 0) }
                .mf(op, "IMFTransform::SetInputType")?;
        }
        let row_bytes = config.width as usize * bytes_per_sample;
        // SAFETY: plain COM calls on the live transform.
        let (stride, alignment) = unsafe {
            let current = transform
                .mft()
                .GetInputCurrentType(0)
                .mf(op, "IMFTransform::GetInputCurrentType")?;
            let stride = current
                .GetUINT32(&MF_MT_DEFAULT_STRIDE)
                .ok()
                .and_then(|stride| usize::try_from(stride).ok())
                .filter(|&stride| stride >= row_bytes)
                .unwrap_or(row_bytes);
            let mut info = MFT_INPUT_STREAM_INFO::default();
            transform
                .mft()
                .GetInputStreamInfo(0, &raw mut info)
                .mf(op, "IMFTransform::GetInputStreamInfo")?;
            (stride, info.cbAlignment.max(16))
        };
        transform.request_low_latency();
        transform
            .set_default_output_size(config.width.saturating_mul(config.height).saturating_mul(2));
        transform.start()?;
        let mut parameter_sets = ParameterSets::new(if hevc {
            NalSyntax::H265
        } else {
            NalSyntax::H264
        });
        // SAFETY: plain COM calls on the live transform.
        if let Ok(current) = unsafe { transform.mft().GetOutputCurrentType(0) }
            && let Some(header) = blob(&current, &MF_MT_MPEG_SEQUENCE_HEADER)
        {
            parameter_sets.remember(&annex_b(header));
        }
        let layout = Nv12Layout {
            width: config.width,
            height: config.height,
            stride,
            chroma_offset: stride * config.height as usize,
        };
        let gpu = match device.take() {
            Some(gpu) if bound => Gpu::Bound(gpu),
            Some(gpu) => Gpu::Refused(gpu),
            None => Gpu::None,
        };
        let mut session = Self {
            search,
            config: config.clone(),
            transform,
            codec_api,
            codec: config.codec,
            input: input_format,
            size: FrameSize::new(config.width, config.height),
            fps,
            colour: config.colour,
            layout,
            alignment,
            gpu,
            scaler: FrameScaler::default(),
            keyframe_requested: false,
            parameter_sets,
            ready: VecDeque::new(),
            packets: 0,
            in_flight: VecDeque::new(),
        };
        session.prime()?;
        Ok(session)
    }

    /// Encodes one black frame and throws its packet away, so the
    /// parameter sets are known before the first real frame (the HLS
    /// segmenter asks for them up front, and `MF_MT_MPEG_SEQUENCE_HEADER`
    /// is usually empty); the next frame is forced to an IDR.
    fn prime(&mut self) -> CodecResult<()> {
        if !self.parameter_sets.annex_b().is_empty() {
            return Ok(());
        }
        let op = Operation::Open;
        let layout = self.layout;
        let limited = self.colour.range == ColourRange::Limited;
        let input = self.input;
        let sample = mf::sample_with(layout.len(), self.alignment, op, |dst| {
            let (luma, chroma) = dst.split_at_mut(layout.chroma_offset);
            if input == PixelFormat::P010 {
                // 10-bit codes in the high bits: black 64 (limited) or 0,
                // neutral chroma 512.
                let black: u16 = if limited { 64 << 6 } else { 0 };
                fill_u16(luma, black);
                fill_u16(chroma, 512 << 6);
            } else {
                luma.fill(if limited { 16 } else { 0 });
                chroma.fill(128);
            }
            Ok(())
        })?;
        // SAFETY: plain attribute writes on the new sample.
        unsafe {
            sample.SetSampleTime(0).mf(op, "IMFSample::SetSampleTime")?;
            sample
                .SetSampleDuration(hns(1_000_000 / u64::from(self.fps)))
                .mf(op, "IMFSample::SetSampleDuration")?;
        }
        let mut outputs = Vec::new();
        self.transform
            .push(&sample, &mut outputs)
            .mf(op, "IMFTransform::ProcessInput (priming)")?;
        self.transform
            .pull(&mut outputs, Some(OUTPUT_WAIT))
            .mf(op, "IMFTransform::ProcessOutput (priming)")?;
        if outputs.is_empty() {
            // A pipelining MFT holds the frame: drain it out.
            self.transform
                .drain(&mut outputs)
                .mf(op, "draining the priming frame")?;
        }
        for output in outputs {
            if let Output::Sample(sample) = output {
                self.parameter_sets
                    .remember(&annex_b(mf::sample_bytes(&sample, op)?));
            }
        }
        if self.parameter_sets.annex_b().is_empty() {
            tracing::warn!(
                transform = self.transform.name(),
                "the priming frame produced no parameter sets; headers come with the first keyframe"
            );
        }
        self.keyframe_requested = true;
        Ok(())
    }

    /// An input type of `subtype` the MFT offers, completed with the
    /// picture.
    fn offered_input(
        transform: &Transform,
        subtype: GUID,
        picture: &dyn Fn(&IMFMediaType) -> CodecResult<()>,
    ) -> Option<IMFMediaType> {
        (0..64).find_map(|index| {
            // SAFETY: plain COM calls on the live transform.
            let offered = unsafe { transform.mft().GetInputAvailableType(0, index) }.ok()?;
            // SAFETY: plain attribute read.
            let offered_subtype = unsafe { offered.GetGUID(&MF_MT_SUBTYPE) }.ok()?;
            (offered_subtype == subtype && picture(&offered).is_ok()).then_some(offered)
        })
    }

    /// Whether `texture` can go in as it is: the encoder's input format
    /// and size, the whole texture visible.
    fn takes_texture(&self, texture: &D3d11Texture) -> bool {
        let visible = texture.visible();
        NativeSurface::format(texture) == self.input
            && texture.size() == (self.size.width, self.size.height)
            && (visible.x, visible.y, visible.width, visible.height)
                == (0, 0, self.size.width, self.size.height)
    }

    /// A sample around `frame`'s texture, rebinding the session to its
    /// device first when needed; `None` when it has to be read back.
    fn texture_sample(&mut self, frame: &NativeFrame) -> CodecResult<Option<IMFSample>> {
        let Some(texture) = frame.surface::<D3d11Texture>() else {
            return Ok(None);
        };
        if !self.takes_texture(texture) {
            return Ok(None);
        }
        let on_device = match &self.gpu {
            Gpu::Bound(device) => device.holds(texture),
            Gpu::Refused(device) if device.holds(texture) => return Ok(None),
            Gpu::Refused(_) | Gpu::None => false,
        };
        if !on_device && !self.rebind(texture)? {
            return Ok(None);
        }
        let op = Operation::Encode;
        let buffer = texture.media_buffer()?;
        // SAFETY: plain MF calls on a new sample.
        let sample = unsafe {
            let sample = MFCreateSample().mf(op, "MFCreateSample")?;
            sample.AddBuffer(&buffer).mf(op, "IMFSample::AddBuffer")?;
            sample
        };
        Ok(Some(sample))
    }

    /// Rebuilds the session on `texture`'s device, keeping the packets
    /// already finished; `false` (textures of that device are then read
    /// back) when the device or its transform cannot take them.
    fn rebind(&mut self, texture: &D3d11Texture) -> CodecResult<bool> {
        let device = Device::of_texture(texture)?;
        let finished = self.flush()?;
        let rebuilt = Self::open(self.search, &self.config, Some(device));
        match rebuilt {
            Ok(mut session) if matches!(session.gpu, Gpu::Bound(_)) => {
                tracing::info!(
                    transform = session.transform.name(),
                    "encoding from Direct3D 11 textures"
                );
                session.packets = self.packets;
                session.ready = finished.into();
                session.ready.extend(self.ready.drain(..));
                *self = session;
                Ok(true)
            }
            Ok(session) => {
                tracing::info!(
                    transform = session.transform.name(),
                    "the transform is not Direct3D 11 aware; textures are read back"
                );
                self.refuse(session.gpu, finished);
                Ok(false)
            }
            Err(error) => {
                tracing::warn!(%error, "rebinding to the texture's device failed; textures are read back");
                self.refuse(Gpu::Refused(Device::of_texture(texture)?), finished);
                Ok(false)
            }
        }
    }

    /// Keeps this session after a refused rebind, never retrying that
    /// device.
    fn refuse(&mut self, gpu: Gpu, finished: Vec<EncodedFrame>) {
        self.gpu = match gpu {
            Gpu::Bound(device) | Gpu::Refused(device) => Gpu::Refused(device),
            Gpu::None => Gpu::None,
        };
        for packet in finished.into_iter().rev() {
            self.ready.push_front(packet);
        }
    }

    /// A system-memory sample holding `frame` in the input layout.
    fn memory_sample(&mut self, frame: CapturedFrame) -> CodecResult<IMFSample> {
        let op = Operation::Encode;
        let raw = frame
            .into_cpu()
            .map_err(|error| CodecError::backend(BACKEND, op, error))?;
        let raw = self
            .scaler
            .fit(raw, self.size)
            .map_err(CodecError::from_backend(BACKEND, op))?;
        let view = FrameView::new(&raw, self.size.width, self.size.height)
            .map_err(CodecError::from_backend(BACKEND, op))?;
        let (colour, layout, input) = (self.colour, self.layout, self.input);
        mf::sample_with(layout.len(), self.alignment, op, |dst| {
            if input == PixelFormat::P010 {
                nv12::write_p010(&view, colour, layout, dst, BACKEND)
            } else {
                nv12::write(&view, colour, layout, dst, BACKEND)
            }
        })
    }

    fn encode(&mut self, frame: CapturedFrame) -> CodecResult<EncodedFrame> {
        let op = Operation::Encode;
        let timestamp_us = frame.timestamp_us();
        let texture = match &frame {
            CapturedFrame::Native(native) => self.texture_sample(native)?,
            CapturedFrame::Cpu(_) => None,
        };
        let sample = match texture {
            Some(sample) => sample,
            None => self.memory_sample(frame)?,
        };
        if self.keyframe_requested {
            self.keyframe_requested = false;
            if let Some(api) = &self.codec_api {
                if let Err(error) = set_codec_u32(api, &CODECAPI_AVEncVideoForceKeyFrame, 1) {
                    tracing::warn!(transform = self.transform.name(), %error, "keyframe request refused");
                }
            } else {
                tracing::warn!(
                    transform = self.transform.name(),
                    "no ICodecAPI: keyframe request dropped"
                );
            }
        }
        let duration_us = 1_000_000 / u64::from(self.fps);
        // SAFETY: plain attribute writes on the new sample.
        unsafe {
            sample
                .SetSampleTime(hns(timestamp_us))
                .mf(op, "IMFSample::SetSampleTime")?;
            self.in_flight.push_back((hns(timestamp_us), timestamp_us));
            sample
                .SetSampleDuration(hns(duration_us))
                .mf(op, "IMFSample::SetSampleDuration")?;
        }
        let mut outputs = Vec::new();
        self.transform
            .push(&sample, &mut outputs)
            .mf(op, "IMFTransform::ProcessInput")?;
        self.transform
            .pull(&mut outputs, Some(OUTPUT_WAIT))
            .mf(op, "IMFTransform::ProcessOutput")?;
        self.collect(outputs)?;
        self.ready.pop_front().ok_or_else(|| {
            CodecError::backend(
                BACKEND,
                op,
                format!(
                    "{} buffered the frame without output yet; its packet comes with a later frame",
                    self.transform.name()
                ),
            )
        })
    }

    fn flush(&mut self) -> CodecResult<Vec<EncodedFrame>> {
        let mut outputs = Vec::new();
        self.transform
            .drain(&mut outputs)
            .mf(Operation::Flush, "draining the encoder")?;
        self.collect(outputs)?;
        Ok(self.ready.drain(..).collect())
    }

    /// The capture timestamp of the frame a packet with sample time
    /// `time` encodes: the submitted frame with that sample time, else
    /// (an MFT that dropped or rewrote it) the oldest frame in flight,
    /// as frames leave in input order without B-frames.
    fn capture_time(&mut self, time: Option<i64>) -> u64 {
        if let Some(time) = time
            && let Some(at) = self.in_flight.iter().position(|&(fed, _)| fed == time)
        {
            let (_, capture) = self.in_flight[at];
            self.in_flight.drain(..=at);
            return capture;
        }
        if let Some((_, capture)) = self.in_flight.pop_front() {
            tracing::debug!(
                ?time,
                capture,
                "packet sample time matches no frame; using the oldest in flight"
            );
            capture
        } else {
            let fallback = time.map_or(0, micros);
            tracing::warn!(?time, fallback, "packet with no frame in flight");
            fallback
        }
    }

    /// Turns MFT output into packets on the ready queue.
    fn collect(&mut self, outputs: Vec<Output>) -> CodecResult<()> {
        let op = Operation::Encode;
        for output in outputs {
            let sample = match output {
                Output::Sample(sample) => sample,
                Output::FormatChanged => {
                    // SAFETY: plain COM call on the live transform.
                    if let Ok(current) = unsafe { self.transform.mft().GetOutputCurrentType(0) }
                        && let Some(header) = blob(&current, &MF_MT_MPEG_SEQUENCE_HEADER)
                    {
                        self.parameter_sets.remember(&annex_b(header));
                    }
                    continue;
                }
            };
            let data = annex_b(mf::sample_bytes(&sample, op)?);
            if data.is_empty() {
                // Some MFTs hand back an empty sample (e.g. while
                // reconfiguring): no packet, and no frame consumed.
                continue;
            }
            // SAFETY: plain attribute reads on the output sample.
            let (time, keyframe) = unsafe {
                (
                    sample.GetSampleTime().ok(),
                    sample.GetUINT32(&MFSampleExtension_CleanPoint).unwrap_or(0) != 0,
                )
            };
            let syntax = if self.codec == Codec::H265 {
                NalSyntax::H265
            } else {
                NalSyntax::H264
            };
            let carries_sets = has_parameter_sets(&data, syntax);
            self.parameter_sets.remember(&data);
            let data = Bytes::from(data);
            let data = if keyframe && !carries_sets {
                self.parameter_sets.prepend(&data)
            } else {
                data
            };
            let index = self.packets;
            self.packets += 1;
            let timestamp_us = self.capture_time(time);
            self.ready.push_back(EncodedFrame {
                codec: self.codec,
                data,
                timestamp_us,
                is_keyframe: keyframe,
                duration_us: Some(1_000_000 / u64::from(self.fps)),
                // No B-frames: decode order is presentation order.
                pts_dts: (index, index),
            });
        }
        Ok(())
    }
}

/// `data` as Annex B: MFTs emit Annex B byte streams, but a sequence
/// header may come length-prefixed (AVCC / HVCC style), so 4-byte
/// length prefixes are rewritten when no start code leads.
fn annex_b(data: Vec<u8>) -> Vec<u8> {
    if data.starts_with(&[0, 0, 1]) || data.starts_with(&[0, 0, 0, 1]) {
        return data;
    }
    bitstream::length_prefixed_to_annex_b(&data, 4).unwrap_or(data)
}

/// Whether `data` carries an SPS.
fn has_parameter_sets(data: &[u8], syntax: NalSyntax) -> bool {
    bitstream::split_annex_b(data)
        .iter()
        .any(|nal| match (syntax, nal.first()) {
            (NalSyntax::H264, Some(header)) => header & 0x1f == 7,
            (NalSyntax::H265, Some(header)) => (header >> 1) & 0x3f == 33,
            (_, None) => false,
        })
}

/// A blob attribute, when present.
fn blob(media: &IMFMediaType, key: &GUID) -> Option<Vec<u8>> {
    // SAFETY: plain attribute reads; the buffer is sized from the blob.
    unsafe {
        let len = media.GetBlobSize(key).ok()?;
        let mut data = vec![0; len as usize];
        media.GetBlob(key, &mut data, None).ok()?;
        Some(data)
    }
}
