//! NVENC hardware encode (H.264, HEVC) through `ferricast-nvcodec`.
//!
//! One generic [`NvencEncoder`] serves both codecs; a sealed
//! [`NvencCodec`] marker fixes the codec and its profile mapping.
//!
//! The session is the low-latency one ferricast streams with: preset P3,
//! low-latency tuning, no B-frames, CBR with the peak locked to the
//! average, an IDR every keyframe interval (and on request) with the
//! parameter sets in front, the configured colour in the VUI and, for
//! HEVC, Main10 from P010 input with HDR10 metadata.
//!
//! Inputs, each without a copy through the CPU when it can:
//!
//! - [`CapturedFrame::Cpu`]: copied on the GPU's copy engine into a pooled
//!   device buffer (BGRA / RGBA, NV12, P010; P010 into an 8-bit session is
//!   narrowed first).
//! - [`CapturedFrame::Native`] holding a [`CudaSurface`] (NVDEC output)
//!   of the session's size and format: encoded in place.
//! - [`CapturedFrame::Gpu`] (Linux): the DMA-BUF is imported into CUDA
//!   through EGL, once per capture buffer, and encoded from the CUDA array
//!   or pitched memory it aliases (tiled and LINEAR buffers). The
//!   frame, and with it the capture buffer's lease, is held until NVENC has
//!   finished reading it. Any failure on this path sends that frame through
//!   the CPU readback instead; the next frame tries again.

use std::marker::PhantomData;
use std::num::NonZeroU32;
use std::sync::{Mutex, PoisonError};

use bytes::Bytes;
use ferricast_codec::cuda::nvcodec::cuda::HostImage;
use ferricast_codec::cuda::nvcodec::encode::{
    self as nv, Capabilities, ColorDescription, ColorPrimaries, Encoder, FrameOptions, FrameRate,
    Gop, Hdr10, HostFrame, InputFormat, MatrixCoefficients, PictureType, Profile, RateControl,
    TransferCharacteristics,
};
use ferricast_codec::cuda::{self, CudaSurface, map_error};
use ferricast_codec::{Candidate, VideoEncoderRegistry, VideoEncoderSession};
use ferricast_core::{
    Acceleration, BackendId, BitDepth, CapturedFrame, Codec, CodecError, CodecResult,
    ColourDescription, ColourRange, EncodedFrame, EncoderConfig, H264Profile, HdrMetadata,
    MatrixCoefficients as Matrix, Operation, PixelFormat, RawFrame, VideoEncoder,
};
use tracing::info;

use crate::config_ext::EncoderConfigExt;
use crate::frame_view::{self, FrameView};
use crate::scale::{FrameScaler, FrameSize};

/// The lowest bitrate a session is opened or retargeted with.
const MIN_BITRATE: u32 = 500_000;

/// Sealed marker pinning a [`NvencEncoder`] to one codec.
pub trait NvencCodec: sealed::Sealed + Send + 'static {
    /// The ferricast codec ([`VideoEncoder::CODEC`]).
    const FERRICAST_CODEC: Codec;
    /// The NVENC codec.
    const NV_CODEC: nv::Codec;

    /// The profile `config` asks for, within its ceiling.
    ///
    /// # Errors
    ///
    /// [`CodecError::UnsupportedConfig`] for a bit depth the codec has no
    /// profile for here.
    fn profile(config: &EncoderConfig) -> CodecResult<Profile>;
}

mod sealed {
    pub trait Sealed {}
}

/// H.264.
#[derive(Debug)]
pub enum H264 {}
impl sealed::Sealed for H264 {}
impl NvencCodec for H264 {
    const FERRICAST_CODEC: Codec = Codec::H264;
    const NV_CODEC: nv::Codec = nv::Codec::H264;

    fn profile(config: &EncoderConfig) -> CodecResult<Profile> {
        if config.bit_depth != BitDepth::Eight {
            return Err(CodecError::UnsupportedConfig {
                backend: BackendId::Nvenc,
                what: format!("{}-bit H.264", config.bit_depth.bits()),
            });
        }
        // Main unless the receiver allows more: the floor every receiver
        // protocol (Chromecast 1st gen and up, Miracast, AirPlay 2)
        // decodes. The manager raises the ceiling from the device's
        // capabilities.
        Ok(match config.max_h264_profile {
            Some(H264Profile::High) => Profile::H264High,
            Some(H264Profile::Baseline) => Profile::H264Baseline,
            Some(H264Profile::Main) | None => Profile::H264Main,
        })
    }
}

/// HEVC.
#[cfg(feature = "nvenc-hevc")]
#[derive(Debug)]
pub enum Hevc {}
#[cfg(feature = "nvenc-hevc")]
impl sealed::Sealed for Hevc {}
#[cfg(feature = "nvenc-hevc")]
impl NvencCodec for Hevc {
    const FERRICAST_CODEC: Codec = Codec::H265;
    const NV_CODEC: nv::Codec = nv::Codec::Hevc;

    fn profile(config: &EncoderConfig) -> CodecResult<Profile> {
        // The bit depth decides the profile: Main10 for 10-bit, Main
        // (decoded everywhere) otherwise.
        Ok(match config.bit_depth {
            BitDepth::Ten => Profile::HevcMain10,
            BitDepth::Eight => Profile::HevcMain,
        })
    }
}

/// The H.264 NVENC encoder.
pub type NvencH264Encoder = NvencEncoder<H264>;
/// The HEVC NVENC encoder.
#[cfg(feature = "nvenc-hevc")]
pub type NvencH265Encoder = NvencEncoder<Hevc>;

/// Adds NVENC for `codec` to `registry` (HEVC needs `nvenc-hevc`).
pub(crate) fn register(registry: VideoEncoderRegistry, codec: Codec) -> VideoEncoderRegistry {
    match codec {
        Codec::H264 => registry.candidate(candidate::<H264>()),
        #[cfg(feature = "nvenc-hevc")]
        Codec::H265 => registry.candidate(candidate::<Hevc>()),
        #[cfg(not(feature = "nvenc-hevc"))]
        Codec::H265 => registry.missing(BackendId::Nvenc, "ferricast-encoder/nvenc-hevc"),
        Codec::Vp8 | Codec::Vp9 | Codec::Av1 => registry,
    }
}

fn candidate<C: NvencCodec>() -> Candidate<EncoderConfig, dyn VideoEncoderSession> {
    Candidate::new(
        BackendId::Nvenc,
        Acceleration::Hardware,
        |config: &EncoderConfig| {
            NvencEncoder::<C>::probe_with(config.clone())
                .map(|encoder| Box::new(encoder) as Box<dyn VideoEncoderSession>)
        },
    )
    .with_probe(NvencEncoder::<C>::probe)
}

/// How the session takes pictures, from the configured pixel format and
/// bit depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Input {
    /// BGRA (NVENC's ARGB) or RGBA (ABGR), converted to YUV by NVENC.
    Rgb(PixelFormat),
    /// NV12; P010 frames are narrowed to their top eight bits.
    Nv12,
    /// P010 into a 10-bit session.
    P010,
}

impl Input {
    fn of(config: &EncoderConfig) -> CodecResult<Self> {
        let unsupported = |what: String| CodecError::UnsupportedConfig {
            backend: BackendId::Nvenc,
            what,
        };
        match (config.pixel_format, config.bit_depth) {
            (format @ (PixelFormat::Bgra | PixelFormat::Rgba), BitDepth::Eight) => {
                Ok(Self::Rgb(format))
            }
            (PixelFormat::Nv12 | PixelFormat::P010, BitDepth::Eight) => Ok(Self::Nv12),
            (PixelFormat::P010, BitDepth::Ten) => Ok(Self::P010),
            (
                format @ (PixelFormat::Bgra | PixelFormat::Rgba | PixelFormat::Nv12),
                BitDepth::Ten,
            ) => Err(unsupported(format!(
                "10-bit encoding from {format:?} input (needs P010)"
            ))),
            (PixelFormat::I420, _) => Err(unsupported("I420 input".into())),
        }
    }

    fn format(self) -> InputFormat {
        match self {
            Self::Rgb(PixelFormat::Rgba) => InputFormat::Abgr,
            Self::Rgb(_) => InputFormat::Argb,
            Self::Nv12 => InputFormat::Nv12,
            Self::P010 => InputFormat::P010,
        }
    }

    /// The pixel format of the frames the session encodes as they are.
    fn pixel_format(self) -> PixelFormat {
        match self {
            Self::Rgb(format) => format,
            Self::Nv12 => PixelFormat::Nv12,
            Self::P010 => PixelFormat::P010,
        }
    }
}

/// The colour the session writes and, for RGB input, converts with.
fn colour(config: &EncoderConfig, input: Input) -> CodecResult<ColorDescription> {
    let colour: ColourDescription = config.colour;
    // NVENC converts RGB with the matrix the VUI names, at limited range;
    // what else is configured cannot be produced from RGB input.
    if matches!(input, Input::Rgb(_))
        && (colour.range == ColourRange::Full
            || !matches!(
                colour.matrix,
                Matrix::Bt709 | Matrix::Bt470Bg | Matrix::Smpte170M
            ))
    {
        return Err(CodecError::UnsupportedConfig {
            backend: BackendId::Nvenc,
            what: format!("{colour} from RGB input (BT.709 / BT.601, limited range only)"),
        });
    }
    Ok(ColorDescription {
        primaries: ColorPrimaries(colour.primaries.code()),
        transfer: TransferCharacteristics(colour.transfer.code()),
        matrix: MatrixCoefficients(colour.matrix.code()),
        full_range: colour.range == ColourRange::Full,
    })
}

/// ferricast's static HDR metadata as NVENC's.
fn hdr10(hdr: HdrMetadata) -> Hdr10 {
    let point = |point: ferricast_core::Chromaticity| nv::Chromaticity {
        x: point.x,
        y: point.y,
    };
    Hdr10 {
        mastering_display: hdr.mastering_display.map(|display| nv::MasteringDisplay {
            red: point(display.red),
            green: point(display.green),
            blue: point(display.blue),
            white_point: point(display.white_point),
            max_luminance: display.max_luminance,
            min_luminance: display.min_luminance,
        }),
        content_light: hdr.content_light.map(|light| nv::ContentLightLevel {
            max_content: light.max_cll,
            max_frame_average: light.max_fall,
        }),
    }
}

/// What NVENC can do on this machine's GPU, queried once (it opens a
/// session); a failed query is not kept, so a later probe retries.
fn capabilities() -> CodecResult<Capabilities> {
    static CAPABILITIES: Mutex<Option<Capabilities>> = Mutex::new(None);
    let mut cached = CAPABILITIES.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(capabilities) = cached.as_ref() {
        return Ok(capabilities.clone());
    }
    let probe = map_error(BackendId::Nvenc, Operation::Probe);
    let context = cuda::context().map_err(probe)?;
    let capabilities =
        Capabilities::query(&context).map_err(map_error(BackendId::Nvenc, Operation::Probe))?;
    *cached = Some(capabilities.clone());
    Ok(capabilities)
}

/// NVENC encoder for the codec `C`.
pub struct NvencEncoder<C: NvencCodec> {
    // Field order is drop order: the session (and the CUDA array
    // registrations it holds) goes before the DMA-BUF importer that owns
    // the arrays.
    encoder: Encoder,
    #[cfg(target_os = "linux")]
    zero_copy: zero_copy::State,
    input: Input,
    /// The configuration the session was opened for (bitrate kept in
    /// sync with live retargets).
    requested: EncoderConfig,
    /// Pictures submitted; an IDR every `keyframe_interval` of them.
    frame_count: u64,
    keyframe_interval: u64,
    /// Set by [`VideoEncoder::request_keyframe`] (and by a failed encode),
    /// consumed by the next picture NVENC encodes.
    pending_keyframe: bool,
    /// Fits frames whose size differs from the session into it.
    scaler: FrameScaler,
    _codec: PhantomData<fn() -> C>,
}

impl<C: NvencCodec> NvencEncoder<C> {
    /// Whether this machine's GPU encodes `C` through NVENC: loads the
    /// driver and queries the GPU once per process.
    ///
    /// # Errors
    ///
    /// [`CodecError::BackendUnavailable`] without an NVIDIA driver or GPU,
    /// or when the GPU's NVENC has no encoder for the codec.
    pub fn probe() -> CodecResult<()> {
        capabilities()?
            .require(C::NV_CODEC)
            .map(drop)
            .map_err(|error| CodecError::BackendUnavailable {
                backend: BackendId::Nvenc,
                reason: error.to_string(),
            })
    }

    /// Opens an NVENC session for `config`.
    ///
    /// # Errors
    ///
    /// [`CodecError::UnsupportedConfig`] for what NVENC cannot produce here
    /// (I420 input, 10-bit H.264, 10-bit without P010 input, colour RGB
    /// input cannot be converted to, 10-bit or a size the GPU lacks);
    /// [`CodecError::BackendUnavailable`] without a driver or GPU, or when
    /// the GPU's session limit is reached.
    pub fn probe_with(config: EncoderConfig) -> CodecResult<Self> {
        let input = Input::of(&config)?;
        let profile = C::profile(&config)?;
        let width = config.width.max(16);
        let height = config.height.max(16);
        let fps = NonZeroU32::new(config.fps).unwrap_or(NonZeroU32::MIN);
        let keyframe_interval = config.keyframe_interval_frames();
        let bitrate = config.bitrate_kbps.saturating_mul(1000).max(MIN_BITRATE);
        let mut nv_config = nv::Config::low_latency(
            C::NV_CODEC,
            width,
            height,
            input.format(),
            FrameRate::per_second(fps),
            bitrate,
        );
        // No B-frames (every segment self-contained for HLS and the cast
        // receivers), CBR with max == average (a Chromecast behind weak
        // Wi-Fi aborts when the peak overshoots its buffer), parameter
        // sets with every IDR (receivers join at any IDR): all from
        // `low_latency`.
        nv_config.gop =
            Gop::idr_every(NonZeroU32::new(keyframe_interval).unwrap_or(NonZeroU32::MIN));
        nv_config.profile = Some(profile);
        nv_config.color = Some(colour(&config, input)?);
        nv_config.hdr10 = config.hdr.filter(|hdr| !hdr.is_empty()).map(hdr10);

        let context = cuda::context().map_err(map_error(BackendId::Nvenc, Operation::Open))?;
        let encoder = Encoder::open(&context, &nv_config)
            .map_err(map_error(BackendId::Nvenc, Operation::Open))?;
        info!(
            codec = %C::NV_CODEC,
            width,
            height,
            fps = fps.get(),
            input = %input.format(),
            api = %encoder.api_version(),
            "NVENC encoder ready"
        );
        Ok(Self {
            encoder,
            #[cfg(target_os = "linux")]
            zero_copy: zero_copy::State::default(),
            input,
            requested: config,
            frame_count: 0,
            keyframe_interval: u64::from(keyframe_interval),
            pending_keyframe: false,
            scaler: FrameScaler::default(),
            _codec: PhantomData,
        })
    }

    fn size(&self) -> (u32, u32) {
        let config = self.encoder.config();
        (config.width, config.height)
    }

    /// Submits one picture; returns what must stay alive until its packet
    /// is out: the frame whose GPU memory NVENC reads, if any.
    fn submit(
        &mut self,
        frame: CapturedFrame,
        options: FrameOptions,
    ) -> CodecResult<Option<CapturedFrame>> {
        match frame {
            CapturedFrame::Cpu(raw) => {
                self.encode_raw(raw, options)?;
                Ok(None)
            }
            CapturedFrame::Native(native) => {
                let (width, height) = self.size();
                let surface = native.surface::<CudaSurface>().filter(|_| {
                    (native.width, native.height) == (width, height)
                        && native.format() == self.input.pixel_format()
                });
                if let Some(surface) = surface {
                    let frame = surface.frame();
                    let device = nv::DeviceFrame {
                        buffer: frame.shared_buffer(),
                        chroma_row: None,
                        ready: Some(frame.ready()),
                    };
                    self.encoder
                        .encode_device(device, options)
                        .map_err(map_error(BackendId::Nvenc, Operation::Encode))?;
                    return Ok(Some(CapturedFrame::Native(native)));
                }
                let raw = CapturedFrame::Native(native).into_cpu().map_err(|error| {
                    CodecError::backend(BackendId::Nvenc, Operation::Encode, error)
                })?;
                self.encode_raw(raw, options)?;
                Ok(None)
            }
            #[cfg(any(target_os = "linux", target_os = "android"))]
            CapturedFrame::Gpu(gpu) => {
                #[cfg(target_os = "linux")]
                if let Input::Rgb(format) = self.input
                    && (gpu.width, gpu.height) == self.size()
                    && gpu.format == format
                    // A failure is logged by the zero-copy path; this frame
                    // is read back below.
                    && self
                        .zero_copy
                        .encode(&mut self.encoder, &gpu, options)
                        .is_ok()
                {
                    return Ok(Some(CapturedFrame::Gpu(gpu)));
                }
                let raw = CapturedFrame::Gpu(gpu).into_cpu().map_err(|error| {
                    CodecError::backend(BackendId::Nvenc, Operation::Encode, error)
                })?;
                self.encode_raw(raw, options)?;
                Ok(None)
            }
        }
    }

    /// Encodes a CPU picture (uploaded on the encoder's stream).
    fn encode_raw(&mut self, raw: RawFrame, options: FrameOptions) -> CodecResult<()> {
        let (width, height) = self.size();
        let encode = |error| CodecError::backend(BackendId::Nvenc, Operation::Encode, error);
        let raw = self
            .scaler
            .fit(raw, FrameSize::new(width, height))
            .map_err(encode)?;
        let raw = match (self.input, raw.format) {
            (Input::Nv12, PixelFormat::P010) => {
                frame_view::eight_bit(raw, width, height).map_err(encode)?
            }
            _ => raw,
        };
        let expected = self.input.pixel_format();
        if raw.format != expected {
            // The session was opened for one layout; another would be read
            // as garbage (or with red and blue swapped).
            return Err(CodecError::UnsupportedInput {
                backend: BackendId::Nvenc,
                expected: format!("{expected:?} (the configured format)"),
                actual: raw.format,
            });
        }
        let view = FrameView::new(&raw, width, height).map_err(encode)?;
        let data = view.data();
        let host = match self.input {
            Input::Rgb(_) => HostFrame::Packed(HostImage {
                data,
                pitch: view.stride(),
            }),
            Input::Nv12 | Input::P010 => {
                let (luma, chroma) = data.split_at(view.chroma_offset().min(data.len()));
                HostFrame::SemiPlanar {
                    luma: HostImage {
                        data: luma,
                        pitch: view.stride(),
                    },
                    chroma: HostImage {
                        data: chroma,
                        pitch: view.chroma_stride(),
                    },
                }
            }
        };
        self.encoder
            .encode_host(host, options)
            .map_err(map_error(BackendId::Nvenc, Operation::Encode))
    }

    /// Submits one picture and collects its packet. `force_idr` is decided
    /// by the caller so it can keep the request alive when this fails.
    fn encode_frame(&mut self, frame: CapturedFrame, force_idr: bool) -> CodecResult<EncodedFrame> {
        let timestamp_us = frame.timestamp_us();
        let options = FrameOptions {
            timestamp: timestamp_us,
            force_idr,
        };
        // Kept until the packet is out: NVENC reads a device or DMA-BUF
        // input until then, and the frame holds the capture buffer's lease
        // and the decoder's pooled buffer.
        let input = self.submit(frame, options)?;
        let packet = self.encoder.next_packet().ok_or_else(|| {
            CodecError::backend(
                BackendId::Nvenc,
                Operation::Encode,
                "NVENC produced no packet for the picture",
            )
        })?;
        drop(input);
        let pts = self.frame_count;
        self.frame_count += 1;
        Ok(EncodedFrame {
            codec: C::FERRICAST_CODEC,
            is_keyframe: matches!(packet.picture_type, PictureType::I | PictureType::Idr),
            data: Bytes::from(packet.data),
            timestamp_us,
            duration_us: Some(1_000_000 / u64::from(self.requested.fps.max(1))),
            pts_dts: (pts, pts),
        })
    }

    fn retarget(&mut self, kbps: u32) -> CodecResult<()> {
        let bitrate = kbps.saturating_mul(1000).max(MIN_BITRATE);
        // CBR keeps max == average.
        self.encoder
            .set_rate_control(RateControl::Cbr { bitrate, vbv: None })
            .map_err(map_error(BackendId::Nvenc, Operation::SetBitrate))?;
        self.requested.bitrate_kbps = kbps;
        info!(codec = %C::NV_CODEC, kbps, "NVENC bitrate retargeted live");
        Ok(())
    }
}

impl<C: NvencCodec> VideoEncoder for NvencEncoder<C> {
    const CODEC: Codec = C::FERRICAST_CODEC;

    fn configure(&mut self, config: &EncoderConfig) -> CodecResult<()> {
        if self.requested.same_session_shape(config) {
            // Same session; only the bitrate may differ, and NVENC
            // retargets that live.
            if config.bitrate_kbps != self.requested.bitrate_kbps {
                self.retarget(config.bitrate_kbps)?;
            }
            return Ok(());
        }
        // Geometry, rate, GOP, input, profile, colour or HDR changed: those
        // are baked into the session (and its registered inputs).
        *self = Self::probe_with(config.clone())?;
        Ok(())
    }

    fn encode(&mut self, frame: CapturedFrame) -> CodecResult<EncodedFrame> {
        let force_idr = self
            .frame_count
            .is_multiple_of(self.keyframe_interval.max(1))
            || self.pending_keyframe;
        let result = self.encode_frame(frame, force_idr);
        // A requested IDR is only consumed by a picture NVENC produced,
        // and a failure asks for a fresh IDR so the receiver never
        // predicts across the gap.
        self.pending_keyframe = result.is_err();
        result
    }

    fn flush(mut self) -> CodecResult<Vec<EncodedFrame>> {
        // Without B-frames or frames in flight every packet is already
        // out; this drains whatever the session still holds anyway.
        self.encoder
            .flush()
            .map_err(map_error(BackendId::Nvenc, Operation::Flush))?;
        let duration = 1_000_000 / u64::from(self.requested.fps.max(1));
        Ok(std::iter::from_fn(|| self.encoder.next_packet())
            .map(|packet| {
                let pts = self.frame_count;
                self.frame_count += 1;
                EncodedFrame {
                    codec: C::FERRICAST_CODEC,
                    is_keyframe: matches!(packet.picture_type, PictureType::I | PictureType::Idr),
                    timestamp_us: packet.timestamp,
                    data: Bytes::from(packet.data),
                    duration_us: Some(duration),
                    pts_dts: (pts, pts),
                }
            })
            .collect())
    }

    fn set_bitrate_kbps(&mut self, kbps: u32) -> CodecResult<()> {
        self.retarget(kbps)
    }

    fn get_headers(&mut self) -> CodecResult<Vec<u8>> {
        // SPS + PPS (and VPS for HEVC) in Annex B, out of band.
        self.encoder
            .sequence_params()
            .map_err(map_error(BackendId::Nvenc, Operation::Headers))
    }

    fn request_keyframe(&mut self) {
        self.pending_keyframe = true;
    }
}

/// DMA-BUF capture frames encoded without a CPU copy (Linux).
#[cfg(target_os = "linux")]
mod zero_copy {
    use std::collections::HashMap;
    use std::os::fd::AsFd as _;
    use std::time::Duration;

    use ferricast_codec::cuda::nvcodec::encode::{Encoder, FrameOptions};
    use ferricast_codec::cuda::nvcodec::interop::{
        BufferKey, DmaBuf, DmaBufImporter, DrmFormat, DrmNode, FenceWait, FrameView, Modifier,
    };
    use ferricast_codec::cuda::{self, nvcodec};
    use ferricast_core::{DmaBufId, GpuFrame, PixelFormat};
    use tracing::info;

    /// How long the producer's write fence is waited for.
    const FENCE_TIMEOUT: Duration = Duration::from_millis(100);
    /// More distinct buffers than any capture pool has: past this, the
    /// producer is not reusing its buffers, so the caches start over.
    const MAX_BUFFERS: usize = 16;
    /// The DRM node the capture renders on, when the user names it (the
    /// same variable the capture's Vulkan importer reads).
    const DRM_NODE_ENV: &str = "FERRICAST_DRM_RENDER_NODE";

    /// Why a frame could not go the zero-copy way.
    #[derive(Debug, thiserror::Error)]
    pub(super) enum Error {
        #[error("DMA-BUF import into CUDA is unavailable here")]
        Unavailable,
        #[error("cannot identify the DMA-BUF: {0}")]
        Identity(#[source] std::io::Error),
        #[error(transparent)]
        Nvcodec(#[from] nvcodec::Error),
    }

    /// The zero-copy path's state, created on the first DMA-BUF.
    #[derive(Default)]
    pub(super) enum State {
        #[default]
        Untried,
        Ready(Box<ZeroCopy>),
        /// The importer cannot be created on this machine (no EGL device
        /// for the GPU, another GPU renders the capture): every frame goes
        /// through the CPU.
        Unavailable,
    }

    impl State {
        /// Encodes `frame` from its DMA-BUF. Errors leave the encoder able
        /// to take the same frame from the CPU.
        pub(super) fn encode(
            &mut self,
            encoder: &mut Encoder,
            frame: &GpuFrame,
            options: FrameOptions,
        ) -> Result<(), Error> {
            if matches!(self, Self::Untried) {
                *self = match ZeroCopy::new() {
                    Ok(zero_copy) => Self::Ready(Box::new(zero_copy)),
                    Err(error) => {
                        info!(%error, "NVENC: no DMA-BUF zero-copy, capture frames are read back");
                        Self::Unavailable
                    }
                };
            }
            match self {
                Self::Ready(zero_copy) => {
                    let result = zero_copy.encode(encoder, frame, options);
                    if let Err(error) = &result {
                        zero_copy.failures += 1;
                        // Loud the first time, then once per minute at 60 fps:
                        // a path that keeps failing costs a readback per frame.
                        if zero_copy.failures % 3600 == 1 {
                            tracing::warn!(
                                %error,
                                failures = zero_copy.failures,
                                "NVENC: DMA-BUF zero-copy failed, reading the frame back"
                            );
                        }
                    }
                    result
                }
                Self::Untried | Self::Unavailable => Err(Error::Unavailable),
            }
        }
    }

    /// What a registration depends on: a buffer whose description changes
    /// under the same identity is registered again.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Description {
        width: u32,
        height: u32,
        format: PixelFormat,
        offset: u32,
        stride: u32,
        modifier: u64,
    }

    impl From<&GpuFrame> for Description {
        fn from(frame: &GpuFrame) -> Self {
            Self {
                width: frame.width,
                height: frame.height,
                format: frame.format,
                offset: frame.plane.offset,
                stride: frame.plane.stride,
                modifier: frame.plane.modifier,
            }
        }
    }

    /// One capture buffer imported into CUDA and known to NVENC.
    #[derive(Debug, Clone, Copy)]
    struct Imported {
        /// NVENC's key for the buffer's CUDA array (tiled layouts) or
        /// pitched memory (LINEAR).
        key: u64,
        description: Description,
    }

    /// The importer and what is registered with it and with NVENC.
    pub(super) struct ZeroCopy {
        importer: DmaBufImporter,
        buffers: HashMap<DmaBufId, Imported>,
        next_key: u64,
        /// The description of the last frame: a change means the producer
        /// reallocated its pool.
        last: Option<Description>,
        /// Frames that fell back to the CPU so far.
        failures: u64,
    }

    impl ZeroCopy {
        fn new() -> Result<Self, nvcodec::Error> {
            let context = cuda::context()?;
            let node =
                std::env::var_os(DRM_NODE_ENV).and_then(|path| DrmNode::from_path(path).ok());
            Ok(Self {
                importer: DmaBufImporter::new(&context, node)?,
                buffers: HashMap::new(),
                next_key: 0,
                last: None,
                failures: 0,
            })
        }

        /// Forgets every buffer, NVENC's registrations first: an array must
        /// stay registered with CUDA while NVENC's registration of it lives.
        fn clear(&mut self, encoder: &mut Encoder) {
            encoder.clear_input_caches();
            self.importer.clear();
            self.buffers.clear();
        }

        fn encode(
            &mut self,
            encoder: &mut Encoder,
            frame: &GpuFrame,
            options: FrameOptions,
        ) -> Result<(), Error> {
            let description = Description::from(frame);
            let id = frame.plane.fd.buffer_id().map_err(Error::Identity)?;
            let pool_changed = self.last.is_some_and(|last| last != description)
                || (self.buffers.len() >= MAX_BUFFERS && !self.buffers.contains_key(&id));
            if pool_changed {
                self.clear(encoder);
            }
            self.last = Some(description);
            let key = BufferKey::from((id.dev, id.ino));
            let imported = match self.buffers.get(&id).copied() {
                Some(imported) if imported.description == description => imported,
                stale => {
                    if let Some(stale) = stale {
                        encoder.invalidate_array(stale.key);
                        self.importer.forget(key);
                    }
                    let imported = Imported {
                        key: self.next_key,
                        description,
                    };
                    self.next_key += 1;
                    self.importer.import(key, &dma_buf(frame))?;
                    self.buffers.insert(id, imported);
                    imported
                }
            };
            let mapped = self.importer.map(key, FenceWait::Implicit(FENCE_TIMEOUT))?;
            // SAFETY (both arms): the memory belongs to the importer's
            // registration under `key`, which stays alive until NVENC's
            // registration of `imported.key` is invalidated (`clear` and the
            // re-import above invalidate NVENC's first; `NvencEncoder` drops
            // the encoder before the importer). The encoder's caller holds
            // the frame, and with it the producer's lease, until the packet
            // is out, so the buffer is not written while NVENC reads it.
            match mapped.view() {
                FrameView::Array(array) => {
                    // SAFETY: see above.
                    unsafe { encoder.encode_array(imported.key, array, options) }?;
                }
                FrameView::Pitched(pitched) => {
                    // SAFETY: see above.
                    unsafe { encoder.encode_pitched(imported.key, pitched, options) }?;
                }
            }
            Ok(())
        }
    }

    /// The frame's DMA-BUF as the importer takes it.
    /// Only BGRA and RGBA frames get here (the session's RGB input).
    fn dma_buf(frame: &GpuFrame) -> DmaBuf<'_> {
        // BGRA in memory is DRM's ARGB8888 (a little-endian word), RGBA
        // its ABGR8888; the importer registers both as CUDA takes them.
        let format = match frame.format {
            PixelFormat::Rgba => DrmFormat::Abgr8888,
            PixelFormat::Bgra | PixelFormat::Nv12 | PixelFormat::I420 | PixelFormat::P010 => {
                DrmFormat::Argb8888
            }
        };
        DmaBuf {
            fd: frame.plane.fd.as_fd(),
            width: frame.width,
            height: frame.height,
            format,
            offset: frame.plane.offset,
            stride: frame.plane.stride,
            modifier: Modifier(frame.plane.modifier),
        }
    }
}
