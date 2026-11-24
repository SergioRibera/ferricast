//! H.264 / H.265 encode through VideoToolbox.
//!
//! VideoToolbox hands compressed frames to a callback. Each `encode`
//! submits one picture and then calls `VTCompressionSessionCompleteFrames`
//! up to its timestamp, which delivers that picture's output before it
//! returns (frame reordering is off, so nothing is held back): the
//! [`VideoEncoder`] contract of one packet per frame holds.
//!
//! Output is Annex B. VideoToolbox emits length-prefixed NAL units and
//! keeps the parameter sets in the sample's format description; they are
//! copied in front of every keyframe, as x264 repeats them in-band.

use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::sync::{Mutex, MutexGuard, PoisonError};

use bytes::Bytes;
use ferricast_codec::bitstream::{length_prefixed_to_annex_b, split_annex_b};
use ferricast_codec::{Candidate, VideoEncoderRegistry};
use ferricast_core::{
    Acceleration, BackendId, CapturedFrame, Codec, CodecError, CodecKind, CodecResult,
    ColourDescription, ColourPrimaries, ColourRange, EncodedFrame, EncoderConfig, H264Profile,
    H265Profile, MatrixCoefficients, Operation, PixelFormat, RawFrame, TransferCharacteristics,
    VideoEncoder,
};
use objc2_core_foundation::{
    CFBoolean, CFDictionary, CFNumber, CFRetained, CFString, CFType, kCFAllocatorDefault,
};
use objc2_core_media::{
    CMFormatDescription, CMSampleBuffer, CMTime, CMTimeFlags, CMVideoCodecType,
    CMVideoFormatDescriptionGetH264ParameterSetAtIndex,
    CMVideoFormatDescriptionGetHEVCParameterSetAtIndex, kCMTimeInvalid, kCMVideoCodecType_H264,
    kCMVideoCodecType_HEVC,
};
use objc2_core_video::{
    CVColorPrimariesGetStringForIntegerCodePoint, CVPixelBuffer, CVPixelBufferCreate,
    CVPixelBufferGetBaseAddress, CVPixelBufferGetBaseAddressOfPlane, CVPixelBufferGetBytesPerRow,
    CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
    CVPixelBufferUnlockBaseAddress, CVTransferFunctionGetStringForIntegerCodePoint,
    CVYCbCrMatrixGetStringForIntegerCodePoint, kCVPixelBufferIOSurfacePropertiesKey,
    kCVPixelFormatType_32BGRA, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
    kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
};
use objc2_video_toolbox::{
    VTCompressionSession, VTEncodeInfoFlags, VTSessionSetProperty,
    kVTCompressionPropertyKey_AllowFrameReordering, kVTCompressionPropertyKey_AverageBitRate,
    kVTCompressionPropertyKey_ColorPrimaries, kVTCompressionPropertyKey_ExpectedFrameRate,
    kVTCompressionPropertyKey_MaxKeyFrameInterval,
    kVTCompressionPropertyKey_MaxKeyFrameIntervalDuration, kVTCompressionPropertyKey_ProfileLevel,
    kVTCompressionPropertyKey_RealTime, kVTCompressionPropertyKey_TransferFunction,
    kVTCompressionPropertyKey_YCbCrMatrix, kVTCouldNotFindVideoEncoderErr,
    kVTEncodeFrameOptionKey_ForceKeyFrame, kVTProfileLevel_H264_Baseline_AutoLevel,
    kVTProfileLevel_H264_High_AutoLevel, kVTProfileLevel_H264_Main_AutoLevel,
    kVTProfileLevel_HEVC_Main_AutoLevel, kVTProfileLevel_HEVC_Main10_AutoLevel,
    kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder,
    kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder,
};
use tracing::{debug, warn};

use super::os_status;
use crate::frame_view::FrameView;
use crate::nv12::{self, Nv12Layout};
use crate::scale::{FrameScaler, FrameSize};

const BACKEND: BackendId = BackendId::VideoToolbox;
/// Timestamps cross VideoToolbox as microseconds.
const MICROS: i32 = 1_000_000;
/// `kVTPropertyNotSupportedErr`: an encoder without the property.
const PROPERTY_NOT_SUPPORTED: i32 = -12_900;

/// Adds the VideoToolbox H.264 / H.265 encoders.
pub(crate) fn video(registry: VideoEncoderRegistry) -> VideoEncoderRegistry {
    if !matches!(
        registry.target().codec,
        CodecKind::Video(Codec::H264 | Codec::H265)
    ) {
        return registry;
    }
    let registry = registry.candidate(Candidate::new(
        BACKEND,
        Acceleration::Hardware,
        |config: &EncoderConfig| {
            VtEncoder::open(config, Acceleration::Hardware).map(|encoder| Box::new(encoder) as _)
        },
    ));
    // Apple ships software H.264 / HEVC encoders on macOS only; iOS
    // encodes in the media engine.
    #[cfg(target_os = "macos")]
    let registry = registry.candidate(Candidate::new(
        BACKEND,
        Acceleration::Software,
        |config: &EncoderConfig| {
            VtEncoder::open(config, Acceleration::Software).map(|encoder| Box::new(encoder) as _)
        },
    ));
    registry
}

fn fail(operation: Operation, what: impl std::fmt::Display) -> CodecError {
    CodecError::backend(BACKEND, operation, what)
}

fn check(status: i32, operation: Operation, what: &str) -> CodecResult<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(fail(operation, format!("{what}: {}", os_status(status))))
    }
}

fn cm_time(value: i64) -> CMTime {
    CMTime {
        value,
        timescale: MICROS,
        flags: CMTimeFlags::Valid,
        epoch: 0,
    }
}

/// What the output callback saw for one submitted picture.
enum Output {
    Frame {
        annex_b: Vec<u8>,
        parameter_sets: Vec<u8>,
        keyframe: bool,
        pts_us: i64,
    },
    Dropped,
    Failed(String),
}

/// Pictures the callback delivered and the session has not handed out.
type Outbox = Mutex<Vec<Output>>;

fn lock(outbox: &Outbox) -> MutexGuard<'_, Vec<Output>> {
    outbox.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One VideoToolbox compression session and the shape it was built for.
struct Session {
    session: CFRetained<VTCompressionSession>,
    /// The callback's reference value; boxed so its address outlives
    /// every call into the session (it is freed after `invalidate`).
    outbox: Box<Outbox>,
    config: EncoderConfig,
    nv12: Vec<u8>,
}

// SAFETY: a VideoToolbox session may be driven from any thread as long as
// calls do not overlap; `Session` is only reached through `&mut
// VtEncoder`, and the output callback only touches the `Mutex` outbox.
unsafe impl Send for Session {}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: the session is live; invalidating it stops callbacks
        // before `outbox` (their reference value) is freed.
        unsafe { self.session.invalidate() };
    }
}

/// The parameter sets (SPS / PPS, plus VPS for HEVC) of a format
/// description, as Annex B, and the NAL length size its samples use.
fn parameter_sets(
    description: &CMFormatDescription,
    codec: Codec,
) -> CodecResult<(Vec<u8>, usize)> {
    let get = match codec {
        Codec::H265 => CMVideoFormatDescriptionGetHEVCParameterSetAtIndex,
        _ => CMVideoFormatDescriptionGetH264ParameterSetAtIndex,
    };
    let mut count = 0usize;
    let mut length_size: i32 = 0;
    // SAFETY: `description` is a live video format description; the out
    // pointers are valid locals and null for what is not asked for.
    let status = unsafe {
        get(
            description,
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            &raw mut count,
            &raw mut length_size,
        )
    };
    check(status, Operation::Encode, "reading the parameter-set count")?;
    let mut out = Vec::new();
    for index in 0..count {
        let mut data: *const u8 = ptr::null();
        let mut size = 0usize;
        // SAFETY: as above; `index` is below the count just read.
        let status = unsafe {
            get(
                description,
                index,
                &raw mut data,
                &raw mut size,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        check(status, Operation::Encode, "reading a parameter set")?;
        if data.is_null() {
            continue;
        }
        // SAFETY: VideoToolbox returned `size` bytes at `data`, owned by
        // `description`, which outlives this copy.
        let set = unsafe { std::slice::from_raw_parts(data, size) };
        out.extend_from_slice(&ferricast_codec::bitstream::START_CODE);
        out.extend_from_slice(set);
    }
    let length_size = usize::try_from(length_size).unwrap_or(4);
    Ok((out, length_size))
}

/// Whether Annex B `data` carries a random-access picture.
fn is_keyframe(data: &[u8], codec: Codec) -> bool {
    split_annex_b(data)
        .iter()
        .any(|nal| match (codec, nal.first()) {
            (Codec::H265, Some(&header)) => (16..=21).contains(&((header >> 1) & 0x3f)),
            (_, Some(&header)) => header & 0x1f == 5,
            (_, None) => false,
        })
}

/// Reads one compressed sample.
///
/// # Safety
/// `sample` must be a live `CMSampleBuffer` VideoToolbox produced.
unsafe fn read_sample(sample: &CMSampleBuffer, codec: Codec) -> CodecResult<Output> {
    // SAFETY: `sample` is live (caller contract).
    let (block, description, pts) = unsafe {
        (
            sample.data_buffer(),
            sample.format_description(),
            sample.presentation_time_stamp(),
        )
    };
    let block = block.ok_or_else(|| fail(Operation::Encode, "the sample has no data"))?;
    let description = description
        .ok_or_else(|| fail(Operation::Encode, "the sample has no format description"))?;
    // SAFETY: `block` is a live block buffer.
    let len = unsafe { block.data_length() };
    let mut avcc = vec![0u8; len];
    if let Some(destination) = NonNull::new(avcc.as_mut_ptr().cast::<c_void>()) {
        // SAFETY: `avcc` holds `len` writable bytes, the block's length.
        let status = unsafe { block.copy_data_bytes(0, len, destination) };
        check(status, Operation::Encode, "copying the compressed sample")?;
    }
    let (parameter_sets, length_size) = parameter_sets(&description, codec)?;
    let annex_b = length_prefixed_to_annex_b(&avcc, length_size).ok_or_else(|| {
        fail(
            Operation::Encode,
            "the sample's NAL unit lengths are inconsistent",
        )
    })?;
    Ok(Output::Frame {
        keyframe: is_keyframe(&annex_b, codec),
        annex_b,
        parameter_sets,
        pts_us: pts.value,
    })
}

unsafe extern "C-unwind" fn on_encoded(
    refcon: *mut c_void,
    source: *mut c_void,
    status: i32,
    flags: VTEncodeInfoFlags,
    sample: *mut CMSampleBuffer,
) {
    // SAFETY: `refcon` is the `Outbox` the session was created with; it
    // outlives the session (`Session::drop` invalidates first). `source`
    // carries the codec as an integer.
    let outbox = unsafe { &*refcon.cast::<Outbox>() };
    let codec = if source as usize == 265 {
        Codec::H265
    } else {
        Codec::H264
    };
    let output = if status != 0 {
        Output::Failed(os_status(status))
    } else if flags.contains(VTEncodeInfoFlags::FrameDropped) || sample.is_null() {
        Output::Dropped
    } else {
        // SAFETY: VideoToolbox passes a live sample buffer for this call.
        match unsafe { read_sample(&*sample, codec) } {
            Ok(output) => output,
            Err(error) => Output::Failed(error.to_string()),
        }
    };
    lock(outbox).push(output);
}

fn codec_type(codec: Codec) -> CodecResult<CMVideoCodecType> {
    match codec {
        Codec::H264 => Ok(kCMVideoCodecType_H264),
        Codec::H265 => Ok(kCMVideoCodecType_HEVC),
        other => Err(CodecError::BackendCannotHandle {
            backend: BACKEND,
            target: ferricast_core::CodecTarget::encode(other),
        }),
    }
}

/// The `ProfileLevel` value for the configuration's profile cap.
fn profile_level(config: &EncoderConfig) -> &'static CFString {
    // SAFETY: VideoToolbox exports these constants for the life of the process.
    unsafe {
        match config.codec {
            Codec::H265 => match config.max_h265_profile {
                Some(H265Profile::Main10) => kVTProfileLevel_HEVC_Main10_AutoLevel,
                Some(H265Profile::Main) | None => kVTProfileLevel_HEVC_Main_AutoLevel,
            },
            _ => match config.max_h264_profile {
                Some(H264Profile::Baseline) => kVTProfileLevel_H264_Baseline_AutoLevel,
                Some(H264Profile::Main) => kVTProfileLevel_H264_Main_AutoLevel,
                Some(H264Profile::High) | None => kVTProfileLevel_H264_High_AutoLevel,
            },
        }
    }
}

/// A colour property: what it is, and its value (`None` to leave it unset).
type ColourProperty = (&'static str, Option<CFRetained<CFString>>);

/// The VideoToolbox colour property values of `colour`; `None` for an
/// unspecified field (left to the encoder).
fn colour_properties(colour: ColourDescription) -> CodecResult<[ColourProperty; 3]> {
    let unsupported = |field: &str, code: u8| CodecError::UnsupportedConfig {
        backend: BACKEND,
        what: format!("signalling {field} code {code}"),
    };
    let primaries = match colour.primaries {
        ColourPrimaries::Unspecified => None,
        other => Some(
            CVColorPrimariesGetStringForIntegerCodePoint(i32::from(other.code()))
                .ok_or_else(|| unsupported("colour_primaries", other.code()))?,
        ),
    };
    let transfer = match colour.transfer {
        TransferCharacteristics::Unspecified => None,
        other => Some(
            CVTransferFunctionGetStringForIntegerCodePoint(i32::from(other.code()))
                .ok_or_else(|| unsupported("transfer_characteristics", other.code()))?,
        ),
    };
    let matrix = match colour.matrix {
        MatrixCoefficients::Unspecified => None,
        other => Some(
            CVYCbCrMatrixGetStringForIntegerCodePoint(i32::from(other.code()))
                .ok_or_else(|| unsupported("matrix_coefficients", other.code()))?,
        ),
    };
    Ok([
        ("colour primaries", primaries),
        ("transfer function", transfer),
        ("YCbCr matrix", matrix),
    ])
}

impl Session {
    fn open(config: &EncoderConfig, acceleration: Acceleration) -> CodecResult<Self> {
        let codec_type = codec_type(config.codec)?;
        let (width, height) = (
            i32::try_from(config.width).map_err(|_| fail(Operation::Open, "width exceeds i32"))?,
            i32::try_from(config.height)
                .map_err(|_| fail(Operation::Open, "height exceeds i32"))?,
        );
        // SAFETY: VideoToolbox exports these keys for the life of the process.
        let specification = unsafe {
            match acceleration {
                Acceleration::Hardware => CFDictionary::<CFString, CFType>::from_slices(
                    &[kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder],
                    &[CFBoolean::new(true).as_ref()],
                ),
                Acceleration::Software => CFDictionary::<CFString, CFType>::from_slices(
                    &[kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder],
                    &[CFBoolean::new(false).as_ref()],
                ),
            }
        };
        let outbox: Box<Outbox> = Box::new(Mutex::new(Vec::new()));
        let mut raw: *mut VTCompressionSession = ptr::null_mut();
        // SAFETY: every pointer is valid for the call; `on_encoded` matches
        // the callback prototype and `outbox` outlives the session.
        let status = unsafe {
            VTCompressionSession::create(
                None,
                width,
                height,
                codec_type,
                Some(specification.as_ref()),
                None,
                None,
                Some(on_encoded),
                ptr::from_ref::<Outbox>(&outbox).cast_mut().cast(),
                NonNull::from(&mut raw),
            )
        };
        if status == kVTCouldNotFindVideoEncoderErr {
            return Err(CodecError::BackendUnavailable {
                backend: BACKEND,
                reason: format!(
                    "no {acceleration} {} encoder for {}x{} on this device ({})",
                    config.codec,
                    config.width,
                    config.height,
                    os_status(status)
                ),
            });
        }
        check(status, Operation::Open, "VTCompressionSessionCreate")?;
        let session = NonNull::new(raw).ok_or_else(|| {
            fail(
                Operation::Open,
                "VTCompressionSessionCreate returned no session",
            )
        })?;
        // SAFETY: `create` succeeded and handed over a +1 reference.
        let session = unsafe { CFRetained::from_raw(session) };
        let session = Self {
            session,
            outbox,
            config: config.clone(),
            nv12: Vec::new(),
        };
        session.apply(config)?;
        // SAFETY: the session is live.
        let status = unsafe { session.session.prepare_to_encode_frames() };
        check(
            status,
            Operation::Open,
            "VTCompressionSessionPrepareToEncodeFrames",
        )?;
        Ok(session)
    }

    fn set(&self, key: &CFString, value: &CFType) -> i32 {
        // SAFETY: the session is live and `value` has the type the key
        // documents (callers pair them).
        unsafe { VTSessionSetProperty(self.session.as_ref(), key, Some(value)) }
    }

    /// Sets a property the stream depends on.
    fn require(&self, key: &CFString, value: &CFType, what: &str) -> CodecResult<()> {
        match self.set(key, value) {
            0 => Ok(()),
            PROPERTY_NOT_SUPPORTED => Err(CodecError::UnsupportedConfig {
                backend: BACKEND,
                what: format!("{what} on this encoder"),
            }),
            status => Err(fail(
                Operation::Configure,
                format!("setting {what}: {}", os_status(status)),
            )),
        }
    }

    /// Sets a property that only tunes the encoder.
    fn prefer(&self, key: &CFString, value: &CFType, what: &str) {
        let status = self.set(key, value);
        if status != 0 {
            debug!(what, status = %os_status(status), "VideoToolbox ignored an encoder hint");
        }
    }

    fn apply(&self, config: &EncoderConfig) -> CodecResult<()> {
        let keyframe_interval =
            i32::try_from(config.keyframe_interval_frames()).unwrap_or(i32::MAX);
        // SAFETY: VideoToolbox exports these keys for the life of the process.
        let keys = unsafe {
            (
                kVTCompressionPropertyKey_RealTime,
                kVTCompressionPropertyKey_AllowFrameReordering,
                kVTCompressionPropertyKey_ProfileLevel,
                kVTCompressionPropertyKey_MaxKeyFrameInterval,
                kVTCompressionPropertyKey_MaxKeyFrameIntervalDuration,
                kVTCompressionPropertyKey_ExpectedFrameRate,
                [
                    kVTCompressionPropertyKey_ColorPrimaries,
                    kVTCompressionPropertyKey_TransferFunction,
                    kVTCompressionPropertyKey_YCbCrMatrix,
                ],
            )
        };
        let (real_time, reordering, profile, interval, duration, rate, colour_keys) = keys;
        self.prefer(
            real_time,
            CFBoolean::new(true).as_ref(),
            "real-time encoding",
        );
        self.require(
            reordering,
            CFBoolean::new(false).as_ref(),
            "disabling B-frames",
        )?;
        self.require(
            profile,
            profile_level(config).as_ref(),
            "the requested profile",
        )?;
        self.set_bitrate(config.bitrate_kbps)?;
        self.require(
            interval,
            CFNumber::new_i32(keyframe_interval).as_ref(),
            "the keyframe interval",
        )?;
        self.prefer(
            duration,
            CFNumber::new_f64(f64::from(config.keyframe_interval_secs)).as_ref(),
            "the keyframe interval duration",
        );
        self.prefer(
            rate,
            CFNumber::new_i32(i32::try_from(config.fps).unwrap_or(i32::MAX)).as_ref(),
            "the expected frame rate",
        );
        for (key, (what, value)) in colour_keys
            .into_iter()
            .zip(colour_properties(config.colour)?)
        {
            if let Some(value) = value {
                self.require(key, value.as_ref(), what)?;
            }
        }
        Ok(())
    }

    fn set_bitrate(&self, kbps: u32) -> CodecResult<()> {
        let bits = i64::from(kbps) * 1000;
        // SAFETY: VideoToolbox exports this key for the life of the process.
        let key = unsafe { kVTCompressionPropertyKey_AverageBitRate };
        self.require(key, CFNumber::new_i64(bits).as_ref(), "the average bitrate")
    }

    /// A CoreVideo buffer holding `view`: BGRA as is when the stream
    /// signals limited range (VideoToolbox converts it with the
    /// configured matrix in hardware), anything else as NV12 in the
    /// configured range.
    fn pixel_buffer(&mut self, view: &FrameView<'_>) -> CodecResult<CFRetained<CVPixelBuffer>> {
        let colour = self.config.colour;
        let direct = view.format() == PixelFormat::Bgra && colour.range == ColourRange::Limited;
        let format = if direct {
            kCVPixelFormatType_32BGRA
        } else if colour.range == ColourRange::Full {
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        } else {
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        };
        // An empty IOSurface dictionary asks for IOSurface backing, which
        // the hardware encoder reads without a copy.
        let empty = CFDictionary::<CFString, CFType>::empty();
        let empty: &CFType = (*empty).as_ref();
        // SAFETY: CoreVideo exports this key for the life of the process.
        let surface_key = unsafe { kCVPixelBufferIOSurfacePropertiesKey };
        let attributes = CFDictionary::<CFString, CFType>::from_slices(&[surface_key], &[empty]);
        let mut raw: *mut CVPixelBuffer = ptr::null_mut();
        // SAFETY: the out pointer is a valid local; the dictionary is live.
        let status = unsafe {
            CVPixelBufferCreate(
                kCFAllocatorDefault,
                view.width() as usize,
                view.height() as usize,
                format,
                Some(attributes.as_ref()),
                NonNull::from(&mut raw),
            )
        };
        check(status, Operation::Encode, "CVPixelBufferCreate")?;
        let buffer = NonNull::new(raw)
            .ok_or_else(|| fail(Operation::Encode, "CVPixelBufferCreate returned no buffer"))?;
        // SAFETY: `CVPixelBufferCreate` handed over a +1 reference.
        let buffer = unsafe { CFRetained::from_raw(buffer) };
        // SAFETY: the buffer is live and unlocked.
        let status = unsafe { CVPixelBufferLockBaseAddress(&buffer, CVPixelBufferLockFlags(0)) };
        check(status, Operation::Encode, "CVPixelBufferLockBaseAddress")?;
        let filled = if direct {
            fill_bgra(&buffer, view)
        } else {
            self.fill_nv12(&buffer, view)
        };
        // SAFETY: locked just above with the same flags.
        let status = unsafe { CVPixelBufferUnlockBaseAddress(&buffer, CVPixelBufferLockFlags(0)) };
        filled?;
        check(status, Operation::Encode, "CVPixelBufferUnlockBaseAddress")?;
        Ok(buffer)
    }

    fn fill_nv12(&mut self, buffer: &CVPixelBuffer, view: &FrameView<'_>) -> CodecResult<()> {
        let layout = Nv12Layout::packed(view.width(), view.height());
        self.nv12.resize(layout.len(), 0);
        nv12::write(view, self.config.colour, layout, &mut self.nv12, BACKEND)?;
        let (luma, chroma) = self.nv12.split_at(layout.chroma_offset);
        let rows = view.height() as usize;
        copy_plane(buffer, 0, luma, layout.stride, rows)?;
        copy_plane(buffer, 1, chroma, layout.stride, rows.div_ceil(2))
    }
}

/// Copies `rows` rows of `stride` bytes into plane `plane` of a locked
/// buffer.
fn copy_plane(
    buffer: &CVPixelBuffer,
    plane: usize,
    src: &[u8],
    stride: usize,
    rows: usize,
) -> CodecResult<()> {
    let base = CVPixelBufferGetBaseAddressOfPlane(buffer, plane).cast::<u8>();
    let pitch = CVPixelBufferGetBytesPerRowOfPlane(buffer, plane);
    if base.is_null() || pitch < stride {
        return Err(fail(
            Operation::Encode,
            format!("pixel buffer plane {plane} is unusable"),
        ));
    }
    for (row, line) in src.chunks(stride).take(rows).enumerate() {
        // SAFETY: the buffer is locked; plane `plane` has at least `rows`
        // rows of `pitch >= stride` bytes, so row `row` holds `line`.
        unsafe { ptr::copy_nonoverlapping(line.as_ptr(), base.add(row * pitch), line.len()) };
    }
    Ok(())
}

/// Copies BGRA rows into a locked single-plane BGRA buffer.
fn fill_bgra(buffer: &CVPixelBuffer, view: &FrameView<'_>) -> CodecResult<()> {
    let base = CVPixelBufferGetBaseAddress(buffer).cast::<u8>();
    let pitch = CVPixelBufferGetBytesPerRow(buffer);
    let row_bytes = view.row_bytes();
    if base.is_null() || pitch < row_bytes {
        return Err(fail(Operation::Encode, "BGRA pixel buffer is unusable"));
    }
    for (row, line) in view
        .data()
        .chunks(view.stride())
        .take(view.height() as usize)
        .enumerate()
    {
        // SAFETY: the buffer is locked and holds `height` rows of `pitch
        // >= row_bytes` bytes; `line` has at least `row_bytes` bytes
        // (`FrameView` validated the geometry).
        unsafe { ptr::copy_nonoverlapping(line.as_ptr(), base.add(row * pitch), row_bytes) };
    }
    Ok(())
}

/// A VideoToolbox H.264 / H.265 encoder.
pub(crate) struct VtEncoder {
    acceleration: Acceleration,
    session: Option<Session>,
    scaler: FrameScaler,
    pending_keyframe: bool,
    frame_count: u64,
    /// The parameter sets of the latest keyframe (Annex B).
    headers: Vec<u8>,
}

impl VtEncoder {
    fn open(config: &EncoderConfig, acceleration: Acceleration) -> CodecResult<Self> {
        let mut encoder = Self {
            acceleration,
            session: None,
            scaler: FrameScaler::default(),
            pending_keyframe: false,
            frame_count: 0,
            headers: Vec::new(),
        };
        encoder.configure(config)?;
        Ok(encoder)
    }

    fn session(&mut self, operation: Operation) -> CodecResult<&mut Session> {
        self.session.as_mut().ok_or(CodecError::NotConfigured {
            target: ferricast_core::CodecTarget::encode(Codec::H264),
            operation,
        })
    }

    /// Submits one picture and waits for its output.
    fn submit(&mut self, view: &FrameView<'_>, pts_us: i64, keyframe: bool) -> CodecResult<Output> {
        let session = self.session(Operation::Encode)?;
        let codec = session.config.codec;
        let fps = i64::from(session.config.fps.max(1));
        let buffer = session.pixel_buffer(view)?;
        // SAFETY: VideoToolbox exports this key for the life of the process.
        let force = unsafe { kVTEncodeFrameOptionKey_ForceKeyFrame };
        let options = keyframe.then(|| {
            CFDictionary::<CFString, CFType>::from_slices(
                &[force],
                &[CFBoolean::new(true).as_ref()],
            )
        });
        let pts = cm_time(pts_us);
        let codec_tag: usize = if codec == Codec::H265 { 265 } else { 264 };
        // SAFETY: the session and buffer are live; `options` outlives the
        // call; the source reference value is an integer, never read as a
        // pointer.
        let status = unsafe {
            session.session.encode_frame(
                &buffer,
                pts,
                cm_time(i64::from(MICROS) / fps),
                options.as_deref().map(AsRef::as_ref),
                codec_tag as *mut c_void,
                ptr::null_mut(),
            )
        };
        check(status, Operation::Encode, "VTCompressionSessionEncodeFrame")?;
        // SAFETY: the session is live; completing up to this picture's
        // timestamp delivers its output before returning.
        let status = unsafe { session.session.complete_frames(pts) };
        check(
            status,
            Operation::Encode,
            "VTCompressionSessionCompleteFrames",
        )?;
        let mut outputs = std::mem::take(&mut *lock(&session.outbox));
        match outputs.len() {
            1 => Ok(outputs.remove(0)),
            0 => Err(fail(
                Operation::Encode,
                "the encoder returned no output for the frame",
            )),
            more => {
                warn!(
                    outputs = more,
                    "VideoToolbox returned several outputs for one frame"
                );
                Ok(outputs.remove(0))
            }
        }
    }

    fn try_encode(&mut self, frame: CapturedFrame) -> CodecResult<EncodedFrame> {
        let session = self.session(Operation::Encode)?;
        let (codec, width, height, fps) = (
            session.config.codec,
            session.config.width,
            session.config.height,
            session.config.fps,
        );
        let raw: RawFrame = frame
            .into_cpu()
            .map_err(|error| fail(Operation::Encode, error))?;
        let raw = self
            .scaler
            .fit(raw, FrameSize::new(width, height))
            .map_err(CodecError::from_backend(BACKEND, Operation::Encode))?;
        let view = FrameView::new(&raw, width, height)
            .map_err(CodecError::from_backend(BACKEND, Operation::Encode))?;
        let pts_us = i64::try_from(raw.timestamp_us)
            .map_err(|_| fail(Operation::Encode, "timestamp exceeds i64"))?;
        let output = self.submit(&view, pts_us, self.pending_keyframe)?;
        self.pending_keyframe = false;
        self.finish(output, codec, fps, raw.timestamp_us)
    }

    fn finish(
        &mut self,
        output: Output,
        codec: Codec,
        fps: u32,
        timestamp_us: u64,
    ) -> CodecResult<EncodedFrame> {
        match output {
            Output::Frame {
                annex_b,
                parameter_sets,
                keyframe,
                pts_us,
            } => {
                let index = self.frame_count;
                self.frame_count += 1;
                let data = if keyframe {
                    self.headers.clone_from(&parameter_sets);
                    let mut data = parameter_sets;
                    data.extend_from_slice(&annex_b);
                    data
                } else {
                    annex_b
                };
                Ok(EncodedFrame {
                    codec,
                    data: Bytes::from(data),
                    timestamp_us: u64::try_from(pts_us).unwrap_or(timestamp_us),
                    is_keyframe: keyframe,
                    duration_us: Some(1_000_000 / u64::from(fps.max(1))),
                    pts_dts: (index, index),
                })
            }
            Output::Dropped => Err(fail(Operation::Encode, "the encoder dropped the frame")),
            Output::Failed(reason) => Err(fail(Operation::Encode, reason)),
        }
    }

    /// Encodes a black picture to learn the parameter sets before the
    /// first real frame (consumers such as the HLS segmenter need them
    /// up front) and discards it; the next real frame is forced to be a
    /// keyframe so the stream still starts with one.
    fn prime(&mut self) -> CodecResult<()> {
        let session = self.session(Operation::Headers)?;
        let (width, height) = (session.config.width, session.config.height);
        let layout = Nv12Layout::packed(width, height);
        let mut black = vec![16u8; layout.len()];
        black[layout.chroma_offset..].fill(128);
        let raw = RawFrame {
            width,
            height,
            stride: width,
            format: PixelFormat::Nv12,
            data: Bytes::from(black),
            timestamp_us: 0,
            colour: None,
        };
        let view = FrameView::new(&raw, width, height)
            .map_err(CodecError::from_backend(BACKEND, Operation::Headers))?;
        match self.submit(&view, 0, true)? {
            Output::Frame { parameter_sets, .. } => self.headers = parameter_sets,
            Output::Dropped => {
                return Err(fail(Operation::Headers, "the priming frame was dropped"));
            }
            Output::Failed(reason) => return Err(fail(Operation::Headers, reason)),
        }
        self.pending_keyframe = true;
        Ok(())
    }
}

impl VideoEncoder for VtEncoder {
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, config: &EncoderConfig) -> CodecResult<()> {
        codec_type(config.codec)?;
        if let Some(session) = &mut self.session {
            let current = &session.config;
            let same_shape = current.codec == config.codec
                && current.width == config.width
                && current.height == config.height
                && current.fps == config.fps
                && current.keyframe_interval_secs.to_bits()
                    == config.keyframe_interval_secs.to_bits()
                && current.max_h264_profile == config.max_h264_profile
                && current.max_h265_profile == config.max_h265_profile
                && current.colour == config.colour;
            if same_shape {
                session.set_bitrate(config.bitrate_kbps)?;
                session.config = config.clone();
                return Ok(());
            }
        }
        // Drop the old session before opening the new one: hardware
        // encoders are a limited resource.
        self.session = None;
        self.session = Some(Session::open(config, self.acceleration)?);
        self.headers.clear();
        self.pending_keyframe = true;
        Ok(())
    }

    fn encode(&mut self, frame: CapturedFrame) -> CodecResult<EncodedFrame> {
        self.try_encode(frame)
    }

    fn flush(mut self) -> CodecResult<Vec<EncodedFrame>> {
        let Some(session) = &self.session else {
            return Ok(Vec::new());
        };
        let (codec, fps) = (session.config.codec, session.config.fps);
        // SAFETY: the session is live; `kCMTimeInvalid` completes every
        // pending frame.
        let status = unsafe { session.session.complete_frames(kCMTimeInvalid) };
        check(
            status,
            Operation::Flush,
            "VTCompressionSessionCompleteFrames",
        )?;
        let outputs = std::mem::take(&mut *lock(&session.outbox));
        outputs
            .into_iter()
            .map(|output| self.finish(output, codec, fps, 0))
            .collect()
    }

    fn get_headers(&mut self) -> CodecResult<Vec<u8>> {
        if self.headers.is_empty() {
            self.prime()?;
        }
        Ok(self.headers.clone())
    }

    fn request_keyframe(&mut self) {
        self.pending_keyframe = true;
    }

    fn set_bitrate_kbps(&mut self, kbps: u32) -> CodecResult<()> {
        let session = self.session(Operation::SetBitrate)?;
        session.set_bitrate(kbps)?;
        session.config.bitrate_kbps = kbps;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyframes_are_found_by_nal_type() {
        assert!(is_keyframe(&[0, 0, 0, 1, 0x65, 1], Codec::H264));
        assert!(!is_keyframe(&[0, 0, 0, 1, 0x41, 1], Codec::H264));
        // IDR_W_RADL (19) and TRAIL_R (1) for HEVC.
        assert!(is_keyframe(&[0, 0, 0, 1, 19 << 1, 1], Codec::H265));
        assert!(!is_keyframe(&[0, 0, 0, 1, 1 << 1, 1], Codec::H265));
    }
}
