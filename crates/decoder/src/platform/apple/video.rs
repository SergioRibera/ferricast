//! H.264 / H.265 decode through VideoToolbox.
//!
//! Packets arrive as Annex B. Their parameter sets build the
//! `CMVideoFormatDescription` the session decodes against (rebuilt, and
//! the session recreated when it cannot take the new one, whenever they
//! change); the remaining NAL units are rewritten length-prefixed into a
//! `CMSampleBuffer` stamped with the packet timestamp. Pictures come back
//! through a callback carrying that timestamp; outside low-latency mode
//! temporal processing is on, so VideoToolbox emits them in display
//! order. Each picture is copied out of its `CVPixelBuffer` (NV12, video
//! range) and handed out in the configured pixel format.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::sync::{Mutex, MutexGuard, PoisonError};

use bytes::Bytes;
use ferricast_codec::bitstream::split_annex_b;
use ferricast_codec::{Candidate, VideoDecoderRegistry};
use ferricast_core::{
    Acceleration, BackendId, CapturedFrame, Codec, CodecError, CodecKind, CodecResult, CodecTarget,
    ColourDescription, ColourRange, DecoderConfig, EncodedFrame, Operation, VideoDecoder,
};
use objc2_core_foundation::{CFBoolean, CFDictionary, CFNumber, CFRetained, CFString, CFType};
use objc2_core_media::{
    CMBlockBuffer, CMFormatDescription, CMSampleBuffer, CMSampleTimingInfo, CMTime, CMTimeFlags,
    CMVideoCodecType, CMVideoFormatDescriptionCreateFromH264ParameterSets,
    CMVideoFormatDescriptionCreateFromHEVCParameterSets, kCMTimeInvalid, kCMVideoCodecType_H264,
    kCMVideoCodecType_HEVC,
};
use objc2_core_video::{
    CVColorPrimariesGetIntegerCodePointForString, CVImageBuffer,
    CVPixelBufferGetBaseAddressOfPlane, CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferGetHeight,
    CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
    CVPixelBufferUnlockBaseAddress, CVTransferFunctionGetIntegerCodePointForString,
    CVYCbCrMatrixGetIntegerCodePointForString, kCVImageBufferColorPrimariesKey,
    kCVImageBufferTransferFunctionKey, kCVImageBufferYCbCrMatrixKey,
    kCVPixelBufferPixelFormatTypeKey, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
};
use objc2_video_toolbox::{
    VTDecodeFrameFlags, VTDecodeInfoFlags, VTDecompressionOutputCallbackRecord,
    VTDecompressionSession, VTIsHardwareDecodeSupported, kVTCouldNotFindVideoDecoderErr,
    kVTVideoDecoderBadDataErr, kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder,
    kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder,
    kVTVideoDecoderUnsupportedDataFormatErr,
};
use tracing::{debug, warn};

use super::os_status;
use crate::nv12::{self, OutputFormat};

const BACKEND: BackendId = BackendId::VideoToolbox;
/// Timestamps cross VideoToolbox as microseconds.
const MICROS: i32 = 1_000_000;

/// Adds the VideoToolbox H.264 / H.265 decoders.
pub(crate) fn video(registry: VideoDecoderRegistry) -> VideoDecoderRegistry {
    if !matches!(
        registry.target().codec,
        CodecKind::Video(Codec::H264 | Codec::H265)
    ) {
        return registry;
    }
    let registry = registry.candidate(Candidate::new(
        BACKEND,
        Acceleration::Hardware,
        |config: &DecoderConfig| {
            VtDecoder::open(config, Acceleration::Hardware).map(|decoder| Box::new(decoder) as _)
        },
    ));
    // Apple's software H.264 / HEVC decoders back the hardware ones on
    // macOS; iOS decodes in the media engine.
    #[cfg(target_os = "macos")]
    let registry = registry.candidate(Candidate::new(
        BACKEND,
        Acceleration::Software,
        |config: &DecoderConfig| {
            VtDecoder::open(config, Acceleration::Software).map(|decoder| Box::new(decoder) as _)
        },
    ));
    registry
}

fn fail(operation: Operation, what: impl std::fmt::Display) -> CodecError {
    CodecError::backend(BACKEND, operation, what)
}

/// A decode status as a codec error: data the decoder cannot handle is
/// a stream it does not support (another backend may), anything else a
/// failure.
fn decode_error(status: i32, what: &str) -> CodecError {
    if status == kVTVideoDecoderUnsupportedDataFormatErr || status == kVTCouldNotFindVideoDecoderErr
    {
        CodecError::UnsupportedStream {
            backend: BACKEND,
            reason: format!("{what}: {}", os_status(status)),
        }
    } else if status == kVTVideoDecoderBadDataErr {
        fail(
            Operation::Decode,
            format!("{what}: the bitstream is corrupt ({})", os_status(status)),
        )
    } else {
        fail(Operation::Decode, format!("{what}: {}", os_status(status)))
    }
}

fn codec_type(codec: Codec) -> CodecResult<CMVideoCodecType> {
    match codec {
        Codec::H264 => Ok(kCMVideoCodecType_H264),
        Codec::H265 => Ok(kCMVideoCodecType_HEVC),
        other => Err(CodecError::BackendCannotHandle {
            backend: BACKEND,
            target: CodecTarget::decode(other),
        }),
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

/// Where a NAL unit goes: a parameter-set slot, dropped, or decoded.
enum Nal {
    ParameterSet(usize),
    Skip,
    Slice,
}

fn classify(codec: Codec, nal: &[u8]) -> Nal {
    let Some(&header) = nal.first() else {
        return Nal::Skip;
    };
    match codec {
        Codec::H265 => match (header >> 1) & 0x3f {
            32 => Nal::ParameterSet(0),
            33 => Nal::ParameterSet(1),
            34 => Nal::ParameterSet(2),
            // Access unit delimiter, end of sequence / bitstream.
            35..=37 => Nal::Skip,
            _ => Nal::Slice,
        },
        _ => match header & 0x1f {
            7 => Nal::ParameterSet(0),
            8 => Nal::ParameterSet(1),
            9..=11 => Nal::Skip,
            _ => Nal::Slice,
        },
    }
}

/// A picture the callback copied out, still NV12.
struct Picture {
    data: Vec<u8>,
    width: usize,
    height: usize,
    timestamp_us: u64,
    colour: Option<ColourDescription>,
}

enum Output {
    Picture(Picture),
    Failed(i32),
}

type Outbox = Mutex<Vec<Output>>;

fn lock(outbox: &Outbox) -> MutexGuard<'_, Vec<Output>> {
    outbox.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The colour attachments of a decoded picture, when it has them.
fn attached_colour(image: &CVImageBuffer) -> Option<ColourDescription> {
    // SAFETY: CoreVideo exports these keys for the life of the process.
    let keys = unsafe {
        [
            kCVImageBufferColorPrimariesKey,
            kCVImageBufferTransferFunctionKey,
            kCVImageBufferYCbCrMatrixKey,
        ]
    };
    let read = |key: &CFString| {
        // SAFETY: `image` is live; the attachment mode out pointer may be null.
        let value = unsafe { image.attachment(key, ptr::null_mut()) }?;
        value.downcast::<CFString>().ok()
    };
    let [primaries, transfer, matrix] = keys.map(read);
    if primaries.is_none() && transfer.is_none() && matrix.is_none() {
        return None;
    }
    let code = |value: i32| u8::try_from(value).unwrap_or(2);
    // VideoToolbox hands out video-range NV12 (requested below).
    Some(ColourDescription::from_codes(
        code(CVColorPrimariesGetIntegerCodePointForString(
            primaries.as_deref(),
        )),
        code(CVTransferFunctionGetIntegerCodePointForString(
            transfer.as_deref(),
        )),
        code(CVYCbCrMatrixGetIntegerCodePointForString(matrix.as_deref())),
        false,
    ))
}

/// Copies a locked NV12 picture into a tightly packed buffer.
fn copy_nv12(image: &CVImageBuffer) -> Option<(Vec<u8>, usize, usize)> {
    let width = CVPixelBufferGetWidth(image);
    let height = CVPixelBufferGetHeight(image);
    let mut out = Vec::with_capacity(width * height * 3 / 2);
    for (plane, rows) in [(0, height), (1, height.div_ceil(2))] {
        let base = CVPixelBufferGetBaseAddressOfPlane(image, plane).cast::<u8>();
        let pitch = CVPixelBufferGetBytesPerRowOfPlane(image, plane);
        let row_bytes = if plane == 0 {
            width
        } else {
            width.div_ceil(2) * 2
        };
        if base.is_null() || pitch < row_bytes {
            return None;
        }
        for row in 0..rows {
            // SAFETY: the buffer is locked; plane `plane` holds `rows` rows
            // of `pitch >= row_bytes` bytes.
            let line = unsafe { std::slice::from_raw_parts(base.add(row * pitch), row_bytes) };
            out.extend_from_slice(line);
        }
    }
    Some((out, width, height))
}

unsafe extern "C-unwind" fn on_decoded(
    refcon: *mut c_void,
    _source: *mut c_void,
    status: i32,
    _flags: VTDecodeInfoFlags,
    image: *mut CVImageBuffer,
    pts: CMTime,
    _duration: CMTime,
) {
    // SAFETY: `refcon` is the `Outbox` the session was created with; it
    // outlives the session (`Session::drop` invalidates first).
    let outbox = unsafe { &*refcon.cast::<Outbox>() };
    if status != 0 {
        lock(outbox).push(Output::Failed(status));
        return;
    }
    // A null image is a frame the decoder dropped on purpose.
    let Some(image) = NonNull::new(image) else {
        return;
    };
    // SAFETY: VideoToolbox passes a live image buffer for this call.
    let image = unsafe { image.as_ref() };
    // SAFETY: the buffer is live and unlocked; read-only lock.
    if unsafe { CVPixelBufferLockBaseAddress(image, CVPixelBufferLockFlags::ReadOnly) } != 0 {
        lock(outbox).push(Output::Failed(kVTVideoDecoderBadDataErr));
        return;
    }
    let copied = copy_nv12(image);
    // SAFETY: locked just above with the same flags.
    unsafe { CVPixelBufferUnlockBaseAddress(image, CVPixelBufferLockFlags::ReadOnly) };
    let output = match copied {
        Some((data, width, height)) => Output::Picture(Picture {
            data,
            width,
            height,
            timestamp_us: u64::try_from(pts.value).unwrap_or(0),
            colour: attached_colour(image),
        }),
        None => Output::Failed(kVTVideoDecoderBadDataErr),
    };
    lock(outbox).push(output);
}

/// One decompression session and the format description it decodes.
struct Session {
    session: CFRetained<VTDecompressionSession>,
    description: CFRetained<CMFormatDescription>,
    /// The callback's reference value; boxed so its address outlives the
    /// session (freed after `invalidate`).
    outbox: Box<Outbox>,
}

// SAFETY: a VideoToolbox session may be driven from any thread as long as
// calls do not overlap; it is only reached through `&mut VtDecoder`, and
// the callback only touches the `Mutex` outbox.
unsafe impl Send for Session {}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: the session is live; invalidating stops callbacks before
        // `outbox` is freed.
        unsafe { self.session.invalidate() };
    }
}

/// A format description from Annex-B-free parameter sets.
fn format_description(
    codec: Codec,
    sets: &[Vec<u8>],
) -> CodecResult<CFRetained<CMFormatDescription>> {
    let mut pointers: Vec<NonNull<u8>> = sets
        .iter()
        .map(|set| {
            NonNull::new(set.as_ptr().cast_mut())
                .ok_or_else(|| fail(Operation::Decode, "an empty parameter set"))
        })
        .collect::<CodecResult<_>>()?;
    let mut sizes: Vec<usize> = sets.iter().map(Vec::len).collect();
    let mut raw: *const CMFormatDescription = ptr::null();
    let pointers_ptr = NonNull::new(pointers.as_mut_ptr())
        .ok_or_else(|| fail(Operation::Decode, "no parameter sets"))?;
    let sizes_ptr = NonNull::new(sizes.as_mut_ptr())
        .ok_or_else(|| fail(Operation::Decode, "no parameter sets"))?;
    // SAFETY: `pointers` / `sizes` describe `sets.len()` live parameter
    // sets; the out pointer is a valid local. 4-byte NAL lengths match
    // what `decode` writes.
    let status = unsafe {
        match codec {
            Codec::H265 => CMVideoFormatDescriptionCreateFromHEVCParameterSets(
                None,
                sets.len(),
                pointers_ptr,
                sizes_ptr,
                4,
                None,
                NonNull::from(&mut raw),
            ),
            _ => CMVideoFormatDescriptionCreateFromH264ParameterSets(
                None,
                sets.len(),
                pointers_ptr,
                sizes_ptr,
                4,
                NonNull::from(&mut raw),
            ),
        }
    };
    if status != 0 {
        return Err(CodecError::UnsupportedStream {
            backend: BACKEND,
            reason: format!("its parameter sets are not accepted: {}", os_status(status)),
        });
    }
    let raw = NonNull::new(raw.cast_mut())
        .ok_or_else(|| fail(Operation::Decode, "no format description was created"))?;
    // SAFETY: the create call handed over a +1 reference.
    Ok(unsafe { CFRetained::from_raw(raw) })
}

impl Session {
    fn open(
        description: CFRetained<CMFormatDescription>,
        acceleration: Acceleration,
    ) -> CodecResult<Self> {
        // SAFETY: VideoToolbox / CoreVideo export these keys for the life
        // of the process.
        let (require, enable, format_key) = unsafe {
            (
                kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder,
                kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder,
                kCVPixelBufferPixelFormatTypeKey,
            )
        };
        let specification = match acceleration {
            Acceleration::Hardware => CFDictionary::<CFString, CFType>::from_slices(
                &[require],
                &[CFBoolean::new(true).as_ref()],
            ),
            Acceleration::Software => CFDictionary::<CFString, CFType>::from_slices(
                &[enable],
                &[CFBoolean::new(false).as_ref()],
            ),
        };
        let pixel_format = CFNumber::new_i32(
            i32::try_from(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange).unwrap_or(i32::MAX),
        );
        let attributes =
            CFDictionary::<CFString, CFType>::from_slices(&[format_key], &[pixel_format.as_ref()]);
        let outbox: Box<Outbox> = Box::new(Mutex::new(Vec::new()));
        let record = VTDecompressionOutputCallbackRecord {
            decompressionOutputCallback: Some(on_decoded),
            decompressionOutputRefCon: ptr::from_ref::<Outbox>(&outbox).cast_mut().cast(),
        };
        let mut raw: *mut VTDecompressionSession = ptr::null_mut();
        // SAFETY: every pointer is valid for the call; `on_decoded` matches
        // the callback prototype and `outbox` outlives the session.
        let status = unsafe {
            VTDecompressionSession::create(
                None,
                &description,
                Some(specification.as_ref()),
                Some(attributes.as_ref()),
                &raw const record,
                NonNull::from(&mut raw),
            )
        };
        if status != 0 {
            return Err(decode_error(status, "VTDecompressionSessionCreate"));
        }
        let session = NonNull::new(raw).ok_or_else(|| {
            fail(
                Operation::Decode,
                "VTDecompressionSessionCreate returned no session",
            )
        })?;
        // SAFETY: the create call handed over a +1 reference.
        let session = unsafe { CFRetained::from_raw(session) };
        Ok(Self {
            session,
            description,
            outbox,
        })
    }
}

/// A VideoToolbox H.264 / H.265 decoder.
pub(crate) struct VtDecoder {
    acceleration: Acceleration,
    codec: Codec,
    format: OutputFormat,
    low_latency: bool,
    container_colour: Option<ColourDescription>,
    /// The latest parameter sets, by slot (SPS, PPS; VPS, SPS, PPS).
    sets: Vec<Option<Vec<u8>>>,
    /// The parameter sets the open session's description was built from.
    session_sets: Vec<Vec<u8>>,
    session: Option<Session>,
    ready: VecDeque<CapturedFrame>,
}

impl VtDecoder {
    fn open(config: &DecoderConfig, acceleration: Acceleration) -> CodecResult<Self> {
        let codec_type = codec_type(config.codec)?;
        // SAFETY: a plain capability query.
        if acceleration == Acceleration::Hardware
            && !unsafe { VTIsHardwareDecodeSupported(codec_type) }
        {
            return Err(CodecError::BackendUnavailable {
                backend: BACKEND,
                reason: format!("this device has no hardware {} decoder", config.codec),
            });
        }
        let mut decoder = Self {
            acceleration,
            codec: config.codec,
            format: OutputFormat::default(),
            low_latency: config.low_latency,
            container_colour: config.colour,
            sets: Vec::new(),
            session_sets: Vec::new(),
            session: None,
            ready: VecDeque::new(),
        };
        decoder.configure(config)?;
        Ok(decoder)
    }

    /// Makes sure a session decodes the current parameter sets; `false`
    /// while some are still missing.
    fn ensure_session(&mut self) -> CodecResult<bool> {
        let Some(sets) = self.sets.iter().cloned().collect::<Option<Vec<_>>>() else {
            return Ok(false);
        };
        if self.session.is_some() && self.session_sets == sets {
            return Ok(true);
        }
        let description = format_description(self.codec, &sets)?;
        if let Some(session) = &mut self.session {
            // SAFETY: both objects are live.
            if unsafe { session.session.can_accept_format_description(&description) } {
                session.description = description;
                self.session_sets = sets;
                return Ok(true);
            }
            self.drain(Operation::Decode)?;
            self.session = None;
        }
        self.session = Some(Session::open(description, self.acceleration)?);
        self.session_sets = sets;
        Ok(true)
    }

    /// Hands out what the callback delivered.
    fn collect(&mut self) -> CodecResult<()> {
        let Some(session) = &self.session else {
            return Ok(());
        };
        let outputs = std::mem::take(&mut *lock(&session.outbox));
        for output in outputs {
            match output {
                Output::Picture(picture) => {
                    let colour = picture
                        .colour
                        .or(self.container_colour)
                        .unwrap_or(ColourDescription::unspecified(ColourRange::Limited))
                        // The session converts to video-range NV12 whatever the
                        // stream used.
                        .with_range(ColourRange::Limited);
                    let frame = nv12::deliver(
                        Bytes::from(picture.data),
                        picture.width,
                        (picture.width, picture.height),
                        picture.timestamp_us,
                        self.format.0,
                        colour,
                    )
                    .ok_or_else(|| fail(Operation::Decode, "converting a picture failed"))?;
                    self.ready.push_back(CapturedFrame::Cpu(frame));
                }
                Output::Failed(status) => return Err(decode_error(status, "decoding a frame")),
            }
        }
        Ok(())
    }

    /// Waits for every frame the session still holds.
    fn drain(&mut self, operation: Operation) -> CodecResult<()> {
        if let Some(session) = &self.session {
            // SAFETY: the session is live.
            let status = unsafe { session.session.finish_delayed_frames() };
            if status != 0 {
                return Err(fail(
                    operation,
                    format!("finishing delayed frames: {}", os_status(status)),
                ));
            }
            // SAFETY: as above.
            let status = unsafe { session.session.wait_for_asynchronous_frames() };
            if status != 0 {
                return Err(fail(
                    operation,
                    format!("waiting for frames: {}", os_status(status)),
                ));
            }
        }
        self.collect()
    }

    /// One sample buffer holding `nals` length-prefixed.
    fn sample(&self, nals: &[&[u8]], timestamp_us: u64) -> CodecResult<CFRetained<CMSampleBuffer>> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| fail(Operation::Decode, "no session"))?;
        let mut avcc = Vec::with_capacity(nals.iter().map(|nal| nal.len() + 4).sum());
        for nal in nals {
            let len = u32::try_from(nal.len())
                .map_err(|_| fail(Operation::Decode, "NAL unit too long"))?;
            avcc.extend_from_slice(&len.to_be_bytes());
            avcc.extend_from_slice(nal);
        }
        let mut block: *mut CMBlockBuffer = ptr::null_mut();
        // SAFETY: a null memory block with the default allocator makes
        // CoreMedia allocate `avcc.len()` bytes; the out pointer is valid.
        let status = unsafe {
            CMBlockBuffer::create_with_memory_block(
                None,
                ptr::null_mut(),
                avcc.len(),
                None,
                ptr::null(),
                0,
                avcc.len(),
                0,
                NonNull::from(&mut block),
            )
        };
        if status != 0 {
            return Err(fail(
                Operation::Decode,
                format!("CMBlockBufferCreateWithMemoryBlock: {}", os_status(status)),
            ));
        }
        let block =
            NonNull::new(block).ok_or_else(|| fail(Operation::Decode, "no block buffer"))?;
        // SAFETY: the create call handed over a +1 reference.
        let block = unsafe { CFRetained::from_raw(block) };
        let source = NonNull::new(avcc.as_mut_ptr().cast::<c_void>())
            .ok_or_else(|| fail(Operation::Decode, "an empty access unit"))?;
        // SAFETY: `block` was allocated with exactly `avcc.len()` bytes.
        let status = unsafe { CMBlockBuffer::replace_data_bytes(source, &block, 0, avcc.len()) };
        if status != 0 {
            return Err(fail(
                Operation::Decode,
                format!("CMBlockBufferReplaceDataBytes: {}", os_status(status)),
            ));
        }
        let pts = i64::try_from(timestamp_us)
            .map_err(|_| fail(Operation::Decode, "timestamp exceeds i64"))?;
        // SAFETY: reading an exported constant.
        let invalid = unsafe { kCMTimeInvalid };
        let timing = CMSampleTimingInfo {
            duration: invalid,
            presentationTimeStamp: cm_time(pts),
            decodeTimeStamp: invalid,
        };
        let size = avcc.len();
        let mut sample: *mut CMSampleBuffer = ptr::null_mut();
        // SAFETY: the block and description are live; `timing` and `size`
        // describe the one sample; the out pointer is valid.
        let status = unsafe {
            CMSampleBuffer::create_ready(
                None,
                Some(&block),
                Some(&session.description),
                1,
                1,
                &raw const timing,
                1,
                &raw const size,
                NonNull::from(&mut sample),
            )
        };
        if status != 0 {
            return Err(fail(
                Operation::Decode,
                format!("CMSampleBufferCreateReady: {}", os_status(status)),
            ));
        }
        let sample =
            NonNull::new(sample).ok_or_else(|| fail(Operation::Decode, "no sample buffer"))?;
        // SAFETY: the create call handed over a +1 reference.
        Ok(unsafe { CFRetained::from_raw(sample) })
    }
}

impl VideoDecoder for VtDecoder {
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, config: &DecoderConfig) -> CodecResult<()> {
        codec_type(config.codec)?;
        if config.codec != self.codec {
            self.session = None;
            self.session_sets.clear();
            self.codec = config.codec;
        }
        self.sets = vec![None; if self.codec == Codec::H265 { 3 } else { 2 }];
        self.format = OutputFormat(config.pixel_format);
        self.low_latency = config.low_latency;
        self.container_colour = config.colour;
        Ok(())
    }

    fn decode(&mut self, frame: EncodedFrame) -> CodecResult<Option<CapturedFrame>> {
        let mut slices = Vec::new();
        for nal in split_annex_b(&frame.data) {
            match classify(self.codec, nal) {
                Nal::ParameterSet(slot) => {
                    if let Some(entry) = self.sets.get_mut(slot) {
                        *entry = Some(nal.to_vec());
                    }
                }
                Nal::Skip => {}
                Nal::Slice => slices.push(nal),
            }
        }
        if slices.is_empty() {
            return Ok(self.ready.pop_front());
        }
        if !self.ensure_session()? {
            debug!(codec = %self.codec, "VideoToolbox: slice before the parameter sets; skipped");
            return Ok(self.ready.pop_front());
        }
        let sample = self.sample(&slices, frame.timestamp_us)?;
        let flags = if self.low_latency {
            VTDecodeFrameFlags(0)
        } else {
            VTDecodeFrameFlags::Frame_EnableTemporalProcessing
        };
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| fail(Operation::Decode, "no session"))?;
        let mut info = VTDecodeInfoFlags(0);
        // SAFETY: the session and sample are live; the info out pointer is
        // a valid local; no per-frame reference value is used.
        let status = unsafe {
            session
                .session
                .decode_frame(&sample, flags, ptr::null_mut(), &raw mut info)
        };
        if status != 0 {
            return Err(decode_error(status, "VTDecompressionSessionDecodeFrame"));
        }
        if info.contains(VTDecodeInfoFlags::FrameDropped) {
            warn!(
                timestamp_us = frame.timestamp_us,
                "VideoToolbox dropped a frame"
            );
        }
        self.collect()?;
        Ok(self.ready.pop_front())
    }

    fn flush(&mut self) -> CodecResult<Vec<CapturedFrame>> {
        self.drain(Operation::Flush)?;
        Ok(self.ready.drain(..).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nal_units_are_sorted_into_parameter_sets_and_slices() {
        assert!(matches!(
            classify(Codec::H264, &[0x67]),
            Nal::ParameterSet(0)
        ));
        assert!(matches!(
            classify(Codec::H264, &[0x68]),
            Nal::ParameterSet(1)
        ));
        assert!(matches!(classify(Codec::H264, &[0x09]), Nal::Skip));
        assert!(matches!(classify(Codec::H264, &[0x65]), Nal::Slice));
        assert!(matches!(
            classify(Codec::H265, &[0x40]),
            Nal::ParameterSet(0)
        ));
        assert!(matches!(
            classify(Codec::H265, &[0x44]),
            Nal::ParameterSet(2)
        ));
        assert!(matches!(classify(Codec::H265, &[0x46]), Nal::Skip));
        assert!(matches!(classify(Codec::H265, &[0x26]), Nal::Slice));
    }
}
