//! H.264 / H.265 / VP9 / AV1 decode through VideoToolbox.
//!
//! [`stream`] turns packets into samples and the format description they
//! decode against; the session is recreated when it cannot take a new
//! description, or when the bit depth changes (8-bit streams decode to
//! NV12, deeper ones to P010). Samples are stamped with the packet
//! timestamp; pictures come back through a callback carrying it. Outside
//! low-latency mode temporal processing is on, so VideoToolbox emits
//! them in display order.
//!
//! With [`FrameMemory::Native`] each picture is handed out as the
//! IOSurface-backed `CVPixelBuffer` VideoToolbox decoded into
//! ([`ferricast_codec::apple::PixelBuffer`]), with no copy; otherwise it
//! is copied out in the configured pixel format.
//!
//! VP9 uses Apple's supplemental decoder, registered on demand (macOS
//! 11+); AV1 needs a hardware decoder (Apple M3 / A17 Pro and later).

mod stream;

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use bytes::Bytes;
use ferricast_codec::apple::PixelBuffer;
use ferricast_codec::{Candidate, VideoDecoderRegistry};
use ferricast_core::{
    Acceleration, BackendId, CapturedFrame, Codec, CodecError, CodecKind, CodecResult,
    ColourDescription, DecoderConfig, EncodedFrame, FrameMemory, HdrMetadata, NativeFrame,
    NativeSurface, Operation, PixelFormat, RawFrame, VideoDecoder,
};
use objc2_core_foundation::{CFBoolean, CFDictionary, CFNumber, CFRetained, CFString, CFType};
use objc2_core_media::{
    CMBlockBuffer, CMFormatDescription, CMSampleBuffer, CMSampleTimingInfo, CMTime, CMTimeFlags,
    CMVideoCodecType, kCMTimeInvalid,
};
use objc2_core_video::{
    CVImageBuffer, kCVPixelBufferIOSurfacePropertiesKey, kCVPixelBufferMetalCompatibilityKey,
    kCVPixelBufferPixelFormatTypeKey, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
    kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
};
use objc2_video_toolbox::{
    VTDecodeFrameFlags, VTDecodeInfoFlags, VTDecompressionOutputCallbackRecord,
    VTDecompressionSession, VTIsHardwareDecodeSupported, kVTCouldNotFindVideoDecoderErr,
    kVTVideoDecoderBadDataErr, kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder,
    kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder,
    kVTVideoDecoderUnsupportedDataFormatErr,
};
use tracing::warn;

use self::stream::{Stream, codec_type};
use super::os_status;
use crate::nv12::{self, OutputFormat};

const BACKEND: BackendId = BackendId::VideoToolbox;
/// Timestamps cross VideoToolbox as microseconds.
const MICROS: i32 = 1_000_000;

/// What a candidate opens.
type Opened = Box<dyn ferricast_codec::VideoDecoderSession>;

/// Adds the VideoToolbox decoders of the registry's codec.
pub(crate) fn video(registry: VideoDecoderRegistry) -> VideoDecoderRegistry {
    let CodecKind::Video(codec) = registry.target().codec else {
        return registry;
    };
    match codec {
        Codec::H264 | Codec::H265 => {
            let registry = registry.candidate(Candidate::new(
                BACKEND,
                Acceleration::Hardware,
                open_hardware,
            ));
            // Apple's software H.264 / HEVC decoders back the hardware
            // ones on macOS; iOS decodes in the media engine.
            #[cfg(target_os = "macos")]
            let registry = registry.candidate(Candidate::new(
                BACKEND,
                Acceleration::Software,
                open_software,
            ));
            registry
        }
        // Apple's VP9 decoder is a supplemental one, registered on
        // request; the call exists on macOS only.
        #[cfg(target_os = "macos")]
        Codec::Vp9 => registry
            .candidate(
                Candidate::new(BACKEND, Acceleration::Hardware, open_hardware)
                    .with_probe(|| hardware_supported(Codec::Vp9)),
            )
            .candidate(Candidate::new(
                BACKEND,
                Acceleration::Software,
                open_software,
            )),
        Codec::Av1 => registry.candidate(
            Candidate::new(BACKEND, Acceleration::Hardware, open_hardware)
                .with_probe(|| hardware_supported(Codec::Av1)),
        ),
        _ => registry,
    }
}

fn open_hardware(config: &DecoderConfig) -> CodecResult<Opened> {
    VtDecoder::open(config, Acceleration::Hardware).map(|decoder| Box::new(decoder) as _)
}

#[cfg_attr(
    not(target_os = "macos"),
    allow(
        dead_code,
        reason = "conditional: the software decoders exist on macOS only"
    )
)]
fn open_software(config: &DecoderConfig) -> CodecResult<Opened> {
    VtDecoder::open(config, Acceleration::Software).map(|decoder| Box::new(decoder) as _)
}

/// Registers Apple's supplemental decoder of `codec_type` when the system
/// has one that is not registered by default (VP9); a no-op elsewhere.
#[cfg(target_os = "macos")]
fn register_supplemental(codec_type: CMVideoCodecType) {
    // SAFETY: a plain registration call, idempotent.
    unsafe { objc2_video_toolbox::VTRegisterSupplementalVideoDecoderIfAvailable(codec_type) };
}

#[cfg(not(target_os = "macos"))]
fn register_supplemental(_codec_type: CMVideoCodecType) {}

/// Whether this device decodes `codec` in hardware.
fn hardware_supported(codec: Codec) -> CodecResult<()> {
    let codec_type = codec_type(codec)?;
    register_supplemental(codec_type);
    // SAFETY: a plain capability query.
    if unsafe { VTIsHardwareDecodeSupported(codec_type) } {
        Ok(())
    } else {
        Err(CodecError::BackendUnavailable {
            backend: BACKEND,
            reason: format!("this device has no hardware {codec} decoder"),
        })
    }
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

fn cm_time(value: i64) -> CMTime {
    CMTime {
        value,
        timescale: MICROS,
        flags: CMTimeFlags::Valid,
        epoch: 0,
    }
}

enum Output {
    Picture {
        buffer: PixelBuffer,
        timestamp_us: u64,
    },
    Failed(CodecError),
}

type Outbox = Mutex<Vec<Output>>;

fn lock(outbox: &Outbox) -> MutexGuard<'_, Vec<Output>> {
    outbox.lock().unwrap_or_else(PoisonError::into_inner)
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
        lock(outbox).push(Output::Failed(decode_error(status, "decoding a frame")));
        return;
    }
    // A null image is a frame the decoder dropped on purpose.
    let Some(image) = NonNull::new(image) else {
        return;
    };
    // SAFETY: VideoToolbox passes a live image buffer for this call; the
    // retain keeps it alive past the callback.
    let image = unsafe { CFRetained::retain(image) };
    let output = match PixelBuffer::new(image) {
        Ok(buffer) => Output::Picture {
            buffer,
            timestamp_us: u64::try_from(pts.value).unwrap_or(0),
        },
        Err(unsupported) => Output::Failed(fail(
            Operation::Decode,
            format!("the decoder produced a picture this crate cannot read: {unsupported}"),
        )),
    };
    lock(outbox).push(output);
}

/// One decompression session.
struct Session {
    session: CFRetained<VTDecompressionSession>,
    /// The generation of the stream format the session decodes.
    generation: u64,
    depth: u8,
    /// The callback's reference value; boxed so its address outlives the
    /// session (freed after `invalidate`).
    outbox: Box<Outbox>,
}

// SAFETY: a VideoToolbox session may be driven from any thread as long as
// calls do not overlap; it is only reached through `&mut VtDecoder`, and
// the callback only touches the `Mutex` outbox, whose pictures are `Send`.
unsafe impl Send for Session {}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: the session is live; invalidating stops callbacks before
        // `outbox` is freed.
        unsafe { self.session.invalidate() };
    }
}

impl Session {
    fn open(
        description: &CMFormatDescription,
        depth: u8,
        generation: u64,
        acceleration: Acceleration,
    ) -> CodecResult<Self> {
        // SAFETY: VideoToolbox / CoreVideo export these keys for the life
        // of the process.
        let (require, enable, format_key, surface_key, metal_key) = unsafe {
            (
                kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder,
                kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder,
                kCVPixelBufferPixelFormatTypeKey,
                kCVPixelBufferIOSurfacePropertiesKey,
                kCVPixelBufferMetalCompatibilityKey,
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
        let os_type = if depth > 8 {
            kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
        } else {
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        };
        let pixel_format = CFNumber::new_i32(i32::try_from(os_type).unwrap_or(i32::MAX));
        // IOSurface-backed, Metal-compatible buffers: what a native frame
        // hands to an encoder or a renderer without a copy.
        let empty = CFDictionary::<CFString, CFType>::empty();
        let attributes = CFDictionary::<CFString, CFType>::from_slices(
            &[format_key, surface_key, metal_key],
            &[
                pixel_format.as_ref(),
                (*empty).as_ref(),
                CFBoolean::new(true).as_ref(),
            ],
        );
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
                description,
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
            generation,
            depth,
            outbox,
        })
    }
}

/// A VideoToolbox decoder.
pub(crate) struct VtDecoder {
    acceleration: Acceleration,
    stream: Stream,
    format: OutputFormat,
    memory: FrameMemory,
    low_latency: bool,
    container_colour: Option<ColourDescription>,
    container_hdr: Option<HdrMetadata>,
    session: Option<Session>,
    ready: VecDeque<CapturedFrame>,
}

impl VtDecoder {
    fn open(config: &DecoderConfig, acceleration: Acceleration) -> CodecResult<Self> {
        register_supplemental(codec_type(config.codec)?);
        if acceleration == Acceleration::Hardware {
            hardware_supported(config.codec)?;
        }
        let mut decoder = Self {
            acceleration,
            stream: Stream::new(config.codec)?,
            format: OutputFormat::default(),
            memory: config.memory,
            low_latency: config.low_latency,
            container_colour: config.colour,
            container_hdr: config.hdr,
            session: None,
            ready: VecDeque::new(),
        };
        decoder.configure(config)?;
        Ok(decoder)
    }

    /// Makes sure a session decodes the stream's current format.
    fn ensure_session(&mut self) -> CodecResult<()> {
        let format = self
            .stream
            .format()
            .ok_or_else(|| fail(Operation::Decode, "no format description"))?;
        if let Some(session) = &mut self.session {
            if session.generation == format.generation {
                return Ok(());
            }
            // SAFETY: both objects are live.
            let accepts = unsafe {
                session
                    .session
                    .can_accept_format_description(&format.description)
            };
            if session.depth == format.depth && accepts {
                session.generation = format.generation;
                return Ok(());
            }
        }
        let (depth, generation) = (format.depth, format.generation);
        let description = format.description.clone();
        if self.session.is_some() {
            self.drain(Operation::Decode)?;
            self.session = None;
        }
        self.session = Some(Session::open(
            &description,
            depth,
            generation,
            self.acceleration,
        )?);
        Ok(())
    }

    /// Hands out what the callback delivered.
    fn collect(&mut self) -> CodecResult<()> {
        let Some(session) = &self.session else {
            return Ok(());
        };
        let outputs = std::mem::take(&mut *lock(&session.outbox));
        for output in outputs {
            match output {
                Output::Picture {
                    buffer,
                    timestamp_us,
                } => {
                    let frame = self.picture(buffer, timestamp_us)?;
                    self.ready.push_back(frame);
                }
                Output::Failed(error) => return Err(error),
            }
        }
        Ok(())
    }

    /// One decoded picture in the configured frame memory and format.
    fn picture(&self, buffer: PixelBuffer, timestamp_us: u64) -> CodecResult<CapturedFrame> {
        let range = buffer.format().range;
        let colour = buffer
            .colour()
            .or(self.container_colour)
            .unwrap_or(ColourDescription::unspecified(range))
            // The session outputs video range whatever the stream used.
            .with_range(range);
        let hdr = buffer.hdr().or(self.container_hdr);
        if self.memory == FrameMemory::Native {
            return Ok(CapturedFrame::Native(NativeFrame {
                width: buffer.width(),
                height: buffer.height(),
                timestamp_us,
                colour: Some(colour),
                hdr,
                surface: Arc::new(buffer),
            }));
        }
        let (width, height) = (buffer.width() as usize, buffer.height() as usize);
        let pixels = buffer
            .readback()
            .map_err(|error| fail(Operation::Decode, error))?;
        let stride = pixels.stride as usize;
        let converted = match (buffer.format().format, self.format.0) {
            // Planar requests get the picture's own layout.
            (PixelFormat::P010, PixelFormat::P010 | PixelFormat::Nv12 | PixelFormat::I420) => {
                Some(RawFrame {
                    width: buffer.width(),
                    height: buffer.height(),
                    stride: pixels.stride,
                    format: PixelFormat::P010,
                    data: pixels.data,
                    timestamp_us,
                    colour: Some(colour),
                    hdr,
                })
            }
            (PixelFormat::P010, requested @ (PixelFormat::Bgra | PixelFormat::Rgba)) => {
                nv12::deliver(
                    Bytes::from(p010_to_nv12(&pixels.data)),
                    stride / 2,
                    (width, height),
                    timestamp_us,
                    requested,
                    colour,
                )
            }
            (_, requested) => nv12::deliver(
                pixels.data,
                stride,
                (width, height),
                timestamp_us,
                requested,
                colour,
            ),
        };
        let mut frame =
            converted.ok_or_else(|| fail(Operation::Decode, "converting a picture failed"))?;
        frame.hdr = hdr;
        Ok(CapturedFrame::Cpu(frame))
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
}

/// One sample buffer holding `data` against `description`.
fn sample_buffer(
    data: &[u8],
    description: &CMFormatDescription,
    timestamp_us: u64,
) -> CodecResult<CFRetained<CMSampleBuffer>> {
    let mut block: *mut CMBlockBuffer = ptr::null_mut();
    // SAFETY: a null memory block with the default allocator makes
    // CoreMedia allocate `data.len()` bytes; the out pointer is valid.
    let status = unsafe {
        CMBlockBuffer::create_with_memory_block(
            None,
            ptr::null_mut(),
            data.len(),
            None,
            ptr::null(),
            0,
            data.len(),
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
    let block = NonNull::new(block).ok_or_else(|| fail(Operation::Decode, "no block buffer"))?;
    // SAFETY: the create call handed over a +1 reference.
    let block = unsafe { CFRetained::from_raw(block) };
    let source = NonNull::new(data.as_ptr().cast_mut().cast::<c_void>())
        .ok_or_else(|| fail(Operation::Decode, "an empty sample"))?;
    // SAFETY: `block` was allocated with exactly `data.len()` bytes; the
    // source is only read.
    let status = unsafe { CMBlockBuffer::replace_data_bytes(source, &block, 0, data.len()) };
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
    let size = data.len();
    let mut sample: *mut CMSampleBuffer = ptr::null_mut();
    // SAFETY: the block and description are live; `timing` and `size`
    // describe the one sample; the out pointer is valid.
    let status = unsafe {
        CMSampleBuffer::create_ready(
            None,
            Some(&block),
            Some(description),
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
    let sample = NonNull::new(sample).ok_or_else(|| fail(Operation::Decode, "no sample buffer"))?;
    // SAFETY: the create call handed over a +1 reference.
    Ok(unsafe { CFRetained::from_raw(sample) })
}

/// P010 (little-endian 16-bit samples, value in the high 10 bits) as NV12
/// of the same layout at half the stride: the high byte of each sample.
fn p010_to_nv12(p010: &[u8]) -> Vec<u8> {
    p010.chunks_exact(2).map(|sample| sample[1]).collect()
}

impl VideoDecoder for VtDecoder {
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, config: &DecoderConfig) -> CodecResult<()> {
        codec_type(config.codec)?;
        // The same codec keeps its stream state and session: a reconfigure
        // mid-stream must not drop pictures until the next in-band
        // configuration.
        if config.codec != self.stream.codec() {
            self.session = None;
            self.stream = Stream::new(config.codec)?;
        }
        self.format = OutputFormat(config.pixel_format);
        self.memory = config.memory;
        self.low_latency = config.low_latency;
        self.container_colour = config.colour;
        self.container_hdr = config.hdr;
        Ok(())
    }

    fn decode(&mut self, frame: EncodedFrame) -> CodecResult<Option<CapturedFrame>> {
        let Some(data) = self.stream.sample(&frame.data, self.container_colour)? else {
            return Ok(self.ready.pop_front());
        };
        self.ensure_session()?;
        let description = &self
            .stream
            .format()
            .ok_or_else(|| fail(Operation::Decode, "no format description"))?
            .description;
        let sample = sample_buffer(&data, description, frame.timestamp_us)?;
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
    fn p010_keeps_the_high_byte_of_each_sample() {
        assert_eq!(p010_to_nv12(&[0x40, 0xEB, 0xC0, 0x10]), [0xEB, 0x10]);
    }
}
