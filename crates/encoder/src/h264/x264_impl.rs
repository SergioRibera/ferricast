//! Software H.264 encoder on libx264.
//!
//! Talks to libx264 through the raw `x264-sys` bindings instead of the
//! safe `x264` crate, which hides the knobs this pipeline depends on:
//!
//! * `tune=zerolatency` — no B-frames, no lookahead, no frame-thread
//!   delay. The HLS segmenter assumes PTS == DTS and one output picture
//!   per input picture.
//! * per-picture `i_type`, so [`VideoEncoder::request_keyframe`] can
//!   force an IDR where the segmenter wants a segment boundary.
//! * draining delayed pictures on [`VideoEncoder::flush`].
//! * never reading `pic_out` when libx264 produced no NAL (the safe
//!   crate `assume_init`s it unconditionally).
//!
//! Every output picture carries the capture timestamp of the input
//! picture libx264 says it encoded (matched through `i_pts`), not the
//! timestamp of whatever input happened to be submitted on that call.

use std::collections::VecDeque;
use std::ffi::{CStr, c_int};
use std::mem::MaybeUninit;
use std::ptr::{self, NonNull};

use bytes::Bytes;
use ferricast_codec::bitstream::NalSyntax;
use ferricast_codec::bitstream::hdr::{insert_before_slices, sei_nal};
use ferricast_core::{BackendId, CodecError, CodecResult, Operation};
use ferricast_core::{
    BitDepth, CapturedFrame, Codec, ColourRange, EncodedFrame, EncoderConfig, FerricastError,
    H264Profile, PixelFormat, Result, VideoEncoder,
};
use tracing::warn;
use x264_sys as sys;
use yuv::{YuvChromaSubsampling, YuvPlanarImageMut, bgra_to_yuv420, rgba_to_yuv420};

use crate::frame_view::FrameView;
use crate::h264::utils::extract_sps_pps;
use crate::scale::{FrameScaler, FrameSize};

const X264_PRESET_VAR: &str = "FERRICAST_X264_PRESET";
const X264_TUNE_VAR: &str = "FERRICAST_TUNE_PRESET";

/// Upper bound on libx264 calls while draining in `flush()`, so a
/// misbehaving library can't spin us forever.
const MAX_FLUSH_CALLS: usize = 256;

#[derive(Default)]
pub struct X264Encoder {
    session: Option<Session>,
    /// Monotonic picture counter handed to libx264 as `i_pts`. Restarts
    /// at 0 with every session: a fresh rate-control model that sees a
    /// large first PTS assumes seconds of frames already went by and
    /// starts in catch-up mode (long stretch of low-quality frames).
    frame_count: i64,
    fps: u32,
    sps_pps: Vec<u8>,
    planar: Option<I420Scratch>,
    /// Capture timestamps of pictures libx264 accepted but hasn't
    /// emitted yet, keyed by the `i_pts` they were submitted with.
    in_flight: TimestampQueue,
    /// Set by [`VideoEncoder::request_keyframe`]; cleared once a
    /// picture typed IDR has been accepted by libx264.
    pending_keyframe: bool,
    /// Fits frames whose size differs from the session into it.
    scaler: FrameScaler,
    /// The matrix and range RGB input is converted with, from
    /// `EncoderConfig::colour`.
    conversion: Conversion,
    /// SEI NAL unit carrying `EncoderConfig::hdr`, put in every keyframe.
    hdr_sei: Option<Vec<u8>>,
}

/// How RGB input becomes YCbCr.
#[derive(Debug, Clone, Copy)]
struct Conversion {
    range: yuv::YuvRange,
    matrix: yuv::YuvStandardMatrix,
}

impl Default for Conversion {
    fn default() -> Self {
        Self {
            range: yuv::YuvRange::Limited,
            matrix: yuv::YuvStandardMatrix::Bt709,
        }
    }
}

impl X264Encoder {
    fn try_configure(&mut self, config: &EncoderConfig) -> Result<()> {
        let preset = env_choice(X264_PRESET_VAR, "veryfast", preset_name)?;
        if preset == c"placebo" {
            warn!(
                "Using the placebo preset is not recommended due to its impact on performance; Use a faster preset."
            );
        }
        let tune = env_choice(X264_TUNE_VAR, "none", zero_latency_tune)?;

        let mut session = Session::open(config, preset, tune)?;
        let sps_pps = extract_sps_pps(&session.headers()?);
        if sps_pps.is_empty() {
            return Err(FerricastError::Encoder(
                "x264: no SPS/PPS in headers".into(),
            ));
        }

        self.sps_pps = sps_pps;
        self.fps = config.fps.max(1);
        self.session = Some(session);
        self.frame_count = 0;
        self.in_flight = TimestampQueue::default();
        // Every session opens on an IDR anyway.
        self.pending_keyframe = false;
        // Re-sized lazily on the next encode.
        self.planar = None;
        Ok(())
    }

    fn try_encode(&mut self, frame: CapturedFrame) -> Result<EncodedFrame> {
        let timestamp_us = frame.timestamp_us();
        let raw = frame.into_cpu()?;
        let session = self.session.as_mut().ok_or_else(|| {
            FerricastError::Encoder("x264: encode called before configure()".into())
        })?;
        let raw = self
            .scaler
            .fit(raw, FrameSize::new(session.width, session.height))?;
        // This build is 8-bit: 10-bit input keeps its top eight bits.
        let raw = crate::frame_view::eight_bit(raw, session.width, session.height)?;
        let view = FrameView::new(&raw, session.width, session.height)?;
        let planar = self
            .planar
            .get_or_insert_with(|| I420Scratch::new(view.width(), view.height()));
        planar.fill_from(&view, self.conversion)?;

        let pts = self.frame_count;
        let mut picture = planar.picture(pts, self.pending_keyframe);
        let output = session.encode(Some(&mut picture))?;
        // libx264 accepted the picture: it now owns the IDR request and
        // will hand the timestamp back through `i_pts`.
        self.frame_count += 1;
        self.pending_keyframe = false;
        self.in_flight.push(pts, timestamp_us);

        match output {
            Some(out) => Ok(self.finish(out, timestamp_us)),
            // Only reachable if libx264 buffers despite `zerolatency`;
            // the picture is emitted by a later call or by `flush()`.
            None => Err(FerricastError::Encoding(
                "x264: picture buffered, no output yet".into(),
            )),
        }
    }

    fn try_flush(mut self) -> Result<Vec<EncodedFrame>> {
        let Some(session) = self.session.as_mut() else {
            return Ok(Vec::new());
        };
        let mut drained = Vec::new();
        for _ in 0..MAX_FLUSH_CALLS {
            if session.delayed_frames() <= 0 {
                break;
            }
            if let Some(out) = session.encode(None)? {
                drained.push(out);
            }
        }
        Ok(drained.into_iter().map(|out| self.finish(out, 0)).collect())
    }

    fn try_get_headers(&mut self) -> Result<Vec<u8>> {
        // Re-pull from the live encoder so any internal SPS/PPS
        // regeneration is reflected. Falls back to the cached copy when
        // the encoder isn't built yet or the call fails, so callers that
        // probe headers before the first encode still get something.
        if let Some(session) = self.session.as_mut()
            && let Ok(hdr) = session.headers()
        {
            let fresh = extract_sps_pps(&hdr);
            if !fresh.is_empty() {
                self.sps_pps = fresh;
            }
        }
        Ok(self.sps_pps.clone())
    }

    fn try_set_bitrate_kbps(&mut self, kbps: u32) -> Result<()> {
        match self.session.as_mut() {
            Some(session) => session.set_bitrate(kbps),
            // Not configured yet: `configure()` takes the bitrate.
            None => Ok(()),
        }
    }
}

impl VideoEncoder for X264Encoder {
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, config: &EncoderConfig) -> CodecResult<()> {
        if config.bit_depth != BitDepth::Eight {
            return Err(CodecError::UnsupportedConfig {
                backend: BackendId::X264,
                what: format!(
                    "{}-bit H.264 (this libx264 build is 8-bit)",
                    config.bit_depth.bits()
                ),
            });
        }
        let (range, matrix) = crate::nv12::conversion(config.colour, BackendId::X264)?;
        self.conversion = Conversion { range, matrix };
        self.hdr_sei = config.hdr.and_then(|hdr| sei_nal(hdr, NalSyntax::H264));
        self.try_configure(config).map_err(CodecError::from_backend(
            BackendId::X264,
            Operation::Configure,
        ))
    }

    fn encode(&mut self, frame: CapturedFrame) -> CodecResult<EncodedFrame> {
        self.try_encode(frame)
            .map_err(CodecError::from_backend(BackendId::X264, Operation::Encode))
    }

    fn flush(self) -> CodecResult<Vec<EncodedFrame>> {
        self.try_flush()
            .map_err(CodecError::from_backend(BackendId::X264, Operation::Flush))
    }

    fn get_headers(&mut self) -> CodecResult<Vec<u8>> {
        self.try_get_headers().map_err(CodecError::from_backend(
            BackendId::X264,
            Operation::Headers,
        ))
    }

    fn set_bitrate_kbps(&mut self, kbps: u32) -> CodecResult<()> {
        self.try_set_bitrate_kbps(kbps)
            .map_err(CodecError::from_backend(
                BackendId::X264,
                Operation::SetBitrate,
            ))
    }

    fn request_keyframe(&mut self) {
        self.pending_keyframe = true;
    }
}

impl X264Encoder {
    /// Wrap one libx264 output picture, recovering the capture
    /// timestamp of the input it encodes. `fallback_us` is used only if
    /// libx264 reports a PTS we never submitted.
    fn finish(&mut self, out: Output, fallback_us: u64) -> EncodedFrame {
        let timestamp_us = self.in_flight.take(out.pts).unwrap_or_else(|| {
            warn!(
                pts = out.pts,
                "x264: output PTS has no matching input timestamp"
            );
            fallback_us
        });
        let data = match &self.hdr_sei {
            Some(sei) if out.keyframe => {
                Bytes::from(insert_before_slices(&out.data, sei, NalSyntax::H264))
            }
            _ => out.data,
        };
        EncodedFrame {
            codec: Codec::H264,
            data,
            timestamp_us,
            is_keyframe: out.keyframe,
            duration_us: Some(1_000_000 / u64::from(self.fps.max(1))),
            // libx264 shifts DTS below zero when B-frames delay decode
            // order; `zerolatency` rules that out, but never let a
            // negative value wrap to ~2^64.
            pts_dts: (non_negative(out.pts), non_negative(out.dts)),
        }
    }
}

fn non_negative(v: i64) -> u64 {
    u64::try_from(v).unwrap_or(0)
}

/// Read `var` (default `default`) and map it through `parse`.
fn env_choice(
    var: &str,
    default: &str,
    parse: fn(&str) -> Option<&'static CStr>,
) -> Result<&'static CStr> {
    let value = std::env::var(var).unwrap_or_else(|_| default.to_owned());
    parse(&value).ok_or_else(|| FerricastError::Encoder(format!("x264: invalid {var}={value:?}")))
}

fn preset_name(name: &str) -> Option<&'static CStr> {
    Some(match name {
        "placebo" => c"placebo",
        "veryslow" => c"veryslow",
        "slower" => c"slower",
        "slow" => c"slow",
        "medium" => c"medium",
        "fast" => c"fast",
        "faster" => c"faster",
        "veryfast" => c"veryfast",
        "superfast" => c"superfast",
        "ultrafast" => c"ultrafast",
        _ => return None,
    })
}

/// `zerolatency` is always applied: live streaming can't afford
/// B-frames, lookahead or frame-thread delay. The user's tune stacks on
/// top of it.
fn zero_latency_tune(name: &str) -> Option<&'static CStr> {
    Some(match name {
        "none" => c"zerolatency",
        "film" => c"zerolatency,film",
        "animation" => c"zerolatency,animation",
        "grain" => c"zerolatency,grain",
        "stillimage" => c"zerolatency,stillimage",
        "psnr" => c"zerolatency,psnr",
        "ssim" => c"zerolatency,ssim",
        _ => return None,
    })
}

/// One encoded picture as reported by libx264.
struct Output {
    data: Bytes,
    pts: i64,
    dts: i64,
    keyframe: bool,
}

/// Owned libx264 encoder handle.
struct Session {
    raw: NonNull<sys::x264_t>,
    width: u32,
    height: u32,
}

// SAFETY: libx264 has no thread affinity; an `x264_t` only must not be
// used from two threads at once. `Session` is reachable solely through
// `&mut X264Encoder`, so at most one thread touches it at a time.
unsafe impl Send for Session {}

impl Session {
    fn open(config: &EncoderConfig, preset: &CStr, tune: &CStr) -> Result<Self> {
        let mut params = MaybeUninit::<sys::x264_param_t>::uninit();
        // SAFETY: `params` is a valid out-pointer and both names are
        // NUL-terminated. The call starts with `x264_param_default`,
        // which writes every field, then applies preset and tune.
        let status = unsafe {
            sys::x264_param_default_preset(params.as_mut_ptr(), preset.as_ptr(), tune.as_ptr())
        };
        // SAFETY: fully initialised by `x264_param_default` above, even
        // when the preset/tune lookup afterwards fails.
        let mut params = unsafe { params.assume_init() };
        if status < 0 {
            return Err(FerricastError::Encoder(format!(
                "x264: unknown preset {preset:?} / tune {tune:?}"
            )));
        }

        params.i_csp = sys::X264_CSP_I420 as c_int;
        params.i_width = to_c_int(config.width);
        params.i_height = to_c_int(config.height);
        params.i_fps_num = config.fps.max(1);
        params.i_fps_den = 1;
        params.i_keyint_max = to_c_int(config.keyframe_interval_frames());
        // Keyframes only at the interval or on request.
        params.i_scenecut_threshold = 0;
        // SPS/PPS in front of every IDR so each HLS segment is
        // self-contained; Annex B start codes for MPEG-TS.
        params.b_repeat_headers = 1;
        params.b_annexb = 1;
        // Signal the colour the RGB conversion produces (or the producer
        // of YUV input promised) instead of letting decoders guess.
        let colour = config.colour;
        params.vui.b_fullrange = c_int::from(colour.range == ColourRange::Full);
        params.vui.i_colorprim = c_int::from(colour.primaries.code());
        params.vui.i_transfer = c_int::from(colour.transfer.code());
        params.vui.i_colmatrix = c_int::from(colour.matrix.code());
        if let Some(location) = colour.chroma_location {
            params.vui.i_chroma_loc = c_int::from(location.code());
        }
        // Presets default to CRF, which ignores `i_bitrate` entirely.
        // ABR with a VBV capped at the target makes the bitrate a real
        // ceiling (≈ CBR) so a weak receiver link isn't flooded.
        params.rc.i_rc_method = sys::X264_RC_ABR as c_int;
        apply_bitrate(&mut params, config.bitrate_kbps);

        let profile = match config.max_h264_profile.unwrap_or(H264Profile::High) {
            H264Profile::Baseline => c"baseline",
            H264Profile::Main => c"main",
            H264Profile::High => c"high",
        };
        // SAFETY: `params` is initialised and `profile` NUL-terminated.
        if unsafe { sys::x264_param_apply_profile(&mut params, profile.as_ptr()) } < 0 {
            return Err(FerricastError::Encoder(format!(
                "x264: settings incompatible with profile {profile:?}"
            )));
        }

        // SAFETY: `params` is initialised; libx264 copies it and returns
        // NULL when the parameters are invalid.
        let raw = unsafe { sys::x264_encoder_open(&mut params) };
        let raw = NonNull::new(raw).ok_or_else(|| {
            FerricastError::Encoder(format!(
                "x264: encoder rejected {}x{} @ {} fps",
                config.width, config.height, config.fps
            ))
        })?;
        Ok(Self {
            raw,
            width: config.width,
            height: config.height,
        })
    }

    /// SPS + PPS (+ SEI) in Annex B form.
    fn headers(&mut self) -> Result<Vec<u8>> {
        let mut nals: *mut sys::x264_nal_t = ptr::null_mut();
        let mut count: c_int = 0;
        // SAFETY: `raw` is a live encoder; out-pointers are valid.
        let size = unsafe { sys::x264_encoder_headers(self.raw.as_ptr(), &mut nals, &mut count) };
        // SAFETY: values come straight from the call above and the
        // handle hasn't been used since.
        let payload = unsafe { nal_payload(nals, count, size) }?;
        Ok(payload.to_vec())
    }

    /// Submit `picture` (or drain a delayed one with `None`) and return
    /// the picture libx264 emitted on this call, if any.
    fn encode(&mut self, picture: Option<&mut sys::x264_picture_t>) -> Result<Option<Output>> {
        let mut nals: *mut sys::x264_nal_t = ptr::null_mut();
        let mut count: c_int = 0;
        let mut pic_out = blank_picture();
        let pic_in = picture.map_or(ptr::null_mut(), ptr::from_mut);
        // SAFETY: `raw` is live; `pic_in` is NULL (drain) or a picture
        // whose planes stay borrowed for the duration of this call
        // (libx264 copies them before returning); out-pointers are valid.
        let size = unsafe {
            sys::x264_encoder_encode(
                self.raw.as_ptr(),
                &mut nals,
                &mut count,
                pic_in,
                &mut pic_out,
            )
        };
        // SAFETY: values come straight from the call above.
        let payload = unsafe { nal_payload(nals, count, size) }?;
        if payload.is_empty() {
            // No picture came out of this call; `pic_out` is meaningless.
            return Ok(None);
        }
        Ok(Some(Output {
            data: Bytes::copy_from_slice(payload),
            pts: pic_out.i_pts,
            dts: pic_out.i_dts,
            keyframe: pic_out.b_keyframe != 0,
        }))
    }

    /// Retarget ABR + VBV without restarting the session.
    fn set_bitrate(&mut self, kbps: u32) -> Result<()> {
        let mut params = MaybeUninit::<sys::x264_param_t>::uninit();
        // SAFETY: `raw` is live; `x264_encoder_parameters` copies the
        // encoder's current parameters into the out-pointer in full.
        let mut params = unsafe {
            sys::x264_encoder_parameters(self.raw.as_ptr(), params.as_mut_ptr());
            params.assume_init()
        };
        apply_bitrate(&mut params, kbps);
        // SAFETY: `raw` is live and `params` initialised; libx264 only
        // applies the fields that may change mid-stream (VBV was enabled
        // at open, which bitrate reconfiguration requires).
        if unsafe { sys::x264_encoder_reconfig(self.raw.as_ptr(), &mut params) } < 0 {
            return Err(FerricastError::Encoder(format!(
                "x264: reconfigure to {kbps} kbps rejected"
            )));
        }
        Ok(())
    }

    /// Pictures accepted but not yet emitted.
    fn delayed_frames(&self) -> c_int {
        // SAFETY: `raw` is a live encoder.
        unsafe { sys::x264_encoder_delayed_frames(self.raw.as_ptr()) }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: `raw` came from `x264_encoder_open` and is closed once.
        unsafe { sys::x264_encoder_close(self.raw.as_ptr()) }
    }
}

/// Borrow the bytes of the NAL units libx264 returned.
///
/// # Safety
///
/// `nals`, `count` and `size` must come from a successful
/// `x264_encoder_encode` / `x264_encoder_headers` call on a handle that
/// has not been used since; the slice is only valid until the next call.
unsafe fn nal_payload<'a>(
    nals: *const sys::x264_nal_t,
    count: c_int,
    size: c_int,
) -> Result<&'a [u8]> {
    let len = usize::try_from(size)
        .map_err(|_| FerricastError::Encoding(format!("x264: encode failed ({size})")))?;
    if len == 0 || count <= 0 || nals.is_null() {
        return Ok(&[]);
    }
    // SAFETY: `nals` points at `count >= 1` NAL descriptors (caller
    // contract). x264.h guarantees "the payloads of all output NALs are
    // sequential in memory", `size` bytes in total starting at the
    // first payload.
    unsafe { Ok(std::slice::from_raw_parts((*nals).p_payload, len)) }
}

/// A fully initialised `x264_picture_t` with libx264's defaults.
fn blank_picture() -> sys::x264_picture_t {
    let mut pic = MaybeUninit::uninit();
    // SAFETY: `x264_picture_init` zeroes the struct and sets defaults,
    // initialising every field.
    unsafe {
        sys::x264_picture_init(pic.as_mut_ptr());
        pic.assume_init()
    }
}

/// Average bitrate, peak rate and a one-second VBV buffer, all at
/// `kbps`: the stream never bursts above the target for longer than
/// the receiver's one-second buffering.
fn apply_bitrate(params: &mut sys::x264_param_t, kbps: u32) {
    let kbps = to_c_int(kbps.max(1));
    params.rc.i_bitrate = kbps;
    params.rc.i_vbv_max_bitrate = kbps;
    params.rc.i_vbv_buffer_size = kbps;
}

fn to_c_int(v: u32) -> c_int {
    c_int::try_from(v).unwrap_or(c_int::MAX)
}

/// Capture timestamps of pictures in flight inside libx264, in
/// submission order.
#[derive(Default)]
struct TimestampQueue(VecDeque<(i64, u64)>);

impl TimestampQueue {
    fn push(&mut self, pts: i64, timestamp_us: u64) {
        self.0.push_back((pts, timestamp_us));
    }

    /// Remove and return the timestamp submitted with `pts`. Older
    /// entries were never emitted (cannot happen without frame drops)
    /// and are discarded so the queue can't grow without bound.
    fn take(&mut self, pts: i64) -> Option<u64> {
        while let Some(&(queued, timestamp_us)) = self.0.front() {
            if queued > pts {
                return None;
            }
            self.0.pop_front();
            if queued == pts {
                return Some(timestamp_us);
            }
        }
        None
    }
}

/// Reused I420 staging buffer. Allocated on the first encode after a
/// `configure()` (which sizes it), then borrowed mutably each frame so
/// the hot path doesn't pay ~1.5×W×H bytes of alloc/free per frame.
struct I420Scratch(YuvPlanarImageMut<'static, u8>);

impl I420Scratch {
    fn new(width: u32, height: u32) -> Self {
        Self(YuvPlanarImageMut::alloc(
            width,
            height,
            YuvChromaSubsampling::Yuv420,
        ))
    }

    /// An input picture pointing at the scratch planes. libx264 only
    /// reads the planes, and only during the `x264_encoder_encode` call
    /// the picture is passed to, so `self` must not change until then.
    fn picture(&self, pts: i64, force_idr: bool) -> sys::x264_picture_t {
        let mut pic = blank_picture();
        pic.i_type = if force_idr {
            sys::X264_TYPE_IDR
        } else {
            sys::X264_TYPE_AUTO
        } as c_int;
        pic.i_pts = pts;
        pic.img.i_csp = sys::X264_CSP_I420 as c_int;
        pic.img.i_plane = 3;
        let planes = [
            (self.0.y_plane.borrow(), self.0.y_stride),
            (self.0.u_plane.borrow(), self.0.u_stride),
            (self.0.v_plane.borrow(), self.0.v_stride),
        ];
        for (i, (data, stride)) in planes.into_iter().enumerate() {
            pic.img.plane[i] = data.as_ptr().cast_mut();
            pic.img.i_stride[i] = to_c_int(stride);
        }
        pic
    }

    /// Convert / copy `src` into the scratch planes, walking `src` with
    /// its own stride (see [`FrameView`] for the planar layouts).
    fn fill_from(&mut self, src: &FrameView<'_>, conversion: Conversion) -> Result<()> {
        let dst = &mut self.0;
        let stride = u32::try_from(src.stride())
            .map_err(|_| FerricastError::Encoder("x264: stride exceeds u32".into()))?;
        match src.format() {
            PixelFormat::Bgra => bgra_to_yuv420(
                dst,
                src.data(),
                stride,
                conversion.range,
                conversion.matrix,
                yuv::YuvConversionMode::Balanced,
            )
            .map_err(|e| FerricastError::Encoding(format!("x264: BGRA → I420: {e}"))),
            PixelFormat::Rgba => rgba_to_yuv420(
                dst,
                src.data(),
                stride,
                conversion.range,
                conversion.matrix,
                yuv::YuvConversionMode::Balanced,
            )
            .map_err(|e| FerricastError::Encoding(format!("x264: RGBA → I420: {e}"))),
            PixelFormat::Nv12 => {
                let (w, cw) = (src.width() as usize, src.width().div_ceil(2) as usize);
                copy_rows(
                    dst.y_plane.borrow_mut(),
                    dst.y_stride as usize,
                    src.data(),
                    src.stride(),
                    w,
                );
                let (u_stride, v_stride) = (dst.u_stride as usize, dst.v_stride as usize);
                let uv = &src.data()[src.chroma_offset()..];
                let u_rows = dst.u_plane.borrow_mut().chunks_mut(u_stride);
                let v_rows = dst.v_plane.borrow_mut().chunks_mut(v_stride);
                for ((u, v), line) in u_rows.zip(v_rows).zip(uv.chunks(src.chroma_stride())) {
                    for (i, pair) in line[..cw * 2].chunks_exact(2).enumerate() {
                        u[i] = pair[0];
                        v[i] = pair[1];
                    }
                }
                Ok(())
            }
            PixelFormat::I420 => {
                let (w, cw) = (src.width() as usize, src.width().div_ceil(2) as usize);
                let data = src.data();
                copy_rows(
                    dst.y_plane.borrow_mut(),
                    dst.y_stride as usize,
                    data,
                    src.stride(),
                    w,
                );
                copy_rows(
                    dst.u_plane.borrow_mut(),
                    dst.u_stride as usize,
                    &data[src.chroma_offset()..],
                    src.chroma_stride(),
                    cw,
                );
                copy_rows(
                    dst.v_plane.borrow_mut(),
                    dst.v_stride as usize,
                    &data[src.v_offset()..],
                    src.chroma_stride(),
                    cw,
                );
                Ok(())
            }
            PixelFormat::P010 => Err(FerricastError::Encoder(
                "x264: P010 input must be narrowed to NV12 first".into(),
            )),
        }
    }
}

/// Copy `width` bytes of every row from `src` (rows `src_stride`
/// apart) into `dst` (rows `dst_stride` apart), for as many rows as
/// both buffers hold.
fn copy_rows(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, width: usize) {
    for (d, s) in dst.chunks_mut(dst_stride).zip(src.chunks(src_stride)) {
        d[..width].copy_from_slice(&s[..width]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferricast_core::RawFrame;

    fn bgra_frame(width: u32, height: u32, stride: u32, timestamp_us: u64) -> CapturedFrame {
        let mut buf = vec![0u8; (stride * height) as usize];
        for row in buf.chunks_mut(stride as usize) {
            for px in row[..(width * 4) as usize].chunks_exact_mut(4) {
                px[3] = 0xff; // opaque alpha; BGR all zero
            }
        }
        CapturedFrame::Cpu(RawFrame {
            hdr: None,
            colour: None,
            width,
            height,
            stride,
            format: PixelFormat::Bgra,
            data: Bytes::from(buf),
            timestamp_us,
        })
    }

    fn count_param_sets(annex_b: &[u8]) -> (u32, u32) {
        let mut i = 0usize;
        let mut sps = 0u32;
        let mut pps = 0u32;
        while i + 3 <= annex_b.len() {
            if annex_b[i..i + 3] != [0, 0, 1] {
                i += 1;
                continue;
            }
            let nal_start = i + 3;
            if nal_start >= annex_b.len() {
                break;
            }
            match annex_b[nal_start] & 0x1f {
                7 => sps += 1,
                8 => pps += 1,
                _ => {}
            }
            i = nal_start + 1;
        }
        (sps, pps)
    }

    fn small_cfg() -> EncoderConfig {
        EncoderConfig {
            codec: Codec::H264,
            width: 320,
            height: 240,
            bitrate_kbps: 500,
            fps: 30,
            keyframe_interval_secs: 1.0,
            pixel_format: PixelFormat::Bgra,
            max_h264_profile: Some(H264Profile::High),
            max_h265_profile: None,
            ..EncoderConfig::default()
        }
    }

    fn configured() -> X264Encoder {
        let mut enc = X264Encoder::default();
        enc.configure(&small_cfg()).expect("configure");
        enc
    }

    /// Zero latency: the very first input comes straight back out as
    /// an IDR carrying SPS/PPS (b_repeat_headers=1).
    #[test]
    fn first_frame_is_immediate_idr_with_param_sets() {
        let mut enc = configured();
        let out = enc.encode(bgra_frame(320, 240, 1280, 7)).expect("encode");
        assert!(out.is_keyframe);
        assert_eq!(out.timestamp_us, 7);
        let (sps, pps) = count_param_sets(&out.data);
        assert!(sps >= 1 && pps >= 1, "SPS={sps} PPS={pps}");
    }

    /// HDR metadata goes into keyframes as SEI; a 10-bit stream and a
    /// matrix the conversion lacks are refused up front.
    #[test]
    fn hdr_metadata_rides_keyframes_and_ten_bit_is_refused() {
        let hdr = ferricast_core::HdrMetadata {
            mastering_display: None,
            content_light: Some(ferricast_core::ContentLightLevel {
                max_cll: 1000,
                max_fall: 400,
            }),
        };
        let mut enc = X264Encoder::default();
        enc.configure(&EncoderConfig {
            hdr: Some(hdr),
            ..small_cfg()
        })
        .expect("configure");
        let key = enc.encode(bgra_frame(320, 240, 1280, 0)).expect("encode");
        assert!(key.is_keyframe);
        assert_eq!(
            ferricast_codec::bitstream::hdr::from_annex_b(&key.data, NalSyntax::H264),
            hdr
        );
        let next = enc.encode(bgra_frame(320, 240, 1280, 1)).expect("encode");
        assert!(!next.is_keyframe);
        assert!(
            ferricast_codec::bitstream::hdr::from_annex_b(&next.data, NalSyntax::H264).is_empty()
        );

        let ten_bit = EncoderConfig {
            bit_depth: BitDepth::Ten,
            ..small_cfg()
        };
        assert!(matches!(
            X264Encoder::default().configure(&ten_bit),
            Err(CodecError::UnsupportedConfig { .. })
        ));
        let ictcp = EncoderConfig {
            colour: ferricast_core::ColourDescription {
                matrix: ferricast_core::MatrixCoefficients::ICtCp,
                ..ferricast_core::ColourDescription::BT709_LIMITED
            },
            ..small_cfg()
        };
        assert!(matches!(
            X264Encoder::default().configure(&ictcp),
            Err(CodecError::UnsupportedConfig { .. })
        ));
    }

    /// Each output carries its own input's timestamp and PTS == DTS.
    #[test]
    fn output_timestamps_track_inputs() {
        let mut enc = configured();
        for i in 0..10u64 {
            let out = enc
                .encode(bgra_frame(320, 240, 1280, 1_000 + i * 33_333))
                .expect("encode");
            assert_eq!(out.timestamp_us, 1_000 + i * 33_333);
            assert_eq!(out.pts_dts, (i, i));
        }
        assert!(enc.flush().expect("flush").is_empty());
    }

    #[test]
    fn requested_keyframe_is_honoured() {
        let mut enc = configured();
        for _ in 0..3 {
            enc.encode(bgra_frame(320, 240, 1280, 0)).expect("encode");
        }
        let p = enc.encode(bgra_frame(320, 240, 1280, 0)).expect("encode");
        assert!(!p.is_keyframe);
        enc.request_keyframe();
        let idr = enc.encode(bgra_frame(320, 240, 1280, 0)).expect("encode");
        assert!(idr.is_keyframe);
    }

    #[test]
    fn size_change_is_scaled_into_the_session() {
        let mut enc = configured();
        assert!(enc.encode(bgra_frame(640, 480, 2560, 0)).is_ok());
        assert!(enc.encode(bgra_frame(500, 240, 2000, 0)).is_ok());
        assert!(enc.encode(bgra_frame(320, 240, 1280, 0)).is_ok());
    }

    #[test]
    fn padded_rows_are_accepted() {
        let mut enc = configured();
        assert!(enc.encode(bgra_frame(320, 240, 1344, 0)).is_ok());
    }

    fn noise_frame(seed: u32) -> CapturedFrame {
        let mut state = seed.wrapping_mul(2_654_435_761).max(1);
        let buf: Vec<u8> = (0..320 * 240 * 4)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state.to_le_bytes()[0]
            })
            .collect();
        CapturedFrame::Cpu(RawFrame {
            hdr: None,
            colour: None,
            width: 320,
            height: 240,
            stride: 1280,
            format: PixelFormat::Bgra,
            data: Bytes::from(buf),
            timestamp_us: 0,
        })
    }

    /// Pure noise at CRF would cost megabits per frame; ABR + VBV must
    /// hold one second of it near the 500 kbps target (≈ 62 KB).
    #[test]
    fn bitrate_caps_incompressible_content() {
        let mut enc = configured();
        let total: usize = (0..30)
            .map(|i| enc.encode(noise_frame(i + 1)).expect("encode").data.len())
            .sum();
        assert!(
            total < 3 * 62_500,
            "{total} bytes for one second at 500 kbps"
        );
        enc.set_bitrate_kbps(250).expect("reconfigure");
        assert!(enc.encode(noise_frame(99)).is_ok());
    }

    #[test]
    fn timestamp_queue_matches_by_pts() {
        let mut q = TimestampQueue::default();
        q.push(0, 100);
        q.push(1, 200);
        q.push(2, 300);
        assert_eq!(q.take(0), Some(100));
        // PTS 1 skipped: its entry is discarded on the way to 2.
        assert_eq!(q.take(2), Some(300));
        assert_eq!(q.take(1), None);
        assert!(q.0.is_empty());
    }

    #[test]
    fn negative_timestamps_clamp_to_zero() {
        assert_eq!(non_negative(-2), 0);
        assert_eq!(non_negative(5), 5);
    }
}
