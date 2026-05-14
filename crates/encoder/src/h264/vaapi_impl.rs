//! VA-API H.264 encoder.
//!
//! Profile / feature decisions:
//!
//! * **Profile**: Constrained Baseline (`66`) when the GPU exposes
//!   it, else Main (`77`). Both are lowest-common-denominator across
//!   Chromecast / Miracast / AirPlay. We don't target High because
//!   only the screen-cast path benefits from it and not every
//!   driver supports CABAC encode.
//! * **GOP**: closed, IPPP… with an IDR every `keyframe_interval`
//!   frames. No B-frames (Constrained Baseline forbids them; even
//!   on Main we skip them — the segmenter cuts segments on
//!   keyframes and B-frames complicate that).
//! * **Surface format**: NV12 with `VA_RT_FORMAT_YUV420`. Universal.
//!   BGRA capture frames are colour-converted to NV12 on the CPU
//!   (see [`super::yuv`]) before upload — VA-API drivers don't
//!   accept BGRA into the H.264 encoder entrypoint directly.
//! * **Rate control**: CBR by default, configured via
//!   `VAEncMiscParameterRateControl` + `FrameRate` + `HRD` chained
//!   buffers.
//! * **Packed headers**: we synthesise our own SPS / PPS NAL units
//!   ([`super::headers`]) and submit them via `VAEncPackedHeader*`
//!   buffers built by raw FFI — cros-libva 0.0.13 doesn't have safe
//!   wrappers for those buffer types yet.
//!
//! Anything in this module that uses unsafe is in service of a
//! `cros_libva::bindings::*` call documented in the libva headers.

use std::cell::RefCell;
use std::os::fd::RawFd;
use std::os::raw::c_void;
use std::path::Path;
use std::rc::Rc;

use bytes::Bytes;
use cros_libva::*;
use ferricast_core::{
    CapturedFrame, Codec, EncodedFrame, EncoderConfig, FerricastError, GpuFrame, PixelFormat,
    Result, VideoEncoder,
};
use tracing::{debug, info, trace, warn};

use super::headers::{self, FrameCrop, PpsParams, SpsParams, VuiParams, profile};
use super::yuv;

const RENDER_NODES: &[&str] = &[
    "/dev/dri/renderD128",
    "/dev/dri/renderD129",
    "/dev/dri/renderD130",
];

/// Standard NV12 V4L2 / DRM fourcc reused across libva.
const VA_FOURCC_NV12: u32 = 0x3231564E; // 'N','V','1','2'

/// VA-API fourccs for the BGRA / RGBA input layouts we accept on
/// the DMA-BUF import path. These name the **byte order in memory**
/// (byte 0 = B for BGRA, etc.). Mapped to DRM fourccs via
/// [`pixel_format_to_drm`].
const VA_FOURCC_BGRA: u32 = 0x41524742; // 'B','G','R','A'
const VA_FOURCC_RGBA: u32 = 0x41424752; // 'R','G','B','A'

/// DRM fourccs are the inverse of VA's: the four letters describe
/// the **integer** as little-endian, not the memory order. So
/// memory B,G,R,A == integer ARGB == `DRM_FORMAT_ARGB8888`.
const DRM_FORMAT_ARGB8888: u32 = 0x34325241; // 'A','R','2','4'
const DRM_FORMAT_ABGR8888: u32 = 0x34324241; // 'A','B','2','4'

/// `VA_RT_FORMAT_*` per `va.h`. `RGB32` is what the VPP input
/// surface uses for BGRA / RGBA imports.
const VA_RT_FORMAT_YUV420: u32 = 0x01;
const VA_RT_FORMAT_RGB32: u32 = 0x04;

/// Number of reconstruction surfaces. One for the current frame +
/// one for the previous reference (used by P frames). Two is the
/// minimum for IPPP without B-frames.
const RECON_POOL: usize = 2;

/// Output coded-buffer size hint. Worst-case picture for 1080p ≈
/// 1.5 × width × height bytes; we round up to a comfortable 8 MB
/// because the driver pads its allocation anyway.
const CODED_BUFFER_SIZE: usize = 8 * 1024 * 1024;

pub struct VaapiH264Encoder {
    display: Rc<Display>,
    /// Held so the FFI handle stays valid; cros-libva drops it
    /// before the context.
    #[allow(dead_code)]
    config: Config,
    context: Rc<Context>,

    /// VPP (`VAEntrypointVideoProc`) plumbing. Used by the GPU
    /// path to convert imported BGRA/RGBA DMA-BUF surfaces into
    /// `self.input` (NV12) inside the driver, without any CPU
    /// readback. `None` means the driver doesn't expose VPP — we
    /// fall back to the CPU path even for GPU frames in that case
    /// (which still beats x264 because the encode itself is HW).
    ///
    /// `vpp_config` is held purely to keep the FFI handle alive
    /// for the lifetime of `vpp_context` — cros-libva drops the
    /// config before the context, which would invalidate the
    /// context. Never read directly.
    #[allow(dead_code)]
    vpp_config: Option<Config>,
    vpp_context: Option<Rc<Context>>,

    /// One NV12 surface that we re-upload every frame as the
    /// encoder's input.
    input: Surface<()>,
    /// Reconstruction surfaces, alternated round-robin so the
    /// previous one can serve as the reference picture for the
    /// next P frame.
    recon: Vec<Surface<()>>,

    /// Latched configuration.
    cfg: EncoderCfg,

    /// Per-stream packed headers (Annex-B `00 00 00 01` prefixed,
    /// emulation-byte-escaped). Re-emitted on every IDR.
    sps_nal: Vec<u8>,
    pps_nal: Vec<u8>,

    /// Mutable per-frame state. Wrapped in `RefCell` so `encode`
    /// can take `&mut self` without bleeding the cell into every
    /// helper signature.
    state: RefCell<FrameState>,
}

#[derive(Clone)]
struct EncoderCfg {
    profile: VAProfile::Type,
    profile_idc: u8,
    /// `constraint_set0..5` packed in the high byte (see SPS
    /// emission).
    constraint_flags: u8,
    /// CABAC iff Main / High; CAVLC for Constrained Baseline.
    cabac: bool,
    width: u32,
    height: u32,
    /// Width in 16-pixel macroblocks.
    width_mbs: u16,
    /// Height in 16-pixel macroblocks (rounded up).
    height_mbs: u16,
    /// Crop offset (in 2-luma units) needed when `height` isn't a
    /// multiple of 16.
    height_crop: u32,
    fps: u32,
    bitrate_bps: u32,
    keyframe_interval: u32,
    initial_qp: u32,
    /// Same level as in the SPS / `level_idc`. 41 = 4.1, the
    /// minimum for 1080p60.
    level_idc: u8,
}

#[derive(Default)]
struct FrameState {
    frame_idx: u32,
    /// Picture-order-count (×2 per frame, mod `MaxPicOrderCntLsb`).
    poc: u32,
    /// Toggles 0..1 every IDR. Sent in the slice header.
    idr_pic_id: u16,
    /// Round-robin index into `recon`.
    next_recon: usize,
    /// Previous reconstruction surface index, or `None` on the very
    /// first frame.
    prev_recon: Option<usize>,
    /// Frame number of the previous reference (for P frame ref
    /// list).
    prev_frame_num: u32,
    /// POC of the previous reference.
    prev_poc: u32,
    /// Set by [`VaapiH264Encoder::request_keyframe`]; OR-ed into the
    /// natural interval-based IDR decision on the next encode so
    /// the HLS segmenter can anchor segment boundaries to wall
    /// clock without reconfiguring the encoder.
    pending_keyframe: bool,
}

unsafe impl Send for VaapiH264Encoder {}

impl VaapiH264Encoder {
    /// Try to bring up VA-API + create everything that's
    /// stream-invariant. Returns `Err` and the factory falls back
    /// to x264 on any problem (no driver, no compatible profile,
    /// surface alloc failed, ...).
    pub fn probe() -> Result<Self> {
        Self::probe_with(EncoderConfig::default())
    }

    /// Real entry point used by the factory's `configure()` retry
    /// path. The `EncoderConfig::default` from `probe` is replaced
    /// here with the caller's actual config so we size surfaces
    /// correctly.
    pub fn probe_with(cfg: EncoderConfig) -> Result<Self> {
        if !matches!(cfg.pixel_format, PixelFormat::Bgra | PixelFormat::Rgba) {
            return Err(FerricastError::Encoder(format!(
                "VA-API: input pixel format {:?} not supported (need Bgra/Rgba)",
                cfg.pixel_format
            )));
        }

        let display = open_render_node()
            .ok_or_else(|| FerricastError::Encoder("VA-API: no usable DRM render node".into()))?;
        let vendor = display.query_vendor_string().unwrap_or_default();
        debug!(%vendor, "VA-API display opened");

        let supported = display
            .query_config_profiles()
            .map_err(|e| FerricastError::Encoder(format!("query_config_profiles: {e}")))?;

        let (profile, profile_idc, constraint_flags, cabac) = if supported
            .contains(&VAProfile::VAProfileH264ConstrainedBaseline)
            && profile_has_enc_slice(&display, VAProfile::VAProfileH264ConstrainedBaseline)
        {
            (
                VAProfile::VAProfileH264ConstrainedBaseline,
                profile::BASELINE,
                0b0100_0000_u8, // constraint_set1_flag = 1 (Constrained Baseline)
                false,
            )
        } else if supported.contains(&VAProfile::VAProfileH264Main)
            && profile_has_enc_slice(&display, VAProfile::VAProfileH264Main)
        {
            (VAProfile::VAProfileH264Main, profile::MAIN, 0_u8, true)
        } else {
            return Err(FerricastError::Encoder(
                "VA-API: no supported H.264 encode profile (need ConstrainedBaseline or Main)"
                    .into(),
            ));
        };
        info!(?profile, %vendor, "VA-API H.264 encoder selected");

        let cfg = build_encoder_cfg(profile, profile_idc, constraint_flags, cabac, &cfg)?;

        // VAConfig with rate control + RT format.
        let cfg_handle = display
            .create_config(
                vec![
                    VAConfigAttrib {
                        type_: VAConfigAttribType::VAConfigAttribRTFormat,
                        value: VA_RT_FORMAT_YUV420,
                    },
                    VAConfigAttrib {
                        type_: VAConfigAttribType::VAConfigAttribRateControl,
                        value: VA_RC_CBR,
                    },
                ],
                profile,
                VAEntrypoint::VAEntrypointEncSlice,
            )
            .map_err(|e| FerricastError::Encoder(format!("vaCreateConfig: {e}")))?;

        // Allocate surfaces: one input + N reconstruction. All NV12.
        let input_descs: Vec<()> = vec![()];
        let mut input_surfaces = display
            .create_surfaces(
                VA_RT_FORMAT_YUV420,
                Some(VA_FOURCC_NV12),
                cfg.width,
                cfg.padded_height(),
                Some(UsageHint::USAGE_HINT_ENCODER),
                input_descs,
            )
            .map_err(|e| FerricastError::Encoder(format!("create input surface: {e}")))?;
        let input = input_surfaces.pop().expect("we asked for 1 surface");

        let recon_descs: Vec<()> = (0..RECON_POOL).map(|_| ()).collect();
        let recon = display
            .create_surfaces(
                VA_RT_FORMAT_YUV420,
                Some(VA_FOURCC_NV12),
                cfg.width,
                cfg.padded_height(),
                Some(UsageHint::USAGE_HINT_ENCODER),
                recon_descs,
            )
            .map_err(|e| FerricastError::Encoder(format!("create recon surfaces: {e}")))?;

        // Hand the recon surfaces (the encoder's render targets) to
        // `vaCreateContext`. The input surface is uploaded every
        // frame and isn't a render target, so it doesn't go in the
        // context's surface list.
        let context = display
            .create_context(
                &cfg_handle,
                cfg.width,
                cfg.padded_height(),
                Some(&recon),
                /* progressive = */ true,
            )
            .map_err(|e| FerricastError::Encoder(format!("vaCreateContext: {e}")))?;

        // Synthesise SPS / PPS once per encoder. They get re-emitted
        // unchanged on every IDR.
        let sps_nal = headers::build_sps(&SpsParams {
            profile_idc: cfg.profile_idc,
            constraint_flags: cfg.constraint_flags,
            level_idc: cfg.level_idc,
            seq_parameter_set_id: 0,
            pic_width_in_mbs_minus1: (cfg.width_mbs as u32) - 1,
            pic_height_in_map_units_minus1: (cfg.height_mbs as u32) - 1,
            log2_max_frame_num_minus4: 4,
            log2_max_pic_order_cnt_lsb_minus4: 4,
            max_num_ref_frames: 1,
            frame_cropping: if cfg.height_crop > 0 {
                Some(FrameCrop {
                    left: 0,
                    right: 0,
                    top: 0,
                    bottom: cfg.height_crop,
                })
            } else {
                None
            },
            vui: Some(VuiParams {
                num_units_in_tick: 1,
                time_scale: 2 * cfg.fps,
                fixed_frame_rate_flag: true,
            }),
        });
        let pps_nal = headers::build_pps(&PpsParams {
            pic_parameter_set_id: 0,
            seq_parameter_set_id: 0,
            entropy_coding_mode_flag: cfg.cabac,
            num_ref_idx_l0_default_active_minus1: 0,
            pic_init_qp_minus26: cfg.initial_qp as i32 - 26,
            deblocking_filter_control_present_flag: true,
            transform_8x8_mode_flag: false,
        });

        // VPP config/context. We try to bring it up but treat any
        // failure as soft: without VPP we lose zero-copy DMA-BUF
        // input but the rest of the encoder still works (CPU path).
        let (vpp_config, vpp_context) = match build_vpp(&display, &cfg) {
            Ok((c, ctx)) => {
                info!("VA-API VPP up; DMA-BUF input will be zero-copy");
                (Some(c), Some(ctx))
            }
            Err(e) => {
                warn!(error = %e, "VA-API VPP unavailable; DMA-BUF frames will fall back to CPU path");
                (None, None)
            }
        };

        Ok(Self {
            display,
            config: cfg_handle,
            context,
            vpp_config,
            vpp_context,
            input,
            recon,
            cfg,
            sps_nal,
            pps_nal,
            state: RefCell::new(FrameState::default()),
        })
    }
}

fn build_vpp(display: &Rc<Display>, cfg: &EncoderCfg) -> Result<(Config, Rc<Context>)> {
    let config = display
        .create_config(
            vec![],
            VAProfile::VAProfileNone,
            VAEntrypoint::VAEntrypointVideoProc,
        )
        .map_err(|e| FerricastError::Encoder(format!("vaCreateConfig(VPP): {e}")))?;
    // VPP contexts don't have a render-target list (the dest
    // surface is passed per-Picture). `create_context` accepts
    // `Option<&Vec<Surface<D>>>`, so we hand it `None` rather than
    // synthesising an empty Vec just to satisfy the API shape.
    let context = display
        .create_context::<()>(
            &config,
            cfg.width,
            cfg.padded_height(),
            None,
            /* progressive = */ true,
        )
        .map_err(|e| FerricastError::Encoder(format!("vaCreateContext(VPP): {e}")))?;
    Ok((config, context))
}

fn open_render_node() -> Option<Rc<Display>> {
    for node in RENDER_NODES {
        let path = Path::new(node);
        if !path.exists() {
            continue;
        }
        match Display::open_drm_display(path) {
            Ok(d) => {
                debug!(node = node, "opened VA-API DRM display");
                return Some(d);
            }
            Err(e) => debug!(node = node, error = %e, "render node open failed"),
        }
    }
    Display::open()
}

fn profile_has_enc_slice(display: &Display, profile: VAProfile::Type) -> bool {
    match display.query_config_entrypoints(profile) {
        Ok(eps) => eps.iter().any(|e| *e == VAEntrypoint::VAEntrypointEncSlice),
        Err(_) => false,
    }
}

impl EncoderCfg {
    /// MB-aligned encoded height. The user-visible height may be
    /// smaller and is communicated via the SPS frame crop.
    fn padded_height(&self) -> u32 {
        (self.height_mbs as u32) * 16
    }
}

fn build_encoder_cfg(
    profile: VAProfile::Type,
    profile_idc: u8,
    constraint_flags: u8,
    cabac: bool,
    cfg: &EncoderConfig,
) -> Result<EncoderCfg> {
    let width = cfg.width.max(16);
    let height = cfg.height.max(16);
    let width_mbs = ((width + 15) / 16) as u16;
    let height_mbs = ((height + 15) / 16) as u16;
    let padded_h = (height_mbs as u32) * 16;
    let height_crop = (padded_h - height) / 2;
    let level_idc = pick_level(width, height, cfg.fps);

    Ok(EncoderCfg {
        profile,
        profile_idc,
        constraint_flags,
        cabac,
        width,
        height,
        width_mbs,
        height_mbs,
        height_crop,
        fps: cfg.fps.max(1),
        bitrate_bps: (cfg.bitrate_kbps.max(1) as u32).saturating_mul(1000),
        keyframe_interval: cfg.keyframe_interval.max(1),
        initial_qp: 26,
        level_idc,
    })
}

/// Pick the smallest H.264 level that contains the requested
/// resolution+framerate (Annex A, Table A-1). 41 covers 1080p60;
/// 50 covers 4K30. Anything larger we just clamp to 51.
fn pick_level(width: u32, height: u32, fps: u32) -> u8 {
    let mb_per_sec = ((width + 15) / 16) * ((height + 15) / 16) * fps.max(1);
    if mb_per_sec <= 245_760 {
        31
    } else if mb_per_sec <= 522_240 {
        40
    } else if mb_per_sec <= 522_240 && (width * height) <= 2_073_600 {
        41
    } else if mb_per_sec <= 589_824 {
        41
    } else if mb_per_sec <= 983_040 {
        42
    } else if mb_per_sec <= 2_073_600 {
        50
    } else {
        51
    }
}

impl VideoEncoder for VaapiH264Encoder {
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, _config: &EncoderConfig) -> Result<()> {
        // `probe_with` already configured everything stream-invariant
        // when the factory built us. A reconfigure-after-the-fact
        // would need to tear down surfaces / context / config; we
        // don't support that today.
        Ok(())
    }

    fn encode(&mut self, frame: CapturedFrame) -> Result<EncodedFrame> {
        // Two paths into the encoder's `input` NV12 surface:
        //
        // 1. `CapturedFrame::Gpu(g)` + VPP available → import `g`'s
        //    DMA-BUF as a BGRA/RGBA surface and run `vaProcPipeline`
        //    to convert it into NV12 directly inside the driver. No
        //    CPU bytes ever touched.
        // 2. Otherwise → readback to CPU, run the existing
        //    BGRA→NV12 conversion + `vaPutImage` upload. Pre-Gpu
        //    behavior, kept for x264-style callers and as a
        //    fallback when the driver doesn't expose VPP.
        let timestamp_us = frame.timestamp_us();
        match frame {
            CapturedFrame::Gpu(g) if self.vpp_context.is_some() => {
                self.upload_dmabuf_via_vpp(&g)?;
            }
            other => {
                let raw = other.into_cpu()?;
                if !matches!(raw.format, PixelFormat::Bgra | PixelFormat::Rgba) {
                    return Err(FerricastError::Encoder(format!(
                        "VA-API: unexpected runtime pixel format {:?}",
                        raw.format
                    )));
                }
                upload_bgra_to_nv12(&self.input, &self.cfg, &raw.data, raw.stride as usize)?;
            }
        }

        let (encoded_bytes, is_keyframe, frame_idx, poc) = {
            let mut state = self.state.borrow_mut();
            run_encode(self, &mut state)?
        };

        Ok(EncodedFrame {
            codec: Codec::H264,
            data: Bytes::from(encoded_bytes),
            timestamp_us,
            duration_us: Some(1_000_000 / self.cfg.fps as u64),
            is_keyframe,
            pts_dts: (poc as u64, frame_idx as u64),
        })
    }

    fn flush(self) -> Result<Vec<EncodedFrame>> {
        // No B-frame queue, no buffered frames — every `encode()`
        // already produced its output.
        Ok(Vec::new())
    }

    fn request_keyframe(&mut self) {
        self.state.borrow_mut().pending_keyframe = true;
    }

    fn get_headers(&mut self) -> Result<Vec<u8>> {
        // SPS + PPS, AnnexB-prefixed, ready to live at the start of
        // every HLS segment. The bitstream we hand out from
        // `encode()` does NOT include these (we emit them via
        // packed headers and the driver writes them into the coded
        // buffer for IDR frames; non-IDR frames don't carry them).
        let mut out = Vec::with_capacity(self.sps_nal.len() + self.pps_nal.len());
        out.extend_from_slice(&self.sps_nal);
        out.extend_from_slice(&self.pps_nal);
        Ok(out)
    }
}

/// Upload a BGRA frame into the encoder's NV12 input surface. Uses
/// `vaDeriveImage` for a zero-copy view when the driver supports
/// it; falls back to `vaCreateImage` + `vaPutImage` otherwise.
fn upload_bgra_to_nv12(
    surface: &Surface<()>,
    cfg: &EncoderCfg,
    bgra: &[u8],
    bgra_stride: usize,
) -> Result<()> {
    let w = cfg.width;
    let h = cfg.height;
    // Pad height up to MB boundary; the extra rows will be black.
    let padded_h = cfg.padded_height();

    // Try derive first.
    let mut image = match Image::derive_from(surface, (w, padded_h)) {
        Ok(img) => img,
        Err(_e) => {
            // Driver doesn't allow derive on this format. Allocate
            // an NV12 image and pay the put-image cost on drop.
            let format = VAImageFormat {
                fourcc: VA_FOURCC_NV12,
                byte_order: VA_LSB_FIRST,
                bits_per_pixel: 12,
                ..Default::default()
            };
            Image::create_from(surface, format, (w, padded_h), (w, padded_h))
                .map_err(|e| FerricastError::Encoder(format!("vaCreateImage(NV12): {e}")))?
        }
    };

    // Lay out the NV12 planes inside the mapped buffer using the
    // image's plane offsets and strides.
    let im = image.image();
    let y_off = im.offsets[0] as usize;
    let uv_off = im.offsets[1] as usize;
    let y_stride = im.pitches[0] as usize;
    let uv_stride = im.pitches[1] as usize;

    let buf = image.as_mut(); // marks dirty so vaPutImage runs on drop
    let len = buf.len();
    if y_off + y_stride * (padded_h as usize) > len
        || uv_off + uv_stride * ((padded_h / 2) as usize) > len
    {
        return Err(FerricastError::Encoder(
            "VA-API NV12 image buffer smaller than expected planes".into(),
        ));
    }
    // Split-borrow to write Y and UV planes.
    let (y_plane, uv_plane) = if y_off < uv_off {
        let (lo, hi) = buf.split_at_mut(uv_off);
        (&mut lo[y_off..], hi)
    } else {
        let (lo, hi) = buf.split_at_mut(y_off);
        (hi, &mut lo[uv_off..])
    };

    yuv::bgra_to_nv12(
        bgra,
        bgra_stride.max((w * 4) as usize),
        w,
        h,
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
    );

    // Zero out any padding rows so they don't leak previous frame's
    // data into reference frames at the bottom edge.
    if padded_h > h {
        for row in h..padded_h {
            let off = (row as usize) * y_stride;
            for b in &mut y_plane[off..off + w as usize] {
                *b = 16;
            }
        }
        for row in (h / 2)..(padded_h / 2) {
            let off = (row as usize) * uv_stride;
            for b in &mut uv_plane[off..off + w as usize] {
                *b = 128;
            }
        }
    }

    Ok(())
    // Image drops here: if !derived && dirty, vaPutImage writes
    // back to the surface; vaUnmapBuffer + vaDestroyImage either
    // way.
}

/// The actual encode call. Bumps frame state, builds parameter
/// buffers, submits, syncs, extracts the coded bitstream.
fn run_encode(enc: &VaapiH264Encoder, state: &mut FrameState) -> Result<(Vec<u8>, bool, u32, u32)> {
    let cfg = &enc.cfg;
    let is_idr =
        state.frame_idx % cfg.keyframe_interval == 0 || std::mem::take(&mut state.pending_keyframe);
    let frame_num = if is_idr {
        0
    } else {
        // frame_num counts reference frames since last IDR (mod
        // 2^(log2_max_frame_num_minus4+4) = 2^8 = 256).
        ((state.frame_idx - state.frame_idx_at_last_idr(cfg.keyframe_interval)) & 0xff) as u16
    };
    let poc = if is_idr { 0 } else { state.poc };

    // Pick this frame's recon surface and remember the previous
    // one as the P-frame reference.
    let cur_recon_idx = state.next_recon;
    let prev_recon_idx = state.prev_recon;
    state.next_recon = (cur_recon_idx + 1) % enc.recon.len();

    let curr_pic = picture_h264(
        enc.recon[cur_recon_idx].id(),
        frame_num as u32,
        0,
        poc as i32,
    );
    // The "previous reference" slot used by P frames. `None` for
    // IDR — built once and cloned into both the picture-param
    // ReferenceFrames[16] and the slice-param ref_pic_list_0[32].
    let prev_ref: Option<PictureH264> = if !is_idr {
        prev_recon_idx.map(|prev_idx| {
            picture_h264(
                enc.recon[prev_idx].id(),
                state.prev_frame_num,
                VA_PICTURE_H264_SHORT_TERM_REFERENCE,
                state.prev_poc as i32,
            )
        })
    } else {
        None
    };
    let reference_frames: [PictureH264; 16] = std::array::from_fn(|i| {
        if i == 0 {
            prev_ref
                .as_ref()
                .map(clone_pic)
                .unwrap_or_else(invalid_picture_h264)
        } else {
            invalid_picture_h264()
        }
    });

    // Allocate the coded output buffer. cros-libva exposes a
    // dedicated `EncCodedBuffer` for this — `MappedCodedBuffer`
    // takes that specific type rather than the generic `Buffer`.
    let coded = enc
        .context
        .create_enc_coded(CODED_BUFFER_SIZE)
        .map_err(|e| FerricastError::Encoder(format!("vaCreateBuffer(coded): {e}")))?;

    // Build parameter buffers. Order matters per the VA-API loop
    // (libva-utils encode/h264encode.c, FFmpeg vaapi_encode.c):
    // sequence (IDR), misc (IDR), picture, packed SPS+data (IDR),
    // packed PPS+data (IDR), slice.
    let mut buffers: Vec<Buffer> = Vec::with_capacity(8);
    let mut packed_buffer_ids: Vec<VABufferID> = Vec::new();

    if is_idr {
        let seq = build_seq_param(cfg);
        buffers.push(
            enc.context
                .create_buffer(cros_libva::BufferType::EncSequenceParameter(
                    EncSequenceParameter::H264(seq),
                ))
                .map_err(|e| FerricastError::Encoder(format!("seq buffer: {e}")))?,
        );

        let rc = build_rate_control(cfg);
        let fr = EncMiscParameterFrameRate::new(cfg.fps, 0);
        let hrd = EncMiscParameterHRD::new(cfg.bitrate_bps / 2, cfg.bitrate_bps);
        for misc in [
            cros_libva::BufferType::EncMiscParameter(EncMiscParameter::RateControl(rc)),
            cros_libva::BufferType::EncMiscParameter(EncMiscParameter::FrameRate(fr)),
            cros_libva::BufferType::EncMiscParameter(EncMiscParameter::HRD(hrd)),
        ] {
            buffers.push(
                enc.context
                    .create_buffer(misc)
                    .map_err(|e| FerricastError::Encoder(format!("misc buffer: {e}")))?,
            );
        }
    }

    let pic = build_pic_param(
        cfg,
        &curr_pic,
        reference_frames,
        coded.id(),
        is_idr,
        frame_num,
    );
    // Rebuild the per-slice ref list from `prev_ref` rather than
    // cloning the picture-param array (those `PictureH264` values
    // were moved into `pic`).
    buffers.push(
        enc.context
            .create_buffer(cros_libva::BufferType::EncPictureParameter(
                EncPictureParameter::H264(pic),
            ))
            .map_err(|e| FerricastError::Encoder(format!("pic buffer: {e}")))?,
    );

    // Packed SPS / PPS. cros-libva 0.0.13 doesn't wrap these; we
    // call vaCreateBuffer ourselves and remember the IDs so they
    // get destroyed at the end of the frame.
    if is_idr {
        let (sp, sd) = unsafe {
            create_packed_header(&enc.context, EncPackedHeaderType::Sequence, &enc.sps_nal)?
        };
        packed_buffer_ids.push(sp);
        packed_buffer_ids.push(sd);
        let (pp, pd) = unsafe {
            create_packed_header(&enc.context, EncPackedHeaderType::Picture, &enc.pps_nal)?
        };
        packed_buffer_ids.push(pp);
        packed_buffer_ids.push(pd);
    }

    let slice = build_slice_param(cfg, is_idr, state.idr_pic_id, poc as u16, prev_ref.as_ref());
    buffers.push(
        enc.context
            .create_buffer(cros_libva::BufferType::EncSliceParameter(
                EncSliceParameter::H264(slice),
            ))
            .map_err(|e| FerricastError::Encoder(format!("slice buffer: {e}")))?,
    );

    // Submit. We bypass cros-libva's typestate Picture API because
    // we need to mix our raw-FFI packed-header buffers with the
    // safe ones in a single vaRenderPicture call.
    // cros-libva's `Buffer.id` is a private field; the helper
    // `Buffer::as_id_vec` is the public way to flatten a slice
    // into a `Vec<VABufferID>`.
    let mut all_ids: Vec<VABufferID> = Buffer::as_id_vec(&buffers);
    all_ids.extend_from_slice(&packed_buffer_ids);

    unsafe {
        let dpy = enc.display.handle();
        let ctx = enc.context.id();
        let target = enc.recon[cur_recon_idx].id();

        check_status(vaBeginPicture(dpy, ctx, target))
            .map_err(|s| FerricastError::Encoder(format!("vaBeginPicture: {s:#x}")))?;

        println!("{:?}", all_ids.len());
        let render_status =
            vaRenderPicture(dpy, ctx, all_ids.as_ptr() as *mut _, all_ids.len() as i32);
        if let Err(s) = check_status(render_status) {
            // Best-effort cleanup on render failure.
            let _ = vaEndPicture(dpy, ctx);
            destroy_packed(&enc.display, &packed_buffer_ids);
            return Err(FerricastError::Encoder(format!("vaRenderPicture: {s:#x}")));
        }

        check_status(vaEndPicture(dpy, ctx))
            .map_err(|s| FerricastError::Encoder(format!("vaEndPicture: {s:#x}")))?;

        check_status(vaSyncSurface(dpy, target))
            .map_err(|s| FerricastError::Encoder(format!("vaSyncSurface: {s:#x}")))?;
    }

    // Pull the bitstream out of the coded buffer. Walks the
    // VACodedBufferSegment linked list under the hood.
    let mut bitstream = Vec::with_capacity(64 * 1024);
    {
        let mapped = MappedCodedBuffer::new(&coded)
            .map_err(|e| FerricastError::Encoder(format!("map coded: {e}")))?;
        for seg in mapped.segments() {
            bitstream.extend_from_slice(seg.buf);
        }
    }
    trace!(
        is_idr,
        frame_num,
        bytes = bitstream.len(),
        "VA-API encoded frame"
    );

    // Destroy the packed-header buffers we allocated by hand.
    destroy_packed(&enc.display, &packed_buffer_ids);

    // Update state for the next frame.
    let frame_idx_now = state.frame_idx;
    state.frame_idx += 1;
    state.poc = if is_idr { 2 } else { state.poc + 2 };
    state.prev_recon = Some(cur_recon_idx);
    state.prev_frame_num = frame_num as u32;
    state.prev_poc = poc;
    if is_idr {
        state.idr_pic_id ^= 1;
    }

    Ok((bitstream, is_idr, frame_idx_now, poc))
}

/// Helper kept inline so `state` updates and `frame_num` math live
/// next to each other.
impl FrameState {
    fn frame_idx_at_last_idr(&self, gop: u32) -> u32 {
        (self.frame_idx / gop) * gop
    }
}

fn build_seq_param(cfg: &EncoderCfg) -> EncSequenceParameterBufferH264 {
    let seq_fields = H264EncSeqFields::new(
        /* chroma_format_idc */ 1, /* frame_mbs_only_flag */ 1,
        /* mb_adaptive_frame_field_flag */ 0, /* seq_scaling_matrix_present_flag */ 0,
        /* direct_8x8_inference_flag */ 1, /* log2_max_frame_num_minus4 */ 4,
        /* pic_order_cnt_type */ 0, /* log2_max_pic_order_cnt_lsb_minus4 */ 4,
        /* delta_pic_order_always_zero_flag */ 0,
    );

    let frame_crop = if cfg.height_crop > 0 {
        Some(H264EncFrameCropOffsets {
            left: 0,
            right: 0,
            top: 0,
            bottom: cfg.height_crop,
        })
    } else {
        None
    };

    let vui = Some(H264VuiFields::new(
        /* aspect_ratio_info_present_flag */ 0, /* timing_info_present_flag */ 1,
        /* bitstream_restriction_flag */ 0, /* log2_max_mv_length_horizontal */ 16,
        /* log2_max_mv_length_vertical */ 16, /* fixed_frame_rate_flag */ 1,
        /* low_delay_hrd_flag */ 0, /* motion_vectors_over_pic_boundaries_flag */ 1,
    ));

    EncSequenceParameterBufferH264::new(
        /* seq_parameter_set_id */ 0,
        cfg.level_idc,
        cfg.keyframe_interval,
        cfg.keyframe_interval,
        /* ip_period */ 1,
        cfg.bitrate_bps,
        /* max_num_ref_frames */ 1,
        cfg.width_mbs,
        cfg.height_mbs,
        &seq_fields,
        /* bit_depth_luma_minus8 */ 0,
        /* bit_depth_chroma_minus8 */ 0,
        /* num_ref_frames_in_pic_order_cnt_cycle */ 0,
        /* offset_for_non_ref_pic */ 0,
        /* offset_for_top_to_bottom_field */ 0,
        [0i32; 256],
        frame_crop,
        vui,
        /* aspect_ratio_idc */ 0,
        /* sar_width */ 1,
        /* sar_height */ 1,
        /* num_units_in_tick */ 1,
        /* time_scale */ 2 * cfg.fps,
    )
}

fn build_pic_param(
    cfg: &EncoderCfg,
    curr_pic: &PictureH264,
    reference_frames: [PictureH264; 16],
    coded_buf: VABufferID,
    is_idr: bool,
    frame_num: u16,
) -> EncPictureParameterBufferH264 {
    let pic_fields = H264EncPicFields::new(
        /* idr_pic_flag */ if is_idr { 1 } else { 0 },
        /* reference_pic_flag */ 1,
        /* entropy_coding_mode_flag */ if cfg.cabac { 1 } else { 0 },
        /* weighted_pred_flag */ 0,
        /* weighted_bipred_idc */ 0,
        /* constrained_intra_pred_flag */ 0,
        /* transform_8x8_mode_flag */ 0,
        /* deblocking_filter_control_present_flag */ 1,
        /* redundant_pic_cnt_present_flag */ 0,
        /* pic_order_present_flag */ 0,
        /* pic_scaling_matrix_present_flag */ 0,
    );

    EncPictureParameterBufferH264::new(
        clone_pic(curr_pic),
        reference_frames,
        coded_buf,
        /* pic_parameter_set_id */ 0,
        /* seq_parameter_set_id */ 0,
        /* last_picture */ 0,
        frame_num,
        cfg.initial_qp as u8,
        /* num_ref_idx_l0_active_minus1 */ 0,
        /* num_ref_idx_l1_active_minus1 */ 0,
        /* chroma_qp_index_offset */ 0,
        /* second_chroma_qp_index_offset */ 0,
        &pic_fields,
    )
}

fn build_slice_param(
    cfg: &EncoderCfg,
    is_idr: bool,
    idr_pic_id: u16,
    poc_lsb: u16,
    prev_ref: Option<&PictureH264>,
) -> EncSliceParameterBufferH264 {
    let ref_pic_list_0: [PictureH264; 32] = std::array::from_fn(|i| {
        if i == 0 {
            prev_ref.map(clone_pic).unwrap_or_else(invalid_picture_h264)
        } else {
            invalid_picture_h264()
        }
    });
    let ref_pic_list_1: [PictureH264; 32] = std::array::from_fn(|_| invalid_picture_h264());
    let _ = is_idr;

    let slice_type: u8 = if is_idr { 2 } else { 0 };
    let num_mbs = (cfg.width_mbs as u32) * (cfg.height_mbs as u32);

    EncSliceParameterBufferH264::new(
        /* macroblock_address */ 0,
        num_mbs,
        VA_INVALID_ID,
        slice_type,
        /* pic_parameter_set_id */ 0,
        idr_pic_id,
        poc_lsb,
        /* delta_pic_order_cnt_bottom */ 0,
        [0, 0],
        /* direct_spatial_mv_pred_flag */ 1,
        /* num_ref_idx_active_override_flag */ 0,
        /* num_ref_idx_l0_active_minus1 */ 0,
        /* num_ref_idx_l1_active_minus1 */ 0,
        ref_pic_list_0,
        ref_pic_list_1,
        0,
        0,
        0,
        [0; 32],
        [0; 32],
        0,
        [[0; 2]; 32],
        [[0; 2]; 32],
        0,
        [0; 32],
        [0; 32],
        0,
        [[0; 2]; 32],
        [[0; 2]; 32],
        /* cabac_init_idc */ 0,
        /* slice_qp_delta */ 0,
        /* disable_deblocking_filter_idc */ 0,
        /* slice_alpha_c0_offset_div2 */ 2,
        /* slice_beta_offset_div2 */ 2,
    )
}

fn build_rate_control(cfg: &EncoderCfg) -> EncMiscParameterRateControl {
    EncMiscParameterRateControl::new(
        cfg.bitrate_bps,
        /* target_percentage */ 100, // 100 = CBR
        /* window_size */ 1500,
        cfg.initial_qp,
        /* min_qp */ 1,
        /* basic_unit_size */ 0,
        RcFlags::new(0, 0, 0, 0, 0, 0, 0, 0, 0),
        /* icq_quality_factor */ 0,
        /* max_qp */ 51,
        /* quality_factor */ 0,
        /* target_frame_size */ 0,
    )
}

fn picture_h264(surface_id: VASurfaceID, frame_idx: u32, flags: u32, poc: i32) -> PictureH264 {
    PictureH264::new(surface_id, frame_idx, flags, poc, poc)
}

fn invalid_picture_h264() -> PictureH264 {
    PictureH264::new(VA_INVALID_ID, 0, VA_PICTURE_H264_INVALID, 0, 0)
}

/// `PictureH264: !Clone`, but the underlying VAPictureH264 is plain
/// `Copy`. We hand-roll a clone-equivalent so callers can pass the
/// same picture into both `pic_param.CurrPic` and `slice_param`.
fn clone_pic(p: &PictureH264) -> PictureH264 {
    // Re-build from raw fields. The constructor above takes the
    // same five values we expose elsewhere; we need to peek at the
    // wrapper internals via its Default impl + raw access.
    // Cheaper: just re-construct from scratch using `picture_h264`
    // — both callsites already know what they put in.
    //
    // Since `PictureH264` doesn't expose its inner struct, we pass
    // it through an intermediate FFI struct via transmute. Both
    // sides are `#[repr(transparent)]` over `VAPictureH264`.
    //
    // SAFETY: `PictureH264` is `pub struct PictureH264(VAPictureH264)`
    // (`cros-libva 0.0.13` `src/buffer/h264.rs:10`). Cloning the
    // backing FFI type is sound.
    unsafe {
        let inner: VAPictureH264 = std::mem::transmute_copy(p);
        std::mem::transmute(inner)
    }
}

unsafe fn create_packed_header(
    context: &Rc<Context>,
    htype: EncPackedHeaderType,
    bytes: &[u8],
) -> Result<(VABufferID, VABufferID)> {
    let dpy = context.display().handle();
    let ctx = context.id();

    let mut params = EncPackedHeaderParameter::new(htype, (bytes.len() as u32) * 8, true);
    let mut p_id: VABufferID = 0;
    unsafe {
        check_status(vaCreateBuffer(
            dpy,
            ctx,
            VABufferType::VAEncPackedHeaderParameterBufferType,
            std::mem::size_of::<EncPackedHeaderParameter>() as u32,
            1,
            &mut params as *mut _ as *mut c_void,
            &mut p_id,
        ))
        .map_err(|s| FerricastError::Encoder(format!("packed header param: {s:#x}")))?;
    }

    let mut d_id: VABufferID = 0;
    let st = unsafe {
        check_status(vaCreateBuffer(
            dpy,
            ctx,
            VABufferType::VAEncPackedHeaderDataBufferType,
            bytes.len() as u32,
            1,
            bytes.as_ptr() as *mut c_void,
            &mut d_id,
        ))
    };
    if let Err(s) = st {
        unsafe {
            vaDestroyBuffer(dpy, p_id);
        }
        return Err(FerricastError::Encoder(format!(
            "packed header data: {s:#x}"
        )));
    }
    Ok((p_id, d_id))
}

fn destroy_packed(display: &Rc<Display>, ids: &[VABufferID]) {
    for id in ids {
        unsafe {
            let _ = vaDestroyBuffer(display.handle(), *id);
        }
    }
}

fn check_status(status: VAStatus) -> std::result::Result<(), VAStatus> {
    if status == 0 {
        // VA_STATUS_SUCCESS
        Ok(())
    } else {
        Err(status)
    }
}

// ── DMA-BUF input path ────────────────────────────────────────────
//
// Zero-copy ingest: import the producer's DMA-BUF as a BGRA/RGBA
// surface, then run a VPP `vaProcPipeline` step in-driver to copy +
// colour-convert into `self.input` (NV12). The encoder then proceeds
// exactly like the CPU path — `self.input` is the same surface
// either way, just populated differently.
//
// The imported surface is **per-frame** because the fd may change
// every frame (PipeWire rotates buffers in a pool, WaylandDirect
// allocates fresh per frame). Caching by fd would let us skip
// re-import when the producer reuses fds; left as a follow-up
// because the import is cheap (a few `vaCreateSurfaces` calls).

/// Descriptor passed to `Display::create_surfaces` to import a single-
/// plane DMA-BUF. Only single-plane formats are supported because
/// every capture path emits BGRA/RGBA (multi-plane formats like
/// NV12-from-source would need `num_planes = 2`).
struct DmaBufImport {
    fd: RawFd,
    width: u32,
    height: u32,
    va_fourcc: u32,
    drm_fourcc: u32,
    modifier: u64,
    offset: u32,
    stride: u32,
    size: u32,
}

impl ExternalBufferDescriptor for DmaBufImport {
    const MEMORY_TYPE: MemoryType = MemoryType::DrmPrime2;
    type DescriptorAttribute = VADRMPRIMESurfaceDescriptor;

    fn va_surface_attribute(&mut self) -> Self::DescriptorAttribute {
        // `VADRMPRIMESurfaceDescriptor` is a C struct: zeroing then
        // filling the fields we care about leaves the unused slots
        // (`objects[1..4]`, `layers[1..4]`) at zero, which the
        // driver ignores per `num_objects = 1` / `num_layers = 1`.
        let mut d: VADRMPRIMESurfaceDescriptor = unsafe { std::mem::zeroed() };
        d.fourcc = self.va_fourcc;
        d.width = self.width;
        d.height = self.height;
        d.num_objects = 1;
        d.objects[0].fd = self.fd;
        d.objects[0].size = self.size;
        d.objects[0].drm_format_modifier = self.modifier;
        d.num_layers = 1;
        d.layers[0].drm_format = self.drm_fourcc;
        d.layers[0].num_planes = 1;
        d.layers[0].object_index[0] = 0;
        d.layers[0].offset[0] = self.offset;
        d.layers[0].pitch[0] = self.stride;
        d
    }
}

impl VaapiH264Encoder {
    fn upload_dmabuf_via_vpp(&self, g: &GpuFrame) -> Result<()> {
        let vpp = self.vpp_context.as_ref().ok_or_else(|| {
            FerricastError::Encoder("VA-API: VPP not initialised".into())
        })?;

        let (va_fourcc, drm_fourcc) = match g.format {
            PixelFormat::Bgra => (VA_FOURCC_BGRA, DRM_FORMAT_ARGB8888),
            PixelFormat::Rgba => (VA_FOURCC_RGBA, DRM_FORMAT_ABGR8888),
            other => {
                return Err(FerricastError::Encoder(format!(
                    "VA-API VPP: unsupported source pixel format {other:?}"
                )));
            }
        };

        let import = DmaBufImport {
            fd: g.plane.fd,
            width: g.width,
            height: g.height,
            va_fourcc,
            drm_fourcc,
            modifier: g.plane.modifier,
            offset: g.plane.offset,
            stride: g.plane.stride,
            size: g.plane.size,
        };

        let mut surfaces = self
            .display
            .create_surfaces(
                VA_RT_FORMAT_RGB32,
                Some(va_fourcc),
                g.width,
                g.height,
                None, // No usage hint — VPP source.
                vec![import],
            )
            .map_err(|e| {
                FerricastError::Encoder(format!("VA-API: create_surfaces(import DMA-BUF): {e}"))
            })?;
        let imported = surfaces.pop().expect("we asked for 1");

        // VAProcPipelineParameterBuffer points at `imported` as the
        // source; the destination is implicit (the picture target,
        // = self.input). We pass color standard `None` and let the
        // driver pick BT.709 — captures from any modern Wayland
        // compositor are already in sRGB / BT.709.
        let pipe = ProcPipelineParameterBuffer::new(
            imported.id(),
            None, // surface_region = full
            0_u8, // VAProcColorStandardNone — driver picks BT.601/709.
            None, // output_region = full
            0,    // output_background_color
            0_u8, // VAProcColorStandardNone — driver picks BT.601/709.
            0, // pipeline_flags
            0, // filter_flags
            None,
            None,
            None,
            0, // rotation_state
            None,
            0, // mirror_state
            None,
            0, // input_surface_flag
            0, // output_surface_flag
            ProcColorProperties::default(),
            ProcColorProperties::default(),
            0,
            None,
        );

        let buffer = vpp
            .create_buffer(BufferType::ProcPipelineParameter(pipe))
            .map_err(|e| FerricastError::Encoder(format!("VA-API: vaCreateBuffer(VPP): {e}")))?;

        // Picture lifecycle: New → add buffer → Begin → Render →
        // End → Sync. After sync, self.input has NV12 data.
        let mut pic = Picture::new(g.timestamp_us, Rc::clone(vpp), &self.input);
        pic.add_buffer(buffer);
        let pic = pic
            .begin::<()>()
            .map_err(|e| FerricastError::Encoder(format!("VA-API: vaBeginPicture(VPP): {e}")))?;
        let pic = pic
            .render()
            .map_err(|e| FerricastError::Encoder(format!("VA-API: vaRenderPicture(VPP): {e}")))?;
        let pic = pic
            .end()
            .map_err(|e| FerricastError::Encoder(format!("VA-API: vaEndPicture(VPP): {e}")))?;
        pic.sync::<()>()
            .map_err(|(e, _)| FerricastError::Encoder(format!("VA-API: vaSyncSurface(VPP): {e}")))?;
        // `imported` drops here, freeing the per-frame surface but
        // the driver has already finished with it.
        drop(imported);
        Ok(())
    }
}

// -------------------------------------------------------------------
// Tests-only access to private helpers (compile-time check only).
// -------------------------------------------------------------------
#[cfg(test)]
fn _build_seq_for_test(cfg: &EncoderCfg) -> EncSequenceParameterBufferH264 {
    build_seq_param(cfg)
}
