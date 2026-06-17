//! VA-API HEVC encoder.
//!
//! Mirrors the shape of [`crate::h264::vaapi_impl::VaapiH264Encoder`]:
//! probe → surfaces → context → per-frame parameter buffers → packed
//! headers → coded buffer extraction. Differences vs H.264:
//!
//! * **Profile**: Main (Main10 follows once the surface format / bit
//!   depth plumbing accepts P010 input). `VAProfileHEVCMain` value `17`
//!   from `va.h`; cros-libva exposes the enum but no convenience
//!   constants, so we name it inline.
//! * **CTU size**: 32 (LCU). Width / height counted in CTUs, not MBs.
//! * **Parameter buffers**: HEVC variants of `EncSequenceParameter`,
//!   `EncPictureParameter`, `EncSliceParameter`.
//! * **NAL headers**: 2 bytes (vs 1 for H.264) — packed VPS/SPS/PPS
//!   built in [`super::headers`] / [`super::bitstream`].
//! * **Picture reference type**: `PictureHEVC` instead of `PictureH264`.
//!   IRAP frame uses `IDR_W_RADL` (nal type 19); trailing P frames use
//!   `TRAIL_R` (nal type 1).
//! * **No B-frames, single sub-layer, IPPP**: matches the H.264 path's
//!   profile constraints and keeps the HLS segmenter happy.

use std::cell::RefCell;
use std::os::fd::RawFd;
use std::os::raw::c_void;
use std::path::Path;
use std::rc::Rc;

use bytes::Bytes;
use cros_libva::*;
use ferricast_core::{
    CapturedFrame, Codec, EncodedFrame, EncoderConfig, FerricastError, GpuFrame, H265Profile,
    PixelFormat, Result, VideoEncoder,
};
use tracing::{debug, info, trace, warn};

use super::headers::{self, ConformanceWindow, StreamParams};

const RENDER_NODES: &[&str] = &[
    "/dev/dri/renderD128",
    "/dev/dri/renderD129",
    "/dev/dri/renderD130",
];

const VA_FOURCC_NV12: u32 = 0x3231564E;
const VA_FOURCC_BGRA: u32 = 0x41524742;
const VA_FOURCC_RGBA: u32 = 0x41424752;
const DRM_FORMAT_ARGB8888: u32 = 0x34325241;
const DRM_FORMAT_ABGR8888: u32 = 0x34324241;
const VA_RT_FORMAT_YUV420: u32 = 0x01;
const VA_RT_FORMAT_RGB32: u32 = 0x04;

const RECON_POOL: usize = 2;
const CODED_BUFFER_SIZE: usize = 8 * 1024 * 1024;

const HEVC_PROFILE_MAIN_IDC: u8 = 1;
const HEVC_PROFILE_MAIN10_IDC: u8 = 2;
const HEVC_NAL_TRAIL_R: u8 = 1;
const HEVC_NAL_IDR_W_RADL: u8 = 19;

/// VA profile enum values from `va.h`. cros-libva re-exports them
/// through `VAProfile::Type`; we name the two we need directly so we
/// don't tie the encoder to a specific path through the cros-libva
/// re-export tree.
const VA_PROFILE_HEVC_MAIN: VAProfile::Type = 17;
const VA_PROFILE_HEVC_MAIN10: VAProfile::Type = 18;

pub struct VaapiH265Encoder {
    display: Rc<Display>,
    #[allow(dead_code)]
    config: Config,
    context: Rc<Context>,

    /// VPP plumbing for zero-copy DMA-BUF → NV12 conversion. Same
    /// fallback story as the H.264 path: `None` means we still work
    /// but DMA-BUF frames take the CPU path.
    #[allow(dead_code)]
    vpp_config: Option<Config>,
    vpp_context: Option<Rc<Context>>,

    input: Surface<()>,
    recon: Vec<Surface<()>>,

    cfg: EncoderCfg,

    vps_nal: Vec<u8>,
    sps_nal: Vec<u8>,
    pps_nal: Vec<u8>,

    state: RefCell<FrameState>,
}

#[derive(Clone, Copy)]
struct EncoderCfg {
    profile_idc: u8,
    width: u32,
    height: u32,
    /// CTU = 32 luma samples. Width / height in CTUs (rounded up).
    width_ctus: u16,
    height_ctus: u16,
    /// Conformance-window crop in luma samples needed when `height`
    /// isn't a multiple of the CTU size.
    height_crop: u32,
    fps: u32,
    bitrate_bps: u32,
    keyframe_interval: u32,
    initial_qp: u32,
    level_idc: u8,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
}

impl EncoderCfg {
    /// CTU-aligned encoded height. The user-visible height may be
    /// smaller and is communicated via the SPS conformance window.
    fn padded_height(&self) -> u32 {
        (self.height_ctus as u32) * 32
    }
}

#[derive(Default)]
struct FrameState {
    frame_idx: u32,
    /// HEVC POC counts in frames (not 2× like H.264 with
    /// pic_order_cnt_type 0 / interlaced semantics).
    poc: i32,
    next_recon: usize,
    prev_recon: Option<usize>,
    prev_poc: i32,
    pending_keyframe: bool,
}

impl FrameState {
    fn frame_idx_at_last_idr(&self, gop: u32) -> u32 {
        (self.frame_idx / gop) * gop
    }
}

unsafe impl Send for VaapiH265Encoder {}

impl VaapiH265Encoder {
    pub fn probe() -> Result<Self> {
        Self::probe_with(EncoderConfig::default())
    }

    pub fn probe_with(cfg: EncoderConfig) -> Result<Self> {
        if !matches!(cfg.pixel_format, PixelFormat::Bgra | PixelFormat::Rgba) {
            return Err(FerricastError::Encoder(format!(
                "VA-API HEVC: input pixel format {:?} not supported (need Bgra/Rgba)",
                cfg.pixel_format
            )));
        }

        let display = open_render_node()
            .ok_or_else(|| FerricastError::Encoder("VA-API HEVC: no usable DRM render node".into()))?;
        let vendor = display.query_vendor_string().unwrap_or_default();
        debug!(%vendor, "VA-API display opened for HEVC encode");

        let supported = display
            .query_config_profiles()
            .map_err(|e| FerricastError::Encoder(format!("query_config_profiles: {e}")))?;

        // Prefer Main10 only when the caller asks for it (so 8-bit
        // capture pipelines don't accidentally get a 10-bit encoder
        // that won't accept their surfaces) and the driver supports
        // the encode entrypoint at Main10.
        let want_main10 = matches!(cfg.max_h265_profile, Some(H265Profile::Main10));
        let (profile, profile_idc, bit_depth) = if want_main10
            && supported.contains(&VA_PROFILE_HEVC_MAIN10)
            && profile_has_enc_slice(&display, VA_PROFILE_HEVC_MAIN10)
        {
            (VA_PROFILE_HEVC_MAIN10, HEVC_PROFILE_MAIN10_IDC, 2_u8)
        } else if supported.contains(&VA_PROFILE_HEVC_MAIN)
            && profile_has_enc_slice(&display, VA_PROFILE_HEVC_MAIN)
        {
            (VA_PROFILE_HEVC_MAIN, HEVC_PROFILE_MAIN_IDC, 0_u8)
        } else {
            return Err(FerricastError::Encoder(
                "VA-API HEVC: no supported encode profile (need Main or Main10)".into(),
            ));
        };
        info!(?profile, %vendor, "VA-API HEVC encoder selected");

        let cfg = build_encoder_cfg(profile_idc, bit_depth, &cfg)?;

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
            .map_err(|e| FerricastError::Encoder(format!("vaCreateConfig(HEVC): {e}")))?;

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

        let context = display
            .create_context(
                &cfg_handle,
                cfg.width,
                cfg.padded_height(),
                Some(&recon),
                /* progressive = */ true,
            )
            .map_err(|e| FerricastError::Encoder(format!("vaCreateContext(HEVC): {e}")))?;

        let stream = StreamParams {
            profile_idc: cfg.profile_idc,
            level_idc: cfg.level_idc,
            tier_flag: 0,
            width: cfg.width,
            height: cfg.height,
            fps: cfg.fps,
            conformance_window: if cfg.height_crop > 0 {
                Some(ConformanceWindow {
                    left: 0,
                    right: 0,
                    top: 0,
                    bottom: cfg.height_crop,
                })
            } else {
                None
            },
            bit_depth_luma_minus8: cfg.bit_depth_luma_minus8,
            bit_depth_chroma_minus8: cfg.bit_depth_chroma_minus8,
            max_num_ref_frames: 1,
        };
        let vps_nal = headers::build_vps(&stream);
        let sps_nal = headers::build_sps(&stream, /* min_cb_log2_minus3 */ 0, /* diff_max_min_cb_log2 */ 3);
        let pps_nal = headers::build_pps(cfg.initial_qp as i32 - 26);

        let (vpp_config, vpp_context) = match build_vpp(&display, &cfg) {
            Ok((c, ctx)) => {
                info!("VA-API HEVC VPP up; DMA-BUF input will be zero-copy");
                (Some(c), Some(ctx))
            }
            Err(e) => {
                warn!(error = %e, "VA-API HEVC VPP unavailable; DMA-BUF frames will fall back to CPU path");
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
            vps_nal,
            sps_nal,
            pps_nal,
            state: RefCell::new(FrameState::default()),
        })
    }
}

fn build_vpp(display: &Rc<Display>, cfg: &EncoderCfg) -> Result<(Config, Rc<Context>)> {
    let config = display
        .create_config(vec![], VAProfile::VAProfileNone, VAEntrypoint::VAEntrypointVideoProc)
        .map_err(|e| FerricastError::Encoder(format!("vaCreateConfig(VPP): {e}")))?;
    let context = display
        .create_context::<()>(&config, cfg.width, cfg.padded_height(), None, true)
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

fn build_encoder_cfg(
    profile_idc: u8,
    bit_depth: u8,
    cfg: &EncoderConfig,
) -> Result<EncoderCfg> {
    let width = cfg.width.max(32);
    let height = cfg.height.max(32);
    // CTU = 32 luma samples. Width / height counted in CTUs, rounded
    // up. The encoder's surface is the CTU-padded height; the user's
    // crop comes back through the SPS conformance window.
    let width_ctus = ((width + 31) / 32) as u16;
    let height_ctus = ((height + 31) / 32) as u16;
    let padded_h = (height_ctus as u32) * 32;
    let height_crop = padded_h - height;
    let level_idc = pick_level(width, height, cfg.fps);

    Ok(EncoderCfg {
        profile_idc,
        width,
        height,
        width_ctus,
        height_ctus,
        height_crop,
        fps: cfg.fps.max(1),
        bitrate_bps: (cfg.bitrate_kbps.max(1) as u32).saturating_mul(1000),
        keyframe_interval: cfg.keyframe_interval_frames(),
        initial_qp: 26,
        level_idc,
        bit_depth_luma_minus8: bit_depth,
        bit_depth_chroma_minus8: bit_depth,
    })
}

/// HEVC level table (Annex A, Table A.6). Levels times 30. We pick
/// the smallest level that contains the requested resolution+framerate.
fn pick_level(width: u32, height: u32, fps: u32) -> u8 {
    let luma_samples = width * height;
    let samples_per_sec = luma_samples * fps.max(1);
    if samples_per_sec <= 33_177_600 {
        90 // 3.0
    } else if samples_per_sec <= 66_846_720 {
        93 // 3.1
    } else if samples_per_sec <= 133_693_440 {
        120 // 4.0
    } else if samples_per_sec <= 267_386_880 {
        123 // 4.1
    } else if samples_per_sec <= 534_773_760 {
        150 // 5.0
    } else if samples_per_sec <= 1_069_547_520 {
        153 // 5.1
    } else {
        156 // 5.2
    }
}

impl VideoEncoder for VaapiH265Encoder {
    const CODEC: Codec = Codec::H265;

    fn configure(&mut self, _config: &EncoderConfig) -> Result<()> {
        // Stream-invariant configuration is committed in `probe_with`.
        // Live resize would need to tear down surfaces / context /
        // config; we don't support that today.
        Ok(())
    }

    fn encode(&mut self, frame: CapturedFrame) -> Result<EncodedFrame> {
        let target_recon_idx = self.state.borrow().next_recon;
        let target_surface = &self.recon[target_recon_idx];

        let timestamp_us = frame.timestamp_us();
        match frame {
            CapturedFrame::Gpu(g) if self.vpp_context.is_some() => {
                self.upload_dmabuf_via_vpp(target_recon_idx, &g)?;
            }
            other => {
                let raw = other.into_cpu()?;
                if !matches!(raw.format, PixelFormat::Bgra | PixelFormat::Rgba) {
                    return Err(FerricastError::Encoder(format!(
                        "VA-API HEVC: unexpected runtime pixel format {:?}",
                        raw.format
                    )));
                }
                upload_bgra_to_nv12(target_surface, &self.cfg, &raw.data, raw.stride as usize)?;
            }
        }

        let (encoded_bytes, is_keyframe, frame_idx, poc) = {
            let mut state = self.state.borrow_mut();
            run_encode(self, &mut state)?
        };

        Ok(EncodedFrame {
            codec: Codec::H265,
            data: Bytes::from(encoded_bytes),
            timestamp_us,
            duration_us: Some(1_000_000 / self.cfg.fps as u64),
            is_keyframe,
            pts_dts: (poc as u64, frame_idx as u64),
        })
    }

    fn flush(self) -> Result<Vec<EncodedFrame>> {
        Ok(Vec::new())
    }

    fn request_keyframe(&mut self) {
        self.state.borrow_mut().pending_keyframe = true;
    }

    fn get_headers(&mut self) -> Result<Vec<u8>> {
        let mut out =
            Vec::with_capacity(self.vps_nal.len() + self.sps_nal.len() + self.pps_nal.len());
        out.extend_from_slice(&self.vps_nal);
        out.extend_from_slice(&self.sps_nal);
        out.extend_from_slice(&self.pps_nal);
        Ok(out)
    }
}

fn upload_bgra_to_nv12(
    surface: &Surface<()>,
    cfg: &EncoderCfg,
    bgra: &[u8],
    bgra_stride: usize,
) -> Result<()> {
    let w = cfg.width;
    let h = cfg.height;
    let padded_h = cfg.padded_height();

    let mut image = match Image::derive_from(surface, (w, padded_h)) {
        Ok(img) => img,
        Err(_e) => {
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

    let im = image.image();
    let y_off = im.offsets[0] as usize;
    let uv_off = im.offsets[1] as usize;
    let y_stride = im.pitches[0] as usize;
    let uv_stride = im.pitches[1] as usize;

    let buf = image.as_mut();
    let len = buf.len();
    if y_off + y_stride * (padded_h as usize) > len
        || uv_off + uv_stride * ((padded_h / 2) as usize) > len
    {
        return Err(FerricastError::Encoder(
            "VA-API HEVC NV12 image buffer smaller than expected planes".into(),
        ));
    }
    let (y_plane, uv_plane) = if y_off < uv_off {
        let (lo, hi) = buf.split_at_mut(uv_off);
        (&mut lo[y_off..], hi)
    } else {
        let (lo, hi) = buf.split_at_mut(y_off);
        (hi, &mut lo[uv_off..])
    };

    crate::h264::yuv::bgra_to_nv12(
        bgra,
        bgra_stride.max((w * 4) as usize),
        w,
        h,
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
    );

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
}

fn run_encode(enc: &VaapiH265Encoder, state: &mut FrameState) -> Result<(Vec<u8>, bool, u32, i32)> {
    let cfg = &enc.cfg;
    let is_idr =
        state.frame_idx % cfg.keyframe_interval == 0 || std::mem::take(&mut state.pending_keyframe);
    let poc = if is_idr {
        0
    } else {
        // POC counts frames since last IDR.
        (state.frame_idx - state.frame_idx_at_last_idr(cfg.keyframe_interval)) as i32
    };

    let cur_recon_idx = state.next_recon;
    let prev_recon_idx = state.prev_recon;
    state.next_recon = (cur_recon_idx + 1) % enc.recon.len();

    let curr_pic = PictureHEVC::new(enc.recon[cur_recon_idx].id(), poc, 0);
    let prev_ref: Option<PictureHEVC> = if !is_idr {
        prev_recon_idx.map(|prev_idx| {
            // VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE (0x10) marks the
            // single short-term-before reference our IPPP slot uses.
            PictureHEVC::new(enc.recon[prev_idx].id(), state.prev_poc, 0x10)
        })
    } else {
        None
    };

    let reference_frames: [PictureHEVC; 15] = std::array::from_fn(|i| {
        if i == 0 {
            prev_ref
                .as_ref()
                .map(clone_pic_hevc)
                .unwrap_or_else(invalid_picture_hevc)
        } else {
            invalid_picture_hevc()
        }
    });

    let coded = enc
        .context
        .create_enc_coded(CODED_BUFFER_SIZE)
        .map_err(|e| FerricastError::Encoder(format!("vaCreateBuffer(coded): {e}")))?;

    let mut buffers: Vec<Buffer> = Vec::with_capacity(8);
    let mut packed_buffer_ids: Vec<VABufferID> = Vec::new();

    if is_idr {
        let seq = build_seq_param(cfg);
        buffers.push(
            enc.context
                .create_buffer(cros_libva::BufferType::EncSequenceParameter(
                    EncSequenceParameter::HEVC(seq),
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

    let pic = build_pic_param(cfg, &curr_pic, reference_frames, coded.id(), is_idr);
    buffers.push(
        enc.context
            .create_buffer(cros_libva::BufferType::EncPictureParameter(
                EncPictureParameter::HEVC(pic),
            ))
            .map_err(|e| FerricastError::Encoder(format!("pic buffer: {e}")))?,
    );

    if is_idr {
        // VPS / SPS / PPS packed headers. Per the VA spec each one is
        // submitted as a Sequence-type packed header — there's no
        // dedicated VPS packed-header type.
        for nal in [&enc.vps_nal, &enc.sps_nal, &enc.pps_nal] {
            let (p, d) = unsafe {
                create_packed_header(&enc.context, EncPackedHeaderType::Sequence, nal)?
            };
            packed_buffer_ids.push(p);
            packed_buffer_ids.push(d);
        }
    }

    let slice = build_slice_param(cfg, is_idr, prev_ref.as_ref());
    buffers.push(
        enc.context
            .create_buffer(cros_libva::BufferType::EncSliceParameter(
                EncSliceParameter::HEVC(slice),
            ))
            .map_err(|e| FerricastError::Encoder(format!("slice buffer: {e}")))?,
    );

    let mut all_ids: Vec<VABufferID> = Buffer::as_id_vec(&buffers);
    all_ids.extend_from_slice(&packed_buffer_ids);

    unsafe {
        let dpy = enc.display.handle();
        let ctx = enc.context.id();
        let target = enc.recon[cur_recon_idx].id();

        check_status(vaBeginPicture(dpy, ctx, target))
            .map_err(|s| FerricastError::Encoder(format!("vaBeginPicture(HEVC): {s:#x}")))?;

        let render_status =
            vaRenderPicture(dpy, ctx, all_ids.as_ptr() as *mut _, all_ids.len() as i32);
        if let Err(s) = check_status(render_status) {
            let _ = vaEndPicture(dpy, ctx);
            destroy_packed(&enc.display, &packed_buffer_ids);
            return Err(FerricastError::Encoder(format!(
                "vaRenderPicture(HEVC): {s:#x}"
            )));
        }

        check_status(vaEndPicture(dpy, ctx))
            .map_err(|s| FerricastError::Encoder(format!("vaEndPicture(HEVC): {s:#x}")))?;
        check_status(vaSyncSurface(dpy, target))
            .map_err(|s| FerricastError::Encoder(format!("vaSyncSurface(HEVC): {s:#x}")))?;
    }

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
        poc,
        bytes = bitstream.len(),
        "VA-API HEVC encoded frame"
    );

    destroy_packed(&enc.display, &packed_buffer_ids);

    let frame_idx_now = state.frame_idx;
    state.frame_idx += 1;
    state.prev_recon = Some(cur_recon_idx);
    state.prev_poc = poc;
    state.poc = poc;

    Ok((bitstream, is_idr, frame_idx_now, poc))
}

fn build_seq_param(cfg: &EncoderCfg) -> EncSequenceParameterBufferHEVC {
    let seq_fields = HEVCEncSeqFields::new(
        /* chroma_format_idc */ 1,
        /* separate_colour_plane_flag */ 0,
        /* bit_depth_luma_minus8 */ cfg.bit_depth_luma_minus8 as u32,
        /* bit_depth_chroma_minus8 */ cfg.bit_depth_chroma_minus8 as u32,
        /* scaling_list_enabled_flag */ 0,
        /* strong_intra_smoothing_enabled_flag */ 1,
        /* amp_enabled_flag */ 1,
        /* sample_adaptive_offset_enabled_flag */ 1,
        /* pcm_enabled_flag */ 0,
        /* pcm_loop_filter_disabled_flag */ 0,
        /* sps_temporal_mvp_enabled_flag */ 1,
        /* low_delay_seq */ 1,
        /* hierachical_flag */ 0,
    );

    let vui = Some(HevcEncVuiFields::new(
        /* aspect_ratio_info_present_flag */ 0,
        /* neutral_chroma_indication_flag */ 0,
        /* field_seq_flag */ 0,
        /* vui_timing_info_present_flag */ 1,
        /* bitstream_restriction_flag */ 0,
        /* tiles_fixed_structure_flag */ 0,
        /* motion_vectors_over_pic_boundaries_flag */ 1,
        /* restricted_ref_pic_lists_flag */ 0,
        /* log2_max_mv_length_horizontal */ 15,
        /* log2_max_mv_length_vertical */ 15,
    ));

    let scc = HevcEncSeqSccFields::new(/* palette_mode_enabled_flag */ 0);

    EncSequenceParameterBufferHEVC::new(
        cfg.profile_idc,
        cfg.level_idc,
        /* general_tier_flag */ 0,
        cfg.keyframe_interval,
        cfg.keyframe_interval,
        /* ip_period */ 1,
        cfg.bitrate_bps,
        cfg.width as u16,
        cfg.height as u16,
        &seq_fields,
        /* log2_min_luma_coding_block_size_minus3 */ 0,
        /* log2_diff_max_min_luma_coding_block_size */ 3,
        /* log2_min_transform_block_size_minus2 */ 0,
        /* log2_diff_max_min_transform_block_size */ 3,
        /* max_transform_hierarchy_depth_inter */ 3,
        /* max_transform_hierarchy_depth_intra */ 3,
        /* pcm_sample_bit_depth_luma_minus1 */ 0,
        /* pcm_sample_bit_depth_chroma_minus1 */ 0,
        /* log2_min_pcm_luma_coding_block_size_minus3 */ 0,
        /* log2_max_pcm_luma_coding_block_size_minus3 */ 0,
        vui,
        /* aspect_ratio_idc */ 0,
        /* sar_width */ 1,
        /* sar_height */ 1,
        /* vui_num_units_in_tick */ 1,
        /* vui_time_scale */ cfg.fps,
        /* min_spatial_segmentation_idc */ 0,
        /* max_bytes_per_pic_denom */ 0,
        /* max_bits_per_min_cu_denom */ 0,
        &scc,
    )
}

fn build_pic_param(
    cfg: &EncoderCfg,
    curr_pic: &PictureHEVC,
    reference_frames: [PictureHEVC; 15],
    coded_buf: VABufferID,
    is_idr: bool,
) -> EncPictureParameterBufferHEVC {
    // coding_type: 1 = I (IRAP), 2 = P, 3 = B. IPPP: IDR = 1, rest = 2.
    let coding_type: u32 = if is_idr { 1 } else { 2 };
    let nal_unit_type = if is_idr { HEVC_NAL_IDR_W_RADL } else { HEVC_NAL_TRAIL_R };

    let pic_fields = HEVCEncPicFields::new(
        /* idr_pic_flag */ if is_idr { 1 } else { 0 },
        coding_type,
        /* reference_pic_flag */ 1,
        /* dependent_slice_segments_enabled_flag */ 0,
        /* sign_data_hiding_enabled_flag */ 0,
        /* constrained_intra_pred_flag */ 0,
        /* transform_skip_enabled_flag */ 0,
        /* cu_qp_delta_enabled_flag */ 0,
        /* weighted_pred_flag */ 0,
        /* weighted_bipred_flag */ 0,
        /* transquant_bypass_enabled_flag */ 0,
        /* tiles_enabled_flag */ 0,
        /* entropy_coding_sync_enabled_flag */ 0,
        /* loop_filter_across_tiles_enabled_flag */ 0,
        /* pps_loop_filter_across_slices_enabled_flag */ 1,
        /* scaling_list_data_present_flag */ 0,
        /* screen_content_flag */ 0,
        /* enable_gpu_weighted_prediction */ 0,
        /* no_output_of_prior_pics_flag */ 0,
    );
    let scc_fields = HevcEncPicSccFields::new(/* pps_curr_pic_ref_enabled_flag */ 0);

    EncPictureParameterBufferHEVC::new(
        clone_pic_hevc(curr_pic),
        reference_frames,
        coded_buf,
        /* collocated_ref_pic_index */ if is_idr { 0xff } else { 0 },
        /* last_picture */ 0,
        cfg.initial_qp as u8,
        /* diff_cu_qp_delta_depth */ 0,
        /* pps_cb_qp_offset */ 0,
        /* pps_cr_qp_offset */ 0,
        /* num_tile_columns_minus1 */ 0,
        /* num_tile_rows_minus1 */ 0,
        [0u8; 19],
        [0u8; 21],
        /* log2_parallel_merge_level_minus2 */ 0,
        /* ctu_max_bitsize_allowed */ 0,
        /* num_ref_idx_l0_default_active_minus1 */ 0,
        /* num_ref_idx_l1_default_active_minus1 */ 0,
        /* slice_pic_parameter_set_id */ 0,
        nal_unit_type,
        &pic_fields,
        /* hierarchical_level_plus1 */ 0,
        /* va_byte_reserved */ 0,
        &scc_fields,
    )
}

fn build_slice_param(
    cfg: &EncoderCfg,
    is_idr: bool,
    prev_ref: Option<&PictureHEVC>,
) -> EncSliceParameterBufferHEVC {
    let ref_pic_list0: [PictureHEVC; 15] = std::array::from_fn(|i| {
        if i == 0 {
            prev_ref
                .map(clone_pic_hevc)
                .unwrap_or_else(invalid_picture_hevc)
        } else {
            invalid_picture_hevc()
        }
    });
    let ref_pic_list1: [PictureHEVC; 15] = std::array::from_fn(|_| invalid_picture_hevc());

    let slice_fields = HevcEncSliceFields::new(
        /* last_slice_of_pic_flag */ 1,
        /* dependent_slice_segment_flag */ 0,
        /* colour_plane_id */ 0,
        /* slice_temporal_mvp_enabled_flag */ if is_idr { 0 } else { 1 },
        /* slice_sao_luma_flag */ 1,
        /* slice_sao_chroma_flag */ 1,
        /* num_ref_idx_active_override_flag */ 0,
        /* mvd_l1_zero_flag */ 0,
        /* cabac_init_flag */ 0,
        /* slice_deblocking_filter_disabled_flag */ 0,
        /* slice_loop_filter_across_slices_enabled_flag */ 1,
        /* collocated_from_l0_flag */ 1,
    );

    // slice_type: 2 = I, 1 = P, 0 = B.
    let slice_type: u8 = if is_idr { 2 } else { 1 };
    let num_ctu = (cfg.width_ctus as u32) * (cfg.height_ctus as u32);

    EncSliceParameterBufferHEVC::new(
        /* slice_segment_address */ 0,
        num_ctu,
        slice_type,
        /* slice_pic_parameter_set_id */ 0,
        /* num_ref_idx_l0_active_minus1 */ 0,
        /* num_ref_idx_l1_active_minus1 */ 0,
        ref_pic_list0,
        ref_pic_list1,
        /* luma_log2_weight_denom */ 0,
        /* delta_chroma_log2_weight_denom */ 0,
        [0; 15],
        [0; 15],
        [[0; 2]; 15],
        [[0; 2]; 15],
        [0; 15],
        [0; 15],
        [[0; 2]; 15],
        [[0; 2]; 15],
        /* max_num_merge_cand */ 5,
        /* slice_qp_delta */ 0,
        /* slice_cb_qp_offset */ 0,
        /* slice_cr_qp_offset */ 0,
        /* slice_beta_offset_div2 */ 0,
        /* slice_tc_offset_div2 */ 0,
        &slice_fields,
        /* pred_weight_table_bit_offset */ 0,
        /* pred_weight_table_bit_length */ 0,
    )
}

fn build_rate_control(cfg: &EncoderCfg) -> EncMiscParameterRateControl {
    EncMiscParameterRateControl::new(
        cfg.bitrate_bps,
        /* target_percentage */ 100,
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

fn invalid_picture_hevc() -> PictureHEVC {
    PictureHEVC::new(VA_INVALID_ID, 0, 0x00000001) // VA_PICTURE_HEVC_INVALID
}

/// `PictureHEVC` is `Copy` (`Default + Copy + Clone + PartialEq + Eq`
/// derived in cros-libva), but the convenience clone-by-value helper
/// makes the callers symmetric with the H.264 path. Stays inline so
/// the optimiser folds it into a `mov`.
fn clone_pic_hevc(p: &PictureHEVC) -> PictureHEVC {
    *p
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
        Ok(())
    } else {
        Err(status)
    }
}

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

impl VaapiH265Encoder {
    fn upload_dmabuf_via_vpp(&self, target_recon_idx: usize, g: &GpuFrame) -> Result<()> {
        let vpp = self
            .vpp_context
            .as_ref()
            .ok_or_else(|| FerricastError::Encoder("VA-API HEVC: VPP not initialised".into()))?;

        let (va_fourcc, drm_fourcc) = match g.format {
            PixelFormat::Bgra => (VA_FOURCC_BGRA, DRM_FORMAT_ARGB8888),
            PixelFormat::Rgba => (VA_FOURCC_RGBA, DRM_FORMAT_ABGR8888),
            other => {
                return Err(FerricastError::Encoder(format!(
                    "VA-API HEVC VPP: unsupported source pixel format {other:?}"
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
                None,
                vec![import],
            )
            .map_err(|e| {
                FerricastError::Encoder(format!(
                    "VA-API HEVC: create_surfaces(import DMA-BUF): {e}"
                ))
            })?;
        let imported = surfaces.pop().expect("we asked for 1");

        let pipe = ProcPipelineParameterBuffer::new(
            imported.id(),
            None,
            0_u8,
            None,
            0,
            0_u8,
            0,
            0,
            None,
            None,
            None,
            0,
            None,
            0,
            None,
            0,
            0,
            ProcColorProperties::default(),
            ProcColorProperties::default(),
            0,
            None,
        );

        let buffer = vpp
            .create_buffer(BufferType::ProcPipelineParameter(pipe))
            .map_err(|e| FerricastError::Encoder(format!("VA-API HEVC: vaCreateBuffer(VPP): {e}")))?;

        let dest = &self.recon[target_recon_idx];
        let mut pic = Picture::new(g.timestamp_us, Rc::clone(vpp), dest);
        pic.add_buffer(buffer);
        let pic = pic
            .begin::<()>()
            .map_err(|e| FerricastError::Encoder(format!("VA-API HEVC: vaBeginPicture(VPP): {e}")))?;
        let pic = pic
            .render()
            .map_err(|e| FerricastError::Encoder(format!("VA-API HEVC: vaRenderPicture(VPP): {e}")))?;
        let pic = pic
            .end()
            .map_err(|e| FerricastError::Encoder(format!("VA-API HEVC: vaEndPicture(VPP): {e}")))?;
        pic.sync::<()>()
            .map_err(|(e, _)| FerricastError::Encoder(format!("VA-API HEVC: vaSyncSurface(VPP): {e}")))?;
        drop(imported);
        Ok(())
    }
}
