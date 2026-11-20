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
use crate::frame_view::FrameView;
use crate::gop::GopState;

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

    /// Size of the surfaces the encoder reads from.
    fn coded_size(&self) -> (u32, u32) {
        (self.width, self.padded_height())
    }
}

struct FrameState {
    gop: GopState,
    /// Pictures encoded so far; reported as DTS.
    frame_idx: u32,
    /// Round-robin index into `recon` for the next picture.
    next_recon: usize,
    /// `(recon index, POC)` of the reference for the next P picture.
    prev_ref: Option<(usize, i32)>,
}

impl FrameState {
    fn new(keyframe_interval: u32) -> Self {
        Self {
            gop: GopState::new(keyframe_interval),
            frame_idx: 0,
            next_recon: 0,
            prev_ref: None,
        }
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

        let display = open_render_node().ok_or_else(|| {
            FerricastError::Encoder("VA-API HEVC: no usable DRM render node".into())
        })?;
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
        let sps_nal = headers::build_sps(
            &stream, /* min_cb_log2_minus3 */ 0, /* diff_max_min_cb_log2 */ 3,
        );
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
            recon,
            cfg,
            vps_nal,
            sps_nal,
            pps_nal,
            state: RefCell::new(FrameState::new(cfg.keyframe_interval)),
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
        Ok(eps) => eps.contains(&VAEntrypoint::VAEntrypointEncSlice),
        Err(_) => false,
    }
}

fn build_encoder_cfg(profile_idc: u8, bit_depth: u8, cfg: &EncoderConfig) -> Result<EncoderCfg> {
    let width = cfg.width.max(32);
    let height = cfg.height.max(32);
    // CTU = 32 luma samples. Width / height counted in CTUs, rounded
    // up. The encoder's surface is the CTU-padded height; the user's
    // crop comes back through the SPS conformance window.
    let width_ctus = width.div_ceil(32) as u16;
    let height_ctus = height.div_ceil(32) as u16;
    let padded_h = (height_ctus as u32) * 32;
    let height_crop = padded_h - height;
    let bitrate_bps = cfg.bitrate_kbps.max(1).saturating_mul(1000);
    let level_idc = pick_level(width, height, cfg.fps.max(1), bitrate_bps);

    Ok(EncoderCfg {
        profile_idc,
        width,
        height,
        width_ctus,
        height_ctus,
        height_crop,
        fps: cfg.fps.max(1),
        bitrate_bps,
        keyframe_interval: cfg.keyframe_interval_frames(),
        initial_qp: 26,
        level_idc,
        bit_depth_luma_minus8: bit_depth,
        bit_depth_chroma_minus8: bit_depth,
    })
}

/// HEVC Tables A.8 / A.9, Main tier: `(general_level_idc, MaxLumaPs,
/// MaxLumaSr, MaxBR)`; level_idc is the level × 30, MaxBR in 1000 bit/s
/// (CpbBrVclFactor of Main / Main10).
const HEVC_LEVELS: &[(u8, u64, u64, u64)] = &[
    (30, 36_864, 552_960, 128),
    (60, 122_880, 3_686_400, 1_500),
    (63, 245_760, 7_372_800, 3_000),
    (90, 552_960, 16_588_800, 6_000),
    (93, 983_040, 33_177_600, 10_000),
    (120, 2_228_224, 66_846_720, 12_000),
    (123, 2_228_224, 133_693_440, 20_000),
    (150, 8_912_896, 267_386_880, 25_000),
    (153, 8_912_896, 534_773_760, 40_000),
    (156, 8_912_896, 1_069_547_520, 60_000),
    (180, 35_651_584, 1_069_547_520, 60_000),
    (183, 35_651_584, 2_139_095_040, 120_000),
    (186, 35_651_584, 4_278_190_080, 240_000),
];

/// Smallest HEVC level (A.4.1) whose MaxLumaPs, MaxLumaSr, picture
/// dimensions (each ≤ √(8·MaxLumaPs)) and MaxBR admit the coded
/// picture size `width × height`. Anything beyond clamps to 6.2.
fn pick_level(width: u32, height: u32, fps: u32, bitrate_bps: u32) -> u8 {
    let (w, h) = (u64::from(width), u64::from(height));
    let pic_size = w * h;
    let sample_rate = pic_size * u64::from(fps);
    HEVC_LEVELS
        .iter()
        .find(|&&(_, max_ps, max_sr, max_br)| {
            let max_dim = (8 * max_ps).isqrt();
            pic_size <= max_ps
                && sample_rate <= max_sr
                && w <= max_dim
                && h <= max_dim
                && u64::from(bitrate_bps) <= max_br * 1000
        })
        .map_or(186, |&(level, ..)| level)
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
        let result = self.encode_frame(frame);
        if result.is_err() {
            // Never predict from a picture the driver may not have
            // finished; restart the GOP.
            self.state.borrow_mut().gop.request_idr();
        }
        result
    }

    fn flush(self) -> Result<Vec<EncodedFrame>> {
        Ok(Vec::new())
    }

    fn request_keyframe(&mut self) {
        self.state.borrow_mut().gop.request_idr();
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

impl VaapiH265Encoder {
    fn encode_frame(&mut self, frame: CapturedFrame) -> Result<EncodedFrame> {
        let target_recon_idx = self.state.borrow().next_recon;
        let target_surface = &self.recon[target_recon_idx];

        let timestamp_us = frame.timestamp_us();
        match frame {
            CapturedFrame::Gpu(g) if self.vpp_context.is_some() => {
                self.upload_dmabuf_via_vpp(target_recon_idx, &g)?;
            }
            other => {
                let raw = other.into_cpu()?;
                let view = FrameView::new(&raw, self.cfg.width, self.cfg.height)?;
                crate::h264::yuv::upload_to_surface(target_surface, self.cfg.coded_size(), &view)?;
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
            pts_dts: (poc.unsigned_abs().into(), frame_idx.into()),
        })
    }
}

fn run_encode(enc: &VaapiH265Encoder, state: &mut FrameState) -> Result<(Vec<u8>, bool, u32, i32)> {
    let cfg = &enc.cfg;
    // State only advances after the driver returned the picture (see
    // `GopState`); POC counts pictures since the last emitted IDR.
    let plan = state.gop.plan();
    let is_idr = plan.is_idr;
    let poc = i32::try_from(plan.position).unwrap_or(i32::MAX);
    let cur_recon_idx = state.next_recon;

    let curr_pic = PictureHEVC::new(enc.recon[cur_recon_idx].id(), poc, 0);
    let prev_ref: Option<PictureHEVC> = if is_idr {
        None
    } else {
        state.prev_ref.map(|(prev_idx, prev_poc)| {
            // VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE (0x10) marks the
            // single short-term-before reference our IPPP slot uses.
            PictureHEVC::new(enc.recon[prev_idx].id(), prev_poc, 0x10)
        })
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
    let mut packed = PackedBuffers::new(&enc.display);

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
            let (p, d) =
                unsafe { create_packed_header(&enc.context, EncPackedHeaderType::Sequence, nal)? };
            packed.ids.push(p);
            packed.ids.push(d);
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
    all_ids.extend_from_slice(&packed.ids);

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

    let frame_idx_now = state.frame_idx;
    state.frame_idx += 1;
    state.gop.commit(plan);
    state.next_recon = (cur_recon_idx + 1) % enc.recon.len();
    state.prev_ref = Some((cur_recon_idx, poc));

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
        /* aspect_ratio_info_present_flag */ 0, /* neutral_chroma_indication_flag */ 0,
        /* field_seq_flag */ 0, /* vui_timing_info_present_flag */ 1,
        /* bitstream_restriction_flag */ 0, /* tiles_fixed_structure_flag */ 0,
        /* motion_vectors_over_pic_boundaries_flag */ 1,
        /* restricted_ref_pic_lists_flag */ 0, /* log2_max_mv_length_horizontal */ 15,
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
    let nal_unit_type = if is_idr {
        HEVC_NAL_IDR_W_RADL
    } else {
        HEVC_NAL_TRAIL_R
    };

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

/// Packed-header buffers created by raw FFI (cros-libva has no safe
/// wrapper for them). Destroyed on drop so every exit path — including
/// a failed `vaEndPicture` / `vaSyncSurface` — releases them.
struct PackedBuffers<'a> {
    display: &'a Display,
    ids: Vec<VABufferID>,
}

impl<'a> PackedBuffers<'a> {
    fn new(display: &'a Display) -> Self {
        Self {
            display,
            ids: Vec::new(),
        }
    }
}

impl Drop for PackedBuffers<'_> {
    fn drop(&mut self) {
        for id in &self.ids {
            // SAFETY: each id came from a successful `vaCreateBuffer` on
            // this display and is destroyed exactly once, here.
            unsafe {
                let _ = vaDestroyBuffer(self.display.handle(), *id);
            }
        }
    }
}

fn check_status(status: VAStatus) -> std::result::Result<(), VAStatus> {
    if status == 0 { Ok(()) } else { Err(status) }
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
            .map_err(|e| {
                FerricastError::Encoder(format!("VA-API HEVC: vaCreateBuffer(VPP): {e}"))
            })?;

        let dest = &self.recon[target_recon_idx];
        let mut pic = Picture::new(g.timestamp_us, Rc::clone(vpp), dest);
        pic.add_buffer(buffer);
        let pic = pic.begin::<()>().map_err(|e| {
            FerricastError::Encoder(format!("VA-API HEVC: vaBeginPicture(VPP): {e}"))
        })?;
        let pic = pic.render().map_err(|e| {
            FerricastError::Encoder(format!("VA-API HEVC: vaRenderPicture(VPP): {e}"))
        })?;
        let pic = pic
            .end()
            .map_err(|e| FerricastError::Encoder(format!("VA-API HEVC: vaEndPicture(VPP): {e}")))?;
        pic.sync::<()>().map_err(|(e, _)| {
            FerricastError::Encoder(format!("VA-API HEVC: vaSyncSurface(VPP): {e}"))
        })?;
        drop(imported);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_follow_tables_a8_a9() {
        // 1080p is 2 073 600 samples: over 3.1's MaxLumaPs (983 040).
        assert_eq!(pick_level(1920, 1088, 30, 8_000_000), 120);
        // 1080p60 needs 124 M samples/s: over 4.0's 66.8 M.
        assert_eq!(pick_level(1920, 1088, 60, 8_000_000), 123);
        assert_eq!(pick_level(1280, 736, 30, 5_000_000), 93);
        assert_eq!(pick_level(3840, 2176, 30, 20_000_000), 150);
        assert_eq!(pick_level(3840, 2176, 60, 20_000_000), 153);
    }

    #[test]
    fn bitrate_raises_level() {
        // 1080p30 at 15 Mbps exceeds 4.0's 12 Mbps Main-tier MaxBR.
        assert_eq!(pick_level(1920, 1088, 30, 15_000_000), 123);
    }
}
