//! VA-API hardware H.264 decode (worker-thread architecture).
//!
//! Engaged via `FERRICAST_H264_DECODE_BACKEND=vaapi`. The facade's
//! default chain still skips this backend until it's been validated
//! against more bitstreams in the wild — set the env var to take it
//! out for a spin against a real HLS / Chromecast cast feed.
//!
//! ## Why a worker thread
//!
//! `cros-libva` uses `Rc` for its display / context / surface
//! handles, so `Display: !Send`. The [`VideoDecoder`] trait requires
//! `Send`. Putting the VA state on a dedicated worker thread and
//! talking to it over `std::sync::mpsc` lets the public struct stay
//! `Send` (it's just a pair of channel handles), while the VA state
//! itself never crosses a thread boundary.
//!
//! The trait surface is synchronous, so each [`VideoDecoder::decode`]
//! call sends a [`Cmd::Decode`] and blocks on a [`Reply`] reading
//! through `mpsc::Receiver::recv`. The manager's pump runs decoders
//! on a dedicated tokio task, so blocking inside that task only
//! affects that one session.
//!
//! ## Per-frame flow
//!
//! HLS streams encode one access unit per [`EncodedFrame`]
//! (mpeg-ts demuxer in `ferricast-hls` aggregates a PES packet into
//! a single delivery), so the worker treats each `Decode` as one
//! complete picture:
//!
//! 1. Walk Annex-B NAL units. Cache SPS (type 7) / PPS (type 8)
//!    in the `h264-reader` context bank. Buffer slice NALs (1, 5)
//!    until end of frame.
//! 2. On first SPS containing dimensions, lazy-create `Config` +
//!    `Context` + a small surface pool.
//! 3. For each slice NAL in this frame, parse a minimal slice header
//!    (inline parser — `h264-reader`'s [`SliceHeader`] fields are
//!    private), build `VAPictureParameterBufferH264`,
//!    `VAIQMatrixBufferH264`, `VASliceParameterBufferH264`, and
//!    `VASliceDataBufferType` from the raw NAL bytes (start code
//!    stripped).
//! 4. Run the typestate sequence
//!    `Picture::new → add_buffer*N → begin → render → end → sync`
//!    against an output surface borrowed from the pool.
//! 5. `vaDeriveImage` on the synced surface, read NV12, convert to
//!    BGRA on the CPU, return as `CapturedFrame::Cpu`. A future
//!    revision will switch to `vaExportSurfaceHandle` → DMA-BUF for
//!    a zero-copy `CapturedFrame::Gpu` path; for now the converted
//!    BGRA matches what `openh264_impl` produces, so the receiver
//!    window's renderer doesn't need to know which backend it came
//!    from.
//! 6. Promote the just-decoded surface to "previous reference" for
//!    P-frame motion compensation on the next call.
//!
//! ## Known limitations (tracked in tasks #9, #13)
//!
//! - **DPB depth = 1.** Reference list construction only carries the
//!   most recently decoded surface. P-frames work; B-frames that
//!   reference distant pictures will produce garbage. Most live
//!   casting streams (VLC, Cast SDK senders, YouTube HLS) ship Main
//!   profile with no B-frames, so this is OK in practice. The
//!   follow-up Vulkan Video decoder (task #13) will manage a proper
//!   DPB.
//! - **Single-slice frames only.** Multi-slice pictures aren't
//!   handled — `vaBeginPicture` is called once and all slice
//!   parameter buffers for the picture are submitted together.
//!   Multi-slice pictures need slice-group / MBAFF support.
//! - **No SEI / FMO / ASO.** Ignored on input.
//! - **CPU readback.** NV12 → BGRA on CPU costs a memcpy + colour
//!   conversion. That's still much cheaper than software H.264
//!   decode, but it isn't the zero-copy path GPU decode promises.
//!   Task #13 (Vulkan Video) produces a `VkImage` ready for
//!   Skia/Freya to sample directly.

use std::io::Cursor;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread;

use bytes::Bytes;
use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, PixelFormat, RawFrame,
    Result, VideoDecoder,
};
use h264_reader::{
    Context as H264Context,
    nal::{
        pps::PicParameterSet,
        sps::{ChromaFormat, FrameMbsFlags, PicOrderCntType, SeqParameterSet},
    },
    rbsp::decode_nal,
};
use libva::{
    BufferType, Config, Context, Display, H264PicFields, H264SeqFields, IQMatrix,
    IQMatrixBufferH264, Picture, PictureH264, PictureParameter, PictureParameterBufferH264,
    SliceParameter, SliceParameterBufferH264, Surface, VA_FOURCC_NV12, VA_INVALID_SURFACE,
    VA_PICTURE_H264_INVALID, VA_PICTURE_H264_SHORT_TERM_REFERENCE, VA_RT_FORMAT_YUV420,
    VA_SLICE_DATA_FLAG_ALL, VAEntrypoint::VAEntrypointVLD, VAProfile,
};

use cros_libva as libva;

const SURFACE_POOL_SIZE: usize = 8;

// ─── public-facing decoder ──────────────────────────────────────────

pub struct VaapiH264Decoder {
    width: u32,
    height: u32,
    cmd_tx: SyncSender<Cmd>,
    reply_rx: Receiver<Reply>,
}

enum Cmd {
    Configure(DecoderConfig),
    Decode(EncodedFrame),
    Flush,
    Shutdown,
}

enum Reply {
    Configured,
    Frame(std::result::Result<Option<CapturedFrame>, String>),
    Flushed,
}

impl VaapiH264Decoder {
    /// Probe whether the host's VA-API driver advertises H.264 decode
    /// and spin up the worker thread. Returns a `Send` decoder handle
    /// that proxies subsequent calls to the worker.
    pub fn probe_with(config: &DecoderConfig) -> Result<Self> {
        if config.codec != Codec::H264 {
            return Err(FerricastError::Decode(format!(
                "VA-API H.264 probe asked for {:?}",
                config.codec
            )));
        }

        let (probe_tx, probe_rx) = mpsc::sync_channel::<std::result::Result<(), String>>(1);
        let (cmd_tx, cmd_rx) = mpsc::sync_channel::<Cmd>(2);
        let (reply_tx, reply_rx) = mpsc::sync_channel::<Reply>(2);

        thread::Builder::new()
            .name("ferricast-vaapi-h264".into())
            .spawn(move || run_worker(probe_tx, cmd_rx, reply_tx))
            .map_err(|e| {
                FerricastError::Decode(format!("VA-API: failed to spawn worker thread: {e}"))
            })?;

        match probe_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                width: config.width,
                height: config.height,
                cmd_tx,
                reply_rx,
            }),
            Ok(Err(e)) => Err(FerricastError::Decode(e)),
            Err(_) => Err(FerricastError::Decode(
                "VA-API worker thread died during probe".into(),
            )),
        }
    }

    fn send(&self, cmd: Cmd) -> Result<Reply> {
        self.cmd_tx.send(cmd).map_err(|_| {
            FerricastError::Decode("VA-API worker thread is gone (cmd_tx send failed)".into())
        })?;
        self.reply_rx.recv().map_err(|_| {
            FerricastError::Decode("VA-API worker thread is gone (reply_rx recv failed)".into())
        })
    }
}

impl Drop for VaapiH264Decoder {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(Cmd::Shutdown);
    }
}

impl VideoDecoder for VaapiH264Decoder {
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, config: &DecoderConfig) -> Result<()> {
        self.width = config.width;
        self.height = config.height;
        match self.send(Cmd::Configure(config.clone()))? {
            Reply::Configured => Ok(()),
            Reply::Frame(Err(e)) => Err(FerricastError::Decode(e)),
            _ => Err(FerricastError::Decode(
                "VA-API worker returned unexpected reply to Configure".into(),
            )),
        }
    }

    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        match self.send(Cmd::Decode(frame))? {
            Reply::Frame(Ok(out)) => Ok(out),
            Reply::Frame(Err(e)) => Err(FerricastError::Decode(e)),
            _ => Err(FerricastError::Decode(
                "VA-API worker returned unexpected reply to Decode".into(),
            )),
        }
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        match self.send(Cmd::Flush)? {
            Reply::Flushed => Ok(Vec::new()),
            Reply::Frame(Err(e)) => Err(FerricastError::Decode(e)),
            _ => Err(FerricastError::Decode(
                "VA-API worker returned unexpected reply to Flush".into(),
            )),
        }
    }
}

// ─── worker thread ──────────────────────────────────────────────────

fn run_worker(
    probe_tx: SyncSender<std::result::Result<(), String>>,
    cmd_rx: Receiver<Cmd>,
    reply_tx: SyncSender<Reply>,
) {
    let display = match Display::open() {
        Some(d) => d,
        None => {
            let _ = probe_tx.send(Err(
                "VA-API: no display (no /dev/dri/renderD* available)".into()
            ));
            return;
        }
    };

    let candidates = [
        (VAProfile::VAProfileH264High, "High"),
        (VAProfile::VAProfileH264Main, "Main"),
        (
            VAProfile::VAProfileH264ConstrainedBaseline,
            "ConstrainedBaseline",
        ),
    ];
    let mut probed: Option<(Config, &'static str)> = None;
    let mut last_err: Option<String> = None;
    for (profile, label) in candidates {
        match display.create_config(Vec::new(), profile, VAEntrypointVLD) {
            Ok(c) => {
                probed = Some((c, label));
                break;
            }
            Err(e) => last_err = Some(format!("{label}: {e:?}")),
        }
    }
    let (config, profile_label) = match probed {
        Some(c) => c,
        None => {
            let _ = probe_tx.send(Err(format!(
                "VA-API: driver does not advertise H.264 decode (last: {last_err:?})"
            )));
            return;
        }
    };
    tracing::info!(
        profile = profile_label,
        "VA-API H.264 worker thread up; awaiting bitstream"
    );
    let _ = probe_tx.send(Ok(()));

    let mut state = WorkerState {
        display,
        config,
        ctx: None,
        surfaces: Vec::new(),
        next_surface: 0,
        prev_ref: None,
        h264_ctx: H264Context::default(),
        sps_seen: false,
        pps_seen: false,
        width: 0,
        height: 0,
        frame_count: 0,
    };

    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            Cmd::Configure(c) => {
                state.width = c.width;
                state.height = c.height;
                let _ = reply_tx.send(Reply::Configured);
            }
            Cmd::Decode(frame) => {
                let result = state
                    .decode_picture(frame)
                    .map_err(|e| e.to_string());
                let _ = reply_tx.send(Reply::Frame(result));
            }
            Cmd::Flush => {
                state.prev_ref = None;
                let _ = reply_tx.send(Reply::Flushed);
            }
            Cmd::Shutdown => break,
        }
    }
}

struct WorkerState {
    display: std::rc::Rc<Display>,
    config: Config,
    ctx: Option<std::rc::Rc<Context>>,
    /// Surface pool plus an `in_flight` deque to round-robin
    /// allocation. Decode targets and references are picked from the
    /// same pool; an unsafe (re)use of a surface still tied up as a
    /// reference would corrupt the next picture's decode.
    surfaces: Vec<Surface<()>>,
    next_surface: usize,
    /// Most recently decoded surface index. Used as the single ref
    /// frame for P-slice motion compensation. `None` before the first
    /// IDR has been processed.
    prev_ref: Option<RefPic>,
    h264_ctx: H264Context,
    sps_seen: bool,
    pps_seen: bool,
    width: u32,
    height: u32,
    frame_count: u64,
}

#[derive(Clone, Copy)]
struct RefPic {
    surface_idx: usize,
    frame_num: u32,
    poc: i32,
}

impl WorkerState {
    fn ensure_context(&mut self, sps: &SeqParameterSet) -> Result<()> {
        if self.ctx.is_some() {
            return Ok(());
        }
        let (w, h) = sps.pixel_dimensions().map_err(|e| {
            FerricastError::Decode(format!("VA-API: SPS pixel_dimensions(): {e:?}"))
        })?;
        // The hint dimensions in DecoderConfig may be 0 (the
        // discovery path in the HLS puller can't know dimensions
        // until after the first SPS). Adopt the bitstream-derived
        // dimensions here once we have them.
        if self.width == 0 || self.height == 0 {
            self.width = w;
            self.height = h;
        }
        let surfaces = self
            .display
            .create_surfaces::<()>(
                VA_RT_FORMAT_YUV420,
                Some(VA_FOURCC_NV12),
                w,
                h,
                None,
                (0..SURFACE_POOL_SIZE).map(|_| ()).collect(),
            )
            .map_err(|e| FerricastError::Decode(format!("VA-API create_surfaces: {e:?}")))?;
        let ctx = self
            .display
            .create_context::<()>(&self.config, w as u32, h as u32, Some(&surfaces), true)
            .map_err(|e| FerricastError::Decode(format!("VA-API create_context: {e:?}")))?;
        self.surfaces = surfaces;
        self.ctx = Some(ctx);
        tracing::info!(width = w, height = h, "VA-API: context + surface pool created");
        Ok(())
    }

    fn take_output_surface(&mut self) -> Option<usize> {
        if self.surfaces.is_empty() {
            return None;
        }
        // Avoid the surface currently holding the previous frame's
        // decoded data — it's needed as a reference input.
        let mut idx = self.next_surface % self.surfaces.len();
        if let Some(prev) = self.prev_ref {
            if idx == prev.surface_idx {
                idx = (idx + 1) % self.surfaces.len();
            }
        }
        self.next_surface = (idx + 1) % self.surfaces.len();
        Some(idx)
    }

    fn decode_picture(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        let nals = split_annex_b(&frame.data);
        let mut slice_nals: Vec<&[u8]> = Vec::new();
        let mut is_idr = false;

        // First pass: register SPS / PPS in the h264_ctx, classify slices.
        for nal in &nals {
            if nal.is_empty() {
                continue;
            }
            let nal_type = nal[0] & 0x1f;
            match nal_type {
                7 => {
                    // EBSP → RBSP, then h264-reader's bit reader.
                    let rbsp = match decode_nal(&nal[1..]) {
                        Ok(r) => r,
                        Err(e) => {
                            tracing::warn!(%e, "VA-API: SPS NAL unescape failed");
                            continue;
                        }
                    };
                    let rb = h264_reader::rbsp::BitReader::new(Cursor::new(rbsp.as_ref()));
                    match SeqParameterSet::from_bits(rb) {
                        Ok(sps) => {
                            let id = sps.seq_parameter_set_id;
                            self.h264_ctx.put_seq_param_set(sps);
                            self.sps_seen = true;
                            // Clone the SPS out of the context bank so we
                            // don't hold a borrow across `ensure_context`,
                            // which needs `&mut self`.
                            let sps_clone = self.h264_ctx.sps_by_id(id).cloned();
                            if let Some(s) = sps_clone {
                                self.ensure_context(&s)?;
                            }
                        }
                        Err(e) => {
                            tracing::warn!(?e, "VA-API: SPS parse failed; skipping NAL");
                        }
                    }
                }
                8 => {
                    let rbsp = match decode_nal(&nal[1..]) {
                        Ok(r) => r,
                        Err(e) => {
                            tracing::warn!(%e, "VA-API: PPS NAL unescape failed");
                            continue;
                        }
                    };
                    let rb = h264_reader::rbsp::BitReader::new(Cursor::new(rbsp.as_ref()));
                    match PicParameterSet::from_bits(&self.h264_ctx, rb) {
                        Ok(pps) => {
                            self.h264_ctx.put_pic_param_set(pps);
                            self.pps_seen = true;
                        }
                        Err(e) => {
                            tracing::warn!(?e, "VA-API: PPS parse failed; skipping NAL");
                        }
                    }
                }
                1 => slice_nals.push(*nal),
                5 => {
                    is_idr = true;
                    slice_nals.push(*nal);
                }
                _ => {}
            }
        }

        if !self.sps_seen || !self.pps_seen {
            // Pre-keyframe arrival, or out-of-band SPS / PPS that
            // didn't make it. The receiver pipeline retries on the
            // next access unit; report None so the sink ignores this
            // packet.
            return Ok(None);
        }
        if slice_nals.is_empty() {
            return Ok(None);
        }

        // Snapshot the most recent SPS/PPS — slices reference them
        // via pps_id; we pluck the active pair from the context bank
        // by parsing the first slice's pps_id below.
        let first_slice = slice_nals[0];
        let nal_hdr = first_slice[0];
        let slice_rbsp_cow = decode_nal(&first_slice[1..])
            .map_err(|e| FerricastError::Decode(format!("slice NAL unescape: {e}")))?;
        let slice_rbsp: &[u8] = slice_rbsp_cow.as_ref();
        let header_info = parse_slice_header(slice_rbsp, nal_hdr, &self.h264_ctx)?;

        // The active SPS / PPS pair for this picture. Cloned because
        // the next call may mutate the context bank.
        let pps = self
            .h264_ctx
            .pps_by_id(h264_reader::nal::pps::ParamSetId::from_u32(
                header_info.pps_id as u32,
            ).map_err(|e| FerricastError::Decode(format!("PPS id: {e:?}")))?)
            .ok_or_else(|| FerricastError::Decode("PPS missing".into()))?
            .clone();
        let sps = self
            .h264_ctx
            .sps_by_id(pps.seq_parameter_set_id)
            .ok_or_else(|| FerricastError::Decode("SPS missing".into()))?
            .clone();

        self.ensure_context(&sps)?;

        let surface_idx = self
            .take_output_surface()
            .ok_or_else(|| FerricastError::Decode("VA-API: no surface available".into()))?;

        // Reset the prev-ref binding on IDR — DPB is cleared.
        if is_idr {
            self.prev_ref = None;
        }

        // Build VA buffers.
        let pic_param = build_pic_param(
            &sps,
            &pps,
            &header_info,
            &self.surfaces,
            surface_idx,
            self.prev_ref,
        );
        let iq_matrix = build_iq_matrix();
        let slice_data_bit_offset =
            rbsp_bits_to_escaped_bits(&first_slice[1..], header_info.header_bits) + 8;
        let slice_param = build_slice_param(
            &header_info,
            first_slice.len() as u32,
            slice_data_bit_offset,
            self.prev_ref,
        );
        let slice_data = first_slice.to_vec();

        // Submit picture.
        let ctx = self
            .ctx
            .clone()
            .ok_or_else(|| FerricastError::Decode("VA-API: context not ready".into()))?;
        // Take the surface out of the pool by index — we put it back
        // after the picture syncs. `std::mem::replace` keeps the Vec
        // shape stable so the index → surface mapping in `prev_ref`
        // stays valid across decodes.
        let placeholder = self
            .display
            .create_surfaces::<()>(
                VA_RT_FORMAT_YUV420,
                Some(VA_FOURCC_NV12),
                self.width,
                self.height,
                None,
                vec![()],
            )
            .map_err(|e| {
                FerricastError::Decode(format!("VA-API placeholder surface: {e:?}"))
            })?
            .into_iter()
            .next()
            .unwrap();
        let output_surface = std::mem::replace(&mut self.surfaces[surface_idx], placeholder);

        let timestamp = frame.timestamp_us;
        let mut picture = Picture::new(timestamp, std::rc::Rc::clone(&ctx), output_surface);

        let pic_buf = ctx
            .create_buffer(BufferType::PictureParameter(PictureParameter::H264(
                pic_param,
            )))
            .map_err(|e| FerricastError::Decode(format!("VA-API pic_param buffer: {e:?}")))?;
        picture.add_buffer(pic_buf);

        let iq_buf = ctx
            .create_buffer(BufferType::IQMatrix(IQMatrix::H264(iq_matrix)))
            .map_err(|e| FerricastError::Decode(format!("VA-API iq_matrix buffer: {e:?}")))?;
        picture.add_buffer(iq_buf);

        let slice_param_buf = ctx
            .create_buffer(BufferType::SliceParameter(SliceParameter::H264(
                slice_param,
            )))
            .map_err(|e| {
                FerricastError::Decode(format!("VA-API slice_param buffer: {e:?}"))
            })?;
        picture.add_buffer(slice_param_buf);

        let slice_data_buf = ctx
            .create_buffer(BufferType::SliceData(slice_data))
            .map_err(|e| FerricastError::Decode(format!("VA-API slice_data buffer: {e:?}")))?;
        picture.add_buffer(slice_data_buf);

        let pic_begin = picture
            .begin::<()>()
            .map_err(|e| FerricastError::Decode(format!("vaBeginPicture: {e:?}")))?;
        let pic_render = pic_begin
            .render()
            .map_err(|e| FerricastError::Decode(format!("vaRenderPicture: {e:?}")))?;
        let pic_end = pic_render
            .end()
            .map_err(|e| FerricastError::Decode(format!("vaEndPicture: {e:?}")))?;
        let pic_sync = pic_end
            .sync::<()>()
            .map_err(|(e, _)| FerricastError::Decode(format!("vaSyncSurface: {e:?}")))?;

        // Read back NV12 + convert to BGRA.
        let derived = pic_sync
            .derive_image::<()>((self.width, self.height))
            .map_err(|e| FerricastError::Decode(format!("vaDeriveImage: {e:?}")))?;
        let bgra = nv12_to_bgra(&derived, self.width, self.height)?;
        drop(derived);

        // Reclaim the surface, swap the placeholder out.
        let reclaimed = pic_sync
            .take_surface()
            .map_err(|_| FerricastError::Decode("VA-API: picture still holds surface".into()))?;
        self.surfaces[surface_idx] = reclaimed;

        self.frame_count += 1;
        if self.frame_count.is_multiple_of(60) {
            tracing::debug!(
                count = self.frame_count,
                w = self.width,
                h = self.height,
                "VA-API: decoded frame"
            );
        }

        // Record this surface as the new reference for the next P-frame.
        self.prev_ref = Some(RefPic {
            surface_idx,
            frame_num: header_info.frame_num,
            poc: header_info.pic_order_cnt_lsb as i32,
        });

        Ok(Some(CapturedFrame::Cpu(RawFrame {
            width: self.width,
            height: self.height,
            stride: self.width * 4,
            format: PixelFormat::Bgra,
            data: Bytes::from(bgra),
            timestamp_us: timestamp,
        })))
    }
}

// ─── parameter buffer construction ──────────────────────────────────

fn build_pic_param(
    sps: &SeqParameterSet,
    pps: &PicParameterSet,
    hdr: &SliceHdr,
    surfaces: &[Surface<()>],
    output_idx: usize,
    prev_ref: Option<RefPic>,
) -> PictureParameterBufferH264 {
    let curr_pic = PictureH264::new(surfaces[output_idx].id(), hdr.frame_num, 0, 0, 0);

    let invalid_ref =
        PictureH264::new(VA_INVALID_SURFACE, 0, VA_PICTURE_H264_INVALID, 0, 0);
    let mut refs: [PictureH264; 16] = std::array::from_fn(|_| {
        PictureH264::new(VA_INVALID_SURFACE, 0, VA_PICTURE_H264_INVALID, 0, 0)
    });
    if let Some(prev) = prev_ref {
        refs[0] = PictureH264::new(
            surfaces[prev.surface_idx].id(),
            prev.frame_num,
            VA_PICTURE_H264_SHORT_TERM_REFERENCE,
            prev.poc,
            prev.poc,
        );
    }
    let _ = invalid_ref;

    let chroma_format_idc = match sps.chroma_info.chroma_format {
        ChromaFormat::Monochrome => 0,
        ChromaFormat::YUV420 => 1,
        ChromaFormat::YUV422 => 2,
        ChromaFormat::YUV444 => 3,
        ChromaFormat::Invalid(v) => v as u32,
    };

    let (frame_mbs_only_flag, mb_adaptive_frame_field_flag) = match sps.frame_mbs_flags {
        FrameMbsFlags::Frames => (1, 0),
        FrameMbsFlags::Fields {
            mb_adaptive_frame_field_flag,
        } => (0, mb_adaptive_frame_field_flag as u32),
    };
    let (log2_max_pic_order_cnt_lsb_minus4, pic_order_cnt_type, delta_pic_order_always_zero_flag) =
        match sps.pic_order_cnt {
            PicOrderCntType::TypeZero {
                log2_max_pic_order_cnt_lsb_minus4,
            } => (log2_max_pic_order_cnt_lsb_minus4 as u32, 0u32, 0u32),
            PicOrderCntType::TypeOne {
                delta_pic_order_always_zero_flag,
                ..
            } => (0, 1, delta_pic_order_always_zero_flag as u32),
            PicOrderCntType::TypeTwo => (0, 2, 0),
        };

    let seq_fields = H264SeqFields::new(
        chroma_format_idc,
        sps.chroma_info.separate_colour_plane_flag as u32,
        sps.gaps_in_frame_num_value_allowed_flag as u32,
        frame_mbs_only_flag,
        mb_adaptive_frame_field_flag,
        sps.direct_8x8_inference_flag as u32,
        // min_luma_bi_pred_size8x8 = (profile_idc == 66) ? 0 : (level_idc >= 31 ? 1 : 0).
        // Baseline / Constrained Baseline never apply 8x8 inference;
        // Main / High at Level 3.1+ allow it. Approximate without
        // dragging the level table — the driver only consults this
        // for direct-8x8 motion compensation, so a misread costs
        // accuracy on B-frames in High@L3.1+ streams, which we don't
        // support yet anyway.
        if sps.level_idc >= 31 { 1 } else { 0 },
        sps.log2_max_frame_num_minus4 as u32,
        pic_order_cnt_type,
        log2_max_pic_order_cnt_lsb_minus4,
        delta_pic_order_always_zero_flag,
    );
    let pic_fields = H264PicFields::new(
        pps.entropy_coding_mode_flag as u32,
        pps.weighted_pred_flag as u32,
        pps.weighted_bipred_idc as u32,
        pps.extension
            .as_ref()
            .map(|e| e.transform_8x8_mode_flag as u32)
            .unwrap_or(0),
        match hdr.field_pic_flag {
            true => 1,
            false => 0,
        },
        pps.constrained_intra_pred_flag as u32,
        pps.bottom_field_pic_order_in_frame_present_flag as u32,
        pps.deblocking_filter_control_present_flag as u32,
        pps.redundant_pic_cnt_present_flag as u32,
        // reference_pic_flag — set when this picture will be used as
        // a reference for later pictures. We mark every decoded
        // picture as a potential ref since our DPB is just "most
        // recent"; the driver uses this flag for its own bookkeeping
        // around marking, which we replicate at the surface level.
        1,
    );

    PictureParameterBufferH264::new(
        curr_pic,
        refs,
        sps.pic_width_in_mbs_minus1 as u16,
        sps.pic_height_in_map_units_minus1 as u16,
        sps.chroma_info.bit_depth_luma_minus8,
        sps.chroma_info.bit_depth_chroma_minus8,
        sps.max_num_ref_frames as u8,
        &seq_fields,
        // num_slice_groups_minus1 = 0 (FMO unsupported, we read PPS
        // slice_groups but fall back to 0 if present — driver will
        // reject otherwise).
        0,
        0,
        0,
        pps.pic_init_qp_minus26 as i8,
        pps.pic_init_qs_minus26 as i8,
        pps.chroma_qp_index_offset as i8,
        pps.extension
            .as_ref()
            .map(|e| e.second_chroma_qp_index_offset as i8)
            .unwrap_or(pps.chroma_qp_index_offset as i8),
        &pic_fields,
        hdr.frame_num as u16,
    )
}

/// Default flat scaling matrices. Used when the bitstream doesn't
/// signal explicit scaling. Every entry is 16 → identity scaling.
fn build_iq_matrix() -> IQMatrixBufferH264 {
    IQMatrixBufferH264::new([[16u8; 16]; 6], [[16u8; 64]; 2])
}

fn build_slice_param(
    hdr: &SliceHdr,
    slice_data_size: u32,
    slice_data_bit_offset: u32,
    prev_ref: Option<RefPic>,
) -> SliceParameterBufferH264 {
    let invalid_pic = || PictureH264::new(VA_INVALID_SURFACE, 0, VA_PICTURE_H264_INVALID, 0, 0);
    let ref_list_0: [PictureH264; 32] = std::array::from_fn(|i| {
        if i == 0 {
            if let Some(prev) = prev_ref {
                return PictureH264::new(
                    VA_INVALID_SURFACE,
                    prev.frame_num,
                    VA_PICTURE_H264_SHORT_TERM_REFERENCE,
                    prev.poc,
                    prev.poc,
                );
            }
        }
        invalid_pic()
    });
    let ref_list_1: [PictureH264; 32] = std::array::from_fn(|_| invalid_pic());

    SliceParameterBufferH264::new(
        slice_data_size,
        0,
        VA_SLICE_DATA_FLAG_ALL,
        slice_data_bit_offset as u16,
        hdr.first_mb_in_slice as u16,
        hdr.slice_type_mod5,
        hdr.direct_spatial_mv_pred_flag as u8,
        hdr.num_ref_idx_l0_minus1.unwrap_or(0) as u8,
        hdr.num_ref_idx_l1_minus1.unwrap_or(0) as u8,
        hdr.cabac_init_idc.unwrap_or(0) as u8,
        hdr.slice_qp_delta as i8,
        hdr.disable_deblocking_filter_idc,
        hdr.slice_alpha_c0_offset_div2,
        hdr.slice_beta_offset_div2,
        ref_list_0,
        ref_list_1,
        // No explicit prediction weights — we don't implement
        // pred_weight_table, so the driver applies default weights.
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
    )
}

// ─── slice header parser (minimal) ──────────────────────────────────

struct SliceHdr {
    first_mb_in_slice: u32,
    /// Raw slice_type as parsed, 0..=9.
    #[allow(dead_code)]
    slice_type_raw: u8,
    /// Canonical slice_type mod 5 — VA-API's `slice_type` field uses
    /// the 0..=4 form (P=0, B=1, I=2, SP=3, SI=4).
    slice_type_mod5: u8,
    pps_id: u8,
    frame_num: u32,
    field_pic_flag: bool,
    pic_order_cnt_lsb: u32,
    slice_qp_delta: i32,
    disable_deblocking_filter_idc: u8,
    slice_alpha_c0_offset_div2: i8,
    slice_beta_offset_div2: i8,
    direct_spatial_mv_pred_flag: bool,
    num_ref_idx_l0_minus1: Option<u32>,
    num_ref_idx_l1_minus1: Option<u32>,
    cabac_init_idc: Option<u32>,
    /// Size of the slice header in bits, starting from the byte after
    /// the NAL header byte. Used to fill `slice_data_bit_offset` for
    /// VA, after offsetting by the NAL header byte (8 bits).
    header_bits: u32,
}

/// Inline slice header parser. `h264-reader` parses the full slice
/// header but keeps most fields private, so we parse the subset we
/// need ourselves.
///
/// `slice_rbsp` is the RBSP (emulation-prevention bytes stripped)
/// starting AT the byte after the NAL header.
fn parse_slice_header(
    slice_rbsp: &[u8],
    nal_hdr: u8,
    ctx: &H264Context,
) -> Result<SliceHdr> {
    let mut br = BitReader::new(slice_rbsp);
    let first_mb_in_slice = br.read_ue()?;
    let slice_type_raw = br.read_ue()? as u8;
    let slice_type_mod5 = slice_type_raw % 5;
    let pps_id = br.read_ue()? as u8;

    let pps = ctx
        .pps_by_id(
            h264_reader::nal::pps::ParamSetId::from_u32(pps_id as u32)
                .map_err(|e| FerricastError::Decode(format!("PPS id: {e:?}")))?,
        )
        .ok_or_else(|| FerricastError::Decode(format!("slice references unknown PPS {pps_id}")))?;
    let sps = ctx
        .sps_by_id(pps.seq_parameter_set_id)
        .ok_or_else(|| FerricastError::Decode("slice references unknown SPS".into()))?;

    // separate_colour_plane_flag is virtually never set in the wild
    // (it requires chroma_format_idc==3 with explicit separation),
    // but skip the colour_plane_id when it is.
    if sps.chroma_info.separate_colour_plane_flag {
        br.read_bits(2)?;
    }

    let frame_num = br.read_bits(sps.log2_max_frame_num_minus4 as u32 + 4)?;

    let field_pic_flag = match sps.frame_mbs_flags {
        FrameMbsFlags::Frames => false,
        FrameMbsFlags::Fields { .. } => {
            let flag = br.read_bit()? != 0;
            if flag {
                let _bottom_field_flag = br.read_bit()?;
            }
            flag
        }
    };

    let nal_unit_type = nal_hdr & 0x1f;
    let is_idr = nal_unit_type == 5;
    if is_idr {
        let _idr_pic_id = br.read_ue()?;
    }

    let pic_order_cnt_lsb = match sps.pic_order_cnt {
        PicOrderCntType::TypeZero {
            log2_max_pic_order_cnt_lsb_minus4,
        } => {
            let lsb = br.read_bits(log2_max_pic_order_cnt_lsb_minus4 as u32 + 4)?;
            if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
                let _delta_pic_order_cnt_bottom = br.read_se()?;
            }
            lsb
        }
        PicOrderCntType::TypeOne {
            delta_pic_order_always_zero_flag,
            ..
        } => {
            if !delta_pic_order_always_zero_flag {
                br.read_se()?;
                if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
                    br.read_se()?;
                }
            }
            0
        }
        PicOrderCntType::TypeTwo => 0,
    };

    if pps.redundant_pic_cnt_present_flag {
        br.read_ue()?;
    }

    let family = slice_family(slice_type_mod5);
    let direct_spatial_mv_pred_flag = if family == SliceFamily::B {
        br.read_bit()? != 0
    } else {
        false
    };

    let (num_ref_idx_l0_minus1, num_ref_idx_l1_minus1) =
        if family == SliceFamily::P || family == SliceFamily::Sp || family == SliceFamily::B {
            let override_flag = br.read_bit()? != 0;
            if override_flag {
                let l0 = br.read_ue()?;
                let l1 = if family == SliceFamily::B {
                    Some(br.read_ue()?)
                } else {
                    None
                };
                (Some(l0), l1)
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

    // ref_pic_list_modification — skip but parse fully so the bit
    // offset stays accurate.
    if family != SliceFamily::I && family != SliceFamily::Si {
        let ref_pic_list_modification_flag_l0 = br.read_bit()? != 0;
        if ref_pic_list_modification_flag_l0 {
            loop {
                let modification_of_pic_nums_idc = br.read_ue()?;
                if modification_of_pic_nums_idc == 3 {
                    break;
                }
                let _ = br.read_ue()?;
            }
        }
    }
    if family == SliceFamily::B {
        let ref_pic_list_modification_flag_l1 = br.read_bit()? != 0;
        if ref_pic_list_modification_flag_l1 {
            loop {
                let modification_of_pic_nums_idc = br.read_ue()?;
                if modification_of_pic_nums_idc == 3 {
                    break;
                }
                let _ = br.read_ue()?;
            }
        }
    }

    // pred_weight_table — skip parsing. The default weights case
    // (most common in HLS / Cast streams) doesn't carry this syntax
    // element. If a stream does use weighted prediction we'll see
    // visual artifacts; document as known TBD.
    if (pps.weighted_pred_flag && (family == SliceFamily::P || family == SliceFamily::Sp))
        || (pps.weighted_bipred_idc == 1 && family == SliceFamily::B)
    {
        return Err(FerricastError::Decode(
            "VA-API: pred_weight_table not yet supported in slice header parser".into(),
        ));
    }

    let nal_ref_idc = (nal_hdr >> 5) & 0x3;
    if nal_ref_idc != 0 {
        // dec_ref_pic_marking
        if is_idr {
            let _no_output_of_prior_pics_flag = br.read_bit()?;
            let _long_term_reference_flag = br.read_bit()?;
        } else {
            let adaptive = br.read_bit()? != 0;
            if adaptive {
                loop {
                    let mmco = br.read_ue()?;
                    if mmco == 0 {
                        break;
                    }
                    match mmco {
                        1 | 3 => {
                            br.read_ue()?;
                            if mmco == 3 {
                                br.read_ue()?;
                            }
                        }
                        2 => {
                            br.read_ue()?;
                        }
                        4 => {
                            br.read_ue()?;
                        }
                        5 => {}
                        6 => {
                            br.read_ue()?;
                        }
                        _ => {
                            return Err(FerricastError::Decode(format!(
                                "VA-API: invalid mmco {mmco}"
                            )));
                        }
                    }
                }
            }
        }
    }

    let cabac_init_idc =
        if pps.entropy_coding_mode_flag && family != SliceFamily::I && family != SliceFamily::Si {
            Some(br.read_ue()?)
        } else {
            None
        };

    let slice_qp_delta = br.read_se()?;

    if family == SliceFamily::Sp || family == SliceFamily::Si {
        if family == SliceFamily::Sp {
            br.read_bit()?;
        }
        br.read_se()?; // slice_qs_delta
    }

    let mut disable_deblocking_filter_idc = 0u8;
    let mut slice_alpha_c0_offset_div2 = 0i8;
    let mut slice_beta_offset_div2 = 0i8;
    if pps.deblocking_filter_control_present_flag {
        disable_deblocking_filter_idc = br.read_ue()? as u8;
        if disable_deblocking_filter_idc != 1 {
            slice_alpha_c0_offset_div2 = br.read_se()? as i8;
            slice_beta_offset_div2 = br.read_se()? as i8;
        }
    }

    let header_bits = br.bit_pos() as u32;

    Ok(SliceHdr {
        first_mb_in_slice,
        slice_type_raw,
        slice_type_mod5,
        pps_id,
        frame_num,
        field_pic_flag,
        pic_order_cnt_lsb,
        slice_qp_delta,
        disable_deblocking_filter_idc,
        slice_alpha_c0_offset_div2,
        slice_beta_offset_div2,
        direct_spatial_mv_pred_flag,
        num_ref_idx_l0_minus1,
        num_ref_idx_l1_minus1,
        cabac_init_idc,
        header_bits,
    })
}

#[derive(PartialEq)]
enum SliceFamily {
    P,
    B,
    I,
    Sp,
    Si,
}

fn slice_family(mod5: u8) -> SliceFamily {
    match mod5 {
        0 => SliceFamily::P,
        1 => SliceFamily::B,
        2 => SliceFamily::I,
        3 => SliceFamily::Sp,
        4 => SliceFamily::Si,
        _ => SliceFamily::I,
    }
}

// ─── bit reader for slice headers ───────────────────────────────────

struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit_pos: 0 }
    }

    fn bit_pos(&self) -> usize {
        self.bit_pos
    }

    fn read_bit(&mut self) -> Result<u8> {
        let byte_idx = self.bit_pos / 8;
        let bit_off = 7 - (self.bit_pos % 8);
        if byte_idx >= self.bytes.len() {
            return Err(FerricastError::Decode(
                "VA-API: slice header bitreader EOF".into(),
            ));
        }
        let b = (self.bytes[byte_idx] >> bit_off) & 1;
        self.bit_pos += 1;
        Ok(b)
    }

    fn read_bits(&mut self, n: u32) -> Result<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.read_bit()? as u32;
        }
        Ok(v)
    }

    /// Exp-Golomb unsigned (ue(v)).
    fn read_ue(&mut self) -> Result<u32> {
        let mut leading_zeros = 0u32;
        while self.read_bit()? == 0 {
            leading_zeros += 1;
            if leading_zeros > 32 {
                return Err(FerricastError::Decode(
                    "VA-API: ue(v) leading zeros overflow".into(),
                ));
            }
        }
        if leading_zeros == 0 {
            return Ok(0);
        }
        let tail = self.read_bits(leading_zeros)?;
        Ok((1u32 << leading_zeros) - 1 + tail)
    }

    /// Exp-Golomb signed (se(v)).
    fn read_se(&mut self) -> Result<i32> {
        let u = self.read_ue()?;
        Ok(if u & 1 == 1 {
            ((u >> 1) + 1) as i32
        } else {
            -((u >> 1) as i32)
        })
    }
}

// ─── Annex-B split + NV12 readback ──────────────────────────────────

fn split_annex_b(buf: &Bytes) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut nal_start: Option<usize> = None;
    while i + 3 <= buf.len() {
        let three = i + 3 <= buf.len() && buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1;
        let four = i + 4 <= buf.len()
            && buf[i] == 0
            && buf[i + 1] == 0
            && buf[i + 2] == 0
            && buf[i + 3] == 1;
        if three || four {
            if let Some(start) = nal_start.take() {
                out.push(&buf[start..i]);
            }
            i += if four { 4 } else { 3 };
            nal_start = Some(i);
            continue;
        }
        i += 1;
    }
    if let Some(start) = nal_start {
        out.push(&buf[start..]);
    }
    out
}

/// Map a bit offset measured in the RBSP (emulation-prevention bytes
/// stripped) back onto the corresponding bit offset in the original
/// escaped NAL byte stream.
///
/// VAAPI's `slice_data_bit_offset` is measured against the bitstream
/// as it sits in the `VASliceDataBuffer` — which is the original
/// (escaped) NAL — so we need this translation if any 0x000003
/// prevention bytes fall before the start of slice_data.
fn rbsp_bits_to_escaped_bits(escaped_after_header: &[u8], rbsp_bits: u32) -> u32 {
    let target_rbsp_byte = (rbsp_bits / 8) as usize;
    let bit_remainder = (rbsp_bits % 8) as u32;
    let mut rbsp_bytes_seen: usize = 0;
    let mut zero_run = 0u32;
    let mut esc_idx: usize = 0;
    while rbsp_bytes_seen < target_rbsp_byte && esc_idx < escaped_after_header.len() {
        let b = escaped_after_header[esc_idx];
        if zero_run >= 2 && b == 0x03 {
            // Emulation prevention byte — present in the escaped
            // stream, absent from the RBSP. Don't advance the RBSP
            // counter; reset the zero run so the *next* byte's
            // interpretation isn't affected.
            zero_run = 0;
        } else {
            rbsp_bytes_seen += 1;
            zero_run = if b == 0 { zero_run + 1 } else { 0 };
        }
        esc_idx += 1;
    }
    esc_idx as u32 * 8 + bit_remainder
}

/// Read an NV12 [`libva::Image`] (Y plane followed by interleaved UV
/// plane), convert to BGRA8888 on the CPU. Uses BT.601 limited-range
/// coefficients — the standard for HLS / mpeg-ts content from web
/// senders.
fn nv12_to_bgra(image: &libva::Image<'_>, width: u32, height: u32) -> Result<Vec<u8>> {
    let img_info = image.image();
    let pitch_y = img_info.pitches[0] as usize;
    let pitch_uv = img_info.pitches[1] as usize;
    let off_y = img_info.offsets[0] as usize;
    let off_uv = img_info.offsets[1] as usize;
    let src: &[u8] = image.as_ref();
    let w = width as usize;
    let h = height as usize;

    if off_y + pitch_y * h > src.len() || off_uv + pitch_uv * (h / 2) > src.len() {
        return Err(FerricastError::Decode(
            "VA-API: NV12 image plane bounds exceed mapped region".into(),
        ));
    }

    let mut out = vec![0u8; w * h * 4];
    for y in 0..h {
        let y_row = &src[off_y + y * pitch_y..off_y + y * pitch_y + w];
        let uv_row = &src[off_uv + (y / 2) * pitch_uv..off_uv + (y / 2) * pitch_uv + w];
        let dst_row = &mut out[y * w * 4..(y + 1) * w * 4];
        for x in 0..w {
            let yy = y_row[x] as i32;
            let uv_x = (x / 2) * 2;
            let u = uv_row[uv_x] as i32;
            let v = uv_row[uv_x + 1] as i32;
            // BT.601 limited range. y in [16,235], u/v in [16,240].
            let c = yy - 16;
            let d = u - 128;
            let e = v - 128;
            let r = (298 * c + 409 * e + 128) >> 8;
            let g = (298 * c - 100 * d - 208 * e + 128) >> 8;
            let b = (298 * c + 516 * d + 128) >> 8;
            let dst = &mut dst_row[x * 4..x * 4 + 4];
            dst[0] = b.clamp(0, 255) as u8;
            dst[1] = g.clamp(0, 255) as u8;
            dst[2] = r.clamp(0, 255) as u8;
            dst[3] = 0xff;
        }
    }
    Ok(out)
}
