use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::io::Cursor;
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread;
use std::time::{Duration, Instant};

use bytes::Bytes;

use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, PixelFormat, Result,
    VideoDecoder,
};
use h264_reader::{
    Context as H264Context,
    nal::{
        pps::PicParameterSet,
        sps::{ChromaFormat, FrameMbsFlags, PicOrderCntType, SeqParameterSet},
    },
    rbsp::decode_nal,
};

use cros_libva as libva;
use libva::{
    BufferType, Config, Context, Display, H264PicFields, H264SeqFields, IQMatrix,
    IQMatrixBufferH264, Image, Picture, PictureH264, PictureNew, PictureParameter,
    PictureParameterBufferH264, SliceParameter, SliceParameterBufferH264, Surface, VA_FOURCC_NV12,
    VA_INVALID_SURFACE, VA_PICTURE_H264_INVALID, VA_PICTURE_H264_LONG_TERM_REFERENCE,
    VA_PICTURE_H264_SHORT_TERM_REFERENCE, VA_RT_FORMAT_YUV420, VA_SLICE_DATA_FLAG_ALL,
    VAEntrypoint::VAEntrypointVLD, VAProfile,
};

use super::dpb::{Dpb, Frame as DpbFrame, Released};
use super::slice::{self, PocKind, Pps, SliceHeader, SliceType, Sps, Unsupported};
use crate::nv12::{self, Nv12Planes};

/// Why a VA-API decode call failed, split so the facade can tell a
/// stream it should hand to another backend from a real failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VaapiH264Error {
    /// Valid H.264 using a feature this backend does not implement
    /// (interlacing, 4:2:2, high bit depth, scaling matrices, FMO).
    Unsupported(String),
    Failed(String),
}

impl std::fmt::Display for VaapiH264Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(what) | Self::Failed(what) => f.write_str(what),
        }
    }
}

impl std::error::Error for VaapiH264Error {}

impl From<FerricastError> for VaapiH264Error {
    fn from(error: FerricastError) -> Self {
        Self::Failed(error.to_string())
    }
}

impl From<Unsupported> for VaapiH264Error {
    fn from(error: Unsupported) -> Self {
        Self::Unsupported(error.to_string())
    }
}

impl From<VaapiH264Error> for FerricastError {
    fn from(error: VaapiH264Error) -> Self {
        FerricastError::Decode(error.to_string())
    }
}

type WorkerResult<T> = std::result::Result<T, VaapiH264Error>;

pub struct VaapiH264Decoder {
    cmd_tx: SyncSender<Cmd>,
    reply_rx: Receiver<Reply>,
    ready: VecDeque<CapturedFrame>,
}

enum Cmd {
    Configure(DecoderConfig),
    Decode(EncodedFrame),
    Flush,
    Shutdown,
}

enum Reply {
    Configured,
    /// Pictures finished before `error` (if any) are still good output.
    Frames {
        frames: Vec<CapturedFrame>,
        error: Option<VaapiH264Error>,
    },
}

impl VaapiH264Decoder {
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
                cmd_tx,
                reply_rx,
                ready: VecDeque::new(),
            }),
            Ok(Err(e)) => Err(FerricastError::Decode(e)),
            Err(_) => Err(FerricastError::Decode(
                "VA-API worker thread died during probe".into(),
            )),
        }
    }

    fn send(&self, cmd: Cmd) -> Result<Reply> {
        let gone = || FerricastError::Decode("VA-API worker thread is gone".into());
        self.cmd_tx.send(cmd).map_err(|_| gone())?;
        self.reply_rx.recv().map_err(|_| gone())
    }

    /// Queues every finished picture, then reports the error that cut the
    /// command short, if any.
    fn frames(&mut self, cmd: Cmd) -> WorkerResult<()> {
        match self.send(cmd)? {
            Reply::Frames { frames, error } => {
                self.ready.extend(frames);
                error.map_or(Ok(()), Err)
            }
            Reply::Configured => Err(VaapiH264Error::Failed(
                "VA-API worker returned an unexpected reply".into(),
            )),
        }
    }

    /// [`VideoDecoder::decode`] with the error class kept, so a caller can
    /// switch backends on [`VaapiH264Error::Unsupported`]. Pictures
    /// finished before an error stay queued for the next call.
    ///
    /// # Errors
    ///
    /// [`VaapiH264Error::Unsupported`] when the stream needs a feature
    /// this backend lacks, [`VaapiH264Error::Failed`] otherwise.
    pub fn decode_packet(&mut self, frame: EncodedFrame) -> WorkerResult<Option<CapturedFrame>> {
        self.frames(Cmd::Decode(frame))?;
        Ok(self.ready.pop_front())
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
        match self.send(Cmd::Configure(config.clone()))? {
            Reply::Configured => Ok(()),
            Reply::Frames { .. } => Err(FerricastError::Decode(
                "VA-API worker returned an unexpected reply".into(),
            )),
        }
    }

    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        Ok(self.decode_packet(frame)?)
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        let flushed = self.frames(Cmd::Flush);
        let frames: Vec<CapturedFrame> = self.ready.drain(..).collect();
        match flushed {
            Err(error) if frames.is_empty() => Err(error.into()),
            Err(error) => {
                tracing::warn!(%error, "VA-API H.264 flush failed part-way; returning what was read back");
                Ok(frames)
            }
            Ok(()) => Ok(frames),
        }
    }
}

fn run_worker(
    probe_tx: SyncSender<std::result::Result<(), String>>,
    cmd_rx: Receiver<Cmd>,
    reply_tx: SyncSender<Reply>,
) {
    let Some(display) = Display::open() else {
        let _ = probe_tx.send(Err(
            "VA-API: no display (no /dev/dri/renderD* available)".into()
        ));
        return;
    };
    let profiles = [
        VAProfile::VAProfileH264High,
        VAProfile::VAProfileH264Main,
        VAProfile::VAProfileH264ConstrainedBaseline,
    ];
    let mut last_error = None;
    let config = profiles.into_iter().find_map(|profile| {
        match display.create_config(Vec::new(), profile, VAEntrypointVLD) {
            Ok(config) => Some(config),
            Err(e) => {
                last_error = Some(format!("{profile:?}: {e:?}"));
                None
            }
        }
    });
    let Some(config) = config else {
        let _ = probe_tx.send(Err(format!(
            "VA-API: driver does not advertise H.264 decode (last: {last_error:?})"
        )));
        return;
    };
    tracing::info!("VA-API H.264 worker thread up; awaiting bitstream");
    let _ = probe_tx.send(Ok(()));
    let mut worker = Worker {
        display,
        config,
        format: PixelFormat::Nv12,
        low_latency: false,
        parser: H264Context::default(),
        sps: HashMap::new(),
        pps: HashMap::new(),
        active: None,
    };
    while let Ok(cmd) = cmd_rx.recv() {
        let reply = match cmd {
            Cmd::Configure(config) => {
                worker.format = config.pixel_format;
                worker.low_latency = config.low_latency;
                Reply::Configured
            }
            Cmd::Decode(frame) => {
                let mut frames = Vec::new();
                let result = worker.decode(&frame, &mut frames);
                Reply::from_partial(frames, result)
            }
            Cmd::Flush => {
                let mut frames = Vec::new();
                let result = worker.flush(&mut frames);
                Reply::from_partial(frames, result)
            }
            Cmd::Shutdown => break,
        };
        if reply_tx.send(reply).is_err() {
            break;
        }
    }
}

impl Reply {
    fn from_partial(frames: Vec<CapturedFrame>, result: WorkerResult<()>) -> Self {
        if let Err(error) = &result
            && !frames.is_empty()
        {
            tracing::warn!(
                %error,
                kept = frames.len(),
                "VA-API H.264 failed part-way through a packet; keeping the pictures already finished"
            );
        }
        Reply::Frames {
            frames,
            error: result.err(),
        }
    }
}

struct Worker {
    display: Rc<Display>,
    config: Config,
    format: PixelFormat,
    low_latency: bool,
    parser: H264Context,
    sps: HashMap<u32, Sps>,
    pps: HashMap<u32, Pps>,
    active: Option<Active>,
}

impl Worker {
    /// Pictures finished before an error are left in `out`.
    fn decode(&mut self, frame: &EncodedFrame, out: &mut Vec<CapturedFrame>) -> WorkerResult<()> {
        let nals = split_annex_b(&frame.data);
        let mut picture: Vec<&[u8]> = Vec::new();
        for nal in nals {
            let Some(&header) = nal.first() else {
                continue;
            };
            match header & 0x1f {
                7 | 8 => {
                    self.picture(&picture, frame.timestamp_us, out)?;
                    picture.clear();
                    if header & 0x1f == 7 {
                        self.add_sps(nal)?;
                    } else {
                        self.add_pps(nal)?;
                    }
                }
                1 | 5 => {
                    if first_mb_in_slice(nal)? == 0 && !picture.is_empty() {
                        self.picture(&picture, frame.timestamp_us, out)?;
                        picture.clear();
                    }
                    picture.push(nal);
                }
                _ => {}
            }
        }
        self.picture(&picture, frame.timestamp_us, out)
    }

    fn add_sps(&mut self, nal: &[u8]) -> Result<()> {
        let rbsp = decode_nal(nal)
            .map_err(|e| FerricastError::Decode(format!("VA-API: SPS unescape: {e}")))?;
        let bits = h264_reader::rbsp::BitReader::new(Cursor::new(rbsp.as_ref()));
        let parsed = SeqParameterSet::from_bits(bits)
            .map_err(|e| FerricastError::Decode(format!("VA-API: SPS parse: {e:?}")))?;
        let id = u32::from(parsed.seq_parameter_set_id.id());
        self.sps.insert(id, convert_sps(&parsed, &rbsp));
        self.parser.put_seq_param_set(parsed);
        Ok(())
    }

    fn add_pps(&mut self, nal: &[u8]) -> Result<()> {
        let rbsp = decode_nal(nal)
            .map_err(|e| FerricastError::Decode(format!("VA-API: PPS unescape: {e}")))?;
        let bits = h264_reader::rbsp::BitReader::new(Cursor::new(rbsp.as_ref()));
        let parsed = PicParameterSet::from_bits(&self.parser, bits)
            .map_err(|e| FerricastError::Decode(format!("VA-API: PPS parse: {e:?}")))?;
        let id = u32::from(parsed.pic_parameter_set_id.id());
        self.pps.insert(id, convert_pps(&parsed));
        self.parser.put_pic_param_set(parsed);
        Ok(())
    }

    fn picture(
        &mut self,
        nals: &[&[u8]],
        timestamp_us: u64,
        out: &mut Vec<CapturedFrame>,
    ) -> WorkerResult<()> {
        let Some(first) = nals.first() else {
            return Ok(());
        };
        let rbsp = decode_nal(first)
            .map_err(|e| FerricastError::Decode(format!("VA-API: slice unescape: {e}")))?;
        let pps = self
            .pps
            .get(&slice::peek_pps_id(&rbsp)?)
            .ok_or_else(|| FerricastError::Decode("VA-API: slice names a missing PPS".into()))?
            .clone();
        let sps = self
            .sps
            .get(&pps.sps_id)
            .ok_or_else(|| FerricastError::Decode("VA-API: PPS names a missing SPS".into()))?
            .clone();
        sps.check_supported(&pps)?;
        sps.check_valid()?;
        let header = slice::parse(&rbsp, first[0], &sps, &pps)?;
        if self.active.as_ref().is_none_or(|active| active.sps != sps) {
            self.flush(out)?;
            self.active = Some(Active::new(
                &self.display,
                &self.config,
                &sps,
                self.low_latency,
            )?);
        }
        let Some(active) = self.active.as_mut() else {
            return Ok(());
        };
        if !header.idr
            && header.slice_type.is_inter()
            && active.dpb.reference_frames().next().is_none()
        {
            return Ok(());
        }
        active.decode(nals, &sps, &pps, timestamp_us, self.format, out)?;
        Ok(())
    }

    fn flush(&mut self, out: &mut Vec<CapturedFrame>) -> WorkerResult<()> {
        match self.active.as_mut() {
            Some(active) => Ok(active.flush(self.format, out)?),
            None => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Readback {
    Unknown,
    Derive,
    Get,
}

struct Active {
    readback: Cell<Readback>,
    display: Rc<Display>,
    sps: Sps,
    ctx: Rc<Context>,
    surfaces: Vec<Rc<Surface<()>>>,
    free: Vec<usize>,
    dpb: Dpb,
}

impl Active {
    fn new(display: &Rc<Display>, config: &Config, sps: &Sps, low_latency: bool) -> Result<Self> {
        let dpb = Dpb::new(sps, low_latency);
        let (width, height) = sps.coded_size();
        let count = dpb.capacity() + 1;
        let surfaces = display
            .create_surfaces::<()>(
                VA_RT_FORMAT_YUV420,
                Some(VA_FOURCC_NV12),
                width,
                height,
                None,
                vec![(); count],
            )
            .map_err(|e| FerricastError::Decode(format!("VA-API create_surfaces: {e:?}")))?;
        let ctx = display
            .create_context::<()>(config, width, height, Some(&surfaces), true)
            .map_err(|e| FerricastError::Decode(format!("VA-API create_context: {e:?}")))?;
        tracing::info!(
            width,
            height,
            surfaces = count,
            "VA-API H.264 context ready"
        );
        Ok(Self {
            readback: Cell::new(Readback::Unknown),
            display: Rc::clone(display),
            sps: sps.clone(),
            ctx,
            surfaces: surfaces.into_iter().map(Rc::new).collect(),
            free: (0..count).rev().collect(),
            dpb,
        })
    }

    fn flush(&mut self, format: PixelFormat, out: &mut Vec<CapturedFrame>) -> Result<()> {
        let released = self.dpb.flush();
        self.release(released, out, format)
    }

    /// Returns surfaces to the pool before reading anything back, and
    /// keeps reading past a failed picture, so one bad readback neither
    /// leaks handles nor drops the pictures after it. The first error is
    /// reported once everything has been released.
    fn release(
        &mut self,
        released: Released,
        out: &mut Vec<CapturedFrame>,
        format: PixelFormat,
    ) -> Result<()> {
        self.free.extend(released.free);
        let mut first_error = None;
        for output in released.output {
            match self.read(output.handle, output.timestamp_us, format) {
                Ok(frame) => out.push(frame),
                Err(error) => {
                    tracing::warn!(%error, handle = output.handle, "VA-API readback failed");
                    first_error.get_or_insert(error);
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn take_surface(&mut self, out: &mut Vec<CapturedFrame>, format: PixelFormat) -> Result<usize> {
        if self.free.is_empty() {
            let released = self.dpb.make_room();
            self.release(released, out, format)?;
        }
        self.free
            .pop()
            .ok_or_else(|| FerricastError::Decode("VA-API: no free surface".into()))
    }

    fn decode(
        &mut self,
        nals: &[&[u8]],
        sps: &Sps,
        pps: &Pps,
        timestamp_us: u64,
        format: PixelFormat,
        out: &mut Vec<CapturedFrame>,
    ) -> Result<()> {
        let mut slices = Vec::with_capacity(nals.len());
        for nal in nals {
            let rbsp = decode_nal(nal)
                .map_err(|e| FerricastError::Decode(format!("VA-API: slice unescape: {e}")))?;
            let header = slice::parse(&rbsp, nal[0], sps, pps)?;
            slices.push((*nal, header));
        }
        let Some((_, first)) = slices.first() else {
            return Ok(());
        };
        let first = first.clone();
        if slices
            .iter()
            .any(|(_, header)| header.pps_id != first.pps_id)
        {
            return Err(FerricastError::Decode(
                "VA-API: slices of one picture use different PPS".into(),
            ));
        }
        let Some((current, released)) = self.dpb.start(&first) else {
            return Ok(());
        };
        self.release(released, out, format)?;
        let handle = self.take_surface(out, format)?;
        if let Err(error) = self.submit(handle, &slices, sps, pps, &current, timestamp_us) {
            // The picture never reached the DPB, so nothing else will
            // hand its surface back.
            self.free.push(handle);
            return Err(error);
        }
        let finished = self.dpb.finish(current, &first, handle, timestamp_us);
        self.release(finished, out, format)
    }

    fn submit(
        &self,
        handle: usize,
        slices: &[(&[u8], SliceHeader)],
        sps: &Sps,
        pps: &Pps,
        current: &super::dpb::Current,
        timestamp_us: u64,
    ) -> Result<()> {
        let Some((_, first)) = slices.first() else {
            return Ok(());
        };
        let surface = Rc::clone(&self.surfaces[handle]);
        let mut picture = Picture::new(timestamp_us, Rc::clone(&self.ctx), Rc::clone(&surface));
        let pic_param = self.pic_param(sps, pps, first, current, surface.id());
        self.add(
            &mut picture,
            BufferType::PictureParameter(PictureParameter::H264(pic_param)),
        )?;
        self.add(
            &mut picture,
            BufferType::IQMatrix(IQMatrix::H264(IQMatrixBufferH264::new(
                [[16; 16]; 6],
                [[16; 64]; 2],
            ))),
        )?;
        for (nal, header) in slices {
            let lists = self.dpb.ref_lists(header, current);
            let param = self.slice_param(nal, header, &lists);
            self.add(
                &mut picture,
                BufferType::SliceParameter(SliceParameter::H264(param)),
            )?;
            self.add(&mut picture, BufferType::SliceData(nal.to_vec()))?;
        }
        let ended = picture
            .begin::<()>()
            .map_err(|e| FerricastError::Decode(format!("vaBeginPicture: {e:?}")))?
            .render()
            .map_err(|e| FerricastError::Decode(format!("vaRenderPicture: {e:?}")))?
            .end()
            .map_err(|e| FerricastError::Decode(format!("vaEndPicture: {e:?}")))?;
        ended
            .sync::<()>()
            .map_err(|(e, _)| FerricastError::Decode(format!("vaSyncSurface: {e:?}")))?;
        Ok(())
    }

    fn add(
        &self,
        picture: &mut Picture<PictureNew, Rc<Surface<()>>>,
        buffer: BufferType,
    ) -> Result<()> {
        let buffer = self
            .ctx
            .create_buffer(buffer)
            .map_err(|e| FerricastError::Decode(format!("VA-API create_buffer: {e:?}")))?;
        picture.add_buffer(buffer);
        Ok(())
    }

    fn va_picture(&self, frame: &DpbFrame) -> PictureH264 {
        let Some(surface) = frame.handle.and_then(|handle| self.surfaces.get(handle)) else {
            return invalid_picture();
        };
        let (index, flags) = match frame.long_term_idx {
            Some(idx) => (idx, VA_PICTURE_H264_LONG_TERM_REFERENCE),
            None => (frame.frame_num, VA_PICTURE_H264_SHORT_TERM_REFERENCE),
        };
        PictureH264::new(surface.id(), index, flags, frame.top_poc, frame.bottom_poc)
    }

    fn pic_param(
        &self,
        sps: &Sps,
        pps: &Pps,
        header: &SliceHeader,
        current: &super::dpb::Current,
        surface_id: u32,
    ) -> PictureParameterBufferH264 {
        let mut references: Vec<PictureH264> = self
            .dpb
            .reference_frames()
            .take(16)
            .map(|frame| self.va_picture(frame))
            .collect();
        references.resize_with(16, invalid_picture);
        let references: [PictureH264; 16] = references
            .try_into()
            .unwrap_or_else(|_| std::array::from_fn(|_| invalid_picture()));
        let (log2_max_pic_order_cnt_lsb_minus4, poc_type, delta_pic_order_always_zero) =
            match &sps.poc {
                PocKind::Lsb { log2_max_lsb } => (log2_max_lsb - 4, 0, 0),
                PocKind::Cycle {
                    delta_always_zero, ..
                } => (0, 1, u32::from(*delta_always_zero)),
                PocKind::FrameNum => (0, 2, 0),
            };
        let seq_fields = H264SeqFields::new(
            u32::from(sps.chroma_format_idc),
            0,
            0,
            u32::from(sps.frame_mbs_only),
            0,
            u32::from(sps.direct_8x8_inference),
            u32::from(sps.profile_idc != 66 && sps.level_idc >= 31),
            sps.log2_max_frame_num - 4,
            poc_type,
            log2_max_pic_order_cnt_lsb_minus4,
            delta_pic_order_always_zero,
        );
        let pic_fields = H264PicFields::new(
            u32::from(pps.entropy_cabac),
            u32::from(pps.weighted_pred),
            u32::from(pps.weighted_bipred_idc),
            u32::from(pps.transform_8x8_mode),
            0,
            u32::from(pps.constrained_intra_pred),
            u32::from(pps.bottom_field_poc_present),
            u32::from(pps.deblocking_control_present),
            u32::from(pps.redundant_pic_cnt_present),
            u32::from(header.nal_ref_idc != 0),
        );
        PictureParameterBufferH264::new(
            PictureH264::new(
                surface_id,
                current.frame_num,
                0,
                current.top_poc,
                current.bottom_poc,
            ),
            references,
            (sps.width_mbs - 1) as u16,
            (sps.height_mbs - 1) as u16,
            sps.bit_depth_luma - 8,
            sps.bit_depth_chroma - 8,
            sps.max_num_ref_frames as u8,
            &seq_fields,
            0,
            0,
            0,
            pps.pic_init_qp_minus26 as i8,
            pps.pic_init_qs_minus26 as i8,
            pps.chroma_qp_index_offset as i8,
            pps.second_chroma_qp_index_offset as i8,
            &pic_fields,
            header.frame_num as u16,
        )
    }

    fn slice_param(
        &self,
        nal: &[u8],
        header: &SliceHeader,
        lists: &[Vec<Option<usize>>; 2],
    ) -> SliceParameterBufferH264 {
        let list = |entries: &[Option<usize>]| -> [PictureH264; 32] {
            std::array::from_fn(|index| {
                entries
                    .get(index)
                    .copied()
                    .flatten()
                    .map_or_else(invalid_picture, |frame| {
                        self.va_picture(self.dpb.frame(frame))
                    })
            })
        };
        let weights = header.weights.clone().unwrap_or_default();
        let weighted = header.weights.is_some();
        let weighted_l1 = weighted && header.slice_type == SliceType::B;
        let luma = |list: usize, offset: bool| -> [i16; 32] {
            std::array::from_fn(|index| {
                let (weight, delta) = weights.luma[list][index];
                if offset { delta } else { weight }
            })
        };
        let chroma = |list: usize, offset: bool| -> [[i16; 2]; 32] {
            std::array::from_fn(|index| {
                std::array::from_fn(|plane| {
                    let (weight, delta) = weights.chroma[list][index][plane];
                    if offset { delta } else { weight }
                })
            })
        };
        SliceParameterBufferH264::new(
            nal.len() as u32,
            0,
            VA_SLICE_DATA_FLAG_ALL,
            (header.header_bits + 8) as u16,
            header.first_mb_in_slice as u16,
            header.slice_type.code(),
            u8::from(header.direct_spatial_mv_pred),
            header.num_ref_idx_active[0].saturating_sub(1) as u8,
            header.num_ref_idx_active[1].saturating_sub(1) as u8,
            header.cabac_init_idc,
            header.slice_qp_delta as i8,
            header.disable_deblocking_filter_idc,
            header.alpha_c0_offset_div2,
            header.beta_offset_div2,
            list(&lists[0]),
            list(&lists[1]),
            weights.luma_log2_denom,
            weights.chroma_log2_denom,
            u8::from(weighted),
            luma(0, false),
            luma(0, true),
            u8::from(weighted && weights.has_chroma),
            chroma(0, false),
            chroma(0, true),
            u8::from(weighted_l1),
            luma(1, false),
            luma(1, true),
            u8::from(weighted_l1 && weights.has_chroma),
            chroma(1, false),
            chroma(1, true),
        )
    }

    fn get_image<'a>(&self, surface: &'a Surface<()>) -> Result<Image<'a>> {
        let failed =
            |what: &str, e: libva::VaError| FerricastError::Decode(format!("VA-API {what}: {e:?}"));
        let (coded_width, coded_height) = self.sps.coded_size();
        let format = self
            .display
            .query_image_formats()
            .map_err(|e| failed("vaQueryImageFormats", e))?
            .into_iter()
            .find(|format| format.fourcc == VA_FOURCC_NV12)
            .ok_or_else(|| {
                FerricastError::Decode("VA-API driver has no NV12 image format".into())
            })?;
        Image::create_from(
            surface,
            format,
            (coded_width, coded_height),
            (coded_width, coded_height),
        )
        .map_err(|e| failed("vaGetImage", e))
    }

    fn copy_out(&self, handle: usize, how: Readback) -> Result<(Bytes, usize)> {
        let surface = &*self.surfaces[handle];
        surface
            .sync()
            .map_err(|e| FerricastError::Decode(format!("VA-API vaSyncSurface: {e:?}")))?;
        let (coded_width, coded_height) = self.sps.coded_size();
        let derived = match how {
            Readback::Get => None,
            _ => Image::derive_from(surface, (coded_width, coded_height))
                .ok()
                .filter(|image| image.image().format.fourcc == VA_FOURCC_NV12),
        };
        let image = match derived {
            Some(image) => image,
            None => self.get_image(surface)?,
        };
        let info = image.image();
        let data: &[u8] = image.as_ref();
        let (y_offset, uv_offset) = (info.offsets[0] as usize, info.offsets[1] as usize);
        let (x, y, width, height) = self.sps.visible_rect();
        nv12::crop(
            &Nv12Planes {
                y: data.get(y_offset..).unwrap_or_default(),
                y_pitch: info.pitches[0] as usize,
                uv: data.get(uv_offset..).unwrap_or_default(),
                uv_pitch: info.pitches[1] as usize,
            },
            (x as usize, y as usize, width as usize, height as usize),
        )
        .ok_or_else(|| {
            FerricastError::Decode("VA-API: NV12 planes are shorter than the picture".into())
        })
    }

    fn pick_readback(&self, handle: usize) -> Result<(Bytes, usize)> {
        let mut best = [Duration::MAX; 2];
        let mut copy = None;
        for _ in 0..2 {
            for (slot, how) in [Readback::Derive, Readback::Get].into_iter().enumerate() {
                let start = Instant::now();
                match self.copy_out(handle, how) {
                    Ok(done) => {
                        best[slot] = best[slot].min(start.elapsed());
                        copy = Some(done);
                    }
                    Err(error) => tracing::debug!(%error, ?how, "VA-API readback failed"),
                }
            }
        }
        let Some(copy) = copy else {
            // Nothing worked yet; probe again on the next picture rather
            // than locking in a method that just failed.
            return Err(FerricastError::Decode(
                "VA-API: no way to read a picture back".into(),
            ));
        };
        let chosen = if best[1] < best[0] {
            Readback::Get
        } else {
            Readback::Derive
        };
        self.readback.set(chosen);
        tracing::info!(
            derive_ms = best[0].as_secs_f64() * 1000.0,
            get_image_ms = best[1].as_secs_f64() * 1000.0,
            ?chosen,
            "VA-API readback chosen"
        );
        Ok(copy)
    }

    fn read(&self, handle: usize, timestamp_us: u64, format: PixelFormat) -> Result<CapturedFrame> {
        let (data, stride) = match self.readback.get() {
            Readback::Unknown => self.pick_readback(handle)?,
            how => self.copy_out(handle, how)?,
        };
        let (_, _, width, height) = self.sps.visible_rect();
        let size = (width as usize, height as usize);
        let colour = nv12::Colour::signalled(
            self.sps.matrix_coefficients.map(u32::from),
            self.sps.full_range,
            size.1,
        );
        nv12::deliver(data, stride, size, timestamp_us, format, colour)
            .map(CapturedFrame::Cpu)
            .ok_or_else(|| FerricastError::Decode("VA-API: NV12 conversion failed".into()))
    }
}

fn invalid_picture() -> PictureH264 {
    PictureH264::new(VA_INVALID_SURFACE, 0, VA_PICTURE_H264_INVALID, 0, 0)
}

/// Stream-controlled sizes saturate here; [`Sps::check_valid`] then
/// rejects whatever ended up out of range.
fn convert_sps(sps: &SeqParameterSet, rbsp: &[u8]) -> Sps {
    let frame_mbs_only = matches!(sps.frame_mbs_flags, FrameMbsFlags::Frames);
    let restrictions = sps
        .vui_parameters
        .as_ref()
        .and_then(|vui| vui.bitstream_restrictions.as_ref());
    let signal = sps
        .vui_parameters
        .as_ref()
        .and_then(|vui| vui.video_signal_type.as_ref());
    Sps {
        profile_idc: u8::from(sps.profile_idc),
        level_idc: sps.level_idc,
        constraint_set3: u8::from(sps.constraint_flags) & 0x10 != 0,
        chroma_format_idc: match sps.chroma_info.chroma_format {
            ChromaFormat::Monochrome => 0,
            ChromaFormat::YUV420 => 1,
            ChromaFormat::YUV422 => 2,
            ChromaFormat::YUV444 => 3,
            ChromaFormat::Invalid(value) => value as u8,
        },
        separate_colour_plane: sps.chroma_info.separate_colour_plane_flag,
        bit_depth_luma: sps.chroma_info.bit_depth_luma_minus8.saturating_add(8),
        bit_depth_chroma: sps.chroma_info.bit_depth_chroma_minus8.saturating_add(8),
        log2_max_frame_num: u32::from(sps.log2_max_frame_num_minus4) + 4,
        poc: match &sps.pic_order_cnt {
            PicOrderCntType::TypeZero {
                log2_max_pic_order_cnt_lsb_minus4,
            } => PocKind::Lsb {
                log2_max_lsb: u32::from(*log2_max_pic_order_cnt_lsb_minus4) + 4,
            },
            PicOrderCntType::TypeOne {
                delta_pic_order_always_zero_flag,
                offset_for_non_ref_pic,
                offset_for_top_to_bottom_field,
                offsets_for_ref_frame,
            } => PocKind::Cycle {
                delta_always_zero: *delta_pic_order_always_zero_flag,
                offset_for_non_ref_pic: *offset_for_non_ref_pic,
                offset_for_top_to_bottom: *offset_for_top_to_bottom_field,
                offsets_for_ref_frame: offsets_for_ref_frame.clone(),
            },
            PicOrderCntType::TypeTwo => PocKind::FrameNum,
        },
        max_num_ref_frames: sps.max_num_ref_frames,
        gaps_allowed: sps.gaps_in_frame_num_value_allowed_flag,
        frame_mbs_only,
        width_mbs: sps.pic_width_in_mbs_minus1.saturating_add(1),
        height_mbs: sps
            .pic_height_in_map_units_minus1
            .saturating_add(1)
            .saturating_mul(if frame_mbs_only { 1 } else { 2 }),
        direct_8x8_inference: sps.direct_8x8_inference_flag,
        crop: sps.frame_cropping.as_ref().map_or((0, 0, 0, 0), |crop| {
            (
                crop.left_offset,
                crop.right_offset,
                crop.top_offset,
                crop.bottom_offset,
            )
        }),
        num_reorder_frames: restrictions.map(|r| r.max_num_reorder_frames),
        max_dec_frame_buffering: restrictions.map(|r| r.max_dec_frame_buffering),
        scaling_matrix_present: slice::sps_has_scaling_matrix(rbsp),
        matrix_coefficients: signal
            .and_then(|signal| signal.colour_description.as_ref())
            .map(|description| description.matrix_coefficients),
        full_range: signal.is_some_and(|signal| signal.video_full_range_flag),
    }
}

fn convert_pps(pps: &PicParameterSet) -> Pps {
    let extension = pps.extension.as_ref();
    Pps {
        sps_id: u32::from(pps.seq_parameter_set_id.id()),
        entropy_cabac: pps.entropy_coding_mode_flag,
        bottom_field_poc_present: pps.bottom_field_pic_order_in_frame_present_flag,
        slice_groups: pps.slice_groups.is_some(),
        num_ref_idx_default: [
            pps.num_ref_idx_l0_default_active_minus1,
            pps.num_ref_idx_l1_default_active_minus1,
        ],
        weighted_pred: pps.weighted_pred_flag,
        weighted_bipred_idc: pps.weighted_bipred_idc,
        pic_init_qp_minus26: pps.pic_init_qp_minus26,
        pic_init_qs_minus26: pps.pic_init_qs_minus26,
        chroma_qp_index_offset: pps.chroma_qp_index_offset,
        second_chroma_qp_index_offset: extension.map_or(pps.chroma_qp_index_offset, |extra| {
            extra.second_chroma_qp_index_offset
        }),
        deblocking_control_present: pps.deblocking_filter_control_present_flag,
        constrained_intra_pred: pps.constrained_intra_pred_flag,
        redundant_pic_cnt_present: pps.redundant_pic_cnt_present_flag,
        transform_8x8_mode: extension.is_some_and(|extra| extra.transform_8x8_mode_flag),
        scaling_matrix_present: extension.is_some_and(|extra| extra.pic_scaling_matrix.is_some()),
    }
}

fn first_mb_in_slice(nal: &[u8]) -> Result<u32> {
    let rbsp = decode_nal(nal)
        .map_err(|e| FerricastError::Decode(format!("VA-API: slice unescape: {e}")))?;
    slice::BitReader::new(&rbsp).read_ue()
}

pub(super) fn split_annex_b(buf: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut start = None;
    let mut index = 0;
    while index + 3 <= buf.len() {
        if buf[index] == 0 && buf[index + 1] == 0 && buf[index + 2] == 1 {
            if let Some(begin) = start {
                let end = if index > begin && buf[index - 1] == 0 {
                    index - 1
                } else {
                    index
                };
                out.push(&buf[begin..end]);
            }
            index += 3;
            start = Some(index);
            continue;
        }
        index += 1;
    }
    if let Some(begin) = start {
        out.push(&buf[begin..]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annex_b_units_are_split_at_three_and_four_byte_start_codes() {
        let stream = [
            0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 0, 0, 0, 1, 0x65, 4,
        ];
        let units = split_annex_b(&stream);
        assert_eq!(units, [&[0x67, 1, 2][..], &[0x68, 3][..], &[0x65, 4][..]]);
    }

    #[test]
    fn a_stream_without_start_codes_has_no_units() {
        assert!(split_annex_b(&[1, 2, 3, 4]).is_empty());
    }

    struct Simulation {
        output_order: Vec<u64>,
        packet_times: Vec<u64>,
        most_references: usize,
        reference_limit: usize,
        lists_with_gaps: usize,
        lists: usize,
        alignment_bits_checked: usize,
    }

    fn simulate(name: &str) -> Simulation {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../demuxer/tests/fixtures")
            .join(name);
        let mut demuxer = ferricast_demuxer::Demuxer::open(&path).unwrap();
        let mut parser = H264Context::default();
        let mut spss = HashMap::new();
        let mut ppss = HashMap::new();
        let mut dpb: Option<Dpb> = None;
        let mut free: Vec<usize> = (0..32).rev().collect();
        let mut simulation = Simulation {
            output_order: Vec::new(),
            packet_times: Vec::new(),
            most_references: 0,
            reference_limit: 0,
            lists_with_gaps: 0,
            lists: 0,
            alignment_bits_checked: 0,
        };
        fn take(released: Released, output: &mut Vec<u64>, free: &mut Vec<usize>) {
            output.extend(released.output.iter().map(|o| o.timestamp_us));
            free.extend(released.free);
        }
        while let Some(packet) = demuxer.next_frame().unwrap() {
            simulation.packet_times.push(packet.timestamp_us);
            let mut slices = Vec::new();
            for nal in split_annex_b(&packet.data) {
                let rbsp = decode_nal(nal).unwrap();
                let bits = h264_reader::rbsp::BitReader::new(Cursor::new(rbsp.as_ref()));
                match nal[0] & 0x1f {
                    7 => {
                        let parsed = SeqParameterSet::from_bits(bits).unwrap();
                        spss.insert(
                            u32::from(parsed.seq_parameter_set_id.id()),
                            convert_sps(&parsed, &rbsp),
                        );
                        parser.put_seq_param_set(parsed);
                    }
                    8 => {
                        let parsed = PicParameterSet::from_bits(&parser, bits).unwrap();
                        ppss.insert(
                            u32::from(parsed.pic_parameter_set_id.id()),
                            convert_pps(&parsed),
                        );
                        parser.put_pic_param_set(parsed);
                    }
                    1 | 5 => slices.push(nal),
                    _ => {}
                }
            }
            let first = slices[0];
            let rbsp = decode_nal(first).unwrap();
            let pps = &ppss[&slice::peek_pps_id(&rbsp).unwrap()];
            let sps = &spss[&pps.sps_id];
            sps.check_supported(pps).unwrap();
            sps.check_valid().unwrap();
            simulation.reference_limit = sps.max_num_ref_frames as usize;
            let dpb = dpb.get_or_insert_with(|| Dpb::new(sps, false));
            let header = slice::parse(&rbsp, first[0], sps, pps).unwrap();
            let (current, released) = dpb.start(&header).expect("fixture has no frame_num gaps");
            take(released, &mut simulation.output_order, &mut free);
            if free.is_empty() {
                take(dpb.make_room(), &mut simulation.output_order, &mut free);
            }
            let handle = free.pop().expect("a free surface");
            for nal in &slices {
                let rbsp = decode_nal(nal).unwrap();
                let header = slice::parse(&rbsp, nal[0], sps, pps).unwrap();
                if pps.entropy_cabac {
                    let start = header.header_bits as usize;
                    for bit in start..start.next_multiple_of(8) {
                        assert_eq!(rbsp[bit / 8] >> (7 - bit % 8) & 1, 1, "{name}: alignment");
                        simulation.alignment_bits_checked += 1;
                    }
                }
                let lists = dpb.ref_lists(&header, &current);
                let references = dpb.reference_frames().count();
                for (list, active) in lists.iter().zip(header.num_ref_idx_active) {
                    if active == 0 {
                        continue;
                    }
                    simulation.lists += 1;
                    let filled = list.iter().flatten().count();
                    if filled < (active as usize).min(references) {
                        simulation.lists_with_gaps += 1;
                    }
                }
            }
            take(
                dpb.finish(current, &header, handle, packet.timestamp_us),
                &mut simulation.output_order,
                &mut free,
            );
            simulation.most_references = simulation
                .most_references
                .max(dpb.reference_frames().count());
        }
        if let Some(dpb) = dpb.as_mut() {
            take(dpb.flush(), &mut simulation.output_order, &mut free);
        }
        simulation
    }

    #[test]
    fn real_x264_streams_come_out_in_display_order_with_complete_reference_lists() {
        for name in ["h264.mp4", "h264.mov", "h264.mkv"] {
            let run = simulate(name);
            let mut presentation = run.packet_times.clone();
            presentation.sort_unstable();
            assert_eq!(run.output_order, presentation, "{name}");
            assert_eq!(run.lists_with_gaps, 0, "{name}");
            assert!(run.lists > 0, "{name}");
            assert!(run.alignment_bits_checked > 0, "{name}");
            assert!(run.most_references > 1, "{name}");
            assert!(
                run.most_references <= run.reference_limit,
                "{name}: {} references for a limit of {}",
                run.most_references,
                run.reference_limit
            );
        }
    }
}
