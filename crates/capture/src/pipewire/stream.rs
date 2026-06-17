//! PipeWire stream worker.
//!
//! Mirrors the wlx-capture flow as closely as possible — that is the
//! production-tested reference for screencast on Mutter / KWin / wlroots:
//!
//! * `Context::connect(None)` — the user's regular PipeWire daemon, NOT
//!   `OpenPipeWireRemote`'s private fd. Once the portal authorizes the
//!   source, the node is also visible on the regular daemon, and going
//!   through the regular daemon avoids the per-portal-implementation
//!   quirks of the private fd.
//! * `MainLoop` cloned by value into closures (it has an internal Rc).
//! * EnumFormat list with one pod per format carrying a
//!   `MANDATORY | DONT_FIXATE` modifier choice + a SHM-only fallback pod.
//! * `param_changed` unconditionally `update_params`'s `[Buffers, Meta]`
//!   without branching on `MODIFIER_FIXATION_REQUIRED`.
//! * `process` drains stale buffers and processes only the newest.
//!
//! Buffer reading: CPU-mapped (MemFd / MemPtr) buffers are handled
//! directly; DmaBuf with a LINEAR / INVALID modifier is `mmap`'d
//! manually. Tiled / compressed modifiers need GPU-import and are
//! dropped with a warning until that path is wired up.

use std::os::fd::RawFd;
use std::ptr;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use ferricast_core::{
    CaptureConfig, CapturedFrame, DmaBufPlane, FerricastError, GpuFrame, RawFrame, Result,
};
use tokio::sync::mpsc;
use tracing::{debug, error, info, trace, warn};

use pipewire as pw;
use pw::context::Context;
use pw::main_loop::MainLoop;
use pw::properties::properties;
use pw::spa::buffer::DataType;
use pw::spa::param::ParamType;
use pw::spa::param::format::{MediaSubtype, MediaType};
use pw::spa::param::format_utils;
use pw::spa::param::video::VideoInfoRaw;
use pw::spa::utils::Direction;
use pw::stream::{Stream, StreamFlags, StreamRef, StreamState};

use super::format::{self, EnumFormatParams, GpuFormat, NegotiatedFormat};
use super::portal::PortalStream;
use super::vulkan::VulkanImporter;

pub(super) type SharedFormat = Arc<RwLock<Option<NegotiatedFormat>>>;

/// Sent over `pipewire::channel` to wake the PW main loop and stop it.
struct Terminate;

/// Per-stream state shared between PipeWire callbacks.
struct UserData {
    video_info: VideoInfoRaw,
    /// Last fully-fixated negotiation, mirrored into `shared` for
    /// outside readers and reused inside `process`.
    negotiated: Option<NegotiatedFormat>,
    shared: SharedFormat,
    /// Set when we've logged the first dequeued buffer (regardless of
    /// whether it had data) so the log doesn't spam.
    first_buffer_logged: bool,
    /// Set when we've logged the first non-empty frame forwarded
    /// downstream — the moment capture is actually working.
    first_frame_logged: bool,
    /// Target frame period derived from `CaptureConfig::fps`. Used
    /// to throttle Vulkan readback so we don't waste CPU on
    /// memcpy'ing 8 MB at 60 fps when the consumer only encodes at
    /// 30 fps. Without this, the worker dutifully runs the full
    /// Vulkan import + blit + memcpy for every buffer PipeWire
    /// delivers (which on a 60 Hz compositor is 60/s = ~480 MB/s of
    /// memory bandwidth) even though the downstream encoder only
    /// consumes every second frame.
    frame_period: Duration,
    /// Wallclock instant of the last buffer we *processed* (not
    /// dropped). Used together with `frame_period` to decide
    /// whether to skip the next arriving buffer.
    last_processed_at: Option<Instant>,
    /// `Some` when Vulkan came up successfully and we can import
    /// the dmabuf modifiers we advertised. The PW worker thread
    /// invokes `importer.readback(...)` directly inside
    /// `handle_process` for non-linear DmaBuf frames so the GPU
    /// blit-and-copy overlaps with the encoder doing CPU work on the
    /// previous frame. `None` collapses the dmabuf path to a warning
    /// + drop and we rely on the SHM fallback.
    importer: Option<Arc<VulkanImporter>>,
}

pub(super) struct WorkerHandle {
    pub frames: mpsc::Receiver<CapturedFrame>,
    pub errors: mpsc::Receiver<String>,
    terminator: pw::channel::Sender<Terminate>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl WorkerHandle {
    pub(super) fn shutdown(&mut self) {
        let _ = self.terminator.send(Terminate);
        if let Some(handle) = self.join.take() {
            if let Err(panic) = handle.join() {
                error!(?panic, "PipeWire worker panicked during shutdown");
            }
        }
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub(super) fn spawn(
    portal: PortalStream,
    config: CaptureConfig,
    shared: SharedFormat,
) -> Result<WorkerHandle> {
    // Capacity 1 with `try_send`: when the consumer (segmenter) falls
    // behind, additional frames are dropped on the producer side
    // instead of queueing. That, combined with `next_frame()`'s
    // drain-to-newest, ensures the encoder always sees the freshest
    // available frame and never a back-to-back burst of stale ones
    // after a capture stall (which the viewer perceives as "freeze
    // then catch-up jump").
    let (frame_tx, frame_rx) = mpsc::channel::<CapturedFrame>(1);
    let (error_tx, error_rx) = mpsc::channel::<String>(1);
    let (term_tx, term_rx) = pw::channel::channel::<Terminate>();

    let join = std::thread::Builder::new()
        .name("ferricast-pw".into())
        .spawn(move || {
            if let Err(e) = run(portal, config, shared, frame_tx, &error_tx, term_rx) {
                error!(error = %e, "PipeWire worker exited with error");
                let _ = error_tx.try_send(e.to_string());
            }
        })
        .map_err(|e| FerricastError::Capture(format!("spawn PW thread: {e}")))?;

    Ok(WorkerHandle {
        frames: frame_rx,
        errors: error_rx,
        terminator: term_tx,
        join: Some(join),
    })
}

fn run(
    portal: PortalStream,
    config: CaptureConfig,
    shared: SharedFormat,
    frame_tx: mpsc::Sender<CapturedFrame>,
    error_tx: &mpsc::Sender<String>,
    term_rx: pw::channel::Receiver<Terminate>,
) -> Result<()> {
    pw::init();

    let mainloop =
        MainLoop::new(None).map_err(|e| FerricastError::Capture(format!("MainLoop: {e}")))?;
    let context =
        Context::new(&mainloop).map_err(|e| FerricastError::Capture(format!("Context: {e}")))?;

    // Connect to the regular per-user PipeWire daemon. The portal has
    // already authorized our access to `node_id` so it shows up here.
    let core = context
        .connect(None)
        .map_err(|e| FerricastError::Capture(format!("Context::connect: {e}")))?;

    let stream = Stream::new(
        &core,
        "ferricast-capture",
        properties! {
            *pw::keys::MEDIA_TYPE => "Video",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Screen",
        },
    )
    .map_err(|e| FerricastError::Capture(format!("Stream::new: {e}")))?;

    let enum_params = EnumFormatParams {
        default_width: portal.size_hint.map(|(w, _)| w).unwrap_or(1920),
        default_height: portal.size_hint.map(|(_, h)| h).unwrap_or(1080),
        default_fps: config.fps.max(1),
    };

    // Try to bring Vulkan up — gives us GPU-supported DRM modifiers
    // for the EnumFormat advertise and a real DmaBuf import path in
    // `process`. Soft-failure: we log and keep going with SHM only.
    let importer = match VulkanImporter::new() {
        Ok(imp) => Some(Arc::new(imp)),
        Err(e) => {
            warn!(error = %e, "Vulkan unavailable, falling back to SHM-only capture");
            None
        }
    };

    // Build the GPU format list (one entry per supported pixel format
    // with the modifiers the GPU actually exposes for it). Empty when
    // Vulkan isn't available — `initial_enum_format_list` then
    // collapses to the SHM fallback pod alone.
    let gpu_formats: Vec<GpuFormat> = match importer.as_ref() {
        Some(imp) => format::SUPPORTED_FORMATS
            .iter()
            .filter_map(|f| {
                let mods = imp.supported_modifiers(*f);
                if mods.is_empty() {
                    None
                } else {
                    debug!(format = ?f, modifiers = mods.len(), "GPU format supported");
                    Some(GpuFormat {
                        format: *f,
                        modifiers: mods,
                    })
                }
            })
            .collect(),
        None => Vec::new(),
    };

    let user_data = UserData {
        video_info: VideoInfoRaw::default(),
        negotiated: None,
        shared: Arc::clone(&shared),
        first_buffer_logged: false,
        first_frame_logged: false,
        // CaptureConfig.fps is the manager-imposed cap (which
        // already factors in DeviceCapabilities::max_fps). Use
        // 0.95 of the period so a buffer that arrives ~one
        // refresh tick early still gets processed instead of
        // being dropped by a hair.
        frame_period: Duration::from_secs_f64(0.95 / (config.fps.max(1) as f64)),
        last_processed_at: None,
        importer,
    };

    let _listener = stream
        .add_local_listener_with_user_data(user_data)
        .state_changed({
            let mainloop = mainloop.clone();
            let error_tx = error_tx.clone();
            move |_, _, old, new| {
                info!(?old, ?new, "stream state changed");
                if let StreamState::Error(msg) = new {
                    error!(%msg, "stream entered Error state");
                    let _ = error_tx.try_send(msg.to_string());
                    mainloop.quit();
                }
            }
        })
        .param_changed(|stream, ud, id, param| {
            handle_param_changed(stream, ud, id, param);
        })
        .process({
            let frame_tx = frame_tx.clone();
            move |stream, ud| {
                handle_process(stream, ud, &frame_tx);
            }
        })
        .register()
        .map_err(|e| FerricastError::Capture(format!("listener register: {e}")))?;

    let pod_storage = format::initial_enum_format_list(&enum_params, &gpu_formats);
    info!(
        pods = pod_storage.len(),
        gpu_formats = gpu_formats.len(),
        "offering EnumFormat pods"
    );
    let mut pods: Vec<&pw::spa::pod::Pod> =
        pod_storage.iter().map(|b| format::pod_view(b)).collect();

    stream
        .connect(
            Direction::Input,
            Some(portal.node_id),
            StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS,
            pods.as_mut_slice(),
        )
        .map_err(|e| FerricastError::Capture(format!("Stream::connect: {e}")))?;
    info!(
        node_id = portal.node_id,
        "stream.connect ok, entering main loop"
    );

    let _term_attach = term_rx.attach(mainloop.loop_(), {
        let mainloop = mainloop.clone();
        move |_| {
            info!("worker received Terminate");
            mainloop.quit();
        }
    });

    mainloop.run();

    info!("PipeWire main loop exited cleanly");
    Ok(())
}

fn handle_param_changed(
    stream: &StreamRef,
    ud: &mut UserData,
    id: u32,
    param: Option<&pw::spa::pod::Pod>,
) {
    let Some(param) = param else { return };
    if id != ParamType::Format.as_raw() {
        return;
    }

    let (media_type, media_subtype) = match format_utils::parse_format(param) {
        Ok(v) => v,
        Err(e) => {
            warn!(error = ?e, "ignoring unparseable Format pod");
            return;
        }
    };
    if media_type != MediaType::Video || media_subtype != MediaSubtype::Raw {
        return;
    }

    if let Err(e) = ud.video_info.parse(param) {
        warn!(error = ?e, "VideoInfoRaw::parse failed");
        return;
    }

    debug!(
        format = ?ud.video_info.format(),
        size = ?ud.video_info.size(),
        modifier = ud.video_info.modifier(),
        flags = ?ud.video_info.flags(),
        "compositor proposed format"
    );

    match NegotiatedFormat::from_video_info(&ud.video_info) {
        Ok(neg) => {
            // Format-relevant change → drop any cached `VkImage`s
            // pinned to the previous fd / size / modifier so the
            // next frame re-imports cleanly.
            let format_changed = match ud.negotiated {
                Some(prev) => {
                    prev.width != neg.width
                        || prev.height != neg.height
                        || prev.spa_format != neg.spa_format
                        || prev.modifier != neg.modifier
                }
                None => true,
            };
            if format_changed {
                if let Some(imp) = ud.importer.as_ref() {
                    imp.reset_cache();
                }
            }

            ud.negotiated = Some(neg);
            if let Ok(mut g) = ud.shared.write() {
                *g = Some(neg);
            }
            info!(
                width = neg.width,
                height = neg.height,
                format = ?neg.spa_format,
                modifier = ?neg.modifier,
                "format accepted, sending Buffers + Meta"
            );
        }
        Err(e) => {
            warn!(error = %e, "negotiated format unsupported, not acknowledging");
            return;
        }
    }

    let buffers_bytes = format::param_buffers_bytes();
    let meta_bytes = format::param_meta_header_bytes();
    let mut pods = [
        format::pod_view(&buffers_bytes),
        format::pod_view(&meta_bytes),
    ];
    if let Err(e) = stream.update_params(&mut pods) {
        warn!(error = %e, "update_params(Buffers + Meta) failed");
    }
}

fn handle_process(stream: &StreamRef, ud: &mut UserData, frame_tx: &mpsc::Sender<CapturedFrame>) {
    let Some(neg) = ud.negotiated else {
        debug!("process tick before format negotiated, dropping all buffers");
        while stream.dequeue_buffer().is_some() {}
        return;
    };

    // Throttle to `ud.frame_period`. The Wayland compositor emits
    // a buffer per refresh (60 Hz typical) regardless of what
    // framerate the consumer wants. Doing the full Vulkan import
    // + blit + ~8 MB readback on every one of those buffers when
    // the encoder only consumes at 30 fps is pure CPU/PCIe waste
    // (≈ 240 MB/s of memcpy + GPU work we throw away). When the
    // arrival is too soon after the last processed frame, drain
    // and discard everything — quick path, no GPU work, no
    // memcpy. PW reuses the buffer slot on the next round.
    let now = Instant::now();
    if let Some(prev) = ud.last_processed_at {
        if now.duration_since(prev) < ud.frame_period {
            let mut dropped = 0;
            while stream.dequeue_buffer().is_some() {
                dropped += 1;
            }
            if dropped > 0 {
                trace!(dropped, "throttle: discarded buffer(s) before Vulkan path");
            }
            return;
        }
    }

    // Drain stale buffers and keep only the newest.
    let mut drained = 0;
    let mut newest = None;
    while let Some(b) = stream.dequeue_buffer() {
        drained += 1;
        newest = Some(b);
    }
    let Some(mut buffer) = newest else {
        trace!("process tick with no buffer available");
        return;
    };
    if drained > 1 {
        trace!(drained, "drained stale buffers, keeping newest");
    }
    // Anchor for the next throttle decision *only* once we've
    // committed to doing real work for this buffer. Otherwise a
    // sequence of quick drops would never re-anchor and we'd
    // accidentally process the very next buffer instead of waiting
    // a full period.
    ud.last_processed_at = Some(now);

    let datas = buffer.datas_mut();
    let Some(plane) = datas.first_mut() else {
        warn!("PipeWire buffer had no data planes");
        return;
    };

    let chunk = plane.chunk();
    let mut stride = chunk.stride() as u32;
    let chunk_size = chunk.size();
    let chunk_offset = chunk.offset();
    let plane_type = plane.type_();
    let plane_fd = plane.as_raw().fd;
    let plane_maxsize = plane.as_raw().maxsize;

    if !ud.first_buffer_logged {
        info!(
            ?plane_type,
            chunk_size,
            chunk_offset,
            stride,
            plane_fd,
            plane_maxsize,
            "first buffer received from PipeWire"
        );
        ud.first_buffer_logged = true;
    } else {
        trace!(
            ?plane_type,
            chunk_size, chunk_offset, stride, "buffer dequeued"
        );
    }

    if chunk_size == 0 {
        trace!("buffer empty (chunk.size == 0), source still warming up");
        return;
    }

    if stride == 0 {
        stride = neg.width.saturating_mul(bytes_per_pixel(neg.pixel_format));
    }

    let plane_size = (stride as usize).saturating_mul(neg.height as usize);
    let timestamp_us = now_us();

    // Hot decision tree:
    //
    // * shm (MemFd/MemPtr) → CPU bytes, emit `Cpu` directly.
    // * DmaBuf with LINEAR/INVALID modifier → mmap-read; emit `Cpu`
    //   (no point in deferring since mmap is cheap and any consumer
    //   needs the bytes anyway).
    // * DmaBuf with tiled/compressed modifier + GPU importer → emit
    //   `Gpu` carrying the fd + importer. The downstream encoder
    //   either consumes the fd directly (VA-API / NVENC after the
    //   dmabuf-input commits land) or calls `into_cpu()`, which
    //   triggers a synchronous Vulkan readback via the importer. We
    //   no longer eagerly pipeline the readback in the producer
    //   thread — that hid ~10-15 ms of blit latency for the x264
    //   path but turned every frame into a wasted memcpy for the
    //   GPU encoder paths. Net: x264 picks up ~10-15 ms per frame,
    //   GPU encoders skip a whole readback. Acceptable trade.
    // * DmaBuf with tiled modifier + no importer → drop with
    //   warning. Should be unreachable because we wouldn't have
    //   negotiated those modifiers in the first place.
    let captured: CapturedFrame = match plane_type {
        DataType::MemFd | DataType::MemPtr => {
            let Some(bytes) = read_cpu_buffer(plane, chunk_offset, plane_size) else {
                return;
            };
            CapturedFrame::Cpu(RawFrame {
                width: neg.width,
                height: neg.height,
                stride,
                format: neg.pixel_format,
                data: bytes,
                timestamp_us,
            })
        }
        DataType::DmaBuf if neg.modifier_is_cpu_readable() => {
            let Some(bytes) = read_dmabuf_mmap(plane, chunk_offset, plane_size) else {
                return;
            };
            CapturedFrame::Cpu(RawFrame {
                width: neg.width,
                height: neg.height,
                stride,
                format: neg.pixel_format,
                data: bytes,
                timestamp_us,
            })
        }
        DataType::DmaBuf => {
            // We require the importer to have come up so that
            // downstream CPU encoders (x264) can still read back. If
            // it didn't, dropping is the safe move — emitting a Gpu
            // frame with `importer: None` would just defer the same
            // failure to `into_cpu()` later.
            let Some(importer) = ud.importer.as_ref() else {
                warn!(
                    modifier = ?neg.modifier,
                    "DmaBuf with non-linear modifier and no Vulkan importer — frame dropped"
                );
                return;
            };
            let modifier = neg.modifier.unwrap_or(0);
            let raw = plane.as_raw();
            if raw.fd < 0 {
                warn!("DmaBuf plane has invalid fd");
                return;
            }
            // DMA-BUF object size: use maxsize directly (the total
            // buffer including tiling padding). Do NOT subtract
            // chunk_offset — VA-API's objects[0].size expects the
            // full fd size, and the plane offset is stored separately
            // in DmaBufPlane.offset.
            let dmabuf_size = if raw.maxsize > 0 {
                raw.maxsize
            } else {
                stride.saturating_mul(neg.height)
            };
            // GPU emission: hand the encoder a `GpuFrame` carrying
            // the raw fd + the importer for lazy readback. GPU-aware
            // encoders (VA-API, NVENC) keep the fd zero-copy;
            // CPU-only encoders (x264) call `frame.into_cpu()` which
            // triggers `VulkanImporter::readback` via the trait. The
            // pipelined-readback we used to do here is gone — see
            // the decision-tree comment above for the trade-off.
            CapturedFrame::Gpu(GpuFrame {
                width: neg.width,
                height: neg.height,
                stride,
                format: neg.pixel_format,
                timestamp_us,
                plane: DmaBufPlane {
                    fd: raw.fd as RawFd,
                    offset: chunk_offset,
                    stride,
                    modifier,
                    size: dmabuf_size,
                },
                importer: Some(importer.clone() as Arc<dyn ferricast_core::DmaBufImporter>),
            })
        }
        other => {
            warn!(?other, "unexpected SPA buffer data type");
            return;
        }
    };

    if !ud.first_frame_logged {
        info!(
            width = neg.width,
            height = neg.height,
            stride,
            kind = match &captured {
                CapturedFrame::Cpu(_) => "cpu",
                CapturedFrame::Gpu(_) => "gpu",
            },
            "first frame ready, forwarding to consumer"
        );
        ud.first_frame_logged = true;
    }

    match frame_tx.try_send(captured) {
        Ok(()) => {}
        Err(mpsc::error::TrySendError::Full(_)) => {
            trace!("frame dropped, downstream consumer fell behind");
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            debug!("frame channel closed");
        }
    }
}

fn bytes_per_pixel(format: ferricast_core::PixelFormat) -> u32 {
    use ferricast_core::PixelFormat;
    match format {
        PixelFormat::Bgra | PixelFormat::Rgba => 4,
        PixelFormat::Nv12 | PixelFormat::I420 => 0,
    }
}

fn read_cpu_buffer(
    plane: &mut pw::spa::buffer::Data,
    chunk_offset: u32,
    plane_size: usize,
) -> Option<Bytes> {
    let slice = plane.data()?;
    let start = (chunk_offset as usize).min(slice.len());
    let end = start.saturating_add(plane_size).min(slice.len());
    if end <= start {
        return None;
    }
    Some(Bytes::copy_from_slice(&slice[start..end]))
}

/// Linear / INVALID-modifier DmaBuf: we can read the bytes directly
/// via `mmap`. Used for the trivial case (Mutter falling back to
/// LINEAR even though it negotiated DmaBuf).
fn read_dmabuf_mmap(
    plane: &mut pw::spa::buffer::Data,
    chunk_offset: u32,
    plane_size: usize,
) -> Option<Bytes> {
    let raw = plane.as_raw();

    let fd = raw.fd as RawFd;
    if fd < 0 {
        warn!("DmaBuf plane has invalid fd");
        return None;
    }

    let mapoffset = raw.mapoffset as i64;
    let maxsize = raw.maxsize as usize;
    if maxsize == 0 {
        warn!("DmaBuf plane has zero maxsize");
        return None;
    }

    let ptr = unsafe {
        libc::mmap(
            ptr::null_mut(),
            maxsize,
            libc::PROT_READ,
            libc::MAP_SHARED,
            fd,
            mapoffset,
        )
    };
    if ptr == libc::MAP_FAILED {
        let errno = std::io::Error::last_os_error();
        warn!(%errno, fd, mapoffset, maxsize, "mmap(DmaBuf) failed");
        return None;
    }

    let bytes = unsafe {
        let slice = std::slice::from_raw_parts(ptr as *const u8, maxsize);
        let start = (chunk_offset as usize).min(slice.len());
        let end = start.saturating_add(plane_size).min(slice.len());
        if end <= start {
            None
        } else {
            Some(Bytes::copy_from_slice(&slice[start..end]))
        }
    };

    unsafe {
        libc::munmap(ptr, maxsize);
    }

    bytes
}

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}
