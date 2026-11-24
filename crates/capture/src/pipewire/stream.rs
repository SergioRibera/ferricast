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
//! Buffer reading: CPU-mapped (MemFd / MemPtr) buffers and DmaBuf
//! with a LINEAR / INVALID modifier are copied out before the buffer
//! goes back to the compositor. Tiled / compressed DmaBufs are emitted
//! as `GpuFrame`s that keep their buffer dequeued until downstream
//! drops them (see [`super::lease`]).

use std::cell::RefCell;
use std::os::fd::{BorrowedFd, RawFd};
use std::ptr;
use std::rc::Rc;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use ferricast_core::{
    CaptureConfig, CapturedFrame, DmaBufFd, DmaBufPlane, FerricastError, GpuFrame, RawFrame, Result,
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
use super::lease::{BufferLeases, DequeuedBuffer, Release};
use super::portal::PortalStream;
use crate::vulkan::VulkanImporter;

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
    /// Set once we've logged hitting the DmaBuf hold limit.
    lease_limit_logged: bool,
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
    /// `true` while buffers sit unprocessed in the stream queue
    /// because they arrived inside the throttle window; the throttle
    /// timer processes the newest once the window closes.
    deferred: bool,
    /// `Some` when Vulkan came up successfully and we can import
    /// the dmabuf modifiers we advertised. The PW worker thread
    /// invokes `importer.readback(...)` directly inside
    /// `handle_process` for non-linear DmaBuf frames so the GPU
    /// blit-and-copy overlaps with the encoder doing CPU work on the
    /// previous frame. `None` collapses the dmabuf path to a warning
    /// + drop and we rely on the SHM fallback.
    importer: Option<Arc<VulkanImporter>>,
}

impl UserData {
    /// Whether a buffer arriving at `now` falls inside the throttle
    /// window opened by the last processed frame.
    fn is_throttled(&self, now: Instant) -> bool {
        is_within_period(self.last_processed_at, now, self.frame_period)
    }
}

fn is_within_period(last: Option<Instant>, now: Instant, period: Duration) -> bool {
    last.is_some_and(|prev| now.saturating_duration_since(prev) < period)
}

pub(super) struct WorkerHandle {
    pub frames: mpsc::Receiver<CapturedFrame>,
    pub errors: mpsc::Receiver<String>,
    terminator: pw::channel::Sender<Terminate>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl WorkerHandle {
    /// Stops the worker and waits for its thread off the async runtime.
    pub(super) async fn shutdown(mut self) {
        let _ = self.terminator.send(Terminate);
        let Some(handle) = self.join.take() else {
            return;
        };
        match tokio::task::spawn_blocking(move || handle.join()).await {
            Ok(Ok(())) => {}
            Ok(Err(panic)) => error!(?panic, "PipeWire worker panicked during shutdown"),
            Err(e) => error!(error = %e, "joining the PipeWire worker failed"),
        }
    }
}

impl Drop for WorkerHandle {
    /// Asks the worker to exit and detaches it, so dropping a capture
    /// never blocks the dropping thread on the PipeWire main loop.
    fn drop(&mut self) {
        let _ = self.terminator.send(Terminate);
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

    // `Rc`: the throttle timer below needs the stream too.
    let stream = Rc::new(
        Stream::new(
            &core,
            "ferricast-capture",
            properties! {
                *pw::keys::MEDIA_TYPE => "Video",
                *pw::keys::MEDIA_CATEGORY => "Capture",
                *pw::keys::MEDIA_ROLE => "Screen",
            },
        )
        .map_err(|e| FerricastError::Capture(format!("Stream::new: {e}")))?,
    );

    let enum_params = EnumFormatParams {
        default_width: portal.size_hint.map(|(w, _)| w).unwrap_or(1920),
        default_height: portal.size_hint.map(|(_, h)| h).unwrap_or(1080),
        default_fps: config.fps.max(1),
    };

    // Try to bring Vulkan up — gives us GPU-supported DRM modifiers
    // for the EnumFormat advertise and a real DmaBuf import path in
    // `process`. Soft-failure: we log and keep going with SHM only.
    let importer = match VulkanImporter::new(None) {
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
                let mods = format::pixel_format(*f)
                    .map(|p| imp.supported_modifiers(p))
                    .unwrap_or_default();
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

    // Shared between the stream listener and the throttle timer; both
    // run on this thread's PipeWire loop, never re-entrantly.
    let user_data = Rc::new(RefCell::new(UserData {
        video_info: VideoInfoRaw::default(),
        negotiated: None,
        shared: Arc::clone(&shared),
        first_buffer_logged: false,
        first_frame_logged: false,
        lease_limit_logged: false,
        // CaptureConfig.fps is the manager-imposed cap (which
        // already factors in DeviceCapabilities::max_fps). Use
        // 0.95 of the period so a buffer that arrives ~one
        // refresh tick early still gets processed instead of
        // being dropped by a hair.
        frame_period: Duration::from_secs_f64(0.95 / (config.fps.max(1) as f64)),
        last_processed_at: None,
        deferred: false,
        importer,
    }));
    let frame_period = user_data.borrow().frame_period;

    // Separate from `user_data` so the buffer callbacks never contend
    // with a `user_data` borrow held across `update_params`.
    let (release_tx, release_rx) = pw::channel::channel::<Release>();
    let leases = Rc::new(RefCell::new(BufferLeases::new(release_tx)));

    let _listener = stream
        .add_local_listener_with_user_data(Rc::clone(&user_data))
        .state_changed({
            let mainloop = mainloop.clone();
            let error_tx = error_tx.clone();
            move |_, _, old, new| {
                info!(?old, ?new, "stream state changed");
                if let Some(msg) = fatal_state_transition(&old, &new) {
                    error!(%msg, "PipeWire stream ended");
                    let _ = error_tx.try_send(msg);
                    mainloop.quit();
                }
            }
        })
        .param_changed(|stream, ud, id, param| {
            handle_param_changed(stream, &mut ud.borrow_mut(), id, param);
        })
        .add_buffer({
            let leases = Rc::clone(&leases);
            move |_, _, _| leases.borrow_mut().buffer_added()
        })
        .remove_buffer({
            let leases = Rc::clone(&leases);
            move |_, _, buffer| leases.borrow_mut().buffer_removed(buffer)
        })
        .process({
            let frame_tx = frame_tx.clone();
            let leases = Rc::clone(&leases);
            move |stream, ud| {
                handle_process(
                    stream,
                    &mut ud.borrow_mut(),
                    &mut leases.borrow_mut(),
                    &frame_tx,
                );
            }
        })
        .register()
        .map_err(|e| FerricastError::Capture(format!("listener register: {e}")))?;

    // Flushes buffers that `handle_process` left queued while
    // throttled, once the frame period has expired. Without it the
    // last buffer of a burst would wait for the *next* damage event —
    // on damage-driven compositors that can be never, leaving the
    // receiver on a stale frame.
    let throttle_timer = mainloop.loop_().add_timer({
        let stream = Rc::clone(&stream);
        let user_data = Rc::clone(&user_data);
        let leases = Rc::clone(&leases);
        let frame_tx = frame_tx.clone();
        move |_| {
            let mut ud = user_data.borrow_mut();
            if ud.deferred && !ud.is_throttled(Instant::now()) {
                handle_process(&stream, &mut ud, &mut leases.borrow_mut(), &frame_tx);
            }
        }
    });
    let tick = (frame_period / 4).max(Duration::from_millis(2));
    throttle_timer
        .update_timer(Some(tick), Some(tick))
        .into_result()
        .map_err(|e| FerricastError::Capture(format!("arm throttle timer: {e}")))?;

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
            // DONT_RECONNECT: when the portal node goes away ("Stop
            // sharing", window closed) the session manager must not
            // re-link us to some other video source; the stream
            // drops to Unconnected instead, which
            // `fatal_state_transition` reports.
            StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS | StreamFlags::DONT_RECONNECT,
            pods.as_mut_slice(),
        )
        .map_err(|e| FerricastError::Capture(format!("Stream::connect: {e}")))?;
    info!(
        node_id = portal.node_id,
        "stream.connect ok, entering main loop"
    );

    // Re-queues buffers whose `GpuFrame`s downstream has dropped.
    // Declared after `stream` and `_listener`, so it is detached before
    // either is dropped and never re-queues into a destroyed stream.
    let _release_attach = release_rx.attach(mainloop.loop_(), {
        let stream = Rc::clone(&stream);
        let leases = Rc::clone(&leases);
        move |release| leases.borrow_mut().release(&stream, release)
    });

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

/// Error message for a stream state transition that ends the capture:
/// an explicit `Error`, or falling back to `Unconnected` after the
/// stream had connected (source node removed, user hit "Stop
/// sharing"). Without the latter `next_frame` would wait forever.
fn fatal_state_transition(old: &StreamState, new: &StreamState) -> Option<String> {
    match (old, new) {
        (_, StreamState::Error(msg)) => Some(msg.clone()),
        (
            StreamState::Connecting | StreamState::Paused | StreamState::Streaming,
            StreamState::Unconnected,
        ) => Some("stream disconnected (screencast source removed or sharing stopped)".into()),
        _ => None,
    }
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
            if format_changed && let Some(imp) = ud.importer.as_ref() {
                imp.reset_cache();
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

fn handle_process(
    stream: &StreamRef,
    ud: &mut UserData,
    leases: &mut BufferLeases,
    frame_tx: &mpsc::Sender<CapturedFrame>,
) {
    let Some(neg) = ud.negotiated else {
        debug!("process tick before format negotiated, dropping all buffers");
        while stream.dequeue_buffer().is_some() {}
        return;
    };

    // Throttle to `ud.frame_period`. The Wayland compositor emits
    // a buffer per refresh (60 Hz typical) regardless of what
    // framerate the consumer wants; copying / importing every one
    // when the encoder only consumes at 30 fps is pure waste. When
    // the arrival is too soon after the last processed frame, leave
    // the buffers queued in the stream and let the throttle timer
    // (see `run`) process the newest one once the period expires —
    // dropping them here would lose the final frame of a burst on
    // damage-driven compositors. Holding them for at most one
    // period also back-pressures the compositor, which skips
    // renders while it has no free buffer instead of rendering
    // frames we'd discard.
    let now = Instant::now();
    if ud.is_throttled(now) {
        if !ud.deferred {
            trace!("throttle: deferring buffer(s) to the period timer");
        }
        ud.deferred = true;
        return;
    }
    ud.deferred = false;

    // Drain stale buffers and keep only the newest. It goes back to
    // the compositor when `buffer` drops at the end of this callback,
    // unless a `GpuFrame` takes a lease on it below.
    let (newest, drained) = DequeuedBuffer::newest(stream);
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
        stride = neg.pixel_format.packed_row_bytes(neg.width);
    }

    // `RawFrame::stride` promises "bytes per row of `data`", so the
    // CPU paths below only emit a frame when the mapped plane really
    // holds `height` rows of `stride` bytes (the last row may omit
    // its trailing padding).
    let layout = PlaneLayout {
        stride,
        row_bytes: neg.pixel_format.packed_row_bytes(neg.width),
        height: neg.height,
    };
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
    //   The frame holds a lease that keeps the buffer dequeued (so
    //   the compositor can't render into it) until every clone of the
    //   frame is dropped; at the hold limit we read back to `Cpu`
    //   instead (see the arm below).
    // * DmaBuf with tiled modifier + no importer → drop with
    //   warning. Should be unreachable because we wouldn't have
    //   negotiated those modifiers in the first place.
    let captured: CapturedFrame = match plane_type {
        DataType::MemFd | DataType::MemPtr => {
            let Some(bytes) = read_cpu_buffer(plane, chunk_offset, layout) else {
                return;
            };
            CapturedFrame::Cpu(RawFrame {
                hdr: None,
                colour: None,
                width: neg.width,
                height: neg.height,
                stride,
                format: neg.pixel_format,
                data: bytes,
                timestamp_us,
            })
        }
        DataType::DmaBuf if neg.modifier_is_cpu_readable() => {
            let Some(bytes) = read_dmabuf_mmap(plane, chunk_offset, layout) else {
                return;
            };
            CapturedFrame::Cpu(RawFrame {
                hdr: None,
                colour: None,
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
            let (raw_fd, raw_maxsize) = (raw.fd, raw.maxsize);
            let Ok(raw_fd) = RawFd::try_from(raw_fd) else {
                warn!(fd = raw_fd, "DmaBuf plane has out-of-range fd");
                return;
            };
            if raw_fd < 0 {
                warn!("DmaBuf plane has invalid fd");
                return;
            }
            // PipeWire closes this fd on renegotiation / stop, possibly
            // while the frame is still alive downstream. The frame
            // therefore owns its own duplicate so a consumer on
            // another thread never imports a closed or recycled
            // descriptor (= a different object). Content stability is
            // the lease's job, below.
            // SAFETY: `raw_fd` is the live descriptor PipeWire owns
            // for this dequeued buffer; it stays open for the whole
            // `process` callback, and we only borrow it long enough to
            // `dup` it.
            let borrowed = unsafe { BorrowedFd::borrow_raw(raw_fd) };
            let fd = match DmaBufFd::dup(borrowed) {
                Ok(fd) => fd,
                Err(e) => {
                    warn!(error = %e, "dup(DmaBuf fd) failed — frame dropped");
                    return;
                }
            };
            // DMA-BUF object size: use maxsize directly (the total
            // buffer including tiling padding). Do NOT subtract
            // chunk_offset — VA-API's objects[0].size expects the
            // full fd size, and the plane offset is stored separately
            // in DmaBufPlane.offset.
            let dmabuf_size = if raw_maxsize > 0 {
                raw_maxsize
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
            let frame = GpuFrame {
                width: neg.width,
                height: neg.height,
                stride,
                format: neg.pixel_format,
                timestamp_us,
                plane: DmaBufPlane {
                    fd,
                    offset: chunk_offset,
                    stride,
                    modifier,
                    size: dmabuf_size,
                },
                importer: Some(importer.clone() as Arc<dyn ferricast_core::DmaBufImporter>),
                lease: None,
            };
            match leases.hold(buffer) {
                Ok(lease) => CapturedFrame::Gpu(GpuFrame {
                    lease: Some(lease),
                    ..frame
                }),
                // Downstream already holds all buffers but the one the
                // compositor needs to keep rendering. Dropping this
                // frame instead could stall capture for good: the
                // segmenter keeps its last frame for re-encoding and
                // only lets go of it when a newer one arrives. A
                // synchronous readback costs ~10-15 ms but emits a
                // lease-free `Cpu` frame, which frees that slot.
                Err(buffer) => {
                    if ud.lease_limit_logged {
                        trace!("DmaBuf hold limit reached, reading frame back to CPU");
                    } else {
                        info!(
                            limit = leases.limit(),
                            held = leases.held(),
                            "DmaBuf hold limit reached, reading frame back to CPU \
                             (consumer slower than the compositor's buffer pool)"
                        );
                        ud.lease_limit_logged = true;
                    }
                    let cpu = CapturedFrame::Gpu(frame).into_cpu();
                    // The readback has completed; the compositor may
                    // reuse the buffer now.
                    drop(buffer);
                    match cpu {
                        Ok(raw) => CapturedFrame::Cpu(raw),
                        Err(e) => {
                            warn!(error = %e, "DmaBuf readback at hold limit failed — frame dropped");
                            return;
                        }
                    }
                }
            }
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
                CapturedFrame::Native(_) => "native",
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

/// Row geometry of a CPU-readable plane, used to cut exactly one
/// frame out of a mapped buffer.
#[derive(Debug, Clone, Copy)]
struct PlaneLayout {
    /// Bytes between the starts of consecutive rows.
    stride: u32,
    /// Bytes of pixel data in one row (`<= stride` when valid).
    row_bytes: u32,
    height: u32,
}

impl PlaneLayout {
    /// Byte range of the frame inside a mapping of `len` bytes, with
    /// the frame starting at `offset`. Covers `height` full strides
    /// when they fit, otherwise accepts a last row trimmed to
    /// `row_bytes`. `None` when the stride can't hold a row or the
    /// mapping is too short — emitting such a frame would break the
    /// `RawFrame::stride` contract for every consumer.
    fn frame_range(self, len: usize, offset: u32) -> Option<std::ops::Range<usize>> {
        let (stride, row_bytes) = (self.stride as usize, self.row_bytes as usize);
        if self.height == 0 || row_bytes == 0 || stride < row_bytes {
            return None;
        }
        let start = offset as usize;
        let full = stride.checked_mul(self.height as usize)?;
        let min = full.checked_sub(stride)?.checked_add(row_bytes)?;
        let available = len.checked_sub(start)?;
        if available < min {
            return None;
        }
        Some(start..start + full.min(available))
    }
}

fn read_cpu_buffer(
    plane: &mut pw::spa::buffer::Data,
    chunk_offset: u32,
    layout: PlaneLayout,
) -> Option<Bytes> {
    let slice = plane.data()?;
    let Some(range) = layout.frame_range(slice.len(), chunk_offset) else {
        warn!(
            ?layout,
            len = slice.len(),
            chunk_offset,
            "SHM plane too small for frame"
        );
        return None;
    };
    slice.get(range).map(Bytes::copy_from_slice)
}

/// Linear / INVALID-modifier DmaBuf: we can read the bytes directly
/// via `mmap`. Used for the trivial case (Mutter falling back to
/// LINEAR even though it negotiated DmaBuf).
fn read_dmabuf_mmap(
    plane: &mut pw::spa::buffer::Data,
    chunk_offset: u32,
    layout: PlaneLayout,
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
        match layout.frame_range(slice.len(), chunk_offset) {
            Some(range) => slice.get(range).map(Bytes::copy_from_slice),
            None => {
                warn!(
                    ?layout,
                    maxsize, chunk_offset, "DmaBuf plane too small for frame"
                );
                None
            }
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

#[cfg(test)]
mod tests {
    use super::{PlaneLayout, StreamState, fatal_state_transition, is_within_period};
    use std::time::{Duration, Instant};

    #[test]
    fn throttle_window() {
        let period = Duration::from_millis(30);
        let t0 = Instant::now();
        assert!(!is_within_period(None, t0, period));
        assert!(is_within_period(
            Some(t0),
            t0 + Duration::from_millis(29),
            period
        ));
        assert!(!is_within_period(Some(t0), t0 + period, period));
        // A clock that appears to go backwards must not panic.
        assert!(is_within_period(Some(t0 + period), t0, period));
    }

    #[test]
    fn unconnected_after_connect_and_errors_are_fatal() {
        use StreamState::{Connecting, Paused, Streaming, Unconnected};
        assert!(fatal_state_transition(&Streaming, &Unconnected).is_some());
        assert!(fatal_state_transition(&Paused, &Unconnected).is_some());
        assert!(fatal_state_transition(&Connecting, &Unconnected).is_some());
        assert_eq!(
            fatal_state_transition(&Streaming, &StreamState::Error("boom".into())).as_deref(),
            Some("boom")
        );
        assert!(fatal_state_transition(&Unconnected, &Connecting).is_none());
        assert!(fatal_state_transition(&Streaming, &Paused).is_none());
        assert!(fatal_state_transition(&Paused, &Streaming).is_none());
    }

    const LAYOUT: PlaneLayout = PlaneLayout {
        stride: 64,
        row_bytes: 40,
        height: 3,
    };

    #[test]
    fn frame_range_takes_full_strides_when_available() {
        assert_eq!(LAYOUT.frame_range(1024, 16), Some(16..16 + 192));
    }

    #[test]
    fn frame_range_accepts_trimmed_last_row() {
        // 2 full strides + 40 bytes of the last row.
        assert_eq!(LAYOUT.frame_range(168, 0), Some(0..168));
    }

    #[test]
    fn frame_range_rejects_short_or_inconsistent_planes() {
        assert_eq!(LAYOUT.frame_range(167, 0), None);
        assert_eq!(LAYOUT.frame_range(100, 200), None);
        let narrow = PlaneLayout {
            stride: 32,
            ..LAYOUT
        };
        assert_eq!(narrow.frame_range(1024, 0), None);
        let empty = PlaneLayout {
            height: 0,
            ..LAYOUT
        };
        assert_eq!(empty.frame_range(1024, 0), None);
    }
}
