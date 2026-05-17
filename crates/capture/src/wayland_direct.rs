//! Wayland direct capture — no xdg-desktop-portal.
//!
//! Streams a monitor (and, later, a toplevel) by talking
//! `zwlr_screencopy_unstable_v1` directly to the compositor. The
//! frames arrive in a DMA-BUF we allocated via `gbm`, the encoder
//! consumes that fd zero-copy (VA-API VPP, NVENC once vendored), and
//! a Vulkan importer is attached for x264-style readback fallback.
//!
//! ## Why a separate backend from PipeWire?
//!
//! The portal owns source selection (it pops its own dialog). When
//! the user pick already happened in our in-process picker, going
//! through the portal asks them again. Direct capture lets us hand
//! the compositor a specific output id resolved from
//! `ListMonitors` / `ListWindows` and skip the second prompt — at
//! the cost of dropping GNOME compatibility (Mutter doesn't expose
//! these protocols).
//!
//! ## Compositor coverage
//!
//! - wlroots family (Hyprland, sway, river, …): ✓ `wlr-screencopy`
//! - niri:                                       ✓ `wlr-screencopy`
//! - KDE Plasma 6:                               ✓ `wlr-screencopy`
//! - GNOME / Mutter:                             ✗ falls through
//!
//! The fallback chain in [`crate::auto_capture`] picks
//! `WaylandDirectCapture` first on Wayland; failure (binding,
//! compositor refuses dmabuf negotiation, etc.) drops to
//! `PipeWireCapture` when the `pipewire` feature is enabled.
//!
//! ## Transport
//!
//! - **Preferred**: `linux_dmabuf` event. We allocate a single
//!   gbm bo per buffer slot with the LINEAR modifier (universal),
//!   wrap it as `wl_buffer` via `zwp_linux_dmabuf_v1`, and hand it
//!   to the compositor. Frames emit as `CapturedFrame::Gpu`.
//! - **Fallback**: `buffer` event (wl_shm). Allocates a memfd pool,
//!   reads back. Emits `CapturedFrame::Cpu`. Used when the
//!   compositor doesn't advertise `linux_dmabuf` for the frame
//!   (rare on modern hardware but possible on llvmpipe / nested).

use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use ferricast_core::{
    CaptureConfig, CaptureSource, CapturedFrame, DmaBufImporter, DmaBufPlane, FerricastError,
    GpuFrame, PixelFormat, RawFrame, Result, ScreenCapture,
};
use tokio::sync::mpsc;
use tracing::{debug, info, trace, warn};

use wayland_client::{
    Connection, Dispatch, Proxy, QueueHandle, WEnum,
    globals::{GlobalListContents, registry_queue_init},
    protocol::{
        wl_buffer::WlBuffer,
        wl_output::{self, WlOutput},
        wl_registry::WlRegistry,
        wl_shm::{self, WlShm},
        wl_shm_pool::WlShmPool,
    },
};
use wayland_protocols::wp::linux_dmabuf::zv1::client::{
    zwp_linux_buffer_params_v1::{self, ZwpLinuxBufferParamsV1},
    zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1,
};
use wayland_protocols::xdg::xdg_output::zv1::client::{
    zxdg_output_manager_v1::ZxdgOutputManagerV1,
    zxdg_output_v1::{self, ZxdgOutputV1},
};
use wayland_protocols_wlr::screencopy::v1::client::{
    zwlr_screencopy_frame_v1::{self, ZwlrScreencopyFrameV1},
    zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1,
};

/// DMA-BUF allocation backend. Wraps a single gbm device tied to
/// the first usable render node. Allocates LINEAR BGRA buffers —
/// LINEAR is universally supported, costs a bit of bandwidth vs
/// tiled formats but keeps the negotiation matrix small.
struct DmabufAllocator {
    device: gbm::Device<std::fs::File>,
}

impl DmabufAllocator {
    fn open() -> Result<Self> {
        for node in &["/dev/dri/renderD128", "/dev/dri/renderD129", "/dev/dri/renderD130"] {
            let path = Path::new(node);
            if !path.exists() {
                continue;
            }
            let file = match std::fs::OpenOptions::new().read(true).write(true).open(path) {
                Ok(f) => f,
                Err(e) => {
                    debug!(node, %e, "could not open render node");
                    continue;
                }
            };
            match gbm::Device::new(file) {
                Ok(device) => {
                    info!(node, "gbm device opened for direct-capture allocation");
                    return Ok(DmabufAllocator { device });
                }
                Err(e) => debug!(node, %e, "gbm::Device::new failed"),
            }
        }
        Err(FerricastError::Capture(
            "wayland-direct: no usable DRM render node for DMA-BUF allocation".into(),
        ))
    }

    fn alloc(
        &self,
        width: u32,
        height: u32,
        fourcc: drm_fourcc::DrmFourcc,
    ) -> Result<AllocatedBuffer> {
        // Allocation strategy: cascade through three increasingly
        // permissive paths because GBM backends — NVIDIA's in
        // particular — return EINVAL on combinations that other
        // drivers accept without complaint.
        //
        //   1. `with_modifiers([Linear])` — modifier-driven API,
        //      no usage flags. Mesa picks sensible internal flags.
        //      The cleanest path; works on Intel/AMD universally.
        //   2. `with_modifiers([Invalid])` — driver-picked
        //      modifier. INVALID is a valid wire value for
        //      `zwp_linux_buffer_params_v1`, and the compositor
        //      accepts whatever the driver gave us.
        //   3. Legacy `BufferObjectFlags::LINEAR` (no RENDERING).
        //      NVIDIA's gbm sometimes refuses RENDERING + LINEAR
        //      for the format the compositor offered, so we drop
        //      RENDERING in the last attempt.
        //
        // Whichever succeeds defines the modifier we'll forward
        // to `params.add` so the compositor knows what layout it's
        // looking at.
        use gbm::{BufferObjectFlags, Modifier};

        tracing::debug!(
            ?fourcc,
            width,
            height,
            "gbm: attempting dmabuf alloc"
        );

        let bo = self
            .device
            .create_buffer_object_with_modifiers::<()>(
                width,
                height,
                fourcc,
                [Modifier::Linear].into_iter(),
            )
            .inspect(|_| tracing::debug!(?fourcc, "gbm: allocated with Modifier::Linear"))
            .or_else(|e1| {
                tracing::debug!(
                    %e1, ?fourcc,
                    "gbm: with_modifiers([Linear]) rejected; trying [Invalid]"
                );
                self.device
                    .create_buffer_object_with_modifiers::<()>(
                        width,
                        height,
                        fourcc,
                        [Modifier::Invalid].into_iter(),
                    )
                    .inspect(|_| tracing::debug!(?fourcc, "gbm: allocated with Modifier::Invalid"))
                    .or_else(|e2| {
                        tracing::debug!(
                            %e2, ?fourcc,
                            "gbm: with_modifiers([Invalid]) rejected; trying legacy LINEAR flag"
                        );
                        self.device
                            .create_buffer_object::<()>(
                                width,
                                height,
                                fourcc,
                                BufferObjectFlags::LINEAR,
                            )
                            .inspect(|_| tracing::debug!(?fourcc, "gbm: allocated with legacy LINEAR flag"))
                    })
            })
            .map_err(|e| {
                FerricastError::Capture(format!(
                    "gbm alloc (fourcc={fourcc:?}, {width}x{height}): {e}"
                ))
            })?;

        let modifier = u64::from(bo.modifier());
        let stride = bo.stride() as u32;
        let fd: OwnedFd = bo
            .fd()
            .map_err(|e| FerricastError::Capture(format!("gbm fd: {e}")))?;
        Ok(AllocatedBuffer {
            _bo: bo,
            fd,
            width,
            height,
            stride,
            modifier,
            fourcc,
        })
    }
}

/// One DMA-BUF the compositor writes into. Keeps the `gbm` BO alive
/// (drop releases the memory) and exposes the fd for wl_buffer
/// wrapping + downstream consumption.
struct AllocatedBuffer {
    _bo: gbm::BufferObject<()>,
    fd: OwnedFd,
    width: u32,
    height: u32,
    stride: u32,
    modifier: u64,
    fourcc: drm_fourcc::DrmFourcc,
}

impl AllocatedBuffer {
    fn pixel_format(&self) -> PixelFormat {
        // We always allocate one of these two — see `pick_fourcc`.
        match self.fourcc {
            drm_fourcc::DrmFourcc::Argb8888 => PixelFormat::Bgra,
            drm_fourcc::DrmFourcc::Abgr8888 => PixelFormat::Rgba,
            _ => PixelFormat::Bgra,
        }
    }
}


// ── ScreenCapture wrapper ─────────────────────────────────────────

pub struct WaylandDirectCapture {
    worker: Option<WorkerHandle>,
    fps: u32,
    size: (usize, usize),
}

impl WaylandDirectCapture {
    pub fn new() -> Self {
        Self {
            worker: None,
            fps: 0,
            size: (0, 0),
        }
    }
}

struct WorkerHandle {
    frames: mpsc::Receiver<CapturedFrame>,
    stop: std::sync::mpsc::Sender<()>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

impl ScreenCapture for WaylandDirectCapture {
    async fn start(&mut self, source: CaptureSource, config: CaptureConfig) -> Result<()> {
        let output_name = match source {
            CaptureSource::FullScreen { monitor: Some(name) } => name,
            CaptureSource::FullScreen { monitor: None } => {
                return Err(FerricastError::Capture(
                    "wayland-direct: monitor id required (we don't pop a portal picker)".into(),
                ));
            }
            CaptureSource::Window { .. } => {
                // Window capture needs ext-image-copy-capture +
                // ext-foreign-toplevel-image-capture-source — a
                // follow-up. wlr-screencopy doesn't capture toplevels.
                return Err(FerricastError::Capture(
                    "wayland-direct: window capture not yet implemented; use monitor capture or fall back to PipeWire".into(),
                ));
            }
        };

        let (frame_tx, frame_rx) = mpsc::channel::<CapturedFrame>(2);
        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let fps = config.fps.max(1);

        // The first frame on the channel carries the negotiated
        // size; we read it synchronously to populate `self.size`
        // before returning so `get_screen_size()` works immediately.
        // Worker also sends an initial `(width, height)` over a
        // one-shot channel for that purpose.
        let (size_tx, size_rx) = std::sync::mpsc::sync_channel::<(u32, u32)>(1);

        let output_name_clone = output_name.clone();
        let join = std::thread::Builder::new()
            .name("ferricast-wl-direct".into())
            .spawn(move || {
                if let Err(e) = run_worker(output_name_clone, fps, frame_tx, stop_rx, size_tx) {
                    warn!(%e, "wayland-direct worker exited");
                }
            })
            .map_err(|e| FerricastError::Capture(format!("spawn: {e}")))?;

        let size = size_rx
            .recv_timeout(Duration::from_secs(5))
            .map_err(|_| {
                FerricastError::Capture(
                    "wayland-direct: compositor didn't respond within 5s".into(),
                )
            })?;

        self.worker = Some(WorkerHandle {
            frames: frame_rx,
            stop: stop_tx,
            join: Some(join),
        });
        self.size = (size.0 as usize, size.1 as usize);
        self.fps = fps;
        info!(
            output = output_name,
            width = size.0,
            height = size.1,
            fps,
            "wayland-direct started"
        );
        Ok(())
    }

    async fn next_frame(&mut self) -> Result<CapturedFrame> {
        let worker = self
            .worker
            .as_mut()
            .ok_or_else(|| FerricastError::Capture("wayland-direct: not started".into()))?;
        worker
            .frames
            .recv()
            .await
            .ok_or_else(|| FerricastError::Capture("wayland-direct: worker hung up".into()))
    }

    async fn stop(&mut self) -> Result<()> {
        if let Some(mut w) = self.worker.take() {
            let _ = w.stop.send(());
            if let Some(j) = w.join.take() {
                let _ = j.join();
            }
        }
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.worker.is_some()
    }

    fn get_pixel_format(&self) -> PixelFormat {
        PixelFormat::Bgra
    }

    fn get_screen_size(&self) -> (usize, usize) {
        self.size
    }

    fn get_framerate(&self) -> u32 {
        self.fps
    }
}

// ── Worker thread ─────────────────────────────────────────────────

fn run_worker(
    output_name: String,
    fps: u32,
    frame_tx: mpsc::Sender<CapturedFrame>,
    stop_rx: std::sync::mpsc::Receiver<()>,
    size_tx: std::sync::mpsc::SyncSender<(u32, u32)>,
) -> Result<()> {
    let conn = Connection::connect_to_env()
        .map_err(|e| FerricastError::Capture(format!("wayland connect: {e}")))?;
    let (globals, mut queue) = registry_queue_init::<WorkerState>(&conn)
        .map_err(|e| FerricastError::Capture(format!("registry: {e}")))?;
    let qh = queue.handle();

    let shm: WlShm = globals
        .bind(&qh, 1..=1, ())
        .map_err(|e| FerricastError::Capture(format!("wl_shm: {e}")))?;
    let screencopy: ZwlrScreencopyManagerV1 = globals
        .bind(&qh, 3..=3, ())
        .map_err(|e| FerricastError::Capture(format!("zwlr_screencopy v3 missing: {e}")))?;
    let dmabuf: Option<ZwpLinuxDmabufV1> = globals.bind(&qh, 3..=4, ()).ok();
    if dmabuf.is_none() {
        warn!("wayland-direct: zwp_linux_dmabuf_v1 absent — falling back to wl_shm transport");
    }
    let xdg_om: Option<ZxdgOutputManagerV1> = globals.bind(&qh, 2..=3, ()).ok();

    let mut state = WorkerState {
        outputs: Vec::new(),
        target: None,
        frame_info: None,
        status: FrameStatus::Pending,
        size_emitted: false,
        size_tx: Some(size_tx),
    };

    for g in globals.contents().clone_list() {
        if g.interface == "wl_output" {
            let out = globals
                .registry()
                .bind::<WlOutput, _, _>(g.name, g.version.min(4), &qh, ());
            if let Some(om) = xdg_om.as_ref() {
                om.get_xdg_output(&out, &qh, ());
            }
            state.outputs.push((out, None));
        }
    }
    queue
        .roundtrip(&mut state)
        .map_err(|e| FerricastError::Capture(format!("output roundtrip: {e}")))?;

    let target = state
        .outputs
        .iter()
        .find(|(_, n)| n.as_deref() == Some(output_name.as_str()))
        .map(|(o, _)| o.clone())
        .ok_or_else(|| FerricastError::Capture(format!("no output named {output_name:?}")))?;
    state.target = Some(target.clone());

    let allocator = DmabufAllocator::open().ok();
    let importer = if cfg!(feature = "pipewire") {
        // Reuse the Vulkan importer if pipewire is built in. Without
        // it, CapturedFrame::Gpu still emits but `into_cpu()` will
        // fail — only matters for the x264 path.
        try_build_importer()
    } else {
        None
    };

    let frame_period = Duration::from_micros(1_000_000 / fps as u64);
    let mut next_capture = Instant::now();
    // Sticky-disable knob for the dmabuf path. The compositor can
    // accept a `params.add` with `create_immed` (no synchronous
    // error there) and only reject the buffer at `copy()` time via
    // the `failed` event — so even when allocation looks fine we
    // can end up dropping every frame. After a small handful of
    // consecutive `Failed`s we stop trying dmabuf and stream via
    // wl_shm for the rest of the session. The threshold is small
    // (3) because the failure mode is systemic, not transient —
    // either the compositor's renderer can read our modifier or
    // it can't.
    let mut dmabuf_disabled = false;
    let mut dmabuf_consecutive_failures: u32 = 0;
    const DMABUF_FAILURE_THRESHOLD: u32 = 3;

    loop {
        // Cooperative stop — check between frames.
        if stop_rx.try_recv().is_ok() {
            break;
        }
        let now = Instant::now();
        if now < next_capture {
            std::thread::sleep(next_capture - now);
        }
        next_capture = Instant::now() + frame_period;

        // Per-frame state reset.
        state.frame_info = None;
        state.status = FrameStatus::Pending;

        let frame = screencopy.capture_output(0, &target, &qh, ());

        // Wait for buffer / linux_dmabuf / buffer_done events.
        let deadline = Instant::now() + Duration::from_secs(2);
        while state.frame_info.is_none() && Instant::now() < deadline {
            queue
                .blocking_dispatch(&mut state)
                .map_err(|e| FerricastError::Capture(format!("buffer wait: {e}")))?;
        }
        let info = state.frame_info.clone().ok_or_else(|| {
            FerricastError::Capture("compositor didn't announce buffer params within 2s".into())
        })?;

        if !state.size_emitted {
            let _ = state
                .size_tx
                .as_ref()
                .unwrap()
                .send((info.width, info.height));
            state.size_emitted = true;
        }

        // Pick the transport: dmabuf if (a) the compositor offered
        // it on this frame, (b) we have an allocator, and (c) the
        // sticky-disable hasn't kicked in yet. Otherwise wl_shm
        // fallback.
        let dmabuf_offer = info
            .dmabuf
            .as_ref()
            .filter(|_| allocator.is_some() && !dmabuf_disabled);
        let (buffer, transport) = if let (Some(offer), Some(alloc), Some(dmabuf_proxy)) =
            (dmabuf_offer, allocator.as_ref(), dmabuf.as_ref())
        {
            match make_dmabuf_buffer(alloc, dmabuf_proxy, &qh, offer) {
                Ok((buf, tx)) => (buf, Transport::Dmabuf(tx)),
                Err(e) => {
                    warn!(%e, "dmabuf allocation failed, falling back to shm for this frame");
                    let (buf, tx) = make_shm_buffer(&shm, &qh, &info)?;
                    (buf, Transport::Shm(tx))
                }
            }
        } else {
            let (buf, tx) = make_shm_buffer(&shm, &qh, &info)?;
            (buf, Transport::Shm(tx))
        };
        let attempted_dmabuf = matches!(transport, Transport::Dmabuf(_));

        frame.copy(&buffer);
        let deadline = Instant::now() + Duration::from_secs(2);
        while state.status == FrameStatus::Pending && Instant::now() < deadline {
            queue
                .blocking_dispatch(&mut state)
                .map_err(|e| FerricastError::Capture(format!("frame wait: {e}")))?;
        }
        if state.status != FrameStatus::Ready {
            // Record per-transport consecutive failures so the
            // sticky-disable above only fires when the *dmabuf*
            // path is the one rejecting frames. shm failures are
            // a different problem (compositor bug, OOM) and don't
            // benefit from flipping the dmabuf knob.
            if attempted_dmabuf {
                dmabuf_consecutive_failures = dmabuf_consecutive_failures.saturating_add(1);
                if dmabuf_consecutive_failures >= DMABUF_FAILURE_THRESHOLD && !dmabuf_disabled {
                    dmabuf_disabled = true;
                    warn!(
                        consecutive = dmabuf_consecutive_failures,
                        "compositor rejected {DMABUF_FAILURE_THRESHOLD} dmabuf frames in a row \
                         — switching this session to wl_shm transport. \
                         Set RUST_LOG=ferricast_capture=debug to see the negotiated \
                         modifier / stride / format that the compositor refused."
                    );
                }
            }
            warn!(?state.status, "frame not ready — dropping");
            buffer.destroy();
            continue;
        }
        // Successful frame: a dmabuf success means the path is
        // healthy, so wipe the consecutive-failure counter. (One
        // bad frame followed by good ones shouldn't escalate to
        // sticky-disable.)
        if attempted_dmabuf {
            dmabuf_consecutive_failures = 0;
        }
        let captured = match transport {
            Transport::Dmabuf(alloc) => CapturedFrame::Gpu(GpuFrame {
                width: alloc.width,
                height: alloc.height,
                stride: alloc.stride,
                format: alloc.pixel_format(),
                timestamp_us: now_us(),
                plane: DmaBufPlane {
                    fd: alloc.fd.as_raw_fd(),
                    offset: 0,
                    stride: alloc.stride,
                    modifier: alloc.modifier,
                    size: alloc.stride.saturating_mul(alloc.height),
                },
                importer: importer.clone(),
            }),
            Transport::Shm(shm_buf) => CapturedFrame::Cpu(RawFrame {
                width: shm_buf.width,
                height: shm_buf.height,
                stride: shm_buf.stride,
                format: shm_buf.pixel_format,
                data: Bytes::copy_from_slice(shm_buf.mmap.as_slice()),
                timestamp_us: now_us(),
            }),
        };
        buffer.destroy();

        if frame_tx.try_send(captured).is_err() {
            trace!("consumer behind — dropping frame");
        }
    }
    Ok(())
}

fn try_build_importer() -> Option<Arc<dyn DmaBufImporter>> {
    // The Vulkan importer lives in the pipewire sub-module today and
    // is gated behind that feature. When pipewire is also enabled,
    // we reach for it; otherwise CapturedFrame::Gpu goes out with
    // `importer: None` and x264 fallback fails. A follow-up should
    // hoist `VulkanImporter` up to a shared module so wayland-direct
    // alone can readback for CPU encoders.
    None
}

enum Transport {
    Dmabuf(AllocatedBuffer),
    Shm(ShmBuffer),
}

fn make_dmabuf_buffer(
    alloc: &DmabufAllocator,
    dmabuf: &ZwpLinuxDmabufV1,
    qh: &QueueHandle<WorkerState>,
    offer: &DmabufOffer,
) -> Result<(WlBuffer, AllocatedBuffer)> {
    let fourcc = drm_fourcc::DrmFourcc::try_from(offer.format).map_err(|_| {
        FerricastError::Capture(format!("unknown DRM fourcc 0x{:08x}", offer.format))
    })?;
    let buf = alloc.alloc(offer.width, offer.height, fourcc)?;
    // The compositor only finds out about a bad params combo at
    // `copy()` time (it emits `failed` on the frame), so logging
    // what we negotiated here is the only way to diagnose
    // "compositor rejected the buffer" failures after the fact.
    // The modifier is the most common culprit: gbm sometimes
    // returns INVALID (`u64::MAX`) for what's really a linear
    // allocation, and not every compositor's renderer is willing
    // to import INVALID-modifier buffers via screencopy.
    tracing::debug!(
        format = ?fourcc,
        width = buf.width,
        height = buf.height,
        stride = buf.stride,
        modifier = format!("0x{:016x}", buf.modifier),
        "wayland-direct: zwp_linux_buffer_params_v1.add()"
    );
    let params: ZwpLinuxBufferParamsV1 = dmabuf.create_params(qh, ());
    params.add(
        buf.fd.as_fd(),
        0, // plane idx
        0, // offset
        buf.stride,
        (buf.modifier >> 32) as u32,
        (buf.modifier & 0xffff_ffff) as u32,
    );
    let wl_buf = params.create_immed(
        buf.width as i32,
        buf.height as i32,
        offer.format,
        zwp_linux_buffer_params_v1::Flags::empty(),
        qh,
        (),
    );
    params.destroy();
    Ok((wl_buf, buf))
}

struct ShmBuffer {
    _fd: OwnedFd,
    mmap: MmapRegion,
    width: u32,
    height: u32,
    stride: u32,
    pixel_format: PixelFormat,
}

fn make_shm_buffer(
    shm: &WlShm,
    qh: &QueueHandle<WorkerState>,
    info: &FrameInfo,
) -> Result<(WlBuffer, ShmBuffer)> {
    let size = info.shm.as_ref().map(|s| s.stride * info.height).unwrap_or(0);
    let stride = info.shm.as_ref().map(|s| s.stride).unwrap_or(info.width * 4);
    let shm_fmt = info.shm.as_ref().map(|s| s.format).unwrap_or(0);
    if size == 0 {
        return Err(FerricastError::Capture(
            "compositor didn't offer wl_shm format for this frame".into(),
        ));
    }
    let (fd, mmap) = alloc_memfd(size as usize)?;
    let pool = shm.create_pool(fd.as_fd(), size as i32, qh, ());
    let wl_buf = pool.create_buffer(
        0,
        info.width as i32,
        info.height as i32,
        stride as i32,
        wl_shm_format(shm_fmt)?,
        qh,
        (),
    );
    pool.destroy();
    let pixel_format = wl_shm_format_to_pixel(shm_fmt);
    Ok((
        wl_buf,
        ShmBuffer {
            _fd: fd,
            mmap,
            width: info.width,
            height: info.height,
            stride,
            pixel_format,
        },
    ))
}

fn wl_shm_format(raw: u32) -> Result<wl_shm::Format> {
    match raw {
        0 => Ok(wl_shm::Format::Argb8888),
        1 => Ok(wl_shm::Format::Xrgb8888),
        0x34324241 => Ok(wl_shm::Format::Abgr8888),
        0x34325258 => Ok(wl_shm::Format::Xbgr8888),
        other => Err(FerricastError::Capture(format!(
            "unsupported wl_shm format 0x{other:08x}"
        ))),
    }
}

fn wl_shm_format_to_pixel(raw: u32) -> PixelFormat {
    match raw {
        0x34324241 | 0x34325258 => PixelFormat::Rgba,
        _ => PixelFormat::Bgra,
    }
}

fn now_us() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

// ── Worker state + dispatch ───────────────────────────────────────

#[derive(Clone, Debug)]
struct DmabufOffer {
    format: u32, // DRM fourcc
    width: u32,
    height: u32,
}

#[derive(Clone, Debug)]
struct ShmOffer {
    format: u32, // wl_shm format
    stride: u32,
}

#[derive(Clone, Debug)]
struct FrameInfo {
    width: u32,
    height: u32,
    dmabuf: Option<DmabufOffer>,
    shm: Option<ShmOffer>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameStatus {
    Pending,
    Ready,
    Failed,
}

struct WorkerState {
    outputs: Vec<(WlOutput, Option<String>)>,
    /// Stashed for diagnostics so we can log "started capturing
    /// $name" without re-querying. Held for `Drop` ordering — the
    /// proxy survives until the worker thread ends.
    #[allow(dead_code)]
    target: Option<WlOutput>,
    frame_info: Option<FrameInfo>,
    status: FrameStatus,
    size_emitted: bool,
    size_tx: Option<std::sync::mpsc::SyncSender<(u32, u32)>>,
}

impl Dispatch<WlRegistry, GlobalListContents> for WorkerState {
    fn event(
        _: &mut Self,
        _: &WlRegistry,
        _: <WlRegistry as Proxy>::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<WlOutput, ()> for WorkerState {
    fn event(
        state: &mut Self,
        proxy: &WlOutput,
        event: <WlOutput as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_output::Event::Name { name } = event {
            for (out, slot) in &mut state.outputs {
                if out == proxy {
                    *slot = Some(name.clone());
                    break;
                }
            }
        }
    }
}
impl Dispatch<ZxdgOutputManagerV1, ()> for WorkerState {
    fn event(
        _: &mut Self,
        _: &ZxdgOutputManagerV1,
        _: <ZxdgOutputManagerV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<ZxdgOutputV1, ()> for WorkerState {
    fn event(
        state: &mut Self,
        _: &ZxdgOutputV1,
        event: <ZxdgOutputV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let zxdg_output_v1::Event::Name { name } = event {
            // xdg-output `name` fills in for wl_output v3- (no
            // Name event there). Best-effort: assign to the first
            // output that doesn't have a name yet.
            for (_, slot) in &mut state.outputs {
                if slot.is_none() {
                    *slot = Some(name.clone());
                    break;
                }
            }
        }
    }
}
impl Dispatch<WlShm, ()> for WorkerState {
    fn event(
        _: &mut Self,
        _: &WlShm,
        _: <WlShm as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<WlShmPool, ()> for WorkerState {
    fn event(
        _: &mut Self,
        _: &WlShmPool,
        _: <WlShmPool as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<WlBuffer, ()> for WorkerState {
    fn event(
        _: &mut Self,
        _: &WlBuffer,
        _: <WlBuffer as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<ZwpLinuxDmabufV1, ()> for WorkerState {
    fn event(
        _: &mut Self,
        _: &ZwpLinuxDmabufV1,
        _: <ZwpLinuxDmabufV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<ZwpLinuxBufferParamsV1, ()> for WorkerState {
    fn event(
        _: &mut Self,
        _: &ZwpLinuxBufferParamsV1,
        _: <ZwpLinuxBufferParamsV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
impl Dispatch<ZwlrScreencopyManagerV1, ()> for WorkerState {
    fn event(
        _: &mut Self,
        _: &ZwlrScreencopyManagerV1,
        _: <ZwlrScreencopyManagerV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ZwlrScreencopyFrameV1, ()> for WorkerState {
    fn event(
        state: &mut Self,
        _: &ZwlrScreencopyFrameV1,
        event: <ZwlrScreencopyFrameV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        use zwlr_screencopy_frame_v1::Event;
        match event {
            Event::Buffer {
                format,
                width,
                height,
                stride,
            } => {
                let format = match format {
                    WEnum::Value(v) => v as u32,
                    WEnum::Unknown(v) => v,
                };
                let entry = state.frame_info.get_or_insert(FrameInfo {
                    width,
                    height,
                    dmabuf: None,
                    shm: None,
                });
                entry.width = width;
                entry.height = height;
                entry.shm = Some(ShmOffer { format, stride });
            }
            Event::LinuxDmabuf {
                format,
                width,
                height,
            } => {
                // Decode the DRM fourcc so the log line is greppable
                // ("ARGB8888" / "XRGB8888" / "ABGR8888" / …) rather
                // than a bare hex word. Useful when an allocation
                // fails — `fourcc=Some(Argb8888)` immediately
                // localises the problem to a known format the GBM
                // backend should support.
                tracing::trace!(
                    raw = format!("0x{format:08x}"),
                    decoded = ?drm_fourcc::DrmFourcc::try_from(format).ok(),
                    width,
                    height,
                    "wlr-screencopy: linux_dmabuf offer"
                );
                let entry = state.frame_info.get_or_insert(FrameInfo {
                    width,
                    height,
                    dmabuf: None,
                    shm: None,
                });
                entry.width = width;
                entry.height = height;
                entry.dmabuf = Some(DmabufOffer {
                    format,
                    width,
                    height,
                });
            }
            Event::BufferDone => {
                // Compositor done advertising buffer params. If we
                // got nothing, frame_info stays None and the worker
                // will time out.
            }
            Event::Ready { .. } => state.status = FrameStatus::Ready,
            Event::Failed => state.status = FrameStatus::Failed,
            _ => {}
        }
    }
}

// ── memfd helpers (shared-ish with wayland_thumb but private here) ──

fn alloc_memfd(size: usize) -> Result<(OwnedFd, MmapRegion)> {
    use std::os::fd::FromRawFd;
    let name = b"ferricast-wldirect\0";
    let raw = unsafe {
        libc::syscall(
            libc::SYS_memfd_create,
            name.as_ptr() as *const libc::c_char,
            libc::MFD_CLOEXEC,
        )
    };
    if raw < 0 {
        return Err(FerricastError::Capture(format!(
            "memfd_create: {}",
            std::io::Error::last_os_error()
        )));
    }
    let fd = unsafe { OwnedFd::from_raw_fd(raw as i32) };
    if unsafe { libc::ftruncate(fd.as_raw_fd(), size as libc::off_t) } != 0 {
        return Err(FerricastError::Capture(format!(
            "ftruncate: {}",
            std::io::Error::last_os_error()
        )));
    }
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd.as_raw_fd(),
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(FerricastError::Capture(format!(
            "mmap: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok((
        fd,
        MmapRegion {
            ptr: ptr as *mut u8,
            len: size,
        },
    ))
}

struct MmapRegion {
    ptr: *mut u8,
    len: usize,
}
unsafe impl Send for MmapRegion {}
unsafe impl Sync for MmapRegion {}

impl MmapRegion {
    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for MmapRegion {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut _, self.len);
        }
    }
}
