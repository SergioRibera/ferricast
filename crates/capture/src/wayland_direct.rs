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

use std::collections::HashMap;
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
    zwp_linux_dmabuf_feedback_v1::{self, ZwpLinuxDmabufFeedbackV1},
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
    /// `(format_fourcc → allowed modifiers)` advertised by the
    /// compositor via the `linux-dmabuf-v1` feedback path. Empty
    /// when feedback wasn't available (older compositor, v3
    /// fallback): in that case we use whatever modifier the gbm
    /// cascade picks and hope.
    accepted: HashMap<u32, Vec<u64>>,
}

impl DmabufAllocator {
    /// Open the GBM device for the render node identified by the
    /// compositor's `linux_dmabuf_feedback_v1.main_device`. The
    /// compositor's renderer can only import dmabufs allocated on
    /// the same device — falling back to "first node we can open"
    /// breaks cross-GPU setups (NVIDIA dGPU + iGPU is the common
    /// failure case: niri is bound to one, we alloc on the other,
    /// every `copy()` rejected).
    ///
    /// When `main_device_dev_t` is `None` (no feedback, very old
    /// compositor) we fall back to the legacy "try every renderD*"
    /// path so the dmabuf attempt at least happens on single-GPU
    /// hosts.
    fn open(main_device_dev_t: Option<u64>) -> Result<Self> {
        let nodes: Vec<String> = render_node_paths();

        if let Some(target_dev_t) = main_device_dev_t {
            for node in &nodes {
                let path = Path::new(node);
                let Ok(meta) = std::fs::metadata(path) else {
                    continue;
                };
                use std::os::unix::fs::MetadataExt;
                if meta.rdev() != target_dev_t {
                    continue;
                }
                match Self::try_open(path, node) {
                    Some(d) => {
                        info!(
                            node,
                            main_device = format!("0x{target_dev_t:016x}"),
                            "gbm device opened for direct-capture allocation (matched feedback main_device)"
                        );
                        return Ok(DmabufAllocator {
                            device: d,
                            accepted: HashMap::new(),
                        });
                    }
                    None => continue,
                }
            }
            warn!(
                main_device = format!("0x{target_dev_t:016x}"),
                "wayland-direct: compositor's main_device doesn't match any /dev/dri/renderD* node we can open — falling back to first-available"
            );
        }

        for node in &nodes {
            if let Some(d) = Self::try_open(Path::new(node), node) {
                info!(node, "gbm device opened (fallback — no feedback main_device)");
                return Ok(DmabufAllocator {
                    device: d,
                    accepted: HashMap::new(),
                });
            }
        }
        Err(FerricastError::Capture(
            "wayland-direct: no usable DRM render node for DMA-BUF allocation".into(),
        ))
    }

    fn try_open(path: &Path, node_str: &str) -> Option<gbm::Device<std::fs::File>> {
        if !path.exists() {
            return None;
        }
        let file = match std::fs::OpenOptions::new().read(true).write(true).open(path) {
            Ok(f) => f,
            Err(e) => {
                debug!(node = node_str, %e, "could not open render node");
                return None;
            }
        };
        match gbm::Device::new(file) {
            Ok(d) => Some(d),
            Err(e) => {
                debug!(node = node_str, %e, "gbm::Device::new failed");
                None
            }
        }
    }

    fn alloc(
        &self,
        width: u32,
        height: u32,
        fourcc: drm_fourcc::DrmFourcc,
    ) -> Result<AllocatedBuffer> {
        // Allocation strategy now goes through the compositor's
        // advertised modifier set first (via linux-dmabuf-v1
        // feedback). This is the difference between "the buffer
        // allocates fine but `copy()` returns failed" and "we
        // never reach `failed` because the layout actually matches
        // what the compositor's renderer can read":
        //
        //   1. If feedback gave us a list for this fourcc, walk
        //      it in preference order. Within the list, LINEAR
        //      first (universal-readable for software fallbacks
        //      and screencopy paths that go through GLES blit),
        //      then everything else.
        //   2. If feedback didn't advertise this fourcc, walk a
        //      fixed cascade: LINEAR → INVALID → legacy LINEAR
        //      flag. INVALID stays in the cascade because old
        //      compositors that don't expose feedback at v4+ may
        //      still accept driver-picked layouts (single-GPU
        //      Intel hosts mostly).
        use gbm::{BufferObjectFlags, Modifier};

        tracing::debug!(
            ?fourcc,
            width,
            height,
            advertised_modifiers = self
                .accepted
                .get(&(fourcc as u32))
                .map(|v| v.len())
                .unwrap_or(0),
            "gbm: attempting dmabuf alloc"
        );

        // Pick the preferred modifier list. When feedback is empty
        // (older compositor / no v4 binding) we synthesise the
        // legacy cascade so the rest of the function shape stays
        // identical.
        let mut wanted: Vec<u64> = self
            .accepted
            .get(&(fourcc as u32))
            .cloned()
            .unwrap_or_default();
        if wanted.is_empty() {
            wanted.push(u64::from(Modifier::Linear));
            wanted.push(u64::from(Modifier::Invalid));
        } else {
            // LINEAR first if it's in the set — keeps the bytes
            // mmap'able for the shm-encode fallback path and is the
            // most widely-supported across compositor renderers.
            let linear = u64::from(Modifier::Linear);
            if let Some(pos) = wanted.iter().position(|&m| m == linear) {
                wanted.swap(0, pos);
            }
        }

        let mut last_err: Option<std::io::Error> = None;
        let mut bo_opt: Option<gbm::BufferObject<()>> = None;
        // Crucial knob: pass `RENDERING` so the compositor's GLES
        // renderer can bind the buffer as a render *target* via
        // EGL. The plain `create_buffer_object_with_modifiers`
        // (v1, no flags) variant lets mesa pick defaults — which
        // on many drivers means SCANOUT-ish flags that EGL refuses
        // to bind for rendering. niri's screencopy backend does
        // `renderer.bind(&mut dmabuf)` and that's where it fails
        // without RENDERING. We use the v2 API to force the right
        // flags; v2 needs libgbm ≥ 22.2 which is present on every
        // distro shipping a recent Wayland stack.
        let usage = BufferObjectFlags::RENDERING;
        // Important: this loop ONLY tries LINEAR (modifier == 0).
        // After-capture we hand the bytes to the encoder via
        // `mmap` + the kernel's dma-buf sync ioctl. That mmap
        // assumes row-major BGRA layout — which is what LINEAR
        // gives. Vendor-tiled modifiers like NVIDIA's
        // `BLOCK_LINEAR_2D` (0x0300_0000_0060_6011) write pixels
        // in a tile pattern; mmap'ing those returns bytes that
        // *look* like BGRA per row but are actually scrambled.
        // The encoder happily compresses the scramble and the
        // chromecast plays unrecognisable garbage — symptom the
        // user hit on NVIDIA when LINEAR+RENDERING got rejected
        // and the cascade fell through to the next advertised
        // modifier.
        //
        // For non-LINEAR modifiers we'd need a Vulkan blit step
        // (tiled→linear on the GPU) followed by readback. That's
        // what `VulkanImporter` from `pipewire/vulkan/` does;
        // hoisting it out of the pipewire submodule is the next
        // step to support GBM allocations that have to be tiled.
        // Until then: if LINEAR+RENDERING can't allocate, fail
        // the dmabuf path so sticky-disable promotes us to shm.
        let linear = u64::from(Modifier::Linear);
        for modifier_u64 in &wanted {
            if *modifier_u64 != linear {
                tracing::trace!(
                    modifier = format!("0x{modifier_u64:016x}"),
                    "gbm: skipping non-LINEAR modifier (mmap-readback requires row-major layout)"
                );
                continue;
            }
            let modifier: Modifier = (*modifier_u64).into();
            match self.device.create_buffer_object_with_modifiers2::<()>(
                width,
                height,
                fourcc,
                [modifier].into_iter(),
                usage,
            ) {
                Ok(bo) => {
                    tracing::debug!(
                        ?fourcc,
                        modifier = format!("0x{modifier_u64:016x}"),
                        flags = format!("{usage:?}"),
                        "gbm: allocated with advertised modifier + RENDERING"
                    );
                    bo_opt = Some(bo);
                    break;
                }
                Err(e) => {
                    tracing::debug!(
                        modifier = format!("0x{modifier_u64:016x}"),
                        %e,
                        "gbm: rejected this modifier; trying next"
                    );
                    last_err = Some(e);
                }
            }
        }
        // Final legacy-flag fallback for the no-feedback case —
        // some drivers (NVIDIA's GBM in particular) only do LINEAR
        // via the flag API, not the modifier API. Same RENDERING
        // requirement, just expressed as flag bits.
        if bo_opt.is_none() {
            match self.device.create_buffer_object::<()>(
                width,
                height,
                fourcc,
                BufferObjectFlags::LINEAR | BufferObjectFlags::RENDERING,
            ) {
                Ok(bo) => {
                    tracing::debug!(?fourcc, "gbm: allocated with legacy LINEAR+RENDERING flags");
                    bo_opt = Some(bo);
                }
                Err(e) => {
                    last_err = Some(e);
                }
            }
        }
        let bo = bo_opt.ok_or_else(|| {
            let cause = last_err
                .map(|e| e.to_string())
                .unwrap_or_else(|| "no modifier worked".into());
            FerricastError::Capture(format!(
                "gbm alloc (fourcc={fourcc:?}, {width}x{height}): {cause}"
            ))
        })?;

        let modifier = u64::from(bo.modifier());
        let stride = bo.stride() as u32;
        let fd: OwnedFd = bo
            .fd()
            .map_err(|e| FerricastError::Capture(format!("gbm fd: {e}")))?;
        Ok(AllocatedBuffer {
            bo,
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
    /// Held to keep the buffer alive while the wl_buffer + the
    /// downstream `CapturedFrame` reference it. Also used as the
    /// handle for `gbm_bo_map` when we have to read the bytes
    /// back for a CPU encoder.
    bo: gbm::BufferObject<()>,
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

    // Probe `linux-dmabuf-v1` feedback for the compositor's
    // preferred device + the (format, modifier) pairs it actually
    // accepts on the screencopy import path. Without this the
    // first frame's `copy()` request hits `failed` whenever the
    // compositor's renderer is on a different GPU than the render
    // node we'd open by default (NVIDIA dGPU + Intel/AMD iGPU
    // setups are the common failure case), or when the modifier
    // gbm picked isn't in the compositor's accept-list. Feedback
    // is best-effort: a v3 compositor doesn't have it and we fall
    // back to "open first available node, allocate LINEAR, hope".
    let feedback_info = dmabuf
        .as_ref()
        .and_then(|d| query_feedback(&conn, d).ok());
    if let Some(info) = feedback_info.as_ref() {
        info!(
            main_device = format!("0x{:016x}", info.main_device.unwrap_or(0)),
            formats = info.accepted.len(),
            "linux-dmabuf feedback collected"
        );
    } else {
        warn!(
            "linux-dmabuf feedback unavailable — dmabuf path will allocate \
             from the first available render node with LINEAR modifier and \
             may be rejected by the compositor (cross-GPU systems usually \
             fall back to wl_shm after a few rejected frames)"
        );
    }
    let allocator = DmabufAllocator::open(feedback_info.as_ref().and_then(|i| i.main_device))
        .ok()
        .map(|mut a| {
            if let Some(info) = feedback_info {
                a.accepted = info.accepted;
            }
            a
        });
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
                    // Alloc-time failure (gbm rejected our format /
                    // modifier / RENDERING combo). Increment the
                    // sticky-disable counter from this side too —
                    // otherwise the per-frame "trying dmabuf,
                    // falling back to shm" cycle just keeps
                    // spamming logs while doing wasted work. Once
                    // we've failed N allocs in a row, flip
                    // dmabuf_disabled and stop trying.
                    dmabuf_consecutive_failures =
                        dmabuf_consecutive_failures.saturating_add(1);
                    if dmabuf_consecutive_failures >= DMABUF_FAILURE_THRESHOLD
                        && !dmabuf_disabled
                    {
                        dmabuf_disabled = true;
                        warn!(
                            %e,
                            consecutive = dmabuf_consecutive_failures,
                            "gbm refused {DMABUF_FAILURE_THRESHOLD} dmabuf allocs in a row \
                             (typical on NVIDIA where LINEAR+RENDERING is rejected and the \
                             driver only offers vendor-tiled modifiers we can't mmap) — \
                             switching this session to wl_shm transport."
                        );
                    } else {
                        warn!(%e, "dmabuf allocation failed, falling back to shm for this frame");
                    }
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
            Transport::Dmabuf(alloc) => {
                // Two cases for what we hand the encoder:
                //
                //   1. Vulkan importer present → emit `Gpu` with
                //      the fd attached. GPU encoders (VA-API VPP,
                //      future NVENC zero-copy) consume the fd
                //      directly; CPU encoders trigger
                //      `into_cpu()` which routes through the
                //      importer's Vulkan readback.
                //   2. No importer → `CapturedFrame::Gpu` with
                //      `importer: None` would just blow up at the
                //      encoder's `into_cpu()` call (the user's
                //      most recent bug: "GpuFrame has no DmaBuf
                //      importer attached"). So we mmap the bo via
                //      gbm here and emit `Cpu` straight away.
                //      `gbm_bo_map(READ)` is allowed on LINEAR
                //      bos including ones flagged RENDERING; the
                //      driver may return a staging copy on
                //      non-coherent hosts, but the bytes are the
                //      same. We pay one CPU memcpy that the
                //      encoder would have paid anyway via Vulkan
                //      readback — net the same end-to-end cost,
                //      and it gets the path working today on
                //      hosts where we haven't hoisted the Vulkan
                //      importer out of the pipewire submodule yet.
                if let Some(importer) = importer.clone() {
                    CapturedFrame::Gpu(GpuFrame {
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
                        importer: Some(importer),
                    })
                } else {
                    match mmap_bo_to_raw(&alloc) {
                        Ok(raw) => CapturedFrame::Cpu(raw),
                        Err(e) => {
                            warn!(
                                %e,
                                "wayland-direct: gbm_bo_map readback failed — dropping frame"
                            );
                            buffer.destroy();
                            continue;
                        }
                    }
                }
            }
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

/// Read a freshly-captured dmabuf back to host bytes. Used when no
/// Vulkan importer is available to attach to a `CapturedFrame::Gpu`
/// — without an importer the encoder's `into_cpu()` call would fail
/// at the wire, and the failure mode is worse than paying for one
/// mmap copy.
///
/// We *don't* go through `gbm_bo_map` here. That call has been
/// observed to return `EAGAIN` ("Resource temporarily unavailable")
/// on dmabufs the compositor just finished writing into, even after
/// the screencopy `Ready` event — mesa's gbm path doesn't do the
/// cross-process CPU-coherence sync internally. Instead we do the
/// kernel-documented dance:
///
///   1. `DMA_BUF_IOCTL_SYNC(START | READ)` — blocks until the GPU
///      releases the buffer to the CPU. Maps internally to the
///      driver's `dma_buf_attach`-style fencing.
///   2. `mmap(PROT_READ)` of the fd. LINEAR-modifier buffers are
///      laid out row-major at `alloc.stride` bytes per row.
///   3. Row-wise copy into a tightly-packed `width * 4 * height`
///      buffer (the alignment padding upstream encoders don't
///      want to know about).
///   4. `munmap` + `DMA_BUF_IOCTL_SYNC(END | READ)` — releases
///      CPU access back to the GPU side.
///
/// Returns a `RawFrame` whose `data` is the tightly-packed bytes.
fn mmap_bo_to_raw(alloc: &AllocatedBuffer) -> std::result::Result<RawFrame, FerricastError> {
    let fd = alloc.fd.as_raw_fd();
    let map_stride = alloc.stride as usize;
    let w = alloc.width as usize;
    let h = alloc.height as usize;
    let dst_stride = w * 4;
    let size = map_stride * h;
    let format = alloc.pixel_format();

    dma_buf_sync(fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ)
        .map_err(|e| FerricastError::Capture(format!("dma_buf_sync(START|READ): {e}")))?;

    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ,
            libc::MAP_SHARED,
            fd,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        let err = std::io::Error::last_os_error();
        // Best-effort release before returning; we can't read but
        // the kernel still needs to know we're done with the
        // acquire side.
        let _ = dma_buf_sync(fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
        return Err(FerricastError::Capture(format!("mmap(dmabuf): {err}")));
    }
    let src = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };

    // Re-pack into `width * 4` so callers see tightly-packed bytes
    // regardless of what alignment the driver picked. The cost is
    // one row-wise memcpy and matches what the Vulkan-importer
    // readback path produces.
    let data = if map_stride == dst_stride {
        Bytes::copy_from_slice(src)
    } else {
        let mut buf = vec![0u8; dst_stride * h];
        for y in 0..h {
            let src_off = y * map_stride;
            let dst_off = y * dst_stride;
            buf[dst_off..dst_off + dst_stride]
                .copy_from_slice(&src[src_off..src_off + dst_stride]);
        }
        Bytes::from(buf)
    };

    unsafe {
        libc::munmap(ptr, size);
    }
    let _ = dma_buf_sync(fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);

    Ok(RawFrame {
        width: alloc.width,
        height: alloc.height,
        stride: dst_stride as u32,
        format,
        data,
        timestamp_us: now_us(),
    })
}

// `linux/dma-buf.h` constants. Hardcoded because `libc` doesn't
// expose them and adding a `nix` dep for one ioctl is overkill.
const DMA_BUF_SYNC_READ: u64 = 1 << 0;
#[allow(dead_code)]
const DMA_BUF_SYNC_WRITE: u64 = 1 << 1;
const DMA_BUF_SYNC_START: u64 = 0;
const DMA_BUF_SYNC_END: u64 = 1 << 2;
// `_IOW('b', 0, struct dma_buf_sync)` = 0x40086200 on every Linux
// arch we target. struct size = 8 bytes, dir = WRITE (userspace
// writes the request struct), type = 'b' = 0x62, nr = 0. The
// ioctl value is ABI-stable across kernel versions.
const DMA_BUF_IOCTL_SYNC: libc::c_ulong = 0x40086200;

fn dma_buf_sync(fd: std::os::fd::RawFd, flags: u64) -> std::io::Result<()> {
    #[repr(C)]
    struct DmaBufSync {
        flags: u64,
    }
    let req = DmaBufSync { flags };
    let res =
        unsafe { libc::ioctl(fd, DMA_BUF_IOCTL_SYNC, &req as *const DmaBufSync as *const _) };
    if res < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// `/dev/dri/renderD128..D192`, in numerical order. We probe in
/// order so single-GPU hosts (where 128 is the only node) take the
/// happy path. Multi-GPU hosts where the compositor picked the
/// second card go through `main_device` matching above to land on
/// the right one regardless of order.
fn render_node_paths() -> Vec<String> {
    let mut paths = Vec::with_capacity(8);
    for n in 128..=192 {
        let p = format!("/dev/dri/renderD{n}");
        if Path::new(&p).exists() {
            paths.push(p);
        }
    }
    paths
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

// ── linux-dmabuf-v1 feedback ──────────────────────────────────────
//
// Spec walk-through (v4):
//
//   1. Compositor sends `format_table { fd, size }` once: an fd
//      pointing to `size` bytes worth of 16-byte entries. Each
//      entry: { u32 format, u32 padding, u64 modifier }. This is
//      the universe of (format, modifier) pairs the compositor
//      *knows about* — not yet what it accepts.
//   2. `main_device { device }` once: dev_t of the GPU the
//      compositor's primary renderer lives on. Allocating dmabufs
//      on this device guarantees the compositor can read them
//      without cross-GPU transfer.
//   3. One or more *tranches*. A tranche is a contiguous group of
//      events that describes a set of formats for a specific
//      target device with a flags field:
//         `tranche_target_device { device }`
//         `tranche_formats { indices }`  // u16 array into format_table
//         `tranche_flags { flags }`
//         `tranche_done`
//   4. Finally `done` signals end-of-batch.
//
// We only consume tranches whose `target_device == main_device`:
// per-output tranches with a different target are for scanout
// optimisations we don't care about, and using them would defeat
// the "same GPU as compositor" guarantee.

/// What `query_feedback` returns. `main_device` is the dev_t of the
/// compositor's primary GPU as a `u64` (matches `st_rdev` from
/// `stat()`). `accepted` is the map of fourcc → modifiers the
/// compositor will import on that device.
struct FeedbackInfo {
    main_device: Option<u64>,
    accepted: HashMap<u32, Vec<u64>>,
}

/// One pass through the feedback protocol on a private event queue.
/// Returns once `Done` fires (single roundtrip is usually enough)
/// or after a 2s deadline. Compositor-side compositors that don't
/// implement v4 leave us with `main_device = None` and an empty
/// `accepted` — the caller falls back gracefully.
fn query_feedback(
    conn: &Connection,
    dmabuf: &ZwpLinuxDmabufV1,
) -> std::result::Result<FeedbackInfo, FerricastError> {
    let mut feedback_queue: wayland_client::EventQueue<FeedbackState> =
        conn.new_event_queue();
    let fb_qh = feedback_queue.handle();

    // `get_default_feedback` only exists since version 4 of the
    // dmabuf global. If we bound a lower version the request just
    // wouldn't be in the proxy's API surface — we wouldn't reach
    // this function at all (the caller filters on `dmabuf.version()`
    // implicitly via the bind range).
    let _feedback = dmabuf.get_default_feedback(&fb_qh, ());

    let mut state = FeedbackState::default();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while !state.done && std::time::Instant::now() < deadline {
        feedback_queue
            .blocking_dispatch(&mut state)
            .map_err(|e| FerricastError::Capture(format!("dmabuf feedback dispatch: {e}")))?;
    }
    if !state.done {
        return Err(FerricastError::Capture(
            "dmabuf feedback timed out — compositor didn't send `done` within 2s".into(),
        ));
    }

    // Resolve every tranche we captured: mmap the format table,
    // pick out the (format, modifier) entries at the indices we
    // collected for tranches whose target matched main_device.
    let mut accepted: HashMap<u32, Vec<u64>> = HashMap::new();
    if let (Some(fd), Some(main_dev)) = (state.format_table_fd.as_ref(), state.main_device) {
        let table_size = state.format_table_size as usize;
        if table_size > 0 && table_size % 16 == 0 {
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    table_size,
                    libc::PROT_READ,
                    libc::MAP_PRIVATE,
                    fd.as_raw_fd(),
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                warn!(
                    err = %std::io::Error::last_os_error(),
                    "dmabuf feedback: mmap of format_table failed — proceeding without per-format accept-list"
                );
            } else {
                let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, table_size) };
                for tranche in &state.tranches {
                    if tranche.target_device != Some(main_dev) {
                        continue;
                    }
                    for &idx in &tranche.indices {
                        let off = (idx as usize) * 16;
                        if off + 16 > bytes.len() {
                            continue;
                        }
                        let format = u32::from_ne_bytes([
                            bytes[off],
                            bytes[off + 1],
                            bytes[off + 2],
                            bytes[off + 3],
                        ]);
                        let modifier = u64::from_ne_bytes([
                            bytes[off + 8],
                            bytes[off + 9],
                            bytes[off + 10],
                            bytes[off + 11],
                            bytes[off + 12],
                            bytes[off + 13],
                            bytes[off + 14],
                            bytes[off + 15],
                        ]);
                        let list = accepted.entry(format).or_default();
                        if !list.contains(&modifier) {
                            list.push(modifier);
                        }
                    }
                }
                unsafe {
                    libc::munmap(ptr, table_size);
                }
            }
        }
    }

    Ok(FeedbackInfo {
        main_device: state.main_device,
        accepted,
    })
}

#[derive(Default)]
struct FeedbackState {
    done: bool,
    main_device: Option<u64>,
    format_table_fd: Option<OwnedFd>,
    format_table_size: u32,
    tranches: Vec<TrancheState>,
    current: Option<TrancheState>,
}

#[derive(Default, Clone)]
struct TrancheState {
    target_device: Option<u64>,
    /// u16 indices into the format_table.
    indices: Vec<u16>,
}

impl Dispatch<ZwpLinuxDmabufFeedbackV1, ()> for FeedbackState {
    fn event(
        state: &mut Self,
        _: &ZwpLinuxDmabufFeedbackV1,
        event: <ZwpLinuxDmabufFeedbackV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        use zwp_linux_dmabuf_feedback_v1::Event;
        match event {
            Event::Done => state.done = true,
            Event::FormatTable { fd, size } => {
                state.format_table_fd = Some(fd);
                state.format_table_size = size;
            }
            Event::MainDevice { device } => {
                state.main_device = decode_dev_t(&device);
            }
            Event::TrancheTargetDevice { device } => {
                let cur = state.current.get_or_insert_with(TrancheState::default);
                cur.target_device = decode_dev_t(&device);
            }
            Event::TrancheFormats { indices } => {
                // The protocol delivers `indices` as a flat byte
                // array; each entry is a little-endian u16.
                let cur = state.current.get_or_insert_with(TrancheState::default);
                for chunk in indices.chunks_exact(2) {
                    cur.indices.push(u16::from_ne_bytes([chunk[0], chunk[1]]));
                }
            }
            Event::TrancheFlags { .. } => {
                // We don't filter on `scanout` here — screencopy
                // doesn't care, and excluding scanout-only tranches
                // would lose perfectly importable LINEAR layouts
                // on some compositors.
            }
            Event::TrancheDone => {
                if let Some(t) = state.current.take() {
                    state.tranches.push(t);
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<ZwpLinuxDmabufV1, ()> for FeedbackState {
    // Required because `get_default_feedback` was called on a
    // proxy whose state-bound dispatch may briefly touch the new
    // queue. The base dmabuf proxy emits no events in v4+ (its
    // format/modifier events are deprecated), so this is just a
    // type-system formality.
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

/// `dev_t` arrives as 8 little-endian bytes per the wayland protocol
/// (`array` of `u8` representing a `dev_t`). Decode unconditionally
/// LE because Wayland is always little-endian on the wire.
fn decode_dev_t(device: &[u8]) -> Option<u64> {
    if device.len() != 8 {
        return None;
    }
    Some(u64::from_ne_bytes([
        device[0], device[1], device[2], device[3],
        device[4], device[5], device[6], device[7],
    ]))
}
