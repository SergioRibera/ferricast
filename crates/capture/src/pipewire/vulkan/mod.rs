//! Vulkan-backed DmaBuf importer.
//!
//! Why this module exists: PipeWire's screencast on Mutter / KWin
//! prefers DmaBuf for performance, but the buffers come back with
//! GPU-vendor-specific DRM format modifiers (tiled, compressed). We
//! can't `mmap` those for CPU read — every byte we'd see would be
//! garbage. The fix is to import the dmabuf into a `VkImage`, blit it
//! to a linearly-laid-out staging buffer, and `mmap` *that*.
//!
//! Public surface:
//!
//! * [`VulkanImporter::new`] — try to bring up Vulkan. Returns `Err`
//!   on systems without a loader / driver / required extensions, in
//!   which case [`super::stream`] falls back to SHM-only EnumFormat
//!   advertising.
//! * [`VulkanImporter::supported_modifiers`] — the actual DRM
//!   modifiers the GPU can produce/consume for a given pixel format.
//!   Used to populate the EnumFormat pod's `VideoModifier` choice.
//! * [`VulkanImporter::import_and_readback`] — import a single dmabuf
//!   plane and copy its decoded pixels back to a CPU [`bytes::Bytes`].
//! * [`VulkanImporter::reset_cache`] — invalidate per-frame caches
//!   (called on format renegotiation).
//!
//! Performance note: the importer keeps a `RING_DEPTH`-slot ring of
//! `(command_buffer, fence, staging)` plus a `HashMap<BufferId,
//! VkImage>` cache, allowing GPU work for frame N+1 to overlap with
//! the CPU memcpy of frame N. PipeWire reuses the same handful of
//! underlying dmabuf objects from a small pool, but compositors
//! (notably Mutter on NVIDIA) routinely hand out a fresh `dup`'d fd
//! for each buffer borrow, so the `RawFd` number is not stable. We
//! key the cache on `(st_dev, st_ino)` of the fd instead, which is
//! invariant across `dup`. With that key the expensive
//! `vkAllocateMemory` + `vkBindImageMemory` only pay once per pool
//! entry; subsequent frames record-and-submit on a free ring slot
//! and return the previous slot's bytes (see
//! [`VulkanImporter::submit_and_take_previous`]).

mod format;
mod import;
mod init;
mod modifiers;

use std::collections::{HashMap, VecDeque};
use std::os::fd::RawFd;
use std::sync::Mutex;

use ash::vk;
use bytes::Bytes;
use ferricast_core::{DmaBufImporter, DmaBufPlane, FerricastError, PixelFormat, Result};
use pipewire::spa::param::video::VideoFormat;
use tracing::{debug, warn};

/// Public handle. The mutable per-frame state lives behind a
/// `Mutex` so the whole importer is `Send + Sync` and can travel
/// across thread boundaries inside an `Arc<dyn DmaBufImporter>` (a
/// `GpuFrame` carries one of these so consumers on any thread can
/// call `readback`). Contention is effectively zero: only one
/// thread at a time uses it (the segmenter task or the PW worker,
/// not both simultaneously).
pub(crate) struct VulkanImporter {
    /// Field order matters: `state` drops first (no-op since it's a
    /// plain struct) then `inner` drops second (and `Inner::drop`
    /// destroys the command pool, device, and instance). Per-frame
    /// Vulkan objects in `state` are destroyed *manually* in
    /// `VulkanImporter::drop` before the device is torn down.
    state: Mutex<State>,
    inner: Inner,
}

/// Owned Vulkan handles created once at importer-bring-up time.
/// `Drop` destroys them in reverse order.
pub(super) struct Inner {
    /// Entry must outlive `instance` — it owns the `libvulkan.so`
    /// dlopen handle. Kept here so the loader stays mapped for as
    /// long as we have any Vulkan state.
    #[allow(dead_code)]
    pub(super) entry: ash::Entry,
    pub(super) instance: ash::Instance,
    pub(super) physical_device: vk::PhysicalDevice,
    pub(super) device: ash::Device,
    pub(super) queue: vk::Queue,
    pub(super) queue_family_index: u32,
    pub(super) command_pool: vk::CommandPool,

    /// Loaded `VK_EXT_image_drm_format_modifier` device fn pointers.
    /// Currently unused (we hand Vulkan an explicit modifier on
    /// import so we don't need to query it back) but cheap to keep
    /// loaded.
    #[allow(dead_code)]
    pub(super) drm_modifier: ash::ext::image_drm_format_modifier::Device,
    #[allow(dead_code)] // used for the future GPU-handoff encoder path
    pub(super) external_memory_fd: ash::khr::external_memory_fd::Device,

    /// Memory type indices grouped by the property flags we care
    /// about. Set up once during init so each import / readback
    /// doesn't re-walk `VkPhysicalDeviceMemoryProperties`.
    pub(super) memory_props: vk::PhysicalDeviceMemoryProperties,
}

/// Stable identity for a dmabuf-backed buffer. PipeWire's screencast
/// portal frequently `dup`s the same underlying dmabuf and hands us a
/// fresh fd number, but `(st_dev, st_ino)` is the same for every dup
/// of the same buffer — so it is a safe key for our import cache.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub(super) struct BufferId {
    pub(super) dev: u64,
    pub(super) ino: u64,
}

/// Cap on how many imported `VkImage`s we keep cached. Real PipeWire
/// pools are 4-8 buffers; this cap exists only as a safety net for
/// pathological compositors that mint a new buffer every frame. When
/// exceeded, the oldest entry (by insertion order) is destroyed.
pub(super) const MAX_IMPORT_CACHE: usize = 16;

/// Number of `(command_buffer, fence, staging)` slots in the
/// pipeline ring. Two is enough to hide the GPU blit behind one
/// CPU memcpy: while the worker memcopies frame N out of slot A,
/// the GPU is already blitting frame N+1 into slot B. A deeper ring
/// would only buy more latency without throughput, since the GPU
/// queue serialises blits anyway.
pub(super) const RING_DEPTH: usize = 2;

/// Per-frame mutable cache. Lives behind a `Mutex` because the PW
/// worker thread is the only consumer and the borrow is scoped to a
/// single `import_and_readback` / `submit_and_take_previous` call.
pub(super) struct State {
    /// `RING_DEPTH` independent submission slots. Entry 0 is also
    /// what the synchronous `import_and_readback` path uses (after
    /// flushing any in-flight pipelined submissions).
    pub(super) slots: Vec<Slot>,
    /// Round-robin counter. The next pipelined submit lands on
    /// `slots[next_slot_idx]`, and `next_slot_idx` advances modulo
    /// `RING_DEPTH` afterwards.
    pub(super) next_slot_idx: usize,
    /// Slot indices with a submission currently in flight, in the
    /// order they were submitted (oldest first). Drained from the
    /// front by [`VulkanImporter::submit_and_take_previous`].
    pub(super) pending: VecDeque<usize>,
    /// Imported dmabuf images keyed by stable buffer identity (see
    /// [`BufferId`]). Cleared by `reset_cache` on renegotiation.
    pub(super) imports: HashMap<BufferId, ImportedImage>,
    /// FIFO of cache keys in insertion order, used to evict the
    /// oldest entry once `imports.len() > MAX_IMPORT_CACHE`.
    pub(super) import_order: VecDeque<BufferId>,
    /// `false` until the very first pipelined submission has been
    /// drained back to the consumer. While `false`,
    /// [`run_pipelined`] drains synchronously instead of returning
    /// `Ok(None)` — this avoids the priming deadlock when the
    /// upstream (event-driven Wayland compositors) only sends a
    /// new buffer when the screen actually changes. Reset to
    /// `false` by [`reset_cache`] on renegotiation.
    pub(super) primed: bool,
}

/// One ring entry: command buffer that records the blit, fence that
/// signals when the GPU is done, persistent host-visible staging the
/// blit copies into, and the metadata of any submission currently in
/// flight on this slot.
pub(super) struct Slot {
    pub(super) command_buffer: vk::CommandBuffer,
    pub(super) fence: vk::Fence,
    pub(super) staging: Option<Staging>,
    /// `Some` while the slot has a submitted-but-not-yet-drained
    /// blit. The metadata travels alongside the bytes when the slot
    /// is drained so the caller can reconstruct a `RawFrame`.
    pub(super) pending_meta: Option<PendingMeta>,
}

/// Everything needed to rebuild a `RawFrame` from a slot's staging
/// buffer once its blit completes.
#[derive(Clone, Copy)]
pub(super) struct PendingMeta {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) stride: u32,
    pub(super) format: PixelFormat,
    pub(super) timestamp_us: u64,
    pub(super) byte_size: u64,
}

/// Result of a pipelined `submit_and_take_previous` call. The bytes
/// belong to the *previously* submitted frame (one ring step back);
/// the metadata travels with them so the caller can rebuild a
/// `RawFrame`.
pub(crate) struct ReadyFrame {
    pub(crate) bytes: Bytes,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) stride: u32,
    pub(crate) format: PixelFormat,
    pub(crate) timestamp_us: u64,
}

pub(super) struct Staging {
    pub(super) buffer: vk::Buffer,
    pub(super) memory: vk::DeviceMemory,
    pub(super) capacity: u64,
    /// Persistent host-visible mapping. Mapped once at creation,
    /// unmapped only on destruction — `vkMapMemory` / `vkUnmapMemory`
    /// per frame is pointless overhead for HOST_COHERENT memory and
    /// the spec explicitly allows leaving the mapping live.
    ///
    /// Stored as raw `*mut u8` so we don't fight Rust's lifetime
    /// model for a pointer whose lifetime is tied to the Vulkan
    /// device, not the borrow checker. Always non-null while the
    /// `Staging` exists; only ever read while the importer's `Mutex`
    /// is held, so the `unsafe impl Send` is sound.
    pub(super) mapped_ptr: *mut u8,
}

// SAFETY: `mapped_ptr` is a stable, thread-agnostic pointer into a
// HOST_COHERENT Vulkan device memory mapping. All reads happen with
// the importer's `Mutex<State>` held, so it never crosses threads
// concurrently. The other Vulkan handles in `Staging` are integer
// IDs (already `Send`), and `*mut u8` would otherwise opt the type
// out of `Send`.
unsafe impl Send for Staging {}

pub(super) struct ImportedImage {
    pub(super) image: vk::Image,
    pub(super) memory: vk::DeviceMemory,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) format: vk::Format,
    pub(super) modifier: u64,
}

impl VulkanImporter {
    /// Try to bring up a Vulkan instance + device with the extensions
    /// we need to import dmabuf and copy-back to CPU.
    ///
    /// Returns `Err(FerricastError::Capture(_))` on:
    /// * Vulkan loader missing (no `libvulkan.so.1`)
    /// * No physical device exposes `VK_EXT_image_drm_format_modifier`
    ///   + `VK_EXT_external_memory_dma_buf`
    /// * Logical device creation failed
    ///
    /// The capture worker treats this as a soft failure and falls
    /// back to SHM-only EnumFormat advertising.
    pub(crate) fn new() -> Result<Self> {
        let inner = init::build()?;
        let state = State::create(&inner)?;
        Ok(Self {
            inner,
            state: Mutex::new(state),
        })
    }

    /// DRM modifiers the GPU can import/export for the given video
    /// format. Pass these to PipeWire's `VideoModifier` choice in the
    /// EnumFormat pod.
    pub(crate) fn supported_modifiers(&self, format: VideoFormat) -> Vec<u64> {
        let Some(vk_format) = format::video_format_to_vk(format) else {
            debug!(?format, "no VkFormat mapping, treating as unsupported");
            return Vec::new();
        };
        match modifiers::query(&self.inner, vk_format) {
            Ok(list) => list,
            Err(e) => {
                warn!(?format, error = %e, "modifier query failed");
                Vec::new()
            }
        }
    }

    /// Synchronous import + readback. Drains any in-flight pipeline
    /// first (so the result is for *this* fd, not the previous one)
    /// and returns the linear pixel bytes. Used by the
    /// `DmaBufImporter` trait for any caller that hasn't switched to
    /// the pipelined API.
    pub(crate) fn import_and_readback(
        &self,
        fd: RawFd,
        plane_offset: u32,
        plane_stride: u32,
        modifier: u64,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<Bytes> {
        let Some(vk_format) = format::pixel_format_to_vk(format) else {
            return Err(FerricastError::Capture(format!(
                "Vulkan: unsupported PixelFormat {format:?}"
            )));
        };
        let mut state = self.state.lock().expect("vulkan state mutex poisoned");
        import::run_sync(
            &self.inner,
            &mut state,
            fd,
            plane_offset,
            plane_stride,
            modifier,
            width,
            height,
            vk_format,
            format,
        )
    }

    /// Pipelined entry point used by the PW worker. Submits a blit
    /// for `fd` to a fresh ring slot and returns the bytes (with
    /// original metadata) of the *previous* frame submitted on
    /// another slot, if any. The very first call after startup or
    /// after `reset_cache` returns `Ok(None)` — there's nothing in
    /// flight yet — and every subsequent call returns one frame.
    /// Net effect: GPU work for frame N+1 overlaps with the CPU
    /// memcpy for frame N.
    ///
    /// Public name spelled `import_pipelined` (instead of
    /// `submit_and_take_previous`) because that's how it reads at
    /// the call site in `handle_process`.
    pub(crate) fn import_pipelined(
        &self,
        fd: RawFd,
        plane_offset: u32,
        plane_stride: u32,
        modifier: u64,
        width: u32,
        height: u32,
        format: PixelFormat,
        timestamp_us: u64,
    ) -> Result<Option<ReadyFrame>> {
        let Some(vk_format) = format::pixel_format_to_vk(format) else {
            return Err(FerricastError::Capture(format!(
                "Vulkan: unsupported PixelFormat {format:?}"
            )));
        };
        let mut state = self.state.lock().expect("vulkan state mutex poisoned");
        import::run_pipelined(
            &self.inner,
            &mut state,
            fd,
            plane_offset,
            plane_stride,
            modifier,
            width,
            height,
            vk_format,
            format,
            timestamp_us,
        )
    }

    /// Wipe the imported-image cache and drop any in-flight
    /// pipelined work. Called when PipeWire renegotiates: the old
    /// fds are about to be closed and any new stream may pick a
    /// different format, so cached `VkImage`s would be stale.
    pub(crate) fn reset_cache(&self) {
        let mut state = self.state.lock().expect("vulkan state mutex poisoned");
        // Wait until any in-flight submit completes before we destroy
        // the images those commands referenced.
        unsafe {
            let _ = self.inner.device.device_wait_idle();
            for img in state.imports.values() {
                self.inner.device.destroy_image(img.image, None);
                self.inner.device.free_memory(img.memory, None);
            }
        }
        state.imports.clear();
        state.import_order.clear();
        // Drop any in-flight pipeline state — the bytes those slots
        // were going to produce are now meaningless because the
        // images they reference are gone.
        for slot in state.slots.iter_mut() {
            slot.pending_meta = None;
        }
        state.pending.clear();
        // Renegotiation = a fresh stream that will need to be
        // primed again. Clear the flag so the next first frame
        // drains synchronously.
        state.primed = false;
        debug!("Vulkan import cache cleared");
    }
}

/// Object-safe view exposed via `Arc<dyn DmaBufImporter>` on every
/// `GpuFrame`. The capture path stores the importer in an `Arc<Self>`
/// once and clones the Arc into each frame; consumers (encoders)
/// call `readback` on demand.
impl DmaBufImporter for VulkanImporter {
    fn readback(
        &self,
        plane: &DmaBufPlane,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<Bytes> {
        self.import_and_readback(
            plane.fd,
            plane.offset,
            plane.stride,
            plane.modifier,
            width,
            height,
            format,
        )
    }
}

impl State {
    fn create(inner: &Inner) -> Result<Self> {
        // Allocate `RING_DEPTH` primary command buffers in one call.
        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(inner.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(RING_DEPTH as u32);
        let cmd_buffers = unsafe { inner.device.allocate_command_buffers(&cmd_alloc) }
            .map_err(|e| FerricastError::Capture(format!("vk allocate_command_buffers: {e}")))?;

        let mut slots = Vec::with_capacity(RING_DEPTH);
        for cmd in cmd_buffers {
            let fence = unsafe {
                inner
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
            }
            .map_err(|e| FerricastError::Capture(format!("vk create_fence: {e}")))?;
            slots.push(Slot {
                command_buffer: cmd,
                fence,
                staging: None,
                pending_meta: None,
            });
        }

        Ok(Self {
            slots,
            next_slot_idx: 0,
            pending: VecDeque::new(),
            imports: HashMap::new(),
            import_order: VecDeque::new(),
            primed: false,
        })
    }
}

impl Drop for VulkanImporter {
    fn drop(&mut self) {
        // Block until the GPU is fully idle so we don't destroy
        // resources while a submitted command buffer might still be
        // referencing them. Errors here are unrecoverable; just log.
        unsafe {
            if let Err(e) = self.inner.device.device_wait_idle() {
                warn!(error = %e, "device_wait_idle on shutdown");
            }
        }

        // Tear down per-frame state explicitly while `inner.device`
        // is still alive. After this returns, `state` drops as a
        // plain struct (no Drop impl) and `inner` drops next, which
        // destroys the command pool, device, and instance.
        let mut state = match self.state.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(), // poisoned — destroy anyway
        };
        let dev = &self.inner.device;
        unsafe {
            for img in state.imports.values() {
                dev.destroy_image(img.image, None);
                dev.free_memory(img.memory, None);
            }
            state.imports.clear();
            state.import_order.clear();
            state.pending.clear();
            // Tear down each ring slot. `device_wait_idle` above
            // guarantees the GPU isn't still touching any of these
            // resources, so destruction order within the slot is
            // free.
            let mut cmd_to_free: Vec<vk::CommandBuffer> = Vec::with_capacity(state.slots.len());
            for slot in state.slots.drain(..) {
                if let Some(s) = slot.staging {
                    if !s.mapped_ptr.is_null() {
                        dev.unmap_memory(s.memory);
                    }
                    dev.destroy_buffer(s.buffer, None);
                    dev.free_memory(s.memory, None);
                }
                if slot.fence != vk::Fence::null() {
                    dev.destroy_fence(slot.fence, None);
                }
                if slot.command_buffer != vk::CommandBuffer::null() {
                    cmd_to_free.push(slot.command_buffer);
                }
            }
            if !cmd_to_free.is_empty() {
                dev.free_command_buffers(self.inner.command_pool, &cmd_to_free);
            }
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        // SAFETY: VulkanImporter::drop already destroyed all
        // per-frame resources and waited for the GPU to idle.
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
