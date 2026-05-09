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
//! Performance note: the importer keeps a persistent command buffer,
//! fence, staging buffer and a `HashMap<fd, VkImage>` cache. PipeWire
//! reuses the same handful of fds within a negotiation cycle, so the
//! `vkAllocateMemory` + `vkBindImageMemory` cost (the most expensive
//! pieces of dmabuf import) only pays for the first frame per fd.
//! Subsequent frames record-and-submit a single command buffer and
//! memcpy from the staging buffer.

mod format;
mod import;
mod init;
mod modifiers;

pub(super) use modifiers::ModifierCaps;

use std::collections::HashMap;
use std::os::fd::RawFd;
use std::sync::Mutex;

use ash::vk;
use bytes::Bytes;
use ferricast_core::{
    DmaBufImporter, DmaBufPlane, FerricastError, PixelFormat, Result,
};
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

/// Per-frame mutable cache. Lives behind a `RefCell` because the PW
/// worker thread is the only consumer and the borrow is scoped to a
/// single `import_and_readback` call.
pub(super) struct State {
    /// Single primary command buffer. Reset and re-recorded every
    /// frame instead of allocating a fresh one (vkAllocate is
    /// surprisingly expensive on some drivers).
    pub(super) command_buffer: vk::CommandBuffer,
    /// Reused fence. Reset after each `wait_for_fences`.
    pub(super) fence: vk::Fence,
    /// Persistent host-visible staging buffer. Recreated only when a
    /// format renegotiation needs more bytes than we have.
    pub(super) staging: Option<Staging>,
    /// Imported dmabuf images keyed by their original fd. Within one
    /// negotiation cycle PipeWire reuses the same handful of fds, so
    /// the expensive `vkAllocateMemory(import)` only happens once
    /// per pool entry. Cleared by `reset_cache` on renegotiation.
    pub(super) imports: HashMap<RawFd, ImportedImage>,
}

pub(super) struct Staging {
    pub(super) buffer: vk::Buffer,
    pub(super) memory: vk::DeviceMemory,
    pub(super) capacity: u64,
}

pub(super) struct ImportedImage {
    pub(super) image: vk::Image,
    /// One `VkDeviceMemory` per memory plane the modifier requires.
    /// Single-plane modifiers store one allocation; multi-plane
    /// modifiers (e.g. AMD DCC retile) store one per plane and the
    /// image was created with `VK_IMAGE_CREATE_DISJOINT_BIT`.
    pub(super) memories: Vec<vk::DeviceMemory>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) format: vk::Format,
    pub(super) modifier: u64,
    pub(super) plane_count: u32,
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
    /// format, paired with their required memory plane count. Pass
    /// these to PipeWire's `VideoModifier` choice in the EnumFormat
    /// pod.
    pub(crate) fn supported_modifiers(&self, format: VideoFormat) -> Vec<ModifierCaps> {
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

    /// Import a dmabuf (one or more memory planes) into a `VkImage`
    /// using the cached entry when available, blit it into the
    /// persistent staging buffer and copy the result out as `Bytes`.
    pub(crate) fn import_and_readback(
        &self,
        planes: &[DmaBufPlane],
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
        if planes.is_empty() {
            return Err(FerricastError::Capture(
                "Vulkan: import_and_readback called with zero planes".into(),
            ));
        }
        let mut state = self.state.lock().expect("vulkan state mutex poisoned");
        import::run(
            &self.inner,
            &mut state,
            planes,
            modifier,
            width,
            height,
            vk_format,
            format,
        )
    }

    /// Wipe the imported-image cache. Called when PipeWire
    /// renegotiates: the old fds are about to be closed and any new
    /// stream may pick a different format, so cached `VkImage`s
    /// would be stale.
    pub(crate) fn reset_cache(&self) {
        let mut state = self.state.lock().expect("vulkan state mutex poisoned");
        // Wait until any in-flight submit completes before we destroy
        // the images those commands referenced.
        unsafe {
            let _ = self.inner.device.device_wait_idle();
            for img in state.imports.values() {
                self.inner.device.destroy_image(img.image, None);
                for mem in &img.memories {
                    self.inner.device.free_memory(*mem, None);
                }
            }
        }
        state.imports.clear();
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
        planes: &[DmaBufPlane],
        modifier: u64,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<Bytes> {
        self.import_and_readback(planes, modifier, width, height, format)
    }
}

impl State {
    fn create(inner: &Inner) -> Result<Self> {
        // Allocate one primary command buffer.
        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(inner.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buffers = unsafe { inner.device.allocate_command_buffers(&cmd_alloc) }
            .map_err(|e| FerricastError::Capture(format!("vk allocate_command_buffers: {e}")))?;

        let fence = unsafe {
            inner
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|e| FerricastError::Capture(format!("vk create_fence: {e}")))?;

        Ok(Self {
            command_buffer: cmd_buffers[0],
            fence,
            staging: None,
            imports: HashMap::new(),
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
                for mem in &img.memories {
                    dev.free_memory(*mem, None);
                }
            }
            state.imports.clear();
            if let Some(s) = state.staging.take() {
                dev.destroy_buffer(s.buffer, None);
                dev.free_memory(s.memory, None);
            }
            if state.fence != vk::Fence::null() {
                dev.destroy_fence(state.fence, None);
            }
            if state.command_buffer != vk::CommandBuffer::null() {
                dev.free_command_buffers(self.inner.command_pool, &[state.command_buffer]);
            }
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        // SAFETY: VulkanImporter::drop already destroyed all
        // per-frame resources and waited for the GPU to idle.
        unsafe {
            self.device
                .destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

