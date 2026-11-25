//! `DmaBuf` import + readback with cross-frame resource caching.
//!
//! Hot-path layout:
//!
//! 1. `fstat(fd)` to get a stable [`BufferId`] for the underlying
//!    dmabuf and look it up in [`State::imports`]. If it's there with
//!    matching dimensions / modifier, reuse the `VkImage` — the
//!    expensive `vkAllocateMemory(import)` + `vkBindImageMemory`
//!    already paid for it on the first frame.
//! 2. If not cached: dup the fd (Vulkan takes ownership of the dup),
//!    create a fresh `VkImage` with the explicit modifier layout,
//!    import the memory, bind, store in the cache.
//! 3. Ensure the persistent staging buffer is at least
//!    `width * height * bpp` bytes; recreate larger if needed.
//! 4. Reset the cached command buffer + fence, record
//!    `vkCmdCopyImageToBuffer`, submit and wait.
//! 5. Map the staging memory, `Bytes::copy_from_slice` to a packed
//!    linear copy, unmap.
//!
//! Steady-state cost is dominated by the GPU blit + the host copy
//! out — orders of magnitude cheaper than the per-frame
//! create/destroy pattern this replaced.

use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd, RawFd};

use ash::vk;
use bytes::Bytes;
use ferricast_core::{FerricastError, PixelFormat, Result};
use tracing::{trace, warn};

use super::{
    BufferId, ImportedImage, Inner, MAX_IMPORT_CACHE, Slot, Staging, State, format as fmt,
};

/// Resolve a dmabuf fd to its stable underlying buffer identity. All
/// `dup`s of the same fd return the same `(st_dev, st_ino)`, which is
/// what we want for the import cache key — fd numbers are not stable.
fn buffer_id(fd: RawFd) -> std::io::Result<BufferId> {
    // SAFETY: `libc::fstat` only writes through `&mut st`; on error it
    // returns -1 and leaves `errno` set, in which case we don't read
    // `st`.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    let r = unsafe { libc::fstat(fd, &raw mut st) };
    if r != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(BufferId {
        dev: st.st_dev as u64,
        ino: st.st_ino as u64,
    })
}

/// One single-plane dmabuf to import as a `VkImage`.
#[derive(Debug, Clone, Copy)]
pub(super) struct PlaneImport {
    /// Borrowed descriptor, valid for the duration of the call.
    pub(super) fd: RawFd,
    pub(super) offset: u32,
    /// Row pitch of the dmabuf (not of the readback).
    pub(super) stride: u32,
    pub(super) modifier: u64,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) vk_format: vk::Format,
}

/// Submit + wait + read on the next ring slot. Returns tightly
/// packed rows (`width * bpp` bytes each). Used by the
/// `DmaBufImporter` trait.
pub(super) fn run_sync(
    inner: &Inner,
    state: &mut State,
    plane: &PlaneImport,
    pixel_format: PixelFormat,
) -> Result<Bytes> {
    let PlaneImport { width, height, .. } = *plane;
    let bpp = fmt::bytes_per_pixel(pixel_format)
        .ok_or_else(|| FerricastError::Capture("vulkan: format has no bpp".into()))?;
    let staging_size = u64::from(width) * u64::from(height) * u64::from(bpp);
    if staging_size == 0 {
        return Err(FerricastError::Capture(
            "vulkan: staging size is zero".into(),
        ));
    }

    let image = ensure_image(inner, state, plane)?;

    let slot_idx = state.next_slot_idx;
    state.next_slot_idx = (slot_idx + 1) % state.slots.len();

    ensure_staging(
        inner,
        &mut state.slots[slot_idx],
        inner_memory_props(inner),
        staging_size,
    )?;
    let slot = &mut state.slots[slot_idx];
    record_and_submit(inner, slot, plane, image, staging_size)?;
    slot.in_flight = true;
    wait_and_reset_fence(inner, slot.fence)?;
    slot.in_flight = false;
    let staging = slot
        .staging
        .as_ref()
        .ok_or_else(|| FerricastError::Capture("vulkan: slot lost its staging buffer".into()))?;
    let bytes = readback_from_staging(staging, staging_size);

    trace!(width, height, len = bytes.len(), "vulkan readback ok");
    Ok(bytes)
}

fn wait_and_reset_fence(inner: &Inner, fence: vk::Fence) -> Result<()> {
    let started = std::time::Instant::now();
    unsafe {
        inner
            .device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(|e| FerricastError::Capture(format!("vk wait_for_fences: {e}")))?;
        inner
            .device
            .reset_fences(&[fence])
            .map_err(|e| FerricastError::Capture(format!("vk reset_fences: {e}")))?;
    }
    let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    // Anything past one frame period (~16 ms at 60 fps) at this
    // step means GPU contention or driver weirdness — log loudly so
    // we know whether a stall is on us or upstream.
    if elapsed_ms > 50 {
        tracing::warn!(
            elapsed_ms,
            "vk wait_for_fences blocked unusually long — GPU contention or driver issue"
        );
    } else {
        tracing::trace!(elapsed_ms, "vk wait_for_fences");
    }
    Ok(())
}

/// Copy `size` bytes out of the persistently mapped staging into a
/// fresh `Bytes`. Single CPU memcpy; the spec guarantees the GPU
/// writes are visible to the host on a `HOST_COHERENT` memory type
/// once the submission's fence has signalled.
fn readback_from_staging(staging: &Staging, size: u64) -> Bytes {
    // SAFETY: `size` ≤ `staging.capacity` (callers prove this via
    // `ensure_staging(size)` directly above), and the mapping stays
    // live for the entire lifetime of the `Staging`.
    unsafe {
        let slice = std::slice::from_raw_parts(
            staging.mapped_ptr.cast_const(),
            usize::try_from(size).expect("invariant: `size` fits the mapped staging buffer"),
        );
        Bytes::copy_from_slice(slice)
    }
}

/// Borrow trick: `record_and_submit` needs the queue + queue family
/// from `Inner` but takes the slot via `&Slot`, so we need a clean
/// way to expose `memory_props` to `ensure_staging` without holding
/// a borrow on `Inner` while also passing `&mut Slot`. This
/// indirection just hands the props back; the alternative is
/// passing `inner` everywhere, which is fine but more verbose.
fn inner_memory_props(inner: &Inner) -> &vk::PhysicalDeviceMemoryProperties {
    &inner.memory_props
}

/// Look up the cached `VkImage` for the dmabuf behind `fd`, or import
/// a fresh one and stash it. Cache is keyed on the underlying buffer's
/// `(st_dev, st_ino)`, not the fd number, because compositors hand out
/// a fresh `dup`'d fd per frame for the same physical buffer.
fn ensure_image(inner: &Inner, state: &mut State, plane: &PlaneImport) -> Result<vk::Image> {
    let PlaneImport {
        fd,
        modifier,
        width,
        height,
        vk_format,
        ..
    } = *plane;
    let key = buffer_id(fd)
        .map_err(|e| FerricastError::Capture(format!("fstat(dmabuf fd) for cache key: {e}")))?;

    if let Some(existing) = state.imports.get(&key) {
        if existing.width == width
            && existing.height == height
            && existing.format == vk_format
            && existing.modifier == modifier
        {
            trace!(?key, "vulkan import cache HIT");
            return Ok(existing.image);
        }
        // Stale — same buffer identity, but reported shape has
        // changed (renegotiation in flight). Destroy and re-import.
        unsafe {
            let _ = inner.device.device_wait_idle();
            inner.device.destroy_image(existing.image, None);
            inner.device.free_memory(existing.memory, None);
        }
        state.imports.remove(&key);
        state.import_order.retain(|k| *k != key);
    }
    trace!(?key, "vulkan import cache MISS — importing");

    // Import: dup the fd (Vulkan owns the dup; PipeWire keeps the
    // original).
    let dup_fd = unsafe { libc::dup(fd) };
    if dup_fd < 0 {
        return Err(FerricastError::Capture(format!(
            "dup(dmabuf fd): {}",
            std::io::Error::last_os_error()
        )));
    }
    let mut fd_guard = OwnedFdGuard::new(dup_fd);

    let imported = unsafe { import_image(inner, &mut fd_guard, plane)? };
    let image = imported.image;
    state.imports.insert(key, imported);
    state.import_order.push_back(key);

    // FIFO evict the oldest entry if the cache grew past its cap.
    // Real PipeWire pools are much smaller than this — only an
    // anomalous compositor that mints fresh buffers every frame would
    // ever trigger it.
    while state.imports.len() > MAX_IMPORT_CACHE {
        let Some(victim) = state.import_order.pop_front() else {
            break;
        };
        if victim == key {
            // Don't evict the entry we just inserted, even in the
            // degenerate case of a 1-element cap.
            state.import_order.push_back(victim);
            break;
        }
        if let Some(old) = state.imports.remove(&victim) {
            unsafe {
                let _ = inner.device.device_wait_idle();
                inner.device.destroy_image(old.image, None);
                inner.device.free_memory(old.memory, None);
            }
        }
    }

    Ok(image)
}

/// SAFETY: `fd_guard` owns a valid duped dmabuf fd. On success the
/// fd ownership transfers to Vulkan (we mark it `forget`).
#[expect(
    clippy::too_many_lines,
    reason = "one linear setup or event sequence over shared state; splitting it would scatter that state"
)]
unsafe fn import_image(
    inner: &Inner,
    fd_guard: &mut OwnedFdGuard,
    plane: &PlaneImport,
) -> Result<ImportedImage> {
    let PlaneImport {
        offset: plane_offset,
        stride: plane_stride,
        modifier,
        width,
        height,
        vk_format,
        ..
    } = *plane;
    let plane_layouts = [vk::SubresourceLayout::default()
        .offset(u64::from(plane_offset))
        .row_pitch(u64::from(plane_stride))];

    let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
        .drm_format_modifier(modifier)
        .plane_layouts(&plane_layouts);

    let mut external = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk_format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(vk::ImageUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut external)
        .push_next(&mut explicit);

    let image = unsafe {
        inner
            .device
            .create_image(&image_info, None)
            .map_err(|e| FerricastError::Capture(format!("vk create_image: {e}")))?
    };

    let mut mem_reqs = vk::MemoryRequirements2::default();
    unsafe {
        inner.device.get_image_memory_requirements2(
            &vk::ImageMemoryRequirementsInfo2::default().image(image),
            &mut mem_reqs,
        );
    }

    let mut fd_props = vk::MemoryFdPropertiesKHR::default();
    let res = unsafe {
        inner.external_memory_fd.get_memory_fd_properties(
            vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            fd_guard.raw(),
            &mut fd_props,
        )
    };
    if let Err(e) = res {
        unsafe { inner.device.destroy_image(image, None) };
        return Err(FerricastError::Capture(format!(
            "vk get_memory_fd_properties: {e}"
        )));
    }

    let memory_type_index = pick_memory_type(
        &inner.memory_props,
        mem_reqs.memory_requirements.memory_type_bits & fd_props.memory_type_bits,
        vk::MemoryPropertyFlags::empty(),
    );
    let Some(memory_type_index) = memory_type_index else {
        unsafe { inner.device.destroy_image(image, None) };
        return Err(FerricastError::Capture(
            "vk: no memory type matches dmabuf import constraints".into(),
        ));
    };

    let mut import_info = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
        .fd(fd_guard.raw());
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_reqs.memory_requirements.size)
        .memory_type_index(memory_type_index)
        .push_next(&mut dedicated)
        .push_next(&mut import_info);

    let memory = unsafe {
        match inner.device.allocate_memory(&alloc_info, None) {
            Ok(m) => m,
            Err(e) => {
                inner.device.destroy_image(image, None);
                return Err(FerricastError::Capture(format!(
                    "vk allocate_memory(import): {e}"
                )));
            }
        }
    };
    fd_guard.forget(); // Vulkan now owns the fd.

    if let Err(e) = unsafe { inner.device.bind_image_memory(image, memory, 0) } {
        unsafe {
            inner.device.free_memory(memory, None);
            inner.device.destroy_image(image, None);
        }
        return Err(FerricastError::Capture(format!(
            "vk bind_image_memory: {e}"
        )));
    }

    Ok(ImportedImage {
        image,
        memory,
        width,
        height,
        format: vk_format,
        modifier,
    })
}

/// (Re)allocate the slot's staging buffer if it doesn't exist or is
/// too small. The new buffer is mapped once and stays mapped for
/// its lifetime — `vkMapMemory` / `vkUnmapMemory` per frame are
/// pure overhead on `HOST_COHERENT` memory (the spec explicitly
/// permits leaving the mapping live across submits).
fn ensure_staging(
    inner: &Inner,
    slot: &mut Slot,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
) -> Result<()> {
    if let Some(s) = &slot.staging {
        if s.capacity >= size {
            return Ok(());
        }
        // Old buffer too small. A slot's fence is only ever signalled
        // by a submission still marked `in_flight`; every completed
        // wait resets it, so waiting on an idle slot's fence would
        // block forever. Callers wait for the slot before reaching
        // here, so an in-flight slot is a bookkeeping bug — refuse
        // rather than free memory the GPU may still be writing.
        if slot.in_flight {
            return Err(FerricastError::Capture(
                "vulkan: ensure_staging on a slot with an undrained submission".into(),
            ));
        }
        // SAFETY: the slot has no submission in flight (checked
        // above), so the GPU no longer references this buffer.
        unsafe {
            inner.device.unmap_memory(s.memory);
            inner.device.destroy_buffer(s.buffer, None);
            inner.device.free_memory(s.memory, None);
        }
        slot.staging = None;
    }

    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { inner.device.create_buffer(&buffer_info, None) }
        .map_err(|e| FerricastError::Capture(format!("vk create_buffer (staging): {e}")))?;

    let reqs = unsafe { inner.device.get_buffer_memory_requirements(buffer) };
    // The CPU reads every byte of this buffer: cached memory reads at
    // memcpy speed, uncached (write-combined) memory several times
    // slower (a 1080p frame took most of a frame period on NVIDIA).
    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let memory_type = pick_memory_type(
        memory_props,
        reqs.memory_type_bits,
        host | vk::MemoryPropertyFlags::HOST_CACHED,
    )
    .or_else(|| pick_memory_type(memory_props, reqs.memory_type_bits, host))
    .ok_or_else(|| FerricastError::Capture("vk: no host-visible memory type for staging".into()))?;

    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(memory_type);
    let memory = unsafe { inner.device.allocate_memory(&alloc, None) }.map_err(|e| {
        unsafe { inner.device.destroy_buffer(buffer, None) };
        FerricastError::Capture(format!("vk allocate_memory (staging): {e}"))
    })?;

    if let Err(e) = unsafe { inner.device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            inner.device.free_memory(memory, None);
            inner.device.destroy_buffer(buffer, None);
        }
        return Err(FerricastError::Capture(format!(
            "vk bind_buffer_memory (staging): {e}"
        )));
    }

    let mapped_ptr = unsafe {
        inner
            .device
            .map_memory(memory, 0, reqs.size, vk::MemoryMapFlags::empty())
    }
    .map_err(|e| {
        unsafe {
            inner.device.free_memory(memory, None);
            inner.device.destroy_buffer(buffer, None);
        }
        FerricastError::Capture(format!("vk map_memory (staging): {e}"))
    })?
    .cast::<u8>();

    slot.staging = Some(Staging {
        buffer,
        memory,
        capacity: reqs.size,
        mapped_ptr,
    });
    Ok(())
}

/// Record the per-frame command buffer (acquire-from-foreign
/// barrier, image-to-buffer copy, host-read barrier) and submit it
/// on `slot`'s fence. Does NOT wait — that's the caller's job.
fn record_and_submit(
    inner: &Inner,
    slot: &Slot,
    plane: &PlaneImport,
    image: vk::Image,
    staging_size: u64,
) -> Result<()> {
    let PlaneImport {
        fd: dmabuf_fd,
        width,
        height,
        ..
    } = *plane;
    let staging = slot.staging.as_ref().ok_or_else(|| {
        FerricastError::Capture("vulkan: submit on a slot without staging".into())
    })?;
    let cmd = slot.command_buffer;
    let device = &inner.device;

    unsafe {
        device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
            .map_err(|e| FerricastError::Capture(format!("vk reset_command_buffer: {e}")))?;

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device
            .begin_command_buffer(cmd, &begin)
            .map_err(|e| FerricastError::Capture(format!("vk begin_command_buffer: {e}")))?;

        // Acquire the image from FOREIGN (the dmabuf producer). Stay
        // in GENERAL, the layout foreign producers hand dmabufs over
        // in: UNDEFINED would allow the driver to discard the
        // contents, and an in-place transition to
        // TRANSFER_SRC_OPTIMAL could rewrite (e.g. decompress) the
        // compositor's shared image.
        let acquire = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
            .dst_queue_family_index(inner.queue_family_index)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[acquire],
        );

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D::default())
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        device.cmd_copy_image_to_buffer(
            cmd,
            image,
            vk::ImageLayout::GENERAL,
            staging.buffer,
            &[region],
        );

        let host_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(staging.buffer)
            .offset(0)
            .size(staging_size);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[],
            &[host_barrier],
            &[],
        );

        device
            .end_command_buffer(cmd)
            .map_err(|e| FerricastError::Capture(format!("vk end_command_buffer: {e}")))?;

        let cmd_buffers = [cmd];
        // Wait for the producer's pending writes (implicit sync)
        // before the copy reads the image, when the kernel and driver
        // let us turn the dmabuf's fence into a semaphore.
        let wait_semaphores = [slot.acquire_semaphore];
        let wait_stages = [vk::PipelineStageFlags::TRANSFER];
        let mut submit = vk::SubmitInfo::default().command_buffers(&cmd_buffers);
        if arm_acquire_semaphore(inner, slot, dmabuf_fd) {
            submit = submit
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages);
        }
        device
            .queue_submit(inner.queue, &[submit], slot.fence)
            .map_err(|e| FerricastError::Capture(format!("vk queue_submit: {e}")))?;
    }

    Ok(())
}

/// `DMA_BUF_IOCTL_EXPORT_SYNC_FILE` =
/// `_IOWR('b', 2, struct dma_buf_export_sync_file)` (Linux >= 6.0).
const DMA_BUF_IOCTL_EXPORT_SYNC_FILE: u32 = 0xC008_6202;
/// Export the fences a *reader* must wait for (pending writes).
const DMA_BUF_SYNC_READ: u32 = 1;

/// `struct dma_buf_export_sync_file` from `<linux/dma-buf.h>`.
#[repr(C)]
struct DmaBufExportSyncFile {
    flags: u32,
    fd: i32,
}

/// The dmabuf's implicit-sync write fence as a `sync_file`, or `None`
/// on kernels / exporters without `DMA_BUF_IOCTL_EXPORT_SYNC_FILE`.
fn export_read_fence(dmabuf_fd: RawFd) -> Option<OwnedFd> {
    let mut arg = DmaBufExportSyncFile {
        flags: DMA_BUF_SYNC_READ,
        fd: -1,
    };
    // SAFETY: `dmabuf_fd` is a live descriptor owned by the caller's
    // frame; `arg` matches the kernel's struct layout and outlives the
    // call. The request number's type differs between libc flavours,
    // hence the inferred cast.
    let rc = unsafe {
        libc::ioctl(
            dmabuf_fd,
            DMA_BUF_IOCTL_EXPORT_SYNC_FILE.into(),
            &raw mut arg,
        )
    };
    if rc != 0 || arg.fd < 0 {
        trace!(
            error = %std::io::Error::last_os_error(),
            "DMA_BUF_IOCTL_EXPORT_SYNC_FILE unavailable; copying without implicit-sync wait"
        );
        return None;
    }
    // SAFETY: on success the kernel returns a fresh fd we now own.
    Some(unsafe { OwnedFd::from_raw_fd(arg.fd) })
}

/// Load the dmabuf's pending-write fence into `slot.acquire_semaphore`
/// as a temporary payload. Returns `true` when the submit must wait
/// on that semaphore. Any failure degrades to "no wait" (the
/// pre-existing behaviour) rather than dropping the frame.
fn arm_acquire_semaphore(inner: &Inner, slot: &Slot, dmabuf_fd: RawFd) -> bool {
    let Some(ext) = inner.external_semaphore_fd.as_ref() else {
        return false;
    };
    let Some(sync_file) = export_read_fence(dmabuf_fd) else {
        return false;
    };
    let raw = sync_file.into_raw_fd();
    let info = vk::ImportSemaphoreFdInfoKHR::default()
        .semaphore(slot.acquire_semaphore)
        .flags(vk::SemaphoreImportFlags::TEMPORARY)
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
        .fd(raw);
    // SAFETY: the semaphore belongs to this device and no pending
    // submission uses it (the slot's previous submit was drained
    // before the slot was reused). On success Vulkan owns `raw`.
    match unsafe { ext.import_semaphore_fd(&info) } {
        Ok(()) => true,
        Err(e) => {
            // SAFETY: import failed, so ownership of `raw` stayed with us.
            drop(unsafe { OwnedFd::from_raw_fd(raw) });
            warn!(error = %e, "vk import_semaphore_fd(SYNC_FD) failed; copying without wait");
            false
        }
    }
}

fn pick_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..props.memory_type_count {
        let bit = 1u32 << i;
        if type_bits & bit == 0 {
            continue;
        }
        let mt = &props.memory_types[i as usize];
        if mt.property_flags.contains(required_flags) {
            return Some(i);
        }
    }
    if required_flags.is_empty() {
        warn!("no memory type satisfies fd/req constraints");
    }
    None
}

/// `dup`'d dmabuf fd that closes itself on drop unless ownership is
/// handed to Vulkan (via `vkAllocateMemory(VkImportMemoryFdInfoKHR)`).
struct OwnedFdGuard {
    fd: RawFd,
    handed_off: bool,
}

impl OwnedFdGuard {
    fn new(fd: RawFd) -> Self {
        Self {
            fd,
            handed_off: false,
        }
    }
    fn raw(&self) -> RawFd {
        self.fd
    }
    fn forget(&mut self) {
        self.handed_off = true;
    }
}

impl Drop for OwnedFdGuard {
    fn drop(&mut self) {
        if !self.handed_off && self.fd >= 0 {
            unsafe { libc::close(self.fd) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{buffer_id, export_read_fence};

    /// A non-dmabuf fd must make the implicit-sync export degrade to
    /// "no wait", not error or panic.
    #[test]
    fn export_read_fence_degrades_on_non_dmabuf() {
        let fd = make_memfd("ferricast-sync-file");
        assert!(export_read_fence(fd).is_none());
        unsafe { libc::close(fd) };
    }
    use std::ffi::CString;
    use std::os::fd::RawFd;

    fn make_memfd(name: &str) -> RawFd {
        let cname = CString::new(name).unwrap();
        // libc 0.2.183 exposes `memfd_create` on Linux. The flag value
        // (0) is fine — we don't need close-on-exec or sealing.
        let fd = unsafe { libc::memfd_create(cname.as_ptr(), 0) };
        assert!(
            fd >= 0,
            "memfd_create failed: {}",
            std::io::Error::last_os_error()
        );
        fd
    }

    /// Two distinct fds for the same underlying buffer (one obtained
    /// by `dup`) must produce the same [`super::BufferId`] — that is
    /// the invariant the import cache relies on to avoid re-importing
    /// every `PipeWire` frame.
    #[test]
    fn dupd_fds_share_buffer_id() {
        let a = make_memfd("ferricast-buffer-id-a");
        let a_dup = unsafe { libc::dup(a) };
        assert!(
            a_dup >= 0,
            "dup failed: {}",
            std::io::Error::last_os_error()
        );
        assert_ne!(a, a_dup, "dup should return a new fd number");

        let id_a = buffer_id(a).expect("fstat a");
        let id_a_dup = buffer_id(a_dup).expect("fstat a_dup");
        assert_eq!(id_a, id_a_dup, "dup'd fds must share BufferId");

        // A separately-created memfd has a different inode and so a
        // different BufferId, even though it's on the same filesystem.
        let b = make_memfd("ferricast-buffer-id-b");
        let id_b = buffer_id(b).expect("fstat b");
        assert_ne!(id_a, id_b, "distinct buffers must have distinct BufferIds");

        unsafe {
            libc::close(a);
            libc::close(a_dup);
            libc::close(b);
        }
    }
}
