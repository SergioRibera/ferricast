//! DmaBuf import + readback with cross-frame resource caching.
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

use std::os::fd::RawFd;

use ash::vk;
use bytes::Bytes;
use ferricast_core::{FerricastError, PixelFormat, Result};
use tracing::{trace, warn};

use super::{
    BufferId, ImportedImage, Inner, MAX_IMPORT_CACHE, PendingMeta, ReadyFrame, Slot, Staging,
    State, format as fmt,
};

/// Resolve a dmabuf fd to its stable underlying buffer identity. All
/// `dup`s of the same fd return the same `(st_dev, st_ino)`, which is
/// what we want for the import cache key — fd numbers are not stable.
fn buffer_id(fd: RawFd) -> std::io::Result<BufferId> {
    // SAFETY: `libc::fstat` only writes through `&mut st`; on error it
    // returns -1 and leaves `errno` set, in which case we don't read
    // `st`.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    let r = unsafe { libc::fstat(fd, &mut st) };
    if r != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(BufferId {
        dev: st.st_dev as u64,
        ino: st.st_ino as u64,
    })
}

/// Synchronous path: drain any in-flight pipelined work, then
/// submit + wait + read on the next ring slot. Used by the
/// `DmaBufImporter` trait.
pub(super) fn run_sync(
    inner: &Inner,
    state: &mut State,
    fd: RawFd,
    plane_offset: u32,
    plane_stride: u32,
    modifier: u64,
    width: u32,
    height: u32,
    vk_format: vk::Format,
    pixel_format: PixelFormat,
) -> Result<Bytes> {
    let bpp = fmt::bytes_per_pixel(pixel_format)
        .ok_or_else(|| FerricastError::Capture("vulkan: format has no bpp".into()))?;
    let staging_size = (width as u64) * (height as u64) * (bpp as u64);
    if staging_size == 0 {
        return Err(FerricastError::Capture("vulkan: staging size is zero".into()));
    }

    // Flush any pipelined submissions so we don't return their bytes
    // mistakenly as ours, and so the slot we pick is free to reuse.
    flush_pending(inner, state)?;

    let image = ensure_image(
        inner,
        state,
        fd,
        plane_offset,
        plane_stride,
        modifier,
        width,
        height,
        vk_format,
    )?;

    let slot_idx = state.next_slot_idx;
    state.next_slot_idx = (slot_idx + 1) % state.slots.len();

    ensure_staging(inner, &mut state.slots[slot_idx], inner_memory_props(inner), staging_size)?;
    let slot = &state.slots[slot_idx];
    let staging = slot.staging.as_ref().expect("ensure_staging populated it");

    record_and_submit(inner, slot, image, staging, width, height, staging_size)?;
    wait_and_reset_fence(inner, slot.fence)?;
    let bytes = readback_from_staging(staging, staging_size);

    trace!(width, height, len = bytes.len(), "vulkan readback ok");
    Ok(bytes)
}

/// Pipelined path: submit a blit for `fd` to the next free ring slot
/// and return the bytes (with original metadata) of the previously
/// submitted frame, if any. The first call after startup or
/// `reset_cache` returns `Ok(None)`; thereafter every call returns
/// one frame back. PW worker thread uses this so GPU work for frame
/// N+1 overlaps with the CPU memcpy of frame N.
pub(super) fn run_pipelined(
    inner: &Inner,
    state: &mut State,
    fd: RawFd,
    plane_offset: u32,
    plane_stride: u32,
    modifier: u64,
    width: u32,
    height: u32,
    vk_format: vk::Format,
    pixel_format: PixelFormat,
    timestamp_us: u64,
) -> Result<Option<ReadyFrame>> {
    let bpp = fmt::bytes_per_pixel(pixel_format)
        .ok_or_else(|| FerricastError::Capture("vulkan: format has no bpp".into()))?;
    let byte_size = (width as u64) * (height as u64) * (bpp as u64);
    if byte_size == 0 {
        return Err(FerricastError::Capture("vulkan: staging size is zero".into()));
    }

    // 1. Cache lookup / import for the new fd.
    let image = ensure_image(
        inner,
        state,
        fd,
        plane_offset,
        plane_stride,
        modifier,
        width,
        height,
        vk_format,
    )?;

    // 2. Pick the next ring slot. If it still has a pending blit
    //    (ring full), drain *that* one as our return value before
    //    overwriting it. Otherwise drain the oldest pending if any
    //    so latency stays bounded at one ring step.
    let slot_idx = state.next_slot_idx;
    state.next_slot_idx = (slot_idx + 1) % state.slots.len();

    let previous = if state.slots[slot_idx].pending_meta.is_some() {
        // Slot we want to write to is still in flight from one or
        // more rings ago. Drain and remove it from the pending
        // queue so the index bookkeeping stays consistent.
        if let Some(pos) = state.pending.iter().position(|&i| i == slot_idx) {
            state.pending.remove(pos);
        }
        Some(drain_slot(inner, &mut state.slots[slot_idx])?)
    } else if let Some(&oldest) = state.pending.front() {
        state.pending.pop_front();
        Some(drain_slot(inner, &mut state.slots[oldest])?)
    } else {
        None
    };

    // 3. Ensure the chosen slot has a big enough staging buffer.
    ensure_staging(inner, &mut state.slots[slot_idx], inner_memory_props(inner), byte_size)?;
    let slot = &state.slots[slot_idx];
    let staging = slot.staging.as_ref().expect("ensure_staging populated it");

    // 4. Record + submit on this slot. DO NOT wait — that's the
    //    whole point of the ring. The next call (or `flush_pending`)
    //    will wait on this slot's fence.
    record_and_submit(inner, slot, image, staging, width, height, byte_size)?;

    // 5. Mark the slot pending so the next ring step knows what to
    //    rebuild when it drains.
    let bgra_stride = (width as u64) * (bpp as u64);
    state.slots[slot_idx].pending_meta = Some(PendingMeta {
        width,
        height,
        stride: bgra_stride as u32,
        format: pixel_format,
        timestamp_us,
        byte_size,
    });
    state.pending.push_back(slot_idx);

    if let Some(prev) = &previous {
        trace!(
            width = prev.width,
            height = prev.height,
            len = prev.bytes.len(),
            "vulkan pipelined readback ok"
        );
    }
    Ok(previous)
}

/// Wait for every in-flight pipelined submission, dropping the bytes
/// (we have no caller waiting on them in the sync path). Used to
/// reset the ring before a synchronous submit so the slot we pick
/// isn't still being read by the GPU.
fn flush_pending(inner: &Inner, state: &mut State) -> Result<()> {
    while let Some(slot_idx) = state.pending.pop_front() {
        wait_and_reset_fence(inner, state.slots[slot_idx].fence)?;
        state.slots[slot_idx].pending_meta = None;
    }
    Ok(())
}

/// Wait on the slot's fence, copy the staging bytes out, clear the
/// pending marker. The bytes are wrapped with the slot's stored
/// metadata so the caller can rebuild a `RawFrame`.
fn drain_slot(inner: &Inner, slot: &mut Slot) -> Result<ReadyFrame> {
    let meta = slot
        .pending_meta
        .take()
        .ok_or_else(|| FerricastError::Capture("vulkan: drain_slot on idle slot".into()))?;
    wait_and_reset_fence(inner, slot.fence)?;
    let staging = slot
        .staging
        .as_ref()
        .ok_or_else(|| FerricastError::Capture("vulkan: drain_slot with no staging".into()))?;
    let bytes = readback_from_staging(staging, meta.byte_size);
    Ok(ReadyFrame {
        bytes,
        width: meta.width,
        height: meta.height,
        stride: meta.stride,
        format: meta.format,
        timestamp_us: meta.timestamp_us,
    })
}

fn wait_and_reset_fence(inner: &Inner, fence: vk::Fence) -> Result<()> {
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
        let slice = std::slice::from_raw_parts(staging.mapped_ptr as *const u8, size as usize);
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
fn ensure_image(
    inner: &Inner,
    state: &mut State,
    fd: RawFd,
    plane_offset: u32,
    plane_stride: u32,
    modifier: u64,
    width: u32,
    height: u32,
    vk_format: vk::Format,
) -> Result<vk::Image> {
    let key = buffer_id(fd).map_err(|e| {
        FerricastError::Capture(format!("fstat(dmabuf fd) for cache key: {e}"))
    })?;

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

    let imported = unsafe {
        import_image(
            inner,
            &mut fd_guard,
            plane_offset,
            plane_stride,
            modifier,
            width,
            height,
            vk_format,
        )?
    };
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
unsafe fn import_image(
    inner: &Inner,
    fd_guard: &mut OwnedFdGuard,
    plane_offset: u32,
    plane_stride: u32,
    modifier: u64,
    width: u32,
    height: u32,
    vk_format: vk::Format,
) -> Result<ImportedImage> {
    let plane_layouts = [vk::SubresourceLayout::default()
        .offset(plane_offset as u64)
        .row_pitch(plane_stride as u64)];

    let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
        .drm_format_modifier(modifier)
        .plane_layouts(&plane_layouts);

    let mut external = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk_format)
        .extent(vk::Extent3D { width, height, depth: 1 })
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
/// pure overhead on HOST_COHERENT memory (the spec explicitly
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
        // Old buffer too small. Wait on this slot's fence so the GPU
        // isn't writing into the buffer we're about to free, then
        // tear it down (unmap before free).
        unsafe {
            let _ = inner.device.wait_for_fences(&[slot.fence], true, u64::MAX);
            let _ = inner.device.reset_fences(&[slot.fence]);
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
    let memory_type = pick_memory_type(
        memory_props,
        reqs.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or_else(|| {
        FerricastError::Capture("vk: no host-visible memory type for staging".into())
    })?;

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
    })? as *mut u8;

    slot.staging = Some(Staging {
        buffer,
        memory,
        capacity: reqs.size,
        mapped_ptr,
    });
    Ok(())
}

/// Record the per-frame command buffer (acquire-from-foreign barrier
/// + image-to-buffer copy + host-read barrier) and submit it on
/// `slot`'s fence. Does NOT wait — that's the caller's job (and the
/// reason for the ring: the wait happens on the *next* iteration so
/// the GPU and CPU overlap).
fn record_and_submit(
    inner: &Inner,
    slot: &Slot,
    image: vk::Image,
    staging: &Staging,
    width: u32,
    height: u32,
    staging_size: u64,
) -> Result<()> {
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

        // Acquire the image from FOREIGN (the dmabuf producer) and
        // transition to TRANSFER_SRC_OPTIMAL.
        let acquire = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
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
            .image_extent(vk::Extent3D { width, height, depth: 1 });
        device.cmd_copy_image_to_buffer(
            cmd,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
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
        let submit = vk::SubmitInfo::default().command_buffers(&cmd_buffers);
        device
            .queue_submit(inner.queue, &[submit], slot.fence)
            .map_err(|e| FerricastError::Capture(format!("vk queue_submit: {e}")))?;
    }

    Ok(())
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
    use super::buffer_id;
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
    /// every PipeWire frame.
    #[test]
    fn dupd_fds_share_buffer_id() {
        let a = make_memfd("ferricast-buffer-id-a");
        let a_dup = unsafe { libc::dup(a) };
        assert!(a_dup >= 0, "dup failed: {}", std::io::Error::last_os_error());
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
