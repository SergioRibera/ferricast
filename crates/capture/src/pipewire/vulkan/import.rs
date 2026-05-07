//! DmaBuf import + readback with cross-frame resource caching.
//!
//! Hot-path layout:
//!
//! 1. Look up the dmabuf fd in [`State::imports`]. If it's there with
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

use super::{format as fmt, ImportedImage, Inner, Staging, State};

pub(super) fn run(
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

    // 1. Look up / import the VkImage for this fd.
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

    // 2. Make sure the staging buffer is large enough.
    ensure_staging(inner, state, staging_size)?;
    let staging = state.staging.as_ref().expect("ensure_staging populated it");

    // 3. Record + submit + wait.
    record_and_submit(inner, state, image, staging, width, height, staging_size)?;

    // 4. Read out.
    let bytes = readback(inner, staging, staging_size)?;

    trace!(width, height, len = bytes.len(), "vulkan readback ok");
    Ok(bytes)
}

/// Look up the cached `VkImage` for the given dmabuf fd, or import a
/// fresh one and stash it.
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
    if let Some(existing) = state.imports.get(&fd) {
        if existing.width == width
            && existing.height == height
            && existing.format == vk_format
            && existing.modifier == modifier
        {
            return Ok(existing.image);
        }
        // Stale — same fd numerical, but the buffer behind it has
        // changed shape. Destroy and re-import.
        unsafe {
            let _ = inner.device.device_wait_idle();
            inner.device.destroy_image(existing.image, None);
            inner.device.free_memory(existing.memory, None);
        }
        state.imports.remove(&fd);
    }

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
    state.imports.insert(fd, imported);
    Ok(state.imports.get(&fd).unwrap().image)
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

/// (Re)allocate the staging buffer if it doesn't exist or is too small.
fn ensure_staging(inner: &Inner, state: &mut State, size: u64) -> Result<()> {
    if let Some(s) = &state.staging {
        if s.capacity >= size {
            return Ok(());
        }
        // Old buffer too small. Wait for any pending GPU op then
        // destroy.
        unsafe {
            let _ = inner.device.device_wait_idle();
            inner.device.destroy_buffer(s.buffer, None);
            inner.device.free_memory(s.memory, None);
        }
        state.staging = None;
    }

    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { inner.device.create_buffer(&buffer_info, None) }
        .map_err(|e| FerricastError::Capture(format!("vk create_buffer (staging): {e}")))?;

    let reqs = unsafe { inner.device.get_buffer_memory_requirements(buffer) };
    let memory_type = pick_memory_type(
        &inner.memory_props,
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

    state.staging = Some(Staging {
        buffer,
        memory,
        capacity: reqs.size,
    });
    Ok(())
}

/// Record the per-frame command buffer (acquire-from-foreign barrier
/// + image-to-buffer copy + host-read barrier), submit, wait on the
/// fence and reset both for the next iteration.
fn record_and_submit(
    inner: &Inner,
    state: &State,
    image: vk::Image,
    staging: &Staging,
    width: u32,
    height: u32,
    staging_size: u64,
) -> Result<()> {
    let cmd = state.command_buffer;
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
            .queue_submit(inner.queue, &[submit], state.fence)
            .map_err(|e| FerricastError::Capture(format!("vk queue_submit: {e}")))?;

        device
            .wait_for_fences(&[state.fence], true, u64::MAX)
            .map_err(|e| FerricastError::Capture(format!("vk wait_for_fences: {e}")))?;
        device
            .reset_fences(&[state.fence])
            .map_err(|e| FerricastError::Capture(format!("vk reset_fences: {e}")))?;
    }

    Ok(())
}

fn readback(inner: &Inner, staging: &Staging, size: u64) -> Result<Bytes> {
    unsafe {
        let ptr = inner
            .device
            .map_memory(staging.memory, 0, size, vk::MemoryMapFlags::empty())
            .map_err(|e| FerricastError::Capture(format!("vk map_memory: {e}")))?;
        let slice = std::slice::from_raw_parts(ptr as *const u8, size as usize);
        let copy = Bytes::copy_from_slice(slice);
        inner.device.unmap_memory(staging.memory);
        Ok(copy)
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
