//! DmaBuf import + readback with cross-frame resource caching.
//!
//! Hot-path layout:
//!
//! 1. Look up the dmabuf in [`State::imports`] (keyed by the first
//!    plane's fd). If it's there with matching dimensions / modifier
//!    / plane count, reuse the `VkImage` — the expensive
//!    `vkAllocateMemory(import)` + `vkBindImageMemory` already paid
//!    for it on the first frame.
//! 2. If not cached: dup the fds (Vulkan takes ownership of the dups),
//!    create a fresh `VkImage` with explicit per-plane modifier
//!    layouts, import each plane's memory, bind, store in the cache.
//!    Multi-plane modifiers (e.g. AMD GFX9+ DCC retile) use
//!    `VK_IMAGE_CREATE_DISJOINT_BIT` and bind one allocation per
//!    plane via `VkBindImagePlaneMemoryInfo`. Single-plane modifiers
//!    use the simpler non-disjoint binding.
//! 3. Ensure the persistent staging buffer is at least
//!    `width * height * bpp` bytes; recreate larger if needed.
//! 4. Reset the cached command buffer + fence, record
//!    `vkCmdCopyImageToBuffer`, submit and wait.
//! 5. Map the staging memory, `Bytes::copy_from_slice` to a packed
//!    linear copy, unmap.

use std::os::fd::RawFd;

use ash::vk;
use bytes::Bytes;
use ferricast_core::{DmaBufPlane, FerricastError, PixelFormat, Result};
use tracing::{trace, warn};

use super::{format as fmt, ImportedImage, Inner, Staging, State};

/// Memory plane aspect bits for `VkBindImagePlaneMemoryInfo`. The
/// modifier extension defines four memory-plane aspects; we index
/// them by plane number.
const MEMORY_PLANE_ASPECTS: [vk::ImageAspectFlags; 4] = [
    vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
    vk::ImageAspectFlags::MEMORY_PLANE_1_EXT,
    vk::ImageAspectFlags::MEMORY_PLANE_2_EXT,
    vk::ImageAspectFlags::MEMORY_PLANE_3_EXT,
];

pub(super) fn run(
    inner: &Inner,
    state: &mut State,
    planes: &[DmaBufPlane],
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
    if planes.is_empty() {
        return Err(FerricastError::Capture("vulkan: zero planes".into()));
    }
    if planes.len() > MEMORY_PLANE_ASPECTS.len() {
        return Err(FerricastError::Capture(format!(
            "vulkan: dmabuf has {} planes, only {} supported",
            planes.len(),
            MEMORY_PLANE_ASPECTS.len()
        )));
    }

    let image = ensure_image(inner, state, planes, modifier, width, height, vk_format)?;
    ensure_staging(inner, state, staging_size)?;
    let staging = state.staging.as_ref().expect("ensure_staging populated it");
    record_and_submit(inner, state, image, staging, width, height, staging_size)?;
    let bytes = readback(inner, staging, staging_size)?;

    trace!(width, height, len = bytes.len(), "vulkan readback ok");
    Ok(bytes)
}

/// Look up the cached `VkImage` for the given dmabuf, or import a
/// fresh one and stash it. Cache key is the first plane's fd —
/// PipeWire reuses fds within a negotiation cycle and never aliases
/// distinct buffers under the same first-plane fd.
fn ensure_image(
    inner: &Inner,
    state: &mut State,
    planes: &[DmaBufPlane],
    modifier: u64,
    width: u32,
    height: u32,
    vk_format: vk::Format,
) -> Result<vk::Image> {
    let key = planes[0].fd;

    if let Some(existing) = state.imports.get(&key) {
        if existing.width == width
            && existing.height == height
            && existing.format == vk_format
            && existing.modifier == modifier
            && existing.plane_count as usize == planes.len()
        {
            return Ok(existing.image);
        }
        // Stale — same fd numerical, but the buffer behind it has
        // changed shape. Destroy and re-import.
        unsafe {
            let _ = inner.device.device_wait_idle();
            inner.device.destroy_image(existing.image, None);
            for mem in &existing.memories {
                inner.device.free_memory(*mem, None);
            }
        }
        state.imports.remove(&key);
    }

    let imported = import_image(inner, planes, modifier, width, height, vk_format)?;
    state.imports.insert(key, imported);
    Ok(state.imports.get(&key).unwrap().image)
}

/// Build the `VkImage` and import every plane's memory.
fn import_image(
    inner: &Inner,
    planes: &[DmaBufPlane],
    modifier: u64,
    width: u32,
    height: u32,
    vk_format: vk::Format,
) -> Result<ImportedImage> {
    let plane_count = planes.len();
    let disjoint = plane_count > 1;

    // Dup every fd up front. If anything below fails we close them
    // via `OwnedFdGuard::Drop`; on success we mark them handed off.
    let mut fd_guards: Vec<OwnedFdGuard> = Vec::with_capacity(plane_count);
    for p in planes {
        let dup_fd = unsafe { libc::dup(p.fd) };
        if dup_fd < 0 {
            return Err(FerricastError::Capture(format!(
                "dup(dmabuf fd): {}",
                std::io::Error::last_os_error()
            )));
        }
        fd_guards.push(OwnedFdGuard::new(dup_fd));
    }

    let plane_layouts: Vec<vk::SubresourceLayout> = planes
        .iter()
        .map(|p| {
            vk::SubresourceLayout::default()
                .offset(p.offset as u64)
                .row_pitch(p.stride as u64)
        })
        .collect();

    let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
        .drm_format_modifier(modifier)
        .plane_layouts(&plane_layouts);

    let mut external = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

    let mut flags = vk::ImageCreateFlags::empty();
    if disjoint {
        flags |= vk::ImageCreateFlags::DISJOINT;
    }

    let image_info = vk::ImageCreateInfo::default()
        .flags(flags)
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

    let image = unsafe { inner.device.create_image(&image_info, None) }
        .map_err(|e| FerricastError::Capture(format!("vk create_image: {e}")))?;

    let memories = match allocate_and_bind(inner, image, &mut fd_guards, disjoint) {
        Ok(m) => m,
        Err(e) => {
            unsafe { inner.device.destroy_image(image, None) };
            return Err(e);
        }
    };

    Ok(ImportedImage {
        image,
        memories,
        width,
        height,
        format: vk_format,
        modifier,
        plane_count: plane_count as u32,
    })
}

/// Allocate one `VkDeviceMemory` per plane (importing the dup'd fd)
/// and bind it to the image. Single-plane uses the classic
/// `vkBindImageMemory`; multi-plane uses `vkBindImageMemory2` with
/// `VkBindImagePlaneMemoryInfo` per plane and the image must have
/// been created with `VK_IMAGE_CREATE_DISJOINT_BIT`.
fn allocate_and_bind(
    inner: &Inner,
    image: vk::Image,
    fd_guards: &mut [OwnedFdGuard],
    disjoint: bool,
) -> Result<Vec<vk::DeviceMemory>> {
    let plane_count = fd_guards.len();
    let mut memories: Vec<vk::DeviceMemory> = Vec::with_capacity(plane_count);

    // Free any successful allocations so far on partial failure.
    let cleanup = |inner: &Inner, mems: &[vk::DeviceMemory]| unsafe {
        for m in mems {
            inner.device.free_memory(*m, None);
        }
    };

    for (i, fd_guard) in fd_guards.iter_mut().enumerate() {
        // Get this plane's memory requirements. The disjoint case
        // needs `VkImagePlaneMemoryRequirementsInfo` chained in to
        // tell Vulkan which plane we're asking about.
        let mut plane_info = vk::ImagePlaneMemoryRequirementsInfo::default()
            .plane_aspect(MEMORY_PLANE_ASPECTS[i]);
        let mut info = vk::ImageMemoryRequirementsInfo2::default().image(image);
        if disjoint {
            info = info.push_next(&mut plane_info);
        }

        let mut reqs = vk::MemoryRequirements2::default();
        unsafe {
            inner
                .device
                .get_image_memory_requirements2(&info, &mut reqs);
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
            cleanup(inner, &memories);
            return Err(FerricastError::Capture(format!(
                "vk get_memory_fd_properties (plane {i}): {e}"
            )));
        }

        let memory_type_index = pick_memory_type(
            &inner.memory_props,
            reqs.memory_requirements.memory_type_bits & fd_props.memory_type_bits,
            vk::MemoryPropertyFlags::empty(),
        );
        let Some(memory_type_index) = memory_type_index else {
            cleanup(inner, &memories);
            return Err(FerricastError::Capture(format!(
                "vk: no memory type matches dmabuf import constraints (plane {i})"
            )));
        };

        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(fd_guard.raw());
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.memory_requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut dedicated)
            .push_next(&mut import_info);

        let memory = match unsafe { inner.device.allocate_memory(&alloc_info, None) } {
            Ok(m) => m,
            Err(e) => {
                cleanup(inner, &memories);
                return Err(FerricastError::Capture(format!(
                    "vk allocate_memory(import) (plane {i}): {e}"
                )));
            }
        };
        fd_guard.forget(); // Vulkan now owns the dup'd fd.
        memories.push(memory);
    }

    // Bind. Single plane: classic `vkBindImageMemory`. Multi-plane:
    // `vkBindImageMemory2` with one `VkBindImagePlaneMemoryInfo` per
    // plane chained into the corresponding `VkBindImageMemoryInfo`.
    if disjoint {
        // Two parallel storage Vecs that outlive the bind call. Each
        // `BindImageMemoryInfo` carries a raw `p_next` pointer into
        // `plane_infos`; the borrow checker can't track this through
        // FFI so we keep both Vecs in scope until after the call.
        let mut plane_infos: Vec<vk::BindImagePlaneMemoryInfo<'_>> = (0..plane_count)
            .map(|i| vk::BindImagePlaneMemoryInfo::default().plane_aspect(MEMORY_PLANE_ASPECTS[i]))
            .collect();

        let binds: Vec<vk::BindImageMemoryInfo<'_>> = memories
            .iter()
            .zip(plane_infos.iter_mut())
            .map(|(mem, plane_info)| {
                vk::BindImageMemoryInfo::default()
                    .image(image)
                    .memory(*mem)
                    .memory_offset(0)
                    .push_next(plane_info)
            })
            .collect();

        let res = unsafe { inner.device.bind_image_memory2(&binds) };
        // Keep `binds` (and the `plane_infos` it borrows from) alive
        // across the FFI call.
        drop(binds);
        drop(plane_infos);
        if let Err(e) = res {
            cleanup(inner, &memories);
            return Err(FerricastError::Capture(format!(
                "vk bind_image_memory2 (disjoint): {e}"
            )));
        }
    } else if let Err(e) = unsafe { inner.device.bind_image_memory(image, memories[0], 0) } {
        cleanup(inner, &memories);
        return Err(FerricastError::Capture(format!(
            "vk bind_image_memory: {e}"
        )));
    }

    Ok(memories)
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
