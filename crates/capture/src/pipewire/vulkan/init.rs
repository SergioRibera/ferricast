//! Vulkan instance / physical device / logical device bring-up.
//!
//! The flow is:
//! 1. Load the Vulkan loader (`ash::Entry::load`).
//! 2. Create an instance with `VK_KHR_get_physical_device_properties2`
//!    and `VK_KHR_external_memory_capabilities` (1.0 fallback; if we
//!    target API 1.1+ those are core).
//! 3. Walk physical devices, pick the first that supports all the
//!    extensions we need and exposes a graphics-or-transfer queue.
//! 4. Create the logical device with the dmabuf-import extensions and
//!    grab the queue + a transient command pool.
//!
//! Anything missing returns `Err`; the caller falls back to SHM.

use std::ffi::CStr;

use ash::vk;
use ferricast_core::{FerricastError, Result};
use tracing::{debug, info, warn};

use super::Inner;

/// Required device-level extensions. Without these we can't import a
/// dmabuf and read it back. Each one is well-supported by Mesa /
/// proprietary drivers from the last several years.
const REQUIRED_DEVICE_EXTS: &[&CStr] = &[
    // Import a dmabuf fd as a `VkDeviceMemory`.
    ash::khr::external_memory_fd::NAME,
    // Tell Vulkan the imported memory is dma-buf shaped.
    ash::ext::external_memory_dma_buf::NAME,
    // Create a `VkImage` with an explicit DRM modifier layout.
    ash::ext::image_drm_format_modifier::NAME,
    // Required by `VK_EXT_external_memory_dma_buf` per the spec.
    ash::khr::external_memory::NAME,
    // Used to query memory & format properties with the `2`
    // structure variant (which the dmabuf flow uses to chain
    // structs).
    ash::khr::get_memory_requirements2::NAME,
    ash::ext::queue_family_foreign::NAME,
];

/// Instance-level extensions. Only required when targeting Vulkan 1.0
/// — we ask for API 1.1, where these are core, but enabling them
/// explicitly does no harm and helps on older loaders.
const INSTANCE_EXTS: &[&CStr] = &[ash::khr::external_memory_capabilities::NAME];

pub(super) fn build() -> Result<Inner> {
    let entry = unsafe {
        ash::Entry::load().map_err(|e| {
            FerricastError::Capture(format!("Vulkan loader missing: {e}"))
        })?
    };

    let instance = create_instance(&entry)?;
    let (physical_device, queue_family_index) = pick_physical_device(&instance)?;
    let device = create_device(&instance, physical_device, queue_family_index)?;

    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    let command_pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family_index)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT),
            None,
        )
    }
    .map_err(|e| FerricastError::Capture(format!("create_command_pool: {e}")))?;

    let drm_modifier =
        ash::ext::image_drm_format_modifier::Device::new(&instance, &device);
    let external_memory_fd =
        ash::khr::external_memory_fd::Device::new(&instance, &device);

    let memory_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    Ok(Inner {
        entry,
        instance,
        physical_device,
        device,
        queue,
        queue_family_index,
        command_pool,
        drm_modifier,
        external_memory_fd,
        memory_props,
    })
}

fn create_instance(entry: &ash::Entry) -> Result<ash::Instance> {
    let app_name = c"ferricast-capture";
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .application_version(0)
        .engine_name(app_name)
        .engine_version(0)
        .api_version(vk::API_VERSION_1_1);

    let ext_ptrs: Vec<*const i8> =
        INSTANCE_EXTS.iter().map(|c| c.as_ptr()).collect();

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&ext_ptrs);

    unsafe { entry.create_instance(&create_info, None) }
        .map_err(|e| FerricastError::Capture(format!("create_instance: {e}")))
}

/// Pick the first physical device that supports the dmabuf-import
/// extensions and has a queue family with `TRANSFER` (or `GRAPHICS`,
/// which implies transfer). Returns `(device, queue_family_index)`.
fn pick_physical_device(
    instance: &ash::Instance,
) -> Result<(vk::PhysicalDevice, u32)> {
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| FerricastError::Capture(format!("enumerate_physical_devices: {e}")))?;

    if devices.is_empty() {
        return Err(FerricastError::Capture(
            "Vulkan: no physical devices".into(),
        ));
    }

    for &dev in &devices {
        let props = unsafe { instance.get_physical_device_properties(dev) };
        let dev_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        if !device_supports_required_extensions(instance, dev) {
            debug!(device = %dev_name, "skipping: missing required extensions");
            continue;
        }

        let queue_family = pick_queue_family(instance, dev);
        let Some(qf) = queue_family else {
            debug!(device = %dev_name, "skipping: no transfer-capable queue");
            continue;
        };

        info!(device = %dev_name, queue_family = qf, "Vulkan ready");
        return Ok((dev, qf));
    }

    Err(FerricastError::Capture(
        "Vulkan: no physical device supports dmabuf import".into(),
    ))
}

fn device_supports_required_extensions(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> bool {
    let exts = match unsafe { instance.enumerate_device_extension_properties(physical_device) } {
        Ok(v) => v,
        Err(e) => {
            warn!(error = %e, "enumerate_device_extension_properties failed");
            return false;
        }
    };

    for required in REQUIRED_DEVICE_EXTS {
        let found = exts.iter().any(|p| {
            // Properties hand back a NUL-terminated array; build a
            // CStr from the pointer to compare against the required
            // name.
            unsafe { CStr::from_ptr(p.extension_name.as_ptr()) == *required }
        });
        if !found {
            debug!(?required, "device missing extension");
            return false;
        }
    }
    true
}

fn pick_queue_family(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Option<u32> {
    let families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    // Prefer a dedicated transfer queue, fall back to graphics
    // (which always supports transfer).
    families
        .iter()
        .enumerate()
        .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::TRANSFER))
        .map(|(i, _)| i as u32)
        .or_else(|| {
            families
                .iter()
                .enumerate()
                .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|(i, _)| i as u32)
        })
}

fn create_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
) -> Result<ash::Device> {
    let priorities = [1.0_f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities);
    let queue_infos = [queue_info];

    let ext_ptrs: Vec<*const i8> =
        REQUIRED_DEVICE_EXTS.iter().map(|c| c.as_ptr()).collect();

    let create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&ext_ptrs);

    unsafe { instance.create_device(physical_device, &create_info, None) }
        .map_err(|e| FerricastError::Capture(format!("create_device: {e}")))
}
