//! Vulkan instance / physical device / logical device bring-up.
//!
//! The flow is:
//! 1. Load the Vulkan loader (`ash::Entry::load`).
//! 2. Create an instance with `VK_KHR_get_physical_device_properties2`
//!    and `VK_KHR_external_memory_capabilities` (1.0 fallback; if we
//!    target API 1.1+ those are core).
//! 3. Walk physical devices that support all the extensions we need
//!    and expose a transfer-capable queue, and pick the best-ranked
//!    one (see [`DeviceRank`]): the device whose DRM node matches
//!    `FERRICAST_DRM_RENDER_NODE` first, then integrated GPUs (on
//!    hybrid laptops the compositor almost always renders on the
//!    iGPU, and its dmabufs/modifiers belong to that device), then
//!    the rest.
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
    ash::khr::image_format_list::NAME,
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

/// Khronos validation layer. Opt-in only: it is rarely installed on
/// user machines (requesting a missing layer makes `vkCreateInstance`
/// fail with `ERROR_LAYER_NOT_PRESENT`, killing the dmabuf path) and
/// costs a lot of CPU per call even where it is.
const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Set to `1` / `true` to request [`VALIDATION_LAYER`].
const VALIDATION_ENV: &str = "FERRICAST_VK_VALIDATION";

fn validation_requested() -> bool {
    std::env::var(VALIDATION_ENV).is_ok_and(|v| env_flag_enabled(&v))
}

fn env_flag_enabled(value: &str) -> bool {
    let value = value.trim();
    value == "1" || value.eq_ignore_ascii_case("true")
}

fn validation_layer_available(entry: &ash::Entry) -> bool {
    // SAFETY: plain loader query; no handles involved.
    match unsafe { entry.enumerate_instance_layer_properties() } {
        Ok(layers) => layers
            .iter()
            .any(|l| l.layer_name_as_c_str().is_ok_and(|n| n == VALIDATION_LAYER)),
        Err(e) => {
            warn!(error = %e, "enumerate_instance_layer_properties failed");
            false
        }
    }
}

pub(super) fn build() -> Result<Inner> {
    let entry = unsafe {
        ash::Entry::load()
            .map_err(|e| FerricastError::Capture(format!("Vulkan loader missing: {e}")))?
    };

    let instance = create_instance(&entry)?;
    let (physical_device, queue_family_index) = pick_physical_device(&instance)?;
    let device = create_device(&instance, physical_device, queue_family_index)?;

    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    let command_pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family_index)
                // TRANSIENT: each command buffer is short-lived
                // (record once, submit once, reset on next use).
                // RESET_COMMAND_BUFFER: required because
                // `record_and_submit` calls `vkResetCommandBuffer`
                // per frame; without this flag the call is
                // undefined behaviour per the Vulkan spec, and on
                // NVIDIA's proprietary driver it manifests as
                // multi-second GPU stalls (validation layer
                // VUID-vkResetCommandBuffer-commandBuffer-00046).
                .flags(
                    vk::CommandPoolCreateFlags::TRANSIENT
                        | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                ),
            None,
        )
    }
    .map_err(|e| FerricastError::Capture(format!("create_command_pool: {e}")))?;

    let drm_modifier = ash::ext::image_drm_format_modifier::Device::new(&instance, &device);
    let external_memory_fd = ash::khr::external_memory_fd::Device::new(&instance, &device);

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

    let ext_ptrs: Vec<*const i8> = INSTANCE_EXTS.iter().map(|c| c.as_ptr()).collect();

    let layer_names_raw: Vec<*const i8> = if validation_requested() {
        if validation_layer_available(entry) {
            info!("enabling {VALIDATION_LAYER:?} ({VALIDATION_ENV}=1)");
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            warn!(
                "{VALIDATION_ENV}=1 but {VALIDATION_LAYER:?} is not installed; continuing without it"
            );
            Vec::new()
        }
    } else {
        Vec::new()
    };

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&ext_ptrs)
        .enabled_layer_names(&layer_names_raw);

    unsafe { entry.create_instance(&create_info, None) }
        .map_err(|e| FerricastError::Capture(format!("create_instance: {e}")))
}

/// Path of the DRM node (e.g. `/dev/dri/renderD128`) the compositor
/// renders on. Optional override for hosts where the integrated-GPU
/// heuristic picks the wrong device.
const DRM_NODE_ENV: &str = "FERRICAST_DRM_RENDER_NODE";

/// Linux `dev_t` split into major / minor, as reported by
/// `VK_EXT_physical_device_drm` and by `stat(2)` on a DRM node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DrmNode {
    major: u64,
    minor: u64,
}

impl From<u64> for DrmNode {
    /// glibc / musl `major()` / `minor()` encoding of a 64-bit `dev_t`.
    fn from(dev: u64) -> Self {
        Self {
            major: ((dev >> 8) & 0xfff) | ((dev >> 32) & 0xffff_f000),
            minor: (dev & 0xff) | ((dev >> 12) & 0xffff_ff00),
        }
    }
}

/// Selection preference among usable devices; higher wins. Field
/// order is the comparison order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct DeviceRank {
    /// The device owns the DRM node named by [`DRM_NODE_ENV`].
    matches_drm_node: bool,
    /// Preference by device type, see [`DeviceRank::type_score`].
    type_score: u8,
}

impl DeviceRank {
    fn new(device_type: vk::PhysicalDeviceType, matches_drm_node: bool) -> Self {
        Self {
            matches_drm_node,
            type_score: Self::type_score(device_type),
        }
    }

    /// Integrated GPUs first: on hybrid laptops the compositor (and so
    /// the dmabufs and modifiers it hands us) lives on the iGPU, and
    /// importing them on the dGPU means a cross-device import with
    /// mismatched modifiers. Software rasterisers last.
    fn type_score(device_type: vk::PhysicalDeviceType) -> u8 {
        match device_type {
            vk::PhysicalDeviceType::INTEGRATED_GPU => 4,
            vk::PhysicalDeviceType::DISCRETE_GPU => 3,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
            vk::PhysicalDeviceType::CPU => 0,
            // OTHER and any future type.
            _ => 1,
        }
    }
}

fn preferred_drm_node() -> Option<DrmNode> {
    use std::os::unix::fs::MetadataExt;

    let path = std::env::var_os(DRM_NODE_ENV)?;
    match std::fs::metadata(&path) {
        Ok(meta) => Some(DrmNode::from(meta.rdev())),
        Err(e) => {
            warn!(path = ?path, error = %e, "{DRM_NODE_ENV}: can't stat DRM node, ignoring");
            None
        }
    }
}

/// Primary and render DRM nodes of `dev`, when the driver exposes
/// `VK_EXT_physical_device_drm` (and Vulkan 1.1 for the `2` query).
fn device_drm_nodes(
    instance: &ash::Instance,
    dev: vk::PhysicalDevice,
    props: &vk::PhysicalDeviceProperties,
    exts: &[vk::ExtensionProperties],
) -> Vec<DrmNode> {
    if props.api_version < vk::API_VERSION_1_1
        || !has_extension(exts, ash::ext::physical_device_drm::NAME)
    {
        return Vec::new();
    }
    let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut drm);
    // SAFETY: `dev` came from this instance, the device reports
    // Vulkan >= 1.1 (core `vkGetPhysicalDeviceProperties2`), and the
    // chained struct is the extension's documented output struct.
    unsafe { instance.get_physical_device_properties2(dev, &mut props2) };
    let node = |major: i64, minor: i64| {
        Some(DrmNode {
            major: u64::try_from(major).ok()?,
            minor: u64::try_from(minor).ok()?,
        })
    };
    [
        (drm.has_primary == vk::TRUE).then(|| node(drm.primary_major, drm.primary_minor)),
        (drm.has_render == vk::TRUE).then(|| node(drm.render_major, drm.render_minor)),
    ]
    .into_iter()
    .flatten()
    .flatten()
    .collect()
}

/// Pick the best-ranked physical device (see [`DeviceRank`]) that
/// supports the dmabuf-import extensions and has a queue family with
/// `TRANSFER` (or `GRAPHICS`, which implies transfer). Ties keep
/// enumeration order. Returns `(device, queue_family_index)`.
fn pick_physical_device(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32)> {
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| FerricastError::Capture(format!("enumerate_physical_devices: {e}")))?;

    if devices.is_empty() {
        return Err(FerricastError::Capture(
            "Vulkan: no physical devices".into(),
        ));
    }

    let preferred = preferred_drm_node();
    let mut best: Option<(DeviceRank, vk::PhysicalDevice, u32, String)> = None;

    for &dev in &devices {
        let props = unsafe { instance.get_physical_device_properties(dev) };
        let dev_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let exts = match unsafe { instance.enumerate_device_extension_properties(dev) } {
            Ok(v) => v,
            Err(e) => {
                warn!(device = %dev_name, error = %e, "enumerate_device_extension_properties failed");
                continue;
            }
        };
        if let Some(missing) = REQUIRED_DEVICE_EXTS
            .iter()
            .find(|required| !has_extension(&exts, required))
        {
            debug!(device = %dev_name, ?missing, "skipping: missing required extension");
            continue;
        }

        let queue_family = pick_queue_family(instance, dev);
        let Some(qf) = queue_family else {
            debug!(device = %dev_name, "skipping: no transfer-capable queue");
            continue;
        };

        let matches_drm_node = preferred
            .is_some_and(|want| device_drm_nodes(instance, dev, &props, &exts).contains(&want));
        let rank = DeviceRank::new(props.device_type, matches_drm_node);
        debug!(device = %dev_name, device_type = ?props.device_type, ?rank, "usable Vulkan device");
        if best.as_ref().is_none_or(|(b, ..)| rank > *b) {
            best = Some((rank, dev, qf, dev_name));
        }
    }

    let Some((rank, dev, qf, dev_name)) = best else {
        return Err(FerricastError::Capture(
            "Vulkan: no physical device supports dmabuf import".into(),
        ));
    };
    if preferred.is_some() && !rank.matches_drm_node {
        warn!("{DRM_NODE_ENV} matched no usable Vulkan device; using the type heuristic");
    }
    info!(device = %dev_name, queue_family = qf, ?rank, "Vulkan ready");
    Ok((dev, qf))
}

fn has_extension(exts: &[vk::ExtensionProperties], name: &CStr) -> bool {
    exts.iter()
        .any(|p| p.extension_name_as_c_str().is_ok_and(|n| n == name))
}

fn pick_queue_family(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Option<u32> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

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

    let ext_ptrs: Vec<*const i8> = REQUIRED_DEVICE_EXTS.iter().map(|c| c.as_ptr()).collect();

    let create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&ext_ptrs);

    unsafe { instance.create_device(physical_device, &create_info, None) }
        .map_err(|e| FerricastError::Capture(format!("create_device: {e}")))
}

#[cfg(test)]
mod tests {
    use super::{DeviceRank, DrmNode, env_flag_enabled};
    use ash::vk::PhysicalDeviceType as T;

    #[test]
    fn drm_node_decodes_linux_dev_t() {
        // renderD128 = char 226:128.
        assert_eq!(
            DrmNode::from(0xe280),
            DrmNode {
                major: 226,
                minor: 128
            }
        );
        // Large major/minor use the high bits (glibc `makedev`).
        let (major, minor) = (0x1_2345_u64, 0x6_789a_u64);
        let dev = ((major & 0xffff_f000) << 32)
            | ((major & 0xfff) << 8)
            | ((minor & 0xffff_ff00) << 12)
            | (minor & 0xff);
        assert_eq!(DrmNode::from(dev), DrmNode { major, minor });
    }

    #[test]
    fn device_rank_prefers_drm_match_then_integrated() {
        let igpu = DeviceRank::new(T::INTEGRATED_GPU, false);
        let dgpu = DeviceRank::new(T::DISCRETE_GPU, false);
        let dgpu_match = DeviceRank::new(T::DISCRETE_GPU, true);
        let cpu = DeviceRank::new(T::CPU, false);
        assert!(igpu > dgpu);
        assert!(dgpu > cpu);
        assert!(dgpu_match > igpu);
    }

    #[test]
    fn validation_env_flag_parsing() {
        assert!(env_flag_enabled("1"));
        assert!(env_flag_enabled(" TRUE "));
        assert!(!env_flag_enabled("0"));
        assert!(!env_flag_enabled(""));
        assert!(!env_flag_enabled("yes please"));
    }
}
