//! Enumerate the DRM modifiers a Vulkan physical device can
//! produce/consume for a given format.
//!
//! The query is the standard two-call pattern of Vulkan's
//! `vkGetPhysicalDeviceFormatProperties2` chained with
//! `VkDrmFormatModifierPropertiesListEXT`:
//!
//! 1. First call with `count = 0`, `null_mut()` — Vulkan fills in the
//!    actual count.
//! 2. Allocate a `Vec` of that size, point the chained list at it,
//!    call again — Vulkan fills in the modifier metadata.
//!
//! We then filter to modifiers whose `drm_format_modifier_tiling_features`
//! advertise `TRANSFER_SRC` — that's the bit we need to blit them into
//! a CPU-readable staging buffer in [`super::import`].

use ash::vk;
use ferricast_core::{FerricastError, Result};
use tracing::trace;

use super::Inner;

pub(super) fn query(inner: &Inner, format: vk::Format) -> Result<Vec<ModifierCaps>> {
    // Pass 1: ask Vulkan for the count. The chained struct holds a
    // `&mut` to `list_pass1`, so we run it in its own scope so that
    // borrow ends before we read `drm_format_modifier_count`.
    let mut list_pass1 = vk::DrmFormatModifierPropertiesListEXT::default();
    {
        let mut props_pass1 =
            vk::FormatProperties2::default().push_next(&mut list_pass1);
        unsafe {
            inner.instance.get_physical_device_format_properties2(
                inner.physical_device,
                format,
                &mut props_pass1,
            );
        }
    }
    let count = list_pass1.drm_format_modifier_count as usize;
    if count == 0 {
        return Ok(Vec::new());
    }

    // Pass 2: allocate the array, point the chained list at it, call
    // again so Vulkan fills the metadata in.
    let mut storage: Vec<vk::DrmFormatModifierPropertiesEXT> =
        vec![vk::DrmFormatModifierPropertiesEXT::default(); count];

    {
        let mut list_pass2 = vk::DrmFormatModifierPropertiesListEXT::default()
            .drm_format_modifier_properties(&mut storage);
        let mut props_pass2 =
            vk::FormatProperties2::default().push_next(&mut list_pass2);
        unsafe {
            inner.instance.get_physical_device_format_properties2(
                inner.physical_device,
                format,
                &mut props_pass2,
            );
        }
    }

    if storage.is_empty() {
        return Err(FerricastError::Capture(
            "DrmFormatModifierPropertiesListEXT returned no entries on second pass".into(),
        ));
    }

    // We only keep modifiers that advertise `TRANSFER_SRC` — anything
    // storage-only / sampled-only would still arrive in the dmabuf
    // path but our staging-copy would fail. Multi-plane modifiers
    // (e.g. AMD GFX9+ DCC retile) are fine: the import path uses
    // `VK_IMAGE_CREATE_DISJOINT_BIT` and binds each plane separately.
    let needed = vk::FormatFeatureFlags::TRANSFER_SRC;
    let modifiers: Vec<ModifierCaps> = storage
        .iter()
        .filter(|p| p.drm_format_modifier_tiling_features.contains(needed))
        .map(|p| ModifierCaps {
            modifier: p.drm_format_modifier,
            plane_count: p.drm_format_modifier_plane_count,
        })
        .collect();

    trace!(
        ?format,
        accepted = modifiers.len(),
        offered = storage.len(),
        "modifier query result"
    );

    Ok(modifiers)
}

/// A modifier the GPU advertises plus the number of memory planes it
/// requires. We pass plane count through so the import path can size
/// its `pPlaneLayouts` array and choose between the single-allocation
/// and disjoint binding paths without re-querying.
#[derive(Debug, Clone, Copy)]
pub struct ModifierCaps {
    pub modifier: u64,
    pub plane_count: u32,
}
