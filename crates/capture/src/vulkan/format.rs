//! Conversions between ferricast's `PixelFormat`, Vulkan's
//! `vk::Format` and the byte stride information the encoder needs.

use ash::vk;
use ferricast_core::PixelFormat;

/// Map a public [`PixelFormat`] to the Vulkan format used to import
/// the corresponding dmabuf.
///
/// SPA names byte order from the LSB up (so `BGRA` means `B` is at
/// byte 0). Vulkan names from MSB down (so `B8G8R8A8` matches the
/// same in-memory layout). The mappings below preserve in-memory
/// order, which is what dmabuf importers expect.
pub(super) fn pixel_format_to_vk(p: PixelFormat) -> Option<vk::Format> {
    Some(match p {
        PixelFormat::Bgra => vk::Format::B8G8R8A8_UNORM,
        PixelFormat::Rgba => vk::Format::R8G8B8A8_UNORM,
        // NV12 / I420 are multi-plane and currently routed through
        // the SHM path; we don't import those via Vulkan.
        PixelFormat::Nv12 | PixelFormat::I420 => return None,
    })
}

/// Bytes-per-pixel for the formats we actually import. NV12 / I420
/// would need plane-aware handling — they aren't supported in the
/// Vulkan path.
pub(super) fn bytes_per_pixel(p: PixelFormat) -> Option<u32> {
    Some(match p {
        PixelFormat::Bgra | PixelFormat::Rgba => 4,
        PixelFormat::Nv12 | PixelFormat::I420 => return None,
    })
}
