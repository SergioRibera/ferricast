//! Conversions between ferricast's `PixelFormat`, Vulkan's
//! `vk::Format` and the byte stride information the encoder needs.

use ash::vk;
use ferricast_core::PixelFormat;
use pipewire::spa::param::video::VideoFormat;

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
        _ => return None,
    })
}

/// SPA `VideoFormat` → Vulkan format. Used to query GPU-supported
/// modifiers for the formats we'd actually import.
pub(super) fn video_format_to_vk(spa: VideoFormat) -> Option<vk::Format> {
    Some(match spa {
        VideoFormat::BGRA | VideoFormat::BGRx => vk::Format::B8G8R8A8_UNORM,
        VideoFormat::RGBA | VideoFormat::RGBx => vk::Format::R8G8B8A8_UNORM,
        _ => return None,
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

/// Map a SPA `VideoFormat` to the public `PixelFormat`. Used by the
/// PipeWire stream worker once the source has negotiated a format.
pub(crate) fn video_format_to_pixel(spa: VideoFormat) -> Option<PixelFormat> {
    Some(match spa {
        VideoFormat::BGRA | VideoFormat::BGRx => PixelFormat::Bgra,
        VideoFormat::RGBA | VideoFormat::RGBx => PixelFormat::Rgba,
        _ => return None,
    })
}
