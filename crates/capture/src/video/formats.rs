//! The format tiers a capture asks the plugin for.

use ferricast_core::{DmaBufModifiers, PixelFormat};
use istmo_screen_capture::{FormatChoice, FormatRequest, FormatTier, PixelFormat as Format};

/// `DRM_FORMAT_MOD_INVALID`: the implicit layout. Never asked for: the
/// importers (Vulkan, NVENC's EGL) need the explicit one.
const IMPLICIT_MODIFIER: u64 = 0x00ff_ffff_ffff_ffff;

/// The plugin formats per ferricast byte order, preferred first. The
/// X variants come first: scanout buffers are opaque, and consumers
/// ignore the alpha byte either way.
const FORMATS: [(Format, PixelFormat); 4] = [
    (Format::Bgrx8, PixelFormat::Bgra),
    (Format::Bgra8, PixelFormat::Bgra),
    (Format::Rgbx8, PixelFormat::Rgba),
    (Format::Rgba8, PixelFormat::Rgba),
];

/// The DMA-BUF layouts the Vulkan importer reads back, per byte order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Readable {
    pub(crate) bgra: Vec<u64>,
    pub(crate) rgba: Vec<u64>,
}

impl Readable {
    fn get(&self, format: PixelFormat) -> &[u64] {
        match format {
            PixelFormat::Bgra => &self.bgra,
            PixelFormat::Rgba => &self.rgba,
            PixelFormat::Nv12 | PixelFormat::I420 | PixelFormat::P010 => &[],
        }
    }
}

/// What the frames' consumer and the readback importer take, turned
/// into format tiers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Formats<'a> {
    /// What the GPU consumer imports without a copy.
    pub(crate) consumer: &'a DmaBufModifiers,
    /// What the importer reads back; `None` without an importer.
    pub(crate) readable: Option<&'a Readable>,
}

impl Formats<'_> {
    /// Tiers, best first:
    ///
    /// 1. DMA-BUFs in a layout the consumer imports *and* the importer
    ///    reads back (so a frame the consumer refuses still reaches the
    ///    CPU);
    /// 2. DMA-BUFs in any layout the importer reads back;
    /// 3. CPU frames in shared memory (the producer's own copy).
    ///
    /// Without an importer only the CPU tier is asked for: a DMA-BUF
    /// nothing here can read back would strand CPU consumers. The
    /// producer intersects each list with what it can allocate.
    pub(crate) fn request(self) -> FormatRequest {
        let Some(readable) = self.readable else {
            return FormatRequest::cpu();
        };
        let mut preferred = Vec::new();
        let mut fallback = Vec::new();
        for (format, order) in FORMATS {
            let readable: Vec<u64> = readable
                .get(order)
                .iter()
                .copied()
                .filter(|&modifier| modifier != IMPLICIT_MODIFIER)
                .collect();
            let consumer = self.consumer.get(order);
            let both: Vec<u64> = readable
                .iter()
                .copied()
                .filter(|modifier| consumer.contains(modifier))
                .collect();
            if both != readable && !readable.is_empty() {
                fallback.push(FormatChoice::dmabuf(format, readable));
            }
            if !both.is_empty() {
                preferred.push(FormatChoice::dmabuf(format, both));
            }
        }
        // An empty modifier list means "any layout" to the plugin, so a
        // tier only ever holds choices with explicit layouts.
        let tiers = [preferred, fallback]
            .into_iter()
            .filter(|choices| !choices.is_empty())
            .map(FormatTier::new)
            .chain(std::iter::once(FormatTier::cpu_rgb()))
            .collect();
        FormatRequest { tiers }
    }
}

/// The ferricast byte order of a plugin format, for the 8-bit RGB
/// formats this crate asks for.
pub(crate) const fn byte_order(format: Format) -> Option<PixelFormat> {
    match format {
        Format::Bgra8 | Format::Bgrx8 => Some(PixelFormat::Bgra),
        Format::Rgba8 | Format::Rgbx8 => Some(PixelFormat::Rgba),
        Format::Xr30
        | Format::Xb30
        | Format::Ar30
        | Format::Rgba16f
        | Format::Nv12
        | Format::P010 => None,
    }
}

#[cfg(test)]
mod tests {
    use istmo_screen_capture::{BufferMemory, CopyStage};

    use super::*;

    const TILED: u64 = 0x0300_0000_0060_6014;
    const OTHER_TILED: u64 = 0x0300_0000_0060_6015;
    const LINEAR: u64 = 0;

    fn readable(bgra: &[u64]) -> Readable {
        Readable {
            bgra: bgra.to_vec(),
            rgba: Vec::new(),
        }
    }

    fn modifiers(tier: &FormatTier, format: Format) -> Option<Vec<u64>> {
        tier.choices
            .iter()
            .find(|choice| choice.format == format)
            .map(|choice| choice.modifiers.clone())
    }

    #[test]
    fn consumer_modifiers_are_offered_first() {
        let consumer = DmaBufModifiers::default().with(PixelFormat::Bgra, vec![TILED]);
        let readable = readable(&[LINEAR, TILED, OTHER_TILED]);
        let request = Formats {
            consumer: &consumer,
            readable: Some(&readable),
        }
        .request();
        assert_eq!(request.tiers.len(), 3);
        assert_eq!(
            modifiers(&request.tiers[0], Format::Bgrx8),
            Some(vec![TILED])
        );
        assert_eq!(
            modifiers(&request.tiers[1], Format::Bgrx8),
            Some(vec![LINEAR, TILED, OTHER_TILED])
        );
        assert!(
            request.tiers[..2]
                .iter()
                .all(|tier| tier.allows_copy.is_empty())
        );
        assert_eq!(request.tiers[2], FormatTier::cpu_rgb());
    }

    #[test]
    fn layout_prefers_what_every_side_takes() {
        // The consumer's other layouts are not readable here: offering
        // them would strand a refused frame on the GPU.
        let consumer = DmaBufModifiers::default().with(PixelFormat::Bgra, vec![OTHER_TILED, TILED]);
        let readable = readable(&[LINEAR, TILED]);
        let request = Formats {
            consumer: &consumer,
            readable: Some(&readable),
        }
        .request();
        assert_eq!(
            modifiers(&request.tiers[0], Format::Bgra8),
            Some(vec![TILED])
        );
        assert!(
            request
                .tiers
                .iter()
                .flat_map(|tier| &tier.choices)
                .all(|choice| !choice.modifiers.contains(&OTHER_TILED))
        );
    }

    #[test]
    fn without_a_consumer_preference_every_readable_layout_is_one_tier() {
        let readable = readable(&[LINEAR, TILED]);
        let request = Formats {
            consumer: &DmaBufModifiers::default(),
            readable: Some(&readable),
        }
        .request();
        assert_eq!(request.tiers.len(), 2);
        assert_eq!(
            modifiers(&request.tiers[0], Format::Bgrx8),
            Some(vec![LINEAR, TILED])
        );
    }

    #[test]
    fn layout_without_importer_is_cpu_only() {
        let consumer = DmaBufModifiers::default().with(PixelFormat::Bgra, vec![TILED]);
        let request = Formats {
            consumer: &consumer,
            readable: None,
        }
        .request();
        assert_eq!(request, FormatRequest::cpu());
        assert!(
            request.tiers[0]
                .choices
                .iter()
                .all(|choice| choice.memory == BufferMemory::Cpu)
        );
        assert_eq!(request.tiers[0].allows_copy, [CopyStage::ShmTransport]);
    }

    #[test]
    fn the_implicit_modifier_is_never_asked_for() {
        let consumer = DmaBufModifiers::default().with(PixelFormat::Bgra, vec![IMPLICIT_MODIFIER]);
        let readable = readable(&[IMPLICIT_MODIFIER]);
        let request = Formats {
            consumer: &consumer,
            readable: Some(&readable),
        }
        .request();
        assert_eq!(request, FormatRequest::cpu());
    }

    #[test]
    fn no_tier_holds_an_any_layout_choice() {
        let readable = Readable {
            bgra: vec![TILED],
            rgba: Vec::new(),
        };
        let request = Formats {
            consumer: &DmaBufModifiers::default(),
            readable: Some(&readable),
        }
        .request();
        assert!(
            request
                .tiers
                .iter()
                .flat_map(|tier| &tier.choices)
                .filter(|choice| choice.memory == BufferMemory::DmaBuf)
                .all(|choice| !choice.modifiers.is_empty())
        );
    }
}
