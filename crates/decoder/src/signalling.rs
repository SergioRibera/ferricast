//! The colour and HDR signalling of an H.264 / H.265 stream, for the
//! decoders whose backend reports neither (openh264, `rust_h265`):
//! read from the parameter sets and SEI messages, over what the
//! container says.

use ferricast_codec::bitstream::NalSyntax;
use ferricast_codec::bitstream::hdr::HdrTracker;
use ferricast_codec::bitstream::vui::ColourTracker;
use ferricast_core::{ColourDescription, DecoderConfig, HdrMetadata};

use crate::nv12::Container;

/// What one stream signals so far.
#[derive(Debug, Clone, Default)]
pub(crate) struct NalSignalling {
    container: Container,
    colour: ColourTracker,
    hdr: HdrTracker,
}

impl NalSignalling {
    /// Nothing read yet from the stream `config` opens.
    pub(crate) fn of(config: &DecoderConfig) -> Self {
        Self {
            container: Container::of(config),
            colour: ColourTracker::default(),
            hdr: HdrTracker::default(),
        }
    }

    /// Reads the parameter sets and SEI messages of an Annex B access
    /// unit, before its pictures are decoded.
    pub(crate) fn observe(&mut self, access_unit: &[u8], syntax: NalSyntax) {
        self.colour.observe_annex_b(access_unit, syntax);
        self.hdr.observe_annex_b(access_unit, syntax);
    }

    /// The colour the pictures are in.
    pub(crate) fn colour(&self) -> ColourDescription {
        self.colour.describe(self.container.colour)
    }

    /// The static HDR metadata the pictures carry.
    pub(crate) fn hdr(&self) -> Option<HdrMetadata> {
        self.hdr.current_or(self.container.hdr)
    }
}
