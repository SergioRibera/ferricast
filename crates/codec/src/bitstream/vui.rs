//! The colour signalling in the VUI of an H.264 / H.265 SPS, for
//! decoders whose backend does not report it (openh264, NVDEC, the
//! pure-Rust HEVC decoder): without it a PQ or HLG stream in an Annex B
//! transport (MPEG-TS, RTP) would be shown with an SDR transfer.

use ferricast_core::{ChromaLocation, ColourDescription, ColourRange};

use super::bits::Bits;
use super::{NalSyntax, h264, hevc, split_annex_b};

/// The colour fields of a VUI: the video signal type (range, primaries,
/// transfer, matrix) and the chroma sample location, each when present.
/// An SPS without a VUI has neither.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct VuiColour {
    signal: Option<ColourDescription>,
    location: Option<ChromaLocation>,
}

impl VuiColour {
    /// The head of `vui_parameters()`, up to the chroma sample location:
    /// the same fields in H.264 (§E.1.1) and H.265 (§E.2.1).
    pub(super) fn read(bits: &mut Bits<'_>) -> Option<Self> {
        /// `aspect_ratio_idc` that spells out the sample aspect ratio.
        const EXTENDED_SAR: u32 = 255;
        /// `colour_primaries` / `transfer_characteristics` /
        /// `matrix_coeffs` of a stream that signals none.
        const UNSPECIFIED: u8 = 2;

        if bits.flag()? && bits.read(8)? == EXTENDED_SAR {
            bits.skip(32)?; // sar_width, sar_height
        }
        if bits.flag()? {
            bits.skip(1)?; // overscan_appropriate_flag
        }
        let signal = if bits.flag()? {
            // video_format.
            bits.skip(3)?;
            let full_range = bits.flag()?;
            let (primaries, transfer, matrix) = if bits.flag()? {
                let mut code = || u8::try_from(bits.read(8)?).ok();
                (code()?, code()?, code()?)
            } else {
                (UNSPECIFIED, UNSPECIFIED, UNSPECIFIED)
            };
            Some(ColourDescription::from_codes(
                primaries, transfer, matrix, full_range,
            ))
        } else {
            None
        };
        let location = if bits.flag()? {
            let top = bits.ue()?;
            bits.ue()?; // chroma_sample_loc_type_bottom_field
            u8::try_from(top)
                .ok()
                .and_then(|code| ChromaLocation::try_from(code).ok())
        } else {
            None
        };
        Some(Self { signal, location })
    }
}

/// The colour signalling of an H.264 / H.265 stream. Feed it every
/// access unit: each SPS replaces what the one before it said.
#[derive(Debug, Clone, Copy, Default)]
pub struct ColourTracker {
    current: VuiColour,
}

impl ColourTracker {
    /// Records the last readable SPS of an Annex B access unit.
    pub fn observe_annex_b(&mut self, access_unit: &[u8], syntax: NalSyntax) {
        let read = match syntax {
            NalSyntax::H264 => h264::sps_colour,
            NalSyntax::H265 => hevc::sps_colour,
        };
        if let Some(colour) = split_annex_b(access_unit).into_iter().rev().find_map(read) {
            self.current = colour;
        }
    }

    /// What the stream signals over `container` (the container's
    /// colour), field group by field group: the bitstream describes the
    /// samples themselves.
    #[must_use]
    pub fn current_or(self, container: Option<ColourDescription>) -> Option<ColourDescription> {
        let VuiColour { signal, location } = self.current;
        match (signal.or(container), location) {
            (Some(colour), Some(location)) => Some(colour.with_chroma_location(location)),
            (None, Some(location)) => Some(
                ColourDescription::unspecified(ColourRange::Limited).with_chroma_location(location),
            ),
            (base, None) => base,
        }
    }

    /// [`Self::current_or`], unspecified (limited range) when neither
    /// the stream nor the container says.
    #[must_use]
    pub fn describe(self, container: Option<ColourDescription>) -> ColourDescription {
        self.current_or(container)
            .unwrap_or(ColourDescription::unspecified(ColourRange::Limited))
    }
}
