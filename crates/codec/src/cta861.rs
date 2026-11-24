//! Reads the CTA-861.3 Static Metadata Descriptor Type 1 that Android's
//! `MediaFormat` `hdr-static-info` key carries back into
//! [`HdrMetadata`]; [`HdrMetadata::to_cta861`] writes it.

use ferricast_core::{Chromaticity, ContentLightLevel, HdrMetadata, MasteringDisplay};

/// The metadata of a CTA-861.3 Type 1 descriptor (see
/// [`HdrMetadata::to_cta861`] for the layout). All-zero halves are
/// unknown and come back `None`; `None` for a buffer that is shorter
/// than [`HdrMetadata::CTA861_LEN`], not Type 1 (first byte not 0), or
/// carries nothing.
#[must_use]
pub fn hdr_from_cta861(bytes: &[u8]) -> Option<HdrMetadata> {
    let bytes = bytes.get(..HdrMetadata::CTA861_LEN)?;
    let (&kind, fields) = bytes.split_first()?;
    if kind != 0 {
        return None;
    }
    let field: Vec<u16> = fields
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect();
    let point = |at: usize| Chromaticity {
        x: field[at],
        y: field[at + 1],
    };
    let display = field[..10]
        .iter()
        .any(|&value| value != 0)
        .then(|| MasteringDisplay {
            red: point(0),
            green: point(2),
            blue: point(4),
            white_point: point(6),
            max_luminance: u32::from(field[8]) * 10_000,
            min_luminance: u32::from(field[9]),
        });
    let light = (field[10] != 0 || field[11] != 0).then_some(ContentLightLevel {
        max_cll: field[10],
        max_fall: field[11],
    });
    let metadata = HdrMetadata {
        mastering_display: display,
        content_light: light,
    };
    (!metadata.is_empty()).then_some(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_descriptor_round_trips() {
        let metadata = HdrMetadata {
            mastering_display: Some(MasteringDisplay {
                red: Chromaticity {
                    x: 35_400,
                    y: 14_600,
                },
                green: Chromaticity {
                    x: 8_500,
                    y: 39_850,
                },
                blue: Chromaticity { x: 6_550, y: 2_300 },
                white_point: Chromaticity {
                    x: 15_635,
                    y: 16_450,
                },
                max_luminance: 10_000_000,
                min_luminance: 50,
            }),
            content_light: Some(ContentLightLevel {
                max_cll: 1000,
                max_fall: 400,
            }),
        };
        assert_eq!(hdr_from_cta861(&metadata.to_cta861()), Some(metadata));
        let light_only = HdrMetadata {
            mastering_display: None,
            ..metadata
        };
        assert_eq!(hdr_from_cta861(&light_only.to_cta861()), Some(light_only));
    }

    #[test]
    fn empty_short_or_foreign_descriptors_are_none() {
        assert_eq!(hdr_from_cta861(&[0; 25]), None);
        assert_eq!(hdr_from_cta861(&[0; 24]), None);
        let mut other = HdrMetadata {
            mastering_display: None,
            content_light: Some(ContentLightLevel {
                max_cll: 1,
                max_fall: 1,
            }),
        }
        .to_cta861();
        other[0] = 1;
        assert_eq!(hdr_from_cta861(&other), None);
    }
}
