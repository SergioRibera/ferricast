//! Static HDR metadata: the mastering display a picture was graded on
//! (SMPTE ST 2086) and the content light levels it reaches (CTA-861.3).
//!
//! Units follow H.265 / MP4 / Matroska-integer signalling, so every
//! container and bitstream maps onto these types without rounding:
//! chromaticities in 0.00002 steps, luminances in 0.0001 cd/m², light
//! levels in cd/m². AV1's fixed-point units convert with
//! [`MasteringDisplay::from_av1`].

/// A CIE 1931 xy chromaticity, each coordinate in units of 0.00002
/// (`50_000` = 1.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub struct Chromaticity {
    pub x: u16,
    pub y: u16,
}

impl Chromaticity {
    /// Coordinates per unit (1 / 0.00002).
    pub const SCALE: f64 = 50_000.0;

    /// The coordinates as fractions.
    #[must_use]
    pub fn xy(self) -> (f64, f64) {
        (
            f64::from(self.x) / Self::SCALE,
            f64::from(self.y) / Self::SCALE,
        )
    }
}

/// SMPTE ST 2086 mastering display colour volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub struct MasteringDisplay {
    pub red: Chromaticity,
    pub green: Chromaticity,
    pub blue: Chromaticity,
    pub white_point: Chromaticity,
    /// Peak luminance, 0.0001 cd/m² units.
    pub max_luminance: u32,
    /// Black level, 0.0001 cd/m² units.
    pub min_luminance: u32,
}

impl MasteringDisplay {
    /// Luminance units per cd/m².
    pub const LUMINANCE_SCALE: f64 = 10_000.0;

    /// Bytes of the H.265 `mastering_display_colour_volume` SEI payload,
    /// which is also the body of an MP4 `mdcv` box.
    pub const SEI_LEN: usize = 24;

    /// Peak luminance in cd/m².
    #[must_use]
    pub fn max_nits(self) -> f64 {
        f64::from(self.max_luminance) / Self::LUMINANCE_SCALE
    }

    /// Black level in cd/m².
    #[must_use]
    pub fn min_nits(self) -> f64 {
        f64::from(self.min_luminance) / Self::LUMINANCE_SCALE
    }

    /// Reads the H.265 SEI payload / MP4 `mdcv` body: big-endian,
    /// primaries in green, blue, red order, then white point and the
    /// max / min luminance. `None` when `payload` is shorter than
    /// [`Self::SEI_LEN`].
    #[must_use]
    pub fn from_sei(payload: &[u8]) -> Option<Self> {
        let bytes: &[u8; Self::SEI_LEN] = payload.get(..Self::SEI_LEN)?.try_into().ok()?;
        let u16_at = |at: usize| u16::from_be_bytes([bytes[at], bytes[at + 1]]);
        let u32_at = |at: usize| {
            u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        let xy = |at: usize| Chromaticity {
            x: u16_at(at),
            y: u16_at(at + 2),
        };
        Some(Self {
            green: xy(0),
            blue: xy(4),
            red: xy(8),
            white_point: xy(12),
            max_luminance: u32_at(16),
            min_luminance: u32_at(20),
        })
    }

    /// The H.265 SEI payload / MP4 `mdcv` body (see [`Self::from_sei`]).
    /// `VideoToolbox` takes the same bytes as
    /// `kCVImageBufferMasteringDisplayColorVolumeKey`.
    #[must_use]
    pub fn to_sei(self) -> [u8; Self::SEI_LEN] {
        let mut out = [0; Self::SEI_LEN];
        let points = [self.green, self.blue, self.red, self.white_point];
        for (chunk, point) in out.chunks_exact_mut(4).zip(points) {
            chunk[..2].copy_from_slice(&point.x.to_be_bytes());
            chunk[2..].copy_from_slice(&point.y.to_be_bytes());
        }
        out[16..20].copy_from_slice(&self.max_luminance.to_be_bytes());
        out[20..24].copy_from_slice(&self.min_luminance.to_be_bytes());
        out
    }

    /// From AV1 `metadata_hdr_mdcv` fields: chromaticities as 0.16
    /// fixed point in R, G, B order, then the white point, max
    /// luminance as 24.8 and min luminance as 18.14 fixed point.
    #[must_use]
    pub fn from_av1(
        primaries: [(u16, u16); 3],
        white_point: (u16, u16),
        max: u32,
        min: u32,
    ) -> Self {
        // 0.16 fixed point → 0.00002 units: v / 65536 * 50000.
        let chroma = |(x, y): (u16, u16)| {
            let scale = |v: u16| {
                u16::try_from((u32::from(v) * 50_000 + 32_768) >> 16)
                    .expect("invariant: 65535 * 50000 / 65536 < 50000 fits u16")
            };
            Chromaticity {
                x: scale(x),
                y: scale(y),
            }
        };
        // 24.8 → 0.0001: v / 256 * 10000; 18.14 → 0.0001: v / 16384 * 10000.
        let max_luminance = u32::try_from((u64::from(max) * 10_000 + 128) >> 8).unwrap_or(u32::MAX);
        let min_luminance =
            u32::try_from((u64::from(min) * 10_000 + 8_192) >> 14).unwrap_or(u32::MAX);
        Self {
            red: chroma(primaries[0]),
            green: chroma(primaries[1]),
            blue: chroma(primaries[2]),
            white_point: chroma(white_point),
            max_luminance,
            min_luminance,
        }
    }
}

/// CTA-861.3 content light level: the brightest pixel and the
/// brightest frame average of the content, in cd/m² (0 = unknown).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub struct ContentLightLevel {
    pub max_cll: u16,
    pub max_fall: u16,
}

impl ContentLightLevel {
    /// Bytes of the H.265 `content_light_level_info` SEI payload, which
    /// is also the body of an MP4 `clli` box.
    pub const SEI_LEN: usize = 4;

    /// Reads the big-endian SEI payload / `clli` body; `None` when
    /// `payload` is shorter than [`Self::SEI_LEN`].
    #[must_use]
    pub fn from_sei(payload: &[u8]) -> Option<Self> {
        let bytes = payload.get(..Self::SEI_LEN)?;
        Some(Self {
            max_cll: u16::from_be_bytes([bytes[0], bytes[1]]),
            max_fall: u16::from_be_bytes([bytes[2], bytes[3]]),
        })
    }

    /// The SEI payload / `clli` body.
    #[must_use]
    pub fn to_sei(self) -> [u8; Self::SEI_LEN] {
        let [a, b] = self.max_cll.to_be_bytes();
        let [c, d] = self.max_fall.to_be_bytes();
        [a, b, c, d]
    }
}

/// The static HDR metadata of a stream. Either half may be missing:
/// streams often carry the mastering display without light levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub struct HdrMetadata {
    pub mastering_display: Option<MasteringDisplay>,
    pub content_light: Option<ContentLightLevel>,
}

impl HdrMetadata {
    /// Bytes of a CTA-861.3 Static Metadata Descriptor Type 1 with its
    /// type byte, the layout Android's `MediaFormat` `hdr-static-info`
    /// key takes.
    pub const CTA861_LEN: usize = 25;

    /// `true` when neither half is present.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.mastering_display.is_none() && self.content_light.is_none()
    }

    /// Reads a CTA-861.3 Type 1 descriptor (see [`Self::to_cta861`]), as
    /// Android's `hdr-static-info` reports it. All-zero halves are unknown
    /// and come back `None`; `None` for a buffer shorter than
    /// [`Self::CTA861_LEN`], not Type 1 (first byte not 0), or carrying
    /// nothing.
    #[must_use]
    pub fn from_cta861(bytes: &[u8]) -> Option<Self> {
        let bytes = bytes.get(..Self::CTA861_LEN)?;
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
        let metadata = Self {
            mastering_display: display,
            content_light: light,
        };
        (!metadata.is_empty()).then_some(metadata)
    }

    /// The CTA-861.3 Type 1 descriptor: type byte 0, then little-endian
    /// R, G, B, white point chromaticities (0.00002 units), max
    /// mastering luminance (cd/m²), min mastering luminance (0.0001
    /// cd/m²), `MaxCLL`, `MaxFALL`. Missing halves are zeros (unknown).
    #[must_use]
    pub fn to_cta861(self) -> [u8; Self::CTA861_LEN] {
        let mut out = [0; Self::CTA861_LEN];
        let mut fields = Vec::with_capacity(12);
        if let Some(display) = self.mastering_display {
            for point in [
                display.red,
                display.green,
                display.blue,
                display.white_point,
            ] {
                fields.extend([point.x, point.y]);
            }
            let max_nits = u16::try_from(display.max_luminance / 10_000).unwrap_or(u16::MAX);
            let min = u16::try_from(display.min_luminance).unwrap_or(u16::MAX);
            fields.extend([max_nits, min]);
        } else {
            fields.extend([0; 10]);
        }
        let light = self.content_light.unwrap_or(ContentLightLevel {
            max_cll: 0,
            max_fall: 0,
        });
        fields.extend([light.max_cll, light.max_fall]);
        for (chunk, field) in out[1..].chunks_exact_mut(2).zip(fields) {
            chunk.copy_from_slice(&field.to_le_bytes());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// BT.2020 primaries, D65, 1000 / 0.005 cd/m², as HDR10 masters use.
    const HDR10: MasteringDisplay = MasteringDisplay {
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
    };

    #[test]
    fn the_sei_payload_round_trips_in_green_blue_red_order() {
        let sei = HDR10.to_sei();
        assert_eq!(&sei[..4], &[0x21, 0x34, 0x9B, 0xAA], "green first");
        assert_eq!(MasteringDisplay::from_sei(&sei), Some(HDR10));
        assert_eq!(MasteringDisplay::from_sei(&sei[..23]), None);
        assert!((HDR10.max_nits() - 1000.0).abs() < f64::EPSILON);

        let light = ContentLightLevel {
            max_cll: 1000,
            max_fall: 400,
        };
        assert_eq!(ContentLightLevel::from_sei(&light.to_sei()), Some(light));
    }

    #[test]
    fn av1_fixed_point_converts_to_sei_units() {
        // 0.708 / 0.292 in 0.16, 1000 cd/m² in 24.8, 0.005 cd/m² in 18.14.
        let display = MasteringDisplay::from_av1(
            [(46_399, 19_136), (11_141, 52_233), (8_585, 3_015)],
            (20_493, 21_561),
            1000 << 8,
            82,
        );
        assert_eq!(
            display.red,
            Chromaticity {
                x: 35_400,
                y: 14_600
            }
        );
        assert_eq!(display.max_luminance, 10_000_000);
        assert_eq!(display.min_luminance, 50);
    }

    #[test]
    fn cta861_descriptor_is_little_endian_with_nits_peak() {
        let info = HdrMetadata {
            mastering_display: Some(HDR10),
            content_light: Some(ContentLightLevel {
                max_cll: 1000,
                max_fall: 400,
            }),
        }
        .to_cta861();
        assert_eq!(info[0], 0, "descriptor type 1 is coded 0");
        assert_eq!(&info[1..3], &35_400_u16.to_le_bytes(), "red x first");
        assert_eq!(&info[17..19], &1000_u16.to_le_bytes(), "peak in cd/m²");
        assert_eq!(&info[19..21], &50_u16.to_le_bytes());
        assert_eq!(&info[21..25], &[0xE8, 0x03, 0x90, 0x01]);
        assert!(HdrMetadata::default().is_empty());
    }

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
        assert_eq!(
            HdrMetadata::from_cta861(&metadata.to_cta861()),
            Some(metadata)
        );
        let light_only = HdrMetadata {
            mastering_display: None,
            ..metadata
        };
        assert_eq!(
            HdrMetadata::from_cta861(&light_only.to_cta861()),
            Some(light_only)
        );
    }

    #[test]
    fn empty_short_or_foreign_descriptors_are_none() {
        assert_eq!(HdrMetadata::from_cta861(&[0; 25]), None);
        assert_eq!(HdrMetadata::from_cta861(&[0; 24]), None);
        let mut other = HdrMetadata {
            mastering_display: None,
            content_light: Some(ContentLightLevel {
                max_cll: 1,
                max_fall: 1,
            }),
        }
        .to_cta861();
        other[0] = 1;
        assert_eq!(HdrMetadata::from_cta861(&other), None);
    }
}
