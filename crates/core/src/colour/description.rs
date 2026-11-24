//! The colour description a bitstream or container signals for a
//! picture, as ITU-T H.273 codes it (the same code points H.264 / H.265
//! VUI, AV1 sequence headers, VP9 `colr`/`vpcC` and Matroska use).

use std::fmt;

use super::{YuvColour, YuvMatrix};

/// An H.273 code point that is reserved, so no enum variant names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("{field} code {code} is reserved by ITU-T H.273")]
pub struct ReservedColourCode {
    /// The H.273 field the code was read for (`colour_primaries`, ...).
    pub field: &'static str,
    pub code: u8,
}

/// Generates an H.273 code-point enum with strict (`TryFrom<u8>`) and
/// code (`code()`) conversions, so the tables are written once.
macro_rules! h273_enum {
    (
        $(#[$meta:meta])*
        $name:ident, $field:literal {
            $( $(#[$vmeta:meta])* $variant:ident = $code:literal, )+
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
        pub enum $name {
            $( $(#[$vmeta])* $variant, )+
        }

        impl $name {
            /// The H.273 field this enum codes.
            pub const FIELD: &'static str = $field;

            /// The H.273 code point.
            #[must_use]
            pub const fn code(self) -> u8 {
                match self {
                    $( Self::$variant => $code, )+
                }
            }
        }

        impl TryFrom<u8> for $name {
            type Error = ReservedColourCode;

            fn try_from(code: u8) -> Result<Self, Self::Error> {
                match code {
                    $( $code => Ok(Self::$variant), )+
                    _ => Err(ReservedColourCode { field: $field, code }),
                }
            }
        }
    };
}

h273_enum! {
    /// Chromaticity of the source primaries and white point
    /// (H.273 `colour_primaries`).
    ColourPrimaries, "colour_primaries" {
        /// BT.709, sRGB: every HD stream.
        Bt709 = 1,
        Unspecified = 2,
        /// BT.470 System M (NTSC 1953).
        Bt470M = 4,
        /// BT.470 System B/G, BT.601 625-line (PAL / SECAM).
        Bt470Bg = 5,
        /// SMPTE 170M, BT.601 525-line (NTSC).
        Smpte170M = 6,
        Smpte240M = 7,
        /// Generic film (colour filters using Illuminant C).
        Film = 8,
        /// BT.2020 / BT.2100: UHD and HDR.
        Bt2020 = 9,
        /// SMPTE ST 428-1 (CIE 1931 XYZ).
        Smpte428 = 10,
        /// SMPTE RP 431-2, DCI-P3.
        DciP3 = 11,
        /// SMPTE EG 432-1, Display P3 (Apple devices).
        DisplayP3 = 12,
        /// EBU Tech. 3213-E.
        Ebu3213 = 22,
    }
}

h273_enum! {
    /// Opto-electronic transfer function (H.273
    /// `transfer_characteristics`).
    TransferCharacteristics, "transfer_characteristics" {
        Bt709 = 1,
        Unspecified = 2,
        /// Assumed display gamma 2.2 (BT.470 System M).
        Gamma22 = 4,
        /// Assumed display gamma 2.8 (BT.470 System B/G).
        Gamma28 = 5,
        Smpte170M = 6,
        Smpte240M = 7,
        Linear = 8,
        /// Logarithmic, 100:1 range.
        Log100 = 9,
        /// Logarithmic, 100·√10:1 range.
        Log316 = 10,
        /// IEC 61966-2-4 (xvYCC).
        Iec61966_2_4 = 11,
        /// BT.1361 extended colour gamut.
        Bt1361 = 12,
        /// IEC 61966-2-1, sRGB / sYCC.
        Srgb = 13,
        /// BT.2020, 10-bit systems.
        Bt2020Ten = 14,
        /// BT.2020, 12-bit systems.
        Bt2020Twelve = 15,
        /// SMPTE ST 2084 perceptual quantizer: HDR10 / Dolby Vision.
        Pq = 16,
        /// SMPTE ST 428-1.
        Smpte428 = 17,
        /// ARIB STD-B67 hybrid log-gamma: broadcast HDR.
        Hlg = 18,
    }
}

h273_enum! {
    /// How luma and chroma derive from the RGB primaries (H.273
    /// `matrix_coefficients`).
    MatrixCoefficients, "matrix_coefficients" {
        /// The planes carry G, B, R directly (no YCbCr conversion).
        Identity = 0,
        Bt709 = 1,
        Unspecified = 2,
        /// FCC Title 47 §73.682.
        Fcc = 4,
        /// BT.470 System B/G, BT.601 625-line.
        Bt470Bg = 5,
        /// SMPTE 170M, BT.601 525-line.
        Smpte170M = 6,
        Smpte240M = 7,
        YCgCo = 8,
        /// BT.2020 non-constant luminance.
        Bt2020Ncl = 9,
        /// BT.2020 constant luminance.
        Bt2020Cl = 10,
        /// SMPTE ST 2085 (Y′D′zD′x).
        Smpte2085 = 11,
        /// Chromaticity-derived, non-constant luminance.
        ChromaDerivedNcl = 12,
        /// Chromaticity-derived, constant luminance.
        ChromaDerivedCl = 13,
        /// BT.2100 ICtCp.
        ICtCp = 14,
    }
}

h273_enum! {
    /// Where 4:2:0 chroma samples sit relative to luma (H.264 / H.265
    /// `chroma_sample_loc_type`, the same codes as H.273
    /// `ChromaSampleLocType`).
    ChromaLocation, "chroma_sample_loc_type" {
        /// Co-sited horizontally with the left luma sample, vertically
        /// between rows: MPEG-2 / H.264 default.
        Left = 0,
        /// Centred between four luma samples: JPEG / MPEG-1.
        Center = 1,
        /// Co-sited with the top-left luma sample: BT.2020 / UHD.
        TopLeft = 2,
        Top = 3,
        BottomLeft = 4,
        Bottom = 5,
    }
}

/// Whether samples use the full code range or leave headroom and
/// footroom (H.273 `video_full_range_flag`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub enum ColourRange {
    /// 16-235 luma, 16-240 chroma at 8 bits ("TV", "studio swing"):
    /// what every broadcast and streaming encoder emits.
    #[default]
    Limited,
    /// 0-255 at 8 bits ("PC", "JPEG").
    Full,
}

impl ColourRange {
    /// From an H.273 `video_full_range_flag`.
    #[must_use]
    pub const fn from_full_range_flag(full_range: bool) -> Self {
        if full_range {
            Self::Full
        } else {
            Self::Limited
        }
    }

    /// The H.273 `video_full_range_flag`.
    #[must_use]
    pub const fn is_full(self) -> bool {
        matches!(self, Self::Full)
    }
}

/// Everything a picture's colour signalling says: which primaries and
/// transfer function the RGB it encodes uses, how YCbCr derives from
/// that RGB, its sample range and where its chroma samples sit.
///
/// Decoders attach the description their bitstream signals to each
/// [`crate::RawFrame`]; encoders write the one their
/// [`crate::EncoderConfig`] asks for into the stream headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub struct ColourDescription {
    pub primaries: ColourPrimaries,
    pub transfer: TransferCharacteristics,
    pub matrix: MatrixCoefficients,
    pub range: ColourRange,
    /// `None` when the stream does not signal it; H.264 and H.265 then
    /// imply [`ChromaLocation::Left`].
    pub chroma_location: Option<ChromaLocation>,
}

impl ColourDescription {
    /// SDR HD: BT.709 primaries, transfer and matrix, limited range.
    /// What the sender encoders signal, and what every receiver expects.
    pub const BT709_LIMITED: Self = Self {
        primaries: ColourPrimaries::Bt709,
        transfer: TransferCharacteristics::Bt709,
        matrix: MatrixCoefficients::Bt709,
        range: ColourRange::Limited,
        chroma_location: None,
    };

    /// SDR SD (NTSC): BT.601 525-line, limited range.
    pub const BT601_LIMITED: Self = Self {
        primaries: ColourPrimaries::Smpte170M,
        transfer: TransferCharacteristics::Smpte170M,
        matrix: MatrixCoefficients::Smpte170M,
        range: ColourRange::Limited,
        chroma_location: None,
    };

    /// Nothing signalled except the sample range.
    #[must_use]
    pub const fn unspecified(range: ColourRange) -> Self {
        Self {
            primaries: ColourPrimaries::Unspecified,
            transfer: TransferCharacteristics::Unspecified,
            matrix: MatrixCoefficients::Unspecified,
            range,
            chroma_location: None,
        }
    }

    /// From the raw H.273 code points a bitstream carries. Reserved
    /// codes read as `Unspecified`, the meaning H.264 / H.265 give a
    /// decoder for values it does not know.
    #[must_use]
    pub fn from_codes(primaries: u8, transfer: u8, matrix: u8, full_range: bool) -> Self {
        Self {
            primaries: ColourPrimaries::try_from(primaries).unwrap_or(ColourPrimaries::Unspecified),
            transfer: TransferCharacteristics::try_from(transfer)
                .unwrap_or(TransferCharacteristics::Unspecified),
            matrix: MatrixCoefficients::try_from(matrix).unwrap_or(MatrixCoefficients::Unspecified),
            range: ColourRange::from_full_range_flag(full_range),
            chroma_location: None,
        }
    }

    /// The same description with `chroma_location` set.
    #[must_use]
    pub const fn with_chroma_location(self, location: ChromaLocation) -> Self {
        Self {
            chroma_location: Some(location),
            ..self
        }
    }

    /// The same description with `range` set.
    #[must_use]
    pub const fn with_range(self, range: ColourRange) -> Self {
        Self { range, ..self }
    }

    /// Whether the transfer function is an HDR one (PQ or HLG).
    #[must_use]
    pub const fn is_hdr(self) -> bool {
        matches!(
            self.transfer,
            TransferCharacteristics::Pq | TransferCharacteristics::Hlg
        )
    }

    /// The YCbCr → RGB conversion that renders this picture.
    ///
    /// Matrices without a conversion of their own here map to the
    /// closest one players use (FCC and BT.470 B/G to BT.601, BT.2020
    /// constant luminance to its non-constant matrix); an unspecified
    /// or non-YCbCr matrix falls back to [`YuvColour::guess`] for a
    /// picture `height` rows tall.
    #[must_use]
    pub fn yuv(self, height: usize) -> YuvColour {
        let matrix = match self.matrix {
            MatrixCoefficients::Bt709 => YuvMatrix::Bt709,
            MatrixCoefficients::Fcc
            | MatrixCoefficients::Bt470Bg
            | MatrixCoefficients::Smpte170M => YuvMatrix::Bt601,
            MatrixCoefficients::Bt2020Ncl | MatrixCoefficients::Bt2020Cl => YuvMatrix::Bt2020,
            MatrixCoefficients::Identity
            | MatrixCoefficients::Unspecified
            | MatrixCoefficients::Smpte240M
            | MatrixCoefficients::YCgCo
            | MatrixCoefficients::Smpte2085
            | MatrixCoefficients::ChromaDerivedNcl
            | MatrixCoefficients::ChromaDerivedCl
            | MatrixCoefficients::ICtCp => YuvColour::guess(height).matrix,
        };
        YuvColour {
            matrix,
            range: self.range,
        }
    }
}

impl Default for ColourDescription {
    fn default() -> Self {
        Self::BT709_LIMITED
    }
}

impl fmt::Display for ColourDescription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "primaries {:?}, transfer {:?}, matrix {:?}, {:?} range",
            self.primaries, self.transfer, self.matrix, self.range
        )?;
        if let Some(location) = self.chroma_location {
            write!(f, ", chroma {location:?}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_code_round_trips_and_reserved_codes_are_rejected() {
        for code in 0..=u8::MAX {
            if let Ok(primaries) = ColourPrimaries::try_from(code) {
                assert_eq!(primaries.code(), code);
            }
            if let Ok(transfer) = TransferCharacteristics::try_from(code) {
                assert_eq!(transfer.code(), code);
            }
            if let Ok(matrix) = MatrixCoefficients::try_from(code) {
                assert_eq!(matrix.code(), code);
            }
        }
        assert_eq!(
            ColourPrimaries::try_from(3),
            Err(ReservedColourCode {
                field: "colour_primaries",
                code: 3
            })
        );
        assert!(MatrixCoefficients::try_from(15).is_err());
        assert!(ChromaLocation::try_from(6).is_err());
    }

    #[test]
    fn reserved_codes_in_a_bitstream_read_as_unspecified() {
        let description = ColourDescription::from_codes(200, 3, 1, true);
        assert_eq!(description.primaries, ColourPrimaries::Unspecified);
        assert_eq!(description.transfer, TransferCharacteristics::Unspecified);
        assert_eq!(description.matrix, MatrixCoefficients::Bt709);
        assert_eq!(description.range, ColourRange::Full);
    }

    #[test]
    fn the_conversion_follows_the_signalled_matrix_and_range() {
        let pal = ColourDescription::from_codes(5, 5, 5, false);
        assert_eq!(
            pal.yuv(1080),
            YuvColour {
                matrix: YuvMatrix::Bt601,
                range: ColourRange::Limited
            }
        );
        let uhd = ColourDescription::from_codes(9, 16, 9, true);
        assert!(uhd.is_hdr());
        assert_eq!(uhd.yuv(2160).matrix, YuvMatrix::Bt2020);
        assert_eq!(
            ColourDescription::unspecified(ColourRange::Limited)
                .yuv(480)
                .matrix,
            YuvMatrix::Bt601
        );
        assert_eq!(
            ColourDescription::unspecified(ColourRange::Limited)
                .yuv(720)
                .matrix,
            YuvMatrix::Bt709
        );
    }
}
