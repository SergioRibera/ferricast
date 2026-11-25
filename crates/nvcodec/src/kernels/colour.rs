//! YCbCr signalling of the kernels' YUV output and the fixed-point RGB →
//! YCbCr coefficients.

/// The RGB → YCbCr matrix.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Matrix {
    /// BT.709 (HD).
    #[default]
    Bt709,
    /// BT.601 (SMPTE 170M / BT.470 BG; Kr 0.299, Kb 0.114).
    Bt601,
}

/// The YCbCr range.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Range {
    /// Limited ("TV"): luma 16–235, chroma 16–240 (8 bits).
    #[default]
    Limited,
    /// Full ("PC"): 0–255.
    Full,
}

/// The colour of YUV output: the matrix RGB input is converted with and the
/// range of the result (also the black the letterbox is filled with).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct YuvColour {
    /// The matrix.
    pub matrix: Matrix,
    /// The range.
    pub range: Range,
}

/// Fractional bits of [`Coefficients`] (`CSC_PRECISION` in `kernels.cu`).
pub(crate) const PRECISION: u32 = 13;

/// RGB → YCbCr at [`PRECISION`] fractional bits: `[yr, yg, yb, cb_r, cb_g,
/// cb_b, cr_r, cr_g, cr_b]` plus the luma and chroma offsets with the
/// rounding bias folded in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Coefficients {
    pub(crate) matrix: [i32; 9],
    pub(crate) bias_y: i32,
    pub(crate) bias_uv: i32,
}

impl YuvColour {
    /// The coefficients ferricast's CPU conversion uses for 8-bit output: the
    /// `yuv` crate's prebuilt "balanced" (13-bit) tables, so GPU and CPU
    /// produce the same samples from the same RGB.
    pub(crate) const fn coefficients(self) -> Coefficients {
        let matrix = match (self.matrix, self.range) {
            (Matrix::Bt709, Range::Limited) => {
                [1496, 5032, 508, -824, -2774, 3598, 3598, -3268, -330]
            }
            (Matrix::Bt709, Range::Full) => [1742, 5859, 591, -939, -3157, 4096, 4096, -3720, -376],
            (Matrix::Bt601, Range::Limited) => {
                [2104, 4130, 802, -1214, -2384, 3598, 3598, -3013, -585]
            }
            (Matrix::Bt601, Range::Full) => {
                [2449, 4809, 934, -1382, -2714, 4096, 4096, -3430, -666]
            }
        };
        // The `yuv` crate rounds with half a unit minus one.
        let rounding = (1 << (PRECISION - 1)) - 1;
        let black = match self.range {
            Range::Limited => 16,
            Range::Full => 0,
        };
        Coefficients {
            matrix,
            bias_y: (black << PRECISION) + rounding,
            bias_uv: (128 << PRECISION) + rounding,
        }
    }

    /// Black in this range as (luma, chroma) at `bits` per sample.
    pub(crate) const fn black(self, bits: u32) -> (u32, u32) {
        let luma = match self.range {
            Range::Limited => 16,
            Range::Full => 0,
        };
        (luma << (bits - 8), 128 << (bits - 8))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tables are the float matrix rounded at 13 bits (what the `yuv`
    /// crate computes for matrices without a prebuilt table).
    #[test]
    fn tables_match_the_float_matrix() {
        for (matrix, kr, kb) in [
            (Matrix::Bt709, 0.2126_f64, 0.0722),
            (Matrix::Bt601, 0.299, 0.114),
        ] {
            for (range, ry, ruv) in [(Range::Limited, 219.0, 224.0), (Range::Full, 255.0, 255.0)] {
                let kg = 1.0 - kr - kb;
                let float = [
                    kr * ry,
                    kg * ry,
                    kb * ry,
                    -0.5 * kr / (1.0 - kb) * ruv,
                    -0.5 * kg / (1.0 - kb) * ruv,
                    0.5 * ruv,
                    0.5 * ruv,
                    -0.5 * kg / (1.0 - kr) * ruv,
                    -0.5 * kb / (1.0 - kr) * ruv,
                ];
                let table = YuvColour { matrix, range }.coefficients().matrix;
                for (got, want) in table.iter().zip(float) {
                    let want = want / 255.0 * 8192.0;
                    assert!(
                        (f64::from(*got) - want).abs() <= 1.0,
                        "{matrix:?} {range:?}: {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn black_levels() {
        let limited = YuvColour::default();
        assert_eq!(limited.black(8), (16, 128));
        assert_eq!(limited.black(16), (4096, 32768));
        let full = YuvColour {
            range: Range::Full,
            ..limited
        };
        assert_eq!(full.black(8), (0, 128));
    }
}
