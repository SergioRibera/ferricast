//! How a picture's YCbCr samples map to RGB.

/// YCbCr → RGB matrix (ITU-T H.273 `matrix_coefficients`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YuvMatrix {
    Bt601,
    Bt709,
    Bt2020,
}

/// The colour description of a decoded YCbCr picture: which matrix
/// turns its samples into RGB and whether they use the full 0-255
/// range or the limited 16-235 (luma) / 16-240 (chroma) one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YuvColour {
    pub matrix: YuvMatrix,
    pub full_range: bool,
}

impl YuvColour {
    /// Pictures at least this tall are assumed HD when nothing is signalled.
    const HD_HEIGHT: usize = 720;

    /// Unsignalled streams follow what players do: BT.709 for HD,
    /// BT.601 for SD, limited range.
    #[must_use]
    pub fn guess(height: usize) -> Self {
        Self {
            matrix: if height >= Self::HD_HEIGHT {
                YuvMatrix::Bt709
            } else {
                YuvMatrix::Bt601
            },
            full_range: false,
        }
    }

    /// From a bitstream's `matrix_coefficients` code and range flag;
    /// codes this crate does not know fall back to [`Self::guess`].
    #[must_use]
    pub fn signalled(matrix_coefficients: Option<u32>, full_range: bool, height: usize) -> Self {
        let matrix = match matrix_coefficients {
            Some(1) => YuvMatrix::Bt709,
            Some(4..=6) => YuvMatrix::Bt601,
            Some(9 | 10) => YuvMatrix::Bt2020,
            _ => Self::guess(height).matrix,
        };
        Self { matrix, full_range }
    }

    /// Fixed-point (×256) coefficients: luma gain, then the Cr→R, Cb→G,
    /// Cr→G and Cb→B weights.
    fn coefficients(self) -> [i32; 5] {
        match (self.matrix, self.full_range) {
            (YuvMatrix::Bt601, false) => [298, 409, 100, 208, 516],
            (YuvMatrix::Bt709, false) => [298, 459, 55, 136, 541],
            (YuvMatrix::Bt2020, false) => [298, 430, 48, 167, 548],
            (YuvMatrix::Bt601, true) => [256, 359, 88, 183, 454],
            (YuvMatrix::Bt709, true) => [256, 403, 48, 120, 475],
            (YuvMatrix::Bt2020, true) => [256, 377, 42, 146, 482],
        }
    }

    /// Converts an NV12 picture (`stride`-pitched luma rows followed by
    /// the interleaved chroma rows) to tightly packed BGRA.
    #[must_use]
    pub fn nv12_to_bgra(
        self,
        data: &[u8],
        stride: usize,
        width: usize,
        height: usize,
    ) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        self.nv12_to_bgra_into(data, stride, (width, height), &mut out)?;
        Some(out)
    }

    /// [`Self::nv12_to_bgra`] into `out`, reusing its allocation.
    /// Returns `None`, leaving `out` unspecified, when `data` is too
    /// short for the picture.
    pub fn nv12_to_bgra_into(
        self,
        data: &[u8],
        stride: usize,
        (width, height): (usize, usize),
        out: &mut Vec<u8>,
    ) -> Option<()> {
        let [gain, cr_r, cb_g, cr_g, cb_b] = self.coefficients();
        let black = if self.full_range { 0 } else { 16 };
        let (luma, chroma) = data.split_at_checked(stride.checked_mul(height)?)?;
        out.clear();
        out.resize(width.checked_mul(height)?.checked_mul(4)?, 0);
        for (row, line) in out.chunks_exact_mut(width * 4).enumerate() {
            let luma_row = luma.get(row * stride..)?.get(..width)?;
            let chroma_row = chroma.get(row / 2 * stride..)?.get(..stride)?;
            let (pixels, _) = line.as_chunks_mut::<4>();
            for (column, pixel) in pixels.iter_mut().enumerate() {
                let y = gain * (i32::from(luma_row[column]) - black) + 128;
                let cb = i32::from(*chroma_row.get(column / 2 * 2)?) - 128;
                let cr = i32::from(*chroma_row.get(column / 2 * 2 + 1)?) - 128;
                pixel[0] = ((y + cb_b * cb) >> 8).clamp(0, 255) as u8;
                pixel[1] = ((y - cb_g * cb - cr_g * cr) >> 8).clamp(0, 255) as u8;
                pixel[2] = ((y + cr_r * cr) >> 8).clamp(0, 255) as u8;
                pixel[3] = 255;
            }
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [YuvColour; 6] = [
        YuvColour {
            matrix: YuvMatrix::Bt601,
            full_range: false,
        },
        YuvColour {
            matrix: YuvMatrix::Bt709,
            full_range: false,
        },
        YuvColour {
            matrix: YuvMatrix::Bt2020,
            full_range: false,
        },
        YuvColour {
            matrix: YuvMatrix::Bt601,
            full_range: true,
        },
        YuvColour {
            matrix: YuvMatrix::Bt709,
            full_range: true,
        },
        YuvColour {
            matrix: YuvMatrix::Bt2020,
            full_range: true,
        },
    ];

    fn one_pixel(y: u8, cb: u8, cr: u8, colour: YuvColour) -> [u8; 4] {
        let nv12 = [y, y, y, y, cb, cr];
        let pixels = colour.nv12_to_bgra(&nv12, 2, 2, 2).unwrap();
        pixels[..4].try_into().unwrap()
    }

    #[test]
    fn white_and_black_map_to_the_rgb_extremes_for_every_matrix_and_range() {
        for colour in ALL {
            let (black, white) = if colour.full_range {
                (0, 255)
            } else {
                (16, 235)
            };
            assert_eq!(one_pixel(white, 128, 128, colour), [255; 4], "{colour:?}");
            assert_eq!(
                one_pixel(black, 128, 128, colour),
                [0, 0, 0, 255],
                "{colour:?}"
            );
        }
    }

    #[test]
    fn the_matrix_changes_how_chroma_becomes_colour() {
        // Limited-range BT.709 pure green is (173, 42, 26); read as BT.601
        // it picks up a visible red cast.
        let green = one_pixel(173, 42, 26, YuvColour::guess(1080));
        assert!(
            green[1] == 255 && green[0] <= 2 && green[2] <= 2,
            "{green:?}"
        );
        let wrong = one_pixel(173, 42, 26, YuvColour::guess(480));
        assert!(wrong[2] >= 15, "{wrong:?}");
    }

    #[test]
    fn unsignalled_colour_is_guessed_from_the_picture_height() {
        assert_eq!(YuvColour::guess(720).matrix, YuvMatrix::Bt709);
        assert_eq!(YuvColour::guess(576).matrix, YuvMatrix::Bt601);
        assert_eq!(
            YuvColour::signalled(Some(6), true, 1080),
            YuvColour {
                matrix: YuvMatrix::Bt601,
                full_range: true
            }
        );
        assert_eq!(
            YuvColour::signalled(Some(2), false, 1080).matrix,
            YuvMatrix::Bt709
        );
        assert_eq!(
            YuvColour::signalled(Some(9), false, 480).matrix,
            YuvMatrix::Bt2020
        );
    }

    #[test]
    fn a_reused_buffer_is_resized_to_the_picture() {
        let mut out = vec![7; 100];
        YuvColour::guess(2)
            .nv12_to_bgra_into(&[235, 235, 235, 235, 128, 128], 2, (2, 2), &mut out)
            .unwrap();
        assert_eq!(out, [255; 16]);
        assert!(
            YuvColour::guess(2)
                .nv12_to_bgra_into(&[0; 5], 2, (2, 2), &mut out)
                .is_none()
        );
    }
}
