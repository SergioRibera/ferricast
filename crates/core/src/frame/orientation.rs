//! How a picture must be turned to stand the way the user sees it.
//!
//! A capture of a rotated or flipped output holds the compositor's buffer
//! as it is scanned out, not as the user sees it. The capture describes the
//! correction as an [`Orientation`] (the eight `wl_output` transforms, plus
//! `Y_INVERT` composed in), and either applies it on the CPU
//! ([`RawFrame::oriented`]) or hands it on with a GPU frame
//! (`GpuFrame::orientation`) for a consumer that turns pictures on the GPU.

use bytes::Bytes;

use super::{FrameError, PixelFormat, RawFrame};

/// A clockwise rotation by a multiple of 90°.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Rotation {
    /// Upright.
    #[default]
    None,
    /// A quarter turn clockwise.
    Clockwise90,
    /// A half turn.
    Half,
    /// Three quarter turns clockwise (a quarter turn counter-clockwise).
    Clockwise270,
}

/// The correction a picture needs: an optional left-to-right mirror, then
/// a clockwise rotation. The eight values are the `wl_output` transforms
/// as ferricast undoes them (`Flipped90`: mirror, then a quarter turn
/// clockwise); composing with [`Orientation::FLIP_VERTICAL`] covers
/// screencopy's `Y_INVERT`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Orientation {
    /// Mirrored left to right before the rotation.
    pub mirrored: bool,
    /// The rotation.
    pub rotation: Rotation,
}

/// The 2 × 2 signed permutation matrix an orientation applies to (x, y),
/// y pointing down, row-major.
type Matrix = [[i64; 2]; 2];

const fn multiply(a: Matrix, b: Matrix) -> Matrix {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

/// Where an oriented picture's pixel (X, Y) comes from in the source:
/// `x = xx X + xy Y + x0`, `y = yx X + yy Y + y0`.
#[derive(Clone, Copy, Debug)]
struct Inverse {
    xx: i64,
    xy: i64,
    x0: i64,
    yx: i64,
    yy: i64,
    y0: i64,
}

impl Orientation {
    /// No change.
    pub const IDENTITY: Self = Self {
        mirrored: false,
        rotation: Rotation::None,
    };
    /// Upside down (a mirror and a half turn).
    pub const FLIP_VERTICAL: Self = Self {
        mirrored: true,
        rotation: Rotation::Half,
    };

    /// Every orientation.
    pub const ALL: [Self; 8] = {
        let rotations = [
            Rotation::None,
            Rotation::Clockwise90,
            Rotation::Half,
            Rotation::Clockwise270,
        ];
        let mut all = [Self::IDENTITY; 8];
        let mut i = 0;
        while i < 8 {
            all[i] = Self {
                mirrored: i >= 4,
                rotation: rotations[i % 4],
            };
            i += 1;
        }
        all
    };

    /// Whether the picture is left as it is.
    #[must_use]
    pub const fn is_identity(self) -> bool {
        !self.mirrored && matches!(self.rotation, Rotation::None)
    }

    /// Whether width and height trade places.
    #[must_use]
    pub const fn swaps_axes(self) -> bool {
        matches!(
            self.rotation,
            Rotation::Clockwise90 | Rotation::Clockwise270
        )
    }

    /// The `(width, height)` of a `width` × `height` picture once oriented.
    #[must_use]
    pub const fn oriented_size(self, width: u32, height: u32) -> (u32, u32) {
        if self.swaps_axes() {
            (height, width)
        } else {
            (width, height)
        }
    }

    /// `self` first, then `next`.
    #[must_use]
    pub fn then(self, next: Self) -> Self {
        let product = multiply(next.matrix(), self.matrix());
        Self::ALL
            .into_iter()
            .find(|candidate| candidate.matrix() == product)
            .unwrap_or(Self::IDENTITY)
    }

    const fn matrix(self) -> Matrix {
        let rotation = match self.rotation {
            Rotation::None => [[1, 0], [0, 1]],
            // (x, y) -> (H - 1 - y, x): the linear part is (-y, x).
            Rotation::Clockwise90 => [[0, -1], [1, 0]],
            Rotation::Half => [[-1, 0], [0, -1]],
            // (x, y) -> (y, W - 1 - x).
            Rotation::Clockwise270 => [[0, 1], [-1, 0]],
        };
        if self.mirrored {
            multiply(rotation, [[-1, 0], [0, 1]])
        } else {
            rotation
        }
    }

    /// The map from oriented coordinates back to a `width` × `height`
    /// source's.
    fn inverse(self, width: u32, height: u32) -> Inverse {
        let m = self.matrix();
        let (w, h) = (i64::from(width) - 1, i64::from(height) - 1);
        // Forward: X = M x + t, t putting the oriented picture's top-left
        // corner at the origin.
        let corners = [(0, 0), (w, 0), (0, h), (w, h)];
        let min_of = |row: [i64; 2]| {
            corners
                .iter()
                .map(|&(x, y)| row[0] * x + row[1] * y)
                .min()
                .unwrap_or(0)
        };
        let t = [-min_of(m[0]), -min_of(m[1])];
        // M is orthogonal: x = Mᵀ (X - t).
        Inverse {
            xx: m[0][0],
            xy: m[1][0],
            x0: -(m[0][0] * t[0] + m[1][0] * t[1]),
            yx: m[0][1],
            yy: m[1][1],
            y0: -(m[0][1] * t[0] + m[1][1] * t[1]),
        }
    }
}

impl RawFrame {
    /// The picture turned by `orientation`, tightly packed
    /// (`stride == width * 4`); unchanged for the identity.
    ///
    /// # Errors
    /// [`FrameError::Unorientable`] for a format other than BGRA / RGBA
    /// or data shorter than the picture.
    pub fn oriented(self, orientation: Orientation) -> Result<Self, FrameError> {
        if orientation.is_identity() {
            return Ok(self);
        }
        let unorientable = || FrameError::Unorientable {
            format: self.format,
            width: self.width,
            height: self.height,
        };
        if !matches!(self.format, PixelFormat::Bgra | PixelFormat::Rgba) {
            return Err(unorientable());
        }
        let (width, height) = (self.width as usize, self.height as usize);
        let stride = self.stride as usize;
        let needed = height
            .checked_sub(1)
            .map_or(0, |rows| rows * stride + width * 4);
        if stride < width * 4 || self.data.len() < needed {
            return Err(unorientable());
        }
        let (out_width, out_height) = orientation.oriented_size(self.width, self.height);
        let map = orientation.inverse(self.width, self.height);
        let mut out = Vec::with_capacity(width * height * 4);
        for y_out in 0..i64::from(out_height) {
            for x_out in 0..i64::from(out_width) {
                let x = map.xx * x_out + map.xy * y_out + map.x0;
                let y = map.yx * x_out + map.yy * y_out + map.y0;
                // The map is a bijection onto the source's pixels.
                let (Ok(x), Ok(y)) = (usize::try_from(x), usize::try_from(y)) else {
                    return Err(unorientable());
                };
                let at = y * stride + x * 4;
                let pixel = self.data.get(at..at + 4).ok_or_else(unorientable)?;
                out.extend_from_slice(pixel);
            }
        }
        Ok(Self {
            width: out_width,
            height: out_height,
            stride: out_width * 4,
            data: Bytes::from(out),
            ..self
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `width` × `height` BGRA picture whose pixel (x, y) is
    /// `[x, y, 0, 255]`, rows padded to `stride`.
    fn picture(width: u32, height: u32, stride: u32) -> RawFrame {
        let mut data = vec![0; (stride * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let at = (y * stride + x * 4) as usize;
                data[at..at + 4].copy_from_slice(&[
                    u8::try_from(x).unwrap(),
                    u8::try_from(y).unwrap(),
                    0,
                    255,
                ]);
            }
        }
        RawFrame {
            width,
            height,
            stride,
            format: PixelFormat::Bgra,
            data: Bytes::from(data),
            timestamp_us: 7,
            colour: None,
            hdr: None,
        }
    }

    /// Where pixel (x, y) of the source lands.
    fn at(frame: &RawFrame, x: u32, y: u32) -> (u8, u8) {
        let at = (y * frame.stride + x * 4) as usize;
        (frame.data[at], frame.data[at + 1])
    }

    #[test]
    fn quarter_turn_clockwise_puts_the_bottom_left_corner_top_left() {
        let turned = picture(3, 2, 16)
            .oriented(Orientation {
                mirrored: false,
                rotation: Rotation::Clockwise90,
            })
            .unwrap();
        assert_eq!((turned.width, turned.height, turned.stride), (2, 3, 8));
        // (x, y) -> (H - 1 - y, x).
        assert_eq!(at(&turned, 0, 0), (0, 1));
        assert_eq!(at(&turned, 1, 0), (0, 0));
        assert_eq!(at(&turned, 1, 2), (2, 0));
        assert_eq!(turned.timestamp_us, 7);
    }

    #[test]
    fn every_orientation_moves_each_pixel_once() {
        let source = picture(5, 3, 24);
        for orientation in Orientation::ALL {
            let turned = source.clone().oriented(orientation).unwrap();
            let mut seen: Vec<(u8, u8)> = turned
                .data
                .chunks_exact(4)
                .map(|pixel| (pixel[0], pixel[1]))
                .collect();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), 15, "{orientation:?}");
        }
    }

    #[test]
    fn mirror_then_vertical_flip_is_a_half_turn() {
        let mirror = Orientation {
            mirrored: true,
            rotation: Rotation::None,
        };
        let half = Orientation {
            mirrored: false,
            rotation: Rotation::Half,
        };
        assert_eq!(mirror.then(Orientation::FLIP_VERTICAL), half);
        assert!(
            Orientation::FLIP_VERTICAL
                .then(Orientation::FLIP_VERTICAL)
                .is_identity()
        );
        let source = picture(4, 3, 16);
        let flipped = source.oriented(Orientation::FLIP_VERTICAL).unwrap();
        assert_eq!(at(&flipped, 1, 0), (1, 2));
    }

    #[test]
    fn short_or_planar_pictures_are_refused() {
        let mut short = picture(4, 4, 16);
        short.data = short.data.slice(..40);
        assert!(short.oriented(Orientation::FLIP_VERTICAL).is_err());
        let mut nv12 = picture(4, 4, 16);
        nv12.format = PixelFormat::Nv12;
        assert!(nv12.oriented(Orientation::FLIP_VERTICAL).is_err());
    }
}
