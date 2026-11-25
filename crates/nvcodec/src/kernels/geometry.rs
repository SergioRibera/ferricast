//! Where an image kernel reads and writes: the source crop, its
//! orientation, the filter it is resampled with, and where it lands in the
//! output.

use crate::error::{Error, Result};

/// Width × height in pixels (or samples, for one plane).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Size {
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}

impl Size {
    /// `width` × `height`.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// The size of a 4:2:0 chroma plane of a picture this size.
    #[must_use]
    pub const fn chroma_420(self) -> Self {
        Self::new(self.width.div_ceil(2), self.height.div_ceil(2))
    }

    /// The same size turned a quarter (width and height swapped).
    #[must_use]
    pub const fn transposed(self) -> Self {
        Self::new(self.height, self.width)
    }
}

/// A rectangle: top-left corner and size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Rect {
    /// Left edge.
    pub x: u32,
    /// Top edge.
    pub y: u32,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}

impl Rect {
    /// The rectangle at (`x`, `y`) of `width` × `height`.
    #[must_use]
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// The whole of a `size` picture.
    #[must_use]
    pub const fn of(size: Size) -> Self {
        Self::new(0, 0, size.width, size.height)
    }

    /// Its size.
    #[must_use]
    pub const fn size(self) -> Size {
        Size::new(self.width, self.height)
    }

    /// Checks that the rectangle is not empty and lies inside `bounds`.
    fn check_within(self, what: &str, bounds: Size) -> Result<()> {
        let invalid = |reason: String| Error::InvalidKernelInput { reason };
        if self.width == 0 || self.height == 0 {
            return Err(invalid(format!("the {what} {self:?} is empty")));
        }
        let right = u64::from(self.x) + u64::from(self.width);
        let bottom = u64::from(self.y) + u64::from(self.height);
        if right > u64::from(bounds.width) || bottom > u64::from(bounds.height) {
            return Err(invalid(format!(
                "the {what} {self:?} leaves the {}x{} picture",
                bounds.width, bounds.height
            )));
        }
        Ok(())
    }

    /// The rectangle in a 4:2:0 chroma plane covering this luma one; the
    /// origin must be even so chroma samples stay aligned.
    fn chroma_420(self, what: &str) -> Result<Self> {
        if !self.x.is_multiple_of(2) || !self.y.is_multiple_of(2) {
            return Err(Error::InvalidKernelInput {
                reason: format!(
                    "the {what} {self:?} of a 4:2:0 picture must start at an even position"
                ),
            });
        }
        Ok(Self::new(
            self.x / 2,
            self.y / 2,
            self.width.div_ceil(2),
            self.height.div_ceil(2),
        ))
    }
}

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

/// How the picture is turned between the source and the output: an
/// optional horizontal mirror, then a rotation. This is how `wl_output`
/// transforms are defined (`Flipped90` is a mirror, then a quarter turn
/// clockwise), and composing with [`Orientation::FLIP_VERTICAL`] covers
/// `Y_INVERT`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Orientation {
    /// Mirrored left to right before the rotation.
    pub mirrored: bool,
    /// The rotation.
    pub rotation: Rotation,
}

/// An orientation as the 2 × 2 signed permutation matrix it applies to
/// (x, y) (y pointing down), row-major.
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

    /// Every orientation, for searches and tests.
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

    /// Whether width and height trade places.
    #[must_use]
    pub const fn swaps_axes(self) -> bool {
        matches!(
            self.rotation,
            Rotation::Clockwise90 | Rotation::Clockwise270
        )
    }

    /// The size of a `source`-sized picture once oriented.
    #[must_use]
    pub const fn oriented_size(self, source: Size) -> Size {
        if self.swaps_axes() {
            source.transposed()
        } else {
            source
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

    /// The map from oriented coordinates back to a `source`-sized picture's.
    fn inverse(self, source: Size) -> Affine {
        let m = self.matrix();
        let (w, h) = (i64::from(source.width) - 1, i64::from(source.height) - 1);
        // Forward: X = M x + t, with t the shift that puts the oriented
        // picture's top-left corner at the origin.
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
        Affine {
            xx: m[0][0],
            xy: m[1][0],
            x0: -(m[0][0] * t[0] + m[1][0] * t[1]),
            yx: m[0][1],
            yy: m[1][1],
            y0: -(m[0][1] * t[0] + m[1][1] * t[1]),
        }
    }
}

/// `x = xx X + xy Y + x0`, `y = yx X + yy Y + y0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Affine {
    pub(crate) xx: i64,
    pub(crate) xy: i64,
    pub(crate) x0: i64,
    pub(crate) yx: i64,
    pub(crate) yy: i64,
    pub(crate) y0: i64,
}

impl Affine {
    /// The same map followed by a shift of the result.
    const fn shifted(self, dx: i64, dy: i64) -> Self {
        Self {
            x0: self.x0 + dx,
            y0: self.y0 + dy,
            ..self
        }
    }
}

/// The resampling filter, used with the support stretched by the
/// downscale ratio (so a downscale averages every source pixel an output
/// pixel covers): `fast_image_resize`'s convolution, which ferricast's CPU
/// scaler runs with [`Filter::Bilinear`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Filter {
    /// Triangle kernel: bilinear interpolation upwards, a tent average
    /// downwards.
    #[default]
    Bilinear,
    /// Box kernel: nearest upwards, an area average downwards.
    Box,
}

/// What an image kernel does: take `crop` of the source, orient it, resample
/// it into `placement` with `filter`, and fill the rest of the `canvas`
/// (the whole output picture) with black.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Geometry {
    /// The part of the source read, in source pixels.
    pub crop: Rect,
    /// How the crop is turned.
    pub orientation: Orientation,
    /// The output picture's size.
    pub canvas: Size,
    /// Where the oriented crop lands, in output pixels.
    pub placement: Rect,
    /// The resampling filter.
    pub filter: Filter,
}

impl Geometry {
    /// All of a `source`-sized picture stretched over all of `canvas`.
    #[must_use]
    pub const fn new(source: Size, canvas: Size) -> Self {
        Self {
            crop: Rect::of(source),
            orientation: Orientation::IDENTITY,
            canvas,
            placement: Rect::of(canvas),
            filter: Filter::Bilinear,
        }
    }

    /// The same, reading only `crop` of the source.
    #[must_use]
    pub const fn with_crop(self, crop: Rect) -> Self {
        Self { crop, ..self }
    }

    /// The same, turned by `orientation`.
    #[must_use]
    pub const fn with_orientation(self, orientation: Orientation) -> Self {
        Self {
            orientation,
            ..self
        }
    }

    /// The same, landing in `placement` of the canvas.
    #[must_use]
    pub const fn with_placement(self, placement: Rect) -> Self {
        Self { placement, ..self }
    }

    /// The same, resampled with `filter`.
    #[must_use]
    pub const fn with_filter(self, filter: Filter) -> Self {
        Self { filter, ..self }
    }

    /// The geometry of the chroma planes of a 4:2:0 picture: crop and
    /// placement must start at even positions.
    pub(crate) fn chroma_420(self) -> Result<Self> {
        Ok(Self {
            crop: self.crop.chroma_420("crop")?,
            canvas: self.canvas.chroma_420(),
            placement: self.placement.chroma_420("placement")?,
            ..self
        })
    }

    /// Checks the geometry against a `source`-sized input and lowers it to
    /// what the kernels take.
    pub(crate) fn lower(self, source: Size) -> Result<Lowered> {
        let invalid = |reason: String| Error::InvalidKernelInput { reason };
        self.crop.check_within("crop", source)?;
        self.placement.check_within("placement", self.canvas)?;
        let to_i32 = |value: i64, what: &str| {
            i32::try_from(value).map_err(|_| invalid(format!("{what} {value} exceeds i32")))
        };
        let oriented = self.orientation.oriented_size(self.crop.size());
        let map = self
            .orientation
            .inverse(self.crop.size())
            .shifted(i64::from(self.crop.x), i64::from(self.crop.y));
        #[expect(
            clippy::cast_precision_loss,
            reason = "sizes below 2^24 are exact in f32; larger ones only blur the ratio by an ulp"
        )]
        let ratio = |from: u32, to: u32| from as f32 / to as f32;
        Ok(Lowered {
            canvas_w: to_i32(self.canvas.width.into(), "the canvas width")?,
            canvas_h: to_i32(self.canvas.height.into(), "the canvas height")?,
            place_x: to_i32(self.placement.x.into(), "the placement x")?,
            place_y: to_i32(self.placement.y.into(), "the placement y")?,
            place_w: to_i32(self.placement.width.into(), "the placement width")?,
            place_h: to_i32(self.placement.height.into(), "the placement height")?,
            oriented_w: to_i32(oriented.width.into(), "the crop width")?,
            oriented_h: to_i32(oriented.height.into(), "the crop height")?,
            xx: to_i32(map.xx, "a map coefficient")?,
            xy: to_i32(map.xy, "a map coefficient")?,
            x0: to_i32(map.x0, "a map offset")?,
            yx: to_i32(map.yx, "a map coefficient")?,
            yy: to_i32(map.yy, "a map coefficient")?,
            y0: to_i32(map.y0, "a map offset")?,
            ratio_x: ratio(oriented.width, self.placement.width),
            ratio_y: ratio(oriented.height, self.placement.height),
            filter: match self.filter {
                Filter::Bilinear => 0,
                Filter::Box => 1,
            },
            reserved: 0,
        })
    }
}

/// [`Geometry`] as the kernels take it (`struct Geometry` in
/// `kernels.cu`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Lowered {
    pub(crate) canvas_w: i32,
    pub(crate) canvas_h: i32,
    pub(crate) place_x: i32,
    pub(crate) place_y: i32,
    pub(crate) place_w: i32,
    pub(crate) place_h: i32,
    pub(crate) oriented_w: i32,
    pub(crate) oriented_h: i32,
    pub(crate) xx: i32,
    pub(crate) xy: i32,
    pub(crate) x0: i32,
    pub(crate) yx: i32,
    pub(crate) yy: i32,
    pub(crate) y0: i32,
    pub(crate) ratio_x: f32,
    pub(crate) ratio_y: f32,
    pub(crate) filter: i32,
    pub(crate) reserved: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where source pixel (x, y) of a `w` × `h` picture ends up, applied
    /// step by step like the `image` crate's `flip_horizontal` /
    /// `rotate90` / `rotate180` / `rotate270` (what ferricast's capture
    /// uses).
    #[expect(
        clippy::many_single_char_names,
        reason = "the pixel maths reads as w, h, x, y"
    )]
    fn forward(o: Orientation, w: i64, h: i64, x: i64, y: i64) -> (i64, i64) {
        let x = if o.mirrored { w - 1 - x } else { x };
        match o.rotation {
            Rotation::None => (x, y),
            Rotation::Clockwise90 => (h - 1 - y, x),
            Rotation::Half => (w - 1 - x, h - 1 - y),
            Rotation::Clockwise270 => (y, w - 1 - x),
        }
    }

    #[test]
    fn inverse_undoes_every_orientation() {
        let (w, h) = (5_i64, 3_i64);
        for o in Orientation::ALL {
            let map = o.inverse(Size::new(5, 3));
            for y in 0..h {
                for x in 0..w {
                    let (ox, oy) = forward(o, w, h, x, y);
                    let back = (
                        map.xx * ox + map.xy * oy + map.x0,
                        map.yx * ox + map.yy * oy + map.y0,
                    );
                    assert_eq!(back, (x, y), "{o:?} at ({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn composition_matches_applying_twice() {
        let (w, h) = (4_i64, 7_i64);
        for a in Orientation::ALL {
            for b in Orientation::ALL {
                let composed = a.then(b);
                let mid = a.oriented_size(Size::new(4, 7));
                for (x, y) in [(0, 0), (3, 0), (1, 5), (3, 6)] {
                    let (ax, ay) = forward(a, w, h, x, y);
                    let twice = forward(b, i64::from(mid.width), i64::from(mid.height), ax, ay);
                    assert_eq!(forward(composed, w, h, x, y), twice, "{a:?} then {b:?}");
                }
            }
        }
        assert_eq!(
            Orientation::FLIP_VERTICAL.then(Orientation::FLIP_VERTICAL),
            Orientation::IDENTITY
        );
    }

    #[test]
    fn lower_checks_bounds() {
        let source = Size::new(8, 4);
        let ok = Geometry::new(source, Size::new(4, 2));
        assert!(ok.lower(source).is_ok());
        let crop = ok.with_crop(Rect::new(4, 0, 5, 4));
        assert!(matches!(
            crop.lower(source),
            Err(Error::InvalidKernelInput { .. })
        ));
        let empty = ok.with_placement(Rect::new(0, 0, 0, 2));
        assert!(empty.lower(source).is_err());
        let outside = ok.with_placement(Rect::new(1, 1, 4, 2));
        assert!(outside.lower(source).is_err());
    }

    #[test]
    fn lower_folds_the_crop_into_the_map() {
        let source = Size::new(10, 10);
        let lowered = Geometry::new(source, Size::new(3, 2))
            .with_crop(Rect::new(4, 6, 2, 3))
            .with_orientation(Orientation {
                mirrored: false,
                rotation: Rotation::Clockwise90,
            })
            .lower(source)
            .unwrap();
        assert_eq!((lowered.oriented_w, lowered.oriented_h), (3, 2));
        // Oriented (0, 0) is the crop's bottom-left pixel (4, 8).
        assert_eq!((lowered.x0, lowered.y0), (4, 8));
        assert!((lowered.ratio_x - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn chroma_needs_even_origins() {
        let g = Geometry::new(Size::new(9, 5), Size::new(7, 3));
        let chroma = g.chroma_420().unwrap();
        assert_eq!(chroma.crop, Rect::new(0, 0, 5, 3));
        assert_eq!(chroma.canvas, Size::new(4, 2));
        assert!(g.with_crop(Rect::new(1, 0, 8, 5)).chroma_420().is_err());
        assert!(
            g.with_placement(Rect::new(2, 1, 4, 2))
                .chroma_420()
                .is_err()
        );
    }
}
