//! VP9 keyframe headers (VP9 bitstream spec §6.2 `uncompressed_header`)
//! and the `vpcC` record (VP Codec ISO-BMFF binding §2.3) built from
//! them.

use ferricast_core::{ColourDescription, ColourRange, MatrixCoefficients};

use super::bits::Bits;

/// `frame_sync_code` of keyframes and intra-only frames.
const SYNC_CODE: u32 = 0x49_83_42;
/// `color_space` `CS_RGB`.
const CS_RGB: u32 = 7;

/// What a keyframe's uncompressed header says about the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Keyframe {
    /// `profile` (0 to 3).
    pub profile: u8,
    /// Bits per sample: 8, 10 or 12.
    pub bit_depth: u8,
    /// `color_space`, the VP9 matrix code.
    pub colour_space: u8,
    /// `color_range`: full-range samples.
    pub full_range: bool,
    /// `subsampling_x`: chroma halved horizontally.
    pub subsampling_x: bool,
    /// `subsampling_y`: chroma halved vertically.
    pub subsampling_y: bool,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

impl Keyframe {
    /// The H.273 matrix `colour_space` names.
    #[must_use]
    pub fn matrix(self) -> MatrixCoefficients {
        match self.colour_space {
            1 | 3 => MatrixCoefficients::Smpte170M,
            2 => MatrixCoefficients::Bt709,
            4 => MatrixCoefficients::Smpte240M,
            5 => MatrixCoefficients::Bt2020Ncl,
            7 => MatrixCoefficients::Identity,
            _ => MatrixCoefficients::Unspecified,
        }
    }

    /// The `vpcC` box body (version 1): profile, a level guessed from
    /// the frame size, bit depth, chroma subsampling, range, and the
    /// colour codes of `colour` (the container's, since VP9 itself only
    /// signals the matrix).
    #[must_use]
    pub fn vpcc(self, colour: Option<ColourDescription>) -> [u8; 12] {
        // 4:2:0 co-located with luma (1), 4:2:2 (2), 4:4:4 (3).
        let subsampling = match (self.subsampling_x, self.subsampling_y) {
            (true, true) => 1,
            (true, false) => 2,
            (false, _) => 3,
        };
        let (primaries, transfer) = colour.map_or((2, 2), |colour| {
            (colour.primaries.code(), colour.transfer.code())
        });
        let matrix = match self.matrix() {
            MatrixCoefficients::Unspecified => colour.map_or(2, |colour| colour.matrix.code()),
            known => known.code(),
        };
        [
            1,
            0,
            0,
            0,
            self.profile,
            self.level(),
            (self.bit_depth << 4) | (subsampling << 1) | u8::from(self.full_range),
            primaries,
            transfer,
            matrix,
            0,
            0,
        ]
    }

    /// The smallest VP9 level (as `vpcC` codes it, ten times the level)
    /// whose maximum picture size holds the frame.
    fn level(self) -> u8 {
        let samples = u64::from(self.width) * u64::from(self.height);
        [
            (36_864, 10),
            (73_728, 11),
            (122_880, 20),
            (245_760, 21),
            (552_960, 30),
            (983_040, 31),
            (2_228_224, 40),
            (8_912_896, 50),
            (35_651_584, 60),
        ]
        .into_iter()
        .find(|&(max, _)| samples <= max)
        .map_or(62, |(_, level)| level)
    }

    /// The colour the header signals, in its range.
    #[must_use]
    pub fn colour(self, container: Option<ColourDescription>) -> ColourDescription {
        let range = if self.full_range {
            ColourRange::Full
        } else {
            ColourRange::Limited
        };
        let base = container.unwrap_or(ColourDescription::unspecified(range));
        let matrix = match self.matrix() {
            MatrixCoefficients::Unspecified => base.matrix,
            known => known,
        };
        ColourDescription { matrix, ..base }.with_range(range)
    }
}

/// The frames of a packet: the frames of a superframe (§B.3), or the
/// packet itself.
#[must_use]
pub fn frames(packet: &[u8]) -> Vec<&[u8]> {
    let Some(&marker) = packet.last() else {
        return Vec::new();
    };
    if marker & 0xe0 != 0xc0 {
        return vec![packet];
    }
    let count = usize::from(marker & 0x07) + 1;
    let size_bytes = usize::from((marker >> 3) & 0x03) + 1;
    let index_len = 2 + size_bytes * count;
    let Some(index_start) = packet.len().checked_sub(index_len) else {
        return vec![packet];
    };
    if packet[index_start] != marker {
        return vec![packet];
    }
    let mut frames = Vec::with_capacity(count);
    let mut at = 0;
    for size in packet[index_start + 1..packet.len() - 1].chunks_exact(size_bytes) {
        let size = size
            .iter()
            .rev()
            .fold(0_usize, |acc, &byte| (acc << 8) | usize::from(byte));
        let Some(frame) = packet
            .get(at..at + size)
            .filter(|_| at + size <= index_start)
        else {
            return vec![packet];
        };
        frames.push(frame);
        at += size;
    }
    frames
}

/// The keyframe header of the first keyframe in `packet`, if it has one.
#[must_use]
pub fn keyframe(packet: &[u8]) -> Option<Keyframe> {
    frames(packet).into_iter().find_map(parse_keyframe)
}

fn parse_keyframe(frame: &[u8]) -> Option<Keyframe> {
    let mut bits = Bits::new(frame);
    if bits.read(2)? != 2 {
        return None;
    }
    let low = bits.read(1)?;
    let high = bits.read(1)?;
    let profile = u8::try_from((high << 1) | low).ok()?;
    if profile == 3 {
        bits.skip(1)?;
    }
    // show_existing_frame.
    if bits.flag()? {
        return None;
    }
    // frame_type: 0 is KEY_FRAME.
    if bits.flag()? {
        return None;
    }
    // show_frame, error_resilient_mode.
    bits.skip(2)?;
    if bits.read(24)? != SYNC_CODE {
        return None;
    }
    let bit_depth = if profile >= 2 {
        if bits.flag()? { 12 } else { 10 }
    } else {
        8
    };
    let colour_space = bits.read(3)?;
    let (full_range, subsampling_x, subsampling_y) = if colour_space == CS_RGB {
        if profile == 1 || profile == 3 {
            bits.skip(1)?;
        }
        (true, false, false)
    } else {
        let full_range = bits.flag()?;
        if profile == 1 || profile == 3 {
            let x = bits.flag()?;
            let y = bits.flag()?;
            bits.skip(1)?;
            (full_range, x, y)
        } else {
            (full_range, true, true)
        }
    };
    let width = bits.read(16)? + 1;
    let height = bits.read(16)? + 1;
    Some(Keyframe {
        profile,
        bit_depth,
        colour_space: u8::try_from(colour_space).ok()?,
        full_range,
        subsampling_x,
        subsampling_y,
        width,
        height,
    })
}

#[cfg(test)]
mod tests {
    use super::super::bits::Writer;
    use super::*;

    /// A keyframe header of `profile` at `width`x`height`.
    fn keyframe_bytes(profile: u32, ten_bit: bool, width: u32, height: u32) -> Vec<u8> {
        let mut frame = Writer::default();
        frame.put(2, 2).put(profile & 1, 1).put(profile >> 1, 1);
        // show_existing_frame 0, KEY_FRAME, show_frame, not error resilient.
        frame.put(0, 1).put(0, 1).put(1, 1).put(0, 1);
        frame.put(SYNC_CODE, 24);
        if profile >= 2 {
            frame.put(u32::from(!ten_bit), 1);
        }
        // BT.709, limited range.
        frame.put(2, 3).put(0, 1);
        frame.put(width - 1, 16).put(height - 1, 16);
        let mut bytes = frame.bytes();
        bytes.extend_from_slice(&[0; 8]);
        bytes
    }

    #[test]
    fn keyframe_headers_give_size_depth_and_matrix() {
        let header = keyframe(&keyframe_bytes(0, false, 1920, 1080)).unwrap();
        assert_eq!(
            (
                header.profile,
                header.bit_depth,
                header.width,
                header.height
            ),
            (0, 8, 1920, 1080)
        );
        assert_eq!(header.matrix(), MatrixCoefficients::Bt709);
        let colour = header.colour(Some(ColourDescription::BT709_LIMITED));
        assert_eq!(
            (colour.matrix, colour.range),
            (MatrixCoefficients::Bt709, ColourRange::Limited)
        );
        let ten = keyframe(&keyframe_bytes(2, true, 3840, 2160)).unwrap();
        assert_eq!((ten.profile, ten.bit_depth), (2, 10));
        let vpcc = ten.vpcc(None);
        assert_eq!(&vpcc[..6], &[1, 0, 0, 0, 2, 50]);
        assert_eq!(vpcc[6], (10 << 4) | (1 << 1));
        assert_eq!(vpcc[9], 1);
    }

    #[test]
    fn inter_frames_and_superframes_are_handled() {
        // An inter frame: frame_type 1.
        assert_eq!(keyframe(&[0b1000_0100, 0, 0, 0]), None);
        let key = keyframe_bytes(0, false, 640, 360);
        let inter = [0b1000_0110, 1, 2];
        let mut superframe = key.clone();
        superframe.extend_from_slice(&inter);
        let marker = 0b1100_1001; // two frames, two-byte sizes
        superframe.push(marker);
        for size in [key.len(), inter.len()] {
            superframe.extend_from_slice(&u16::try_from(size).unwrap().to_le_bytes());
        }
        superframe.push(marker);
        assert_eq!(frames(&superframe), vec![&key[..], &inter[..]]);
        assert_eq!(keyframe(&superframe).map(|k| k.width), Some(640));
        assert_eq!(frames(&[]), Vec::<&[u8]>::new());
    }
}
