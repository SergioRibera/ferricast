//! HDR10 SEI payloads for the API versions (and codecs) whose NVENC cannot
//! write them itself: they go through `seiPayloadArray`, which NVENC wraps in
//! an SEI NAL unit (header, size, emulation prevention).
//!
//! Syntax from ITU-T H.265 D.2.28 / D.2.35 (identical in H.264 D.1.29 /
//! D.1.31): big-endian fields, primaries in G, B, R order.

use super::config::{ContentLightLevel, Hdr10, MasteringDisplay};

/// `mastering_display_colour_volume` payload type.
pub(crate) const MASTERING_DISPLAY_TYPE: u32 = 137;
/// `content_light_level_info` payload type.
pub(crate) const CONTENT_LIGHT_TYPE: u32 = 144;

/// The SEI payloads of `hdr10`, in a fixed buffer the pictures point into.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HdrSei {
    /// Mastering display payload, if any.
    pub(crate) mastering_display: Option<[u8; 24]>,
    /// Content light payload, if any.
    pub(crate) content_light: Option<[u8; 4]>,
}

impl From<&Hdr10> for HdrSei {
    fn from(hdr10: &Hdr10) -> Self {
        Self {
            mastering_display: hdr10.mastering_display.as_ref().map(mastering_display),
            content_light: hdr10.content_light.map(content_light),
        }
    }
}

fn mastering_display(display: &MasteringDisplay) -> [u8; 24] {
    let mut payload = [0; 24];
    let fields = [
        display.green.x,
        display.green.y,
        display.blue.x,
        display.blue.y,
        display.red.x,
        display.red.y,
        display.white_point.x,
        display.white_point.y,
    ];
    let (primaries, luminance) = payload.split_at_mut(16);
    for (chunk, value) in primaries.chunks_exact_mut(2).zip(fields) {
        chunk.copy_from_slice(&value.to_be_bytes());
    }
    let (max, min) = luminance.split_at_mut(4);
    max.copy_from_slice(&display.max_luminance.to_be_bytes());
    min.copy_from_slice(&display.min_luminance.to_be_bytes());
    payload
}

fn content_light(level: ContentLightLevel) -> [u8; 4] {
    let [a, b] = level.max_content.to_be_bytes();
    let [c, d] = level.max_frame_average.to_be_bytes();
    [a, b, c, d]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::config::Chromaticity;

    #[test]
    fn payloads_follow_the_sei_syntax() {
        let hdr = Hdr10 {
            mastering_display: Some(MasteringDisplay {
                red: Chromaticity {
                    x: 0x0102,
                    y: 0x0304,
                },
                green: Chromaticity {
                    x: 0x0506,
                    y: 0x0708,
                },
                blue: Chromaticity {
                    x: 0x090a,
                    y: 0x0b0c,
                },
                white_point: Chromaticity {
                    x: 0x0d0e,
                    y: 0x0f10,
                },
                max_luminance: 0x1112_1314,
                min_luminance: 0x1516_1718,
            }),
            content_light: Some(ContentLightLevel {
                max_content: 1000,
                max_frame_average: 400,
            }),
        };
        let sei = HdrSei::from(&hdr);
        assert_eq!(
            sei.mastering_display,
            Some([
                5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 13, 14, 15, 16, 0x11, 0x12, 0x13, 0x14,
                0x15, 0x16, 0x17, 0x18
            ])
        );
        assert_eq!(sei.content_light, Some([0x03, 0xe8, 0x01, 0x90]));
        assert_eq!(HdrSei::from(&Hdr10::default()), HdrSei::default());
    }
}
