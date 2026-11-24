//! SEI messages the parser hands over per picture (`pfnGetSEIMsg`), and the
//! two HDR ones decoded.

use std::ops::Range;

use crate::sys::cuvid as sys;

/// `mastering_display_colour_volume` (H.264 / HEVC SEI payload type).
pub const MASTERING_DISPLAY_COLOUR_VOLUME: u8 = 137;
/// `content_light_level_info` (H.264 / HEVC SEI payload type).
pub const CONTENT_LIGHT_LEVEL_INFO: u8 = 144;

/// The SEI payloads of one picture, kept in reusable buffers.
#[derive(Debug, Default)]
pub(super) struct SeiStore {
    payload: Vec<u8>,
    messages: Vec<(u8, Range<usize>)>,
}

impl SeiStore {
    pub(super) fn clear(&mut self) {
        self.payload.clear();
        self.messages.clear();
    }

    /// Replaces the contents with the messages of one picture: `data` holds
    /// their payloads back to back, `messages` their types and sizes. A size
    /// running past `data` ends the list.
    pub(super) fn fill(&mut self, data: &[u8], messages: &[sys::CUSEIMESSAGE]) {
        self.clear();
        self.payload.extend_from_slice(data);
        let mut start = 0_usize;
        for message in messages {
            let Some(end) = usize::try_from(message.sei_message_size)
                .ok()
                .and_then(|size| start.checked_add(size))
                .filter(|end| *end <= data.len())
            else {
                break;
            };
            self.messages.push((message.sei_message_type, start..end));
            start = end;
        }
    }

    pub(super) fn messages(&self) -> SeiMessages<'_> {
        SeiMessages {
            payload: &self.payload,
            messages: self.messages.iter(),
        }
    }
}

/// One SEI message of a picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeiMessage<'a> {
    /// `payloadType` (137 mastering display, 144 content light level, ...).
    pub payload_type: u8,
    /// The payload as the parser delivers it: the `sei_payload()` bytes, except
    /// for the types NVIDIA decodes into structs (137 and 144: see
    /// [`MasteringDisplay::parse`] and [`ContentLightLevel::parse`]).
    pub payload: &'a [u8],
}

/// The SEI messages of a picture, in bitstream order.
#[derive(Clone, Debug)]
pub struct SeiMessages<'a> {
    payload: &'a [u8],
    messages: std::slice::Iter<'a, (u8, Range<usize>)>,
}

impl<'a> Iterator for SeiMessages<'a> {
    type Item = SeiMessage<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.messages.by_ref().find_map(|(payload_type, range)| {
            self.payload.get(range.clone()).map(|payload| SeiMessage {
                payload_type: *payload_type,
                payload,
            })
        })
    }
}

impl<'a> SeiMessages<'a> {
    /// The first message of `payload_type`.
    #[must_use]
    pub fn payload(self, payload_type: u8) -> Option<&'a [u8]> {
        let mut messages = self;
        Iterator::find(&mut messages, |message| {
            message.payload_type == payload_type
        })
        .map(|message| message.payload)
    }

    /// The mastering display colour volume, if the picture carries one.
    #[must_use]
    pub fn mastering_display(self) -> Option<MasteringDisplay> {
        self.payload(MASTERING_DISPLAY_COLOUR_VOLUME)
            .and_then(MasteringDisplay::parse)
    }

    /// The content light level, if the picture carries one.
    #[must_use]
    pub fn content_light_level(self) -> Option<ContentLightLevel> {
        self.payload(CONTENT_LIGHT_LEVEL_INFO)
            .and_then(ContentLightLevel::parse)
    }
}

/// Reads the little-endian fields of the structs the NVDEC parser decodes
/// these payloads into.
struct Fields<'a>(&'a [u8]);

impl Fields<'_> {
    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let (head, rest) = self.0.split_first_chunk::<N>()?;
        self.0 = rest;
        Some(*head)
    }

    fn u16(&mut self) -> Option<u16> {
        self.take().map(u16::from_le_bytes)
    }

    fn u32(&mut self) -> Option<u32> {
        self.take().map(u32::from_le_bytes)
    }
}

/// `mastering_display_colour_volume` (SMPTE ST 2086), in the units the SEI
/// codes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MasteringDisplay {
    /// The three primaries as (x, y) in units of 0.00002, in bitstream order
    /// (HDR10 streams code green, blue, red).
    pub primaries: [(u16, u16); 3],
    /// White point (x, y) in units of 0.00002.
    pub white_point: (u16, u16),
    /// Maximum luminance in units of 0.0001 cd/m².
    pub max_luminance: u32,
    /// Minimum luminance in units of 0.0001 cd/m².
    pub min_luminance: u32,
}

impl MasteringDisplay {
    /// Decodes the payload as the NVDEC parser delivers it: not the SEI
    /// bytes but NVIDIA's decoded struct (`display_primaries_x[3]`,
    /// `display_primaries_y[3]`, `white_point_x`, `white_point_y` as
    /// little-endian `u16`, then the maximum and minimum luminance as
    /// little-endian `u32`; 24 bytes). `None` when it is shorter.
    #[must_use]
    pub fn parse(payload: &[u8]) -> Option<Self> {
        let mut fields = Fields(payload);
        let mut x = [0; 3];
        for value in &mut x {
            *value = fields.u16()?;
        }
        let mut primaries = [(0, 0); 3];
        for (primary, x) in primaries.iter_mut().zip(x) {
            *primary = (x, fields.u16()?);
        }
        Some(Self {
            primaries,
            white_point: (fields.u16()?, fields.u16()?),
            max_luminance: fields.u32()?,
            min_luminance: fields.u32()?,
        })
    }
}

/// `content_light_level_info` (CTA-861.3), in cd/m².
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ContentLightLevel {
    /// `MaxCLL`: brightest pixel of the content.
    pub max_content: u16,
    /// `MaxFALL`: brightest frame average of the content.
    pub max_frame_average: u16,
}

impl ContentLightLevel {
    /// Decodes the payload as the NVDEC parser delivers it: NVIDIA's decoded
    /// struct (`max_content_light_level`, `max_pic_average_light_level` as
    /// little-endian `u16`). `None` when it is shorter than 4 bytes.
    #[must_use]
    pub fn parse(payload: &[u8]) -> Option<Self> {
        let mut fields = Fields(payload);
        Some(Self {
            max_content: fields.u16()?,
            max_frame_average: fields.u16()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(payload_type: u8, size: u32) -> sys::CUSEIMESSAGE {
        sys::CUSEIMESSAGE {
            sei_message_type: payload_type,
            sei_message_size: size,
            ..Default::default()
        }
    }

    #[test]
    fn payloads_are_split_by_size() {
        let mut store = SeiStore::default();
        let data = [0xe8, 0x03, 0x90, 0x01, 9, 9];
        store.fill(&data, &[message(144, 4), message(5, 2), message(6, 1)]);
        let messages: Vec<_> = store.messages().collect();
        assert_eq!(messages.len(), 2, "a size past the data ends the list");
        assert_eq!(messages[1].payload, &[9, 9]);
        assert_eq!(
            store.messages().content_light_level(),
            Some(ContentLightLevel {
                max_content: 1000,
                max_frame_average: 400
            })
        );
        assert_eq!(store.messages().mastering_display(), None);
        store.clear();
        assert_eq!(store.messages().count(), 0);
    }

    #[test]
    fn mastering_display_fields() {
        // What the driver handed over for x265's
        // G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1).
        let payload = [
            0xc2, 0x33, 0x4c, 0x1d, 0xd0, 0x84, // x: G, B, R
            0xc4, 0x86, 0xb8, 0x0b, 0x80, 0x3e, // y: G, B, R
            0x13, 0x3d, 0x42, 0x40, // white point
            0x80, 0x96, 0x98, 0x00, // 10000000
            0x01, 0x00, 0x00, 0x00, // 1
        ];
        let display = MasteringDisplay::parse(&payload).unwrap();
        assert_eq!(
            display.primaries,
            [(13250, 34500), (7500, 3000), (34000, 16000)]
        );
        assert_eq!(display.white_point, (15635, 16450));
        assert_eq!(
            (display.max_luminance, display.min_luminance),
            (10_000_000, 1)
        );
        assert_eq!(MasteringDisplay::parse(&payload[..23]), None);
    }
}
