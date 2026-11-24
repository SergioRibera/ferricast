//! AV1 OBUs (AV1 spec §5.3) and the sequence header (§5.5), enough to
//! build the `av1C` record (AV1 ISO-BMFF binding §2.3) a decoder is set
//! up from.

use ferricast_core::{ColourDescription, ColourRange};

use super::bits::{Bits, leb128, to_leb128};

/// `OBU_SEQUENCE_HEADER`.
pub(crate) const SEQUENCE_HEADER: u8 = 1;
/// `OBU_TEMPORAL_DELIMITER`.
pub(crate) const TEMPORAL_DELIMITER: u8 = 2;
/// `OBU_PADDING`.
const PADDING: u8 = 15;

/// One OBU of a packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Obu<'a> {
    pub(crate) kind: u8,
    /// The header byte(s): type, extension flag, and the extension.
    header: &'a [u8],
    pub(crate) payload: &'a [u8],
}

impl Obu<'_> {
    /// The OBU with `obu_has_size_field` set, as ISO-BMFF samples and
    /// `av1C` carry it.
    pub(crate) fn sized(&self) -> Vec<u8> {
        let size = to_leb128(self.payload.len() as u64);
        let mut out = Vec::with_capacity(self.header.len() + size.len() + self.payload.len());
        out.extend_from_slice(self.header);
        if let Some(first) = out.first_mut() {
            *first |= 0b10;
        }
        out.extend_from_slice(&size);
        out.extend_from_slice(self.payload);
        out
    }
}

/// The OBUs of a low-overhead-format packet; `None` when it is malformed.
pub(crate) fn obus(packet: &[u8]) -> Option<Vec<Obu<'_>>> {
    let mut out = Vec::new();
    let mut rest = packet;
    while let Some(&first) = rest.first() {
        if first & 0x80 != 0 {
            return None;
        }
        let kind = (first >> 3) & 0x0f;
        let header_len = if first & 0b100 != 0 { 2 } else { 1 };
        let header = rest.get(..header_len)?;
        let after_header = &rest[header_len..];
        let (payload, next) = if first & 0b10 != 0 {
            let (size, size_len) = leb128(after_header)?;
            let size = usize::try_from(size).ok()?;
            let body = after_header.get(size_len..)?;
            (body.get(..size)?, &body[size..])
        } else {
            (after_header, &after_header[after_header.len()..])
        };
        out.push(Obu {
            kind,
            header,
            payload,
        });
        rest = next;
    }
    Some(out)
}

/// The packet as a decoder sample: every OBU sized, without temporal
/// delimiters and padding.
pub(crate) fn sample(obus: &[Obu<'_>]) -> Vec<u8> {
    obus.iter()
        .filter(|obu| !matches!(obu.kind, TEMPORAL_DELIMITER | PADDING))
        .flat_map(Obu::sized)
        .collect()
}

/// What a sequence header says about the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SequenceHeader {
    pub(crate) profile: u8,
    pub(crate) level: u8,
    pub(crate) tier: bool,
    pub(crate) bit_depth: u8,
    pub(crate) monochrome: bool,
    pub(crate) subsampling_x: bool,
    pub(crate) subsampling_y: bool,
    pub(crate) chroma_sample_position: u8,
    pub(crate) max_width: u32,
    pub(crate) max_height: u32,
    pub(crate) colour: ColourDescription,
}

impl SequenceHeader {
    /// Reads a sequence header OBU payload.
    pub(crate) fn parse(payload: &[u8]) -> Option<Self> {
        let mut bits = Bits::new(payload);
        let profile = u8::try_from(bits.read(3)?).ok()?;
        // still_picture.
        bits.skip(1)?;
        let reduced = bits.flag()?;
        let (level, tier) = if reduced {
            (bits.read(5)?, false)
        } else {
            operating_points(&mut bits)?
        };
        let width_bits = usize::try_from(bits.read(4)?).ok()? + 1;
        let height_bits = usize::try_from(bits.read(4)?).ok()? + 1;
        let max_width = bits.read(width_bits)?.checked_add(1)?;
        let max_height = bits.read(height_bits)?.checked_add(1)?;
        if !reduced && bits.flag()? {
            // delta_frame_id_length_minus_2, additional_frame_id_length_minus_1.
            bits.skip(7)?;
        }
        // use_128x128_superblock, enable_filter_intra, enable_intra_edge_filter.
        bits.skip(3)?;
        if !reduced {
            skip_inter_tools(&mut bits)?;
        }
        // enable_superres, enable_cdef, enable_restoration.
        bits.skip(3)?;
        let colour = ColourConfig::parse(&mut bits, profile)?;
        Some(Self {
            profile,
            level: u8::try_from(level).ok()?,
            tier,
            bit_depth: colour.bit_depth,
            monochrome: colour.monochrome,
            subsampling_x: colour.subsampling_x,
            subsampling_y: colour.subsampling_y,
            chroma_sample_position: colour.chroma_sample_position,
            max_width,
            max_height,
            colour: colour.description,
        })
    }

    /// The `av1C` box body, with `sequence_obu` (the sized sequence
    /// header OBU) as its `configOBUs`.
    pub(crate) fn av1c(&self, sequence_obu: &[u8]) -> Vec<u8> {
        let high_bitdepth = self.bit_depth > 8;
        let twelve_bit = self.bit_depth == 12;
        let mut out = vec![
            // marker 1, version 1.
            0x81,
            (self.profile << 5) | (self.level & 0x1f),
            (u8::from(self.tier) << 7)
                | (u8::from(high_bitdepth) << 6)
                | (u8::from(twelve_bit) << 5)
                | (u8::from(self.monochrome) << 4)
                | (u8::from(self.subsampling_x) << 3)
                | (u8::from(self.subsampling_y) << 2)
                | (self.chroma_sample_position & 0b11),
            // No initial presentation delay.
            0,
        ];
        out.extend_from_slice(sequence_obu);
        out
    }
}

/// The level and tier of the first operating point, past timing and
/// decoder model information.
fn operating_points(bits: &mut Bits<'_>) -> Option<(u32, bool)> {
    let mut buffer_delay_bits = 0;
    let mut decoder_model = false;
    if bits.flag()? {
        // timing_info: num_units_in_display_tick, time_scale.
        bits.skip(64)?;
        if bits.flag()? {
            // equal_picture_interval: num_ticks_per_picture_minus_1.
            bits.uvlc()?;
        }
        decoder_model = bits.flag()?;
        if decoder_model {
            buffer_delay_bits = usize::try_from(bits.read(5)?).ok()? + 1;
            // num_units_in_decoding_tick, buffer_removal_time_length_minus_1,
            // frame_presentation_time_length_minus_1.
            bits.skip(42)?;
        }
    }
    let display_delay = bits.flag()?;
    let count = bits.read(5)? + 1;
    let mut first = None;
    for _ in 0..count {
        // operating_point_idc.
        bits.skip(12)?;
        let level = bits.read(5)?;
        let tier = level > 7 && bits.flag()?;
        if decoder_model && bits.flag()? {
            // decoder_buffer_delay, encoder_buffer_delay, low_delay_mode_flag.
            bits.skip(2 * buffer_delay_bits + 1)?;
        }
        if display_delay && bits.flag()? {
            // initial_display_delay_minus_1.
            bits.skip(4)?;
        }
        first.get_or_insert((level, tier));
    }
    first
}

/// The inter-prediction tool flags of a non-reduced sequence header.
fn skip_inter_tools(bits: &mut Bits<'_>) -> Option<()> {
    // enable_interintra_compound, enable_masked_compound,
    // enable_warped_motion, enable_dual_filter.
    bits.skip(4)?;
    let order_hint = bits.flag()?;
    if order_hint {
        // enable_jnt_comp, enable_ref_frame_mvs.
        bits.skip(2)?;
    }
    let screen_content = if bits.flag()? {
        // seq_choose_screen_content_tools: SELECT_SCREEN_CONTENT_TOOLS.
        2
    } else {
        bits.read(1)?
    };
    if screen_content > 0 && !bits.flag()? {
        // seq_force_integer_mv.
        bits.skip(1)?;
    }
    if order_hint {
        // order_hint_bits_minus_1.
        bits.skip(3)?;
    }
    Some(())
}

/// `color_config()` (§5.5.2).
struct ColourConfig {
    bit_depth: u8,
    monochrome: bool,
    subsampling_x: bool,
    subsampling_y: bool,
    chroma_sample_position: u8,
    description: ColourDescription,
}

impl ColourConfig {
    fn parse(bits: &mut Bits<'_>, profile: u8) -> Option<Self> {
        let high_bitdepth = bits.flag()?;
        let bit_depth = match (profile, high_bitdepth) {
            (2, true) => {
                if bits.flag()? {
                    12
                } else {
                    10
                }
            }
            (_, true) => 10,
            (_, false) => 8,
        };
        let monochrome = profile != 1 && bits.flag()?;
        let (primaries, transfer, matrix) = if bits.flag()? {
            (bits.read(8)?, bits.read(8)?, bits.read(8)?)
        } else {
            (2, 2, 2)
        };
        let code = |value: u32| u8::try_from(value).unwrap_or(2);
        let describe = |full: bool| {
            ColourDescription::from_codes(code(primaries), code(transfer), code(matrix), full)
        };
        if monochrome {
            let full = bits.flag()?;
            return Some(Self {
                bit_depth,
                monochrome,
                subsampling_x: true,
                subsampling_y: true,
                chroma_sample_position: 0,
                description: describe(full),
            });
        }
        // sRGB (BT.709 primaries, sRGB transfer, identity matrix) is 4:4:4
        // full range with no range bit.
        let srgb = (primaries, transfer, matrix) == (1, 13, 0);
        let (full, subsampling_x, subsampling_y) = if srgb {
            (true, false, false)
        } else {
            let full = bits.flag()?;
            let (x, y) = match profile {
                0 => (true, true),
                1 => (false, false),
                _ if bit_depth == 12 => {
                    let x = bits.flag()?;
                    (x, x && bits.flag()?)
                }
                _ => (true, false),
            };
            (full, x, y)
        };
        let chroma_sample_position = if subsampling_x && subsampling_y {
            u8::try_from(bits.read(2)?).ok()?
        } else {
            0
        };
        Some(Self {
            bit_depth,
            monochrome,
            subsampling_x,
            subsampling_y,
            chroma_sample_position,
            description: describe(full).with_range(if full {
                ColourRange::Full
            } else {
                ColourRange::Limited
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headers::bits::Writer;
    use ferricast_core::TransferCharacteristics;

    /// A non-reduced Main (profile 0) sequence header, one operating
    /// point at level 8 (4.0), timing info, order hints, then a colour
    /// config.
    fn sequence_header(ten_bit: bool, pq: bool) -> Vec<u8> {
        let mut seq = Writer::default();
        // profile 0, not still, not reduced; timing info present.
        seq.put(0, 3).put(0, 1).put(0, 1).put(1, 1);
        seq.put(1001, 32).put(60_000, 32).put(0, 1);
        // no decoder model, no display delay, one operating point.
        seq.put(0, 1).put(0, 1).put(0, 5);
        seq.put(0, 12).put(8, 5).put(0, 1);
        // 12-bit sizes, 1920x1080.
        seq.put(11, 4).put(11, 4).put(1919, 12).put(1079, 12);
        // frame ids off, 128x128, filter intra, edge filter.
        seq.put(0, 1).put(0b111, 3);
        // interintra, masked, warped, dual filter; order hint, jnt, mvs;
        // choose screen content; choose integer mv; order hint bits.
        seq.put(0b1111, 4)
            .put(0b111, 3)
            .put(1, 1)
            .put(1, 1)
            .put(6, 3);
        // superres, cdef, restoration.
        seq.put(0b011, 3);
        // colour: high bitdepth, not mono, description present.
        seq.put(u32::from(ten_bit), 1).put(0, 1).put(1, 1);
        seq.put(9, 8).put(if pq { 16 } else { 1 }, 8).put(9, 8);
        // limited range, chroma sample position 1, separate uv delta q.
        seq.put(0, 1).put(1, 2).put(0, 1);
        // film grain.
        seq.put(0, 1);
        seq.bytes()
    }

    #[test]
    fn a_sequence_header_gives_profile_level_depth_and_colour() {
        let header = SequenceHeader::parse(&sequence_header(true, true)).unwrap();
        assert_eq!((header.profile, header.level, header.bit_depth), (0, 8, 10));
        assert_eq!((header.max_width, header.max_height), (1920, 1080));
        assert_eq!(header.colour.transfer, TransferCharacteristics::Pq);
        assert_eq!(header.chroma_sample_position, 1);
        let eight = SequenceHeader::parse(&sequence_header(false, false)).unwrap();
        assert_eq!(eight.bit_depth, 8);
        assert!(SequenceHeader::parse(&[0]).is_none());
    }

    #[test]
    fn obus_are_split_sized_and_delimiters_dropped() {
        let payload = sequence_header(true, true);
        let mut packet = vec![TEMPORAL_DELIMITER << 3 | 0b10, 0];
        packet.push(SEQUENCE_HEADER << 3 | 0b10);
        packet.extend_from_slice(&to_leb128(payload.len() as u64));
        packet.extend_from_slice(&payload);
        // A frame OBU without a size field runs to the end.
        packet.extend_from_slice(&[6 << 3, 0xAA, 0xBB]);
        let parsed = obus(&packet).unwrap();
        assert_eq!(
            parsed.iter().map(|obu| obu.kind).collect::<Vec<_>>(),
            [TEMPORAL_DELIMITER, SEQUENCE_HEADER, 6]
        );
        let sample = sample(&parsed);
        assert_eq!(sample[0], SEQUENCE_HEADER << 3 | 0b10);
        assert_eq!(&sample[sample.len() - 4..], &[6 << 3 | 0b10, 2, 0xAA, 0xBB]);

        let header = SequenceHeader::parse(parsed[1].payload).unwrap();
        let av1c = header.av1c(&parsed[1].sized());
        assert_eq!(&av1c[..4], &[0x81, 8, 0b0100_1101, 0]);
        assert!(obus(&[0x80]).is_none());
        assert!(obus(&[SEQUENCE_HEADER << 3 | 0b10, 5, 1]).is_none());
    }
}
