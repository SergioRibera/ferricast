//! A check of AV1 temporal units before they reach the parser.
//!
//! NVIDIA's AV1 parser trusts OBU sizes: a temporal unit whose last OBU
//! claims more bytes than the packet holds (a truncated packet) makes
//! `cuvidDecodePicture` fault inside the driver (SIGBUS, seen on driver 595)
//! instead of failing. It also faults inside `cuvidParseVideoData` on a
//! sequence header with damaged fields. So the decoder walks the OBU
//! headers of each packet, refuses a packet whose sizes do not add up, and
//! parses every sequence header in full (§5.5) to its trailing bits.

use super::nal::Bits;
use crate::error::{Error, Result};

/// Checks that every OBU header of `data` is well formed and its size fits.
pub(super) fn check_temporal_unit(data: &[u8]) -> Result<()> {
    let mut rest = data;
    while let Some((&header, tail)) = rest.split_first() {
        let invalid = |reason: &str| Error::InvalidPacket {
            reason: format!("AV1 OBU at byte {}: {reason}", data.len() - rest.len()),
        };
        if header & 0x80 != 0 {
            return Err(invalid("forbidden bit set"));
        }
        let extension = header & 0x04 != 0;
        let has_size = header & 0x02 != 0;
        let mut body = tail;
        if extension {
            body = body
                .get(1..)
                .ok_or_else(|| invalid("missing extension header"))?;
        }
        let kind = (header >> 3) & 0x0f;
        let check = |payload: &[u8]| {
            if kind == OBU_SEQUENCE_HEADER {
                check_sequence_header(payload)
                    .map_err(|reason| invalid(&format!("sequence header: {reason}")))?;
            }
            Ok(())
        };
        if !has_size {
            // Runs to the end of the temporal unit.
            return check(body);
        }
        let (size, read) = leb128(body).ok_or_else(|| invalid("bad size field"))?;
        let payload = body.get(read..).unwrap_or_default();
        let size = usize::try_from(size).map_err(|_| invalid("size overflows"))?;
        if size > payload.len() {
            return Err(invalid(&format!(
                "size {size} runs past the packet ({} bytes left)",
                payload.len()
            )));
        }
        check(&payload[..size])?;
        rest = payload.get(size..).unwrap_or_default();
    }
    Ok(())
}

/// `leb128()`: the value and the bytes it took (at most 8).
fn leb128(data: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0_u64;
    for (index, byte) in data.iter().take(8).enumerate() {
        value |= u64::from(byte & 0x7f) << (7 * index);
        if byte & 0x80 == 0 {
            return Some((value, index + 1));
        }
    }
    None
}

/// `obu_type` of a sequence header.
const OBU_SEQUENCE_HEADER: u8 = 1;
/// `seq_profile` values (Main, High, Professional).
const MAX_PROFILE: u32 = 2;
/// `seq_level_idx` 24-30 are reserved (31 means no level).
const RESERVED_LEVELS: std::ops::RangeInclusive<u32> = 24..=30;

/// Checks a sequence header OBU payload (AV1 §5.5): every field present
/// and in range, ending exactly at its `trailing_bits()`.
fn check_sequence_header(payload: &[u8]) -> std::result::Result<(), &'static str> {
    let mut bits = Bits::new(payload);
    let stop = bits.stop_bit().ok_or("no trailing bits")?;
    sequence_header(&mut bits).ok_or("cut short or out of range")?;
    if bits.position() == stop {
        Ok(())
    } else {
        Err("its syntax does not end at its trailing bits")
    }
}

/// `sequence_header_obu()`; `None` when the data ends first or a value is
/// out of range.
fn sequence_header(bits: &mut Bits<'_>) -> Option<()> {
    let profile = bits.bits(3).filter(|profile| *profile <= MAX_PROFILE)?;
    bits.skip(1)?; // still_picture
    let reduced = bits.flag()?;
    if reduced {
        level(bits)?;
    } else {
        operating_points(bits)?;
    }
    let width_bits = bits.bits(4)? + 1;
    let height_bits = bits.bits(4)? + 1;
    bits.bits(width_bits)?; // max_frame_width_minus_1
    bits.bits(height_bits)?; // max_frame_height_minus_1
    if !reduced && bits.flag()? {
        // delta_frame_id_length_minus_2, additional_frame_id_length_minus_1
        bits.skip(4 + 3)?;
    }
    // use_128x128_superblock, enable_filter_intra, enable_intra_edge_filter
    bits.skip(3)?;
    if !reduced {
        // enable_interintra_compound, enable_masked_compound,
        // enable_warped_motion, enable_dual_filter
        bits.skip(4)?;
        let order_hint = bits.flag()?;
        if order_hint {
            bits.skip(2)?; // enable_jnt_comp, enable_ref_frame_mvs
        }
        let force_screen_content = if bits.flag()? { 2 } else { bits.bits(1)? };
        if force_screen_content > 0 && !bits.flag()? {
            bits.skip(1)?; // seq_force_integer_mv
        }
        if order_hint {
            bits.skip(3)?; // order_hint_bits_minus_1
        }
    }
    // enable_superres, enable_cdef, enable_restoration
    bits.skip(3)?;
    colour_config(bits, profile)?;
    bits.skip(1) // film_grain_params_present
}

/// `seq_level_idx`, not a reserved one.
fn level(bits: &mut Bits<'_>) -> Option<u32> {
    bits.bits(5)
        .filter(|level| !RESERVED_LEVELS.contains(level))
}

/// Timing and decoder model information and the operating points.
fn operating_points(bits: &mut Bits<'_>) -> Option<()> {
    let mut decoder_model = false;
    let mut buffer_delay_bits = 0;
    if bits.flag()? {
        // timing_info(): num_units_in_display_tick, time_scale
        bits.skip(64)?;
        if bits.flag()? {
            uvlc(bits)?; // num_ticks_per_picture_minus_1
        }
        decoder_model = bits.flag()?;
        if decoder_model {
            buffer_delay_bits = bits.bits(5)? as usize + 1;
            // num_units_in_decoding_tick, buffer_removal_time_length_minus_1,
            // frame_presentation_time_length_minus_1
            bits.skip(32 + 5 + 5)?;
        }
    }
    let display_delay = bits.flag()?;
    for _ in 0..=bits.bits(5)? {
        bits.skip(12)?; // operating_point_idc
        if level(bits)? > 7 {
            bits.skip(1)?; // seq_tier
        }
        if decoder_model && bits.flag()? {
            // decoder_buffer_delay, encoder_buffer_delay, low_delay_mode_flag
            bits.skip(2 * buffer_delay_bits + 1)?;
        }
        if display_delay && bits.flag()? {
            bits.skip(4)?; // initial_display_delay_minus_1
        }
    }
    Some(())
}

/// `color_config()` (§5.5.2).
fn colour_config(bits: &mut Bits<'_>, profile: u32) -> Option<()> {
    let high_bitdepth = bits.flag()?;
    let twelve_bit = profile == 2 && high_bitdepth && bits.flag()?;
    let monochrome = profile != 1 && bits.flag()?;
    let (mut primaries, mut transfer, mut matrix) = (2, 2, 2);
    if bits.flag()? {
        (primaries, transfer, matrix) = (bits.bits(8)?, bits.bits(8)?, bits.bits(8)?);
    }
    if monochrome {
        return bits.skip(1); // color_range
    }
    let (subsampling_x, subsampling_y) = if (primaries, transfer, matrix) == (1, 13, 0) {
        // sRGB: 4:4:4, full range, only in profiles that have 4:4:4.
        if profile == 0 {
            return None;
        }
        (false, false)
    } else {
        bits.skip(1)?; // color_range
        match profile {
            0 => (true, true),
            1 => (false, false),
            _ if twelve_bit => {
                let x = bits.flag()?;
                (x, x && bits.flag()?)
            }
            _ => (true, false),
        }
    };
    if subsampling_x && subsampling_y {
        bits.skip(2)?; // chroma_sample_position
    }
    bits.skip(1) // separate_uv_delta_q
}

/// `uvlc()`: at most 32 leading zeros.
fn uvlc(bits: &mut Bits<'_>) -> Option<u32> {
    let mut zeros = 0;
    while !bits.flag()? {
        zeros += 1;
        if zeros >= 32 {
            return None;
        }
    }
    bits.bits(zeros)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_must_fit() {
        // Temporal delimiter (size 0), then a frame OBU of 3 bytes.
        let unit = [0x12, 0x00, 0x32, 0x03, 1, 2, 3];
        assert!(check_temporal_unit(&unit).is_ok());
        assert!(matches!(
            check_temporal_unit(&unit[..6]),
            Err(Error::InvalidPacket { .. })
        ));
        assert!(check_temporal_unit(&[0x92, 0x00]).is_err(), "forbidden bit");
        assert!(
            check_temporal_unit(&[0x32, 0x80]).is_err(),
            "unterminated size"
        );
        // Without a size field an OBU runs to the end.
        assert!(check_temporal_unit(&[0x30, 9, 9, 9]).is_ok());
        assert_eq!(leb128(&[0x80, 0x01]), Some((128, 2)));
    }

    #[test]
    fn sequence_headers_are_checked_to_their_trailing_bits() {
        // The sequence header of the `av1_8bit.ivf` fixture (Main, 256x144).
        let header = [
            0x00, 0x00, 0x00, 0x03, 0xbf, 0xfc, 0x78, 0xd5, 0xf3, 0x00, 0x80,
        ];
        assert_eq!(check_sequence_header(&header), Ok(()));
        for cut in 0..header.len() {
            assert!(
                check_sequence_header(&header[..cut]).is_err(),
                "cut at {cut}"
            );
        }
        let mut unit = vec![0x12, 0x00, 0x0a, 0x0b];
        unit.extend_from_slice(&header);
        assert!(check_temporal_unit(&unit).is_ok());
        // seq_profile 7.
        unit[4] = 0xe0;
        assert!(check_temporal_unit(&unit).is_err());
    }
}
