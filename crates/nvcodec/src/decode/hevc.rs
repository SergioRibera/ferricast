//! The full syntax check of an HEVC video parameter set (H.265 §7.3.2.1),
//! for the parser that does not bounds-check it.
//!
//! Only the VPS: a truncated VPS makes NVIDIA's parser fault (SIGSEGV
//! inside `cuvidParseVideoData`, driver 595), while every truncation of an
//! SPS or PPS was refused cleanly (`CUDA_ERROR_INVALID_VALUE` or a sequence
//! the decoder refuses).

use super::nal::{Bits, unescape};

/// `MaxDpbSize`: the largest DPB of any level (§A.4.2).
const MAX_DPB_SIZE: u32 = 16;
/// `vps_num_layer_sets_minus1` is at most 1023.
const MAX_LAYER_SETS_MINUS1: u32 = 1023;
/// `nuh_layer_id` values are 0-62; 63 is reserved.
const MAX_LAYER_ID: u32 = 62;
/// `cpb_cnt_minus1` is at most 31.
const MAX_CPB_CNT_MINUS1: u32 = 31;
/// `elemental_duration_in_tc_minus1` is at most 2047.
const MAX_ELEMENTAL_DURATION: u32 = 2047;
/// The most values a `ue(v)` of a 32-bit count may take (`2^32 - 2`).
const MAX_UE_32: u32 = u32::MAX - 1;

/// Checks `unit` (the NAL unit from its 2-byte header to the next start
/// code) as a VPS: every syntax element present and in range, ending
/// exactly at `rbsp_trailing_bits()`. `rbsp` is scratch space.
pub(super) fn check_vps(unit: &[u8], rbsp: &mut Vec<u8>) -> Result<(), String> {
    let end = unit
        .iter()
        .rposition(|&byte| byte != 0)
        .map_or(0, |last| last + 1);
    let Some((header, payload)) = unit[..end].split_first_chunk::<2>() else {
        return Err("shorter than its NAL unit header".into());
    };
    let (forbidden_zero_bit, temporal_id_plus1) = (header[0] >> 7, header[1] & 0x07);
    if forbidden_zero_bit != 0 || temporal_id_plus1 == 0 {
        return Err("invalid NAL unit header".into());
    }
    if !unescape(payload, rbsp) {
        return Err("a byte sequence no NAL unit may contain".into());
    }
    let mut bits = Bits::new(rbsp);
    let stop = bits.stop_bit().ok_or("no rbsp_trailing_bits()")?;
    let extension = vps(&mut bits).ok_or("cut short or out of range")?;
    let position = bits.position();
    if position > stop || (!extension && position != stop) {
        return Err(format!(
            "its syntax ends at bit {position}, its rbsp_trailing_bits() at bit {stop}"
        ));
    }
    Ok(())
}

/// `video_parameter_set_rbsp()` up to `vps_extension_flag`, which it
/// returns. `None` when the data ends first or a value is out of range.
fn vps(bits: &mut Bits<'_>) -> Option<bool> {
    bits.skip(4)?; // vps_video_parameter_set_id
    let base_layer_internal = bits.flag()?;
    bits.skip(1)?; // vps_base_layer_available_flag
    bits.bits(6).filter(|layers| *layers <= MAX_LAYER_ID)?; // vps_max_layers_minus1
    let max_sub_layers_minus1 = bits.bits(3).filter(|count| *count <= 6)?;
    bits.skip(1 + 16)?; // vps_temporal_id_nesting_flag, vps_reserved_0xffff_16bits
    profile_tier_level(bits, max_sub_layers_minus1)?;
    let ordering_for_all = bits.flag()?;
    let first = if ordering_for_all {
        0
    } else {
        max_sub_layers_minus1
    };
    for _ in first..=max_sub_layers_minus1 {
        let buffering = bits.ue_max(MAX_DPB_SIZE - 1)?;
        bits.ue_max(buffering)?; // vps_max_num_reorder_pics
        bits.ue_max(MAX_UE_32)?; // vps_max_latency_increase_plus1
    }
    let max_layer_id = bits.bits(6).filter(|id| *id <= MAX_LAYER_ID)?;
    let layer_sets_minus1 = bits.ue_max(MAX_LAYER_SETS_MINUS1)?;
    let included = usize::try_from(layer_sets_minus1).ok()? * (max_layer_id as usize + 1);
    bits.skip(included)?; // layer_id_included_flag[i][j]
    if bits.flag()? {
        // vps_timing_info_present_flag
        bits.skip(64)?; // vps_num_units_in_tick, vps_time_scale
        if bits.flag()? {
            bits.ue_max(MAX_UE_32)?; // vps_num_ticks_poc_diff_one_minus1
        }
        let hrd_count = bits.ue_max(layer_sets_minus1 + 1)?;
        let first_set = u32::from(!base_layer_internal);
        for index in 0..hrd_count {
            let set = bits.ue()?;
            if !(first_set..=layer_sets_minus1).contains(&set) {
                return None;
            }
            let common = index == 0 || bits.flag()?;
            hrd_parameters(bits, common, max_sub_layers_minus1)?;
        }
    }
    bits.flag()
}

/// `profile_tier_level(1, maxNumSubLayersMinus1)` (§7.3.3).
fn profile_tier_level(bits: &mut Bits<'_>, max_sub_layers_minus1: u32) -> Option<()> {
    bits.skip(88 + 8)?; // general profile, tier, flags; general_level_idc
    let mut present = [(false, false); 6];
    for layer in present.iter_mut().take(max_sub_layers_minus1 as usize) {
        *layer = (bits.flag()?, bits.flag()?);
    }
    if max_sub_layers_minus1 > 0 {
        bits.skip(2 * (8 - max_sub_layers_minus1 as usize))?; // reserved_zero_2bits
    }
    for (profile, level) in present.iter().take(max_sub_layers_minus1 as usize) {
        if *profile {
            bits.skip(88)?;
        }
        if *level {
            bits.skip(8)?;
        }
    }
    Some(())
}

/// `hrd_parameters()` (§E.2.2).
fn hrd_parameters(bits: &mut Bits<'_>, common: bool, max_sub_layers_minus1: u32) -> Option<()> {
    let (mut nal, mut vcl, mut sub_picture) = (false, false, false);
    if common {
        nal = bits.flag()?;
        vcl = bits.flag()?;
        if nal || vcl {
            sub_picture = bits.flag()?;
            if sub_picture {
                // tick_divisor_minus2, du_cpb_removal_delay_increment_length_minus1,
                // sub_pic_cpb_params_in_pic_timing_sei_flag,
                // dpb_output_delay_du_length_minus1
                bits.skip(8 + 5 + 1 + 5)?;
            }
            bits.skip(4 + 4)?; // bit_rate_scale, cpb_size_scale
            if sub_picture {
                bits.skip(4)?; // cpb_size_du_scale
            }
            // initial_cpb_removal_delay_length_minus1,
            // au_cpb_removal_delay_length_minus1, dpb_output_delay_length_minus1
            bits.skip(5 + 5 + 5)?;
        }
    }
    for _ in 0..=max_sub_layers_minus1 {
        let fixed_general = bits.flag()?;
        let fixed_within_cvs = fixed_general || bits.flag()?;
        let mut low_delay = false;
        if fixed_within_cvs {
            bits.ue_max(MAX_ELEMENTAL_DURATION)?;
        } else {
            low_delay = bits.flag()?;
        }
        let cpb_count = if low_delay {
            1
        } else {
            bits.ue_max(MAX_CPB_CNT_MINUS1)? + 1
        };
        for present in [nal, vcl] {
            if present {
                sub_layer_hrd_parameters(bits, cpb_count, sub_picture)?;
            }
        }
    }
    Some(())
}

/// `sub_layer_hrd_parameters()` (§E.2.3).
fn sub_layer_hrd_parameters(bits: &mut Bits<'_>, cpb_count: u32, sub_picture: bool) -> Option<()> {
    for _ in 0..cpb_count {
        bits.ue_max(MAX_UE_32)?; // bit_rate_value_minus1
        bits.ue_max(MAX_UE_32)?; // cpb_size_value_minus1
        if sub_picture {
            bits.ue_max(MAX_UE_32)?; // cpb_size_du_value_minus1
            bits.ue_max(MAX_UE_32)?; // bit_rate_du_value_minus1
        }
        bits.skip(1)?; // cbr_flag
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The VPS of the `hevc_main8.hevc` fixture.
    const VPS: [u8; 24] = [
        0x40, 0x01, 0x0c, 0x01, 0xff, 0xff, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x03, 0x00, 0x3c, 0x95, 0x94, 0x09,
    ];

    #[test]
    fn a_real_vps_passes_and_every_truncation_fails() {
        let mut rbsp = Vec::new();
        assert_eq!(check_vps(&VPS, &mut rbsp), Ok(()));
        // Trailing zero bytes belong to the byte stream.
        let mut padded = VPS.to_vec();
        padded.extend_from_slice(&[0, 0]);
        assert_eq!(check_vps(&padded, &mut rbsp), Ok(()));
        for cut in 0..VPS.len() {
            assert!(check_vps(&VPS[..cut], &mut rbsp).is_err(), "cut at {cut}");
        }
    }

    #[test]
    fn out_of_range_values_fail() {
        let mut rbsp = Vec::new();
        // vps_max_sub_layers_minus1 = 7.
        let mut sub_layers = VPS;
        sub_layers[3] |= 0x0e;
        assert!(check_vps(&sub_layers, &mut rbsp).is_err());
        // Data after the stop bit without vps_extension_flag.
        let mut longer = VPS.to_vec();
        *longer.last_mut().unwrap() = 0x08;
        longer.push(0x80);
        assert!(check_vps(&longer, &mut rbsp).is_err());
    }
}
