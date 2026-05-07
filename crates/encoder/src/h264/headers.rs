//! Build SPS / PPS NAL units that match a given VAAPI parameter
//! buffer set.
//!
//! H.264 §7.3.2.1 (SPS) and §7.3.2.2 (PPS). The values below mirror
//! what we ask the VAAPI driver to encode — they have to match the
//! `VAEncSequenceParameterBufferH264` / `VAEncPictureParameterBufferH264`
//! we hand to the driver, otherwise decoders downstream will reject
//! the stream with "non-existing SPS/PPS referenced".
//!
//! The shape of these builders is informed by Intel's
//! `libva-utils/encode/h264encode.c`, ffmpeg's
//! `libavcodec/vaapi_encode_h264.c` and ChromiumOS' cros-codecs
//! H.264 encoder.

use super::bitstream::{finalize_nal, BitWriter};

/// Profile IDCs we target. See §A.2 (profile definitions).
pub(super) mod profile {
    pub const BASELINE: u8 = 66;
    pub const MAIN: u8 = 77;
    pub const HIGH: u8 = 100;
}

/// Common subset of the H.264 sequence parameters we'll commit to.
#[derive(Clone, Copy)]
pub(super) struct SpsParams {
    pub profile_idc: u8,
    /// `constraint_set0_flag`..`constraint_set5_flag` packed into
    /// the top byte (§7.3.2.1.1, "constraint_setN_flag"). Bit 7 =
    /// constraint_set0_flag.
    pub constraint_flags: u8,
    pub level_idc: u8,
    /// Default 0; only one SPS today.
    pub seq_parameter_set_id: u32,
    /// In macroblocks (each 16 px). For 1920×1080: width=120,
    /// height=68 (1920/16=120, ceil(1080/16)=68 with frame_crop
    /// trimming the bottom 8 px).
    pub pic_width_in_mbs_minus1: u32,
    pub pic_height_in_map_units_minus1: u32,
    pub log2_max_frame_num_minus4: u32,
    pub log2_max_pic_order_cnt_lsb_minus4: u32,
    pub max_num_ref_frames: u32,
    /// Frame cropping (§7.4.2.1.1). Used when image height isn't a
    /// multiple of 16: e.g. 1080 needs 8 px crop from the bottom.
    pub frame_cropping: Option<FrameCrop>,
    /// VUI present — emit timing info / fixed-frame-rate so the
    /// player honours the encoder's clock instead of guessing.
    pub vui: Option<VuiParams>,
}

#[derive(Clone, Copy)]
pub(super) struct FrameCrop {
    pub left: u32,
    pub right: u32,
    pub top: u32,
    pub bottom: u32,
}

#[derive(Clone, Copy)]
pub(super) struct VuiParams {
    pub num_units_in_tick: u32,
    pub time_scale: u32,
    pub fixed_frame_rate_flag: bool,
}

/// Build the SPS NAL byte stream (Annex B). `nal_ref_idc=3` is the
/// canonical value for SPS (§7.4.1).
pub(super) fn build_sps(p: &SpsParams) -> Vec<u8> {
    let mut w = BitWriter::new();

    // profile_idc u(8)
    w.write_bits(p.profile_idc as u32, 8);
    // constraint_set0..5_flag u(1)*6 + reserved_zero_2bits u(2)
    w.write_bits(p.constraint_flags as u32, 8);
    // level_idc u(8)
    w.write_bits(p.level_idc as u32, 8);

    // seq_parameter_set_id ue(v)
    w.write_ue(p.seq_parameter_set_id);

    // Profile-specific seq fields. For Baseline / Main these aren't
    // emitted; only High and above carry chroma_format_idc, bit
    // depth, etc.
    if matches!(p.profile_idc, profile::HIGH | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128) {
        w.write_ue(1); // chroma_format_idc = 1 (4:2:0)
        w.write_ue(0); // bit_depth_luma_minus8
        w.write_ue(0); // bit_depth_chroma_minus8
        w.write_flag(false); // qpprime_y_zero_transform_bypass_flag
        w.write_flag(false); // seq_scaling_matrix_present_flag
    }

    // log2_max_frame_num_minus4 ue(v)
    w.write_ue(p.log2_max_frame_num_minus4);

    // pic_order_cnt_type ue(v) — we always use type 0
    w.write_ue(0);
    // log2_max_pic_order_cnt_lsb_minus4 ue(v) (only for type 0)
    w.write_ue(p.log2_max_pic_order_cnt_lsb_minus4);

    // num_ref_frames ue(v)
    w.write_ue(p.max_num_ref_frames);
    // gaps_in_frame_num_value_allowed_flag u(1)
    w.write_flag(false);

    // pic_width_in_mbs_minus1 ue(v)
    w.write_ue(p.pic_width_in_mbs_minus1);
    // pic_height_in_map_units_minus1 ue(v)
    w.write_ue(p.pic_height_in_map_units_minus1);
    // frame_mbs_only_flag u(1) — we always emit progressive
    w.write_flag(true);
    // direct_8x8_inference_flag u(1) — required when
    // frame_mbs_only_flag=1 and level >= 3.0
    w.write_flag(true);

    // frame_cropping_flag u(1) + offsets
    if let Some(c) = &p.frame_cropping {
        w.write_flag(true);
        w.write_ue(c.left);
        w.write_ue(c.right);
        w.write_ue(c.top);
        w.write_ue(c.bottom);
    } else {
        w.write_flag(false);
    }

    // vui_parameters_present_flag u(1)
    if let Some(v) = &p.vui {
        w.write_flag(true);
        write_vui(&mut w, v);
    } else {
        w.write_flag(false);
    }

    finalize_nal(w, /* nal_ref_idc = */ 3, /* nal_unit_type SPS = */ 7)
}

fn write_vui(w: &mut BitWriter, v: &VuiParams) {
    // aspect_ratio_info_present_flag u(1) = 0
    w.write_flag(false);
    // overscan_info_present_flag u(1) = 0
    w.write_flag(false);
    // video_signal_type_present_flag u(1) = 0
    w.write_flag(false);
    // chroma_loc_info_present_flag u(1) = 0
    w.write_flag(false);
    // timing_info_present_flag u(1) = 1
    w.write_flag(true);
    // num_units_in_tick u(32)
    w.write_bits(v.num_units_in_tick, 32);
    // time_scale u(32)
    w.write_bits(v.time_scale, 32);
    // fixed_frame_rate_flag u(1)
    w.write_flag(v.fixed_frame_rate_flag);
    // nal_hrd_parameters_present_flag u(1) = 0
    w.write_flag(false);
    // vcl_hrd_parameters_present_flag u(1) = 0
    w.write_flag(false);
    // pic_struct_present_flag u(1) = 0
    w.write_flag(false);
    // bitstream_restriction_flag u(1) = 0
    w.write_flag(false);
}

#[derive(Clone, Copy)]
pub(super) struct PpsParams {
    pub pic_parameter_set_id: u32,
    pub seq_parameter_set_id: u32,
    pub entropy_coding_mode_flag: bool,
    pub num_ref_idx_l0_default_active_minus1: u32,
    pub pic_init_qp_minus26: i32,
    pub deblocking_filter_control_present_flag: bool,
    pub transform_8x8_mode_flag: bool,
}

/// Build the PPS NAL byte stream. Single slice group, no scaling
/// matrices.
pub(super) fn build_pps(p: &PpsParams) -> Vec<u8> {
    let mut w = BitWriter::new();

    w.write_ue(p.pic_parameter_set_id);
    w.write_ue(p.seq_parameter_set_id);
    // entropy_coding_mode_flag u(1)
    w.write_flag(p.entropy_coding_mode_flag);
    // bottom_field_pic_order_in_frame_present_flag u(1)
    w.write_flag(false);
    // num_slice_groups_minus1 ue(v) = 0
    w.write_ue(0);
    // num_ref_idx_l0_default_active_minus1 ue(v)
    w.write_ue(p.num_ref_idx_l0_default_active_minus1);
    // num_ref_idx_l1_default_active_minus1 ue(v) = 0
    w.write_ue(0);
    // weighted_pred_flag u(1)
    w.write_flag(false);
    // weighted_bipred_idc u(2)
    w.write_bits(0, 2);
    // pic_init_qp_minus26 se(v)
    w.write_se(p.pic_init_qp_minus26);
    // pic_init_qs_minus26 se(v)
    w.write_se(0);
    // chroma_qp_index_offset se(v)
    w.write_se(0);
    // deblocking_filter_control_present_flag u(1)
    w.write_flag(p.deblocking_filter_control_present_flag);
    // constrained_intra_pred_flag u(1)
    w.write_flag(false);
    // redundant_pic_cnt_present_flag u(1)
    w.write_flag(false);

    // Profile-dependent extensions. We emit `transform_8x8_mode_flag`
    // only when a meaningful caller asks for it (High-profile
    // path). Constrained Baseline / Main keep this segment empty,
    // which matches what x264 emits in baseline.
    if p.transform_8x8_mode_flag {
        // more_rbsp_data: emit transform_8x8_mode_flag u(1) +
        // pic_scaling_matrix_present_flag u(1) +
        // second_chroma_qp_index_offset se(v).
        w.write_flag(true);
        w.write_flag(false); // pic_scaling_matrix_present_flag
        w.write_se(0); // second_chroma_qp_index_offset
    }

    finalize_nal(w, /* nal_ref_idc = */ 3, /* nal_unit_type PPS = */ 8)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: an SPS for 1920×1080 Baseline @ 30 fps should at
    /// least decode without error in a reference parser. We can't
    /// import a parser into unit tests, so just check the byte
    /// stream starts with the Annex B prefix + an SPS header.
    #[test]
    fn sps_annexb_prefix_and_type() {
        let sps = build_sps(&SpsParams {
            profile_idc: profile::BASELINE,
            // constraint_set1_flag = 1 → "Constrained Baseline"
            constraint_flags: 0b0100_0000,
            level_idc: 41,
            seq_parameter_set_id: 0,
            pic_width_in_mbs_minus1: 119, // 1920/16 - 1
            pic_height_in_map_units_minus1: 67, // ceil(1080/16) - 1 = 68 - 1
            log2_max_frame_num_minus4: 4,
            log2_max_pic_order_cnt_lsb_minus4: 4,
            max_num_ref_frames: 1,
            frame_cropping: Some(FrameCrop {
                left: 0,
                right: 0,
                top: 0,
                bottom: 4, // 4 * 2 = 8 px crop for 1080
            }),
            vui: Some(VuiParams {
                num_units_in_tick: 1,
                time_scale: 60,
                fixed_frame_rate_flag: true,
            }),
        });
        assert_eq!(&sps[..4], &[0x00, 0x00, 0x00, 0x01], "Annex B prefix");
        // header byte: nal_ref_idc=3 (top 2 bits=11 in 0bX_XXX_XXXXX after the
        // forbidden_zero_bit), nal_unit_type=7. So byte = (3<<5)|7 = 0x67.
        assert_eq!(sps[4], 0x67, "SPS NAL header");
    }

    #[test]
    fn pps_annexb_prefix_and_type() {
        let pps = build_pps(&PpsParams {
            pic_parameter_set_id: 0,
            seq_parameter_set_id: 0,
            entropy_coding_mode_flag: false, // CAVLC
            num_ref_idx_l0_default_active_minus1: 0,
            pic_init_qp_minus26: 0,
            deblocking_filter_control_present_flag: true,
            transform_8x8_mode_flag: false,
        });
        assert_eq!(&pps[..4], &[0x00, 0x00, 0x00, 0x01]);
        // (3<<5)|8 = 0x68
        assert_eq!(pps[4], 0x68);
    }
}
