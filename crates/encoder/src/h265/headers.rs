//! Build VPS / SPS / PPS NAL units matching the parameter buffers
//! the VA-API HEVC encoder is configured with.
//!
//! HEVC §7.3.2.1 (VPS), §7.3.2.2 (SPS), §7.3.2.3 (PPS). Scope is
//! deliberately limited to what the VA-API encoder actually emits:
//! Main profile (later Main10), single-layer, single-sublayer,
//! IPPP-only — no B-frames, no tiles, no SCC extensions, no scalability.
//! Every flag we don't need stays at its standards-default value so
//! the resulting bitstream is parseable by any conformant Main-profile
//! decoder (Chromecast Ultra, modern Android TV, hardware HEVC blocks
//! on RDNA2 / Ampere / etc).

use super::bitstream::{nal, BitWriter, finalize_nal};

/// Common parameters shared between VPS / SPS / PPS construction.
#[derive(Clone, Copy)]
pub(super) struct StreamParams {
    /// `general_profile_idc`: 1 = Main, 2 = Main10.
    pub profile_idc: u8,
    /// `general_level_idc`: Annex A levels times 30. 1080p60 = 4.0
    /// (level_idc 120) or 4.1 (123). 4K30 = 5.0 (150). 4K60 = 5.1 (153).
    pub level_idc: u8,
    /// `general_tier_flag`: 0 = Main tier, 1 = High tier. We stay on
    /// Main tier — every decoder we care about supports it; High tier
    /// only unlocks higher bitrate caps we don't approach.
    pub tier_flag: u8,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    /// Output of [`crate::h264::headers::FrameCrop`] equivalent —
    /// HEVC encodes the cropping window as offsets in *luma samples*
    /// (already in pixels, no 2× scaling like H.264's frame_crop).
    pub conformance_window: Option<ConformanceWindow>,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub max_num_ref_frames: u32,
}

/// Conformance cropping window in luma samples (§7.4.3.2.1).
#[derive(Clone, Copy)]
pub(super) struct ConformanceWindow {
    pub left: u32,
    pub right: u32,
    pub top: u32,
    pub bottom: u32,
}

/// HEVC general profile_tier_level (§7.3.3) at sub-layer 0.
/// Most compatibility-flag bits are 0; we only set the
/// `general_profile_compatibility_flag[profile_idc]` bit so old
/// decoders that pattern-match by compatibility flag still accept us.
fn write_profile_tier_level(w: &mut BitWriter, p: &StreamParams, max_sub_layers_minus1: u8) {
    // general_profile_space u(2) = 0
    w.write_bits(0, 2);
    // general_tier_flag u(1)
    w.write_bits(p.tier_flag as u32, 1);
    // general_profile_idc u(5)
    w.write_bits(p.profile_idc as u32, 5);
    // general_profile_compatibility_flag[32] u(1) each
    let mut compat = 0u32;
    if (p.profile_idc as usize) < 32 {
        compat |= 1 << (31 - p.profile_idc as usize);
    }
    w.write_bits(compat >> 16, 16);
    w.write_bits(compat & 0xffff, 16);
    // general_progressive_source_flag u(1) = 1
    w.write_bits(1, 1);
    // general_interlaced_source_flag u(1) = 0
    w.write_bits(0, 1);
    // general_non_packed_constraint_flag u(1) = 1
    w.write_bits(1, 1);
    // general_frame_only_constraint_flag u(1) = 1
    w.write_bits(1, 1);
    // general_reserved_zero_43bits — write zeros
    w.write_bits(0, 22);
    w.write_bits(0, 21);
    // general_inbld_flag (or reserved_zero_bit for Main) = 0
    w.write_bits(0, 1);
    // general_level_idc u(8)
    w.write_bits(p.level_idc as u32, 8);

    // sub_layer_profile_present_flag / sub_layer_level_present_flag
    // pairs for each sub-layer below the top. With max_sub_layers_minus1
    // == 0 we emit zero pairs but still must pad to byte boundary
    // before sub_layer_level_idc[]. The spec mandates a pad: 2 bits per
    // sub-layer up to 8, with reserved_zero_2bits[i] for i in
    // max_sub_layers_minus1..8.
    for _ in 0..max_sub_layers_minus1 {
        w.write_bits(0, 1); // sub_layer_profile_present_flag[i]
        w.write_bits(0, 1); // sub_layer_level_present_flag[i]
    }
    if max_sub_layers_minus1 > 0 {
        for _ in max_sub_layers_minus1..8 {
            w.write_bits(0, 2); // reserved_zero_2bits[i]
        }
    }
    // No sub-layer profile/level info to emit (all flags zero above).
}

/// Build a VPS for Main-profile, single-layer, single-sublayer stream.
pub(super) fn build_vps(p: &StreamParams) -> Vec<u8> {
    let mut w = BitWriter::new();
    // vps_video_parameter_set_id u(4) = 0
    w.write_bits(0, 4);
    // vps_base_layer_internal_flag u(1) = 1
    w.write_bits(1, 1);
    // vps_base_layer_available_flag u(1) = 1
    w.write_bits(1, 1);
    // vps_max_layers_minus1 u(6) = 0
    w.write_bits(0, 6);
    // vps_max_sub_layers_minus1 u(3) = 0
    w.write_bits(0, 3);
    // vps_temporal_id_nesting_flag u(1) = 1
    w.write_bits(1, 1);
    // vps_reserved_0xffff_16bits u(16) = 0xffff
    w.write_bits(0xffff, 16);

    write_profile_tier_level(&mut w, p, 0);

    // vps_sub_layer_ordering_info_present_flag u(1) = 0
    w.write_bits(0, 1);
    // vps_max_dec_pic_buffering_minus1[i] ue(v)
    w.write_ue(p.max_num_ref_frames);
    // vps_max_num_reorder_pics[i] ue(v) = 0 (IPPP, no reordering)
    w.write_ue(0);
    // vps_max_latency_increase_plus1[i] ue(v) = 0
    w.write_ue(0);

    // vps_max_layer_id u(6) = 0
    w.write_bits(0, 6);
    // vps_num_layer_sets_minus1 ue(v) = 0
    w.write_ue(0);
    // vps_timing_info_present_flag u(1) = 1
    w.write_bits(1, 1);
    // vps_num_units_in_tick u(32) = 1
    w.write_bits(1, 32);
    // vps_time_scale u(32) = fps
    w.write_bits(p.fps.max(1), 32);
    // vps_poc_proportional_to_timing_flag u(1) = 0
    w.write_bits(0, 1);
    // vps_num_hrd_parameters ue(v) = 0
    w.write_ue(0);
    // vps_extension_flag u(1) = 0
    w.write_bits(0, 1);

    w.rbsp_trailing();
    finalize_nal(nal::VPS_NUT, w.into_inner())
}

/// Build an SPS matching the encoder's surface dimensions, profile,
/// and IPPP / single-sublayer assumptions.
pub(super) fn build_sps(p: &StreamParams, min_cb_log2_minus3: u8, diff_max_min_cb_log2: u8) -> Vec<u8> {
    let mut w = BitWriter::new();
    // sps_video_parameter_set_id u(4) = 0
    w.write_bits(0, 4);
    // sps_max_sub_layers_minus1 u(3) = 0
    w.write_bits(0, 3);
    // sps_temporal_id_nesting_flag u(1) = 1
    w.write_bits(1, 1);

    write_profile_tier_level(&mut w, p, 0);

    // sps_seq_parameter_set_id ue(v) = 0
    w.write_ue(0);
    // chroma_format_idc ue(v) = 1 (4:2:0)
    w.write_ue(1);
    // pic_width_in_luma_samples ue(v)
    w.write_ue(p.width);
    // pic_height_in_luma_samples ue(v)
    w.write_ue(p.height);
    // conformance_window_flag u(1)
    if let Some(cw) = p.conformance_window {
        w.write_bits(1, 1);
        w.write_ue(cw.left);
        w.write_ue(cw.right);
        w.write_ue(cw.top);
        w.write_ue(cw.bottom);
    } else {
        w.write_bits(0, 1);
    }
    // bit_depth_luma_minus8 ue(v)
    w.write_ue(p.bit_depth_luma_minus8 as u32);
    // bit_depth_chroma_minus8 ue(v)
    w.write_ue(p.bit_depth_chroma_minus8 as u32);
    // log2_max_pic_order_cnt_lsb_minus4 ue(v) = 4
    w.write_ue(4);
    // sps_sub_layer_ordering_info_present_flag u(1) = 0
    w.write_bits(0, 1);
    // sps_max_dec_pic_buffering_minus1[0] ue(v)
    w.write_ue(p.max_num_ref_frames);
    // sps_max_num_reorder_pics[0] ue(v) = 0
    w.write_ue(0);
    // sps_max_latency_increase_plus1[0] ue(v) = 0
    w.write_ue(0);

    // log2_min_luma_coding_block_size_minus3 ue(v)
    w.write_ue(min_cb_log2_minus3 as u32);
    // log2_diff_max_min_luma_coding_block_size ue(v)
    w.write_ue(diff_max_min_cb_log2 as u32);
    // log2_min_luma_transform_block_size_minus2 ue(v) = 0 (4x4 TU min)
    w.write_ue(0);
    // log2_diff_max_min_luma_transform_block_size ue(v) = 3 (up to 32x32)
    w.write_ue(3);
    // max_transform_hierarchy_depth_inter ue(v) = 3
    w.write_ue(3);
    // max_transform_hierarchy_depth_intra ue(v) = 3
    w.write_ue(3);

    // scaling_list_enabled_flag u(1) = 0
    w.write_bits(0, 1);
    // amp_enabled_flag u(1) = 1
    w.write_bits(1, 1);
    // sample_adaptive_offset_enabled_flag u(1) = 1
    w.write_bits(1, 1);
    // pcm_enabled_flag u(1) = 0
    w.write_bits(0, 1);
    // num_short_term_ref_pic_sets ue(v) = 1
    w.write_ue(1);

    // ─ short_term_ref_pic_set(0) inline (§7.3.7) ──────────────
    // We emit a single ST RPS describing the IPPP P-frame
    // reference: one negative reference one frame back.
    // inter_ref_pic_set_prediction_flag is skipped because rpsIdx
    // == 0.
    // num_negative_pics ue(v) = 1
    w.write_ue(1);
    // num_positive_pics ue(v) = 0
    w.write_ue(0);
    // delta_poc_s0_minus1[0] ue(v) = 0 (delta = -1)
    w.write_ue(0);
    // used_by_curr_pic_s0_flag[0] u(1) = 1
    w.write_bits(1, 1);
    // ──────────────────────────────────────────────────────────

    // long_term_ref_pics_present_flag u(1) = 0
    w.write_bits(0, 1);
    // sps_temporal_mvp_enabled_flag u(1) = 1
    w.write_bits(1, 1);
    // strong_intra_smoothing_enabled_flag u(1) = 1
    w.write_bits(1, 1);

    // vui_parameters_present_flag u(1) = 1
    w.write_bits(1, 1);
    write_vui(&mut w, p);

    // sps_extension_present_flag u(1) = 0
    w.write_bits(0, 1);

    w.rbsp_trailing();
    finalize_nal(nal::SPS_NUT, w.into_inner())
}

/// VUI subset — just enough to publish timing info (num_units_in_tick /
/// time_scale / fixed_frame_rate_flag) and aspect ratio = square. The
/// HLS players downstream key off timing info to honour the encoder's
/// clock; without VUI they fall back to guessing from PTS deltas.
fn write_vui(w: &mut BitWriter, p: &StreamParams) {
    // aspect_ratio_info_present_flag u(1) = 0
    w.write_bits(0, 1);
    // overscan_info_present_flag u(1) = 0
    w.write_bits(0, 1);
    // video_signal_type_present_flag u(1) = 0
    w.write_bits(0, 1);
    // chroma_loc_info_present_flag u(1) = 0
    w.write_bits(0, 1);
    // neutral_chroma_indication_flag u(1) = 0
    w.write_bits(0, 1);
    // field_seq_flag u(1) = 0
    w.write_bits(0, 1);
    // frame_field_info_present_flag u(1) = 0
    w.write_bits(0, 1);
    // default_display_window_flag u(1) = 0
    w.write_bits(0, 1);
    // vui_timing_info_present_flag u(1) = 1
    w.write_bits(1, 1);
    // vui_num_units_in_tick u(32) = 1
    w.write_bits(1, 32);
    // vui_time_scale u(32) = fps
    w.write_bits(p.fps.max(1), 32);
    // vui_poc_proportional_to_timing_flag u(1) = 0
    w.write_bits(0, 1);
    // vui_hrd_parameters_present_flag u(1) = 0
    w.write_bits(0, 1);
    // bitstream_restriction_flag u(1) = 0
    w.write_bits(0, 1);
}

/// Build a minimal PPS for IPPP, single-tile, no transform skip.
pub(super) fn build_pps(init_qp_minus26: i32) -> Vec<u8> {
    let mut w = BitWriter::new();
    // pps_pic_parameter_set_id ue(v) = 0
    w.write_ue(0);
    // pps_seq_parameter_set_id ue(v) = 0
    w.write_ue(0);
    // dependent_slice_segments_enabled_flag u(1) = 0
    w.write_bits(0, 1);
    // output_flag_present_flag u(1) = 0
    w.write_bits(0, 1);
    // num_extra_slice_header_bits u(3) = 0
    w.write_bits(0, 3);
    // sign_data_hiding_enabled_flag u(1) = 0
    w.write_bits(0, 1);
    // cabac_init_present_flag u(1) = 0
    w.write_bits(0, 1);
    // num_ref_idx_l0_default_active_minus1 ue(v) = 0
    w.write_ue(0);
    // num_ref_idx_l1_default_active_minus1 ue(v) = 0
    w.write_ue(0);
    // init_qp_minus26 se(v)
    w.write_se(init_qp_minus26);
    // constrained_intra_pred_flag u(1) = 0
    w.write_bits(0, 1);
    // transform_skip_enabled_flag u(1) = 0
    w.write_bits(0, 1);
    // cu_qp_delta_enabled_flag u(1) = 0
    w.write_bits(0, 1);
    // pps_cb_qp_offset se(v) = 0
    w.write_se(0);
    // pps_cr_qp_offset se(v) = 0
    w.write_se(0);
    // pps_slice_chroma_qp_offsets_present_flag u(1) = 0
    w.write_bits(0, 1);
    // weighted_pred_flag u(1) = 0
    w.write_bits(0, 1);
    // weighted_bipred_flag u(1) = 0
    w.write_bits(0, 1);
    // transquant_bypass_enabled_flag u(1) = 0
    w.write_bits(0, 1);
    // tiles_enabled_flag u(1) = 0
    w.write_bits(0, 1);
    // entropy_coding_sync_enabled_flag u(1) = 0
    w.write_bits(0, 1);
    // pps_loop_filter_across_slices_enabled_flag u(1) = 1
    w.write_bits(1, 1);
    // deblocking_filter_control_present_flag u(1) = 0
    w.write_bits(0, 1);
    // pps_scaling_list_data_present_flag u(1) = 0
    w.write_bits(0, 1);
    // lists_modification_present_flag u(1) = 0
    w.write_bits(0, 1);
    // log2_parallel_merge_level_minus2 ue(v) = 0
    w.write_ue(0);
    // slice_segment_header_extension_present_flag u(1) = 0
    w.write_bits(0, 1);
    // pps_extension_present_flag u(1) = 0
    w.write_bits(0, 1);

    w.rbsp_trailing();
    finalize_nal(nal::PPS_NUT, w.into_inner())
}
