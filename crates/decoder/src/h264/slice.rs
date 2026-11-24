use ferricast_core::{ColourDescription, FerricastError, Result};

#[derive(Debug, Clone, PartialEq)]
pub(super) enum PocKind {
    Lsb {
        log2_max_lsb: u32,
    },
    Cycle {
        delta_always_zero: bool,
        offset_for_non_ref_pic: i32,
        offset_for_top_to_bottom: i32,
        offsets_for_ref_frame: Vec<i32>,
    },
    FrameNum,
}

#[expect(clippy::struct_excessive_bools, reason = "mirrors the H.264 SPS flags")]
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Sps {
    pub profile_idc: u8,
    pub level_idc: u8,
    pub constraint_set3: bool,
    pub chroma_format_idc: u8,
    pub separate_colour_plane: bool,
    pub bit_depth_luma: u8,
    pub bit_depth_chroma: u8,
    pub log2_max_frame_num: u32,
    pub poc: PocKind,
    pub max_num_ref_frames: u32,
    pub gaps_allowed: bool,
    pub frame_mbs_only: bool,
    pub width_mbs: u32,
    pub height_mbs: u32,
    pub direct_8x8_inference: bool,
    pub crop: (u32, u32, u32, u32),
    pub num_reorder_frames: Option<u32>,
    pub max_dec_frame_buffering: Option<u32>,
    pub scaling_matrix_present: bool,
    /// The VUI colour signalling; unspecified when the SPS has none.
    pub colour: ColourDescription,
}

#[expect(clippy::struct_excessive_bools, reason = "mirrors the H.264 PPS flags")]
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Pps {
    pub sps_id: u32,
    pub entropy_cabac: bool,
    pub bottom_field_poc_present: bool,
    pub slice_groups: bool,
    pub num_ref_idx_default: [u32; 2],
    pub weighted_pred: bool,
    pub weighted_bipred_idc: u8,
    pub pic_init_qp_minus26: i32,
    pub pic_init_qs_minus26: i32,
    pub chroma_qp_index_offset: i32,
    pub second_chroma_qp_index_offset: i32,
    pub deblocking_control_present: bool,
    pub constrained_intra_pred: bool,
    pub redundant_pic_cnt_present: bool,
    pub transform_8x8_mode: bool,
    pub scaling_matrix_present: bool,
}

/// Largest picture dimension the backend accepts, in macroblocks
/// (8192 pixels, the H.264 level 6.2 ceiling).
pub(super) const MAX_DIMENSION_MBS: u32 = 8192 / 16;
/// Spec ceiling for `max_num_ref_frames` and the DPB size (A.3.1).
pub(super) const MAX_DPB_FRAMES: u32 = 16;
/// Spec ceiling for `log2_max_frame_num` and `log2_max_pic_order_cnt_lsb`.
const MAX_LOG2: u32 = 16;
/// Spec ceiling for `num_ref_frames_in_pic_order_cnt_cycle`.
const MAX_POC_CYCLE: usize = 255;

/// Why a stream cannot go through the VA-API backend even though it is
/// valid H.264; another backend may still decode it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Unsupported(pub &'static str);

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VA-API H.264: {} is not supported", self.0)
    }
}

impl std::error::Error for Unsupported {}

impl Sps {
    pub(super) fn coded_size(&self) -> (u32, u32) {
        (
            self.width_mbs.saturating_mul(16),
            self.height_mbs.saturating_mul(16),
        )
    }

    /// Crop units are two samples in both directions: [`Self::check_supported`]
    /// only admits progressive 4:2:0.
    pub(super) fn visible_rect(&self) -> (u32, u32, u32, u32) {
        let (left, right, top, bottom) = self.crop;
        let (width, height) = self.coded_size();
        let units = |a: u32, b: u32| a.saturating_add(b).saturating_mul(2);
        (
            left.saturating_mul(2),
            top.saturating_mul(2),
            width.saturating_sub(units(left, right)),
            height.saturating_sub(units(top, bottom)),
        )
    }

    pub(super) fn chroma_array_type(&self) -> u8 {
        if self.separate_colour_plane {
            0
        } else {
            self.chroma_format_idc
        }
    }

    /// Features the VA-API path does not implement. Failing here does not
    /// make the stream broken: a software decoder can take over.
    pub(super) fn check_supported(&self, pps: &Pps) -> std::result::Result<(), Unsupported> {
        let unsupported = |what: &'static str| Err(Unsupported(what));
        if !self.frame_mbs_only {
            return unsupported("interlaced coding");
        }
        if self.chroma_format_idc != 1 || self.separate_colour_plane {
            return unsupported("anything but 4:2:0");
        }
        if self.bit_depth_luma != 8 || self.bit_depth_chroma != 8 {
            return unsupported("bit depths above 8");
        }
        if pps.slice_groups {
            return unsupported("flexible macroblock ordering");
        }
        if self.scaling_matrix_present || pps.scaling_matrix_present {
            return unsupported("scaling matrices");
        }
        if self.width_mbs > MAX_DIMENSION_MBS || self.height_mbs > MAX_DIMENSION_MBS {
            return unsupported("pictures larger than 8192 pixels");
        }
        Ok(())
    }

    /// Range checks on stream-controlled values that later feed DPB and
    /// buffer arithmetic. Failing here means the SPS is corrupt.
    pub(super) fn check_valid(&self) -> Result<()> {
        let invalid = |what: String| Err(FerricastError::Decode(format!("H.264 SPS: {what}")));
        if self.width_mbs == 0 || self.height_mbs == 0 {
            return invalid("zero-sized picture".into());
        }
        if self.max_num_ref_frames > MAX_DPB_FRAMES {
            return invalid(format!(
                "max_num_ref_frames {} exceeds {MAX_DPB_FRAMES}",
                self.max_num_ref_frames
            ));
        }
        if !(4..=MAX_LOG2).contains(&self.log2_max_frame_num) {
            return invalid(format!(
                "log2_max_frame_num {} outside 4..=16",
                self.log2_max_frame_num
            ));
        }
        match &self.poc {
            PocKind::Lsb { log2_max_lsb } if !(4..=MAX_LOG2).contains(log2_max_lsb) => {
                return invalid(format!(
                    "log2_max_pic_order_cnt_lsb {log2_max_lsb} outside 4..=16"
                ));
            }
            PocKind::Cycle {
                offsets_for_ref_frame,
                ..
            } if offsets_for_ref_frame.len() > MAX_POC_CYCLE => {
                return invalid(format!(
                    "{} reference frames in the POC cycle exceed {MAX_POC_CYCLE}",
                    offsets_for_ref_frame.len()
                ));
            }
            PocKind::Lsb { .. } | PocKind::Cycle { .. } | PocKind::FrameNum => {}
        }
        let (width, height) = self.coded_size();
        let (left, right, top, bottom) = self.crop;
        let inside = |a: u32, b: u32, size: u32| {
            a.checked_add(b)
                .and_then(|sum| sum.checked_mul(2))
                .is_some_and(|cropped| cropped < size)
        };
        if !inside(left, right, width) || !inside(top, bottom, height) {
            return invalid(format!(
                "crop {:?} leaves nothing of the {width}x{height} picture",
                self.crop
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum SliceType {
    P,
    B,
    #[default]
    I,
    Sp,
    Si,
}

impl SliceType {
    pub(super) fn code(self) -> u8 {
        match self {
            SliceType::P => 0,
            SliceType::B => 1,
            SliceType::I => 2,
            SliceType::Sp => 3,
            SliceType::Si => 4,
        }
    }

    fn from_code(code: u32) -> Self {
        match code % 5 {
            0 => SliceType::P,
            1 => SliceType::B,
            2 => SliceType::I,
            3 => SliceType::Sp,
            _ => SliceType::Si,
        }
    }

    pub(super) fn is_inter(self) -> bool {
        matches!(self, SliceType::P | SliceType::B | SliceType::Sp)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RefMod {
    ShortMinus(u32),
    ShortPlus(u32),
    Long(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Mmco {
    UnmarkShort {
        diff_minus1: u32,
    },
    UnmarkLong {
        long_term_pic_num: u32,
    },
    ShortToLong {
        diff_minus1: u32,
        long_term_idx: u32,
    },
    MaxLongTermIdxPlus1(u32),
    ClearAll,
    CurrentToLong {
        long_term_idx: u32,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct Marking {
    pub no_output_of_prior_pics: bool,
    pub long_term_reference: bool,
    pub adaptive: Option<Vec<Mmco>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Weights {
    pub luma_log2_denom: u8,
    pub chroma_log2_denom: u8,
    pub has_chroma: bool,
    pub luma: [[(i16, i16); 32]; 2],
    pub chroma: [[[(i16, i16); 2]; 32]; 2],
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            luma_log2_denom: 0,
            chroma_log2_denom: 0,
            has_chroma: false,
            luma: [[(1, 0); 32]; 2],
            chroma: [[[(1, 0); 2]; 32]; 2],
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct SliceHeader {
    pub nal_ref_idc: u8,
    pub idr: bool,
    pub first_mb_in_slice: u32,
    pub slice_type: SliceType,
    pub pps_id: u32,
    pub frame_num: u32,
    pub idr_pic_id: u32,
    pub pic_order_cnt_lsb: u32,
    pub delta_pic_order_cnt_bottom: i32,
    pub delta_pic_order_cnt: [i32; 2],
    pub direct_spatial_mv_pred: bool,
    pub num_ref_idx_active: [u32; 2],
    pub ref_mods: [Vec<RefMod>; 2],
    pub weights: Option<Weights>,
    pub marking: Marking,
    pub cabac_init_idc: u8,
    pub slice_qp_delta: i32,
    pub disable_deblocking_filter_idc: u8,
    pub alpha_c0_offset_div2: i8,
    pub beta_offset_div2: i8,
    pub header_bits: u32,
}

pub(super) struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit_pos: 0 }
    }

    pub(super) fn bit_pos(&self) -> usize {
        self.bit_pos
    }

    pub(super) fn read_bit(&mut self) -> Result<u8> {
        let byte = self
            .bytes
            .get(self.bit_pos / 8)
            .ok_or_else(|| FerricastError::Decode("H.264 slice header ended early".into()))?;
        let bit = (byte >> (7 - self.bit_pos % 8)) & 1;
        self.bit_pos += 1;
        Ok(bit)
    }

    pub(super) fn read_flag(&mut self) -> Result<bool> {
        Ok(self.read_bit()? != 0)
    }

    pub(super) fn read_bits(&mut self, count: u32) -> Result<u32> {
        let mut value = 0;
        for _ in 0..count {
            value = value << 1 | u32::from(self.read_bit()?);
        }
        Ok(value)
    }

    pub(super) fn read_ue(&mut self) -> Result<u32> {
        let mut zeros = 0;
        while self.read_bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return Err(FerricastError::Decode("H.264 ue(v) is too long".into()));
            }
        }
        if zeros == 0 {
            return Ok(0);
        }
        Ok((1 << zeros) - 1 + self.read_bits(zeros)?)
    }

    pub(super) fn read_se(&mut self) -> Result<i32> {
        let code = self.read_ue()?;
        Ok(if code & 1 == 1 {
            (code / 2 + 1).cast_signed()
        } else {
            -((code / 2).cast_signed())
        })
    }

    /// A `ue(v)` syntax element that must fit `T` (`what` names it in the error).
    pub(super) fn read_ue_as<T: TryFrom<u32>>(&mut self, what: &str) -> Result<T> {
        let value = self.read_ue()?;
        T::try_from(value)
            .map_err(|_| FerricastError::Decode(format!("H.264 {what} {value} is out of range")))
    }

    /// An `se(v)` syntax element that must fit `T` (`what` names it in the error).
    pub(super) fn read_se_as<T: TryFrom<i32>>(&mut self, what: &str) -> Result<T> {
        let value = self.read_se()?;
        T::try_from(value)
            .map_err(|_| FerricastError::Decode(format!("H.264 {what} {value} is out of range")))
    }
}

pub(super) fn sps_has_scaling_matrix(rbsp: &[u8]) -> bool {
    let Some(&profile) = rbsp.first() else {
        return false;
    };
    if !matches!(
        profile,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        return false;
    }
    let mut r = BitReader::new(&rbsp[3.min(rbsp.len())..]);
    let present = (|| -> Result<bool> {
        r.read_ue()?;
        if r.read_ue()? == 3 {
            r.read_bit()?;
        }
        r.read_ue()?;
        r.read_ue()?;
        r.read_bit()?;
        r.read_flag()
    })();
    present.unwrap_or(false)
}

pub(super) fn peek_pps_id(rbsp: &[u8]) -> Result<u32> {
    let mut reader = BitReader::new(rbsp);
    reader.read_ue()?;
    reader.read_ue()?;
    reader.read_ue()
}

pub(super) fn parse(rbsp: &[u8], nal_header: u8, sps: &Sps, pps: &Pps) -> Result<SliceHeader> {
    let mut r = BitReader::new(rbsp);
    let nal_unit_type = nal_header & 0x1f;
    let mut header = SliceHeader {
        nal_ref_idc: (nal_header >> 5) & 3,
        idr: nal_unit_type == 5,
        first_mb_in_slice: r.read_ue()?,
        slice_type: SliceType::from_code(r.read_ue()?),
        pps_id: r.read_ue()?,
        ..SliceHeader::default()
    };
    if sps.separate_colour_plane {
        r.read_bits(2)?;
    }
    header.frame_num = r.read_bits(sps.log2_max_frame_num)?;
    if !sps.frame_mbs_only && r.read_flag()? {
        return Err(FerricastError::Decode(
            "VA-API H.264: field pictures are not supported".into(),
        ));
    }
    if header.idr {
        header.idr_pic_id = r.read_ue()?;
    }
    match &sps.poc {
        PocKind::Lsb { log2_max_lsb } => {
            header.pic_order_cnt_lsb = r.read_bits(*log2_max_lsb)?;
            if pps.bottom_field_poc_present {
                header.delta_pic_order_cnt_bottom = r.read_se()?;
            }
        }
        PocKind::Cycle {
            delta_always_zero: false,
            ..
        } => {
            header.delta_pic_order_cnt[0] = r.read_se()?;
            if pps.bottom_field_poc_present {
                header.delta_pic_order_cnt[1] = r.read_se()?;
            }
        }
        _ => {}
    }
    if pps.redundant_pic_cnt_present {
        r.read_ue()?;
    }
    let slice_type = header.slice_type;
    if slice_type == SliceType::B {
        header.direct_spatial_mv_pred = r.read_flag()?;
    }
    if slice_type.is_inter() {
        header.num_ref_idx_active = pps.num_ref_idx_default.map(|n| n.saturating_add(1));
        if r.read_flag()? {
            header.num_ref_idx_active[0] = r.read_ue()? + 1;
            if slice_type == SliceType::B {
                header.num_ref_idx_active[1] = r.read_ue()? + 1;
            }
        }
        if slice_type != SliceType::B {
            header.num_ref_idx_active[1] = 0;
        }
        if header.num_ref_idx_active.iter().any(|&n| n > 32) {
            return Err(FerricastError::Decode(
                "H.264 slice has more than 32 active references".into(),
            ));
        }
        header.ref_mods = parse_ref_mods(&mut r, slice_type)?;
        let weighted = (pps.weighted_pred && matches!(slice_type, SliceType::P | SliceType::Sp))
            || (pps.weighted_bipred_idc == 1 && slice_type == SliceType::B);
        if weighted {
            header.weights = Some(parse_weights(&mut r, sps, &header)?);
        }
    }
    if header.nal_ref_idc != 0 {
        header.marking = parse_marking(&mut r, header.idr)?;
    }
    if pps.entropy_cabac && slice_type.is_inter() {
        header.cabac_init_idc = r.read_ue_as("cabac_init_idc")?;
    }
    header.slice_qp_delta = r.read_se()?;
    if matches!(slice_type, SliceType::Sp | SliceType::Si) {
        if slice_type == SliceType::Sp {
            r.read_bit()?;
        }
        r.read_se()?;
    }
    if pps.deblocking_control_present {
        header.disable_deblocking_filter_idc = r.read_ue_as("disable_deblocking_filter_idc")?;
        if header.disable_deblocking_filter_idc != 1 {
            header.alpha_c0_offset_div2 = r.read_se_as("slice_alpha_c0_offset_div2")?;
            header.beta_offset_div2 = r.read_se_as("slice_beta_offset_div2")?;
        }
    }
    header.header_bits = u32::try_from(r.bit_pos())
        .map_err(|_| FerricastError::Decode("H.264 slice header is too long".into()))?;
    Ok(header)
}

fn parse_ref_mods(r: &mut BitReader, slice_type: SliceType) -> Result<[Vec<RefMod>; 2]> {
    let mut mods: [Vec<RefMod>; 2] = Default::default();
    let lists = if slice_type == SliceType::B { 2 } else { 1 };
    for list in mods.iter_mut().take(lists) {
        if !r.read_flag()? {
            continue;
        }
        loop {
            let operation = match r.read_ue()? {
                0 => RefMod::ShortMinus(r.read_ue()?),
                1 => RefMod::ShortPlus(r.read_ue()?),
                2 => RefMod::Long(r.read_ue()?),
                3 => break,
                other => {
                    return Err(FerricastError::Decode(format!(
                        "H.264 modification_of_pic_nums_idc {other}"
                    )));
                }
            };
            if list.len() >= 66 {
                return Err(FerricastError::Decode(
                    "H.264 slice has too many list modifications".into(),
                ));
            }
            list.push(operation);
        }
    }
    Ok(mods)
}

fn parse_weights(r: &mut BitReader, sps: &Sps, header: &SliceHeader) -> Result<Weights> {
    let mut weights = Weights {
        luma_log2_denom: r.read_ue_as("luma_log2_weight_denom")?,
        has_chroma: sps.chroma_array_type() != 0,
        ..Weights::default()
    };
    if weights.has_chroma {
        weights.chroma_log2_denom = r.read_ue_as("chroma_log2_weight_denom")?;
    }
    if weights.luma_log2_denom > 7 || weights.chroma_log2_denom > 7 {
        return Err(FerricastError::Decode(
            "H.264 weight denominator is out of range".into(),
        ));
    }
    let lists = if header.slice_type == SliceType::B {
        2
    } else {
        1
    };
    for list in 0..lists {
        for entry in 0..header.num_ref_idx_active[list] as usize {
            weights.luma[list][entry] = (1 << weights.luma_log2_denom, 0);
            if r.read_flag()? {
                weights.luma[list][entry] =
                    (r.read_se_as("luma_weight")?, r.read_se_as("luma_offset")?);
            }
            if weights.has_chroma {
                weights.chroma[list][entry] = [(1 << weights.chroma_log2_denom, 0); 2];
                if r.read_flag()? {
                    for plane in &mut weights.chroma[list][entry] {
                        *plane = (
                            r.read_se_as("chroma_weight")?,
                            r.read_se_as("chroma_offset")?,
                        );
                    }
                }
            }
        }
    }
    Ok(weights)
}

fn parse_marking(r: &mut BitReader, idr: bool) -> Result<Marking> {
    let mut marking = Marking::default();
    if idr {
        marking.no_output_of_prior_pics = r.read_flag()?;
        marking.long_term_reference = r.read_flag()?;
        return Ok(marking);
    }
    if !r.read_flag()? {
        return Ok(marking);
    }
    let mut operations = Vec::new();
    loop {
        let operation = match r.read_ue()? {
            0 => break,
            1 => Mmco::UnmarkShort {
                diff_minus1: r.read_ue()?,
            },
            2 => Mmco::UnmarkLong {
                long_term_pic_num: r.read_ue()?,
            },
            3 => Mmco::ShortToLong {
                diff_minus1: r.read_ue()?,
                long_term_idx: r.read_ue()?,
            },
            4 => Mmco::MaxLongTermIdxPlus1(r.read_ue()?),
            5 => Mmco::ClearAll,
            6 => Mmco::CurrentToLong {
                long_term_idx: r.read_ue()?,
            },
            other => {
                return Err(FerricastError::Decode(format!(
                    "H.264 memory_management_control_operation {other}"
                )));
            }
        };
        if operations.len() >= 66 {
            return Err(FerricastError::Decode(
                "H.264 slice has too many marking operations".into(),
            ));
        }
        operations.push(operation);
    }
    marking.adaptive = Some(operations);
    Ok(marking)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    #[derive(Default)]
    pub(in crate::h264) struct BitWriter {
        bits: Vec<bool>,
    }

    impl BitWriter {
        pub(in crate::h264) fn bits(&mut self, value: u32, count: u32) -> &mut Self {
            for shift in (0..count).rev() {
                self.bits.push(value >> shift & 1 == 1);
            }
            self
        }

        pub(in crate::h264) fn flag(&mut self, value: bool) -> &mut Self {
            self.bits(u32::from(value), 1)
        }

        pub(in crate::h264) fn ue(&mut self, value: u32) -> &mut Self {
            let coded = value + 1;
            let length = 32 - coded.leading_zeros();
            self.bits(0, length - 1).bits(coded, length)
        }

        pub(in crate::h264) fn se(&mut self, value: i32) -> &mut Self {
            let code = if value > 0 {
                value.cast_unsigned() * 2 - 1
            } else {
                (-value).cast_unsigned() * 2
            };
            self.ue(code)
        }

        pub(in crate::h264) fn bytes(&self) -> Vec<u8> {
            let mut out = vec![0; self.bits.len().div_ceil(8) + 1];
            for (index, bit) in self.bits.iter().enumerate() {
                if *bit {
                    out[index / 8] |= 0x80 >> (index % 8);
                }
            }
            out
        }

        pub(in crate::h264) fn len(&self) -> usize {
            self.bits.len()
        }
    }

    pub(in crate::h264) fn sps() -> Sps {
        Sps {
            profile_idc: 100,
            level_idc: 40,
            constraint_set3: false,
            chroma_format_idc: 1,
            separate_colour_plane: false,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            log2_max_frame_num: 4,
            poc: PocKind::Lsb { log2_max_lsb: 6 },
            max_num_ref_frames: 3,
            gaps_allowed: false,
            frame_mbs_only: true,
            width_mbs: 120,
            height_mbs: 68,
            direct_8x8_inference: true,
            crop: (0, 0, 0, 4),
            num_reorder_frames: Some(2),
            max_dec_frame_buffering: Some(4),
            scaling_matrix_present: false,
            colour: ColourDescription::unspecified(ferricast_core::ColourRange::Limited),
        }
    }

    pub(in crate::h264) fn pps() -> Pps {
        Pps {
            sps_id: 0,
            entropy_cabac: true,
            bottom_field_poc_present: false,
            slice_groups: false,
            num_ref_idx_default: [0, 0],
            weighted_pred: false,
            weighted_bipred_idc: 0,
            pic_init_qp_minus26: 0,
            pic_init_qs_minus26: 0,
            chroma_qp_index_offset: 0,
            second_chroma_qp_index_offset: 0,
            deblocking_control_present: true,
            constrained_intra_pred: false,
            redundant_pic_cnt_present: false,
            transform_8x8_mode: true,
            scaling_matrix_present: false,
        }
    }

    #[test]
    fn an_idr_slice_reads_its_marking_flags_and_ends_after_the_deblocking_fields() {
        let mut w = BitWriter::default();
        w.ue(0).ue(7).ue(0).bits(0, 4).ue(5).bits(12, 6);
        w.flag(true).flag(false);
        w.se(-3).ue(0).se(2).se(-1);
        let header = parse(&w.bytes(), 0x65, &sps(), &pps()).unwrap();
        assert!(header.idr);
        assert_eq!(header.slice_type, SliceType::I);
        assert_eq!(header.idr_pic_id, 5);
        assert_eq!(header.pic_order_cnt_lsb, 12);
        assert!(header.marking.no_output_of_prior_pics);
        assert!(!header.marking.long_term_reference);
        assert_eq!(header.slice_qp_delta, -3);
        assert_eq!(
            (header.alpha_c0_offset_div2, header.beta_offset_div2),
            (2, -1)
        );
        assert_eq!(header.header_bits as usize, w.len());
    }

    #[test]
    fn a_p_slice_reads_its_override_modifications_and_marking_operations() {
        let mut w = BitWriter::default();
        w.ue(0).ue(5).ue(0).bits(7, 4).bits(20, 6);
        w.flag(true).ue(1);
        w.flag(true).ue(0).ue(2).ue(2).ue(1).ue(3);
        w.flag(true).ue(1).ue(0).ue(3).ue(1).ue(0).ue(4).ue(2).ue(0);
        w.ue(1).se(4).ue(1);
        let header = parse(&w.bytes(), 0x41, &sps(), &pps()).unwrap();
        assert_eq!(header.slice_type, SliceType::P);
        assert_eq!(header.frame_num, 7);
        assert_eq!(header.num_ref_idx_active, [2, 0]);
        assert_eq!(header.ref_mods[0], [RefMod::ShortMinus(2), RefMod::Long(1)]);
        assert_eq!(
            header.marking.adaptive,
            Some(vec![
                Mmco::UnmarkShort { diff_minus1: 0 },
                Mmco::ShortToLong {
                    diff_minus1: 1,
                    long_term_idx: 0
                },
                Mmco::MaxLongTermIdxPlus1(2),
            ])
        );
        assert_eq!(header.cabac_init_idc, 1);
        assert_eq!(header.slice_qp_delta, 4);
        assert_eq!(header.disable_deblocking_filter_idc, 1);
        assert_eq!(header.header_bits as usize, w.len());
    }

    #[test]
    fn explicit_weights_fill_defaults_for_entries_without_flags() {
        let mut weighted = pps();
        weighted.weighted_pred = true;
        weighted.num_ref_idx_default = [1, 0];
        let mut w = BitWriter::default();
        w.ue(0).ue(0).ue(0).bits(3, 4).bits(8, 6);
        w.flag(false).flag(false);
        w.ue(5).ue(3);
        w.flag(true).se(30).se(-4).flag(false);
        w.flag(false).flag(true).se(9).se(1).se(7).se(-2);
        w.ue(0).se(0).ue(1);
        let header = parse(&w.bytes(), 0x01, &sps(), &weighted).unwrap();
        let weights = header.weights.unwrap();
        assert_eq!((weights.luma_log2_denom, weights.chroma_log2_denom), (5, 3));
        assert_eq!(weights.luma[0][0], (30, -4));
        assert_eq!(weights.chroma[0][0], [(8, 0), (8, 0)]);
        assert_eq!(weights.luma[0][1], (32, 0));
        assert_eq!(weights.chroma[0][1], [(9, 1), (7, -2)]);
    }

    #[test]
    fn field_pictures_are_refused() {
        let mut interlaced = sps();
        interlaced.frame_mbs_only = false;
        let mut w = BitWriter::default();
        w.ue(0).ue(7).ue(0).bits(0, 4).flag(true).flag(false);
        assert!(parse(&w.bytes(), 0x65, &interlaced, &pps()).is_err());
    }

    #[test]
    fn a_high_profile_sps_reports_its_scaling_matrix_flag() {
        for (flag, expected) in [(false, false), (true, true)] {
            let mut w = BitWriter::default();
            w.bits(100, 8).bits(0, 8).bits(40, 8);
            w.ue(0).ue(1).ue(0).ue(0).flag(false).flag(flag);
            assert_eq!(sps_has_scaling_matrix(&w.bytes()), expected);
        }
        let mut main = BitWriter::default();
        main.bits(77, 8).bits(0, 8).bits(40, 8).ue(0);
        assert!(!sps_has_scaling_matrix(&main.bytes()));
    }

    #[test]
    fn the_pps_id_is_readable_before_the_rest_of_the_header() {
        let mut w = BitWriter::default();
        w.ue(3).ue(7).ue(9);
        assert_eq!(peek_pps_id(&w.bytes()).unwrap(), 9);
    }

    #[test]
    fn exp_golomb_codes_round_trip() {
        let mut w = BitWriter::default();
        w.ue(0).ue(1).ue(2).ue(254).se(0).se(1).se(-1).se(-40);
        let bytes = w.bytes();
        let mut r = BitReader::new(&bytes);
        assert_eq!(
            [
                r.read_ue().unwrap(),
                r.read_ue().unwrap(),
                r.read_ue().unwrap(),
                r.read_ue().unwrap()
            ],
            [0, 1, 2, 254]
        );
        assert_eq!(
            [
                r.read_se().unwrap(),
                r.read_se().unwrap(),
                r.read_se().unwrap(),
                r.read_se().unwrap()
            ],
            [0, 1, -1, -40]
        );
    }

    #[test]
    fn supported_streams_are_four_two_zero_eight_bit_progressive_without_fmo_or_scaling() {
        assert!(sps().check_supported(&pps()).is_ok());
        let mut tenbit = sps();
        tenbit.bit_depth_luma = 10;
        assert!(tenbit.check_supported(&pps()).is_err());
        let mut scaled = pps();
        scaled.scaling_matrix_present = true;
        assert!(sps().check_supported(&scaled).is_err());
        let mut fmo = pps();
        fmo.slice_groups = true;
        assert!(sps().check_supported(&fmo).is_err());
    }

    #[test]
    fn stream_controlled_sps_values_are_range_checked() {
        assert!(sps().check_valid().is_ok());
        let mut refs = sps();
        refs.max_num_ref_frames = 17;
        assert!(refs.check_valid().is_err());
        let mut crop = sps();
        crop.crop = (0, 0, 0, 544);
        assert!(crop.check_valid().is_err());
        crop.crop = (u32::MAX, u32::MAX, 0, 0);
        assert!(crop.check_valid().is_err());
        let mut frame_num = sps();
        frame_num.log2_max_frame_num = 17;
        assert!(frame_num.check_valid().is_err());
        let lsb = Sps {
            poc: PocKind::Lsb { log2_max_lsb: 40 },
            ..sps()
        };
        assert!(lsb.check_valid().is_err());
        let cycle = Sps {
            poc: PocKind::Cycle {
                delta_always_zero: false,
                offset_for_non_ref_pic: 0,
                offset_for_top_to_bottom: 0,
                offsets_for_ref_frame: vec![1; 256],
            },
            ..sps()
        };
        assert!(cycle.check_valid().is_err());
    }

    #[test]
    fn huge_pictures_are_unsupported_and_their_sizes_saturate() {
        let mut huge = sps();
        huge.width_mbs = u32::MAX;
        assert!(huge.check_supported(&pps()).is_err());
        assert_eq!(huge.coded_size().0, u32::MAX);
        huge.crop = (u32::MAX, u32::MAX, u32::MAX, u32::MAX);
        assert_eq!(huge.visible_rect().2, 0);
    }

    #[test]
    fn the_visible_rectangle_removes_the_cropped_rows() {
        assert_eq!(sps().coded_size(), (1920, 1088));
        assert_eq!(sps().visible_rect(), (0, 0, 1920, 1080));
    }
}
