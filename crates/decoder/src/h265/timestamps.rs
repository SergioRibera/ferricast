//! Presentation timestamps for HEVC pictures.
//!
//! [`Headers`] reads what every HEVC backend needs to put a time on a
//! picture: the parameter sets and the first slice segment of each
//! picture, which say whether the picture is shown, which coded video
//! sequence it belongs to, its POC LSBs and whether a decoder skips it.
//!
//! `rust_h265` releases pictures in display order but carries no
//! timestamps, only each picture's POC. Every packet's pictures are noted
//! with the POC LSBs their first slice segment carries, and a released
//! picture takes the timestamp of the pending picture whose LSBs match.
//! Pictures the decoder drops (skipped RASL pictures, pictures before the
//! first IRAP, prior pictures a new sequence discards) are noticed when a
//! later one comes out, so they never shift the timestamps after them.

use std::collections::HashMap;

use h264_reader::rbsp::{BitRead, BitReader, BitReaderError};
use rust_h265::{NalUnit, NalUnitType};

/// Pictures noted at most. A stream whose pictures never come out cannot
/// grow the list without bound; the earliest are forgotten first.
#[cfg(feature = "hevc-decode")]
const MAX_PENDING: usize = 64;
/// `sps_max_sub_layers_minus1` ceiling (H.265 7.4.3.2.1).
const MAX_SUB_LAYERS_MINUS1: u8 = 6;
/// `log2_max_pic_order_cnt_lsb` ceiling (H.265 7.4.3.2.1).
const MAX_LOG2_POC_LSB: u32 = 16;

/// Parameter sets and sequence state read from the bitstream.
#[derive(Default)]
pub struct Headers {
    sps: HashMap<u32, Sps>,
    pps: HashMap<u32, Pps>,
    /// Coded video sequences started so far; POCs restart with each.
    sequence: u32,
    /// An end-of-sequence NAL unit came last, so a CRA starts a new
    /// sequence.
    after_sequence_end: bool,
    /// An IRAP picture came already: decoding can start.
    after_irap: bool,
    /// The latest IRAP picture has `NoRaslOutputFlag` set, so the RASL
    /// pictures that follow it are skipped (H.265 8.1.3).
    skipping_rasl: bool,
}

/// A picture whose first slice segment says it is shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Picture {
    pub(crate) sequence: u32,
    /// `None` when the slice header could not be read: its parameter sets
    /// were never seen.
    pub(crate) poc_lsb: Option<PocLsb>,
    /// A conforming decoder never outputs it: it precedes the first IRAP
    /// picture, or is a RASL picture of an IRAP that starts decoding.
    pub(crate) skipped: bool,
    /// `sps_max_num_reorder_pics` of the highest sub-layer, when the SPS
    /// could be read that far.
    pub(crate) max_num_reorder: Option<u32>,
}

/// What a slice header needs from the SPS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Sps {
    log2_max_poc_lsb: u32,
    separate_colour_plane: bool,
    max_num_reorder: Option<u32>,
}

/// What a slice header needs from the PPS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Pps {
    sps_id: u32,
    output_flag_present: bool,
    num_extra_slice_header_bits: u8,
}

/// `slice_pic_order_cnt_lsb` and the power of two it wraps at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PocLsb {
    value: u32,
    log2_max: u32,
}

impl PocLsb {
    #[cfg(feature = "hevc-decode")]
    fn matches(self, poc: i32) -> bool {
        i64::from(poc).rem_euclid(1 << self.log2_max) == i64::from(self.value)
    }
}

/// What the first slice segment of a picture says about its output.
enum Output {
    /// `pic_output_flag` is 0: the picture is decoded but never shown.
    Hidden,
    Shown {
        poc_lsb: PocLsb,
        max_num_reorder: Option<u32>,
    },
}

impl Headers {
    /// Notes the parameter sets in one packet and returns the pictures
    /// it starts that are meant to be shown.
    pub(crate) fn pictures(&mut self, nals: &[NalUnit<'_>]) -> Vec<Picture> {
        let mut pictures = Vec::new();
        // Enhancement layers are not decoded.
        for nal in nals.iter().filter(|nal| nal.nuh_layer_id == 0) {
            match nal.nal_unit_type {
                NalUnitType::Sps => {
                    if let Ok((id, sps)) = Sps::parse(&nal.rbsp) {
                        self.sps.insert(id, sps);
                    }
                }
                NalUnitType::Pps => {
                    if let Ok((id, pps)) = Pps::parse(&nal.rbsp) {
                        self.pps.insert(id, pps);
                    }
                }
                NalUnitType::Eos => self.after_sequence_end = true,
                kind if kind.is_vcl() => pictures.extend(self.slice(kind, &nal.rbsp)),
                _ => {}
            }
        }
        pictures
    }

    fn slice(&mut self, kind: NalUnitType, rbsp: &[u8]) -> Option<Picture> {
        let mut bits = BitReader::new(rbsp);
        // Only the first slice segment of a picture starts one.
        if !bits
            .read_bool("first_slice_segment_in_pic_flag")
            .unwrap_or(false)
        {
            return None;
        }
        let starts_sequence = matches!(
            kind,
            NalUnitType::IdrWRadl
                | NalUnitType::IdrNLp
                | NalUnitType::BlaWLp
                | NalUnitType::BlaWRadl
                | NalUnitType::BlaNLp
        ) || kind == NalUnitType::Cra && self.after_sequence_end;
        if starts_sequence || self.sequence == 0 {
            self.sequence += 1;
        }
        if kind.is_irap() {
            // IDR and BLA always discard their RASL pictures; a CRA does
            // when decoding starts at it (H.265 8.1.3).
            self.skipping_rasl = starts_sequence || !self.after_irap;
            self.after_irap = true;
        }
        self.after_sequence_end = false;
        let skipped = !self.after_irap
            || self.skipping_rasl && matches!(kind, NalUnitType::RaslN | NalUnitType::RaslR);
        let (poc_lsb, max_num_reorder) = match self.first_slice(kind, &mut bits) {
            Ok(Output::Hidden) => return None,
            Ok(Output::Shown {
                poc_lsb,
                max_num_reorder,
            }) => (Some(poc_lsb), max_num_reorder),
            Err(_) => (None, None),
        };
        Some(Picture {
            sequence: self.sequence,
            poc_lsb,
            skipped,
            max_num_reorder,
        })
    }

    /// Reads a first slice segment header up to `slice_pic_order_cnt_lsb`
    /// (H.265 7.3.6.1).
    fn first_slice(
        &self,
        kind: NalUnitType,
        bits: &mut impl BitRead,
    ) -> Result<Output, Unreadable> {
        if kind.is_irap() {
            bits.read_bool("no_output_of_prior_pics_flag")?;
        }
        let pps_id = bits.read_ue("slice_pic_parameter_set_id")?;
        let pps = self.pps.get(&pps_id).ok_or(Unreadable::ParameterSets)?;
        let sps = self.sps.get(&pps.sps_id).ok_or(Unreadable::ParameterSets)?;
        skip(bits, u32::from(pps.num_extra_slice_header_bits))?;
        bits.read_ue("slice_type")?;
        if pps.output_flag_present && !bits.read_bool("pic_output_flag")? {
            return Ok(Output::Hidden);
        }
        if sps.separate_colour_plane {
            bits.read_u8(2, "colour_plane_id")?;
        }
        let value = if kind.is_idr() {
            0
        } else {
            bits.read_u32(sps.log2_max_poc_lsb, "slice_pic_order_cnt_lsb")?
        };
        Ok(Output::Shown {
            poc_lsb: PocLsb {
                value,
                log2_max: sps.log2_max_poc_lsb,
            },
            max_num_reorder: sps.max_num_reorder,
        })
    }
}

/// Timestamps for a decoder that reports each released picture's POC.
#[cfg(feature = "hevc-decode")]
#[derive(Default)]
pub(super) struct Timestamps {
    headers: Headers,
    pending: Vec<Pending>,
}

/// A picture decoded from a packet and not released yet.
#[cfg(feature = "hevc-decode")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Pending {
    sequence: u32,
    /// `None` when the slice header could not be read.
    poc_lsb: Option<PocLsb>,
    timestamp_us: u64,
}

#[cfg(feature = "hevc-decode")]
impl Timestamps {
    /// Notes the parameter sets and the pictures in one packet.
    pub(super) fn observe(&mut self, nals: &[NalUnit<'_>], timestamp_us: u64) {
        for picture in self.headers.pictures(nals) {
            self.note(picture, timestamp_us);
        }
    }

    /// The timestamp of the picture with `poc` the decoder just released.
    ///
    /// Pictures come out in display order, so a pending picture of the
    /// same sequence with an earlier timestamp, or of an earlier sequence,
    /// can never come out after it: the decoder dropped them.
    pub(super) fn claim(&mut self, poc: i32) -> Option<u64> {
        let candidates = || self.pending.iter().enumerate();
        let earliest = |(_, pending): &(usize, &Pending)| (pending.sequence, pending.timestamp_us);
        let (index, _) = candidates()
            .filter(|(_, pending)| pending.poc_lsb.is_some_and(|lsb| lsb.matches(poc)))
            .min_by_key(earliest)
            .or_else(|| {
                candidates()
                    .filter(|(_, pending)| pending.poc_lsb.is_none())
                    .min_by_key(earliest)
            })?;
        let claimed = self.pending.swap_remove(index);
        self.pending.retain(|pending| {
            pending.sequence > claimed.sequence
                || pending.sequence == claimed.sequence
                    && pending.timestamp_us >= claimed.timestamp_us
        });
        Some(claimed.timestamp_us)
    }

    /// Skipped pictures are noted too: the POC match forgets them once a
    /// later picture comes out.
    fn note(&mut self, picture: Picture, timestamp_us: u64) {
        if self.pending.len() >= MAX_PENDING
            && let Some((index, _)) = self
                .pending
                .iter()
                .enumerate()
                .min_by_key(|(_, pending)| (pending.sequence, pending.timestamp_us))
        {
            self.pending.swap_remove(index);
        }
        self.pending.push(Pending {
            sequence: picture.sequence,
            poc_lsb: picture.poc_lsb,
            timestamp_us,
        });
    }
}

impl Sps {
    /// Reads an SPS up to `sps_max_num_reorder_pics`
    /// (H.265 7.3.2.2.1), returning its id.
    fn parse(rbsp: &[u8]) -> Result<(u32, Self), Unreadable> {
        let mut bits = BitReader::new(rbsp);
        bits.read_u8(4, "sps_video_parameter_set_id")?;
        let max_sub_layers_minus1 = bits.read_u8(3, "sps_max_sub_layers_minus1")?;
        if max_sub_layers_minus1 > MAX_SUB_LAYERS_MINUS1 {
            return Err(Unreadable::OutOfRange);
        }
        bits.read_bool("sps_temporal_id_nesting_flag")?;
        skip_profile_tier_level(&mut bits, max_sub_layers_minus1)?;
        let id = bits.read_ue("sps_seq_parameter_set_id")?;
        let separate_colour_plane = bits.read_ue("chroma_format_idc")? == 3
            && bits.read_bool("separate_colour_plane_flag")?;
        bits.read_ue("pic_width_in_luma_samples")?;
        bits.read_ue("pic_height_in_luma_samples")?;
        if bits.read_bool("conformance_window_flag")? {
            for _ in 0..4 {
                bits.read_ue("conf_win_offset")?;
            }
        }
        bits.read_ue("bit_depth_luma_minus8")?;
        bits.read_ue("bit_depth_chroma_minus8")?;
        let log2_max_poc_lsb = bits.read_ue("log2_max_pic_order_cnt_lsb_minus4")? + 4;
        if log2_max_poc_lsb > MAX_LOG2_POC_LSB {
            return Err(Unreadable::OutOfRange);
        }
        // Only the timestamp bookkeeping of NVDEC wants it; a truncated
        // SPS still serves the slice headers.
        let max_num_reorder = Self::max_num_reorder(&mut bits, max_sub_layers_minus1).ok();
        Ok((
            id,
            Self {
                log2_max_poc_lsb,
                separate_colour_plane,
                max_num_reorder,
            },
        ))
    }

    /// `sps_max_num_reorder_pics` of the highest sub-layer, the last one
    /// the sub-layer ordering info lists.
    fn max_num_reorder(
        bits: &mut impl BitRead,
        max_sub_layers_minus1: u8,
    ) -> Result<u32, Unreadable> {
        let listed = if bits.read_bool("sps_sub_layer_ordering_info_present_flag")? {
            0..=max_sub_layers_minus1
        } else {
            max_sub_layers_minus1..=max_sub_layers_minus1
        };
        let mut max_num_reorder = 0;
        for _ in listed {
            bits.read_ue("sps_max_dec_pic_buffering_minus1")?;
            max_num_reorder = bits.read_ue("sps_max_num_reorder_pics")?;
            bits.read_ue("sps_max_latency_increase_plus1")?;
        }
        Ok(max_num_reorder)
    }
}

impl Pps {
    /// Reads the head of a PPS (H.265 7.3.2.3.1), returning its id.
    fn parse(rbsp: &[u8]) -> Result<(u32, Self), Unreadable> {
        let mut bits = BitReader::new(rbsp);
        let id = bits.read_ue("pps_pic_parameter_set_id")?;
        let sps_id = bits.read_ue("pps_seq_parameter_set_id")?;
        bits.read_bool("dependent_slice_segments_enabled_flag")?;
        let output_flag_present = bits.read_bool("output_flag_present_flag")?;
        let num_extra_slice_header_bits = bits.read_u8(3, "num_extra_slice_header_bits")?;
        Ok((
            id,
            Self {
                sps_id,
                output_flag_present,
                num_extra_slice_header_bits,
            },
        ))
    }
}

/// Why a header could not be read; the picture is then matched by time.
#[derive(Debug)]
enum Unreadable {
    Bits,
    OutOfRange,
    ParameterSets,
}

impl From<BitReaderError> for Unreadable {
    fn from(_: BitReaderError) -> Self {
        Self::Bits
    }
}

/// Skips `profile_tier_level(1, max_sub_layers_minus1)` (H.265 7.3.3).
fn skip_profile_tier_level(
    bits: &mut impl BitRead,
    max_sub_layers_minus1: u8,
) -> Result<(), Unreadable> {
    // General profile (88 bits) and general_level_idc.
    skip(bits, 96)?;
    let mut sub_layers = Vec::with_capacity(usize::from(max_sub_layers_minus1));
    for _ in 0..max_sub_layers_minus1 {
        sub_layers.push((
            bits.read_bool("sub_layer_profile_present_flag")?,
            bits.read_bool("sub_layer_level_present_flag")?,
        ));
    }
    if max_sub_layers_minus1 > 0 {
        skip(bits, 2 * u32::from(8 - max_sub_layers_minus1))?;
    }
    for (profile, level) in sub_layers {
        skip(
            bits,
            if profile { 88 } else { 0 } + if level { 8 } else { 0 },
        )?;
    }
    Ok(())
}

fn skip(bits: &mut impl BitRead, mut count: u32) -> Result<(), BitReaderError> {
    while count > 0 {
        let chunk = count.min(32);
        bits.read_u32(chunk, "skipped")?;
        count -= chunk;
    }
    Ok(())
}

#[cfg(all(test, feature = "hevc-decode"))]
mod tests {
    use super::*;

    fn pending(sequence: u32, lsb: Option<u32>, timestamp_us: u64) -> Pending {
        Pending {
            sequence,
            poc_lsb: lsb.map(|value| PocLsb { value, log2_max: 8 }),
            timestamp_us,
        }
    }

    fn with(pending: &[Pending]) -> Timestamps {
        Timestamps {
            pending: pending.to_vec(),
            ..Timestamps::default()
        }
    }

    #[test]
    fn released_pictures_take_the_timestamp_of_their_poc() {
        // I0 P4 B2 in decode order, released as 0 2 4.
        let mut timestamps = with(&[
            pending(1, Some(0), 0),
            pending(1, Some(4), 66),
            pending(1, Some(2), 33),
        ]);
        assert_eq!(timestamps.claim(0), Some(0));
        assert_eq!(timestamps.claim(2), Some(33));
        assert_eq!(timestamps.claim(4), Some(66));
        assert_eq!(timestamps.claim(6), None);
    }

    #[test]
    fn dropped_pictures_do_not_shift_the_ones_after_them() {
        // Two pictures before the first IDR never come out.
        let mut timestamps = with(&[
            pending(1, Some(9), 0),
            pending(1, Some(10), 33),
            pending(2, Some(0), 66),
            pending(2, Some(1), 100),
        ]);
        assert_eq!(timestamps.claim(0), Some(66));
        assert_eq!(timestamps.claim(1), Some(100));
        assert!(timestamps.pending.is_empty());
    }

    #[test]
    fn skipped_leading_pictures_are_forgotten_once_a_later_one_comes_out() {
        // CRA 8 with RASL 6 and 7 the decoder skips.
        let mut timestamps = with(&[
            pending(1, Some(8), 266),
            pending(1, Some(6), 200),
            pending(1, Some(7), 233),
            pending(1, Some(9), 300),
        ]);
        assert_eq!(timestamps.claim(8), Some(266));
        assert_eq!(timestamps.pending, [pending(1, Some(9), 300)]);
    }

    #[test]
    fn lsbs_wrap_and_negative_pocs_match() {
        let lsb = PocLsb {
            value: 250,
            log2_max: 8,
        };
        assert!(lsb.matches(250));
        assert!(lsb.matches(506));
        assert!(lsb.matches(-6));
        assert!(!lsb.matches(6));
    }

    #[test]
    fn a_picture_whose_header_was_unreadable_is_matched_by_time() {
        let mut timestamps = with(&[pending(1, None, 40), pending(1, None, 20)]);
        assert_eq!(timestamps.claim(3), Some(20));
        assert_eq!(timestamps.claim(4), Some(40));
    }

    #[test]
    fn an_earlier_sequence_is_released_before_a_later_one_restarting_its_clock() {
        let mut timestamps = with(&[pending(1, Some(4), 1_000), pending(2, Some(0), 0)]);
        assert_eq!(timestamps.claim(4), Some(1_000));
        assert_eq!(timestamps.claim(0), Some(0));
    }

    #[test]
    fn the_fixture_parameter_sets_are_read() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../demuxer/tests/fixtures/hevc.mp4");
        let mut demuxer = ferricast_demuxer::Demuxer::open(&path).unwrap();
        let packet = demuxer.next_frame().unwrap().unwrap();
        let mut timestamps = Timestamps::default();
        timestamps.observe(&rust_h265::parse_annex_b(&packet.data), 7);
        assert!(!timestamps.headers.sps.is_empty());
        assert!(!timestamps.headers.pps.is_empty());
        assert!(
            timestamps
                .headers
                .sps
                .values()
                .all(|sps| sps.max_num_reorder.is_some())
        );
        assert_eq!(timestamps.pending.len(), 1);
        assert_eq!(
            timestamps.pending[0].poc_lsb.map(|lsb| lsb.value),
            Some(0),
            "an IDR starts at POC 0"
        );
    }
}

#[cfg(test)]
mod header_tests {
    use super::*;

    const TRAIL_R: u8 = 1;
    const RASL_N: u8 = 8;
    const IDR_W_RADL: u8 = 19;
    const CRA: u8 = 21;
    const EOS: u8 = 36;

    /// One Annex B NAL unit; a slice starts its picture and carries no
    /// readable header past that, as if its parameter sets were lost.
    fn nal(kind: u8) -> Vec<u8> {
        vec![0, 0, 1, kind << 1, 1, 0x80]
    }

    /// `(sequence, skipped)` of each picture, one packet per NAL type.
    fn pictures(kinds: &[u8]) -> Vec<(u32, bool)> {
        let mut headers = Headers::default();
        kinds
            .iter()
            .flat_map(|&kind| headers.pictures(&rust_h265::parse_annex_b(&nal(kind))))
            .map(|picture| (picture.sequence, picture.skipped))
            .collect()
    }

    #[test]
    fn pictures_before_the_first_irap_are_skipped() {
        assert_eq!(
            pictures(&[TRAIL_R, TRAIL_R, IDR_W_RADL, TRAIL_R]),
            [(1, true), (1, true), (2, false), (2, false)]
        );
    }

    #[test]
    fn rasl_pictures_are_skipped_only_when_decoding_starts_at_their_cra() {
        assert_eq!(
            pictures(&[CRA, RASL_N, TRAIL_R, CRA, RASL_N]),
            [(1, false), (1, true), (1, false), (1, false), (1, false)]
        );
    }

    #[test]
    fn a_cra_after_an_end_of_sequence_starts_a_sequence_and_skips_its_rasl() {
        assert_eq!(
            pictures(&[IDR_W_RADL, TRAIL_R, EOS, CRA, RASL_N, TRAIL_R]),
            [(1, false), (1, false), (2, false), (2, true), (2, false)]
        );
    }
}
