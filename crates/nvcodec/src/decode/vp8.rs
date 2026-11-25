//! A check of VP8 frames before they reach the parser.
//!
//! NVIDIA's VP8 parser faults (SIGSEGV inside `cuvidParseVideoData`,
//! driver 595) on an inter frame that refreshes the golden frame and
//! copies it to the alternate reference (`refresh_golden_frame = 1`,
//! `copy_buffer_to_alternate = 2`) when no key frame came before it: the
//! copy reads reference buffers that were never set up. Found by cutting
//! a crashing fuzz case down to an unmodified frame of the test clip, fed
//! first: every such frame of the clip faults, no other frame does. So the
//! decoder refuses inter frames until it has passed a key frame on.
//!
//! The check also reads the frame tag (RFC 6386 §9.1), the key frame start
//! code and dimensions, and the first partition's header up to the
//! partition count (§9.2-9.5, §19.2), and refuses a frame whose partitions
//! do not fit in the packet, so the parser never has to trust those sizes.
use crate::error::{Error, Result};

/// The 3-byte frame tag and, on key frames, the start code and dimensions.
const KEY_FRAME_HEADER: usize = 10;
const INTER_FRAME_HEADER: usize = 3;
/// `9d 01 2a`, after a key frame's tag.
const START_CODE: [u8; 3] = [0x9d, 0x01, 0x2a];
/// Each DCT partition but the last is preceded by its size, 3 bytes.
const PARTITION_SIZE_BYTES: usize = 3;

/// The layout of a VP8 frame, as far as the parser relies on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FrameLayout {
    pub(super) key_frame: bool,
    pub(super) first_partition: usize,
    pub(super) partitions: usize,
}

/// Checks that `data` is a VP8 frame whose partitions all lie inside it.
pub(super) fn check_frame(data: &[u8]) -> Result<FrameLayout> {
    let invalid = |reason: String| Error::InvalidPacket {
        reason: format!("VP8 frame: {reason}"),
    };
    let [a, b, c, ..] = *data else {
        return Err(invalid(format!(
            "{} bytes, shorter than a frame tag",
            data.len()
        )));
    };
    let tag = u32::from_le_bytes([a, b, c, 0]);
    let key_frame = tag & 1 == 0;
    let version = (tag >> 1) & 7;
    let first_partition = (tag >> 5) as usize;
    if version > 3 {
        return Err(invalid(format!("version {version}")));
    }
    let header = if key_frame {
        if data.len() < KEY_FRAME_HEADER {
            return Err(invalid("key frame shorter than its header".into()));
        }
        if data[3..6] != START_CODE {
            return Err(invalid("key frame without the 9d 01 2a start code".into()));
        }
        let width = u16::from_le_bytes([data[6], data[7]]) & 0x3fff;
        let height = u16::from_le_bytes([data[8], data[9]]) & 0x3fff;
        if width == 0 || height == 0 {
            return Err(invalid(format!("key frame of {width}x{height}")));
        }
        KEY_FRAME_HEADER
    } else {
        INTER_FRAME_HEADER
    };
    let first_end = header
        .checked_add(first_partition)
        .filter(|end| *end <= data.len())
        .ok_or_else(|| {
            invalid(format!(
                "first partition of {first_partition} bytes runs past the {}-byte packet",
                data.len()
            ))
        })?;
    let partitions = partition_count(&data[header..first_end], key_frame)
        .ok_or_else(|| invalid("first partition ends inside its header".into()))?;
    // The sizes of all partitions but the last, then the partitions.
    let sizes_end = first_end
        .checked_add((partitions - 1) * PARTITION_SIZE_BYTES)
        .filter(|end| *end <= data.len())
        .ok_or_else(|| invalid(format!("{partitions} partition sizes run past the packet")))?;
    let mut end = sizes_end;
    for size in data[first_end..sizes_end].chunks_exact(PARTITION_SIZE_BYTES) {
        let size = u32::from_le_bytes([size[0], size[1], size[2], 0]) as usize;
        end = end
            .checked_add(size)
            .filter(|end| *end <= data.len())
            .ok_or_else(|| {
                invalid(format!(
                    "DCT partitions of {partitions} run past the {}-byte packet",
                    data.len()
                ))
            })?;
    }
    Ok(FrameLayout {
        key_frame,
        first_partition,
        partitions,
    })
}

/// Reads the frame header in the first partition (§19.2) up to
/// `log2_nbr_of_dct_partitions`; `None` when the partition ends first.
fn partition_count(first: &[u8], key_frame: bool) -> Option<usize> {
    let mut bits = BoolDecoder::new(first)?;
    if key_frame {
        bits.literal(2)?; // color_space, clamping_type
    }
    if bits.flag()? {
        // update_segmentation()
        let update_map = bits.flag()?;
        if bits.flag()? {
            // update_segment_feature_data: segment_feature_mode, then the
            // quantizer and loop filter values of each segment.
            bits.literal(1)?;
            for (_, magnitude) in [(0, 7); 4].into_iter().chain([(0, 6); 4]) {
                if bits.flag()? {
                    bits.literal(magnitude + 1)?; // value and sign
                }
            }
        }
        if update_map {
            for _ in 0..3 {
                if bits.flag()? {
                    bits.literal(8)?; // segment_prob
                }
            }
        }
    }
    bits.literal(1 + 6 + 3)?; // filter_type, loop_filter_level, sharpness_level
    if bits.flag()? && bits.flag()? {
        // loop_filter_adj_enable, mode_ref_lf_delta_update: the deltas of
        // four reference frames and four modes.
        for _ in 0..8 {
            if bits.flag()? {
                bits.literal(6 + 1)?;
            }
        }
    }
    let log2 = bits.literal(2)?;
    Some(1 << log2)
}

/// The boolean entropy decoder of RFC 6386 §7.3, for literals only.
struct BoolDecoder<'a> {
    data: &'a [u8],
    /// Next byte of `data` to shift in.
    next: usize,
    value: u32,
    range: u32,
    bit_count: u32,
    /// Bytes shifted in past the end of `data` (as zeros).
    overrun: usize,
}

impl<'a> BoolDecoder<'a> {
    fn new(data: &'a [u8]) -> Option<Self> {
        let [first, second, ..] = *data else {
            return None;
        };
        Some(Self {
            data,
            next: 2,
            value: (u32::from(first) << 8) | u32::from(second),
            range: 255,
            bit_count: 0,
            overrun: 0,
        })
    }

    fn bool(&mut self, probability: u32) -> Option<bool> {
        let split = 1 + (((self.range - 1) * probability) >> 8);
        let big_split = split << 8;
        let bit = if self.value >= big_split {
            self.range -= split;
            self.value -= big_split;
            true
        } else {
            self.range = split;
            false
        };
        while self.range < 128 {
            self.value <<= 1;
            self.range <<= 1;
            self.bit_count += 1;
            if self.bit_count == 8 {
                self.bit_count = 0;
                match self.data.get(self.next) {
                    Some(&byte) => self.value |= u32::from(byte),
                    None => self.overrun += 1,
                }
                self.next += 1;
            }
        }
        // The decoder looks two bytes ahead; reading further past the end
        // means the header does not fit in the partition.
        (self.overrun <= 2).then_some(bit)
    }

    fn flag(&mut self) -> Option<bool> {
        self.bool(128)
    }

    /// `L(n)`: an unsigned n-bit literal, most significant bit first.
    fn literal(&mut self, bits: u32) -> Option<u32> {
        (0..bits).try_fold(0, |value, _| Some((value << 1) | u32::from(self.flag()?)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 16x16 key frame tag, start code and size, then a first partition
    /// of zeros: every flag 0, one partition.
    fn key_frame(first_partition: u32, extra: usize) -> Vec<u8> {
        let tag = first_partition << 5 | 1 << 4;
        let mut frame = tag.to_le_bytes()[..3].to_vec();
        frame.extend_from_slice(&START_CODE);
        frame.extend_from_slice(&[16, 0, 16, 0]);
        frame.resize(KEY_FRAME_HEADER + first_partition as usize + extra, 0);
        frame
    }

    #[test]
    fn layouts_must_fit() {
        let layout = check_frame(&key_frame(8, 4)).unwrap();
        assert_eq!(
            layout,
            FrameLayout {
                key_frame: true,
                first_partition: 8,
                partitions: 1
            }
        );
        let mut short = key_frame(8, 0);
        short.truncate(short.len() - 1);
        assert!(check_frame(&short).is_err(), "first partition past the end");
        let mut no_start = key_frame(8, 4);
        no_start[4] = 0;
        assert!(check_frame(&no_start).is_err(), "start code");
        assert!(check_frame(&[0x11, 0x02]).is_err(), "shorter than a tag");
        // An inter frame: only the tag before the first partition.
        let inter = [0x11 | 2 << 5, 0, 0, 0, 0, 0];
        assert!(!check_frame(&inter).unwrap().key_frame);
    }

    #[test]
    fn the_bool_decoder_reads_literals() {
        // With probability 128 each bit of a zero stream decodes as 0.
        let mut zeros = BoolDecoder::new(&[0; 8]).unwrap();
        assert_eq!(zeros.literal(16), Some(0));
        // It runs out a little past the end, not before.
        let mut short = BoolDecoder::new(&[0; 2]).unwrap();
        assert!(short.literal(16).is_some());
        assert!(short.literal(64).is_none());
    }
}
