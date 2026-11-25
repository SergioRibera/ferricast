//! A framing check of VP9 packets before they reach the parser.
//!
//! NVIDIA's VP9 parser trusts a superframe index (VP9 bitstream
//! specification, Annex B): a packet that ends in what reads as one, with
//! frame sizes past the packet, makes `cuvidParseVideoData` read out of
//! bounds and fault (SIGSEGV, driver 595; a random 91-byte packet whose
//! last byte happened to be a superframe marker). So the decoder reads the
//! index and refuses a packet whose frames do not fit before it.

use crate::error::{Error, Result};

/// Checks the superframe index of `data`, if it has one.
pub(super) fn check_packet(data: &[u8]) -> Result<()> {
    let Some(&marker) = data.last() else {
        return Ok(());
    };
    if marker & 0xe0 != 0xc0 {
        return Ok(());
    }
    let size_bytes = usize::from((marker >> 3) & 0x03) + 1;
    let frames = usize::from(marker & 0x07) + 1;
    let index_size = 2 + size_bytes * frames;
    let Some(index_start) = data.len().checked_sub(index_size) else {
        return Ok(());
    };
    if data[index_start] != marker {
        // Not an index: the last byte of an ordinary frame.
        return Ok(());
    }
    let mut total = 0_usize;
    for size in data[index_start + 1..data.len() - 1].chunks_exact(size_bytes) {
        let size = size
            .iter()
            .rev()
            .fold(0_usize, |value, &byte| (value << 8) | usize::from(byte));
        total = total.saturating_add(size);
    }
    if total > index_start {
        return Err(Error::InvalidPacket {
            reason: format!(
                "VP9 superframe of {frames} frames, {total} bytes, before an index at byte {index_start}"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn superframe_sizes_must_fit_before_the_index() {
        // Two frames of 2 and 1 bytes, 1-byte sizes: marker 0b1100_0001.
        let superframe = [9, 9, 8, 0xc1, 2, 1, 0xc1];
        assert!(check_packet(&superframe).is_ok());
        let mut long = superframe;
        long[4] = 3;
        assert!(check_packet(&long).is_err());
        // A last byte that looks like a marker without a matching first
        // index byte is an ordinary frame.
        assert!(check_packet(&[1, 2, 3, 0xc1]).is_ok());
        assert!(check_packet(&[]).is_ok());
    }
}
