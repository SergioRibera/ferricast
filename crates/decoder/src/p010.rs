//! P010 pictures (16-bit little-endian samples, the 10-bit code in the
//! high bits) for consumers that asked for 8 bits.

/// The high byte of every sample of a P010 picture whose luma rows are
/// followed by its interleaved chroma rows, `stride` bytes apart: the
/// same picture as NV12, rows packed. Returns the bytes and their stride.
pub(crate) fn downshift_p010(
    data: &[u8],
    stride: usize,
    width: usize,
    height: usize,
) -> (Vec<u8>, usize) {
    let row = width.div_ceil(2) * 2;
    let rows = height + height.div_ceil(2);
    let mut out = Vec::with_capacity(row * rows);
    for line in data.chunks(stride).take(rows) {
        out.extend(line.iter().skip(1).step_by(2).take(row));
    }
    (out, row)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_high_byte_of_each_sample_and_drops_padding() {
        // 2x2 P010, stride 6 (2 bytes padding): luma rows, one chroma row.
        let data = [
            0x00, 0x40, 0xC0, 0xEB, 0xFF, 0xFF, //
            0x11, 0x10, 0x22, 0x20, 0xFF, 0xFF, //
            0x00, 0x80, 0x00, 0x7F, 0xFF, 0xFF,
        ];
        let (nv12, stride) = downshift_p010(&data, 6, 2, 2);
        assert_eq!(stride, 2);
        assert_eq!(nv12, [0x40, 0xEB, 0x10, 0x20, 0x80, 0x7F]);
    }
}
