//! The second hash mixed into the session key (`sap_hash.c`).

use super::garble::Garble;

/// Bytes of the garbled `buffer3` that feed the first 11 key bytes, in
/// units of 4 bytes.
const BUFFER3_PICKS: [usize; 11] = [18, 22, 23, 0, 5, 19, 32, 31, 10, 21, 30];

/// Index `i - back` reduced like the C code: the subtraction wraps as a
/// 32-bit unsigned integer before the modulo.
fn wrapped_index(i: u32, back: u32, len: u32) -> usize {
    (i.wrapping_sub(back) % len) as usize
}

pub(super) fn sap_hash(block: &[u8; 64]) -> [u8; 16] {
    // Spread the block over 210 bytes, byte-swapping each 32-bit word.
    let mut buffer1: [u8; 210] = std::array::from_fn(|i| {
        let i = i % 64;
        block[(i & !3) | (3 - (i % 4))]
    });
    for i in 0..840_u32 {
        let x = buffer1[wrapped_index(i, 155, 210)];
        let y = buffer1[wrapped_index(i, 57, 210)];
        let z = buffer1[wrapped_index(i, 13, 210)];
        let w = buffer1[wrapped_index(i, 0, 210)];
        buffer1[wrapped_index(i, 0, 210)] = y
            .rotate_left(5)
            .wrapping_add(z.rotate_left(3) ^ w)
            .wrapping_sub(x.rotate_left(7));
    }

    let garbled = Garble::new(&buffer1).run();

    let mut key = [0xe1_u8; 16];
    for (i, pick) in BUFFER3_PICKS.into_iter().enumerate() {
        key[i] = if i == 3 {
            0x3d
        } else {
            key[i].wrapping_add(garbled.buffer3[pick * 4])
        };
    }
    let buffers: [&[u8]; 3] = [&garbled.buffer0, &garbled.buffer2, &garbled.buffer1];
    for buffer in buffers {
        for (byte, slot) in buffer.iter().zip((0..16).cycle()) {
            key[slot] ^= byte;
        }
    }

    for _ in 0..16 {
        for i in 0..16_u32 {
            let x = key[wrapped_index(i, 7, 16)];
            let y = key[wrapped_index(i, 0, 16)];
            let z = key[wrapped_index(i, 37, 16)];
            let w = key[wrapped_index(i, 177, 16)];
            key[wrapped_index(i, 0, 16)] =
                x.rotate_left(1) ^ y ^ z.rotate_left(6) ^ w.rotate_left(5);
        }
    }
    key
}
