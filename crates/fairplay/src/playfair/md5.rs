//! MD5 rounds with a mid-way shuffle of the input (`modified_md5.c`).

/// Per-round left rotation, as in MD5.
const SHIFT: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, //
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, //
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, //
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

/// `floor(2^32 * |sin(i + 1)|)`, the MD5 round constants the C code
/// computes at run time.
const SINE: [u32; 64] = [
    0xd76a_a478,
    0xe8c7_b756,
    0x2420_70db,
    0xc1bd_ceee,
    0xf57c_0faf,
    0x4787_c62a,
    0xa830_4613,
    0xfd46_9501,
    0x6980_98d8,
    0x8b44_f7af,
    0xffff_5bb1,
    0x895c_d7be,
    0x6b90_1122,
    0xfd98_7193,
    0xa679_438e,
    0x49b4_0821,
    0xf61e_2562,
    0xc040_b340,
    0x265e_5a51,
    0xe9b6_c7aa,
    0xd62f_105d,
    0x0244_1453,
    0xd8a1_e681,
    0xe7d3_fbc8,
    0x21e1_cde6,
    0xc337_07d6,
    0xf4d5_0d87,
    0x455a_14ed,
    0xa9e3_e905,
    0xfcef_a3f8,
    0x676f_02d9,
    0x8d2a_4c8a,
    0xfffa_3942,
    0x8771_f681,
    0x6d9d_6122,
    0xfde5_380c,
    0xa4be_ea44,
    0x4bde_cfa9,
    0xf6bb_4b60,
    0xbebf_bc70,
    0x289b_7ec6,
    0xeaa1_27fa,
    0xd4ef_3085,
    0x0488_1d05,
    0xd9d4_d039,
    0xe6db_99e5,
    0x1fa2_7cf8,
    0xc4ac_5665,
    0xf429_2244,
    0x432a_ff97,
    0xab94_23a7,
    0xfc93_a039,
    0x655b_59c3,
    0x8f0c_cc92,
    0xffef_f47d,
    0x8584_5dd1,
    0x6fa8_7e4f,
    0xfe2c_e6e0,
    0xa301_4314,
    0x4e08_11a1,
    0xf753_7e82,
    0xbd3a_f235,
    0x2ad7_d2bb,
    0xeb86_d391,
];

/// Runs the 64 MD5 steps over `block`, starting from `key` instead of the
/// MD5 initial state, and returns `key` plus the final state. After step
/// 31 some input words are swapped, picked by the state at that point.
pub(super) fn modified_md5(block: &[u8; 64], key: &[u8; 16]) -> [u8; 16] {
    // Input words are read big-endian; the key and output little-endian.
    let (words, _) = block.as_chunks::<4>();
    let mut input: [u32; 16] = std::array::from_fn(|i| u32::from_be_bytes(words[i]));
    let (key_words, _) = key.as_chunks::<4>();
    let key_words: [u32; 4] = std::array::from_fn(|i| u32::from_le_bytes(key_words[i]));

    let [mut a, mut b, mut c, mut d] = key_words;
    for (i, (shift, sine)) in SHIFT.into_iter().zip(SINE).enumerate() {
        let (mix, j) = match i {
            0..16 => ((b & c) | (!b & d), i),
            16..32 => ((b & d) | (c & !d), (5 * i + 1) % 16),
            32..48 => (b ^ c ^ d, (3 * i + 5) % 16),
            _ => (c ^ (b | !d), (7 * i) % 16),
        };
        let z = a
            .wrapping_add(input[j])
            .wrapping_add(sine)
            .wrapping_add(mix)
            .rotate_left(shift)
            .wrapping_add(b);
        (a, b, c, d) = (d, z, b, c);
        if i == 31 {
            let nibble = |x: u32, n: u32| ((x >> (4 * n)) & 15) as usize;
            input.swap(nibble(a, 0), nibble(b, 0));
            input.swap(nibble(c, 0), nibble(d, 0));
            input.swap(nibble(a, 1), nibble(b, 1));
            input.swap(nibble(a, 2), nibble(b, 2));
            input.swap(nibble(a, 3), nibble(b, 3));
        }
    }

    let state = [a, b, c, d];
    let mut out = [0_u8; 16];
    for ((chunk, key_word), word) in out.chunks_exact_mut(4).zip(key_words).zip(state) {
        chunk.copy_from_slice(&key_word.wrapping_add(word).to_le_bytes());
    }
    out
}
