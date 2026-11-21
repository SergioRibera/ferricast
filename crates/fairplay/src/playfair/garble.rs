//! The byte-scrambling step of `sap_hash` (`hand_garble.c`).
//!
//! Every statement mirrors one in the C function `garble`, in the same
//! order. The C code computes in `int`/`unsigned int` and truncates on each
//! store to an `unsigned char`; here the buffers hold their bytes as
//! wrapping `u32` values and every store masks to 8 bits, which gives the
//! same bits. The C code only divides, takes remainders of, or right-shifts
//! values that are non-negative, so unsigned arithmetic matches it there
//! too.

use std::num::Wrapping;

type W = Wrapping<u32>;

const fn w(value: u32) -> W {
    Wrapping(value)
}

/// Truncates to the `unsigned char` the C code stores.
fn byte(value: W) -> W {
    value & w(0xff)
}

/// Index into a buffer of `len` bytes: `value % len`.
fn ix(value: W, len: usize) -> usize {
    value.0 as usize % len
}

/// Shift amount; every caller passes a value below 8.
fn sh(value: W) -> usize {
    value.0 as usize
}

/// `rol8` from `sap_hash.c`: an 8-bit rotation of an `unsigned char`.
fn rol8(input: W, count: usize) -> W {
    let input = byte(input);
    byte(input << count) | (input >> (8 - count))
}

/// `rol8x`: the same rotation without truncating the result.
fn rol8x(input: W, count: usize) -> W {
    let input = byte(input);
    (input << count) | (input >> (8 - count))
}

/// `weird_ror8`: right rotation that leaves the high bits in and yields 0
/// for a zero count.
fn weird_ror8(input: W, count: W) -> W {
    let (input, count) = (byte(input), sh(count));
    if count == 0 {
        return w(0);
    }
    byte(input >> count) | (input << (8 - count))
}

/// `weird_rol8`: 8-bit left rotation that yields 0 for a zero count.
fn weird_rol8(input: W, count: W) -> W {
    let (input, count) = (byte(input), sh(count));
    if count == 0 {
        return w(0);
    }
    byte(input << count) | (input >> (8 - count))
}

/// `weird_rol32`: left rotation folded with XOR, 0 for a zero count.
fn weird_rol32(input: W, count: W) -> W {
    let (input, count) = (byte(input), sh(count));
    if count == 0 {
        return w(0);
    }
    (input << count) ^ (input >> (8 - count))
}

/// The buffers `garble` reads and writes, as left by it.
pub(super) struct Garbled {
    pub(super) buffer0: [u8; 20],
    pub(super) buffer1: [u8; 210],
    pub(super) buffer2: [u8; 35],
    pub(super) buffer3: [u8; 132],
}

/// Inputs of `garble`: the scrambled block plus `sap_hash`'s constant
/// buffers.
pub(super) struct Garble {
    b0: [W; 20],
    b1: [W; 210],
    b2: [W; 35],
    b3: [W; 132],
    b4: [W; 21],
}

const BUFFER0: [u8; 20] = [
    0x96, 0x5F, 0xC6, 0x53, 0xF8, 0x46, 0xCC, 0x18, 0xDF, 0xBE, 0xB2, 0xF8, 0x38, 0xD7, 0xEC, 0x22,
    0x03, 0xD1, 0x20, 0x8F,
];
const BUFFER2: [u8; 35] = [
    0x43, 0x54, 0x62, 0x7A, 0x18, 0xC3, 0xD6, 0xB3, 0x9A, 0x56, 0xF6, 0x1C, 0x14, 0x3F, 0x0C, 0x1D,
    0x3B, 0x36, 0x83, 0xB1, 0x39, 0x51, 0x4A, 0xAA, 0x09, 0x3E, 0xFE, 0x44, 0xAF, 0xDE, 0xC3, 0x20,
    0x9D, 0x42, 0x3A,
];
const BUFFER4: [u8; 21] = [
    0xED, 0x25, 0xD1, 0xBB, 0xBC, 0x27, 0x9F, 0x02, 0xA2, 0xA9, 0x11, 0x00, 0x0C, 0xB3, 0x52, 0xC0,
    0xBD, 0xE3, 0x1B, 0x49, 0xC7,
];

fn widen<const N: usize>(bytes: &[u8; N]) -> [W; N] {
    bytes.map(|b| w(u32::from(b)))
}

fn narrow<const N: usize>(values: &[W; N]) -> [u8; N] {
    // Every stored value was masked to 8 bits.
    values.map(|v| v.0.to_le_bytes()[0])
}

impl Garble {
    pub(super) fn new(buffer1: &[u8; 210]) -> Self {
        Self {
            b0: widen(&BUFFER0),
            b1: widen(buffer1),
            b2: widen(&BUFFER2),
            // Uninitialised in C; `garble` writes every byte it later reads.
            b3: [w(0); 132],
            b4: widen(&BUFFER4),
        }
    }

    pub(super) fn run(self) -> Garbled {
        let Self {
            mut b0,
            mut b1,
            mut b2,
            mut b3,
            b4,
        } = self;

        b2[12] = byte(
            w(0x14)
                + (((b1[64] & w(92)) | ((b1[99] / w(3)) & w(35)))
                    & b4[ix(rol8x(b4[ix(b1[206], 21)], 4), 21)]),
        );
        b1[4] = byte((b1[99] / w(5)) * (b1[99] / w(5)) * w(2));
        b2[34] = w(0xb8);
        {
            let x = b2[ix(b1[203], 35)];
            b1[153] = byte(b1[153] ^ (x * x * b1[190]));
        }
        b0[3] = byte(b0[3] - (((b4[ix(b1[205], 21)] >> 1) & w(80)) | w(0xe6440)));
        b0[16] = w(0x93);
        b0[13] = w(0x62);
        b1[33] = byte(b1[33] - (b4[ix(b1[36], 21)] & w(0xf6)));

        let tmp2 = b2[ix(b1[67], 35)];
        b2[12] = w(0x07);

        let tmp = b0[ix(b1[181], 20)];
        b1[2] = byte(b1[2] - w(3136));
        b0[19] = b4[ix(b1[58], 21)];
        b3[0] = byte(w(92) - b2[ix(b1[32], 35)]);
        b3[4] = byte(b2[ix(b1[15], 35)] + w(0x9e));
        b1[34] = byte(b1[34] + (b4[ix((b2[ix(b1[15], 35)] + w(0x9e)) & w(0xff), 21)] / w(5)));
        b0[19] = byte(b0[19] + (w(0xffff_fee6) - ((b0[ix(b3[4], 20)] >> 1) & w(102))));

        {
            let s = b4[ix(b1[190], 21)];
            let rotated = (b1[72] >> sh(s & w(7))) ^ (b1[72] << sh((w(7) - (s - w(1))) & w(7)));
            b1[15] = byte((w(3) * (rotated - w(3) * b4[ix(b1[126], 21)])) ^ b1[15]);
        }
        {
            let x = b2[ix(b1[181], 35)];
            b0[15] = byte(b0[15] ^ (x * x * x));
        }
        b2[4] = byte(b2[4] ^ (b1[202] / w(3)));

        {
            let a = w(92) - b0[ix(b3[0], 20)];
            let e = (a & w(0xc6)) | (!b1[105] & w(0xc6)) | (a & !b1[105]);
            b2[1] = byte(b2[1] + e * e * e);
        }
        b0[19] =
            byte(b0[19] ^ (((w(224) | (b4[ix(b1[92], 21)] & w(27))) * b2[ix(b1[41], 35)]) / w(3)));
        b1[140] = byte(b1[140] + weird_ror8(w(92), b1[5] & w(7)));

        {
            let x = !b1[4] ^ b2[ix(b1[12], 35)];
            b2[12] = byte(b2[12] + (((x | b1[182]) & w(192)) | (x & b1[182])));
        }
        b1[36] = byte(b1[36] + w(125));

        {
            let p = (w(74) & b1[138]) | ((w(74) | b1[138]) & b0[15]);
            let q = b0[ix(b1[43], 20)];
            b1[124] = byte(rol8x((p & q) | ((p | q) & w(95)), 4));
        }
        b3[8] = byte(
            (((b0[ix(b3[4], 20)] & w(95)) & ((b4[ix(b1[68], 21)] & w(46)) << 1)) | w(16)) ^ w(92),
        );

        {
            let a = b1[177] + b4[ix(b1[79], 21)];
            let m = (w(3) * b1[148]) / w(5);
            let d = (((a >> 1) | m) & b2[1]) | ((a >> 1) & m);
            b3[12] = byte(w(0) - w(34) - d);
        }

        {
            let a = w(8) - (b2[22] & w(7));
            let b = b1[33] >> sh(a & w(7));
            let c = b1[33] << sh(b2[22] & w(7));
            b2[16] = byte(
                b2[16]
                    + (((b2[ix(b3[0], 35)] & w(159)) | b0[ix(b3[4], 20)] | w(8))
                        - ((b ^ c) | w(128))),
            );
        }

        b0[14] = byte(b0[14] ^ b2[ix(b3[12], 35)]);

        {
            let a = weird_rol8(
                b4[ix(b0[ix(b1[201], 20)], 21)],
                (b2[ix(b1[112], 35)] << 1) & w(7),
            );
            let d = (b0[ix(b1[208], 20)] & w(131)) | (b0[ix(b1[164], 20)] & w(124));
            b1[19] = byte(b1[19] + ((a & (d / w(5))) | ((a | (d / w(5))) & w(37))));
        }

        {
            let x = b4[ix(b1[45], 21)] + w(92);
            b2[8] = byte(weird_ror8(w(140), (x * x) & w(7)));
        }
        b1[190] = w(56);
        b2[8] = byte(b2[8] ^ b3[0]);
        b1[53] = byte(!((b0[ix(b1[83], 20)] | w(204)) / w(5)));
        b0[13] = byte(b0[13] + b0[ix(b1[41], 20)]);
        {
            let x = b2[ix(b3[0], 35)];
            b0[10] = byte(((x & b1[2]) | ((x | b1[2]) & b3[12])) / w(15));
        }

        {
            let p = b4[ix(b1[2], 21)] & w(68);
            let q = b2[ix(b3[8], 35)];
            let a = (((w(56) | p) | q) & w(42)) | ((p | w(56)) & q);
            b3[16] = byte(a * a + w(110));
        }
        b3[20] = byte(w(202) - b3[16]);
        b3[24] = b1[151];
        b2[13] = byte(b2[13] ^ b4[ix(b3[0], 21)]);

        {
            let x = b2[ix(b1[179], 35)] - w(38);
            let b = (x & w(177)) | (b3[12] & w(177));
            let c = x & b3[12];
            b3[28] = byte(w(30) + (b | c) * (b | c));
        }
        b3[32] = byte(b3[28] + w(62));

        let tmp3;
        {
            let s = b3[20] + (b3[0] & w(74));
            let n = !b4[ix(b3[0], 21)];
            let a = (s | n) & w(121);
            let b = s & n;
            tmp3 = a | b;
            let x = (a | b) ^ w(0xffff_ffa6);
            let c = ((x | b3[0]) & w(4)) | (x & b3[0]);
            b1[47] = byte((b2[ix(b1[89], 35)] + c) ^ b1[47]);
        }

        b3[36] = byte(((rol8((tmp & w(179)) + w(68), 2) & b0[3]) | (tmp2 & !b0[3])) - w(15));
        b1[123] = byte(b1[123] ^ w(221));

        {
            let a = (b4[ix(b3[0], 21)] / w(3)) - b2[ix(b3[4], 35)];
            let c = (((b3[0] & w(163)) + w(92)) & w(246)) | (b3[0] & w(92));
            let e = ((c | b3[24]) & w(54)) | (c & b3[24]);
            b3[40] = byte(a - e);
        }
        b3[44] = byte(tmp3 ^ w(81) ^ (((b3[0] >> 1) & w(101)) + w(26)));
        b3[48] = byte(b2[ix(b3[4], 35)] & w(27));
        b3[52] = w(27);
        b3[56] = w(199);

        {
            // `buffer4[buffer3[0] % 20]` (not 21) is what the C code reads.
            let b4_20 = b4[ix(b3[0], 20)];
            let b4_21 = b4[ix(b3[0], 21)];
            let t1 = ((b3[40] | b3[24]) & w(177)) | (b3[40] & b3[24]);
            let t2 = ((b4_20 & w(177)) | w(176)) | (b4_21 & !w(3));
            let q1 = ((b3[40] & b3[24]) | ((b3[40] | b3[24]) & w(177))) & w(199);
            let q2 = (((b4_21 & w(1)) + w(176)) | (b4_21 & !w(3))) & b3[56];
            b3[64] = byte(b3[4] + ((((t1 & t2) | (q1 | q2)) & !b3[52]) | b3[48]));
        }

        b2[33] = byte(b2[33] ^ b1[26]);
        b1[106] = byte(b1[106] ^ b3[20] ^ w(133));
        b2[30] = byte(((b3[64] / w(3)) - (w(275) | (b3[0] & w(247)))) ^ b0[ix(b1[122], 20)]);
        b1[22] = byte((b2[ix(b1[90], 35)] & w(95)) | w(68));

        {
            let a = (b4[ix(b3[36], 21)] & w(184)) | (b2[ix(b3[44], 35)] & !w(184));
            b2[18] = byte(b2[18] + ((a * a * a) >> 1));
        }
        b2[5] = byte(b2[5] - b4[ix(b1[92], 21)]);

        {
            let a = (((b1[41] & !w(24)) | (b2[ix(b1[183], 35)] & w(24))) & (b3[16] + w(53)))
                | (b3[20] & b2[ix(b3[20], 35)]);
            let b = (b1[17] & !b3[44]) | (b0[ix(b1[59], 20)] & b3[44]);
            b2[18] = byte(b2[18] ^ (a * b));
        }

        {
            let a = weird_ror8(b1[11], b2[ix(b1[28], 35)] & w(7)) & w(7);
            let b =
                (((b0[ix(b1[93], 20)] & !b0[14]) | (b0[14] & w(150))) & !w(28)) | (b1[7] & w(28));
            let r = weird_rol8(b2[ix(b3[0], 35)], a);
            b2[22] = byte((((b | r) & b2[33]) | (b & r)) + w(74));
        }

        {
            let a = b4[ix(b0[ix(b1[39], 20)] ^ w(217), 21)];
            let q = ((b3[20] | b3[0]) & w(214)) | (b3[20] & b3[0]);
            b0[15] = byte(b0[15] - ((q & a) | ((q | a) & b3[32])));
        }

        let t;
        {
            let x = b2[ix(b1[57], 35)];
            let y = b0[ix(b3[64], 20)];
            let k = (b3[64] & w(45)) | w(82);
            let b = ((x & y) | ((y | x) & w(95)) | (b3[64] & w(45)) | w(82)) & w(32);
            let c = ((x & y) | ((x | y) & w(95))) & k;
            let d = ((b3[0] / w(3)) - (b3[64] | b1[22])) ^ (b3[28] + w(62)) ^ (b | c);
            t = b0[ix(d & w(0xff), 20)];
        }

        {
            let x = b0[ix(b1[99], 20)];
            b3[68] = byte((x * x * x * x) | b2[ix(b3[64], 35)]);
        }

        let u = b0[ix(b1[50], 20)];
        let w_ = b2[ix(b1[138], 35)];
        let x_ = b4[ix(b1[39], 21)];
        let y_ = b0[ix(b1[4], 20)];
        let z_ = b4[ix(b1[202], 21)];
        let v = b0[ix(b1[151], 20)];
        let s = b2[ix(b1[14], 35)];
        let r = b0[ix(b1[145], 20)];

        {
            let p = b2[ix(b3[68], 35)];
            let q = b0[ix(b1[209], 20)];
            let a = (p & q) | ((p | q) & w(24));
            let b = weird_rol8(b4[ix(b1[127], 21)], p & w(7));
            let c = (a & b0[10]) | (b & !b0[10]);
            let d = w(7) ^ (b4[ix(b2[ix(b3[36], 35)], 21)] << 1);
            b3[72] = byte((c & w(71)) | (d & !w(71)));
        }

        b2[2] = byte(
            b2[2]
                + ((((b0[ix(b3[20], 20)] << 1) & w(159)) | (b4[ix(b1[190], 21)] & !w(159)))
                    & ((((b4[ix(b3[64], 21)] & w(110)) | (b0[ix(b1[25], 20)] & !w(110)))
                        & !w(150))
                        | (b1[25] & w(150)))),
        );
        b2[14] = byte(
            b2[14]
                - (((b2[ix(b3[20], 35)] & (b3[72] ^ b2[ix(b1[100], 35)])) & !w(34))
                    | (b1[97] & w(34))),
        );
        b0[17] = w(115);

        {
            let p = b4[ix(b1[17], 21)];
            let q = b0[ix(b3[20], 20)];
            let m = ((p | q) & b3[72]) | (p & q);
            let d = b1[50] / w(3);
            b1[23] = byte(b1[23] ^ (((m & d) | ((m | d) & w(246))) << 1));
        }

        {
            let p = b0[ix(b3[40], 20)];
            b0[13] = byte(
                (((((p | b1[10]) & w(82)) | (p & b1[10])) & w(209))
                    | ((b0[ix(b1[39], 20)] << 1) & w(46)))
                    >> 1,
            );
        }
        b2[33] = byte(b2[33] - (b1[113] & w(9)));
        b2[28] =
            byte(b2[28] - ((((w(2) | (b1[110] & w(222))) >> 1) & !w(223)) | (b3[20] & w(223))));

        let j = weird_rol8(v | z_, u & w(7));
        let e;
        let m;
        {
            let a = (b2[16] & t) | (w_ & !b2[16]);
            let b = (b1[33] & w(17)) | (x_ & !w(17));
            e = ((y_ | ((a + b) / w(5))) & w(147)) | (y_ & ((a + b) / w(5)));
            let k = b4[ix((b3[8] + j + e) & w(0xff), 21)];
            m = (b3[40] & k) | ((b3[40] | k) & b2[23]);
        }
        let jeb = b3[8] + j + e;

        {
            let p = b4[ix(b3[20], 21)] - w(48);
            b0[15] = byte(((p & !b1[184]) | (p & w(189)) | (w(189) & !b1[184])) & (m * m * m));
        }
        b2[22] = byte(b2[22] + b1[183]);
        b3[76] = byte((w(3) * b4[ix(b1[1], 21)]) ^ b3[0]);

        {
            let a = b2[ix(jeb & w(0xff), 35)];
            let p = b4[ix(b1[178], 21)];
            let f =
                (((p & a) | ((p | a) & w(209))) * b0[ix(b1[13], 20)]) * (b4[ix(b1[26], 21)] >> 1);
            let g = (f + w(0x733f_fff9)) * w(198)
                - (((f + w(0x733f_fff9)) * w(396) + w(212)) & w(212))
                + w(85);
            b3[80] = byte(b3[36] + (g ^ w(148)) + ((g ^ w(107)) << 1) - w(127));
        }

        b3[84] = byte((b2[ix(b3[64], 35)] & w(245)) | (b2[ix(b3[20], 35)] & w(10)));

        {
            let a = b0[ix(b3[68], 20)] | w(81);
            b2[18] = byte(b2[18] - (((a * a * a) & !b0[15]) | ((b3[80] / w(15)) & b0[15])));
        }

        b3[88] = byte(jeb - b0[ix(b1[160], 20)] + (b4[ix(b0[ix(jeb & w(255), 20)], 21)] / w(3)));

        {
            let b = ((r ^ b3[72]) & !w(198)) | ((s * s) & w(198));
            let p = b4[ix(b1[69], 21)];
            let f = (p & b1[172]) | ((p | b1[172]) & ((b3[12] - b) + w(77)));
            b0[16] = byte(
                w(147) - ((b3[72] & ((f & w(251)) | w(1))) | (((f & w(250)) | b3[72]) & w(198))),
            );
        }

        {
            let p = b4[ix(b1[168], 21)];
            let q = b0[ix(b1[29], 20)];
            let c = (p & q & w(7)) | ((p | q) & w(6));
            let p = b4[ix(b1[155], 21)];
            let f = (p & b1[105]) | ((p | b1[105]) & w(141));
            b0[3] = byte(b0[3] - b4[ix(weird_rol32(f, c), 21)]);
        }

        b1[5] = byte(
            weird_ror8(b0[12], (b0[ix(b1[61], 20)] / w(5)) & w(7)) ^ (!b2[ix(b3[84], 35)] / w(5)),
        );
        b1[198] = byte(b1[198] + b1[3]);
        {
            let a = w(162) | b2[ix(b3[64], 35)];
            b1[164] = byte(b1[164] + (a * a) / w(5));
        }

        {
            let g = weird_ror8(w(139), b3[80] & w(7));
            let p = b4[ix(b3[64], 21)];
            let c = ((p * p * p) & w(95)) | (b0[ix(b3[40], 20)] & !w(95));
            let q = b0[ix(b3[20], 20)];
            b3[92] = byte((g & w(12)) | (q & w(12)) | (g & q) | c);
        }

        b2[12] = byte(b2[12] + (((b1[103] & w(32)) | (b3[92] & (b1[103] | w(60))) | w(16)) / w(3)));
        b3[96] = b1[143];
        b3[100] = w(27);

        let picked = ((b3[40] & !b2[8]) | (b1[35] & b2[8])) & b3[64];
        let third = b3[84] / w(3);
        b3[104] = byte(picked ^ w(119));
        b3[108] = byte(w(238) & (picked << 1));
        b3[112] = byte((!b3[64] & third) ^ w(49));
        b3[116] = byte(w(98) & ((!b3[64] & third) << 1));

        {
            let a = (b1[35] & b2[8]) | (b3[40] & !b2[8]);
            let b = (a & b3[64]) | (third & !b3[64]);
            let lhs = b & (w(86) + ((b1[172] & w(64)) >> 1));
            let rhs = ((((b1[172] & w(65)) >> 1) ^ w(86)) | ((!b3[64] & third) | picked)) & b3[100];
            b1[143] = byte(b3[96] - (lhs | rhs));
        }

        b2[29] = w(162);

        {
            let a = ((b4[ix(b3[88], 21)] & w(160)) | (b0[ix(b1[125], 20)] & w(95))) >> 1;
            let b = b2[ix(b1[149], 35)] ^ (b1[43] * b1[43]);
            b0[15] = byte(b0[15] + ((b & a) | ((a | b) & w(115))));
        }

        b3[120] = byte(b3[64] - b0[ix(b3[40], 20)]);
        b1[95] = b4[ix(b3[20], 21)];

        {
            let p = b2[ix(b1[17], 35)];
            let a = weird_ror8(b2[ix(b3[80], 35)], (p * p * p) & w(7));
            b0[7] = byte(b0[7] - a * a);
        }
        {
            let p = b4[ix(b1[202], 21)];
            b2[8] = byte(b2[8] - b1[184] + p * p * p);
        }
        b0[16] = byte((b2[ix(b1[102], 35)] << 1) & w(132));
        b3[124] = byte((b4[ix(b3[40], 21)] >> 1) ^ b3[68]);
        b0[7] = byte(
            b0[7]
                - (b0[ix(b1[191], 20)]
                    - (((b4[ix(b1[80], 21)] << 1) & !w(177))
                        | (b4[ix(b4[ix(b3[88], 21)], 21)] & w(177)))),
        );
        b0[6] = b0[ix(b1[119], 20)];

        {
            let a = (b4[ix(b1[190], 21)] & !w(209)) | (b1[118] & w(209));
            let p = b0[ix(b3[120], 20)];
            let b = p * p;
            b0[12] = byte(
                (b0[ix(b3[84], 20)] ^ (b2[ix(b1[71], 35)] + b2[ix(b1[15], 35)]))
                    & ((a & b) | ((a | b) & w(27))),
            );
        }

        {
            let p = b2[ix(b3[88], 35)];
            let b = (b1[32] & p) | ((b1[32] | p) & w(23));
            let d = ((b4[ix(b1[57], 21)] * w(231)) & w(169)) | (b & w(86));
            let f = (((b0[ix(b1[82], 20)] & !w(29)) | (b4[ix(b3[124], 21)] & w(29))) & w(190))
                | (b4[ix(d / w(5), 21)] & !w(190));
            let q = b0[ix(b3[40], 20)];
            let h = q * q * q;
            let k = (h & b1[82]) | (h & w(92)) | (b1[82] & w(92));
            b3[128] = byte(((f & k) | ((f | k) & w(192))) ^ (d / w(5)));
        }

        b2[25] = byte(
            b2[25]
                ^ ((b0[ix(b3[120], 20)] << 1) * b1[5]
                    - (weird_rol8(b3[76], b4[ix(b3[124], 21)] & w(7)) & (b3[20] + w(110)))),
        );

        Garbled {
            buffer0: narrow(&b0),
            buffer1: narrow(&b1),
            buffer2: narrow(&b2),
            buffer3: narrow(&b3),
        }
    }
}
