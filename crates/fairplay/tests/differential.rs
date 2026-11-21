//! Compares the Rust port against outputs of UxPlay's C FairPlay code.
//!
//! `data/playfair_vectors.txt` was printed by `data/gen_vectors.c` linked
//! against UxPlay 1.73.6 (`lib/fairplay_playfair.c` and `lib/playfair`);
//! that file documents how to rebuild it. The inputs are not stored: both
//! sides draw them from the same splitmix64 stream.

use ferricast_fairplay::{
    EKEY_LEN, FP_SETUP2_LEN, FairPlayError, decrypt_ekey, setup_message1, setup1_reply,
    setup2_reply,
};

const VECTORS: &str = include_str!("data/playfair_vectors.txt");
const CASES: usize = 400;

/// splitmix64 seeded like `gen_vectors.c`.
struct SplitMix64(u64);

impl Default for SplitMix64 {
    fn default() -> Self {
        Self(0x4650_4c59)
    }
}

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn fill(&mut self, out: &mut [u8]) {
        for byte in out {
            *byte = self.next_u64().to_be_bytes()[0];
        }
    }
}

const KEYMSG_HEADER: [u8; 12] = [b'F', b'P', b'L', b'Y', 3, 1, 3, 0, 0, 0, 0, 0x98];
const EKEY_HEADER: [u8; 12] = [b'F', b'P', b'L', b'Y', 1, 2, 1, 0, 0, 0, 0, 0x3c];

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

/// The hex payloads of the lines tagged `tag`, in file order.
fn tagged(tag: &str) -> Vec<Vec<u8>> {
    VECTORS
        .lines()
        .filter_map(|line| line.split_once(' '))
        .filter(|(line_tag, _)| *line_tag == tag)
        .map(|(_, payload)| unhex(payload))
        .collect()
}

#[test]
fn setup1_replies_match_c_for_every_mode() {
    let expected = tagged("setup1");
    assert_eq!(expected.len(), 4);
    for (mode, reply) in (0..4).zip(&expected) {
        assert_eq!(
            &setup1_reply(&setup_message1(mode)).unwrap()[..],
            &reply[..]
        );
    }
}

#[test]
fn setup2_replies_and_keys_match_c_for_random_messages() {
    let replies = tagged("setup2");
    let keys = tagged("key");
    assert_eq!((replies.len(), keys.len()), (CASES, CASES));

    let mut rng = SplitMix64::default();
    for (case, (reply, key)) in replies.iter().zip(&keys).enumerate() {
        let mut keymsg = [0_u8; FP_SETUP2_LEN];
        rng.fill(&mut keymsg);
        keymsg[..12].copy_from_slice(&KEYMSG_HEADER);
        keymsg[12] = u8::try_from(case % 4).unwrap();
        let mut ekey = [0_u8; EKEY_LEN];
        rng.fill(&mut ekey);
        ekey[..12].copy_from_slice(&EKEY_HEADER);

        assert_eq!(
            &setup2_reply(&keymsg).unwrap()[..],
            &reply[..],
            "case {case}"
        );
        assert_eq!(
            &decrypt_ekey(&keymsg, &ekey).unwrap()[..],
            &key[..],
            "case {case}"
        );
    }
}

#[test]
fn modes_without_tables_are_errors_not_reads_out_of_bounds() {
    let mut keymsg = [0_u8; FP_SETUP2_LEN];
    keymsg[..12].copy_from_slice(&KEYMSG_HEADER);
    let ekey = [0_u8; EKEY_LEN];
    for mode in 4..=u8::MAX {
        keymsg[12] = mode;
        assert_eq!(decrypt_ekey(&keymsg, &ekey), Err(FairPlayError::Mode(mode)));
        let mut message1 = setup_message1(0);
        message1[14] = mode;
        assert_eq!(setup1_reply(&message1), Err(FairPlayError::Mode(mode)));
    }
}
