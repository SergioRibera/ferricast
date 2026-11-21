//! Receiver-side FairPlay key decryption (`playfair_decrypt`).
//!
//! Port of UxPlay 1.73.6 `lib/playfair` (`playfair.c`, `omg_hax.c`,
//! `modified_md5.c`, `sap_hash.c`, `hand_garble.c`), originally by Esteban
//! Kubata and licensed GPL-3.0. The C code reinterprets byte buffers as
//! host-endian `uint32_t` words; this port spells those views out as
//! little-endian, the byte order of the x86 and ARM hosts UxPlay runs on.

mod garble;
mod md5;
mod sap_hash;
mod tables;

use crate::{EKEY_LEN, FP_SETUP2_LEN, FairPlayError};
use tables::{
    DEFAULT_SAP, INDEX_MANGLE, INITIAL_SESSION_KEY, MESSAGE_IV, MESSAGE_KEY, STATIC_SOURCE_1,
    STATIC_SOURCE_2, T_KEY, TABLE_S1, TABLE_S2, TABLE_S3, TABLE_S4, TABLE_S5, TABLE_S6, TABLE_S7,
    TABLE_S8, TABLE_S9, TABLE_S10, X_KEY, Z_KEY,
};

/// One 16-byte cipher block.
type Block = [u8; 16];

/// Round keys of the AES-like cipher, as four little-endian words each.
type KeySchedule = [[u32; 4]; 11];

/// Offset of the mode byte in the key message.
const KEYMSG_MODE_OFFSET: usize = 12;

/// Where the session AES key and its mask sit inside the `ekey`.
const EKEY_KEY_RANGE: std::ops::Range<usize> = 16..32;
const EKEY_MASK_RANGE: std::ops::Range<usize> = 56..72;

/// For each output byte, the input byte the cipher's row shift moves there.
const SHIFT_SOURCE: [usize; 16] = [0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3];

/// Which 256-byte slice of `TABLE_S3` substitutes each output byte.
const S3_SLICE: [usize; 16] = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11];

/// The `fp-setup` mode a key message was produced for; selects the message
/// key and IV. Only modes 0 to 3 have tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Mode(usize);

impl TryFrom<u8> for Mode {
    type Error = FairPlayError;

    fn try_from(mode: u8) -> Result<Self, Self::Error> {
        let index = usize::from(mode);
        if index < MESSAGE_KEY.len() {
            Ok(Self(index))
        } else {
            Err(FairPlayError::Mode(mode))
        }
    }
}

impl Mode {
    /// Mode 3 chains the key message blocks back to front.
    fn is_reversed(self) -> bool {
        self.0 == 3
    }
}

/// Recovers the session AES key from the `ekey`, using the key message to
/// derive the wrapping key.
pub(crate) fn decrypt(
    keymsg: &[u8; FP_SETUP2_LEN],
    ekey: &[u8; EKEY_LEN],
) -> Result<Block, FairPlayError> {
    let mode = Mode::try_from(keymsg[KEYMSG_MODE_OFFSET])?;
    let sap_key = session_key(keymsg, mode);
    let schedule = key_schedule(&sap_key);

    let mut block = xor(&block_at(ekey, EKEY_MASK_RANGE.start), &Z_KEY);
    cycle(&mut block, &schedule);
    let wrapped = block_at(ekey, EKEY_KEY_RANGE.start);
    let mut key = xor(&block, &wrapped);
    key = xor(&key, &X_KEY);
    Ok(xor(&key, &Z_KEY))
}

/// Copies the 16 bytes of `bytes` starting at `offset`.
fn block_at(bytes: &[u8], offset: usize) -> Block {
    let mut block = [0_u8; 16];
    block.copy_from_slice(&bytes[offset..offset + 16]);
    block
}

fn xor(a: &Block, b: &Block) -> Block {
    std::array::from_fn(|i| a[i] ^ b[i])
}

fn word(block: &Block, index: usize) -> u32 {
    let at = 4 * index;
    u32::from_le_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]])
}

fn set_word(block: &mut Block, index: usize, value: u32) {
    block[4 * index..4 * index + 4].copy_from_slice(&value.to_le_bytes());
}

/// Moves every byte to its row-shifted position and substitutes it:
/// output byte `p` becomes `sbox(p, input[SHIFT_SOURCE[p]])`.
fn shift_substitute(block: &mut Block, sbox: impl Fn(usize, u8) -> u8) {
    let input = *block;
    for (position, byte) in block.iter_mut().enumerate() {
        *byte = sbox(position, input[SHIFT_SOURCE[position]]);
    }
}

/// The 256-byte substitution box `index` of a table made of such boxes.
fn sbox_slice(table: &[u8], index: usize) -> &[u8] {
    &table[index << 8..(index + 1) << 8]
}

/// Rijndael-like key expansion with custom S-boxes (`generate_key_schedule`).
fn key_schedule(key_material: &Block) -> KeySchedule {
    let mut schedule = [[0_u32; 4]; 11];
    let mut buffer = xor(key_material, &T_KEY);
    for (round, (round_keys, mangle)) in schedule.iter_mut().zip(INDEX_MANGLE).enumerate() {
        let sbox = |k: usize| sbox_slice(&TABLE_S1, (31 * (4 * round + k)) % 0x28);
        round_keys[0] = word(&buffer, 0);
        buffer[0] ^= sbox(0)[usize::from(buffer[13])] ^ mangle;
        buffer[1] ^= sbox(1)[usize::from(buffer[14])];
        buffer[2] ^= sbox(2)[usize::from(buffer[15])];
        buffer[3] ^= sbox(3)[usize::from(buffer[12])];
        for (index, round_key) in round_keys.iter_mut().enumerate().skip(1) {
            *round_key = word(&buffer, index);
            let chained = word(&buffer, index) ^ word(&buffer, index - 1);
            set_word(&mut buffer, index, chained);
        }
    }
    schedule
}

fn xor_round_key(block: &mut Block, round_keys: &[u32; 4]) {
    for (index, round_key) in round_keys.iter().enumerate() {
        let value = word(block, index) ^ round_key;
        set_word(block, index, value);
    }
}

/// One block of the AES-like decryption (`cycle`).
fn cycle(block: &mut Block, schedule: &KeySchedule) {
    xor_round_key(block, &schedule[10]);
    shift_substitute(block, |position, byte| {
        sbox_slice(&TABLE_S3, S3_SLICE[position])[usize::from(byte)]
    });
    for round in 0..9 {
        let round_keys = &schedule[9 - round];
        for (index, round_key) in round_keys.iter().enumerate() {
            let key = round_key.to_le_bytes();
            let at = 4 * index;
            let lookup = |table: &[u32; 256], k: usize| table[usize::from(block[at + k] ^ key[k])];
            let value = lookup(&TABLE_S5, 3)
                ^ lookup(&TABLE_S6, 2)
                ^ lookup(&TABLE_S7, 1)
                ^ lookup(&TABLE_S8, 0);
            set_word(block, index, value);
        }
        let table_round = 8 - round;
        shift_substitute(block, |position, byte| {
            let index = (71 * (table_round * 16 + position)) % 144;
            sbox_slice(&TABLE_S4, index)[usize::from(byte)]
        });
    }
    xor_round_key(block, &schedule[0]);
}

/// Decrypts the eight cipher blocks of the key message (`decryptMessage`).
fn decrypt_message(keymsg: &[u8; FP_SETUP2_LEN], mode: Mode) -> [u8; 128] {
    let message_key = &MESSAGE_KEY[mode.0];
    let mut decrypted = [0_u8; 128];
    for i in 0..8 {
        // Mode 3 walks the blocks from the last to the first; the others
        // start after the 16-byte header. Either way each block is chained
        // to the message block in front of its output slot.
        let (input, output) = if mode.is_reversed() {
            (0x80 - 0x10 * i, 0x70 - 0x10 * i)
        } else {
            (0x10 * (i + 1), 0x10 * i)
        };
        let mut buffer = block_at(keymsg, input);
        for j in 0..9 {
            let base = 0x80 - 0x10 * j;
            shift_substitute(&mut buffer, |position, byte| {
                let index = (97 * (base + position)) % 144;
                sbox_slice(&TABLE_S2, index)[usize::from(byte)] ^ message_key[base + position]
            });
            for index in 0..4 {
                let at = 4 * index;
                let value = TABLE_S9[usize::from(buffer[at])]
                    ^ TABLE_S9[0x100 + usize::from(buffer[at + 1])]
                    ^ TABLE_S9[0x200 + usize::from(buffer[at + 2])]
                    ^ TABLE_S9[0x300 + usize::from(buffer[at + 3])];
                set_word(&mut buffer, index, value);
            }
        }
        shift_substitute(&mut buffer, |position, byte| {
            sbox_slice(&TABLE_S10, position)[usize::from(byte)]
        });
        let chain = if output == 0 {
            MESSAGE_IV[mode.0]
        } else {
            block_at(keymsg, output)
        };
        decrypted[output..output + 16].copy_from_slice(&xor(&buffer, &chain));
    }
    decrypted
}

/// Derives the key that wraps the session AES key (`generate_session_key`
/// with UxPlay's `default_sap`).
fn session_key(keymsg: &[u8; FP_SETUP2_LEN], mode: Mode) -> Block {
    let decrypted = decrypt_message(keymsg, mode);
    let mut sap = [0_u8; 320];
    let (static1, rest) = sap.split_at_mut(STATIC_SOURCE_1.len());
    static1.copy_from_slice(&STATIC_SOURCE_1);
    let (message, rest) = rest.split_at_mut(decrypted.len());
    message.copy_from_slice(&decrypted);
    let (old_sap, static2) = rest.split_at_mut(0x80);
    old_sap.copy_from_slice(&DEFAULT_SAP[0x80..0x100]);
    static2.copy_from_slice(&STATIC_SOURCE_2);

    let mut key = INITIAL_SESSION_KEY;
    let (rounds, _) = sap.as_chunks::<64>();
    for round in rounds {
        let digest = md5::modified_md5(round, &key);
        key = sap_hash::sap_hash(round);
        for index in 0..4 {
            let sum = word(&key, index).wrapping_add(word(&digest, index));
            set_word(&mut key, index, sum);
        }
    }
    for index in 0..4 {
        // Swap each word's byte order, then mask with 121.
        let swapped = word(&key, index).swap_bytes();
        set_word(&mut key, index, swapped);
    }
    key.map(|byte| byte ^ 121)
}
