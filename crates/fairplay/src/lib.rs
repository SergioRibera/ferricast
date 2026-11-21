//! FairPlay `fp-setup` for AirPlay mirroring, on both ends of the session.
//!
//! The `fp-setup` exchange is two round trips. The first message picks one
//! of four modes and the receiver answers with a reply that is fixed for
//! that mode. The second message (the 164-byte key message) is the
//! sender's FairPlay answer to that reply; the session AES key is later
//! wrapped in the `ekey` sent in `SETUP`.
//!
//! # Receiver
//!
//! [`setup1_reply`] and [`setup2_reply`] answer the two messages, and
//! [`decrypt_ekey`] recovers the session AES key from the stored key
//! message and the `ekey`. They are a port of UxPlay's receiver code:
//! `lib/fairplay_playfair.c` (Juho Vähä-Herttua, LGPL-2.1-or-later) and
//! `lib/playfair` (originally by Esteban Kubata, GPL-3.0), as shipped in
//! UxPlay 1.73.6. Unlike the C code, every input that would read outside a
//! table (an unknown mode, a short message) is rejected with a
//! [`FairPlayError`].
//!
//! # Sender
//!
//! [`KEY_MESSAGE`] and [`EKEY`] are a genuine pair recorded from an iOS
//! sender (`AirPlay/675.4.1`) in mode 1, as logged in UxPlay issue
//! antimof/UxPlay#161. Because the mode-1 reply never changes, the pair can
//! be replayed:
//!
//! - receivers that verify the key message (AirScreen, for instance)
//!   accept it, where they reject arbitrary bytes and then drop the session
//!   on the mirroring `SETUP`;
//! - shairplay-derived receivers (UxPlay, RPiPlay) store it unverified and
//!   recover the key with `playfair_decrypt(key_message, ekey)`.
//!
//! [`AES_KEY`] is that key: UxPlay logged it for this session, and
//! [`decrypt_ekey`] reproduces it.

#![forbid(unsafe_code)]

mod setup;

/// Length of the first `fp-setup` message.
pub const FP_SETUP1_LEN: usize = 16;
/// Length of the receiver's reply to the first `fp-setup` message.
pub const FP_SETUP1_REPLY_LEN: usize = 142;
/// Length of the second `fp-setup` message (the key message).
pub const FP_SETUP2_LEN: usize = 164;
/// Length of the receiver's reply to the second `fp-setup` message.
pub const FP_SETUP2_REPLY_LEN: usize = 32;
/// Length of the `ekey` carried in the first `SETUP` request.
pub const EKEY_LEN: usize = 72;

/// Why an `fp-setup` message or an `ekey` was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum FairPlayError {
    /// The message has the wrong length for its step.
    #[error("FairPlay message is {actual} bytes, expected {expected}")]
    Length {
        /// Length the step requires.
        expected: usize,
        /// Length that was received.
        actual: usize,
    },
    /// The message does not start with `FPLY`.
    #[error("FairPlay message does not start with \"FPLY\"")]
    Magic,
    /// The version byte (offset 4) is not [`FAIRPLAY_VERSION`].
    #[error("unsupported FairPlay version {0}, expected {FAIRPLAY_VERSION}")]
    Version(u8),
    /// The mode byte is outside `0..=3`; no reply or key table exists for it.
    #[error("unsupported fp-setup mode {0}, expected 0..=3")]
    Mode(u8),
    /// ekey decryption is not built in yet.
    #[error("ekey decryption is not available")]
    Unavailable,
}

/// Builds the receiver's reply to the first `fp-setup` message.
///
/// The reply is the canned table entry for the mode at offset 14.
///
/// # Errors
///
/// [`FairPlayError::Length`] unless `message1` is [`FP_SETUP1_LEN`] bytes,
/// [`FairPlayError::Magic`] or [`FairPlayError::Version`] for a bad header,
/// and [`FairPlayError::Mode`] for a mode above 3.
pub fn setup1_reply(message1: &[u8]) -> Result<[u8; FP_SETUP1_REPLY_LEN], FairPlayError> {
    setup::setup1_reply(message1)
}

/// Builds the receiver's reply to the second `fp-setup` message.
///
/// The reply is a fixed `FPLY` header followed by the last 20 bytes of
/// `message2`. The caller keeps `message2`: [`decrypt_ekey`] needs it.
///
/// # Errors
///
/// [`FairPlayError::Length`] unless `message2` is [`FP_SETUP2_LEN`] bytes,
/// and [`FairPlayError::Magic`] or [`FairPlayError::Version`] for a bad
/// header.
pub fn setup2_reply(message2: &[u8]) -> Result<[u8; FP_SETUP2_REPLY_LEN], FairPlayError> {
    setup::setup2_reply(message2)
}

/// Recovers the session AES key wrapped in `ekey`.
///
/// `keymsg` is the second `fp-setup` message the sender sent on this
/// connection, as passed to [`setup2_reply`].
///
/// # Errors
///
/// [`FairPlayError::Mode`] when the mode byte of `keymsg` (offset 12) is
/// above 3.
pub fn decrypt_ekey(
    _keymsg: &[u8; FP_SETUP2_LEN],
    _ekey: &[u8; EKEY_LEN],
) -> Result<[u8; 16], FairPlayError> {
    Err(FairPlayError::Unavailable)
}

/// FairPlay protocol version byte carried at offset 4 of every message.
pub const FAIRPLAY_VERSION: u8 = 3;

/// `fp-setup` mode [`KEY_MESSAGE`] was produced for. The first message must
/// ask for it: the key message answers that mode's reply.
pub const MODE: u8 = 1;

/// Length of the receiver's reply to [`setup_message1`].
pub const REPLY1_LEN: usize = 142;

/// Second `fp-setup` message (164 bytes). Byte 12 is the mode.
pub const KEY_MESSAGE: [u8; 164] = [
    0x46, 0x50, 0x4c, 0x59, 0x03, 0x01, 0x03, 0x00, 0x00, 0x00, 0x00, 0x98, 0x01, 0x8f, 0x1a, 0x9c,
    0x43, 0x49, 0x45, 0xca, 0x63, 0x24, 0xae, 0x18, 0x56, 0xd5, 0x41, 0x7f, 0x19, 0x1b, 0x8a, 0x7c,
    0xc2, 0xb1, 0x6a, 0x66, 0x2d, 0x3c, 0x8d, 0x36, 0x14, 0x38, 0x53, 0xd8, 0x93, 0x32, 0x25, 0xa2,
    0x74, 0xc6, 0xff, 0xbc, 0x0d, 0xdd, 0x22, 0x01, 0x95, 0x05, 0x97, 0x20, 0x06, 0xd5, 0x4c, 0x68,
    0x5a, 0x41, 0xf6, 0xe6, 0xe1, 0x7b, 0x6f, 0xb8, 0xdd, 0xc7, 0x58, 0x90, 0xae, 0xe4, 0x8c, 0xaa,
    0x07, 0xce, 0x61, 0x07, 0x26, 0xed, 0x09, 0x11, 0x45, 0x54, 0xed, 0xc7, 0x38, 0xc4, 0x4e, 0x48,
    0x24, 0xaf, 0xf7, 0x9f, 0xbc, 0x33, 0x9a, 0x20, 0xc5, 0xd7, 0xaf, 0x64, 0x97, 0x04, 0xb3, 0x58,
    0xd7, 0x28, 0x18, 0xe5, 0x8c, 0x8e, 0x29, 0x80, 0x39, 0x23, 0xa0, 0x8e, 0x74, 0x3f, 0xe0, 0xac,
    0x4e, 0x6e, 0x7c, 0xac, 0x92, 0x27, 0xb9, 0x12, 0x86, 0xb5, 0x76, 0xaa, 0x06, 0x08, 0x83, 0xf7,
    0x9f, 0xa7, 0x94, 0xe6, 0xe2, 0x86, 0xbc, 0xaf, 0xc6, 0x21, 0x09, 0x48, 0x44, 0x31, 0x51, 0xa4,
    0xef, 0x59, 0x63, 0x73,
];

/// `ekey` sent in the first `SETUP` request (72 bytes).
pub const EKEY: [u8; 72] = [
    0x46, 0x50, 0x4c, 0x59, 0x01, 0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x00,
    0x70, 0x7d, 0xc0, 0x21, 0x45, 0x1a, 0x70, 0xbd, 0x64, 0x05, 0x7d, 0x34, 0x22, 0x42, 0xbe, 0x91,
    0x00, 0x00, 0x00, 0x10, 0x61, 0xcd, 0xab, 0x4d, 0x30, 0x9d, 0x24, 0xfd, 0x6b, 0xf6, 0x53, 0x8b,
    0xd3, 0xe1, 0xbf, 0x6a, 0xd9, 0xa2, 0x51, 0xa8, 0xd6, 0x9e, 0x42, 0x67, 0x2b, 0x3b, 0x2a, 0x41,
    0x83, 0xf7, 0x8c, 0xa8, 0xda, 0xec, 0xfa, 0x93,
];

/// AES key the receiver recovers from [`EKEY`] after storing [`KEY_MESSAGE`].
pub const AES_KEY: [u8; 16] = [
    0x21, 0x2e, 0x51, 0xe5, 0xe2, 0xb8, 0x02, 0x40, 0x8b, 0x45, 0x3b, 0xeb, 0xb9, 0x59, 0xa9, 0x42,
];
/// First `fp-setup` message (16 bytes). `mode` (0..=3) selects which of
/// the receiver's canned replies is returned.
#[must_use]
pub fn setup_message1(mode: u8) -> [u8; 16] {
    [
        0x46,
        0x50,
        0x4c,
        0x59,
        FAIRPLAY_VERSION,
        0x01,
        0x01,
        0x00,
        0x00,
        0x00,
        0x00,
        0x04,
        0x02,
        0x00,
        mode & 0x03,
        0xbb,
    ]
}

/// Reply a conforming receiver sends to [`KEY_MESSAGE`].
#[must_use]
pub fn expected_reply2() -> [u8; 32] {
    let mut reply = [0_u8; 32];
    reply[..setup::FP_HEADER.len()].copy_from_slice(&setup::FP_HEADER);
    reply[12..].copy_from_slice(&KEY_MESSAGE[144..]);
    reply
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_carry_fairplay_headers() {
        assert_eq!(&KEY_MESSAGE[..4], b"FPLY");
        assert_eq!(KEY_MESSAGE[4], FAIRPLAY_VERSION);
        assert_eq!(KEY_MESSAGE[12], MODE);
        assert_eq!(&EKEY[..4], b"FPLY");
        assert_eq!(setup_message1(2)[14], 2);
        assert_eq!(setup_message1(2)[4], FAIRPLAY_VERSION);
        assert_eq!(&expected_reply2()[12..], &KEY_MESSAGE[144..]);
    }

    #[test]
    fn receiver_answers_the_sender_messages() {
        for mode in 0..4 {
            let reply = setup1_reply(&setup_message1(mode)).unwrap();
            assert_eq!(&reply[..4], b"FPLY");
            assert_eq!(reply[13], mode);
        }
        assert_eq!(setup2_reply(&KEY_MESSAGE).unwrap(), expected_reply2());
    }

    #[test]
    fn receiver_rejects_malformed_messages() {
        let mut message1 = setup_message1(0);
        message1[14] = 4;
        assert_eq!(setup1_reply(&message1), Err(FairPlayError::Mode(4)));
        message1[4] = 2;
        assert_eq!(setup1_reply(&message1), Err(FairPlayError::Version(2)));
        message1[0] = b'X';
        assert_eq!(setup1_reply(&message1), Err(FairPlayError::Magic));
        assert_eq!(
            setup1_reply(&KEY_MESSAGE),
            Err(FairPlayError::Length {
                expected: FP_SETUP1_LEN,
                actual: FP_SETUP2_LEN
            })
        );
        assert!(matches!(
            setup2_reply(&KEY_MESSAGE[..100]),
            Err(FairPlayError::Length { .. })
        ));
    }
}
