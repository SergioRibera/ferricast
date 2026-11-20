//! TLV8 encoding used by HomeKit pairing.
//!
//! Values longer than 255 bytes are split into consecutive items with the
//! same tag; only *adjacent* items with the same tag are merged on decode.

use ferricast_core::{FerricastError, Result};

pub const TAG_METHOD: u8 = 0x00;
pub const TAG_IDENTIFIER: u8 = 0x01;
pub const TAG_SALT: u8 = 0x02;
pub const TAG_PUBLIC_KEY: u8 = 0x03;
pub const TAG_PROOF: u8 = 0x04;
pub const TAG_ENCRYPTED_DATA: u8 = 0x05;
pub const TAG_STATE: u8 = 0x06;
pub const TAG_ERROR: u8 = 0x07;
pub const TAG_SIGNATURE: u8 = 0x0A;
pub const TAG_FLAGS: u8 = 0x13;

/// Encode `(tag, value)` items.
pub fn encode(items: &[(u8, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    for &(tag, value) in items {
        if value.is_empty() {
            out.extend_from_slice(&[tag, 0]);
            continue;
        }
        for chunk in value.chunks(255) {
            out.push(tag);
            // `chunks(255)` bounds the length.
            out.push(u8::try_from(chunk.len()).unwrap_or(u8::MAX));
            out.extend_from_slice(chunk);
        }
    }
    out
}

/// Decoded TLV8 items in wire order.
#[derive(Debug, Default)]
pub struct Tlv(Vec<(u8, Vec<u8>)>);

impl Tlv {
    /// Decode a TLV8 buffer. A truncated item is an error.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut items: Vec<(u8, Vec<u8>)> = Vec::new();
        let mut rest = bytes;
        let mut prev_was_full = false;
        while !rest.is_empty() {
            let [tag, len, tail @ ..] = rest else {
                return Err(FerricastError::Tlv("truncated TLV header".into()));
            };
            let len = usize::from(*len);
            let value = tail
                .get(..len)
                .ok_or_else(|| FerricastError::Tlv(format!("truncated TLV item {tag:#04x}")))?;
            match items.last_mut() {
                // A 255-byte item followed by the same tag is a fragment.
                Some((last_tag, data)) if *last_tag == *tag && prev_was_full => {
                    data.extend_from_slice(value);
                }
                _ => items.push((*tag, value.to_vec())),
            }
            prev_was_full = len == 255;
            rest = &tail[len..];
        }
        Ok(Self(items))
    }

    /// First value with `tag`.
    pub fn get(&self, tag: u8) -> Option<&[u8]> {
        self.0
            .iter()
            .find(|(t, _)| *t == tag)
            .map(|(_, v)| v.as_slice())
    }

    /// Value with `tag`, or a protocol error naming `what`.
    pub fn require(&self, tag: u8, what: &str) -> Result<&[u8]> {
        self.get(tag)
            .ok_or_else(|| FerricastError::Protocol(format!("{what}: missing TLV {tag:#04x}")))
    }

    /// Fails if the accessory reported an error, or the state is not `state`.
    pub fn expect_state(&self, state: u8, what: &str) -> Result<()> {
        if let Some(code) = self.get(TAG_ERROR) {
            let code = code.first().copied().unwrap_or_default();
            let meaning = match code {
                1 => "unknown",
                2 => "authentication failed (wrong PIN?)",
                3 => "backoff, retry later",
                4 => "max peers",
                5 => "max tries",
                6 => "unavailable",
                7 => "busy",
                _ => "unrecognised",
            };
            return Err(FerricastError::Protocol(format!(
                "{what}: receiver error {code} ({meaning})"
            )));
        }
        match self.get(TAG_STATE) {
            Some([s]) if *s == state => Ok(()),
            other => Err(FerricastError::Protocol(format!(
                "{what}: expected state {state}, got {other:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_values_round_trip() {
        let long = vec![9_u8; 600];
        let bytes = encode(&[
            (TAG_STATE, &[2]),
            (TAG_PUBLIC_KEY, &long),
            (TAG_SALT, &[1, 2]),
        ]);
        let tlv = Tlv::decode(&bytes).unwrap();
        assert_eq!(tlv.get(TAG_PUBLIC_KEY), Some(long.as_slice()));
        assert_eq!(tlv.get(TAG_SALT), Some(&[1_u8, 2][..]));
        tlv.expect_state(2, "test").unwrap();
    }

    #[test]
    fn non_adjacent_same_tags_stay_separate() {
        let bytes = encode(&[
            (TAG_IDENTIFIER, b"a"),
            (TAG_STATE, &[1]),
            (TAG_IDENTIFIER, b"b"),
        ]);
        let tlv = Tlv::decode(&bytes).unwrap();
        assert_eq!(tlv.get(TAG_IDENTIFIER), Some(&b"a"[..]));
    }

    #[test]
    fn truncation_and_errors_are_reported() {
        assert!(Tlv::decode(&[TAG_STATE]).is_err());
        assert!(Tlv::decode(&[TAG_STATE, 4, 1]).is_err());
        let tlv = Tlv::decode(&encode(&[(TAG_STATE, &[4]), (TAG_ERROR, &[2])])).unwrap();
        assert!(tlv.expect_state(4, "test").is_err());
    }
}
