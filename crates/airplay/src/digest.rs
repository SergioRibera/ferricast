//! RFC 2617 Digest authentication for password-protected receivers.
//!
//! UxPlay `-pw` answers the first `SETUP` with `401 Unauthorized` and
//! `WWW-Authenticate: Digest realm="raop", nonce="<hex>"`; the sender
//! repeats the request with an `Authorization` header (UxPlay
//! `pairing_digest_verify`, no `qop`):
//! `response = MD5(MD5(user:realm:password):nonce:MD5(method:uri))`, all
//! as lowercase hex. UxPlay reads the fields in the order username,
//! realm, nonce, uri, response, so they are written in that order.

use md5::{Digest, Md5};

/// User name senders put in the `Authorization` header; receivers do not
/// check it beyond hashing it.
const USERNAME: &str = "AirPlay";

/// A `WWW-Authenticate: Digest …` challenge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DigestChallenge {
    realm: String,
    nonce: String,
}

fn md5_hex(input: &str) -> String {
    hex::encode(Md5::digest(input.as_bytes()))
}

/// Value of `key="…"` in a comma-separated parameter list.
fn quoted_param<'a>(params: &'a str, key: &str) -> Option<&'a str> {
    params.split(',').find_map(|part| {
        let (k, v) = part.trim().split_once('=')?;
        if !k.trim().eq_ignore_ascii_case(key) {
            return None;
        }
        let v = v.trim();
        Some(v.strip_prefix('"')?.strip_suffix('"').unwrap_or(v))
    })
}

impl DigestChallenge {
    /// Parse a `WWW-Authenticate` header value; `None` unless it is a
    /// Digest challenge with a nonce.
    pub(crate) fn parse(header: &str) -> Option<Self> {
        let (scheme, params) = header.trim().split_once(' ')?;
        if !scheme.eq_ignore_ascii_case("Digest") {
            return None;
        }
        Some(Self {
            realm: quoted_param(params, "realm").unwrap_or_default().to_owned(),
            nonce: quoted_param(params, "nonce")?.to_owned(),
        })
    }

    /// `Authorization` header value answering this challenge.
    pub(crate) fn authorization(&self, method: &str, uri: &str, password: &str) -> String {
        let ha1 = md5_hex(&format!("{USERNAME}:{}:{password}", self.realm));
        let ha2 = md5_hex(&format!("{method}:{uri}"));
        let response = md5_hex(&format!("{ha1}:{}:{ha2}", self.nonce));
        format!(
            "Digest username=\"{USERNAME}\", realm=\"{}\", nonce=\"{}\", uri=\"{uri}\", response=\"{response}\"",
            self.realm, self.nonce
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_uxplay_challenge() {
        let c = DigestChallenge::parse("Digest realm=\"raop\", nonce=\"0123abcd\"").unwrap();
        assert_eq!(c.realm, "raop");
        assert_eq!(c.nonce, "0123abcd");
        assert!(DigestChallenge::parse("Basic realm=\"x\"").is_none());
        assert!(DigestChallenge::parse("Digest realm=\"raop\"").is_none());
    }

    #[test]
    fn response_matches_rfc_2617_without_qop() {
        // RFC 2069-style digest (no qop): MD5(HA1:nonce:HA2).
        let c = DigestChallenge {
            realm: "raop".into(),
            nonce: "abc".into(),
        };
        let auth = c.authorization("SETUP", "rtsp://10.0.0.2/1234", "secret");
        let ha1 = md5_hex("AirPlay:raop:secret");
        let ha2 = md5_hex("SETUP:rtsp://10.0.0.2/1234");
        let expected = md5_hex(&format!("{ha1}:abc:{ha2}"));
        assert!(auth.ends_with(&format!("response=\"{expected}\"")));
        let order: Vec<usize> = ["username=", "realm=", "nonce=", "uri=", "response="]
            .iter()
            .map(|k| auth.find(k).unwrap())
            .collect();
        assert!(order.windows(2).all(|w| w[0] < w[1]), "{auth}");
    }

    #[test]
    fn md5_matches_the_rfc_1321_vector() {
        assert_eq!(md5_hex("abc"), "900150983cd24fb0d6963f7d28e17f72");
    }
}
