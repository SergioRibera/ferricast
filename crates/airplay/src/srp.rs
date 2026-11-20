//! SRP-6a client (RFC 5054 style) for the two AirPlay pairing flavours.
//!
//! * Legacy `/pair-setup-pin`: SHA-1, 2048-bit group, username = sender
//!   device id, and Apple's 40-byte session key
//!   `K = H(S ‖ 0x00000000) ‖ H(S ‖ 0x00000001)`.
//! * HomeKit pair-setup: SHA-512, 3072-bit group, username `Pair-Setup`,
//!   `K = H(S)`.
//!
//! Both use `k = H(N ‖ PAD(g))`, `u = H(PAD(A) ‖ PAD(B))`,
//! `x = H(s ‖ H(I ":" P))`, `M1 = H(H(N) ⊕ H(g) ‖ H(I) ‖ s ‖ A ‖ B ‖ K)` and
//! `M2 = H(A ‖ M1 ‖ K)`.

use std::marker::PhantomData;

use ferricast_core::{FerricastError, Result};
use num_bigint::BigUint;
use rand::RngCore;
use sha2::Digest;

const N_2048: &str = "AC6BDB41324A9A9BF166DE5E1389582FAF72B6651987EE07FC3192943DB56050A37329CBB4A099ED8193E0757767A13DD52312AB4B03310DCD7F48A9DA04FD50E8083969EDB767B0CF6095179A163AB3661A05FBD5FAAAE82918A9962F0B93B855F97993EC975EEAA80D740ADBF4FF747359D041D5C33EA71D281E446B14773BCA97B43A23FB801676BD207A436C6481F1D2B9078717461A5B9D32E688F87748544523B524B0D57D5EA77A2775D2ECFA032CFBDBF52FB3786160279004E57AE6AF874E7303CE53299CCC041C7BC308D82A5698F3A8D0C38271AE35F8E9DBFBB694B5C803D89F7AE435DE236D525F54759B65E372FCD68EF20FA7111F9E4AFF73";

const N_3072: &str = "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6BF12FFA06D98A0864D87602733EC86A64521F2B18177B200CBBE117577A615D6C770988C0BAD946E208E24FA074E5AB3143DB5BFCE0FD108E4B82D120A93AD2CAFFFFFFFFFFFFFFFF";

/// SRP group and session-key convention.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SrpGroup {
    /// 2048-bit group, g = 2, Apple's interleaved 2×hash session key.
    Apple2048,
    /// 3072-bit group, g = 5, `K = H(S)`.
    Hap3072,
}

impl SrpGroup {
    fn params(self) -> (BigUint, BigUint) {
        let (n, g) = match self {
            Self::Apple2048 => (N_2048, 2_u32),
            Self::Hap3072 => (N_3072, 5_u32),
        };
        // The constants above are valid hex.
        let n = BigUint::parse_bytes(n.as_bytes(), 16).unwrap_or_default();
        (n, BigUint::from(g))
    }
}

/// Result of processing the server's salt and public key.
pub(crate) struct SrpProof {
    /// Client public key `A`, padded to the group size.
    pub public_a: Vec<u8>,
    /// Client proof `M1`.
    pub m1: Vec<u8>,
    /// Session key `K`.
    pub session_key: Vec<u8>,
    expected_m2: Vec<u8>,
}

impl SrpProof {
    /// Check the server proof `M2` (compared over the server's length, which
    /// UxPlay may truncate to the hash size).
    pub fn verify_server(&self, m2: &[u8]) -> Result<()> {
        if !m2.is_empty() && self.expected_m2.starts_with(m2) {
            Ok(())
        } else {
            Err(FerricastError::Protocol(
                "SRP server proof mismatch: the receiver does not know the PIN".into(),
            ))
        }
    }
}

/// SRP-6a client over hash `D`.
pub(crate) struct SrpClient<D> {
    group: SrpGroup,
    n: BigUint,
    g: BigUint,
    _hash: PhantomData<D>,
}

impl<D: Digest> SrpClient<D> {
    pub fn new(group: SrpGroup) -> Self {
        let (n, g) = group.params();
        Self {
            group,
            n,
            g,
            _hash: PhantomData,
        }
    }

    fn n_len(&self) -> usize {
        self.n.bits().div_ceil(8) as usize
    }

    fn pad(&self, v: &BigUint) -> Vec<u8> {
        let bytes = v.to_bytes_be();
        let mut out = vec![0_u8; self.n_len().saturating_sub(bytes.len())];
        out.extend_from_slice(&bytes);
        out
    }

    fn hash(parts: &[&[u8]]) -> Vec<u8> {
        let mut h = D::new();
        for p in parts {
            h.update(p);
        }
        h.finalize().to_vec()
    }

    fn session_key(&self, s: &BigUint) -> Vec<u8> {
        let s = s.to_bytes_be();
        match self.group {
            SrpGroup::Apple2048 => {
                let mut k = Self::hash(&[&s, &[0, 0, 0, 0]]);
                k.extend(Self::hash(&[&s, &[0, 0, 0, 1]]));
                k
            }
            SrpGroup::Hap3072 => Self::hash(&[&s]),
        }
    }

    /// Compute `A`, `M1` and `K` with a random private exponent.
    pub fn process(
        &self,
        username: &str,
        password: &str,
        salt: &[u8],
        server_b: &[u8],
    ) -> Result<SrpProof> {
        // Keep `A` full-width so the receiver hashes the same bytes whether
        // it pads or not.
        loop {
            let mut a = [0_u8; 32];
            rand::rngs::OsRng.fill_bytes(&mut a);
            let a = BigUint::from_bytes_be(&a);
            let public_a = self.g.modpow(&a, &self.n);
            if public_a.to_bytes_be().len() == self.n_len() {
                return self.process_with(username, password, salt, server_b, &a);
            }
        }
    }

    fn process_with(
        &self,
        username: &str,
        password: &str,
        salt: &[u8],
        server_b: &[u8],
        a: &BigUint,
    ) -> Result<SrpProof> {
        let n = &self.n;
        let b = BigUint::from_bytes_be(server_b);
        if (&b % n) == BigUint::default() {
            return Err(FerricastError::Protocol(
                "invalid SRP server public key".into(),
            ));
        }
        let public_a = self.g.modpow(a, n);
        let k = BigUint::from_bytes_be(&Self::hash(&[&n.to_bytes_be(), &self.pad(&self.g)]));
        let u = BigUint::from_bytes_be(&Self::hash(&[&self.pad(&public_a), &self.pad(&b)]));
        if u == BigUint::default() {
            return Err(FerricastError::Protocol(
                "invalid SRP scrambling parameter".into(),
            ));
        }
        let inner = Self::hash(&[username.as_bytes(), b":", password.as_bytes()]);
        let x = BigUint::from_bytes_be(&Self::hash(&[salt, &inner]));

        // S = (B - k·g^x)^(a + u·x) mod N, computed without going negative.
        let kgx = (&k * self.g.modpow(&x, n)) % n;
        let base = ((&b % n) + n - kgx) % n;
        let shared = base.modpow(&(a + &u * &x), n);
        let session_key = self.session_key(&shared);

        let h_n = Self::hash(&[&n.to_bytes_be()]);
        let h_g = Self::hash(&[&self.g.to_bytes_be()]);
        let h_xor: Vec<u8> = h_n.iter().zip(&h_g).map(|(x, y)| x ^ y).collect();
        let h_i = Self::hash(&[username.as_bytes()]);
        let a_bytes = public_a.to_bytes_be();
        let m1 = Self::hash(&[&h_xor, &h_i, salt, &a_bytes, &b.to_bytes_be(), &session_key]);
        let expected_m2 = Self::hash(&[&a_bytes, &m1, &session_key]);
        Ok(SrpProof {
            public_a: self.pad(&public_a),
            m1,
            session_key,
            expected_m2,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Server side of SRP-6a, used only to check the client end to end.
    fn server<D: Digest>(
        group: SrpGroup,
        user: &str,
        pin: &str,
        salt: &[u8],
    ) -> (BigUint, BigUint, Vec<u8>) {
        let c = SrpClient::<D>::new(group);
        let inner = SrpClient::<D>::hash(&[user.as_bytes(), b":", pin.as_bytes()]);
        let x = BigUint::from_bytes_be(&SrpClient::<D>::hash(&[salt, &inner]));
        let v = c.g.modpow(&x, &c.n);
        let b = BigUint::from(0x1234_5678_9abc_def0_u64);
        let k = BigUint::from_bytes_be(&SrpClient::<D>::hash(&[&c.n.to_bytes_be(), &c.pad(&c.g)]));
        let big_b = (&k * &v + c.g.modpow(&b, &c.n)) % &c.n;
        (b, big_b, v.to_bytes_be())
    }

    fn round_trip<D: Digest>(group: SrpGroup, user: &str) {
        let salt = [7_u8; 16];
        let (b, big_b, v) = server::<D>(group, user, "1234", &salt);
        let c = SrpClient::<D>::new(group);
        let proof = c
            .process(user, "1234", &salt, &big_b.to_bytes_be())
            .unwrap();
        // Server: S = (A·v^u)^b.
        let a = BigUint::from_bytes_be(&proof.public_a);
        let u = BigUint::from_bytes_be(&SrpClient::<D>::hash(&[&c.pad(&a), &c.pad(&big_b)]));
        let v = BigUint::from_bytes_be(&v);
        let s = (&a * v.modpow(&u, &c.n)).modpow(&b, &c.n);
        assert_eq!(c.session_key(&s), proof.session_key);
        let m2 = SrpClient::<D>::hash(&[&a.to_bytes_be(), &proof.m1, &proof.session_key]);
        proof.verify_server(&m2).unwrap();
        assert!(proof.verify_server(&[0; 20]).is_err());
        // Wrong PIN gives a different key.
        let wrong = c
            .process(user, "0000", &salt, &big_b.to_bytes_be())
            .unwrap();
        assert_ne!(wrong.session_key, proof.session_key);
    }

    #[test]
    fn apple_variant_agrees_with_server() {
        round_trip::<sha1::Sha1>(SrpGroup::Apple2048, "AA:BB:CC:DD:EE:FF");
        assert_eq!(
            SrpClient::<sha1::Sha1>::new(SrpGroup::Apple2048).n_len(),
            256
        );
    }

    #[test]
    fn hap_variant_agrees_with_server() {
        round_trip::<sha2::Sha512>(SrpGroup::Hap3072, "Pair-Setup");
        assert_eq!(
            SrpClient::<sha2::Sha512>::new(SrpGroup::Hap3072).n_len(),
            384
        );
    }
}
