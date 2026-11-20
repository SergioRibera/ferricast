//! Small cryptographic helpers shared by the pairing and streaming code.

use ferricast_core::{FerricastError, Result};
use hkdf::Hkdf;
use sha2::{Digest, Sha512};

/// AES-128 in big-endian 128-bit counter mode, as used by every AirPlay
/// AES-CTR stream (pair-verify signatures, mirroring payloads).
pub(crate) type Aes128Ctr = ctr::Ctr128BE<aes::Aes128>;

/// First 16 bytes of `SHA-512(parts…)`.
pub(crate) fn sha512_16(parts: &[&[u8]]) -> [u8; 16] {
    let mut hasher = Sha512::new();
    for part in parts {
        hasher.update(part);
    }
    let digest = hasher.finalize();
    let mut out = [0_u8; 16];
    out.copy_from_slice(&digest[..16]);
    out
}

/// HKDF-SHA-512 expand of `ikm` into `N` bytes.
pub(crate) fn hkdf_sha512<const N: usize>(ikm: &[u8], salt: &[u8], info: &[u8]) -> Result<[u8; N]> {
    let mut out = [0_u8; N];
    Hkdf::<Sha512>::new(Some(salt), ikm)
        .expand(info, &mut out)
        .map_err(|_| FerricastError::Protocol("HKDF output length invalid".into()))?;
    Ok(out)
}

/// 32-byte array from a slice of exactly that length.
pub(crate) fn array32(bytes: &[u8], what: &str) -> Result<[u8; 32]> {
    bytes.try_into().map_err(|_| {
        FerricastError::Protocol(format!("{what}: expected 32 bytes, got {}", bytes.len()))
    })
}
