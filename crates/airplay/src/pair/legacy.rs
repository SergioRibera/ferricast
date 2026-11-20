//! Legacy AirPlay pairing (`/pair-setup` + `/pair-verify`).
//!
//! Wire format as implemented by UxPlay `lib/pairing.c`:
//!
//! * pair-setup: client sends its 32-byte Ed25519 public key, the receiver
//!   answers with its own.
//! * pair-verify 1: `[1,0,0,0] ‖ client X25519 ‖ client Ed25519` →
//!   `server X25519 ‖ AES-CTR(sign(server X25519 ‖ client X25519))`.
//! * pair-verify 2: `[0,0,0,0] ‖ AES-CTR(sign(client X25519 ‖ server X25519))`
//!   where the CTR stream continues after the first 64 bytes.
//!
//! The AES key / IV are the first 16 bytes of `SHA-512("Pair-Verify-AES-Key"
//! ‖ secret)` / `SHA-512("Pair-Verify-AES-IV" ‖ secret)`.

use aes::cipher::{KeyIvInit, StreamCipher};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use ferricast_core::{FerricastError, Result};
use rand::rngs::OsRng;
use x25519_dalek::{PublicKey, StaticSecret};

use crate::crypto::{Aes128Ctr, array32, sha512_16};
use crate::rtsp::{CONTENT_OCTET, RtspClient};

/// Register our Ed25519 key and learn the receiver's.
pub(crate) async fn pair_setup(rtsp: &mut RtspClient, identity: &SigningKey) -> Result<[u8; 32]> {
    let response = rtsp
        .request(
            "POST",
            "/pair-setup",
            Some(CONTENT_OCTET),
            identity.verifying_key().as_bytes(),
        )
        .await?
        .ensure_success("pair-setup")?;
    array32(&response.body, "pair-setup response")
}

/// Run pair-verify against `server_key` and return the X25519 secret.
pub(crate) async fn pair_verify(
    rtsp: &mut RtspClient,
    identity: &SigningKey,
    server_key: &[u8; 32],
) -> Result<[u8; 32]> {
    let server_key = VerifyingKey::from_bytes(server_key)
        .map_err(|_| FerricastError::Protocol("receiver Ed25519 key is invalid".into()))?;
    let secret = StaticSecret::random_from_rng(OsRng);
    let public = PublicKey::from(&secret);

    let mut m1 = Vec::with_capacity(68);
    m1.extend_from_slice(&[1, 0, 0, 0]);
    m1.extend_from_slice(public.as_bytes());
    m1.extend_from_slice(identity.verifying_key().as_bytes());
    let response = rtsp
        .request("POST", "/pair-verify", Some(CONTENT_OCTET), &m1)
        .await?
        .ensure_success("pair-verify")?;
    if response.body.len() != 96 {
        // UxPlay answers 200 with an empty body when it does not recognise
        // this sender (no pair-setup on this connection, or not registered).
        return Err(FerricastError::Protocol(format!(
            "pair-verify refused by receiver ({} byte reply)",
            response.body.len()
        )));
    }
    let server_public = array32(&response.body[..32], "pair-verify server key")?;
    let shared = secret.diffie_hellman(&PublicKey::from(server_public));
    let ecdh = *shared.as_bytes();

    let key = sha512_16(&[b"Pair-Verify-AES-Key", &ecdh]);
    let iv = sha512_16(&[b"Pair-Verify-AES-IV", &ecdh]);
    let mut cipher = Aes128Ctr::new(&key.into(), &iv.into());

    let mut server_sig = [0_u8; 64];
    server_sig.copy_from_slice(&response.body[32..]);
    cipher.apply_keystream(&mut server_sig);
    let mut signed = [0_u8; 64];
    signed[..32].copy_from_slice(&server_public);
    signed[32..].copy_from_slice(public.as_bytes());
    server_key
        .verify(&signed, &Signature::from_bytes(&server_sig))
        .map_err(|_| {
            FerricastError::Protocol("pair-verify: receiver signature is invalid".into())
        })?;

    let mut to_sign = [0_u8; 64];
    to_sign[..32].copy_from_slice(public.as_bytes());
    to_sign[32..].copy_from_slice(&server_public);
    let mut client_sig = identity.sign(&to_sign).to_bytes();
    // The keystream continues after the 64 bytes used for the server's
    // signature.
    cipher.apply_keystream(&mut client_sig);
    let mut m2 = Vec::with_capacity(68);
    m2.extend_from_slice(&[0, 0, 0, 0]);
    m2.extend_from_slice(&client_sig);
    rtsp.request("POST", "/pair-verify", Some(CONTENT_OCTET), &m2)
        .await?
        .ensure_success("pair-verify (signature)")?;
    Ok(ecdh)
}
