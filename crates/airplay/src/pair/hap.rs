//! HomeKit pairing over `/pair-setup` and `/pair-verify` (TLV8 bodies).
//!
//! * Normal pair-setup (M1–M6): SRP-6a with the PIN shown on the receiver,
//!   then an exchange of long-term Ed25519 keys. The receiver's key is
//!   stored so later sessions only run pair-verify.
//! * Transient pair-setup (M1–M4): fixed PIN `3939`, flag `0x10`; the SRP
//!   session key is used directly and nothing is stored.
//! * Pair-verify (M1–M4): X25519 exchange authenticated with the stored
//!   long-term keys.

use chacha20poly1305::ChaCha20Poly1305;
use chacha20poly1305::aead::{Aead, KeyInit};
use ed25519_dalek::{Signature, Signer, Verifier, VerifyingKey};
use ferricast_core::{FerricastError, Result};
use rand::rngs::OsRng;
use x25519_dalek::{PublicKey, StaticSecret};

use super::hap_nonce;
use crate::crypto::{array32, hkdf_sha512};
use crate::rtsp::{CONTENT_OCTET, RtspClient};
use crate::srp::{SrpClient, SrpGroup};
use crate::store::{Identity, ReceiverKey};
use crate::tlv::{
    self, TAG_ENCRYPTED_DATA, TAG_FLAGS, TAG_IDENTIFIER, TAG_METHOD, TAG_PROOF, TAG_PUBLIC_KEY,
    TAG_SALT, TAG_SIGNATURE, TAG_STATE, Tlv,
};

/// PIN used by transient pairing.
const TRANSIENT_PIN: &str = "3939";
/// `Flags` TLV value requesting transient pairing.
const FLAG_TRANSIENT: u8 = 0x10;

async fn post(rtsp: &mut RtspClient, path: &str, body: &[u8], what: &str) -> Result<Tlv> {
    let response = rtsp
        .request("POST", path, Some(CONTENT_OCTET), body)
        .await?;
    // Receivers report pairing errors inside the TLV with a 200 status, but
    // some use 470 on authentication failure.
    if !response.is_success() && response.body.is_empty() {
        return Err(FerricastError::Protocol(format!(
            "{what}: {} {}",
            response.status, response.reason
        )));
    }
    Tlv::decode(&response.body)
}

fn aead(key: &[u8; 32]) -> ChaCha20Poly1305 {
    ChaCha20Poly1305::new(key.into())
}

/// Run pair-setup. Returns the SRP session key and, for normal pairing, the
/// receiver's long-term key.
pub(crate) async fn pair_setup(
    rtsp: &mut RtspClient,
    identity: &Identity,
    pin: Option<&str>,
) -> Result<(Vec<u8>, Option<ReceiverKey>)> {
    let transient = pin.is_none();
    let pin = pin.unwrap_or(TRANSIENT_PIN);

    let mut m1 = vec![(TAG_METHOD, &[0_u8][..]), (TAG_STATE, &[1_u8][..])];
    if transient {
        m1.push((TAG_FLAGS, &[FLAG_TRANSIENT][..]));
    }
    let m2 = post(rtsp, "/pair-setup", &tlv::encode(&m1), "pair-setup M2").await?;
    m2.expect_state(2, "pair-setup M2")?;
    let salt = m2.require(TAG_SALT, "pair-setup M2")?;
    let server_b = m2.require(TAG_PUBLIC_KEY, "pair-setup M2")?;

    let proof = SrpClient::<sha2::Sha512>::new(SrpGroup::Hap3072).process(
        "Pair-Setup",
        pin,
        salt,
        server_b,
    )?;
    let m3 = tlv::encode(&[
        (TAG_STATE, &[3]),
        (TAG_PUBLIC_KEY, &proof.public_a),
        (TAG_PROOF, &proof.m1),
    ]);
    let m4 = post(rtsp, "/pair-setup", &m3, "pair-setup M4").await?;
    m4.expect_state(4, "pair-setup M4")?;
    proof.verify_server(m4.require(TAG_PROOF, "pair-setup M4")?)?;
    let session_key = proof.session_key;
    if transient {
        return Ok((session_key, None));
    }

    // M5: our long-term key, signed over HKDF-derived controller material.
    let encrypt_key: [u8; 32] = hkdf_sha512(
        &session_key,
        b"Pair-Setup-Encrypt-Salt",
        b"Pair-Setup-Encrypt-Info",
    )?;
    let controller_x: [u8; 32] = hkdf_sha512(
        &session_key,
        b"Pair-Setup-Controller-Sign-Salt",
        b"Pair-Setup-Controller-Sign-Info",
    )?;
    let ltpk = identity.key.verifying_key().to_bytes();
    let mut signed = Vec::with_capacity(32 + identity.pairing_id.len() + 32);
    signed.extend_from_slice(&controller_x);
    signed.extend_from_slice(identity.pairing_id.as_bytes());
    signed.extend_from_slice(&ltpk);
    let signature = identity.key.sign(&signed).to_bytes();
    let inner = tlv::encode(&[
        (TAG_IDENTIFIER, identity.pairing_id.as_bytes()),
        (TAG_PUBLIC_KEY, &ltpk),
        (TAG_SIGNATURE, &signature),
    ]);
    let sealed = aead(&encrypt_key)
        .encrypt(&hap_nonce(b"PS-Msg05"), inner.as_slice())
        .map_err(|_| FerricastError::Protocol("pair-setup M5: encryption failed".into()))?;
    let m5 = tlv::encode(&[(TAG_STATE, &[5]), (TAG_ENCRYPTED_DATA, &sealed)]);
    let m6 = post(rtsp, "/pair-setup", &m5, "pair-setup M6").await?;
    m6.expect_state(6, "pair-setup M6")?;

    // M6: the receiver's long-term key, signed the same way.
    let plain = aead(&encrypt_key)
        .decrypt(
            &hap_nonce(b"PS-Msg06"),
            m6.require(TAG_ENCRYPTED_DATA, "pair-setup M6")?,
        )
        .map_err(|_| FerricastError::Protocol("pair-setup M6: authentication failed".into()))?;
    let inner = Tlv::decode(&plain)?;
    let accessory_id = inner.require(TAG_IDENTIFIER, "pair-setup M6")?;
    let accessory_ltpk = array32(
        inner.require(TAG_PUBLIC_KEY, "pair-setup M6")?,
        "pair-setup M6 key",
    )?;
    let accessory_sig = inner.require(TAG_SIGNATURE, "pair-setup M6")?;
    let accessory_x: [u8; 32] = hkdf_sha512(
        &session_key,
        b"Pair-Setup-Accessory-Sign-Salt",
        b"Pair-Setup-Accessory-Sign-Info",
    )?;
    let mut signed = Vec::new();
    signed.extend_from_slice(&accessory_x);
    signed.extend_from_slice(accessory_id);
    signed.extend_from_slice(&accessory_ltpk);
    verify(&accessory_ltpk, &signed, accessory_sig, "pair-setup M6")?;

    let key = ReceiverKey {
        id: String::from_utf8_lossy(accessory_id).into_owned(),
        public_key: hex::encode(accessory_ltpk),
    };
    Ok((session_key, Some(key)))
}

fn verify(public: &[u8; 32], message: &[u8], signature: &[u8], what: &str) -> Result<()> {
    let key = VerifyingKey::from_bytes(public)
        .map_err(|_| FerricastError::Protocol(format!("{what}: invalid receiver key")))?;
    let signature = Signature::from_slice(signature)
        .map_err(|_| FerricastError::Protocol(format!("{what}: malformed signature")))?;
    key.verify(message, &signature)
        .map_err(|_| FerricastError::Protocol(format!("{what}: receiver signature is invalid")))
}

/// Run pair-verify with stored credentials; returns the session secret.
pub(crate) async fn pair_verify(
    rtsp: &mut RtspClient,
    identity: &Identity,
    receiver: &ReceiverKey,
) -> Result<Vec<u8>> {
    let receiver_ltpk = receiver
        .public_key_bytes()
        .ok_or_else(|| FerricastError::Protocol("stored receiver key is corrupt".into()))?;
    let secret = StaticSecret::random_from_rng(OsRng);
    let public = PublicKey::from(&secret);

    let m1 = tlv::encode(&[(TAG_STATE, &[1]), (TAG_PUBLIC_KEY, public.as_bytes())]);
    let m2 = post(rtsp, "/pair-verify", &m1, "pair-verify M2").await?;
    m2.expect_state(2, "pair-verify M2")?;
    let server_public = array32(
        m2.require(TAG_PUBLIC_KEY, "pair-verify M2")?,
        "pair-verify M2 key",
    )?;
    let shared = secret.diffie_hellman(&PublicKey::from(server_public));
    let session_key: [u8; 32] = hkdf_sha512(
        shared.as_bytes(),
        b"Pair-Verify-Encrypt-Salt",
        b"Pair-Verify-Encrypt-Info",
    )?;
    let plain = aead(&session_key)
        .decrypt(
            &hap_nonce(b"PV-Msg02"),
            m2.require(TAG_ENCRYPTED_DATA, "pair-verify M2")?,
        )
        .map_err(|_| FerricastError::Protocol("pair-verify M2: authentication failed".into()))?;
    let inner = Tlv::decode(&plain)?;
    let accessory_id = inner.require(TAG_IDENTIFIER, "pair-verify M2")?;
    if accessory_id != receiver.id.as_bytes() {
        return Err(FerricastError::Protocol(
            "pair-verify M2: receiver identity does not match the stored pairing".into(),
        ));
    }
    let mut signed = Vec::new();
    signed.extend_from_slice(&server_public);
    signed.extend_from_slice(accessory_id);
    signed.extend_from_slice(public.as_bytes());
    verify(
        &receiver_ltpk,
        &signed,
        inner.require(TAG_SIGNATURE, "pair-verify M2")?,
        "pair-verify M2",
    )?;

    let mut signed = Vec::new();
    signed.extend_from_slice(public.as_bytes());
    signed.extend_from_slice(identity.pairing_id.as_bytes());
    signed.extend_from_slice(&server_public);
    let signature = identity.key.sign(&signed).to_bytes();
    let inner = tlv::encode(&[
        (TAG_IDENTIFIER, identity.pairing_id.as_bytes()),
        (TAG_SIGNATURE, &signature),
    ]);
    let sealed = aead(&session_key)
        .encrypt(&hap_nonce(b"PV-Msg03"), inner.as_slice())
        .map_err(|_| FerricastError::Protocol("pair-verify M3: encryption failed".into()))?;
    let m3 = tlv::encode(&[(TAG_STATE, &[3]), (TAG_ENCRYPTED_DATA, &sealed)]);
    let m4 = post(rtsp, "/pair-verify", &m3, "pair-verify M4").await?;
    m4.expect_state(4, "pair-verify M4")?;
    Ok(shared.as_bytes().to_vec())
}

/// Control-channel keys `(write, read)` derived from a HomeKit session secret.
pub(crate) fn control_keys(secret: &[u8]) -> Result<([u8; 32], [u8; 32])> {
    Ok((
        hkdf_sha512(secret, b"Control-Salt", b"Control-Write-Encryption-Key")?,
        hkdf_sha512(secret, b"Control-Salt", b"Control-Read-Encryption-Key")?,
    ))
}
