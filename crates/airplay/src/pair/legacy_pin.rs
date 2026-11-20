//! PIN pairing for legacy receivers (`/pair-pin-start` + `/pair-setup-pin`).
//!
//! Bodies are binary plists. As implemented by UxPlay
//! (`raop_handler_pairsetup_pin`, `srp_confirm_pair_setup`):
//!
//! 1. `{method: "pin", user: <device id>}` → `{pk: B, salt}`
//! 2. `{pk: A, proof: M1}` → `{proof: M2}` (SRP-6a, SHA-1, 2048-bit group)
//! 3. `{epk, authTag}` → `{epk, authTag}`: each side's Ed25519 public key
//!    under AES-128-GCM with a 16-byte IV. Key / IV are the first 16 bytes
//!    of `SHA-512("Pair-Setup-AES-Key" ‖ K)` / `SHA-512("Pair-Setup-AES-IV"
//!    ‖ K)`; the IV's last byte is incremented once for our message and once
//!    more for the receiver's.

use aes_gcm::aead::consts::U16;
use aes_gcm::aead::{AeadInPlace, KeyInit};
use aes_gcm::{AesGcm, Tag};
use ed25519_dalek::SigningKey;
use ferricast_core::{FerricastError, Result};
use plist::{Dictionary, Value};

use crate::crypto::{array32, sha512_16};
use crate::rtsp::{CONTENT_BPLIST, RtspClient};
use crate::srp::{SrpClient, SrpGroup};

type Aes128Gcm16 = AesGcm<aes::Aes128, U16>;

/// Ask the receiver to display a PIN.
pub(crate) async fn pin_start(rtsp: &mut RtspClient) -> Result<()> {
    rtsp.request("POST", "/pair-pin-start", None, &[])
        .await?
        .ensure_success("pair-pin-start")?;
    Ok(())
}

fn to_bplist(dict: Dictionary) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    plist::to_writer_binary(&mut out, &Value::Dictionary(dict))
        .map_err(|e| FerricastError::Protocol(format!("encode plist: {e}")))?;
    Ok(out)
}

fn from_bplist(bytes: &[u8], what: &str) -> Result<Dictionary> {
    plist::from_bytes::<Value>(bytes)
        .ok()
        .and_then(Value::into_dictionary)
        .ok_or_else(|| FerricastError::Protocol(format!("{what}: reply is not a plist dictionary")))
}

fn data<'a>(dict: &'a Dictionary, key: &str, what: &str) -> Result<&'a [u8]> {
    dict.get(key)
        .and_then(Value::as_data)
        .ok_or_else(|| FerricastError::Protocol(format!("{what}: missing `{key}`")))
}

async fn post(rtsp: &mut RtspClient, dict: Dictionary, what: &str) -> Result<Dictionary> {
    let body = to_bplist(dict)?;
    let response = rtsp
        .request("POST", "/pair-setup-pin", Some(CONTENT_BPLIST), &body)
        .await?;
    if response.status == 470 {
        return Err(FerricastError::Protocol(format!(
            "{what}: receiver rejected the PIN"
        )));
    }
    let response = response.ensure_success(what)?;
    from_bplist(&response.body, what)
}

/// Run the SRP exchange with `pin` and return the receiver's Ed25519 key.
pub(crate) async fn pin_setup(
    rtsp: &mut RtspClient,
    device_id: &str,
    pin: &str,
    identity: &SigningKey,
) -> Result<[u8; 32]> {
    let mut m1 = Dictionary::new();
    m1.insert("method".into(), Value::String("pin".into()));
    m1.insert("user".into(), Value::String(device_id.into()));
    let reply = post(rtsp, m1, "pair-setup-pin (1)").await?;
    let salt = data(&reply, "salt", "pair-setup-pin (1)")?;
    let server_b = data(&reply, "pk", "pair-setup-pin (1)")?;

    let proof = SrpClient::<sha1::Sha1>::new(SrpGroup::Apple2048)
        .process(device_id, pin, salt, server_b)?;
    let mut m2 = Dictionary::new();
    m2.insert("pk".into(), Value::Data(proof.public_a.clone()));
    m2.insert("proof".into(), Value::Data(proof.m1.clone()));
    let reply = post(rtsp, m2, "pair-setup-pin (2)").await?;
    proof.verify_server(data(&reply, "proof", "pair-setup-pin (2)")?)?;

    let key = sha512_16(&[b"Pair-Setup-AES-Key", &proof.session_key]);
    let mut iv = sha512_16(&[b"Pair-Setup-AES-IV", &proof.session_key]);
    let cipher = Aes128Gcm16::new(&key.into());

    iv[15] = iv[15].wrapping_add(1);
    let mut epk = identity.verifying_key().to_bytes();
    let tag = cipher
        .encrypt_in_place_detached(&iv.into(), &[], &mut epk)
        .map_err(|_| FerricastError::Protocol("pair-setup-pin: encryption failed".into()))?;
    let mut m3 = Dictionary::new();
    m3.insert("epk".into(), Value::Data(epk.to_vec()));
    m3.insert("authTag".into(), Value::Data(tag.to_vec()));
    let reply = post(rtsp, m3, "pair-setup-pin (3)").await?;

    let mut server_pk = array32(
        data(&reply, "epk", "pair-setup-pin (3)")?,
        "pair-setup-pin epk",
    )?;
    let server_tag = data(&reply, "authTag", "pair-setup-pin (3)")?;
    let server_tag = Tag::<U16>::from_exact_iter(server_tag.iter().copied())
        .ok_or_else(|| FerricastError::Protocol("pair-setup-pin: bad authTag length".into()))?;
    iv[15] = iv[15].wrapping_add(1);
    cipher
        .decrypt_in_place_detached(&iv.into(), &[], &mut server_pk, &server_tag)
        .map_err(|_| {
            FerricastError::Protocol("pair-setup-pin: receiver key failed authentication".into())
        })?;
    Ok(server_pk)
}
