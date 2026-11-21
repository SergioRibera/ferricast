//! Receiver side of legacy pairing: `/pair-setup`, `/pair-verify` and the
//! PIN variant (`/pair-pin-start` + `/pair-setup-pin`). The sender side
//! and the wire format are in [`crate::pair::legacy`] and
//! [`crate::pair::legacy_pin`].

use aes::cipher::{KeyIvInit, StreamCipher};
use aes_gcm::aead::consts::U16;
use aes_gcm::aead::{AeadInPlace, KeyInit};
use aes_gcm::{AesGcm, Tag};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use ferricast_core::{FerricastError, Result};
use plist::{Dictionary, Value};
use rand::rngs::OsRng;
use x25519_dalek::{PublicKey, StaticSecret};

use crate::crypto::{Aes128Ctr, array32, sha512_16};
use crate::srp::{SrpGroup, SrpServer};

type Aes128Gcm16 = AesGcm<aes::Aes128, U16>;

fn protocol(msg: &str) -> FerricastError {
    FerricastError::Protocol(msg.into())
}

/// A pair-verify in progress (after its first message).
struct Verify {
    client_ed: VerifyingKey,
    client_x: [u8; 32],
    server_x: [u8; 32],
    ecdh: [u8; 32],
    /// The AES-CTR stream, already advanced past the server signature.
    cipher: Aes128Ctr,
}

/// A PIN pair-setup in progress.
enum PinSetup {
    /// `pair-pin-start` showed this PIN; waiting for the SRP start.
    Shown(String),
    /// SRP started for this sender.
    Srp {
        server: Box<SrpServer<sha1::Sha1>>,
        device_id: String,
    },
    /// SRP proof accepted; waiting for the sender's key.
    Proven { key: Vec<u8>, device_id: String },
}

/// What a completed PIN pair-setup established.
pub(super) struct PinPaired {
    pub device_id: String,
    pub public_key: [u8; 32],
}

/// Pairing state of one control connection.
#[derive(Default)]
pub(super) struct Pairing {
    verify: Option<Verify>,
    pin: Option<PinSetup>,
    /// Shared secret of a completed pair-verify.
    ecdh: Option<[u8; 32]>,
}

impl Pairing {
    /// Shared secret of a completed pair-verify.
    pub fn ecdh(&self) -> Option<[u8; 32]> {
        self.ecdh
    }

    /// `pair-verify` step 1: `[1,0,0,0] ‖ client X25519 ‖ client Ed25519`.
    /// Returns `server X25519 ‖ encrypted signature`.
    pub fn verify_start(&mut self, identity: &SigningKey, body: &[u8]) -> Result<Vec<u8>> {
        if body.len() != 68 {
            return Err(protocol("pair-verify: bad first message"));
        }
        let client_x = array32(&body[4..36], "pair-verify client key")?;
        let client_ed = VerifyingKey::from_bytes(&array32(&body[36..68], "pair-verify identity")?)
            .map_err(|_| protocol("pair-verify: invalid client Ed25519 key"))?;
        let secret = StaticSecret::random_from_rng(OsRng);
        let server_x = *PublicKey::from(&secret).as_bytes();
        let ecdh = *secret.diffie_hellman(&PublicKey::from(client_x)).as_bytes();
        let key = sha512_16(&[b"Pair-Verify-AES-Key", &ecdh]);
        let iv = sha512_16(&[b"Pair-Verify-AES-IV", &ecdh]);
        let mut cipher = Aes128Ctr::new(&key.into(), &iv.into());
        let mut signed = [0_u8; 64];
        signed[..32].copy_from_slice(&server_x);
        signed[32..].copy_from_slice(&client_x);
        let mut signature = identity.sign(&signed).to_bytes();
        cipher.apply_keystream(&mut signature);
        let mut reply = Vec::with_capacity(96);
        reply.extend_from_slice(&server_x);
        reply.extend_from_slice(&signature);
        self.verify = Some(Verify {
            client_ed,
            client_x,
            server_x,
            ecdh,
            cipher,
        });
        self.ecdh = None;
        Ok(reply)
    }

    /// The Ed25519 key the sender is pair-verifying with.
    pub fn verifying_client(&self) -> Option<[u8; 32]> {
        self.verify.as_ref().map(|v| v.client_ed.to_bytes())
    }

    /// `pair-verify` step 2: `[0,0,0,0] ‖ encrypted client signature`.
    pub fn verify_finish(&mut self, body: &[u8]) -> Result<()> {
        let mut verify = self
            .verify
            .take()
            .ok_or_else(|| protocol("pair-verify: signature without a handshake"))?;
        if body.len() != 68 {
            return Err(protocol("pair-verify: bad second message"));
        }
        let mut signature = [0_u8; 64];
        signature.copy_from_slice(&body[4..]);
        verify.cipher.apply_keystream(&mut signature);
        let mut signed = [0_u8; 64];
        signed[..32].copy_from_slice(&verify.client_x);
        signed[32..].copy_from_slice(&verify.server_x);
        verify
            .client_ed
            .verify(&signed, &Signature::from_bytes(&signature))
            .map_err(|_| protocol("pair-verify: sender signature is invalid"))?;
        self.ecdh = Some(verify.ecdh);
        Ok(())
    }

    /// `pair-pin-start`: remember the PIN the user is shown.
    pub fn pin_start(&mut self, pin: String) {
        self.pin = Some(PinSetup::Shown(pin));
    }

    /// One `pair-setup-pin` message (a binary plist dictionary). Returns
    /// the reply dictionary, and the paired sender once the exchange is
    /// complete.
    pub fn pin_setup(
        &mut self,
        identity: &SigningKey,
        request: &Dictionary,
    ) -> Result<(Dictionary, Option<PinPaired>)> {
        let data = |key: &str| request.get(key).and_then(Value::as_data);
        let mut reply = Dictionary::new();
        if let (Some(method), Some(user)) = (
            request.get("method").and_then(Value::as_string),
            request.get("user").and_then(Value::as_string),
        ) {
            if method != "pin" {
                return Err(protocol("pair-setup-pin: unsupported method"));
            }
            let Some(PinSetup::Shown(pin)) = self.pin.take() else {
                return Err(protocol("pair-setup-pin before pair-pin-start"));
            };
            let server = SrpServer::<sha1::Sha1>::new(SrpGroup::Apple2048, user, &pin);
            reply.insert("pk".into(), Value::Data(server.public_b()));
            reply.insert("salt".into(), Value::Data(server.salt().to_vec()));
            self.pin = Some(PinSetup::Srp {
                server: Box::new(server),
                device_id: user.to_owned(),
            });
            return Ok((reply, None));
        }
        if let (Some(pk), Some(proof)) = (data("pk"), data("proof")) {
            let Some(PinSetup::Srp { server, device_id }) = self.pin.take() else {
                return Err(protocol("pair-setup-pin: proof out of order"));
            };
            let session = server.verify_client(pk, proof)?;
            reply.insert("proof".into(), Value::Data(session.m2));
            self.pin = Some(PinSetup::Proven {
                key: session.session_key,
                device_id,
            });
            return Ok((reply, None));
        }
        if let (Some(epk), Some(tag)) = (data("epk"), data("authTag")) {
            let Some(PinSetup::Proven { key, device_id }) = self.pin.take() else {
                return Err(protocol("pair-setup-pin: key exchange out of order"));
            };
            let aes_key = sha512_16(&[b"Pair-Setup-AES-Key", &key]);
            let mut iv = sha512_16(&[b"Pair-Setup-AES-IV", &key]);
            let cipher = Aes128Gcm16::new(&aes_key.into());
            let mut client_pk = array32(epk, "pair-setup-pin epk")?;
            let tag = Tag::<U16>::from_exact_iter(tag.iter().copied())
                .ok_or_else(|| protocol("pair-setup-pin: bad authTag length"))?;
            iv[15] = iv[15].wrapping_add(1);
            cipher
                .decrypt_in_place_detached(&iv.into(), &[], &mut client_pk, &tag)
                .map_err(|_| protocol("pair-setup-pin: sender key failed authentication"))?;
            iv[15] = iv[15].wrapping_add(1);
            let mut server_pk = identity.verifying_key().to_bytes();
            let server_tag = cipher
                .encrypt_in_place_detached(&iv.into(), &[], &mut server_pk)
                .map_err(|_| protocol("pair-setup-pin: encryption failed"))?;
            reply.insert("epk".into(), Value::Data(server_pk.to_vec()));
            reply.insert("authTag".into(), Value::Data(server_tag.to_vec()));
            return Ok((
                reply,
                Some(PinPaired {
                    device_id,
                    public_key: client_pk,
                }),
            ));
        }
        Err(protocol("pair-setup-pin: unrecognised message"))
    }
}

/// A random four-digit PIN.
pub(super) fn random_pin() -> String {
    format!("{:04}", rand::random::<u16>() % 10_000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::srp::SrpClient;

    /// The sender side of pair-verify, as `pair::legacy` runs it.
    #[test]
    fn verify_round_trip_with_the_sender_algorithm() {
        let server_id = SigningKey::generate(&mut OsRng);
        let client_id = SigningKey::generate(&mut OsRng);
        let client_secret = StaticSecret::random_from_rng(OsRng);
        let client_x = PublicKey::from(&client_secret);
        let mut m1 = vec![1, 0, 0, 0];
        m1.extend_from_slice(client_x.as_bytes());
        m1.extend_from_slice(client_id.verifying_key().as_bytes());

        let mut pairing = Pairing::default();
        let reply = pairing.verify_start(&server_id, &m1).unwrap();
        let server_x: [u8; 32] = reply[..32].try_into().unwrap();
        let ecdh = *client_secret
            .diffie_hellman(&PublicKey::from(server_x))
            .as_bytes();
        let key = sha512_16(&[b"Pair-Verify-AES-Key", &ecdh]);
        let iv = sha512_16(&[b"Pair-Verify-AES-IV", &ecdh]);
        let mut cipher = Aes128Ctr::new(&key.into(), &iv.into());
        let mut sig = [0_u8; 64];
        sig.copy_from_slice(&reply[32..]);
        cipher.apply_keystream(&mut sig);
        let mut signed = [0_u8; 64];
        signed[..32].copy_from_slice(&server_x);
        signed[32..].copy_from_slice(client_x.as_bytes());
        server_id
            .verifying_key()
            .verify(&signed, &Signature::from_bytes(&sig))
            .unwrap();
        let mut to_sign = [0_u8; 64];
        to_sign[..32].copy_from_slice(client_x.as_bytes());
        to_sign[32..].copy_from_slice(&server_x);
        let mut client_sig = client_id.sign(&to_sign).to_bytes();
        cipher.apply_keystream(&mut client_sig);
        let mut m2 = vec![0, 0, 0, 0];
        m2.extend_from_slice(&client_sig);
        pairing.verify_finish(&m2).unwrap();
        assert_eq!(pairing.ecdh(), Some(ecdh));

        // A forged signature fails.
        let mut pairing = Pairing::default();
        pairing.verify_start(&server_id, &m1).unwrap();
        assert!(pairing.verify_finish(&[0; 68]).is_err());
    }

    #[test]
    fn pin_setup_runs_srp_and_swaps_keys() {
        let server_id = SigningKey::generate(&mut OsRng);
        let mut pairing = Pairing::default();
        pairing.pin_start("1234".into());
        let mut m1 = Dictionary::new();
        m1.insert("method".into(), Value::String("pin".into()));
        m1.insert("user".into(), Value::String("AA:BB".into()));
        let (r1, done) = pairing.pin_setup(&server_id, &m1).unwrap();
        assert!(done.is_none());
        let proof = SrpClient::<sha1::Sha1>::new(SrpGroup::Apple2048)
            .process(
                "AA:BB",
                "1234",
                r1["salt"].as_data().unwrap(),
                r1["pk"].as_data().unwrap(),
            )
            .unwrap();
        let mut m2 = Dictionary::new();
        m2.insert("pk".into(), Value::Data(proof.public_a.clone()));
        m2.insert("proof".into(), Value::Data(proof.m1.clone()));
        let (r2, _) = pairing.pin_setup(&server_id, &m2).unwrap();
        proof.verify_server(r2["proof"].as_data().unwrap()).unwrap();
        // The epk step is covered end to end by the loop test.
    }
}
