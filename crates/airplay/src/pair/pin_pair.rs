use std::io::Read;

use chacha20poly1305::{ChaCha20Poly1305, KeyInit, Nonce, aead::Aead};
use ed25519_dalek::Signer;
use ferricast_core::{FerricastError, PairingChallenge};
use hkdf::Hkdf;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{FromPrimitive, Num};
use rand::{Rng, rngs::OsRng};
use sha2::{Digest, Sha512};
use tokio::{
    io::BufReader,
    net::tcp::{ReadHalf, WriteHalf},
};
use uuid::Uuid;

use crate::{
    rtsp::{RtspManager, RtspResponse},
    tlv::{
        self, TLV_TYPE_ENCRYPTED_DATA, TLV_TYPE_IDENTIFIER, TLV_TYPE_METHOD, TLV_TYPE_PROOF, TLV_TYPE_PUBLIC_KEY, TLV_TYPE_SALT, TLV_TYPE_SIGNATURE, TLV_TYPE_STATE
    },
};

const SRP_N: &str = concat!(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1",
    "29024E088A67CC74020BBEA63B139B22514A08798E3404DD",
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245",
    "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED",
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D",
    "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F",
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D",
    "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B",
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9",
    "DE2BCBF6955817183995497CEA956AE515D2261898FA0510",
    "15728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64",
    "ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7",
    "ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6B",
    "F12FFA06D98A0864D87602733EC86A64521F2B18177B200C",
    "BBE117577A615D6C770988C0BAD946E208E24FA074E5AB31",
    "43DB5BFCE0FD108E4B82D120A93AD2CAFFFFFFFFFFFFFFFF"
);

const SRP_G: u64 = 5;

// TODO(to SergioRibera): Request Pin in a UI-way
pub async fn pair_pin(
    challenge: &PairingChallenge,
    manager: &mut RtspManager,
    write_half: &mut WriteHalf<'_>,
    buf_reader: &mut BufReader<ReadHalf<'_>>,
    uuid: &Uuid
) -> Result<(), FerricastError> {
    let mut csprng = OsRng;
    let mut signing_key = ed25519_dalek::SigningKey::generate(&mut csprng);

    let ed_public_key = signing_key.verifying_key().to_bytes();

    let mut pin = String::new();

    if !matches!(
        challenge,
        PairingChallenge::None | PairingChallenge::Credential
    ) {
        manager
            .builder()
            .path("/pair-pin-start".to_string())
            .post()
            .write(write_half)
            .await?;

        match RtspResponse::read(buf_reader).await {
            Ok(v) => {
                v.is_ok()?;
            }
            Err(_) => {}
        };

        let mut b = vec![0_u8; 4];

        print!("Pin: ");
        std::io::stdin().read(&mut b)?;

        // TODO(Juanperias): Remove this unwrap
        pin = String::from_utf8(b).unwrap().trim().to_string();
    }

    pin_pair_setup(
        signing_key,
        &ed_public_key,
        manager,
        write_half,
        buf_reader,
        pin,
        uuid
    )
    .await?;

    Ok(())
}

pub async fn pin_pair_setup(
    signing_key: ed25519_dalek::SigningKey,
    client_public_key: &[u8],
    manager: &mut RtspManager,
    write_half: &mut WriteHalf<'_>,
    buf_reader: &mut BufReader<ReadHalf<'_>>,
    pin: String,
    id: &Uuid
) -> Result<(), FerricastError> {
    let m1 = tlv::encode(vec![(TLV_TYPE_METHOD, &[0x0]), (TLV_TYPE_STATE, &[0x01])])?;

    let m2 = {
        let mut m2 = Vec::new();

        for _ in 0..3 {
            manager
                .builder()
                .path("/pair-setup".to_string())
                .content_type("application/octet-stream".to_string())
                .body(m1.clone())
                .post()
                .write(write_half)
                .await?;

            let res = RtspResponse::read(buf_reader).await?;

            match res.content() {
                Ok(v) => {
                    if v.is_empty() {
                        continue;
                    }

                    m2 = v.clone();
                    break;
                }
                Err(_) => continue,
            }
        }

        if m2.is_empty() {
            Err(FerricastError::Protocol(
                "Airplay did not send a valid response with m2".to_string(),
            ))
        } else {
            Ok(m2)
        }
    }?;

    let m2 = tlv::decode(&m2)?;

    let server_salt = m2.get(&TLV_TYPE_SALT).ok_or(FerricastError::Protocol(
        "Invalid AirPlay Response no salt in TLV".to_string(),
    ))?;

    let server_pub = m2
        .get(&TLV_TYPE_PUBLIC_KEY)
        .ok_or(FerricastError::Protocol(
            "Invalid airplay Response no public key in TLV".to_string(),
        ))?;

    tracing::info!("Salt {:?} Server public key {:?}", server_salt, server_pub);

    let n_2048 = BigInt::from_str_radix(SRP_N, 16).map_err(|_| {
        FerricastError::Protocol("Invalid SRP_N, This is an internal ferricast error".to_string())
    })?;

    let g_2048 = BigInt::from_u64(SRP_G).unwrap();

    let inner_hash = sha2::Sha512::digest(format!("Pair-Setup:{pin}").as_bytes());

    let mut x_input = Vec::new();

    x_input.extend_from_slice(server_salt);
    x_input.extend_from_slice(&inner_hash);

    let x_hash = sha2::Sha512::digest(&x_input);

    let x = num_bigint::BigInt::from_bytes_be(num_bigint::Sign::Plus, &x_hash);

    let pad_n = pad_to(n_2048.to_bytes_be().1, 384);
    let pad_g = pad_to(g_2048.to_bytes_be().1, 384);

    let mut k_input = Vec::new();

    k_input.extend_from_slice(&pad_n);
    k_input.extend_from_slice(&pad_g);

    let k_hash = sha2::Sha512::digest(&k_input);

    let k = num_bigint::BigInt::from_bytes_be(num_bigint::Sign::Plus, &k_hash);

    let mut a_bytes = [0_u8; 32];

    rand::thread_rng().fill(&mut a_bytes);

    let mut a = num_bigint::BigInt::from_bytes_be(num_bigint::Sign::Plus, &a_bytes);

    use num_traits::Zero;

    if a.is_zero() {
        a = BigInt::from_u32(1).unwrap();
    }

    let A = g_2048.modpow(&a, &n_2048);

    let client_public = A.to_bytes_be();

    let B = BigInt::from_bytes_be(Sign::Plus, server_pub);

    if B.sign() == Sign::NoSign || B.sign() == Sign::Minus || B == n_2048 {
        return Err(FerricastError::Protocol(
            "Invalid server public key".to_string(),
        ));
    }

    let server_public = B.to_bytes_be().1;

    let mut u_input = Vec::new();

    u_input.extend(pad_to(client_public.1.clone(), 384));
    u_input.extend(pad_to(server_public, 384));

    let u_hash = Sha512::digest(&u_input);

    let u = BigInt::from_bytes_be(Sign::Plus, &u_hash);

    let gx = g_2048.modpow(&x, &n_2048);

    let kgx = (k * gx) % n_2048.clone();

    let mut diff = B - kgx;

    if diff.sign() == Sign::Minus {
        diff += n_2048.clone();
    }

    let exp = u * x + a;

    let S = diff.modpow(&exp, &n_2048);

    let K = Sha512::digest(S.to_bytes_be().1);

    let hn_hash = Sha512::digest(n_2048.to_bytes_be().1);
    let hg_hash = Sha512::digest(g_2048.to_bytes_be().1);

    let mut h_xor = vec![0_u8; 64];

    for i in 0..64 {
        h_xor[i] = hn_hash[i] ^ hg_hash[i];
    }

    let hu_hash = Sha512::digest("Pair-Setup");

    let mut proof_input = Vec::new();

    proof_input.extend(h_xor);
    proof_input.extend_from_slice(&hu_hash);
    proof_input.extend_from_slice(server_salt);
    proof_input.extend_from_slice(&client_public.1);
    proof_input.extend_from_slice(&server_pub);
    proof_input.extend_from_slice(&K);

    let m1_proof = Sha512::digest(&proof_input);

    let m3 = tlv::encode(vec![
        (TLV_TYPE_STATE, &[0x03_u8]),
        (TLV_TYPE_PUBLIC_KEY, &pad_to(client_public.1.clone(), 384)),
        (TLV_TYPE_PROOF, &m1_proof),
    ])?;

    manager
        .builder()
        .post()
        .content_type("application/octet-stream".to_string())
        .path("/pair-setup".to_string())
        .body(m3)
        .write(write_half)
        .await?;

    let res = RtspResponse::read(buf_reader).await?;

    res.is_ok()?;

    let m4 = res.content()?;
    let m4 = tlv::decode(m4)?;

    let m4_proof = m4.get(&TLV_TYPE_PROOF)
        .ok_or(FerricastError::Protocol("Invalid TLV, no proof".to_string()))?;



    tracing::info!("M4!");


    let mut m2_proof_input = Vec::new();

    m2_proof_input.extend_from_slice(&client_public.1);
    m2_proof_input.extend_from_slice(&m1_proof);
    m2_proof_input.extend_from_slice(&K);

    let m2_proof_expected = Sha512::digest(&m2_proof_input);

    if m2_proof_expected.as_slice() != m4_proof {
        return Err(FerricastError::Protocol("server proof mismatch".to_string()));
    }


 
    let k_bytes = K.as_slice();

    let session_key = hkdf(k_bytes, b"Pair-Setup-Encrypt-Salt", b"Pair-Setup-Encrypt-Info", 32)?;
    let sig_key = hkdf(k_bytes, b"Pair-Setup-Controller-Sign-Salt", b"Pair-Setup-Controller-Sign-Info", 32)?;


    let mut sig_input = Vec::new();

    sig_input.extend_from_slice(&sig_key);  
    sig_input.extend_from_slice(&id.into_bytes());
    sig_input.extend_from_slice(client_public_key);

    let signature = signing_key.sign(&sig_input);

    let body = tlv::encode(vec![
        (TLV_TYPE_IDENTIFIER, &id.into_bytes()),
        (TLV_TYPE_PUBLIC_KEY, client_public_key),
        (TLV_TYPE_SIGNATURE, &signature.to_bytes())
    ])?;

    let aead = ChaCha20Poly1305::new_from_slice(&session_key)
        .map_err(|e| FerricastError::Protocol(format!("Failed to create ChaCha20Poly1305, {:?}", e)))?;

    let mut nonce = vec![0_u8; 12];
    nonce[4..].copy_from_slice(b"PS-Msg05");

    let encrypted = aead.encrypt(Nonce::from_slice(&nonce), body.as_slice())
        .map_err(|e| FerricastError::Protocol(format!("Failed to encrypt TLV msg, {:?}", e)))?;
   

    let m5 = tlv::encode(vec![
        (TLV_TYPE_ENCRYPTED_DATA, &encrypted),
        (TLV_TYPE_STATE, &[5]),
    ])?;

    manager
        .builder()
        .post()
        .content_type("application/octet-stream".to_string())
        .path("/pair-setup".to_string())
        .body(m5)
        .write(write_half)
        .await?;

    let res = RtspResponse::read(buf_reader).await?;

    res.is_ok()?;

    let m6 = res.content()?;
    let m6 = tlv::decode(m6);

    println!("M6: {:?}", m6);
    println!("Shared Secret: {:?}", K);



    Ok(())
}

fn pad_to(data: Vec<u8>, size: usize) -> Vec<u8> {
    if data.len() >= size {
        return data;
    }

    let mut padded = vec![0_u8; size];

    padded[size - data.len()..].copy_from_slice(&data);

    padded
}

fn hkdf(secret: &[u8], salt: &[u8], info: &[u8], len: usize) -> Result<Vec<u8>, FerricastError> {
    let hkdf = Hkdf::<Sha512>::new(Some(salt), secret);

    let mut buf = vec![0_u8; len];

    hkdf.expand(info, &mut buf)
        .map_err(|_| FerricastError::Protocol("Cannot expand HKDF".to_string()))?;

    Ok(buf)
}
