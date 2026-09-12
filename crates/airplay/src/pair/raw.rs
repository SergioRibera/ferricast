use aes::{Aes128, cipher::{KeyIvInit, StreamCipher}};
use ctr::Ctr128BE;
use sha2::{Digest, Sha512};
use tokio::net::TcpStream;

use ferricast_core::{FerricastError, device::Features};
use rand::rngs::OsRng;
use tokio::io::{BufReader, WriteHalf};
use x25519_dalek::{PublicKey, StaticSecret};

use crate::rtsp::{RtspManager, RtspResponse};


pub async fn raw_pair(features: &Features, manager: &mut RtspManager, write_half: &mut WriteHalf<TcpStream>, buf_reader: &mut BufReader<TcpStream>) -> Result<(), FerricastError> { 
    let mut csprng = OsRng; 
    let signing_key = ed25519_dalek::SigningKey::generate(&mut csprng);

    let ed_public_key = signing_key.verifying_key().to_bytes();

    manager.builder()
            .post()
            .path("/pair-setup".to_string())
            .content_type("application/octet-stream".to_string())
            .body(ed_public_key.to_vec())
            .write(write_half).await?;

    let req = RtspResponse::read(buf_reader).await?;

    req.is_ok()?;    

    let server_pub = req.content()?;

    let mix_fairplay_key = !features.supports_legacy_pairing();

    let verify_headers = {
        if mix_fairplay_key {
            vec![
                ("X-Apple-PD".to_string(), "1".to_string())
            ]
        } else {
            vec![]
        }
    };

    let client_secret = StaticSecret::new(OsRng);
    let public = PublicKey::from(&client_secret);
    
    let mut v1 = vec![0_u8; 68];

    v1[0] = 1;

    v1[4..36]
        .copy_from_slice(public.as_bytes());

    v1[36..68]
        .copy_from_slice(&ed_public_key);

    
    manager.builder()
            .post()
            .path("/pair-setup".to_string())
            .content_type("application/octet-stream".to_string())
            .body(v1)
            .write(write_half).await?;


    let res = RtspResponse::read(buf_reader).await?;

    res.is_ok()?;

    let v2 = res.content()?;

    if v2.len() != 96 {
        return Err(FerricastError::Protocol("Raw Pairing Failed, V2 package is not 96 bytes".to_string()));
    } 

    let server_public = &v2[..32];

    let encrypted_server_sig = &v2[32..96];

    let server_p: [u8; 32] = server_public.try_into().unwrap();

    
    let shared_secret = client_secret.diffie_hellman(&PublicKey::from(server_p));
    let shared_secret_bytes =  shared_secret.as_bytes();

    let aes_key = sha512_derive_key("Pair-Verify-AES-Key", shared_secret_bytes);
    let aes_iv = sha512_derive_key("Pair-Verify-AES-IV", shared_secret_bytes);

    let mut cipher: ctr::Ctr128BE<Aes128> = Ctr128BE::new(aes_key.as_slice().try_into().unwrap(), aes_iv.as_slice().try_into().unwrap());

    let mut server_sig = encrypted_server_sig.to_vec();
    
    cipher.apply_keystream(&mut server_sig);

    let mut server_sig_msg = vec![0_u8; 64];

    server_sig_msg[..32]
        .copy_from_slice(server_public);

    server_sig_msg[32..]
        .copy_from_slice(public.as_bytes());


    // TODO: implement get_info()
    
    Ok(())
}

fn sha512_derive_key(salt: &str, secret: &[u8]) -> Vec<u8> {
    let mut hasher = Sha512::new();

    hasher.update(salt.as_bytes());
    hasher.update(secret);

    let result = hasher.finalize();

    result[..16].to_vec()
}
