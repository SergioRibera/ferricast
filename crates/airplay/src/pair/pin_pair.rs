use ferricast_core::{FerricastError, PairingChallenge};
use rand::rngs::OsRng;
use tokio::{io::{BufReader, WriteHalf}, net::TcpStream};

use crate::{rtsp::{RtspManager, RtspResponse}, tlv::{self, TLV_TYPE_METHOD, TLV_TYPE_PUBLIC_KEY, TLV_TYPE_SALT, TLV_TYPE_STATE}};

// TODO(to SergioRibera): Request Pin in a UI-way
pub async fn pair_pin(challenge: &PairingChallenge, manager: &mut RtspManager, write_half: &mut WriteHalf<TcpStream>, buf_reader: &mut BufReader<TcpStream>) -> Result<(), FerricastError> {
    let mut csprng = OsRng; 
    let mut signing_key = ed25519_dalek::SigningKey::generate(&mut csprng);

    let ed_public_key = signing_key.verifying_key().to_bytes();


    if !matches!(challenge, PairingChallenge::None | PairingChallenge::Credential)  {
        manager.builder()
            .path("/pair-pin-start".to_string())
            .post()
            .write(write_half)
            .await?;

        RtspResponse::read(buf_reader).await?.is_ok()?;
    }

    pin_pair_setup(signing_key, &ed_public_key, manager, write_half, buf_reader).await?;

    Ok(())

}

pub async fn pin_pair_setup(signing_key: ed25519_dalek::SigningKey, client_public_key: &[u8], manager: &mut RtspManager, write_half: &mut WriteHalf<TcpStream>, buf_reader: &mut BufReader<TcpStream>) -> Result<(), FerricastError> {
   let m1 = tlv::encode(vec![
       (TLV_TYPE_METHOD, &[0x0]),
       (TLV_TYPE_STATE, &[0x01]),
   ])?; 

   let m2 = {
       let mut m2 = Vec::new();

       for retry in 0..3 {
            manager.builder()
                .path("/pair-setup".to_string())
                .content_type("application/octet-stream".to_string())
                .body(m1.clone())
                .post()
                .write(write_half)
                .await?;
    
            let res = RtspResponse::read(buf_reader)
              .await?;
        

           match res.content() {
                Ok(v) => {
                    if v.is_empty() { continue; }


                    m2 = v.clone();
                    break;
                }
                Err(_) => { continue },
           } 
        }

        if m2.is_empty() { 
            Err(FerricastError::Protocol("Airplay did not send a valid response with m2".to_string()))
        } else {
            Ok(m2)
        }
   }?;

   let m2 = tlv::decode(&m2)?;

   let salt = m2.get(&TLV_TYPE_SALT)
       .ok_or(FerricastError::Protocol("Invalid AirPlay Response no salt in TLV".to_string()))?;
   
    let server_pub = m2.get(&TLV_TYPE_PUBLIC_KEY)
        .ok_or(FerricastError::Protocol("Invalid airplay Response no public key in TLV".to_string()))?;

   Ok(())
}

