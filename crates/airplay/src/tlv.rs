use byteorder::WriteBytesExt;
use ferricast_core::FerricastError;
use std::collections::HashMap;
use std::io::Write;


pub const TLV_TYPE_IDENTIFIER: u8 = 0x1;

pub const TLV_TYPE_STATE: u8 = 6;
pub const TLV_TYPE_METHOD: u8 = 0;
pub const TLV_TYPE_FLAGS: u8 = 0x13;

pub const TLV_TYPE_SALT: u8 = 0x2;
pub const TLV_TYPE_PUBLIC_KEY: u8 = 0x3;
pub const TLV_TYPE_PROOF: u8 = 0x04;

pub const TLV_TYPE_SIGNATURE: u8 = 0x0A;
pub const TLV_TYPE_ENCRYPTED_DATA: u8 = 0x05;

pub const TLV_TYPE_ERROR: u8 = 0x7;

pub fn encode(items: Vec<(u8, &[u8])>) -> Result<Vec<u8>, FerricastError> {
    let mut bytes = Vec::new();

    for (tag, mut value) in items {
        if value.is_empty() {
            bytes.write_u8(tag)?;
            bytes.write_u8(0)?;
            continue;
        }

        while !value.is_empty() {
            let chunk_len = std::cmp::min(value.len(), 255);
            let (chunk, rest) = value.split_at(chunk_len);

            bytes.write_u8(tag)?;
            bytes.write_u8(chunk_len as u8)?;
            bytes.write_all(chunk)?;

            value = rest;
        }
    }

    Ok(bytes)
}

pub fn decode(bytes: &[u8]) -> Result<HashMap<u8, Vec<u8>>, FerricastError> {
    let mut result: HashMap<u8, Vec<u8>> = HashMap::new();
    let mut offset = 0;

    while bytes.len() - offset >= 2 {
        let tag = bytes[offset];
        let data_len = bytes[offset + 1] as usize;

        if offset + 2 + data_len > bytes.len() {
            break;
        }

        let data = &bytes[offset + 2..offset + 2 + data_len];

        result.entry(tag).or_default().extend_from_slice(data);

        offset += 2 + data_len;
    }

    if let Some(err_code) = result.get(&TLV_TYPE_ERROR) {
        return Err(FerricastError::Tlv(format!(
            "Airplay device send an error, with code {:?}",
            err_code
        )));
    }

    Ok(result)
}
