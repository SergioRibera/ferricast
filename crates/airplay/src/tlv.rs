use std::io::Write;
use std::collections::HashMap;
use byteorder::WriteBytesExt;
use ferricast_core::FerricastError;


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


pub fn decode(bytes: &[u8]) -> HashMap<u8, Vec<u8>> {
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

    result
}

