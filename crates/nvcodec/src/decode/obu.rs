//! A framing check of AV1 temporal units before they reach the parser.
//!
//! NVIDIA's AV1 parser trusts OBU sizes: a temporal unit whose last OBU
//! claims more bytes than the packet holds (a truncated packet) makes
//! `cuvidDecodePicture` fault inside the driver (SIGBUS, seen on driver 595)
//! instead of failing. So the decoder walks the OBU headers of each packet,
//! reads nothing else, and refuses a packet whose sizes do not add up.

use crate::error::{Error, Result};

/// Checks that every OBU header of `data` is well formed and its size fits.
pub(super) fn check_temporal_unit(data: &[u8]) -> Result<()> {
    let mut rest = data;
    while let Some((&header, tail)) = rest.split_first() {
        let invalid = |reason: &str| Error::InvalidPacket {
            reason: format!("AV1 OBU at byte {}: {reason}", data.len() - rest.len()),
        };
        if header & 0x80 != 0 {
            return Err(invalid("forbidden bit set"));
        }
        let extension = header & 0x04 != 0;
        let has_size = header & 0x02 != 0;
        let mut body = tail;
        if extension {
            body = body
                .get(1..)
                .ok_or_else(|| invalid("missing extension header"))?;
        }
        if !has_size {
            // Runs to the end of the temporal unit.
            return Ok(());
        }
        let (size, read) = leb128(body).ok_or_else(|| invalid("bad size field"))?;
        let payload = body.get(read..).unwrap_or_default();
        let size = usize::try_from(size).map_err(|_| invalid("size overflows"))?;
        if size > payload.len() {
            return Err(invalid(&format!(
                "size {size} runs past the packet ({} bytes left)",
                payload.len()
            )));
        }
        rest = payload.get(size..).unwrap_or_default();
    }
    Ok(())
}

/// `leb128()`: the value and the bytes it took (at most 8).
fn leb128(data: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0_u64;
    for (index, byte) in data.iter().take(8).enumerate() {
        value |= u64::from(byte & 0x7f) << (7 * index);
        if byte & 0x80 == 0 {
            return Some((value, index + 1));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_must_fit() {
        // Temporal delimiter (size 0), then a frame OBU of 3 bytes.
        let unit = [0x12, 0x00, 0x32, 0x03, 1, 2, 3];
        assert!(check_temporal_unit(&unit).is_ok());
        assert!(matches!(
            check_temporal_unit(&unit[..6]),
            Err(Error::InvalidPacket { .. })
        ));
        assert!(check_temporal_unit(&[0x92, 0x00]).is_err(), "forbidden bit");
        assert!(
            check_temporal_unit(&[0x32, 0x80]).is_err(),
            "unterminated size"
        );
        // Without a size field an OBU runs to the end.
        assert!(check_temporal_unit(&[0x30, 9, 9, 9]).is_ok());
        assert_eq!(leb128(&[0x80, 0x01]), Some((128, 2)));
    }
}
