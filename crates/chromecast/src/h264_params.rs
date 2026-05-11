//! H.264 parameter-set extraction from an Annex B bitstream.
//!
//! The HLS segmenter wants SPS + PPS (NAL types 7 and 8) prefixed
//! with start codes so it can inject them at every segment boundary.
//! Most encoders we wrap (VA-API, NVENC) emit them inline at every
//! IDR, so we recover them by scanning the first keyframe payload.

/// Walk an Annex B bitstream and return the concatenation of every
/// SPS (NAL type 7) and PPS (NAL type 8) NAL unit found, each
/// prefixed with the canonical 4-byte `00 00 00 01` start code.
///
/// Returns an empty `Vec` if the stream contains no parameter sets —
/// the caller should treat that as a configuration error.
pub fn extract_sps_pps(annex_b: &[u8]) -> Vec<u8> {
    let positions = find_start_codes(annex_b);
    if positions.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    for (i, &(start, sc_len)) in positions.iter().enumerate() {
        let nal_start = start + sc_len;
        if nal_start >= annex_b.len() {
            continue;
        }
        let nal_type = annex_b[nal_start] & 0x1f;
        if nal_type != 7 && nal_type != 8 {
            continue;
        }
        let nal_end = positions
            .get(i + 1)
            .map(|(s, _)| *s)
            .unwrap_or(annex_b.len());
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&annex_b[nal_start..nal_end]);
    }
    out
}

/// Returns `(start_code_offset, start_code_len)` for every Annex B
/// start code in the buffer. Both 3-byte (`00 00 01`) and 4-byte
/// (`00 00 00 01`) variants are recognised.
fn find_start_codes(buf: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 3 <= buf.len() {
        if buf[i] == 0 && buf[i + 1] == 0 {
            if buf[i + 2] == 1 {
                out.push((i, 3));
                i += 3;
                continue;
            }
            if i + 4 <= buf.len() && buf[i + 2] == 0 && buf[i + 3] == 1 {
                out.push((i, 4));
                i += 4;
                continue;
            }
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_sps_pps_from_idr() {
        // SPS (type 7), PPS (type 8), IDR slice (type 5) — all
        // prefixed with 4-byte start codes.
        let stream = [
            0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1e, // SPS
            0, 0, 0, 1, 0x68, 0xce, 0x3c, 0x80, // PPS
            0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00, // IDR
        ];
        let out = extract_sps_pps(&stream);
        let expected = [
            0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1e, 0, 0, 0, 1, 0x68, 0xce, 0x3c, 0x80,
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn empty_when_no_params() {
        let stream = [0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00];
        assert!(extract_sps_pps(&stream).is_empty());
    }
}
