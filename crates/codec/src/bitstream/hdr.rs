//! Static HDR metadata in elementary streams: H.264 / H.265 SEI messages
//! 137 (mastering display colour volume) and 144 (content light level),
//! and AV1 metadata OBUs (`METADATA_TYPE_HDR_MDCV` / `HDR_CLL`).
//!
//! Readers never fail: malformed messages are skipped, so a damaged
//! stream only loses its metadata. [`HdrTracker`] keeps the latest
//! values seen in a stream, since encoders send them with keyframes.

use ferricast_core::{ContentLightLevel, HdrMetadata, MasteringDisplay};

use super::{NalSyntax, START_CODE, split_annex_b};

/// SEI `payloadType` of `mastering_display_colour_volume`.
const SEI_MASTERING_DISPLAY: u32 = 137;
/// SEI `payloadType` of `content_light_level_info`.
const SEI_CONTENT_LIGHT: u32 = 144;

/// H.264 `nal_unit_type` of an SEI NAL unit.
const H264_SEI: u8 = 6;
/// H.265 `nal_unit_type` of a prefix SEI NAL unit.
const H265_PREFIX_SEI: u8 = 39;

/// AV1 `obu_type` of a metadata OBU.
const OBU_METADATA: u8 = 5;
/// AV1 `metadata_type` codes.
const METADATA_HDR_CLL: u64 = 1;
const METADATA_HDR_MDCV: u64 = 2;

/// `nal` (header included) with emulation-prevention bytes removed.
#[must_use]
pub fn unescape(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len());
    let mut zeros = 0;
    for &byte in nal {
        if zeros >= 2 && byte == 3 {
            zeros = 0;
            continue;
        }
        zeros = if byte == 0 { zeros + 1 } else { 0 };
        out.push(byte);
    }
    out
}

/// `rbsp` with emulation-prevention bytes inserted, so no `00 00 0x`
/// (x ≤ 3) sequence appears in the NAL unit.
#[must_use]
pub fn escape(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + rbsp.len() / 64 + 1);
    let mut zeros = 0;
    for &byte in rbsp {
        if zeros >= 2 && byte <= 3 {
            out.push(3);
            zeros = 0;
        }
        zeros = if byte == 0 { zeros + 1 } else { 0 };
        out.push(byte);
    }
    out
}

/// Merges the halves `update` carries into `into`.
fn merge(into: &mut HdrMetadata, update: HdrMetadata) {
    if update.mastering_display.is_some() {
        into.mastering_display = update.mastering_display;
    }
    if update.content_light.is_some() {
        into.content_light = update.content_light;
    }
}

/// The HDR metadata in one SEI NAL unit (`nal` starts at the NAL
/// header, emulation prevention still in place). Empty when `nal` is
/// not an SEI unit of `syntax` or carries no HDR message.
#[must_use]
pub fn from_sei_nal(nal: &[u8], syntax: NalSyntax) -> HdrMetadata {
    let mut found = HdrMetadata::default();
    let Some(&header) = nal.first() else {
        return found;
    };
    let header_len = match syntax {
        NalSyntax::H264 if header & 0x1f == H264_SEI => 1,
        NalSyntax::H265 if (header >> 1) & 0x3f == H265_PREFIX_SEI => 2,
        NalSyntax::H264 | NalSyntax::H265 => return found,
    };
    let rbsp = unescape(nal);
    let mut rest = rbsp.get(header_len..).unwrap_or_default();
    // sei_message() until only rbsp_trailing_bits (0x80) is left.
    while rest.len() > 1 || rest.first().is_some_and(|&byte| byte != 0x80) {
        let Some((payload_type, tail)) = sei_varint(rest) else {
            break;
        };
        let Some((size, tail)) = sei_varint(tail) else {
            break;
        };
        let Some((payload, tail)) = usize::try_from(size)
            .ok()
            .and_then(|size| tail.split_at_checked(size))
        else {
            break;
        };
        match payload_type {
            SEI_MASTERING_DISPLAY => {
                if let Some(display) = MasteringDisplay::from_sei(payload) {
                    found.mastering_display = Some(display);
                }
            }
            SEI_CONTENT_LIGHT => {
                if let Some(light) = ContentLightLevel::from_sei(payload) {
                    found.content_light = Some(light);
                }
            }
            _ => {}
        }
        rest = tail;
    }
    found
}

/// An SEI `payloadType` / `payloadSize`: a run of `0xFF` bytes, each
/// adding 255, then the last byte.
fn sei_varint(bytes: &[u8]) -> Option<(u32, &[u8])> {
    let mut value: u32 = 0;
    for (index, &byte) in bytes.iter().enumerate() {
        value = value.checked_add(u32::from(byte))?;
        if byte != 0xFF {
            return Some((value, &bytes[index + 1..]));
        }
    }
    None
}

/// The HDR metadata in the SEI units of an Annex B access unit.
#[must_use]
pub fn from_annex_b(access_unit: &[u8], syntax: NalSyntax) -> HdrMetadata {
    let mut found = HdrMetadata::default();
    for nal in split_annex_b(access_unit) {
        merge(&mut found, from_sei_nal(nal, syntax));
    }
    found
}

/// The HDR metadata in the metadata OBUs of an AV1 temporal unit
/// (low-overhead bitstream format, as Matroska / MP4 / RTP carry it).
#[must_use]
pub fn from_av1_obus(temporal_unit: &[u8]) -> HdrMetadata {
    let mut found = HdrMetadata::default();
    let mut rest = temporal_unit;
    while let Some((&header, tail)) = rest.split_first() {
        let obu_type = (header >> 3) & 0x0f;
        let has_extension = header & 0b100 != 0;
        let has_size = header & 0b10 != 0;
        let Some(tail) = tail.get(usize::from(has_extension)..) else {
            break;
        };
        let (payload, next) = if has_size {
            let Some((size, tail)) = leb128(tail) else {
                break;
            };
            let Some(split) = usize::try_from(size)
                .ok()
                .and_then(|size| tail.split_at_checked(size))
            else {
                break;
            };
            split
        } else {
            (tail, &[][..])
        };
        if obu_type == OBU_METADATA {
            merge(&mut found, av1_metadata(payload));
        }
        rest = next;
    }
    found
}

/// One AV1 `metadata_obu()` payload.
fn av1_metadata(payload: &[u8]) -> HdrMetadata {
    let mut found = HdrMetadata::default();
    let Some((metadata_type, body)) = leb128(payload) else {
        return found;
    };
    let u16_at = |at: usize| {
        body.get(at..at + 2)
            .map(|bytes| u16::from_be_bytes([bytes[0], bytes[1]]))
    };
    let u32_at = |at: usize| {
        body.get(at..at + 4)
            .map(|bytes| u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    };
    match metadata_type {
        METADATA_HDR_CLL => {
            if let (Some(max_cll), Some(max_fall)) = (u16_at(0), u16_at(2)) {
                found.content_light = Some(ContentLightLevel { max_cll, max_fall });
            }
        }
        METADATA_HDR_MDCV => {
            let point = |at: usize| Some((u16_at(at)?, u16_at(at + 2)?));
            if let (Some(r), Some(g), Some(b), Some(white), Some(max), Some(min)) = (
                point(0),
                point(4),
                point(8),
                point(12),
                u32_at(16),
                u32_at(20),
            ) {
                found.mastering_display =
                    Some(MasteringDisplay::from_av1([r, g, b], white, max, min));
            }
        }
        _ => {}
    }
    found
}

/// An unsigned LEB128 value (AV1 `leb128()`, at most 8 bytes).
fn leb128(bytes: &[u8]) -> Option<(u64, &[u8])> {
    let mut value = 0_u64;
    for (index, &byte) in bytes.iter().enumerate().take(8) {
        value |= u64::from(byte & 0x7f) << (index * 7);
        if byte & 0x80 == 0 {
            return Some((value, &bytes[index + 1..]));
        }
    }
    None
}

/// An Annex B SEI NAL unit (start code included) carrying `hdr` as
/// SEI messages 137 / 144, for encoders whose backend cannot signal
/// HDR itself: prefix SEI for H.265, SEI for H.264. `None` when `hdr`
/// is empty.
#[must_use]
pub fn sei_nal(hdr: HdrMetadata, syntax: NalSyntax) -> Option<Vec<u8>> {
    if hdr.is_empty() {
        return None;
    }
    let mut rbsp = match syntax {
        NalSyntax::H264 => vec![H264_SEI],
        // nal_unit_type 39, nuh_layer_id 0, nuh_temporal_id_plus1 1.
        NalSyntax::H265 => vec![H265_PREFIX_SEI << 1, 1],
    };
    let mut message = |payload_type: u32, payload: &[u8]| {
        for value in [payload_type, u32::try_from(payload.len()).unwrap_or(0)] {
            rbsp.extend(std::iter::repeat_n(0xFF, (value / 255) as usize));
            rbsp.push(u8::try_from(value % 255).unwrap_or(0));
        }
        rbsp.extend_from_slice(payload);
    };
    if let Some(display) = hdr.mastering_display {
        message(SEI_MASTERING_DISPLAY, &display.to_sei());
    }
    if let Some(light) = hdr.content_light {
        message(SEI_CONTENT_LIGHT, &light.to_sei());
    }
    rbsp.push(0x80);
    let mut out = START_CODE.to_vec();
    out.extend(escape(&rbsp));
    Some(out)
}

/// The latest HDR metadata of a stream. Feed it every access unit;
/// a half once seen stays until the stream sends a new one.
#[derive(Debug, Clone, Default)]
pub struct HdrTracker {
    current: HdrMetadata,
}

impl HdrTracker {
    /// Records the SEI messages of an H.264 / H.265 Annex B access unit.
    pub fn observe_annex_b(&mut self, access_unit: &[u8], syntax: NalSyntax) {
        merge(&mut self.current, from_annex_b(access_unit, syntax));
    }

    /// Records the metadata OBUs of an AV1 temporal unit.
    pub fn observe_av1(&mut self, temporal_unit: &[u8]) {
        merge(&mut self.current, from_av1_obus(temporal_unit));
    }

    /// What the stream signalled so far, else `fallback` (the
    /// container's metadata).
    #[must_use]
    pub fn current_or(&self, fallback: Option<HdrMetadata>) -> Option<HdrMetadata> {
        if self.current.is_empty() {
            fallback
        } else {
            Some(self.current)
        }
    }
}

#[cfg(test)]
mod tests {
    use ferricast_core::Chromaticity;

    use super::*;

    const DISPLAY: MasteringDisplay = MasteringDisplay {
        red: Chromaticity {
            x: 35_400,
            y: 14_600,
        },
        green: Chromaticity {
            x: 8_500,
            y: 39_850,
        },
        blue: Chromaticity { x: 6_550, y: 2_300 },
        white_point: Chromaticity {
            x: 15_635,
            y: 16_450,
        },
        max_luminance: 10_000_000,
        min_luminance: 50,
    };
    const LIGHT: ContentLightLevel = ContentLightLevel {
        max_cll: 1000,
        max_fall: 400,
    };
    const HDR: HdrMetadata = HdrMetadata {
        mastering_display: Some(DISPLAY),
        content_light: Some(LIGHT),
    };

    #[test]
    fn written_sei_reads_back_for_both_syntaxes() {
        for syntax in [NalSyntax::H264, NalSyntax::H265] {
            let nal = sei_nal(HDR, syntax).unwrap();
            assert_eq!(&nal[..4], &START_CODE);
            assert_eq!(from_annex_b(&nal, syntax), HDR, "{syntax:?}");
        }
        let light_only = HdrMetadata {
            mastering_display: None,
            content_light: Some(LIGHT),
        };
        let nal = sei_nal(light_only, NalSyntax::H265).unwrap();
        assert_eq!(from_annex_b(&nal, NalSyntax::H265), light_only);
        assert!(sei_nal(HdrMetadata::default(), NalSyntax::H265).is_none());
    }

    #[test]
    fn escaping_removes_start_code_emulation_and_round_trips() {
        let rbsp = [0x4E, 1, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0x80];
        let escaped = escape(&rbsp);
        assert_eq!(escaped, [0x4E, 1, 0, 0, 3, 0, 0, 3, 0, 1, 0, 0, 3, 3, 0x80]);
        assert_eq!(unescape(&escaped), rbsp);
    }

    #[test]
    fn hand_written_hevc_sei_with_an_unrelated_message_first() {
        // Prefix SEI: a 2-byte user_data payload (type 5), then CLL.
        let nal = [
            0x4E, 0x01, 5, 2, 0xAA, 0xBB, 144, 4, 0x03, 0xE8, 0x01, 0x90, 0x80,
        ];
        let found = from_sei_nal(&nal, NalSyntax::H265);
        assert_eq!(found.content_light, Some(LIGHT));
        assert_eq!(found.mastering_display, None);
        // The same bytes as an H.264 NAL are not an SEI unit there.
        assert!(from_sei_nal(&nal, NalSyntax::H264).is_empty());
    }

    #[test]
    fn av1_metadata_obus_are_read_with_and_without_size_fields() {
        let mut mdcv = vec![u8::try_from(METADATA_HDR_MDCV).unwrap()];
        for (x, y) in [
            (46_399_u16, 19_136_u16),
            (11_141, 52_233),
            (8_585, 3_015),
            (20_493, 21_561),
        ] {
            mdcv.extend(x.to_be_bytes());
            mdcv.extend(y.to_be_bytes());
        }
        mdcv.extend((1000_u32 << 8).to_be_bytes());
        mdcv.extend(82_u32.to_be_bytes());
        let cll = [
            u8::try_from(METADATA_HDR_CLL).unwrap(),
            0x03,
            0xE8,
            0x01,
            0x90,
        ];
        let mut unit = vec![0x12, 0]; // temporal delimiter, size 0
        unit.push((OBU_METADATA << 3) | 0b10);
        unit.push(u8::try_from(mdcv.len()).unwrap());
        unit.extend(&mdcv);
        unit.push(OBU_METADATA << 3); // no size field: runs to the end
        unit.extend(cll);
        let found = from_av1_obus(&unit);
        assert_eq!(found.content_light, Some(LIGHT));
        let display = found.mastering_display.unwrap();
        assert_eq!(display.red, DISPLAY.red);
        assert_eq!(display.max_luminance, DISPLAY.max_luminance);
    }

    #[test]
    fn the_tracker_keeps_halves_and_prefers_the_stream() {
        let mut tracker = HdrTracker::default();
        let container = HdrMetadata {
            mastering_display: None,
            content_light: Some(ContentLightLevel {
                max_cll: 1,
                max_fall: 1,
            }),
        };
        assert_eq!(tracker.current_or(Some(container)), Some(container));
        tracker.observe_annex_b(&sei_nal(HDR, NalSyntax::H265).unwrap(), NalSyntax::H265);
        tracker.observe_annex_b(&[0, 0, 1, 0x26, 1, 2], NalSyntax::H265);
        assert_eq!(tracker.current_or(Some(container)), Some(HDR));
    }

    #[test]
    fn garbage_never_panics() {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state.to_le_bytes()[0]
        };
        for len in 0..64 {
            for _ in 0..200 {
                let mut bytes: Vec<u8> = (0..len).map(|_| next()).collect();
                if let Some(first) = bytes.first_mut() {
                    // Bias half the inputs towards SEI / metadata headers.
                    if len % 2 == 0 {
                        *first = 0x4E;
                    }
                }
                let _ = from_sei_nal(&bytes, NalSyntax::H265);
                let _ = from_sei_nal(&bytes, NalSyntax::H264);
                let _ = from_annex_b(&bytes, NalSyntax::H264);
                let _ = from_av1_obus(&bytes);
                assert_eq!(unescape(&escape(&bytes)), bytes);
            }
        }
    }
}
