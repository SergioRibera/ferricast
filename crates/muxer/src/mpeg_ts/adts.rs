//! ADTS header parsing and silent AAC frame synthesis for the audio PID.

/// Sampling frequencies indexed by the ADTS `sampling_frequency_index`
/// (ISO/IEC 14496-3 table 1.18).
const SAMPLE_RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

/// Samples per AAC frame (no SBR, 1024-sample frame length).
const SAMPLES_PER_FRAME: u64 = 1024;
const ADTS_HEADER_LEN: usize = 7;

/// Global gain written into every silent channel element; any value
/// decodes to silence when the spectrum is all zero.
const SILENT_GLOBAL_GAIN: u32 = 140;

/// Syntactic element ids (ISO/IEC 14496-3 table 4.85).
const ID_SCE: u32 = 0;
const ID_CPE: u32 = 1;
const ID_LFE: u32 = 3;
const ID_END: u32 = 7;

/// AAC stream parameters as signalled in an ADTS header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AdtsFormat {
    /// `profile_ObjectType` (audio object type minus one; 1 = AAC LC).
    profile: u8,
    sampling_index: u8,
    /// `channel_configuration`, 1..=7.
    channels: u8,
}

/// AAC LC, 48 kHz, stereo: what the muxer assumes before it has seen a
/// real audio frame (and the only format used in silent-only mode).
impl Default for AdtsFormat {
    fn default() -> Self {
        Self {
            profile: 1,
            sampling_index: 3,
            channels: 2,
        }
    }
}

impl AdtsFormat {
    /// Reads the format from the start of an ADTS frame. `None` when the
    /// bytes are not an ADTS header or use a layout this module cannot
    /// synthesise silence for (reserved rates, PCE-defined channels).
    pub(super) fn parse(frame: &[u8]) -> Option<Self> {
        let header = frame.get(..ADTS_HEADER_LEN)?;
        if header[0] != 0xFF || header[1] & 0xF6 != 0xF0 {
            return None;
        }
        let format = Self {
            profile: header[2] >> 6,
            sampling_index: (header[2] >> 2) & 0x0F,
            channels: (header[2] & 0x01) << 2 | header[3] >> 6,
        };
        let rate_known = usize::from(format.sampling_index) < SAMPLE_RATES.len();
        (rate_known && (1..=7).contains(&format.channels)).then_some(format)
    }

    fn sample_rate(self) -> u32 {
        SAMPLE_RATES[usize::from(self.sampling_index)]
    }

    /// Duration of one AAC frame in 90 kHz ticks (1920 at 48 kHz),
    /// rounded down for rates that do not divide evenly (44.1 kHz).
    pub(super) fn frame_ticks_90k(self) -> u64 {
        SAMPLES_PER_FRAME * 90_000 / u64::from(self.sample_rate())
    }

    /// One complete ADTS frame of digital silence in this format.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the frame length fields are masked to their bit widths"
    )]
    pub(super) fn silent_frame(self) -> Vec<u8> {
        let raw = silent_raw_data_block(self.channels);
        let len = ADTS_HEADER_LEN + raw.len();
        let mut frame = Vec::with_capacity(len);
        frame.extend_from_slice(&[
            0xFF,
            0xF1, // MPEG-4, layer 0, protection_absent = 1 (no CRC)
            self.profile << 6 | self.sampling_index << 2 | (self.channels >> 2) & 0x01,
            (self.channels & 0x03) << 6 | ((len >> 11) & 0x03) as u8,
            ((len >> 3) & 0xFF) as u8,
            ((len & 0x07) as u8) << 5 | 0x1F, // + buffer fullness 0x7FF (VBR)...
            0xFC,                             // ...and one raw_data_block
        ]);
        frame.extend_from_slice(&raw);
        frame
    }
}

/// A `raw_data_block` holding one silent element per channel of the
/// given `channel_configuration` (ISO/IEC 14496-3 table 1.19), then END.
///
/// Every element has `max_sfb = 0`, so it carries no section, scale
/// factor or spectral data: the decoder reconstructs all-zero spectra.
fn silent_raw_data_block(channels: u8) -> Vec<u8> {
    let layout: &[u32] = match channels {
        1 => &[ID_SCE],
        2 => &[ID_CPE],
        3 => &[ID_SCE, ID_CPE],
        4 => &[ID_SCE, ID_CPE, ID_SCE],
        5 => &[ID_SCE, ID_CPE, ID_CPE],
        6 => &[ID_SCE, ID_CPE, ID_CPE, ID_LFE],
        _ => &[ID_SCE, ID_CPE, ID_CPE, ID_CPE, ID_LFE],
    };
    let mut bits = BitWriter::default();
    let mut instance_tags = [0u32; 8];
    for &id in layout {
        let tag = &mut instance_tags[id as usize];
        bits.put(id, 3);
        bits.put(*tag, 4);
        *tag += 1;
        if id == ID_CPE {
            bits.put(1, 1); // common_window
            put_ics_info(&mut bits);
            bits.put(0, 2); // ms_mask_present
            put_silent_ics(&mut bits, false);
            put_silent_ics(&mut bits, false);
        } else {
            put_silent_ics(&mut bits, true);
        }
    }
    bits.put(ID_END, 3);
    bits.finish()
}

/// `ics_info` for a long window with no scale factor bands.
fn put_ics_info(bits: &mut BitWriter) {
    bits.put(0, 1); // ics_reserved_bit
    bits.put(0, 2); // window_sequence = ONLY_LONG_SEQUENCE
    bits.put(1, 1); // window_shape
    bits.put(0, 6); // max_sfb
    bits.put(0, 1); // predictor_data_present
}

/// `individual_channel_stream` with no spectral content; `with_ics_info`
/// is false inside a CPE that shares its window through `common_window`.
fn put_silent_ics(bits: &mut BitWriter, with_ics_info: bool) {
    bits.put(SILENT_GLOBAL_GAIN, 8);
    if with_ics_info {
        put_ics_info(bits);
    }
    bits.put(0, 1); // pulse_data_present
    bits.put(0, 1); // tns_data_present
    bits.put(0, 1); // gain_control_data_present
}

/// MSB-first bit packer.
#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    used: u32,
}

impl BitWriter {
    fn put(&mut self, value: u32, width: u32) {
        for shift in (0..width).rev() {
            if self.used.is_multiple_of(8) {
                self.bytes.push(0);
            }
            let bit = ((value >> shift) & 1) as u8;
            if let Some(last) = self.bytes.last_mut() {
                *last |= bit << (7 - self.used % 8);
            }
            self.used += 1;
        }
    }

    /// The packed bytes, zero-padded to a byte boundary.
    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hand-crafted 48 kHz stereo frame the muxer shipped with
    /// before silence was derived from the real stream's format.
    const LEGACY_STEREO_48K: [u8; 13] = [
        0xFF, 0xF1, 0x4C, 0x80, 0x01, 0xBF, 0xFC, 0x21, 0x10, 0x04, 0x60, 0x8C, 0x1C,
    ];

    #[test]
    fn default_format_reproduces_the_legacy_silent_frame() {
        assert_eq!(AdtsFormat::default().silent_frame(), LEGACY_STEREO_48K);
        assert_eq!(AdtsFormat::default().frame_ticks_90k(), 1920);
    }

    #[test]
    fn the_legacy_frame_parses_as_the_default_format() {
        assert_eq!(
            AdtsFormat::parse(&LEGACY_STEREO_48K),
            Some(AdtsFormat::default())
        );
    }

    #[test]
    fn mono_silence_is_a_single_channel_element() {
        let format = AdtsFormat {
            profile: 1,
            sampling_index: 4,
            channels: 1,
        };
        let frame = format.silent_frame();
        // SCE tag 0 | gain 140 | ics_info | 3 flags | END: 32 bits.
        assert_eq!(&frame[7..], [0x01, 0x18, 0x20, 0x07]);
        assert_eq!(usize::from(frame[4]) << 3 | usize::from(frame[5] >> 5), 11);
        assert_eq!(AdtsFormat::parse(&frame), Some(format));
        assert_eq!(format.frame_ticks_90k(), 2089);
    }

    #[test]
    fn surround_layouts_round_trip_through_their_header() {
        for channels in 1..=7 {
            let format = AdtsFormat {
                profile: 1,
                sampling_index: 3,
                channels,
            };
            let frame = format.silent_frame();
            let len = usize::from(frame[3] & 0x03) << 11
                | usize::from(frame[4]) << 3
                | usize::from(frame[5] >> 5);
            assert_eq!(len, frame.len(), "channels={channels}");
            assert_eq!(AdtsFormat::parse(&frame), Some(format));
        }
    }

    #[test]
    fn five_one_silence_lists_its_elements_in_order() {
        let raw = silent_raw_data_block(6);
        // First element: SCE (000) tag 0 (0000), then the gain's top bit.
        assert_eq!(raw[0] >> 1, 0b000_0000);
        // SCE (29 bits) + 2 × CPE (43) + LFE (29) + END (3) = 147 bits.
        assert_eq!(raw.len(), 147_usize.div_ceil(8));
        // The LFE element (id 011, tag 0) starts at bit 115 = byte 14, bit 3.
        let window = u32::from(raw[14]) << 8 | u32::from(raw[15]);
        assert_eq!((window >> 10) & 0b111, ID_LFE);
        assert_eq!((window >> 6) & 0b1111, 0);
    }

    #[test]
    fn non_adts_or_pce_defined_layouts_are_not_parsed() {
        assert_eq!(AdtsFormat::parse(&[0x00; 7]), None);
        assert_eq!(AdtsFormat::parse(&LEGACY_STEREO_48K[..5]), None);
        let mut pce = LEGACY_STEREO_48K;
        pce[3] &= 0x3F; // channel_configuration = 0
        assert_eq!(AdtsFormat::parse(&pce), None);
    }
}
