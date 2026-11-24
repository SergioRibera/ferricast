//! What NVDEC's H.264 pictures need from the bitstream to be timed.

use std::io::Cursor;

use h264_reader::nal::sps::SeqParameterSet;
use h264_reader::rbsp::{BitReader, decode_nal};

use super::timestamps::Picture;
use ferricast_codec::bitstream::split_annex_b;

const NAL_SLICE: u8 = 1;
const NAL_SLICE_DATA_PARTITION_A: u8 = 2;
const NAL_IDR_SLICE: u8 = 5;
const NAL_SPS: u8 = 7;

/// Sequence state read from an H.264 bitstream.
///
/// Every coded picture is shown (H.264 has no `pic_output_flag`), so the
/// only pictures left out are those before the first IDR picture, which
/// NVDEC cannot decode for lack of references. An MMCO 5 picture also
/// empties the DPB but is not read: it keeps display order and
/// timestamp order in step, so treating it as part of its sequence only
/// delays when pictures dropped before it are noticed.
#[derive(Debug, Default)]
pub struct Headers {
    /// IDR pictures seen; 0 until decoding can start.
    sequence: u32,
    /// `max_dec_frame_buffering` of the latest SPS, when its VUI says.
    max_dec_frame_buffering: Option<u32>,
}

impl Headers {
    /// Notes the SPS in one packet and returns the picture it carries,
    /// if NVDEC is expected to show one. A packet carries one access
    /// unit: its slices all belong to the same picture.
    pub(super) fn picture(&mut self, packet: &[u8]) -> Option<Picture> {
        let mut slices = false;
        let mut idr = false;
        for nal in split_annex_b(packet) {
            match nal.first().map(|header| header & 0x1f) {
                Some(NAL_SPS) => self.max_dec_frame_buffering = max_dec_frame_buffering(nal),
                Some(NAL_IDR_SLICE) => idr = true,
                Some(NAL_SLICE | NAL_SLICE_DATA_PARTITION_A) => slices = true,
                _ => {}
            }
        }
        if idr {
            self.sequence += 1;
        }
        ((slices || idr) && self.sequence > 0).then_some(Picture {
            sequence: self.sequence,
            max_waiting: self.max_dec_frame_buffering,
        })
    }
}

/// The VUI's `max_dec_frame_buffering`, the most frames the DPB keeps.
fn max_dec_frame_buffering(nal: &[u8]) -> Option<u32> {
    let rbsp = decode_nal(nal).ok()?;
    SeqParameterSet::from_bits(BitReader::new(Cursor::new(rbsp.as_ref())))
        .ok()?
        .vui_parameters?
        .bitstream_restrictions
        .map(|restrictions| restrictions.max_dec_frame_buffering)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NON_IDR: [u8; 5] = [0, 0, 1, 0x41, 0x9a];
    const IDR: [u8; 5] = [0, 0, 1, 0x65, 0x88];

    fn pictures(packets: &[&[u8]]) -> Vec<Option<u32>> {
        let mut headers = Headers::default();
        packets
            .iter()
            .map(|packet| headers.picture(packet).map(|picture| picture.sequence))
            .collect()
    }

    #[test]
    fn pictures_before_the_first_idr_are_left_out() {
        assert_eq!(
            pictures(&[&NON_IDR, &NON_IDR, &IDR, &NON_IDR]),
            [None, None, Some(1), Some(1)]
        );
    }

    #[test]
    fn every_idr_starts_a_sequence_once_however_many_slices_it_has() {
        let two_slices = [IDR, IDR].concat();
        assert_eq!(
            pictures(&[&IDR, &NON_IDR, &two_slices, &NON_IDR]),
            [Some(1), Some(1), Some(2), Some(2)]
        );
    }

    #[test]
    fn a_packet_without_slices_carries_no_picture() {
        let parameter_sets = [0, 0, 1, 0x68, 0xce];
        assert_eq!(pictures(&[&IDR, &parameter_sets]), [Some(1), None]);
    }

    #[test]
    fn the_fixture_opens_with_an_idr() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../demuxer/tests/fixtures/h264.mp4");
        let mut demuxer = ferricast_demuxer::Demuxer::open(&path).unwrap();
        let packet = demuxer.next_frame().unwrap().unwrap();
        let mut headers = Headers::default();
        let picture = headers.picture(&packet.data);
        assert_eq!(picture.map(|picture| picture.sequence), Some(1));
    }
}
