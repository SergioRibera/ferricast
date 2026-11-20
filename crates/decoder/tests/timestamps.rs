//! Every decoder stamps a picture with the timestamp of the packet it was
//! decoded from: reordered pictures come out in presentation order, and
//! packets a decoder drops or rejects never shift the pictures after them.
#![cfg(feature = "hevc-decode")]

use std::path::PathBuf;

use ferricast_core::{CapturedFrame, DecoderConfig, EncodedFrame, PixelFormat, VideoDecoder};
use ferricast_demuxer::Demuxer;

const WIDTH: u32 = 128;
const HEIGHT: u32 = 72;

fn packets(name: &str) -> Vec<EncodedFrame> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../demuxer/tests/fixtures")
        .join(name);
    let mut demuxer = Demuxer::open(&path).unwrap();
    std::iter::from_fn(|| demuxer.next_frame().unwrap()).collect()
}

fn configured<D: VideoDecoder + Default>(low_latency: bool) -> D {
    let mut decoder = D::default();
    decoder
        .configure(&DecoderConfig {
            codec: D::CODEC,
            width: WIDTH,
            height: HEIGHT,
            pixel_format: PixelFormat::Nv12,
            low_latency,
        })
        .unwrap();
    decoder
}

/// Timestamps of every picture decoded from `packets`, flush included.
fn decode_all<D: VideoDecoder>(decoder: &mut D, packets: &[EncodedFrame]) -> Vec<u64> {
    let mut stamps: Vec<u64> = packets
        .iter()
        .filter_map(|packet| decoder.decode(packet.clone()).unwrap())
        .map(|frame| frame.timestamp_us())
        .collect();
    stamps.extend(
        decoder
            .flush()
            .unwrap()
            .iter()
            .map(CapturedFrame::timestamp_us),
    );
    stamps
}

fn presentation_order(packets: &[EncodedFrame]) -> Vec<u64> {
    let mut stamps: Vec<u64> = packets.iter().map(|packet| packet.timestamp_us).collect();
    stamps.sort_unstable();
    stamps
}

/// The packet's leading parameter sets, up to its first slice.
fn parameter_sets(packet: &EncodedFrame, is_slice: impl Fn(u8) -> bool) -> EncodedFrame {
    let slice = packet
        .data
        .windows(4)
        .position(|window| window[..3] == [0, 0, 1] && is_slice(window[3]))
        .unwrap();
    EncodedFrame {
        data: packet.data.slice(..slice),
        ..packet.clone()
    }
}

#[cfg(feature = "hevc-decode")]
mod hevc {
    use super::*;
    use ferricast_decoder::h265::SoftwareH265Decoder;

    /// `hevc.mp4` opens with an IDR GOP; packet 10 is a CRA whose two RASL
    /// pictures are packets 11 and 12.
    const CRA: usize = 10;
    const RASL: [usize; 2] = [11, 12];

    #[test]
    fn b_frames_come_out_in_presentation_order() {
        let packets = packets("hevc.mp4");
        let mut decoder: SoftwareH265Decoder = configured(false);
        assert_eq!(
            decode_all(&mut decoder, &packets),
            presentation_order(&packets)
        );
    }

    #[test]
    fn skipped_leading_pictures_do_not_shift_the_pictures_after_them() {
        let packets = packets("hevc.mp4");
        // Starting at the CRA, the decoder skips its RASL pictures.
        let mut decoder: SoftwareH265Decoder = configured(false);
        let shown: Vec<EncodedFrame> = (CRA..packets.len())
            .filter(|index| !RASL.contains(index))
            .map(|index| packets[index].clone())
            .collect();
        assert_eq!(
            decode_all(&mut decoder, &packets[CRA..]),
            presentation_order(&shown)
        );
    }

    #[test]
    fn joining_mid_gop_keeps_every_picture_on_its_own_time() {
        let packets = packets("hevc.mp4");
        let mut decoder: SoftwareH265Decoder = configured(false);
        let irap = |header: u8| (16..=21).contains(&(header >> 1 & 0x3f));
        let mut joined = vec![parameter_sets(&packets[0], irap)];
        joined.extend_from_slice(&packets[4..]);
        assert_eq!(
            decode_all(&mut decoder, &joined),
            presentation_order(&packets[4..])
        );
    }
}
