//! NVDEC decodes H.264 / HEVC packets that are not one access unit each
//! (an MPEG-TS PES may hold several, or part of one) bit-exactly: the same
//! pictures, in the same order, as from one access unit per packet.
#![cfg(feature = "nvdec-decode")]

use std::path::PathBuf;

use ferricast_core::{
    BackendSelection, CapturedFrame, CodecError, DecoderConfig, EncodedFrame, FrameMemory,
    PixelFormat, VideoDecoder,
};
use ferricast_decoder::nvdec::{NvdecCodec, NvdecDecoder};
use ferricast_demuxer::Demuxer;

fn packets(name: &str) -> Vec<EncodedFrame> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../demuxer/tests/fixtures")
        .join(name);
    let mut demuxer = Demuxer::open(&path).unwrap();
    std::iter::from_fn(|| demuxer.next_frame().unwrap()).collect()
}

fn fnv(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The hash of every picture's planes, in output order, or `None`
/// (printed) when this machine has no NVDEC for `C`.
fn pictures<C: NvdecCodec>(name: &str, packets: &[EncodedFrame]) -> Option<Vec<u64>> {
    let mut decoder = NvdecDecoder::<C>::default();
    let config = DecoderConfig {
        gpu: ferricast_core::GpuPreference::Auto,
        hdr: None,
        memory: FrameMemory::Cpu,
        codec: C::FERRICAST_CODEC,
        width: 128,
        height: 72,
        pixel_format: PixelFormat::Nv12,
        low_latency: false,
        colour: None,
        backend: BackendSelection::default(),
    };
    match decoder.configure(&config) {
        Ok(()) => {}
        Err(error @ CodecError::BackendUnavailable { .. }) => {
            eprintln!("{name}: skipped, {error}");
            return None;
        }
        Err(error) => panic!("{name}: {error}"),
    }
    let hash = |frame: CapturedFrame| match frame {
        CapturedFrame::Cpu(raw) => fnv(&raw.data),
        other => panic!("{name}: a host decoder returned {other:?}"),
    };
    let mut out = Vec::new();
    for packet in packets {
        match decoder.decode(packet.clone()) {
            Ok(frame) => out.extend(frame.map(hash)),
            // NVDEC has minimum sizes per codec.
            Err(error @ CodecError::UnsupportedStream { .. }) => {
                eprintln!("{name}: skipped, {error}");
                return None;
            }
            Err(error) => panic!("{name}: {error}"),
        }
    }
    out.extend(decoder.flush().unwrap().into_iter().map(hash));
    Some(out)
}

/// Every `count` packets joined into one, timed as the first.
fn joined(packets: &[EncodedFrame], count: usize) -> Vec<EncodedFrame> {
    packets
        .chunks(count)
        .map(|group| EncodedFrame {
            data: group
                .iter()
                .flat_map(|packet| packet.data.iter().copied())
                .collect::<Vec<u8>>()
                .into(),
            ..group[0].clone()
        })
        .collect()
}

/// Every packet cut in two at its middle byte (inside a slice).
fn halves(packets: &[EncodedFrame]) -> Vec<EncodedFrame> {
    packets
        .iter()
        .flat_map(|packet| {
            let middle = packet.data.len() / 2;
            [
                EncodedFrame {
                    data: packet.data.slice(..middle),
                    ..packet.clone()
                },
                EncodedFrame {
                    data: packet.data.slice(middle..),
                    is_keyframe: false,
                    ..packet.clone()
                },
            ]
        })
        .collect()
}

/// Whole access units up to the last keyframe, so the decoder trusts
/// packets to end pictures, then the rest of the stream cut in halves.
/// `None` for a stream with one keyframe.
fn whole_then_halves(packets: &[EncodedFrame]) -> Option<Vec<EncodedFrame>> {
    let keyframe = packets.iter().rposition(|packet| packet.is_keyframe)?;
    if keyframe == 0 {
        return None;
    }
    let (whole, rest) = packets.split_at(keyframe);
    let mut out = whole.to_vec();
    out.extend(halves(rest));
    Some(out)
}

fn check<C: NvdecCodec>(name: &str, packets: &[EncodedFrame]) {
    let Some(reference) = pictures::<C>(name, packets) else {
        return;
    };
    assert_eq!(
        reference.len(),
        packets
            .iter()
            .filter(|packet| !packet.data.is_empty())
            .count(),
        "{name}: one picture per packet"
    );
    let mut cases = vec![
        ("two access units per packet", joined(packets, 2)),
        ("three access units per packet", joined(packets, 3)),
        ("each access unit in two halves", halves(packets)),
    ];
    if let Some(packets) = whole_then_halves(packets) {
        cases.push((
            "whole access units, then halves from a keyframe on",
            packets,
        ));
    }
    for (case, repacketised) in cases {
        let decoded = pictures::<C>(name, &repacketised).unwrap();
        assert_eq!(decoded, reference, "{name}, {case}");
    }
}

#[test]
fn h264_packets_of_several_or_part_of_an_access_unit_decode_bit_exact() {
    check::<ferricast_decoder::nvdec::H264>("h264.mp4", &packets("h264.mp4"));
}

#[cfg(feature = "nvdec-hevc-decode")]
#[test]
fn hevc_packets_of_several_or_part_of_an_access_unit_decode_bit_exact() {
    // 256x144: NVDEC decodes no HEVC below 144x144, which `hevc.mp4` is.
    let delimited = annex_b_packets("hevc_main8.hevc");
    check::<ferricast_decoder::nvdec::Hevc>("hevc_main8.hevc", &delimited);
    let bare: Vec<EncodedFrame> = delimited
        .iter()
        .map(|packet| EncodedFrame {
            data: without_hevc_delimiters(&packet.data).into(),
            ..packet.clone()
        })
        .collect();
    check::<ferricast_decoder::nvdec::Hevc>("hevc_main8.hevc without delimiters", &bare);
}

/// Offsets of the `00 00 01` start codes in `data`.
fn start_codes(data: &[u8]) -> Vec<usize> {
    (0..data.len().saturating_sub(2))
        .filter(|&at| data[at..at + 3] == [0, 0, 1])
        .collect()
}

/// An HEVC Annex B clip of `ferricast-nvcodec`'s fixtures as one packet per
/// access unit (cut at its delimiters), 30 fps.
fn annex_b_packets(name: &str) -> Vec<EncodedFrame> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../nvcodec/tests/fixtures")
        .join(name);
    let data = std::fs::read(path).unwrap();
    let mut cuts: Vec<usize> = start_codes(&data)
        .into_iter()
        .filter(|&at| (data[at + 3] >> 1) & 0x3f == 35)
        .map(|at| {
            if at > 0 && data[at - 1] == 0 {
                at - 1
            } else {
                at
            }
        })
        .collect();
    cuts.insert(0, 0);
    cuts.push(data.len());
    cuts.dedup();
    cuts.windows(2)
        .enumerate()
        .map(|(index, cut)| {
            let unit = &data[cut[0]..cut[1]];
            let keyframe = start_codes(unit)
                .iter()
                .any(|&at| (16..=21).contains(&((unit[at + 3] >> 1) & 0x3f)));
            EncodedFrame {
                codec: ferricast_core::Codec::H265,
                data: unit.to_vec().into(),
                timestamp_us: u64::try_from(index).unwrap() * 33_333,
                is_keyframe: keyframe,
                duration_us: None,
                pts_dts: (0, 0),
            }
        })
        .collect()
}

/// `data` without its HEVC access unit delimiters, as MP4 and Matroska
/// carry it.
fn without_hevc_delimiters(data: &[u8]) -> Vec<u8> {
    let mut starts = start_codes(data);
    starts.push(data.len());
    starts
        .windows(2)
        .map(|pair| &data[pair[0]..pair[1]])
        .filter(|unit| (unit.get(3).copied().unwrap_or(0) >> 1) & 0x3f != 35)
        .flat_map(|unit| unit.iter().copied())
        .collect()
}
