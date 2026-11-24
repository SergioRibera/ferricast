//! HE-AAC (SBR) and HE-AAC v2 (PS) decode at their full output rate and
//! channel count. The streams are encoded with libfdk-aac, the only
//! HE-AAC encoder at hand.

#![cfg(feature = "fdk-aac-decode")]

use bytes::Bytes;
use fdk_aac::enc::{AudioObjectType, BitRate, ChannelMode, Encoder, EncoderParams, Transport};
use ferricast_core::{AudioCodec, AudioDecoder, AudioDecoderConfig, AudioFrame};
use ferricast_decoder::AacDecoder;

const RATE: u32 = 48_000;
const SECONDS: usize = 2;

/// `SECONDS` of a 1 kHz tone (left) and a 3 kHz tone (right) at 48 kHz,
/// encoded as ADTS with `aot`; returns the ADTS frames.
fn encode(aot: AudioObjectType) -> Vec<Vec<u8>> {
    let encoder = Encoder::new(EncoderParams {
        bit_rate: BitRate::Cbr(48_000),
        sample_rate: RATE,
        transport: Transport::Adts,
        channels: ChannelMode::Stereo,
        audio_object_type: aot,
    })
    .unwrap();
    let frame = encoder.info().unwrap().frameLength as usize * 2;
    let pcm: Vec<i16> = (0..RATE as usize * SECONDS)
        .flat_map(|n| {
            let t = n as f32 / RATE as f32;
            let l = (t * 1_000.0 * std::f32::consts::TAU).sin();
            let r = (t * 3_000.0 * std::f32::consts::TAU).sin();
            [(l * 12_000.0) as i16, (r * 12_000.0) as i16]
        })
        .collect();
    let mut frames = Vec::new();
    let mut out = vec![0u8; 8192];
    for chunk in pcm.chunks(frame) {
        let mut input = chunk;
        while !input.is_empty() {
            let info = encoder.encode(input, &mut out).unwrap();
            if info.output_size > 0 {
                frames.push(out[..info.output_size].to_vec());
            }
            input = &input[info.input_consumed..];
            if info.input_consumed == 0 {
                break;
            }
        }
    }
    frames
}

fn decode(frames: &[Vec<u8>]) -> (u32, u16, usize) {
    let mut decoder = AacDecoder::default();
    decoder
        .configure(&AudioDecoderConfig {
            codec: AudioCodec::Aac,
            sample_rate: 0,
            channels: 0,
            backend: ferricast_core::BackendSelection::default(),
        })
        .unwrap();
    let mut shape = None;
    let mut samples = 0;
    for (i, data) in frames.iter().enumerate() {
        let frame = AudioFrame {
            codec: AudioCodec::Aac,
            data: Bytes::from(data.clone()),
            timestamp_us: i as u64 * 42_666,
            sample_rate: 0,
            channels: 0,
        };
        let out = decoder.decode(frame).unwrap().into_iter();
        for audio in out.chain(decoder.flush().unwrap()) {
            shape = Some((audio.sample_rate, audio.channels));
            samples += audio.pcm.len() / usize::from(audio.channels);
        }
    }
    let (rate, channels) = shape.expect("decoded audio");
    (rate, channels, samples)
}

#[test]
fn he_aac_decodes_at_the_sbr_rate() {
    let frames = encode(AudioObjectType::Mpeg4HeAac);
    // The ADTS headers name only the 24 kHz core.
    assert_eq!((frames[0][2] >> 2) & 0xF, 6, "core rate index");
    let (rate, channels, samples) = decode(&frames);
    eprintln!("HE-AAC v1: {rate} Hz, {channels} ch, {samples} samples");
    assert_eq!((rate, channels), (48_000, 2));
    assert!(
        samples.abs_diff(RATE as usize * SECONDS) < 4_096,
        "{samples}"
    );
}

#[test]
fn he_aac_v2_decodes_to_stereo_at_the_sbr_rate() {
    let frames = encode(AudioObjectType::Mpeg4HeAacV2);
    // Mono 24 kHz core: parametric stereo carries the rest.
    let channel_config = ((frames[0][2] & 1) << 2) | (frames[0][3] >> 6);
    assert_eq!(channel_config, 1, "mono core");
    let (rate, channels, samples) = decode(&frames);
    eprintln!("HE-AAC v2: {rate} Hz, {channels} ch, {samples} samples");
    assert_eq!((rate, channels), (48_000, 2));
    assert!(
        samples.abs_diff(RATE as usize * SECONDS) < 4_096,
        "{samples}"
    );
}
