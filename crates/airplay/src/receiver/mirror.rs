//! Receiving end of the screen-mirroring data stream (`SETUP` stream
//! type 110); the wire format is described in [`crate::mirror`].
//!
//! Codec packets carry the parameter sets (`avcC`, or an `hvc1` record for
//! HEVC) and the picture size; they are turned into Annex-B and put in
//! front of the next video packet, which carries the same timestamp and
//! is the keyframe they belong to. Video payloads are length-prefixed NAL
//! units under one continuous AES-CTR stream, rewritten to Annex-B.

use aes::cipher::StreamCipher;
use bytes::Bytes;
use ferricast_core::{Codec, EncodedFrame, FerricastError, Result};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::sync::mpsc;
use tracing::{debug, trace, warn};

use super::assembler::MediaEvent;
use super::clock::{RemoteClock, fixed_point_to_ns};
use crate::crypto::Aes128Ctr;

const HEADER_LEN: usize = 128;
const TYPE_VIDEO: u8 = 0;
const TYPE_CODEC: u8 = 1;
/// Largest payload accepted; a 4K keyframe is well below this.
const MAX_PAYLOAD: usize = 32 * 1024 * 1024;
const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// Parameter sets from a codec packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CodecConfig {
    pub codec: Codec,
    /// The parameter sets as Annex-B NAL units.
    pub annexb: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// NAL units of a big-endian 16-bit-length-prefixed list.
fn take_sized(data: &[u8], at: usize) -> Option<(&[u8], usize)> {
    let len = usize::from(u16::from_be_bytes([*data.get(at)?, *data.get(at + 1)?]));
    let nal = data.get(at + 2..at + 2 + len)?;
    Some((nal, at + 2 + len))
}

/// Annex-B parameter sets of an `AVCDecoderConfigurationRecord`.
fn avcc_to_annexb(record: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let sps_count = usize::from(*record.get(5)? & 0x1f);
    let mut at = 6;
    for _ in 0..sps_count {
        let (sps, next) = take_sized(record, at)?;
        out.extend_from_slice(&START_CODE);
        out.extend_from_slice(sps);
        at = next;
    }
    let pps_count = usize::from(*record.get(at)?);
    at += 1;
    for _ in 0..pps_count {
        let (pps, next) = take_sized(record, at)?;
        out.extend_from_slice(&START_CODE);
        out.extend_from_slice(pps);
        at = next;
    }
    (!out.is_empty()).then_some(out)
}

/// Annex-B VPS / SPS / PPS of an `HEVCDecoderConfigurationRecord`.
fn hvcc_to_annexb(record: &[u8]) -> Option<Vec<u8>> {
    let arrays = usize::from(*record.get(22)?);
    let mut at = 23;
    let mut out = Vec::new();
    for _ in 0..arrays {
        let count = usize::from(u16::from_be_bytes([
            *record.get(at + 1)?,
            *record.get(at + 2)?,
        ]));
        at += 3;
        for _ in 0..count {
            let (nal, next) = take_sized(record, at)?;
            out.extend_from_slice(&START_CODE);
            out.extend_from_slice(nal);
            at = next;
        }
    }
    (!out.is_empty()).then_some(out)
}

/// The decoder configuration record inside an HEVC codec packet. Senders
/// wrap it in an `hvc1` sample entry whose `hvcC` box holds the record;
/// it is found by its box type rather than at a fixed offset.
fn hevc_record(payload: &[u8]) -> Option<&[u8]> {
    let pos = payload.windows(4).position(|w| w == b"hvcC")?;
    let start = pos.checked_sub(4)?;
    let size = u32::from_be_bytes(payload.get(start..pos)?.try_into().ok()?) as usize;
    payload.get(pos + 4..(start + size).min(payload.len()))
}

fn header_f32(header: &[u8; HEADER_LEN], at: usize) -> u32 {
    let v = f32::from_le_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]]);
    if (1.0..65_536.0).contains(&v) {
        // Picture sizes are small integers stored as floats.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "range checked just above"
        )]
        let n = v as u32;
        n
    } else {
        0
    }
}

/// Parse a codec packet.
pub(super) fn parse_codec(header: &[u8; HEADER_LEN], payload: &[u8]) -> Result<CodecConfig> {
    let mut width = header_f32(header, 56);
    let mut height = header_f32(header, 60);
    if width == 0 || height == 0 {
        width = header_f32(header, 16);
        height = header_f32(header, 20);
    }
    let is_hevc = payload.get(4..8) == Some(b"hvc1") || payload.windows(4).any(|w| w == b"hvcC");
    let (codec, annexb) = if is_hevc {
        (Codec::H265, hevc_record(payload).and_then(hvcc_to_annexb))
    } else {
        (Codec::H264, avcc_to_annexb(payload))
    };
    let annexb = annexb.ok_or_else(|| {
        FerricastError::Protocol(format!(
            "unreadable {codec} codec packet ({} bytes)",
            payload.len()
        ))
    })?;
    Ok(CodecConfig {
        codec,
        annexb,
        width,
        height,
    })
}

/// Rewrite length-prefixed NAL units to Annex-B in place. Returns whether
/// one of them is a keyframe, or `None` when the lengths do not add up
/// (a decryption with the wrong key).
fn avcc_to_annexb_in_place(data: &mut [u8], codec: Codec) -> Option<bool> {
    let mut at = 0;
    let mut keyframe = false;
    while at < data.len() {
        let len = u32::from_be_bytes(data.get(at..at + 4)?.try_into().ok()?) as usize;
        let nal_start = at + 4;
        let header = *data.get(nal_start)?;
        if header & 0x80 != 0 || nal_start + len > data.len() {
            return None;
        }
        keyframe |= match codec {
            Codec::H265 => (16..=21).contains(&((header >> 1) & 0x3f)),
            _ => header & 0x1f == 5,
        };
        data[at..nal_start].copy_from_slice(&START_CODE);
        at = nal_start + len;
    }
    Some(keyframe)
}

/// Read the mirroring stream from `stream` until it ends, decrypting with
/// `cipher` and reporting to `events`.
pub(super) async fn receive<S: AsyncRead + Unpin>(
    mut stream: S,
    mut cipher: Aes128Ctr,
    clock: RemoteClock,
    events: mpsc::Sender<MediaEvent>,
) -> Result<()> {
    let mut config: Option<CodecConfig> = None;
    // Parameter sets from the last codec packet, not yet handed to the
    // decoder. Senders differ in how they time codec packets (iOS stamps
    // them like the following frame, others do not), so they go in front
    // of the next video packet whatever its timestamp.
    let mut pending_params: Option<Vec<u8>> = None;
    let mut frames = 0_u64;
    let mut bad = 0_u64;
    loop {
        let mut header = [0_u8; HEADER_LEN];
        match stream.read_exact(&mut header).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let len = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        if len > MAX_PAYLOAD {
            return Err(FerricastError::Protocol(format!(
                "mirroring payload of {len} bytes"
            )));
        }
        let mut payload = vec![0_u8; len];
        stream.read_exact(&mut payload).await?;
        let raw_ts = u64::from_le_bytes(header[8..16].try_into().unwrap_or_default());
        match header[4] {
            TYPE_VIDEO => {
                cipher.apply_keystream(&mut payload);
                let codec = config.as_ref().map_or(Codec::H264, |c| c.codec);
                let Some(keyframe) = avcc_to_annexb_in_place(&mut payload, codec) else {
                    bad += 1;
                    if bad == 1 || bad.is_power_of_two() {
                        warn!(
                            bad,
                            "mirroring video packet does not decrypt to NAL units; wrong key?"
                        );
                    }
                    continue;
                };
                // Every keyframe also carries the current parameter sets, so
                // a decoder that missed or dropped them can start there.
                let params = pending_params.take().or_else(|| {
                    keyframe
                        .then(|| config.as_ref().map(|c| c.annexb.clone()))
                        .flatten()
                });
                let data = match params {
                    Some(mut params) => {
                        params.extend_from_slice(&payload);
                        params
                    }
                    None => payload,
                };
                let timestamp_us = clock.to_local_us(fixed_point_to_ns(raw_ts));
                frames += 1;
                let frame = EncodedFrame {
                    codec,
                    data: Bytes::from(data),
                    timestamp_us,
                    is_keyframe: keyframe,
                    duration_us: None,
                    pts_dts: (timestamp_us, timestamp_us),
                };
                if events.send(MediaEvent::Video(frame)).await.is_err() {
                    break;
                }
            }
            TYPE_CODEC => {
                if payload.is_empty() {
                    warn!(
                        "empty codec packet: the sender uses a codec this receiver did not offer"
                    );
                    continue;
                }
                let parsed = parse_codec(&header, &payload)?;
                debug!(
                    codec = %parsed.codec,
                    width = parsed.width,
                    height = parsed.height,
                    "mirroring codec packet"
                );
                pending_params = Some(parsed.annexb.clone());
                let changed = config.as_ref().is_none_or(|c| {
                    (c.codec, c.width, c.height) != (parsed.codec, parsed.width, parsed.height)
                });
                if changed
                    && events
                        .send(MediaEvent::VideoFormat {
                            codec: parsed.codec,
                            width: parsed.width,
                            height: parsed.height,
                        })
                        .await
                        .is_err()
                {
                    break;
                }
                config = Some(parsed);
            }
            other => trace!(kind = other, len, "mirroring packet ignored"),
        }
    }
    debug!(frames, "mirroring data stream ended");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mirror::avcc_record;

    #[test]
    fn avcc_becomes_annexb() {
        let sps = [0x67, 0x64, 0x00, 0x1f, 0xac];
        let pps = [0x68, 0xee, 0x3c];
        let rec = avcc_record(&sps, &pps).unwrap();
        let out = avcc_to_annexb(&rec).unwrap();
        assert_eq!(out, [&START_CODE[..], &sps, &START_CODE, &pps].concat());
    }

    #[test]
    fn hvcc_becomes_annexb() {
        let mut rec = vec![0_u8; 22];
        rec.push(3);
        for (kind, nal) in [
            (32_u8, &[0x40, 1][..]),
            (33, &[0x42, 1, 2]),
            (34, &[0x44, 1]),
        ] {
            rec.push(kind);
            rec.extend_from_slice(&1_u16.to_be_bytes());
            rec.extend_from_slice(&u16::try_from(nal.len()).unwrap().to_be_bytes());
            rec.extend_from_slice(nal);
        }
        let mut payload = vec![0, 0, 0, 0];
        payload.extend_from_slice(b"hvc1");
        payload.extend_from_slice(&[0; 20]);
        payload.extend_from_slice(&u32::try_from(rec.len() + 8).unwrap().to_be_bytes());
        payload.extend_from_slice(b"hvcC");
        payload.extend_from_slice(&rec);
        let header = [0_u8; HEADER_LEN];
        let c = parse_codec(&header, &payload).unwrap();
        assert_eq!(c.codec, Codec::H265);
        assert_eq!(
            c.annexb,
            [
                &START_CODE[..],
                &[0x40, 1],
                &START_CODE,
                &[0x42, 1, 2],
                &START_CODE,
                &[0x44, 1]
            ]
            .concat()
        );
    }

    #[test]
    fn length_prefixes_become_start_codes() {
        let mut data = vec![0, 0, 0, 2, 0x65, 0xaa, 0, 0, 0, 1, 0x41];
        assert_eq!(avcc_to_annexb_in_place(&mut data, Codec::H264), Some(true));
        assert_eq!(data, [0, 0, 0, 1, 0x65, 0xaa, 0, 0, 0, 1, 0x41]);
        let mut garbage = vec![0xff, 0, 0, 2, 0x65];
        assert_eq!(avcc_to_annexb_in_place(&mut garbage, Codec::H264), None);
    }
}
