//! Screen-mirroring data stream (`SETUP` stream type 110).
//!
//! Each packet is a 128-byte header followed by a payload:
//!
//! | bytes  | meaning                                                   |
//! |--------|-----------------------------------------------------------|
//! | 0..4   | payload length, u32 LE                                    |
//! | 4      | type: 0 video, 1 codec (avcC)                             |
//! | 5      | 0x10 on IDR video packets                                 |
//! | 6..8   | option bytes (`0x16 0x01` on codec packets)               |
//! | 8..16  | timestamp, 32.32 fixed-point seconds, u64 LE              |
//! | 16..64 | codec packets: source / display size as f32 LE            |
//!
//! Video payloads are length-prefixed (AVCC) NAL units, encrypted with one
//! continuous AES-128-CTR stream for the whole connection. Key and IV are
//! the first 16 bytes of `SHA-512("AirPlayStreamKey<id>" ‖ base key)` and
//! `SHA-512("AirPlayStreamIV<id>" ‖ base key)` (UxPlay `mirror_buffer.c`).

use aes::cipher::{KeyIvInit, StreamCipher};
use bytes::Bytes;
use ferricast_core::{EncodedFrame, FerricastError, Result};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

use crate::crypto::{Aes128Ctr, sha512_16};
use crate::ntp::fixed_point_from_micros;

const HEADER_LEN: usize = 128;
const TYPE_VIDEO: u8 = 0;
const TYPE_CODEC: u8 = 1;

const NAL_IDR: u8 = 5;
const NAL_SPS: u8 = 7;
const NAL_PPS: u8 = 8;
const NAL_AUD: u8 = 9;

/// Split an Annex-B byte stream into NAL units (start codes removed).
pub(crate) fn annexb_nals(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut nals = Vec::with_capacity(starts.len());
    for (n, &start) in starts.iter().enumerate() {
        let mut end = starts.get(n + 1).map_or(data.len(), |&next| next - 3);
        // A 4-byte start code leaves one extra zero before the next NAL;
        // trailing zeros are never part of a NAL.
        while end > start && data[end - 1] == 0 {
            end -= 1;
        }
        if end > start {
            nals.push(&data[start..end]);
        }
    }
    nals
}

/// `AVCDecoderConfigurationRecord` for one SPS / PPS pair.
pub(crate) fn avcc_record(sps: &[u8], pps: &[u8]) -> Result<Vec<u8>> {
    let too_long = |_| FerricastError::Streaming("SPS/PPS longer than 65535 bytes".into());
    let sps_len = u16::try_from(sps.len()).map_err(too_long)?;
    let pps_len = u16::try_from(pps.len()).map_err(too_long)?;
    let [_, profile, compat, level, ..] = *sps else {
        return Err(FerricastError::Streaming("SPS too short".into()));
    };
    let mut out = Vec::with_capacity(11 + sps.len() + pps.len());
    out.extend_from_slice(&[1, profile, compat, level, 0xff, 0xe1]);
    out.extend_from_slice(&sps_len.to_be_bytes());
    out.extend_from_slice(sps);
    out.push(1);
    out.extend_from_slice(&pps_len.to_be_bytes());
    out.extend_from_slice(pps);
    Ok(out)
}

fn header(
    payload_len: usize,
    kind: u8,
    flags: [u8; 3],
    timestamp: u64,
) -> Result<[u8; HEADER_LEN]> {
    let len = u32::try_from(payload_len)
        .map_err(|_| FerricastError::Streaming("mirroring payload too large".into()))?;
    let mut h = [0_u8; HEADER_LEN];
    h[..4].copy_from_slice(&len.to_le_bytes());
    h[4] = kind;
    h[5..8].copy_from_slice(&flags);
    h[8..16].copy_from_slice(&timestamp.to_le_bytes());
    Ok(h)
}

fn codec_header(
    payload_len: usize,
    timestamp: u64,
    width: u32,
    height: u32,
) -> Result<[u8; HEADER_LEN]> {
    let mut h = header(payload_len, TYPE_CODEC, [0x00, 0x16, 0x01], timestamp)?;
    // Sizes are small integers, exactly representable as f32.
    #[expect(
        clippy::cast_precision_loss,
        reason = "video dimensions fit in f32 exactly"
    )]
    let (w, hgt) = ((width as f32).to_le_bytes(), (height as f32).to_le_bytes());
    for offset in [16, 40, 48, 56] {
        h[offset..offset + 4].copy_from_slice(&w);
        h[offset + 4..offset + 8].copy_from_slice(&hgt);
    }
    Ok(h)
}

/// Mirroring stream key / IV for `stream_connection_id`.
pub(crate) fn stream_cipher(base_key: &[u8; 16], stream_connection_id: u64) -> Aes128Ctr {
    let key = sha512_16(&[
        format!("AirPlayStreamKey{stream_connection_id}").as_bytes(),
        base_key,
    ]);
    let iv = sha512_16(&[
        format!("AirPlayStreamIV{stream_connection_id}").as_bytes(),
        base_key,
    ]);
    Aes128Ctr::new(&key.into(), &iv.into())
}

/// Writer for the mirroring TCP connection.
pub(crate) struct MirrorStream {
    stream: TcpStream,
    cipher: Aes128Ctr,
    width: u32,
    height: u32,
    /// Last SPS / PPS sent in a codec packet.
    params: Option<(Bytes, Bytes)>,
    /// Nothing is sent until the first IDR; the receiver needs SPS/PPS first.
    started: bool,
    frames_sent: u64,
    /// A frame was dropped while waiting for an IDR; the encoder should
    /// be asked for one.
    keyframe_wanted: bool,
}

impl MirrorStream {
    pub fn new(stream: TcpStream, cipher: Aes128Ctr, width: u32, height: u32) -> Self {
        // Frames are written whole; Nagle only adds latency.
        let _ = stream.set_nodelay(true);
        Self {
            stream,
            cipher,
            width,
            height,
            params: None,
            started: false,
            frames_sent: 0,
            keyframe_wanted: false,
        }
    }

    /// Returns (and clears) whether frames were dropped waiting for an IDR.
    pub fn take_keyframe_request(&mut self) -> bool {
        std::mem::take(&mut self.keyframe_wanted)
    }

    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    /// Send one encoded H.264 access unit (Annex-B).
    pub async fn send(&mut self, frame: &EncodedFrame) -> Result<()> {
        let nals = annexb_nals(&frame.data);
        let mut sps = None;
        let mut pps = None;
        let mut idr = false;
        let mut payload = Vec::with_capacity(frame.data.len() + 16);
        for nal in &nals {
            match nal[0] & 0x1f {
                NAL_SPS => sps = Some(*nal),
                NAL_PPS => pps = Some(*nal),
                NAL_AUD => {}
                kind => {
                    idr |= kind == NAL_IDR;
                    let len = u32::try_from(nal.len())
                        .map_err(|_| FerricastError::Streaming("NAL unit too large".into()))?;
                    payload.extend_from_slice(&len.to_be_bytes());
                    payload.extend_from_slice(nal);
                }
            }
        }
        if let (Some(sps), Some(pps)) = (sps, pps) {
            let changed = self
                .params
                .as_ref()
                .is_none_or(|(s, p)| s.as_ref() != sps || p.as_ref() != pps);
            if changed {
                self.params = Some((Bytes::copy_from_slice(sps), Bytes::copy_from_slice(pps)));
            }
        }
        let can_start = idr && self.params.is_some();
        if !(self.started || can_start) {
            self.keyframe_wanted = true;
            return Ok(());
        }
        if payload.is_empty() {
            return Ok(());
        }

        let timestamp = fixed_point_from_micros(frame.timestamp_us);
        let mut wire = Vec::with_capacity(2 * HEADER_LEN + payload.len() + 64);
        // The receiver prepends the codec packet's SPS/PPS to the next video
        // packet with the same timestamp, so send it before every IDR.
        if idr && let Some((sps, pps)) = &self.params {
            let record = avcc_record(sps, pps)?;
            wire.extend_from_slice(&codec_header(
                record.len(),
                timestamp,
                self.width,
                self.height,
            )?);
            wire.extend_from_slice(&record);
        }
        self.cipher.apply_keystream(&mut payload);
        let flags = [if idr { 0x10 } else { 0x00 }, 0x00, 0x00];
        wire.extend_from_slice(&header(payload.len(), TYPE_VIDEO, flags, timestamp)?);
        wire.extend_from_slice(&payload);
        self.stream.write_all(&wire).await?;
        self.started = true;
        self.frames_sent += 1;
        Ok(())
    }

    pub async fn shutdown(&mut self) {
        let _ = self.stream.shutdown().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_annexb_with_mixed_start_codes() {
        let data = [
            0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 0, 0, 0, 1, 0x65, 4, 5,
        ];
        let nals = annexb_nals(&data);
        assert_eq!(
            nals,
            vec![&[0x67, 1, 2][..], &[0x68, 3][..], &[0x65, 4, 5][..]]
        );
    }

    #[test]
    fn avcc_layout_matches_receiver_parser() {
        let sps = [0x67, 0x64, 0x00, 0x1f, 0xac];
        let pps = [0x68, 0xee, 0x3c];
        let rec = avcc_record(&sps, &pps).unwrap();
        // UxPlay: sps_size at 6, SPS at 8, pps_size at sps_size + 9, PPS at sps_size + 11.
        assert_eq!(u16::from_be_bytes([rec[6], rec[7]]), 5);
        assert_eq!(&rec[8..13], &sps);
        assert_eq!(u16::from_be_bytes([rec[5 + 9], rec[5 + 10]]), 3);
        assert_eq!(&rec[5 + 11..], &pps);
        assert_eq!(rec[1], 0x64);
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "the header stores the exact integer sizes"
    )]
    fn header_layout() {
        let h = codec_header(20, 0x0102_0304_0506_0708, 1280, 720).unwrap();
        assert_eq!(&h[..4], &20_u32.to_le_bytes());
        assert_eq!(h[4], TYPE_CODEC);
        assert_eq!(&h[6..8], &[0x16, 0x01]);
        assert_eq!(&h[8..16], &0x0102_0304_0506_0708_u64.to_le_bytes());
        assert_eq!(f32::from_le_bytes(h[56..60].try_into().unwrap()), 1280.0);
        assert_eq!(f32::from_le_bytes(h[60..64].try_into().unwrap()), 720.0);
    }

    #[test]
    fn stream_cipher_is_continuous_across_packets() {
        // The receiver decrypts each packet by continuing one CTR stream,
        // so encrypting packets separately must equal encrypting them joined.
        let key = [5_u8; 16];
        let mut joined = vec![0xab_u8; 37 + 50];
        stream_cipher(&key, 42).apply_keystream(&mut joined);
        let mut c = stream_cipher(&key, 42);
        let mut a = vec![0xab_u8; 37];
        let mut b = vec![0xab_u8; 50];
        c.apply_keystream(&mut a);
        c.apply_keystream(&mut b);
        a.extend(b);
        assert_eq!(a, joined);
    }
}
