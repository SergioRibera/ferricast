//! Receiving end of the audio stream (`SETUP` stream type 96); the wire
//! format is described in [`crate::audio`].
//!
//! Packets are put back in sequence order in a small jitter buffer. A
//! gap asks the sender to retransmit the missing packets once; if they
//! do not arrive within [`GAP_WAIT`] playback skips them. Timestamps come
//! from the RTP clock, mapped to the sender's clock by the sync packets
//! and from there to local time by the [`RemoteClock`].

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Duration;

use aes::Aes128;
use aes::cipher::{BlockDecrypt, KeyInit, generic_array::GenericArray};
use bytes::Bytes;
use ferricast_core::{AudioCodec, AudioFrame};
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tracing::{debug, trace};

use super::assembler::MediaEvent;
use super::clock::{RemoteClock, local_now_ns, ntp_to_ns};

const RTP_HEADER_LEN: usize = 12;
/// Payload of the placeholder packets some senders send before audio.
const NO_DATA_MARKER: [u8; 4] = [0x00, 0x68, 0x34, 0x00];
/// How long a gap waits for its retransmission before it is skipped.
const GAP_WAIT: Duration = Duration::from_millis(100);
/// A packet this far ahead of the expected one means the stream jumped.
const MAX_AHEAD: u16 = 1024;
/// Most packets asked for in one retransmission request.
const MAX_RESEND: u16 = 64;
const TICK: Duration = Duration::from_millis(20);

/// What the stream carries, from its `SETUP` entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AudioFormat {
    pub codec: AudioCodec,
    pub sample_rate: u32,
    pub channels: u16,
}

/// AES-128-CBC over the whole 16-byte blocks of a payload, from the same
/// IV every packet; the tail is clear.
pub(super) struct PayloadDecipher {
    cipher: Aes128,
    iv: [u8; 16],
}

impl PayloadDecipher {
    pub fn new(key: &[u8; 16], iv: [u8; 16]) -> Self {
        Self {
            cipher: Aes128::new(GenericArray::from_slice(key)),
            iv,
        }
    }

    pub fn decrypt(&self, payload: &mut [u8]) {
        let mut chain = self.iv;
        for block in payload.chunks_exact_mut(16) {
            let mut next = [0_u8; 16];
            next.copy_from_slice(block);
            self.cipher
                .decrypt_block(GenericArray::from_mut_slice(block));
            for (b, c) in block.iter_mut().zip(chain) {
                *b ^= c;
            }
            chain = next;
        }
    }
}

/// Where RTP time 0 of the stream sits.
#[derive(Debug, Clone, Copy)]
enum Anchor {
    /// From a sync packet: `rtp` was sent at sender time `remote_ns`.
    Remote { rtp: u32, remote_ns: u64 },
    /// No sync packet yet: `rtp` arrived at local time `local_ns`.
    Arrival { rtp: u32, local_ns: u64 },
}

/// `a - b` in RTP sequence space.
fn seq_diff(a: u16, b: u16) -> i32 {
    i32::from(a.wrapping_sub(b) as i16)
}

/// The audio receiver of one stream.
pub(super) struct AudioReceiver {
    pub data: UdpSocket,
    pub control: UdpSocket,
    pub decipher: PayloadDecipher,
    pub format: AudioFormat,
    pub clock: RemoteClock,
    pub events: mpsc::Sender<MediaEvent>,
    /// Sender's control port from `SETUP`; replaced by the address its
    /// control packets come from.
    pub remote_control: Option<SocketAddr>,
    /// `FLUSH` requests, with the next sequence number when known.
    pub flushes: mpsc::Receiver<Option<u16>>,
}

struct Jitter {
    packets: HashMap<u16, (u32, Vec<u8>)>,
    next: Option<u16>,
    gap_since: Option<Instant>,
}

impl Jitter {
    fn insert(&mut self, seq: u16, rtp: u32, payload: Vec<u8>) {
        let next = *self.next.get_or_insert(seq);
        let ahead = seq_diff(seq, next);
        if ahead < 0 {
            trace!(seq, "late audio packet dropped");
            return;
        }
        if ahead > i32::from(MAX_AHEAD) {
            debug!(seq, expected = next, "audio sequence jumped; restarting");
            self.packets.clear();
            self.next = Some(seq);
            self.gap_since = None;
        }
        self.packets.entry(seq).or_insert((rtp, payload));
    }

    /// The earliest buffered sequence number.
    fn first(&self) -> Option<u16> {
        let next = self.next?;
        self.packets
            .keys()
            .copied()
            .min_by_key(|s| seq_diff(*s, next))
    }

    fn flush(&mut self, next: Option<u16>) {
        self.packets.clear();
        self.next = next;
        self.gap_since = None;
    }
}

impl AudioReceiver {
    pub async fn run(mut self) {
        let mut jitter = Jitter {
            packets: HashMap::new(),
            next: None,
            gap_since: None,
        };
        let mut anchor: Option<Anchor> = None;
        let mut resend_seq = 0_u16;
        let mut data_buf = vec![0_u8; 4096];
        let mut control_buf = vec![0_u8; 4096];
        let mut tick = tokio::time::interval(TICK);
        let mut frames = 0_u64;
        loop {
            tokio::select! {
                received = self.data.recv(&mut data_buf) => {
                    let Ok(len) = received else { break };
                    if let Some((seq, rtp, payload)) = self.packet(&data_buf[..len]) {
                        jitter.insert(seq, rtp, payload);
                    }
                }
                received = self.control.recv_from(&mut control_buf) => {
                    let Ok((len, from)) = received else { break };
                    self.remote_control = Some(from);
                    let packet = &control_buf[..len];
                    match packet.get(1).map(|b| b & 0x7f) {
                        // Sync: RTP time ↔ sender time.
                        Some(0x54) if len >= 16 => {
                            let rtp = u32::from_be_bytes(packet[4..8].try_into().unwrap_or_default());
                            let ntp = u64::from_be_bytes(packet[8..16].try_into().unwrap_or_default());
                            trace!(rtp, "audio sync");
                            anchor = Some(Anchor::Remote { rtp, remote_ns: ntp_to_ns(ntp) });
                        }
                        // Retransmitted packet, after a 4-byte header.
                        Some(0x56) if len >= 4 + RTP_HEADER_LEN => {
                            if let Some((seq, rtp, payload)) = self.packet(&packet[4..]) {
                                jitter.insert(seq, rtp, payload);
                            }
                        }
                        _ => trace!(len, "audio control packet ignored"),
                    }
                }
                flush = self.flushes.recv() => match flush {
                    Some(next) => {
                        debug!(?next, "audio flushed");
                        jitter.flush(next);
                    }
                    None => break,
                },
                _ = tick.tick() => {}
            }
            // Hand out everything that is in order.
            while let Some(next) = jitter.next {
                if let Some((rtp, mut payload)) = jitter.packets.remove(&next) {
                    jitter.next = Some(next.wrapping_add(1));
                    jitter.gap_since = None;
                    self.decipher.decrypt(&mut payload);
                    let anchor = *anchor.get_or_insert(Anchor::Arrival {
                        rtp,
                        local_ns: local_now_ns(),
                    });
                    let timestamp_us = self.timestamp_us(anchor, rtp);
                    frames += 1;
                    let frame = AudioFrame {
                        codec: self.format.codec,
                        data: Bytes::from(payload),
                        timestamp_us,
                        sample_rate: self.format.sample_rate,
                        channels: self.format.channels,
                    };
                    if self.events.send(MediaEvent::Audio(frame)).await.is_err() {
                        return;
                    }
                    continue;
                }
                let Some(first) = jitter.first() else { break };
                match jitter.gap_since {
                    None => {
                        jitter.gap_since = Some(Instant::now());
                        let missing = u16::try_from(seq_diff(first, next)).unwrap_or(MAX_RESEND);
                        self.request_resend(&mut resend_seq, next, missing.min(MAX_RESEND))
                            .await;
                        break;
                    }
                    Some(since) if since.elapsed() >= GAP_WAIT => {
                        debug!(from = next, to = first, "audio packets lost");
                        jitter.next = Some(first);
                        jitter.gap_since = None;
                    }
                    Some(_) => break,
                }
            }
        }
        debug!(frames, "audio stream ended");
    }

    /// Sequence number, RTP time and (encrypted) payload of a data
    /// packet; `None` for placeholders.
    fn packet(&self, packet: &[u8]) -> Option<(u16, u32, Vec<u8>)> {
        if packet.len() <= RTP_HEADER_LEN {
            return None;
        }
        let payload = &packet[RTP_HEADER_LEN..];
        if payload == NO_DATA_MARKER {
            return None;
        }
        // ALAC streams start with 32-byte packets that only describe the
        // format.
        if self.format.codec == AudioCodec::Alac && payload.len() == 32 {
            return None;
        }
        let seq = u16::from_be_bytes([packet[2], packet[3]]);
        let rtp = u32::from_be_bytes([packet[4], packet[5], packet[6], packet[7]]);
        Some((seq, rtp, payload.to_vec()))
    }

    fn timestamp_us(&self, anchor: Anchor, rtp: u32) -> u64 {
        let rate = i128::from(self.format.sample_rate.max(1));
        let offset_ns =
            |base: u32| i128::from(rtp.wrapping_sub(base) as i32) * 1_000_000_000 / rate;
        match anchor {
            Anchor::Remote {
                rtp: base,
                remote_ns,
            } => {
                let remote = (i128::from(remote_ns) + offset_ns(base)).max(0);
                self.clock
                    .to_local_us(u64::try_from(remote).unwrap_or(u64::MAX))
            }
            Anchor::Arrival {
                rtp: base,
                local_ns,
            } => {
                let local = (i128::from(local_ns) + offset_ns(base)).max(0) / 1_000;
                u64::try_from(local).unwrap_or(u64::MAX)
            }
        }
    }

    async fn request_resend(&self, our_seq: &mut u16, first: u16, count: u16) {
        let Some(to) = self.remote_control else {
            return;
        };
        *our_seq = our_seq.wrapping_add(1);
        let mut p = [0_u8; 8];
        p[0] = 0x80;
        p[1] = 0xd5;
        p[2..4].copy_from_slice(&our_seq.to_be_bytes());
        p[4..6].copy_from_slice(&first.to_be_bytes());
        p[6..8].copy_from_slice(&count.to_be_bytes());
        debug!(first, count, "asking for audio retransmission");
        if let Err(e) = self.control.send_to(&p, to).await {
            debug!(error = %e, "audio retransmission request failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::PayloadCipher;

    #[test]
    fn decipher_inverts_the_sender_cipher() {
        let key = [3_u8; 16];
        let iv = [4_u8; 16];
        let plain: Vec<u8> = (0..45).collect();
        let mut wire = plain.clone();
        PayloadCipher::new(&key, iv).encrypt(&mut wire);
        assert_ne!(wire[..32], plain[..32]);
        PayloadDecipher::new(&key, iv).decrypt(&mut wire);
        assert_eq!(wire, plain);
    }

    #[test]
    fn jitter_buffer_orders_and_drops_late_packets() {
        let mut j = Jitter {
            packets: HashMap::new(),
            next: None,
            gap_since: None,
        };
        j.insert(65_535, 0, vec![1]);
        j.insert(1, 0, vec![3]);
        assert_eq!(j.first(), Some(65_535));
        j.packets.remove(&65_535);
        j.next = Some(0);
        assert_eq!(j.first(), Some(1));
        j.insert(65_534, 0, vec![9]);
        assert!(!j.packets.contains_key(&65_534));
        j.insert(5000, 0, vec![7]);
        assert_eq!(j.next, Some(5000));
        assert_eq!(j.packets.len(), 1);
    }
}
