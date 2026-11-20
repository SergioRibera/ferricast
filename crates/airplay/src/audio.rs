//! Screen-mirroring audio stream (`SETUP` stream type 96).
//!
//! AAC-ELD access units (44.1 kHz stereo, 480 samples each) travel as RTP
//! over UDP to the receiver's `dataPort`, as UxPlay (`raop_rtp.c`,
//! `raop_buffer.c`) expects them:
//!
//! * RTP header `0x80 0x60 <seq> <timestamp> 0x00000000`; the timestamp
//!   counts samples at 44.1 kHz.
//! * Payload encrypted with AES-128-CBC, restarting from `eiv` for every
//!   packet. Only whole 16-byte blocks are encrypted; the tail stays clear.
//!   The key is the session's base AES key (the one mirroring derives its
//!   stream key from) and the IV is the `eiv` of the first `SETUP`.
//! * Sync packets (`0x80 0xd4`) on the control port map an RTP timestamp
//!   to the sender's NTP clock, which the receiver tracks through the
//!   timing port; that is how audio and video share one timeline.
//! * The receiver asks for lost packets with `0x80 0xd5 <seq> <first>
//!   <count>` on the control port; they go back as `0x80 0xd6 0x00 0x01`
//!   followed by the original RTP packet.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aes::Aes128;
use aes::cipher::{BlockEncrypt, KeyInit, generic_array::GenericArray};
use ferricast_core::{AudioFrame, FerricastError, Result};
use plist::{Dictionary, Value};
use tokio::net::UdpSocket;
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tracing::{debug, trace, warn};

use crate::ntp::ntp_from_micros;

/// `SETUP` stream type for real-time (mirroring) audio.
pub(crate) const STREAM_TYPE_AUDIO: u64 = 96;
/// `ct` (compression type) of AAC-ELD.
const COMPRESSION_AAC_ELD: u64 = 8;
/// `audioFormat` bit of AAC-ELD 44100 Hz stereo, 480 samples per frame.
const AUDIO_FORMAT_AAC_ELD_44100_2: u64 = 1 << 24;
const SAMPLES_PER_FRAME: u64 = 480;
const SAMPLE_RATE: u64 = 44_100;
/// RTP payload type of AirPlay audio data packets.
const PAYLOAD_TYPE: u8 = 0x60;
const RTP_HEADER_LEN: usize = 12;
/// How often the RTP ↔ NTP mapping is refreshed.
const SYNC_INTERVAL: Duration = Duration::from_secs(1);
/// Offset senders put between the two RTP timestamps of a sync packet
/// for AAC-ELD (0.17 s); receivers only use the first one.
const SYNC_NEXT_RTP_OFFSET: u32 = 7_497;
/// Packets kept for retransmission: about 5.5 s of audio.
const HISTORY_LEN: usize = 512;
/// Receiver latency bounds announced in `SETUP`, in samples.
const LATENCY_MIN: u64 = 11_025;
const LATENCY_MAX: u64 = 88_200;

/// The `streams` entry of the audio `SETUP` request.
pub(crate) fn setup_stream_dict(control_port: u16) -> Dictionary {
    let mut stream = Dictionary::new();
    stream.insert("type".into(), Value::Integer(STREAM_TYPE_AUDIO.into()));
    stream.insert("ct".into(), Value::Integer(COMPRESSION_AAC_ELD.into()));
    stream.insert(
        "audioFormat".into(),
        Value::Integer(AUDIO_FORMAT_AAC_ELD_44100_2.into()),
    );
    stream.insert("spf".into(), Value::Integer(SAMPLES_PER_FRAME.into()));
    stream.insert("sr".into(), Value::Integer(SAMPLE_RATE.into()));
    stream.insert("controlPort".into(), Value::Integer(control_port.into()));
    stream.insert("latencyMin".into(), Value::Integer(LATENCY_MIN.into()));
    stream.insert("latencyMax".into(), Value::Integer(LATENCY_MAX.into()));
    stream.insert("audioMode".into(), Value::String("default".into()));
    stream.insert("isMedia".into(), Value::Boolean(true));
    stream.insert("usingScreen".into(), Value::Boolean(true));
    stream
}

/// AES-128-CBC over the whole 16-byte blocks of each payload, restarting
/// from the same IV every packet.
pub(crate) struct PayloadCipher {
    cipher: Aes128,
    iv: [u8; 16],
}

impl PayloadCipher {
    pub(crate) fn new(key: &[u8; 16], iv: [u8; 16]) -> Self {
        Self {
            cipher: Aes128::new(GenericArray::from_slice(key)),
            iv,
        }
    }

    fn encrypt(&self, payload: &mut [u8]) {
        let mut chain = self.iv;
        for block in payload.chunks_exact_mut(16) {
            for (b, c) in block.iter_mut().zip(chain) {
                *b ^= c;
            }
            let block = GenericArray::from_mut_slice(block);
            self.cipher.encrypt_block(block);
            chain.copy_from_slice(block);
        }
    }
}

/// Maps capture timestamps onto the 44.1 kHz RTP clock.
#[derive(Debug, Clone, Copy)]
struct RtpClock {
    origin_us: u64,
    origin_rtp: u32,
}

impl RtpClock {
    fn new(origin_us: u64) -> Self {
        Self {
            origin_us,
            origin_rtp: rand::random(),
        }
    }

    /// RTP timestamp of a sample captured at `timestamp_us`. Rounded, so
    /// timestamps truncated to whole microseconds land on exact sample
    /// counts.
    fn rtp(&self, timestamp_us: u64) -> u32 {
        let elapsed = timestamp_us.saturating_sub(self.origin_us);
        let samples = (u128::from(elapsed) * u128::from(SAMPLE_RATE) + 500_000) / 1_000_000;
        // RTP timestamps wrap modulo 2^32 by design.
        #[expect(clippy::cast_possible_truncation, reason = "RTP time wraps")]
        self.origin_rtp.wrapping_add(samples as u32)
    }
}

/// Recently sent packets, for retransmission.
#[derive(Default)]
struct History(VecDeque<(u16, Vec<u8>)>);

impl History {
    fn push(&mut self, seq: u16, packet: Vec<u8>) {
        if self.0.len() == HISTORY_LEN {
            self.0.pop_front();
        }
        self.0.push_back((seq, packet));
    }

    fn get(&self, seq: u16) -> Option<&[u8]> {
        self.0
            .iter()
            .rev()
            .find(|(s, _)| *s == seq)
            .map(|(_, p)| p.as_slice())
    }
}

fn sync_packet(first: bool, rtp: u32, ntp: u64) -> [u8; 20] {
    let mut p = [0_u8; 20];
    p[0] = if first { 0x90 } else { 0x80 };
    p[1] = 0xd4;
    p[2..4].copy_from_slice(&4_u16.to_be_bytes());
    p[4..8].copy_from_slice(&rtp.to_be_bytes());
    p[8..16].copy_from_slice(&ntp.to_be_bytes());
    p[16..20].copy_from_slice(&rtp.wrapping_add(SYNC_NEXT_RTP_OFFSET).to_be_bytes());
    p
}

/// Answer retransmission requests until the socket fails.
async fn serve_resends(control: Arc<UdpSocket>, history: Arc<Mutex<History>>) {
    let mut buf = [0_u8; 64];
    loop {
        let len = match control.recv(&mut buf).await {
            Ok(len) => len,
            Err(e) => {
                debug!(error = %e, "audio control socket closed");
                break;
            }
        };
        let Some(request) = buf.get(..len) else {
            continue;
        };
        if len < 8 || request[1] & 0x7f != 0x55 {
            trace!(len, "ignoring audio control packet");
            continue;
        }
        let first = u16::from_be_bytes([request[4], request[5]]);
        let count = u16::from_be_bytes([request[6], request[7]]);
        debug!(first, count, "receiver asked for audio retransmission");
        let resends: Vec<Vec<u8>> = {
            let Ok(history) = history.lock() else { break };
            (0..count)
                .filter_map(|i| history.get(first.wrapping_add(i)))
                .map(|packet| {
                    let mut out = Vec::with_capacity(4 + packet.len());
                    out.extend_from_slice(&[0x80, 0xd6, 0x00, 0x01]);
                    out.extend_from_slice(packet);
                    out
                })
                .collect()
        };
        for packet in resends {
            if let Err(e) = control.send(&packet).await {
                warn!(error = %e, "audio retransmission failed");
            }
        }
    }
}

/// Sender side of the mirroring audio stream.
pub(crate) struct AudioStream {
    data: UdpSocket,
    control: Arc<UdpSocket>,
    cipher: PayloadCipher,
    seq: u16,
    clock: Option<RtpClock>,
    last_sync: Option<Instant>,
    history: Arc<Mutex<History>>,
    resends: JoinHandle<()>,
    frames_sent: u64,
}

impl AudioStream {
    /// Start streaming to the receiver's `data_addr` / `control_addr`.
    /// `control` is the socket whose port went into `SETUP`.
    pub(crate) async fn start(
        data: UdpSocket,
        control: UdpSocket,
        data_addr: SocketAddr,
        control_addr: SocketAddr,
        cipher: PayloadCipher,
    ) -> Result<Self> {
        data.connect(data_addr).await.map_err(|e| {
            FerricastError::Connection(format!("audio data socket to {data_addr}: {e}"))
        })?;
        control.connect(control_addr).await.map_err(|e| {
            FerricastError::Connection(format!("audio control socket to {control_addr}: {e}"))
        })?;
        let control = Arc::new(control);
        let history = Arc::new(Mutex::new(History::default()));
        let resends = tokio::spawn(serve_resends(control.clone(), history.clone()));
        Ok(Self {
            data,
            control,
            cipher,
            seq: rand::random(),
            clock: None,
            last_sync: None,
            history,
            resends,
            frames_sent: 0,
        })
    }

    pub(crate) fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    /// Send one AAC-ELD access unit.
    pub(crate) async fn send(&mut self, frame: &AudioFrame) -> Result<()> {
        let clock = *self
            .clock
            .get_or_insert_with(|| RtpClock::new(frame.timestamp_us));
        let rtp = clock.rtp(frame.timestamp_us);

        let due = self
            .last_sync
            .is_none_or(|at| at.elapsed() >= SYNC_INTERVAL);
        if due {
            let first = self.last_sync.is_none();
            let sync = sync_packet(first, rtp, ntp_from_micros(frame.timestamp_us));
            self.control.send(&sync).await?;
            self.last_sync = Some(Instant::now());
        }

        let mut packet = Vec::with_capacity(RTP_HEADER_LEN + frame.data.len());
        packet.extend_from_slice(&[0x80, PAYLOAD_TYPE]);
        packet.extend_from_slice(&self.seq.to_be_bytes());
        packet.extend_from_slice(&rtp.to_be_bytes());
        packet.extend_from_slice(&[0; 4]);
        packet.extend_from_slice(&frame.data);
        self.cipher.encrypt(&mut packet[RTP_HEADER_LEN..]);
        self.data.send(&packet).await?;
        if let Ok(mut history) = self.history.lock() {
            history.push(self.seq, packet);
        }
        self.seq = self.seq.wrapping_add(1);
        self.frames_sent += 1;
        Ok(())
    }
}

impl Drop for AudioStream {
    fn drop(&mut self) {
        self.resends.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cbc_leaves_the_partial_block_clear() {
        let key = [7_u8; 16];
        let iv = [9_u8; 16];
        let mut payload: Vec<u8> = (0..37).collect();
        PayloadCipher::new(&key, iv).encrypt(&mut payload);
        assert_eq!(&payload[32..], &[32, 33, 34, 35, 36]);

        // Reference: CBC by hand with the raw block cipher.
        let aes = Aes128::new(GenericArray::from_slice(&key));
        let mut b0: [u8; 16] = std::array::from_fn(|i| i as u8 ^ iv[i]);
        aes.encrypt_block(GenericArray::from_mut_slice(&mut b0));
        let mut b1: [u8; 16] = std::array::from_fn(|i| (i as u8 + 16) ^ b0[i]);
        aes.encrypt_block(GenericArray::from_mut_slice(&mut b1));
        assert_eq!(&payload[..16], &b0);
        assert_eq!(&payload[16..32], &b1);
    }

    #[test]
    fn cbc_restarts_from_the_iv_for_every_packet() {
        let c = PayloadCipher::new(&[1; 16], [2; 16]);
        let mut a = vec![5_u8; 32];
        let mut b = vec![5_u8; 32];
        c.encrypt(&mut a);
        c.encrypt(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn rtp_clock_counts_480_samples_per_frame() {
        let clock = RtpClock {
            origin_us: 1_000_000,
            origin_rtp: u32::MAX - 100,
        };
        // Encoder timestamps: origin + floor(n * 480 / 44100 s).
        for n in 0..2000_u64 {
            let ts = 1_000_000 + n * 480 * 1_000_000 / 44_100;
            let expected = (u32::MAX - 100).wrapping_add((n * 480) as u32);
            assert_eq!(clock.rtp(ts), expected, "frame {n}");
        }
    }

    #[test]
    fn sync_packet_layout() {
        let p = sync_packet(true, 0x0102_0304, 0x1112_1314_1516_1718);
        assert_eq!(&p[..4], &[0x90, 0xd4, 0x00, 0x04]);
        assert_eq!(&p[4..8], &[1, 2, 3, 4]);
        assert_eq!(&p[8..16], &0x1112_1314_1516_1718_u64.to_be_bytes());
        assert_eq!(
            u32::from_be_bytes(p[16..20].try_into().unwrap()),
            0x0102_0304 + SYNC_NEXT_RTP_OFFSET
        );
        assert_eq!(sync_packet(false, 0, 0)[0], 0x80);
    }

    #[test]
    fn history_drops_the_oldest_packets() {
        let mut h = History::default();
        for seq in 0..(HISTORY_LEN as u16 + 10) {
            h.push(seq, vec![seq as u8]);
        }
        assert!(h.get(5).is_none());
        assert_eq!(
            h.get(HISTORY_LEN as u16 + 9),
            Some(&[(HISTORY_LEN as u16 + 9) as u8][..])
        );
    }
}
