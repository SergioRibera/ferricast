//! The sender's clock, as seen from the receiver.
//!
//! Mirroring packets and audio sync packets are stamped on the sender's
//! clock. The receiver asks the sender's `timingPort` for the time (an
//! NTP-style exchange every few seconds, the receiver being the client)
//! and keeps the offset of the exchange with the smallest round trip.
//! [`RemoteClock::to_local_us`] then maps a sender timestamp onto local
//! wall-clock time (microseconds since the UNIX epoch, the clock every
//! timestamp in the pipeline uses).

use std::net::SocketAddr;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::net::UdpSocket;
use tokio::task::JoinHandle;
use tracing::{debug, trace};

/// Seconds between 1900-01-01 (NTP epoch) and 1970-01-01.
const NTP_EPOCH_OFFSET: u64 = 2_208_988_800;
const NANOS_PER_SEC: u64 = 1_000_000_000;
/// Exchanges kept to pick the offset from.
const SAMPLES: usize = 8;
/// First exchanges go out quickly, so an offset is known before media
/// arrives; later ones every few seconds track drift.
const FAST_INTERVAL: Duration = Duration::from_millis(200);
const FAST_COUNT: u32 = 4;
const INTERVAL: Duration = Duration::from_secs(3);
const REPLY_TIMEOUT: Duration = Duration::from_secs(1);

/// Local wall-clock time in nanoseconds since the UNIX epoch.
pub(super) fn local_now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or_default()
}

/// Nanoseconds of a 32.32 fixed-point seconds value.
pub(super) fn fixed_point_to_ns(value: u64) -> u64 {
    let secs = value >> 32;
    let frac = value & 0xffff_ffff;
    secs * NANOS_PER_SEC + ((frac * NANOS_PER_SEC) >> 32)
}

/// Nanoseconds since the UNIX epoch of an NTP timestamp (seconds since
/// 1900). Senders whose clock does not start in 1900 still add the
/// offset, so subtracting it puts every sender timestamp on one scale.
pub(super) fn ntp_to_ns(value: u64) -> u64 {
    fixed_point_to_ns(value.wrapping_sub(NTP_EPOCH_OFFSET << 32))
}

fn ns_to_ntp(ns: u64) -> u64 {
    let secs = ns / NANOS_PER_SEC + NTP_EPOCH_OFFSET;
    let frac = ((ns % NANOS_PER_SEC) << 32) / NANOS_PER_SEC;
    (secs << 32) | frac
}

/// One timing exchange.
#[derive(Debug, Clone, Copy)]
struct Sample {
    /// Sender clock minus local clock.
    offset_ns: i64,
    round_trip_ns: i64,
}

#[derive(Debug, Default)]
struct State {
    samples: Vec<Sample>,
    next: usize,
    /// Offset estimated from when media arrived, used until a timing
    /// exchange succeeds.
    arrival_offset_ns: Option<i64>,
}

impl State {
    fn offset_ns(&self) -> Option<i64> {
        self.samples
            .iter()
            .min_by_key(|s| s.round_trip_ns)
            .map(|s| s.offset_ns)
            .or(self.arrival_offset_ns)
    }
}

/// Maps sender timestamps to local time. Cheap to clone.
#[derive(Debug, Clone, Default)]
pub(super) struct RemoteClock {
    state: Arc<Mutex<State>>,
}

impl RemoteClock {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn add_sample(&self, sample: Sample) {
        let mut state = self.lock();
        if state.samples.len() < SAMPLES {
            state.samples.push(sample);
        } else {
            let at = state.next;
            state.samples[at] = sample;
        }
        state.next = (state.next + 1) % SAMPLES;
    }

    /// Whether a timing exchange succeeded.
    #[cfg(test)]
    pub fn is_synced(&self) -> bool {
        !self.lock().samples.is_empty()
    }

    /// Local time, µs since the UNIX epoch, of sender time `remote_ns`.
    /// Before any timing exchange the first call assumes the sample is
    /// from now.
    pub fn to_local_us(&self, remote_ns: u64) -> u64 {
        let mut state = self.lock();
        let offset = match state.offset_ns() {
            Some(offset) => offset,
            None => {
                let offset = signed_diff(remote_ns, local_now_ns());
                state.arrival_offset_ns = Some(offset);
                offset
            }
        };
        let local_ns = i128::from(remote_ns) - i128::from(offset);
        u64::try_from(local_ns.max(0) / 1_000).unwrap_or(u64::MAX)
    }
}

fn signed_diff(a: u64, b: u64) -> i64 {
    i64::try_from(i128::from(a) - i128::from(b)).unwrap_or(0)
}

/// Request: `0x80 0xd2`, sequence 7, our transmit time at bytes 24..32.
fn request(transmit_ns: u64) -> [u8; 32] {
    let mut p = [0_u8; 32];
    p[..4].copy_from_slice(&[0x80, 0xd2, 0x00, 0x07]);
    p[24..32].copy_from_slice(&ns_to_ntp(transmit_ns).to_be_bytes());
    p
}

/// The sample of a reply received at `received_ns`, or `None` if it is
/// not a timing reply.
fn sample(reply: &[u8], received_ns: u64) -> Option<Sample> {
    if reply.len() < 32 || reply[1] & 0x7f != 0x53 {
        return None;
    }
    let field = |at: usize| {
        reply
            .get(at..at + 8)
            .and_then(|b| b.try_into().ok())
            .map(u64::from_be_bytes)
    };
    let origin = ntp_to_ns(field(8)?);
    let remote_received = ntp_to_ns(field(16)?);
    let remote_sent = ntp_to_ns(field(24)?);
    let t0 = i128::from(origin);
    let t1 = i128::from(remote_received);
    let t2 = i128::from(remote_sent);
    let t3 = i128::from(received_ns);
    Some(Sample {
        offset_ns: i64::try_from(((t1 - t0) + (t2 - t3)) / 2).ok()?,
        round_trip_ns: i64::try_from((t3 - t0) - (t2 - t1)).ok()?,
    })
}

/// Queries the sender's timing port until dropped.
pub(super) struct TimingClient {
    task: JoinHandle<()>,
    port: u16,
}

impl TimingClient {
    /// Start querying `remote` from `socket`, feeding `clock`.
    pub fn start(socket: UdpSocket, remote: SocketAddr, clock: RemoteClock) -> Self {
        let port = socket.local_addr().map(|a| a.port()).unwrap_or_default();
        let task = tokio::spawn(async move {
            let mut buf = [0_u8; 128];
            let mut sent = 0_u32;
            loop {
                let transmit = local_now_ns();
                if let Err(e) = socket.send_to(&request(transmit), remote).await {
                    debug!(%remote, error = %e, "timing request failed");
                }
                sent += 1;
                let deadline = tokio::time::Instant::now() + REPLY_TIMEOUT;
                while let Ok(Ok((len, from))) =
                    tokio::time::timeout_at(deadline, socket.recv_from(&mut buf)).await
                {
                    if from.ip() != remote.ip() {
                        continue;
                    }
                    if let Some(s) = sample(&buf[..len], local_now_ns()) {
                        trace!(
                            offset_ns = s.offset_ns,
                            rtt_ns = s.round_trip_ns,
                            "timing reply"
                        );
                        clock.add_sample(s);
                        break;
                    }
                }
                let wait = if sent < FAST_COUNT {
                    FAST_INTERVAL
                } else {
                    INTERVAL
                };
                tokio::time::sleep(wait).await;
            }
        });
        Self { task, port }
    }

    pub fn port(&self) -> u16 {
        self.port
    }
}

impl Drop for TimingClient {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ntp_round_trips_through_nanoseconds() {
        let ns = 1_700_000_000_123_456_789_u64;
        let back = ntp_to_ns(ns_to_ntp(ns));
        assert!(back.abs_diff(ns) < 2, "{back} vs {ns}");
        assert_eq!(fixed_point_to_ns((3 << 32) | (1 << 31)), 3_500_000_000);
    }

    #[test]
    fn reply_gives_offset_and_round_trip() {
        // Sender 5 s ahead, 2 ms each way, 1 ms in the sender.
        let t0 = 1_000_000_000_000_u64;
        let ahead = 5_000_000_000_u64;
        let mut reply = [0_u8; 32];
        reply[..4].copy_from_slice(&[0x80, 0xd3, 0, 7]);
        reply[8..16].copy_from_slice(&ns_to_ntp(t0).to_be_bytes());
        reply[16..24].copy_from_slice(&ns_to_ntp(t0 + ahead + 2_000_000).to_be_bytes());
        reply[24..32].copy_from_slice(&ns_to_ntp(t0 + ahead + 3_000_000).to_be_bytes());
        let s = sample(&reply, t0 + 5_000_000).unwrap();
        assert!((s.offset_ns - 5_000_000_000).abs() < 10, "{s:?}");
        assert!((s.round_trip_ns - 4_000_000).abs() < 10, "{s:?}");
        assert!(sample(&reply[..20], 0).is_none());
    }

    #[test]
    fn clock_prefers_the_shortest_round_trip() {
        let clock = RemoteClock::default();
        clock.add_sample(Sample {
            offset_ns: 2_000,
            round_trip_ns: 9_000_000,
        });
        clock.add_sample(Sample {
            offset_ns: 1_000_000,
            round_trip_ns: 100_000,
        });
        assert_eq!(clock.to_local_us(11_000_000), 10_000);
    }

    #[tokio::test]
    async fn queries_a_timing_server() {
        let server = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let remote = server.local_addr().unwrap();
        let clock = RemoteClock::default();
        let _client = TimingClient::start(
            UdpSocket::bind("127.0.0.1:0").await.unwrap(),
            remote,
            clock.clone(),
        );
        let mut buf = [0_u8; 64];
        let (n, from) = server.recv_from(&mut buf).await.unwrap();
        assert_eq!(n, 32);
        // Answer as a sender whose clock runs 10 s ahead.
        let now = local_now_ns() + 10 * NANOS_PER_SEC;
        let mut reply = [0_u8; 32];
        reply[..4].copy_from_slice(&[0x80, 0xd3, 0, 7]);
        reply[8..16].copy_from_slice(&buf[24..32]);
        reply[16..24].copy_from_slice(&ns_to_ntp(now).to_be_bytes());
        reply[24..32].copy_from_slice(&ns_to_ntp(now).to_be_bytes());
        server.send_to(&reply, from).await.unwrap();
        for _ in 0..100 {
            if clock.is_synced() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(clock.is_synced());
        let local = clock.to_local_us(local_now_ns() + 10 * NANOS_PER_SEC);
        let now_us = local_now_ns() / 1_000;
        assert!(local.abs_diff(now_us) < 50_000, "{local} vs {now_us}");
    }
}
