//! Sender clock and the NTP-style timing responder.
//!
//! The receiver periodically sends a 32-byte timing request to the
//! `timingPort` announced in `SETUP` and estimates our clock offset from
//! the reply. Mirroring packets are stamped on the same clock (wall-clock
//! time, so capture timestamps in UNIX microseconds map onto it directly).

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use ferricast_core::{FerricastError, Result};
use tokio::net::UdpSocket;
use tokio::task::JoinHandle;
use tracing::{debug, trace};

/// Seconds between 1900-01-01 (NTP epoch) and 1970-01-01.
const NTP_EPOCH_OFFSET: u64 = 2_208_988_800;

/// 32.32 fixed-point seconds for `micros` since the UNIX epoch, *without*
/// the NTP epoch offset (the format of mirroring packet timestamps).
pub(crate) fn fixed_point_from_micros(micros: u64) -> u64 {
    let secs = micros / 1_000_000;
    let frac = ((micros % 1_000_000) << 32) / 1_000_000;
    (secs << 32) | frac
}

fn now_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_micros()).unwrap_or(u64::MAX))
        .unwrap_or_default()
}

/// Current time as an NTP timestamp (seconds since 1900, 32.32).
fn ntp_now() -> u64 {
    fixed_point_from_micros(now_micros()) + (NTP_EPOCH_OFFSET << 32)
}

/// Reply to one timing request (`0x80 0xd2 …`), or `None` if it is not one.
fn timing_reply(request: &[u8], received: u64) -> Option<[u8; 32]> {
    if request.len() < 32 || request[1] & 0x7f != 0x52 {
        return None;
    }
    let mut reply = [0_u8; 32];
    reply[0] = 0x80;
    reply[1] = 0xd3;
    reply[2..4].copy_from_slice(&request[2..4]);
    // Origin = the requester's transmit time.
    reply[8..16].copy_from_slice(&request[24..32]);
    reply[16..24].copy_from_slice(&received.to_be_bytes());
    reply[24..32].copy_from_slice(&ntp_now().to_be_bytes());
    Some(reply)
}

/// UDP timing responder; stops when dropped.
pub(crate) struct TimingServer {
    port: u16,
    task: JoinHandle<()>,
}

impl TimingServer {
    /// Start answering timing requests on `socket`.
    pub fn start(socket: UdpSocket) -> Result<Self> {
        let port = socket
            .local_addr()
            .map_err(|e| FerricastError::Connection(format!("timing socket: {e}")))?
            .port();
        let socket = Arc::new(socket);
        let task = tokio::spawn(async move {
            let mut buf = [0_u8; 128];
            loop {
                let (len, peer): (usize, SocketAddr) = match socket.recv_from(&mut buf).await {
                    Ok(v) => v,
                    Err(e) => {
                        debug!(error = %e, "timing socket closed");
                        break;
                    }
                };
                let received = ntp_now();
                if let Some(reply) = timing_reply(&buf[..len], received) {
                    trace!(%peer, "answering timing request");
                    if let Err(e) = socket.send_to(&reply, peer).await {
                        debug!(error = %e, "timing reply failed");
                    }
                }
            }
        });
        Ok(Self { port, task })
    }

    pub fn port(&self) -> u16 {
        self.port
    }
}

impl Drop for TimingServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point_conversion() {
        assert_eq!(fixed_point_from_micros(1_500_000), (1 << 32) | (1 << 31));
        assert_eq!(fixed_point_from_micros(0), 0);
    }

    #[test]
    fn reply_echoes_origin_and_stamps_times() {
        let mut request = [0_u8; 32];
        request[..4].copy_from_slice(&[0x80, 0xd2, 0x00, 0x07]);
        request[24..32].copy_from_slice(&0x1122_3344_5566_7788_u64.to_be_bytes());
        let reply = timing_reply(&request, 42).unwrap();
        assert_eq!(&reply[..4], &[0x80, 0xd3, 0x00, 0x07]);
        assert_eq!(&reply[8..16], &request[24..32]);
        assert_eq!(u64::from_be_bytes(reply[16..24].try_into().unwrap()), 42);
        let transmit = u64::from_be_bytes(reply[24..32].try_into().unwrap());
        assert!(transmit >> 32 > NTP_EPOCH_OFFSET);
        assert!(timing_reply(&[0x80, 0xd3], 0).is_none());
    }

    #[tokio::test]
    async fn answers_over_udp() {
        let server = TimingServer::start(UdpSocket::bind("127.0.0.1:0").await.unwrap()).unwrap();
        let client = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let mut request = [0_u8; 32];
        request[..4].copy_from_slice(&[0x80, 0xd2, 0x00, 0x07]);
        client
            .send_to(&request, ("127.0.0.1", server.port()))
            .await
            .unwrap();
        let mut buf = [0_u8; 64];
        let n = tokio::time::timeout(std::time::Duration::from_secs(2), client.recv(&mut buf))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(n, 32);
        assert_eq!(buf[1], 0xd3);
    }
}
