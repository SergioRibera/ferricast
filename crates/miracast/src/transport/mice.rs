//! MS-MICE (Miracast over Infrastructure) discovery + preamble.
//!
//! Sink advertises `_display._tcp.local.` with TXT `mice=1`,
//! `rtsp_port=7236`. Source browses the service, resolves one host,
//! opens TCP to its MICE control port (7250 by convention) and does
//! the 4-byte probe:
//!
//! ```text
//! client → server : 00 00 00 00
//! server → client : 20-byte response (16-byte session nonce + 4-byte status LE)
//! ```
//!
//! On a non-zero status the session is refused (sink busy, policy
//! denied). On success the source opens a *separate* TCP to the sink's
//! RTSP port 7236 and runs the normal WFD M1–M7.

use std::net::SocketAddr;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use super::TransportError;

/// Default MICE pre-association TCP port on the sink.
pub const MICE_CONTROL_PORT: u16 = 7250;
/// Default RTSP TCP port on the sink.
pub const WFD_RTSP_PORT: u16 = 7236;
/// mDNS service type for MICE discovery.
pub const MICE_SERVICE_TYPE: &str = "_display._tcp.local.";

/// Response to the MICE preamble.
#[derive(Debug, Clone)]
pub struct MiceHandshake {
    pub session_nonce: [u8; 16],
    pub status: u32,
}

impl MiceHandshake {
    #[must_use]
    pub const fn is_ok(&self) -> bool {
        self.status == 0
    }
}

/// Perform the client-side MICE preamble against `addr`. The caller
/// passes an already-resolved MICE control endpoint; `wall_timeout`
/// bounds the overall exchange.
pub async fn mice_probe(
    addr: SocketAddr,
    wall_timeout: Duration,
) -> Result<MiceHandshake, TransportError> {
    let fut = async move {
        let mut tcp = TcpStream::connect(addr).await?;
        tcp.write_all(&[0u8; 4]).await?;
        let mut resp = [0u8; 20];
        tcp.read_exact(&mut resp).await?;
        let mut nonce = [0u8; 16];
        nonce.copy_from_slice(&resp[..16]);
        let status = u32::from_le_bytes(resp[16..20].try_into().expect("len 4"));
        Ok::<_, TransportError>(MiceHandshake {
            session_nonce: nonce,
            status,
        })
    };
    tokio::time::timeout(wall_timeout, fut)
        .await
        .map_err(|_| TransportError::Mice(format!("probe timed out after {wall_timeout:?}")))?
}

/// Serve a single MICE probe on an accepted TCP connection: read the
/// 4-byte client preamble, reply with our nonce + status. The sink
/// RTSP server listens on a *separate* socket (port 7236).
pub async fn serve_mice_probe(
    mut tcp: TcpStream,
    status: u32,
) -> Result<MiceHandshake, TransportError> {
    let mut client = [0u8; 4];
    tcp.read_exact(&mut client).await?;
    if client != [0u8; 4] {
        return Err(TransportError::Mice(format!(
            "unexpected preamble bytes {client:02x?}"
        )));
    }
    let mut nonce = [0u8; 16];
    rand::Rng::fill(&mut rand::thread_rng(), &mut nonce);
    let mut resp = [0u8; 20];
    resp[..16].copy_from_slice(&nonce);
    resp[16..20].copy_from_slice(&status.to_le_bytes());
    tcp.write_all(&resp).await?;
    tcp.flush().await?;
    Ok(MiceHandshake {
        session_nonce: nonce,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn probe_round_trip_on_loopback() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            serve_mice_probe(stream, 0).await.unwrap();
        });

        let handshake = mice_probe(addr, Duration::from_secs(2)).await.unwrap();
        server.await.unwrap();
        assert!(handshake.is_ok());
    }
}
