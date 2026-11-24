//! `ProgressivePuller` over a local HTTP server, on a current-thread
//! runtime: the HTTP reader must never need the caller's runtime.

#![cfg(feature = "http")]

use std::path::Path;
use std::time::Duration;

use ferricast_core::{MediaPacket, MediaPuller, PullSpec};
use ferricast_demuxer::ProgressivePuller;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Serves `body` for every path, honouring `Range: bytes=a-b`.
async fn serve(body: Vec<u8>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let body = body.clone();
            tokio::spawn(async move {
                let mut head = Vec::new();
                let mut chunk = [0u8; 2048];
                while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                    match sock.read(&mut chunk).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => head.extend_from_slice(&chunk[..n]),
                    }
                }
                let head = String::from_utf8_lossy(&head).to_string();
                let range = head
                    .lines()
                    .filter_map(|l| l.split_once(':'))
                    .find(|(k, _)| k.eq_ignore_ascii_case("range"))
                    .and_then(|(_, v)| v.trim().strip_prefix("bytes=")?.split_once('-'))
                    .and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?)));
                let out = if let Some((start, end)) = range {
                    let end = end.min(body.len() - 1);
                    let mut r = format!(
                        "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\n\
                         Content-Range: bytes {start}-{end}/{}\r\nConnection: close\r\n\r\n",
                        end + 1 - start,
                        body.len()
                    )
                    .into_bytes();
                    r.extend_from_slice(&body[start..=end]);
                    r
                } else {
                    let mut r = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    )
                    .into_bytes();
                    r.extend_from_slice(&body);
                    r
                };
                let _ = sock.write_all(&out).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    format!("http://{addr}/clip.mp4")
}

#[tokio::test(flavor = "current_thread")]
async fn plays_and_seeks_on_a_current_thread_runtime() {
    let body = std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/h264.mp4"))
        .unwrap();
    let url = serve(body).await;
    let mut puller = ProgressivePuller::default();
    let run = async {
        let info = puller
            .open(PullSpec {
                url,
                headers: std::collections::HashMap::default(),
            })
            .await
            .unwrap();
        assert!(info.video.is_some());
        let mut frames = 0;
        loop {
            match puller.next().await.unwrap() {
                MediaPacket::Video(_) => frames += 1,
                MediaPacket::Eos => break,
                _ => {}
            }
        }
        assert!(frames > 0);
        puller.seek(0).await.unwrap();
        let mut again = 0;
        loop {
            match puller.next().await.unwrap() {
                MediaPacket::Video(_) => again += 1,
                MediaPacket::Eos => break,
                _ => {}
            }
        }
        assert_eq!(again, frames);
        puller.close().await.unwrap();
    };
    tokio::time::timeout(Duration::from_secs(20), run)
        .await
        .expect("the puller hung on a current-thread runtime");
}
