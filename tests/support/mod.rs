//! Test support shared by the receiver integration tests: a tiny
//! HTTP/1.1 file server with byte-range support and helpers that
//! generate HLS fixtures with ffmpeg.
//!
//! ffmpeg is looked up in `FERRICAST_FFMPEG`, then on `PATH`. Tests
//! that need it skip (with a message) when it is missing.

#![allow(dead_code, reason = "each test binary uses a different subset")]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// What the handler answers for one request path.
pub enum Reply {
    Ok(Vec<u8>),
    /// The whole body with 200, whatever `Range` asked for (a server
    /// without range support).
    OkIgnoringRange(Vec<u8>),
    Status(u16),
    Redirect(String),
}

pub type Handler = Arc<dyn Fn(&str) -> Reply + Send + Sync>;

pub struct TestServer {
    pub addr: SocketAddr,
    /// Every request path, in arrival order.
    pub log: Arc<Mutex<Vec<String>>>,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl TestServer {
    pub async fn start(handler: Handler) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let log = Arc::new(Mutex::new(Vec::new()));
        let log_task = log.clone();
        let task = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let handler = handler.clone();
                let log = log_task.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 4096];
                    while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
                        match sock.read(&mut chunk).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => buf.extend_from_slice(&chunk[..n]),
                        }
                    }
                    let head = String::from_utf8_lossy(&buf).to_string();
                    let mut lines = head.lines();
                    let path = lines
                        .next()
                        .and_then(|l| l.split_whitespace().nth(1))
                        .unwrap_or("/")
                        .to_string();
                    let range = lines
                        .filter_map(|l| l.split_once(':'))
                        .find(|(k, _)| k.eq_ignore_ascii_case("range"))
                        .map(|(_, v)| v.trim().to_string());
                    log.lock().unwrap().push(path.clone());
                    let response = match handler(&path) {
                        Reply::Ok(body) => {
                            if let Some((start, end)) = range.as_deref().and_then(parse_range) {
                                let end = end.min(body.len().saturating_sub(1));
                                let part = &body[start.min(body.len())..=end];
                                let mut r = format!(
                                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\n\
                                 Content-Range: bytes {start}-{end}/{}\r\n\
                                 Connection: close\r\n\r\n",
                                    part.len(),
                                    body.len()
                                )
                                .into_bytes();
                                r.extend_from_slice(part);
                                r
                            } else {
                                let mut r = format!(
                                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\
                                 Connection: close\r\n\r\n",
                                    body.len()
                                )
                                .into_bytes();
                                r.extend_from_slice(&body);
                                r
                            }
                        }
                        Reply::OkIgnoringRange(body) => {
                            let mut r = format!(
                                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\
                                 Connection: close\r\n\r\n",
                                body.len()
                            )
                            .into_bytes();
                            r.extend_from_slice(&body);
                            r
                        }
                        Reply::Status(code) => format!(
                            "HTTP/1.1 {code} Status\r\nContent-Length: 0\r\n\
                             Connection: close\r\n\r\n"
                        )
                        .into_bytes(),
                        Reply::Redirect(to) => format!(
                            "HTTP/1.1 302 Found\r\nLocation: {to}\r\nContent-Length: 0\r\n\
                             Connection: close\r\n\r\n"
                        )
                        .into_bytes(),
                    };
                    let _ = sock.write_all(&response).await;
                    let _ = sock.shutdown().await;
                });
            }
        });
        Self { addr, log, task }
    }

    pub fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }
}

fn parse_range(v: &str) -> Option<(usize, usize)> {
    let spec = v.strip_prefix("bytes=")?;
    let (a, b) = spec.split_once('-')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

/// Serves every file of `dir` under `/<name>`.
pub fn dir_handler(dir: &Path) -> Handler {
    let files: HashMap<String, Vec<u8>> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(std::result::Result::ok)
        .map(|e| {
            (
                format!("/{}", e.file_name().to_string_lossy()),
                std::fs::read(e.path()).unwrap(),
            )
        })
        .collect();
    Arc::new(move |path| match files.get(path) {
        Some(body) => Reply::Ok(body.clone()),
        None => Reply::Status(404),
    })
}

pub fn ffmpeg() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("FERRICAST_FFMPEG") {
        return Some(PathBuf::from(p));
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|d| d.join("ffmpeg"))
            .find(|p| p.is_file())
    })
}

/// Generates an H.264 (with B-frames) + AAC HLS stream of `seconds`
/// into `dir` as `index.m3u8`, segments of `segment_secs`. `extra`
/// goes before the output path (e.g. HLS flags). Returns `false`
/// when ffmpeg is unavailable.
pub fn generate_hls(dir: &Path, seconds: u32, segment_secs: u32, extra: &[&str]) -> bool {
    let Some(ffmpeg) = ffmpeg() else {
        eprintln!("ffmpeg not found (set FERRICAST_FFMPEG) — skipping");
        return false;
    };
    let fps = 30;
    let gop = (fps * segment_secs).to_string();
    let duration = seconds.to_string();
    let hls_time = segment_secs.to_string();
    let mut cmd = Command::new(ffmpeg);
    cmd.env_remove("LD_LIBRARY_PATH")
        .current_dir(dir)
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30"])
        .args(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"])
        .args(["-t", &duration])
        .args(["-c:v", "libx264", "-profile:v", "high", "-bf", "2"])
        .args(["-g", &gop, "-keyint_min", &gop, "-sc_threshold", "0"])
        .args(["-pix_fmt", "yuv420p"])
        .args(["-c:a", "aac", "-ac", "2", "-b:a", "96k"])
        .args([
            "-f",
            "hls",
            "-hls_time",
            &hls_time,
            "-hls_playlist_type",
            "vod",
        ])
        .args(extra)
        .arg("index.m3u8");
    let status = cmd.status().expect("spawn ffmpeg");
    assert!(status.success(), "ffmpeg failed: {status}");
    true
}

/// Parses `index.m3u8` written by [`generate_hls`] into
/// `(duration, segment file name)` pairs.
pub fn segments(dir: &Path) -> Vec<(f32, String)> {
    let text = std::fs::read_to_string(dir.join("index.m3u8")).unwrap();
    let mut out = Vec::new();
    let mut duration = None;
    for line in text.lines() {
        if let Some(d) = line.strip_prefix("#EXTINF:") {
            duration = d
                .trim_end_matches(',')
                .split(',')
                .next()
                .and_then(|d| d.parse().ok());
        } else if !line.starts_with('#') && !line.is_empty() {
            out.push((duration.take().unwrap_or(0.0), line.to_string()));
        }
    }
    out
}

/// A live source simulated from VOD segments: a sliding window of
/// `WINDOW` segments that advances by one segment per wall-clock
/// second. `restart_at` restarts the media sequence.
pub struct LiveSim {
    segments: Vec<(f32, String)>,
    started: Instant,
    restart_at: Option<Duration>,
}

pub const WINDOW: usize = 6;

impl LiveSim {
    pub fn playlist(&self) -> String {
        let elapsed = self.started.elapsed();
        let elapsed = match self.restart_at {
            Some(at) if elapsed >= at => elapsed.checked_sub(at).unwrap(),
            _ => elapsed,
        };
        let head = usize::try_from(elapsed.as_secs())
            .unwrap()
            .min(self.segments.len() - WINDOW);
        let mut text = format!(
            "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:1\n#EXT-X-MEDIA-SEQUENCE:{head}\n"
        );
        for (d, name) in &self.segments[head..head + WINDOW] {
            writeln!(text, "#EXTINF:{d:.3},\n{name}").unwrap();
        }
        text
    }
}

pub async fn live_server(restart_at: Option<Duration>) -> Option<(tempfile::TempDir, TestServer)> {
    let dir = tempfile::tempdir().unwrap();
    if !generate_hls(dir.path(), 24, 1, &[]) {
        return None;
    }
    let sim = Arc::new(Mutex::new(LiveSim {
        segments: segments(dir.path()),
        started: Instant::now(),
        restart_at,
    }));
    let files = dir_handler(dir.path());
    let server = TestServer::start(Arc::new(move |path: &str| {
        if path == "/live.m3u8" {
            Reply::Ok(sim.lock().unwrap().playlist().into_bytes())
        } else {
            files(path)
        }
    }))
    .await;
    Some((dir, server))
}

/// Encodes a `seconds`-long test clip (320x240 at 30 fps, a keyframe
/// every second, 48 kHz stereo audio) into `dir/name`; the container
/// follows the extension. `codecs` are the ffmpeg codec arguments and
/// `extra` goes before the output path. Returns `false` when ffmpeg is
/// unavailable.
pub fn generate_progressive(
    dir: &Path,
    name: &str,
    seconds: u32,
    codecs: &[&str],
    extra: &[&str],
) -> bool {
    let Some(ffmpeg) = ffmpeg() else {
        eprintln!("ffmpeg not found (set FERRICAST_FFMPEG) — skipping");
        return false;
    };
    let status = Command::new(ffmpeg)
        .env_remove("LD_LIBRARY_PATH")
        .current_dir(dir)
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30"])
        .args(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"])
        .args(["-t", &seconds.to_string()])
        .args([
            "-g",
            "30",
            "-keyint_min",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-ac",
            "2",
        ])
        .args(codecs)
        .args(extra)
        .arg(name)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "ffmpeg failed: {status}");
    true
}

pub const H264_AAC: &[&str] = &[
    "-c:v",
    "libx264",
    "-profile:v",
    "high",
    "-bf",
    "2",
    "-c:a",
    "aac",
    "-b:a",
    "96k",
];
pub const VP9_OPUS: &[&str] = &[
    "-c:v",
    "libvpx-vp9",
    "-deadline",
    "realtime",
    "-cpu-used",
    "8",
    "-b:v",
    "300k",
    "-c:a",
    "libopus",
    "-b:a",
    "64k",
];

/// Encodes `seconds` of a 440 Hz stereo tone at 44.1 kHz into `dir/name`
/// with the ffmpeg arguments `args`. Returns `false` when ffmpeg is
/// unavailable.
pub fn generate_audio(dir: &Path, name: &str, seconds: u32, args: &[&str]) -> bool {
    let Some(ffmpeg) = ffmpeg() else {
        eprintln!("ffmpeg not found (set FERRICAST_FFMPEG) — skipping");
        return false;
    };
    let status = Command::new(ffmpeg)
        .env_remove("LD_LIBRARY_PATH")
        .current_dir(dir)
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100"])
        .args(["-t", &seconds.to_string(), "-ac", "2"])
        .args(args)
        .arg(name)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "ffmpeg failed: {status}");
    true
}
