//! A blocking [`Read`] + [`Seek`] view of a remote file, for the
//! demuxer to run on over HTTP.
//!
//! Servers that honour `Range` are read in fixed blocks, fetched on
//! demand and kept in a small LRU cache, so a demuxer can jump to an MP4
//! `moov` at the end of the file, or between the audio and video of a
//! loosely interleaved one, without downloading what it skips. Servers
//! that ignore `Range` are downloaded front to back into an anonymous
//! temporary file, and reads wait for the bytes they need.
//!
//! Every request runs on a small runtime of its own, on a dedicated
//! thread ([`HttpRuntime`]), so the source never borrows the caller's
//! runtime: it behaves the same under a multi-thread runtime, a
//! current-thread runtime or none at all. Opening and reading block the
//! calling thread until the bytes arrive, so the source belongs on a
//! thread that is not running async code.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::future::Future;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex, PoisonError, mpsc as std_mpsc};
use std::time::Duration;

use bytes::Bytes;
use ferricast_core::{FerricastError, Result};
use reqwest::header::{CONTENT_RANGE, HeaderMap, HeaderName, HeaderValue, RANGE};
use reqwest::{Client, StatusCode};
use tokio::sync::mpsc;

/// Bytes fetched per range request.
const BLOCK: u64 = 1 << 20;
/// Blocks kept in memory.
const CACHED_BLOCKS: usize = 16;
/// Attempts per block before a read fails.
const FETCH_ATTEMPTS: u32 = 3;
const RETRY_BACKOFF: Duration = Duration::from_millis(300);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Covers one block on a slow link.
const BLOCK_TIMEOUT: Duration = Duration::from_secs(30);
/// Longest silence tolerated while downloading without `Range`.
const READ_TIMEOUT: Duration = Duration::from_secs(30);
/// Largest file downloaded from a server without `Range` support. The
/// whole file lands on disk, so this bounds the temporary file.
pub(super) const MAX_DOWNLOAD: u64 = 4 << 30;

/// Work queued for the HTTP thread.
type Job = Pin<Box<dyn Future<Output = ()> + Send>>;

/// A current-thread runtime on a dedicated thread that runs the HTTP
/// requests of one source. The thread exits once every handle is gone.
#[derive(Clone)]
struct HttpRuntime {
    jobs: mpsc::UnboundedSender<Job>,
}

impl HttpRuntime {
    fn start() -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| FerricastError::Pull(format!("HTTP runtime: {e}")))?;
        let (jobs, mut queue) = mpsc::unbounded_channel::<Job>();
        std::thread::Builder::new()
            .name("ferricast-http".into())
            .spawn(move || {
                runtime.block_on(async move {
                    while let Some(job) = queue.recv().await {
                        tokio::spawn(job);
                    }
                });
            })
            .map_err(|e| FerricastError::Pull(format!("spawn HTTP thread: {e}")))?;
        Ok(Self { jobs })
    }

    /// Runs `future` on the HTTP thread and blocks until it finishes.
    fn block_on<T, F>(&self, future: F) -> io::Result<T>
    where
        T: Send + 'static,
        F: Future<Output = T> + Send + 'static,
    {
        let (done_tx, done_rx) = std_mpsc::sync_channel(1);
        self.spawn(async move {
            let _ = done_tx.send(future.await);
        })?;
        done_rx
            .recv()
            .map_err(|_| io::Error::other("the HTTP thread stopped"))
    }

    /// Starts `future` on the HTTP thread without waiting for it.
    fn spawn(&self, future: impl Future<Output = ()> + Send + 'static) -> io::Result<()> {
        self.jobs
            .send(Box::pin(future))
            .map_err(|_| io::Error::other("the HTTP thread stopped"))
    }
}

/// A remote file, readable and seekable.
pub(super) struct HttpSource {
    position: u64,
    len: Option<u64>,
    backend: Backend,
}

enum Backend {
    Ranged(Ranged),
    Download {
        download: Arc<Download>,
        /// Kept alive for the download task running on it.
        _runtime: HttpRuntime,
    },
}

/// Blocks of a file served with `Range` support.
struct Ranged {
    runtime: HttpRuntime,
    client: Client,
    url: String,
    len: u64,
    blocks: HashMap<u64, Bytes>,
    /// Block indices, least recently used first.
    recency: VecDeque<u64>,
}

/// How the server answered the first request.
enum Probe {
    /// `206`, with the first block.
    Ranged { len: u64, first: Bytes },
    /// `200`: the whole file follows in `response`.
    Whole {
        len: Option<u64>,
        response: Box<reqwest::Response>,
    },
}

impl Probe {
    async fn run(client: Client, url: String) -> Result<Self> {
        let response = client
            .get(&url)
            .header(RANGE, range_header(0, BLOCK - 1))
            .timeout(BLOCK_TIMEOUT)
            .send()
            .await
            .map_err(|e| pull_error(&url, &e))?;
        match response.status() {
            StatusCode::PARTIAL_CONTENT => {
                let (start, len) = content_range(response.headers()).ok_or_else(|| {
                    FerricastError::Pull(format!("GET {url}: 206 without a usable Content-Range"))
                })?;
                if start != 0 {
                    return Err(FerricastError::Pull(format!(
                        "GET {url}: asked for byte 0, got a range starting at {start}"
                    )));
                }
                let first = response.bytes().await.map_err(|e| pull_error(&url, &e))?;
                Ok(Self::Ranged { len, first })
            }
            StatusCode::OK => {
                let len = response.content_length();
                if len.is_some_and(|len| len > MAX_DOWNLOAD) {
                    return Err(FerricastError::Pull(format!(
                        "GET {url}: the server does not support range requests and the file \
                         is larger than {} MiB",
                        MAX_DOWNLOAD >> 20
                    )));
                }
                Ok(Self::Whole {
                    len,
                    response: Box::new(response),
                })
            }
            status => Err(FerricastError::Pull(format!("GET {url}: HTTP {status}"))),
        }
    }
}

impl HttpSource {
    /// Probes `url` with a request for its first block and picks ranged
    /// reads or a full download depending on how the server answers.
    /// Blocks the calling thread.
    pub(super) fn open(url: &str, headers: &HashMap<String, String>) -> Result<Self> {
        let client = client(headers)?;
        let runtime = HttpRuntime::start()?;
        let probe = runtime.block_on(Probe::run(client.clone(), url.to_string()))??;
        match probe {
            Probe::Ranged { len, first } => {
                let mut ranged = Ranged {
                    runtime,
                    client,
                    url: url.to_string(),
                    len,
                    blocks: HashMap::new(),
                    recency: VecDeque::new(),
                };
                ranged.insert(0, first);
                Ok(Self {
                    position: 0,
                    len: Some(len),
                    backend: Backend::Ranged(ranged),
                })
            }
            Probe::Whole { len, response } => {
                tracing::info!(
                    %url,
                    ?len,
                    "progressive: server ignores Range; downloading the whole file"
                );
                let download = Download::start(&runtime, url.to_string(), *response)?;
                Ok(Self {
                    position: 0,
                    len,
                    backend: Backend::Download {
                        download,
                        _runtime: runtime,
                    },
                })
            }
        }
    }

    /// Whether reads are served by range requests.
    pub(super) fn is_ranged(&self) -> bool {
        matches!(self.backend, Backend::Ranged(_))
    }
}

impl Read for HttpSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let n = match &mut self.backend {
            Backend::Ranged(ranged) => ranged.read_at(self.position, buf)?,
            Backend::Download { download, .. } => download.read_at(self.position, buf)?,
        };
        self.position += n as u64;
        Ok(n)
    }
}

impl Seek for HttpSource {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let target = match pos {
            SeekFrom::Start(at) => Some(at),
            SeekFrom::Current(delta) => self.position.checked_add_signed(delta),
            SeekFrom::End(delta) => {
                let len = match (&self.backend, self.len) {
                    (_, Some(len)) => len,
                    // Unknown length: only the finished download tells.
                    (Backend::Download { download, .. }, None) => download.wait_done()?,
                    (Backend::Ranged(ranged), None) => ranged.len,
                };
                self.len = Some(len);
                len.checked_add_signed(delta)
            }
        };
        self.position = target.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek before the start of the file",
            )
        })?;
        Ok(self.position)
    }
}

impl Ranged {
    fn read_at(&mut self, at: u64, buf: &mut [u8]) -> io::Result<usize> {
        if at >= self.len {
            return Ok(0);
        }
        let index = at / BLOCK;
        let block = self.block(index)?;
        let offset = usize::try_from(at - index * BLOCK).unwrap_or(usize::MAX);
        let available = block.get(offset..).unwrap_or_default();
        let n = available.len().min(buf.len());
        buf[..n].copy_from_slice(&available[..n]);
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("block {index} of {} is shorter than expected", self.url),
            ));
        }
        Ok(n)
    }

    fn block(&mut self, index: u64) -> io::Result<Bytes> {
        if let Some(block) = self.blocks.get(&index).cloned() {
            self.recency.retain(|&i| i != index);
            self.recency.push_back(index);
            return Ok(block);
        }
        let start = index * BLOCK;
        let end = (start + BLOCK).min(self.len) - 1;
        let mut last_error = String::new();
        for attempt in 0..FETCH_ATTEMPTS {
            if attempt > 0 {
                std::thread::sleep(RETRY_BACKOFF * attempt);
            }
            let request = RangeRequest {
                client: self.client.clone(),
                url: self.url.clone(),
                start,
                end,
            };
            match self.runtime.block_on(request.fetch())? {
                Ok(block) => {
                    self.insert(index, block.clone());
                    return Ok(block);
                }
                Err(FetchError::Fatal(message)) => return Err(io::Error::other(message)),
                Err(FetchError::Transient(message)) => {
                    tracing::debug!(url = %self.url, start, %message, "progressive: range fetch failed; retrying");
                    last_error = message;
                }
            }
        }
        Err(io::Error::other(last_error))
    }

    fn insert(&mut self, index: u64, block: Bytes) {
        if self.blocks.len() >= CACHED_BLOCKS
            && let Some(evicted) = self.recency.pop_front()
        {
            self.blocks.remove(&evicted);
        }
        self.blocks.insert(index, block);
        self.recency.push_back(index);
    }
}

/// One range request, owned so it can run on the HTTP thread.
struct RangeRequest {
    client: Client,
    url: String,
    start: u64,
    end: u64,
}

impl RangeRequest {
    async fn fetch(self) -> std::result::Result<Bytes, FetchError> {
        let Self {
            client,
            url,
            start,
            end,
        } = self;
        let response = client
            .get(&url)
            .header(RANGE, range_header(start, end))
            .timeout(BLOCK_TIMEOUT)
            .send()
            .await
            .map_err(|e| FetchError::Transient(format!("GET {url}: {e}")))?;
        let status = response.status();
        if status != StatusCode::PARTIAL_CONTENT {
            let message = format!(
                "GET {url} bytes {start}-{end}: HTTP {status} instead of a partial response"
            );
            return Err(if status.is_server_error() {
                FetchError::Transient(message)
            } else {
                FetchError::Fatal(message)
            });
        }
        if content_range(response.headers()).map(|(s, _)| s) != Some(start) {
            return Err(FetchError::Fatal(format!(
                "GET {url} bytes {start}-{end}: the server answered a different range"
            )));
        }
        let body = response
            .bytes()
            .await
            .map_err(|e| FetchError::Transient(format!("GET {url}: {e}")))?;
        if (body.len() as u64) < end - start + 1 {
            return Err(FetchError::Transient(format!(
                "GET {url} bytes {start}-{end}: short body of {} bytes",
                body.len()
            )));
        }
        Ok(body)
    }
}

enum FetchError {
    Transient(String),
    Fatal(String),
}

/// A whole-file download into an anonymous temporary file, fed by a
/// background task.
struct Download {
    file: File,
    state: Mutex<DownloadState>,
    progress: Condvar,
}

#[derive(Default)]
struct DownloadState {
    written: u64,
    done: bool,
    error: Option<String>,
}

impl Download {
    fn start(runtime: &HttpRuntime, url: String, response: reqwest::Response) -> Result<Arc<Self>> {
        let file = tempfile::tempfile()?;
        let writer = file.try_clone()?;
        let download = Arc::new(Self {
            file,
            state: Mutex::new(DownloadState::default()),
            progress: Condvar::new(),
        });
        let task = download.clone();
        runtime.spawn(async move {
            let result = task.fill(writer, response).await;
            let mut state = task.state.lock().unwrap_or_else(PoisonError::into_inner);
            state.done = true;
            if let Err(message) = result {
                tracing::warn!(%url, %message, "progressive: download failed");
                state.error = Some(message);
            }
            task.progress.notify_all();
        })?;
        Ok(download)
    }

    async fn fill(
        &self,
        mut writer: File,
        mut response: reqwest::Response,
    ) -> std::result::Result<(), String> {
        let mut written = 0u64;
        while let Some(chunk) = response.chunk().await.map_err(|e| e.to_string())? {
            written += chunk.len() as u64;
            if written > MAX_DOWNLOAD {
                return Err(format!(
                    "the server does not support range requests and the file is larger than {} MiB",
                    MAX_DOWNLOAD >> 20
                ));
            }
            writer.write_all(&chunk).map_err(|e| e.to_string())?;
            self.state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .written = written;
            self.progress.notify_all();
        }
        Ok(())
    }

    /// Waits until byte `at` is downloaded (or the download ended), then
    /// reads what is there.
    fn read_at(&self, at: u64, buf: &mut [u8]) -> io::Result<usize> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let state = self
            .progress
            .wait_while(state, |s| s.written <= at && !s.done)
            .unwrap_or_else(PoisonError::into_inner);
        if state.written <= at {
            return match &state.error {
                Some(message) => Err(io::Error::other(message.clone())),
                None => Ok(0),
            };
        }
        let available = usize::try_from(state.written - at).unwrap_or(usize::MAX);
        drop(state);
        let len = buf.len().min(available);
        self.file.read_at(&mut buf[..len], at)
    }

    /// Waits for the download to finish and returns the file length.
    fn wait_done(&self) -> io::Result<u64> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let state = self
            .progress
            .wait_while(state, |s| !s.done)
            .unwrap_or_else(PoisonError::into_inner);
        match &state.error {
            Some(message) => Err(io::Error::other(message.clone())),
            None => Ok(state.written),
        }
    }
}

fn client(headers: &HashMap<String, String>) -> Result<Client> {
    let mut map = HeaderMap::new();
    for (name, value) in headers {
        let name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|e| FerricastError::Pull(format!("invalid header name {name}: {e}")))?;
        let value = HeaderValue::from_str(value)
            .map_err(|e| FerricastError::Pull(format!("invalid header value: {e}")))?;
        map.insert(name, value);
    }
    Client::builder()
        .default_headers(map)
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_TIMEOUT)
        .build()
        .map_err(|e| FerricastError::Pull(format!("HTTP client: {e}")))
}

fn range_header(start: u64, end: u64) -> String {
    format!("bytes={start}-{end}")
}

/// `(first byte, total length)` of a `Content-Range: bytes a-b/total`.
fn content_range(headers: &HeaderMap) -> Option<(u64, u64)> {
    let value = headers.get(CONTENT_RANGE)?.to_str().ok()?;
    let (range, total) = value.strip_prefix("bytes ")?.split_once('/')?;
    let (start, _) = range.split_once('-')?;
    Some((start.trim().parse().ok()?, total.trim().parse().ok()?))
}

fn pull_error(url: &str, error: &reqwest::Error) -> FerricastError {
    FerricastError::Pull(format!("GET {url}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_ranges_are_parsed() {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_RANGE,
            HeaderValue::from_static("bytes 1048576-2097151/5000000"),
        );
        assert_eq!(content_range(&headers), Some((1_048_576, 5_000_000)));
        headers.insert(CONTENT_RANGE, HeaderValue::from_static("bytes 0-9/*"));
        assert_eq!(content_range(&headers), None);
    }
}
