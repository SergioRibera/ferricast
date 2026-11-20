//! HTTP layer of the HLS puller: bounded GETs that report the final
//! (post-redirect) URL and classify failures so the worker can tell
//! a retryable hiccup from a permanent error.

use std::collections::HashMap;
use std::time::Duration;

use bytes::{Bytes, BytesMut};
use ferricast_core::{FerricastError, Result};
use futures_util::StreamExt;
use reqwest::{Client, StatusCode};
use url::Url;

/// Largest playlist accepted. Real playlists are a few KiB; a long
/// VOD with thousands of segments stays well under this.
pub(super) const PLAYLIST_LIMIT: usize = 1 << 20;
/// Largest media segment accepted. A 10 s segment at 35 Mbit/s is
/// ~44 MiB.
pub(super) const SEGMENT_LIMIT: usize = 64 << 20;
/// AES-128 keys are 16 bytes; anything much larger is not a key.
pub(super) const KEY_LIMIT: usize = 1024;

/// A successful GET.
pub(super) struct Fetched {
    pub(super) body: Bytes,
    /// URL the body was served from after redirects. Relative URIs
    /// inside a playlist resolve against this, not the request URL.
    pub(super) url: Url,
}

#[derive(Debug)]
pub(super) enum FetchError {
    /// 404 / 410: the resource is gone (an expired live segment).
    NotFound,
    /// Network error, timeout, 5xx, 408 or 429: worth retrying.
    Transient(String),
    /// Anything else (403, oversized body, …): retrying won't help.
    Fatal(String),
}

impl FetchError {
    pub(super) fn into_pull_error(self, url: &Url) -> FerricastError {
        match self {
            FetchError::NotFound => FerricastError::Pull(format!("GET {url}: not found")),
            FetchError::Transient(msg) | FetchError::Fatal(msg) => {
                FerricastError::Pull(format!("GET {url}: {msg}"))
            }
        }
    }
}

/// Byte sub-range of a resource (`EXT-X-BYTERANGE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ByteRange {
    pub(super) offset: u64,
    pub(super) length: u64,
}

pub(super) struct Fetcher {
    client: Client,
}

impl Fetcher {
    pub(super) fn new(headers: &HashMap<String, String>) -> Result<Self> {
        let mut header_map = reqwest::header::HeaderMap::new();
        for (k, v) in headers {
            let name = reqwest::header::HeaderName::from_bytes(k.as_bytes())
                .map_err(|e| FerricastError::Pull(format!("invalid header name {k}: {e}")))?;
            let value = reqwest::header::HeaderValue::from_str(v)
                .map_err(|e| FerricastError::Pull(format!("invalid header value: {e}")))?;
            header_map.insert(name, value);
        }
        let client = Client::builder()
            .default_headers(header_map)
            .connect_timeout(Duration::from_secs(5))
            // Whole-request timeout. Long enough for a single 10s
            // segment to download even on a 1 Mbps link.
            .timeout(Duration::from_secs(30))
            // Accept self-signed TLS certs. The HLS sender in this
            // workspace (`HlsFrameSink` with `HlsConfig::tls`) uses a
            // per-session `rcgen` cert that no CA signs; a cast link
            // on the LAN has no identity to verify anyway. The TLS
            // handshake still runs, so the data stays encrypted.
            .danger_accept_invalid_certs(true)
            .build()
            .map_err(|e| FerricastError::Pull(format!("reqwest client: {e}")))?;
        Ok(Self { client })
    }

    /// GETs `url` (or a sub-range of it), refusing bodies over
    /// `limit` bytes.
    pub(super) async fn get(
        &self,
        url: &Url,
        range: Option<ByteRange>,
        limit: usize,
    ) -> std::result::Result<Fetched, FetchError> {
        let mut request = self.client.get(url.clone());
        if let Some(r) = range {
            let last = r.offset + r.length.saturating_sub(1);
            request = request.header(reqwest::header::RANGE, format!("bytes={}-{last}", r.offset));
        }
        let resp = request.send().await.map_err(classify_reqwest)?;
        let status = resp.status();
        match status {
            StatusCode::NOT_FOUND | StatusCode::GONE => return Err(FetchError::NotFound),
            StatusCode::REQUEST_TIMEOUT | StatusCode::TOO_MANY_REQUESTS => {
                return Err(FetchError::Transient(format!("HTTP {status}")));
            }
            s if s.is_server_error() => return Err(FetchError::Transient(format!("HTTP {s}"))),
            s if !s.is_success() => return Err(FetchError::Fatal(format!("HTTP {s}"))),
            _ => {}
        }
        if range.is_some() && status != StatusCode::PARTIAL_CONTENT {
            return Err(FetchError::Fatal(
                "server ignored the byte-range request".into(),
            ));
        }
        if resp.content_length().is_some_and(|len| len > limit as u64) {
            return Err(FetchError::Fatal(format!("body larger than {limit} bytes")));
        }
        let final_url = resp.url().clone();
        let mut stream = resp.bytes_stream();
        let mut buf = BytesMut::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(classify_reqwest)?;
            if buf.len() + chunk.len() > limit {
                return Err(FetchError::Fatal(format!("body larger than {limit} bytes")));
            }
            buf.extend_from_slice(&chunk);
        }
        Ok(Fetched {
            body: buf.freeze(),
            url: final_url,
        })
    }
}

fn classify_reqwest(e: reqwest::Error) -> FetchError {
    if e.is_builder() || e.is_redirect() {
        FetchError::Fatal(e.to_string())
    } else {
        FetchError::Transient(e.to_string())
    }
}
