//! The body of `POST /play`. Senders use one of two formats:
//!
//! - a binary plist (iOS, macOS, most third-party apps):
//!   `{"Content-Location": url, "Start-Position": 0.25, "uuid": …}`;
//! - `text/parameters` lines (older iOS, some Android apps):
//!   `Content-Location: url\nStart-Position: 0.25\n`.

use plist::{Dictionary, Value};
use thiserror::Error;

/// URL scheme of the YouTube app's playlists, which a receiver has to
/// fetch through the sender (`unhandledURLRequest` events answered with
/// `/action`). Not supported.
const YOUTUBE_SCHEME: &str = "mlhls://";

/// Why a `/play` body cannot be played.
#[derive(Debug, Error, PartialEq, Eq)]
pub(in crate::receiver) enum PlayError {
    #[error("the /play body has no Content-Location")]
    NoLocation,
    #[error("the /play body is neither a plist nor text parameters")]
    Unreadable,
}

/// Where playback starts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::receiver) enum StartPosition {
    /// A fraction of the duration (`Start-Position`), in `0.0..=1.0`.
    Fraction(f64),
    /// Seconds from the start (`Start-Position-Seconds`).
    Seconds(f64),
}

impl StartPosition {
    /// The position in microseconds, given the duration a fraction refers
    /// to; `None` while that duration is unknown.
    pub fn resolve_us(self, duration_us: Option<u64>) -> Option<u64> {
        let secs_to_us = |s: f64| {
            // Clamped to a finite, non-negative value first.
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "clamped to 0..=u64::MAX microseconds"
            )]
            let us = (s.max(0.0) * 1_000_000.0).min(u64::MAX as f64) as u64;
            us
        };
        match self {
            Self::Seconds(s) => Some(secs_to_us(s)),
            Self::Fraction(f) => {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "a media duration fits an f64 mantissa"
                )]
                let duration = duration_us? as f64 / 1_000_000.0;
                Some(secs_to_us(f.clamp(0.0, 1.0) * duration))
            }
        }
    }

    /// Whether playback starts somewhere other than the beginning.
    pub fn is_beginning(self) -> bool {
        match self {
            Self::Fraction(v) | Self::Seconds(v) => v <= 0.0 || !v.is_finite(),
        }
    }
}

/// A parsed `POST /play`.
#[derive(Debug, Clone, PartialEq)]
pub(in crate::receiver) struct PlayRequest {
    pub url: String,
    pub start: StartPosition,
    /// The sender's id for this playback (`uuid` or `playbackSessionId`),
    /// echoed in events.
    pub playback_id: Option<String>,
    /// Name of the app that sent it (`clientProcName`), for logs.
    pub client: Option<String>,
}

impl PlayRequest {
    /// Why the receiver cannot play this URL by itself, if so.
    pub fn unsupported_reason(&self) -> Option<&'static str> {
        let url = self.url.to_ascii_lowercase();
        if url.starts_with(YOUTUBE_SCHEME) {
            Some("YouTube playlists fetched through the sender (mlhls://) are not supported")
        } else if !(url.starts_with("http://") || url.starts_with("https://")) {
            Some("only http(s) media URLs can be played")
        } else {
            None
        }
    }

    /// Parses the body in whichever format it comes; the content type is
    /// only a hint, as senders get it wrong.
    pub fn parse(body: &[u8]) -> Result<Self, PlayError> {
        if body.starts_with(b"bplist") || body.starts_with(b"<?xml") {
            let dict = plist::from_bytes::<Value>(body)
                .ok()
                .and_then(Value::into_dictionary)
                .ok_or(PlayError::Unreadable)?;
            return Self::from_plist(&dict);
        }
        let text = std::str::from_utf8(body).map_err(|_| PlayError::Unreadable)?;
        Self::from_text(text)
    }

    fn from_plist(dict: &Dictionary) -> Result<Self, PlayError> {
        let string = |key: &str| {
            dict.get(key)
                .and_then(Value::as_string)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
        };
        let real = |key: &str| {
            dict.get(key).and_then(|v| {
                v.as_real()
                    .or_else(|| v.as_signed_integer().map(|i| i as f64))
                    .or_else(|| v.as_string().and_then(|s| s.trim().parse().ok()))
            })
        };
        let url = string("Content-Location").ok_or(PlayError::NoLocation)?;
        let start = match (real("Start-Position-Seconds"), real("Start-Position")) {
            (Some(secs), _) => StartPosition::Seconds(secs),
            (None, Some(fraction)) => StartPosition::Fraction(fraction),
            (None, None) => StartPosition::Seconds(0.0),
        };
        Ok(Self {
            url,
            start,
            playback_id: string("uuid").or_else(|| string("playbackSessionId")),
            client: string("clientProcName").or_else(|| string("clientBundleID")),
        })
    }

    fn from_text(text: &str) -> Result<Self, PlayError> {
        let mut url = None;
        let mut fraction = None;
        let mut seconds = None;
        let mut playback_id = None;
        for line in text.lines() {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let value = value.trim();
            match key.trim().to_ascii_lowercase().as_str() {
                "content-location" if !value.is_empty() => url = Some(value.to_owned()),
                "start-position" => fraction = value.parse::<f64>().ok(),
                "start-position-seconds" => seconds = value.parse::<f64>().ok(),
                "uuid" | "playbacksessionid" if !value.is_empty() => {
                    playback_id = Some(value.to_owned());
                }
                _ => {}
            }
        }
        let start = match (seconds, fraction) {
            (Some(secs), _) => StartPosition::Seconds(secs),
            (None, Some(f)) => StartPosition::Fraction(f),
            (None, None) => StartPosition::Seconds(0.0),
        };
        Ok(Self {
            url: url.ok_or(PlayError::NoLocation)?,
            start,
            playback_id,
            client: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bplist(entries: &[(&str, Value)]) -> Vec<u8> {
        let dict: Dictionary = entries
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect();
        let mut out = Vec::new();
        plist::to_writer_binary(&mut out, &Value::Dictionary(dict)).unwrap();
        out
    }

    #[test]
    fn ios_binary_plist() {
        // What iOS Safari / Photos send (keys as seen from iOS 16).
        let body = bplist(&[
            (
                "Content-Location",
                Value::String("http://192.168.0.10:7001/master.m3u8".into()),
            ),
            ("Start-Position", Value::Real(0.25)),
            ("clientProcName", Value::String("MobileSafari".into())),
            (
                "uuid",
                Value::String("9F5C3E2A-0B61-4F6B-9C55-6D2D7A1A9E10".into()),
            ),
            ("rate", Value::Real(1.0)),
            ("volume", Value::Real(1.0)),
        ]);
        let play = PlayRequest::parse(&body).unwrap();
        assert_eq!(play.url, "http://192.168.0.10:7001/master.m3u8");
        assert_eq!(play.start, StartPosition::Fraction(0.25));
        assert_eq!(
            play.playback_id.as_deref(),
            Some("9F5C3E2A-0B61-4F6B-9C55-6D2D7A1A9E10")
        );
        assert_eq!(play.client.as_deref(), Some("MobileSafari"));
        assert_eq!(play.start.resolve_us(Some(20_000_000)), Some(5_000_000));
        assert_eq!(play.start.resolve_us(None), None);
    }

    #[test]
    fn youtube_style_plist_with_seconds_and_session_id() {
        let body = bplist(&[
            (
                "Content-Location",
                Value::String("mlhls://localhost/master.m3u8".into()),
            ),
            ("Start-Position-Seconds", Value::Real(12.5)),
            ("Start-Position", Value::Real(0.1)),
            ("playbackSessionId", Value::String("abc-123".into())),
        ]);
        let play = PlayRequest::parse(&body).unwrap();
        assert_eq!(play.start, StartPosition::Seconds(12.5));
        assert_eq!(play.start.resolve_us(None), Some(12_500_000));
        assert_eq!(play.playback_id.as_deref(), Some("abc-123"));
    }

    #[test]
    fn android_text_parameters() {
        // Third-party Android senders (AllCast-style) send CRLF text.
        let body =
            b"Content-Location: http://192.168.0.200:8080/video.mp4\r\nStart-Position: 0.0\r\n";
        let play = PlayRequest::parse(body).unwrap();
        assert_eq!(play.url, "http://192.168.0.200:8080/video.mp4");
        assert_eq!(play.start, StartPosition::Fraction(0.0));
        assert!(play.start.is_beginning());
        assert_eq!(play.playback_id, None);

        let body = b"Content-Location: http://h/a.m3u8\nStart-Position-Seconds: 30\n";
        let play = PlayRequest::parse(body).unwrap();
        assert_eq!(play.start, StartPosition::Seconds(30.0));
        assert!(!play.start.is_beginning());
    }

    #[test]
    fn xml_plist_and_integer_positions() {
        let body = br#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Content-Location</key><string>http://h/movie.mp4</string>
<key>Start-Position</key><integer>0</integer>
</dict></plist>"#;
        let play = PlayRequest::parse(body).unwrap();
        assert_eq!(play.url, "http://h/movie.mp4");
        assert_eq!(play.start, StartPosition::Fraction(0.0));
    }

    #[test]
    fn unsupported_urls_are_named() {
        let mut play = PlayRequest::parse(b"Content-Location: https://h/a.m3u8\n").unwrap();
        assert_eq!(play.unsupported_reason(), None);
        play.url = "mlhls://localhost/master.m3u8".into();
        assert!(play.unsupported_reason().unwrap().contains("YouTube"));
        play.url = "file:///etc/passwd".into();
        assert!(play.unsupported_reason().is_some());
    }

    #[test]
    fn rejects_bodies_without_a_url() {
        assert_eq!(
            PlayRequest::parse(b"Start-Position: 0.5\n"),
            Err(PlayError::NoLocation)
        );
        let body = bplist(&[("Start-Position", Value::Real(0.5))]);
        assert_eq!(PlayRequest::parse(&body), Err(PlayError::NoLocation));
        assert_eq!(
            PlayRequest::parse(b"bplist00garbage"),
            Err(PlayError::Unreadable)
        );
        assert_eq!(
            PlayRequest::parse(&[0xff, 0xfe]),
            Err(PlayError::Unreadable)
        );
    }
}
