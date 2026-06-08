//! Receiver window — one top-level Freya window per incoming
//! transmission.
//!
//! Pattern mirrors [`crate::picker`]: each window is its own OS-level
//! top-level so users can still reach it when the main Ferricast
//! window is minimised. Window opens on
//! [`ferricast::ManagerEvent::ReceiverIncoming`]; closes on
//! [`ferricast::ManagerEvent::ReceiverStopped`] or
//! [`ferricast::ManagerEvent::ReceiverError`].
//!
//! UI shape today:
//!
//! - Audio-only (`MediaInfo::video.is_none()`): card view with the
//!   friendly metadata the sender supplied (title, artist, artwork
//!   URL when set) plus the sender's address and a stop button.
//! - Video: same card with a "Video: WIDTHxHEIGHT — codec" stamp.
//!   Actual frame rendering follows in a separate change — needs
//!   Freya `Canvas` (Skia surface) integration so we can blit a
//!   decoded BGRA buffer at fps without PNG-encoding every frame.
//!
//! [`LoggingSink`] is the placeholder sink the manager pipes decoded
//! frames into. Today it counts frames and forwards counters to the
//! window for a "received N video / M audio frames" indicator;
//! real audio playback (cpal → PipeWire) and Skia video rendering
//! are deferred.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ferricast::{
    AudioCodec, CapturedFrame, Codec, DecodedAudio, FerricastError, FrameSink, MediaInfo,
    RemoteSender, Result,
};
use freya::prelude::*;
use freya::winit::window::WindowLevel;

/// Frame counters shared between [`LoggingSink`] and the window's
/// reactive read side. Both writers (the pipeline) and the reader
/// (the window's render loop) are decoupled — atomic counters keep
/// the wiring trivial.
#[derive(Default, Debug)]
pub struct ReceiverCounters {
    pub video_frames: AtomicU64,
    pub audio_frames: AtomicU64,
}

/// Placeholder sink. Counts everything it receives so the window can
/// show progress. Video frames carry width/height; audio frames
/// carry sample-rate hints — both are dropped after counting.
///
/// When the audio output path (cpal → PipeWire) lands, push the
/// `DecodedAudio` PCM bytes into the cpal ring buffer here instead
/// of discarding. When the video render path (Freya Canvas + Skia
/// image upload) lands, blit `CapturedFrame::Cpu` into the surface
/// here instead of discarding.
pub struct LoggingSink {
    counters: Arc<ReceiverCounters>,
    label: String,
}

impl LoggingSink {
    pub fn new(label: impl Into<String>, counters: Arc<ReceiverCounters>) -> Self {
        Self {
            counters,
            label: label.into(),
        }
    }
}

#[async_trait::async_trait]
impl FrameSink for LoggingSink {
    async fn push_video(&mut self, frame: CapturedFrame) -> Result<()> {
        let n = self.counters.video_frames.fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_multiple_of(60) {
            tracing::info!(
                receiver = %self.label,
                count = n,
                width = frame.width(),
                height = frame.height(),
                "receiver sink: video frames received"
            );
        }
        Ok(())
    }

    async fn push_audio(&mut self, audio: DecodedAudio) -> Result<()> {
        let n = self.counters.audio_frames.fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_multiple_of(120) {
            tracing::info!(
                receiver = %self.label,
                count = n,
                sample_rate = audio.sample_rate,
                channels = audio.channels,
                "receiver sink: audio frames received"
            );
        }
        Ok(())
    }
}

/// Construct an error to surface when the sink_factory closure is
/// invoked before the GUI's reactive infrastructure is ready. Kept
/// as a free function so call sites read naturally.
pub fn err_not_ready() -> FerricastError {
    FerricastError::Receiver(
        "receiver sink_factory invoked before GUI was ready — \
         this is a startup ordering bug; report it"
            .into(),
    )
}

#[derive(Clone)]
pub struct ReceiverWindowApp {
    remote: RemoteSender,
    info: MediaInfo,
    counters: Arc<ReceiverCounters>,
}

impl ReceiverWindowApp {
    pub fn new(remote: RemoteSender, info: MediaInfo, counters: Arc<ReceiverCounters>) -> Self {
        Self {
            remote,
            info,
            counters,
        }
    }
}

impl App for ReceiverWindowApp {
    fn render(&self) -> impl IntoElement {
        let remote = self.remote.clone();
        let info = self.info.clone();
        let counters = self.counters.clone();

        // Counter snapshot at render time. Live tick (re-render
        // every second to reflect new frames) needs a Freya signal
        // wired through; tracked as follow-up alongside the proper
        // video render path.
        let video_n = counters.video_frames.load(Ordering::Relaxed);
        let audio_n = counters.audio_frames.load(Ordering::Relaxed);

        let title = receiver_title(&remote, &info);
        let codec_line = codec_line(&info);
        let video_label = match &info.video {
            Some(v) => format!(
                "Video: {} — {} frames decoded",
                codec_label_video(v.codec),
                video_n
            ),
            None => "No video".to_string(),
        };
        let audio_label = match &info.audio {
            Some(a) => format!(
                "Audio: {} {}Hz/{}ch — {} frames decoded",
                codec_label_audio(a.codec),
                a.sample_rate,
                a.channels,
                audio_n
            ),
            None => "No audio".to_string(),
        };

        rect()
            .expanded()
            .background((20, 20, 28))
            .vertical()
            .padding(24.)
            .spacing(12.)
            .child(
                label()
                    .text(title)
                    .font_size(20.)
                    .color((230, 230, 240)),
            )
            .child(
                label()
                    .text(codec_line)
                    .font_size(13.)
                    .color((180, 180, 200)),
            )
            .child(
                rect()
                    .vertical()
                    .spacing(6.)
                    .child(
                        label()
                            .text(video_label)
                            .font_size(13.)
                            .color((200, 200, 220)),
                    )
                    .child(
                        label()
                            .text(audio_label)
                            .font_size(13.)
                            .color((200, 200, 220)),
                    ),
            )
            .child(
                label()
                    .text(format!("from {}", remote.addr))
                    .font_size(12.)
                    .color((150, 150, 170)),
            )
    }
}

/// Launch a new top-level Freya window for an incoming receiver
/// session. The window persists until the caller closes it via
/// `Platform::close_window(window_id)` (typically on `ReceiverStopped`).
pub async fn open_receiver_window(
    platform: Platform,
    remote: RemoteSender,
    info: MediaInfo,
    counters: Arc<ReceiverCounters>,
) -> freya::winit::window::WindowId {
    let title = receiver_title(&remote, &info);
    let app = ReceiverWindowApp::new(remote, info, counters);
    let config = WindowConfig::new_app(app)
        // `WindowConfig::with_title` takes a `&'static str`, which
        // can't carry the dynamic sender name. The winit attribute
        // builder accepts an owned String, so we set the title there
        // and skip the static-only API. `with_app_id` keeps the
        // desktop-entry / wmclass stable across sessions.
        .with_size(480., 280.)
        .with_background((20, 20, 28))
        .with_app_id("rs.sergioribera.ferricast.Receiver")
        .with_window_attributes(move |attrs, _| {
            attrs
                .with_title(title.clone())
                .with_window_level(WindowLevel::Normal)
        });
    platform.launch_window(config).await
}

fn receiver_title(remote: &RemoteSender, info: &MediaInfo) -> String {
    if let Some(name) = &remote.name {
        format!("Cast from {name}")
    } else if info.audio.is_some() && info.video.is_none() {
        "Cast (audio)".to_string()
    } else {
        "Cast".to_string()
    }
}

fn codec_line(info: &MediaInfo) -> String {
    let live = if info.is_live { "live" } else { "VOD" };
    let dur = match info.duration_us {
        Some(us) => format!(", {:.1}s", us as f64 / 1_000_000.0),
        None => String::new(),
    };
    format!("{live}{dur}")
}

fn codec_label_video(c: Codec) -> &'static str {
    match c {
        Codec::H264 => "H.264",
        Codec::H265 => "H.265",
        Codec::Vp8 => "VP8",
        Codec::Vp9 => "VP9",
    }
}

fn codec_label_audio(c: AudioCodec) -> &'static str {
    match c {
        AudioCodec::Aac => "AAC",
        AudioCodec::Opus => "Opus",
        AudioCodec::Pcm => "PCM",
        AudioCodec::Alac => "ALAC",
    }
}
