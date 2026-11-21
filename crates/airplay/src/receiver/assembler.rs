//! Turns the separately set up mirroring and audio streams into one
//! [`PushedStream`] for the receiver pipeline.
//!
//! The pipeline needs to know up front which streams it decodes, but the
//! sender sets them up one after the other: the video stream's format is
//! only known from its first codec packet, and the audio stream may be
//! set up after video already flows. So the stream is announced once the
//! video format is known and audio is set up, or [`AUDIO_GRACE`] after the
//! video format when no audio comes. Audio-only sessions are announced on
//! their first packet. Media that arrives before the announcement is kept
//! (video from its last keyframe) and replayed into the stream, marked
//! decode-only up to the newest frame so playback starts live.
//!
//! If audio shows up after the announcement, or the video codec changes,
//! a new stream replaces the old one, again replaying video from the last
//! keyframe.

use std::collections::VecDeque;
use std::time::Duration;

use ferricast_core::{
    AudioFrame, AudioStreamInfo, Codec, EncodedFrame, MediaInfo, MediaPacket, PushError,
    PushSender, PushedStream, VideoStreamInfo,
};
use tokio::time::Instant;
use tracing::{debug, info, warn};

use super::audio::AudioFormat;

/// How long the stream waits for audio once the video format is known.
pub(super) const AUDIO_GRACE: Duration = Duration::from_millis(500);
/// Packets queued towards the pipeline.
const STREAM_CAPACITY: usize = 1024;
/// Video frames kept since the last keyframe, at most.
const MAX_GOP: usize = 600;
/// Audio frames kept before the announcement, at most (about 2 s).
const MAX_PENDING_AUDIO: usize = 200;
/// Nominal frame rate reported for mirroring; the real cadence follows
/// the screen's activity.
const NOMINAL_FPS: u32 = 60;

/// What the media receivers report.
#[derive(Debug)]
pub(super) enum MediaEvent {
    /// A codec packet with a new codec or picture size.
    VideoFormat {
        codec: Codec,
        width: u32,
        height: u32,
    },
    Video(EncodedFrame),
    Audio(AudioFrame),
}

/// Builds and feeds the pushed stream of one session.
#[derive(Default)]
pub(super) struct Assembler {
    /// A mirroring stream was set up.
    video_expected: bool,
    video: Option<VideoStreamInfo>,
    audio: Option<AudioStreamInfo>,
    /// When the video format became known (the audio grace starts).
    video_since: Option<Instant>,
    gop: Vec<EncodedFrame>,
    pending_audio: VecDeque<AudioFrame>,
    /// The announced stream and what it was announced with.
    out: Option<(PushSender, Option<Codec>, bool)>,
    dropped: u64,
}

impl Assembler {
    /// A mirroring stream was set up (`true`) or torn down.
    pub fn set_video_expected(&mut self, expected: bool) {
        self.video_expected = expected;
        if !expected {
            self.video = None;
            self.video_since = None;
            self.gop.clear();
        }
    }

    /// The audio stream was set up with `format`, or torn down (`None`).
    /// May announce a stream.
    pub fn set_audio(&mut self, format: Option<AudioFormat>) -> Option<PushedStream> {
        self.audio = format.map(|f| AudioStreamInfo {
            codec: f.codec,
            sample_rate: f.sample_rate,
            channels: f.channels,
            source: None,
        });
        if self.audio.is_none() {
            self.pending_audio.clear();
            return None;
        }
        self.maybe_announce()
    }

    /// When the audio grace ends, if the stream is waiting on it.
    pub fn deadline(&self) -> Option<Instant> {
        if self.out.is_some() || self.audio.is_some() {
            return None;
        }
        self.video_since.map(|t| t + AUDIO_GRACE)
    }

    /// The audio grace ended: announce without audio.
    pub fn on_deadline(&mut self) -> Option<PushedStream> {
        self.maybe_announce()
    }

    pub fn on_event(&mut self, event: MediaEvent) -> Option<PushedStream> {
        match event {
            MediaEvent::VideoFormat {
                codec,
                width,
                height,
            } => {
                self.video = Some(VideoStreamInfo {
                    codec,
                    width,
                    height,
                    fps: NOMINAL_FPS,
                });
                self.video_since.get_or_insert_with(Instant::now);
                self.maybe_announce()
            }
            MediaEvent::Video(frame) => {
                if frame.is_keyframe {
                    self.gop.clear();
                }
                if self.gop.len() < MAX_GOP {
                    self.gop.push(frame.clone());
                }
                self.push(MediaPacket::Video(frame));
                None
            }
            MediaEvent::Audio(frame) => {
                if self.out.is_none() {
                    if self.pending_audio.len() == MAX_PENDING_AUDIO {
                        self.pending_audio.pop_front();
                    }
                    self.pending_audio.push_back(frame);
                    return self.maybe_announce();
                }
                self.push(MediaPacket::Audio(frame));
                None
            }
        }
    }

    fn push(&mut self, packet: MediaPacket) {
        let Some((tx, ..)) = &self.out else {
            return;
        };
        match tx.try_push(packet) {
            Ok(()) => {}
            Err(PushError::Full) => {
                self.dropped += 1;
                if self.dropped.is_power_of_two() {
                    warn!(
                        dropped = self.dropped,
                        "receiver pipeline is behind; media dropped"
                    );
                }
            }
            Err(PushError::Closed) => {
                debug!("pipeline stopped consuming the pushed stream");
                self.out = None;
            }
        }
    }

    /// Whether what the stream would carry now differs from what was
    /// announced (or nothing was).
    fn wanted(&self) -> Option<(Option<Codec>, bool)> {
        let has_audio = self.audio.is_some();
        let ready = match (&self.video, self.video_expected) {
            (Some(_), _) => {
                has_audio || self.video_since.is_some_and(|t| t.elapsed() >= AUDIO_GRACE)
            }
            (None, true) => false,
            (None, false) => has_audio && !self.pending_audio.is_empty(),
        };
        if !ready {
            return None;
        }
        let shape = (self.video.as_ref().map(|v| v.codec), has_audio);
        match &self.out {
            Some((tx, codec, audio)) if !tx.is_closed() && (*codec, *audio) == shape => None,
            // Once announced, audio going away does not restart video.
            Some((tx, codec, true)) if !tx.is_closed() && *codec == shape.0 && !has_audio => None,
            _ => Some(shape),
        }
    }

    fn maybe_announce(&mut self) -> Option<PushedStream> {
        let (codec, has_audio) = self.wanted()?;
        let info = MediaInfo {
            video: self.video.clone(),
            audio: self.audio.clone(),
            duration_us: None,
            is_live: true,
        };
        info!(?info, "announcing the AirPlay stream");
        let (tx, stream) = PushedStream::channel(info, STREAM_CAPACITY);
        let live_from = self
            .gop
            .last()
            .map(|f| f.timestamp_us)
            .or_else(|| self.pending_audio.back().map(|a| a.timestamp_us));
        if let Some(timestamp_us) = live_from {
            let _ = tx.try_push(MediaPacket::PresentFrom { timestamp_us });
        }
        if codec.is_some() {
            for frame in &self.gop {
                let _ = tx.try_push(MediaPacket::Video(frame.clone()));
            }
        }
        for frame in self.pending_audio.drain(..) {
            let _ = tx.try_push(MediaPacket::Audio(frame));
        }
        self.out = Some((tx, codec, has_audio));
        Some(stream)
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use ferricast_core::{AudioCodec, MediaPuller as _, PullSpec};

    use super::*;

    fn video(ts: u64, key: bool) -> MediaEvent {
        MediaEvent::Video(EncodedFrame {
            codec: Codec::H264,
            data: Bytes::from_static(&[0, 0, 0, 1, 0x65]),
            timestamp_us: ts,
            is_keyframe: key,
            duration_us: None,
            pts_dts: (ts, ts),
        })
    }

    fn format() -> MediaEvent {
        MediaEvent::VideoFormat {
            codec: Codec::H264,
            width: 1280,
            height: 720,
        }
    }

    fn eld() -> AudioFormat {
        AudioFormat {
            codec: AudioCodec::AacEld,
            sample_rate: 44_100,
            channels: 2,
        }
    }

    fn spec() -> PullSpec {
        PullSpec {
            url: String::new(),
            headers: std::collections::HashMap::new(),
        }
    }

    #[tokio::test]
    async fn waits_for_audio_then_replays_from_the_keyframe() {
        let mut a = Assembler::default();
        a.set_video_expected(true);
        assert!(a.on_event(format()).is_none());
        assert!(a.on_event(video(1, true)).is_none());
        assert!(a.on_event(video(2, false)).is_none());
        let mut stream = a.set_audio(Some(eld())).unwrap();
        let info = stream.open(spec()).await.unwrap();
        assert!(info.video.is_some() && info.audio.is_some());
        assert!(matches!(
            stream.next().await.unwrap(),
            MediaPacket::PresentFrom { timestamp_us: 2 }
        ));
        assert!(
            matches!(stream.next().await.unwrap(), MediaPacket::Video(f) if f.timestamp_us == 1)
        );
        assert!(
            matches!(stream.next().await.unwrap(), MediaPacket::Video(f) if f.timestamp_us == 2)
        );
        a.on_event(video(3, false));
        assert!(
            matches!(stream.next().await.unwrap(), MediaPacket::Video(f) if f.timestamp_us == 3)
        );
        // Nothing changes: no new stream.
        assert!(a.set_audio(Some(eld())).is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn announces_without_audio_after_the_grace() {
        let mut a = Assembler::default();
        a.set_video_expected(true);
        a.on_event(format());
        let deadline = a.deadline().unwrap();
        tokio::time::sleep_until(deadline).await;
        let mut stream = a.on_deadline().unwrap();
        assert!(stream.open(spec()).await.unwrap().audio.is_none());
        assert!(a.deadline().is_none());
        // Audio arriving later replaces the stream.
        let mut second = a.set_audio(Some(eld())).unwrap();
        assert!(second.open(spec()).await.unwrap().audio.is_some());
    }

    #[tokio::test]
    async fn audio_only_announces_on_the_first_packet() {
        let mut a = Assembler::default();
        assert!(a.set_audio(Some(eld())).is_none());
        let frame = AudioFrame {
            codec: AudioCodec::AacEld,
            data: Bytes::from_static(&[1]),
            timestamp_us: 9,
            sample_rate: 44_100,
            channels: 2,
        };
        let mut stream = a.on_event(MediaEvent::Audio(frame)).unwrap();
        assert!(stream.open(spec()).await.unwrap().video.is_none());
        assert!(matches!(
            stream.next().await.unwrap(),
            MediaPacket::PresentFrom { timestamp_us: 9 }
        ));
        assert!(matches!(
            stream.next().await.unwrap(),
            MediaPacket::Audio(_)
        ));
    }
}
