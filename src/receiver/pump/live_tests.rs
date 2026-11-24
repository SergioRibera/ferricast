//! The pump on a live pushed stream, end to end with stand-in decoders
//! and a recording sink, in real time.

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use ferricast_core::{
    AudioCodec, AudioDecoder, AudioDecoderConfig, AudioFrame, AudioStreamInfo, CapturedFrame,
    Codec, DecodedAudio, DecoderConfig, EncodedFrame, FrameSink, MediaInfo, MediaPacket,
    PixelFormat, PushSender, PushedStream, RawFrame, VideoDecoder, VideoStreamInfo,
};

use super::*;

const FRAME_US: u64 = 33_333;
/// 10 ms of 48 kHz stereo per audio packet.
const AUDIO_US: u64 = 10_000;
const AUDIO_FRAMES: usize = 480;

fn wall_us() -> u64 {
    u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros(),
    )
    .unwrap()
}

struct PassVideo;

impl VideoDecoder for PassVideo {
    const CODEC: Codec = Codec::H264;
    fn configure(&mut self, _config: &DecoderConfig) -> ferricast_core::CodecResult<()> {
        Ok(())
    }
    fn decode(
        &mut self,
        frame: EncodedFrame,
    ) -> ferricast_core::CodecResult<Option<CapturedFrame>> {
        Ok(Some(CapturedFrame::Cpu(RawFrame {
            hdr: None,
            width: 2,
            height: 2,
            stride: 8,
            format: PixelFormat::Bgra,
            data: Bytes::new(),
            timestamp_us: frame.timestamp_us,
            colour: None,
        })))
    }
    fn flush(&mut self) -> ferricast_core::CodecResult<Vec<CapturedFrame>> {
        Ok(Vec::new())
    }
}

struct PassAudio;

impl AudioDecoder for PassAudio {
    const CODEC: AudioCodec = AudioCodec::Pcm;
    fn configure(&mut self, _config: &AudioDecoderConfig) -> ferricast_core::CodecResult<()> {
        Ok(())
    }
    fn decode(&mut self, frame: AudioFrame) -> ferricast_core::CodecResult<Option<DecodedAudio>> {
        Ok(Some(DecodedAudio {
            pcm: vec![1; AUDIO_FRAMES * 2],
            sample_rate: 48_000,
            channels: 2,
            timestamp_us: frame.timestamp_us,
        }))
    }
    fn flush(&mut self) -> ferricast_core::CodecResult<Vec<DecodedAudio>> {
        Ok(Vec::new())
    }
}

/// Capture-to-output lag of everything the sink got, in arrival order.
#[derive(Default)]
struct Seen {
    video: Vec<Duration>,
    /// Lag of each audio buffer when it starts to be heard.
    audio: Vec<Duration>,
}

/// Plays audio back to back from when it is pushed, like a sound card.
struct RecordingSink {
    seen: Arc<Mutex<Seen>>,
    play_head: Instant,
}

#[async_trait::async_trait]
impl FrameSink for RecordingSink {
    async fn push_video(&mut self, frame: CapturedFrame) -> ferricast_core::Result<()> {
        let lag = wall_us().saturating_sub(frame.timestamp_us());
        self.seen
            .lock()
            .unwrap()
            .video
            .push(Duration::from_micros(lag));
        Ok(())
    }

    async fn push_audio(&mut self, audio: DecodedAudio) -> ferricast_core::Result<()> {
        let now = Instant::now();
        let start = self.play_head.max(now);
        let frames = audio.pcm.len() / usize::from(audio.channels);
        self.play_head = start + Duration::from_secs_f64(frames as f64 / 48_000.0);
        let heard_us = wall_us() + u64::try_from((start - now).as_micros()).unwrap();
        self.seen.lock().unwrap().audio.push(Duration::from_micros(
            heard_us.saturating_sub(audio.timestamp_us),
        ));
        Ok(())
    }

    fn audio_backlog(&self) -> Option<Duration> {
        Some(self.play_head.saturating_duration_since(Instant::now()))
    }

    async fn flush_audio(&mut self) -> ferricast_core::Result<()> {
        self.play_head = Instant::now();
        Ok(())
    }
}

fn info(audio: bool) -> MediaInfo {
    MediaInfo {
        video: Some(VideoStreamInfo {
            hdr: None,
            codec: Codec::H264,
            width: 2,
            height: 2,
            fps: 30,
            colour: None,
        }),
        audio: audio.then_some(AudioStreamInfo {
            codec: AudioCodec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            source: None,
        }),
        duration_us: None,
        is_live: true,
    }
}

fn video(pts: u64) -> MediaPacket {
    MediaPacket::Video(EncodedFrame {
        codec: Codec::H264,
        data: Bytes::from_static(&[0]),
        timestamp_us: pts,
        is_keyframe: true,
        duration_us: None,
        pts_dts: (pts, pts),
    })
}

fn audio(pts: u64) -> MediaPacket {
    MediaPacket::Audio(AudioFrame {
        codec: AudioCodec::Pcm,
        data: Bytes::from_static(&[0]),
        timestamp_us: pts,
        sample_rate: 48_000,
        channels: 2,
    })
}

/// A live sender: media stamped with its capture time, `delay` before it
/// reaches the receiver.
struct Sender {
    tx: PushSender,
    with_audio: bool,
    next_video: u64,
    next_audio: u64,
}

impl Sender {
    /// Sends everything captured up to `delay` ago.
    fn send_until_now(&mut self, delay: Duration) {
        let until = wall_us() - u64::try_from(delay.as_micros()).unwrap();
        while self.next_video <= until {
            self.tx.try_push(video(self.next_video)).unwrap();
            self.next_video += FRAME_US;
        }
        while self.with_audio && self.next_audio <= until {
            self.tx.try_push(audio(self.next_audio)).unwrap();
            self.next_audio += AUDIO_US;
        }
    }

    /// Streams in real time for `length`, `delay` behind capture.
    async fn stream(&mut self, length: Duration, delay: Duration) {
        let end = Instant::now() + length;
        while Instant::now() < end {
            self.send_until_now(delay);
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
}

/// Starts a pump on a live stream whose sender began capturing
/// `backlog` ago, everything of which is queued when the pump starts.
/// The stream announces audio when `with_audio`; the sender sends it when
/// `sends_audio`.
fn start(
    with_audio: bool,
    sends_audio: bool,
    backlog: Duration,
) -> (Sender, Arc<Mutex<Seen>>, mpsc::Sender<PumpCmd>) {
    let (tx, stream) = PushedStream::channel(info(with_audio), 4096);
    let start_us = wall_us() - u64::try_from(backlog.as_micros()).unwrap();
    let mut sender = Sender {
        tx,
        with_audio: sends_audio,
        next_video: start_us,
        next_audio: start_us,
    };
    sender.send_until_now(Duration::ZERO);
    let seen = Arc::new(Mutex::new(Seen::default()));
    let parts = PumpParts {
        puller: Box::new(stream),
        video_dec: Some(Box::new(PassVideo)),
        audio_dec: with_audio.then(|| Box::new(PassAudio) as Box<dyn ErasedAudioDecoder>),
        sink: Box::new(RecordingSink {
            seen: seen.clone(),
            play_head: Instant::now(),
        }),
        playing: true,
        gain: 1.0,
        start_us: 0,
        position: Arc::new(AtomicU64::new(0)),
        timeline: Timeline::Live,
    };
    let (cmd_tx, cmd_rx) = mpsc::channel(4);
    tokio::spawn(run(parts, cmd_rx));
    (sender, seen, cmd_tx)
}

fn last(lags: &[Duration], n: usize) -> &[Duration] {
    &lags[lags.len().saturating_sub(n)..]
}

/// Where a live stream should be presented: the target after capture
/// (transit is nil here), give or take the tolerance and some slack for
/// a loaded test machine.
fn assert_near_the_edge(lags: &[Duration], what: &str) {
    let bound = LIVE_TARGET + LIVE_TOLERANCE + Duration::from_millis(60);
    assert!(!lags.is_empty(), "no {what}");
    for lag in lags {
        assert!(*lag < bound, "{what} {lag:?} behind capture: {lags:?}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_startup_backlog_is_skipped_not_replayed() {
    let (mut sender, seen, cmd) = start(true, true, Duration::from_millis(1_500));
    sender
        .stream(Duration::from_millis(800), Duration::ZERO)
        .await;
    let _ = cmd.send(PumpCmd::Stop).await;
    let seen = seen.lock().unwrap();
    // Not one frame of the backlog was shown, from the first on.
    assert_near_the_edge(&seen.video, "video");
    assert_near_the_edge(last(&seen.audio, 20), "audio");
    // And it plays on in sync.
    let video = last(&seen.video, 5)[0].as_secs_f64();
    let heard = last(&seen.audio, 5)[0].as_secs_f64();
    assert!(
        (video - heard).abs() < 0.03,
        "video {video} s, audio {heard} s"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn video_starts_without_waiting_for_audio() {
    // Audio is announced but never comes.
    let (mut sender, seen, cmd) = start(true, false, Duration::from_millis(300));
    sender
        .stream(Duration::from_millis(500), Duration::ZERO)
        .await;
    let _ = cmd.send(PumpCmd::Stop).await;
    let seen = seen.lock().unwrap();
    assert!(seen.video.len() >= 8, "{} frames", seen.video.len());
    assert_near_the_edge(&seen.video, "video");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_slower_network_moves_the_timeline_later() {
    let (mut sender, seen, cmd) = start(false, false, Duration::ZERO);
    sender
        .stream(Duration::from_millis(500), Duration::ZERO)
        .await;
    // From now on every packet arrives 400 ms after capture: within two
    // transit windows the timeline follows, and frames are shown again.
    sender
        .stream(Duration::from_millis(3_000), Duration::from_millis(400))
        .await;
    let _ = cmd.send(PumpCmd::Stop).await;
    let seen = seen.lock().unwrap();
    let recent = last(&seen.video, 10);
    assert_eq!(recent.len(), 10);
    for lag in recent {
        let expected = Duration::from_millis(400) + LIVE_TARGET;
        assert!(
            lag.abs_diff(expected) < LIVE_TOLERANCE + Duration::from_millis(30),
            "{lag:?}"
        );
    }
}
