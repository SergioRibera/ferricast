//! The playback pump: pulls packets, decodes them and hands the
//! output to the sink paced in real time.
//!
//! Containers interleave audio and video loosely (an MPEG-TS muxer
//! may emit audio half a second after the video with the same PTS),
//! so the pump does not present in arrival order. Compressed packets
//! are queued per stream; each stream keeps one decoded item ready,
//! and whichever item is due first is pushed. Reading continues while
//! a stream has nothing ready, concurrently with waiting for the next
//! due item, so a slow network never delays what is already decoded.
//!
//! Pacing: the timeline is anchored once every stream has an item
//! ready, at `now + AUDIO_LEAD` for the earliest PTS. Video frames are
//! pushed at their presentation instant; frames already late are not
//! shown so playback catches up. Audio is pushed `AUDIO_LEAD` ahead
//! of its presentation instant, and every timeline start is preceded
//! by `AUDIO_LEAD` of silence: a sink that plays samples back-to-back
//! as they arrive then has exactly `AUDIO_LEAD` queued, so its output
//! lines up with the video and absorbs scheduling jitter. The
//! timeline re-anchors on discontinuities, seeks, resume after pause,
//! stalls and timestamp jumps.

mod clock;

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use ferricast_core::{
    AudioFrame, CapturedFrame, DecodedAudio, EncodedFrame, FrameSink, MediaPacket,
};

use super::{ErasedAudioDecoder, ErasedPuller, ErasedVideoDecoder};
use clock::Clock;

/// Audio queued ahead of its presentation time; also the start-up
/// delay of every timeline.
const AUDIO_LEAD: Duration = Duration::from_millis(200);
/// Video frames later than this are dropped instead of shown.
const MAX_VIDEO_LATENESS: Duration = Duration::from_millis(100);
/// Output this late means a stall (slow network, suspended process):
/// re-anchor instead of racing to catch up.
const RESYNC_LATENESS: Duration = Duration::from_secs(1);
/// Compressed packets queued per stream at most. Bounds read-ahead
/// when one stream runs dry (an audio track shorter than the video).
const MAX_QUEUED_PACKETS: usize = 600;

pub(super) enum PumpCmd {
    Play,
    Pause,
    Seek(u64),
    /// Output gain, `0.0`–`1.0` (zero when muted).
    Volume(f32),
    Stop,
}

/// Why a pump ended.
#[derive(Debug)]
pub(super) enum PumpEnd {
    /// The media played to its end.
    Finished,
    /// Stopped on request.
    Stopped,
    Failed(String),
}

pub(super) struct PumpParts {
    pub(super) puller: Box<dyn ErasedPuller>,
    pub(super) video_dec: Option<Box<dyn ErasedVideoDecoder>>,
    pub(super) audio_dec: Option<Box<dyn ErasedAudioDecoder>>,
    pub(super) sink: Box<dyn FrameSink>,
    pub(super) playing: bool,
    pub(super) gain: f32,
    /// Media position the playback starts from (LOAD start time).
    pub(super) start_us: u64,
    /// Where playback is, in media time; read by the session for status.
    pub(super) position: Arc<AtomicU64>,
    /// How packet timestamps relate to media time.
    pub(super) timeline: Timeline,
}

/// How packet timestamps relate to the media position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Timeline {
    /// Timestamps are media time (progressive files start at zero).
    MediaTime,
    /// Timestamps have an arbitrary origin (MPEG-TS), so the position is
    /// counted from the start or seek target.
    Offset,
    /// A live stream the sender pushes (screen mirroring): timestamps
    /// have an arbitrary origin and the timeline starts as soon as any
    /// stream has something to present.
    Live,
}

pub(super) async fn run(parts: PumpParts, cmd_rx: mpsc::Receiver<PumpCmd>) -> PumpEnd {
    let mut pump = Pump {
        puller: parts.puller,
        video: parts.video_dec.map(Stream::new),
        audio: parts.audio_dec.map(Stream::new),
        sink: parts.sink,
        cmd_rx,
        playing: parts.playing,
        clock: Clock::new(AUDIO_LEAD),
        position: Position::new(parts.position, parts.start_us, parts.timeline),
        audio_primed: false,
        input: Input::Open,
        late_drops: 0,
    };
    if let Err(e) = pump.sink.set_volume(parts.gain).await {
        tracing::warn!(%e, "pump: sink.set_volume failed");
    }
    let end = pump.run().await;
    if pump.late_drops > 0 {
        tracing::info!(frames = pump.late_drops, "pump: late video frames dropped");
    }
    let _ = pump.puller.close().await;
    end
}

/// Media position of what is being presented. Container timestamps
/// need not start at zero (MPEG-TS rarely does), so the position is
/// counted from a known base: the start or seek target, or the position
/// reached when the timeline restarted.
struct Position {
    shared: Arc<AtomicU64>,
    timeline: Timeline,
    base_us: u64,
    /// PTS of the first item presented since the base was set.
    first_pts: Option<u64>,
}

impl Position {
    fn new(shared: Arc<AtomicU64>, start_us: u64, timeline: Timeline) -> Self {
        shared.store(start_us, Ordering::Relaxed);
        Self {
            shared,
            timeline,
            base_us: start_us,
            first_pts: None,
        }
    }

    /// Continues from the current position on a new timeline.
    fn restart(&mut self) {
        self.base_us = self.shared.load(Ordering::Relaxed);
        self.first_pts = None;
    }

    /// Restarts at `position_us` (a seek).
    fn jump(&mut self, position_us: u64) {
        self.base_us = position_us;
        self.first_pts = None;
        self.shared.store(position_us, Ordering::Relaxed);
    }

    /// Records that the item with `pts_us` is being presented, `behind`
    /// before it is actually seen or heard.
    fn presented(&mut self, pts_us: u64, behind: Duration) {
        if self.timeline == Timeline::MediaTime {
            let at = pts_us.saturating_sub(behind.as_micros() as u64);
            self.shared.store(at, Ordering::Relaxed);
            return;
        }
        let first = *self.first_pts.get_or_insert(pts_us);
        let elapsed = pts_us
            .saturating_sub(first)
            .saturating_sub(behind.as_micros() as u64);
        self.shared
            .store(self.base_us.saturating_add(elapsed), Ordering::Relaxed);
    }
}

/// Decoder operations the pump needs, over both stream kinds.
trait Decode {
    type Packet;
    type Output;
    fn decode(&mut self, packet: Self::Packet) -> ferricast_core::Result<Option<Self::Output>>;
    fn flush(&mut self) -> Vec<Self::Output>;
    fn pts_us(output: &Self::Output) -> u64;
    /// What of `output` may be presented when nothing before
    /// `from_us` may: all of it, a trimmed part, or nothing.
    fn trim(output: Self::Output, from_us: u64) -> Option<Self::Output>;
}

impl Decode for Box<dyn ErasedVideoDecoder> {
    type Packet = EncodedFrame;
    type Output = CapturedFrame;
    fn decode(&mut self, packet: EncodedFrame) -> ferricast_core::Result<Option<CapturedFrame>> {
        ErasedVideoDecoder::decode(self.as_mut(), packet)
    }
    fn flush(&mut self) -> Vec<CapturedFrame> {
        ErasedVideoDecoder::flush(self.as_mut()).unwrap_or_default()
    }
    fn pts_us(output: &CapturedFrame) -> u64 {
        output.timestamp_us()
    }
    fn trim(output: CapturedFrame, from_us: u64) -> Option<CapturedFrame> {
        (output.timestamp_us() >= from_us).then_some(output)
    }
}

impl Decode for Box<dyn ErasedAudioDecoder> {
    type Packet = AudioFrame;
    type Output = DecodedAudio;
    fn decode(&mut self, packet: AudioFrame) -> ferricast_core::Result<Option<DecodedAudio>> {
        ErasedAudioDecoder::decode(self.as_mut(), packet)
    }
    fn flush(&mut self) -> Vec<DecodedAudio> {
        ErasedAudioDecoder::flush(self.as_mut()).unwrap_or_default()
    }
    fn pts_us(output: &DecodedAudio) -> u64 {
        output.timestamp_us
    }
    fn trim(output: DecodedAudio, from_us: u64) -> Option<DecodedAudio> {
        trim_audio(output, from_us)
    }
}

/// One elementary stream: queued packets, its decoder, and decoded
/// output waiting for its turn (in presentation order).
struct Stream<D: Decode> {
    decoder: D,
    packets: VecDeque<D::Packet>,
    ready: VecDeque<D::Output>,
    /// Output before this timestamp is decode-only (see
    /// [`MediaPacket::PresentFrom`]).
    present_from: Option<u64>,
}

impl<D: Decode> Stream<D> {
    fn new(decoder: D) -> Self {
        Self {
            decoder,
            packets: VecDeque::new(),
            ready: VecDeque::new(),
            present_from: None,
        }
    }

    /// Decodes queued packets until an output is ready.
    fn fill(&mut self) {
        while self.ready.is_empty() {
            let Some(packet) = self.packets.pop_front() else {
                return;
            };
            match self.decoder.decode(packet) {
                Ok(Some(out)) => self.accept(out),
                Ok(None) => {}
                Err(e) => tracing::warn!(%e, "pump: decode failed"),
            }
        }
    }

    /// Moves everything the decoder still buffers into `ready`.
    fn drain(&mut self) {
        while let Some(packet) = self.packets.pop_front() {
            match self.decoder.decode(packet) {
                Ok(Some(out)) => self.accept(out),
                Ok(None) => {}
                Err(e) => tracing::warn!(%e, "pump: decode failed"),
            }
        }
        for out in self.decoder.flush() {
            self.accept(out);
        }
    }

    /// Queues decoded output for presentation, minus any decode-only
    /// part.
    fn accept(&mut self, out: D::Output) {
        let out = match self.present_from {
            Some(from_us) => D::trim(out, from_us),
            None => Some(out),
        };
        self.ready.extend(out);
    }

    fn clear(&mut self) {
        self.packets.clear();
        self.ready.clear();
        self.present_from = None;
        let _ = self.decoder.flush();
    }

    fn front_pts(&self) -> Option<u64> {
        self.ready.front().map(D::pts_us)
    }
}

/// State of the packet input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Input {
    Open,
    /// A discontinuity was read: finish presenting the old timeline
    /// before reading on.
    Discontinuity,
    Eos,
}

/// What to do after handling a command.
enum Flow {
    Continue,
    Exit(PumpEnd),
}

#[derive(Clone, Copy)]
enum Which {
    Video,
    Audio,
}

struct Pump {
    puller: Box<dyn ErasedPuller>,
    video: Option<Stream<Box<dyn ErasedVideoDecoder>>>,
    audio: Option<Stream<Box<dyn ErasedAudioDecoder>>>,
    sink: Box<dyn FrameSink>,
    cmd_rx: mpsc::Receiver<PumpCmd>,
    playing: bool,
    clock: Clock,
    position: Position,
    /// Silence was queued for the current timeline.
    audio_primed: bool,
    input: Input,
    late_drops: u64,
}

impl Pump {
    async fn run(&mut self) -> PumpEnd {
        loop {
            if let Some(v) = self.video.as_mut() {
                v.fill();
            }
            if let Some(a) = self.audio.as_mut() {
                a.fill();
            }
            let video_ready = self.video.as_ref().map(|s| !s.ready.is_empty());
            let audio_ready = self.audio.as_ref().map(|s| !s.ready.is_empty());
            let nothing_ready = video_ready != Some(true) && audio_ready != Some(true);

            if self.input != Input::Open && nothing_ready {
                // Old timeline fully presented (or nothing left).
                match self.input {
                    Input::Eos => return PumpEnd::Finished,
                    Input::Discontinuity | Input::Open => {
                        self.restart_timeline().await;
                        self.input = Input::Open;
                        continue;
                    }
                }
            }

            let starved = video_ready == Some(false) || audio_ready == Some(false);
            let queues_full = self
                .video
                .as_ref()
                .is_some_and(|s| s.packets.len() >= MAX_QUEUED_PACKETS)
                || self
                    .audio
                    .as_ref()
                    .is_some_and(|s| s.packets.len() >= MAX_QUEUED_PACKETS);
            let read = self.playing && self.input == Input::Open && starved && !queues_full;
            // Anchor only once every stream has something ready, so
            // the earliest PTS of either stream starts the timeline. A
            // live stream starts with whatever arrives first: its other
            // stream may stay silent for as long as the sender likes.
            let live = self.position.timeline == Timeline::Live;
            let can_present =
                self.playing && !nothing_ready && (self.clock.is_anchored() || !read || live);
            let next = if can_present { self.next_due() } else { None };

            tokio::select! {
                biased;
                cmd = self.cmd_rx.recv() => {
                    if let Flow::Exit(end) = self.on_cmd(cmd).await {
                        return end;
                    }
                }
                () = sleep_until_opt(next.map(|(due, _)| due)) => {
                    if let Some((_, which)) = next {
                        self.present(which).await;
                    }
                }
                packet = self.puller.next(), if read => {
                    if let Some(end) = self.on_packet(packet) {
                        return end;
                    }
                }
            }
        }
    }

    fn on_packet(&mut self, packet: ferricast_core::Result<MediaPacket>) -> Option<PumpEnd> {
        match packet {
            Ok(MediaPacket::Video(f)) => {
                if let Some(v) = self.video.as_mut() {
                    v.packets.push_back(f);
                }
            }
            Ok(MediaPacket::Audio(a)) => {
                if let Some(s) = self.audio.as_mut() {
                    s.packets.push_back(a);
                }
            }
            Ok(MediaPacket::Discontinuity) => {
                self.drain_decoders();
                self.set_present_from(None);
                self.input = Input::Discontinuity;
            }
            Ok(MediaPacket::PresentFrom { timestamp_us }) => {
                self.set_present_from(Some(timestamp_us));
            }
            Ok(MediaPacket::Eos) => {
                self.drain_decoders();
                self.input = Input::Eos;
            }
            Err(e) => return Some(PumpEnd::Failed(e.to_string())),
        }
        None
    }

    fn set_present_from(&mut self, from_us: Option<u64>) {
        if let Some(v) = self.video.as_mut() {
            v.present_from = from_us;
        }
        if let Some(a) = self.audio.as_mut() {
            a.present_from = from_us;
        }
    }

    fn drain_decoders(&mut self) {
        if let Some(v) = self.video.as_mut() {
            v.drain();
        }
        if let Some(a) = self.audio.as_mut() {
            a.drain();
        }
    }

    /// Starts a new timeline: whatever audio the sink still queues
    /// belongs to the old one and is dropped.
    async fn restart_timeline(&mut self) {
        self.flush_sink_audio().await;
        self.clock.reset();
        self.position.restart();
        self.audio_primed = false;
    }

    async fn flush_sink_audio(&mut self) {
        if let Err(e) = self.sink.flush_audio().await {
            tracing::warn!(%e, "pump: sink.flush_audio failed");
        }
    }

    /// The earliest due item, anchoring the timeline if needed.
    fn next_due(&mut self) -> Option<(Instant, Which)> {
        let video_pts = self.video.as_ref().and_then(Stream::front_pts);
        let audio_pts = self.audio.as_ref().and_then(Stream::front_pts);
        let now = Instant::now();
        if !self.clock.is_anchored() {
            let first = video_pts.into_iter().chain(audio_pts).min()?;
            self.clock.anchor_at(first, now);
        }
        let video = video_pts.map(|p| (self.clock.due(p, now), Which::Video));
        let audio = audio_pts.map(|p| {
            let due = self.clock.due(p, now);
            (due.checked_sub(AUDIO_LEAD).unwrap_or(due), Which::Audio)
        });
        match (video, audio) {
            (Some(v), Some(a)) => Some(if a.0 <= v.0 { a } else { v }),
            (v, a) => v.or(a),
        }
    }

    async fn on_cmd(&mut self, cmd: Option<PumpCmd>) -> Flow {
        match cmd {
            Some(PumpCmd::Play) => {
                if !self.playing {
                    self.playing = true;
                    self.restart_timeline().await;
                }
            }
            Some(PumpCmd::Pause) => {
                self.playing = false;
                self.flush_sink_audio().await;
            }
            Some(PumpCmd::Seek(position_us)) => {
                if let Err(e) = self.puller.seek(position_us).await {
                    tracing::warn!(%e, "pump: seek failed");
                    return Flow::Continue;
                }
                // Everything queued belongs to the old position.
                if let Some(v) = self.video.as_mut() {
                    v.clear();
                }
                if let Some(a) = self.audio.as_mut() {
                    a.clear();
                }
                self.input = Input::Open;
                self.restart_timeline().await;
                self.position.jump(position_us);
            }
            Some(PumpCmd::Volume(gain)) => {
                if let Err(e) = self.sink.set_volume(gain).await {
                    tracing::warn!(%e, "pump: sink.set_volume failed");
                }
            }
            Some(PumpCmd::Stop) | None => {
                self.flush_sink_audio().await;
                return Flow::Exit(PumpEnd::Stopped);
            }
        }
        Flow::Continue
    }

    /// Pushes the front item of `which`, or drops it when it is too
    /// late to be worth showing.
    async fn present(&mut self, which: Which) {
        let lateness = |due: Instant| Instant::now().saturating_duration_since(due);
        match which {
            Which::Video => {
                let Some(stream) = self.video.as_mut() else {
                    return;
                };
                let Some(frame) = stream.ready.pop_front() else {
                    return;
                };
                let pts = frame.timestamp_us();
                let late = lateness(self.clock.due(pts, Instant::now()));
                if late > RESYNC_LATENESS {
                    tracing::debug!(?late, "pump: stalled — re-anchoring");
                    stream.ready.push_front(frame);
                    self.restart_timeline().await;
                    return;
                }
                if late > MAX_VIDEO_LATENESS {
                    self.late_drops += 1;
                    return;
                }
                self.position.presented(pts, Duration::ZERO);
                if let Err(e) = self.sink.push_video(frame).await {
                    tracing::warn!(%e, "pump: sink.push_video failed");
                }
            }
            Which::Audio => {
                let Some(stream) = self.audio.as_mut() else {
                    return;
                };
                let Some(pcm) = stream.ready.pop_front() else {
                    return;
                };
                let now = Instant::now();
                let due = self.clock.due(pcm.timestamp_us, now);
                let late = lateness(due.checked_sub(AUDIO_LEAD).unwrap_or(due));
                if late > RESYNC_LATENESS {
                    tracing::debug!(?late, "pump: audio stalled — re-anchoring");
                    stream.ready.push_front(pcm);
                    self.restart_timeline().await;
                    return;
                }
                if self.audio_primed {
                    if let Some(backlog) = self.sink.audio_backlog() {
                        self.clock.follow_audio(backlog, now);
                    }
                } else {
                    self.audio_primed = true;
                    if let Err(e) = self.sink.push_audio(silence(&pcm, AUDIO_LEAD)).await {
                        tracing::warn!(%e, "pump: sink.push_audio failed");
                    }
                }
                if self.video.is_none() {
                    self.position.presented(pcm.timestamp_us, AUDIO_LEAD);
                }
                if let Err(e) = self.sink.push_audio(pcm).await {
                    tracing::warn!(%e, "pump: sink.push_audio failed");
                }
            }
        }
    }
}

async fn sleep_until_opt(due: Option<Instant>) {
    match due {
        Some(due) => tokio::time::sleep_until(due.into()).await,
        None => std::future::pending().await,
    }
}

/// The part of `audio` from `from_us` on, or `None` when it ends
/// before that.
fn trim_audio(mut audio: DecodedAudio, from_us: u64) -> Option<DecodedAudio> {
    if audio.timestamp_us >= from_us {
        return Some(audio);
    }
    let channels = usize::from(audio.channels.max(1));
    let rate = u64::from(audio.sample_rate.max(1));
    let frames = (audio.pcm.len() / channels) as u64;
    let skip = (from_us - audio.timestamp_us).saturating_mul(rate) / 1_000_000;
    if skip >= frames {
        return None;
    }
    let skip_samples = usize::try_from(skip).unwrap_or(usize::MAX) * channels;
    audio.pcm.drain(..skip_samples);
    audio.timestamp_us += skip * 1_000_000 / rate;
    Some(audio)
}

/// `duration` of silence in `like`'s format, timed just before it.
fn silence(like: &DecodedAudio, duration: Duration) -> DecodedAudio {
    let frames = u64::from(like.sample_rate) * duration.as_millis() as u64 / 1000;
    let samples = usize::try_from(frames).unwrap_or(0) * usize::from(like.channels);
    DecodedAudio {
        pcm: vec![0; samples],
        sample_rate: like.sample_rate,
        channels: like.channels,
        timestamp_us: like
            .timestamp_us
            .saturating_sub(duration.as_micros() as u64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_before_the_presentation_start_is_trimmed() {
        let audio = |timestamp_us| DecodedAudio {
            pcm: vec![7; 4_800 * 2],
            sample_rate: 48_000,
            channels: 2,
            timestamp_us,
        };
        // Wholly before: dropped.
        assert!(trim_audio(audio(0), 100_000).is_none());
        // Wholly after: kept as is.
        assert_eq!(
            trim_audio(audio(200_000), 100_000).unwrap().pcm.len(),
            9_600
        );
        // Straddling: the first 25 ms are cut.
        let cut = trim_audio(audio(1_000_000), 1_025_000).unwrap();
        assert_eq!(cut.pcm.len(), (4_800 - 1_200) * 2);
        assert_eq!(cut.timestamp_us, 1_025_000);
    }

    #[test]
    fn silence_matches_format_and_duration() {
        let like = DecodedAudio {
            pcm: vec![1; 2048],
            sample_rate: 48_000,
            channels: 2,
            timestamp_us: 1_000_000,
        };
        let s = silence(&like, Duration::from_millis(200));
        assert_eq!(s.pcm.len(), 9_600 * 2);
        assert!(s.pcm.iter().all(|&x| x == 0));
        assert_eq!(s.timestamp_us, 800_000);
    }
}
