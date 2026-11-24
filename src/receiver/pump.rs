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
//!
//! Live streams (screen mirroring) are not buffered: they are read as
//! they arrive, the timeline starts at the newest packet queued while
//! the pipeline came up, and it follows the live edge measured from the
//! arrivals ([`live`]) instead of the first packet. Whatever is too late
//! for that schedule is dropped, video and audio alike, and the audio
//! lead is shorter.

mod clock;
mod latency;
mod live;
#[cfg(test)]
mod live_tests;

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
use latency::{Kind, LatencyMeter, Stage};
use live::{Drift, LiveEdge};

/// Audio queued ahead of its presentation time; also the start-up
/// delay of every timeline.
const AUDIO_LEAD: Duration = Duration::from_millis(200);
/// Audio lead of a live stream: kept short, since it adds to the latency.
const LIVE_AUDIO_LEAD: Duration = Duration::from_millis(100);
/// Time a live stream keeps between the fastest packet arrivals and
/// presentation, for network jitter and decoding.
const LIVE_TARGET: Duration = Duration::from_millis(200);
/// How far a live stream may drift from that schedule before it is
/// moved back to it.
const LIVE_TOLERANCE: Duration = Duration::from_millis(100);
/// A live sink holding this much more (or less) audio than the lead no
/// longer lines up with the timeline, typically because it queued audio
/// before its output device opened: what it holds is dropped.
const LIVE_MAX_AUDIO_ERROR: Duration = Duration::from_millis(100);
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
    let live = parts.timeline == Timeline::Live;
    let lead = if live { LIVE_AUDIO_LEAD } else { AUDIO_LEAD };
    let mut pump = Pump {
        puller: parts.puller,
        video: parts.video_dec.map(|d| Stream::new(d, Kind::Video)),
        audio: parts.audio_dec.map(|d| Stream::new(d, Kind::Audio)),
        sink: parts.sink,
        cmd_rx,
        playing: parts.playing,
        clock: Clock::new(lead),
        lead,
        position: Position::new(parts.position, parts.start_us, parts.timeline),
        audio_primed: false,
        audio_stale: false,
        input: Input::Open,
        late_drops: 0,
        meter: live.then(LatencyMeter::default),
        edge: live.then(|| LiveEdge::new(LIVE_TARGET, LIVE_TOLERANCE)),
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
            let at = pts_us.saturating_sub(u64::try_from(behind.as_micros()).unwrap_or(u64::MAX));
            self.shared.store(at, Ordering::Relaxed);
            return;
        }
        let first = *self.first_pts.get_or_insert(pts_us);
        let elapsed = pts_us
            .saturating_sub(first)
            .saturating_sub(u64::try_from(behind.as_micros()).unwrap_or(u64::MAX));
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
        Ok(ErasedVideoDecoder::decode(self.as_mut(), packet)?)
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
        Ok(ErasedAudioDecoder::decode(self.as_mut(), packet)?)
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
    kind: Kind,
    packets: VecDeque<D::Packet>,
    ready: VecDeque<D::Output>,
    /// Output before this timestamp is decode-only (see
    /// [`MediaPacket::PresentFrom`]).
    present_from: Option<u64>,
}

impl<D: Decode> Stream<D> {
    fn new(decoder: D, kind: Kind) -> Self {
        Self {
            decoder,
            kind,
            packets: VecDeque::new(),
            ready: VecDeque::new(),
            present_from: None,
        }
    }

    /// Decodes queued packets until an output is ready.
    fn fill(&mut self, meter: &mut Option<LatencyMeter>) {
        while self.ready.is_empty() {
            let Some(packet) = self.packets.pop_front() else {
                return;
            };
            match self.decoder.decode(packet) {
                Ok(Some(out)) => self.accept(out, meter),
                Ok(None) => {}
                Err(e) => tracing::warn!(%e, "pump: decode failed"),
            }
        }
    }

    /// Moves everything the decoder still buffers into `ready`.
    fn drain(&mut self, meter: &mut Option<LatencyMeter>) {
        while let Some(packet) = self.packets.pop_front() {
            match self.decoder.decode(packet) {
                Ok(Some(out)) => self.accept(out, meter),
                Ok(None) => {}
                Err(e) => tracing::warn!(%e, "pump: decode failed"),
            }
        }
        for out in self.decoder.flush() {
            self.accept(out, meter);
        }
    }

    /// Queues decoded output for presentation, minus any decode-only
    /// part.
    fn accept(&mut self, out: D::Output, meter: &mut Option<LatencyMeter>) {
        let out = match self.present_from {
            Some(from_us) => D::trim(out, from_us),
            None => Some(out),
        };
        if let (Some(meter), Some(out)) = (meter.as_mut(), out.as_ref()) {
            meter.record(self.kind, Stage::Decoded, D::pts_us(out), Duration::ZERO);
        }
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

struct Pump {
    puller: Box<dyn ErasedPuller>,
    video: Option<Stream<Box<dyn ErasedVideoDecoder>>>,
    audio: Option<Stream<Box<dyn ErasedAudioDecoder>>>,
    sink: Box<dyn FrameSink>,
    cmd_rx: mpsc::Receiver<PumpCmd>,
    playing: bool,
    clock: Clock,
    /// How far ahead of its presentation audio is pushed.
    lead: Duration,
    position: Position,
    /// Silence was queued for the current timeline.
    audio_primed: bool,
    /// What the sink holds no longer matches the timeline: flush it
    /// before the next audio push.
    audio_stale: bool,
    input: Input,
    late_drops: u64,
    /// Capture-to-output latency, for live streams (their timestamps
    /// are on the local wall clock).
    meter: Option<LatencyMeter>,
    /// Where the live edge is, for live streams.
    edge: Option<LiveEdge>,
}

impl Pump {
    async fn run(&mut self) -> PumpEnd {
        loop {
            if self.edge.is_some()
                && self.playing
                && self.input == Input::Open
                && !self.clock.is_anchored()
                && let Some(end) = self.read_backlog().await
            {
                return end;
            }
            if let Some(v) = self.video.as_mut() {
                v.fill(&mut self.meter);
            }
            if let Some(a) = self.audio.as_mut() {
                a.fill(&mut self.meter);
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
            // A live stream is read as it arrives, so arrival times are
            // known and nothing waits upstream.
            let live = self.position.timeline == Timeline::Live;
            let read =
                self.playing && self.input == Input::Open && (starved || live) && !queues_full;
            // Anchor only once every stream has something ready, so
            // the earliest PTS of either stream starts the timeline. A
            // live stream starts with whatever arrives first: its other
            // stream may stay silent for as long as the sender likes.
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
                self.arrived(Kind::Video, f.timestamp_us);
                if let Some(v) = self.video.as_mut() {
                    v.packets.push_back(f);
                }
            }
            Ok(MediaPacket::Audio(a)) => {
                self.arrived(Kind::Audio, a.timestamp_us);
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

    /// Notes when a packet of a live stream arrived, and moves the
    /// timeline back to the live edge when it drifted off.
    fn arrived(&mut self, kind: Kind, pts_us: u64) {
        if let Some(meter) = self.meter.as_mut() {
            meter.record(kind, Stage::Arrival, pts_us, Duration::ZERO);
        }
        let Some(edge) = self.edge.as_mut() else {
            return;
        };
        let now = Instant::now();
        edge.observe(kind, pts_us, now);
        if !self.clock.is_anchored() {
            return;
        }
        let scheduled = self.clock.due(pts_us, now);
        match edge.drift(pts_us, scheduled) {
            Drift::Fine => return,
            Drift::Behind(by) => {
                tracing::info!(
                    behind_ms = by.as_millis(),
                    "pump: live stream behind its live edge — skipping ahead"
                );
                self.clock.shift(by, true);
            }
            Drift::Ahead(by) => {
                tracing::info!(
                    ahead_ms = by.as_millis(),
                    "pump: live stream arrives later than presented — delaying"
                );
                self.clock.shift(by, false);
            }
        }
        self.audio_stale = true;
    }

    /// Reads every packet a live stream already queued, so its timeline
    /// starts at the newest one. Yields between rounds, so that packets
    /// the protocol still holds come through as well.
    async fn read_backlog(&mut self) -> Option<PumpEnd> {
        let mut read = 0;
        while read < MAX_QUEUED_PACKETS {
            let round = match self.read_queued(MAX_QUEUED_PACKETS - read) {
                Ok(round) => round,
                Err(end) => return Some(end),
            };
            read += round;
            if round == 0 || self.input != Input::Open {
                break;
            }
            tokio::task::yield_now().await;
        }
        tracing::debug!(packets = read, "pump: live backlog read");
        None
    }

    /// Reads up to `limit` packets that are ready without waiting.
    fn read_queued(&mut self, limit: usize) -> Result<usize, PumpEnd> {
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        for n in 0..limit {
            // Polled once and dropped: reading a packet is cancellation-safe.
            let std::task::Poll::Ready(packet) = self.puller.next().as_mut().poll(&mut cx) else {
                return Ok(n);
            };
            if let Some(end) = self.on_packet(packet) {
                return Err(end);
            }
            if self.input != Input::Open {
                return Ok(n + 1);
            }
        }
        Ok(limit)
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
            v.drain(&mut self.meter);
        }
        if let Some(a) = self.audio.as_mut() {
            a.drain(&mut self.meter);
        }
    }

    /// Starts a new timeline: whatever audio the sink still queues
    /// belongs to the old one and is dropped.
    async fn restart_timeline(&mut self) {
        self.flush_sink_audio().await;
        self.clock.reset();
        self.position.restart();
        self.audio_primed = false;
        self.audio_stale = false;
        if let Some(edge) = self.edge.as_mut() {
            edge.reset();
        }
    }

    async fn flush_sink_audio(&mut self) {
        if let Err(e) = self.sink.flush_audio().await {
            tracing::warn!(%e, "pump: sink.flush_audio failed");
        }
    }

    /// The earliest due item, anchoring the timeline if needed.
    fn next_due(&mut self) -> Option<(Instant, Kind)> {
        let video_pts = self.video.as_ref().and_then(Stream::front_pts);
        let audio_pts = self.audio.as_ref().and_then(Stream::front_pts);
        let now = Instant::now();
        if !self.clock.is_anchored() {
            let first = video_pts.into_iter().chain(audio_pts).min()?;
            match self.edge.as_ref().and_then(|edge| edge.due(first, now)) {
                // Live: at the live edge, even if the first item is then
                // too late to present.
                Some(at) => self.clock.anchor_due(first, at),
                None => self.clock.anchor_at(first, now),
            }
        }
        let video = video_pts.map(|p| (self.clock.due(p, now), Kind::Video));
        let audio = audio_pts.map(|p| {
            let due = self.clock.due(p, now);
            (due.checked_sub(self.lead).unwrap_or(due), Kind::Audio)
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
    async fn present(&mut self, which: Kind) {
        match which {
            Kind::Video => self.present_video().await,
            Kind::Audio => self.present_audio().await,
        }
    }

    async fn present_video(&mut self) {
        let Some(stream) = self.video.as_mut() else {
            return;
        };
        let Some(frame) = stream.ready.pop_front() else {
            return;
        };
        let pts = frame.timestamp_us();
        let late = Instant::now().saturating_duration_since(self.clock.due(pts, Instant::now()));
        // A live stream is kept at its edge by the arrivals instead.
        if late > RESYNC_LATENESS && self.edge.is_none() {
            tracing::debug!(?late, "pump: stalled — re-anchoring");
            stream.ready.push_front(frame);
            self.restart_timeline().await;
            return;
        }
        if late > MAX_VIDEO_LATENESS {
            self.late_drops += 1;
            if let Some(meter) = self.meter.as_mut() {
                meter.stale(Kind::Video);
            }
            return;
        }
        self.position.presented(pts, Duration::ZERO);
        if let Some(meter) = self.meter.as_mut() {
            meter.record(Kind::Video, Stage::Output, pts, Duration::ZERO);
        }
        if let Err(e) = self.sink.push_video(frame).await {
            tracing::warn!(%e, "pump: sink.push_video failed");
        }
    }

    async fn present_audio(&mut self) {
        let Some(stream) = self.audio.as_mut() else {
            return;
        };
        let Some(pcm) = stream.ready.pop_front() else {
            return;
        };
        let now = Instant::now();
        let due = self.clock.due(pcm.timestamp_us, now);
        let late = now.saturating_duration_since(due.checked_sub(self.lead).unwrap_or(due));
        let pcm = if self.edge.is_some() {
            match self.prime_live_audio(pcm, due, late, now).await {
                Some(pcm) => pcm,
                None => return,
            }
        } else {
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
                self.push_audio(silence(&pcm, self.lead)).await;
            }
            pcm
        };
        if self.video.is_none() {
            self.position.presented(pcm.timestamp_us, self.lead);
        }
        if let Some(meter) = self.meter.as_mut() {
            // Heard once what the sink holds has played, or, when it
            // cannot tell, at its presentation time.
            let ahead = self
                .sink
                .audio_backlog()
                .unwrap_or_else(|| due.saturating_duration_since(now));
            meter.record(Kind::Audio, Stage::Output, pcm.timestamp_us, ahead);
        }
        self.push_audio(pcm).await;
    }

    /// Live audio: what of `pcm` to push now, priming the sink with
    /// silence up to its presentation time `due` when the sink starts
    /// over. Audio too late to be heard on time is dropped, and so is
    /// what the sink holds when it no longer lines up with the timeline
    /// (the timeline moved, the sink queued audio before its device
    /// opened, or it ran dry).
    async fn prime_live_audio(
        &mut self,
        pcm: DecodedAudio,
        due: Instant,
        late: Duration,
        now: Instant,
    ) -> Option<DecodedAudio> {
        let backlog = self.sink.audio_backlog();
        let off_timeline = backlog.is_some_and(|b| b.abs_diff(self.lead) > LIVE_MAX_AUDIO_ERROR);
        if self.audio_stale || (self.audio_primed && (late > self.lead || off_timeline)) {
            if self.audio_primed && !self.audio_stale {
                tracing::debug!(
                    ?late,
                    ?backlog,
                    "pump: live audio off its timeline — starting over"
                );
            }
            self.audio_stale = false;
            self.audio_primed = false;
            self.flush_sink_audio().await;
        }
        if self.audio_primed {
            if let Some(backlog) = backlog {
                self.clock.follow_audio(backlog, now);
            }
            return Some(pcm);
        }
        let pcm = if let Some(gap) = due.checked_duration_since(now) {
            if !gap.is_zero() {
                self.push_audio(silence(&pcm, gap)).await;
            }
            Some(pcm)
        } else {
            // Already due: only the part still ahead is pushed.
            let late_us = u64::try_from(now.duration_since(due).as_micros()).unwrap_or(u64::MAX);
            let from_us = pcm.timestamp_us.saturating_add(late_us);
            trim_audio(pcm, from_us)
        };
        let Some(pcm) = pcm else {
            if let Some(meter) = self.meter.as_mut() {
                meter.stale(Kind::Audio);
            }
            return None;
        };
        self.audio_primed = true;
        Some(pcm)
    }

    async fn push_audio(&mut self, pcm: DecodedAudio) {
        if let Err(e) = self.sink.push_audio(pcm).await {
            tracing::warn!(%e, "pump: sink.push_audio failed");
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
    let frames = u64::from(like.sample_rate)
        * u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
        / 1000;
    let samples = usize::try_from(frames).unwrap_or(0) * usize::from(like.channels);
    DecodedAudio {
        pcm: vec![0; samples],
        sample_rate: like.sample_rate,
        channels: like.channels,
        timestamp_us: like
            .timestamp_us
            .saturating_sub(u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)),
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
