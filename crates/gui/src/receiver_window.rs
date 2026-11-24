//! Receiver window — one Freya top-level window per incoming cast.
//!
//! Three subsystems work together:
//!
//! 1. [`WindowSink`] — the [`FrameSink`] the manager pipeline pipes
//!    decoded frames into. It asks the decoders for NV12 and splits
//!    the stream: video frames go to a video mpsc; PCM s16le audio
//!    goes to an audio mpsc.
//! 2. [`ReceiverWindowApp`] — the Freya `App` that owns the audio
//!    output (rodio `Sink` over `PipeWire`), the latest video frame
//!    slot, and a render callback that draws that frame with Skia
//!    each render pass.
//! 3. Window opener ([`open_receiver_window`]) — wires the rx ends
//!    into the App and calls `Platform::launch_window`.
//!
//! Audio output uses rodio because it bundles cpal + symphonia and
//! gives us a no-fuss `Sink::append(SamplesBuffer)`. We pre-decode
//! AAC → PCM in the pipeline (`ferricast-decoder::AacDecoder`) and
//! hand rodio bare PCM, so the `symphonia-aac` feature isn't on
//! rodio's decode path today — it's there to keep the door open for
//! a future "skip the decoder, feed rodio ADTS directly" shortcut.
//!
//! Video drawing lives in [`video`]: the YCbCr → RGB conversion and
//! the scaling to the window both run on the GPU.

mod video;

use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use ferricast::{
    AudioCodec, CapturedFrame, Codec, DecodedAudio, FrameSink, MediaInfo, PixelFormat,
    RemoteSender, Result,
};
use freya::prelude::*;
use freya::winit::window::WindowLevel;
use tokio::sync::mpsc;

pub use video::VideoFrame;
use video::{FrameSlot, VideoRenderer};

/// Frame counters surfaced to the receiver-side telemetry — kept
/// public because `main.rs` collects them across active sessions
/// for a debugging overlay we may add later.
#[derive(Debug)]
pub struct ReceiverCounters {
    pub video_frames: AtomicU64,
    pub audio_frames: AtomicU64,
    /// Audio output gain as `f32` bits, written by the sink when the
    /// sender changes volume or mute and applied by the window's
    /// audio task.
    audio_gain: AtomicU32,
    /// Interleaved samples the sink handed to the window.
    audio_queued: AtomicU64,
    /// Interleaved samples the output device consumed (or the window
    /// dropped); `audio_queued - audio_played` is the backlog.
    audio_played: AtomicU64,
    /// Interleaved samples per second of the latest buffer.
    audio_rate: AtomicU32,
    /// Whether an output device is playing what is queued.
    audio_output: AtomicBool,
    /// Bumped by every flush: buffers and playing sources of an older
    /// epoch are stale and neither played nor counted.
    audio_epoch: AtomicU64,
    /// Wakes the audio task to empty the output queue after a flush.
    audio_flushed: tokio::sync::Notify,
}

impl Default for ReceiverCounters {
    fn default() -> Self {
        Self {
            video_frames: AtomicU64::new(0),
            audio_frames: AtomicU64::new(0),
            audio_gain: AtomicU32::new(1.0f32.to_bits()),
            audio_queued: AtomicU64::new(0),
            audio_played: AtomicU64::new(0),
            audio_rate: AtomicU32::new(0),
            audio_output: AtomicBool::new(false),
            audio_epoch: AtomicU64::new(0),
            audio_flushed: tokio::sync::Notify::new(),
        }
    }
}

impl ReceiverCounters {
    pub fn audio_gain(&self) -> f32 {
        f32::from_bits(self.audio_gain.load(Ordering::Relaxed))
    }

    pub fn set_audio_gain(&self, gain: f32) {
        self.audio_gain
            .store(gain.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }

    /// Audio queued for the device but not played yet, once a device is
    /// open and the format is known.
    #[expect(
        clippy::cast_precision_loss,
        reason = "a sample backlog, far below 2^52"
    )]
    fn audio_backlog(&self) -> Option<Duration> {
        let rate = self.audio_rate.load(Ordering::Relaxed);
        if rate == 0 || !self.audio_output.load(Ordering::Relaxed) {
            return None;
        }
        let queued = self.audio_queued.load(Ordering::Relaxed);
        let played = self.audio_played.load(Ordering::Relaxed);
        Some(Duration::from_secs_f64(
            queued.saturating_sub(played) as f64 / f64::from(rate),
        ))
    }

    fn add_played(&self, samples: usize) {
        self.audio_played
            .fetch_add(samples as u64, Ordering::Relaxed);
    }

    fn audio_epoch(&self) -> u64 {
        self.audio_epoch.load(Ordering::Relaxed)
    }

    /// Forgets every queued sample: the backlog drops to zero and the
    /// audio task empties the output device queue.
    fn flush_audio(&self) {
        self.audio_epoch.fetch_add(1, Ordering::Relaxed);
        self.audio_played
            .store(self.audio_queued.load(Ordering::Relaxed), Ordering::Relaxed);
        self.audio_flushed.notify_one();
    }
}

/// Cross-thread payload for one decoded audio buffer. Already s16
/// interleaved as `Vec<i16>` so rodio's [`rodio::buffer::SamplesBuffer`]
/// can take it without further conversion. The samples vec moves
/// through the channel directly (no `Arc` clone) — the audio drain
/// task is the single consumer.
#[derive(Debug)]
pub struct AudioBuf {
    pub samples: Vec<i16>,
    pub sample_rate: u32,
    pub channels: u16,
    /// Flush epoch the buffer was queued in.
    epoch: u64,
}

/// The sink the manager pipeline writes into. Owns the tx ends of
/// the per-window video / audio channels. When the receiver
/// disconnects the channels close on drop; the window drains
/// whatever was pending and quiesces.
pub struct WindowSink {
    counters: Arc<ReceiverCounters>,
    label: String,
    video_tx: mpsc::Sender<VideoFrame>,
    audio_tx: mpsc::Sender<AudioBuf>,
    /// Channels of the output device the window plays to.
    output_channels: u16,
}

impl WindowSink {
    pub fn new(
        label: impl Into<String>,
        counters: Arc<ReceiverCounters>,
        video_tx: mpsc::Sender<VideoFrame>,
        audio_tx: mpsc::Sender<AudioBuf>,
    ) -> Self {
        Self {
            counters,
            label: label.into(),
            video_tx,
            audio_tx,
            output_channels: default_output_channels(),
        }
    }
}

/// Channels of the default output device, the one
/// `rodio::OutputStream::try_default` opens; stereo when it cannot be
/// queried.
fn default_output_channels() -> u16 {
    use rodio::cpal::traits::{DeviceTrait, HostTrait};
    rodio::cpal::default_host()
        .default_output_device()
        .and_then(|device| device.default_output_config().ok())
        .map_or(2, |config| config.channels())
}

/// Where one input channel goes in a stereo downmix.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Gains {
    left: f32,
    right: f32,
}

/// ITU-R BS.775 stereo downmix of 3 to 8 interleaved channels in WAV
/// order (the order libfdk-aac and symphonia output): centre and
/// surrounds go to their side at -3 dB, LFE is dropped, and the result
/// is scaled so a full-scale signal on every contributing channel
/// cannot clip.
#[derive(Debug, Clone, PartialEq)]
struct StereoDownmix {
    gains: Vec<Gains>,
}

impl StereoDownmix {
    /// The downmix for `channels` input channels; `None` when there is
    /// nothing to mix (mono, stereo) or the layout is unknown.
    fn new(channels: u16) -> Option<Self> {
        const K: f32 = std::f32::consts::FRAC_1_SQRT_2;
        const L: Gains = Gains {
            left: 1.0,
            right: 0.0,
        };
        const R: Gains = Gains {
            left: 0.0,
            right: 1.0,
        };
        const C: Gains = Gains { left: K, right: K };
        const LFE: Gains = Gains {
            left: 0.0,
            right: 0.0,
        };
        const LS: Gains = Gains {
            left: K,
            right: 0.0,
        };
        const RS: Gains = Gains {
            left: 0.0,
            right: K,
        };
        let layout: &[Gains] = match channels {
            // FL FR FC
            3 => &[L, R, C],
            // FL FR FC BC
            4 => &[L, R, C, C],
            // FL FR FC BL BR
            5 => &[L, R, C, LS, RS],
            // FL FR FC LFE BL BR
            6 => &[L, R, C, LFE, LS, RS],
            // FL FR FC LFE BC SL SR
            7 => &[L, R, C, LFE, C, LS, RS],
            // FL FR FC LFE BL BR SL SR
            8 => &[L, R, C, LFE, LS, RS, LS, RS],
            _ => return None,
        };
        let total: f32 = layout.iter().map(|g| g.left).sum();
        let gains = layout
            .iter()
            .map(|g| Gains {
                left: g.left / total,
                right: g.right / total,
            })
            .collect();
        Some(Self { gains })
    }

    /// Mixes interleaved `pcm` down to interleaved stereo.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "clamped to the i16 range first"
    )]
    fn apply(&self, pcm: &[i16]) -> Vec<i16> {
        let to_i16 = |s: f32| s.round().clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16;
        pcm.chunks_exact(self.gains.len())
            .flat_map(|frame| {
                let (left, right) = frame
                    .iter()
                    .zip(&self.gains)
                    .fold((0.0f32, 0.0f32), |(l, r), (&s, g)| {
                        (l + f32::from(s) * g.left, r + f32::from(s) * g.right)
                    });
                [to_i16(left), to_i16(right)]
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl FrameSink for WindowSink {
    fn video_format(&self) -> PixelFormat {
        // The window converts YCbCr on the GPU. P010 is the preference for
        // 10-bit streams (they keep their depth and HDR signalling); 8-bit
        // ones still arrive as NV12.
        PixelFormat::P010
    }

    async fn push_video(&mut self, frame: CapturedFrame) -> Result<()> {
        let frame = VideoFrame::try_from(frame.into_cpu()?)?;
        let n = self.counters.video_frames.fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_multiple_of(120) {
            tracing::debug!(
                receiver = %self.label,
                count = n,
                width = frame.width,
                height = frame.height,
                "WindowSink: video"
            );
        }
        // try_send drops on backpressure: a frame older than the
        // newest in flight is useless for display anyway, and
        // blocking the decoder would back-pressure into the puller
        // and then into the sender (bad UX — the sender notices a
        // stuck connection and disconnects). Drop instead.
        let _ = self.video_tx.try_send(frame);
        Ok(())
    }

    async fn push_audio(&mut self, audio: DecodedAudio) -> Result<()> {
        let n = self.counters.audio_frames.fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_multiple_of(240) {
            tracing::debug!(
                receiver = %self.label,
                count = n,
                sample_rate = audio.sample_rate,
                channels = audio.channels,
                "WindowSink: audio"
            );
        }
        // rodio fits more channels than the device has by dropping the
        // extra ones, centre and surrounds included: mix them in first.
        let (pcm, channels) = match StereoDownmix::new(audio.channels) {
            Some(downmix) if audio.channels > self.output_channels => {
                (downmix.apply(&audio.pcm), 2)
            }
            _ => (audio.pcm, audio.channels),
        };
        let samples = pcm.len() as u64;
        let rate = audio.sample_rate * u32::from(channels);
        let buf = AudioBuf {
            samples: pcm,
            sample_rate: audio.sample_rate,
            channels,
            epoch: self.counters.audio_epoch(),
        };
        // Drop on backpressure (matches video). The window-side
        // drain task is bottlenecked by rodio's playback clock —
        // `Sink::append` only blocks the decoder if the channel
        // fills past the drain task's pace. We used to `send().await`
        // here so audio frames never dropped, but a slow window
        // startup (Freya + Skia + alsa init, hundreds of ms) drained
        // the puller via the manager pump and tripped HLS segment
        // eviction on the sender. Capacity 256 (~5 s @ 21 ms/buffer)
        // absorbs the startup spike; under steady-state overload
        // we'd rather glitch audio than stall the whole pipeline.
        match self.audio_tx.try_send(buf) {
            Ok(()) => {
                self.counters.audio_rate.store(rate, Ordering::Relaxed);
                self.counters
                    .audio_queued
                    .fetch_add(samples, Ordering::Relaxed);
            }
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                tracing::warn!(receiver = %self.label, "audio buffer dropped — channel full");
            }
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {}
        }
        Ok(())
    }

    async fn set_volume(&mut self, gain: f32) -> Result<()> {
        self.counters.set_audio_gain(gain);
        Ok(())
    }

    fn audio_backlog(&self) -> Option<Duration> {
        self.counters.audio_backlog()
    }

    async fn flush_audio(&mut self) -> Result<()> {
        self.counters.flush_audio();
        Ok(())
    }
}

#[derive(Clone)]
pub struct ReceiverWindowApp {
    remote: RemoteSender,
    info: MediaInfo,
    counters: Arc<ReceiverCounters>,
    /// `Rc<RefCell<Option<_>>>` because the inner receivers aren't
    /// `Clone` and `App::render` takes `&self`. First mount takes
    /// them out; subsequent renders find `None` and skip the
    /// drain-task setup.
    video_rx: std::rc::Rc<std::cell::RefCell<Option<mpsc::Receiver<VideoFrame>>>>,
    audio_rx: std::rc::Rc<std::cell::RefCell<Option<mpsc::Receiver<AudioBuf>>>>,
}

impl ReceiverWindowApp {
    pub fn new(
        remote: RemoteSender,
        info: MediaInfo,
        counters: Arc<ReceiverCounters>,
        video_rx: mpsc::Receiver<VideoFrame>,
        audio_rx: mpsc::Receiver<AudioBuf>,
    ) -> Self {
        Self {
            remote,
            info,
            counters,
            video_rx: std::rc::Rc::new(std::cell::RefCell::new(Some(video_rx))),
            audio_rx: std::rc::Rc::new(std::cell::RefCell::new(Some(audio_rx))),
        }
    }
}

impl App for ReceiverWindowApp {
    #[expect(
        clippy::too_many_lines,
        reason = "the window's layout reads top to bottom"
    )]
    fn render(&self) -> impl IntoElement {
        let remote = self.remote.clone();
        let info = self.info.clone();
        let counters = self.counters.clone();

        // Shared video slot: the drain task writes the most recent
        // frame, the render callback reads it and draws. Stored in
        // `use_hook` so the same Arc is returned on every render
        // pass — if we allocate a fresh slot each render the task
        // keeps writing to the first-render slot while the callback
        // reads from the current-render slot (always None).
        // `StdMutex` because the render callback runs sync on the
        // Freya render thread.
        let video_slot = use_hook(|| Arc::new(StdMutex::new(FrameSlot::default())));

        // Tick signal: drain task increments per video frame. The
        // render() body reads it so Freya schedules a redraw on
        // each frame and the canvas closure re-runs.
        let mut tick = use_state(|| 0u64);
        let _ = tick.read();

        // Stream-ended signal: flipped by whichever drain task sees
        // its channel close first (sender disconnect, manager pump
        // stop, or normal end-of-stream). The render branch below
        // shows a "Stream ended" banner and grays out the metadata.
        // Keeping the last frame visible is intentional — closing
        // the window or hiding the canvas would feel like the
        // receiver app crashed; freezing on the last frame is the
        // expected receiver-side UX (matches a Chromecast TV showing
        // the last frame when the casting phone leaves).
        let ended = use_state(|| false);
        let ended_now = *ended.read();

        // Drain video into the slot.
        let video_rx_take = self.video_rx.clone();
        let video_slot_for_task = video_slot.clone();
        let mut ended_for_video = ended;
        use_hook(move || {
            if let Some(mut rx) = video_rx_take.borrow_mut().take() {
                let slot = video_slot_for_task;
                spawn(async move {
                    while let Some(frame) = rx.recv().await {
                        tracing::trace!(pts_us = frame.pts_us, "receiver window: frame");
                        if let Ok(mut g) = slot.lock() {
                            g.replace(frame);
                        }
                        *tick.write() += 1;
                    }
                    *ended_for_video.write() = true;
                    *tick.write() += 1;
                });
            }
        });

        // Drain audio into rodio. The `OutputStream` has to live
        // for the whole window lifetime — drop it and PipeWire
        // tears the stream down mid-playback. Stash it in a hook
        // closure so it's owned by the task.
        let audio_rx_take = self.audio_rx.clone();
        let audio_counters = counters.clone();
        let mut ended_for_audio = ended;
        use_hook(move || {
            if let Some(mut rx) = audio_rx_take.borrow_mut().take() {
                spawn(async move {
                    let (stream, handle) = match rodio::OutputStream::try_default() {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::warn!(%e, "no rodio default output — receiver window will be silent");
                            return;
                        }
                    };
                    let sink = match rodio::Sink::try_new(&handle) {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::warn!(%e, "rodio sink init failed");
                            return;
                        }
                    };
                    // Keep `stream` alive: drop = device closed.
                    let _stream_keepalive = stream;
                    audio_counters.audio_output.store(true, Ordering::Relaxed);
                    loop {
                        let buf = tokio::select! {
                            buf = rx.recv() => buf,
                            () = audio_counters.audio_flushed.notified() => {
                                // Stale audio (before a seek or a pause) must
                                // not be heard; `clear` also pauses the sink.
                                sink.clear();
                                sink.play();
                                continue;
                            }
                        };
                        let Some(buf) = buf else { break };
                        if buf.epoch != audio_counters.audio_epoch() {
                            continue;
                        }
                        // Drop frames when the sink queue is too deep
                        // (~2 s worth). Rodio's internal queue is
                        // unbounded; if playback stalls (PipeWire
                        // disconnect, device switch) the queue grows
                        // without limit. 100 × ~20 ms ≈ 2 s headroom
                        // is enough to absorb startup latency without
                        // accumulating an ever-growing backlog.
                        if sink.len() > 100 {
                            audio_counters.add_played(buf.samples.len());
                            continue;
                        }
                        sink.set_volume(audio_counters.audio_gain());
                        let source = rodio::buffer::SamplesBuffer::new(
                            buf.channels,
                            buf.sample_rate,
                            buf.samples,
                        );
                        sink.append(Counted {
                            inner: source,
                            counters: audio_counters.clone(),
                            epoch: buf.epoch,
                        });
                    }
                    audio_counters.audio_output.store(false, Ordering::Relaxed);
                    *ended_for_audio.write() = true;
                });
            }
        });

        let video_n = counters.video_frames.load(Ordering::Relaxed);
        let audio_n = counters.audio_frames.load(Ordering::Relaxed);

        let raw_title = receiver_title(&remote, &info);
        let title = if ended_now {
            format!("● Stream ended — {raw_title}")
        } else {
            raw_title
        };
        let codec_line = if ended_now {
            "sender disconnected — last frame frozen".to_string()
        } else {
            codec_line(&info)
        };
        let title_color = if ended_now {
            (200, 140, 140)
        } else {
            (230, 230, 240)
        };
        let meta_color = if ended_now {
            (120, 120, 130)
        } else {
            (200, 200, 220)
        };
        let video_meta = info.video.as_ref().map_or_else(
            || "no video".to_string(),
            |v| format!("{} — {} frames", codec_label_video(v.codec), video_n),
        );
        let audio_meta = info.audio.as_ref().map_or_else(
            || "no audio".to_string(),
            |a| {
                format!(
                    "{} {}Hz/{}ch — {} frames",
                    a.source.map_or(codec_label_audio(a.codec), |s| s.label()),
                    a.sample_rate,
                    a.channels,
                    audio_n
                )
            },
        );

        let audio_only = info.video.is_none();
        // One renderer per window, kept across renders so a frame is
        // uploaded once however often the window redraws.
        let render_slot = video_slot.clone();
        let on_render = use_hook(move || {
            let mut renderer = VideoRenderer::default();
            RenderCallback::new(move |ctx| {
                let RenderTarget::Background { canvas, .. } = &ctx.target else {
                    return;
                };
                let Ok(slot) = render_slot.lock() else { return };
                renderer.draw(canvas, &slot, ctx.size.width, ctx.size.height);
            })
        });
        rect()
            .expanded()
            .background((10, 10, 14))
            .vertical()
            .child(
                // Video region — fills as much height as possible.
                rect()
                    .width(Size::fill())
                    .height(Size::flex(1.))
                    .background((0, 0, 0))
                    .center()
                    // Audio-only media: no picture will come, say so
                    // instead of leaving the region blank.
                    .maybe_child(audio_only.then(|| {
                        label()
                            .text("Audio only")
                            .font_size(20.)
                            .color((150, 150, 170))
                    }))
                    .maybe_child((!audio_only).then(|| rect().expanded().background(on_render))),
            )
            .child(
                // Metadata strip across the bottom.
                rect()
                    .width(Size::fill())
                    .background((18, 18, 24))
                    .padding(Gaps::new(12., 16., 12., 16.))
                    .vertical()
                    .spacing(4.)
                    .child(label().text(title).font_size(16.).color(title_color))
                    .child(
                        label()
                            .text(codec_line)
                            .font_size(11.)
                            .color((160, 160, 180)),
                    )
                    .child(
                        rect()
                            .horizontal()
                            .spacing(16.)
                            .child(label().text(video_meta).font_size(11.).color(meta_color))
                            .child(label().text(audio_meta).font_size(11.).color(meta_color)),
                    )
                    .child(
                        label()
                            .text(format!("from {}", remote.addr))
                            .font_size(10.)
                            .color((130, 130, 150)),
                    ),
            )
    }
}

/// Launch a top-level receiver window. Returns the window id so the
/// pump can close it when the receiver stops.
pub async fn open_receiver_window(
    platform: Platform,
    remote: RemoteSender,
    info: MediaInfo,
    counters: Arc<ReceiverCounters>,
    video_rx: mpsc::Receiver<VideoFrame>,
    audio_rx: mpsc::Receiver<AudioBuf>,
) -> freya::winit::window::WindowId {
    let title = receiver_title(&remote, &info);
    let app = ReceiverWindowApp::new(remote, info, counters, video_rx, audio_rx);
    let config = WindowConfig::new_app(app)
        .with_size(720., 480.)
        .with_background((10, 10, 14))
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

#[expect(clippy::cast_precision_loss, reason = "a display-only duration")]
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
        Codec::Av1 => "AV1",
    }
}

fn codec_label_audio(c: AudioCodec) -> &'static str {
    match c {
        AudioCodec::Aac => "AAC",
        AudioCodec::AacEld => "AAC-ELD",
        AudioCodec::Opus => "Opus",
        AudioCodec::Pcm => "PCM",
        AudioCodec::Alac => "ALAC",
    }
}

/// A rodio source that counts the samples the output device pulls, so
/// the sink can tell how much queued audio is still unplayed.
struct Counted<S> {
    inner: S,
    counters: Arc<ReceiverCounters>,
    /// Samples of a flushed epoch are not counted: the flush already
    /// wrote them off.
    epoch: u64,
}

impl<S: Iterator<Item = i16>> Iterator for Counted<S> {
    type Item = i16;

    fn next(&mut self) -> Option<i16> {
        let sample = self.inner.next()?;
        if self.epoch == self.counters.audio_epoch() {
            self.counters.add_played(1);
        }
        Some(sample)
    }
}

impl<S: rodio::Source<Item = i16>> rodio::Source for Counted<S> {
    fn current_frame_len(&self) -> Option<usize> {
        self.inner.current_frame_len()
    }

    fn channels(&self) -> u16 {
        self.inner.channels()
    }

    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    fn total_duration(&self) -> Option<Duration> {
        self.inner.total_duration()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the gains keep both values inside i16"
    )]
    fn surround_downmix_follows_itu_gains_without_clipping() {
        let downmix = StereoDownmix::new(6).unwrap();
        let scale = 1.0 / (1.0 + 2.0 * std::f32::consts::FRAC_1_SQRT_2);
        // One frame per channel: FL, FR, FC, LFE, BL, BR at 10 000.
        let mut pcm = Vec::new();
        for ch in 0..6 {
            let mut frame = [0i16; 6];
            frame[ch] = 10_000;
            pcm.extend(frame);
        }
        let out = downmix.apply(&pcm);
        let front = (10_000.0 * scale).round() as i16;
        let side = (10_000.0 * scale * std::f32::consts::FRAC_1_SQRT_2).round() as i16;
        assert_eq!(
            out,
            [front, 0, 0, front, side, side, 0, 0, side, 0, 0, side]
        );
        // Full scale everywhere stays in range.
        let loud = downmix.apply(&[i16::MAX; 6]);
        assert!(loud.iter().all(|&s| s >= i16::MAX - 1), "{loud:?}");
    }

    #[test]
    fn mono_and_stereo_are_not_mixed() {
        assert!(StereoDownmix::new(1).is_none());
        assert!(StereoDownmix::new(2).is_none());
        assert_eq!(StereoDownmix::new(8).unwrap().gains.len(), 8);
    }
}
