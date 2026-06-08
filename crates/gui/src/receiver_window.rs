//! Receiver window — one Freya top-level window per incoming cast.
//!
//! Three subsystems work together:
//!
//! 1. [`WindowSink`] — the [`FrameSink`] the manager pipeline pipes
//!    decoded frames into. It splits the stream: BGRA video frames
//!    go to a video mpsc; PCM s16le audio goes to an audio mpsc.
//! 2. [`ReceiverWindowApp`] — the Freya `App` that owns the audio
//!    output (rodio `Sink` over PipeWire), the latest video frame
//!    slot, and a [`canvas`] element that blits the latest decoded
//!    frame using Skia each render pass.
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
//! Video render uses Freya's [`canvas`] which exposes a Skia
//! `SkCanvas` per render. We build a raster `SkImage` wrapping the
//! latest BGRA buffer and draw it scaled to the canvas area. Skia
//! does the resampling on whatever backend (Vulkan / OpenGL) the
//! window happens to be running on.

use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use ferricast::{
    AudioCodec, CapturedFrame, Codec, DecodedAudio, FerricastError, FrameSink, MediaInfo,
    PixelFormat, RemoteSender, Result,
};
use freya::engine::prelude::{
    AlphaType, ColorType, Data, ISize, Image as SkImage, ImageInfo, Paint, SamplingOptions,
};
use freya::prelude::*;
use freya::winit::window::WindowLevel;
use tokio::sync::mpsc;

/// Frame counters surfaced to the receiver-side telemetry — kept
/// public because `main.rs` collects them across active sessions
/// for a debugging overlay we may add later.
#[derive(Default, Debug)]
pub struct ReceiverCounters {
    pub video_frames: AtomicU64,
    pub audio_frames: AtomicU64,
}

/// Cross-thread payload for one decoded video frame. Skia images
/// can't be `Send` across thread boundaries in all configurations,
/// so we ship raw bytes + dims and reconstruct the `SkImage` inside
/// the render closure on the Freya thread.
#[derive(Clone, Debug)]
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    /// Tightly packed BGRA8888 — what openh264's CPU path emits
    /// after we swap R↔B in `OpenH264Decoder::decode`. When a GPU
    /// decoder lands and produces `CapturedFrame::Gpu` (DMA-BUF),
    /// the sink readbacks here too — the canvas only knows about
    /// the CPU shape.
    pub bgra: Bytes,
    pub stride: u32,
}

/// Cross-thread payload for one decoded audio buffer. Already s16le
/// interleaved as `Vec<i16>` so rodio's [`rodio::buffer::SamplesBuffer`]
/// can take it without further conversion.
#[derive(Clone, Debug)]
pub struct AudioBuf {
    pub samples: Arc<Vec<i16>>,
    pub sample_rate: u32,
    pub channels: u16,
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
        }
    }
}

#[async_trait::async_trait]
impl FrameSink for WindowSink {
    async fn push_video(&mut self, frame: CapturedFrame) -> Result<()> {
        let raw = frame.into_cpu()?;
        if raw.format != PixelFormat::Bgra {
            // Future GPU decoders may emit NV12; convert there when
            // they land. The openh264 path always produces BGRA so
            // we don't carry a conversion table here yet.
            return Err(FerricastError::Decode(format!(
                "WindowSink: expected BGRA frame, got {:?}",
                raw.format
            )));
        }
        let n = self.counters.video_frames.fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_multiple_of(120) {
            tracing::debug!(
                receiver = %self.label,
                count = n,
                width = raw.width,
                height = raw.height,
                "WindowSink: video"
            );
        }
        let frame = VideoFrame {
            width: raw.width,
            height: raw.height,
            stride: raw.stride,
            bgra: raw.data,
        };
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
        // Pack the s16le bytes into Vec<i16>. We could keep them as
        // bytes and have the audio task interleave — but rodio's
        // `SamplesBuffer` constructor wants typed samples, so doing
        // the cast here keeps the audio task simple.
        let samples: Vec<i16> = audio
            .pcm
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        let buf = AudioBuf {
            samples: Arc::new(samples),
            sample_rate: audio.sample_rate,
            channels: audio.channels,
        };
        // Audio backpressure is real: dropping audio buffers
        // produces audible glitches. Bound to 64 and `send` rather
        // than `try_send` so the decoder waits if the playback task
        // falls behind.
        let _ = self.audio_tx.send(buf).await;
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
    fn render(&self) -> impl IntoElement {
        let remote = self.remote.clone();
        let info = self.info.clone();
        let counters = self.counters.clone();

        // Shared video slot: the drain task writes the most recent
        // frame, the canvas closure reads it and draws. `StdMutex`
        // (not tokio) because the canvas closure runs sync on the
        // Freya render thread and we don't want to wait on a tokio
        // lock there.
        let video_slot: Arc<StdMutex<Option<VideoFrame>>> = Arc::new(StdMutex::new(None));

        // Tick signal: drain task increments per video frame. The
        // render() body reads it so Freya schedules a redraw on
        // each frame and the canvas closure re-runs.
        let mut tick = use_state(|| 0u64);
        let _ = tick.read();

        // Drain video into the slot.
        let video_rx_take = self.video_rx.clone();
        let video_slot_for_task = video_slot.clone();
        use_hook(move || {
            if let Some(mut rx) = video_rx_take.borrow_mut().take() {
                let slot = video_slot_for_task;
                spawn(async move {
                    while let Some(frame) = rx.recv().await {
                        if let Ok(mut g) = slot.lock() {
                            *g = Some(frame);
                        }
                        *tick.write() += 1;
                    }
                });
            }
        });

        // Drain audio into rodio. The `OutputStream` has to live
        // for the whole window lifetime — drop it and PipeWire
        // tears the stream down mid-playback. Stash it in a hook
        // closure so it's owned by the task.
        let audio_rx_take = self.audio_rx.clone();
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
                    while let Some(buf) = rx.recv().await {
                        let samples: Vec<i16> = (*buf.samples).clone();
                        let source = rodio::buffer::SamplesBuffer::new(
                            buf.channels,
                            buf.sample_rate,
                            samples,
                        );
                        sink.append(source);
                    }
                });
            }
        });

        let video_n = counters.video_frames.load(Ordering::Relaxed);
        let audio_n = counters.audio_frames.load(Ordering::Relaxed);

        let title = receiver_title(&remote, &info);
        let codec_line = codec_line(&info);
        let video_meta = info
            .video
            .as_ref()
            .map(|v| {
                format!(
                    "{} — {} frames",
                    codec_label_video(v.codec),
                    video_n
                )
            })
            .unwrap_or_else(|| "no video".to_string());
        let audio_meta = info
            .audio
            .as_ref()
            .map(|a| {
                format!(
                    "{} {}Hz/{}ch — {} frames",
                    codec_label_audio(a.codec),
                    a.sample_rate,
                    a.channels,
                    audio_n
                )
            })
            .unwrap_or_else(|| "no audio".to_string());

        let render_slot = video_slot.clone();
        let on_render = RenderCallback::new(move |ctx| {
            let Ok(g) = render_slot.lock() else { return };
            let Some(frame) = g.as_ref() else { return };
            // Build a raster SkImage that wraps the bytes — Skia
            // copies into its own surface on draw, so we don't have
            // to keep the bytes alive past this closure call.
            let info = ImageInfo::new(
                ISize::new(frame.width as i32, frame.height as i32),
                ColorType::BGRA8888,
                AlphaType::Premul,
                None,
            );
            let data = Data::new_copy(&frame.bgra);
            let Some(image) = SkImage::from_raster_data(&info, data, frame.stride as usize)
            else {
                return;
            };
            // Use the visible-area dimensions Freya gave us; the
            // canvas() layout determines the rect. We scale-to-fit
            // letterbox-style by computing the largest src→dst
            // aspect-preserving rectangle.
            let area = ctx.layout_node.visible_area();
            let dst = aspect_fit(
                area.width(),
                area.height(),
                frame.width as f32,
                frame.height as f32,
            );
            let paint = Paint::default();
            ctx.canvas.draw_image_rect_with_sampling_options(
                &image,
                None,
                freya::engine::prelude::SkRect::new(
                    dst.0,
                    dst.1,
                    dst.0 + dst.2,
                    dst.1 + dst.3,
                ),
                SamplingOptions::default(),
                &paint,
            );
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
                    .child(canvas(on_render).expanded()),
            )
            .child(
                // Metadata strip across the bottom.
                rect()
                    .width(Size::fill())
                    .background((18, 18, 24))
                    .padding(Gaps::new(12., 16., 12., 16.))
                    .vertical()
                    .spacing(4.)
                    .child(
                        label()
                            .text(title)
                            .font_size(16.)
                            .color((230, 230, 240)),
                    )
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
                            .child(
                                label()
                                    .text(video_meta)
                                    .font_size(11.)
                                    .color((200, 200, 220)),
                            )
                            .child(
                                label()
                                    .text(audio_meta)
                                    .font_size(11.)
                                    .color((200, 200, 220)),
                            ),
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

/// Aspect-fit `src` into `dst`, return `(x, y, w, h)` of the inset
/// rectangle. Used by the canvas closure to letterbox the decoded
/// frame inside the visible-area Freya measured for the canvas.
fn aspect_fit(
    dst_w: f32,
    dst_h: f32,
    src_w: f32,
    src_h: f32,
) -> (f32, f32, f32, f32) {
    if src_w <= 0.0 || src_h <= 0.0 || dst_w <= 0.0 || dst_h <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let src_aspect = src_w / src_h;
    let dst_aspect = dst_w / dst_h;
    let (w, h) = if src_aspect > dst_aspect {
        (dst_w, dst_w / src_aspect)
    } else {
        (dst_h * src_aspect, dst_h)
    };
    let x = (dst_w - w) / 2.0;
    let y = (dst_h - h) / 2.0;
    (x, y, w, h)
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
