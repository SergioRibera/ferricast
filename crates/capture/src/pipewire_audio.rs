//! PipeWire audio capture backend.
//!
//! Symmetric counterpart to [`crate::PipeWireCapture`]: a dedicated
//! PipeWire main loop runs on its own OS thread, pulls audio buffers
//! from either the system's default sink monitor (when [`AudioSource
//! ::DefaultMonitor`] is selected) or a caller-picked node id (when
//! [`AudioSource::Node`] is used), and forwards them as PCM frames
//! through an async channel.
//!
//! Output format is locked to S16LE / configured sample rate / configured
//! channels — the chromecast HLS pipeline expects 48 kHz stereo PCM and
//! the downstream AAC-LC encoder works on the same shape, so there's
//! no benefit to negotiating other formats here.
//!
//! Mute semantics: a shared [`AudioMuteHandle`] is read on the hot
//! path. When the flag is set the captured samples are zeroed before
//! being forwarded; the encoder keeps producing AAC at the same
//! cadence, so the audio elementary stream on the receiver side keeps
//! advancing without ever going silent in PTS terms.

use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::{Bytes, BytesMut};
use ferricast_core::{
    AudioCapture, AudioCaptureConfig, AudioCodec, AudioFrame, AudioMuteHandle, AudioSource,
    FerricastError, Result,
};
use tokio::sync::mpsc;
use tracing::{debug, error, info, trace, warn};

use pipewire as pw;
use pw::context::Context;
use pw::main_loop::MainLoop;
use pw::properties::properties;
use pw::spa::param::ParamType;
use pw::spa::param::audio::AudioInfoRaw;
use pw::spa::param::format::{FormatProperties, MediaSubtype, MediaType};
use pw::spa::param::format_utils;
use pw::spa::pod::{Object, Pod, Property, PropertyFlags, Value, serialize::PodSerializer};
use pw::spa::sys::{SPA_AUDIO_CHANNEL_FL, SPA_AUDIO_CHANNEL_FR, SPA_AUDIO_FORMAT_S16_LE};
use pw::spa::utils::{Id, SpaTypes};
use pw::stream::{Stream, StreamFlags, StreamRef, StreamState};

const BYTES_PER_SAMPLE: usize = 2; // S16LE

struct Terminate;

struct AudioWorkerHandle {
    frames: mpsc::Receiver<AudioFrame>,
    errors: mpsc::Receiver<String>,
    negotiated: Arc<NegotiatedAudio>,
    terminator: pw::channel::Sender<Terminate>,
    join: Option<std::thread::JoinHandle<()>>,
}

#[derive(Default)]
struct NegotiatedAudio {
    sample_rate: AtomicU32,
    channels: AtomicU32,
}

impl AudioWorkerHandle {
    fn shutdown(&mut self) {
        let _ = self.terminator.send(Terminate);
        if let Some(h) = self.join.take() {
            if let Err(panic) = h.join() {
                error!(?panic, "PipeWire audio worker panicked during shutdown");
            }
        }
    }
}

impl Drop for AudioWorkerHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// PipeWire audio capture. Implements [`AudioCapture`].
pub struct PipeWireAudioCapture {
    worker: Option<AudioWorkerHandle>,
}

impl PipeWireAudioCapture {
    pub fn new() -> Self {
        Self { worker: None }
    }
}

impl Default for PipeWireAudioCapture {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioCapture for PipeWireAudioCapture {
    async fn start(
        &mut self,
        source: AudioSource,
        config: AudioCaptureConfig,
        mute: AudioMuteHandle,
    ) -> Result<()> {
        if self.worker.is_some() {
            return Err(FerricastError::Capture(
                "PipeWire audio capture already running".into(),
            ));
        }
        info!(?source, ?config, "starting PipeWire audio capture");

        let worker = spawn_worker(source, config, mute)?;
        self.worker = Some(worker);
        Ok(())
    }

    async fn next_frame(&mut self) -> Result<AudioFrame> {
        let worker = self
            .worker
            .as_mut()
            .ok_or_else(|| FerricastError::Capture("audio capture not started".into()))?;

        tokio::select! {
            biased;
            err = worker.errors.recv() => {
                Err(FerricastError::Capture(
                    err.unwrap_or_else(|| "audio worker exited".into()),
                ))
            }
            frame = worker.frames.recv() => {
                frame.ok_or_else(|| FerricastError::Capture(
                    "audio frame channel closed".into(),
                ))
            }
        }
    }

    async fn stop(&mut self) -> Result<()> {
        if let Some(mut worker) = self.worker.take() {
            worker.shutdown();
        }
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.worker.is_some()
    }

    fn sample_rate(&self) -> u32 {
        self.worker
            .as_ref()
            .map(|w| w.negotiated.sample_rate.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    fn channels(&self) -> u16 {
        self.worker
            .as_ref()
            .map(|w| w.negotiated.channels.load(Ordering::Relaxed) as u16)
            .unwrap_or(0)
    }
}

fn spawn_worker(
    source: AudioSource,
    config: AudioCaptureConfig,
    mute: AudioMuteHandle,
) -> Result<AudioWorkerHandle> {
    // Moderate buffer: 32 chunks × ~21 ms ≈ 680 ms. The PipeWire
    // callback runs on the stream's main loop, which must not block;
    // `try_send` returning `Full` is the only realistic backpressure
    // response.
    // 32 is large enough to ride out short supervisor stalls (HTTP
    // read locks, segment-close ring writes) without holding so
    // many chunks in flight that, once the supervisor catches up,
    // the segmenter's staleness check drops them anyway.
    let (frame_tx, frame_rx) = mpsc::channel::<AudioFrame>(32);
    let (error_tx, error_rx) = mpsc::channel::<String>(1);
    let (term_tx, term_rx) = pw::channel::channel::<Terminate>();
    let negotiated: Arc<NegotiatedAudio> = Arc::new(NegotiatedAudio::default());
    let neg_for_thread = negotiated.clone();

    let join = std::thread::Builder::new()
        .name("ferricast-pw-audio".into())
        .spawn(move || {
            if let Err(e) = run_loop(
                source,
                config,
                mute,
                frame_tx,
                &error_tx,
                term_rx,
                neg_for_thread,
            ) {
                error!(error = %e, "PipeWire audio worker exited with error");
                let _ = error_tx.try_send(e.to_string());
            }
        })
        .map_err(|e| FerricastError::Capture(format!("spawn PW audio thread: {e}")))?;

    Ok(AudioWorkerHandle {
        frames: frame_rx,
        errors: error_rx,
        negotiated,
        terminator: term_tx,
        join: Some(join),
    })
}

struct UserData {
    sample_rate: u32,
    channels: u16,
    mute: AudioMuteHandle,
    negotiated: Arc<NegotiatedAudio>,
    first_frame_logged: bool,
    /// Sample count we've processed so far. Logged periodically so
    /// the operator can sanity-check that PipeWire's delivered
    /// sample-rate matches our claimed `sample_rate` field. After 5 s
    /// of wall time, we should have `≈ 5 × sample_rate` samples; a
    /// significant deviation means PipeWire is either resampling
    /// behind our back (wrong) or being throttled by the consumer
    /// (drops we should hear).
    log_sample_counter: u64,
    log_last_at: Option<std::time::Instant>,
    pts: PtsClock,
}

/// Max |sample-count PTS − wall clock| tolerated before re-anchoring.
/// Normal delivery jitter is well under one ~21 ms quantum; 100 ms is
/// far above that yet below the HLS segmenter's 300 ms staleness cut
/// (`MAX_AUDIO_STALENESS_US`), so a stream that fell behind gets
/// pulled back before its frames start being dropped as stale.
const PTS_REANCHOR_THRESHOLD_US: u64 = 100_000;

/// Chunk timestamps for the PCM stream.
///
/// Anchored to `now_us()` (wall clock, `UNIX_EPOCH`) so audio shares
/// a clock origin with the screen-capture path, then advanced by
/// `samples / sample_rate` per chunk so deltas are exact in
/// 1/sample-rate ticks. Counting samples alone never notices that the
/// stream paused (sink idle / suspended, xrun) or that the sound
/// card's clock drifts against the system clock (~50 ppm ≈ 180 ms/h),
/// so whenever the counted PTS strays more than
/// [`PTS_REANCHOR_THRESHOLD_US`] from the wall clock the anchor is
/// reset to "now".
#[derive(Debug, Default)]
struct PtsClock {
    anchor_us: Option<u64>,
    samples_since_anchor: u64,
}

/// Result of [`PtsClock::stamp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Stamp {
    timestamp_us: u64,
    /// `Some(counted − wall)` in µs when this chunk re-anchored the
    /// clock (positive: counted PTS was ahead of the wall clock).
    reanchored_from_drift_us: Option<i64>,
}

impl PtsClock {
    /// Timestamp a chunk of `samples` frames that arrived at
    /// `now_us`, then advance the clock past it.
    fn stamp(&mut self, samples: u64, sample_rate: u32, now_us: u64) -> Stamp {
        let rate = u64::from(sample_rate.max(1));
        let mut reanchored_from_drift_us = None;
        let timestamp_us = match self.anchor_us {
            Some(anchor) => {
                let counted = anchor
                    .saturating_add(self.samples_since_anchor.saturating_mul(1_000_000) / rate);
                let drift = i128::from(counted) - i128::from(now_us);
                if drift.unsigned_abs() > u128::from(PTS_REANCHOR_THRESHOLD_US) {
                    reanchored_from_drift_us = Some(i64::try_from(drift).unwrap_or(if drift > 0 {
                        i64::MAX
                    } else {
                        i64::MIN
                    }));
                    self.anchor_us = Some(now_us);
                    self.samples_since_anchor = 0;
                    now_us
                } else {
                    counted
                }
            }
            None => {
                self.anchor_us = Some(now_us);
                now_us
            }
        };
        self.samples_since_anchor = self.samples_since_anchor.saturating_add(samples);
        Stamp {
            timestamp_us,
            reanchored_from_drift_us,
        }
    }
}

fn run_loop(
    source: AudioSource,
    config: AudioCaptureConfig,
    mute: AudioMuteHandle,
    frame_tx: mpsc::Sender<AudioFrame>,
    error_tx: &mpsc::Sender<String>,
    term_rx: pw::channel::Receiver<Terminate>,
    negotiated: Arc<NegotiatedAudio>,
) -> Result<()> {
    pw::init();

    let mainloop =
        MainLoop::new(None).map_err(|e| FerricastError::Capture(format!("audio MainLoop: {e}")))?;
    let context = Context::new(&mainloop)
        .map_err(|e| FerricastError::Capture(format!("audio Context: {e}")))?;
    let core = context
        .connect(None)
        .map_err(|e| FerricastError::Capture(format!("audio Context::connect: {e}")))?;

    // Stream properties: capture role + monitor flag for "system
    // output". The monitor flag tells PipeWire that when the target
    // is a sink, we want its post-mix monitor stream (= what's
    // currently coming out of the speakers) rather than nothing.
    //
    // STREAM_CAPTURE_SINK = "true" is the modern equivalent of the
    // pulseaudio `pa_stream_flags_t::PA_STREAM_RECORD` + monitoring
    // a sink — it makes the default-sink monitor work without us
    // having to resolve the monitor source name explicitly.
    let mut props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Music",
        *pw::keys::STREAM_CAPTURE_SINK => "true",
        *pw::keys::NODE_NAME => "ferricast-audio-capture",
    };
    // When the caller pinned a specific node, route there. Otherwise
    // PipeWire falls back to following the default sink, which is
    // exactly the "global PC audio" behaviour the user expects.
    if let AudioSource::Node(node_id) = source {
        props.insert(*pw::keys::TARGET_OBJECT, node_id.to_string());
    }

    let stream = Stream::new(&core, "ferricast-audio-capture", props)
        .map_err(|e| FerricastError::Capture(format!("audio Stream::new: {e}")))?;

    let user_data = UserData {
        sample_rate: config.sample_rate,
        channels: config.channels,
        mute: mute.clone(),
        negotiated: negotiated.clone(),
        first_frame_logged: false,
        log_sample_counter: 0,
        log_last_at: None,
        pts: PtsClock::default(),
    };

    let _listener = stream
        .add_local_listener_with_user_data(user_data)
        .state_changed({
            let mainloop = mainloop.clone();
            let error_tx = error_tx.clone();
            move |_, _, old, new| {
                info!(?old, ?new, "audio stream state changed");
                if let StreamState::Error(msg) = new {
                    error!(%msg, "audio stream entered Error state");
                    let _ = error_tx.try_send(msg.to_string());
                    mainloop.quit();
                }
            }
        })
        .param_changed(|_, ud, id, param| handle_param_changed(ud, id, param))
        .process({
            let frame_tx = frame_tx.clone();
            move |stream, ud| handle_process(stream, ud, &frame_tx)
        })
        .register()
        .map_err(|e| FerricastError::Capture(format!("audio listener register: {e}")))?;

    let pod = build_audio_format_pod(config.sample_rate, config.channels);
    let mut pods = [pod_view(&pod)];

    // No `RT_PROCESS`: with it, `process` runs on PipeWire's data
    // thread while `param_changed` runs on this main-loop thread, and
    // both get `&mut UserData` — aliased mutable access across threads.
    // `handle_process` also allocates, logs and `try_send`s, none of
    // which is RT-safe. Without the flag PipeWire invokes `process`
    // on the main loop too; at ~21 ms quanta the extra hop is
    // irrelevant.
    stream
        .connect(
            pw::spa::utils::Direction::Input,
            None,
            StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS,
            pods.as_mut_slice(),
        )
        .map_err(|e| FerricastError::Capture(format!("audio Stream::connect: {e}")))?;
    info!("audio stream.connect ok, entering main loop");

    let _term_attach = term_rx.attach(mainloop.loop_(), {
        let mainloop = mainloop.clone();
        move |_| {
            info!("audio worker received Terminate");
            mainloop.quit();
        }
    });

    mainloop.run();

    info!("PipeWire audio main loop exited cleanly");
    Ok(())
}

fn handle_param_changed(ud: &mut UserData, id: u32, param: Option<&Pod>) {
    let Some(param) = param else { return };
    if id != ParamType::Format.as_raw() {
        return;
    }
    let (media_type, media_subtype) = match format_utils::parse_format(param) {
        Ok(v) => v,
        Err(e) => {
            warn!(error = ?e, "audio: ignoring unparseable Format pod");
            return;
        }
    };
    if media_type != MediaType::Audio || media_subtype != MediaSubtype::Raw {
        return;
    }
    let mut info = AudioInfoRaw::new();
    if let Err(e) = info.parse(param) {
        warn!(error = ?e, "audio: AudioInfoRaw::parse failed");
        return;
    }
    let negotiated_rate = info.rate();
    let negotiated_channels = info.channels() as u16;
    let negotiated_format = info.format();

    // Sanity check: with MANDATORY flags on the format pod PipeWire
    // shouldn't ever hand us a different shape, but log loudly if
    // it does so a regression in the negotiation path is obvious in
    // the field instead of presenting as mysterious wrong-pitch /
    // "feo" audio on the receiver.
    if negotiated_rate != ud.sample_rate || negotiated_channels != ud.channels {
        warn!(
            requested_rate = ud.sample_rate,
            negotiated_rate,
            requested_channels = ud.channels,
            negotiated_channels,
            ?negotiated_format,
            "audio: PipeWire negotiated DIFFERENT shape than requested — \
             AAC encoder will mislabel the bitstream, the receiver will \
             play at wrong pitch / speed"
        );
    } else {
        info!(
            sample_rate = negotiated_rate,
            channels = negotiated_channels,
            ?negotiated_format,
            "audio: PipeWire negotiated exact requested format"
        );
    }

    ud.negotiated
        .sample_rate
        .store(negotiated_rate, Ordering::Relaxed);
    ud.negotiated
        .channels
        .store(negotiated_channels as u32, Ordering::Relaxed);
}

fn handle_process(stream: &StreamRef, ud: &mut UserData, frame_tx: &mpsc::Sender<AudioFrame>) {
    let Some(mut buffer) = stream.dequeue_buffer() else {
        trace!("audio process tick with no buffer");
        return;
    };

    let datas = buffer.datas_mut();
    let Some(plane) = datas.first_mut() else {
        warn!("PipeWire audio buffer had no data planes");
        return;
    };

    let chunk = plane.chunk();
    let chunk_size = chunk.size() as usize;
    let chunk_offset = chunk.offset() as usize;
    if chunk_size == 0 {
        trace!("audio buffer empty (chunk.size == 0)");
        return;
    }

    let Some(slice) = plane.data() else {
        warn!("PipeWire audio buffer has no mapped data");
        return;
    };
    let end = chunk_offset.saturating_add(chunk_size).min(slice.len());
    if end <= chunk_offset {
        return;
    }

    // Copy out + (optionally) zero on mute. `BytesMut` so we own the
    // memory; the SPA buffer's `data()` slice is only valid until we
    // drop the buffer back to PipeWire at end-of-scope.
    let mut owned = BytesMut::with_capacity(end - chunk_offset);
    owned.extend_from_slice(&slice[chunk_offset..end]);
    if ud.mute.is_muted() {
        for b in owned.iter_mut() {
            *b = 0;
        }
    }
    let bytes = owned.freeze();

    // Carry the *negotiated* rate/channels on every frame, not the
    // values we requested. PipeWire is free to land on something
    // close-but-not-equal (e.g. 44.1 kHz against a 44.1 kHz sink
    // even though we asked for 48 kHz); the consumer side reads
    // these per-frame fields to configure its AAC encoder
    // accordingly so the ADTS header advertises the right
    // sample-rate index. The PTS math below uses them too, otherwise
    // a rate mismatch would read as constant drift. Falls back to the
    // requested values when `param_changed` hasn't run yet (= first
    // chunk delivered before the format callback, unusual but cheap
    // to guard).
    let negotiated_rate = ud.negotiated.sample_rate.load(Ordering::Relaxed);
    let negotiated_channels = ud.negotiated.channels.load(Ordering::Relaxed) as u16;
    let frame_rate = if negotiated_rate > 0 {
        negotiated_rate
    } else {
        ud.sample_rate
    };
    let frame_channels = if negotiated_channels > 0 {
        negotiated_channels
    } else {
        ud.channels
    };

    let samples_in_chunk = (bytes.len() / BYTES_PER_SAMPLE / frame_channels.max(1) as usize) as u64;
    if samples_in_chunk == 0 {
        return;
    }

    // Sample-count PTS on the wall-clock base shared with video,
    // re-anchored on gaps / drift (see `PtsClock`).
    let stamp = ud.pts.stamp(samples_in_chunk, frame_rate, now_us());
    if let Some(drift_us) = stamp.reanchored_from_drift_us {
        info!(
            drift_us,
            "audio: sample-count PTS strayed from the wall clock (stream gap, xrun or \
             clock drift) — re-anchored"
        );
    }
    let timestamp_us = stamp.timestamp_us;
    ud.log_sample_counter = ud.log_sample_counter.saturating_add(samples_in_chunk);

    if !ud.first_frame_logged {
        info!(
            samples = samples_in_chunk,
            bytes = bytes.len(),
            channels = ud.channels,
            sample_rate = ud.sample_rate,
            "first audio buffer received from PipeWire"
        );
        ud.first_frame_logged = true;
        ud.log_last_at = Some(std::time::Instant::now());
    } else if let Some(prev) = ud.log_last_at {
        let elapsed = prev.elapsed();
        if elapsed >= std::time::Duration::from_secs(5) {
            // Compare effective sample rate vs configured. If
            // they diverge by > 1% something is wrong (PipeWire
            // backpressure dropping chunks, hidden resample,
            // sample-counting bug, etc.).
            let effective_rate = (ud.log_sample_counter as f64 / elapsed.as_secs_f64()) as u32;
            let expected = ud.sample_rate;
            let deviation_pct = if expected > 0 {
                ((effective_rate as f64 - expected as f64).abs() / expected as f64) * 100.0
            } else {
                0.0
            };
            if deviation_pct > 1.0 {
                warn!(
                    expected_sample_rate = expected,
                    effective_sample_rate = effective_rate,
                    deviation_pct,
                    samples_observed = ud.log_sample_counter,
                    elapsed_ms = elapsed.as_millis() as u64,
                    "audio: effective sample rate deviates from configured rate \
                     — either PipeWire is silently delivering a different rate \
                     or some chunks were dropped"
                );
            } else {
                debug!(
                    sample_rate = effective_rate,
                    samples = ud.log_sample_counter,
                    elapsed_ms = elapsed.as_millis() as u64,
                    "audio: 5 s sample-rate health check"
                );
            }
            ud.log_sample_counter = 0;
            ud.log_last_at = Some(std::time::Instant::now());
        }
    }

    let frame = AudioFrame {
        codec: AudioCodec::Pcm,
        data: Bytes::from(bytes),
        timestamp_us,
        sample_rate: frame_rate,
        channels: frame_channels,
    };
    match frame_tx.try_send(frame) {
        Ok(()) => {}
        Err(mpsc::error::TrySendError::Full(_)) => {
            // Audible: a dropped PCM chunk shows up as a 21 ms hole
            // in the audio elementary stream. Surfacing at `warn!`
            // makes the failure mode discoverable in field logs.
            warn!("audio PCM chunk dropped (capture channel full)");
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            debug!("audio frame channel closed");
        }
    }
}

fn pod_view(bytes: &[u8]) -> &Pod {
    Pod::from_bytes(bytes).expect("audio format pod is always valid")
}

/// Build a fixed audio format pod: S16LE @ `sample_rate` × `channels`.
///
/// Properties go in with `flags::empty()` — PipeWire treats them as
/// hints rather than hard constraints, so its audioconvert graph
/// can resample / remix transparently when the upstream sink runs at
/// a different rate (44.1 kHz hardware against our 48 kHz request,
/// say). The actual negotiated shape is read back from the SPA
/// Format pod in `param_changed` and surfaced via
/// [`AudioCapture::sample_rate`] / [`AudioCapture::channels`]; the
/// caller (`run_audio_pipeline` in the binary) reads the first PCM
/// chunk's `AudioFrame::sample_rate` and configures the AAC encoder
/// to match, so the emitted ADTS header carries the right
/// sf_index and the chromecast plays at the correct pitch even
/// when our requested rate didn't survive negotiation verbatim.
///
/// (An earlier attempt at `MANDATORY` flags on every property
/// turned out to wedge the chromecast in a permanent "LOADING"
/// state on at least one PipeWire / sink combination — whatever
/// the precise mechanism, the constraint was too tight to be
/// safe across hardware.)
fn build_audio_format_pod(sample_rate: u32, channels: u16) -> Vec<u8> {
    // SPA's audio "position" array tells the graph which speaker
    // each channel represents. Stereo = FL,FR. Mono = MONO (we map
    // to FL when channels == 1 — close enough; the graph upmixes).
    let position: Vec<u32> = if channels == 1 {
        vec![SPA_AUDIO_CHANNEL_FL]
    } else {
        vec![SPA_AUDIO_CHANNEL_FL, SPA_AUDIO_CHANNEL_FR]
    };

    // Encode the position array as an Int array. SPA POD `Array` of
    // `Id` values uses the `Value::ValueArray` variant; the pipewire
    // crate exposes it via `Value::ValueArray` + `ValueArray::Id`.
    let position_value = Value::ValueArray(pw::spa::pod::ValueArray::Id(
        position.into_iter().map(Id).collect(),
    ));

    let obj = Object {
        type_: SpaTypes::ObjectParamFormat.as_raw(),
        id: ParamType::EnumFormat.as_raw(),
        properties: vec![
            Property {
                key: FormatProperties::MediaType.as_raw(),
                flags: PropertyFlags::empty(),
                value: Value::Id(Id(MediaType::Audio.as_raw())),
            },
            Property {
                key: FormatProperties::MediaSubtype.as_raw(),
                flags: PropertyFlags::empty(),
                value: Value::Id(Id(MediaSubtype::Raw.as_raw())),
            },
            Property {
                key: FormatProperties::AudioFormat.as_raw(),
                flags: PropertyFlags::empty(),
                value: Value::Id(Id(SPA_AUDIO_FORMAT_S16_LE)),
            },
            Property {
                key: FormatProperties::AudioRate.as_raw(),
                flags: PropertyFlags::empty(),
                value: Value::Int(sample_rate as i32),
            },
            Property {
                key: FormatProperties::AudioChannels.as_raw(),
                flags: PropertyFlags::empty(),
                value: Value::Int(channels as i32),
            },
            Property {
                key: FormatProperties::AudioPosition.as_raw(),
                flags: PropertyFlags::empty(),
                value: position_value,
            },
        ],
    };

    PodSerializer::serialize(Cursor::new(Vec::new()), &Value::Object(obj))
        .expect("audio pod serialization is infallible for our inputs")
        .0
        .into_inner()
}

/// Wall clock (`UNIX_EPOCH`) in µs. Deliberately not
/// `CLOCK_MONOTONIC` despite the `AudioCapture` trait doc: the HLS
/// segmenter's staleness filter compares audio timestamps against
/// `SystemTime::now()`, and video (`wayland_direct::now_us`) uses the
/// same base. Change all three together or none.
fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;
    /// 1024 samples at 48 kHz.
    const CHUNK: u64 = 1024;
    const CHUNK_US: u64 = 21_333;

    #[test]
    fn first_chunk_anchors_to_now() {
        let mut clock = PtsClock::default();
        let stamp = clock.stamp(CHUNK, RATE, 5_000_000);
        assert_eq!(stamp.timestamp_us, 5_000_000);
        assert_eq!(stamp.reanchored_from_drift_us, None);
    }

    #[test]
    fn steady_stream_follows_sample_count_not_jittery_wall() {
        let mut clock = PtsClock::default();
        clock.stamp(CHUNK, RATE, 1_000_000);
        // Arrives 3 ms late: PTS still comes from the sample count.
        let stamp = clock.stamp(CHUNK, RATE, 1_000_000 + CHUNK_US + 3_000);
        assert_eq!(stamp.timestamp_us, 1_000_000 + CHUNK_US);
        assert_eq!(stamp.reanchored_from_drift_us, None);
    }

    #[test]
    fn gap_reanchors_to_now() {
        let mut clock = PtsClock::default();
        clock.stamp(CHUNK, RATE, 1_000_000);
        // Sink went idle for 2 s.
        let now = 3_000_000;
        let stamp = clock.stamp(CHUNK, RATE, now);
        assert_eq!(stamp.timestamp_us, now);
        assert!(stamp.reanchored_from_drift_us.is_some_and(|d| d < 0));
        // Continues from the new anchor.
        let next = clock.stamp(CHUNK, RATE, now + CHUNK_US);
        assert_eq!(next.timestamp_us, now + CHUNK_US);
        assert_eq!(next.reanchored_from_drift_us, None);
    }

    #[test]
    fn counted_clock_ahead_of_wall_reanchors() {
        let mut clock = PtsClock::default();
        clock.stamp(48_000 * 10, RATE, 0);
        // 10 s of samples delivered after only 9.8 s of wall time.
        let stamp = clock.stamp(CHUNK, RATE, 9_800_000);
        assert_eq!(stamp.timestamp_us, 9_800_000);
        assert_eq!(stamp.reanchored_from_drift_us, Some(200_000));
    }

    #[test]
    fn drift_within_threshold_is_tolerated() {
        let mut clock = PtsClock::default();
        clock.stamp(48_000 * 10, RATE, 0);
        let stamp = clock.stamp(CHUNK, RATE, 10_000_000 + PTS_REANCHOR_THRESHOLD_US);
        assert_eq!(stamp.timestamp_us, 10_000_000);
        assert_eq!(stamp.reanchored_from_drift_us, None);
    }
}
