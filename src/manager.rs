use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::task::JoinHandle;
use uuid::Uuid;

use ferricast_core::{
    CaptureConfig, CaptureSource, CastSession, CapturedFrame, Device, Discovery, DiscoveryEvent,
    EncodedFrame, EncoderConfig, FerricastError, ProtocolHandler, Result, ScreenCapture,
    StreamConfig, VideoEncoder,
};

type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

trait ErasedDiscovery: Send + Sync {
    fn start(&mut self, tx: mpsc::Sender<DiscoveryEvent>) -> BoxFut<'_, Result<()>>;
    fn stop(&mut self) -> BoxFut<'_, Result<()>>;
}

impl<T: Discovery> ErasedDiscovery for T {
    fn start(&mut self, tx: mpsc::Sender<DiscoveryEvent>) -> BoxFut<'_, Result<()>> {
        Box::pin(Discovery::start(self, tx))
    }
    fn stop(&mut self) -> BoxFut<'_, Result<()>> {
        Box::pin(Discovery::stop(self))
    }
}

trait ErasedSession: Send + Sync {
    fn connect<'a>(&'a mut self, device: &'a Device) -> BoxFut<'a, Result<()>>;
    fn setup_stream<'a>(&'a mut self, config: &'a StreamConfig) -> BoxFut<'a, Result<()>>;
    fn send_frame<'a>(&'a mut self, frame: &'a EncodedFrame) -> BoxFut<'a, Result<()>>;
    fn stop(&mut self) -> BoxFut<'_, Result<()>>;
    fn is_alive(&self) -> bool;
}

impl<T: CastSession> ErasedSession for T {
    fn connect<'a>(&'a mut self, device: &'a Device) -> BoxFut<'a, Result<()>> {
        Box::pin(CastSession::connect(self, device))
    }
    fn setup_stream<'a>(&'a mut self, config: &'a StreamConfig) -> BoxFut<'a, Result<()>> {
        Box::pin(CastSession::setup_stream(self, config))
    }
    fn send_frame<'a>(&'a mut self, frame: &'a EncodedFrame) -> BoxFut<'a, Result<()>> {
        Box::pin(CastSession::send_frame(self, frame))
    }
    fn stop(&mut self) -> BoxFut<'_, Result<()>> {
        Box::pin(CastSession::stop(self))
    }
    fn is_alive(&self) -> bool {
        CastSession::is_alive(self)
    }
}

struct RegisteredProtocol {
    protocol: &'static str,
    create_discovery: Box<dyn Fn() -> Box<dyn ErasedDiscovery> + Send + Sync>,
    create_session: Box<dyn Fn() -> Result<Box<dyn ErasedSession>> + Send + Sync>,
}

struct ActiveStream {
    device: Device,
    cancel_tx: mpsc::Sender<()>,
    /// Handle to the spawned streaming task. Kept so `stop_stream`
    /// can `await` until the task has actually torn down its
    /// capture/encoder/session — without this, `shutdown()` can
    /// return while PipeWire/CASTv2 cleanup is still racing the
    /// process exit and the OS reaps fds + threads abruptly.
    task: JoinHandle<()>,
}

#[derive(Debug, Clone)]
pub enum ManagerEvent {
    DeviceFound(Device),
    DeviceLost(Uuid),
    StreamStarted {
        device_id: Uuid,
        device_name: String,
    },
    StreamStopped {
        device_id: Uuid,
    },
    StreamError {
        device_id: Uuid,
        message: String,
    },
    DiscoveryError {
        protocol: &'static str,
        message: String,
    },
}

pub struct StreamManager {
    protocols: Vec<RegisteredProtocol>,
    devices: Arc<RwLock<HashMap<Uuid, Device>>>,
    active_streams: Arc<Mutex<HashMap<Uuid, ActiveStream>>>,
    discovery_handles: Vec<Box<dyn ErasedDiscovery>>,
    event_tx: mpsc::Sender<ManagerEvent>,
    event_rx: Option<mpsc::Receiver<ManagerEvent>>,
    running: bool,
}

impl Default for StreamManager {
    fn default() -> Self {
        let (event_tx, event_rx) = mpsc::channel(256);
        Self {
            protocols: Vec::new(),
            devices: Arc::new(RwLock::new(HashMap::new())),
            active_streams: Arc::new(Mutex::new(HashMap::new())),
            discovery_handles: Vec::new(),
            event_tx,
            event_rx: Some(event_rx),
            running: false,
        }
    }
}

impl StreamManager {
    pub fn register<H>(&mut self)
    where
        H: ProtocolHandler + Clone + Default + 'static,
    {
        let h_sess = H::default();
        let h_disc = h_sess.clone();

        self.protocols.push(RegisteredProtocol {
            protocol: H::PROTOCOL,
            create_discovery: Box::new(move || {
                Box::new(h_disc.create_discovery()) as Box<dyn ErasedDiscovery>
            }),
            create_session: Box::new(move || {
                let session = h_sess.create_session()?;
                Ok(Box::new(session) as Box<dyn ErasedSession>)
            }),
        });
    }

    pub fn take_event_rx(&mut self) -> Option<mpsc::Receiver<ManagerEvent>> {
        self.event_rx.take()
    }

    pub fn registered_protocols(&self) -> Vec<&'static str> {
        self.protocols.iter().map(|p| p.protocol).collect()
    }

    pub async fn start_discovery(&mut self) -> Result<()> {
        if self.running {
            return Ok(());
        }

        let (disc_tx, mut disc_rx) = mpsc::channel::<DiscoveryEvent>(256);

        for proto in &self.protocols {
            let mut discovery = (proto.create_discovery)();
            let tx = disc_tx.clone();
            match discovery.start(tx).await {
                Ok(()) => {
                    self.discovery_handles.push(discovery);
                }
                Err(e) => {
                    tracing::warn!(
                        protocol = proto.protocol,
                        %e,
                        "Failed to start discovery, skipping"
                    );
                    let _ = self
                        .event_tx
                        .send(ManagerEvent::DiscoveryError {
                            protocol: proto.protocol,
                            message: e.to_string(),
                        })
                        .await;
                }
            }
        }

        let devices = self.devices.clone();
        let event_tx = self.event_tx.clone();

        tokio::spawn(async move {
            while let Some(event) = disc_rx.recv().await {
                match event {
                    DiscoveryEvent::DeviceFound(device) => {
                        let id = device.id;
                        tracing::info!(name = %device.name, protocol = device.protocol, "Device found");
                        devices.write().await.insert(id, device.clone());
                        let _ = event_tx.send(ManagerEvent::DeviceFound(device)).await;
                    }
                    DiscoveryEvent::DeviceLost(id) => {
                        tracing::info!(?id, "Device lost");
                        devices.write().await.remove(&id);
                        let _ = event_tx.send(ManagerEvent::DeviceLost(id)).await;
                    }
                    DiscoveryEvent::Error { protocol, message } => {
                        tracing::warn!(protocol, %message, "Discovery error");
                        let _ = event_tx
                            .send(ManagerEvent::DiscoveryError { protocol, message })
                            .await;
                    }
                }
            }
        });

        self.running = true;
        Ok(())
    }

    pub async fn stop_discovery(&mut self) -> Result<()> {
        for discovery in &mut self.discovery_handles {
            discovery.stop().await?;
        }
        self.discovery_handles.clear();
        self.running = false;
        Ok(())
    }

    pub async fn devices(&self) -> Vec<Device> {
        self.devices.read().await.values().cloned().collect()
    }

    pub async fn start_stream(
        &self,
        device_id: Uuid,
        source: CaptureSource,
        mut capture: impl ScreenCapture + 'static,
        mut encoder: impl VideoEncoder + 'static,
        config: StreamConfig,
    ) -> Result<()> {
        let device = {
            let devices = self.devices.read().await;
            devices
                .get(&device_id)
                .cloned()
                .ok_or_else(|| FerricastError::DeviceNotFound(device_id.to_string()))?
        };

        let proto = self
            .protocols
            .iter()
            .find(|p| p.protocol == device.protocol)
            .ok_or_else(|| {
                FerricastError::Protocol(format!("No handler for {}", device.protocol))
            })?;

        // Apply the device's fps ceiling BEFORE start so the
        // PipeWire worker's internal throttle anchors to the right
        // rate from buffer 0 — otherwise the worker burns CPU on
        // Vulkan readbacks at the compositor's refresh (60 Hz
        // typical) for frames the encoder will never consume.
        let device_fps_cap = device.capabilities.max_fps;
        let initial_fps = match device_fps_cap {
            Some(cap) => config.fps.min(cap),
            None => config.fps,
        };

        // Start capture first so the portal picker shows before connecting.
        let capture_config = CaptureConfig {
            fps: initial_fps,
            width: Some(config.width),
            height: Some(config.height),
            show_cursor: true,
        };

        capture.start(source, capture_config).await?;

        // Block on the first frame: PipeWire's format negotiation and
        // buffer setup happen on its main-loop thread asynchronously,
        // so `capture.start().await` returning doesn't mean the
        // negotiated format is available yet. Pulling a frame is the
        // simplest barrier — backends only emit frames once the
        // format is acked, so by the time this resolves
        // `get_screen_size` / `get_framerate` are populated.
        //
        // Without this the encoder gets configured with (0, 0) and
        // x264 dies with "invalid width x height (0x0)" while
        // PipeWire tears its own stream down with "no more input
        // formats" — same root cause.
        tracing::info!(device_id = %device_id, "awaiting first frame from capture (30s ceiling)");
        let first_frame = match tokio::time::timeout(
            Duration::from_secs(30),
            capture.next_frame(),
        )
        .await
        {
            Ok(Ok(f)) => {
                tracing::info!(
                    device_id = %device_id,
                    "first frame received from capture"
                );
                f
            }
            Ok(Err(e)) => {
                tracing::error!(
                    device_id = %device_id,
                    %e,
                    "capture returned error before first frame"
                );
                return Err(e);
            }
            Err(_) => {
                tracing::error!(
                    device_id = %device_id,
                    "capture.next_frame() did not return within 30s — \
                     the capture worker isn't delivering buffers \
                     (compositor idle, or worker stalled)"
                );
                return Err(FerricastError::Timeout(
                    "capture.next_frame() exceeded 30s".into(),
                ));
            }
        };
        tracing::info!(device_id = %device_id, "Capture negotiated, connecting to device");

        let (width, height) = capture.get_screen_size();
        let pixel_format = capture.get_pixel_format();
        // Prefer the framerate the source actually delivers; fall
        // back to the user's hint if the backend doesn't report one
        // (X11 polling or a backend that hasn't negotiated yet).
        let negotiated_fps = capture.get_framerate();
        let mut effective_fps = if negotiated_fps > 0 {
            negotiated_fps
        } else {
            config.fps
        };
        let mut effective_bitrate_kbps = config.bitrate_kbps;
        let mut effective_h264_profile: Option<ferricast_core::H264Profile> = None;

        // Apply receiver-side hardware limits before we configure
        // the encoder. `DeviceCapabilities` is populated by the
        // protocol's discovery code (chromecast/discovery.rs reads
        // the `md` mDNS field and maps it to a known device class)
        // so this caps for the right reason on the right device:
        // 1st-gen Chromecast → 1080p@30 Main, Ultra → 4K@30 High,
        // Google TV → 1080p@60 High, etc.
        let caps = &device.capabilities;
        if let Some(max_fps) = caps.max_fps {
            if effective_fps > max_fps {
                tracing::warn!(
                    device = %device.name,
                    requested_fps = effective_fps,
                    max_fps,
                    "device cap: lowering encode fps to receiver's decoder ceiling"
                );
                effective_fps = max_fps;
            }
        }
        if let Some(max_bitrate) = caps.max_bitrate_kbps {
            if effective_bitrate_kbps > max_bitrate {
                tracing::warn!(
                    device = %device.name,
                    requested_kbps = effective_bitrate_kbps,
                    max_kbps = max_bitrate,
                    "device cap: lowering bitrate to receiver's decoder ceiling"
                );
                effective_bitrate_kbps = max_bitrate;
            }
        }
        if let Some(prof) = caps.max_h264_profile {
            effective_h264_profile = Some(prof);
        }

        tracing::info!(
            device = %device.name,
            model = %device.model.as_deref().unwrap_or(""),
            width,
            height,
            negotiated_fps,
            effective_fps,
            effective_bitrate_kbps,
            ?effective_h264_profile,
            requires_audio = caps.requires_audio,
            ?pixel_format,
            "encoder configured against device capabilities"
        );

        encoder.configure(&EncoderConfig {
            pixel_format,
            width: width as u32,
            height: height as u32,
            fps: effective_fps,
            bitrate_kbps: effective_bitrate_kbps,
            max_h264_profile: effective_h264_profile,
            ..Default::default()
        })?;


        let mut session = (proto.create_session)()?;
        session.connect(&device).await?;
        session.setup_stream(&config).await?;

        let (cancel_tx, mut cancel_rx) = mpsc::channel::<()>(1);

        let active_streams = self.active_streams.clone();
        let event_tx = self.event_tx.clone();
        let device_name = device.name.clone();
        let did = device.id;

        let task = tokio::spawn(async move {

            let _ = event_tx
                .send(ManagerEvent::StreamStarted {
                    device_id: did,
                    device_name,
                })
                .await;

            // The frame we already pulled to barrier on negotiation
            // is the first frame of this loop — dropping it would
            // waste an entire keyframe interval on backends that
            // emit IDR on the very first frame.
            let mut seed = Some(first_frame);

            // Pacing for event-driven capture sources. Wayland
            // compositors only emit a new dmabuf when the screen
            // actually changes — meaning a stream-paused user
            // (e.g. waiting for the chromecast to start displaying)
            // can go many seconds with no new frame. Receivers
            // that need a steady cadence (HLS chromecast, DASH,
            // anything muxed into MPEG-TS) starve under that:
            // their segmenter expects 1 keyframe / second and
            // can't close a segment without two of them.
            //
            // Pace at `effective_fps`: if `capture.next_frame()`
            // doesn't deliver inside one frame period, re-emit
            // the most recent real frame with its timestamp
            // bumped forward. The encoder treats it as a fresh
            // input and produces another encoded frame; the
            // downstream session sees a steady stream regardless
            // of what's happening on screen.
            let frame_period = Duration::from_secs_f64(1.0 / (effective_fps.max(1) as f64));
            let frame_period_us = (1_000_000.0 / (effective_fps.max(1) as f64)) as u64;
            let mut last_frame: Option<CapturedFrame> = None;

            // Wall-clock keyframe pacing. The HLS segmenter closes a
            // segment at the *next* keyframe after the configured
            // target_secs elapses. NVENC's natural \"every N frames\"
            // keyframe schedule drifts vs wallclock — at 30 fps a
            // 60-frame interval is *nominally* 2 s but actually
            // lands anywhere from 1.95 s to 2.05 s depending on
            // pacing jitter. When NVENC's IDR lands a few ms
            // before the segmenter's 2 s mark, the segmenter
            // pushes the keyframe through and waits another full
            // interval — producing a 4 s segment instead of 2 s.
            // The chromecast then has to fetch ~2 MB chunks and
            // visible BUFFERING events appear.
            //
            // Fix: ask the encoder for an IDR on wall-clock
            // boundaries instead of relying on its natural
            // interval. Segments come out at exactly target_secs
            // wide, sizes stay tight, BUFFERING events disappear.
            //
            // 4 s matches HlsConfig::default().segment_target_secs.
            // Could be plumbed through later if we ship a
            // non-default HlsConfig.
            let keyframe_interval = Duration::from_secs(4);
            let mut next_keyframe_at = Instant::now();

            loop {
                tokio::select! {
                    _ = cancel_rx.recv() => {
                        tracing::info!(?did, "Stream cancelled");
                        break;
                    }
                    frame_result = async {
                        if let Some(f) = seed.take() {
                            last_frame = Some(f.clone());
                            return Ok(f);
                        }
                        match tokio::time::timeout(frame_period, capture.next_frame()).await {
                            Ok(Ok(f)) => {
                                last_frame = Some(f.clone());
                                Ok(f)
                            }
                            Ok(Err(e)) => Err(e),
                            Err(_) => match last_frame.as_mut() {
                                Some(stored) => {
                                    bump_timestamp(stored, frame_period_us);
                                    Ok(stored.clone())
                                }
                                None => capture.next_frame().await,
                            },
                        }
                    } => {
                        match frame_result {
                            Ok(raw_frame) => {
                                // Wall-clock keyframe pacing — see comment at
                                // `keyframe_interval` declaration. Without this
                                // segments alternate 2 s / 4 s and the
                                // chromecast hits BUFFERING on the 4 s ones.
                                let now = Instant::now();
                                if now >= next_keyframe_at {
                                    encoder.request_keyframe();
                                    // Step forward in fixed increments so a
                                    // late frame doesn't shift the whole
                                    // schedule and accumulate drift.
                                    while next_keyframe_at <= now {
                                        next_keyframe_at += keyframe_interval;
                                    }
                                }
                                match encoder.encode(raw_frame) {
                                    Ok(encoded) => {
                                        if let Err(e) = session.send_frame(&encoded).await {
                                            tracing::error!(%e, "Failed to send frame");
                                            let _ = event_tx.send(ManagerEvent::StreamError {
                                                device_id: did,
                                                message: e.to_string(),
                                            }).await;
                                            break;
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!(%e, "Encoding error");
                                        let _ = event_tx.send(ManagerEvent::StreamError {
                                            device_id: did,
                                            message: e.to_string(),
                                        }).await;
                                        break;
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::error!(%e, "Capture error");
                                let _ = event_tx.send(ManagerEvent::StreamError {
                                    device_id: did,
                                    message: e.to_string(),
                                }).await;
                                break;
                            }
                        }
                    }
                }
            }

            let _ = session.stop().await;
            let _ = capture.stop().await;
            active_streams.lock().await.remove(&did);
            let _ = event_tx
                .send(ManagerEvent::StreamStopped { device_id: did })
                .await;
        });

        self.active_streams.lock().await.insert(
            device_id,
            ActiveStream {
                device,
                cancel_tx,
                task,
            },
        );

        Ok(())
    }

    pub async fn stop_stream(&self, device_id: Uuid) -> Result<()> {
        let stream = self.active_streams.lock().await.remove(&device_id);
        if let Some(stream) = stream {
            tracing::info!(name = %stream.device.name, "Stopping stream");
            // Signal the loop to break, then wait for the task to
            // actually finish — capture.stop() / session.stop() run
            // *inside* the task and we want them done before we
            // return so the caller (e.g. ctrl-c handler) can rely on
            // "stop_stream returned" meaning "no more PipeWire
            // worker thread / TLS connection / HLS listener alive".
            let _ = stream.cancel_tx.send(()).await;
            // 5s ceiling: if cleanup wedges (PipeWire main loop
            // refusing to quit, TLS write blocked behind a dead
            // peer) we'd rather log + move on than hang forever.
            match tokio::time::timeout(std::time::Duration::from_secs(5), stream.task).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::warn!(?e, "stream task panicked during shutdown"),
                Err(_) => tracing::warn!(name = %stream.device.name, "stream task did not finish in 5s, abandoning"),
            }
            Ok(())
        } else {
            Err(FerricastError::NoActiveSession)
        }
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        let ids: Vec<Uuid> = self.active_streams.lock().await.keys().copied().collect();
        for id in ids {
            let _ = self.stop_stream(id).await;
        }
        self.stop_discovery().await
    }
}

/// Forward `frame`'s timestamp by `delta_us` microseconds. Used by
/// the manager loop's pacing fallback: when the upstream capture
/// stalls (Wayland compositor not emitting frames because nothing
/// on screen has changed) we re-emit the most recent real frame
/// with a synthetic monotonic timestamp so the downstream encoder
/// + segmenter keeps producing output at the configured fps.
fn bump_timestamp(frame: &mut CapturedFrame, delta_us: u64) {
    match frame {
        CapturedFrame::Cpu(r) => r.timestamp_us = r.timestamp_us.saturating_add(delta_us),
        CapturedFrame::Gpu(g) => g.timestamp_us = g.timestamp_us.saturating_add(delta_us),
    }
}
