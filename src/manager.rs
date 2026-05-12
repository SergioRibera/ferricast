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
    /// Wrapped in `Arc` instead of `Box` so the streaming task can
    /// hold its own clone — needed for auto-reconnect, which builds
    /// a fresh session on `is_alive()` going false (chromecast 301,
    /// receiver-app crash) without tearing down capture/encoder.
    create_session: Arc<dyn Fn() -> Result<Box<dyn ErasedSession>> + Send + Sync>,
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
    /// Fatal stream error — the stream task has terminated and won't
    /// recover on its own. Apps typically surface this to the user
    /// (toast, retry button) since at this point the manager has
    /// already exhausted its own auto-reconnect budget.
    StreamError {
        device_id: Uuid,
        message: String,
    },
    /// Non-fatal: the receiver session went away mid-stream (e.g.
    /// Chromecast `detailedErrorCode=301`, app exit), and the
    /// manager is attempting to recreate it. Capture/encoder keep
    /// running; the receiver will go dark for a few seconds.
    /// Apps can use this to show a "reconnecting…" indicator
    /// without tearing down their own state.
    StreamReconnecting {
        device_id: Uuid,
        attempt: u32,
        reason: String,
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
            create_session: Arc::new(move || {
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


        // Adaptive bitrate state — shared between the receiver's HLS
        // server (records per-segment delivery pressure) and this
        // stream task (reads the recommended target on the hot path).
        // The HLS endpoint inside `session` will pick it up from
        // `StreamConfig::adaptive` in `setup_stream`. Initial target
        // is the already-clamped `effective_bitrate_kbps` so the
        // controller's ceiling matches what the manager negotiated.
        let adaptive = ferricast_core::AdaptiveBitrateState::new(effective_bitrate_kbps);
        tracing::info!(
            initial_kbps = effective_bitrate_kbps,
            ceiling_kbps = adaptive.ceiling_kbps,
            floor_kbps = adaptive.floor_kbps,
            "adaptive: controller initialised; bandwidth probe will fire after first {} segments",
            ferricast_hls::PROBE_SAMPLES_FOR_DOC,
        );
        let session_config = StreamConfig {
            adaptive: Some(adaptive.clone()),
            ..config.clone()
        };

        let mut session = (proto.create_session)()?;
        session.connect(&device).await?;
        session.setup_stream(&session_config).await?;

        // Factory the supervisor uses to rebuild a session on
        // receiver-side disconnect. Cloned `Arc` — cheap and `Send`.
        let create_session = proto.create_session.clone();

        let (cancel_tx, mut cancel_rx) = mpsc::channel::<()>(1);

        let active_streams = self.active_streams.clone();
        let event_tx = self.event_tx.clone();
        let device_name = device.name.clone();
        let did = device.id;

        let device_clone = device.clone();

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

            // Wall-clock keyframe pacing — see longer comment below.
            let keyframe_interval = Duration::from_secs(4);
            let mut next_keyframe_at = Instant::now();

            // Mirror of `adaptive.target_kbps()` we've actually
            // pushed into the encoder. The HLS server may mutate
            // the atomic between any two frames; we only need to
            // call `encoder.set_bitrate_kbps` when it differs from
            // what's currently configured, otherwise every frame
            // would round-trip into NVENC's reconfigure call.
            let mut last_applied_kbps: u32 = effective_bitrate_kbps;

            // Auto-reconnect supervisor state. The receiver can die
            // mid-stream (Chromecast detailedErrorCode=301, Default
            // Media Receiver app exit, transient TCP RST burst on
            // a flaky 2.4 GHz link) without it being a permanent
            // failure of capture or encoder. The pattern here is
            // session-scoped retry: only the receiver session is
            // recreated, capture and encoder keep their internal
            // state. After `MAX_CONSECUTIVE_FAILURES` quick deaths
            // in a row we surface a fatal `StreamError` so the app
            // can fall back to whatever its policy is.
            const MAX_CONSECUTIVE_FAILURES: u32 = 5;
            // Reset the failure counter when the stream has been
            // healthy for at least this long — a stream that ran
            // fine for 10 min then hit one transient is not the
            // same case as five 301s in 20 s.
            const HEALTHY_RESET_AFTER: Duration = Duration::from_secs(60);

            let mut consecutive_failures: u32 = 0;
            let mut last_failure_at: Option<Instant> = None;
            let mut active_session: Option<Box<dyn ErasedSession>> = Some(session);
            let mut fatal_error: Option<String> = None;
            let mut cancelled = false;
   

            // Outer (supervisor) loop. Acquires a session, runs the
            // inner frame loop until that session dies or the user
            // cancels, then decides whether to retry or surface.
            'supervisor: loop {
                let mut session: Box<dyn ErasedSession> = match active_session.take() {
                    Some(s) => s,
                    None => {
                        // Reconnect path.
                        // Reset counter on long-healthy streaks.
                        if let Some(t) = last_failure_at {
                            if t.elapsed() > HEALTHY_RESET_AFTER {
                                consecutive_failures = 0;
                            }
                        }
                        consecutive_failures += 1;
                        if consecutive_failures > MAX_CONSECUTIVE_FAILURES {
                            fatal_error = Some(format!(
                                "receiver-side disconnect; gave up after {} consecutive reconnect attempts",
                                MAX_CONSECUTIVE_FAILURES
                            ));
                            break 'supervisor;
                        }
                        // Force the adaptive controller to the floor
                        // before reconnecting. The previous bitrate
                        // didn't keep the receiver alive; starting
                        // low gives us the best chance of getting
                        // through the soft window after the receiver
                        // app relaunches, when its buffer is still
                        // empty and any further pressure would
                        // trigger 301 again.
                        let floored = adaptive.drop_to_floor();
                        // Apply immediately so the next encode is
                        // already at the lower rate (no waiting for
                        // the inner loop's first iteration to notice).
                        if let Err(e) = encoder.set_bitrate_kbps(floored) {
                            tracing::warn!(%e, "reconnect: encoder set_bitrate_kbps failed");
                        }
                        last_applied_kbps = floored;
                        // Request an IDR so the new HLS sink's
                        // bootstrap finds SPS/PPS on the very first
                        // frame instead of waiting for the encoder's
                        // natural GOP boundary.
                        encoder.request_keyframe();

                        // Backoff: 1 s, 2 s, 4 s, 8 s, 16 s.
                        let backoff = Duration::from_secs(
                            1u64 << (consecutive_failures.saturating_sub(1).min(4) as u64),
                        );
                        let reason = format!(
                            "reconnecting (attempt {}/{}, backing off {:?})",
                            consecutive_failures, MAX_CONSECUTIVE_FAILURES, backoff
                        );
                        tracing::warn!(?did, attempt = consecutive_failures, ?backoff, "reconnect: retrying session");
                        let _ = event_tx
                            .send(ManagerEvent::StreamReconnecting {
                                device_id: did,
                                attempt: consecutive_failures,
                                reason: reason.clone(),
                            })
                            .await;
                        tokio::select! {
                            _ = cancel_rx.recv() => {
                                cancelled = true;
                                break 'supervisor;
                            }
                            _ = tokio::time::sleep(backoff) => {}
                        }

                        let mut s = match (create_session)() {
                            Ok(s) => s,
                            Err(e) => {
                                tracing::warn!(%e, "reconnect: create_session failed");
                                continue 'supervisor;
                            }
                        };
                        if let Err(e) = s.connect(&device_clone).await {
                            tracing::warn!(%e, "reconnect: session.connect failed");
                            continue 'supervisor;
                        }
                        if let Err(e) = s.setup_stream(&session_config).await {
                            tracing::warn!(%e, "reconnect: session.setup_stream failed");
                            let _ = s.stop().await;
                            continue 'supervisor;
                        }
                        tracing::info!(?did, attempt = consecutive_failures, "reconnect: session re-established");
                        last_failure_at = Some(Instant::now());
                        s
                    }
                };

                // Inner frame loop. Same hot path as before; on
                // session death we drop out and let the supervisor
                // recreate.
                'inner: loop {
                    tokio::select! {
                        _ = cancel_rx.recv() => {
                            tracing::info!(?did, "Stream cancelled");
                            cancelled = true;
                            let _ = session.stop().await;
                            break 'supervisor;
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
                                    let now = Instant::now();
                                    if now >= next_keyframe_at {
                                        encoder.request_keyframe();
                                        while next_keyframe_at <= now {
                                            next_keyframe_at += keyframe_interval;
                                        }
                                    }
                                    let want_kbps = adaptive.target_kbps();
                                    if want_kbps != last_applied_kbps {
                                        match encoder.set_bitrate_kbps(want_kbps) {
                                            Ok(()) => {
                                                tracing::info!(
                                                    from_kbps = last_applied_kbps,
                                                    to_kbps = want_kbps,
                                                    "adaptive: encoder bitrate updated"
                                                );
                                                last_applied_kbps = want_kbps;
                                            }
                                            Err(e) => {
                                                tracing::warn!(%e, want_kbps, "adaptive: set_bitrate_kbps failed");
                                                last_applied_kbps = want_kbps;
                                            }
                                        }
                                    }

                                    match encoder.encode(raw_frame) {
                                        Ok(encoded) => {
                                            if let Err(e) = session.send_frame(&encoded).await {
                                                tracing::error!(%e, "Failed to send frame");
                                                fatal_error = Some(e.to_string());
                                                let _ = session.stop().await;
                                                break 'supervisor;
                                            }
                                            // Receiver-side liveness: when
                                            // the chromecast app dies it
                                            // sends CLOSE / ERROR 301; the
                                            // session marks itself not
                                            // alive but `send_frame` keeps
                                            // accepting (HLS sink is local
                                            // and stays up). Catching it
                                            // here pivots us back to the
                                            // supervisor for a clean
                                            // reconnect instead of feeding
                                            // a dead pipe.
                                            if !session.is_alive() {
                                                tracing::warn!(?did, "session reports not alive — entering reconnect supervisor");
                                                let _ = session.stop().await;
                                                break 'inner;
                                            }
                                        }
                                        Err(e) => {
                                            tracing::error!(%e, "Encoding error");
                                            fatal_error = Some(e.to_string());
                                            let _ = session.stop().await;
                                            break 'supervisor;
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::error!(%e, "Capture error");
                                    fatal_error = Some(e.to_string());
                                    let _ = session.stop().await;
                                    break 'supervisor;
                                }
                            }
                        }
                    }
                }
            }

            let _ = capture.stop().await;
            active_streams.lock().await.remove(&did);
            // Surface a final error if the supervisor gave up.
            // `StreamStopped` is always emitted afterwards so the
            // app's "playing → stopped" state machine doesn't need
            // to special-case error paths — same terminal event,
            // optional error explainer just before it.
            if let Some(msg) = fatal_error {
                let _ = event_tx
                    .send(ManagerEvent::StreamError {
                        device_id: did,
                        message: msg,
                    })
                    .await;
            }
            let _ = cancelled; // bookkeeping; not all paths read it
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
