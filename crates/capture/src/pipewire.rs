//! PipeWire capture implementation using xdg-desktop-portal ScreenCast.

use std::os::fd::OwnedFd;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use tokio::sync::mpsc;
use tracing::{debug, error, info, trace, warn};

use pipewire as pw;
use pw::properties::properties;
use pw::stream::{Stream, StreamFlags, StreamState};

use ferricast_core::{
    CaptureConfig, CaptureSource, FerricastError, PixelFormat, RawFrame, Result, ScreenCapture,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a SPA video format fourcc to our [`PixelFormat`].
fn spa_format_to_pixel(format: u32) -> Option<PixelFormat> {
    match format {
        // BGRx / BGRA
        8 | 9 => Some(PixelFormat::Bgra),
        // RGBx / RGBA
        6 | 7 => Some(PixelFormat::Rgba),
        // NV12
        25 => Some(PixelFormat::Nv12),
        // I420
        2 => Some(PixelFormat::I420),
        other => {
            warn!(spa_format = other, "unmapped SPA video format");
            None
        }
    }
}

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

// ---------------------------------------------------------------------------
// Portal helpers (ashpd)
// ---------------------------------------------------------------------------

struct PortalSession {
    pw_fd: OwnedFd,
    node_id: u32,
}


async fn open_portal_session(
    source: &CaptureSource,
    config: &CaptureConfig,
    shared: Arc<SharedState>,
) -> Result<PortalSession> {
    use ashpd::desktop::screencast::{CursorMode, Screencast, SourceType};

    let proxy = Screencast::new().await.map_err(portal_err)?;

    let session = proxy.create_session().await.map_err(portal_err)?;
    debug!("portal session created");

    let source_type = match source {
        CaptureSource::FullScreen { .. } => SourceType::Monitor,
        CaptureSource::Window { .. } => SourceType::Window,
    };

    let cursor_mode = if config.show_cursor {
        CursorMode::Embedded
    } else {
        CursorMode::Hidden
    };

    proxy
        .select_sources(
            &session,
            cursor_mode,
            source_type.into(),
            false,
            None,
            ashpd::desktop::PersistMode::DoNot,
        )
        .await
        .map_err(portal_err)?;

    info!("portal source selected, starting cast");


    let response = proxy.start(&session, None).await.map_err(portal_err)?;
    let response = response.response().map_err(portal_err)?;

    let streams = response.streams();
    let stream = streams.first().ok_or_else(|| {
        FerricastError::Capture("no streams returned from portal".into())
    })?;
    let size = stream.size().unwrap_or((0, 0));
    
    
    shared.width.store(size.0 as usize, Ordering::SeqCst);
    shared.height.store(size.1 as usize, Ordering::SeqCst);

    let node_id = stream.pipe_wire_node_id();
    info!(node_id, "portal returned PipeWire node");

    let fd = proxy
        .open_pipe_wire_remote(&session)
        .await
        .map_err(portal_err)?;

    Ok(PortalSession { pw_fd: fd, node_id })
}

fn portal_err(e: impl std::fmt::Display) -> FerricastError {
    FerricastError::Capture(format!("portal: {e}"))
}

// ---------------------------------------------------------------------------
// PipeWire thread
// ---------------------------------------------------------------------------

struct SharedState {
    running: AtomicBool,
    width: AtomicUsize,
    height: AtomicUsize,
}

enum PwEvent {
    Frame(RawFrame),
    Error(String),
}

/// Negotiated video format info.
struct NegotiatedFormat {
    width: u32,
    height: u32,
    stride: u32,
    format: u32,
}

impl Default for NegotiatedFormat {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            stride: 0,
            format: 0,
        }
    }
}

fn run_pw_thread(
    pw_fd: OwnedFd,
    node_id: u32,
    _config: CaptureConfig,
    frame_tx: mpsc::Sender<PwEvent>,
    shared: Arc<SharedState>,
) {
    pw::init();

    let mainloop =
        pw::main_loop::MainLoop::new(None).expect("failed to create PipeWire MainLoop");
    let context =
        pw::context::Context::new(&mainloop).expect("failed to create PipeWire Context");

    let core = context
        .connect_fd(pw_fd, None)
        .expect("failed to connect to PipeWire via portal fd");

    let stream = Stream::new(
        &core,
        "ferricast-capture",
        properties! {
            *pw::keys::MEDIA_TYPE => "Video",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Screen",
        },
    )
    .expect("failed to create PipeWire Stream");

    let loop_quit = mainloop.clone();
    let shared_state_changed = Arc::clone(&shared);
    let tx_error = frame_tx.clone();

    // Use a pipe to signal the PW main loop to quit from the async side.
    // When the running flag is cleared, the process callback will quit
    // the loop. We also add a signal source via the loop's fd mechanism.

    let _listener = stream
        .add_local_listener_with_user_data(NegotiatedFormat::default())
        .state_changed(move |_, _, old, new| {
            debug!(?old, ?new, "PipeWire stream state changed");
            match new {
                StreamState::Error(msg) => {
                    error!(%msg, "PipeWire stream error");
                    shared_state_changed.running.store(false, Ordering::SeqCst);
                    let _ = tx_error.blocking_send(PwEvent::Error(msg.to_string()));
                    loop_quit.quit();
                }
                StreamState::Unconnected => {
                    debug!("PipeWire stream unconnected");
                }
                _ => {}
            }
        })
        .param_changed(move |_, user_data, id, param| {
            let Some(param) = param else { return };

            // SPA_PARAM_Format = 4
            if id != 4 {
                return;
            }

            debug!(id, "PipeWire param_changed");

            // Parse format info from the raw pod bytes.
            // The SPA format object contains properties for size, format, etc.
            // We extract them using the raw pod data.
            let pod_data = unsafe {
                let pod_ptr = param as *const _ as *const u8;
                let pod_header = &*(pod_ptr as *const pipewire::spa::sys::spa_pod);
                let total_size = std::mem::size_of::<pipewire::spa::sys::spa_pod>() + pod_header.size as usize;
                std::slice::from_raw_parts(pod_ptr, total_size)
            };

            // Simple heuristic: scan for video size and format in the pod.
            // In practice we use spa_format_video_raw_parse, but the Rust
            // bindings don't expose it cleanly. We'll get the format from
            // the first buffer instead and use a reasonable default here.
            let _ = pod_data;

            // The actual format will be determined from buffer metadata
            // when we receive the first frame. For now store zeros.
             user_data.width = 0;
            user_data.height = 0;
        })
        .process({
            let tx = frame_tx.clone();
            let shared_proc = Arc::clone(&shared);
            let loop_quit_proc = mainloop.clone();
            move |stream, user_data| {
                if !shared_proc.running.load(Ordering::SeqCst) {
                    loop_quit_proc.quit();
                    return;
                }

                let mut maybe_buffer = stream.dequeue_buffer();
                let Some(ref mut buffer) = maybe_buffer else {
                    trace!("no buffer dequeued");
                    return;
                };

                let datas = buffer.datas_mut();
                if datas.is_empty() {
                    trace!("buffer has no data planes");
                    return;
                }

                let data = &mut datas[0];

                let chunk = data.chunk();
                let size = chunk.size() as usize;
                let stride = chunk.stride() as u32;

                let Some(slice) = data.data() else {
                    trace!("data plane has no mapped memory");
                    return;
                };

                if size == 0 {
                    trace!("chunk size is 0, skipping");
                    return;
                }

                // Determine dimensions from stride and size if not yet known
                let width = if user_data.width > 0 {
                    user_data.width
                } else if stride > 0 {
                    // Assume 4 bytes per pixel (BGRA/RGBA)
                    let w = stride / 4;
                    user_data.width = w;
                    w
                } else {
                    tracing::trace!("Cannot get width defaulting to 1920");
                    1920 // fallback
                };

                let height = if user_data.height > 0 {
                    user_data.height
                } else if stride > 0 && size > 0 {
                    let h = (size as u32) / stride;
                    user_data.height = h;
                    h
                } else {
                    tracing::trace!("Cannot get height defaulting to 1080");
                    1080 // fallback
                };

                // Default to BGRA - most common PipeWire screen capture format
                let format = if user_data.format > 0 {
                    spa_format_to_pixel(user_data.format).unwrap_or(PixelFormat::Bgra)
                } else {
                    PixelFormat::Bgra
                };

                let bytes = if size <= slice.len() {
                    Bytes::copy_from_slice(&slice[..size])
                } else {
                    Bytes::copy_from_slice(slice)
                };

                let frame = RawFrame {
                    width,
                    height,
                    stride,
                    format,
                    data: bytes,
                    timestamp_us: now_us(),
                };

                if let Err(e) = tx.try_send(PwEvent::Frame(frame)) {
                    match e {
                        mpsc::error::TrySendError::Full(_) => {
                            trace!("frame dropped, channel full");
                        }
                        mpsc::error::TrySendError::Closed(_) => {
                            debug!("frame channel closed, quitting PipeWire loop");
                            shared_proc.running.store(false, Ordering::SeqCst);
                        }
                    }
                }
            }
        })
        .register();

    // Connect the stream without explicit format params – let PipeWire negotiate.
    let params: &mut [&pw::spa::pod::Pod] = &mut [];
    stream
        .connect(
            pw::spa::utils::Direction::Input,
            Some(node_id),
            StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS,
            params,
        )
        .expect("failed to connect PipeWire stream");

    info!("PipeWire stream connected, entering main loop");

    mainloop.run();

    info!("PipeWire main loop exited");
    shared.running.store(false, Ordering::SeqCst);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Screen capture backend using PipeWire via the xdg-desktop-portal ScreenCast
/// interface.
pub struct PipeWireCapture {
    frame_rx: Option<mpsc::Receiver<PwEvent>>,
    pw_thread: Option<thread::JoinHandle<()>>,
    shared: Option<Arc<SharedState>>,
}

impl PipeWireCapture {
    pub fn new() -> Self {
        Self {
            frame_rx: None,
            pw_thread: None,
            shared: None,
        }
    }
}

impl Default for PipeWireCapture {
    fn default() -> Self {
        Self::new()
    }
}

impl ScreenCapture for PipeWireCapture {
    async fn start(&mut self, source: CaptureSource, config: CaptureConfig) -> Result<()> {
        if self.is_running() {
            return Err(FerricastError::Capture(
                "capture session already running".into(),
            ));
        }

        info!(?source, ?config, "starting PipeWire capture");

        

        let (tx, rx) = mpsc::channel::<PwEvent>(8);
        let shared = Arc::new(SharedState {
            running: AtomicBool::new(true),
            width: AtomicUsize::new(0),
            height: AtomicUsize::new(0)
        });
        let portal = open_portal_session(&source, &config, Arc::clone(&shared)).await?;
        let shared_clone = Arc::clone(&shared);

        let handle = thread::Builder::new()
            .name("ferricast-pw".into())
            .spawn(move || {
                run_pw_thread(portal.pw_fd, portal.node_id, config, tx, shared_clone);
            })
            .map_err(|e| FerricastError::Capture(format!("failed to spawn PW thread: {e}")))?;

        self.frame_rx = Some(rx);
        self.pw_thread = Some(handle);
        self.shared = Some(shared);

        info!("PipeWire capture started");
        Ok(())
    }

    async fn next_frame(&mut self) -> Result<RawFrame> {
        let rx = self
            .frame_rx
            .as_mut()
            .ok_or_else(|| FerricastError::Capture("capture not started".into()))?;

        loop {
            match rx.try_recv() {
                Ok(PwEvent::Frame(frame)) => return Ok(frame),
                Ok(PwEvent::Error(msg)) => {
                    return Err(FerricastError::Capture(format!("PipeWire error: {msg}")));
                }
                Err(e) => {
                    println!("{:?}", e);
                    return Err(FerricastError::Capture(
                        "PipeWire stream ended unexpectedly".into(),
                    ));
                }
            }
        }
    }

    async fn stop(&mut self) -> Result<()> {
        info!("stopping PipeWire capture");

        if let Some(ref shared) = self.shared {
            shared.running.store(false, Ordering::SeqCst);
        }
        self.shared.take();
        self.frame_rx.take();

        if let Some(handle) = self.pw_thread.take() {
            let _ = handle.join();
        }

        info!("PipeWire capture stopped");
        Ok(())
    }

    fn get_pixel_format(&self) -> PixelFormat {
        PixelFormat::Bgra
    }
    
    fn get_screen_size(&self) -> (usize, usize) {
        let (w, h) = self.shared.as_ref().map(|s| (s.width.load(Ordering::SeqCst), s.height.load(Ordering::SeqCst))).unwrap_or((0, 0));
        

        (w, h)    
    }

    fn is_running(&self) -> bool {
        self.shared
            .as_ref()
            .map(|s| s.running.load(Ordering::SeqCst))
            .unwrap_or(false)
    }
}

impl Drop for PipeWireCapture {
    fn drop(&mut self) {
        if let Some(ref shared) = self.shared {
            shared.running.store(false, Ordering::SeqCst);
        }
        self.frame_rx.take();
        if let Some(handle) = self.pw_thread.take() {
            let _ = handle.join();
        }
    }
}
