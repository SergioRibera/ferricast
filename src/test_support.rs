//! In-memory capture, encoder and session used by the pipeline tests.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use ferricast_core::{
    AudioCodec, CaptureConfig, CaptureSource, CapturedFrame, CastSession, Codec, ConnectOutcome,
    Device, DeviceCapabilities, EncodedFrame, EncoderConfig, FerricastError, PairingResponse,
    PixelFormat, RawFrame, Result, ScreenCapture, StreamConfig, VideoEncoder,
};
use uuid::Uuid;

use crate::stream::{ErasedSession, SessionFactory};

pub(crate) fn device(caps: DeviceCapabilities) -> Device {
    Device {
        id: Uuid::new_v4(),
        name: "test receiver".into(),
        protocol: "fake",
        protocol_icon: Bytes::new(),
        addresses: std::net::SocketAddr::from(([127, 0, 0, 1], 0)).into(),
        model: None,
        capabilities: caps,
        metadata: std::collections::HashMap::new(),
    }
}

/// Emits solid BGRA frames; the size follows `sizes` (one entry per
/// frame, the last one repeats).
pub(crate) struct FakeCapture {
    sizes: VecDeque<(u32, u32)>,
    current: (u32, u32),
    ts: u64,
    first_frame_delay: Duration,
}

impl FakeCapture {
    pub(crate) fn new(sizes: &[(u32, u32)]) -> Self {
        let sizes: VecDeque<_> = sizes.iter().copied().collect();
        let current = sizes.front().copied().unwrap_or((64, 36));
        Self {
            sizes,
            current,
            ts: 1_700_000_000_000_000,
            first_frame_delay: Duration::ZERO,
        }
    }

    /// Hold the first frame back, like a portal dialog left open.
    pub(crate) fn with_first_frame_delay(mut self, delay: Duration) -> Self {
        self.first_frame_delay = delay;
        self
    }
}

impl ScreenCapture for FakeCapture {
    async fn start(&mut self, _source: CaptureSource, _config: CaptureConfig) -> Result<()> {
        Ok(())
    }

    async fn next_frame(&mut self) -> Result<CapturedFrame> {
        tokio::time::sleep(Duration::from_millis(5) + std::mem::take(&mut self.first_frame_delay))
            .await;
        if let Some(size) = self.sizes.pop_front() {
            self.current = size;
        }
        let (w, h) = self.current;
        self.ts += 5_000;
        Ok(CapturedFrame::Cpu(RawFrame {
            colour: None,
            width: w,
            height: h,
            stride: w * 4,
            format: PixelFormat::Bgra,
            data: Bytes::from(vec![0x80; (w * h * 4) as usize]),
            timestamp_us: self.ts,
        }))
    }

    async fn stop(&mut self) -> Result<()> {
        Ok(())
    }

    fn is_running(&self) -> bool {
        true
    }

    fn get_pixel_format(&self) -> PixelFormat {
        PixelFormat::Bgra
    }

    fn get_screen_size(&self) -> (usize, usize) {
        (self.current.0 as usize, self.current.1 as usize)
    }

    fn get_framerate(&self) -> u32 {
        30
    }
}

/// What the fake encoder saw.
#[derive(Debug, Default)]
pub(crate) struct EncoderLog {
    pub configured: Vec<EncoderConfig>,
    pub frame_sizes: Vec<(u32, u32)>,
    pub keyframe_requests: usize,
    pub bitrates: Vec<u32>,
}

#[derive(Clone, Default)]
pub(crate) struct FakeEncoder(pub Arc<Mutex<EncoderLog>>);

impl VideoEncoder for FakeEncoder {
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, config: &EncoderConfig) -> ferricast_core::CodecResult<()> {
        self.0.lock().unwrap().configured.push(config.clone());
        Ok(())
    }

    fn encode(&mut self, frame: CapturedFrame) -> ferricast_core::CodecResult<EncodedFrame> {
        self.0
            .lock()
            .unwrap()
            .frame_sizes
            .push((frame.width(), frame.height()));
        Ok(EncodedFrame {
            codec: Codec::H264,
            data: Bytes::from_static(&[0, 0, 0, 1, 0x65]),
            timestamp_us: frame.timestamp_us(),
            is_keyframe: false,
            duration_us: None,
            pts_dts: (0, 0),
        })
    }

    fn flush(self) -> ferricast_core::CodecResult<Vec<EncodedFrame>> {
        Ok(Vec::new())
    }

    fn get_headers(&mut self) -> ferricast_core::CodecResult<Vec<u8>> {
        Ok(Vec::new())
    }

    fn request_keyframe(&mut self) {
        self.0.lock().unwrap().keyframe_requests += 1;
    }

    fn set_bitrate_kbps(&mut self, kbps: u32) -> ferricast_core::CodecResult<()> {
        self.0.lock().unwrap().bitrates.push(kbps);
        Ok(())
    }
}

/// How one session (in creation order) behaves.
#[derive(Debug, Clone)]
pub(crate) struct SessionScript {
    /// `is_alive` turns false after this many frames.
    pub dies_after: Option<usize>,
    /// `send_frame` fails with `RemoteStopped` after this many frames.
    pub remote_stop_after: Option<usize>,
    pub audio_codec: AudioCodec,
    pub feeds_adaptive: bool,
    pub forced_keyframes: Option<Duration>,
    pub connect_fails: bool,
    /// When the session dies (`dies_after`), `end_reason` reports a
    /// receiver-side stop.
    pub dies_remotely: bool,
    /// `setup_stream` fails with `CredentialRequired` until this password
    /// was submitted.
    pub password: Option<&'static str>,
}

impl Default for SessionScript {
    fn default() -> Self {
        Self {
            dies_after: None,
            remote_stop_after: None,
            audio_codec: AudioCodec::Aac,
            feeds_adaptive: false,
            forced_keyframes: None,
            connect_fails: false,
            dies_remotely: false,
            password: None,
        }
    }
}

/// What the fake sessions saw, across reconnects.
#[derive(Debug, Default)]
pub(crate) struct SessionLog {
    pub created: usize,
    pub setups: Vec<StreamConfig>,
    pub frames: usize,
    /// Timestamp of the first frame sent.
    pub first_frame_us: Option<u64>,
    pub stops: usize,
    /// Passwords submitted through `submit_pairing`.
    pub passwords: Vec<String>,
}

pub(crate) struct FakeSession {
    script: SessionScript,
    log: Arc<Mutex<SessionLog>>,
    frames: usize,
    remote_end: bool,
    submitted: Option<String>,
}

impl CastSession for FakeSession {
    async fn connect(&mut self, _device: &Device) -> Result<ConnectOutcome> {
        if self.script.connect_fails {
            return Err(FerricastError::Connection("unreachable".into()));
        }
        Ok(ConnectOutcome::Ready)
    }

    async fn submit_pairing(&mut self, response: PairingResponse) -> Result<()> {
        match response {
            PairingResponse::Password(password) => {
                self.log.lock().unwrap().passwords.push(password.clone());
                self.submitted = Some(password);
                Ok(())
            }
            PairingResponse::Cancelled => Err(FerricastError::Protocol("cancelled".into())),
            PairingResponse::Pin(_) | PairingResponse::Confirmed => Ok(()),
        }
    }

    async fn setup_stream(&mut self, config: &StreamConfig) -> Result<()> {
        if let Some(want) = self.script.password
            && self.submitted.as_deref() != Some(want)
        {
            return Err(FerricastError::CredentialRequired {
                rejected: self.submitted.is_some(),
            });
        }
        self.log.lock().unwrap().setups.push(config.clone());
        Ok(())
    }

    async fn send_frame(&mut self, frame: &EncodedFrame) -> Result<()> {
        self.frames += 1;
        let mut log = self.log.lock().unwrap();
        log.frames += 1;
        log.first_frame_us.get_or_insert(frame.timestamp_us);
        drop(log);
        if self
            .script
            .remote_stop_after
            .is_some_and(|n| self.frames > n)
        {
            return Err(FerricastError::RemoteStopped("stopped on the TV".into()));
        }
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        self.log.lock().unwrap().stops += 1;
        Ok(())
    }

    fn is_alive(&self) -> bool {
        self.script.dies_after.is_none_or(|n| self.frames < n)
    }

    fn end_reason(&self) -> Option<FerricastError> {
        if CastSession::is_alive(self) || !self.remote_end {
            return None;
        }
        Some(FerricastError::RemoteStopped(
            "another app took over".into(),
        ))
    }

    fn preferred_audio_codec(&self) -> AudioCodec {
        self.script.audio_codec
    }

    fn feeds_adaptive_bitrate(&self) -> bool {
        self.script.feeds_adaptive
    }

    fn forced_keyframe_interval(&self) -> Option<Duration> {
        self.script.forced_keyframes
    }
}

/// Session factory handing out sessions that follow `scripts` in
/// order (the last one repeats).
pub(crate) fn session_factory(
    scripts: Vec<SessionScript>,
) -> (SessionFactory, Arc<Mutex<SessionLog>>) {
    let log = Arc::new(Mutex::new(SessionLog::default()));
    let log_for_factory = log.clone();
    let factory: SessionFactory = Arc::new(move |_device: &Device| {
        let mut log = log_for_factory.lock().unwrap();
        let script = scripts
            .get(log.created)
            .or(scripts.last())
            .cloned()
            .unwrap_or_default();
        log.created += 1;
        let remote_end = script.dies_remotely;
        Ok(Box::new(FakeSession {
            script,
            log: log_for_factory.clone(),
            frames: 0,
            remote_end,
            submitted: None,
        }) as Box<dyn ErasedSession>)
    });
    (factory, log)
}
