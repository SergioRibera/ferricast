//! Media Foundation plumbing shared by the Windows encoders
//! (`ferricast-encoder`) and decoders (`ferricast-decoder`), so both
//! drive MFTs the same way.
//!
//! # Threading model
//!
//! Every codec session runs its MFT on a [`Worker`] thread of its own:
//! the thread enters the COM multithreaded apartment, starts Media
//! Foundation, creates the transform and owns every COM pointer until
//! the session drops. Calls reach it as jobs over a channel, so the
//! session handle is `Send` without asserting anything about the
//! threading of COM objects, and a transform is only ever called from
//! the thread that created it. The cost is one channel round trip per
//! call.
//!
//! `MFStartup` runs once per process and `MFShutdown` never: sessions
//! may live until exit, and Media Foundation's own refcount would make
//! a shutdown from one session tear down the others.
//!
//! # Synchronous and asynchronous MFTs
//!
//! Software MFTs are synchronous: `ProcessInput`, then `ProcessOutput`
//! until it asks for more input. Hardware MFTs are asynchronous: they
//! are unlocked with `MF_TRANSFORM_ASYNC_UNLOCK` and announce through
//! their event generator when they accept input (`METransformNeedInput`)
//! and when output is ready (`METransformHaveOutput`). [`Transform`]
//! hides the difference behind [`Transform::push`], [`Transform::pull`]
//! and [`Transform::drain`].

use std::mem::ManuallyDrop;
use std::sync::OnceLock;
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ferricast_core::{
    BackendId, ChromaLocation, Chromaticity, CodecError, CodecResult, ColourDescription,
    ColourPrimaries, ColourRange, ContentLightLevel, HdrMetadata, MasteringDisplay,
    MatrixCoefficients, Operation, TransferCharacteristics,
};
use windows::Win32::Foundation::RPC_E_TIMEOUT;
use windows::Win32::Media::MediaFoundation::{
    ICodecAPI, IMFActivate, IMFMediaBuffer, IMFMediaEventGenerator, IMFMediaType, IMFSample,
    IMFTransform, METransformDrainComplete, METransformHaveOutput, METransformNeedInput,
    MF_E_NO_EVENTS_AVAILABLE, MF_E_NOTACCEPTING, MF_E_TRANSFORM_NEED_MORE_INPUT,
    MF_E_TRANSFORM_STREAM_CHANGE, MF_EVENT_FLAG_NO_WAIT, MF_EVENT_TYPE, MF_LOW_LATENCY,
    MF_MT_CUSTOM_VIDEO_PRIMARIES, MF_MT_MAJOR_TYPE, MF_MT_MAX_FRAME_AVERAGE_LUMINANCE_LEVEL,
    MF_MT_MAX_LUMINANCE_LEVEL, MF_MT_MAX_MASTERING_LUMINANCE, MF_MT_MIN_MASTERING_LUMINANCE,
    MF_MT_SUBTYPE, MF_MT_TRANSFER_FUNCTION, MF_MT_VIDEO_CHROMA_SITING, MF_MT_VIDEO_NOMINAL_RANGE,
    MF_MT_VIDEO_PRIMARIES, MF_MT_YUV_MATRIX, MF_TRANSFORM_ASYNC, MF_TRANSFORM_ASYNC_UNLOCK,
    MF_VERSION, MFCreateAlignedMemoryBuffer, MFCreateMediaType, MFCreateSample, MFSTARTUP_NOSOCKET,
    MFStartup, MFT_ENUM_FLAG, MFT_ENUM_FLAG_HARDWARE, MFT_ENUM_FLAG_SORTANDFILTER,
    MFT_ENUM_FLAG_SYNCMFT, MFT_FRIENDLY_NAME_Attribute, MFT_MESSAGE_COMMAND_DRAIN,
    MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, MFT_MESSAGE_NOTIFY_END_OF_STREAM,
    MFT_MESSAGE_NOTIFY_START_OF_STREAM, MFT_OUTPUT_DATA_BUFFER,
    MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES, MFT_OUTPUT_STREAM_PROVIDES_SAMPLES,
    MFT_REGISTER_TYPE_INFO, MFTEnumEx,
};
use windows::Win32::System::Com::{
    COINIT_MULTITHREADED, CoInitializeEx, CoTaskMemFree, CoUninitialize,
};
use windows::Win32::System::Variant::{VARIANT, VARIANT_0, VARIANT_0_0, VARIANT_0_0_0, VT_UI4};
use windows::core::{GUID, Interface, PWSTR};

mod d3d11;

pub use d3d11::{D3d11Texture, Device, Rect};

/// How long an asynchronous MFT may go without sending an event the
/// session waits for before the call fails.
pub const EVENT_TIMEOUT: Duration = Duration::from_secs(2);

/// How long dropping a session waits for its worker to finish.
const JOIN_TIMEOUT: Duration = Duration::from_secs(5);

/// The backend every Media Foundation codec reports.
pub const BACKEND: BackendId = BackendId::MediaFoundation;

/// A failed Media Foundation call as a codec error naming the call and
/// its HRESULT.
#[must_use]
pub fn failure(operation: Operation, call: &str, error: &windows::core::Error) -> CodecError {
    CodecError::backend(
        BACKEND,
        operation,
        format!(
            "{call} failed with HRESULT {:#010X}: {}",
            error.code().0,
            error.message()
        ),
    )
}

/// Attaches the failing call to a Media Foundation result.
pub trait MfResult<T> {
    /// The value, or a [`CodecError::Backend`] naming `call`.
    ///
    /// # Errors
    /// The failure, as a codec error naming the call.
    fn mf(self, operation: Operation, call: &str) -> CodecResult<T>;
}

impl<T> MfResult<T> for windows::core::Result<T> {
    fn mf(self, operation: Operation, call: &str) -> CodecResult<T> {
        self.map_err(|error| failure(operation, call, &error))
    }
}

/// The COM multithreaded apartment of the current thread, left on drop.
struct Apartment;

impl Apartment {
    fn enter() -> CodecResult<Self> {
        // SAFETY: called once on a thread this module spawned; the
        // matching `CoUninitialize` runs in `Drop` on the same thread.
        let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        hr.ok()
            .mf(Operation::Open, "CoInitializeEx(COINIT_MULTITHREADED)")?;
        Ok(Self)
    }
}

impl Drop for Apartment {
    fn drop(&mut self) {
        // SAFETY: balances the successful `CoInitializeEx` of `enter`,
        // on the same thread, after every COM object of the thread is
        // gone (the worker drops its state first).
        unsafe { CoUninitialize() };
    }
}

/// Starts Media Foundation once per process. The first outcome sticks:
/// a machine without Media Foundation (Windows N editions without the
/// Media Feature Pack) reports it for every session.
fn startup() -> CodecResult<()> {
    static STARTED: OnceLock<Result<(), String>> = OnceLock::new();
    STARTED
        .get_or_init(|| {
            // SAFETY: plain call; COM is initialised on this thread.
            unsafe { MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET) }.map_err(|error| {
                format!(
                    "MFStartup failed with HRESULT {:#010X}: {} (on Windows N editions, \
                     install the Media Feature Pack)",
                    error.code().0,
                    error.message()
                )
            })
        })
        .clone()
        .map_err(|reason| CodecError::BackendUnavailable {
            backend: BACKEND,
            reason,
        })
}

type Job<S> = Box<dyn FnOnce(&mut S) + Send>;

/// A thread that owns one codec session's COM objects (see the module
/// docs). Dropping it ends the session and joins the thread.
#[allow(
    missing_debug_implementations,
    reason = "holds COM interfaces, which have no useful Debug"
)]
pub struct Worker<S> {
    jobs: Option<mpsc::Sender<Job<S>>>,
    thread: Option<JoinHandle<()>>,
}

impl<S: 'static> Worker<S> {
    /// Spawns the thread and builds the session state on it with `init`.
    ///
    /// # Errors
    /// The thread cannot start, COM or Media Foundation fail to start, or
    /// `init` fails.
    pub fn spawn(
        name: &str,
        init: impl FnOnce() -> CodecResult<S> + Send + 'static,
    ) -> CodecResult<Self> {
        let (jobs_tx, jobs_rx) = mpsc::channel::<Job<S>>();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<CodecResult<()>>(1);
        let thread = std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                let apartment = match Apartment::enter() {
                    Ok(apartment) => apartment,
                    Err(error) => {
                        ready_tx.send(Err(error)).ok();
                        return;
                    }
                };
                match startup().and_then(|()| init()) {
                    Ok(mut state) => {
                        ready_tx.send(Ok(())).ok();
                        while let Ok(job) = jobs_rx.recv() {
                            job(&mut state);
                        }
                        // The COM objects go before the apartment.
                        drop(state);
                    }
                    Err(error) => {
                        ready_tx.send(Err(error)).ok();
                    }
                }
                drop(apartment);
            })
            .map_err(|error| {
                CodecError::backend(
                    BACKEND,
                    Operation::Open,
                    format!("spawning {name}: {error}"),
                )
            })?;
        let worker = Self {
            jobs: Some(jobs_tx),
            thread: Some(thread),
        };
        ready_rx.recv().map_err(|_| {
            CodecError::backend(
                BACKEND,
                Operation::Open,
                format!("{name} exited while starting"),
            )
        })??;
        Ok(worker)
    }

    /// Runs `job` on the worker and waits for its result.
    ///
    /// # Errors
    /// `job`'s error, or [`CodecError::Backend`] when the worker is gone
    /// (a previous job panicked).
    pub fn run<R: Send + 'static>(
        &self,
        operation: Operation,
        job: impl FnOnce(&mut S) -> CodecResult<R> + Send + 'static,
    ) -> CodecResult<R> {
        let gone = || {
            CodecError::backend(
                BACKEND,
                operation,
                "the Media Foundation worker thread has exited",
            )
        };
        let (tx, rx) = mpsc::sync_channel(1);
        self.jobs
            .as_ref()
            .ok_or_else(gone)?
            .send(Box::new(move |state: &mut S| {
                tx.send(job(state)).ok();
            }))
            .map_err(|_| gone())?;
        rx.recv().map_err(|_| gone())?
    }
}

impl<S> Drop for Worker<S> {
    fn drop(&mut self) {
        drop(self.jobs.take());
        let Some(thread) = self.thread.take() else {
            return;
        };
        // Every wait inside a job is bounded, but a driver call that never
        // returns must not hang the session owner: past the timeout the
        // thread is left to finish (and free its COM objects) on its own.
        let deadline = Instant::now() + JOIN_TIMEOUT;
        while !thread.is_finished() {
            if Instant::now() >= deadline {
                tracing::warn!(
                    "a Media Foundation worker thread did not stop within {} s; detaching it",
                    JOIN_TIMEOUT.as_secs()
                );
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        if thread.join().is_err() {
            tracing::warn!("a Media Foundation worker thread panicked");
        }
    }
}

/// Whether to enumerate hardware (asynchronous, vendor) or software
/// (synchronous, mostly Microsoft's) transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A hardware (vendor) transform, usually asynchronous.
    Hardware,
    /// A software transform, synchronous.
    Software,
}

impl Kind {
    fn flags(self) -> MFT_ENUM_FLAG {
        match self {
            Self::Hardware => MFT_ENUM_FLAG_HARDWARE | MFT_ENUM_FLAG_SORTANDFILTER,
            Self::Software => MFT_ENUM_FLAG_SYNCMFT | MFT_ENUM_FLAG_SORTANDFILTER,
        }
    }

    #[must_use]
    /// The label logs and errors use.
    pub fn label(self) -> &'static str {
        match self {
            Self::Hardware => "hardware",
            Self::Software => "software",
        }
    }
}

/// What an enumeration looks for: a category and the stream types.
#[derive(Debug, Clone, Copy)]
pub struct Search {
    /// The MFT category to enumerate.
    pub category: GUID,
    /// Hardware or software transforms.
    pub kind: Kind,
    /// `(major type, subtype)` of the input, `None` for any.
    pub input: Option<(GUID, GUID)>,
    /// `(major type, subtype)` of the output, `None` for any.
    pub output: Option<(GUID, GUID)>,
    /// What the transform does, for messages ("H.264 encoder").
    pub job: &'static str,
}

impl Search {
    /// The registered transforms that match, best first. Call on a
    /// thread in a COM apartment (a [`Worker`]).
    ///
    /// # Errors
    /// [`CodecError::BackendUnavailable`] when none is installed.
    pub fn enumerate(&self) -> CodecResult<Vec<IMFActivate>> {
        let info = |(major, sub): (GUID, GUID)| MFT_REGISTER_TYPE_INFO {
            guidMajorType: major,
            guidSubtype: sub,
        };
        let input = self.input.map(info);
        let output = self.output.map(info);
        let mut array: *mut Option<IMFActivate> = std::ptr::null_mut();
        let mut count = 0u32;
        // SAFETY: the type infos outlive the call; `array` and `count`
        // receive a CoTaskMem array of `count` activation objects.
        unsafe {
            MFTEnumEx(
                self.category,
                self.kind.flags(),
                input.as_ref().map(std::ptr::from_ref),
                output.as_ref().map(std::ptr::from_ref),
                &raw mut array,
                &raw mut count,
            )
        }
        .mf(Operation::Probe, "MFTEnumEx")?;
        let mut found = Vec::new();
        if !array.is_null() {
            // SAFETY: MFTEnumEx returned `count` initialised entries at
            // `array`; each is taken (moved out) exactly once, then the
            // array itself is freed with the allocator that made it.
            unsafe {
                let entries = std::slice::from_raw_parts_mut(array, count as usize);
                found.extend(entries.iter_mut().filter_map(Option::take));
                CoTaskMemFree(Some(array.cast_const().cast()));
            }
        }
        if found.is_empty() {
            return Err(CodecError::BackendUnavailable {
                backend: BACKEND,
                reason: format!(
                    "no {} {} transform is installed",
                    self.kind.label(),
                    self.job
                ),
            });
        }
        Ok(found)
    }

    /// Activates each match in turn and keeps the first `setup` accepts.
    ///
    /// # Errors
    /// [`CodecError::BackendUnavailable`] when nothing is installed, or
    /// the last transform's `setup` error (every rejection is in the log).
    pub fn open<T>(&self, mut setup: impl FnMut(Transform) -> CodecResult<T>) -> CodecResult<T> {
        let mut last = None;
        for activate in self.enumerate()? {
            let name = friendly_name(&activate);
            let attempt = Transform::activate(&activate, name.clone()).and_then(&mut setup);
            match attempt {
                Ok(value) => {
                    tracing::info!(transform = %name, kind = self.kind.label(), job = self.job, "Media Foundation transform opened");
                    return Ok(value);
                }
                Err(error) => {
                    tracing::debug!(transform = %name, %error, "Media Foundation transform rejected");
                    last = Some(error);
                }
            }
        }
        Err(last.unwrap_or_else(|| CodecError::BackendUnavailable {
            backend: BACKEND,
            reason: format!(
                "no {} {} transform could be opened",
                self.kind.label(),
                self.job
            ),
        }))
    }

    /// A configuration-free availability check for capability reports.
    ///
    /// # Errors
    /// The enumeration's error (nothing installed, Media Foundation
    /// missing).
    pub fn probe(self) -> CodecResult<()> {
        Worker::spawn("ferricast-mf-probe", move || self.enumerate().map(drop)).map(drop)
    }
}

/// The transform's display name, for logs.
fn friendly_name(activate: &IMFActivate) -> String {
    let mut text = PWSTR::null();
    let mut len = 0u32;
    // SAFETY: on success `text` is a CoTaskMem string of `len` UTF-16
    // units, read once and freed below.
    let read = unsafe {
        activate.GetAllocatedString(&MFT_FRIENDLY_NAME_Attribute, &raw mut text, &raw mut len)
    };
    if read.is_err() || text.is_null() {
        return "unnamed transform".to_owned();
    }
    // SAFETY: see above; `text` is valid for `len` units until freed.
    unsafe {
        let name = String::from_utf16_lossy(std::slice::from_raw_parts(text.0, len as usize));
        CoTaskMemFree(Some(text.0.cast_const().cast()));
        name
    }
}

/// What an MFT handed back.
#[allow(
    missing_debug_implementations,
    reason = "holds COM interfaces, which have no useful Debug"
)]
pub enum Output {
    /// A finished sample.
    Sample(IMFSample),
    /// The output type changed (and was renegotiated); re-read it.
    FormatChanged,
}

/// Picks a new output type after `MF_E_TRANSFORM_STREAM_CHANGE`.
pub type Renegotiate = fn(&IMFTransform) -> windows::core::Result<()>;

/// One activated MFT (see the module docs).
#[allow(
    missing_debug_implementations,
    reason = "holds COM interfaces, which have no useful Debug"
)]
pub struct Transform {
    mft: IMFTransform,
    name: String,
    /// The event generator of an asynchronous MFT.
    events: Option<IMFMediaEventGenerator>,
    /// `METransformNeedInput` events not yet answered with input.
    need_input: u32,
    /// Whether the MFT allocates its output samples.
    provides_samples: bool,
    /// Output buffer size the MFT asks for when it does not.
    output_size: u32,
    /// Fallback output buffer size when the MFT names none.
    default_output_size: u32,
    renegotiate: Renegotiate,
    /// An asynchronous drain has completed.
    drained: bool,
}

/// A renegotiation that keeps the first output type the MFT offers.
///
/// # Errors
/// The MFT's refusal of the type.
pub fn first_available(mft: &IMFTransform) -> windows::core::Result<()> {
    // SAFETY: plain COM calls on a live transform.
    unsafe {
        let offered = mft.GetOutputAvailableType(0, 0)?;
        mft.SetOutputType(0, &offered, 0)
    }
}

impl Transform {
    fn activate(activate: &IMFActivate, name: String) -> CodecResult<Self> {
        // SAFETY: plain COM calls on a live activation object.
        let mft: IMFTransform = unsafe { activate.ActivateObject() }
            .mf(Operation::Open, "IMFActivate::ActivateObject")?;
        // SAFETY: plain COM calls on the live transform.
        let events = unsafe {
            match mft.GetAttributes() {
                Ok(attributes) if attributes.GetUINT32(&MF_TRANSFORM_ASYNC).unwrap_or(0) != 0 => {
                    attributes
                        .SetUINT32(&MF_TRANSFORM_ASYNC_UNLOCK, 1)
                        .mf(Operation::Open, "setting MF_TRANSFORM_ASYNC_UNLOCK")?;
                    Some(
                        mft.cast::<IMFMediaEventGenerator>()
                            .mf(Operation::Open, "QueryInterface(IMFMediaEventGenerator)")?,
                    )
                }
                _ => None,
            }
        };
        Ok(Self {
            mft,
            name,
            events,
            need_input: 0,
            provides_samples: false,
            output_size: 0,
            default_output_size: 1 << 20,
            renegotiate: first_available,
            drained: false,
        })
    }

    /// The transform.
    #[must_use]
    pub fn mft(&self) -> &IMFTransform {
        &self.mft
    }

    /// Its display name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// How to pick the output type after a stream change.
    #[allow(
        dead_code,
        reason = "shared with ferricast-decoder, which uses it; the encoders do not"
    )]
    pub fn set_renegotiate(&mut self, renegotiate: Renegotiate) {
        self.renegotiate = renegotiate;
    }

    /// Output buffer size used when the MFT does not state one.
    pub fn set_default_output_size(&mut self, bytes: u32) {
        self.default_output_size = bytes;
    }

    /// Asks for low-latency processing (`MF_LOW_LATENCY`, the attribute
    /// form of `CODECAPI_AVLowLatencyMode`); transforms without the
    /// attribute store are left as they are.
    pub fn request_low_latency(&self) {
        // SAFETY: plain COM calls on the live transform.
        let set = unsafe {
            self.mft
                .GetAttributes()
                .and_then(|a| a.SetUINT32(&MF_LOW_LATENCY, 1))
        };
        if let Err(error) = set {
            tracing::debug!(transform = %self.name, %error, "MF_LOW_LATENCY not accepted");
        }
    }

    /// The transform's `ICodecAPI`, when it has one.
    #[allow(
        dead_code,
        reason = "shared with ferricast-encoder, which uses it; the decoders do not"
    )]
    #[must_use]
    pub fn codec_api(&self) -> Option<ICodecAPI> {
        self.mft.cast::<ICodecAPI>().ok()
    }

    /// Starts streaming once the media types are set.
    ///
    /// # Errors
    /// The `ProcessMessage` failure.
    pub fn start(&mut self) -> CodecResult<()> {
        self.refresh_output_info()?;
        // SAFETY: plain COM calls on the live transform.
        unsafe {
            self.mft
                .ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0)
                .mf(Operation::Open, "MFT_MESSAGE_NOTIFY_BEGIN_STREAMING")?;
            self.mft
                .ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0)
                .mf(Operation::Open, "MFT_MESSAGE_NOTIFY_START_OF_STREAM")?;
        }
        Ok(())
    }

    fn refresh_output_info(&mut self) -> CodecResult<()> {
        // SAFETY: plain COM call on the live transform.
        let info = unsafe { self.mft.GetOutputStreamInfo(0) }
            .mf(Operation::Configure, "IMFTransform::GetOutputStreamInfo")?;
        let provides = flag(MFT_OUTPUT_STREAM_PROVIDES_SAMPLES.0)
            | flag(MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES.0);
        self.provides_samples = info.dwFlags & provides != 0;
        self.output_size = info.cbSize;
        Ok(())
    }

    /// Feeds one input sample, collecting into `outputs` whatever the
    /// MFT finishes while it waits for room.
    ///
    /// # Errors
    /// The `ProcessInput` / `ProcessOutput` / event failure, the raw
    /// HRESULT kept so the caller can classify it.
    pub fn push(
        &mut self,
        sample: &IMFSample,
        outputs: &mut Vec<Output>,
    ) -> windows::core::Result<()> {
        if self.events.is_some() {
            while self.need_input == 0 {
                self.wait_event("METransformNeedInput", outputs)?;
            }
            self.need_input -= 1;
            // SAFETY: the MFT asked for input; plain COM call.
            return unsafe { self.mft.ProcessInput(0, sample, 0) };
        }
        // SAFETY: plain COM calls on the live transform.
        match unsafe { self.mft.ProcessInput(0, sample, 0) } {
            Err(error) if error.code() == MF_E_NOTACCEPTING => {
                // Full: take its output, then it takes the input.
                self.pull(outputs, None)?;
                // SAFETY: as above.
                unsafe { self.mft.ProcessInput(0, sample, 0) }
            }
            other => other,
        }
    }

    /// Collects every output the MFT has ready. An asynchronous MFT is
    /// given up to `wait` for its first output when none is ready yet,
    /// unless it asks for more input first (it then buffers the frame).
    ///
    /// # Errors
    /// The `ProcessOutput` / event failure.
    pub fn pull(
        &mut self,
        outputs: &mut Vec<Output>,
        wait: Option<Duration>,
    ) -> windows::core::Result<()> {
        if self.events.is_none() {
            while self.process_output(outputs)? {}
            return Ok(());
        }
        let deadline = wait.map(|wait| Instant::now() + wait);
        loop {
            let before = (outputs.len(), self.need_input);
            if !self.next_event(outputs)? {
                let waiting = deadline.is_some_and(|deadline| Instant::now() < deadline);
                if outputs.is_empty() && self.need_input == 0 && waiting {
                    std::thread::sleep(Duration::from_millis(1));
                    continue;
                }
                return Ok(());
            }
            if wait.is_some()
                && outputs.len() == before.0
                && self.need_input > before.1
                && outputs.is_empty()
            {
                // It wants more input before it outputs: the frame is
                // buffered inside the MFT.
                return Ok(());
            }
        }
    }

    /// Drains the MFT at the end of the stream.
    ///
    /// # Errors
    /// The message / `ProcessOutput` / event failure.
    pub fn drain(&mut self, outputs: &mut Vec<Output>) -> windows::core::Result<()> {
        // SAFETY: plain COM calls on the live transform.
        unsafe {
            self.mft
                .ProcessMessage(MFT_MESSAGE_NOTIFY_END_OF_STREAM, 0)?;
            self.mft.ProcessMessage(MFT_MESSAGE_COMMAND_DRAIN, 0)?;
        }
        if self.events.is_none() {
            self.pull(outputs, None)?;
            return self.restart();
        }
        self.drained = false;
        while !self.drained {
            self.wait_event("METransformDrainComplete", outputs)?;
        }
        self.restart()
    }

    /// Opens the next stream after a drain, so the session stays usable.
    fn restart(&mut self) -> windows::core::Result<()> {
        // SAFETY: plain COM call on the live transform.
        unsafe {
            self.mft
                .ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0)
        }
    }

    /// Handles the next event of an asynchronous MFT, polling for up to
    /// [`EVENT_TIMEOUT`]: a hardware MFT that stops sending events (a
    /// driver hang) fails the call instead of blocking the worker forever.
    fn wait_event(
        &mut self,
        awaited: &str,
        outputs: &mut Vec<Output>,
    ) -> windows::core::Result<()> {
        let deadline = Instant::now() + EVENT_TIMEOUT;
        loop {
            if self.next_event(outputs)? {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(windows::core::Error::new(
                    RPC_E_TIMEOUT,
                    format!(
                        "{} sent no {awaited} event within {} ms",
                        self.name,
                        EVENT_TIMEOUT.as_millis()
                    ),
                ));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Handles one pending event of an asynchronous MFT; `false` when
    /// none was pending.
    fn next_event(&mut self, outputs: &mut Vec<Output>) -> windows::core::Result<bool> {
        let Some(events) = &self.events else {
            return Ok(false);
        };
        // SAFETY: plain COM calls on the live event generator.
        let event = match unsafe { events.GetEvent(MF_EVENT_FLAG_NO_WAIT) } {
            Ok(event) => event,
            Err(error) if error.code() == MF_E_NO_EVENTS_AVAILABLE => return Ok(false),
            Err(error) => return Err(error),
        };
        // SAFETY: plain COM calls on the live event.
        let (kind, status) = unsafe { (event.GetType()?, event.GetStatus()?) };
        status.ok()?;
        if is(kind, METransformNeedInput) {
            self.need_input += 1;
        } else if is(kind, METransformHaveOutput) {
            self.process_output(outputs)?;
        } else if is(kind, METransformDrainComplete) {
            self.drained = true;
        }
        Ok(true)
    }

    /// One `ProcessOutput`; `false` once the MFT needs more input.
    fn process_output(&mut self, outputs: &mut Vec<Output>) -> windows::core::Result<bool> {
        let sample = if self.provides_samples {
            None
        } else {
            let size = if self.output_size == 0 {
                self.default_output_size
            } else {
                self.output_size
            };
            // SAFETY: plain MF allocation calls.
            let sample = unsafe {
                let sample = MFCreateSample()?;
                sample.AddBuffer(&MFCreateAlignedMemoryBuffer(size, 15)?)?;
                sample
            };
            Some(sample)
        };
        let mut buffers = [MFT_OUTPUT_DATA_BUFFER {
            dwStreamID: 0,
            pSample: ManuallyDrop::new(sample),
            dwStatus: 0,
            pEvents: ManuallyDrop::new(None),
        }];
        let mut status = 0u32;
        // SAFETY: `buffers` is a valid one-entry array the MFT fills;
        // the sample and events it holds are taken out right after,
        // whatever the outcome, so nothing leaks or drops twice.
        let result = unsafe { self.mft.ProcessOutput(0, &mut buffers, &raw mut status) };
        // SAFETY: see above; each field is taken exactly once.
        let (sample, events) = unsafe {
            (
                ManuallyDrop::take(&mut buffers[0].pSample),
                ManuallyDrop::take(&mut buffers[0].pEvents),
            )
        };
        drop(events);
        match result {
            Ok(()) => {
                if let Some(sample) = sample {
                    outputs.push(Output::Sample(sample));
                }
                Ok(true)
            }
            Err(error) if error.code() == MF_E_TRANSFORM_NEED_MORE_INPUT => Ok(false),
            Err(error) if error.code() == MF_E_TRANSFORM_STREAM_CHANGE => {
                (self.renegotiate)(&self.mft)?;
                self.refresh_output_info().map_err(|_| error)?;
                outputs.push(Output::FormatChanged);
                Ok(true)
            }
            Err(error) => Err(error),
        }
    }
}

fn is(kind: u32, event: MF_EVENT_TYPE) -> bool {
    i32::try_from(kind).is_ok_and(|kind| kind == event.0)
}

/// An MF flag constant (declared `i32`) as the `u32` bit mask fields use.
#[must_use]
pub fn flag(value: i32) -> u32 {
    u32::try_from(value).unwrap_or(0)
}

/// A media type of `major` / `subtype`.
///
/// # Errors
/// The MF failure, attributed to `operation`.
pub fn media_type(major: &GUID, subtype: &GUID, operation: Operation) -> CodecResult<IMFMediaType> {
    // SAFETY: plain MF calls.
    unsafe {
        let media = MFCreateMediaType().mf(operation, "MFCreateMediaType")?;
        media
            .SetGUID(&MF_MT_MAJOR_TYPE, major)
            .mf(operation, "setting MF_MT_MAJOR_TYPE")?;
        media
            .SetGUID(&MF_MT_SUBTYPE, subtype)
            .mf(operation, "setting MF_MT_SUBTYPE")?;
        Ok(media)
    }
}

/// Two `u32`s packed as MF packs sizes and ratios (`MFSetAttributeSize`).
#[must_use]
pub fn pack(high: u32, low: u32) -> u64 {
    (u64::from(high) << 32) | u64::from(low)
}

/// The two halves of a packed attribute.
#[allow(
    dead_code,
    reason = "shared with ferricast-decoder, which uses it; the encoders do not"
)]
#[must_use]
pub fn unpack(value: u64) -> (u32, u32) {
    let high = u32::try_from(value >> 32).unwrap_or(0);
    let low = u32::try_from(value & u64::from(u32::MAX)).unwrap_or(0);
    (high, low)
}

/// Microseconds as MF's 100 ns units.
#[must_use]
pub fn hns(micros: u64) -> i64 {
    i64::try_from(micros.saturating_mul(10)).unwrap_or(i64::MAX)
}

/// MF's 100 ns units as microseconds.
#[must_use]
pub fn micros(hns: i64) -> u64 {
    u64::try_from(hns / 10).unwrap_or(0)
}

/// A sample around a fresh buffer of `len` bytes that `fill` writes.
///
/// # Errors
/// The MF allocation failure, or `fill`'s.
pub fn sample_with(
    len: usize,
    alignment: u32,
    operation: Operation,
    fill: impl FnOnce(&mut [u8]) -> CodecResult<()>,
) -> CodecResult<IMFSample> {
    let size = u32::try_from(len)
        .map_err(|_| CodecError::backend(BACKEND, operation, "input sample exceeds 4 GiB"))?;
    // SAFETY: plain MF allocation calls; the buffer is locked for `fill`
    // and unlocked by the guard.
    unsafe {
        let buffer = MFCreateAlignedMemoryBuffer(size, alignment.saturating_sub(1))
            .mf(operation, "MFCreateAlignedMemoryBuffer")?;
        {
            let mut locked = Locked::new(&buffer, operation)?;
            fill(locked.bytes_mut(len)?)?;
        }
        buffer
            .SetCurrentLength(size)
            .mf(operation, "IMFMediaBuffer::SetCurrentLength")?;
        let sample = MFCreateSample().mf(operation, "MFCreateSample")?;
        sample
            .AddBuffer(&buffer)
            .mf(operation, "IMFSample::AddBuffer")?;
        Ok(sample)
    }
}

/// The bytes of `sample`, copied out.
///
/// # Errors
/// The MF failure.
pub fn sample_bytes(sample: &IMFSample, operation: Operation) -> CodecResult<Vec<u8>> {
    // SAFETY: plain COM call on a live sample.
    let buffer = unsafe { sample.ConvertToContiguousBuffer() }
        .mf(operation, "IMFSample::ConvertToContiguousBuffer")?;
    let locked = Locked::new(&buffer, operation)?;
    Ok(locked.bytes().to_vec())
}

/// A locked media buffer, unlocked on drop.
#[allow(
    missing_debug_implementations,
    reason = "holds COM interfaces, which have no useful Debug"
)]
pub struct Locked<'a> {
    buffer: &'a IMFMediaBuffer,
    data: *mut u8,
    max: u32,
    current: u32,
    operation: Operation,
}

impl<'a> Locked<'a> {
    /// Locks `buffer`.
    ///
    /// # Errors
    /// The `Lock` failure.
    pub fn new(buffer: &'a IMFMediaBuffer, operation: Operation) -> CodecResult<Self> {
        let mut data = std::ptr::null_mut();
        let (mut max, mut current) = (0u32, 0u32);
        // SAFETY: the out-pointers are valid; on success the buffer stays
        // locked until `Drop` unlocks it.
        unsafe { buffer.Lock(&raw mut data, Some(&raw mut max), Some(&raw mut current)) }
            .mf(operation, "IMFMediaBuffer::Lock")?;
        Ok(Self {
            buffer,
            data,
            max,
            current,
            operation,
        })
    }

    /// The valid bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        if self.data.is_null() {
            return &[];
        }
        // SAFETY: a locked buffer holds `current` valid bytes at `data`
        // until it is unlocked, which needs `self`.
        unsafe { std::slice::from_raw_parts(self.data, self.current as usize) }
    }

    /// The first `len` bytes of the buffer's capacity, for writing.
    fn bytes_mut(&mut self, len: usize) -> CodecResult<&mut [u8]> {
        if self.data.is_null() || len > self.max as usize {
            return Err(CodecError::backend(
                BACKEND,
                self.operation,
                "media buffer smaller than its contents",
            ));
        }
        // SAFETY: a locked buffer is writable for `max` bytes at `data`
        // until it is unlocked; `len <= max`.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.data, len) })
    }
}

impl Drop for Locked<'_> {
    fn drop(&mut self) {
        // SAFETY: balances the successful `Lock` of `new`.
        if let Err(error) = unsafe { self.buffer.Unlock() } {
            tracing::warn!(%error, "IMFMediaBuffer::Unlock failed");
        }
    }
}

#[allow(
    dead_code,
    reason = "shared with ferricast-encoder, which uses it; the decoders do not"
)]
/// Sets a `VT_UI4` codec property.
///
/// # Errors
/// The `ICodecAPI::SetValue` failure.
pub fn set_codec_u32(api: &ICodecAPI, key: &GUID, value: u32) -> windows::core::Result<()> {
    let variant = VARIANT {
        Anonymous: VARIANT_0 {
            Anonymous: ManuallyDrop::new(VARIANT_0_0 {
                vt: VT_UI4,
                wReserved1: 0,
                wReserved2: 0,
                wReserved3: 0,
                Anonymous: VARIANT_0_0_0 { ulVal: value },
            }),
        },
    };
    // SAFETY: `variant` is a valid VT_UI4 VARIANT (nothing to clear).
    unsafe { api.SetValue(key, &raw const variant) }
}

#[allow(
    dead_code,
    reason = "shared with ferricast-encoder, which uses it; the decoders do not"
)]
/// The MF colour attributes of `colour`: `(primaries, transfer,
/// matrix, nominal range, chroma siting)`.
///
/// # Errors
/// [`CodecError::UnsupportedConfig`] for a value MF has no code for.
pub fn colour_attributes(colour: ColourDescription) -> CodecResult<[(GUID, Option<u32>); 5]> {
    let unsupported = |what: String| CodecError::UnsupportedConfig {
        backend: BACKEND,
        what,
    };
    let primaries = match colour.primaries {
        ColourPrimaries::Unspecified => None,
        ColourPrimaries::Bt709 => Some(2),
        ColourPrimaries::Bt470M => Some(3),
        ColourPrimaries::Bt470Bg => Some(4),
        ColourPrimaries::Smpte170M => Some(5),
        ColourPrimaries::Smpte240M => Some(6),
        ColourPrimaries::Ebu3213 => Some(7),
        ColourPrimaries::Bt2020 => Some(9),
        ColourPrimaries::Smpte428 => Some(10),
        ColourPrimaries::DciP3 => Some(11),
        other @ (ColourPrimaries::Film | ColourPrimaries::DisplayP3) => {
            return Err(unsupported(format!("signalling {other:?} primaries")));
        }
    };
    let transfer = match colour.transfer {
        TransferCharacteristics::Unspecified => None,
        TransferCharacteristics::Bt709 | TransferCharacteristics::Smpte170M => Some(5),
        TransferCharacteristics::Bt2020Ten | TransferCharacteristics::Bt2020Twelve => Some(13),
        TransferCharacteristics::Gamma22 => Some(4),
        TransferCharacteristics::Gamma28 => Some(8),
        TransferCharacteristics::Smpte240M => Some(6),
        TransferCharacteristics::Linear => Some(1),
        TransferCharacteristics::Log100 => Some(9),
        TransferCharacteristics::Log316 => Some(10),
        TransferCharacteristics::Srgb => Some(7),
        TransferCharacteristics::Pq => Some(15),
        TransferCharacteristics::Hlg => Some(16),
        other @ (TransferCharacteristics::Iec61966_2_4
        | TransferCharacteristics::Bt1361
        | TransferCharacteristics::Smpte428) => {
            return Err(unsupported(format!(
                "signalling the {other:?} transfer function"
            )));
        }
    };
    let matrix = match colour.matrix {
        MatrixCoefficients::Unspecified => None,
        MatrixCoefficients::Bt709 => Some(1),
        MatrixCoefficients::Bt470Bg | MatrixCoefficients::Smpte170M | MatrixCoefficients::Fcc => {
            Some(2)
        }
        MatrixCoefficients::Smpte240M => Some(3),
        MatrixCoefficients::Bt2020Ncl => Some(4),
        other @ (MatrixCoefficients::Identity
        | MatrixCoefficients::YCgCo
        | MatrixCoefficients::Bt2020Cl
        | MatrixCoefficients::Smpte2085
        | MatrixCoefficients::ChromaDerivedNcl
        | MatrixCoefficients::ChromaDerivedCl
        | MatrixCoefficients::ICtCp) => {
            return Err(unsupported(format!("signalling the {other:?} matrix")));
        }
    };
    let range = match colour.range {
        ColourRange::Full => 1,
        ColourRange::Limited => 2,
    };
    let siting = match colour.chroma_location {
        None => None,
        Some(ChromaLocation::Left) => Some(5),
        Some(ChromaLocation::Center) => Some(1),
        Some(ChromaLocation::TopLeft) => Some(7),
        Some(
            other @ (ChromaLocation::Top | ChromaLocation::BottomLeft | ChromaLocation::Bottom),
        ) => {
            return Err(unsupported(format!("signalling {other:?} chroma siting")));
        }
    };
    Ok([
        (MF_MT_VIDEO_PRIMARIES, primaries),
        (MF_MT_TRANSFER_FUNCTION, transfer),
        (MF_MT_YUV_MATRIX, matrix),
        (MF_MT_VIDEO_NOMINAL_RANGE, Some(range)),
        (MF_MT_VIDEO_CHROMA_SITING, siting),
    ])
}

/// The colour a media type describes, when it describes any.
#[allow(
    dead_code,
    reason = "shared with ferricast-decoder, which uses it; the encoders do not"
)]
#[must_use]
pub fn colour_of(media: &IMFMediaType) -> Option<ColourDescription> {
    // SAFETY: plain attribute reads on a live media type.
    let read = |key: &GUID| unsafe { media.GetUINT32(key) }.ok();
    let primaries = read(&MF_MT_VIDEO_PRIMARIES);
    let transfer = read(&MF_MT_TRANSFER_FUNCTION);
    let matrix = read(&MF_MT_YUV_MATRIX);
    let range = read(&MF_MT_VIDEO_NOMINAL_RANGE);
    if primaries.is_none() && transfer.is_none() && matrix.is_none() && range.is_none() {
        return None;
    }
    let primaries = match primaries {
        Some(2) => ColourPrimaries::Bt709,
        Some(3) => ColourPrimaries::Bt470M,
        Some(4) => ColourPrimaries::Bt470Bg,
        Some(5 | 8) => ColourPrimaries::Smpte170M,
        Some(6) => ColourPrimaries::Smpte240M,
        Some(7) => ColourPrimaries::Ebu3213,
        Some(9) => ColourPrimaries::Bt2020,
        Some(10) => ColourPrimaries::Smpte428,
        Some(11) => ColourPrimaries::DciP3,
        _ => ColourPrimaries::Unspecified,
    };
    let transfer = match transfer {
        Some(1) => TransferCharacteristics::Linear,
        Some(4) => TransferCharacteristics::Gamma22,
        Some(5 | 11) => TransferCharacteristics::Bt709,
        Some(6) => TransferCharacteristics::Smpte240M,
        Some(7) => TransferCharacteristics::Srgb,
        Some(8) => TransferCharacteristics::Gamma28,
        Some(9) => TransferCharacteristics::Log100,
        Some(10) => TransferCharacteristics::Log316,
        Some(13) => TransferCharacteristics::Bt2020Ten,
        Some(15) => TransferCharacteristics::Pq,
        Some(16) => TransferCharacteristics::Hlg,
        _ => TransferCharacteristics::Unspecified,
    };
    let matrix = match matrix {
        Some(1) => MatrixCoefficients::Bt709,
        Some(2) => MatrixCoefficients::Smpte170M,
        Some(3) => MatrixCoefficients::Smpte240M,
        Some(4 | 5) => MatrixCoefficients::Bt2020Ncl,
        _ => MatrixCoefficients::Unspecified,
    };
    let range = if range == Some(1) {
        ColourRange::Full
    } else {
        ColourRange::Limited
    };
    let colour = ColourDescription {
        primaries,
        transfer,
        matrix,
        range,
        chroma_location: None,
    };
    let siting = match read(&MF_MT_VIDEO_CHROMA_SITING) {
        Some(5) => Some(ChromaLocation::Left),
        Some(1) => Some(ChromaLocation::Center),
        Some(7) => Some(ChromaLocation::TopLeft),
        _ => None,
    };
    Some(siting.map_or(colour, |location| colour.with_chroma_location(location)))
}

/// Bytes of an `MT_CUSTOM_VIDEO_PRIMARIES` blob: eight `f32`s, red,
/// green, blue and white point, `x` then `y`.
const PRIMARIES_BLOB_LEN: usize = 32;

/// Writes `hdr` as Media Foundation's static HDR attributes: the
/// mastering luminances (`MF_MT_MAX_MASTERING_LUMINANCE` in cd/m²,
/// `MF_MT_MIN_MASTERING_LUMINANCE` in 0.0001 cd/m²), the mastering
/// primaries as `MF_MT_CUSTOM_VIDEO_PRIMARIES` (the colour primaries
/// themselves stay in `MF_MT_VIDEO_PRIMARIES`), and `MaxCLL` / `MaxFALL`.
///
/// # Errors
/// The attribute write failure.
pub fn set_hdr(media: &IMFMediaType, hdr: HdrMetadata, operation: Operation) -> CodecResult<()> {
    // SAFETY: plain attribute writes on a live media type.
    unsafe {
        if let Some(display) = hdr.mastering_display {
            media
                .SetUINT32(
                    &MF_MT_MAX_MASTERING_LUMINANCE,
                    display.max_luminance / 10_000,
                )
                .mf(operation, "setting MF_MT_MAX_MASTERING_LUMINANCE")?;
            media
                .SetUINT32(&MF_MT_MIN_MASTERING_LUMINANCE, display.min_luminance)
                .mf(operation, "setting MF_MT_MIN_MASTERING_LUMINANCE")?;
            let mut blob = [0u8; PRIMARIES_BLOB_LEN];
            let points = [
                display.red,
                display.green,
                display.blue,
                display.white_point,
            ];
            for (chunk, point) in blob.chunks_exact_mut(8).zip(points) {
                let (x, y) = point.xy();
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "chromaticities are in 0..=1.31, far inside f32"
                )]
                let (x, y) = (x as f32, y as f32);
                chunk[..4].copy_from_slice(&x.to_le_bytes());
                chunk[4..].copy_from_slice(&y.to_le_bytes());
            }
            media
                .SetBlob(&MF_MT_CUSTOM_VIDEO_PRIMARIES, &blob)
                .mf(operation, "setting MF_MT_CUSTOM_VIDEO_PRIMARIES")?;
        }
        if let Some(light) = hdr.content_light {
            media
                .SetUINT32(&MF_MT_MAX_LUMINANCE_LEVEL, u32::from(light.max_cll))
                .mf(operation, "setting MF_MT_MAX_LUMINANCE_LEVEL")?;
            media
                .SetUINT32(
                    &MF_MT_MAX_FRAME_AVERAGE_LUMINANCE_LEVEL,
                    u32::from(light.max_fall),
                )
                .mf(operation, "setting MF_MT_MAX_FRAME_AVERAGE_LUMINANCE_LEVEL")?;
        }
    }
    Ok(())
}

/// The static HDR metadata a media type carries (see [`set_hdr`]). The
/// mastering display needs both luminances and the primaries blob;
/// Media Foundation has no other place for its primaries.
#[must_use]
pub fn hdr_of(media: &IMFMediaType) -> Option<HdrMetadata> {
    // SAFETY: plain attribute reads on a live media type.
    let read = |key: &GUID| unsafe { media.GetUINT32(key) }.ok();
    let mut blob = [0u8; PRIMARIES_BLOB_LEN];
    // SAFETY: the blob is read into a buffer of its declared size.
    let primaries = unsafe { media.GetBlob(&MF_MT_CUSTOM_VIDEO_PRIMARIES, &mut blob, None) }
        .is_ok()
        .then_some(blob);
    let mastering_display = match (
        read(&MF_MT_MAX_MASTERING_LUMINANCE),
        read(&MF_MT_MIN_MASTERING_LUMINANCE),
        primaries,
    ) {
        (Some(max), Some(min), Some(blob)) => {
            let coordinate = |at: usize| {
                let value =
                    f32::from_le_bytes([blob[at], blob[at + 1], blob[at + 2], blob[at + 3]]);
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "clamped to the u16 range first"
                )]
                let units = (f64::from(value) * Chromaticity::SCALE)
                    .round()
                    .clamp(0.0, f64::from(u16::MAX)) as u16;
                units
            };
            let point = |at: usize| Chromaticity {
                x: coordinate(at),
                y: coordinate(at + 4),
            };
            Some(MasteringDisplay {
                red: point(0),
                green: point(8),
                blue: point(16),
                white_point: point(24),
                max_luminance: max.saturating_mul(10_000),
                min_luminance: min,
            })
        }
        _ => None,
    };
    let content_light = match (
        read(&MF_MT_MAX_LUMINANCE_LEVEL),
        read(&MF_MT_MAX_FRAME_AVERAGE_LUMINANCE_LEVEL),
    ) {
        (None, None) => None,
        (cll, fall) => Some(ContentLightLevel {
            max_cll: cll.map_or(0, |v| u16::try_from(v).unwrap_or(u16::MAX)),
            max_fall: fall.map_or(0, |v| u16::try_from(v).unwrap_or(u16::MAX)),
        }),
    };
    let hdr = HdrMetadata {
        mastering_display,
        content_light,
    };
    (!hdr.is_empty()).then_some(hdr)
}
