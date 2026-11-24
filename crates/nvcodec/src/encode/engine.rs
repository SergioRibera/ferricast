//! The encoder for one API version: session, stream, pooled bitstream
//! buffers and input slots, input registration caches and the queue of
//! pictures in flight. Version-independent logic; the structs it hands NVENC
//! are built by [`EncodeApi`].

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Weak};

use super::caps::{CodecCapabilities, codec_caps};
use super::config::{Config, RateControl, Tuning};
use super::raw::{ApiFeature, EncodeApi, Mapped, Picture, Registration, Resource, Submitted};
use super::{DeviceFrame, FrameOptions, HostFrame, Packet, PictureType, Session};
use crate::cuda::{ArrayView, Context, DeviceBuffer, Event, Point, Region, Stream};
use crate::error::{Error, Result};
use crate::sys::cuda::{CUarray, CUstream};
use crate::version::Api;

/// Spare packet buffers kept for reuse.
const SPARE_BUFFERS: usize = 16;

/// One position of the pipeline: a bitstream buffer and, once host input
/// has used it, a registered device buffer the host pictures are copied to.
struct Slot {
    bitstream: NonNull<c_void>,
    host: Option<HostSlot>,
}

struct HostSlot {
    buffer: DeviceBuffer,
    registered: NonNull<c_void>,
}

/// A submitted picture whose bitstream has not been locked yet.
struct Pending {
    seq: u64,
    slot: usize,
    registered: NonNull<c_void>,
    mapped: Mapped,
    /// A caller's device buffer, kept alive until NVENC has read it.
    _keep: Option<Arc<DeviceBuffer>>,
}

/// A registered caller device buffer.
struct DeviceEntry {
    /// Holding the `Weak` keeps the `Arc`'s allocation, hence the key (its
    /// address), from being reused by another buffer while the entry
    /// exists; `strong_count() == 0` says the buffer was dropped.
    buffer: Weak<DeviceBuffer>,
    registered: NonNull<c_void>,
    chroma_offset: Option<u32>,
}

/// A registered CUDA array.
struct ArrayEntry {
    array: CUarray,
    width: u32,
    height: u32,
    registered: NonNull<c_void>,
}

/// A registration dropped from a cache, unregistered once no pending
/// picture can be reading it.
struct Retired {
    registered: NonNull<c_void>,
    /// The last picture submitted when it was retired.
    after: u64,
}

pub(crate) struct Engine<A: EncodeApi> {
    // Field order is drop order: the session (encoder) goes first, then
    // what it was told about (the stream pointer, the stream), then the
    // memory it read. `Drop::drop` unmaps, unregisters and destroys the
    // bitstream buffers before any of that.
    session: Session<A>,
    _io_stream: Box<CUstream>,
    stream: Stream,
    copy_done: Event,
    slots: Vec<Slot>,
    config: Config,
    caps: CodecCapabilities,
    next_slot: usize,
    pending: VecDeque<Pending>,
    /// How many of the front pending pictures NVENC reported encoded.
    ready: usize,
    device_inputs: HashMap<usize, DeviceEntry>,
    array_inputs: HashMap<u64, ArrayEntry>,
    retired: Vec<Retired>,
    packets: VecDeque<Packet>,
    spare: Vec<Vec<u8>>,
    idr_requested: bool,
    submitted: u64,
    /// Pictures since the last expected IDR, `None` before the first.
    since_idr: Option<u64>,
    registrations: u64,
}

// SAFETY: the NVENC handles (encoder, bitstream buffers, registrations,
// mappings) have no thread affinity and are only used through `&mut self`;
// the CUDA context is pushed around every call. The session, stream, event
// and buffers are `Send` themselves. Not `Sync`: NVENC calls on one session
// must not overlap.
unsafe impl<A: EncodeApi> Send for Engine<A> {}

impl<A: EncodeApi> Engine<A> {
    pub(crate) fn open(api: Api<A>, context: &Arc<Context>, config: &Config) -> Result<Self> {
        config.validate()?;
        Self::check_version(config)?;
        let session = super::open_session(api, context)?;
        let _current = context.guard()?;
        let caps = Self::check_caps(&session, config)?;

        A::initialize(&session, config)?;
        let stream = Stream::new(context)?;
        let copy_done = Event::new(context)?;
        let io_stream = Box::new(stream.raw());
        // SAFETY: the box and the stream are fields declared after the
        // session, so they are dropped after it.
        unsafe { A::set_io_stream(&session, NonNull::from(io_stream.as_ref()))? };

        let mut engine = Self {
            session,
            _io_stream: io_stream,
            stream,
            copy_done,
            slots: Vec::new(),
            config: config.clone(),
            caps,
            next_slot: 0,
            pending: VecDeque::new(),
            ready: 0,
            device_inputs: HashMap::new(),
            array_inputs: HashMap::new(),
            retired: Vec::new(),
            packets: VecDeque::new(),
            spare: Vec::new(),
            idr_requested: false,
            submitted: 0,
            since_idr: None,
            registrations: 0,
        };
        for _ in 0..config.pipeline_depth() {
            let bitstream = A::create_bitstream(&engine.session)?;
            engine.slots.push(Slot {
                bitstream,
                host: None,
            });
        }
        tracing::debug!(
            api = %A::VERSION,
            codec = %config.codec,
            width = config.width,
            height = config.height,
            format = %config.input_format,
            depth = engine.slots.len(),
            "NVENC encoder initialised"
        );
        Ok(engine)
    }

    /// Refuses what the API version lacks, naming the version that adds it.
    fn check_version(config: &Config) -> Result<()> {
        let refuse = |feature: ApiFeature, what| {
            if A::has(feature) {
                Ok(())
            } else {
                Err(feature.unsupported(what, A::VERSION))
            }
        };
        if config.codec == super::Codec::Av1 {
            refuse(ApiFeature::Av1, "AV1 encoding")?;
            if config.hdr10.is_some() {
                refuse(ApiFeature::NativeHdrMetadata, "AV1 HDR10 metadata")?;
            }
        }
        if config.tuning == Tuning::UltraHighQuality {
            refuse(
                ApiFeature::UltraHighQualityTuning,
                "ultra-high-quality tuning",
            )?;
        }
        Ok(())
    }

    /// Refuses what the GPU lacks.
    fn check_caps(session: &Session<A>, config: &Config) -> Result<CodecCapabilities> {
        let codec = config.codec;
        if !A::codecs(session)?.contains(&codec) {
            return Err(Error::Unsupported {
                what: "this codec",
                reason: format!("this GPU's NVENC has no {codec} encoder"),
            });
        }
        let caps = codec_caps(session, codec)?;
        let unsupported = |what, reason: String| Err(Error::Unsupported { what, reason });
        if config.width > caps.max_width
            || config.height > caps.max_height
            || config.width < caps.min_width
            || config.height < caps.min_height
        {
            return unsupported(
                "this picture size",
                format!(
                    "{}x{} is outside {}x{}..{}x{} for {codec}",
                    config.width,
                    config.height,
                    caps.min_width,
                    caps.min_height,
                    caps.max_width,
                    caps.max_height
                ),
            );
        }
        if !caps.input_formats.contains(&config.input_format) {
            return unsupported(
                "this input format",
                format!(
                    "{codec} takes {:?}, not {}",
                    caps.input_formats, config.input_format
                ),
            );
        }
        if config.input_format.bit_depth() > 8 && !caps.ten_bit {
            return unsupported("10-bit encoding", format!("this GPU has no 10-bit {codec}"));
        }
        if u32::from(config.b_frames) > caps.max_b_frames {
            return unsupported(
                "B frames",
                format!(
                    "{codec} allows {} B frames, {} asked",
                    caps.max_b_frames, config.b_frames
                ),
            );
        }
        if config.lookahead.is_some() && !caps.lookahead {
            return unsupported("lookahead", format!("no {codec} lookahead on this GPU"));
        }
        if config.rate_control.vbv().is_some() && !caps.custom_vbv {
            return unsupported("a custom VBV", format!("no custom {codec} VBV on this GPU"));
        }
        if config.tuning == Tuning::Lossless && !caps.lossless {
            return unsupported(
                "lossless encoding",
                format!("no lossless {codec} on this GPU"),
            );
        }
        Ok(caps)
    }

    pub(crate) fn config(&self) -> &Config {
        &self.config
    }

    pub(crate) fn capabilities(&self) -> &CodecCapabilities {
        &self.caps
    }

    pub(crate) fn registrations(&self) -> u64 {
        self.registrations
    }

    pub(crate) fn request_idr(&mut self) {
        self.idr_requested = true;
    }

    pub(crate) fn next_packet(&mut self) -> Option<Packet> {
        self.packets.pop_front()
    }

    pub(crate) fn recycle(&mut self, packet: Packet) {
        if self.spare.len() < SPARE_BUFFERS {
            self.spare.push(packet.data);
        }
    }

    fn context(&self) -> &Arc<Context> {
        self.session.context()
    }

    fn register(&mut self, registration: &Registration) -> Result<NonNull<c_void>> {
        // SAFETY: every caller passes a live allocation of this context that
        // it keeps until it unregisters (see the call sites).
        let registered = unsafe { A::register(&self.session, registration)? };
        self.registrations += 1;
        Ok(registered)
    }

    /// The slot the next picture goes to, checked free.
    fn free_slot(&self) -> Result<usize> {
        let slot = self.next_slot;
        if self.pending.iter().any(|pending| pending.slot == slot) {
            return Err(Error::InvalidEncodeInput {
                reason: format!(
                    "{} pictures are pending, more than the {} the configuration allows",
                    self.pending.len(),
                    self.slots.len()
                ),
            });
        }
        Ok(slot)
    }

    pub(crate) fn encode_host(
        &mut self,
        frame: HostFrame<'_>,
        options: FrameOptions,
    ) -> Result<()> {
        let config = &self.config;
        let format = config.input_format;
        let (luma, chroma) = match (frame, format.is_semi_planar()) {
            (HostFrame::Packed(image), false) => (image, None),
            (HostFrame::SemiPlanar { luma, chroma }, true) => (luma, Some(chroma)),
            (HostFrame::Packed(_), true) | (HostFrame::SemiPlanar { .. }, false) => {
                return Err(Error::InvalidEncodeInput {
                    reason: format!("a {format} encoder got a picture of another layout"),
                });
            }
        };
        let width_bytes = format.row_bytes(config.width);
        let height = config.height as usize;
        let chroma_rows = format.chroma_rows(config.height);

        let context = Arc::clone(self.context());
        let _current = context.guard()?;
        let slot = self.free_slot()?;
        if self.slots[slot].host.is_none() {
            let buffer = DeviceBuffer::pitched(&context, width_bytes, height + chroma_rows)?;
            let registration = Registration {
                resource: Resource::DevicePtr(buffer.device_ptr()),
                format,
                width: config.width,
                height: config.height,
                pitch: to_u32(buffer.pitch(), "a device pitch")?,
                chroma_offset: None,
            };
            let registered = self.register(&registration)?;
            self.slots[slot].host = Some(HostSlot { buffer, registered });
        }
        let Some(host) = self.slots[slot].host.as_mut() else {
            unreachable!("the slot's host buffer was created above");
        };
        let pitch = to_u32(host.buffer.pitch(), "a device pitch")?;

        // SAFETY: the host images are borrowed for this call and the copies
        // are waited for (`copy_done`) before it returns, whatever happens
        // in between.
        let copied = unsafe {
            host.buffer
                .upload_async(Region::new(width_bytes, height), luma, &self.stream)
                .and_then(|()| match chroma {
                    Some(chroma) => host.buffer.upload_async(
                        Region::new(width_bytes, chroma_rows).at(Point {
                            x_bytes: 0,
                            row: height,
                        }),
                        chroma,
                        &self.stream,
                    ),
                    None => Ok(()),
                })
        };
        let waited = self
            .copy_done
            .record(&self.stream)
            .and_then(|()| self.copy_done.synchronize());
        copied?;
        waited?;

        let registered = host.registered;
        // SAFETY: the slot's registration is not mapped: its previous
        // picture was locked and unmapped before the slot came free.
        let mapped = unsafe { A::map(&self.session, registered)? };
        self.submit(slot, registered, mapped, pitch, None, options)
    }

    pub(crate) fn encode_device(
        &mut self,
        frame: DeviceFrame<'_>,
        options: FrameOptions,
    ) -> Result<()> {
        let buffer = frame.buffer;
        let config = &self.config;
        let format = config.input_format;
        if buffer.context().raw() != self.context().raw() {
            return Err(Error::InvalidEncodeInput {
                reason: "the device buffer belongs to another CUDA context".into(),
            });
        }
        let height = config.height as usize;
        let chroma_row = frame.chroma_row.unwrap_or(height);
        let needed_rows = if format.is_semi_planar() {
            if chroma_row < height {
                return Err(Error::InvalidEncodeInput {
                    reason: format!("chroma row {chroma_row} overlaps the {height} luma rows"),
                });
            }
            chroma_row + format.chroma_rows(config.height)
        } else {
            height
        };
        let width_bytes = format.row_bytes(config.width);
        if buffer.width_bytes() < width_bytes || buffer.rows() < needed_rows {
            return Err(Error::InvalidEncodeInput {
                reason: format!(
                    "a {}x{} {format} picture needs {width_bytes} bytes x {needed_rows} rows, the buffer has {} x {}",
                    config.width,
                    config.height,
                    buffer.width_bytes(),
                    buffer.rows()
                ),
            });
        }
        let pitch = to_u32(buffer.pitch(), "a device pitch")?;
        if pitch % 4 != 0 {
            return Err(Error::InvalidEncodeInput {
                reason: format!("NVENC needs a pitch multiple of 4, the buffer's is {pitch}"),
            });
        }
        let chroma_offset = (format.is_semi_planar() && chroma_row != height)
            .then(|| to_u32(chroma_row * buffer.pitch(), "a chroma offset"))
            .transpose()?;

        let context = Arc::clone(self.context());
        let _current = context.guard()?;
        if let Some(ready) = frame.ready {
            self.stream.wait(ready)?;
        }
        self.sweep_device_inputs();
        let slot = self.free_slot()?;
        let key = Arc::as_ptr(buffer) as usize;
        let cached = self
            .device_inputs
            .get(&key)
            .filter(|entry| entry.chroma_offset == chroma_offset)
            .map(|entry| entry.registered);
        let registered = if let Some(registered) = cached {
            registered
        } else {
            if let Some(stale) = self.device_inputs.remove(&key) {
                self.retire(stale.registered);
            }
            let registration = Registration {
                resource: Resource::DevicePtr(buffer.device_ptr()),
                format,
                width: self.config.width,
                height: self.config.height,
                pitch,
                chroma_offset,
            };
            // The buffer stays allocated while the entry can be used:
            // the entry goes as soon as the last `Arc` does
            // (`sweep_device_inputs`), and pending pictures hold one.
            let registered = self.register(&registration)?;
            self.device_inputs.insert(
                key,
                DeviceEntry {
                    buffer: Arc::downgrade(buffer),
                    registered,
                    chroma_offset,
                },
            );
            registered
        };
        self.check_unmapped(registered)?;
        // SAFETY: registered on this session and not mapped (checked above:
        // only pending pictures hold mappings).
        let mapped = unsafe { A::map(&self.session, registered)? };
        self.submit(
            slot,
            registered,
            mapped,
            pitch,
            Some(Arc::clone(buffer)),
            options,
        )
    }

    /// # Safety
    ///
    /// See [`super::Encoder::encode_array`].
    pub(crate) unsafe fn encode_array(
        &mut self,
        key: u64,
        array: ArrayView<'_>,
        options: FrameOptions,
    ) -> Result<()> {
        let config = &self.config;
        let format = config.input_format;
        if format.is_semi_planar() {
            return Err(Error::InvalidEncodeInput {
                reason: format!("CUDA array input takes packed RGB, not {format}"),
            });
        }
        if array.context().raw() != self.context().raw() {
            return Err(Error::InvalidEncodeInput {
                reason: "the CUDA array belongs to another CUDA context".into(),
            });
        }
        if array.width() < config.width || array.height() < config.height {
            return Err(Error::InvalidEncodeInput {
                reason: format!(
                    "a {}x{} array cannot hold a {}x{} picture",
                    array.width(),
                    array.height(),
                    config.width,
                    config.height
                ),
            });
        }
        let pitch = to_u32(format.row_bytes(array.width()), "an array pitch")?;

        let context = Arc::clone(self.context());
        let _current = context.guard()?;
        let slot = self.free_slot()?;
        let cached = self
            .array_inputs
            .get(&key)
            .filter(|entry| {
                entry.array == array.raw()
                    && entry.width == array.width()
                    && entry.height == array.height()
            })
            .map(|entry| entry.registered);
        let registered = if let Some(registered) = cached {
            registered
        } else {
            if let Some(stale) = self.array_inputs.remove(&key) {
                self.retire(stale.registered);
            }
            let registration = Registration {
                resource: Resource::Array(array.raw()),
                format,
                width: self.config.width,
                height: self.config.height,
                pitch,
                chroma_offset: None,
            };
            // The caller keeps the array allocated while it is cached
            // under `key` (this function's contract).
            let registered = self.register(&registration)?;
            self.array_inputs.insert(
                key,
                ArrayEntry {
                    array: array.raw(),
                    width: array.width(),
                    height: array.height(),
                    registered,
                },
            );
            registered
        };
        self.check_unmapped(registered)?;
        // SAFETY: as in `encode_device`.
        let mapped = unsafe { A::map(&self.session, registered)? };
        self.submit(slot, registered, mapped, pitch, None, options)
    }

    /// Refuses an input that a pending picture still has mapped (the same
    /// buffer submitted again before its packet came out).
    fn check_unmapped(&self, registered: NonNull<c_void>) -> Result<()> {
        if self
            .pending
            .iter()
            .any(|pending| pending.registered == registered)
        {
            return Err(Error::InvalidEncodeInput {
                reason: "this input is still being encoded: wait for its packet before submitting it again".into(),
            });
        }
        Ok(())
    }

    /// Submits a mapped picture and hands out what is ready.
    fn submit(
        &mut self,
        slot: usize,
        registered: NonNull<c_void>,
        mapped: Mapped,
        pitch: u32,
        keep: Option<Arc<DeviceBuffer>>,
        options: FrameOptions,
    ) -> Result<()> {
        let force_idr = options.force_idr || self.idr_requested;
        let period_idr = match (self.since_idr, self.config.gop.effective_idr_period()) {
            (None, _) => true,
            (Some(since), Some(period)) => since + 1 >= u64::from(period.get()),
            (Some(_), None) => false,
        };
        let expect_idr = force_idr || period_idr;
        let picture = Picture {
            codec: self.config.codec,
            input: mapped,
            output: self.slots[slot].bitstream,
            width: self.config.width,
            height: self.config.height,
            pitch,
            timestamp: options.timestamp,
            force_idr,
            parameter_sets: force_idr && self.config.repeat_parameter_sets,
            hdr10: self.config.hdr10.as_ref(),
            expect_idr,
        };
        // SAFETY: `mapped` was just mapped on this session and the slot's
        // bitstream buffer is free (`free_slot`).
        let submitted = unsafe { A::encode(&self.session, &picture) };
        let submitted = match submitted {
            Ok(submitted) => submitted,
            Err(error) => {
                // SAFETY: the picture was not accepted, nothing reads it.
                if let Err(unmap) = unsafe { A::unmap(&self.session, mapped) } {
                    tracing::warn!(%unmap, "unmapping a rejected NVENC input");
                }
                return Err(error);
            }
        };
        self.idr_requested = false;
        self.since_idr = Some(if expect_idr {
            0
        } else {
            self.since_idr.map_or(0, |since| since + 1)
        });
        self.pending.push_back(Pending {
            seq: self.submitted,
            slot,
            registered,
            mapped,
            _keep: keep,
        });
        self.submitted += 1;
        self.next_slot = (slot + 1) % self.slots.len();
        if submitted == Submitted::OutputReady {
            // B frames make NVENC answer "need more input" until it can
            // output; lookahead does not: it answers success while the last
            // `depth` pictures are still being analysed, and locking those
            // fails. Their packets come with later pictures or the flush.
            let lookahead = usize::from(self.config.lookahead.unwrap_or(0));
            self.ready = self.ready.max(self.pending.len().saturating_sub(lookahead));
        }
        self.drain(usize::from(self.config.frames_in_flight))
    }

    /// Locks ready pictures, oldest first, until at most `keep` are pending.
    fn drain(&mut self, keep: usize) -> Result<()> {
        while self.ready > 0 && self.pending.len() > keep {
            let Some(pending) = self.pending.pop_front() else {
                break;
            };
            self.ready -= 1;
            let mut data = self.spare.pop().unwrap_or_default();
            let output = self.slots[pending.slot].bitstream;
            // SAFETY: NVENC reported the picture encoded (`ready`), its
            // bitstream buffer is not locked.
            let locked = unsafe { A::lock(&self.session, output, &mut data) };
            // SAFETY: the bitstream is locked (or the lock failed), NVENC no
            // longer reads the input.
            let unmapped = unsafe { A::unmap(&self.session, pending.mapped) };
            let locked = locked?;
            unmapped?;
            let picture_type = PictureType::from_raw(locked.picture_type);
            self.packets.push_back(Packet {
                data,
                timestamp: locked.timestamp,
                picture_type,
                index: pending.seq,
            });
        }
        self.release_retired();
        Ok(())
    }

    /// Queues `registered` for unregistration once no pending picture uses
    /// it.
    fn retire(&mut self, registered: NonNull<c_void>) {
        self.retired.push(Retired {
            registered,
            after: self.submitted,
        });
        self.release_retired();
    }

    fn release_retired(&mut self) {
        let oldest = self.pending.front().map_or(u64::MAX, |pending| pending.seq);
        let session = &self.session;
        self.retired.retain(|retired| {
            if retired.after > oldest {
                return true;
            }
            // SAFETY: retired registrations left their cache, and every
            // picture submitted before they did has been unmapped.
            if let Err(error) = unsafe { A::unregister(session, retired.registered) } {
                tracing::warn!(%error, "unregistering an NVENC input");
            }
            false
        });
    }

    /// Drops the registrations of caller buffers that no longer exist.
    fn sweep_device_inputs(&mut self) {
        let dead: Vec<usize> = self
            .device_inputs
            .iter()
            .filter(|(_, entry)| entry.buffer.strong_count() == 0)
            .map(|(key, _)| *key)
            .collect();
        for key in dead {
            if let Some(entry) = self.device_inputs.remove(&key) {
                self.retire(entry.registered);
            }
        }
    }

    pub(crate) fn forget_device_buffer(&mut self, buffer: &Arc<DeviceBuffer>) {
        if let Some(entry) = self.device_inputs.remove(&(Arc::as_ptr(buffer) as usize)) {
            self.retire(entry.registered);
        }
    }

    pub(crate) fn invalidate_array(&mut self, key: u64) {
        if let Some(entry) = self.array_inputs.remove(&key) {
            self.retire(entry.registered);
        }
    }

    pub(crate) fn clear_input_caches(&mut self) {
        let device: Vec<_> = self
            .device_inputs
            .drain()
            .map(|(_, entry)| entry.registered)
            .collect();
        let arrays: Vec<_> = self
            .array_inputs
            .drain()
            .map(|(_, entry)| entry.registered)
            .collect();
        for registered in device.into_iter().chain(arrays) {
            self.retire(registered);
        }
    }

    pub(crate) fn flush(&mut self) -> Result<()> {
        let context = Arc::clone(self.context());
        let _current = context.guard()?;
        if self.pending.is_empty() {
            return Ok(());
        }
        A::end_of_stream(&self.session)?;
        self.ready = self.pending.len();
        self.drain(0)
    }

    pub(crate) fn set_rate_control(&mut self, rate_control: RateControl) -> Result<()> {
        let current = self.config.rate_control;
        if !current.same_mode(rate_control) && !self.caps.dynamic_rate_control_mode {
            return Err(Error::Unsupported {
                what: "changing the rate-control mode",
                reason: "this GPU needs a new session for it".into(),
            });
        }
        if current.bitrates() != rate_control.bitrates() && !self.caps.dynamic_bitrate {
            return Err(Error::Unsupported {
                what: "changing the bitrate",
                reason: "this GPU needs a new session for it".into(),
            });
        }
        if rate_control.vbv().is_some() && !self.caps.custom_vbv {
            return Err(Error::Unsupported {
                what: "a custom VBV",
                reason: format!("no custom {} VBV on this GPU", self.config.codec),
            });
        }
        let config = Config {
            rate_control,
            ..self.config.clone()
        };
        config.validate()?;
        let context = Arc::clone(self.context());
        let _current = context.guard()?;
        A::reconfigure(&self.session, &config, false)?;
        self.config = config;
        Ok(())
    }

    pub(crate) fn sequence_params(&mut self) -> Result<Vec<u8>> {
        let _current = self.context().guard()?;
        let mut out = Vec::new();
        A::sequence_params(&self.session, &mut out)?;
        Ok(out)
    }
}

impl<A: EncodeApi> Drop for Engine<A> {
    fn drop(&mut self) {
        let context = Arc::clone(self.context());
        let current = context.guard();
        if let Err(error) = &current {
            tracing::warn!(%error, "making the CUDA context current to close an NVENC encoder");
        }
        let session = &self.session;
        if !self.pending.is_empty() {
            // NVENC wants its pending pictures drained before teardown.
            if let Err(error) = A::end_of_stream(session) {
                tracing::warn!(%error, "flushing an NVENC encoder");
            }
            let mut sink = Vec::new();
            for pending in self.pending.drain(..) {
                let output = self.slots[pending.slot].bitstream;
                // SAFETY: after EOS every pending picture is lockable.
                if let Err(error) = unsafe { A::lock(session, output, &mut sink) } {
                    tracing::warn!(%error, "locking a pending NVENC bitstream");
                }
                // SAFETY: locked (or failed), nothing reads the input.
                if let Err(error) = unsafe { A::unmap(session, pending.mapped) } {
                    tracing::warn!(%error, "unmapping an NVENC input");
                }
            }
        }
        let registered = self
            .slots
            .iter()
            .filter_map(|slot| slot.host.as_ref().map(|host| host.registered))
            .chain(self.device_inputs.values().map(|entry| entry.registered))
            .chain(self.array_inputs.values().map(|entry| entry.registered))
            .chain(self.retired.iter().map(|retired| retired.registered));
        for registered in registered {
            // SAFETY: nothing is mapped any more and the handles are not
            // used again.
            if let Err(error) = unsafe { A::unregister(session, registered) } {
                tracing::warn!(%error, "unregistering an NVENC input");
            }
        }
        for slot in &self.slots {
            // SAFETY: no bitstream is pending or locked; not used again.
            if let Err(error) = unsafe { A::destroy_bitstream(session, slot.bitstream) } {
                tracing::warn!(%error, "destroying an NVENC bitstream buffer");
            }
        }
        drop(current);
    }
}

fn to_u32(value: usize, what: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::SizeOverflow { what })
}
