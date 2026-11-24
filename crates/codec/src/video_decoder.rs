//! A video decoder that picks its backend, and moves the stream to the
//! next backend when the one it picked fails on it.
//!
//! Hardware decoders can accept a configuration and still reject the
//! bitstream: NVDEC only opens its decode session when its parser
//! reaches the first sequence header, a VA-API driver may fail on the
//! first slices, a `MediaCodec` or Media Foundation decoder may refuse a
//! profile it advertised. So until a backend has produced its first
//! picture the decoder keeps the packets it fed it ("probation"), and a
//! failure replays them into the next backend from the start of the
//! stream. A backend that reports [`CodecError::UnsupportedStream`]
//! (a stream feature it does not implement, met mid-stream) hands over
//! at any time: its finished pictures are kept, and the next backend
//! starts at the failing packet with the stream's parameter sets in
//! front of it.

use std::collections::VecDeque;

use bytes::Bytes;
use ferricast_core::{
    Acceleration, BackendId, CapturedFrame, CodecError, CodecResult, CodecTarget, DecoderConfig,
    EncodedFrame, Operation, RuntimeFallback,
};
use tracing::warn;

use crate::bitstream::{NalSyntax, ParameterSets};
use crate::registry::{Plan, Registry};
use crate::session::VideoDecoderSession;

/// The registry of one video decode job.
pub type VideoDecoderRegistry = Registry<DecoderConfig, dyn VideoDecoderSession>;

/// Packets kept, until the first picture, for replay into another backend.
#[derive(Debug, Default)]
struct Probation {
    packets: Vec<EncodedFrame>,
    limit: usize,
    settled: bool,
}

impl Probation {
    fn new(fallback: RuntimeFallback) -> Self {
        match fallback {
            RuntimeFallback::NextBackend { packets } => Self {
                packets: Vec::new(),
                limit: usize::from(packets),
                settled: false,
            },
            RuntimeFallback::Disabled => Self {
                settled: true,
                ..Self::default()
            },
        }
    }

    /// Keeps `frame` for a possible replay while still on probation; a
    /// backend that has not produced a picture after `limit` packets is
    /// decoding, just slowly to start, and probation ends.
    fn record(&mut self, frame: &EncodedFrame) {
        if self.settled {
            return;
        }
        if self.packets.len() >= self.limit {
            self.settle();
            return;
        }
        self.packets.push(frame.clone());
    }

    /// The backend produced a picture: from here on its errors are
    /// ordinary decode errors, not a rejected stream.
    fn settle(&mut self) {
        self.settled = true;
        self.packets = Vec::new();
    }

    /// Packets to replay into the next backend, or `None` once the
    /// backend has proven itself. Ends probation either way.
    fn take_replay(&mut self) -> Option<Vec<EncodedFrame>> {
        if self.settled {
            return None;
        }
        self.settled = true;
        Some(std::mem::take(&mut self.packets))
    }
}

struct Active {
    session: Box<dyn VideoDecoderSession>,
    backend: BackendId,
    acceleration: Acceleration,
    position: usize,
    plan: Plan,
    config: DecoderConfig,
    probation: Probation,
}

/// A [`ferricast_core::VideoDecoder`] body that selects its backend from
/// a [`VideoDecoderRegistry`] according to [`DecoderConfig::backend`].
///
/// The backend is chosen on the first `configure`, and chosen again
/// when a later `configure` asks for a different selection.
pub struct SelectingVideoDecoder {
    registry: VideoDecoderRegistry,
    parameter_sets: Option<ParameterSets>,
    active: Option<Active>,
    /// Pictures a backend finished before the stream moved to the next
    /// one; served ahead of anything the next backend produces.
    handed_over: VecDeque<CapturedFrame>,
}

impl std::fmt::Debug for SelectingVideoDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectingVideoDecoder")
            .field("target", &self.registry.target())
            .field("backend", &self.backend())
            .field("handed_over", &self.handed_over.len())
            .finish_non_exhaustive()
    }
}

impl SelectingVideoDecoder {
    /// A decoder over `registry`. `syntax` names the parameter sets to
    /// replay when a backend hands over mid-stream (`None` for codecs
    /// whose packets are self-contained).
    #[must_use]
    pub fn new(registry: VideoDecoderRegistry, syntax: Option<NalSyntax>) -> Self {
        Self {
            registry,
            parameter_sets: syntax.map(ParameterSets::new),
            active: None,
            handed_over: VecDeque::new(),
        }
    }

    /// The job this decoder does.
    #[must_use]
    pub fn target(&self) -> CodecTarget {
        self.registry.target()
    }

    /// The registry, for capability reports.
    #[must_use]
    pub fn registry(&self) -> &VideoDecoderRegistry {
        &self.registry
    }

    /// The backend decoding now, once configured.
    #[must_use]
    pub fn backend(&self) -> Option<(BackendId, Acceleration)> {
        self.active
            .as_ref()
            .map(|active| (active.backend, active.acceleration))
    }

    /// Selects (or re-selects) the backend, or reconfigures the open one.
    ///
    /// # Errors
    /// The selection's errors (see [`Registry::plan`] and
    /// [`Registry::open`]) or the open backend's `configure` error.
    pub fn configure(&mut self, config: &DecoderConfig) -> CodecResult<()> {
        if let Some(active) = &mut self.active
            && active.config.backend == config.backend
        {
            active.session.configure(config)?;
            active.config = config.clone();
            return Ok(());
        }
        let plan = self.registry.plan(&config.backend)?;
        let opened = self.registry.open(&plan, 0, config)?;
        self.active = Some(Active {
            probation: Probation::new(plan.runtime_fallback()),
            session: opened.session,
            backend: opened.backend,
            acceleration: opened.acceleration,
            position: opened.position,
            plan,
            config: config.clone(),
        });
        Ok(())
    }

    /// Decodes one packet; see [`ferricast_core::VideoDecoder::decode`].
    ///
    /// # Errors
    /// [`CodecError::NotConfigured`] before `configure`; otherwise the
    /// backend's error once no fallback applies.
    pub fn decode(&mut self, frame: EncodedFrame) -> CodecResult<Option<CapturedFrame>> {
        let target = self.target();
        let active = self.active.as_mut().ok_or(CodecError::NotConfigured {
            target,
            operation: Operation::Decode,
        })?;
        if let Some(sets) = &mut self.parameter_sets {
            sets.remember(&frame.data);
        }
        active.probation.record(&frame);
        let failing = frame.clone();
        match active.session.decode(frame) {
            Ok(picture) => {
                if picture.is_some() {
                    active.probation.settle();
                }
                Ok(self.serve(picture))
            }
            Err(error) => {
                self.recover(error, Some(failing))?;
                Ok(self.handed_over.pop_front())
            }
        }
    }

    /// Drains the open backend; see [`ferricast_core::VideoDecoder::flush`].
    ///
    /// # Errors
    /// The backend's error once no fallback applies.
    pub fn flush(&mut self) -> CodecResult<Vec<CapturedFrame>> {
        let Some(active) = &mut self.active else {
            return Ok(self.handed_over.drain(..).collect());
        };
        match active.session.flush() {
            Ok(frames) => {
                self.handed_over.extend(frames);
                Ok(self.handed_over.drain(..).collect())
            }
            Err(error) => {
                self.recover(error, None)?;
                let active = self.active.as_mut().ok_or(CodecError::NotConfigured {
                    target: self.registry.target(),
                    operation: Operation::Flush,
                })?;
                let rest = active.session.flush()?;
                self.handed_over.extend(rest);
                Ok(self.handed_over.drain(..).collect())
            }
        }
    }

    /// Keeps output order across a hand-over: while earlier pictures
    /// are queued, a new one goes behind them.
    fn serve(&mut self, picture: Option<CapturedFrame>) -> Option<CapturedFrame> {
        if self.handed_over.is_empty() {
            return picture;
        }
        self.handed_over.extend(picture);
        self.handed_over.pop_front()
    }

    /// Moves the stream to the next backend after `error`, when the
    /// selection allows it; returns `error` when it does not.
    fn recover(&mut self, error: CodecError, failing: Option<EncodedFrame>) -> CodecResult<()> {
        let Some(active) = &mut self.active else {
            return Err(error);
        };
        if matches!(active.plan.runtime_fallback(), RuntimeFallback::Disabled)
            || !active.plan.has_after(active.position)
        {
            return Err(error);
        }
        let mid_stream = matches!(error, CodecError::UnsupportedStream { .. });
        let replay = if mid_stream {
            // Finished pictures stay valid; the next backend picks up at
            // the failing packet with the parameter sets in front.
            match active.session.flush() {
                Ok(frames) => self.handed_over.extend(frames),
                Err(flush_error) => {
                    warn!(backend = %active.backend, %flush_error, "flush before the hand-over failed");
                }
            }
            failing
                .map(|packet| {
                    let data = self
                        .parameter_sets
                        .as_ref()
                        .map_or_else(|| packet.data.clone(), |sets| sets.prepend(&packet.data));
                    vec![EncodedFrame { data, ..packet }]
                })
                .unwrap_or_default()
        } else {
            match active.probation.take_replay() {
                Some(packets) => packets,
                None => return Err(error),
            }
        };
        warn!(
            target_job = %self.registry.target(),
            backend = %active.backend,
            %error,
            "codec backend failed on the stream; moving to the next backend"
        );
        let opened = self
            .registry
            .open(&active.plan, active.position + 1, &active.config)
            .map_err(|_| error)?;
        active.session = opened.session;
        active.backend = opened.backend;
        active.acceleration = opened.acceleration;
        active.position = opened.position;
        // The next backend is on probation too, over the replayed packets,
        // so it can hand over again if it fails the same way.
        active.probation = Probation::new(active.plan.runtime_fallback());
        for packet in replay {
            active.probation.record(&packet);
            match active.session.decode(packet) {
                Ok(picture) => {
                    if picture.is_some() {
                        active.probation.settle();
                    }
                    self.handed_over.extend(picture);
                }
                // The next backend would have skipped it decoding from
                // the start, e.g. a P slice before the first IDR.
                Err(skip) => {
                    warn!(backend = %active.backend, error = %skip, "replayed packet skipped")
                }
            }
        }
        Ok(())
    }

    /// The parameter sets a mid-stream hand-over replays, as Annex B.
    #[must_use]
    pub fn remembered_parameter_sets(&self) -> Option<Bytes> {
        self.parameter_sets
            .as_ref()
            .map(|sets| Bytes::from(sets.annex_b()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use ferricast_core::{AutoSelect, Codec, PixelFormat, RawFrame};

    use super::*;
    use crate::registry::Candidate;

    fn picture(tag: u8) -> CapturedFrame {
        CapturedFrame::Cpu(RawFrame {
            width: 1,
            height: 1,
            stride: 1,
            format: PixelFormat::Nv12,
            data: Bytes::from(vec![tag]),
            timestamp_us: u64::from(tag),
            colour: None,
        })
    }

    fn packet(tag: u8) -> EncodedFrame {
        EncodedFrame {
            codec: Codec::H264,
            data: Bytes::from(vec![0, 0, 1, 0x41, tag]),
            timestamp_us: u64::from(tag),
            is_keyframe: false,
            duration_us: None,
            pts_dts: (0, 0),
        }
    }

    fn tag(frame: &CapturedFrame) -> u8 {
        let CapturedFrame::Cpu(raw) = frame else {
            panic!("CPU frames only");
        };
        raw.data[0]
    }

    /// Fails every packet without producing a picture.
    struct Rejecting(CodecError);

    impl VideoDecoderSession for Rejecting {
        fn configure(&mut self, _: &DecoderConfig) -> CodecResult<()> {
            Ok(())
        }
        fn decode(&mut self, _: EncodedFrame) -> CodecResult<Option<CapturedFrame>> {
            Err(self.0.clone())
        }
        fn flush(&mut self) -> CodecResult<Vec<CapturedFrame>> {
            Ok(Vec::new())
        }
    }

    /// Echoes each packet's last byte as a picture; refuses tag 9 as a
    /// stream feature it lacks once it has decoded `fails_after` packets.
    struct Echo {
        decoded: usize,
        fails_after: Option<usize>,
    }

    impl VideoDecoderSession for Echo {
        fn configure(&mut self, _: &DecoderConfig) -> CodecResult<()> {
            Ok(())
        }
        fn decode(&mut self, frame: EncodedFrame) -> CodecResult<Option<CapturedFrame>> {
            if self.fails_after.is_some_and(|after| self.decoded >= after) {
                return Err(CodecError::UnsupportedStream {
                    backend: BackendId::Vaapi,
                    reason: "interlaced".into(),
                });
            }
            self.decoded += 1;
            Ok(frame.data.last().map(|&tag| picture(tag)))
        }
        fn flush(&mut self) -> CodecResult<Vec<CapturedFrame>> {
            Ok(Vec::new())
        }
    }

    static OPENED_SOFTWARE: AtomicUsize = AtomicUsize::new(0);

    use ferricast_core::BackendSelection;

    fn config(backend: BackendSelection) -> DecoderConfig {
        DecoderConfig {
            codec: Codec::H264,
            width: 1,
            height: 1,
            pixel_format: PixelFormat::Nv12,
            low_latency: true,
            colour: None,
            backend,
        }
    }

    #[expect(clippy::unnecessary_wraps, reason = "an `OpenFn`")]
    fn rejecting(_: &DecoderConfig) -> CodecResult<Box<dyn VideoDecoderSession>> {
        Ok(Box::new(Rejecting(CodecError::backend(
            BackendId::Nvdec,
            Operation::Decode,
            "CUVID_ERROR",
        ))))
    }

    #[expect(clippy::unnecessary_wraps, reason = "an `OpenFn`")]
    fn quitting(_: &DecoderConfig) -> CodecResult<Box<dyn VideoDecoderSession>> {
        Ok(Box::new(Echo {
            decoded: 0,
            fails_after: Some(2),
        }))
    }

    #[expect(clippy::unnecessary_wraps, reason = "an `OpenFn`")]
    fn software(_: &DecoderConfig) -> CodecResult<Box<dyn VideoDecoderSession>> {
        OPENED_SOFTWARE.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Echo {
            decoded: 0,
            fails_after: None,
        }))
    }

    fn decoder(
        first: crate::registry::OpenFn<DecoderConfig, dyn VideoDecoderSession>,
    ) -> SelectingVideoDecoder {
        let registry = VideoDecoderRegistry::new(CodecTarget::decode(Codec::H264))
            .candidate(Candidate::new(
                BackendId::Nvdec,
                Acceleration::Hardware,
                first,
            ))
            .candidate(Candidate::new(
                BackendId::OpenH264,
                Acceleration::Software,
                software,
            ));
        SelectingVideoDecoder::new(registry, Some(NalSyntax::H264))
    }

    #[test]
    fn a_backend_that_fails_before_its_first_picture_replays_into_the_next() {
        let mut decoder = decoder(rejecting);
        decoder
            .configure(&config(BackendSelection::default()))
            .unwrap();
        let first = decoder.decode(packet(1)).unwrap();
        assert_eq!(first.as_ref().map(tag), Some(1));
        assert_eq!(decoder.backend().map(|b| b.0), Some(BackendId::OpenH264));
        assert_eq!(
            decoder.decode(packet(2)).unwrap().as_ref().map(tag),
            Some(2)
        );
    }

    #[test]
    fn a_mid_stream_refusal_hands_over_with_the_parameter_sets() {
        let mut decoder = decoder(quitting);
        decoder
            .configure(&config(BackendSelection::default()))
            .unwrap();
        let mut sps = packet(1);
        sps.data = Bytes::from_static(&[0, 0, 1, 0x67, 5, 0, 0, 1, 0x65, 1]);
        assert_eq!(decoder.decode(sps).unwrap().as_ref().map(tag), Some(1));
        assert_eq!(
            decoder.decode(packet(2)).unwrap().as_ref().map(tag),
            Some(2)
        );
        assert_eq!(
            decoder.decode(packet(3)).unwrap().as_ref().map(tag),
            Some(3)
        );
        assert_eq!(decoder.backend().map(|b| b.0), Some(BackendId::OpenH264));
        assert_eq!(
            decoder.remembered_parameter_sets().as_deref(),
            Some(&[0, 0, 0, 1, 0x67, 5][..])
        );
    }

    #[test]
    fn a_fixed_backend_reports_its_failure() {
        let mut decoder = decoder(rejecting);
        decoder.configure(&config(BackendId::Nvdec.into())).unwrap();
        assert_eq!(
            decoder.decode(packet(1)).unwrap_err(),
            CodecError::backend(BackendId::Nvdec, Operation::Decode, "CUVID_ERROR")
        );
    }

    #[test]
    fn decoding_before_configure_is_an_explicit_error() {
        let mut decoder = decoder(rejecting);
        assert_eq!(
            decoder.decode(packet(1)).unwrap_err(),
            CodecError::NotConfigured {
                target: CodecTarget::decode(Codec::H264),
                operation: Operation::Decode,
            }
        );
    }

    #[test]
    fn a_new_selection_reopens_the_backend() {
        let mut decoder = decoder(rejecting);
        decoder
            .configure(&config(BackendSelection::default()))
            .unwrap();
        assert_eq!(decoder.backend().map(|b| b.0), Some(BackendId::Nvdec));
        let software_only = AutoSelect::default()
            .with_acceleration(ferricast_core::AccelerationPolicy::SoftwareOnly);
        decoder.configure(&config(software_only.into())).unwrap();
        assert_eq!(decoder.backend().map(|b| b.0), Some(BackendId::OpenH264));
    }
}
