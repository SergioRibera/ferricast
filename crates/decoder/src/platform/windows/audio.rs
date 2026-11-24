//! AAC decoding through Microsoft's AAC decoder MFT.
//!
//! Input is ADTS or raw access units (raw ones take their shape from
//! the configuration). Every unit is handed to the MFT as one ADTS
//! frame (`MF_MT_AAC_PAYLOAD_TYPE` 1), so a stream whose shape changes
//! mid-way simply reopens the MFT for the new shape. Output is
//! interleaved 16-bit PCM at the rate and channel count the MFT
//! reports (HE-AAC at its full SBR rate).

use std::collections::VecDeque;

use ferricast_codec::AudioDecoderSession;
use ferricast_codec::aac::AacLc;
use ferricast_core::{
    AudioCodec, AudioDecoderConfig, AudioFrame, CodecError, CodecResult, CodecTarget, DecodedAudio,
    Operation,
};
use windows::Win32::Media::MediaFoundation::{
    IMFTransform, MF_E_INVALIDMEDIATYPE, MF_MT_AAC_PAYLOAD_TYPE, MF_MT_AUDIO_BITS_PER_SAMPLE,
    MF_MT_AUDIO_NUM_CHANNELS, MF_MT_AUDIO_SAMPLES_PER_SECOND, MF_MT_SUBTYPE, MF_MT_USER_DATA,
    MFAudioFormat_AAC, MFAudioFormat_PCM, MFMediaType_Audio, MFT_CATEGORY_AUDIO_DECODER,
};

use super::mf::{
    self, BACKEND, Kind, MfResult, Output, Search, Transform, Worker, hns, media_type, micros,
};
use crate::adts::{self, Shape};

/// Samples per channel in one AAC-LC access unit.
const FRAME_SAMPLES: u64 = 1024;

fn search() -> Search {
    Search {
        category: MFT_CATEGORY_AUDIO_DECODER,
        kind: Kind::Software,
        input: Some((MFMediaType_Audio, MFAudioFormat_AAC)),
        output: None,
        job: "AAC decoder",
    }
}

pub(super) fn probe_aac() -> CodecResult<()> {
    search().probe()
}

pub(super) fn open_aac(config: &AudioDecoderConfig) -> CodecResult<Box<dyn AudioDecoderSession>> {
    Ok(Box::new(MfAacDecoder::open(config)?))
}

struct MfAacDecoder {
    worker: Worker<Session>,
}

impl MfAacDecoder {
    fn open(config: &AudioDecoderConfig) -> CodecResult<Self> {
        if config.codec != AudioCodec::Aac {
            return Err(CodecError::BackendCannotHandle {
                backend: BACKEND,
                target: CodecTarget::decode(config.codec),
            });
        }
        let configured = (config.sample_rate > 0 && config.channels > 0).then_some(Shape {
            sample_rate: config.sample_rate,
            channels: config.channels,
        });
        let worker = Worker::spawn("ferricast-mf-aac-decoder", move || {
            let mut session = Session {
                configured,
                active: None,
                ready: VecDeque::new(),
                in_flight: VecDeque::new(),
            };
            // Fail at open, not on the first packet, when the shape is
            // known and the MFT refuses it (or none is installed).
            match configured {
                Some(shape) => session.reopen(shape)?,
                None => search().enumerate().map(drop)?,
            }
            Ok(session)
        })?;
        Ok(Self { worker })
    }
}

impl AudioDecoderSession for MfAacDecoder {
    fn configure(&mut self, config: &AudioDecoderConfig) -> CodecResult<()> {
        *self = Self::open(config)?;
        Ok(())
    }

    fn decode(&mut self, frame: AudioFrame) -> CodecResult<Option<DecodedAudio>> {
        self.worker
            .run(Operation::Decode, move |session| session.decode(&frame))
    }

    fn flush(&mut self) -> CodecResult<Vec<DecodedAudio>> {
        self.worker.run(Operation::Flush, Session::flush)
    }
}

struct Session {
    /// The shape the stream metadata states, for raw input.
    configured: Option<Shape>,
    /// The MFT and the shape it was opened for.
    active: Option<(Shape, Transform)>,
    ready: VecDeque<DecodedAudio>,
    /// Timestamps of the access units fed, oldest first: the fallback
    /// for output the MFT does not stamp.
    in_flight: VecDeque<u64>,
}

impl Session {
    fn reopen(&mut self, shape: Shape) -> CodecResult<()> {
        let aac = AacLc::new(shape.sample_rate, shape.channels).ok_or_else(|| {
            CodecError::UnsupportedConfig {
                backend: BACKEND,
                what: format!(
                    "AAC at {} Hz / {} channels",
                    shape.sample_rate, shape.channels
                ),
            }
        })?;
        let transform = search().open(|transform| setup(transform, shape, aac))?;
        self.active = Some((shape, transform));
        Ok(())
    }

    fn decode(&mut self, frame: &AudioFrame) -> CodecResult<Option<DecodedAudio>> {
        let op = Operation::Decode;
        let mut decoded = 0u64;
        for unit in adts::units(&frame.data) {
            let unit =
                unit.map_err(|error| CodecError::backend(BACKEND, op, format!("AAC: {error}")))?;
            let shape = unit.shape.or(self.configured).ok_or_else(|| {
                CodecError::backend(
                    BACKEND,
                    op,
                    "raw AAC input with no sample rate / channels configured",
                )
            })?;
            if self
                .active
                .as_ref()
                .is_none_or(|(active, _)| *active != shape)
            {
                self.flush_into_ready()?;
                self.reopen(shape)?;
            }
            let aac = AacLc::new(shape.sample_rate, shape.channels).ok_or_else(|| {
                CodecError::backend(BACKEND, op, "AAC shape without an ADTS code")
            })?;
            let adts = aac.adts_frame(unit.payload).ok_or_else(|| {
                CodecError::backend(BACKEND, op, "AAC access unit too long for ADTS")
            })?;
            let offset_us = decoded * 1_000_000 / u64::from(shape.sample_rate.max(1));
            let sample = mf::sample_with(adts.len(), 16, op, |dst| {
                dst.copy_from_slice(&adts);
                Ok(())
            })?;
            // SAFETY: plain attribute write on the new sample.
            unsafe { sample.SetSampleTime(hns(frame.timestamp_us + offset_us)) }
                .mf(op, "IMFSample::SetSampleTime")?;
            self.in_flight.push_back(frame.timestamp_us + offset_us);
            if self.in_flight.len() > 64 {
                self.in_flight.pop_front();
            }
            decoded += FRAME_SAMPLES;
            let Some((_, transform)) = &mut self.active else {
                continue;
            };
            let mut outputs = Vec::new();
            transform
                .push(&sample, &mut outputs)
                .mf(op, "IMFTransform::ProcessInput")?;
            transform
                .pull(&mut outputs, None)
                .mf(op, "IMFTransform::ProcessOutput")?;
            self.collect(outputs)?;
        }
        Ok(self.ready.pop_front())
    }

    fn flush(&mut self) -> CodecResult<Vec<DecodedAudio>> {
        self.flush_into_ready()?;
        Ok(self.ready.drain(..).collect())
    }

    fn flush_into_ready(&mut self) -> CodecResult<()> {
        let Some((_, transform)) = &mut self.active else {
            return Ok(());
        };
        let mut outputs = Vec::new();
        transform
            .drain(&mut outputs)
            .mf(Operation::Flush, "draining the AAC decoder")?;
        self.collect(outputs)
    }

    fn collect(&mut self, outputs: Vec<Output>) -> CodecResult<()> {
        let op = Operation::Decode;
        let Some((shape, transform)) = &self.active else {
            return Ok(());
        };
        let (sample_rate, channels) =
            pcm_shape(transform.mft()).unwrap_or((shape.sample_rate, shape.channels));
        for output in outputs {
            let Output::Sample(sample) = output else {
                continue;
            };
            // SAFETY: plain attribute read on the output sample.
            let reported = unsafe { sample.GetSampleTime() }.ok().map(micros);
            let timestamp_us = match reported {
                Some(time) => {
                    // Units up to this one are decoded.
                    while self.in_flight.front().is_some_and(|&fed| fed <= time) {
                        self.in_flight.pop_front();
                    }
                    time
                }
                None => {
                    let fallback = self.in_flight.pop_front();
                    tracing::warn!(
                        ?fallback,
                        "Media Foundation AAC output carries no sample time; using the oldest \
                         pending access unit timestamp"
                    );
                    fallback
                        .or(self.ready.back().map(|last| last.timestamp_us))
                        .unwrap_or(0)
                }
            };
            let bytes = mf::sample_bytes(&sample, op)?;
            let pcm: Vec<i16> = bytes
                .chunks_exact(2)
                .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
                .collect();
            if pcm.is_empty() {
                continue;
            }
            self.ready.push_back(DecodedAudio {
                pcm,
                sample_rate,
                channels,
                timestamp_us,
            });
        }
        Ok(())
    }
}

/// Configures `transform` for ADTS AAC of `shape` in, 16-bit PCM out.
fn setup(transform: Transform, shape: Shape, aac: AacLc) -> CodecResult<Transform> {
    let op = Operation::Open;
    let input = media_type(&MFMediaType_Audio, &MFAudioFormat_AAC, op)?;
    // HEAACWAVEINFO after its WAVEFORMATEX: payload type (1 = ADTS),
    // profile/level (0xFE = unspecified), struct type, two reserved
    // fields; then the AudioSpecificConfig.
    let mut user_data = Vec::with_capacity(14);
    user_data.extend_from_slice(&1u16.to_le_bytes());
    user_data.extend_from_slice(&0xFEu16.to_le_bytes());
    user_data.extend_from_slice(&[0; 8]);
    user_data.extend_from_slice(&aac.audio_specific_config());
    // SAFETY: plain attribute writes and COM calls on live objects.
    unsafe {
        input
            .SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, shape.sample_rate)
            .mf(op, "setting MF_MT_AUDIO_SAMPLES_PER_SECOND")?;
        input
            .SetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, u32::from(shape.channels))
            .mf(op, "setting MF_MT_AUDIO_NUM_CHANNELS")?;
        input
            .SetUINT32(&MF_MT_AAC_PAYLOAD_TYPE, 1)
            .mf(op, "setting MF_MT_AAC_PAYLOAD_TYPE")?;
        input
            .SetBlob(&MF_MT_USER_DATA, &user_data)
            .mf(op, "setting MF_MT_USER_DATA")?;
        transform
            .mft()
            .SetInputType(0, &input, 0)
            .mf(op, "IMFTransform::SetInputType(AAC)")?;
    }
    select_pcm16(transform.mft()).mf(op, "selecting 16-bit PCM output")?;
    let mut transform = transform;
    transform.set_default_output_size(64 * 1024);
    transform.start()?;
    Ok(transform)
}

/// Picks the MFT's 16-bit PCM output type.
fn select_pcm16(mft: &IMFTransform) -> windows::core::Result<()> {
    for index in 0.. {
        // SAFETY: plain COM calls on the live transform; the loop ends
        // when the MFT runs out of types (an error).
        let offered = unsafe { mft.GetOutputAvailableType(0, index)? };
        // SAFETY: plain attribute reads.
        let (subtype, bits) = unsafe {
            (
                offered.GetGUID(&MF_MT_SUBTYPE)?,
                offered.GetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE).unwrap_or(0),
            )
        };
        if subtype == MFAudioFormat_PCM && bits == 16 {
            // SAFETY: as above.
            return unsafe { mft.SetOutputType(0, &offered, 0) };
        }
    }
    Err(MF_E_INVALIDMEDIATYPE.into())
}

/// The rate and channel count of the MFT's current PCM output.
fn pcm_shape(mft: &IMFTransform) -> Option<(u32, u16)> {
    // SAFETY: plain COM calls on the live transform and its type.
    unsafe {
        let current = mft.GetOutputCurrentType(0).ok()?;
        let rate = current.GetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND).ok()?;
        let channels = current.GetUINT32(&MF_MT_AUDIO_NUM_CHANNELS).ok()?;
        Some((rate, u16::try_from(channels).ok()?))
    }
}
