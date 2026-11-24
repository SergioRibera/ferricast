//! AAC-LC encoding through Microsoft's AAC encoder MFT.
//!
//! The MFT takes 16-bit PCM at 44.1 or 48 kHz in 1, 2 or 6 channels and
//! encodes at 96, 128, 160 or 192 kbps only: a configured bitrate maps
//! to the nearest of those (the choice is logged). It emits raw AAC
//! access units, framed here as ADTS like the libfdk-aac encoder's
//! output; [`ferricast_codec::AudioEncoderSession::codec_config`]
//! returns the matching `AudioSpecificConfig`.

use bytes::Bytes;
use ferricast_codec::AudioEncoderSession;
use ferricast_codec::aac::AacLc;
use ferricast_core::{
    AudioCodec, AudioEncoderConfig, AudioFrame, CodecError, CodecResult, CodecTarget, Operation,
};
use windows::Win32::Media::MediaFoundation::IMFMediaType;
use windows::Win32::Media::MediaFoundation::{
    MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION, MF_MT_AAC_PAYLOAD_TYPE,
    MF_MT_AUDIO_AVG_BYTES_PER_SECOND, MF_MT_AUDIO_BITS_PER_SAMPLE, MF_MT_AUDIO_BLOCK_ALIGNMENT,
    MF_MT_AUDIO_NUM_CHANNELS, MF_MT_AUDIO_SAMPLES_PER_SECOND, MFAudioFormat_AAC, MFAudioFormat_PCM,
    MFMediaType_Audio, MFT_CATEGORY_AUDIO_ENCODER,
};

use super::mf::{
    self, BACKEND, Kind, MfResult, Output, Search, Transform, Worker, hns, media_type,
};
use crate::audio_clock::{AudioClock, ClockEvent};

/// Samples per channel in one AAC-LC access unit.
const FRAME_SAMPLES: u64 = 1024;

/// Bitrates (kbps) the Microsoft AAC encoder accepts.
const BITRATES_KBPS: [u32; 4] = [96, 128, 160, 192];

fn search() -> Search {
    Search {
        category: MFT_CATEGORY_AUDIO_ENCODER,
        kind: Kind::Software,
        input: Some((MFMediaType_Audio, MFAudioFormat_PCM)),
        output: Some((MFMediaType_Audio, MFAudioFormat_AAC)),
        job: "AAC encoder",
    }
}

pub(super) fn probe_aac() -> CodecResult<()> {
    search().probe()
}

pub(super) fn open_aac(config: &AudioEncoderConfig) -> CodecResult<Box<dyn AudioEncoderSession>> {
    Ok(Box::new(MfAacEncoder::open(config)?))
}

/// A Media Foundation AAC-LC encoder session.
struct MfAacEncoder {
    worker: Worker<Session>,
    shape: AacLc,
}

impl MfAacEncoder {
    fn open(config: &AudioEncoderConfig) -> CodecResult<Self> {
        if config.codec != AudioCodec::Aac {
            return Err(CodecError::BackendCannotHandle {
                backend: BACKEND,
                target: CodecTarget::encode(config.codec),
            });
        }
        if !matches!(config.sample_rate, 44_100 | 48_000) {
            return Err(CodecError::UnsupportedConfig {
                backend: BACKEND,
                what: format!("AAC at {} Hz (44100 or 48000 only)", config.sample_rate),
            });
        }
        if !matches!(config.channels, 1 | 2 | 6) {
            return Err(CodecError::UnsupportedConfig {
                backend: BACKEND,
                what: format!("AAC with {} channels (1, 2 or 6 only)", config.channels),
            });
        }
        let shape = AacLc::new(config.sample_rate, config.channels).ok_or_else(|| {
            CodecError::UnsupportedConfig {
                backend: BACKEND,
                what: format!(
                    "AAC at {} Hz / {} channels",
                    config.sample_rate, config.channels
                ),
            }
        })?;
        let config = config.clone();
        let worker = Worker::spawn("ferricast-mf-aac-encoder", move || {
            Session::open(&config, shape)
        })?;
        Ok(Self { worker, shape })
    }
}

impl AudioEncoderSession for MfAacEncoder {
    fn configure(&mut self, config: &AudioEncoderConfig) -> CodecResult<()> {
        *self = Self::open(config)?;
        Ok(())
    }

    fn encode(&mut self, frame: &AudioFrame) -> CodecResult<()> {
        let frame = frame.clone();
        self.worker
            .run(Operation::Encode, move |session| session.encode(&frame))
    }

    fn take_output(&mut self) -> Vec<AudioFrame> {
        self.worker
            .run(Operation::Encode, |session| {
                Ok(std::mem::take(&mut session.pending))
            })
            .unwrap_or_else(|error| {
                tracing::warn!(%error, "AAC output lost");
                Vec::new()
            })
    }

    fn codec_config(&self) -> Vec<u8> {
        self.shape.audio_specific_config().to_vec()
    }

    fn flush(self: Box<Self>) -> CodecResult<Vec<AudioFrame>> {
        self.worker.run(Operation::Flush, Session::flush)
    }
}

struct Session {
    transform: Transform,
    shape: AacLc,
    sample_rate: u32,
    channels: u16,
    clock: AudioClock,
    /// Samples (per channel) fed to the MFT.
    fed: u64,
    /// Access units it returned.
    emitted: u64,
    pending: Vec<AudioFrame>,
}

impl Session {
    fn open(config: &AudioEncoderConfig, shape: AacLc) -> CodecResult<Self> {
        let kbps = BITRATES_KBPS
            .into_iter()
            .min_by_key(|&rate| rate.abs_diff(config.bitrate_kbps))
            .unwrap_or(128);
        if kbps != config.bitrate_kbps {
            tracing::info!(
                requested_kbps = config.bitrate_kbps,
                kbps,
                "Media Foundation AAC encodes at 96/128/160/192 kbps only; using the nearest"
            );
        }
        let config = config.clone();
        search().open(|transform| Self::setup(transform, &config, kbps, shape))
    }

    fn setup(
        transform: Transform,
        config: &AudioEncoderConfig,
        kbps: u32,
        shape: AacLc,
    ) -> CodecResult<Self> {
        let op = Operation::Open;
        let channels = u32::from(config.channels);
        let set_shape = |media: &IMFMediaType| -> CodecResult<()> {
            // SAFETY: plain attribute writes on a live media type.
            unsafe {
                media
                    .SetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE, 16)
                    .mf(op, "setting MF_MT_AUDIO_BITS_PER_SAMPLE")?;
                media
                    .SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, config.sample_rate)
                    .mf(op, "setting MF_MT_AUDIO_SAMPLES_PER_SECOND")?;
                media
                    .SetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, channels)
                    .mf(op, "setting MF_MT_AUDIO_NUM_CHANNELS")?;
            }
            Ok(())
        };
        let input = media_type(&MFMediaType_Audio, &MFAudioFormat_PCM, op)?;
        set_shape(&input)?;
        let output = media_type(&MFMediaType_Audio, &MFAudioFormat_AAC, op)?;
        set_shape(&output)?;
        // SAFETY: plain attribute writes and COM calls on live objects.
        unsafe {
            input
                .SetUINT32(&MF_MT_AUDIO_BLOCK_ALIGNMENT, channels * 2)
                .mf(op, "setting MF_MT_AUDIO_BLOCK_ALIGNMENT")?;
            input
                .SetUINT32(
                    &MF_MT_AUDIO_AVG_BYTES_PER_SECOND,
                    config.sample_rate * channels * 2,
                )
                .mf(op, "setting MF_MT_AUDIO_AVG_BYTES_PER_SECOND")?;
            output
                .SetUINT32(&MF_MT_AUDIO_AVG_BYTES_PER_SECOND, kbps * 1000 / 8)
                .mf(op, "setting MF_MT_AUDIO_AVG_BYTES_PER_SECOND")?;
            // Raw access units; ADTS is added here.
            output
                .SetUINT32(&MF_MT_AAC_PAYLOAD_TYPE, 0)
                .mf(op, "setting MF_MT_AAC_PAYLOAD_TYPE")?;
            // AAC profile, level 2 (ISO/IEC 14496-3 audioProfileLevelIndication 0x29).
            output
                .SetUINT32(&MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION, 0x29)
                .mf(op, "setting MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION")?;
            transform
                .mft()
                .SetInputType(0, &input, 0)
                .mf(op, "IMFTransform::SetInputType(PCM)")?;
            transform
                .mft()
                .SetOutputType(0, &output, 0)
                .mf(op, "IMFTransform::SetOutputType(AAC)")?;
        }
        let mut transform = transform;
        transform.set_default_output_size(8 * 1024);
        transform.start()?;
        Ok(Self {
            transform,
            shape,
            sample_rate: config.sample_rate,
            channels: config.channels,
            clock: AudioClock::new(config.sample_rate),
            fed: 0,
            emitted: 0,
            pending: Vec::new(),
        })
    }

    fn encode(&mut self, frame: &AudioFrame) -> CodecResult<()> {
        let op = Operation::Encode;
        if frame.codec != AudioCodec::Pcm {
            return Err(CodecError::backend(
                BACKEND,
                op,
                format!("expected PCM input, got {}", frame.codec),
            ));
        }
        if frame.sample_rate != self.sample_rate || frame.channels != self.channels {
            return Err(CodecError::backend(
                BACKEND,
                op,
                format!(
                    "input shape changed mid-stream: configured {} Hz / {} ch, got {} Hz / {} ch",
                    self.sample_rate, self.channels, frame.sample_rate, frame.channels
                ),
            ));
        }
        let frame_bytes = usize::from(self.channels.max(1)) * 2;
        let usable = frame.data.len() / frame_bytes * frame_bytes;
        let chunk = (usable / frame_bytes) as u64;
        let buffered = self.fed.saturating_sub(self.emitted * FRAME_SAMPLES);
        let event = self.clock.observe(frame.timestamp_us, buffered, chunk);
        match event {
            ClockEvent::Resynced { gap_us } => {
                tracing::debug!(
                    gap_us,
                    "Media Foundation AAC: capture gap, re-anchoring output timestamps"
                );
            }
            ClockEvent::Trimmed {
                ahead_us,
                report: true,
                ..
            } => {
                tracing::info!(
                    ahead_us,
                    "Media Foundation AAC: output clock ahead of capture, discarding input samples"
                );
            }
            ClockEvent::Anchored | ClockEvent::Continuous | ClockEvent::Trimmed { .. } => {}
        }
        let skip = usize::try_from(event.skip())
            .map_or(usable, |skip| skip.saturating_mul(frame_bytes).min(usable));
        let pcm = &frame.data[skip..usable];
        if pcm.is_empty() {
            return Ok(());
        }
        let sample = mf::sample_with(pcm.len(), 16, op, |dst| {
            dst.copy_from_slice(pcm);
            Ok(())
        })?;
        // SAFETY: plain attribute write on the new sample; the MFT
        // needs some time, the output is stamped by `clock`.
        unsafe { sample.SetSampleTime(hns(frame.timestamp_us)) }
            .mf(op, "IMFSample::SetSampleTime")?;
        let mut outputs = Vec::new();
        self.transform
            .push(&sample, &mut outputs)
            .mf(op, "IMFTransform::ProcessInput")?;
        self.fed += (pcm.len() / frame_bytes) as u64;
        self.transform
            .pull(&mut outputs, None)
            .mf(op, "IMFTransform::ProcessOutput")?;
        self.collect(outputs)
    }

    fn flush(&mut self) -> CodecResult<Vec<AudioFrame>> {
        let mut outputs = Vec::new();
        self.transform
            .drain(&mut outputs)
            .mf(Operation::Flush, "draining the AAC encoder")?;
        self.collect(outputs)?;
        Ok(std::mem::take(&mut self.pending))
    }

    fn collect(&mut self, outputs: Vec<Output>) -> CodecResult<()> {
        for output in outputs {
            let Output::Sample(sample) = output else {
                continue;
            };
            let payload = mf::sample_bytes(&sample, Operation::Encode)?;
            if payload.is_empty() {
                continue;
            }
            let adts = self.shape.adts_frame(&payload).ok_or_else(|| {
                CodecError::backend(
                    BACKEND,
                    Operation::Encode,
                    format!(
                        "an AAC access unit of {} bytes does not fit ADTS",
                        payload.len()
                    ),
                )
            })?;
            self.pending.push(AudioFrame {
                codec: AudioCodec::Aac,
                data: Bytes::from(adts),
                timestamp_us: self.clock.next_timestamp_us(),
                sample_rate: self.sample_rate,
                channels: self.channels,
            });
            self.clock.advance(FRAME_SAMPLES);
            self.emitted += 1;
        }
        Ok(())
    }
}
