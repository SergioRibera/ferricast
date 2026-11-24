//! AAC-LC and Opus encode through `AMediaCodec`.
//!
//! PCM is cut into whole codec frames (1024 samples for AAC-LC, 20 ms
//! for Opus), each queued with the timestamp the shared
//! [`AudioClock`] gives its first sample; `MediaCodec` carries that time
//! to the packet it encodes. AAC comes out as raw access units and is
//! wrapped in ADTS here, like the libfdk-aac backend emits it.

use std::time::Duration;

use bytes::Bytes;
use ferricast_codec::aac::AacLc;
use ferricast_core::{
    AudioCodec, AudioEncoder, AudioEncoderConfig, AudioFrame, BackendId, CodecError, CodecResult,
    Operation,
};
use ndk::media::media_codec::MediaFormat;
use tracing::{debug, info, warn};

use super::codec::{
    self, Codec as MediaCodec, FLAG_CODEC_CONFIG, FLAG_END_OF_STREAM, Output, Pick, fail, key,
};
use crate::audio_clock::{AudioClock, ClockEvent};

/// Samples per channel in one AAC-LC frame.
const AAC_FRAME_SAMPLES: usize = 1024;
/// Opus frames per second (20 ms frames, what Cast Streaming expects).
const OPUS_FRAMES_PER_SECOND: u32 = 50;
/// The rates Opus encodes natively.
const OPUS_RATES: [u32; 5] = [8_000, 12_000, 16_000, 24_000, 48_000];

/// `MediaCodecInfo.CodecProfileLevel.AACObjectLC`.
const AAC_OBJECT_LC: i32 = 2;

/// The MIME type of `codec`, when `MediaCodec` encodes it.
pub(crate) fn mime(codec: AudioCodec) -> Option<&'static str> {
    match codec {
        AudioCodec::Aac => Some("audio/mp4a-latm"),
        AudioCodec::Opus => Some("audio/opus"),
        AudioCodec::AacEld | AudioCodec::Pcm | AudioCodec::Alac => None,
    }
}

/// The Google software encoder of `codec`.
pub(crate) fn software_name(codec: AudioCodec) -> Option<&'static str> {
    match codec {
        AudioCodec::Aac => Some("c2.android.aac.encoder"),
        // `c2.android.opus.encoder` ships from Android 10 (API 29).
        AudioCodec::Opus => Some("c2.android.opus.encoder"),
        AudioCodec::AacEld | AudioCodec::Pcm | AudioCodec::Alac => None,
    }
}

struct Session {
    codec: MediaCodec,
    kind: AudioCodec,
    sample_rate: u32,
    channels: u16,
    frame_samples: usize,
    adts: Option<AacLc>,
    clock: AudioClock,
    pcm: Vec<u8>,
    pending: Vec<AudioFrame>,
}

/// A `MediaCodec` audio encoder.
pub(crate) struct MediaCodecAudioEncoder {
    pick: Pick,
    session: Option<Session>,
}

/// Opens an encoder for `config`; the candidate's open function.
pub(crate) fn open(pick: Pick, config: &AudioEncoderConfig) -> CodecResult<MediaCodecAudioEncoder> {
    let mut encoder = MediaCodecAudioEncoder {
        pick,
        session: None,
    };
    AudioEncoder::configure(&mut encoder, config)?;
    Ok(encoder)
}

fn unsupported(what: String) -> CodecError {
    CodecError::UnsupportedConfig {
        backend: BackendId::MediaCodec,
        what,
    }
}

impl Session {
    fn open(pick: Pick, config: &AudioEncoderConfig) -> CodecResult<Self> {
        let mime = mime(config.codec).ok_or_else(|| CodecError::BackendCannotHandle {
            backend: BackendId::MediaCodec,
            target: ferricast_core::CodecTarget::encode(config.codec),
        })?;
        let (frame_samples, adts) = if config.codec == AudioCodec::Opus {
            if !OPUS_RATES.contains(&config.sample_rate) || !matches!(config.channels, 1 | 2) {
                return Err(unsupported(format!(
                    "Opus from {} Hz / {} channels (8-48 kHz Opus rates, mono or stereo)",
                    config.sample_rate, config.channels
                )));
            }
            ((config.sample_rate / OPUS_FRAMES_PER_SECOND) as usize, None)
        } else {
            let shape = AacLc::new(config.sample_rate, config.channels).ok_or_else(|| {
                unsupported(format!(
                    "AAC-LC at {} Hz / {} channels",
                    config.sample_rate, config.channels
                ))
            })?;
            (AAC_FRAME_SAMPLES, Some(shape))
        };
        let int = |value: u32, what: &str| {
            i32::try_from(value).map_err(|_| unsupported(format!("a {what} of {value}")))
        };
        let mut format = MediaFormat::new();
        format.set_str(key::MIME, mime);
        format.set_i32(key::SAMPLE_RATE, int(config.sample_rate, "sample rate")?);
        format.set_i32(key::CHANNEL_COUNT, i32::from(config.channels));
        format.set_i32(
            key::BIT_RATE,
            int(config.bitrate_kbps.saturating_mul(1000), "bitrate")?,
        );
        if adts.is_some() {
            format.set_i32(key::AAC_PROFILE, AAC_OBJECT_LC);
        }
        let frame_bytes = frame_samples * usize::from(config.channels) * 2;
        format.set_i32(
            key::MAX_INPUT_SIZE,
            i32::try_from(frame_bytes).unwrap_or(i32::MAX),
        );
        let codec = MediaCodec::create(mime, pick)?;
        codec.start_encoder(&format).map_err(|error| {
            unsupported(format!(
                "{} at {} Hz / {} channels / {} kbps on {} ({error})",
                config.codec,
                config.sample_rate,
                config.channels,
                config.bitrate_kbps,
                codec.label()
            ))
        })?;
        info!(codec = %codec.label(), sample_rate = config.sample_rate, channels = config.channels, "MediaCodec audio encoder opened");
        Ok(Self {
            codec,
            kind: config.codec,
            sample_rate: config.sample_rate,
            channels: config.channels,
            frame_samples,
            adts,
            clock: AudioClock::new(config.sample_rate),
            pcm: Vec::with_capacity(frame_bytes * 4),
            pending: Vec::new(),
        })
    }

    fn frame_bytes(&self) -> usize {
        self.frame_samples * usize::from(self.channels.max(1)) * 2
    }

    /// Files one output buffer as a packet (codec configuration buffers
    /// carry nothing a packet consumer needs: ADTS describes itself).
    fn accept(&mut self, output: Output) -> CodecResult<()> {
        if output.is(FLAG_CODEC_CONFIG) || output.data.is_empty() {
            return Ok(());
        }
        let data = match self.adts {
            Some(shape) => shape
                .adts_frame(&output.data)
                .ok_or_else(|| fail(Operation::Encode, "AAC access unit too long for ADTS"))?,
            None => output.data,
        };
        self.pending.push(AudioFrame {
            codec: self.kind,
            data: Bytes::from(data),
            timestamp_us: output.pts_us,
            sample_rate: self.sample_rate,
            channels: self.channels,
        });
        Ok(())
    }

    fn drain(&mut self, operation: Operation, timeout: Duration) -> CodecResult<bool> {
        let mut ended = false;
        while let Some(output) = self.codec.next_output(operation, timeout)? {
            ended |= output.is(FLAG_END_OF_STREAM);
            self.accept(output)?;
            if ended {
                break;
            }
        }
        Ok(ended)
    }

    fn encode(&mut self, frame: &AudioFrame) -> CodecResult<()> {
        if frame.codec != AudioCodec::Pcm {
            return Err(fail(
                Operation::Encode,
                format!("expected PCM input, got {}", frame.codec),
            ));
        }
        if frame.sample_rate != self.sample_rate || frame.channels != self.channels {
            return Err(fail(
                Operation::Encode,
                format!(
                    "input shape changed mid-stream: configured {} Hz / {} ch, got {} Hz / {} ch",
                    self.sample_rate, self.channels, frame.sample_rate, frame.channels
                ),
            ));
        }
        // Same clock policy as the other encoders: anchor on the first
        // chunk, re-anchor after capture gaps, trim when running ahead.
        let channels = usize::from(self.channels.max(1));
        let usable = frame.data.len() & !1;
        let buffered = (self.pcm.len() / 2 / channels) as u64;
        let chunk = (usable / 2 / channels) as u64;
        let event = self.clock.observe(frame.timestamp_us, buffered, chunk);
        match event {
            ClockEvent::Resynced { gap_us } => {
                debug!(
                    gap_us,
                    "MediaCodec audio: capture gap, re-anchoring output timestamps"
                );
            }
            ClockEvent::Trimmed {
                ahead_us,
                report: true,
                ..
            } => info!(
                ahead_us,
                "MediaCodec audio: output clock ahead of capture, discarding input"
            ),
            ClockEvent::Anchored | ClockEvent::Continuous | ClockEvent::Trimmed { .. } => {}
        }
        let skip = usize::try_from(event.skip())
            .map_or(usable, |skip| skip.saturating_mul(channels * 2).min(usable));
        self.pcm.extend_from_slice(&frame.data[skip..usable]);

        let frame_bytes = self.frame_bytes();
        while self.pcm.len() >= frame_bytes {
            let pts = self.clock.next_timestamp_us();
            let mut drained = Vec::new();
            let pcm = &self.pcm[..frame_bytes];
            self.codec.queue(
                Operation::Encode,
                pts,
                0,
                |buffer| codec::copy_into(buffer, pcm, Operation::Encode),
                |codec| {
                    while let Some(output) = codec.next_output(Operation::Encode, Duration::ZERO)? {
                        drained.push(output);
                    }
                    Ok(())
                },
            )?;
            self.pcm.drain(..frame_bytes);
            self.clock.advance(self.frame_samples as u64);
            for output in drained {
                self.accept(output)?;
            }
        }
        self.drain(Operation::Encode, Duration::ZERO)?;
        Ok(())
    }

    fn flush(mut self) -> CodecResult<Vec<AudioFrame>> {
        let pts = self.clock.next_timestamp_us();
        self.codec.queue(
            Operation::Flush,
            pts,
            FLAG_END_OF_STREAM,
            |_| Ok(0),
            |_| Ok(()),
        )?;
        if !self.drain(Operation::Flush, Duration::from_millis(250))? {
            warn!("MediaCodec audio encoder did not signal the end of the stream");
        }
        if let Err(error) = self.codec.stop() {
            warn!(%error, "AMediaCodec_stop failed after the flush");
        }
        Ok(self.pending)
    }
}

impl AudioEncoder for MediaCodecAudioEncoder {
    fn configure(&mut self, config: &AudioEncoderConfig) -> CodecResult<()> {
        // Release the previous instance first: codecs are a scarce resource.
        self.session = None;
        self.session = Some(Session::open(self.pick, config)?);
        Ok(())
    }

    fn encode(&mut self, frame: &AudioFrame) -> CodecResult<()> {
        self.session
            .as_mut()
            .ok_or_else(|| fail(Operation::Encode, "the encoder is not configured"))?
            .encode(frame)
    }

    fn take_output(&mut self) -> Vec<AudioFrame> {
        self.session
            .as_mut()
            .map(|session| std::mem::take(&mut session.pending))
            .unwrap_or_default()
    }

    /// Empty, like the libfdk-aac backend: ADTS-framed AAC describes
    /// itself, and Opus packets need no out-of-band header here.
    fn codec_config(&self) -> Vec<u8> {
        Vec::new()
    }

    fn flush(self) -> CodecResult<Vec<AudioFrame>> {
        self.session.map_or_else(|| Ok(Vec::new()), Session::flush)
    }
}
