//! AAC and Opus decode through `AMediaCodec` into 16-bit PCM.
//!
//! AAC arrives ADTS-framed or as raw access units ([`crate::adts`]); the
//! codec is configured with the stream's `AudioSpecificConfig` and fed
//! raw access units, each stamped 1024 samples after the one before it
//! in the same payload. A new ADTS shape reopens the codec. HE-AAC is
//! signalled implicitly in such streams; the codec's output format then
//! reports the doubled rate, which the PCM is labelled with.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use ferricast_codec::aac::AacLc;
use ferricast_core::{
    AudioCodec, AudioDecoder, AudioDecoderConfig, AudioFrame, BackendId, CodecError, CodecResult,
    DecodedAudio, Operation,
};
use ndk::media::media_codec::MediaFormat;
use tracing::{info, warn};

use super::codec::{
    Codec as MediaCodec, FLAG_CODEC_CONFIG, FLAG_END_OF_STREAM, Output, Pick, WAIT, copy_into,
    fail, key,
};
use crate::adts::{self, Shape};

/// Samples per channel in one AAC-LC access unit.
const AAC_FRAME_SAMPLES: u64 = 1024;
/// Opus always decodes at 48 kHz.
const OPUS_RATE: u32 = 48_000;
/// Opus seek pre-roll the codec is told about (RFC 7845 §4.6), in ns.
const OPUS_SEEK_PREROLL_NS: u64 = 80_000_000;
/// How long draining to the end of the stream waits at most.
const FLUSH_DEADLINE: Duration = Duration::from_secs(1);

/// The MIME type of `codec`, when `MediaCodec` decodes it.
pub(crate) fn mime(codec: AudioCodec) -> Option<&'static str> {
    match codec {
        AudioCodec::Aac => Some("audio/mp4a-latm"),
        AudioCodec::Opus => Some("audio/opus"),
        AudioCodec::AacEld | AudioCodec::Pcm | AudioCodec::Alac => None,
    }
}

/// The Google software decoder of `codec`.
pub(crate) fn software_name(codec: AudioCodec) -> Option<&'static str> {
    match codec {
        AudioCodec::Aac => Some("c2.android.aac.decoder"),
        AudioCodec::Opus => Some("c2.android.opus.decoder"),
        AudioCodec::AacEld | AudioCodec::Pcm | AudioCodec::Alac => None,
    }
}

struct Open {
    codec: MediaCodec,
    /// The AAC shape the codec was configured for (`None` for Opus).
    shape: Option<Shape>,
    /// Output shape from the codec's output format.
    output: Option<(u32, u16)>,
}

/// A `MediaCodec` audio decoder.
pub(crate) struct MediaCodecAudioDecoder {
    pick: Pick,
    kind: AudioCodec,
    configured: Option<Shape>,
    open: Option<Open>,
    ready: VecDeque<DecodedAudio>,
}

/// Opens a decoder for `config`; the candidate's open function.
pub(crate) fn open(pick: Pick, config: &AudioDecoderConfig) -> CodecResult<MediaCodecAudioDecoder> {
    let mut decoder = MediaCodecAudioDecoder {
        pick,
        kind: config.codec,
        configured: None,
        open: None,
        ready: VecDeque::new(),
    };
    AudioDecoder::configure(&mut decoder, config)?;
    Ok(decoder)
}

/// The Opus identification header (RFC 7845 §5.1) for `channels` at
/// `rate`, mapping family 0, no pre-skip (a live stream starts where it
/// is joined).
fn opus_head(channels: u16, rate: u32) -> Vec<u8> {
    let mut head = b"OpusHead".to_vec();
    head.push(1);
    head.push(u8::try_from(channels).unwrap_or(2));
    head.extend_from_slice(&0u16.to_le_bytes());
    head.extend_from_slice(&rate.to_le_bytes());
    head.extend_from_slice(&0i16.to_le_bytes());
    head.push(0);
    head
}

impl MediaCodecAudioDecoder {
    fn start(&self, shape: Option<Shape>) -> CodecResult<Open> {
        let mime = mime(self.kind).ok_or_else(|| CodecError::BackendCannotHandle {
            backend: BackendId::MediaCodec,
            target: ferricast_core::CodecTarget::decode(self.kind),
        })?;
        let mut format = MediaFormat::new();
        format.set_str(key::MIME, mime);
        match (self.kind, shape) {
            (AudioCodec::Aac, Some(shape)) => {
                let asc = AacLc::new(shape.sample_rate, shape.channels)
                    .ok_or_else(|| CodecError::UnsupportedConfig {
                        backend: BackendId::MediaCodec,
                        what: format!(
                            "AAC at {} Hz / {} channels",
                            shape.sample_rate, shape.channels
                        ),
                    })?
                    .audio_specific_config();
                format.set_i32(
                    key::SAMPLE_RATE,
                    i32::try_from(shape.sample_rate).unwrap_or(48_000),
                );
                format.set_i32(key::CHANNEL_COUNT, i32::from(shape.channels));
                format.set_i32(key::IS_ADTS, 0);
                format.set_buffer(key::CSD_0, &asc);
            }
            (AudioCodec::Opus, _) => {
                let channels = shape.map_or(2, |shape| shape.channels.max(1));
                let rate = shape.map_or(OPUS_RATE, |shape| shape.sample_rate);
                format.set_i32(key::SAMPLE_RATE, i32::try_from(OPUS_RATE).unwrap_or(48_000));
                format.set_i32(key::CHANNEL_COUNT, i32::from(channels));
                format.set_buffer(key::CSD_0, &opus_head(channels, rate));
                format.set_buffer(key::CSD_1, &0u64.to_le_bytes());
                format.set_buffer(key::CSD_2, &OPUS_SEEK_PREROLL_NS.to_le_bytes());
            }
            _ => {
                return Err(fail(
                    Operation::Open,
                    "AAC decode needs the stream shape before the first frame",
                ));
            }
        }
        format.set_i32(key::MAX_INPUT_SIZE, 8192);
        let codec = MediaCodec::create(mime, self.pick)?;
        codec
            .start_decoder(&format)
            .map_err(|error| CodecError::UnsupportedConfig {
                backend: BackendId::MediaCodec,
                what: format!("{} decode on {} ({error})", self.kind, codec.label()),
            })?;
        info!(codec = %codec.label(), "MediaCodec audio decoder opened");
        Ok(Open {
            codec,
            shape,
            output: None,
        })
    }

    /// Files one output buffer as PCM.
    fn accept(&mut self, output: Output) -> CodecResult<()> {
        if output.is(FLAG_CODEC_CONFIG) || output.data.is_empty() {
            return Ok(());
        }
        let open = self
            .open
            .as_mut()
            .ok_or_else(|| fail(Operation::Decode, "output from a closed codec"))?;
        let (sample_rate, channels) = match open.output {
            Some(shape) => shape,
            None => {
                let shape = output_shape(&open.codec)?;
                open.output = Some(shape);
                shape
            }
        };
        let pcm = output
            .data
            .chunks_exact(2)
            .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        self.ready.push_back(DecodedAudio {
            pcm,
            sample_rate,
            channels,
            timestamp_us: output.pts_us,
        });
        Ok(())
    }

    fn drain(&mut self, timeout: Duration) -> CodecResult<bool> {
        loop {
            let Some(open) = self.open.as_mut() else {
                return Ok(false);
            };
            let mut changed = false;
            let output = open
                .codec
                .next_output(Operation::Decode, timeout, || changed = true)?;
            if changed {
                open.output = None;
            }
            let Some(output) = output else {
                return Ok(false);
            };
            let ended = output.is(FLAG_END_OF_STREAM);
            self.accept(output)?;
            if ended {
                return Ok(true);
            }
        }
    }

    fn queue(&mut self, payload: &[u8], pts_us: u64) -> CodecResult<()> {
        let open = self
            .open
            .as_mut()
            .ok_or_else(|| fail(Operation::Decode, "the decoder is not configured"))?;
        // PCM drained while waiting for an input buffer keeps the output
        // shape current when it was dequeued: a format change part-way
        // through the batch only relabels the buffers after it.
        let mut early = Vec::new();
        let mut shape = open.output;
        open.codec.queue(
            Operation::Decode,
            pts_us,
            0,
            |buffer| copy_into(buffer, payload, Operation::Decode),
            |codec| loop {
                let mut changed = false;
                let output =
                    codec.next_output(Operation::Decode, Duration::ZERO, || changed = true)?;
                if changed {
                    shape = Some(output_shape(codec)?);
                }
                let Some(output) = output else {
                    return Ok(());
                };
                early.push((output, shape));
            },
        )?;
        for (output, snapshot) in early {
            if let Some(open) = self.open.as_mut() {
                open.output = snapshot;
            }
            self.accept(output)?;
        }
        if let Some(open) = self.open.as_mut() {
            open.output = shape;
        }
        Ok(())
    }

    /// Signals the end of the stream to the open codec and files every
    /// frame it still holds; the deadline only guards against a codec
    /// that never signals the end.
    fn drain_to_end(&mut self) -> CodecResult<()> {
        let Some(open) = &self.open else {
            return Ok(());
        };
        open.codec.queue(
            Operation::Flush,
            0,
            FLAG_END_OF_STREAM,
            |_| Ok(0),
            |_| Ok(()),
        )?;
        let deadline = Instant::now() + FLUSH_DEADLINE;
        while !self.drain(WAIT)? {
            if Instant::now() >= deadline {
                warn!("MediaCodec audio decoder did not signal the end of the stream within 1 s");
                break;
            }
        }
        Ok(())
    }

    fn decode_aac(&mut self, frame: &AudioFrame) -> CodecResult<()> {
        let mut pts = frame.timestamp_us;
        for unit in adts::units(&frame.data) {
            let unit = unit.map_err(|error| fail(Operation::Decode, format!("AAC: {error}")))?;
            let shape = unit.shape.or(self.configured).ok_or_else(|| {
                fail(
                    Operation::Decode,
                    "raw AAC with no sample rate / channels configured",
                )
            })?;
            if self
                .open
                .as_ref()
                .is_none_or(|open| open.shape != Some(shape))
            {
                // Finish the previous instance (its last frames are good
                // audio), then drop it before opening the next.
                self.drain_to_end()?;
                self.open = None;
                self.open = Some(self.start(Some(shape))?);
            }
            self.queue(unit.payload, pts)?;
            pts += AAC_FRAME_SAMPLES * 1_000_000 / u64::from(shape.sample_rate.max(1));
        }
        Ok(())
    }
}

impl AudioDecoder for MediaCodecAudioDecoder {
    // The facade reports the job's codec; a session decodes the one its
    // configuration names.
    const CODEC: AudioCodec = AudioCodec::Aac;

    fn configure(&mut self, config: &AudioDecoderConfig) -> CodecResult<()> {
        self.kind = config.codec;
        self.configured = (config.sample_rate > 0 && config.channels > 0).then_some(Shape {
            sample_rate: config.sample_rate,
            channels: config.channels,
        });
        self.ready.clear();
        self.open = None;
        match self.kind {
            // Opened on the first frame, once ADTS settles the shape,
            // unless the stream metadata already gave it.
            AudioCodec::Aac => {
                if self.configured.is_some() {
                    self.open = Some(self.start(self.configured)?);
                }
            }
            AudioCodec::Opus => self.open = Some(self.start(self.configured)?),
            AudioCodec::AacEld | AudioCodec::Pcm | AudioCodec::Alac => {
                return Err(CodecError::BackendCannotHandle {
                    backend: BackendId::MediaCodec,
                    target: ferricast_core::CodecTarget::decode(self.kind),
                });
            }
        }
        Ok(())
    }

    fn decode(&mut self, frame: AudioFrame) -> CodecResult<Option<DecodedAudio>> {
        match self.kind {
            AudioCodec::Aac => self.decode_aac(&frame)?,
            _ => self.queue(&frame.data, frame.timestamp_us)?,
        }
        self.drain(Duration::ZERO)?;
        Ok(self.ready.pop_front())
    }

    fn flush(&mut self) -> CodecResult<Vec<DecodedAudio>> {
        if let Some(open) = &self.open {
            open.codec.queue(
                Operation::Flush,
                0,
                FLAG_END_OF_STREAM,
                |_| Ok(0),
                |_| Ok(()),
            )?;
            for _ in 0..25 {
                if self.drain(WAIT)? {
                    break;
                }
            }
            if let Some(open) = &self.open {
                open.codec
                    .flush()
                    .map_err(super::codec::media(Operation::Flush, "AMediaCodec_flush"))?;
            }
        }
        Ok(self.ready.drain(..).collect())
    }
}

/// The PCM shape `codec`'s current output format describes.
fn output_shape(codec: &MediaCodec) -> CodecResult<(u32, u16)> {
    let format = codec.output_format();
    let rate = format
        .i32(key::SAMPLE_RATE)
        .and_then(|r| u32::try_from(r).ok())
        .unwrap_or(OPUS_RATE);
    let channels = format
        .i32(key::CHANNEL_COUNT)
        .and_then(|c| u16::try_from(c).ok())
        .unwrap_or(2);
    // PCM 16-bit is the default; anything else is unreadable here.
    if let Some(encoding) = format.i32(key::PCM_ENCODING)
        && encoding != 2
    {
        return Err(fail(
            Operation::Decode,
            format!("decoder outputs PCM encoding {encoding}, not 16-bit"),
        ));
    }
    Ok((rate, channels))
}
