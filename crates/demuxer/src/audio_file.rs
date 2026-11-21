//! Audio-only files read through symphonia: MP3 (ID3v2 tags, Xing/VBR
//! headers), FLAC, Ogg (Vorbis and Opus) and WAV (16/24-bit integer and
//! 32-bit float PCM).
//!
//! Opus packets are handed out as they are, for the Opus decoder. The
//! other codecs have no decoder in the receiver pipeline and need their
//! container's setup data (Vorbis and FLAC headers) to decode, so they
//! are decoded here, on the demuxer thread, and handed out as
//! [`AudioCodec::Pcm`]: interleaved signed 16-bit little-endian samples.

use std::io::{self, Read, Seek, SeekFrom};
use std::sync::{Mutex, PoisonError};

use bytes::Bytes;
use ferricast_core::frame::SourceAudioFormat;
use ferricast_core::{AudioCodec, AudioFrame, FerricastError, Result};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, CODEC_TYPE_OPUS, Decoder, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::units::{Time, TimeBase};

use crate::audio::OPUS_RATE;
use crate::{AudioStream, BoxedSource, Packet};

/// How far before a seek target decoding restarts. Vorbis and Opus
/// decoders overlap each packet with the one before it, so the first
/// packet after a reset yields no (or unusable) audio; starting this much
/// earlier keeps the target itself covered. Consumers drop what precedes
/// the target.
const SEEK_PREROLL_US: u64 = 100_000;
/// Undecodable packets tolerated in a row before the file is given up.
const MAX_DECODE_ERRORS: u32 = 32;

pub struct AudioFileDemuxer {
    format: Box<dyn FormatReader>,
    track_id: u32,
    time_base: TimeBase,
    stream: AudioStream,
    output: Output,
}

/// What becomes of the track's packets.
enum Output {
    /// Opus: passed through.
    Opus,
    /// Decoded to 16-bit PCM.
    Pcm {
        decoder: Box<dyn Decoder>,
        samples: Option<SampleBuffer<i16>>,
    },
}

/// Adapts a demuxer source to symphonia, which wants `Sync` sources.
struct SyncSource {
    inner: Mutex<BoxedSource>,
    len: Option<u64>,
}

impl SyncSource {
    fn lock(&self) -> std::sync::MutexGuard<'_, BoxedSource> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Read for SyncSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.lock().read(buf)
    }
}

impl Seek for SyncSource {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.lock().seek(pos)
    }
}

impl symphonia::core::io::MediaSource for SyncSource {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        self.len
    }
}

impl AudioFileDemuxer {
    pub(crate) fn open(mut source: BoxedSource) -> Result<Self> {
        let len = source.seek(SeekFrom::End(0)).ok();
        source.seek(SeekFrom::Start(0))?;
        let stream = MediaSourceStream::new(
            Box::new(SyncSource {
                inner: Mutex::new(source),
                len,
            }),
            MediaSourceStreamOptions::default(),
        );
        let options = FormatOptions {
            enable_gapless: true,
            ..FormatOptions::default()
        };
        let probed = symphonia::default::get_probe()
            .format(&Hint::new(), stream, &options, &MetadataOptions::default())
            .map_err(|e| FerricastError::Demux(format!("unrecognised audio file: {e}")))?;
        let format = probed.format;
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| FerricastError::Demux("no audio track".into()))?;
        let params = &track.codec_params;
        let track_id = track.id;
        let channels = params
            .channels
            .map(|c| c.count())
            .and_then(|c| u16::try_from(c).ok())
            .filter(|&c| c > 0)
            .ok_or_else(|| FerricastError::Demux("audio track without a channel count".into()))?;
        let sample_rate = params
            .sample_rate
            .ok_or_else(|| FerricastError::Demux("audio track without a sample rate".into()))?;
        let time_base = params
            .time_base
            .unwrap_or_else(|| TimeBase::new(1, sample_rate));
        let duration_us = params
            .n_frames
            .map(|frames| micros(time_base, frames.saturating_add(params.start_ts)));
        let (codec, output_rate, output, source) = if params.codec == CODEC_TYPE_OPUS {
            (AudioCodec::Opus, OPUS_RATE, Output::Opus, None)
        } else {
            let codecs = symphonia::default::get_codecs();
            let decoder = codecs
                .make(params, &DecoderOptions::default())
                .map_err(|e| FerricastError::Demux(format!("unsupported audio codec: {e}")))?;
            let source = codecs
                .get_codec(params.codec)
                .and_then(|descriptor| source_format(descriptor.short_name));
            (
                AudioCodec::Pcm,
                sample_rate,
                Output::Pcm {
                    decoder,
                    samples: None,
                },
                source,
            )
        };
        Ok(Self {
            format,
            track_id,
            time_base,
            stream: AudioStream {
                codec,
                sample_rate: output_rate,
                channels,
                duration_us,
                source,
            },
            output,
        })
    }

    pub(crate) fn audio(&self) -> &AudioStream {
        &self.stream
    }

    pub(crate) fn duration_us(&self) -> Option<u64> {
        self.stream.duration_us
    }

    pub(crate) fn next_packet(&mut self) -> Result<Option<Packet>> {
        let mut errors = 0;
        loop {
            let packet = match self.format.next_packet() {
                Ok(packet) => packet,
                Err(SymphoniaError::IoError(e)) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    return Ok(None);
                }
                Err(SymphoniaError::ResetRequired) => return Ok(None),
                Err(e) => return Err(FerricastError::Demux(format!("audio file: {e}"))),
            };
            if packet.track_id() != self.track_id {
                continue;
            }
            let timestamp_us = micros(self.time_base, packet.ts());
            let data = match &mut self.output {
                Output::Opus => Bytes::copy_from_slice(packet.buf()),
                Output::Pcm { decoder, samples } => match decoder.decode(&packet) {
                    Ok(decoded) => {
                        if decoded.frames() == 0 {
                            continue;
                        }
                        let spec = *decoded.spec();
                        let capacity = decoded.capacity() as u64;
                        let buffer = match samples {
                            Some(buffer)
                                if buffer.capacity()
                                    >= decoded.capacity() * spec.channels.count() =>
                            {
                                buffer
                            }
                            _ => samples.insert(SampleBuffer::new(capacity, spec)),
                        };
                        buffer.copy_interleaved_ref(decoded);
                        buffer
                            .samples()
                            .iter()
                            .flat_map(|s| s.to_le_bytes())
                            .collect()
                    }
                    Err(SymphoniaError::DecodeError(e)) => {
                        errors += 1;
                        if errors > MAX_DECODE_ERRORS {
                            return Err(FerricastError::Demux(format!("audio decode: {e}")));
                        }
                        tracing::debug!(error = e, "audio file: skipping an undecodable packet");
                        continue;
                    }
                    Err(e) => return Err(FerricastError::Demux(format!("audio decode: {e}"))),
                },
            };
            return Ok(Some(Packet::Audio(AudioFrame {
                codec: self.stream.codec,
                data,
                timestamp_us,
                sample_rate: self.stream.sample_rate,
                channels: self.stream.channels,
            })));
        }
    }

    /// Moves to the packet holding `timestamp_us` (sample-accurate
    /// where the container allows) and returns where it landed.
    pub(crate) fn seek(&mut self, timestamp_us: u64) -> Result<u64> {
        let timestamp_us = timestamp_us.saturating_sub(SEEK_PREROLL_US);
        let time = Time::new(
            timestamp_us / 1_000_000,
            (timestamp_us % 1_000_000) as f64 / 1e6,
        );
        let seeked = self
            .format
            .seek(
                SeekMode::Accurate,
                SeekTo::Time {
                    time,
                    track_id: Some(self.track_id),
                },
            )
            .map_err(|e| FerricastError::Demux(format!("audio seek: {e}")))?;
        if let Output::Pcm { decoder, .. } = &mut self.output {
            decoder.reset();
        }
        Ok(micros(self.time_base, seeked.actual_ts))
    }

    pub(crate) fn rewind(&mut self) -> Result<()> {
        self.seek(0).map(drop)
    }
}

fn micros(time_base: TimeBase, ts: u64) -> u64 {
    let time = time_base.calc_time(ts);
    time.seconds
        .saturating_mul(1_000_000)
        .saturating_add((time.frac * 1e6) as u64)
}

/// The stored format behind a symphonia decoder's short name. Only the
/// WAV reader yields raw PCM among the formats enabled here.
fn source_format(short_name: &str) -> Option<SourceAudioFormat> {
    match short_name {
        "mp3" => Some(SourceAudioFormat::Mp3),
        "flac" => Some(SourceAudioFormat::Flac),
        "vorbis" => Some(SourceAudioFormat::Vorbis),
        name if name.starts_with("pcm_") => Some(SourceAudioFormat::Wav),
        _ => None,
    }
}
