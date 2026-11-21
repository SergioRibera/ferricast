//! Raw ADTS AAC files (`.aac`), optionally behind an ID3v2 tag.
//!
//! The frames are handed out as they are: the AAC decoder reads ADTS.
//! There is no index, so seeking walks the frame headers (seven bytes
//! each) from the nearest point already visited; positions visited are
//! remembered every [`INDEX_EVERY`] frames. The duration is estimated
//! from the average size of the first frames.

use std::io::SeekFrom;

use bytes::Bytes;
use ferricast_core::{AudioCodec, AudioFrame, FerricastError, Result};

use crate::{AudioStream, BoxedSource, Packet};

const HEADER_LEN: usize = 7;
/// Frame headers read at open to estimate the bitrate.
const PROBE_FRAMES: usize = 32;
/// Bytes searched for the next sync word after garbage.
const RESYNC_WINDOW: usize = 64 << 10;
/// A visited frame position is remembered every this many frames.
const INDEX_EVERY: u64 = 32;
/// Remembered positions at most (about 37 hours at 48 kHz).
const MAX_INDEX: usize = 1 << 16;
const RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

/// Whether `head` starts with an ADTS sync word (MPEG layer 0).
pub(crate) fn is_adts(head: &[u8]) -> bool {
    matches!(head, [0xFF, second, ..] if second & 0xF6 == 0xF0)
}

/// Length of the ID3v2 tag `head` starts with, if any.
pub(crate) fn id3v2_len(head: &[u8; 10]) -> Option<u64> {
    if &head[..3] != b"ID3" || head[6..].iter().any(|b| b & 0x80 != 0) {
        return None;
    }
    let size = head[6..]
        .iter()
        .fold(0u64, |acc, &b| (acc << 7) | u64::from(b));
    let footer = if head[5] & 0x10 != 0 { 10 } else { 0 };
    Some(10 + size + footer)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Header {
    sample_rate: u32,
    channels: u8,
    frame_len: usize,
    /// Samples per channel in the frame.
    samples: u64,
}

impl Header {
    fn parse(data: &[u8; HEADER_LEN]) -> Option<Self> {
        if !is_adts(data) {
            return None;
        }
        let rate = *RATES.get(usize::from((data[2] >> 2) & 0xF))?;
        let channels = ((data[2] & 1) << 2) | (data[3] >> 6);
        let frame_len = (usize::from(data[3] & 3) << 11)
            | (usize::from(data[4]) << 3)
            | usize::from(data[5] >> 5);
        let blocks = u64::from(data[6] & 3) + 1;
        (frame_len > HEADER_LEN && channels > 0).then_some(Self {
            sample_rate: rate,
            channels,
            frame_len,
            samples: 1024 * blocks,
        })
    }
}

pub struct AdtsDemuxer {
    source: BoxedSource,
    /// First frame (after any ID3v2 tag).
    start: u64,
    len: u64,
    stream: AudioStream,
    /// Offset of the next frame.
    offset: u64,
    /// Samples per channel before the next frame.
    position: u64,
    /// `(samples before, offset)` of frames visited, ascending.
    index: Vec<(u64, u64)>,
    frames_read: u64,
}

impl AdtsDemuxer {
    pub(crate) fn open(mut source: BoxedSource, start: u64) -> Result<Self> {
        let len = source.seek(SeekFrom::End(0))?;
        let mut offset = start;
        let mut first = None;
        let mut probed = 0usize;
        let mut probed_bytes = 0u64;
        let mut probed_samples = 0u64;
        while probed < PROBE_FRAMES {
            let Some(header) = read_header(&mut source, offset, len)? else {
                break;
            };
            first.get_or_insert(header);
            probed += 1;
            probed_bytes += header.frame_len as u64;
            probed_samples += header.samples;
            offset += header.frame_len as u64;
        }
        let first: Header =
            first.ok_or_else(|| FerricastError::Demux("no ADTS frame at the start".into()))?;
        let duration_us = (probed_bytes > 0).then(|| {
            let bytes = u128::from(len - start);
            let samples = bytes * u128::from(probed_samples) / u128::from(probed_bytes);
            (samples * 1_000_000 / u128::from(first.sample_rate)) as u64
        });
        Ok(Self {
            source,
            start,
            len,
            stream: AudioStream {
                codec: AudioCodec::Aac,
                sample_rate: first.sample_rate,
                channels: u16::from(first.channels),
                duration_us,
            },
            offset: start,
            position: 0,
            index: vec![(0, start)],
            frames_read: 0,
        })
    }

    pub(crate) fn audio(&self) -> &AudioStream {
        &self.stream
    }

    pub(crate) fn duration_us(&self) -> Option<u64> {
        self.stream.duration_us
    }

    fn micros(&self, samples: u64) -> u64 {
        samples * 1_000_000 / u64::from(self.stream.sample_rate.max(1))
    }

    /// The header of the frame at `self.offset`, resynchronising past
    /// garbage; `None` at the end.
    fn header(&mut self) -> Result<Option<Header>> {
        if let Some(header) = read_header(&mut self.source, self.offset, self.len)? {
            return Ok(Some(header));
        }
        // Not a frame: look for the next sync word followed by a valid
        // header.
        let window = (self.len.saturating_sub(self.offset)).min(RESYNC_WINDOW as u64) as usize;
        if window < HEADER_LEN {
            return Ok(None);
        }
        let mut buf = vec![0; window];
        self.source.seek(SeekFrom::Start(self.offset))?;
        self.source.read_exact(&mut buf)?;
        for at in 1..window - HEADER_LEN {
            let Some(head) = buf[at..].first_chunk::<HEADER_LEN>() else {
                break;
            };
            if let Some(header) = Header::parse(head) {
                self.offset += at as u64;
                return Ok(Some(header));
            }
        }
        Ok(None)
    }

    fn remember(&mut self) {
        let last = self.index.last().map_or(0, |&(samples, _)| samples);
        if self.frames_read.is_multiple_of(INDEX_EVERY)
            && self.position > last
            && self.index.len() < MAX_INDEX
        {
            self.index.push((self.position, self.offset));
        }
    }

    pub(crate) fn next_packet(&mut self) -> Result<Option<Packet>> {
        let Some(header) = self.header()? else {
            return Ok(None);
        };
        if self.offset + header.frame_len as u64 > self.len {
            return Ok(None);
        }
        self.remember();
        let mut data = vec![0; header.frame_len];
        self.source.seek(SeekFrom::Start(self.offset))?;
        self.source.read_exact(&mut data)?;
        let frame = AudioFrame {
            codec: AudioCodec::Aac,
            data: Bytes::from(data),
            timestamp_us: self.micros(self.position),
            sample_rate: header.sample_rate,
            channels: u16::from(header.channels),
        };
        self.offset += header.frame_len as u64;
        self.position += header.samples;
        self.frames_read += 1;
        Ok(Some(Packet::Audio(frame)))
    }

    /// Moves to the frame playing at `timestamp_us` and returns its
    /// start.
    pub(crate) fn seek(&mut self, timestamp_us: u64) -> Result<u64> {
        let target = timestamp_us * u64::from(self.stream.sample_rate) / 1_000_000;
        let at = self
            .index
            .partition_point(|&(samples, _)| samples <= target);
        let (samples, offset) = self.index[at.saturating_sub(1)];
        self.position = samples;
        self.offset = offset;
        self.frames_read = 0;
        while let Some(header) = self.header()? {
            if self.position + header.samples > target {
                break;
            }
            self.remember();
            self.offset += header.frame_len as u64;
            self.position += header.samples;
            self.frames_read += 1;
        }
        Ok(self.micros(self.position))
    }

    pub(crate) fn rewind(&mut self) {
        self.offset = self.start;
        self.position = 0;
        self.frames_read = 0;
    }
}

fn read_header(source: &mut BoxedSource, offset: u64, len: u64) -> Result<Option<Header>> {
    if offset + HEADER_LEN as u64 > len {
        return Ok(None);
    }
    let mut head = [0; HEADER_LEN];
    source.seek(SeekFrom::Start(offset))?;
    source.read_exact(&mut head)?;
    Ok(Header::parse(&head))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headers_describe_their_frame() {
        // LC, 44.1 kHz (index 4), stereo, 371 bytes, one raw block.
        let len = 371usize;
        let head = [
            0xFF,
            0xF1,
            (1 << 6) | (4 << 2),
            (2 << 6) | ((len >> 11) & 3) as u8,
            (len >> 3) as u8,
            (((len & 7) as u8) << 5) | 0x1F,
            0xFC,
        ];
        let header = Header::parse(&head).unwrap();
        assert_eq!(header.sample_rate, 44_100);
        assert_eq!(header.channels, 2);
        assert_eq!(header.frame_len, 371);
        assert_eq!(header.samples, 1024);
    }

    #[test]
    fn mp3_sync_is_not_adts() {
        assert!(is_adts(&[0xFF, 0xF1]));
        assert!(!is_adts(&[0xFF, 0xFB]));
    }

    #[test]
    fn id3v2_tags_are_measured() {
        let head = *b"ID3\x04\x00\x00\x00\x00\x02\x01";
        assert_eq!(id3v2_len(&head), Some(10 + 257));
        assert_eq!(id3v2_len(b"RIFF\0\0\0\0WA"), None);
    }
}
