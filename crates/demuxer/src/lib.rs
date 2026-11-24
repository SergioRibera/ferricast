//! MP4/MOV (plain and fragmented), Matroska/WebM and audio-file demuxers.
//!
//! [`Demuxer`] reads from any [`MediaSource`] (a file, an in-memory
//! buffer, an HTTP range reader) and hands out the packets of the first
//! supported video track and, optionally, the first supported audio
//! track (AAC or Opus), interleaved in decode order, with seeking to the
//! keyframe at or before a target time. Audio-only files are read too:
//! raw ADTS AAC natively, MP3, FLAC, Ogg (Vorbis, Opus) and WAV through
//! symphonia. [`demux_segment`] reads fragmented MP4 media segments (HLS
//! `EXT-X-MAP`, CMAF).

mod adts;
mod audio;
#[cfg(feature = "audio-files")]
mod audio_file;
mod mp4;
mod nal;

#[cfg(feature = "matroska")]
mod matroska;
#[cfg(feature = "http")]
mod puller;

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use ferricast_core::{AudioFrame, Codec, ColourDescription, EncodedFrame, FerricastError, Result};

pub use adts::AdtsDemuxer;
pub use audio::AudioStream;
#[cfg(feature = "audio-files")]
pub use audio_file::AudioFileDemuxer;
pub use mp4::Mp4Demuxer;
#[cfg(feature = "http")]
pub use puller::ProgressivePuller;

#[cfg(feature = "matroska")]
pub use matroska::MatroskaDemuxer;

/// Anything a demuxer can read from.
pub trait MediaSource: Read + Seek + Send {}

impl<T: Read + Seek + Send> MediaSource for T {}

pub(crate) type BoxedSource = Box<dyn MediaSource>;

#[derive(Debug, Clone)]
pub struct VideoStream {
    pub codec: Codec,
    pub width: u32,
    pub height: u32,
    /// The colour signalling the container carries for the track.
    pub colour: Option<ColourDescription>,
    pub duration_us: Option<u64>,
    /// Timestamp of the first frame meant to be shown. Frames before it are
    /// pre-roll (samples ahead of an MP4 edit list's start): they must be
    /// decoded, because later frames reference them, but not displayed.
    /// Zero when the stream has no pre-roll.
    pub presentation_start_us: u64,
}

/// One demuxed packet.
#[derive(Debug, Clone)]
pub enum Packet {
    Video(EncodedFrame),
    /// AAC packets come ADTS-framed; Opus packets as they are stored.
    Audio(AudioFrame),
}

/// Which tracks a demuxer hands out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrackSelection {
    /// Video only; a file without a supported video track is an error.
    VideoOnly,
    /// Video and audio; either may be missing, but not both.
    All,
}

pub enum Demuxer {
    Mp4(Box<Mp4Demuxer>),
    #[cfg(feature = "matroska")]
    Matroska(Box<MatroskaDemuxer>),
    /// A raw ADTS AAC file.
    Adts(Box<AdtsDemuxer>),
    /// An MP3, FLAC, Ogg or WAV file.
    #[cfg(feature = "audio-files")]
    AudioFile(Box<AudioFileDemuxer>),
}

impl Demuxer {
    /// Opens the video track of a local file. Audio tracks are ignored.
    ///
    /// # Errors
    ///
    /// Fails when the file cannot be read, its container is not
    /// recognised, or it has no supported video track.
    pub fn open(path: &Path) -> Result<Self> {
        let file = BufReader::new(File::open(path)?);
        Self::open_source(Box::new(file), TrackSelection::VideoOnly).map_err(|e| match e {
            FerricastError::Demux(msg) if msg == UNRECOGNISED => {
                FerricastError::Demux(format!("unrecognised container in {}", path.display()))
            }
            other => other,
        })
    }

    /// Opens the video and audio tracks of any seekable source.
    ///
    /// # Errors
    ///
    /// Fails when the source cannot be read, its container is not
    /// recognised, or it has neither a supported
    /// video nor a supported audio track.
    pub fn from_reader(source: impl MediaSource + 'static) -> Result<Self> {
        Self::open_source(Box::new(source), TrackSelection::All)
    }

    fn open_source(mut source: BoxedSource, selection: TrackSelection) -> Result<Self> {
        let mut head = [0; 10];
        source.seek(SeekFrom::Start(0))?;
        let mut filled = 0;
        while filled < head.len() {
            match source.read(&mut head[filled..])? {
                0 => break,
                n => filled += n,
            }
        }
        source.seek(SeekFrom::Start(0))?;
        if mp4::is_mp4(&head) {
            return Ok(Self::Mp4(Box::new(Mp4Demuxer::open(source, selection)?)));
        }
        #[cfg(feature = "matroska")]
        if matroska::is_matroska(&head) {
            return Ok(Self::Matroska(Box::new(MatroskaDemuxer::open(
                source, selection,
            )?)));
        }
        if selection == TrackSelection::All {
            return Self::open_audio_file(source, &head);
        }
        Err(FerricastError::Demux(UNRECOGNISED.into()))
    }

    /// Opens audio-only files: ADTS AAC natively, the rest through
    /// symphonia.
    fn open_audio_file(mut source: BoxedSource, head: &[u8; 10]) -> Result<Self> {
        let start = adts::id3v2_len(head).unwrap_or(0);
        let mut sync = [0; 2];
        source.seek(SeekFrom::Start(start))?;
        let adts = source.read_exact(&mut sync).is_ok() && adts::is_adts(&sync);
        source.seek(SeekFrom::Start(0))?;
        if adts {
            return Ok(Self::Adts(Box::new(AdtsDemuxer::open(source, start)?)));
        }
        #[cfg(feature = "audio-files")]
        return AudioFileDemuxer::open(source)
            .map(|d| Self::AudioFile(Box::new(d)))
            .map_err(|_| FerricastError::Demux(UNRECOGNISED.into()));
        #[cfg(not(feature = "audio-files"))]
        Err(FerricastError::Demux(UNRECOGNISED.into()))
    }

    #[must_use]
    pub fn video(&self) -> Option<&VideoStream> {
        match self {
            Demuxer::Mp4(d) => d.video(),
            #[cfg(feature = "matroska")]
            Demuxer::Matroska(d) => d.video(),
            Demuxer::Adts(_) => None,
            #[cfg(feature = "audio-files")]
            Demuxer::AudioFile(_) => None,
        }
    }

    #[must_use]
    pub fn audio(&self) -> Option<&AudioStream> {
        match self {
            Demuxer::Mp4(d) => d.audio(),
            #[cfg(feature = "matroska")]
            Demuxer::Matroska(d) => d.audio(),
            Demuxer::Adts(d) => Some(d.audio()),
            #[cfg(feature = "audio-files")]
            Demuxer::AudioFile(d) => Some(d.audio()),
        }
    }

    /// Duration of the longest track, when the container states it.
    #[must_use]
    pub fn duration_us(&self) -> Option<u64> {
        match self {
            Demuxer::Mp4(d) => d.duration_us(),
            #[cfg(feature = "matroska")]
            Demuxer::Matroska(d) => d.duration_us(),
            Demuxer::Adts(d) => d.duration_us(),
            #[cfg(feature = "audio-files")]
            Demuxer::AudioFile(d) => d.duration_us(),
        }
    }

    /// The next packet of either track, in decode order.
    ///
    /// # Errors
    ///
    /// Fails on read errors and malformed samples.
    pub fn next_packet(&mut self) -> Result<Option<Packet>> {
        match self {
            Demuxer::Mp4(d) => d.next_packet(),
            #[cfg(feature = "matroska")]
            Demuxer::Matroska(d) => d.next_packet(),
            Demuxer::Adts(d) => d.next_packet(),
            #[cfg(feature = "audio-files")]
            Demuxer::AudioFile(d) => d.next_packet(),
        }
    }

    /// The next video frame, skipping audio packets.
    ///
    /// # Errors
    ///
    /// Fails on read errors and malformed samples.
    pub fn next_frame(&mut self) -> Result<Option<EncodedFrame>> {
        loop {
            match self.next_packet()? {
                Some(Packet::Video(frame)) => return Ok(Some(frame)),
                Some(Packet::Audio(_)) => {}
                None => return Ok(None),
            }
        }
    }

    /// Moves to the last video keyframe presented at or before
    /// `timestamp_us` (for audio-only media, to `timestamp_us` itself);
    /// audio resumes from the same point. Returns the timestamp it
    /// landed on.
    ///
    /// # Errors
    ///
    /// Fails on read errors while searching for the keyframe.
    pub fn seek(&mut self, timestamp_us: u64) -> Result<u64> {
        match self {
            Demuxer::Mp4(d) => d.seek(timestamp_us),
            #[cfg(feature = "matroska")]
            Demuxer::Matroska(d) => d.seek(timestamp_us),
            Demuxer::Adts(d) => d.seek(timestamp_us),
            #[cfg(feature = "audio-files")]
            Demuxer::AudioFile(d) => d.seek(timestamp_us),
        }
    }

    /// Starts over from the first packet.
    ///
    /// # Errors
    ///
    /// Fails when the source cannot seek back.
    pub fn rewind(&mut self) -> Result<()> {
        match self {
            Demuxer::Mp4(d) => d.rewind(),
            #[cfg(feature = "matroska")]
            Demuxer::Matroska(d) => d.rewind(),
            Demuxer::Adts(d) => {
                d.rewind();
                Ok(())
            }
            #[cfg(feature = "audio-files")]
            Demuxer::AudioFile(d) => d.rewind(),
        }
    }
}

/// The tracks and packets of one fragmented MP4 media segment.
#[derive(Debug, Clone)]
pub struct Segment {
    pub video: Option<VideoStream>,
    pub audio: Option<AudioStream>,
    /// In decode order. Timestamps share one timeline across the
    /// segments of a stream: the decode times the segments carry, from
    /// the edit list start.
    pub packets: Vec<Packet>,
}

/// Demuxes one fragmented MP4 media segment (an HLS `EXT-X-MAP` or CMAF
/// segment) against the initialization segment holding its `moov`.
///
/// # Errors
///
/// Fails when the initialization segment has no supported track or the
/// media segment is malformed.
pub fn demux_segment(init: &[u8], segment: &[u8]) -> Result<Segment> {
    let mut file = Vec::with_capacity(init.len() + segment.len());
    file.extend_from_slice(init);
    file.extend_from_slice(segment);
    let mut demuxer = Mp4Demuxer::open_with(
        Box::new(std::io::Cursor::new(file)),
        TrackSelection::All,
        mp4::Origin::EditList,
    )?;
    let mut packets = Vec::new();
    while let Some(packet) = demuxer.next_packet()? {
        packets.push(packet);
    }
    Ok(Segment {
        video: demuxer.video().cloned(),
        audio: demuxer.audio().cloned(),
        packets,
    })
}

const UNRECOGNISED: &str = "unrecognised container";

pub(crate) fn micros(ticks: i64, timescale: u32) -> u64 {
    (i128::from(ticks.max(0)) * 1_000_000 / i128::from(timescale.max(1))) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_convert_to_microseconds() {
        assert_eq!(micros(512, 15360), 33_333);
        assert_eq!(micros(90_000, 90_000), 1_000_000);
    }

    #[test]
    fn negative_ticks_clamp_to_zero() {
        assert_eq!(micros(-5, 1000), 0);
    }
}
