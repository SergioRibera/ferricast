mod fragment;

use std::io::SeekFrom;

use ferricast_core::{AudioFrame, Codec, EncodedFrame, FerricastError, Result};

use crate::audio::{AacConfig, AudioFormat, AudioStream, OPUS_RATE, opus_channels};
use crate::nal::{Framing, LengthPrefixed};
use crate::{BoxedSource, Color, Packet, TrackSelection, VideoStream, micros};

const VISUAL_ENTRY_HEADER: usize = 78;
/// `SampleEntry` (8 bytes) plus the version 0 `AudioSampleEntry`
/// fields (20 bytes). QuickTime version 1 and 2 entries append 16 and
/// 36 more bytes before their child boxes.
const AUDIO_ENTRY_HEADER: usize = 28;
// Upper bound on the samples taken from one track: about 19 hours at 30 fps.
// Sample counts come straight from the file, so without a cap a few bytes of
// `stsz` could ask for gigabytes of sample entries.
const MAX_SAMPLES: usize = 1 << 21;
/// Largest `moov` box read into memory. Sample tables of a multi-hour
/// movie stay well below this; the cap bounds what a hostile file
/// header can make the demuxer allocate.
const MAX_MOOV: u64 = 256 << 20;
/// Largest sample read into memory.
const MAX_SAMPLE_SIZE: u32 = 64 << 20;
/// Largest `moof` box read into memory.
const MAX_MOOF: u64 = 16 << 20;
/// Largest `sidx` or `mfra` box read into memory.
const MAX_INDEX_BOX: u64 = 16 << 20;
/// Samples queued on one track before fragments stop being read ahead
/// for another track that has none.
const MAX_PENDING: usize = 2048;
/// Levels of a hierarchical `sidx` index followed at most.
const MAX_SIDX_DEPTH: usize = 4;
/// `sidx` boxes read at most while following one index.
const MAX_SIDX_BOXES: usize = 4096;

pub struct Mp4Demuxer {
    source: BoxedSource,
    video: Option<VideoTrack>,
    audio: Option<AudioTrack>,
    duration_us: Option<u64>,
    /// Movie fragments still to read, for fragmented files.
    fragments: Option<Fragments>,
}

struct VideoTrack {
    id: u32,
    stream: VideoStream,
    framing: Framing,
    timescale: u32,
    samples: Vec<Sample>,
    /// Media time (ticks) that maps to timestamp 0.
    origin: i64,
    /// Media time (ticks) at which the presentation starts (edit list).
    shift: i64,
    next: usize,
    /// The next sample is the first one after open or a seek and must
    /// carry the parameter sets whatever its NAL types.
    at_start: bool,
}

struct AudioTrack {
    id: u32,
    stream: AudioStream,
    format: AudioFormat,
    timescale: u32,
    samples: Vec<Sample>,
    /// Media time (ticks) at which the presentation starts (edit list).
    shift: i64,
    /// Timestamp of the presentation start on the shared timeline: the
    /// video track's pre-roll, so both tracks line up.
    offset_us: u64,
    /// First sample that plays: encoder priming entirely before the
    /// edit start (the AAC encoder delay) is skipped, so no two packets
    /// share timestamp 0.
    first: usize,
    next: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct Sample {
    offset: u64,
    size: u32,
    dts: u64,
    duration: u32,
    cts_offset: i32,
    sync: bool,
}

#[derive(Default)]
struct Tables {
    stts: Vec<(u32, u32)>,
    ctts: Vec<(u32, i32)>,
    stsc: Vec<ChunkRun>,
    sizes: Vec<u32>,
    chunk_offsets: Vec<u64>,
    sync: Option<Vec<u32>>,
}

/// One `stsc` entry: chunks from `first_chunk` up to the next run's first
/// chunk each hold `samples_per_chunk` samples described by stsd entry
/// `description_index`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChunkRun {
    first_chunk: u32,
    samples_per_chunk: u32,
    description_index: u32,
}

struct VideoEntry {
    codec: Codec,
    width: u32,
    height: u32,
    framing: Framing,
    color: Option<Color>,
}

struct AudioEntry {
    format: AudioFormat,
    stream: AudioStream,
}

/// A track of either kind, as found in the `moov` box.
enum Parsed {
    Video(Track<VideoEntry>),
    Audio(Track<AudioEntry>),
}

struct Track<E> {
    id: u32,
    entry: E,
    timescale: u32,
    duration: u64,
    tables: Tables,
    shift: i64,
}

impl<E> Track<E> {
    fn duration_us(&self) -> u64 {
        micros(
            i64::try_from(self.duration).unwrap_or(i64::MAX),
            self.timescale,
        )
    }

    fn with<F>(self, entry: F) -> Track<F> {
        Track {
            id: self.id,
            entry,
            timescale: self.timescale,
            duration: self.duration,
            tables: self.tables,
            shift: self.shift,
        }
    }
}

impl Sample {
    fn decode_ticks(&self) -> i64 {
        i64::try_from(self.dts).unwrap_or(i64::MAX)
    }

    fn presentation_ticks(&self) -> i64 {
        self.decode_ticks()
            .saturating_add(i64::from(self.cts_offset))
    }

    fn end(&self) -> Option<u64> {
        self.offset.checked_add(u64::from(self.size))
    }
}

impl Tables {
    /// Sum of every `stts` sample delta, in media ticks.
    fn decode_duration(&self) -> u64 {
        self.stts.iter().fold(0u64, |total, &(count, delta)| {
            total.saturating_add(u64::from(count) * u64::from(delta))
        })
    }
}

impl VideoTrack {
    fn pts_us(&self, sample: &Sample) -> u64 {
        micros(
            sample.presentation_ticks().saturating_sub(self.origin),
            self.timescale,
        )
    }

    fn dts_us(&self, sample: &Sample) -> u64 {
        micros(
            sample.decode_ticks().saturating_sub(self.origin),
            self.timescale,
        )
    }

    fn next_dts_us(&self) -> Option<u64> {
        self.samples.get(self.next).map(|s| self.dts_us(s))
    }

    /// Moves to the last sync sample presented at or before
    /// `timestamp_us` (the first sample when there is none) and returns
    /// its timestamp.
    fn seek(&mut self, timestamp_us: u64) -> u64 {
        let target = self
            .samples
            .iter()
            .rposition(|s| s.sync && self.pts_us(s) <= timestamp_us)
            .unwrap_or(0);
        self.next = target;
        self.at_start = true;
        self.samples.get(target).map_or(0, |s| self.pts_us(s))
    }

    fn frame(&mut self, source: &mut BoxedSource) -> Result<Option<EncodedFrame>> {
        let Some(sample) = self.samples.get(self.next) else {
            return Ok(None);
        };
        let data = read_sample(source, sample)?;
        let (data, idr) = self.framing.annex_b(&data, sample.sync || self.at_start)?;
        let pts = self.pts_us(sample);
        let frame = EncodedFrame {
            codec: self.stream.codec,
            data,
            timestamp_us: pts,
            is_keyframe: sample.sync || idr,
            duration_us: Some(micros(i64::from(sample.duration), self.timescale)),
            pts_dts: (pts, self.dts_us(sample)),
        };
        self.next += 1;
        self.at_start = false;
        Ok(Some(frame))
    }
}

impl AudioTrack {
    fn ts_us(&self, sample: &Sample) -> u64 {
        micros(
            sample.decode_ticks().saturating_sub(self.shift),
            self.timescale,
        )
        .saturating_add(self.offset_us)
    }

    fn next_ts_us(&self) -> Option<u64> {
        self.samples.get(self.next).map(|s| self.ts_us(s))
    }

    /// Decode ticks of the sample timestamped `timestamp_us`: the inverse
    /// of [`Self::ts_us`].
    fn ticks(&self, timestamp_us: u64) -> u64 {
        let us = timestamp_us.saturating_sub(self.offset_us);
        let t = u128::from(us) * u128::from(self.timescale) / 1_000_000;
        i64::try_from(t)
            .unwrap_or(i64::MAX)
            .saturating_add(self.shift)
            .max(0) as u64
    }

    /// Moves to the first sample still playing at `timestamp_us`.
    fn seek(&mut self, timestamp_us: u64) {
        self.next = self
            .samples
            .partition_point(|s| {
                self.ts_us(s)
                    .saturating_add(micros(i64::from(s.duration), self.timescale))
                    <= timestamp_us
            })
            .max(self.first);
    }

    fn frame(&mut self, source: &mut BoxedSource) -> Result<Option<AudioFrame>> {
        let Some(sample) = self.samples.get(self.next) else {
            return Ok(None);
        };
        let data = read_sample(source, sample)?;
        let frame = AudioFrame {
            codec: self.stream.codec,
            data: self.format.packet(&data)?,
            timestamp_us: self.ts_us(sample),
            sample_rate: self.stream.sample_rate,
            channels: self.stream.channels,
        };
        self.next += 1;
        Ok(Some(frame))
    }
}

struct Boxes<'a>(&'a [u8]);

impl<'a> Iterator for Boxes<'a> {
    type Item = ([u8; 4], &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        let data = self.0;
        let kind: [u8; 4] = data.get(4..8)?.try_into().ok()?;
        let (header, size) = match be32(data, 0)? {
            0 => (8, data.len()),
            1 => (16, usize::try_from(be64(data, 8)?).ok()?),
            size => (8, size as usize),
        };
        if size < header || size > data.len() {
            return None;
        }
        self.0 = &data[size..];
        Some((kind, &data[header..size]))
    }
}

/// How the media timeline is anchored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Origin {
    /// A whole file: timestamp 0 is its earliest frame or the edit
    /// list start, whichever is first.
    File,
    /// One media segment of a longer stream: timestamp 0 is the edit
    /// list start, so every segment of the stream shares one timeline.
    EditList,
}

/// Movie fragments of a fragmented file still to be read, and what
/// locates them by time.
struct Fragments {
    trex: Vec<fragment::Trex>,
    file_len: u64,
    /// Offset of the next top-level box to look at.
    next_box: u64,
    /// No `moof` after `next_box`.
    ended: bool,
    /// Fragments of the track seeks are made on (video, else audio).
    index: FragmentIndex,
    /// Fragments of the audio track when video leads seeks: files that
    /// give each track its own `moof` place them independently.
    audio_index: FragmentIndex,
    /// Where each track's decode time continues after the last fragment.
    next_dts: Vec<(u32, u64)>,
}

/// What one walk over every `moof` of a file finds.
#[derive(Debug, Default)]
struct Scan {
    /// Fragments holding a keyframe of the track seeks are made on.
    primary: Vec<fragment::IndexEntry>,
    /// Fragments of the audio track when video leads seeks.
    secondary: Vec<fragment::IndexEntry>,
    /// Where the primary track ends, in its timescale.
    end: Option<u64>,
}

/// Fragments of one track by time, ascending.
#[derive(Debug, Default)]
struct FragmentIndex {
    entries: Vec<fragment::IndexEntry>,
    /// The index covers the whole file (from `sidx` or `mfra`, or a full
    /// scan); otherwise it holds the fragments read so far and grows as
    /// more are read.
    complete: bool,
}

impl FragmentIndex {
    fn complete(entries: Vec<fragment::IndexEntry>) -> Self {
        Self {
            entries,
            complete: true,
        }
    }

    fn remember(&mut self, entry: fragment::IndexEntry) {
        if self.complete || self.entries.len() >= fragment::MAX_INDEX {
            return;
        }
        if self
            .entries
            .last()
            .is_none_or(|last| last.moof < entry.moof)
        {
            self.entries.push(entry);
        }
    }

    /// The `moof` offset of the last fragment starting at or before `time`.
    fn at_or_before(&self, time: u64) -> Option<u64> {
        let at = self.entries.partition_point(|e| e.time <= time);
        at.checked_sub(1)
            .and_then(|i| self.entries.get(i))
            .map(|e| e.moof)
    }
}

impl Fragments {
    fn continue_dts(&self, track_id: u32) -> u64 {
        self.next_dts
            .iter()
            .find(|(id, _)| *id == track_id)
            .map_or(0, |&(_, dts)| dts)
    }

    fn set_dts(&mut self, track_id: u32, dts: u64) {
        match self.next_dts.iter_mut().find(|(id, _)| *id == track_id) {
            Some(entry) => entry.1 = dts,
            None => self.next_dts.push((track_id, dts)),
        }
    }

    /// Reads the next `moof` from `next_box` on: its offset and the
    /// samples it adds per track. `None` at the end of the file.
    fn next_moof(
        &mut self,
        source: &mut BoxedSource,
    ) -> Result<Option<(u64, Vec<fragment::TrackRun>)>> {
        while !self.ended {
            let offset = self.next_box;
            let Some((kind, header_len, size)) = box_header(source, offset, self.file_len)? else {
                self.ended = true;
                break;
            };
            self.next_box = offset + size;
            if &kind != b"moof" {
                continue;
            }
            let body_len = size - header_len;
            if body_len > MAX_MOOF {
                return Err(FerricastError::Demux(format!(
                    "moof box of {body_len} bytes exceeds the {MAX_MOOF} byte limit"
                )));
            }
            let mut body = vec![0; body_len as usize];
            source.seek(SeekFrom::Start(offset + header_len))?;
            source.read_exact(&mut body)?;
            let runs = fragment::parse_moof(&body, offset, &self.trex, |id| self.continue_dts(id));
            for run in &runs {
                if let Some(last) = run.samples.last() {
                    self.set_dts(
                        run.track_id,
                        last.dts.saturating_add(u64::from(last.duration)),
                    );
                }
            }
            return Ok(Some((offset, runs)));
        }
        Ok(None)
    }
}

impl Mp4Demuxer {
    pub(crate) fn open(source: BoxedSource, selection: TrackSelection) -> Result<Self> {
        Self::open_with(source, selection, Origin::File)
    }

    pub(crate) fn open_with(
        mut source: BoxedSource,
        selection: TrackSelection,
        origin_mode: Origin,
    ) -> Result<Self> {
        let file_len = source.seek(SeekFrom::End(0))?;
        let (moov, moov_end) = read_moov(&mut source, file_len)?;
        let (video, audio) = parse_moov(&moov, selection)?;
        let mvex = child(&moov, b"mvex");
        let mut duration_us = video
            .as_ref()
            .map(Track::duration_us)
            .into_iter()
            .chain(audio.as_ref().map(Track::duration_us))
            .max()
            .filter(|&d| d > 0);
        let readable = |tables: &Tables| -> Result<Vec<Sample>> {
            let mut samples = build_samples(tables)?;
            let readable = samples
                .iter()
                .take_while(|s| s.end().is_some_and(|end| end <= file_len))
                .count();
            samples.truncate(readable);
            Ok(samples)
        };

        let video = video
            .map(|track| -> Result<VideoTrack> {
                let samples = readable(&track.tables)?;
                let entry = track.entry;
                Ok(VideoTrack {
                    id: track.id,
                    stream: VideoStream {
                        codec: entry.codec,
                        width: entry.width,
                        height: entry.height,
                        color: entry.color,
                        duration_us: Some(micros(
                            i64::try_from(track.duration).unwrap_or(i64::MAX),
                            track.timescale,
                        )),
                        presentation_start_us: 0,
                    },
                    framing: entry.framing,
                    timescale: track.timescale,
                    samples,
                    origin: track.shift,
                    shift: track.shift,
                    next: 0,
                    at_start: true,
                })
            })
            .transpose()?;
        let audio = audio
            .map(|track| -> Result<AudioTrack> {
                Ok(AudioTrack {
                    id: track.id,
                    stream: track.entry.stream,
                    format: track.entry.format,
                    timescale: track.timescale,
                    samples: readable(&track.tables)?,
                    shift: track.shift,
                    offset_us: 0,
                    first: 0,
                    next: 0,
                })
            })
            .transpose()?;

        let no_samples = video.as_ref().is_none_or(|v| v.samples.is_empty())
            && audio.as_ref().is_none_or(|a| a.samples.is_empty());
        let fragments = match mvex {
            Some(mvex) if no_samples => Some(Fragments {
                trex: fragment::parse_trex(mvex),
                file_len,
                next_box: moov_end,
                ended: false,
                index: FragmentIndex::default(),
                audio_index: FragmentIndex::default(),
                next_dts: Vec::new(),
            }),
            _ => None,
        };
        let fragment_duration = mvex.and_then(fragment::parse_mehd).map(|d| {
            micros(
                i64::try_from(d).unwrap_or(i64::MAX),
                mvhd_timescale(&moov).unwrap_or(1000),
            )
        });
        let mut demuxer = Self {
            source,
            video,
            audio,
            duration_us,
            fragments,
        };
        if demuxer.fragments.is_some() {
            demuxer.fill()?;
            if let Some(index_duration) = demuxer.locate_fragments()? {
                duration_us = duration_us.max(Some(index_duration));
            }
            duration_us = duration_us.max(fragment_duration);
            demuxer.duration_us = duration_us;
        }
        demuxer.anchor(origin_mode);
        Ok(demuxer)
    }

    /// Sets the timeline origin from the samples known now (the whole
    /// file, or its first fragments).
    fn anchor(&mut self, origin_mode: Origin) {
        if let Some(video) = self.video.as_mut() {
            video.origin = match origin_mode {
                Origin::File => timeline_origin(&video.samples, video.shift),
                Origin::EditList => video.shift,
            };
            video.stream.presentation_start_us =
                micros(video.shift.saturating_sub(video.origin), video.timescale);
        }
        let offset_us = self
            .video
            .as_ref()
            .map_or(0, |v| v.stream.presentation_start_us);
        if let Some(audio) = self.audio.as_mut() {
            audio.offset_us = offset_us;
            audio.first = audio.samples.partition_point(|s| {
                s.decode_ticks().saturating_add(i64::from(s.duration)) <= audio.shift
            });
            audio.next = audio.first;
        }
    }

    /// Id and timescale of the track fragments are located by.
    fn primary(&self) -> Option<(u32, u32)> {
        self.video
            .as_ref()
            .map(|v| (v.id, v.timescale))
            .or_else(|| self.audio.as_ref().map(|a| (a.id, a.timescale)))
    }

    /// Id and timescale of the audio track when video leads seeks.
    fn secondary(&self) -> Option<(u32, u32)> {
        self.video.as_ref()?;
        self.audio.as_ref().map(|a| (a.id, a.timescale))
    }

    /// Builds the fragment indexes from `sidx` or `mfra` boxes when the
    /// file has them (else from one scan of its fragments), and returns
    /// the duration they reveal.
    fn locate_fragments(&mut self) -> Result<Option<u64>> {
        let Some((track_id, timescale)) = self.primary() else {
            return Ok(None);
        };
        let secondary = self.secondary();
        let Some(file_len) = self.fragments.as_ref().map(|f| f.file_len) else {
            return Ok(None);
        };
        // `sidx` boxes sit between the `moov` and the first `moof`: one
        // for the whole file or one per track.
        let mut primary_sidx = None;
        let mut audio_sidx = None;
        let mut offset = 0;
        while let Some((kind, header_len, size)) = box_header(&mut self.source, offset, file_len)? {
            match &kind {
                b"sidx" => {
                    let reference_id = self
                        .read_box_body(offset, header_len, size)?
                        .and_then(|body| be32(&body, 4));
                    if reference_id == Some(track_id) && primary_sidx.is_none() {
                        primary_sidx = self.sidx_index(offset, timescale)?;
                    } else if let Some((audio_id, audio_scale)) = secondary
                        && reference_id == Some(audio_id)
                        && audio_sidx.is_none()
                    {
                        audio_sidx = self.sidx_index(offset, audio_scale)?;
                    }
                    offset += size;
                }
                b"moof" | b"mdat" => break,
                _ => offset += size,
            }
        }

        let mut duration = None;
        let mut audio_entries = audio_sidx.map(|(entries, _)| entries);
        let mfra = match primary_sidx {
            Some(_) => None,
            None => self.read_mfra(file_len)?,
        };
        let from_mfra = |id: u32| mfra.as_deref().and_then(|m| fragment::parse_mfra(m, id));
        let entries = if let Some((entries, total)) = primary_sidx {
            duration = Some(micros(i64::try_from(total).unwrap_or(i64::MAX), timescale));
            entries
        } else if let Some(entries) = from_mfra(track_id) {
            if audio_entries.is_none() {
                audio_entries = secondary.and_then(|(id, _)| from_mfra(id));
            }
            entries
        } else {
            // No index in the file: walk its fragments once.
            let scan = self.scan_fragments(track_id, secondary.map(|(id, _)| id))?;
            duration = scan
                .end
                .map(|ticks| micros(i64::try_from(ticks).unwrap_or(i64::MAX), timescale));
            if audio_entries.is_none() {
                audio_entries = Some(scan.secondary);
            }
            scan.primary
        };
        if let Some(fragments) = self.fragments.as_mut()
            && let Some(audio_entries) = audio_entries.filter(|e| !e.is_empty())
        {
            fragments.audio_index = FragmentIndex::complete(audio_entries);
        }
        if entries.is_empty() {
            return Ok(duration);
        }
        if duration.is_none()
            && let Some(last) = entries.last()
        {
            // The last fragment's end is the end of the media.
            duration = self.fragment_end_us(last.moof)?;
        }
        if let Some(fragments) = self.fragments.as_mut() {
            fragments.index = FragmentIndex::complete(entries);
        }
        Ok(duration)
    }

    /// The body of the box at `offset`, unless it is over
    /// [`MAX_INDEX_BOX`].
    fn read_box_body(
        &mut self,
        offset: u64,
        header_len: u64,
        size: u64,
    ) -> Result<Option<Vec<u8>>> {
        let len = size - header_len;
        if len > MAX_INDEX_BOX {
            return Ok(None);
        }
        let mut body = vec![0; len as usize];
        self.source.seek(SeekFrom::Start(offset + header_len))?;
        self.source.read_exact(&mut body)?;
        Ok(Some(body))
    }

    /// The fragments indexed by the `sidx` box at `offset`, following
    /// references to further `sidx` boxes, and the duration the box
    /// covers, in `timescale`. `None` when the index is unreadable or
    /// does not point at fragments.
    fn sidx_index(
        &mut self,
        offset: u64,
        timescale: u32,
    ) -> Result<Option<(Vec<fragment::IndexEntry>, u64)>> {
        let Some(file_len) = self.fragments.as_ref().map(|f| f.file_len) else {
            return Ok(None);
        };
        let Some(root) = self.read_sidx(offset, file_len, timescale)? else {
            return Ok(None);
        };
        let reference_id = root.reference_id;
        let duration = root.duration;
        let mut entries = Vec::new();
        // Depth-first over the index tree: (references, next one, depth).
        let mut stack = vec![(root.refs, 0usize, 0usize)];
        let mut boxes_read = 1;
        while let Some((refs, at, depth)) = stack.last_mut() {
            let Some(reference) = refs.get(*at).copied() else {
                stack.pop();
                continue;
            };
            *at += 1;
            let depth = *depth;
            match reference.target {
                fragment::SidxTarget::Media => {
                    if entries.len() >= fragment::MAX_INDEX {
                        break;
                    }
                    entries.push(fragment::IndexEntry {
                        time: reference.time,
                        moof: reference.offset,
                    });
                }
                fragment::SidxTarget::Index => {
                    if depth + 1 >= MAX_SIDX_DEPTH || boxes_read >= MAX_SIDX_BOXES {
                        return Ok(None);
                    }
                    boxes_read += 1;
                    match self.read_sidx(reference.offset, file_len, timescale)? {
                        Some(child) if child.reference_id == reference_id => {
                            stack.push((child.refs, 0, depth + 1));
                        }
                        _ => return Ok(None),
                    }
                }
            }
        }
        if !entries.is_sorted_by_key(|e| e.moof) || !self.entries_point_at_fragments(&entries)? {
            return Ok(None);
        }
        Ok(Some((entries, duration)))
    }

    /// The `sidx` box at `offset`, parsed; `None` when something else
    /// is there.
    fn read_sidx(
        &mut self,
        offset: u64,
        file_len: u64,
        timescale: u32,
    ) -> Result<Option<fragment::Sidx>> {
        let Some((kind, header_len, size)) = box_header(&mut self.source, offset, file_len)? else {
            return Ok(None);
        };
        if &kind != b"sidx" {
            return Ok(None);
        }
        let body = self.read_box_body(offset, header_len, size)?;
        Ok(body.and_then(|body| fragment::parse_sidx(&body, offset + size, timescale)))
    }

    /// Whether the first, middle and last of `entries` start a fragment.
    /// Some muxers write `sidx` boxes whose sizes skip the fragments of
    /// other tracks; their offsets then land inside media data.
    fn entries_point_at_fragments(&mut self, entries: &[fragment::IndexEntry]) -> Result<bool> {
        let Some(file_len) = self.fragments.as_ref().map(|f| f.file_len) else {
            return Ok(false);
        };
        let probes = [0, entries.len() / 2, entries.len().saturating_sub(1)];
        for entry in probes.iter().filter_map(|&i| entries.get(i)) {
            let kind = box_header(&mut self.source, entry.moof, file_len)?.map(|(kind, ..)| kind);
            if !matches!(
                kind.as_ref(),
                Some(b"moof" | b"styp" | b"sidx" | b"emsg" | b"prft" | b"free" | b"skip")
            ) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Reads every `moof` of the file (not the media): the index entry of
    /// each fragment holding a keyframe of `track_id`, of each fragment of
    /// `secondary`, and where `track_id` ends, in its timescale.
    fn scan_fragments(&mut self, track_id: u32, secondary: Option<u32>) -> Result<Scan> {
        let mut scan = Scan::default();
        let Some(fragments) = self.fragments.as_mut() else {
            return Ok(scan);
        };
        let saved = (
            fragments.next_box,
            fragments.ended,
            fragments.next_dts.clone(),
        );
        fragments.next_box = 0;
        fragments.ended = false;
        fragments.next_dts.clear();
        while scan.primary.len() < fragment::MAX_INDEX
            && scan.secondary.len() < fragment::MAX_INDEX
            && let Some((moof, runs)) = fragments.next_moof(&mut self.source)?
        {
            for run in runs {
                if run.track_id == track_id {
                    if let Some(first) = run.samples.iter().find(|s| s.sync) {
                        scan.primary.push(fragment::IndexEntry {
                            time: u64::try_from(first.presentation_ticks()).unwrap_or(0),
                            moof,
                        });
                    }
                    if let Some(last) = run.samples.last() {
                        scan.end = Some(last.dts.saturating_add(u64::from(last.duration)));
                    }
                } else if Some(run.track_id) == secondary
                    && let Some(first) = run.samples.first()
                {
                    scan.secondary.push(fragment::IndexEntry {
                        time: first.dts,
                        moof,
                    });
                }
            }
        }
        (fragments.next_box, fragments.ended, fragments.next_dts) = saved;
        Ok(scan)
    }

    /// The body of the `mfra` box located through the `mfro` box that
    /// closes the file.
    fn read_mfra(&mut self, file_len: u64) -> Result<Option<Vec<u8>>> {
        if file_len < 16 {
            return Ok(None);
        }
        let mut mfro = [0; 16];
        self.source.seek(SeekFrom::Start(file_len - 16))?;
        self.source.read_exact(&mut mfro)?;
        if &mfro[4..8] != b"mfro" {
            return Ok(None);
        }
        let size = u64::from(be32(&mfro, 12).unwrap_or(0));
        if !(16..=MAX_INDEX_BOX).contains(&size) || size > file_len {
            return Ok(None);
        }
        let mut mfra = vec![0; size as usize];
        self.source.seek(SeekFrom::Start(file_len - size))?;
        self.source.read_exact(&mut mfra)?;
        if mfra.get(4..8) != Some(&b"mfra"[..]) {
            return Ok(None);
        }
        mfra.drain(..8);
        Ok(Some(mfra))
    }

    /// Where the media of the `moof` at `offset` ends, on the primary
    /// track's timeline.
    fn fragment_end_us(&mut self, offset: u64) -> Result<Option<u64>> {
        let Some((track_id, timescale)) = self.primary() else {
            return Ok(None);
        };
        let Some(fragments) = self.fragments.as_mut() else {
            return Ok(None);
        };
        let saved = (
            fragments.next_box,
            fragments.ended,
            fragments.next_dts.clone(),
        );
        fragments.next_box = offset;
        fragments.ended = false;
        let moof = fragments.next_moof(&mut self.source)?;
        (fragments.next_box, fragments.ended, fragments.next_dts) = saved;
        let end = moof.and_then(|(_, runs)| {
            runs.into_iter()
                .find(|r| r.track_id == track_id)
                .and_then(|r| r.samples.last().map(|s| s.dts + u64::from(s.duration)))
        });
        Ok(end.map(|ticks| micros(i64::try_from(ticks).unwrap_or(i64::MAX), timescale)))
    }

    /// Reads fragments until every track has a sample waiting, unless
    /// another track already has [`MAX_PENDING`] queued (a track that
    /// ends early must not pull the whole file in).
    fn fill(&mut self) -> Result<()> {
        loop {
            let Some(fragments) = self.fragments.as_ref() else {
                return Ok(());
            };
            if fragments.ended {
                return Ok(());
            }
            let left = |next: usize, len: usize| len.saturating_sub(next);
            let video_left = self.video.as_ref().map(|v| left(v.next, v.samples.len()));
            let audio_left = self.audio.as_ref().map(|a| left(a.next, a.samples.len()));
            let dry = video_left == Some(0) || audio_left == Some(0);
            let queued = video_left.unwrap_or(0).max(audio_left.unwrap_or(0));
            if !dry || queued >= MAX_PENDING {
                return Ok(());
            }
            if !self.load_fragment()? {
                return Ok(());
            }
        }
    }

    /// Reads the next fragment into the tracks; `false` at the end.
    fn load_fragment(&mut self) -> Result<bool> {
        let Some(fragments) = self.fragments.as_mut() else {
            return Ok(false);
        };
        let Some((moof, runs)) = fragments.next_moof(&mut self.source)? else {
            return Ok(false);
        };
        let file_len = fragments.file_len;
        let primary = self.primary().map(|(id, _)| id);
        let secondary = self.secondary().map(|(id, _)| id);
        for run in runs {
            let mut samples = run.samples;
            let readable = samples
                .iter()
                .take_while(|s| s.end().is_some_and(|end| end <= file_len))
                .count();
            samples.truncate(readable);
            if let Some(fragments) = self.fragments.as_mut() {
                if primary == Some(run.track_id)
                    && let Some(first) = samples.iter().find(|s| s.sync)
                {
                    fragments.index.remember(fragment::IndexEntry {
                        time: u64::try_from(first.presentation_ticks()).unwrap_or(0),
                        moof,
                    });
                } else if secondary == Some(run.track_id)
                    && let Some(first) = samples.first()
                {
                    fragments.audio_index.remember(fragment::IndexEntry {
                        time: first.dts,
                        moof,
                    });
                }
            }
            if let Some(video) = self.video.as_mut().filter(|v| v.id == run.track_id) {
                video.samples.drain(..video.next.min(video.samples.len()));
                video.next = 0;
                video.samples.extend(samples);
            } else if let Some(audio) = self.audio.as_mut().filter(|a| a.id == run.track_id) {
                audio.samples.drain(..audio.next.min(audio.samples.len()));
                audio.next = 0;
                audio.first = 0;
                let shift = audio.shift;
                // Encoder priming wholly before the edit start never plays.
                audio.samples.extend(
                    samples
                        .into_iter()
                        .filter(|s| s.decode_ticks().saturating_add(i64::from(s.duration)) > shift),
                );
            }
        }
        Ok(true)
    }

    pub(crate) fn video(&self) -> Option<&VideoStream> {
        self.video.as_ref().map(|v| &v.stream)
    }

    pub(crate) fn audio(&self) -> Option<&AudioStream> {
        self.audio.as_ref().map(|a| &a.stream)
    }

    pub(crate) fn duration_us(&self) -> Option<u64> {
        self.duration_us
    }

    /// The next packet of either track, in decode-time order.
    pub(crate) fn next_packet(&mut self) -> Result<Option<Packet>> {
        self.fill()?;
        let video_next = self.video.as_ref().and_then(VideoTrack::next_dts_us);
        let audio_next = self.audio.as_ref().and_then(AudioTrack::next_ts_us);
        let audio_first = match (video_next, audio_next) {
            (Some(v), Some(a)) => a < v,
            (None, Some(_)) => true,
            (Some(_) | None, None) => false,
        };
        if audio_first && let Some(audio) = self.audio.as_mut() {
            return Ok(audio.frame(&mut self.source)?.map(Packet::Audio));
        }
        match self.video.as_mut() {
            Some(video) => Ok(video.frame(&mut self.source)?.map(Packet::Video)),
            None => Ok(None),
        }
    }

    /// Moves to the last video keyframe presented at or before
    /// `timestamp_us` (audio-only: the audio packet playing then) and
    /// returns its timestamp.
    ///
    /// # Errors
    ///
    /// Fails on read errors while locating a fragment.
    pub(crate) fn seek(&mut self, timestamp_us: u64) -> Result<u64> {
        if self.fragments.is_some() {
            return self.seek_fragmented(timestamp_us);
        }
        let landed = self
            .video
            .as_mut()
            .map_or(timestamp_us, |v| v.seek(timestamp_us));
        if let Some(audio) = self.audio.as_mut() {
            audio.seek(landed);
        }
        Ok(landed)
    }

    /// `timestamp_us` as presentation ticks of the primary track.
    fn primary_ticks(&self, timestamp_us: u64) -> u64 {
        let ticks = |us: u64, timescale: u32, origin: i64| {
            let t = u128::from(us) * u128::from(timescale) / 1_000_000;
            (t as i64).saturating_add(origin).max(0) as u64
        };
        match (&self.video, &self.audio) {
            (Some(v), _) => ticks(timestamp_us, v.timescale, v.origin),
            (None, Some(a)) => ticks(
                timestamp_us.saturating_sub(a.offset_us),
                a.timescale,
                a.shift,
            ),
            (None, None) => 0,
        }
    }

    fn seek_fragmented(&mut self, timestamp_us: u64) -> Result<u64> {
        let target = self.primary_ticks(timestamp_us);
        // Scan forward until the index reaches past the target (or the
        // file ends) when it only holds what was read so far.
        loop {
            let Some(fragments) = self.fragments.as_mut() else {
                return Ok(timestamp_us);
            };
            let index = &fragments.index;
            if index.complete
                || index.entries.last().is_some_and(|e| e.time > target)
                || index.entries.len() >= fragment::MAX_INDEX
            {
                break;
            }
            if let Some(last) = index.entries.last() {
                fragments.next_box = fragments.next_box.max(last.moof);
            }
            if fragments.ended {
                break;
            }
            // Only the index matters while scanning: the samples read
            // are discarded so they cannot pile up.
            if let Some(video) = self.video.as_mut() {
                video.samples.clear();
                video.next = 0;
            }
            if let Some(audio) = self.audio.as_mut() {
                audio.samples.clear();
                audio.next = 0;
            }
            if !self.load_fragment()? {
                break;
            }
        }
        let Some(fragments) = self.fragments.as_ref() else {
            return Ok(timestamp_us);
        };
        let at = fragments
            .index
            .entries
            .partition_point(|e| e.time <= target);
        let mut candidate = at.saturating_sub(1);
        loop {
            let moof = self.primary_fragment(candidate);
            self.restart_at(moof)?;
            self.fill()?;
            if self.has_keyframe_by(timestamp_us) || candidate == 0 {
                let landed = self.position_tracks(timestamp_us);
                return self.align_audio(timestamp_us, landed, moof, candidate);
            }
            // The fragment holds no keyframe at or before the target:
            // try the one before.
            candidate -= 1;
        }
    }

    /// The `moof` offset of entry `i` of the primary index.
    fn primary_fragment(&self, i: usize) -> Option<u64> {
        self.fragments
            .as_ref()
            .and_then(|f| f.index.entries.get(i))
            .map(|e| e.moof)
    }

    /// When the tracks are fragmented separately, the audio playing at
    /// `landed` can sit in a fragment before `restarted_at`, the video
    /// fragment reading restarted from; audio would then resume late, at
    /// its next fragment. Restarts from the audio fragment instead (from
    /// the previous video fragment when the audio index does not reach
    /// back) and returns where playback resumes.
    fn align_audio(
        &mut self,
        timestamp_us: u64,
        landed: u64,
        restarted_at: Option<u64>,
        candidate: usize,
    ) -> Result<u64> {
        let (Some(restarted_at), Some(_), Some(audio)) =
            (restarted_at, self.video.as_ref(), self.audio.as_ref())
        else {
            return Ok(landed);
        };
        // No audio left to play also counts: its fragment may lie behind.
        let late = audio
            .samples
            .get(audio.next)
            .is_none_or(|s| audio.ts_us(s) > landed);
        if !late {
            return Ok(landed);
        }
        let ticks = audio.ticks(landed);
        let earlier = self
            .fragments
            .as_ref()
            .and_then(|f| f.audio_index.at_or_before(ticks))
            .filter(|&moof| moof < restarted_at)
            .or_else(|| {
                candidate
                    .checked_sub(1)
                    .and_then(|c| self.primary_fragment(c))
            });
        let Some(earlier) = earlier else {
            return Ok(landed);
        };
        self.restart_at(Some(earlier))?;
        self.load_through(landed)?;
        Ok(self.position_tracks(timestamp_us))
    }

    /// Reads fragments until the video keyframe presented at `landed` is
    /// loaded, then until every track has a sample waiting; bounded like
    /// [`Self::fill`].
    fn load_through(&mut self, landed: u64) -> Result<()> {
        loop {
            let loaded = self.video.as_ref().is_none_or(|video| {
                video
                    .samples
                    .iter()
                    .any(|s| s.sync && video.pts_us(s) >= landed)
            });
            let queued = self.video.as_ref().map_or(0, |v| v.samples.len())
                + self.audio.as_ref().map_or(0, |a| a.samples.len());
            if loaded || queued >= 2 * MAX_PENDING || !self.load_fragment()? {
                break;
            }
        }
        self.fill()
    }

    /// Drops every queued sample and continues reading at the `moof` at
    /// `moof` (the first one when `None`).
    fn restart_at(&mut self, moof: Option<u64>) -> Result<()> {
        if let Some(fragments) = self.fragments.as_mut() {
            fragments.next_box = moof.unwrap_or(0);
            fragments.ended = false;
        }
        if let Some(video) = self.video.as_mut() {
            video.samples.clear();
            video.next = 0;
            video.at_start = true;
        }
        if let Some(audio) = self.audio.as_mut() {
            audio.samples.clear();
            audio.next = 0;
            audio.first = 0;
        }
        Ok(())
    }

    /// Whether the loaded video has a keyframe at or before
    /// `timestamp_us` (always true for audio-only media).
    fn has_keyframe_by(&self, timestamp_us: u64) -> bool {
        self.video.as_ref().is_none_or(|video| {
            video
                .samples
                .iter()
                .any(|s| s.sync && video.pts_us(s) <= timestamp_us)
        })
    }

    /// Positions the loaded samples for `timestamp_us` and returns where
    /// playback resumes.
    fn position_tracks(&mut self, timestamp_us: u64) -> u64 {
        let landed = self
            .video
            .as_mut()
            .map_or(timestamp_us, |video| video.seek(timestamp_us));
        if let Some(audio) = self.audio.as_mut() {
            audio.seek(landed);
        }
        landed
    }

    pub(crate) fn rewind(&mut self) -> Result<()> {
        if self.fragments.is_some() {
            return self.seek(0).map(drop);
        }
        if let Some(video) = self.video.as_mut() {
            video.next = 0;
            video.at_start = true;
        }
        if let Some(audio) = self.audio.as_mut() {
            audio.next = audio.first;
        }
        Ok(())
    }
}

pub(crate) fn is_mp4(head: &[u8]) -> bool {
    matches!(
        head.get(4..8),
        Some(
            b"ftyp" | b"moov" | b"mdat" | b"free" | b"wide" | b"skip" | b"styp" | b"sidx" | b"moof"
        )
    )
}

fn read_sample(source: &mut BoxedSource, sample: &Sample) -> Result<Vec<u8>> {
    if sample.size > MAX_SAMPLE_SIZE {
        return Err(FerricastError::Demux(format!(
            "sample of {} bytes exceeds the {MAX_SAMPLE_SIZE} byte limit",
            sample.size
        )));
    }
    let mut data = vec![0; sample.size as usize];
    source.seek(SeekFrom::Start(sample.offset))?;
    source.read_exact(&mut data)?;
    Ok(data)
}

/// The body of the `moov` box and the offset where the box ends.
fn read_moov(source: &mut BoxedSource, len: u64) -> Result<(Vec<u8>, u64)> {
    let mut offset = 0;
    while offset + 8 <= len {
        let mut header = [0; 16];
        let available = (len - offset).min(16) as usize;
        source.seek(SeekFrom::Start(offset))?;
        source.read_exact(&mut header[..available])?;
        let (header_len, size) = match be32(&header, 0).unwrap_or(0) {
            0 => (8, len - offset),
            1 => (16, be64(&header, 8).ok_or_else(truncated)?),
            size => (8, u64::from(size)),
        };
        if size < header_len || size > len - offset {
            return Err(truncated());
        }
        if &header[4..8] == b"moov" {
            let body_len = size - header_len;
            if body_len > MAX_MOOV {
                return Err(FerricastError::Demux(format!(
                    "moov box of {body_len} bytes exceeds the {MAX_MOOV} byte limit"
                )));
            }
            let mut body = vec![0; body_len as usize];
            source.seek(SeekFrom::Start(offset + header_len))?;
            source.read_exact(&mut body)?;
            return Ok((body, offset + size));
        }
        offset += size;
    }
    Err(FerricastError::Demux("no moov box".into()))
}

/// Kind, header length and size of the box at `offset`; `None` past the
/// end or for a box that does not fit the file (a truncated download).
fn box_header(
    source: &mut BoxedSource,
    offset: u64,
    len: u64,
) -> Result<Option<([u8; 4], u64, u64)>> {
    if offset.saturating_add(8) > len {
        return Ok(None);
    }
    let mut header = [0; 16];
    let available = (len - offset).min(16) as usize;
    source.seek(SeekFrom::Start(offset))?;
    source.read_exact(&mut header[..available])?;
    let (header_len, size) = match be32(&header, 0).unwrap_or(0) {
        0 => (8, len - offset),
        1 => match be64(&header, 8) {
            Some(size) if available == 16 => (16, size),
            _ => return Ok(None),
        },
        size => (8, u64::from(size)),
    };
    if size < header_len || size > len - offset {
        return Ok(None);
    }
    let kind = [header[4], header[5], header[6], header[7]];
    Ok(Some((kind, header_len, size)))
}

/// Timescale of the `mvhd` box.
fn mvhd_timescale(moov: &[u8]) -> Option<u32> {
    let mvhd = child(moov, b"mvhd")?;
    match *mvhd.first()? {
        1 => be32(mvhd, 20),
        _ => be32(mvhd, 12),
    }
    .filter(|&t| t > 0)
}

fn truncated() -> FerricastError {
    FerricastError::Demux("truncated MP4 box".into())
}

/// The first supported video track and (unless only video is wanted)
/// the first supported audio track.
type Tracks = (Option<Track<VideoEntry>>, Option<Track<AudioEntry>>);

fn parse_moov(moov: &[u8], selection: TrackSelection) -> Result<Tracks> {
    let mut video = None;
    let mut audio = None;
    for track in Boxes(moov)
        .filter(|(kind, _)| kind == b"trak")
        .filter_map(|(_, trak)| parse_track(trak))
    {
        match track {
            Parsed::Video(track) => {
                video.get_or_insert(track);
            }
            Parsed::Audio(track) => {
                if selection == TrackSelection::All {
                    audio.get_or_insert(track);
                }
            }
        }
    }
    match (&video, &audio, selection) {
        (None, _, TrackSelection::VideoOnly) => {
            Err(FerricastError::Demux("no supported video track".into()))
        }
        (None, None, TrackSelection::All) => Err(FerricastError::Demux(
            "no supported video or audio track".into(),
        )),
        _ => Ok((video, audio)),
    }
}

fn parse_track(trak: &[u8]) -> Option<Parsed> {
    let mdia = child(trak, b"mdia")?;
    let handler = child(mdia, b"hdlr")?.get(8..12)?;
    let (timescale, duration) = parse_mdhd(child(mdia, b"mdhd")?)?;
    if timescale == 0 {
        return None;
    }
    let stbl = child(child(mdia, b"minf")?, b"stbl")?;
    let stsd = child(stbl, b"stsd")?;
    let (kind, body) = Boxes(stsd.get(8..)?).next()?;
    let tables = parse_tables(stbl)?;
    let duration = duration.unwrap_or_else(|| tables.decode_duration());
    let duration_us = micros(i64::try_from(duration).unwrap_or(i64::MAX), timescale);
    let track = Track {
        id: parse_tkhd_id(child(trak, b"tkhd")?)?,
        entry: (),
        timescale,
        duration,
        tables,
        shift: child(trak, b"edts")
            .and_then(|edts| child(edts, b"elst"))
            .and_then(parse_elst)
            .unwrap_or(0),
    };
    match handler {
        b"vide" => Some(Parsed::Video(track.with(parse_entry(kind, body)?))),
        b"soun" => Some(Parsed::Audio(track.with(parse_audio_entry(
            kind,
            body,
            duration_us,
        )?))),
        _ => None,
    }
}

/// `track_ID` of a `tkhd` box.
fn parse_tkhd_id(tkhd: &[u8]) -> Option<u32> {
    match *tkhd.first()? {
        1 => be32(tkhd, 20),
        _ => be32(tkhd, 12),
    }
}

/// Returns the media timescale and duration; an all-ones duration means
/// "unknown" (ISO/IEC 14496-12 §8.4.2.3) and comes back as `None`.
fn parse_mdhd(mdhd: &[u8]) -> Option<(u32, Option<u64>)> {
    match *mdhd.first()? {
        1 => {
            let duration = be64(mdhd, 24)?;
            Some((be32(mdhd, 20)?, (duration != u64::MAX).then_some(duration)))
        }
        _ => {
            let duration = be32(mdhd, 16)?;
            Some((
                be32(mdhd, 12)?,
                (duration != u32::MAX).then_some(u64::from(duration)),
            ))
        }
    }
}

fn parse_entry(kind: [u8; 4], entry: &[u8]) -> Option<VideoEntry> {
    let children = entry.get(VISUAL_ENTRY_HEADER..)?;
    let (codec, framing) = match &kind {
        b"avc1" | b"avc3" => (
            Codec::H264,
            Framing::LengthPrefixed(LengthPrefixed::from_avcc(child(children, b"avcC")?).ok()?),
        ),
        b"hvc1" | b"hev1" => (
            Codec::H265,
            Framing::LengthPrefixed(LengthPrefixed::from_hvcc(child(children, b"hvcC")?).ok()?),
        ),
        b"vp08" => (Codec::Vp8, Framing::Raw),
        b"vp09" => (Codec::Vp9, Framing::Raw),
        b"av01" => (Codec::Av1, Framing::Raw),
        _ => return None,
    };
    let color = child(children, b"colr")
        .and_then(parse_colr)
        .or_else(|| child(children, b"vpcC").and_then(parse_vpcc));
    Some(VideoEntry {
        codec,
        width: u32::from(be16(entry, 24)?),
        height: u32::from(be16(entry, 26)?),
        framing,
        color,
    })
}

/// Reads an `mp4a` (AAC) or `Opus` sample entry.
fn parse_audio_entry(kind: [u8; 4], entry: &[u8], duration_us: u64) -> Option<AudioEntry> {
    let extra = match be16(entry, 8)? {
        0 => 0,
        1 => 16,
        2 => 36,
        _ => return None,
    };
    let children = entry.get(AUDIO_ENTRY_HEADER + extra..)?;
    let channels = be16(entry, 16)?;
    let duration_us = Some(duration_us);
    match &kind {
        b"mp4a" => {
            // QuickTime nests the esds inside a `wave` box.
            let esds =
                child(children, b"esds").or_else(|| child(child(children, b"wave")?, b"esds"))?;
            let config = AacConfig::from_asc(esds_audio_config(esds)?).ok()?;
            Some(AudioEntry {
                format: AudioFormat::Aac(config),
                stream: config.stream(duration_us),
            })
        }
        b"Opus" => {
            let channels = child(children, b"dOps")
                .and_then(opus_channels)
                .unwrap_or(channels);
            Some(AudioEntry {
                format: AudioFormat::Opus,
                stream: AudioStream {
                    codec: AudioFormat::Opus.codec(),
                    sample_rate: OPUS_RATE,
                    channels,
                    duration_us,
                },
            })
        }
        _ => None,
    }
}

/// `ES_DescrTag`, `DecoderConfigDescrTag` and `DecSpecificInfoTag`
/// (ISO/IEC 14496-1 §7.2.2.1).
const ES_DESCRIPTOR: u8 = 0x03;
const DECODER_CONFIG: u8 = 0x04;
const DECODER_SPECIFIC_INFO: u8 = 0x05;
/// `objectTypeIndication` values for MPEG-4 AAC and MPEG-2 AAC
/// (Main, LC, SSR).
const OTI_MPEG4_AUDIO: u8 = 0x40;
const OTI_MPEG2_AAC: [u8; 3] = [0x66, 0x67, 0x68];

/// The AAC `AudioSpecificConfig` inside an `esds` box.
fn esds_audio_config(esds: &[u8]) -> Option<&[u8]> {
    let (tag, es) = descriptor(esds.get(4..)?)?;
    if tag != ES_DESCRIPTOR {
        return None;
    }
    let flags = *es.get(2)?;
    let mut at = 3;
    if flags & 0x80 != 0 {
        at += 2;
    }
    if flags & 0x40 != 0 {
        at += 1 + usize::from(*es.get(at)?);
    }
    if flags & 0x20 != 0 {
        at += 2;
    }
    let (tag, config) = descriptor(es.get(at..)?)?;
    let oti = *config.first()?;
    if tag != DECODER_CONFIG || (oti != OTI_MPEG4_AUDIO && !OTI_MPEG2_AAC.contains(&oti)) {
        return None;
    }
    let (tag, info) = descriptor(config.get(13..)?)?;
    (tag == DECODER_SPECIFIC_INFO).then_some(info)
}

/// Splits one descriptor (tag, then a size of up to four 7-bit groups).
fn descriptor(data: &[u8]) -> Option<(u8, &[u8])> {
    let (&tag, mut rest) = data.split_first()?;
    let mut len = 0usize;
    for _ in 0..4 {
        let (&byte, tail) = rest.split_first()?;
        rest = tail;
        len = (len << 7) | usize::from(byte & 0x7F);
        if byte & 0x80 == 0 {
            break;
        }
    }
    Some((tag, rest.get(..len)?))
}

fn parse_colr(colr: &[u8]) -> Option<Color> {
    let matrix = u8::try_from(be16(colr, 8)?).ok()?;
    match colr.get(..4)? {
        b"nclx" => Some(Color {
            matrix_coefficients: matrix,
            full_range: colr.get(10)? & 0x80 != 0,
        }),
        b"nclc" => Some(Color {
            matrix_coefficients: matrix,
            full_range: false,
        }),
        _ => None,
    }
}

fn parse_vpcc(vpcc: &[u8]) -> Option<Color> {
    Some(Color {
        matrix_coefficients: *vpcc.get(9)?,
        full_range: vpcc.get(6)? & 1 != 0,
    })
}

fn parse_elst(elst: &[u8]) -> Option<i64> {
    let version = *elst.first()?;
    let count = be32(elst, 4)? as usize;
    let entry_len = if version == 1 { 20 } else { 12 };
    elst.get(8..)?
        .chunks_exact(entry_len)
        .take(count)
        .find_map(|entry| {
            let media_time = if version == 1 {
                be64(entry, 8)? as i64
            } else {
                i64::from(be32(entry, 4)? as i32)
            };
            (media_time >= 0).then_some(media_time)
        })
}

fn parse_tables(stbl: &[u8]) -> Option<Tables> {
    // Entry counts are capped like the sample count: a table never needs more
    // entries than there can be samples, and the cap bounds every walk below.
    let counted = |kind: &[u8; 4]| {
        child(stbl, kind)
            .and_then(|body| Some((body.get(8..)?, (be32(body, 4)? as usize).min(MAX_SAMPLES))))
    };
    let (stts, count) = counted(b"stts")?;
    let (stsc, stsc_count) = counted(b"stsc")?;
    let stsz = child(stbl, b"stsz")?;
    let constant_size = be32(stsz, 4)?;
    let sample_count = (be32(stsz, 8)? as usize).min(MAX_SAMPLES);
    let sizes = if constant_size == 0 {
        words(stsz.get(12..)?).take(sample_count).collect()
    } else {
        vec![constant_size; sample_count]
    };
    let chunk_offsets = if let Some((body, count)) = counted(b"stco") {
        words(body).take(count).map(u64::from).collect()
    } else {
        let (body, count) = counted(b"co64")?;
        let (offsets, _) = body.as_chunks::<8>();
        offsets
            .iter()
            .take(count)
            .map(|c| u64::from_be_bytes(*c))
            .collect()
    };
    Some(Tables {
        stts: pairs(stts, count),
        ctts: counted(b"ctts")
            .map(|(body, count)| pairs(body, count))
            .unwrap_or_default()
            .into_iter()
            .map(|(count, offset)| (count, offset as i32))
            .collect(),
        stsc: chunk_runs(stsc, stsc_count),
        sizes,
        chunk_offsets,
        sync: counted(b"stss").map(|(body, count)| words(body).take(count).collect()),
    })
}

fn words(body: &[u8]) -> impl Iterator<Item = u32> + '_ {
    let (words, _) = body.as_chunks::<4>();
    words.iter().map(|c| u32::from_be_bytes(*c))
}

fn pairs(body: &[u8], count: usize) -> Vec<(u32, u32)> {
    let (entries, _) = body.as_chunks::<8>();
    entries
        .iter()
        .take(count)
        .map(|c| {
            (
                u32::from_be_bytes([c[0], c[1], c[2], c[3]]),
                u32::from_be_bytes([c[4], c[5], c[6], c[7]]),
            )
        })
        .collect()
}

fn chunk_runs(body: &[u8], count: usize) -> Vec<ChunkRun> {
    let (entries, _) = body.as_chunks::<12>();
    entries
        .iter()
        .take(count)
        .map(|c| ChunkRun {
            first_chunk: u32::from_be_bytes([c[0], c[1], c[2], c[3]]),
            samples_per_chunk: u32::from_be_bytes([c[4], c[5], c[6], c[7]]),
            description_index: u32::from_be_bytes([c[8], c[9], c[10], c[11]]),
        })
        .collect()
}

/// Checks that `stsc` runs start at chunk 1 and that their first chunks
/// strictly increase, which lets [`build_samples`] walk them with a single
/// forward cursor, and that every chunk uses the one sample description
/// that was parsed.
fn validate_stsc(stsc: &[ChunkRun]) -> Result<()> {
    let mut previous = 0;
    for run in stsc {
        let first_chunk = run.first_chunk;
        let bad_start = previous == 0 && first_chunk != 1;
        if bad_start || first_chunk <= previous {
            return Err(FerricastError::Demux(format!(
                "stsc runs must start at chunk 1 and strictly increase (found {first_chunk} after {previous})"
            )));
        }
        if run.description_index != 1 {
            return Err(FerricastError::Demux(format!(
                "chunk {first_chunk} uses sample description {}; only a single stsd entry is supported",
                run.description_index
            )));
        }
        previous = first_chunk;
    }
    Ok(())
}

fn build_samples(tables: &Tables) -> Result<Vec<Sample>> {
    validate_stsc(&tables.stsc)?;
    let mut samples = Vec::with_capacity(tables.sizes.len());
    let mut durations = expand(&tables.stts);
    let mut cts_offsets = expand(&tables.ctts);
    let mut runs = tables.stsc.iter().peekable();
    let mut per_chunk = None;
    let mut dts = 0u64;
    'chunks: for (chunk_number, &chunk_offset) in (1u32..).zip(&tables.chunk_offsets) {
        if samples.len() == tables.sizes.len() {
            break;
        }
        while let Some(run) = runs.next_if(|run| run.first_chunk <= chunk_number) {
            per_chunk = Some(run.samples_per_chunk);
        }
        let Some(per_chunk) = per_chunk else {
            return Err(FerricastError::Demux(format!(
                "stsc does not cover chunk {chunk_number}"
            )));
        };
        let mut offset = chunk_offset;
        for _ in 0..per_chunk {
            let Some(&size) = tables.sizes.get(samples.len()) else {
                break 'chunks;
            };
            let duration = durations.next().unwrap_or(0);
            samples.push(Sample {
                offset,
                size,
                dts,
                duration,
                cts_offset: cts_offsets.next().unwrap_or(0),
                sync: tables
                    .sync
                    .as_ref()
                    .is_none_or(|sync| sync.binary_search(&(samples.len() as u32 + 1)).is_ok()),
            });
            offset = offset.saturating_add(u64::from(size));
            dts = dts.saturating_add(u64::from(duration));
        }
    }
    Ok(samples)
}

/// Media time that maps to timestamp 0.
///
/// Normally the edit list's media time (`shift`), but when some samples
/// present before the edit starts (pre-roll) the origin moves back to the
/// earliest of them, so their timestamps stay distinct and non-negative
/// instead of all clamping to 0. Consumers skip displaying anything before
/// [`VideoStream::presentation_start_us`].
fn timeline_origin(samples: &[Sample], shift: i64) -> i64 {
    samples
        .iter()
        .map(Sample::presentation_ticks)
        .min()
        .map_or(shift, |earliest| earliest.min(shift))
}

fn expand<T: Copy>(runs: &[(u32, T)]) -> impl Iterator<Item = T> + '_ {
    runs.iter()
        .flat_map(|&(count, value)| std::iter::repeat_n(value, count as usize))
}

fn child<'a>(data: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    Boxes(data)
        .find(|(found, _)| found == kind)
        .map(|(_, body)| body)
}

fn be16(data: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes(data.get(at..at + 2)?.try_into().ok()?))
}

fn be32(data: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(data.get(at..at + 4)?.try_into().ok()?))
}

fn be64(data: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_be_bytes(data.get(at..at + 8)?.try_into().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = (body.len() as u32 + 8).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }

    fn runs(runs: &[(u32, u32)]) -> Vec<ChunkRun> {
        runs.iter()
            .map(|&(first_chunk, samples_per_chunk)| ChunkRun {
                first_chunk,
                samples_per_chunk,
                description_index: 1,
            })
            .collect()
    }

    fn sample(offset: u64, size: u32, dts: u64, duration: u32, sync: bool) -> Sample {
        Sample {
            offset,
            size,
            dts,
            duration,
            cts_offset: 0,
            sync,
        }
    }

    /// A version 0 `sidx` for track 1 at timescale 1000 whose references
    /// are `(points at a further sidx, referenced size, duration)`.
    fn sidx_box(ept: u32, refs: &[(bool, u32, u32)]) -> Vec<u8> {
        let mut body = [0u32, 1, 1000, ept, 0]
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect::<Vec<u8>>();
        body.extend_from_slice(&(refs.len() as u32).to_be_bytes());
        for &(index, size, duration) in refs {
            let reference = size | if index { 0x8000_0000 } else { 0 };
            for word in [reference, duration, 0x9000_0000] {
                body.extend_from_slice(&word.to_be_bytes());
            }
        }
        boxed(b"sidx", &body)
    }

    /// One fragment: an empty `moof` and a small `mdat`.
    fn segment() -> Vec<u8> {
        let mut out = boxed(b"moof", &[0; 8]);
        out.extend(boxed(b"mdat", &[0; 16]));
        out
    }

    fn fragmented(bytes: Vec<u8>) -> Mp4Demuxer {
        let file_len = bytes.len() as u64;
        Mp4Demuxer {
            source: Box::new(std::io::Cursor::new(bytes)),
            video: None,
            audio: None,
            duration_us: None,
            fragments: Some(Fragments {
                trex: Vec::new(),
                file_len,
                next_box: 0,
                ended: false,
                index: FragmentIndex::default(),
                audio_index: FragmentIndex::default(),
                next_dts: Vec::new(),
            }),
        }
    }

    #[test]
    fn hierarchical_sidx_is_followed_to_every_fragment() {
        let seg = segment();
        let seg_len = seg.len() as u32;
        let child = |ept| sidx_box(ept, &[(false, seg_len, 1000), (false, seg_len, 1000)]);
        let child_len = child(0).len() as u32;
        let subtree = child_len + 2 * seg_len;
        let mut file = sidx_box(0, &[(true, subtree, 2000), (true, subtree, 2000)]);
        let top_len = file.len() as u64;
        for ept in [0, 2000] {
            file.extend(child(ept));
            file.extend(&seg);
            file.extend(&seg);
        }
        let mut demuxer = fragmented(file);
        let (entries, duration) = demuxer.sidx_index(0, 1000).unwrap().unwrap();
        let (child_len, seg_len) = (u64::from(child_len), u64::from(seg_len));
        let first = top_len + child_len;
        let second = first + 2 * seg_len + child_len;
        let expected = [
            (0, first),
            (1000, first + seg_len),
            (2000, second),
            (3000, second + seg_len),
        ]
        .map(|(time, moof)| fragment::IndexEntry { time, moof });
        assert_eq!(entries, expected);
        assert_eq!(duration, 4000);
    }

    #[test]
    fn sidx_trees_too_deep_or_pointing_elsewhere_are_rejected() {
        // Each level points at the next sidx, one level more than allowed.
        let seg = segment();
        let mut levels = vec![sidx_box(0, &[(false, seg.len() as u32, 1000)])];
        for _ in 0..MAX_SIDX_DEPTH {
            let below: u32 = levels.iter().map(|l| l.len() as u32).sum::<u32>() + seg.len() as u32;
            levels.insert(0, sidx_box(0, &[(true, below, 1000)]));
        }
        let mut file = levels.concat();
        file.extend(&seg);
        assert_eq!(fragmented(file).sidx_index(0, 1000).unwrap(), None);

        // Sizes that skip part of the file land inside media data.
        let mut file = sidx_box(0, &[(false, 12, 1000), (false, 12, 1000)]);
        file.extend(segment());
        file.extend(segment());
        assert_eq!(fragmented(file).sidx_index(0, 1000).unwrap(), None);
    }

    #[test]
    fn chunk_runs_expand_to_one_offset_per_sample() {
        let tables = Tables {
            stts: vec![(5, 100)],
            stsc: runs(&[(1, 2), (3, 1)]),
            sizes: vec![10, 20, 30, 40, 50],
            chunk_offsets: vec![1000, 2000, 3000],
            ..Tables::default()
        };
        let samples = build_samples(&tables).unwrap();
        let located: Vec<_> = samples.iter().map(|s| (s.offset, s.size)).collect();
        assert_eq!(
            located,
            [(1000, 10), (1010, 20), (2000, 30), (2030, 40), (3000, 50)]
        );
    }

    #[test]
    fn decode_times_accumulate_across_stts_runs() {
        let tables = Tables {
            stts: vec![(2, 100), (1, 50)],
            stsc: runs(&[(1, 3)]),
            sizes: vec![1, 1, 1],
            chunk_offsets: vec![0],
            ..Tables::default()
        };
        let samples = build_samples(&tables).unwrap();
        assert_eq!(samples[0], sample(0, 1, 0, 100, true));
        assert_eq!(samples[1], sample(1, 1, 100, 100, true));
        assert_eq!(samples[2], sample(2, 1, 200, 50, true));
    }

    #[test]
    fn composition_offsets_may_be_negative() {
        let tables = Tables {
            stts: vec![(2, 10)],
            ctts: vec![(1, 20), (1, -10)],
            stsc: runs(&[(1, 2)]),
            sizes: vec![1, 1],
            chunk_offsets: vec![0],
            ..Tables::default()
        };
        let offsets: Vec<_> = build_samples(&tables)
            .unwrap()
            .iter()
            .map(|s| s.cts_offset)
            .collect();
        assert_eq!(offsets, [20, -10]);
    }

    #[test]
    fn only_listed_samples_are_sync_when_stss_is_present() {
        let tables = Tables {
            stts: vec![(3, 1)],
            stsc: runs(&[(1, 3)]),
            sizes: vec![1, 1, 1],
            chunk_offsets: vec![0],
            sync: Some(vec![1, 3]),
            ..Tables::default()
        };
        let sync: Vec<_> = build_samples(&tables)
            .unwrap()
            .iter()
            .map(|s| s.sync)
            .collect();
        assert_eq!(sync, [true, false, true]);
    }

    #[test]
    fn sizes_shorter_than_the_chunk_table_end_the_samples() {
        let tables = Tables {
            stts: vec![(9, 1)],
            stsc: runs(&[(1, 4)]),
            sizes: vec![1, 1, 1, 1, 1],
            chunk_offsets: vec![0, 100, 200],
            ..Tables::default()
        };
        assert_eq!(build_samples(&tables).unwrap().len(), 5);
    }

    #[test]
    fn edit_shift_matching_the_composition_delay_keeps_the_origin() {
        let mut first = sample(0, 1, 0, 10, true);
        first.cts_offset = 20;
        let mut second = sample(1, 1, 10, 10, false);
        second.cts_offset = 30;
        assert_eq!(timeline_origin(&[first, second], 20), 20);
    }

    #[test]
    fn samples_before_the_edit_move_the_origin_back() {
        let samples = [
            sample(0, 1, 0, 10, true),
            sample(1, 1, 10, 10, false),
            sample(2, 1, 20, 10, false),
        ];
        let origin = timeline_origin(&samples, 15);
        assert_eq!(origin, 0);
        let start = 15 - origin;
        let shown: Vec<_> = samples
            .iter()
            .map(|s| s.presentation_ticks() - origin)
            .filter(|&ticks| ticks >= start)
            .collect();
        assert_eq!(shown, [20]);
    }

    #[test]
    fn all_ones_mdhd_duration_is_unknown() {
        let mut v0 = vec![0; 12];
        v0.extend_from_slice(&1000u32.to_be_bytes());
        v0.extend_from_slice(&u32::MAX.to_be_bytes());
        assert_eq!(parse_mdhd(&v0), Some((1000, None)));
        let mut v1 = vec![1, 0, 0, 0];
        v1.extend_from_slice(&[0; 16]);
        v1.extend_from_slice(&90_000u32.to_be_bytes());
        v1.extend_from_slice(&u64::MAX.to_be_bytes());
        assert_eq!(parse_mdhd(&v1), Some((90_000, None)));
        v0[16..20].copy_from_slice(&5000u32.to_be_bytes());
        assert_eq!(parse_mdhd(&v0), Some((1000, Some(5000))));
    }

    #[test]
    fn decode_duration_sums_the_stts_runs() {
        let tables = Tables {
            stts: vec![(2, 100), (3, 50)],
            ..Tables::default()
        };
        assert_eq!(tables.decode_duration(), 350);
    }

    #[test]
    fn unsorted_chunk_runs_are_rejected() {
        let tables = Tables {
            stsc: runs(&[(1, 1), (5, 1), (3, 1)]),
            sizes: vec![1; 4],
            chunk_offsets: vec![0; 4],
            ..Tables::default()
        };
        assert!(build_samples(&tables).is_err());
    }

    #[test]
    fn chunk_runs_not_starting_at_chunk_one_are_rejected() {
        let tables = Tables {
            stsc: runs(&[(2, 1)]),
            sizes: vec![1; 2],
            chunk_offsets: vec![0; 2],
            ..Tables::default()
        };
        assert!(build_samples(&tables).is_err());
    }

    #[test]
    fn chunks_using_a_second_sample_description_are_rejected() {
        let mut stsc = runs(&[(1, 1), (2, 1)]);
        stsc[1].description_index = 2;
        let tables = Tables {
            stsc,
            sizes: vec![1; 2],
            chunk_offsets: vec![0; 2],
            ..Tables::default()
        };
        let error = build_samples(&tables).unwrap_err();
        assert!(
            error.to_string().contains("sample description 2"),
            "{error}"
        );
    }

    #[test]
    fn stsc_entries_keep_their_description_index() {
        let body = [0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 3];
        assert_eq!(
            chunk_runs(&body, 1),
            [ChunkRun {
                first_chunk: 1,
                samples_per_chunk: 4,
                description_index: 3,
            }]
        );
    }

    #[test]
    fn many_empty_chunks_against_many_runs_finish_quickly() {
        let chunks = 1 << 20;
        let tables = Tables {
            stsc: runs(&(1..=chunks).map(|chunk| (chunk, 0)).collect::<Vec<_>>()),
            sizes: vec![1],
            chunk_offsets: vec![0; chunks as usize],
            ..Tables::default()
        };
        let started = std::time::Instant::now();
        assert!(build_samples(&tables).unwrap().is_empty());
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }

    #[test]
    fn declared_sample_count_is_capped() {
        let counted = |count: u32| {
            let mut body = vec![0; 4];
            body.extend_from_slice(&count.to_be_bytes());
            body
        };
        let mut stsz = vec![0; 4];
        stsz.extend_from_slice(&1u32.to_be_bytes());
        stsz.extend_from_slice(&u32::MAX.to_be_bytes());
        let mut stbl = boxed(b"stts", &counted(0));
        stbl.extend(boxed(b"stsc", &counted(0)));
        stbl.extend(boxed(b"stsz", &stsz));
        stbl.extend(boxed(b"stco", &counted(0)));
        let tables = parse_tables(&stbl).unwrap();
        assert_eq!(tables.sizes.len(), MAX_SAMPLES);
    }

    #[test]
    fn sample_offsets_near_the_end_of_u64_do_not_overflow() {
        let tables = Tables {
            stts: vec![(2, u32::MAX)],
            stsc: runs(&[(1, 2)]),
            sizes: vec![u32::MAX, u32::MAX],
            chunk_offsets: vec![u64::MAX - 1],
            ..Tables::default()
        };
        let samples = build_samples(&tables).unwrap();
        assert_eq!(samples[1].offset, u64::MAX);
    }

    #[test]
    fn edit_list_count_is_bounded_by_its_entries() {
        let mut elst = vec![0, 0, 0, 0];
        elst.extend_from_slice(&u32::MAX.to_be_bytes());
        elst.extend_from_slice(&[0, 0, 0, 5, 0xFF, 0xFF, 0xFF, 0xFF, 0, 1, 0, 0]);
        assert_eq!(parse_elst(&elst), None);
    }

    #[test]
    fn extended_size_boxes_are_read() {
        let mut data = 1u32.to_be_bytes().to_vec();
        data.extend_from_slice(b"free");
        data.extend_from_slice(&19u64.to_be_bytes());
        data.extend_from_slice(&[7, 8, 9]);
        let found: Vec<_> = Boxes(&data).collect();
        assert_eq!(found, [(*b"free", &[7, 8, 9][..])]);
    }

    #[test]
    fn box_larger_than_its_parent_ends_the_walk() {
        let mut data = boxed(b"free", &[1]);
        data.extend_from_slice(&100u32.to_be_bytes());
        data.extend_from_slice(b"moov");
        assert_eq!(Boxes(&data).count(), 1);
    }

    #[test]
    fn nclx_range_flag_selects_full_range() {
        let mut colr = b"nclx".to_vec();
        colr.extend_from_slice(&[0, 1, 0, 1, 0, 1, 0x80]);
        let color = parse_colr(&colr).unwrap();
        assert_eq!(color.matrix_coefficients, 1);
        assert!(color.full_range);
    }

    #[test]
    fn nclc_is_always_limited_range() {
        let mut colr = b"nclc".to_vec();
        colr.extend_from_slice(&[0, 1, 0, 1, 0, 6]);
        let color = parse_colr(&colr).unwrap();
        assert_eq!(color.matrix_coefficients, 6);
        assert!(!color.full_range);
    }

    #[test]
    fn edit_list_skips_the_leading_empty_edit() {
        let mut elst = vec![0, 0, 0, 0, 0, 0, 0, 2];
        elst.extend_from_slice(&[0, 0, 0, 5, 0xFF, 0xFF, 0xFF, 0xFF, 0, 1, 0, 0]);
        elst.extend_from_slice(&[0, 0, 0, 9, 0, 0, 2, 0, 0, 1, 0, 0]);
        assert_eq!(parse_elst(&elst), Some(512));
    }

    #[test]
    fn container_sniffing_accepts_ftyp_and_rejects_ebml() {
        assert!(is_mp4(&[0, 0, 0, 0x20, b'f', b't', b'y', b'p']));
        assert!(!is_mp4(&[0x1A, 0x45, 0xDF, 0xA3, 0, 0, 0, 0]));
    }
}
