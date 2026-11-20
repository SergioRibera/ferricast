use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use ferricast_core::{Codec, EncodedFrame, FerricastError, Result};

use crate::nal::{Framing, LengthPrefixed};
use crate::{Color, VideoStream, micros};

const VISUAL_ENTRY_HEADER: usize = 78;
// Upper bound on the samples taken from one track: about 19 hours at 30 fps.
// Sample counts come straight from the file, so without a cap a few bytes of
// `stsz` could ask for gigabytes of sample entries.
const MAX_SAMPLES: usize = 1 << 21;

pub struct Mp4Demuxer {
    file: File,
    stream: VideoStream,
    framing: Framing,
    timescale: u32,
    samples: Vec<Sample>,
    shift: i64,
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

struct Entry {
    codec: Codec,
    width: u32,
    height: u32,
    framing: Framing,
    color: Option<Color>,
}

struct Track {
    entry: Entry,
    timescale: u32,
    duration: u64,
    tables: Tables,
    shift: i64,
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

impl Mp4Demuxer {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let moov = read_moov(&file)?;
        let track = parse_moov(&moov)?;
        let file_len = file.metadata()?.len();
        let mut samples = build_samples(&track.tables)?;
        if samples.is_empty() && child(&moov, b"mvex").is_some() {
            return Err(FerricastError::Demux(
                "fragmented MP4 is not supported".into(),
            ));
        }
        let readable = samples
            .iter()
            .take_while(|s| {
                s.offset
                    .checked_add(u64::from(s.size))
                    .is_some_and(|end| end <= file_len)
            })
            .count();
        samples.truncate(readable);
        Ok(Self {
            file,
            stream: VideoStream {
                codec: track.entry.codec,
                width: track.entry.width,
                height: track.entry.height,
                color: track.entry.color,
                duration_us: Some(micros(track.duration as i64, track.timescale)),
            },
            framing: track.entry.framing,
            timescale: track.timescale,
            samples,
            shift: track.shift,
            next: 0,
        })
    }

    pub fn video(&self) -> &VideoStream {
        &self.stream
    }

    pub fn next_frame(&mut self) -> Result<Option<EncodedFrame>> {
        let Some(sample) = self.samples.get(self.next) else {
            return Ok(None);
        };
        let mut data = vec![0; sample.size as usize];
        self.file.read_exact_at(&mut data, sample.offset)?;
        let (data, idr) = self.framing.annex_b(&data, sample.sync || self.next == 0)?;
        let dts = micros(sample.dts as i64 - self.shift, self.timescale);
        let pts = micros(
            sample.dts as i64 + i64::from(sample.cts_offset) - self.shift,
            self.timescale,
        );
        let frame = EncodedFrame {
            codec: self.stream.codec,
            data,
            timestamp_us: pts,
            is_keyframe: sample.sync || idr,
            duration_us: Some(micros(i64::from(sample.duration), self.timescale)),
            pts_dts: (pts, dts),
        };
        self.next += 1;
        Ok(Some(frame))
    }

    pub fn rewind(&mut self) -> Result<()> {
        self.next = 0;
        Ok(())
    }
}

pub(crate) fn is_mp4(head: &[u8]) -> bool {
    matches!(
        head.get(4..8),
        Some(b"ftyp" | b"moov" | b"mdat" | b"free" | b"wide" | b"skip")
    )
}

fn read_moov(file: &File) -> Result<Vec<u8>> {
    let len = file.metadata()?.len();
    let mut offset = 0;
    while offset + 8 <= len {
        let mut header = [0; 16];
        let available = (len - offset).min(16) as usize;
        file.read_exact_at(&mut header[..available], offset)?;
        let (header_len, size) = match be32(&header, 0).unwrap_or(0) {
            0 => (8, len - offset),
            1 => (16, be64(&header, 8).ok_or_else(truncated)?),
            size => (8, u64::from(size)),
        };
        if size < header_len || size > len - offset {
            return Err(truncated());
        }
        if &header[4..8] == b"moov" {
            let mut body = vec![0; (size - header_len) as usize];
            file.read_exact_at(&mut body, offset + header_len)?;
            return Ok(body);
        }
        offset += size;
    }
    Err(FerricastError::Demux("no moov box".into()))
}

fn truncated() -> FerricastError {
    FerricastError::Demux("truncated MP4 box".into())
}

fn parse_moov(moov: &[u8]) -> Result<Track> {
    Boxes(moov)
        .filter(|(kind, _)| kind == b"trak")
        .find_map(|(_, trak)| parse_track(trak))
        .ok_or_else(|| FerricastError::Demux("no supported video track".into()))
}

fn parse_track(trak: &[u8]) -> Option<Track> {
    let mdia = child(trak, b"mdia")?;
    if child(mdia, b"hdlr")?.get(8..12)? != b"vide" {
        return None;
    }
    let (timescale, duration) = parse_mdhd(child(mdia, b"mdhd")?)?;
    if timescale == 0 {
        return None;
    }
    let stbl = child(child(mdia, b"minf")?, b"stbl")?;
    let stsd = child(stbl, b"stsd")?;
    let (kind, entry) = Boxes(stsd.get(8..)?).next()?;
    Some(Track {
        entry: parse_entry(kind, entry)?,
        timescale,
        duration,
        tables: parse_tables(stbl)?,
        shift: child(trak, b"edts")
            .and_then(|edts| child(edts, b"elst"))
            .and_then(parse_elst)
            .unwrap_or(0),
    })
}

fn parse_mdhd(mdhd: &[u8]) -> Option<(u32, u64)> {
    match *mdhd.first()? {
        1 => Some((be32(mdhd, 20)?, be64(mdhd, 24)?)),
        _ => Some((be32(mdhd, 12)?, u64::from(be32(mdhd, 16)?))),
    }
}

fn parse_entry(kind: [u8; 4], entry: &[u8]) -> Option<Entry> {
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
    Some(Entry {
        codec,
        width: u32::from(be16(entry, 24)?),
        height: u32::from(be16(entry, 26)?),
        framing,
        color,
    })
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
        child(stbl, kind).and_then(|body| {
            Some((body.get(8..)?, (be32(body, 4)? as usize).min(MAX_SAMPLES)))
        })
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
        assert!(error.to_string().contains("sample description 2"), "{error}");
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
