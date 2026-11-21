//! Fragmented MP4 (ISO/IEC 14496-12 §8.8): movie fragments (`moof` with
//! `traf`/`tfhd`/`tfdt`/`trun`) whose samples default to the `trex`
//! boxes of the `moov`, and the indexes that locate fragments by time:
//! the segment index (`sidx`) and the movie fragment random access box
//! (`mfra`/`tfra`).
//!
//! Everything here reads untrusted bytes: counts are capped and every
//! read is bounds-checked.

use super::{Boxes, Sample, be32, be64, child};

/// Samples taken from one `trun` at most.
const MAX_TRUN_SAMPLES: usize = 1 << 16;
/// Index entries kept at most, from any source.
pub(super) const MAX_INDEX: usize = 1 << 18;

/// `tfhd` flags.
const BASE_DATA_OFFSET: u32 = 0x01;
const SAMPLE_DESCRIPTION_INDEX: u32 = 0x02;
const DEFAULT_DURATION: u32 = 0x08;
const DEFAULT_SIZE: u32 = 0x10;
const DEFAULT_FLAGS: u32 = 0x20;
/// `trun` flags.
const DATA_OFFSET: u32 = 0x01;
const FIRST_SAMPLE_FLAGS: u32 = 0x04;
const SAMPLE_DURATION: u32 = 0x100;
const SAMPLE_SIZE: u32 = 0x200;
const SAMPLE_FLAGS: u32 = 0x400;
const SAMPLE_CTS: u32 = 0x800;
/// `sample_is_non_sync_sample` in sample flags.
const NON_SYNC: u32 = 0x1_0000;

/// Per-track sample defaults from a `trex` box.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Trex {
    pub track_id: u32,
    duration: u32,
    size: u32,
    flags: u32,
}

pub(super) fn parse_trex(mvex: &[u8]) -> Vec<Trex> {
    Boxes(mvex)
        .filter(|(kind, _)| kind == b"trex")
        .filter_map(|(_, body)| {
            Some(Trex {
                track_id: be32(body, 4)?,
                duration: be32(body, 12)?,
                size: be32(body, 16)?,
                flags: be32(body, 20)?,
            })
        })
        .collect()
}

/// `fragment_duration` of the `mehd` box, in movie timescale units.
pub(super) fn parse_mehd(mvex: &[u8]) -> Option<u64> {
    let mehd = child(mvex, b"mehd")?;
    match *mehd.first()? {
        1 => be64(mehd, 4),
        _ => be32(mehd, 4).map(u64::from),
    }
    .filter(|&d| d > 0)
}

/// The samples one `traf` adds to a track.
#[derive(Debug, Default)]
pub(super) struct TrackRun {
    pub track_id: u32,
    pub samples: Vec<Sample>,
}

/// Parses the body of a `moof` found at file offset `moof_offset`.
/// `next_dts` gives each track's decode time where the previous fragment
/// ended, for fragments without a `tfdt`.
pub(super) fn parse_moof(
    moof: &[u8],
    moof_offset: u64,
    trex: &[Trex],
    next_dts: impl Fn(u32) -> u64,
) -> Vec<TrackRun> {
    Boxes(moof)
        .filter(|(kind, _)| kind == b"traf")
        .filter_map(|(_, traf)| parse_traf(traf, moof_offset, trex, &next_dts))
        .collect()
}

fn parse_traf(
    traf: &[u8],
    moof_offset: u64,
    trex: &[Trex],
    next_dts: &impl Fn(u32) -> u64,
) -> Option<TrackRun> {
    let tfhd = child(traf, b"tfhd")?;
    let flags = be32(tfhd, 0)? & 0x00FF_FFFF;
    let track_id = be32(tfhd, 4)?;
    let defaults = trex
        .iter()
        .find(|t| t.track_id == track_id)
        .copied()
        .unwrap_or_default();
    let mut at = 8;
    let mut field64 = |present: bool| -> Option<Option<u64>> {
        if !present {
            return Some(None);
        }
        let value = be64(tfhd, at)?;
        at += 8;
        Some(Some(value))
    };
    let base = field64(flags & BASE_DATA_OFFSET != 0)?;
    let mut field32 = |present: bool| -> Option<Option<u32>> {
        if !present {
            return Some(None);
        }
        let value = be32(tfhd, at)?;
        at += 4;
        Some(Some(value))
    };
    field32(flags & SAMPLE_DESCRIPTION_INDEX != 0)?;
    let duration = field32(flags & DEFAULT_DURATION != 0)?.unwrap_or(defaults.duration);
    let size = field32(flags & DEFAULT_SIZE != 0)?.unwrap_or(defaults.size);
    let sample_flags = field32(flags & DEFAULT_FLAGS != 0)?.unwrap_or(defaults.flags);
    // Without an explicit base, offsets count from the `moof` (the
    // `default-base-is-moof` behaviour, also what single-track fragments
    // written without it resolve to).
    let base = base.unwrap_or(moof_offset);

    let mut dts = match child(traf, b"tfdt") {
        Some(tfdt) => match *tfdt.first()? {
            1 => be64(tfdt, 4)?,
            _ => u64::from(be32(tfdt, 4)?),
        },
        None => next_dts(track_id),
    };
    let mut samples = Vec::new();
    let mut data_end = base;
    for (_, trun) in Boxes(traf).filter(|(kind, _)| kind == b"trun") {
        let version = *trun.first()?;
        let flags = be32(trun, 0)? & 0x00FF_FFFF;
        let count = (be32(trun, 4)? as usize).min(MAX_TRUN_SAMPLES);
        let mut at = 8;
        let mut offset = data_end;
        if flags & DATA_OFFSET != 0 {
            let delta = i64::from(be32(trun, at)? as i32);
            offset = base.checked_add_signed(delta)?;
            at += 4;
        }
        let mut first_flags = None;
        if flags & FIRST_SAMPLE_FLAGS != 0 {
            first_flags = Some(be32(trun, at)?);
            at += 4;
        }
        for index in 0..count {
            let mut next = || -> Option<u32> {
                let value = be32(trun, at)?;
                at += 4;
                Some(value)
            };
            let sample_duration = if flags & SAMPLE_DURATION != 0 {
                next()?
            } else {
                duration
            };
            let sample_size = if flags & SAMPLE_SIZE != 0 {
                next()?
            } else {
                size
            };
            let mut this_flags = if flags & SAMPLE_FLAGS != 0 {
                next()?
            } else {
                sample_flags
            };
            if index == 0
                && let Some(first) = first_flags
            {
                this_flags = first;
            }
            let cts_offset = if flags & SAMPLE_CTS != 0 {
                let raw = next()?;
                if version == 0 {
                    i32::try_from(raw).unwrap_or(i32::MAX)
                } else {
                    raw as i32
                }
            } else {
                0
            };
            samples.push(Sample {
                offset,
                size: sample_size,
                dts,
                duration: sample_duration,
                cts_offset,
                sync: this_flags & NON_SYNC == 0,
            });
            offset = offset.saturating_add(u64::from(sample_size));
            dts = dts.saturating_add(u64::from(sample_duration));
        }
        data_end = offset;
    }
    Some(TrackRun { track_id, samples })
}

/// A fragment located by time: the presentation time (in the track's
/// timescale, before any edit) of its first sample and the file offset
/// of its `moof`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct IndexEntry {
    pub time: u64,
    pub moof: u64,
}

/// A `sidx` box at file offset `end` (where the box ends): its entries
/// converted to `timescale`, and the total duration it covers in that
/// timescale. Hierarchical indexes (references to further `sidx` boxes)
/// are not followed: `None`.
pub(super) fn parse_sidx(sidx: &[u8], end: u64, timescale: u32) -> Option<(Vec<IndexEntry>, u64)> {
    let version = *sidx.first()?;
    let sidx_scale = u64::from(be32(sidx, 8)?.max(1));
    let (mut time, first_offset, mut at) = if version == 0 {
        (u64::from(be32(sidx, 12)?), u64::from(be32(sidx, 16)?), 20)
    } else {
        (be64(sidx, 12)?, be64(sidx, 20)?, 28)
    };
    let count = usize::from(u16::from_be_bytes(
        sidx.get(at + 2..at + 4)?.try_into().ok()?,
    ));
    at += 4;
    let convert = |t: u64| (u128::from(t) * u128::from(timescale) / u128::from(sidx_scale)) as u64;
    let mut offset = end.checked_add(first_offset)?;
    let mut entries = Vec::with_capacity(count.min(MAX_INDEX));
    let mut total = 0u64;
    for _ in 0..count.min(MAX_INDEX) {
        let reference = be32(sidx, at)?;
        let duration = u64::from(be32(sidx, at + 4)?);
        at += 12;
        if reference & 0x8000_0000 != 0 {
            return None;
        }
        entries.push(IndexEntry {
            time: convert(time),
            moof: offset,
        });
        offset = offset.checked_add(u64::from(reference & 0x7FFF_FFFF))?;
        time = time.saturating_add(duration);
        total = total.saturating_add(duration);
    }
    Some((entries, convert(total)))
}

/// The `tfra` entries of `track_id` in an `mfra` body.
pub(super) fn parse_mfra(mfra: &[u8], track_id: u32) -> Option<Vec<IndexEntry>> {
    let tfra = Boxes(mfra)
        .filter(|(kind, _)| kind == b"tfra")
        .map(|(_, body)| body)
        .find(|body| be32(body, 4) == Some(track_id))?;
    let version = *tfra.first()?;
    let sizes = be32(tfra, 8)?;
    let traf_len = ((sizes >> 4) & 3) as usize + 1;
    let trun_len = ((sizes >> 2) & 3) as usize + 1;
    let sample_len = (sizes & 3) as usize + 1;
    let count = (be32(tfra, 12)? as usize).min(MAX_INDEX);
    let mut at = 16;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let (time, moof) = if version == 1 {
            let pair = (be64(tfra, at)?, be64(tfra, at + 8)?);
            at += 16;
            pair
        } else {
            let pair = (u64::from(be32(tfra, at)?), u64::from(be32(tfra, at + 4)?));
            at += 8;
            pair
        };
        at += traf_len + trun_len + sample_len;
        if tfra.len() < at {
            return None;
        }
        entries.push(IndexEntry { time, moof });
    }
    Some(entries)
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

    fn words(values: &[u32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_be_bytes()).collect()
    }

    #[test]
    fn truns_resolve_offsets_times_and_sync_flags() {
        // tfhd: track 1, default-base-is-moof, default duration 512.
        let tfhd = boxed(b"tfhd", &words(&[0x02_0008, 1, 512]));
        let tfdt = boxed(b"tfdt", &words(&[0, 9000]));
        // trun: data offset 100, first-sample flags, per-sample size and
        // cts; two samples.
        let trun = boxed(
            b"trun",
            &words(&[
                DATA_OFFSET | FIRST_SAMPLE_FLAGS | SAMPLE_SIZE | SAMPLE_CTS,
                2,
                100,
                0,
                10,
                1024,
                20,
                0,
            ]),
        );
        let mut traf = tfhd;
        traf.extend(tfdt);
        traf.extend(trun);
        let moof = boxed(b"traf", &traf);
        let trex = [Trex {
            track_id: 1,
            duration: 0,
            size: 0,
            flags: NON_SYNC,
        }];
        let runs = parse_moof(&moof, 5000, &trex, |_| 0);
        assert_eq!(runs.len(), 1);
        let samples = &runs[0].samples;
        assert_eq!(samples.len(), 2);
        assert_eq!((samples[0].offset, samples[0].size), (5100, 10));
        assert_eq!((samples[1].offset, samples[1].size), (5110, 20));
        assert_eq!((samples[0].dts, samples[1].dts), (9000, 9512));
        assert_eq!(samples[0].cts_offset, 1024);
        assert!(samples[0].sync, "first-sample flags mark a sync sample");
        assert!(!samples[1].sync, "trex default flags say non-sync");
    }

    #[test]
    fn huge_trun_counts_are_capped_by_the_data() {
        let tfhd = boxed(b"tfhd", &words(&[0, 1]));
        let trun = boxed(b"trun", &words(&[SAMPLE_SIZE, u32::MAX, 7]));
        let mut traf = tfhd;
        traf.extend(trun);
        let runs = parse_moof(&boxed(b"traf", &traf), 0, &[], |_| 0);
        // The table ends after one entry: the traf is dropped rather than
        // read past its end.
        assert!(runs.is_empty());
    }

    #[test]
    fn sidx_entries_are_converted_and_located() {
        // version 0, reference 1, timescale 1000, EPT 0, first offset 0,
        // two references of 300 and 400 bytes, 2 s each.
        let mut body = words(&[0, 1, 1000, 0, 0]);
        body.extend_from_slice(&[0, 0, 0, 2]);
        body.extend(words(&[300, 2000, 0x9000_0000, 400, 2000, 0x9000_0000]));
        let (entries, total) = parse_sidx(&body, 1000, 90_000).unwrap();
        assert_eq!(
            entries,
            [
                IndexEntry {
                    time: 0,
                    moof: 1000
                },
                IndexEntry {
                    time: 180_000,
                    moof: 1300
                }
            ]
        );
        assert_eq!(total, 360_000);
    }

    #[test]
    fn tfra_entries_are_read() {
        let mut tfra = words(&[0, 2, 0, 1]);
        tfra.extend(words(&[3000, 777]));
        tfra.extend_from_slice(&[1, 1, 1]);
        let mfra = boxed(b"tfra", &tfra);
        assert_eq!(
            parse_mfra(&mfra, 2),
            Some(vec![IndexEntry {
                time: 3000,
                moof: 777
            }])
        );
        assert_eq!(parse_mfra(&mfra, 1), None);
    }
}
