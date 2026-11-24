//! Works around how `matroska-demuxer` 0.8 seeks with Cues.
//!
//! When a cue point carries a `CueRelativePosition`, the library adds it
//! to the data offset of the *first* cluster instead of the cluster the
//! cue names, so any seek past the first cluster lands in the middle of
//! unrelated data. Without the relative position it seeks to the cue's
//! (absolute) cluster position, which is correct. [`CueMask`] wraps the
//! source and turns every `CueRelativePosition` element into a `Void`
//! element of the same size as the library reads it; the element IDs are
//! both one byte long, so nothing else moves.

use std::io::{self, Read, Seek, SeekFrom};

use crate::BoxedSource;

const SEGMENT: u32 = 0x1853_8067;
const SEEK_HEAD: u32 = 0x114D_9B74;
const SEEK: u32 = 0x4DBB;
const SEEK_ID: u32 = 0x53AB;
const SEEK_POSITION: u32 = 0x53AC;
const CUES: u32 = 0x1C53_BB6B;
const CUE_POINT: u32 = 0xBB;
const CUE_TRACK_POSITIONS: u32 = 0xB7;
const CUE_RELATIVE_POSITION: u8 = 0xF0;
const VOID: u8 = 0xEC;
/// Largest `Cues` element walked. An hour of video with a cue per second
/// takes well under a megabyte.
const MAX_CUES: u64 = 32 << 20;
/// Largest `SeekHead` walked; real ones list a handful of elements.
const MAX_SEEK_HEAD: u64 = 64 << 10;

/// The source, with every `CueRelativePosition` ID byte read as `Void`.
pub(super) struct CueMask {
    inner: BoxedSource,
    /// Absolute offsets of the ID bytes to mask, sorted.
    masked: Vec<u64>,
    position: u64,
}

impl CueMask {
    /// Locates the `Cues` through the first `SeekHead` and records the
    /// relative positions to mask. A file whose `Cues` cannot be found or
    /// parsed is passed through unchanged.
    pub(super) fn new(mut inner: BoxedSource) -> io::Result<Self> {
        let masked = relative_positions(&mut inner).unwrap_or_default();
        inner.seek(SeekFrom::Start(0))?;
        Ok(Self {
            inner,
            masked,
            position: 0,
        })
    }
}

impl Read for CueMask {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        let start = self.position;
        let end = start + n as u64;
        let first = self.masked.partition_point(|&at| at < start);
        for &at in self.masked[first..].iter().take_while(|&&at| at < end) {
            if let Some(byte) = usize::try_from(at - start)
                .ok()
                .and_then(|i| buf.get_mut(i))
            {
                *byte = VOID;
            }
        }
        self.position = end;
        Ok(n)
    }
}

impl Seek for CueMask {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.position = self.inner.seek(pos)?;
        Ok(self.position)
    }
}

/// An EBML element header: where it starts, its ID, data offset and data size (`None` when
/// the size is unknown).
struct Header {
    start: u64,
    id: u32,
    data: u64,
    size: Option<u64>,
}

fn read_header(r: &mut BoxedSource, at: u64) -> io::Result<Header> {
    r.seek(SeekFrom::Start(at))?;
    let mut first = [0u8; 1];
    r.read_exact(&mut first)?;
    let id_len = first[0].leading_zeros() as usize + 1;
    if id_len > 4 {
        return Err(invalid("element ID longer than four bytes"));
    }
    let mut id = u32::from(first[0]);
    for _ in 1..id_len {
        r.read_exact(&mut first)?;
        id = (id << 8) | u32::from(first[0]);
    }
    r.read_exact(&mut first)?;
    let size_len = first[0].leading_zeros() as usize + 1;
    if size_len > 8 {
        return Err(invalid("element size longer than eight bytes"));
    }
    let mut size = u64::from(first[0]) & (0xFF >> size_len);
    let mut all_ones = size == 0xFF >> size_len;
    for _ in 1..size_len {
        r.read_exact(&mut first)?;
        size = (size << 8) | u64::from(first[0]);
        all_ones &= first[0] == 0xFF;
    }
    Ok(Header {
        start: at,
        id,
        data: at + (id_len + size_len) as u64,
        size: (!all_ones).then_some(size),
    })
}

fn invalid(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what.to_string())
}

/// Calls `f` with each child header of the element whose data spans
/// `data..data + size`.
fn children(
    r: &mut BoxedSource,
    data: u64,
    size: u64,
    mut f: impl FnMut(&mut BoxedSource, &Header) -> io::Result<()>,
) -> io::Result<()> {
    let end = data.saturating_add(size);
    let mut at = data;
    while at < end {
        let child = read_header(r, at)?;
        let child_size = child.size.ok_or_else(|| invalid("child of unknown size"))?;
        f(r, &child)?;
        at = child.data.saturating_add(child_size);
    }
    Ok(())
}

fn read_uint(r: &mut BoxedSource, header: &Header) -> io::Result<u64> {
    let size = header.size.unwrap_or(0);
    if size > 8 {
        return Err(invalid("integer longer than eight bytes"));
    }
    r.seek(SeekFrom::Start(header.data))?;
    let mut value = 0u64;
    let mut byte = [0u8; 1];
    for _ in 0..size {
        r.read_exact(&mut byte)?;
        value = (value << 8) | u64::from(byte[0]);
    }
    Ok(value)
}

fn relative_positions(r: &mut BoxedSource) -> io::Result<Vec<u64>> {
    let ebml = read_header(r, 0)?;
    let ebml_size = ebml
        .size
        .ok_or_else(|| invalid("EBML header of unknown size"))?;
    let segment = read_header(r, ebml.data + ebml_size)?;
    if segment.id != SEGMENT {
        return Err(invalid("no Segment after the EBML header"));
    }
    let seek_head = read_header(r, segment.data)?;
    let seek_head_size = seek_head.size.unwrap_or(0);
    if seek_head.id != SEEK_HEAD || seek_head_size > MAX_SEEK_HEAD {
        return Ok(Vec::new());
    }
    let mut cues_at = None;
    children(r, seek_head.data, seek_head_size, |r, seek| {
        if seek.id != SEEK {
            return Ok(());
        }
        let (mut id, mut position) = (None, None);
        children(r, seek.data, seek.size.unwrap_or(0), |r, field| {
            match field.id {
                SEEK_ID => id = Some(read_uint(r, field)?),
                SEEK_POSITION => position = Some(read_uint(r, field)?),
                _ => {}
            }
            Ok(())
        })?;
        if id == Some(u64::from(CUES)) {
            cues_at = position.map(|p| segment.data + p);
        }
        Ok(())
    })?;
    let Some(cues_at) = cues_at else {
        return Ok(Vec::new());
    };
    let cues = read_header(r, cues_at)?;
    let cues_size = cues.size.unwrap_or(0);
    if cues.id != CUES || cues_size > MAX_CUES {
        return Ok(Vec::new());
    }
    let mut masked = Vec::new();
    children(r, cues.data, cues_size, |r, point| {
        if point.id != CUE_POINT {
            return Ok(());
        }
        children(r, point.data, point.size.unwrap_or(0), |r, positions| {
            if positions.id != CUE_TRACK_POSITIONS {
                return Ok(());
            }
            children(
                r,
                positions.data,
                positions.size.unwrap_or(0),
                |_, field| {
                    if field.id == u32::from(CUE_RELATIVE_POSITION) {
                        masked.push(field.start);
                    }
                    Ok(())
                },
            )
        })
    })?;
    masked.sort_unstable();
    Ok(masked)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn element(id: &[u8], body: &[u8]) -> Vec<u8> {
        let mut out = id.to_vec();
        out.push(0x80 | u8::try_from(body.len()).unwrap());
        out.extend_from_slice(body);
        out
    }

    /// EBML header, then a `Segment` whose `SeekHead` points at `Cues` with
    /// one cue point carrying a relative position.
    fn file() -> (Vec<u8>, usize) {
        let ebml = element(&[0x1A, 0x45, 0xDF, 0xA3], &element(&[0x42, 0x86], &[1]));
        let positions = [
            element(&[0xF7], &[1]),
            element(&[0xF1], &[0x10]),
            element(&[0xF0], &[0x22]),
        ]
        .concat();
        let point = [element(&[0xB3], &[0]), element(&[0xB7], &positions)].concat();
        let cues = element(&[0x1C, 0x53, 0xBB, 0x6B], &element(&[0xBB], &point));
        // SeekHead of fixed length so the Cues offset is known up front.
        let seek_head_len = 4 + 1 + (2 + 1 + (2 + 1 + 4) + (2 + 1 + 1));
        let seek = [
            element(&[0x53, 0xAB], &[0x1C, 0x53, 0xBB, 0x6B]),
            element(&[0x53, 0xAC], &[u8::try_from(seek_head_len).unwrap()]),
        ]
        .concat();
        let seek_head = element(&[0x11, 0x4D, 0x9B, 0x74], &element(&[0x4D, 0xBB], &seek));
        assert_eq!(seek_head.len(), seek_head_len);
        let body = [seek_head, cues].concat();
        let segment = element(&[0x18, 0x53, 0x80, 0x67], &body);
        let file = [ebml, segment].concat();
        let relative = file
            .windows(3)
            .position(|w| w == [0xF0, 0x81, 0x22])
            .unwrap();
        (file, relative)
    }

    #[test]
    fn relative_positions_read_as_void() {
        let (bytes, relative) = file();
        let mut mask = CueMask::new(Box::new(Cursor::new(bytes.clone()))).unwrap();
        assert_eq!(mask.masked, [relative as u64]);
        let mut read = Vec::new();
        mask.read_to_end(&mut read).unwrap();
        assert_eq!(read[relative], VOID);
        let mut expected = bytes;
        expected[relative] = VOID;
        assert_eq!(read, expected);
    }

    #[test]
    fn partial_reads_mask_only_their_window() {
        let (bytes, relative) = file();
        let mut mask = CueMask::new(Box::new(Cursor::new(bytes))).unwrap();
        mask.seek(SeekFrom::Start(relative as u64)).unwrap();
        let mut one = [0u8; 1];
        mask.read_exact(&mut one).unwrap();
        assert_eq!(one, [VOID]);
        mask.read_exact(&mut one).unwrap();
        assert_eq!(one, [0x81]);
    }

    #[test]
    fn files_without_cues_pass_through() {
        let bytes = vec![0u8; 64];
        let mut mask = CueMask::new(Box::new(Cursor::new(bytes.clone()))).unwrap();
        assert!(mask.masked.is_empty());
        let mut read = Vec::new();
        mask.read_to_end(&mut read).unwrap();
        assert_eq!(read, bytes);
    }
}
