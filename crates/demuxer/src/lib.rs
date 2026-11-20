mod mp4;
mod nal;

#[cfg(feature = "matroska")]
mod matroska;

use std::fs::File;
use std::io::Read;
use std::path::Path;

use ferricast_core::{Codec, EncodedFrame, FerricastError, Result};

pub use mp4::Mp4Demuxer;

#[cfg(feature = "matroska")]
pub use matroska::MatroskaDemuxer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    pub matrix_coefficients: u8,
    pub full_range: bool,
}

#[derive(Debug, Clone)]
pub struct VideoStream {
    pub codec: Codec,
    pub width: u32,
    pub height: u32,
    pub color: Option<Color>,
    pub duration_us: Option<u64>,
}

pub enum Demuxer {
    Mp4(Mp4Demuxer),
    #[cfg(feature = "matroska")]
    Matroska(Box<MatroskaDemuxer>),
}

impl Demuxer {
    pub fn open(path: &Path) -> Result<Self> {
        let mut head = [0; 8];
        File::open(path)?.read_exact(&mut head)?;
        if mp4::is_mp4(&head) {
            return Ok(Self::Mp4(Mp4Demuxer::open(path)?));
        }
        #[cfg(feature = "matroska")]
        if matroska::is_matroska(&head) {
            return Ok(Self::Matroska(Box::new(MatroskaDemuxer::open(path)?)));
        }
        Err(FerricastError::Demux(format!(
            "unrecognised container in {}",
            path.display()
        )))
    }

    pub fn video(&self) -> &VideoStream {
        match self {
            Demuxer::Mp4(d) => d.video(),
            #[cfg(feature = "matroska")]
            Demuxer::Matroska(d) => d.video(),
        }
    }

    pub fn next_frame(&mut self) -> Result<Option<EncodedFrame>> {
        match self {
            Demuxer::Mp4(d) => d.next_frame(),
            #[cfg(feature = "matroska")]
            Demuxer::Matroska(d) => d.next_frame(),
        }
    }

    pub fn rewind(&mut self) -> Result<()> {
        match self {
            Demuxer::Mp4(d) => d.rewind(),
            #[cfg(feature = "matroska")]
            Demuxer::Matroska(d) => d.rewind(),
        }
    }
}

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
