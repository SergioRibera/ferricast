use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use ferricast_core::{Codec, EncodedFrame, FerricastError, Result};
use matroska_demuxer::{Frame, MatrixCoefficients, MatroskaFile, Range, TrackEntry, TrackType};

use crate::nal::{Framing, LengthPrefixed};
use crate::{Color, VideoStream};

const EBML_MAGIC: [u8; 4] = [0x1A, 0x45, 0xDF, 0xA3];

pub struct MatroskaDemuxer {
    file: MatroskaFile<BufReader<File>>,
    stream: VideoStream,
    framing: Framing,
    track: u64,
    scale_ns: u64,
    frame: Frame,
}

impl MatroskaDemuxer {
    pub fn open(path: &Path) -> Result<Self> {
        let file = MatroskaFile::open(BufReader::new(File::open(path)?)).map_err(demux)?;
        let scale_ns = file.info().timestamp_scale().get();
        let (entry, codec, framing) = file
            .tracks()
            .iter()
            .find_map(|entry| {
                Some((entry, codec(entry)?)).and_then(|(e, c)| Some((e, c, framing(e, c)?)))
            })
            .ok_or_else(|| FerricastError::Demux("no supported video track".into()))?;
        let video = entry
            .video()
            .ok_or_else(|| FerricastError::Demux("video track without video settings".into()))?;
        let stream = VideoStream {
            codec,
            width: u32::try_from(video.pixel_width().get()).unwrap_or(0),
            height: u32::try_from(video.pixel_height().get()).unwrap_or(0),
            color: video.colour().and_then(|colour| {
                Some(Color {
                    matrix_coefficients: matrix_code(colour.matrix_coefficients()?),
                    full_range: colour.range() == Some(Range::Full),
                })
            }),
            duration_us: file
                .info()
                .duration()
                .map(|ticks| (ticks * scale_ns as f64 / 1000.0) as u64),
        };
        let track = entry.track_number().get();
        Ok(Self {
            file,
            stream,
            framing,
            track,
            scale_ns,
            frame: Frame::default(),
        })
    }

    pub fn video(&self) -> &VideoStream {
        &self.stream
    }

    pub fn next_frame(&mut self) -> Result<Option<EncodedFrame>> {
        loop {
            if !self.file.next_frame(&mut self.frame).map_err(demux)? {
                return Ok(None);
            }
            if self.frame.track != self.track {
                continue;
            }
            let (data, idr) = self.framing.annex_b(&self.frame.data)?;
            let pts = self.frame.timestamp * self.scale_ns / 1000;
            return Ok(Some(EncodedFrame {
                codec: self.stream.codec,
                data,
                timestamp_us: pts,
                is_keyframe: self.frame.is_keyframe.unwrap_or(false) || idr,
                duration_us: self.frame.duration.map(|d| d * self.scale_ns / 1000),
                pts_dts: (pts, pts),
            }));
        }
    }

    pub fn rewind(&mut self) -> Result<()> {
        self.file.seek(0).map_err(demux)
    }
}

pub(crate) fn is_matroska(head: &[u8]) -> bool {
    head.starts_with(&EBML_MAGIC)
}

fn demux(error: matroska_demuxer::DemuxError) -> FerricastError {
    FerricastError::Demux(error.to_string())
}

fn codec(entry: &TrackEntry) -> Option<Codec> {
    if entry.track_type() != TrackType::Video {
        return None;
    }
    match entry.codec_id() {
        "V_MPEG4/ISO/AVC" => Some(Codec::H264),
        "V_MPEGH/ISO/HEVC" => Some(Codec::H265),
        "V_VP8" => Some(Codec::Vp8),
        "V_VP9" => Some(Codec::Vp9),
        "V_AV1" => Some(Codec::Av1),
        _ => None,
    }
}

fn framing(entry: &TrackEntry, codec: Codec) -> Option<Framing> {
    match codec {
        Codec::H264 => LengthPrefixed::from_avcc(entry.codec_private()?).ok(),
        Codec::H265 => LengthPrefixed::from_hvcc(entry.codec_private()?).ok(),
        _ => return Some(Framing::Raw),
    }
    .map(Framing::LengthPrefixed)
}

fn matrix_code(matrix: MatrixCoefficients) -> u8 {
    match matrix {
        MatrixCoefficients::Identity => 0,
        MatrixCoefficients::Bt709 => 1,
        MatrixCoefficients::Fcc73682 => 4,
        MatrixCoefficients::Bt470bg => 5,
        MatrixCoefficients::Smpte170 => 6,
        MatrixCoefficients::Smpte240 => 7,
        MatrixCoefficients::YCoCg => 8,
        MatrixCoefficients::Bt2020Ncl => 9,
        MatrixCoefficients::Bt2020Cl => 10,
        MatrixCoefficients::SmpteSt2085 => 11,
        MatrixCoefficients::ChromaDerivedNcl => 12,
        MatrixCoefficients::ChromaDerivedCl => 13,
        MatrixCoefficients::Bt2100 => 14,
        MatrixCoefficients::Unknown => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_sniffing_accepts_ebml_and_rejects_mp4() {
        assert!(is_matroska(&[
            0x1A, 0x45, 0xDF, 0xA3, 0x9F, 0x42, 0x86, 0x81
        ]));
        assert!(!is_matroska(&[0, 0, 0, 0x20, b'f', b't', b'y', b'p']));
    }
}
