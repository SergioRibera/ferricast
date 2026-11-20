use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use ferricast_core::{Codec, EncodedFrame, FerricastError, Result};
use matroska_demuxer::{
    ContentCompAlgo, ContentEncAlgo, ContentEncodingValue, Frame, MatrixCoefficients,
    MatroskaFile, Range, TrackEntry, TrackType,
};

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
    /// The next frame is the first one after open/rewind and must carry the
    /// parameter sets whatever its NAL types.
    at_start: bool,
    /// Bytes removed from the front of every frame by header-stripping
    /// content compression; restored before the frame is handed out.
    stripped_header: Vec<u8>,
}

/// One `ContentEncoding` of a track, reduced to what this demuxer can undo.
#[derive(Debug, PartialEq, Eq)]
enum Encoding<'a> {
    HeaderStripping(&'a [u8]),
    /// Signed but not encrypted: the frames are untouched.
    SignedOnly,
    Unsupported(&'static str),
}

impl<'a> From<&'a ContentEncodingValue> for Encoding<'a> {
    fn from(value: &'a ContentEncodingValue) -> Self {
        match value {
            ContentEncodingValue::Compression(compression) => match compression.algo() {
                ContentCompAlgo::Stripping => {
                    Encoding::HeaderStripping(compression.settings().unwrap_or_default())
                }
                ContentCompAlgo::Zlib => Encoding::Unsupported("zlib compression"),
                ContentCompAlgo::Bzlib => Encoding::Unsupported("bzlib compression"),
                ContentCompAlgo::Lzo1x => Encoding::Unsupported("lzo1x compression"),
                ContentCompAlgo::Unknown => Encoding::Unsupported("an unknown compression"),
            },
            ContentEncodingValue::Encryption(encryption) => match encryption.algo() {
                ContentEncAlgo::NotEncrypted => Encoding::SignedOnly,
                _ => Encoding::Unsupported("encryption"),
            },
            ContentEncodingValue::Unknown => Encoding::Unsupported("an unknown content encoding"),
        }
    }
}

/// `ContentEncodingScope` bit: the transform applies to frame contents.
const SCOPE_FRAMES: u64 = 1;
/// `ContentEncodingScope` bit: the transform applies to `CodecPrivate`.
const SCOPE_CODEC_PRIVATE: u64 = 2;

/// Returns the bytes to prepend to every frame, or an error when the track
/// uses an encoding this demuxer cannot undo. Encodings are given as
/// `(order, scope, encoding)`; stripping applied in ascending order is undone
/// by prepending the stripped headers in that same order.
fn stripped_header<'a>(
    encodings: impl IntoIterator<Item = (u64, u64, Encoding<'a>)>,
) -> Result<Vec<u8>> {
    let mut encodings: Vec<_> = encodings.into_iter().collect();
    encodings.sort_by_key(|&(order, _, _)| order);
    let mut header = Vec::new();
    for (_, scope, encoding) in encodings {
        match encoding {
            Encoding::SignedOnly => {}
            Encoding::HeaderStripping(_) if scope & SCOPE_CODEC_PRIVATE != 0 => {
                return Err(FerricastError::Demux(
                    "header-stripped CodecPrivate is not supported".into(),
                ));
            }
            Encoding::HeaderStripping(bytes) => {
                if scope & SCOPE_FRAMES != 0 {
                    header.extend_from_slice(bytes);
                }
            }
            Encoding::Unsupported(what) => {
                return Err(FerricastError::Demux(format!(
                    "video track uses {what}, which is not supported"
                )));
            }
        }
    }
    Ok(header)
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
            presentation_start_us: 0,
        };
        let track = entry.track_number().get();
        let stripped_header = stripped_header(
            entry
                .content_encodings()
                .unwrap_or_default()
                .iter()
                .map(|e| (e.order(), e.scope(), Encoding::from(e.encoding()))),
        )?;
        Ok(Self {
            file,
            stream,
            framing,
            track,
            scale_ns,
            frame: Frame::default(),
            at_start: true,
            stripped_header,
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
            if !self.stripped_header.is_empty() {
                self.frame
                    .data
                    .splice(0..0, self.stripped_header.iter().copied());
            }
            let sync = self.frame.is_keyframe.unwrap_or(false);
            let (data, idr) = self
                .framing
                .annex_b(&self.frame.data, sync || self.at_start)?;
            self.at_start = false;
            let pts = scaled_micros(self.frame.timestamp, self.scale_ns)?;
            return Ok(Some(EncodedFrame {
                codec: self.stream.codec,
                data,
                timestamp_us: pts,
                is_keyframe: sync || idr,
                duration_us: self
                    .frame
                    .duration
                    .map(|d| scaled_micros(d, self.scale_ns))
                    .transpose()?,
                pts_dts: (pts, pts),
            }));
        }
    }

    pub fn rewind(&mut self) -> Result<()> {
        self.file.seek(0).map_err(demux)?;
        self.at_start = true;
        Ok(())
    }
}

pub(crate) fn is_matroska(head: &[u8]) -> bool {
    head.starts_with(&EBML_MAGIC)
}

/// Converts `ticks` of `scale_ns` nanoseconds to microseconds; both come from
/// the file, so the product is computed wide and an unrepresentable result
/// is a demux error rather than a silent wrap.
fn scaled_micros(ticks: u64, scale_ns: u64) -> Result<u64> {
    u64::try_from(u128::from(ticks) * u128::from(scale_ns) / 1000).map_err(|_| {
        FerricastError::Demux(format!(
            "timestamp {ticks} at {scale_ns} ns per tick does not fit in microseconds"
        ))
    })
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

    #[test]
    fn stripped_headers_are_restored_in_encoding_order() {
        let header = stripped_header([
            (1, SCOPE_FRAMES, Encoding::HeaderStripping(&[3, 4])),
            (0, SCOPE_FRAMES, Encoding::HeaderStripping(&[1, 2])),
            (2, SCOPE_FRAMES, Encoding::SignedOnly),
        ])
        .unwrap();
        assert_eq!(header, [1, 2, 3, 4]);
    }

    #[test]
    fn a_track_without_encodings_needs_no_header() {
        assert!(stripped_header([]).unwrap().is_empty());
    }

    #[test]
    fn compressed_or_encrypted_tracks_are_rejected() {
        for what in ["zlib compression", "encryption"] {
            let error = stripped_header([(0, SCOPE_FRAMES, Encoding::Unsupported(what))])
                .unwrap_err();
            assert!(error.to_string().contains(what), "{error}");
        }
        assert!(
            stripped_header([(0, SCOPE_CODEC_PRIVATE, Encoding::HeaderStripping(&[0]))]).is_err()
        );
    }

    #[test]
    fn timestamps_scale_without_overflowing() {
        assert_eq!(scaled_micros(40, 1_000_000).unwrap(), 40_000);
        assert_eq!(
            scaled_micros(u64::MAX, 1_000).unwrap(),
            u64::MAX,
            "a product past u64 is still exact"
        );
        assert!(scaled_micros(u64::MAX, u64::MAX).is_err());
    }
}
