use std::io::BufReader;

use ferricast_core::{AudioCodec, AudioFrame, Codec, EncodedFrame, FerricastError, Result};
use matroska_demuxer::{
    ContentCompAlgo, ContentEncAlgo, ContentEncodingValue, Frame, MatrixCoefficients, MatroskaFile,
    Range, TrackEntry, TrackType,
};

mod cues;

use crate::audio::{AacConfig, AudioFormat, AudioStream, OPUS_RATE, opus_channels};
use crate::nal::{Framing, LengthPrefixed};
use crate::{BoxedSource, Color, Packet, TrackSelection, VideoStream};

const EBML_MAGIC: [u8; 4] = [0x1A, 0x45, 0xDF, 0xA3];
/// How far before a seek target the keyframe search starts, tried in
/// turn until a keyframe at or before the target turns up.
const SEEK_LOOKBACK_US: [u64; 5] = [1_000_000, 4_000_000, 10_000_000, 30_000_000, 120_000_000];

pub struct MatroskaDemuxer {
    file: MatroskaFile<BufReader<cues::CueMask>>,
    scale_ns: u64,
    video: Option<VideoTrack>,
    audio: Option<AudioTrack>,
    duration_us: Option<u64>,
    frame: Frame,
}

struct VideoTrack {
    stream: VideoStream,
    framing: Framing,
    track: u64,
    /// The next frame is the first one after open, rewind or a seek and
    /// must carry the parameter sets whatever its NAL types.
    at_start: bool,
    /// Bytes removed from the front of every frame by header-stripping
    /// content compression; restored before the frame is handed out.
    stripped_header: Vec<u8>,
}

struct AudioTrack {
    stream: AudioStream,
    format: AudioFormat,
    track: u64,
    stripped_header: Vec<u8>,
    /// Timestamp and duration of the previous frame. Laced frames share
    /// their block's timestamp, so each one after the first is placed
    /// right after its predecessor.
    previous: Option<(u64, u64)>,
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
                    "track uses {what}, which is not supported"
                )));
            }
        }
    }
    Ok(header)
}

impl VideoTrack {
    fn open(entry: &TrackEntry) -> Result<Option<Self>> {
        let Some((codec, framing)) =
            codec(entry).and_then(|codec| Some((codec, framing(entry, codec)?)))
        else {
            return Ok(None);
        };
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
            duration_us: None,
            presentation_start_us: 0,
        };
        Ok(Some(Self {
            stream,
            framing,
            track: entry.track_number().get(),
            at_start: true,
            stripped_header: track_stripped_header(entry)?,
        }))
    }

    fn restore(&self, frame: &mut Frame) {
        if !self.stripped_header.is_empty() {
            frame
                .data
                .splice(0..0, self.stripped_header.iter().copied());
        }
    }

    /// Whether `frame` (with its stripped header restored) starts a GOP.
    fn is_sync(&self, frame: &Frame) -> bool {
        // BlockGroups carry no keyframe flag (and the library does not
        // expose their ReferenceBlocks), so fall back to the payload.
        frame.is_keyframe.unwrap_or_else(|| {
            payload_is_keyframe(self.stream.codec, &frame.data)
                || self
                    .framing
                    .annex_b(&frame.data, false)
                    .is_ok_and(|(_, idr)| idr)
        })
    }

    fn encoded(&mut self, frame: &Frame, scale_ns: u64) -> Result<EncodedFrame> {
        let sync = frame
            .is_keyframe
            .unwrap_or_else(|| payload_is_keyframe(self.stream.codec, &frame.data));
        let (data, idr) = self.framing.annex_b(&frame.data, sync || self.at_start)?;
        self.at_start = false;
        let pts = scaled_micros(frame.timestamp, scale_ns)?;
        Ok(EncodedFrame {
            codec: self.stream.codec,
            data,
            timestamp_us: pts,
            is_keyframe: sync || idr,
            duration_us: frame
                .duration
                .map(|d| scaled_micros(d, scale_ns))
                .transpose()?,
            pts_dts: (pts, pts),
        })
    }
}

impl AudioTrack {
    /// An AAC or Opus track; other codecs and encodings this demuxer
    /// cannot undo are skipped.
    fn open(entry: &TrackEntry) -> Option<Self> {
        if entry.track_type() != TrackType::Audio {
            return None;
        }
        let settings = entry.audio()?;
        let channels = u16::try_from(settings.channels().get()).ok()?;
        let (format, stream) = match entry.codec_id() {
            "A_OPUS" => {
                let channels = entry
                    .codec_private()
                    .and_then(opus_channels)
                    .unwrap_or(channels);
                (
                    AudioFormat::Opus,
                    AudioStream {
                        codec: AudioCodec::Opus,
                        sample_rate: OPUS_RATE,
                        channels,
                        duration_us: None,
                        source: None,
                    },
                )
            }
            id if id.starts_with("A_AAC") => {
                let config = match entry.codec_private() {
                    Some(asc) => AacConfig::from_asc(asc),
                    // Legacy `A_AAC/MPEG4/LC`-style ids carry no config.
                    None => {
                        AacConfig::low_complexity(settings.sampling_frequency() as u32, channels)
                    }
                }
                .ok()?;
                (AudioFormat::Aac(config), config.stream(None))
            }
            _ => return None,
        };
        Some(Self {
            stream,
            format,
            track: entry.track_number().get(),
            stripped_header: track_stripped_header(entry).ok()?,
            previous: None,
        })
    }

    fn audio_frame(&mut self, frame: &Frame, scale_ns: u64) -> Result<AudioFrame> {
        let mut data = frame.data.as_slice();
        let restored;
        if !self.stripped_header.is_empty() {
            restored = [self.stripped_header.as_slice(), data].concat();
            data = &restored;
        }
        let mut timestamp_us = scaled_micros(frame.timestamp, scale_ns)?;
        if let Some((previous, duration)) = self.previous
            && timestamp_us == previous
        {
            timestamp_us = previous + duration;
        }
        let duration = self.format.packet_duration_us(data).unwrap_or(0);
        self.previous = Some((timestamp_us, duration));
        Ok(AudioFrame {
            codec: self.stream.codec,
            data: self.format.packet(data)?,
            timestamp_us,
            sample_rate: self.stream.sample_rate,
            channels: self.stream.channels,
        })
    }
}

fn track_stripped_header(entry: &TrackEntry) -> Result<Vec<u8>> {
    stripped_header(
        entry
            .content_encodings()
            .unwrap_or_default()
            .iter()
            .map(|e| (e.order(), e.scope(), Encoding::from(e.encoding()))),
    )
}

impl MatroskaDemuxer {
    pub(crate) fn open(source: BoxedSource, selection: TrackSelection) -> Result<Self> {
        let file =
            MatroskaFile::open(BufReader::new(cues::CueMask::new(source)?)).map_err(demux)?;
        let scale_ns = file.info().timestamp_scale().get();
        let duration_us = file
            .info()
            .duration()
            .map(|ticks| (ticks * scale_ns as f64 / 1000.0) as u64);
        let mut video = None;
        for entry in file.tracks() {
            if let Some(track) = VideoTrack::open(entry)? {
                video = Some(track);
                break;
            }
        }
        if let Some(video) = video.as_mut() {
            video.stream.duration_us = duration_us;
        }
        let audio = match selection {
            TrackSelection::All => file
                .tracks()
                .iter()
                .find_map(AudioTrack::open)
                .map(|mut a| {
                    a.stream.duration_us = duration_us;
                    a
                }),
            TrackSelection::VideoOnly => None,
        };
        match (&video, &audio, selection) {
            (None, _, TrackSelection::VideoOnly) => {
                return Err(FerricastError::Demux("no supported video track".into()));
            }
            (None, None, TrackSelection::All) => {
                return Err(FerricastError::Demux(
                    "no supported video or audio track".into(),
                ));
            }
            _ => {}
        }
        Ok(Self {
            file,
            scale_ns,
            video,
            audio,
            duration_us,
            frame: Frame::default(),
        })
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

    /// The next packet of the selected tracks, in file order.
    pub(crate) fn next_packet(&mut self) -> Result<Option<Packet>> {
        loop {
            if !self.file.next_frame(&mut self.frame).map_err(demux)? {
                return Ok(None);
            }
            if let Some(video) = self.video.as_mut()
                && self.frame.track == video.track
            {
                video.restore(&mut self.frame);
                return video
                    .encoded(&self.frame, self.scale_ns)
                    .map(|f| Some(Packet::Video(f)));
            }
            if let Some(audio) = self.audio.as_mut()
                && self.frame.track == audio.track
            {
                return audio
                    .audio_frame(&self.frame, self.scale_ns)
                    .map(|f| Some(Packet::Audio(f)));
            }
        }
    }

    /// Positions the demuxer on the last video keyframe at or before
    /// `timestamp_us` (audio-only files: on `timestamp_us` itself) and
    /// returns where it landed. Without Cues the library walks the
    /// clusters linearly, so the keyframe search reads forward from a
    /// little before the target, widening the window until it finds one.
    pub(crate) fn seek(&mut self, timestamp_us: u64) -> Result<u64> {
        let past_end = self.duration_us.is_some_and(|d| timestamp_us >= d);
        let landed = if self.video.is_none() || past_end {
            timestamp_us
        } else {
            self.find_keyframe(timestamp_us)?
        };
        self.file.seek(self.ticks(landed)).map_err(demux)?;
        if let Some(video) = self.video.as_mut() {
            video.at_start = true;
        }
        if let Some(audio) = self.audio.as_mut() {
            audio.previous = None;
        }
        Ok(landed)
    }

    fn find_keyframe(&mut self, timestamp_us: u64) -> Result<u64> {
        let mut from = timestamp_us;
        for lookback in SEEK_LOOKBACK_US {
            from = timestamp_us.saturating_sub(lookback);
            self.file.seek(self.ticks(from)).map_err(demux)?;
            if let Some(keyframe) = self.last_keyframe_until(timestamp_us)? {
                return Ok(keyframe);
            }
            if from == 0 {
                break;
            }
        }
        // No keyframe within reach: start from the widest window and let
        // the decoder wait for the next one.
        Ok(from)
    }

    /// Reads forward and returns the timestamp of the last video keyframe
    /// not after `timestamp_us`.
    fn last_keyframe_until(&mut self, timestamp_us: u64) -> Result<Option<u64>> {
        let Some(video) = self.video.as_ref() else {
            return Ok(None);
        };
        let mut found = None;
        while self.file.next_frame(&mut self.frame).map_err(demux)? {
            let at = scaled_micros(self.frame.timestamp, self.scale_ns)?;
            if at > timestamp_us {
                break;
            }
            if self.frame.track == video.track {
                video.restore(&mut self.frame);
                if video.is_sync(&self.frame) {
                    found = Some(at);
                }
            }
        }
        Ok(found)
    }

    fn ticks(&self, timestamp_us: u64) -> u64 {
        u64::try_from(u128::from(timestamp_us) * 1000 / u128::from(self.scale_ns.max(1)))
            .unwrap_or(u64::MAX)
    }

    pub(crate) fn rewind(&mut self) -> Result<()> {
        self.file.seek(0).map_err(demux)?;
        if let Some(video) = self.video.as_mut() {
            video.at_start = true;
        }
        if let Some(audio) = self.audio.as_mut() {
            audio.previous = None;
        }
        Ok(())
    }
}

pub(crate) fn is_matroska(head: &[u8]) -> bool {
    head.starts_with(&EBML_MAGIC)
}

/// Reads the keyframe flag from the codec's own frame header.
///
/// H.264/HEVC keyframes are found from their NAL types during Annex B
/// conversion.
fn payload_is_keyframe(codec: Codec, data: &[u8]) -> bool {
    match codec {
        // RFC 6386 §9.1: frame tag bit 0 clear, then the 9d 01 2a start code.
        Codec::Vp8 => {
            data.first().is_some_and(|tag| tag & 1 == 0)
                && data.get(3..6) == Some(&[0x9d, 0x01, 0x2a])
        }
        Codec::Vp9 => vp9_is_keyframe(data),
        Codec::Av1 => av1_is_keyframe(data),
        Codec::H264 | Codec::H265 => false,
    }
}

/// AV1 OBU types (bitstream spec §6.2.2) that decide a keyframe.
const OBU_SEQUENCE_HEADER: u8 = 1;
const OBU_FRAME_HEADER: u8 = 3;
const OBU_FRAME: u8 = 6;

/// An AV1 temporal unit is a keyframe (AV1-in-Matroska §4.4) when it
/// carries a sequence header and its first frame header is a KEY_FRAME
/// that is decoded rather than shown from a reference: show_existing_frame
/// (1 bit) and frame_type (2 bits, 0 = KEY_FRAME) open the uncompressed
/// header (§5.9.2), unless reduced_still_picture_header (bit 4 of the
/// sequence header, after seq_profile and still_picture) makes every
/// frame a key frame with neither field coded.
fn av1_is_keyframe(data: &[u8]) -> bool {
    let mut reduced_still_picture = None;
    for obu in Obus(data) {
        match obu.kind {
            OBU_SEQUENCE_HEADER => {
                reduced_still_picture = Some(obu.payload.first().is_some_and(|b| b & 0x08 != 0));
            }
            OBU_FRAME_HEADER | OBU_FRAME => {
                return reduced_still_picture.is_some_and(|reduced| {
                    reduced || obu.payload.first().is_some_and(|b| b & 0xe0 == 0)
                });
            }
            _ => {}
        }
    }
    false
}

/// One OBU of an AV1 temporal unit.
struct Obu<'a> {
    kind: u8,
    payload: &'a [u8],
}

/// Walks the OBUs of a low-overhead AV1 bitstream (§5.3), ending at the
/// first malformed one. Every OBU consumes at least its header byte, so
/// the walk is bounded by the input length.
struct Obus<'a>(&'a [u8]);

impl<'a> Obus<'a> {
    /// Splits the leading OBU off `data`: obu_header (forbidden bit,
    /// obu_type, extension flag, has_size_field, reserved bit), the
    /// optional extension byte, then a leb128 obu_size or, without a size
    /// field, everything that is left.
    fn split(data: &'a [u8]) -> Option<(Obu<'a>, &'a [u8])> {
        let (&header, rest) = data.split_first()?;
        if header & 0x80 != 0 {
            return None;
        }
        let rest = if header & 0x04 != 0 {
            rest.get(1..)?
        } else {
            rest
        };
        let (payload, tail) = if header & 0x02 != 0 {
            let (size, rest) = leb128(rest)?;
            rest.split_at_checked(usize::try_from(size).ok()?)?
        } else {
            (rest, &[][..])
        };
        let kind = (header >> 3) & 0x0f;
        Some((Obu { kind, payload }, tail))
    }
}

impl<'a> Iterator for Obus<'a> {
    type Item = Obu<'a>;

    fn next(&mut self) -> Option<Obu<'a>> {
        let Some((obu, tail)) = Self::split(self.0) else {
            self.0 = &[];
            return None;
        };
        self.0 = tail;
        Some(obu)
    }
}

/// Unsigned LEB128 (§4.10.5): at most 8 bytes, little-endian 7-bit groups.
fn leb128(data: &[u8]) -> Option<(u64, &[u8])> {
    let mut value = 0;
    for (i, &byte) in data.iter().take(8).enumerate() {
        value |= u64::from(byte & 0x7f) << (7 * i);
        if byte & 0x80 == 0 {
            return Some((value, data.get(i + 1..)?));
        }
    }
    None
}

/// VP9 uncompressed header (bitstream spec §6.2): frame_marker (2 bits, = 2),
/// profile_low_bit, profile_high_bit, a reserved zero bit for profile 3, then
/// show_existing_frame and frame_type (0 = KEY_FRAME).
fn vp9_is_keyframe(data: &[u8]) -> bool {
    let Some(&byte) = data.first() else {
        return false;
    };
    let bit = |at: u32| (byte >> (7 - at)) & 1;
    if byte >> 6 != 0b10 {
        return false;
    }
    let profile = bit(3) << 1 | bit(2);
    let show_existing_at = if profile == 3 { 5 } else { 4 };
    bit(show_existing_at) == 0 && bit(show_existing_at + 1) == 0
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
            let error =
                stripped_header([(0, SCOPE_FRAMES, Encoding::Unsupported(what))]).unwrap_err();
            assert!(error.to_string().contains(what), "{error}");
        }
        assert!(
            stripped_header([(0, SCOPE_CODEC_PRIVATE, Encoding::HeaderStripping(&[0]))]).is_err()
        );
    }

    #[test]
    fn vp8_keyframes_are_read_from_the_frame_tag() {
        let key = [0x50, 0x2d, 0x00, 0x9d, 0x01, 0x2a, 0x80, 0x00];
        let inter = [0x51, 0x2d, 0x00, 0x00, 0x00, 0x00];
        assert!(payload_is_keyframe(Codec::Vp8, &key));
        assert!(!payload_is_keyframe(Codec::Vp8, &inter));
        assert!(!payload_is_keyframe(Codec::Vp8, &[]));
    }

    #[test]
    fn vp9_keyframes_are_read_from_the_uncompressed_header() {
        // marker 10, profile 0, show_existing 0, frame_type 0
        assert!(payload_is_keyframe(Codec::Vp9, &[0b1000_0010]));
        // frame_type 1: inter frame
        assert!(!payload_is_keyframe(Codec::Vp9, &[0b1000_0110]));
        // show_existing_frame 1
        assert!(!payload_is_keyframe(Codec::Vp9, &[0b1000_1000]));
        // profile 3 has a reserved bit before show_existing_frame
        assert!(payload_is_keyframe(Codec::Vp9, &[0b1011_0001]));
        assert!(!payload_is_keyframe(Codec::Vp9, &[0b1011_0010]));
        // wrong frame marker
        assert!(!payload_is_keyframe(Codec::Vp9, &[0b0000_0000]));
    }

    /// Temporal delimiter, sequence header with `seq` as its first
    /// payload byte, then a frame OBU whose header starts with `frame`.
    fn av1_unit(seq: u8, frame: u8) -> [u8; 8] {
        [0x12, 0x00, 0x0a, 0x01, seq, 0x32, 0x01, frame]
    }

    #[test]
    fn av1_keyframes_are_read_from_the_frame_header() {
        // show_existing_frame 0, frame_type KEY_FRAME, show_frame 1
        assert!(payload_is_keyframe(Codec::Av1, &av1_unit(0, 0b0001_0000)));
        // frame_type INTER_FRAME
        assert!(!payload_is_keyframe(Codec::Av1, &av1_unit(0, 0b0011_0000)));
        // show_existing_frame 1
        assert!(!payload_is_keyframe(Codec::Av1, &av1_unit(0, 0b1000_0000)));
        // reduced_still_picture_header: no frame_type, always a key frame
        assert!(payload_is_keyframe(Codec::Av1, &av1_unit(0x18, 0xff)));
        // a KEY_FRAME without a sequence header is not a sync point
        assert!(!payload_is_keyframe(Codec::Av1, &[0x32, 0x01, 0x10]));
        // a frame header OBU works like a frame OBU
        assert!(payload_is_keyframe(
            Codec::Av1,
            &[0x0a, 0x01, 0x00, 0x1a, 0x01, 0x10]
        ));
    }

    #[test]
    fn av1_obu_headers_are_walked_correctly() {
        // sequence header with an extension byte, then a last frame OBU
        // without a size field that runs to the end of the block
        let unit = [0x0e, 0x00, 0x01, 0x00, 0x30, 0x10, 0xaa];
        assert!(payload_is_keyframe(Codec::Av1, &unit));
        // a two-byte leb128 size (128) skips a padding OBU
        let mut padded = vec![0x0a, 0x01, 0x00, 0x7a, 0x80, 0x01];
        padded.extend([0; 128]);
        padded.extend([0x32, 0x01, 0x10]);
        assert!(payload_is_keyframe(Codec::Av1, &padded));
        assert_eq!(leb128(&[0x80, 0x01, 0xff]), Some((128, &[0xff][..])));
    }

    #[test]
    fn malformed_av1_units_are_not_keyframes() {
        let cases: [&[u8]; 7] = [
            &[],
            // forbidden bit set on the sequence header
            &[0x8a, 0x01, 0x00, 0x32, 0x01, 0x10],
            // leb128 cut off mid-number
            &[0x0a, 0x80],
            // leb128 longer than eight bytes
            &[0x0a, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00],
            // obu_size past the end of the block
            &[0x0a, 0xff, 0xff, 0xff, 0xff, 0x0f, 0x00],
            // extension flag without the extension byte
            &[0x0e],
            // frame OBU cut off before its header
            &[0x0a, 0x01, 0x00, 0x32, 0x01],
        ];
        for case in cases {
            assert!(!payload_is_keyframe(Codec::Av1, case), "{case:02x?}");
        }
    }

    /// Garbage never panics or loops; xorshift keeps the run reproducible.
    #[test]
    fn random_bytes_never_panic_the_av1_parser() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..20_000 {
            let len = usize::try_from(next() % 64).unwrap();
            let data: Vec<u8> = (0..len).map(|_| next().to_le_bytes()[0]).collect();
            let _ = payload_is_keyframe(Codec::Av1, &data);
        }
    }

    /// The fixture's first block is a keyframe by its own bitstream, not
    /// only by the container flag; so is the second keyframe it carries.
    #[test]
    fn av1_fixture_keyframes_are_found_in_the_bitstream() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/av1.webm");
        let file = std::fs::File::open(&path).unwrap();
        let mut demuxer = MatroskaDemuxer::open(Box::new(file), TrackSelection::VideoOnly).unwrap();
        let mut keyframes = Vec::new();
        while let Some(Packet::Video(frame)) = demuxer.next_packet().unwrap() {
            keyframes.push(payload_is_keyframe(Codec::Av1, &frame.data));
        }
        assert_eq!(keyframes.first(), Some(&true));
        assert_eq!(keyframes.iter().filter(|&&key| key).count(), 2);
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
