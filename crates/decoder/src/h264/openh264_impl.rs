use std::io::Cursor;
use std::marker::PhantomData;
use std::os::raw::c_int;
use std::ptr::{addr_of_mut, from_mut, null_mut};

use ferricast_core::{
    CapturedFrame, Codec, ColourDescription, ColourRange, DecoderConfig, EncodedFrame,
    FerricastError, PixelFormat, Result, VideoDecoder,
};
use h264_reader::nal::sps::SeqParameterSet;
use h264_reader::rbsp::{BitReader, decode_nal};
use openh264::{
    OpenH264API,
    decoder::{Decoder, DecoderConfig as Oh264DecoderConfig, Flush},
};
use openh264_sys2::{
    DECODER_OPTION, DECODER_OPTION_NUM_OF_FRAMES_REMAINING_IN_BUFFER, SBufferInfo,
};

use crate::nv12::{self, OutputFormat, Planar};

/// `profile_idc` of Baseline and Constrained Baseline, which have no B
/// slices and so never reorder pictures.
const PROFILE_BASELINE: u8 = 66;

#[derive(Default)]
pub struct OpenH264Decoder {
    session: Option<Session>,
    format: OutputFormat,
    low_latency: bool,
    release: Release,
}

/// When openh264 hands a decoded picture over.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Release {
    /// openh264's own reordering puts pictures out in display order, one
    /// picture late when the stream turns out to have no B slices.
    #[default]
    DisplayOrder,
    /// Every picture leaves as soon as it is decoded: no added latency,
    /// but decode order, so only right for a stream that cannot reorder.
    Immediately,
}

impl Release {
    /// The policy the last SPS in an Annex B packet calls for, if the
    /// packet carries one.
    fn for_packet(data: &[u8]) -> Option<Self> {
        openh264::nal_units(data)
            .filter_map(strip_start_code)
            .filter(|nal| nal.first().is_some_and(|header| header & 0x1f == 7))
            .filter_map(Self::for_sps)
            .last()
    }

    fn for_sps(nal: &[u8]) -> Option<Self> {
        let rbsp = decode_nal(nal).ok()?;
        let profile_idc = *rbsp.first()?;
        // An SPS h264-reader cannot read still names its profile, which is
        // enough to tell whether B slices are allowed.
        let max_num_reorder_frames =
            SeqParameterSet::from_bits(BitReader::new(Cursor::new(rbsp.as_ref())))
                .ok()
                .and_then(|sps| sps.vui_parameters)
                .and_then(|vui| vui.bitstream_restrictions)
                .map(|restrictions| restrictions.max_num_reorder_frames);
        Some(Self::for_stream(profile_idc, max_num_reorder_frames))
    }

    /// Display order whenever the stream may reorder: its VUI says so or,
    /// without a VUI bound, its profile allows B slices.
    fn for_stream(profile_idc: u8, max_num_reorder_frames: Option<u32>) -> Self {
        let reorders =
            max_num_reorder_frames.map_or(profile_idc != PROFILE_BASELINE, |frames| frames > 0);
        if reorders {
            Self::DisplayOrder
        } else {
            Self::Immediately
        }
    }
}

/// An Annex B unit from `openh264::nal_units` without its start code.
fn strip_start_code(unit: &[u8]) -> Option<&[u8]> {
    let one = unit.iter().position(|&byte| byte != 0)?;
    (unit[one] == 1).then(|| &unit[one + 1..])
}

impl VideoDecoder for OpenH264Decoder {
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, config: &DecoderConfig) -> Result<()> {
        if config.codec != Codec::H264 {
            return Err(FerricastError::Decode(format!(
                "OpenH264Decoder asked to decode {:?}; only H.264 supported",
                config.codec
            )));
        }
        self.session = Some(Session::open()?);
        self.format = OutputFormat(config.pixel_format);
        self.low_latency = config.low_latency;
        // Low latency releases pictures at once until an SPS says the
        // stream may reorder.
        self.release = if config.low_latency {
            Release::Immediately
        } else {
            Release::DisplayOrder
        };
        Ok(())
    }

    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        if self.low_latency
            && let Some(release) = Release::for_packet(&frame.data)
            && release != self.release
        {
            tracing::debug!(?release, "openh264 picture release follows the new SPS");
            self.release = release;
        }
        let format = self.format.0;
        let release = self.release;
        let session = self.session.as_mut().ok_or_else(|| {
            FerricastError::Decode("OpenH264Decoder::decode before configure()".into())
        })?;
        let decoded = session.decode(&frame.data, frame.timestamp_us)?;
        if decoded.is_ready() {
            return decoded.picture(format);
        }
        if release == Release::Immediately && session.buffered()? > 0 {
            return session.release_one()?.picture(format);
        }
        Ok(None)
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        let format = self.format.0;
        let Some(session) = self.session.as_mut() else {
            return Ok(Vec::new());
        };
        let mut frames = Vec::new();
        for _ in 0..session.buffered()? {
            frames.extend(session.release_one()?.picture(format)?);
        }
        Ok(frames)
    }
}

/// openh264 driven through its raw entry points: the safe wrapper never
/// sets the input timestamp, which openh264 carries from each packet to
/// the picture decoded from it, through its own reordering.
struct Session {
    decoder: Decoder,
    /// What the last call handed back; its planes stay valid until the
    /// decoder is called again. Boxed: the buffer description is large.
    last: Box<Output>,
}

// SAFETY: `Output` only points into buffers owned by the decoder, which
// moves between threads together with them.
unsafe impl Send for Session {}

/// One call's output: plane pointers and their description.
struct Output {
    planes: [*mut u8; 3],
    info: SBufferInfo,
}

/// The output of the last decoder call. It borrows the session so the
/// decoder cannot run again, and recycle the planes, while it is read.
struct Picture<'a> {
    output: &'a Output,
    _session: PhantomData<&'a mut Session>,
}

impl Session {
    fn open() -> Result<Self> {
        // Releasing and flushing are driven here, never by the wrapper.
        let config = Oh264DecoderConfig::new().flush_after_decode(Flush::NoFlush);
        let decoder = Decoder::with_api_config(OpenH264API::from_source(), config)
            .map_err(|e| FerricastError::Decode(format!("openh264 decoder init: {e:?}")))?;
        Ok(Self {
            decoder,
            last: Box::new(Output {
                planes: [null_mut(); 3],
                info: SBufferInfo::default(),
            }),
        })
    }

    fn decode(&mut self, data: &[u8], timestamp_us: u64) -> Result<Picture<'_>> {
        let length = c_int::try_from(data.len())
            .map_err(|_| FerricastError::Decode("H.264 packet too large for openh264".into()))?;
        self.last.planes = [null_mut(); 3];
        self.last.info = SBufferInfo {
            uiInBsTimeStamp: timestamp_us,
            ..SBufferInfo::default()
        };
        // SAFETY: the decoder was initialised by `Decoder::with_api_config`
        // and none of its options are touched; `data` is valid for `length`
        // bytes and the output slots live in `self`.
        let state = unsafe {
            self.decoder.raw_api().decode_frame_no_delay(
                data.as_ptr(),
                length,
                from_mut(&mut self.last.planes).cast(),
                &raw mut self.last.info,
            )
        };
        if state != 0 {
            return Err(FerricastError::Decode(format!(
                "openh264 decode failed with state {state:#x}"
            )));
        }
        Ok(self.last())
    }

    /// Releases the earliest picture openh264 still holds back.
    fn release_one(&mut self) -> Result<Picture<'_>> {
        self.last.planes = [null_mut(); 3];
        self.last.info = SBufferInfo::default();
        // SAFETY: as in `decode`; the output slots live in `self`.
        let state = unsafe {
            self.decoder.raw_api().flush_frame(
                from_mut(&mut self.last.planes).cast(),
                &raw mut self.last.info,
            )
        };
        if state != 0 {
            return Err(FerricastError::Decode(format!(
                "openh264 flush failed with state {state:#x}"
            )));
        }
        Ok(self.last())
    }

    /// Pictures openh264 decoded but has not released yet.
    fn buffered(&mut self) -> Result<usize> {
        let mut count: DECODER_OPTION = 0;
        // SAFETY: this option writes a single `int` through the pointer.
        let state = unsafe {
            self.decoder.raw_api().get_option(
                DECODER_OPTION_NUM_OF_FRAMES_REMAINING_IN_BUFFER,
                addr_of_mut!(count).cast(),
            )
        };
        if state != 0 {
            return Err(FerricastError::Decode(format!(
                "openh264 buffered-picture query failed with {state}"
            )));
        }
        Ok(usize::try_from(count).unwrap_or(0))
    }

    fn last(&mut self) -> Picture<'_> {
        Picture {
            output: &self.last,
            _session: PhantomData,
        }
    }
}

impl Picture<'_> {
    fn is_ready(&self) -> bool {
        self.output.info.iBufferStatus == 1
    }

    /// The released picture in `format`, stamped with the timestamp of the
    /// packet it was decoded from.
    fn picture(&self, format: PixelFormat) -> Result<Option<CapturedFrame>> {
        let output = self.output;
        let [y, u, v] = output.planes;
        // openh264 may report a picture yet hand out no planes; decoding
        // simply goes on.
        if !self.is_ready() || y.is_null() || u.is_null() || v.is_null() {
            return Ok(None);
        }
        let short = || FerricastError::Decode("openh264 returned short planes".into());
        // SAFETY: the system-memory buffer is the union's only variant.
        let buffer = unsafe { output.info.UsrData.sSystemBuffer };
        let size = |value: c_int| usize::try_from(value).map_err(|_| short());
        let (width, height) = (size(buffer.iWidth)?, size(buffer.iHeight)?);
        let (y_stride, uv_stride) = (size(buffer.iStride[0])?, size(buffer.iStride[1])?);
        let chroma = uv_stride * height.div_ceil(2);
        // SAFETY: openh264 hands out padded, macroblock-aligned planes of at
        // least `stride * rows` bytes (half the rows, rounded up, for
        // chroma). They stay valid until the decoder is called again, which
        // the borrow on the session rules out until they are copied below.
        let (y, u, v) = unsafe {
            (
                std::slice::from_raw_parts(y.cast_const(), y_stride * height),
                std::slice::from_raw_parts(u.cast_const(), chroma),
                std::slice::from_raw_parts(v.cast_const(), chroma),
            )
        };
        let (data, stride) = nv12::from_i420(&Planar {
            width,
            height,
            y,
            y_stride,
            u,
            v,
            uv_stride,
        })
        .ok_or_else(short)?;
        // openh264 does not expose the VUI colour description.
        nv12::deliver(
            data,
            stride,
            (width, height),
            output.info.uiOutYuvTimeStamp,
            format,
            ColourDescription::unspecified(ColourRange::Limited),
        )
        .map(|raw| Some(CapturedFrame::Cpu(raw)))
        .ok_or_else(short)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stream_reorders_when_its_vui_says_so_or_its_profile_allows_b_slices() {
        assert_eq!(Release::for_stream(100, Some(2)), Release::DisplayOrder);
        assert_eq!(Release::for_stream(100, Some(0)), Release::Immediately);
        assert_eq!(Release::for_stream(77, None), Release::DisplayOrder);
        assert_eq!(Release::for_stream(100, None), Release::DisplayOrder);
        assert_eq!(Release::for_stream(66, None), Release::Immediately);
        assert_eq!(Release::for_stream(66, Some(1)), Release::DisplayOrder);
    }

    #[test]
    fn the_release_follows_the_last_sps_in_a_packet() {
        // Constrained Baseline 3.0 and High 3.0 SPS headers, no VUI.
        let baseline = [0, 0, 0, 1, 0x67, 0x42, 0xc0, 0x1e, 0xf4, 0x41, 0x04, 0x60];
        let high = [
            0, 0, 0, 1, 0x67, 0x64, 0x00, 0x1e, 0xac, 0xd9, 0x41, 0x04, 0x60,
        ];
        let slice = [0, 0, 1, 0x65, 0x88];
        assert_eq!(Release::for_packet(&baseline), Some(Release::Immediately));
        assert_eq!(Release::for_packet(&high), Some(Release::DisplayOrder));
        assert_eq!(
            Release::for_packet(&[&high[..], &baseline[..]].concat()),
            Some(Release::Immediately)
        );
        assert_eq!(Release::for_packet(&slice), None);
    }

    #[test]
    fn start_codes_of_either_length_are_stripped() {
        assert_eq!(
            strip_start_code(&[0, 0, 0, 1, 0x67, 9]),
            Some(&[0x67, 9][..])
        );
        assert_eq!(strip_start_code(&[0, 0, 1, 0x68]), Some(&[0x68][..]));
        assert_eq!(strip_start_code(&[0, 0, 2]), None);
    }
}
