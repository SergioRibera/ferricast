use std::marker::PhantomData;
use std::os::raw::c_int;
use std::ptr::{addr_of_mut, from_mut, null_mut};

use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, PixelFormat, Result,
    VideoDecoder,
};
use openh264::{
    OpenH264API,
    decoder::{Decoder, DecoderConfig as Oh264DecoderConfig, Flush},
};
use openh264_sys2::{
    DECODER_OPTION, DECODER_OPTION_NUM_OF_FRAMES_REMAINING_IN_BUFFER, SBufferInfo,
};

use crate::nv12::{self, Colour, OutputFormat, Planar};

#[derive(Default)]
pub struct OpenH264Decoder {
    session: Option<Session>,
    format: OutputFormat,
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
        // Low latency trades display order for not waiting on reordering.
        self.release = if config.low_latency {
            Release::Immediately
        } else {
            Release::DisplayOrder
        };
        Ok(())
    }

    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
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
            Colour::guess(height),
        )
        .map(|raw| Some(CapturedFrame::Cpu(raw)))
        .ok_or_else(short)
    }
}
