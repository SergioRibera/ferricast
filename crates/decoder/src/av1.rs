use std::collections::VecDeque;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

use ferricast_core::ContentLightLevel;
use ferricast_core::{BackendId, CodecError, CodecResult, Operation};
use ferricast_core::{
    CapturedFrame, ChromaLocation, Codec, ColourDescription, ColourRange, DecoderConfig,
    EncodedFrame, FerricastError, HdrMetadata, MasteringDisplay, Result, VideoDecoder,
};
use rav1d::include::dav1d::data::Dav1dData;
use rav1d::include::dav1d::dav1d::{Dav1dContext, Dav1dSettings};
use rav1d::include::dav1d::picture::Dav1dPicture;
use rav1d::src::lib::{
    dav1d_close, dav1d_data_create, dav1d_data_unref, dav1d_default_settings, dav1d_get_picture,
    dav1d_open, dav1d_picture_unref, dav1d_send_data,
};

use crate::nv12::{self, OutputFormat, Planar, WidePicture, WidePlane, WideSamples};

const I420: u32 = 1;
const MAX_FRAME_PIXELS: u32 = 8192 * 8192;

#[derive(Default)]
pub struct Rav1dDecoder {
    context: Option<Dav1dContext>,
    ready: VecDeque<CapturedFrame>,
    format: OutputFormat,
    /// The container's HDR metadata, for pictures whose stream sends none.
    container_hdr: Option<HdrMetadata>,
}

// SAFETY: the context is an Arc-backed rav1d decoder that is only ever used through
// `&mut self`, and rav1d synchronises its own worker threads internally.
unsafe impl Send for Rav1dDecoder {}

impl Rav1dDecoder {
    fn try_configure(&mut self, config: &DecoderConfig) -> Result<()> {
        if config.codec != Codec::Av1 {
            return Err(FerricastError::Decode(format!(
                "Rav1dDecoder asked to decode {:?}; only AV1 supported",
                config.codec
            )));
        }
        self.close();
        self.format = OutputFormat(config.pixel_format);
        self.container_hdr = config.hdr;
        let mut settings = MaybeUninit::<Dav1dSettings>::uninit();
        // SAFETY: `settings` is valid for writes and dav1d_default_settings initialises all of it.
        let mut settings = unsafe {
            dav1d_default_settings(NonNull::from(&mut settings).cast());
            settings.assume_init()
        };
        settings.n_threads = 1;
        settings.max_frame_delay = 1;
        // A corrupt or hostile sequence header can claim 65536x65536;
        // refuse anything past 8K-square before rav1d allocates for it.
        settings.frame_size_limit = MAX_FRAME_PIXELS;
        let mut context = None;
        // SAFETY: both pointers come from live references and `context` is only read back once
        // dav1d_open has reported success.
        let opened = unsafe {
            dav1d_open(
                Some(NonNull::from(&mut context)),
                Some(NonNull::from(&mut settings)),
            )
        };
        match context {
            Some(context) if opened.0 == 0 => {
                self.context = Some(context);
                Ok(())
            }
            _ => Err(FerricastError::Decode(format!(
                "rav1d open failed with {}",
                opened.0
            ))),
        }
    }

    fn try_decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        let context = self.context.ok_or_else(|| {
            FerricastError::Decode("Rav1dDecoder::decode before configure()".into())
        })?;
        let mut data = Dav1dData::default();
        // SAFETY: `data` is valid for writes; on success dav1d owns a buffer of the requested size.
        let buffer = unsafe { dav1d_data_create(Some(NonNull::from(&mut data)), frame.data.len()) };
        if buffer.is_null() {
            return Err(FerricastError::Decode(
                "rav1d could not allocate a packet".into(),
            ));
        }
        // SAFETY: `buffer` has room for `frame.data.len()` bytes and cannot overlap `frame.data`.
        unsafe { std::ptr::copy_nonoverlapping(frame.data.as_ptr(), buffer, frame.data.len()) };
        data.m.timestamp = frame.timestamp_us as i64;
        let result = self.feed(context, &mut data);
        // SAFETY: `data` was filled by dav1d_data_create and is unreferenced exactly once, here.
        unsafe { dav1d_data_unref(Some(NonNull::from(&mut data))) };
        result?;
        Ok(self.ready.pop_front())
    }

    fn try_flush(&mut self) -> Result<Vec<CapturedFrame>> {
        Ok(self.ready.drain(..).collect())
    }
}

impl VideoDecoder for Rav1dDecoder {
    const CODEC: Codec = Codec::Av1;

    fn configure(&mut self, config: &DecoderConfig) -> CodecResult<()> {
        self.try_configure(config).map_err(CodecError::from_backend(
            BackendId::Rav1d,
            Operation::Configure,
        ))
    }

    fn decode(&mut self, frame: EncodedFrame) -> CodecResult<Option<CapturedFrame>> {
        self.try_decode(frame).map_err(CodecError::from_backend(
            BackendId::Rav1d,
            Operation::Decode,
        ))
    }

    fn flush(&mut self) -> CodecResult<Vec<CapturedFrame>> {
        self.try_flush()
            .map_err(CodecError::from_backend(BackendId::Rav1d, Operation::Flush))
    }
}

impl Rav1dDecoder {
    fn feed(&mut self, context: Dav1dContext, data: &mut Dav1dData) -> Result<()> {
        while data.sz > 0 {
            // SAFETY: `context` comes from dav1d_open and has not been closed; `data` is valid.
            let sent = unsafe { dav1d_send_data(Some(context), Some(NonNull::from(&mut *data))) };
            if sent.0 != 0 && sent.0 != -libc::EAGAIN {
                return Err(FerricastError::Decode(format!(
                    "rav1d send failed with {}",
                    sent.0
                )));
            }
            let pulled = self.pull(context)?;
            if sent.0 != 0 && !pulled {
                return Err(FerricastError::Decode("rav1d stalled on a packet".into()));
            }
        }
        Ok(())
    }

    fn pull(&mut self, context: Dav1dContext) -> Result<bool> {
        let mut pulled = false;
        loop {
            let mut picture = Dav1dPicture::default();
            // SAFETY: `context` is open and `picture` is valid for writes.
            let got =
                unsafe { dav1d_get_picture(Some(context), Some(NonNull::from(&mut picture))) };
            if got.0 == -libc::EAGAIN {
                return Ok(pulled);
            }
            if got.0 != 0 {
                return Err(FerricastError::Decode(format!(
                    "rav1d get_picture failed with {}",
                    got.0
                )));
            }
            let frame = nv12_frame(&picture, self.format, self.container_hdr);
            // SAFETY: `picture` was filled by dav1d_get_picture and is unreferenced exactly once.
            unsafe { dav1d_picture_unref(Some(NonNull::from(&mut picture))) };
            self.ready.push_back(frame?);
            pulled = true;
        }
    }

    fn close(&mut self) {
        if self.context.is_some() {
            // SAFETY: the context came from dav1d_open and dav1d_close clears it so it cannot be
            // closed twice.
            unsafe { dav1d_close(Some(NonNull::from(&mut self.context))) };
            self.context = None;
        }
        self.ready.clear();
    }
}

impl Drop for Rav1dDecoder {
    fn drop(&mut self) {
        self.close();
    }
}

fn nv12_frame(
    picture: &Dav1dPicture,
    format: OutputFormat,
    container_hdr: Option<HdrMetadata>,
) -> Result<CapturedFrame> {
    let short = || FerricastError::Decode("rav1d returned short planes".into());
    if picture.p.layout != I420 {
        return Err(FerricastError::Decode(format!(
            "rav1d pixel layout {} is not 4:2:0",
            picture.p.layout
        )));
    }
    let (width, height) = (
        usize::try_from(picture.p.w).map_err(|_| short())?,
        usize::try_from(picture.p.h).map_err(|_| short())?,
    );
    let chroma_height = height.div_ceil(2);
    let plane = |index: usize, rows: usize| {
        let stride = usize::try_from(picture.stride[usize::from(index > 0)]).ok()?;
        let base = picture.data[index]?;
        // SAFETY: dav1d allocates every plane as `stride * rows` bytes at least, and the picture
        // stays referenced until the caller unreferences it after this function returns.
        let samples =
            unsafe { std::slice::from_raw_parts(base.as_ptr().cast::<u8>(), stride * rows) };
        Some((samples, stride))
    };
    let (y, y_stride) = plane(0, height).ok_or_else(short)?;
    let (u, uv_stride) = plane(1, chroma_height).ok_or_else(short)?;
    let (v, _) = plane(2, chroma_height).ok_or_else(short)?;
    let colour = match picture.seq_hdr {
        Some(header) => {
            // SAFETY: a picture returned by dav1d_get_picture holds a reference on its sequence
            // header, and the caller unreferences the picture only after this function returns.
            let header = unsafe { header.as_ref() };
            av1_colour(header)
        }
        None => ColourDescription::unspecified(ColourRange::Limited),
    };
    let timestamp_us = u64::try_from(picture.m.timestamp).unwrap_or(0);
    let raw = if picture.p.bpc > 8 {
        let wide = |samples| WidePlane {
            samples: WideSamples::LeBytes(samples),
            stride: uv_stride,
        };
        WidePicture {
            width,
            height,
            depth: u32::try_from(picture.p.bpc).map_err(|_| short())?,
            y: WidePlane {
                samples: WideSamples::LeBytes(y),
                stride: y_stride,
            },
            u: wide(u),
            v: wide(v),
        }
        .deliver(timestamp_us, format.0, colour)
    } else {
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
        nv12::deliver(
            data,
            stride,
            (width, height),
            timestamp_us,
            format.0,
            colour,
        )
    };
    let mut raw = raw.ok_or_else(short)?;
    raw.hdr = picture_hdr(picture).or(container_hdr);
    Ok(CapturedFrame::Cpu(raw))
}

/// The HDR metadata dav1d attached to `picture` (from the stream's
/// metadata OBUs, carried forward by dav1d to every later picture).
fn picture_hdr(picture: &Dav1dPicture) -> Option<HdrMetadata> {
    // SAFETY: a picture returned by dav1d_get_picture holds references on its
    // metadata, released only when the caller unreferences the picture,
    // after this function returns.
    let (display, light) = unsafe {
        (
            picture.mastering_display.map(|meta| meta.as_ref()),
            picture.content_light.map(|meta| meta.as_ref()),
        )
    };
    let hdr = HdrMetadata {
        mastering_display: display.map(|display| {
            let [r, g, b] = display.primaries;
            let point = |[x, y]: [u16; 2]| (x, y);
            MasteringDisplay::from_av1(
                [point(r), point(g), point(b)],
                point(display.white_point),
                display.max_luminance,
                display.min_luminance,
            )
        }),
        content_light: light.map(|light| ContentLightLevel {
            max_cll: light.max_content_light_level,
            max_fall: light.max_frame_average_light_level,
        }),
    };
    (!hdr.is_empty()).then_some(hdr)
}

/// The colour description of an AV1 sequence header. AV1 codes
/// primaries, transfer and matrix as H.273 does; its chroma sample
/// position has its own codes (1 vertical, 2 co-located).
fn av1_colour(header: &rav1d::include::dav1d::headers::Dav1dSequenceHeader) -> ColourDescription {
    let code = |value: u32| u8::try_from(value).unwrap_or(u8::MAX);
    let description = ColourDescription::from_codes(
        code(header.pri),
        code(header.trc),
        code(header.mtrx),
        header.color_range != 0,
    );
    match header.chr {
        1 => description.with_chroma_location(ChromaLocation::Left),
        2 => description.with_chroma_location(ChromaLocation::TopLeft),
        _ => description,
    }
}
