//! What the Android encoders share: an owned, `Send` `AMediaCodec`, how
//! one is created (by type, or a named Google software codec), the
//! `AMediaFormat` keys they set, and error mapping.

use std::time::Duration;

use ferricast_core::{BackendId, CodecError, Operation};
use ndk::media::media_codec::{
    DequeuedInputBufferResult, DequeuedOutputBufferInfoResult, MediaCodec, MediaCodecDirection,
    MediaFormat,
};
use ndk::media_error::MediaError;

/// `AMediaFormat` keys (`NdkMediaFormat.h`). Spelled out rather than read
/// from the NDK's exported `AMEDIAFORMAT_KEY_*` symbols: several of them
/// only exist from API 29 / 30, and the strings are stable API.
pub(crate) mod key {
    pub(crate) const MIME: &str = "mime";
    pub(crate) const WIDTH: &str = "width";
    pub(crate) const HEIGHT: &str = "height";
    pub(crate) const STRIDE: &str = "stride";
    pub(crate) const SLICE_HEIGHT: &str = "slice-height";
    pub(crate) const COLOR_FORMAT: &str = "color-format";
    pub(crate) const BIT_RATE: &str = "bitrate";
    pub(crate) const BITRATE_MODE: &str = "bitrate-mode";
    pub(crate) const FRAME_RATE: &str = "frame-rate";
    pub(crate) const I_FRAME_INTERVAL: &str = "i-frame-interval";
    pub(crate) const PROFILE: &str = "profile";
    pub(crate) const PRIORITY: &str = "priority";
    pub(crate) const LATENCY: &str = "latency";
    pub(crate) const MAX_B_FRAMES: &str = "max-bframes";
    pub(crate) const PREPEND_HEADERS_TO_SYNC_FRAMES: &str = "prepend-sps-pps-to-idr-frames";
    pub(crate) const COLOR_STANDARD: &str = "color-standard";
    pub(crate) const COLOR_RANGE: &str = "color-range";
    pub(crate) const COLOR_TRANSFER: &str = "color-transfer";
    pub(crate) const REQUEST_SYNC_FRAME: &str = "request-sync";
    pub(crate) const VIDEO_BITRATE: &str = "video-bitrate";
    pub(crate) const SAMPLE_RATE: &str = "sample-rate";
    pub(crate) const CHANNEL_COUNT: &str = "channel-count";
    pub(crate) const AAC_PROFILE: &str = "aac-profile";
    pub(crate) const MAX_INPUT_SIZE: &str = "max-input-size";
    /// CTA-861.3 static HDR metadata (`MediaFormat.KEY_HDR_STATIC_INFO`).
    pub(crate) const HDR_STATIC_INFO: &str = "hdr-static-info";
}

/// `AMediaCodec` buffer flags.
pub(crate) const FLAG_KEY_FRAME: u32 = 1;
pub(crate) const FLAG_CODEC_CONFIG: u32 = 2;
pub(crate) const FLAG_END_OF_STREAM: u32 = 4;

/// `MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar`: NV12.
pub(crate) const COLOR_FORMAT_NV12: i32 = 21;

/// `MediaCodecInfo.CodecCapabilities.COLOR_FormatYUVP010`: P010.
pub(crate) const COLOR_FORMAT_P010: i32 = 54;

/// How long one wait on the codec may block.
pub(crate) const WAIT: Duration = Duration::from_millis(20);

/// An owned codec instance.
pub(crate) struct Codec(MediaCodec);

// SAFETY: an `AMediaCodec` is not bound to the thread that created it;
// the NDK only forbids concurrent calls on one instance, which `&mut`
// ownership through the encoder rules out. No async callback is
// installed, so no NDK thread touches it either.
unsafe impl Send for Codec {}

impl std::ops::Deref for Codec {
    type Target = MediaCodec;

    fn deref(&self) -> &MediaCodec {
        &self.0
    }
}

/// Which codec of a type to instantiate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pick {
    /// The device's preferred encoder for the type, required to be a
    /// vendor (hardware) one.
    Hardware,
    /// This Google software encoder (`c2.android.*`).
    Software(&'static str),
}

/// Whether a codec name is one of the Google / AOSP software codecs.
pub(crate) fn is_software(name: &str) -> bool {
    ["c2.android.", "omx.google.", "c2.google."]
        .iter()
        .any(|prefix| name.to_ascii_lowercase().starts_with(prefix))
}

/// A codec error naming `MediaCodec` and `operation`.
pub(crate) fn fail(operation: Operation, reason: impl std::fmt::Display) -> CodecError {
    CodecError::backend(BackendId::MediaCodec, operation, reason)
}

/// Maps an NDK status into a codec error for `what`.
pub(crate) fn media(
    operation: Operation,
    what: &'static str,
) -> impl FnOnce(MediaError) -> CodecError {
    move |error| fail(operation, format!("{what}: {error}"))
}

impl Codec {
    /// Instantiates the encoder `pick` names for `mime`.
    ///
    /// # Errors
    /// [`CodecError::BackendUnavailable`] when the device has no such
    /// encoder, or (for [`Pick::Hardware`]) only a software one.
    pub(crate) fn create(mime: &str, pick: Pick) -> Result<Self, CodecError> {
        let unavailable = |reason: String| CodecError::BackendUnavailable {
            backend: BackendId::MediaCodec,
            reason,
        };
        match pick {
            Pick::Hardware => {
                let codec = MediaCodec::from_encoder_type(mime)
                    .ok_or_else(|| unavailable(format!("the device has no {mime} encoder")))?;
                let name = codec.name().unwrap_or_default();
                if is_software(&name) {
                    return Err(unavailable(format!(
                        "the device's {mime} encoder is the software codec {name}, not hardware"
                    )));
                }
                Ok(Self(codec))
            }
            Pick::Software(name) => MediaCodec::from_codec_name(name)
                .map(Self)
                .ok_or_else(|| unavailable(format!("the device has no {name} codec"))),
        }
    }

    /// The codec's name, for logs.
    pub(crate) fn label(&self) -> String {
        self.name().unwrap_or_else(|_| "unnamed codec".to_owned())
    }

    /// Configures the codec as an encoder for `format` and starts it.
    pub(crate) fn start_encoder(&self, format: &MediaFormat) -> Result<(), CodecError> {
        self.configure(format, None, MediaCodecDirection::Encoder)
            .map_err(media(Operation::Open, "AMediaCodec_configure"))?;
        self.start()
            .map_err(media(Operation::Open, "AMediaCodec_start"))
    }

    /// Waits for an input buffer, copies `fill` into it and queues it
    /// with `pts_us` and `flags`; `fill` writes into the buffer and
    /// returns the byte count. Output that blocks the input is drained
    /// through `drain` while waiting.
    pub(crate) fn queue(
        &self,
        operation: Operation,
        pts_us: u64,
        flags: u32,
        fill: impl FnOnce(&mut [std::mem::MaybeUninit<u8>]) -> Result<usize, CodecError>,
        mut drain: impl FnMut(&Self) -> Result<(), CodecError>,
    ) -> Result<(), CodecError> {
        // A codec whose output is full hands out no input buffer until
        // some output is released, so drain between waits.
        for _ in 0..25 {
            match self
                .dequeue_input_buffer(WAIT)
                .map_err(media(operation, "AMediaCodec_dequeueInputBuffer"))?
            {
                DequeuedInputBufferResult::Buffer(mut buffer) => {
                    let len = fill(buffer.buffer_mut())?;
                    return self
                        .queue_input_buffer(buffer, 0, len, pts_us, flags)
                        .map_err(media(operation, "AMediaCodec_queueInputBuffer"));
                }
                DequeuedInputBufferResult::TryAgainLater => drain(self)?,
            }
        }
        Err(fail(
            operation,
            "the codec offered no input buffer within 500 ms",
        ))
    }

    /// Takes the next output buffer within `timeout`: `Some` with its
    /// bytes, flags and presentation time, `None` when nothing came.
    /// Format and buffer-set changes are consumed silently.
    pub(crate) fn next_output(
        &self,
        operation: Operation,
        timeout: Duration,
    ) -> Result<Option<Output>, CodecError> {
        loop {
            match self
                .dequeue_output_buffer(timeout)
                .map_err(media(operation, "AMediaCodec_dequeueOutputBuffer"))?
            {
                DequeuedOutputBufferInfoResult::Buffer(buffer) => {
                    let info = *buffer.info();
                    let start = usize::try_from(info.offset()).unwrap_or(0);
                    let len = usize::try_from(info.size()).unwrap_or(0);
                    let data = buffer
                        .buffer()
                        .get(start..start.saturating_add(len))
                        .map(<[u8]>::to_vec)
                        .ok_or_else(|| fail(operation, "output buffer range out of bounds"));
                    self.release_output_buffer(buffer, false)
                        .map_err(media(operation, "AMediaCodec_releaseOutputBuffer"))?;
                    return Ok(Some(Output {
                        data: data?,
                        flags: info.flags(),
                        pts_us: u64::try_from(info.presentation_time_us()).unwrap_or(0),
                    }));
                }
                DequeuedOutputBufferInfoResult::TryAgainLater => return Ok(None),
                DequeuedOutputBufferInfoResult::OutputFormatChanged
                | DequeuedOutputBufferInfoResult::OutputBuffersChanged => {}
            }
        }
    }
}

/// One output buffer, copied out of the codec.
pub(crate) struct Output {
    pub(crate) data: Vec<u8>,
    pub(crate) flags: u32,
    pub(crate) pts_us: u64,
}

impl Output {
    pub(crate) fn is(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }
}

/// Copies `src` into a codec input buffer.
pub(crate) fn copy_into(
    dst: &mut [std::mem::MaybeUninit<u8>],
    src: &[u8],
    operation: Operation,
) -> Result<usize, CodecError> {
    if dst.len() < src.len() {
        return Err(fail(
            operation,
            format!(
                "input buffer of {} bytes is too small for {} bytes",
                dst.len(),
                src.len()
            ),
        ));
    }
    for (slot, &byte) in dst.iter_mut().zip(src) {
        slot.write(byte);
    }
    Ok(src.len())
}
