//! Object-safe views of the codec traits, so a registry can hold
//! backends of different types. Every [`VideoEncoder`] / [`VideoDecoder`]
//! / [`AudioEncoder`] / [`AudioDecoder`] gets one through a blanket impl.

use ferricast_core::{
    AudioDecoder, AudioDecoderConfig, AudioEncoder, AudioEncoderConfig, AudioFrame, CapturedFrame,
    CodecResult, DecodedAudio, DecoderConfig, EncodedFrame, EncoderConfig, VideoDecoder,
    VideoEncoder,
};

/// A running video encoder backend.
pub trait VideoEncoderSession: Send {
    /// See [`VideoEncoder::configure`].
    ///
    /// # Errors
    /// The backend's.
    fn configure(&mut self, config: &EncoderConfig) -> CodecResult<()>;
    /// See [`VideoEncoder::reset`].
    ///
    /// # Errors
    /// The backend's.
    fn reset(&mut self, config: &EncoderConfig) -> CodecResult<()>;
    /// See [`VideoEncoder::encode`].
    ///
    /// # Errors
    /// The backend's.
    fn encode(&mut self, frame: CapturedFrame) -> CodecResult<EncodedFrame>;
    /// See [`VideoEncoder::flush`].
    ///
    /// # Errors
    /// The backend's.
    fn flush(self: Box<Self>) -> CodecResult<Vec<EncodedFrame>>;
    /// See [`VideoEncoder::get_headers`].
    ///
    /// # Errors
    /// The backend's.
    fn headers(&mut self) -> CodecResult<Vec<u8>>;
    /// See [`VideoEncoder::request_keyframe`].
    fn request_keyframe(&mut self);
    /// See [`VideoEncoder::set_bitrate_kbps`].
    ///
    /// # Errors
    /// The backend's.
    fn set_bitrate_kbps(&mut self, kbps: u32) -> CodecResult<()>;
}

impl<E: VideoEncoder> VideoEncoderSession for E {
    fn configure(&mut self, config: &EncoderConfig) -> CodecResult<()> {
        VideoEncoder::configure(self, config)
    }

    fn reset(&mut self, config: &EncoderConfig) -> CodecResult<()> {
        VideoEncoder::reset(self, config)
    }

    fn encode(&mut self, frame: CapturedFrame) -> CodecResult<EncodedFrame> {
        VideoEncoder::encode(self, frame)
    }

    fn flush(self: Box<Self>) -> CodecResult<Vec<EncodedFrame>> {
        VideoEncoder::flush(*self)
    }

    fn headers(&mut self) -> CodecResult<Vec<u8>> {
        VideoEncoder::get_headers(self)
    }

    fn request_keyframe(&mut self) {
        VideoEncoder::request_keyframe(self);
    }

    fn set_bitrate_kbps(&mut self, kbps: u32) -> CodecResult<()> {
        VideoEncoder::set_bitrate_kbps(self, kbps)
    }
}

/// A running video decoder backend.
pub trait VideoDecoderSession: Send {
    /// See [`VideoDecoder::configure`].
    ///
    /// # Errors
    /// The backend's.
    fn configure(&mut self, config: &DecoderConfig) -> CodecResult<()>;
    /// See [`VideoDecoder::decode`].
    ///
    /// # Errors
    /// The backend's.
    fn decode(&mut self, frame: EncodedFrame) -> CodecResult<Option<CapturedFrame>>;
    /// See [`VideoDecoder::flush`].
    ///
    /// # Errors
    /// The backend's.
    fn flush(&mut self) -> CodecResult<Vec<CapturedFrame>>;
}

impl<D: VideoDecoder> VideoDecoderSession for D {
    fn configure(&mut self, config: &DecoderConfig) -> CodecResult<()> {
        VideoDecoder::configure(self, config)
    }

    fn decode(&mut self, frame: EncodedFrame) -> CodecResult<Option<CapturedFrame>> {
        VideoDecoder::decode(self, frame)
    }

    fn flush(&mut self) -> CodecResult<Vec<CapturedFrame>> {
        VideoDecoder::flush(self)
    }
}

/// A running audio encoder backend.
pub trait AudioEncoderSession: Send {
    /// See [`AudioEncoder::configure`].
    ///
    /// # Errors
    /// The backend's.
    fn configure(&mut self, config: &AudioEncoderConfig) -> CodecResult<()>;
    /// See [`AudioEncoder::encode`].
    ///
    /// # Errors
    /// The backend's.
    fn encode(&mut self, frame: &AudioFrame) -> CodecResult<()>;
    /// See [`AudioEncoder::take_output`].
    fn take_output(&mut self) -> Vec<AudioFrame>;
    /// See [`AudioEncoder::codec_config`].
    fn codec_config(&self) -> Vec<u8>;
    /// See [`AudioEncoder::flush`].
    ///
    /// # Errors
    /// The backend's.
    fn flush(self: Box<Self>) -> CodecResult<Vec<AudioFrame>>;
}

impl<E: AudioEncoder> AudioEncoderSession for E {
    fn configure(&mut self, config: &AudioEncoderConfig) -> CodecResult<()> {
        AudioEncoder::configure(self, config)
    }

    fn encode(&mut self, frame: &AudioFrame) -> CodecResult<()> {
        AudioEncoder::encode(self, frame)
    }

    fn take_output(&mut self) -> Vec<AudioFrame> {
        AudioEncoder::take_output(self)
    }

    fn codec_config(&self) -> Vec<u8> {
        AudioEncoder::codec_config(self)
    }

    fn flush(self: Box<Self>) -> CodecResult<Vec<AudioFrame>> {
        AudioEncoder::flush(*self)
    }
}

/// A running audio decoder backend.
pub trait AudioDecoderSession: Send {
    /// See [`AudioDecoder::configure`].
    ///
    /// # Errors
    /// The backend's.
    fn configure(&mut self, config: &AudioDecoderConfig) -> CodecResult<()>;
    /// See [`AudioDecoder::decode`].
    ///
    /// # Errors
    /// The backend's.
    fn decode(&mut self, frame: AudioFrame) -> CodecResult<Option<DecodedAudio>>;
    /// See [`AudioDecoder::flush`].
    ///
    /// # Errors
    /// The backend's.
    fn flush(&mut self) -> CodecResult<Vec<DecodedAudio>>;
}

impl<D: AudioDecoder> AudioDecoderSession for D {
    fn configure(&mut self, config: &AudioDecoderConfig) -> CodecResult<()> {
        AudioDecoder::configure(self, config)
    }

    fn decode(&mut self, frame: AudioFrame) -> CodecResult<Option<DecodedAudio>> {
        AudioDecoder::decode(self, frame)
    }

    fn flush(&mut self) -> CodecResult<Vec<DecodedAudio>> {
        AudioDecoder::flush(self)
    }
}

/// Opens a backend of type `T` by configuring its `Default` value: the
/// [`crate::OpenFn`] for every backend whose `configure` does the work.
///
/// # Errors
/// The backend's `configure` error.
pub fn open_video_encoder<T: VideoEncoder + Default + 'static>(
    config: &EncoderConfig,
) -> CodecResult<Box<dyn VideoEncoderSession>> {
    let mut encoder = T::default();
    VideoEncoder::configure(&mut encoder, config)?;
    Ok(Box::new(encoder))
}

/// [`open_video_encoder`] for decoders.
///
/// # Errors
/// The backend's `configure` error.
pub fn open_video_decoder<T: VideoDecoder + Default + 'static>(
    config: &DecoderConfig,
) -> CodecResult<Box<dyn VideoDecoderSession>> {
    let mut decoder = T::default();
    VideoDecoder::configure(&mut decoder, config)?;
    Ok(Box::new(decoder))
}

/// [`open_video_encoder`] for audio encoders.
///
/// # Errors
/// The backend's `configure` error.
pub fn open_audio_encoder<T: AudioEncoder + Default + 'static>(
    config: &AudioEncoderConfig,
) -> CodecResult<Box<dyn AudioEncoderSession>> {
    let mut encoder = T::default();
    AudioEncoder::configure(&mut encoder, config)?;
    Ok(Box::new(encoder))
}

/// [`open_video_encoder`] for audio decoders.
///
/// # Errors
/// The backend's `configure` error.
pub fn open_audio_decoder<T: AudioDecoder + Default + 'static>(
    config: &AudioDecoderConfig,
) -> CodecResult<Box<dyn AudioDecoderSession>> {
    let mut decoder = T::default();
    AudioDecoder::configure(&mut decoder, config)?;
    Ok(Box::new(decoder))
}
