//! Codec-generic NVDEC decoder.
//!
//! Single generic implementation that targets H.264 today and HEVC
//! behind `nvdec-hevc-decode`. Mirrors the encoder's
//! [`ferricast_encoder::nvenc::NvencEncoder<C>`] shape: a sealed
//! [`NvdecCodec`] marker fixes the codec at compile time, the rest of
//! the pipeline (`shiguredo_nvcodec::Decoder` setup, Annex-B parsing,
//! NV12 output assembly) is identical across codecs.
//!
//! Output: `CapturedFrame::Cpu(RawFrame)` with
//! `PixelFormat::Nv12`. NVDEC produces NV12 natively (Y plane + tightly
//! interleaved UV plane); we forward it as-is. Downstream consumers
//! that need BGRA pay the conversion cost on their side — sinks that
//! upload directly to a GPU texture (eframe / wgpu) can consume NV12
//! through a fragment-shader colour conversion at zero CPU cost.
//!
//! The binding drops CUVID timestamps, so each picture's is worked out
//! from the slice headers instead (see the `timestamps` module).

mod h264;

mod timestamps;

use std::collections::VecDeque;
use std::marker::PhantomData;

use bytes::Bytes;
use ferricast_core::{BackendId, CodecError, CodecResult, Operation};
use ferricast_core::{
    CapturedFrame, Codec, ColourDescription, ColourRange, DecoderConfig, EncodedFrame,
    FerricastError, Result, VideoDecoder,
};
use shiguredo_nvcodec::{
    Decoder as NvDecoder, DecoderCodec, DecoderConfig as NvDecCfg, SurfaceFormat,
};
use tracing::{debug, info};

use self::sealed::Headers as _;
use crate::nv12::{self, OutputFormat};
use timestamps::{Picture, Timestamps};

/// Sealed marker pinning a [`NvdecDecoder`] to one bitstream codec.
pub trait NvdecCodec: sealed::Sealed + Send + 'static {
    /// Ferricast-side codec identity. Drives [`VideoDecoder::CODEC`]
    /// and is what the facade matches against `DecoderConfig::codec`
    /// before configuring the underlying NVDEC session.
    const FERRICAST_CODEC: Codec;
    /// shiguredo codec selector used by `query_caps` + `Decoder::new`.
    const NV_CODEC: DecoderCodec;
    const NAME: &'static str;
}

mod sealed {
    use super::Picture;

    pub trait Sealed {
        /// Reads what timing the pictures needs from the bitstream.
        type Headers: Headers;
    }

    pub trait Headers: Default {
        /// The pictures in one packet that NVDEC is expected to show.
        fn shown(&mut self, packet: &[u8]) -> Vec<Picture>;
    }
}

/// H.264 marker.
pub struct H264;
impl sealed::Sealed for H264 {
    type Headers = h264::Headers;
}
impl NvdecCodec for H264 {
    const FERRICAST_CODEC: Codec = Codec::H264;
    const NV_CODEC: DecoderCodec = DecoderCodec::H264;
    const NAME: &'static str = "H.264";
}

/// HEVC marker.
#[cfg(feature = "nvdec-hevc-decode")]
pub struct Hevc;
#[cfg(feature = "nvdec-hevc-decode")]
impl sealed::Sealed for Hevc {
    type Headers = crate::h265::timestamps::Headers;
}
#[cfg(feature = "nvdec-hevc-decode")]
impl NvdecCodec for Hevc {
    const FERRICAST_CODEC: Codec = Codec::H265;
    const NV_CODEC: DecoderCodec = DecoderCodec::Hevc;
    const NAME: &'static str = "HEVC";
}

impl sealed::Headers for h264::Headers {
    fn shown(&mut self, packet: &[u8]) -> Vec<Picture> {
        self.picture(packet).into_iter().collect()
    }
}

#[cfg(feature = "nvdec-hevc-decode")]
impl sealed::Headers for crate::h265::timestamps::Headers {
    fn shown(&mut self, packet: &[u8]) -> Vec<Picture> {
        self.pictures(&rust_h265::parse_annex_b(packet))
            .into_iter()
            .filter(|picture| !picture.skipped)
            .map(|picture| Picture {
                sequence: picture.sequence,
                max_waiting: picture.max_num_reorder,
            })
            .collect()
    }
}

/// Codec-generic NVDEC decoder.
pub struct NvdecDecoder<C: NvdecCodec> {
    decoder: Option<NvDecoder>,
    /// One packet can release several pictures (a B-frame run ending);
    /// they wait here so none is dropped.
    ready: VecDeque<CapturedFrame>,
    headers: C::Headers,
    timestamps: Timestamps,
    format: OutputFormat,
    _codec: PhantomData<C>,
}

impl<C: NvdecCodec> Default for NvdecDecoder<C> {
    fn default() -> Self {
        Self {
            decoder: None,
            ready: VecDeque::new(),
            headers: C::Headers::default(),
            timestamps: Timestamps::default(),
            format: OutputFormat::default(),
            _codec: PhantomData,
        }
    }
}

impl<C: NvdecCodec> NvdecDecoder<C> {
    /// Moves every picture NVDEC has finished into `ready`.
    fn drain(&mut self) -> Result<()> {
        let Some(decoder) = self.decoder.as_mut() else {
            return Ok(());
        };
        while let Some(nv_frame) = decoder
            .next_frame()
            .map_err(|e| FerricastError::Decode(format!("NVDEC ({}) next_frame: {e}", C::NAME)))?
        {
            // NV12: Y plane (full res) + interleaved UV (half height),
            // both at the CUDA pitch, forwarded as one contiguous buffer.
            let y_stride = nv_frame.y_stride();
            let (width, height) = (nv_frame.width(), nv_frame.height());
            let mut data = Vec::with_capacity(y_stride * height * 3 / 2);
            data.extend_from_slice(nv_frame.y_plane());
            data.extend_from_slice(nv_frame.uv_plane());
            // The binding does not surface the VUI colour description.
            let frame = nv12::deliver(
                Bytes::from(data),
                y_stride,
                (width, height),
                self.timestamps.claim(),
                self.format.0,
                ColourDescription::unspecified(ColourRange::Limited),
            )
            .ok_or_else(|| {
                FerricastError::Decode(format!("NVDEC ({}) returned short planes", C::NAME))
            })?;
            self.ready.push_back(CapturedFrame::Cpu(frame));
        }
        Ok(())
    }
}

impl<C: NvdecCodec> NvdecDecoder<C> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Capability probe — fails when libcuda / libnvcuvid aren't on
    /// the runtime LD path, or the GPU doesn't support this codec.
    /// Cheap: no decoder session is created.
    pub fn probe() -> Result<()> {
        let caps = NvDecoder::query_caps(C::NV_CODEC, /* device_id = */ 0)
            .map_err(|e| FerricastError::Decode(format!("NVDEC ({}): query_caps: {e}", C::NAME)))?;
        if !caps.is_supported {
            return Err(FerricastError::Decode(format!(
                "NVDEC ({}): codec not supported by the active GPU",
                C::NAME
            )));
        }
        debug!(
            codec = C::NAME,
            max_width = caps.max_width,
            max_height = caps.max_height,
            "NVDEC caps probe OK"
        );
        Ok(())
    }
}

// SAFETY: `shiguredo_nvcodec::Decoder` holds a CUDA context handle
// that's only valid on the creating thread, but every method push/pop
// the context internally before calling NVDEC. `VideoDecoder: Send`
// is satisfied because the trait's `&mut self` contract guarantees
// no concurrent access.
unsafe impl<C: NvdecCodec> Send for NvdecDecoder<C> {}

impl<C: NvdecCodec> NvdecDecoder<C> {
    fn try_configure(&mut self, config: &DecoderConfig) -> Result<()> {
        if config.codec != C::FERRICAST_CODEC {
            return Err(FerricastError::Decode(format!(
                "NVDEC ({}) asked to decode {:?}",
                C::NAME,
                config.codec
            )));
        }

        // Cap session at 4× decode surfaces. NVDEC keeps reference
        // frames internally; 4 is the documented minimum that handles
        // any IPBB pattern. HEVC and H.264 both fit under this for the
        // single-stream low-latency use case ferricast targets.
        let cfg = NvDecCfg {
            codec: C::NV_CODEC,
            device_id: 0,
            max_num_decode_surfaces: 4,
            // Low latency: emit as soon as the picture is reconstructed.
            // `0` is documented as "no display delay" in CUDA's NVDEC
            // sample headers.
            max_display_delay: 0,
            surface_format: SurfaceFormat::Nv12,
        };
        let decoder = NvDecoder::new(cfg)
            .map_err(|e| FerricastError::Decode(format!("NVDEC ({}) open: {e}", C::NAME)))?;
        info!(codec = C::NAME, "NVDEC decoder ready");
        self.decoder = Some(decoder);
        self.ready.clear();
        self.format = OutputFormat(config.pixel_format);
        self.headers = C::Headers::default();
        self.timestamps = Timestamps::default();
        Ok(())
    }

    fn try_decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        let decoder = self.decoder.as_mut().ok_or_else(|| {
            FerricastError::Decode(format!("NVDEC ({}) decode before configure()", C::NAME))
        })?;

        // Read before NVDEC sees the packet so its parameter sets are
        // known whatever happens to the decode.
        let pictures = self.headers.shown(&frame.data);
        // NVDEC's parser owns picture-boundary detection; one packet may
        // release 0, 1 or N pictures depending on the GOP structure.
        decoder
            .decode(&frame.data)
            .map_err(|e| FerricastError::Decode(format!("NVDEC ({}) decode: {e}", C::NAME)))?;
        for picture in pictures {
            self.timestamps.push(picture, frame.timestamp_us);
        }
        self.drain()?;
        self.timestamps.settle();
        Ok(self.ready.pop_front())
    }

    fn try_flush(&mut self) -> Result<Vec<CapturedFrame>> {
        if let Some(decoder) = self.decoder.as_mut() {
            decoder
                .finish()
                .map_err(|e| FerricastError::Decode(format!("NVDEC ({}) finish: {e}", C::NAME)))?;
            // Everything left comes out now, in display order, so the
            // pending timestamps are handed out earliest first.
            self.drain()?;
        }
        self.timestamps = Timestamps::default();
        Ok(self.ready.drain(..).collect())
    }
}

impl<C: NvdecCodec> VideoDecoder for NvdecDecoder<C> {
    const CODEC: Codec = C::FERRICAST_CODEC;

    fn configure(&mut self, config: &DecoderConfig) -> CodecResult<()> {
        self.try_configure(config).map_err(CodecError::from_backend(
            BackendId::Nvdec,
            Operation::Configure,
        ))
    }

    fn decode(&mut self, frame: EncodedFrame) -> CodecResult<Option<CapturedFrame>> {
        self.try_decode(frame).map_err(CodecError::from_backend(
            BackendId::Nvdec,
            Operation::Decode,
        ))
    }

    fn flush(&mut self) -> CodecResult<Vec<CapturedFrame>> {
        self.try_flush()
            .map_err(CodecError::from_backend(BackendId::Nvdec, Operation::Flush))
    }
}

/// Public alias preserving an "this is the H.264 NVDEC backend" type
/// name in the H.264 facade.
pub type NvdecH264Decoder = NvdecDecoder<H264>;

/// HEVC alias. Only present when `nvdec-hevc-decode` is on.
#[cfg(feature = "nvdec-hevc-decode")]
pub type NvdecH265Decoder = NvdecDecoder<Hevc>;
