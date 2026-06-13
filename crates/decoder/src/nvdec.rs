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

use std::marker::PhantomData;

use bytes::Bytes;
use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, PixelFormat, RawFrame,
    Result, VideoDecoder,
};
use shiguredo_nvcodec::{
    Decoder as NvDecoder, DecoderCodec, DecoderConfig as NvDecCfg, SurfaceFormat,
};
use tracing::{debug, info};

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
    pub trait Sealed {}
}

/// H.264 marker.
pub struct H264;
impl sealed::Sealed for H264 {}
impl NvdecCodec for H264 {
    const FERRICAST_CODEC: Codec = Codec::H264;
    const NV_CODEC: DecoderCodec = DecoderCodec::H264;
    const NAME: &'static str = "H.264";
}

/// HEVC marker.
#[cfg(feature = "nvdec-hevc-decode")]
pub struct Hevc;
#[cfg(feature = "nvdec-hevc-decode")]
impl sealed::Sealed for Hevc {}
#[cfg(feature = "nvdec-hevc-decode")]
impl NvdecCodec for Hevc {
    const FERRICAST_CODEC: Codec = Codec::H265;
    const NV_CODEC: DecoderCodec = DecoderCodec::Hevc;
    const NAME: &'static str = "HEVC";
}

/// Codec-generic NVDEC decoder.
pub struct NvdecDecoder<C: NvdecCodec> {
    decoder: Option<NvDecoder>,
    _codec: PhantomData<C>,
}

impl<C: NvdecCodec> Default for NvdecDecoder<C> {
    fn default() -> Self {
        Self {
            decoder: None,
            _codec: PhantomData,
        }
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

impl<C: NvdecCodec> VideoDecoder for NvdecDecoder<C> {
    const CODEC: Codec = C::FERRICAST_CODEC;

    fn configure(&mut self, config: &DecoderConfig) -> Result<()> {
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
        Ok(())
    }

    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        let decoder = self.decoder.as_mut().ok_or_else(|| {
            FerricastError::Decode(format!("NVDEC ({}) decode before configure()", C::NAME))
        })?;

        // Feed the bitstream packet. NVDEC's parser owns demuxing — we
        // hand it Annex-B and it sorts the NALs / slices / picture
        // boundaries internally. May produce 0, 1 or N output frames
        // per call depending on the GOP structure.
        decoder
            .decode(&frame.data)
            .map_err(|e| FerricastError::Decode(format!("NVDEC ({}) decode: {e}", C::NAME)))?;

        let nv_frame = decoder
            .next_frame()
            .map_err(|e| FerricastError::Decode(format!("NVDEC ({}) next_frame: {e}", C::NAME)))?;
        let Some(nv_frame) = nv_frame else {
            return Ok(None);
        };

        // NV12: Y plane (full res) + interleaved UV (half h × full w
        // when stride aligned, but the CUDA pitch may exceed width).
        // We forward as a single contiguous CPU buffer with the Y
        // stride; downstream sinks read planes via standard NV12
        // offsets (`Y` = [0, pitch*h), `UV` = [pitch*h, pitch*h*3/2)).
        let y_stride = nv_frame.y_stride();
        let height = nv_frame.height();
        let mut data = Vec::with_capacity(y_stride * height * 3 / 2);
        data.extend_from_slice(nv_frame.y_plane());
        data.extend_from_slice(nv_frame.uv_plane());

        Ok(Some(CapturedFrame::Cpu(RawFrame {
            width: nv_frame.width() as u32,
            height: height as u32,
            stride: y_stride as u32,
            format: PixelFormat::Nv12,
            data: Bytes::from(data),
            timestamp_us: frame.timestamp_us,
        })))
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        let Some(decoder) = self.decoder.as_mut() else {
            return Ok(Vec::new());
        };
        decoder
            .finish()
            .map_err(|e| FerricastError::Decode(format!("NVDEC ({}) finish: {e}", C::NAME)))?;

        let mut out = Vec::new();
        // Drain whatever frames the decoder buffered before EOS.
        loop {
            let Some(nv_frame) = decoder.next_frame().map_err(|e| {
                FerricastError::Decode(format!("NVDEC ({}) flush next_frame: {e}", C::NAME))
            })?
            else {
                break;
            };
            let y_stride = nv_frame.y_stride();
            let height = nv_frame.height();
            let mut data = Vec::with_capacity(y_stride * height * 3 / 2);
            data.extend_from_slice(nv_frame.y_plane());
            data.extend_from_slice(nv_frame.uv_plane());
            out.push(CapturedFrame::Cpu(RawFrame {
                width: nv_frame.width() as u32,
                height: height as u32,
                stride: y_stride as u32,
                format: PixelFormat::Nv12,
                data: Bytes::from(data),
                // Flushed frames have no input-side timestamp — leave 0
                // and let the sink interpolate from cadence.
                timestamp_us: 0,
            }));
        }
        Ok(out)
    }
}

/// Public alias preserving an "this is the H.264 NVDEC backend" type
/// name in the H.264 facade.
pub type NvdecH264Decoder = NvdecDecoder<H264>;

/// HEVC alias. Only present when `nvdec-hevc-decode` is on.
#[cfg(feature = "nvdec-hevc-decode")]
pub type NvdecH265Decoder = NvdecDecoder<Hevc>;
