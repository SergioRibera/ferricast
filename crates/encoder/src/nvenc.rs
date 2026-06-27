//! Codec-generic NVENC encoder.
//!
//! Single implementation that targets H.264 today and HEVC behind the
//! `nvenc-hevc` Cargo feature. The bitstream-specific bits (codec
//! GUID for `query_caps`, the `CodecConfig` enum variant) are
//! supplied by a sealed [`NvencCodec`] marker; everything else —
//! session creation, BGRA / RGBA ingest, dmabuf import + caching,
//! CBR lock, live bitrate reconfigure, IDR forcing, parameter-set
//! retrieval — is identical.
//!
//! Why a single generic instead of two parallel files: shiguredo's
//! `Encoder` already abstracts the codec at runtime via
//! `CodecConfig::H264(_)` vs `CodecConfig::Hevc(_)`, and every other
//! call site (`encode`, `next_frame`, `reconfigure`,
//! `get_sequence_params`, `register_external_dmabuf`,
//! `encode_external`) is the same shape. Keeping two copies would
//! double the surface area for every NVENC fix going forward (CBR
//! lock, profile mapping, zero-copy cache eviction, picture-type
//! mapping) with zero behavioural divergence.
//!
//! Public surface is the two aliases [`NvencH264Encoder`] +
//! [`NvencH265Encoder`]. The H.264 facade in [`super::h264`]
//! re-exports the H.264 alias for backwards compatibility; the H.265
//! facade in [`super::h265`] re-exports the H.265 one.

use std::marker::PhantomData;

use bytes::Bytes;
use ferricast_core::{
    CapturedFrame, Codec, EncodedFrame, EncoderConfig, FerricastError, PixelFormat, Result,
    VideoEncoder,
};
use shiguredo_nvcodec::{
    BufferFormat, CodecConfig, EncodeOptions, Encoder as NvEncoder, EncoderCodec,
    EncoderConfig as NvCfg, H264EncoderConfig, H264Profile as NvH264Profile, PictureType, Preset,
    RateControlMode, ReconfigureParams, TuningInfo,
};
#[cfg(feature = "nvenc-hevc")]
use shiguredo_nvcodec::{HevcEncoderConfig, HevcProfile as NvHevcProfile};
use tracing::{debug, info};

#[cfg(feature = "nvenc-zero-copy")]
use std::collections::HashMap;
#[cfg(feature = "nvenc-zero-copy")]
use std::os::fd::RawFd;
#[cfg(feature = "nvenc-zero-copy")]
use ferricast_core::GpuFrame;
#[cfg(feature = "nvenc-zero-copy")]
use shiguredo_nvcodec::RegisteredResource;
#[cfg(feature = "nvenc-zero-copy")]
use tracing::warn;

/// Sealed marker that pins a [`NvencEncoder`] to one bitstream codec.
///
/// The sealed boundary is enforced by `mod sealed` below — only the
/// codec markers defined in this module implement [`NvencCodec`]. Down-
/// stream crates can't add new variants; new NVENC-supported codecs
/// (AV1 is the next obvious one) get added here, alongside the
/// `CodecConfig` mapping, in one place.
pub trait NvencCodec: sealed::Sealed + Send + Sync + 'static {
    /// Ferricast-side codec identity. Drives [`VideoEncoder::CODEC`]
    /// and the [`EncodedFrame::codec`] tag the muxer reads.
    const FERRICAST_CODEC: Codec;
    /// Codec selector for `Encoder::query_caps`. Pinned at the marker
    /// so the capability probe matches the codec we'll actually
    /// configure.
    const NV_ENCODER_CODEC: EncoderCodec;
    /// Human-readable label for tracing.
    const NAME: &'static str;

    /// Build the codec-specific NVENC config for this encoder session.
    /// Profile selection reads the matching `EncoderConfig` ceiling
    /// (`max_h264_profile` / `max_h265_profile`); IDR period is the
    /// resolved frame count from [`EncoderConfig::keyframe_interval_frames`].
    fn make_codec_config(cfg: &EncoderConfig) -> CodecConfig;
}

mod sealed {
    pub trait Sealed {}
}

/// H.264 marker. Selects the H.264 codec variant of NVENC.
pub struct H264;
impl sealed::Sealed for H264 {}
impl NvencCodec for H264 {
    const FERRICAST_CODEC: Codec = Codec::H264;
    const NV_ENCODER_CODEC: EncoderCodec = EncoderCodec::H264;
    const NAME: &'static str = "H.264";

    fn make_codec_config(cfg: &EncoderConfig) -> CodecConfig {
        // Map the cross-crate profile constraint onto NVENC's GUID
        // enum. `max_h264_profile == None` defaults to Main — the
        // conservative floor every receiver protocol (Chromecast 1st
        // gen and up, Miracast, AirPlay 2) can decode. The manager
        // fills this in from the target device's `DeviceCapabilities`
        // so newer hardware gets High (~10-20% better compression).
        let profile = match cfg.max_h264_profile {
            Some(ferricast_core::H264Profile::High) => NvH264Profile::High,
            Some(ferricast_core::H264Profile::Baseline) => NvH264Profile::Baseline,
            _ => NvH264Profile::Main,
        };
        CodecConfig::H264(H264EncoderConfig {
            profile: Some(profile),
            idr_period: Some(cfg.keyframe_interval_frames()),
        })
    }
}

/// HEVC marker. Selects the HEVC codec variant of NVENC.
#[cfg(feature = "nvenc-hevc")]
pub struct Hevc;
#[cfg(feature = "nvenc-hevc")]
impl sealed::Sealed for Hevc {}
#[cfg(feature = "nvenc-hevc")]
impl NvencCodec for Hevc {
    const FERRICAST_CODEC: Codec = Codec::H265;
    const NV_ENCODER_CODEC: EncoderCodec = EncoderCodec::Hevc;
    const NAME: &'static str = "HEVC";

    fn make_codec_config(cfg: &EncoderConfig) -> CodecConfig {
        // Map cross-crate H.265 profile onto NVENC's GUID enum.
        // `max_h265_profile == None` defaults to Main (8-bit 4:2:0),
        // the universally-decoded floor for every HEVC-capable
        // receiver in the field. Main10 (10-bit) is opt-in via
        // `DeviceCapabilities::max_h265_profile`.
        let profile = match cfg.max_h265_profile {
            Some(ferricast_core::H265Profile::Main10) => NvHevcProfile::Main10,
            _ => NvHevcProfile::Main,
        };
        CodecConfig::Hevc(HevcEncoderConfig {
            profile: Some(profile),
            idr_period: Some(cfg.keyframe_interval_frames()),
        })
    }
}

/// Codec-generic NVENC encoder. Two public aliases:
/// [`NvencH264Encoder`] (always present when `nvenc` is on) and
/// [`NvencH265Encoder`] (present when `nvenc-hevc` is on).
pub struct NvencEncoder<C: NvencCodec> {
    encoder: NvEncoder,
    cfg: NvencCfg,
    /// Bumped every successful `encode()`; used as the bitstream
    /// timestamp the muxer translates into PTS/DTS.
    frame_count: u64,
    /// Set by [`request_keyframe`]; OR-ed into the natural
    /// interval-based IDR decision on the next [`encode`] call so
    /// the segmenter can anchor segment boundaries to wall clock
    /// when the capture stalls.
    pending_keyframe: bool,
    /// `NvEncRegisterResource` is expensive (driver round-trip plus
    /// `cuExternalMemoryImport`), so we cache the registration
    /// keyed by `(fd, modifier, dims)`. PipeWire and our wayland-direct
    /// dmabuf path both reuse fds across frames (buffer pools), so
    /// the steady-state hit rate is ~100% — only the first frame
    /// after each pool slot pays the registration cost.
    ///
    /// Eviction: on `configure()` (dimension change). Stale fds
    /// fail registration on next encode and we drop the entry
    /// implicitly via the soft-fallback path.
    #[cfg(feature = "nvenc-zero-copy")]
    dmabuf_cache: HashMap<DmabufKey, RegisteredResource>,
    _codec: PhantomData<C>,
}

#[derive(Clone, Copy)]
struct NvencCfg {
    width: u32,
    height: u32,
    fps: u32,
    keyframe_interval: u32,
}

/// Stable identity of a DMA-BUF for registration caching. Keyed on
/// `(fd, modifier, dims)` rather than `fd` alone because some
/// compositors recycle fd numbers after `close(2)` and a stale
/// `RegisteredResource` would happily encode garbage.
#[cfg(feature = "nvenc-zero-copy")]
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct DmabufKey {
    fd: RawFd,
    modifier: u64,
    width: u32,
    height: u32,
    stride: u32,
}

// SAFETY: the underlying `shiguredo_nvcodec::Encoder` holds a CUDA
// context handle that's only valid on the thread it was created
// on, BUT all method calls re-activate the context via
// `cu_ctx_push_current` before invoking NVENC, so transferring the
// encoder to another thread is safe as long as no two threads use
// it simultaneously. The `VideoEncoder: Send` contract is met by
// our `&mut self` API: only one caller can hold the encoder at a
// time. Sync is not implemented and not needed.
unsafe impl<C: NvencCodec> Send for NvencEncoder<C> {}

impl<C: NvencCodec> NvencEncoder<C> {
    /// Try to bring up NVENC at the default resolution. Mostly useful
    /// for capability probes from the factory; production code uses
    /// [`Self::probe_with`] with the caller's actual config.
    pub fn probe() -> Result<Self> {
        Self::probe_with(EncoderConfig::default())
    }

    pub fn probe_with(cfg: EncoderConfig) -> Result<Self> {
        if !matches!(cfg.pixel_format, PixelFormat::Bgra | PixelFormat::Rgba) {
            return Err(FerricastError::Encoder(format!(
                "NVENC: input pixel format {:?} not supported (need Bgra/Rgba)",
                cfg.pixel_format
            )));
        }

        // Quick capability probe — fails fast on systems without NVENC
        // so the factory drops through cleanly.
        let _caps = NvEncoder::query_caps(C::NV_ENCODER_CODEC, /* device_id = */ 0)
            .map_err(|e| FerricastError::Encoder(format!("NVENC ({}): query_caps: {e}", C::NAME)))?;
        debug!(codec = C::NAME, "NVENC caps query OK");

        let nvcfg = NvCfg {
            codec: C::make_codec_config(&cfg),
            width: cfg.width.max(16),
            height: cfg.height.max(16),
            max_encode_width: None,
            max_encode_height: None,
            framerate_num: cfg.fps.max(1),
            framerate_den: 1,
            average_bitrate: Some((cfg.bitrate_kbps).saturating_mul(1000).max(500_000)),
            preset: Preset::P3,
            tuning_info: TuningInfo::LOW_LATENCY,
            rate_control_mode: RateControlMode::Cbr,
            gop_length: Some(cfg.keyframe_interval_frames()),
            // No B-frames — keeps every segment self-contained for
            // the HLS / cast-protocol consumers.
            frame_interval_p: 1,
            buffer_format: match cfg.pixel_format {
                PixelFormat::Bgra => BufferFormat::Argb,
                PixelFormat::Rgba => BufferFormat::Abgr,
                _ => unreachable!(), // gated above
            },
            device_id: 0,
        };

        let mut encoder = NvEncoder::new(nvcfg).map_err(|e| {
            FerricastError::Encoder(format!("NVENC ({}): open session: {e}", C::NAME))
        })?;
        // Lock the peak bitrate to the average so NVENC can't spike
        // 2× into scene changes. Same story documented at length in
        // the original H.264 impl: HLS over weak Wi-Fi to a 1st-/2nd-
        // gen Chromecast aborts with `detailedErrorCode=301` when the
        // peak overshoots the receiver's buffer. Forcing max ==
        // average gives true CBR; the encoder lowers per-frame quality
        // on busy frames instead of spending bandwidth, which is the
        // right trade-off for screen share where most frames are easy.
        let avg = (cfg.bitrate_kbps).saturating_mul(1000).max(500_000);
        if let Err(e) = encoder.reconfigure(ReconfigureParams {
            average_bitrate: Some(avg),
            max_bitrate: Some(avg),
            ..Default::default()
        }) {
            tracing::warn!(
                codec = C::NAME,
                error = %e,
                "NVENC: failed to lock max_bitrate to average; CBR will allow peaks"
            );
        }
        info!(
            codec = C::NAME,
            width = cfg.width,
            height = cfg.height,
            fps = cfg.fps,
            "NVENC encoder ready"
        );

        Ok(Self {
            encoder,
            cfg: NvencCfg {
                width: cfg.width.max(16),
                height: cfg.height.max(16),
                fps: cfg.fps.max(1),
                keyframe_interval: cfg.keyframe_interval_frames(),
            },
            frame_count: 0,
            pending_keyframe: false,
            #[cfg(feature = "nvenc-zero-copy")]
            dmabuf_cache: HashMap::new(),
            _codec: PhantomData,
        })
    }

    fn encode_cpu(&mut self, host: &[u8], options: &EncodeOptions) -> Result<()> {
        let expected = (self.cfg.width as usize)
            .saturating_mul(self.cfg.height as usize)
            .saturating_mul(4);
        if host.len() < expected {
            return Err(FerricastError::Encoder(format!(
                "NVENC ({}): short frame {} bytes, want {expected}",
                C::NAME,
                host.len()
            )));
        }
        self.encoder
            .encode(&host[..expected], options)
            .map_err(|e| FerricastError::Encoder(format!("NVENC ({}) encode (cpu): {e}", C::NAME)))
    }

    /// Zero-copy encode via the forked `shiguredo_nvcodec` API.
    /// Registers the DMA-BUF (or hits the per-fd cache) and tells
    /// the encoder to use that resource as the input surface for
    /// this frame. No CPU bytes touched.
    #[cfg(feature = "nvenc-zero-copy")]
    fn encode_dmabuf(&mut self, g: &GpuFrame, options: &EncodeOptions) -> Result<()> {
        let format = match g.format {
            PixelFormat::Bgra => BufferFormat::Argb,
            PixelFormat::Rgba => BufferFormat::Abgr,
            other => {
                return Err(FerricastError::Encoder(format!(
                    "NVENC ({}): unsupported GPU pixel format {other:?}",
                    C::NAME
                )));
            }
        };

        let key = DmabufKey {
            fd: g.plane.fd,
            modifier: g.plane.modifier,
            width: g.width,
            height: g.height,
            stride: g.plane.stride,
        };

        if !self.dmabuf_cache.contains_key(&key) {
            let registered = self
                .encoder
                .register_external_dmabuf(
                    g.plane.fd,
                    g.plane.modifier,
                    g.width,
                    g.height,
                    g.plane.stride,
                    format,
                )
                .map_err(|e| {
                    FerricastError::Encoder(format!(
                        "NVENC ({}) register_external_dmabuf: {e}",
                        C::NAME
                    ))
                })?;
            self.dmabuf_cache.insert(key, registered);
        }
        let registered = self
            .dmabuf_cache
            .get(&key)
            .expect("just-inserted above");

        self.encoder
            .encode_external(registered, options)
            .map_err(|e| {
                FerricastError::Encoder(format!("NVENC ({}) encode_external: {e}", C::NAME))
            })?;
        Ok(())
    }
}

impl<C: NvencCodec> VideoEncoder for NvencEncoder<C> {
    const CODEC: Codec = C::FERRICAST_CODEC;

    fn configure(&mut self, _config: &EncoderConfig) -> Result<()> {
        // The `shiguredo_nvcodec::Encoder` is built fully configured
        // by `probe_with`. A reconfigure-after-the-fact would mean
        // tearing down the CUDA context and rebuilding the session;
        // we don't support that today (the factory hands us a freshly-
        // probed instance whenever the resolution changes).
        //
        // Drop the dmabuf cache anyway — `register_external_dmabuf`
        // bakes the dimensions into NVENC's internal surface, so a
        // cached `RegisteredResource` from a previous size is junk
        // even if the fd happens to match. The cache will refill
        // organically on the first encode at the new size.
        #[cfg(feature = "nvenc-zero-copy")]
        self.dmabuf_cache.clear();
        Ok(())
    }

    fn encode(&mut self, frame: CapturedFrame) -> Result<EncodedFrame> {
        let force_idr = self.frame_count % (self.cfg.keyframe_interval as u64) == 0
            || std::mem::take(&mut self.pending_keyframe);
        let options = EncodeOptions {
            force_intra: false,
            force_idr,
            // Repeat parameter sets on every IDR. For H.264 this is
            // SPS/PPS; for HEVC the NVENC flag also emits VPS. HLS
            // players join at any IDR and Chromecast/Miracast receivers
            // reset between segments, so always shipping the headers
            // means the player can re-acquire the stream mid-flight.
            output_spspps: force_idr,
        };

        let timestamp_us = frame.timestamp_us();
        #[cfg(feature = "nvenc-zero-copy")]
        {
            match frame {
                CapturedFrame::Gpu(g) => {
                    if let Err(e) = self.encode_dmabuf(&g, &options) {
                        // Soft fall-back: log and try the CPU path
                        // on the same frame. Costs one Vulkan readback
                        // for this frame; subsequent frames retry
                        // the GPU path so single-frame failures don't
                        // sticky.
                        warn!(codec = C::NAME, %e, "NVENC dmabuf path failed; falling back to CPU readback for this frame");
                        let raw = CapturedFrame::Gpu(g).into_cpu()?;
                        self.encode_cpu(&raw.data, &options)?;
                    }
                }
                CapturedFrame::Cpu(raw) => {
                    self.encode_cpu(&raw.data, &options)?;
                }
            }
        }
        #[cfg(not(feature = "nvenc-zero-copy"))]
        {
            let raw = frame.into_cpu()?;
            self.encode_cpu(&raw.data, &options)?;
        }

        let nv_frame = self.encoder.next_frame().ok_or_else(|| {
            FerricastError::Encoder(format!("NVENC ({}): encoder produced no output", C::NAME))
        })?;

        let pts = self.frame_count;
        self.frame_count += 1;
        let is_keyframe = matches!(nv_frame.picture_type(), PictureType::I | PictureType::Idr);

        Ok(EncodedFrame {
            codec: C::FERRICAST_CODEC,
            data: Bytes::from(nv_frame.into_data()),
            timestamp_us,
            duration_us: Some(1_000_000 / self.cfg.fps as u64),
            is_keyframe,
            pts_dts: (pts, pts),
        })
    }

    fn flush(self) -> Result<Vec<EncodedFrame>> {
        // Without B-frames there's no reorder queue, so an end-of-
        // stream flush wouldn't yield anything. Return empty.
        Ok(Vec::new())
    }

    fn request_keyframe(&mut self) {
        self.pending_keyframe = true;
    }

    fn set_bitrate_kbps(&mut self, kbps: u32) -> Result<()> {
        // NVENC supports live bitrate reconfiguration without a GOP
        // boundary. Pass both `average_bitrate` and `max_bitrate` set
        // to the same value so the true-CBR invariant from
        // `probe_with` is preserved.
        let bps = (kbps).saturating_mul(1000).max(500_000);
        self.encoder
            .reconfigure(ReconfigureParams {
                average_bitrate: Some(bps),
                max_bitrate: Some(bps),
                ..Default::default()
            })
            .map_err(|e| {
                FerricastError::Encoder(format!(
                    "NVENC ({}) live reconfigure (bitrate): {e}",
                    C::NAME
                ))
            })?;
        info!(codec = C::NAME, kbps, "NVENC bitrate live-reconfigured");
        Ok(())
    }

    fn get_headers(&mut self) -> Result<Vec<u8>> {
        // NVENC builds the parameter sets inside the bitstream when we
        // ask for `output_spspps`. The HLS segmenter prepends them
        // already via this method, so we synthesise them here via
        // `nvEncGetSequenceParams` — which returns SPS/PPS for H.264
        // and VPS/SPS/PPS for HEVC.
        let a = self.encoder.get_sequence_params().map_err(|e| {
            FerricastError::Encoder(format!("NVENC ({}): get_sequence_params: {e}", C::NAME))
        });

        // TEMP CHANGE
        println!("nvenc sps/pps {:?}", a);

        a
    }
}

/// Public alias preserving the old type name. The H.264 facade in
/// [`super::h264`] re-exports this.
pub type NvencH264Encoder = NvencEncoder<H264>;

/// HEVC encoder alias. Only present when the `nvenc-hevc` feature is
/// on; the H.265 facade re-exports this.
#[cfg(feature = "nvenc-hevc")]
pub type NvencH265Encoder = NvencEncoder<Hevc>;
