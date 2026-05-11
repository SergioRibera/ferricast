//! NVENC H.264 encoder.
//!
//! NVIDIA's VAAPI implementation (`nvidia-vaapi-driver`) is decode-
//! only — every entrypoint exposed in `vainfo` is `VAEntrypointVLD`.
//! Hardware encode on NVIDIA happens through NVENC, a completely
//! separate API that lives in `libnvidia-encode.so` and uses CUDA
//! contexts (no kernels, just a context handle).
//!
//! We use `shiguredo_nvcodec`, which wraps the public Video Codec
//! SDK headers and loads `libcuda.so.1` + `libnvidia-encode.so.1`
//! dynamically. Build doesn't need the NVIDIA SDK installed; the
//! binary stays runnable on hardware without NVIDIA — `Encoder::new`
//! errors and our factory falls through to x264.
//!
//! Profile / preset choices match what receivers (Chromecast /
//! Miracast / AirPlay) expect:
//!
//! * **Codec / profile**: H.264 High. Chromecast (Gen 1+),
//!   Miracast and AirPlay 2 all accept High up to Level 4.1
//!   (1080p60). High gives ~10-20% better compression than Main
//!   at the same quality through CABAC and 8×8 transforms — both
//!   already enabled by NVENC's preset defaults. We only step
//!   down to Constrained Baseline if a downstream protocol
//!   handler tells us its receiver is too old; that's a future
//!   knob.
//! * **Preset**: P3 with `LOW_LATENCY` tuning. P1..P7 trade speed
//!   for quality (P1 fastest, P7 best). For live screencast at
//!   60 fps, P3 is the sweet spot — it lets the encoder keep up
//!   without dropping frames while keeping the bitrate manageable.
//! * **Rate control**: CBR at the user-configured bitrate.
//! * **GOP**: closed, IDR every `keyframe_interval` frames, no
//!   B-frames (`frame_interval_p = 1`). Same shape as the VA-API
//!   path — keeps the HLS segmenter happy because every segment
//!   starts at a keyframe.
//! * **Input**: `BufferFormat::Argb` is NVENC's name for
//!   `A8R8G8B8` packed pixels — same in-memory byte order as our
//!   public `PixelFormat::Bgra` (`B`, `G`, `R`, `A` from low to
//!   high). No CPU NV12 swizzle needed; `shiguredo_nvcodec` does
//!   the host→device copy and the colour conversion lives in
//!   NVENC's hardware path.

use bytes::Bytes;
use ferricast_core::{
    CapturedFrame, Codec, EncodedFrame, EncoderConfig, FerricastError, PixelFormat, Result,
    VideoEncoder,
};
use shiguredo_nvcodec::{
    BufferFormat, CodecConfig, EncodeOptions, Encoder as NvEncoder, EncoderCodec, EncoderConfig as NvCfg,
    H264EncoderConfig, H264Profile, PictureType, Preset, RateControlMode, ReconfigureParams,
    TuningInfo,
};
use tracing::{debug, info};

pub struct NvencH264Encoder {
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
}

#[derive(Clone, Copy)]
struct NvencCfg {
    width: u32,
    height: u32,
    fps: u32,
    keyframe_interval: u32,
}

// SAFETY: the underlying `shiguredo_nvcodec::Encoder` holds a CUDA
// context handle that's only valid on the thread it was created
// on, BUT all method calls re-activate the context via
// `cu_ctx_push_current` before invoking NVENC, so transferring the
// encoder to another thread is safe as long as no two threads use
// it simultaneously. The `VideoEncoder: Send` contract is met by
// our `&mut self` API: only one caller can hold the encoder at a
// time. Sync is not implemented and not needed.
unsafe impl Send for NvencH264Encoder {}

impl NvencH264Encoder {
    /// Try to bring up NVENC. Returns `Err` if `libcuda` /
    /// `libnvidia-encode` aren't available, no NVIDIA device is
    /// present, or the requested resolution / profile isn't
    /// supported by the GPU.
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

        // Quick capability probe — fails fast on systems without
        // NVENC so the factory drops through cleanly.
        let _caps = NvEncoder::query_caps(EncoderCodec::H264, /* device_id = */ 0)
            .map_err(|e| FerricastError::Encoder(format!("NVENC: query_caps: {e}")))?;
        debug!("NVENC H.264 caps query OK");

        // Map the cross-crate profile constraint onto NVENC's
        // GUID-based profile enum. `max_h264_profile == None`
        // defaults to Main — the conservative floor that every
        // receiver protocol (Chromecast 1st gen and up, Miracast,
        // AirPlay 2) can decode. The manager fills this in from
        // the target device's `DeviceCapabilities` so a
        // Chromecast Ultra / Google TV / Android TV gets High
        // (≈10-20% better compression) while a 1st-gen
        // Chromecast falls back to Main automatically.
        let profile = match cfg.max_h264_profile {
            Some(ferricast_core::H264Profile::High) => H264Profile::High,
            Some(ferricast_core::H264Profile::Baseline) => H264Profile::Baseline,
            // Main or unspecified → Main (safe default).
            _ => H264Profile::Main,
        };

        let nvcfg = NvCfg {
            codec: CodecConfig::H264(H264EncoderConfig {
                // Profile from `DeviceCapabilities`. Old generic
                // Chromecasts (md = \"Chromecast\") choke on High-
                // profile features (CABAC, 8x8 transform, weighted
                // prediction); they get Main from the device-side
                // capability table. Newer receivers (Ultra,
                // Android TV, Google TV) negotiate High and benefit
                // from the better compression. Symptom of getting
                // this wrong on an old device was the
                // LOADING-forever state we hit in the field:
                // receiver accepts the LOAD, transitions to
                // playerState=IDLE / extendedStatus=LOADING, hardware
                // decoder silently rejects the bitstream, never
                // progresses.
                profile: Some(profile),
                idr_period: Some(cfg.keyframe_interval.max(1)),
            }),
            width: cfg.width.max(16),
            height: cfg.height.max(16),
            max_encode_width: None,
            max_encode_height: None,
            framerate_num: cfg.fps.max(1),
            framerate_den: 1,
            average_bitrate: Some((cfg.bitrate_kbps as u32).saturating_mul(1000).max(500_000)),
            preset: Preset::P3,
            tuning_info: TuningInfo::LOW_LATENCY,
            rate_control_mode: RateControlMode::Cbr,
            gop_length: Some(cfg.keyframe_interval.max(1)),
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

        let mut encoder = NvEncoder::new(nvcfg)
            .map_err(|e| FerricastError::Encoder(format!("NVENC: open session: {e}")))?;
        // Lock the peak bitrate to the average so NVENC can't spike
        // 2× into scene changes. NVENC's CBR with only
        // `average_bitrate` set still allows large per-frame peaks
        // (default `maxBitRate` is implementation-defined and in
        // practice ~2× average). On a 1st-/2nd-gen Chromecast
        // pulling HLS over 2.4 GHz Wi-Fi those peaks blow past what
        // the receiver's buffer can absorb and the player aborts
        // with `detailedErrorCode=301` (MEDIA_NETWORK_ERROR) after a
        // few segments. Forcing max == average gives true CBR; the
        // encoder lowers per-frame quality on busy frames instead of
        // spending bandwidth, which is the right trade-off for screen
        // share where most frames are easy.
        let avg = (cfg.bitrate_kbps as u32).saturating_mul(1000).max(500_000);
        if let Err(e) = encoder.reconfigure(ReconfigureParams {
            average_bitrate: Some(avg),
            max_bitrate: Some(avg),
            ..Default::default()
        }) {
            // Non-fatal: the encoder is usable, just with default
            // peak. Worst case we still hit 301 on weak Wi-Fi
            // chromecasts; log loud so it's obvious if this
            // regresses.
            tracing::warn!(error = %e, "NVENC: failed to lock max_bitrate to average; CBR will allow peaks");
        }
        info!(
            width = cfg.width,
            height = cfg.height,
            fps = cfg.fps,
            "NVENC H.264 encoder ready"
        );

        Ok(Self {
            encoder,
            cfg: NvencCfg {
                width: cfg.width.max(16),
                height: cfg.height.max(16),
                fps: cfg.fps.max(1),
                keyframe_interval: cfg.keyframe_interval.max(1),
            },
            frame_count: 0,
            pending_keyframe: false,
        })
    }
}

impl VideoEncoder for NvencH264Encoder {
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, _config: &EncoderConfig) -> Result<()> {
        // The `shiguredo_nvcodec::Encoder` is built fully configured
        // by `probe_with`. A reconfigure-after-the-fact would mean
        // tearing down the CUDA context and rebuilding the session;
        // we don't support that today (the factory hands us a
        // freshly-probed instance whenever the resolution changes).
        Ok(())
    }

    fn encode(&mut self, frame: CapturedFrame) -> Result<EncodedFrame> {
        // NVENC accepts BGRA/RGBA directly via `BufferFormat::Argb`/
        // `::Abgr`. For `Gpu` frames we still need CPU bytes —
        // `shiguredo_nvcodec`'s `encode()` does its own host→device
        // copy via `cuMemcpyHtoD`.
        //
        // TODO(zero-copy): replace this with the
        // `cuImportExternalMemory(OPAQUE_FD)` →
        // `cuExternalMemoryGetMappedBuffer` →
        // `NvEncRegisterResource(CUDADEVICEPTR)` chain so the
        // PipeWire dmabuf goes straight into NVENC. shiguredo
        // doesn't expose the registration step (its
        // `register_input_resource` is `pub(crate)`); when the
        // perf gap matters we vendor the crate or fork it the
        // same way we did with cros-libva, mirror its
        // `CudaLibrary` with the two extra `cu*ExternalMemory*`
        // function pointers, and add a `register_external_dmabuf`
        // method on `Encoder`. ~200 lines once we commit. The
        // current readback path costs ~5 ms / frame at 1080p
        // (Vulkan blit + memcpy + libcuda host->device); a future
        // commit eliminates it entirely.
        let raw = frame.into_cpu()?;

        let expected = (self.cfg.width as usize)
            .saturating_mul(self.cfg.height as usize)
            .saturating_mul(4);
        if raw.data.len() < expected {
            return Err(FerricastError::Encoder(format!(
                "NVENC: short frame {} bytes, want {expected}",
                raw.data.len()
            )));
        }

        let force_idr = self.frame_count % (self.cfg.keyframe_interval as u64) == 0
            || std::mem::take(&mut self.pending_keyframe);
        let options = EncodeOptions {
            force_intra: false,
            force_idr,
            // Repeat SPS/PPS on every IDR — Chromecast/Miracast
            // receivers reset between segments, and HLS players
            // re-fetch the playlist mid-stream. Always shipping the
            // headers means the player can join at any IDR.
            output_spspps: force_idr,
        };

        self.encoder
            .encode(&raw.data[..expected], &options)
            .map_err(|e| FerricastError::Encoder(format!("NVENC encode: {e}")))?;

        // shiguredo's encoder buffers encoded frames internally
        // (NVENC may emit multiple frames per call when it
        // catches up on lookahead). Drain everything available;
        // we return the freshest, push the rest back into a
        // first-out queue handled by `flush()` if any are pending.
        let nv_frame = self
            .encoder
            .next_frame()
            .ok_or_else(|| FerricastError::Encoder("NVENC: encoder produced no output".into()))?;

        let pts = self.frame_count;
        self.frame_count += 1;
        let is_keyframe =
            matches!(nv_frame.picture_type(), PictureType::I | PictureType::Idr);

        Ok(EncodedFrame {
            codec: Codec::H264,
            data: Bytes::from(nv_frame.into_data()),
            timestamp_us: raw.timestamp_us,
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
        // `probe_with` is preserved — without this, max would keep
        // its previous (possibly higher) value and the controller's
        // downshift would be subverted by NVENC's peak headroom.
        let bps = (kbps as u32).saturating_mul(1000).max(500_000);
        self.encoder
            .reconfigure(ReconfigureParams {
                average_bitrate: Some(bps),
                max_bitrate: Some(bps),
                ..Default::default()
            })
            .map_err(|e| {
                FerricastError::Encoder(format!("NVENC live reconfigure (bitrate): {e}"))
            })?;
        info!(kbps, "NVENC bitrate live-reconfigured");
        Ok(())
    }

    fn get_headers(&mut self) -> Result<Vec<u8>> {
        // NVENC builds SPS/PPS inside the bitstream when we ask
        // for `output_spspps`. The HLS segmenter prepends them
        // already via this method, so we synthesise them here
        // through `nvEncGetSequenceParams`.
        self.encoder
            .get_sequence_params()
            .map_err(|e| FerricastError::Encoder(format!("NVENC: get_sequence_params: {e}")))
    }
}
