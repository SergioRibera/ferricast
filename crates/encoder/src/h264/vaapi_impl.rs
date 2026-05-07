//! VA-API H.264 encoder.
//!
//! ## Status
//!
//! **Detection + plumbing landed; the encode loop body is a TODO.**
//!
//! Implementing a complete H.264 encoder through `cros-libva` is on
//! the order of 800+ lines of careful code:
//!
//! * `VAEncSequenceParameterBufferH264` — 25+ fields, mostly
//!   per-stream invariants (level_idc, picture dimensions in MBs,
//!   bitrate, GOP structure, VUI flags).
//! * `VAEncPictureParameterBufferH264` — per-frame state including
//!   reference picture list management, IDR / P frame fields, and a
//!   reference to the `VABufferID` of the coded buffer Vulkan will
//!   write the bitstream into.
//! * `VAEncSliceParameterBufferH264` — 30+ fields including weight
//!   tables and ref-picture lists.
//! * Packed SPS / PPS NAL units — H.264 syntax with Exp-Golomb
//!   coding for the integer fields, RBSP trailing bits, and the
//!   AnnexB start-code prefix. ~200 lines on its own.
//! * Surface management — either allocate `VASurface`s natively or
//!   import the dmabuf carried by `GpuFrame` via
//!   `VASurfaceAttribExternalBuffers`.
//! * Coded buffer extraction with `VACodedBufferSegment`.
//!
//! Doing all of that correctly without a way to test on the actual
//! target hardware would invite subtle bugs. The factory in
//! [`super`] is wired up so adding the encode body is now an
//! isolated change — `configure` already opens the display, picks
//! the profile and creates the config; what's missing is the
//! per-frame buffer construction and submit/sync flow inside
//! [`X264H264Encoder::encode`]'s VA-API counterpart here.
//!
//! Until then, [`VaapiH264Encoder::probe`] returns `Err` so the
//! factory falls back to x264 transparently.

use std::path::Path;
use std::rc::Rc;

use bytes::Bytes;
use cros_libva::{Display, VAProfile};
use ferricast_core::{
    CapturedFrame, Codec, EncodedFrame, EncoderConfig, FerricastError, PixelFormat, Result,
    VideoEncoder,
};
use tracing::{debug, info, warn};

/// Render-node candidates we'll try, in order. The portal-screencast
/// dmabuf usually lives on the primary GPU; falling through to the
/// next node lets us cope with multi-GPU systems where the iGPU is
/// the dmabuf producer.
const RENDER_NODES: &[&str] = &[
    "/dev/dri/renderD128",
    "/dev/dri/renderD129",
    "/dev/dri/renderD130",
];

/// H.264 profiles we can target, in preference order.
///
/// Constrained Baseline is the lowest-common-denominator across
/// Chromecast / Miracast / AirPlay receivers; we prefer Main when
/// the GPU exposes it (better compression, still widely accepted)
/// and fall back to CB when nothing else matches.
const H264_PROFILES: &[VAProfile::Type] = &[
    VAProfile::VAProfileH264Main,
    VAProfile::VAProfileH264ConstrainedBaseline,
];

pub struct VaapiH264Encoder {
    /// `Rc` because cros-libva's `Display` is not Sync and the
    /// `Picture` / `Context` types it spawns hold an `Rc<Display>`.
    /// We keep one per encoder; the encoder itself is `Send` via
    /// the unsafe impl below (see comment).
    display: Rc<Display>,
    /// Profile cros-libva agreed on. Stored so the encode body can
    /// consult it when building parameter buffers.
    #[allow(dead_code)]
    profile: VAProfile::Type,
    /// Encoder configuration, latched in `configure()` and consulted
    /// by `encode()`.
    config: Option<EncoderConfig>,
}

// SAFETY: cros-libva uses `Rc<Display>` internally because its
// `Picture` types borrow the display non-thread-safely. We never
// share the `Rc` across threads — the encoder is consumed by
// `&mut self` from a single thread (the segmenter task), and the
// `tokio::spawn` that owns it ensures it stays on that one thread.
// Marking Send manually lets us put the encoder behind the
// `VideoEncoder: Send` trait bound.
unsafe impl Send for VaapiH264Encoder {}

impl VaapiH264Encoder {
    /// Try to bring up VA-API. Returns `Ok(Self)` only when:
    /// * `libva.so` loads,
    /// * a render node opens,
    /// * one of [`H264_PROFILES`] is supported with the
    ///   `VAEntrypointEncSlice` entrypoint.
    ///
    /// Otherwise returns `Err(FerricastError::Encoder(_))` and the
    /// caller falls back to x264.
    pub fn probe() -> Result<Self> {
        let display = open_render_node().ok_or_else(|| {
            FerricastError::Encoder("VA-API: no usable DRM render node".into())
        })?;

        let vendor = display.query_vendor_string().unwrap_or_default();
        debug!(%vendor, "VA-API display opened");

        let supported_profiles = display.query_config_profiles().map_err(|e| {
            FerricastError::Encoder(format!("VA-API: query_config_profiles failed: {e}"))
        })?;

        let profile = H264_PROFILES
            .iter()
            .copied()
            .find(|p| supported_profiles.contains(p))
            .ok_or_else(|| {
                FerricastError::Encoder(
                    "VA-API: no compatible H.264 profile (Main / Constrained Baseline)".into(),
                )
            })?;

        // Make sure the encode entrypoint actually exists for this
        // profile — some drivers expose the profile for decode only.
        let entrypoints = display
            .query_config_entrypoints(profile)
            .map_err(|e| FerricastError::Encoder(format!("VA-API: query_entrypoints: {e}")))?;
        if !entrypoints
            .iter()
            .any(|e| *e == cros_libva::VAEntrypoint::VAEntrypointEncSlice)
        {
            return Err(FerricastError::Encoder(format!(
                "VA-API: profile {profile:?} has no VAEntrypointEncSlice"
            )));
        }

        info!(?profile, %vendor, "VA-API H.264 encoder available");

        // Phase 2 ends here: the actual encode loop is the next
        // chunk of work. Surface this as a clear error so the
        // factory falls back to x264 instead of silently dropping
        // frames.
        Err(FerricastError::Encoder(
            "VA-API encoder body not implemented yet; falling back to x264".into(),
        ))
        // When the encode loop lands, replace the line above with:
        //   Ok(Self { display, profile, config: None })
    }
}

fn open_render_node() -> Option<Rc<Display>> {
    for node in RENDER_NODES {
        let path = Path::new(node);
        if !path.exists() {
            continue;
        }
        match Display::open_drm_display(path) {
            Ok(d) => {
                debug!(%node, "opened VA-API DRM display");
                return Some(d);
            }
            Err(e) => {
                debug!(%node, error = %e, "render node open failed");
            }
        }
    }
    // Last resort: let cros-libva pick.
    Display::open()
}

impl VideoEncoder for VaapiH264Encoder {
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, config: &EncoderConfig) -> Result<()> {
        // Store the config; once the encode body is implemented this
        // is where we'd build the `VAConfig` and `VAContext`.
        self.config = Some(config.clone());
        if !matches!(config.pixel_format, PixelFormat::Bgra | PixelFormat::Nv12) {
            return Err(FerricastError::Encoder(format!(
                "VA-API: unsupported pixel format {:?}",
                config.pixel_format
            )));
        }
        Ok(())
    }

    fn encode(&mut self, _frame: CapturedFrame) -> Result<EncodedFrame> {
        // TODO(vaapi): build VAEncSequenceParameterBufferH264 +
        // VAEncPictureParameterBufferH264 + VAEncSliceParameterBufferH264,
        // pack SPS / PPS NAL units, submit Picture, sync, extract
        // CodedBuffer.
        let _ = &self.display; // silence dead-field once probe Errs out
        warn!("VA-API encode body not implemented yet");
        Err(FerricastError::Encoder(
            "VA-API encoder body not implemented".into(),
        ))
    }

    fn flush(self) -> Result<Vec<EncodedFrame>> {
        Ok(Vec::new())
    }

    fn get_headers(&mut self) -> Result<Vec<u8>> {
        // SPS + PPS NAL units would be packed here. Returning an
        // empty Vec is fine for the unimplemented path (the factory
        // never reaches this code today).
        Ok(Vec::new())
    }
}

/// Unused for now; kept so the future encode body has a place to
/// drop the bitstream-packing helper.
#[allow(dead_code)]
fn _bytes_placeholder() -> Bytes {
    Bytes::new()
}
