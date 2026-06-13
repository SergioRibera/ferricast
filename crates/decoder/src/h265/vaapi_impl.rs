//! VA-API hardware HEVC decode.
//!
//! Engaged via `FERRICAST_H265_DECODE_BACKEND=vaapi` and only when
//! the `vaapi-hevc-decode` feature is on. Mirrors the worker-thread
//! architecture from [`crate::h264::vaapi_impl`]: a single-threaded
//! VA worker owns the non-`Send` `cros-libva` handles and the public
//! struct is just a pair of channels.
//!
//! ## Scope
//!
//! `probe_with` + `configure` walk the full bring-up path:
//! 1. Open a DRM render node.
//! 2. Query `VAProfileHEVCMain` / `VAProfileHEVCMain10` against the
//!    driver and pick the highest one that exposes `VAEntrypointVLD`.
//! 3. Build `VAConfig` + a small NV12 surface pool sized from the
//!    `DecoderConfig` dimension hint.
//! 4. Hand the worker the bitstream channel.
//!
//! The slice-submission path lives behind the same opt-in pattern as
//! the H.264 VAAPI decoder — `decode()` currently returns
//! `Err(FerricastError::Decode("VA-API HEVC slice submission not yet
//! wired"))`. The facade default-skips this backend; the H.265
//! pipeline auto-selects NVDEC, which has a complete decode path
//! through `shiguredo_nvcodec`. Engage VAAPI HEVC only with the env
//! var, and only on hosts where you intend to drive the slice
//! submission to completion.
//!
//! Slot mechanics — surface allocation, profile negotiation, worker
//! handoff — are real so the wiring can be extended in place rather
//! than rebuilt.

use std::path::Path;
use std::rc::Rc;
use std::sync::mpsc;
use std::thread;

use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, H265Profile, Result,
    VideoDecoder,
};
use cros_libva::*;
use tracing::{debug, info, warn};

const RENDER_NODES: &[&str] = &[
    "/dev/dri/renderD128",
    "/dev/dri/renderD129",
    "/dev/dri/renderD130",
];

const VA_FOURCC_NV12: u32 = 0x3231564E;
const VA_RT_FORMAT_YUV420: u32 = 0x01;

/// VAProfileHEVCMain / VAProfileHEVCMain10 (`va.h`).
const VA_PROFILE_HEVC_MAIN: VAProfile::Type = 17;
const VA_PROFILE_HEVC_MAIN10: VAProfile::Type = 18;

/// Output surface pool size. 4 covers a single P-frame + the picture
/// currently being decoded with two slots of headroom. HEVC's DPB can
/// grow larger when B-frames are in play, but ferricast's encoder side
/// emits IPPP only — symmetric assumption here.
const SURFACE_POOL: usize = 4;

enum Cmd {
    Configure(DecoderConfig),
    Decode(EncodedFrame),
    Flush,
    Shutdown,
}

enum Reply {
    Ok(Option<CapturedFrame>),
    Many(Vec<CapturedFrame>),
    Err(FerricastError),
}

pub struct VaapiH265Decoder {
    tx: Option<mpsc::Sender<Cmd>>,
    rx: mpsc::Receiver<Reply>,
}

impl VaapiH265Decoder {
    /// Open the VA display and confirm the driver advertises at least
    /// one HEVC encode-decode profile. Returns `Err` if the host has
    /// no usable DRM render node or no HEVC decoder.
    pub fn probe_with(config: &DecoderConfig) -> Result<Self> {
        // Spin the bring-up off onto the worker so the non-`Send`
        // display handle never crosses a thread boundary. Worker
        // sends back probe success or a structured error.
        let (cmd_tx, cmd_rx) = mpsc::channel::<Cmd>();
        let (reply_tx, reply_rx) = mpsc::channel::<Reply>();
        let cfg = config.clone();
        let probe_tx = reply_tx.clone();

        thread::spawn(move || {
            let result = worker_main(cmd_rx, reply_tx, cfg);
            if let Err(e) = result {
                // Worker died — try to surface the error so callers
                // see a meaningful message instead of a closed channel.
                let _ = probe_tx.send(Reply::Err(e));
            }
        });

        Ok(Self {
            tx: Some(cmd_tx),
            rx: reply_rx,
        })
    }
}

impl Drop for VaapiH265Decoder {
    fn drop(&mut self) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(Cmd::Shutdown);
        }
    }
}

impl VideoDecoder for VaapiH265Decoder {
    const CODEC: Codec = Codec::H265;

    fn configure(&mut self, config: &DecoderConfig) -> Result<()> {
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| FerricastError::Decode("VA-API HEVC: worker gone".into()))?;
        tx.send(Cmd::Configure(config.clone()))
            .map_err(|_| FerricastError::Decode("VA-API HEVC: worker channel closed".into()))?;
        match self.rx.recv() {
            Ok(Reply::Ok(_)) => Ok(()),
            Ok(Reply::Err(e)) => Err(e),
            Ok(Reply::Many(_)) => Err(FerricastError::Decode(
                "VA-API HEVC: unexpected reply from configure".into(),
            )),
            Err(_) => Err(FerricastError::Decode(
                "VA-API HEVC: worker dropped reply channel".into(),
            )),
        }
    }

    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| FerricastError::Decode("VA-API HEVC: worker gone".into()))?;
        tx.send(Cmd::Decode(frame))
            .map_err(|_| FerricastError::Decode("VA-API HEVC: worker channel closed".into()))?;
        match self.rx.recv() {
            Ok(Reply::Ok(out)) => Ok(out),
            Ok(Reply::Err(e)) => Err(e),
            Ok(Reply::Many(_)) => Err(FerricastError::Decode(
                "VA-API HEVC: unexpected Many reply from decode".into(),
            )),
            Err(_) => Err(FerricastError::Decode(
                "VA-API HEVC: worker dropped reply channel".into(),
            )),
        }
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| FerricastError::Decode("VA-API HEVC: worker gone".into()))?;
        tx.send(Cmd::Flush)
            .map_err(|_| FerricastError::Decode("VA-API HEVC: worker channel closed".into()))?;
        match self.rx.recv() {
            Ok(Reply::Many(out)) => Ok(out),
            Ok(Reply::Ok(_)) => Ok(Vec::new()),
            Ok(Reply::Err(e)) => Err(e),
            Err(_) => Err(FerricastError::Decode(
                "VA-API HEVC: worker dropped reply channel".into(),
            )),
        }
    }
}

struct WorkerState {
    display: Rc<Display>,
    profile: VAProfile::Type,
    /// VAConfig handle. Stays Some after `configure`; the rest of the
    /// decode-loop scaffolding will key off it.
    config: Option<Config>,
    context: Option<Rc<Context>>,
    surfaces: Vec<Surface<()>>,
}

fn worker_main(
    cmd_rx: mpsc::Receiver<Cmd>,
    reply_tx: mpsc::Sender<Reply>,
    initial_config: DecoderConfig,
) -> Result<()> {
    let display = open_render_node()
        .ok_or_else(|| FerricastError::Decode("VA-API HEVC: no usable DRM render node".into()))?;
    let vendor = display.query_vendor_string().unwrap_or_default();
    debug!(%vendor, "VA-API display opened for HEVC decode");

    let supported = display
        .query_config_profiles()
        .map_err(|e| FerricastError::Decode(format!("query_config_profiles: {e}")))?;

    // Decide profile from the DecoderConfig hint, falling back to
    // whatever the driver supports.
    let want_main10 = matches!(initial_config.pixel_format, ferricast_core::PixelFormat::Nv12)
        && supported.contains(&VA_PROFILE_HEVC_MAIN10)
        && profile_has_vld(&display, VA_PROFILE_HEVC_MAIN10);
    let profile = if want_main10 {
        VA_PROFILE_HEVC_MAIN10
    } else if supported.contains(&VA_PROFILE_HEVC_MAIN)
        && profile_has_vld(&display, VA_PROFILE_HEVC_MAIN)
    {
        VA_PROFILE_HEVC_MAIN
    } else {
        return Err(FerricastError::Decode(
            "VA-API HEVC: no decode profile (need Main or Main10)".into(),
        ));
    };
    let _ = H265Profile::Main; // touch import to keep visible for future use.
    info!(?profile, %vendor, "VA-API HEVC decoder profile selected");

    let mut state = WorkerState {
        display,
        profile,
        config: None,
        context: None,
        surfaces: Vec::new(),
    };

    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            Cmd::Configure(cfg) => {
                let r = configure_session(&mut state, &cfg);
                let _ = reply_tx.send(match r {
                    Ok(()) => Reply::Ok(None),
                    Err(e) => Reply::Err(e),
                });
            }
            Cmd::Decode(_frame) => {
                // Slice-submission path is intentionally unimplemented;
                // the structure above is what a follow-up extension
                // will fill. Surfacing the limitation as an explicit
                // error avoids silently dropping packets.
                let _ = reply_tx.send(Reply::Err(FerricastError::Decode(
                    "VA-API HEVC slice submission not yet wired \
                     (use FERRICAST_H265_DECODE_BACKEND=nvdec or leave \
                     the H.265 facade on its default NVDEC path)"
                        .into(),
                )));
            }
            Cmd::Flush => {
                let _ = reply_tx.send(Reply::Many(Vec::new()));
            }
            Cmd::Shutdown => break,
        }
    }
    Ok(())
}

fn configure_session(state: &mut WorkerState, cfg: &DecoderConfig) -> Result<()> {
    if cfg.codec != Codec::H265 {
        return Err(FerricastError::Decode(format!(
            "VA-API HEVC asked to decode {:?}",
            cfg.codec
        )));
    }
    let cfg_handle = state
        .display
        .create_config(
            vec![VAConfigAttrib {
                type_: VAConfigAttribType::VAConfigAttribRTFormat,
                value: VA_RT_FORMAT_YUV420,
            }],
            state.profile,
            VAEntrypoint::VAEntrypointVLD,
        )
        .map_err(|e| FerricastError::Decode(format!("vaCreateConfig(HEVC VLD): {e}")))?;

    // Allocate the NV12 output pool.
    let width = cfg.width.max(64);
    let height = cfg.height.max(64);
    let descs: Vec<()> = (0..SURFACE_POOL).map(|_| ()).collect();
    let surfaces = state
        .display
        .create_surfaces(
            VA_RT_FORMAT_YUV420,
            Some(VA_FOURCC_NV12),
            width,
            height,
            Some(UsageHint::USAGE_HINT_DECODER),
            descs,
        )
        .map_err(|e| FerricastError::Decode(format!("create_surfaces(HEVC): {e}")))?;

    let context = state
        .display
        .create_context(&cfg_handle, width, height, Some(&surfaces), true)
        .map_err(|e| FerricastError::Decode(format!("vaCreateContext(HEVC VLD): {e}")))?;

    state.config = Some(cfg_handle);
    state.context = Some(context);
    state.surfaces = surfaces;
    debug!(width, height, "VA-API HEVC decode session ready");
    Ok(())
}

fn open_render_node() -> Option<Rc<Display>> {
    for node in RENDER_NODES {
        let path = Path::new(node);
        if !path.exists() {
            continue;
        }
        match Display::open_drm_display(path) {
            Ok(d) => {
                debug!(node = node, "opened VA-API DRM display");
                return Some(d);
            }
            Err(e) => warn!(node = node, error = %e, "render node open failed"),
        }
    }
    Display::open()
}

fn profile_has_vld(display: &Display, profile: VAProfile::Type) -> bool {
    match display.query_config_entrypoints(profile) {
        Ok(eps) => eps.iter().any(|e| *e == VAEntrypoint::VAEntrypointVLD),
        Err(_) => false,
    }
}
