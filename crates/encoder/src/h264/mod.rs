//! H.264 encoder facade.
//!
//! Picks a backend at runtime, in order of preference:
//! 1. **VA-API** via `cros-libva` — Intel iGPU + AMD Mesa. Encode
//!    on the GPU, no CPU readback needed once the dmabuf import
//!    path is wired.
//! 2. **NVENC** via `shiguredo_nvcodec` — NVIDIA proprietary
//!    driver. NVIDIA's VA-API implementation is decode-only, so
//!    this is the path that actually engages the GPU on NVIDIA
//!    hardware.
//! 3. **openh264** — software fallback. Always available because
//!    `openh264` is a build-time dep *when its feature is on*.
//!
//! Each backend is gated behind its own Cargo feature (`openh264`,
//! `vaapi`, `nvenc`). At least one must be enabled — see the
//! `compile_error!` below. Backends not in the feature set are
//! removed at compile time from the enum, the dispatcher, and the
//! preference list; their files don't even get `mod`-included.
//!
//! All compiled-in backends implement the same [`VideoEncoder`]
//! trait so consumers don't need to care which one is active.
//! Output is plain H.264 — every receiver protocol we ship
//! (Chromecast / Miracast / AirPlay) decodes any of the three
//! backends' bitstreams identically.
//!
//! Construction has two phases:
//! * `H264Encoder::new()` returns an enum stuck in `Pending` —
//!   detection is deferred so we can size surfaces / SPS / NVENC
//!   session correctly from the user's actual resolution.
//! * `configure(&EncoderConfig)` walks the preference list with
//!   that config; on any failure at any backend (no driver, profile
//!   not supported, session creation refused, ...) it transparently
//!   tries the next one.

#[cfg(not(any(feature = "openh264", feature = "vaapi", feature = "nvenc")))]
compile_error!(
    "ferricast-encoder requires at least one of the `openh264` / `vaapi` / `nvenc` \
     features to be enabled — the H.264 facade has nothing to dispatch to otherwise. \
     Re-enable defaults or pick one explicitly, e.g. \
     `ferricast-encoder = { default-features = false, features = [\"openh264\"] }`."
);

// Helper modules used only by the VA-API backend today. Gated so a
// no-vaapi build doesn't compile them just to drop dead code.
#[cfg(feature = "vaapi")]
mod bitstream;
#[cfg(feature = "vaapi")]
mod headers;
#[cfg(feature = "vaapi")]
pub(crate) mod yuv;

#[cfg(feature = "openh264")]
mod openh264_impl;
#[cfg(feature = "vaapi")]
mod vaapi_impl;

use ferricast_core::{
    CapturedFrame, Codec, EncodedFrame, EncoderConfig, FerricastError, Result, VideoEncoder,
};
#[allow(unused_imports)]
use tracing::{info, warn};

#[cfg(feature = "nvenc")]
pub use crate::nvenc::NvencH264Encoder;
#[cfg(feature = "openh264")]
pub use openh264_impl::OpenH264Encoder;
#[cfg(feature = "vaapi")]
pub use vaapi_impl::VaapiH264Encoder;

const H264_BACKEND_VAR: &'static str = "FERRICAST_H264_BACKEND";

/// Backend-agnostic H.264 encoder. Variants are conditional on
/// their feature flag — disabled backends are removed from the enum
/// at compile time so disabling one doesn't leave dead `match`
/// arms.
pub enum H264Encoder {
    /// Pre-configure placeholder. Replaced with one of the concrete
    /// variants on first `configure()` call.
    Pending,
    /// Hardware VA-API path — Intel / AMD.
    #[cfg(feature = "vaapi")]
    Vaapi(VaapiH264Encoder),
    /// Hardware NVENC path — NVIDIA.
    #[cfg(feature = "nvenc")]
    Nvenc(NvencH264Encoder),
    /// Software openh264 path — always works when compiled in.
    #[cfg(feature = "openh264")]
    OpenH264(OpenH264Encoder),
}

impl H264Encoder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for H264Encoder {
    fn default() -> Self {
        H264Encoder::Pending
    }
}

impl VideoEncoder for H264Encoder {
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, config: &EncoderConfig) -> Result<()> {
        match self {
            H264Encoder::Pending => {
                let var = std::env::var(H264_BACKEND_VAR).unwrap_or_default();

                // Try NVENC first; if that fails (no NVIDIA, libs
                // missing) try VA-API; if that fails fall back to
                // openh264. NVENC takes priority over VA-API because
                // multi-GPU systems (NVIDIA dGPU + AMD iGPU is a
                // very common laptop / Ryzen-with-graphics combo)
                // would otherwise pick the iGPU's VA-API and force
                // every frame across PCIe twice — capture on NVIDIA
                // → readback to CPU → upload to AMD VA-API. NVENC
                // keeps the entire pipeline on the discrete GPU.
                //
                // Each branch is feature-gated. When a backend isn't
                // compiled in, the chain skips its probe and moves
                // to the next candidate. If all hardware backends
                // are disabled AND openh264 is off too, the
                // `compile_error!` at the top of the module would
                // already have triggered — so by the time we reach
                // the final fallback we're guaranteed *some* backend
                // exists.
                #[cfg(feature = "nvenc")]
                {
                    if var.is_empty() || var == "nvenc" {
                        match NvencH264Encoder::probe_with(config.clone()) {
                            Ok(enc) => {
                                info!(
                                    width = config.width,
                                    height = config.height,
                                    fps = config.fps,
                                    "H.264 encoder backend: NVENC"
                                );
                                *self = H264Encoder::Nvenc(enc);
                                return Ok(());
                            }
                            Err(e) => info!(
                                error = %e,
                                "NVENC unavailable, trying VA-API. \
                                 If you expected NVENC, ensure libcuda.so.1 + \
                                 libnvidia-encode.so.1 are on LD_LIBRARY_PATH \
                                 (NixOS: /run/opengl-driver/lib)."
                            ),
                        }
                    }
                }

                #[cfg(feature = "vaapi")]
                {
                    if var.is_empty() || var == "vaapi" {
                        match VaapiH264Encoder::probe_with(config.clone()) {
                            Ok(enc) => {
                                info!(
                                    width = config.width,
                                    height = config.height,
                                    fps = config.fps,
                                    "H.264 encoder backend: VA-API"
                                );
                                *self = H264Encoder::Vaapi(enc);
                                return Ok(());
                            }
                            Err(e) => info!(error = %e, "VA-API unavailable, falling back to openh264"),
                        }
                    }
                }

                #[cfg(feature = "openh264")]
                {
                    if var.is_empty() || var == "openh264" {
                        info!("H.264 encoder backend: openh264 (software)");
                        let mut x = OpenH264Encoder::default();
                        x.configure(config)?;
                        *self = H264Encoder::OpenH264(x);
                        return Ok(());
                    }
                }

                Err(FerricastError::Encoder(format!(
                    "no H.264 encoder backend available for FERRICAST_H264_BACKEND={var:?}"
                )))
            }
            #[cfg(feature = "vaapi")]
            H264Encoder::Vaapi(e) => match e.configure(config) {
                Ok(()) => Ok(()),
                Err(err) => fall_back_to_openh264_or_pending(self, config, err, "VA-API"),
            },
            #[cfg(feature = "nvenc")]
            H264Encoder::Nvenc(e) => match e.configure(config) {
                Ok(()) => Ok(()),
                Err(err) => fall_back_to_openh264_or_pending(self, config, err, "NVENC"),
            },
            #[cfg(feature = "openh264")]
            H264Encoder::OpenH264(e) => e.configure(config),
        }
    }

    fn encode(&mut self, frame: CapturedFrame) -> Result<EncodedFrame> {
        match self {
            H264Encoder::Pending => Err(FerricastError::Encoder(
                "H264Encoder::encode called before configure()".into(),
            )),
            #[cfg(feature = "vaapi")]
            H264Encoder::Vaapi(e) => e.encode(frame),
            #[cfg(feature = "nvenc")]
            H264Encoder::Nvenc(e) => e.encode(frame),
            #[cfg(feature = "openh264")]
            H264Encoder::OpenH264(e) => e.encode(frame),
        }
    }

    fn flush(self) -> Result<Vec<EncodedFrame>> {
        match self {
            H264Encoder::Pending => Ok(Vec::new()),
            #[cfg(feature = "vaapi")]
            H264Encoder::Vaapi(e) => e.flush(),
            #[cfg(feature = "nvenc")]
            H264Encoder::Nvenc(e) => e.flush(),
            #[cfg(feature = "openh264")]
            H264Encoder::OpenH264(e) => e.flush(),
        }
    }

    fn get_headers(&mut self) -> Result<Vec<u8>> {
        match self {
            H264Encoder::Pending => Err(FerricastError::Encoder(
                "H264Encoder::get_headers called before configure()".into(),
            )),
            #[cfg(feature = "vaapi")]
            H264Encoder::Vaapi(e) => e.get_headers(),
            #[cfg(feature = "nvenc")]
            H264Encoder::Nvenc(e) => e.get_headers(),
            #[cfg(feature = "openh264")]
            H264Encoder::OpenH264(e) => e.get_headers(),
        }
    }

    fn request_keyframe(&mut self) {
        match self {
            // No backend wired up yet; the request is dropped.
            // Once `configure()` runs the next call lands on a
            // real backend.
            H264Encoder::Pending => {}
            #[cfg(feature = "vaapi")]
            H264Encoder::Vaapi(e) => e.request_keyframe(),
            #[cfg(feature = "nvenc")]
            H264Encoder::Nvenc(e) => e.request_keyframe(),
            #[cfg(feature = "openh264")]
            H264Encoder::OpenH264(e) => e.request_keyframe(),
        }
    }
}

/// Reconfiguration-failure helper: switch the encoder to openh264 when
/// it's compiled in, otherwise reset to `Pending` so the caller's
/// next `configure()` runs the discovery chain from scratch. Keeps
/// the per-backend match arms short and the policy in one place.
#[cfg(any(feature = "vaapi", feature = "nvenc"))]
fn fall_back_to_openh264_or_pending(
    slot: &mut H264Encoder,
    config: &EncoderConfig,
    err: FerricastError,
    backend_name: &'static str,
) -> Result<()> {
    #[cfg(feature = "openh264")]
    {
        warn!(error = %err, backend = backend_name, "reconfigure failed, switching to openh264");
        let mut x = OpenH264Encoder::default();
        x.configure(config)?;
        *slot = H264Encoder::OpenH264(x);
        Ok(())
    }
    #[cfg(not(feature = "openh264"))]
    {
        let _ = (slot, config);
        warn!(
            error = %err,
            backend = backend_name,
            "reconfigure failed and `openh264` feature is off — no software fallback"
        );
        Err(err)
    }
}

/// Returns true when a VA-API H.264 encoder can be brought up on
/// this system at the given resolution / fps. Useful for telemetry
/// and config UIs. Always `false` when the `vaapi` feature is off.
#[allow(dead_code)]
pub fn vaapi_available(config: &EncoderConfig) -> bool {
    #[cfg(feature = "vaapi")]
    {
        VaapiH264Encoder::probe_with(config.clone()).is_ok()
    }
    #[cfg(not(feature = "vaapi"))]
    {
        let _ = config;
        false
    }
}

pub mod utils {
    pub fn extract_sps_pps(annex_b: &[u8]) -> Vec<u8> {
        let positions = find_start_codes(annex_b);
        if positions.is_empty() {
            return Vec::new();
        }

        let mut out = Vec::new();
        for (i, &(start, sc_len)) in positions.iter().enumerate() {
            let nal_start = start + sc_len;
            if nal_start >= annex_b.len() {
                continue;
            }
            let nal_type = annex_b[nal_start] & 0x1f;
            if nal_type != 7 && nal_type != 8 {
                continue;
            }
            let nal_end = positions
                .get(i + 1)
                .map(|(s, _)| *s)
                .unwrap_or(annex_b.len());
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(&annex_b[nal_start..nal_end]);
        }
        out
    }

    fn find_start_codes(buf: &[u8]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut i = 0;
        while i + 3 <= buf.len() {
            if buf[i] == 0 && buf[i + 1] == 0 {
                if buf[i + 2] == 1 {
                    out.push((i, 3));
                    i += 3;
                    continue;
                }
                if i + 4 <= buf.len() && buf[i + 2] == 0 && buf[i + 3] == 1 {
                    out.push((i, 4));
                    i += 4;
                    continue;
                }
            }
            i += 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::utils::*;

    #[test]
    fn extracts_sps_pps_from_idr() {
        // SPS (type 7), PPS (type 8), IDR slice (type 5) — all
        // prefixed with 4-byte start codes.
        let stream = [
            0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1e, // SPS
            0, 0, 0, 1, 0x68, 0xce, 0x3c, 0x80, // PPS
            0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00, // IDR
        ];
        let out = extract_sps_pps(&stream);
        let expected = [
            0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1e, 0, 0, 0, 1, 0x68, 0xce, 0x3c, 0x80,
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn empty_when_no_params() {
        let stream = [0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00];
        assert!(extract_sps_pps(&stream).is_empty());
    }
}
