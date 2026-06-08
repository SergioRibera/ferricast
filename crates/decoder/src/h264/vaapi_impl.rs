//! VA-API hardware H.264 decode.
//!
//! Opt-in via `FERRICAST_H264_DECODE_BACKEND=vaapi` — the facade's
//! auto-select chain skips this backend by default until the
//! slice-submission path has been verified against a broad set of
//! real-world bitstreams.
//!
//! Overall flow on each `decode(EncodedFrame)`:
//!
//! 1. Split the input into NAL units along Annex-B start codes
//!    (`0x000001` / `0x00000001`). HLS-muxed H.264 is uniformly
//!    Annex-B, so the AVCC length-prefixed form doesn't appear here.
//! 2. For each NAL, dispatch by type:
//!    - 7 (SPS) → parse with `h264-reader`, record sequence params.
//!      Lazy-create `VAContext` + surface pool the first time SPS
//!      gives us dimensions.
//!    - 8 (PPS) → parse, record picture params.
//!    - 1 / 5 (P/B / IDR slice) → parse slice header (minimal:
//!      first_mb_in_slice, slice_type, slice_qp_delta, etc.), build
//!      `VAPictureParameterBufferH264`, `VAIQMatrixBufferH264`,
//!      `VASliceParameterBufferH264`, `VASliceDataBuffer`, run
//!      `vaBeginPicture` / `vaRenderPicture` / `vaEndPicture`.
//!    - 9 (AUD) → access-unit boundary, flush current picture.
//!    - other → ignored for now (SEI, filler).
//! 3. Once the picture finishes (next AUD or next IDR), export the
//!    output `VASurface` as a DMA-BUF via `vaExportSurfaceHandle`,
//!    wrap as `CapturedFrame::Gpu`.
//!
//! Known limitations of this first pass (all tracked as follow-ups):
//!
//! - **No DPB management for B-frames** — reference list construction
//!   only handles the most recent reference. Streams with B-frames
//!   referencing distant pictures will produce garbage. Baseline
//!   profile and Main profile without B-frames are the safe targets.
//! - **Single-slice frames only** — multi-slice pictures need slice-
//!   group / MBAFF awareness in the buffer construction.
//! - **No SEI / no recovery point hints** — we trust the IDR
//!   anchoring the puller emits.
//! - **No FMO / no ASO** — both rare in HLS streams but worth
//!   knowing.

use bytes::Bytes;
use ferricast_core::{
    CapturedFrame, Codec, DecoderConfig, EncodedFrame, FerricastError, Result, VideoDecoder,
};
use libva::{Display, VAEntrypoint::VAEntrypointVLD, VAProfile};

use cros_libva as libva;

/// Probed VA-API H.264 decoder. The actual `Display` + `Config`
/// handles aren't held on this struct because cros-libva's
/// internals use `Rc` and so `Display: !Send` — and the
/// `VideoDecoder` trait requires `Send`. The slice-submission path
/// (when it lands) will own its VA-API state on a dedicated worker
/// thread and talk to it via `std::sync::mpsc`, so the struct still
/// reduces to a `Send` channel handle. Until that lands, the struct
/// is a flag carrier: probe established the host can do VA-API
/// H.264 decode, `decode()` returns a typed TBD error, the facade
/// falls through to openh264.
pub struct VaapiH264Decoder {
    width: u32,
    height: u32,
    /// Most recent SPS / PPS captured from the Annex-B stream.
    /// Kept across calls because the slice-submission path will
    /// need them for `VAPictureParameterBufferH264` construction.
    last_sps: Option<Vec<u8>>,
    last_pps: Option<Vec<u8>>,
}

impl VaapiH264Decoder {
    /// Probe whether the host's VA-API driver advertises H.264
    /// decode. Returns the decoder on success; the facade then
    /// routes `configure()` + `decode()` through it.
    pub fn probe_with(config: &DecoderConfig) -> Result<Self> {
        if config.codec != Codec::H264 {
            return Err(FerricastError::Decode(format!(
                "VA-API H.264 probe asked for {:?}",
                config.codec
            )));
        }
        let display = Display::open().ok_or_else(|| {
            FerricastError::Decode(
                "VA-API: no display could be opened (no /dev/dri/renderD* available)".into(),
            )
        })?;
        // Most receivers stream Main profile (no B-frames, CAVLC).
        // Some Google TV senders emit High profile (CABAC, 8x8). We
        // probe in feature-rich → conservative order so we get the
        // best profile the driver accepts; runtime SPS parsing
        // confirms whether the bitstream actually uses features
        // outside our slice path's supported subset.
        let candidates = [
            VAProfile::VAProfileH264High,
            VAProfile::VAProfileH264Main,
            VAProfile::VAProfileH264ConstrainedBaseline,
        ];
        let mut last_err = None;
        let vacfg = candidates.iter().find_map(|p| {
            match display.create_config(Vec::new(), *p, VAEntrypointVLD) {
                Ok(c) => Some(c),
                Err(e) => {
                    last_err = Some(e);
                    None
                }
            }
        });
        let _vacfg = vacfg.ok_or_else(|| {
            FerricastError::Decode(format!(
                "VA-API: driver does not advertise H.264 decode (last error: {:?})",
                last_err
            ))
        })?;
        // Drop the Display + Config here — neither is held on Self
        // (see struct doc above). The next session that wires the
        // slice-submission path will move both behind a worker
        // thread + channel handle so the struct stays `Send`.
        drop(display);
        Ok(Self {
            width: config.width,
            height: config.height,
            last_sps: None,
            last_pps: None,
        })
    }
}

impl VideoDecoder for VaapiH264Decoder {
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, config: &DecoderConfig) -> Result<()> {
        self.width = config.width;
        self.height = config.height;
        Ok(())
    }

    fn decode(&mut self, frame: EncodedFrame) -> Result<Option<CapturedFrame>> {
        // Walk Annex-B NAL units. Record the SPS / PPS we see, then
        // surface a "slice-submission TBD" error so the facade falls
        // through to openh264 for the actual decoded frame. This
        // keeps the env-var-engaged path from silently producing
        // garbage frames until the buffer-construction code lands.
        for nal in split_annex_b(&frame.data) {
            if nal.is_empty() {
                continue;
            }
            let nal_type = nal[0] & 0x1f;
            match nal_type {
                7 => self.last_sps = Some(nal.to_vec()),
                8 => self.last_pps = Some(nal.to_vec()),
                _ => {}
            }
        }
        Err(FerricastError::Decode(
            "VA-API H.264 slice-submission TBD — SPS/PPS captured but \
             VAPictureParameterBufferH264 + VASliceParameterBufferH264 + \
             vaBeginPicture/vaRenderPicture/vaEndPicture + vaExportSurfaceHandle \
             not yet wired. Set FERRICAST_H264_DECODE_BACKEND=openh264 to use \
             the working CPU path."
                .into(),
        ))
    }

    fn flush(&mut self) -> Result<Vec<CapturedFrame>> {
        Ok(Vec::new())
    }
}

/// Split an Annex-B buffer into a sequence of NAL unit payloads
/// (start codes stripped). Each returned slice borrows from `buf`.
fn split_annex_b(buf: &Bytes) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut nal_start: Option<usize> = None;
    while i + 3 <= buf.len() {
        let three = i + 3 <= buf.len() && buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1;
        let four = i + 4 <= buf.len()
            && buf[i] == 0
            && buf[i + 1] == 0
            && buf[i + 2] == 0
            && buf[i + 3] == 1;
        if three || four {
            if let Some(start) = nal_start.take() {
                out.push(&buf[start..i]);
            }
            i += if four { 4 } else { 3 };
            nal_start = Some(i);
            continue;
        }
        i += 1;
    }
    if let Some(start) = nal_start {
        out.push(&buf[start..]);
    }
    out
}
