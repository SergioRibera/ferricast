use bytes::Bytes;
use ferricast_core::{Codec, EncodedFrame, FerricastError, H264Profile, PixelFormat, VideoEncoder};
use x264::{Colorspace, Encoder, Image, Plane, Preset, Setup, Tune};
use yuv::{YuvChromaSubsampling, YuvPlanarImageMut, bgra_to_yuv420, rgba_to_yuv420};

use crate::h264::utils::extract_sps_pps;

const X264_PRESET_VAR: &'static str = "FERRICAST_X264_PRESET";
const X264_TUNE_VAR: &'static str = "FERRICAST_TUNE_PRESET";

#[derive(Default)]
pub struct X264Encoder {
    pub encoder: Option<Encoder>,
    pub frame_count: i64,
    pub fps: u32,
    pub sps_pps: Vec<u8>,
    /// Reused BGRA→YUV scratch. Allocated on first encode (and on
    /// dimension change), then borrowed mutably each frame so the
    /// hot path doesn't pay ~1.5×W×H bytes of alloc/free per frame.
    planar: Option<YuvPlanarImageMut<'static, u8>>,
}

// SAFETY: x264::Encoder wraps a `*mut x264_t` from libx264. The
// underlying handle isn't Sync (libx264 expects single-threaded
// access per encoder), but Send is fine: a VideoEncoder is owned by
// at most one task at a time and only ever called via `&mut self`,
// matching libx264's threading contract.
unsafe impl Send for X264Encoder {}

impl VideoEncoder for X264Encoder {
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, config: &ferricast_core::EncoderConfig) -> ferricast_core::Result<()> {
        let preset = match std::env::var(X264_PRESET_VAR)
            .unwrap_or("veryfast".to_string())
            .as_str()
        {
            "placebo" => {
                tracing::warn!(
                    "Using the placebo preset is not recommended due to its impact on performance; Use a faster preset."
                );
                Preset::Placebo
            }
            "veryslow" => Preset::Veryslow,
            "slower" => Preset::Slower,
            "slow" => Preset::Slow,
            "medium" => Preset::Medium,
            "fast" => Preset::Fast,
            "faster" => Preset::Faster,
            "veryfast" => Preset::Veryfast,
            "superfast" => Preset::Superfast,
            "ultrafast" => Preset::Ultrafast,
            _ => {
                return Err(FerricastError::Encoder("Invalid preset".to_string()));
            }
        };

        let tune = match std::env::var(X264_TUNE_VAR)
            .unwrap_or("none".to_string())
            .as_str()
        {
            "none" => Tune::None,
            "film" => Tune::Film,
            "animation" => Tune::Animation,
            "grain" => Tune::Grain,
            "stillimage" => Tune::StillImage,
            "psnr" => Tune::Psnr,
            "ssim" => Tune::Ssim,
            _ => return Err(FerricastError::Encoder("Invalid tune".to_string())),
        };

        let builder = Setup::preset(preset, tune, false, false);
        let builder = match config.max_h264_profile.unwrap_or(H264Profile::High) {
            H264Profile::Baseline => builder.baseline(),
            H264Profile::Main => builder.main(),
            H264Profile::High => builder.high(),
        };

        let keyframe_interval = config.keyframe_interval_frames() as i32;
        let mut encoder = builder
            .fps(config.fps, 1)
            .bitrate(config.bitrate_kbps as i32)
            .max_keyframe_interval(keyframe_interval)
            .scenecut_threshold(0)
            .build(Colorspace::I420, config.width as _, config.height as _)
            .map_err(|_| FerricastError::Encoding("Cannot create encoder".to_string()))?;

        let header = encoder
            .headers()
            .map_err(|_| FerricastError::Encoding("Cannot get sps/pps".to_string()))?
            .entirety()
            .to_vec();

        let sps_pps = extract_sps_pps(&header);

        if sps_pps.is_empty() {
            return Err(FerricastError::Encoder("No Sps/Pps found".to_string()));
        }

        self.sps_pps = sps_pps;
        self.fps = config.fps.max(1);
        self.encoder = Some(encoder);
        // Fresh encoder session → PTS must restart at 0, otherwise
        // x264's internal rate-control bit budget thinks several
        // seconds of frames already went by and starts the new
        // session in catch-up mode (visible as a long stretch of
        // very low-quality frames after a reconnect).
        self.frame_count = 0;
        // Drop the scratch buffer so a width/height change between
        // configures re-allocates with the right dimensions on the
        // next encode. Same size: cheap re-alloc, won't happen on
        // bitrate-only reconfigures because we don't call configure()
        // for those.
        self.planar = None;

        Ok(())
    }

    fn encode(
        &mut self,
        frame: ferricast_core::CapturedFrame,
    ) -> ferricast_core::Result<ferricast_core::EncodedFrame> {
        let timestamp_us = frame.timestamp_us();
        let frame = frame.into_cpu()?;
        let encoder = self
            .encoder
            .as_mut()
            .expect("Ferricast(X264) bug: use of an encoder that has not been configured");

        let needs_alloc = match &self.planar {
            Some(p) => p.width != frame.width || p.height != frame.height,
            None => true,
        };
        if needs_alloc {
            self.planar = Some(YuvPlanarImageMut::<u8>::alloc(
                frame.width,
                frame.height,
                YuvChromaSubsampling::Yuv420,
            ));
        }
        let planar = self.planar.as_mut().expect("just allocated above");

        match frame.format {
            PixelFormat::Bgra => {
                bgra_to_yuv420(
                    planar,
                    &frame.data,
                    frame.width * 4,
                    yuv::YuvRange::Limited,
                    yuv::YuvStandardMatrix::Bt601,
                    yuv::YuvConversionMode::Balanced,
                )
                .map_err(|_| {
                    FerricastError::Encoding("Cannot convert BGRA to YUV 4:2:0".to_string())
                })?;
            }
            PixelFormat::Rgba => {
                rgba_to_yuv420(
                    planar,
                    &frame.data,
                    frame.width * 4,
                    yuv::YuvRange::Limited,
                    yuv::YuvStandardMatrix::Bt601,
                    yuv::YuvConversionMode::Balanced,
                )
                .map_err(|_| {
                    FerricastError::Encoding("Cannot convert RGBA to YUV 4:2:0".to_string())
                })?;
            }
            PixelFormat::Nv12 => {
                return Err(FerricastError::Encoding(
                    "Nv12 is not supported".to_string(),
                ));
            }
            PixelFormat::I420 => {}
        };

        let image = match frame.format {
            PixelFormat::I420 => {
                let y_size = (frame.width * frame.height) as usize;
                let uv_size = ((frame.width / 2) * (frame.height / 2)) as usize;

                let (y, tmp) = frame.data.split_at(y_size);
                let (u, v) = tmp.split_at(uv_size);

                Image::new(
                    Colorspace::I420,
                    frame.width as i32,
                    frame.height as i32,
                    &[
                        Plane {
                            stride: frame.width as i32,
                            data: y,
                        },
                        Plane {
                            stride: (frame.width / 2) as i32,
                            data: u,
                        },
                        Plane {
                            stride: (frame.width / 2) as i32,
                            data: v,
                        },
                    ],
                )
            }
            _ => Image::new(
                Colorspace::I420,
                frame.width as i32,
                frame.height as i32,
                &[
                    Plane {
                        stride: planar.y_stride as i32,
                        data: planar.y_plane.borrow(),
                    },
                    Plane {
                        stride: planar.u_stride as i32,
                        data: planar.u_plane.borrow(),
                    },
                    Plane {
                        stride: planar.v_stride as i32,
                        data: planar.v_plane.borrow(),
                    },
                ],
            ),
        };

        // x264 only requires monotonic PTS for ordering; the rate
        // model is driven by the `fps` we configured in `Setup::fps`.
        // Using the raw frame index avoids the integer-division drift
        // of `frame_count * 1000 / fps` (e.g. fps=30 → 33 ms steps
        // instead of 33.33), matching how NVENC tags PTS.
        let pts = self.frame_count;
        self.frame_count += 1;

        let (data, plane) = encoder
            .encode(pts, image)
            .map_err(|_| FerricastError::Encoding("Cannot encode frame".to_string()))?;

        let content = data.entirety();

        // libx264 default `b_repeat_headers=1` already prepends
        // SPS+PPS to every IDR, so the encoder output is already
        // self-contained — no manual prepend needed. The cached
        // `self.sps_pps` is retained for `get_headers()` callers
        // (e.g. HLS init segments).

        Ok(EncodedFrame {
            codec: Codec::H264,
            data: Bytes::copy_from_slice(content),
            timestamp_us,
            is_keyframe: plane.keyframe(),
            duration_us: Some(1_000_000 / self.fps as u64),
            pts_dts: (plane.pts() as u64, plane.dts() as u64),
        })
    }
    fn flush(self) -> ferricast_core::Result<Vec<ferricast_core::EncodedFrame>> {
        Ok(vec![])
    }
    fn get_headers(&mut self) -> ferricast_core::Result<Vec<u8>> {
        // Re-pull from the live encoder so any internal SPS/PPS
        // regeneration (profile change on reconfigure, etc.) is
        // reflected. Falls back to the cached copy when the encoder
        // isn't built yet or the call fails, so callers that probe
        // headers before the first encode still get something usable.
        if let Some(enc) = self.encoder.as_mut() {
            if let Ok(hdr) = enc.headers() {
                let bytes = hdr.entirety().to_vec();
                let fresh = extract_sps_pps(&bytes);
                if !fresh.is_empty() {
                    self.sps_pps = fresh;
                }
            }
        }
        Ok(self.sps_pps.clone())
    }
    fn request_keyframe(&mut self) {
        tracing::debug!(
            "X264: on-demand IDR not supported; relying on configured keyframe interval"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use ferricast_core::{CapturedFrame, EncoderConfig, PixelFormat, RawFrame};

    fn black_bgra_frame(width: u32, height: u32) -> CapturedFrame {
        let mut buf = vec![0u8; (width * height * 4) as usize];
        
        for px in buf.chunks_exact_mut(4) {
            px[3] = 0xff; // opaque alpha; BGR all zero
        }

        CapturedFrame::Cpu(RawFrame {
            width,
            height,
            stride: width * 4,
            format: PixelFormat::Bgra,
            data: Bytes::from(buf),
            timestamp_us: 0,
        })
    }

    fn small_cfg() -> EncoderConfig {
        EncoderConfig {
            codec: ferricast_core::Codec::H264,
            width: 320,
            height: 240,
            bitrate_kbps: 500,
            fps: 30,
            keyframe_interval_secs: 1.0,
            pixel_format: PixelFormat::Bgra,
            max_h264_profile: Some(H264Profile::High),
            max_h265_profile: None,
        }
    }

    #[test]
    fn b5_checks_if_x264_already_emits_sps_pps_per_idr() {
        // Encode the first frame (always an IDR), then count SPS/PPS
        // in the *raw* encoder output (before our prepend). If
        // (raw_sps, raw_pps) == (1, 1) the safe `x264` crate runs
        // libx264 with `b_repeat_headers=1` default → our prepend
        // duplicates them. If (0, 0) the prepend is required.
        let mut enc = X264Encoder::default();
        enc.configure(&small_cfg()).expect("configure");
        let frame = black_bgra_frame(320, 240);

        // Reach into the encoder to encode without the prepend so we
        // can inspect the raw bytes. Mirrors the production `encode`
        // body up to (but not including) the prepend step.
        let raw = frame.into_cpu().expect("cpu");
        let internal = enc.encoder.as_mut().expect("encoder built");

        let mut planar =
            YuvPlanarImageMut::<u8>::alloc(raw.width, raw.height, YuvChromaSubsampling::Yuv420);
        bgra_to_yuv420(
            &mut planar,
            &raw.data,
            raw.width * 4,
            yuv::YuvRange::Limited,
            yuv::YuvStandardMatrix::Bt601,
            yuv::YuvConversionMode::Balanced,
        )
        .expect("bgra→yuv");

        // libx264 buffers a few input frames before emitting output
        // (lookahead). Push the same frame repeatedly until the
        // encoder produces a keyframe, then inspect that one. Cap
        // tries so a misconfigured encoder can't hang the test.
        let mut bitstream_bytes: Vec<u8> = Vec::new();
        let mut got_keyframe = false;
        for pts in 0..32 {
            // Re-build the Image each iteration because borrows.
            let image = Image::new(
                Colorspace::I420,
                raw.width as i32,
                raw.height as i32,
                &[
                    Plane {
                        stride: planar.y_stride as i32,
                        data: planar.y_plane.borrow(),
                    },
                    Plane {
                        stride: planar.u_stride as i32,
                        data: planar.u_plane.borrow(),
                    },
                    Plane {
                        stride: planar.v_stride as i32,
                        data: planar.v_plane.borrow(),
                    },
                ],
            );
            let (data, plane) = internal.encode(pts as i64, image).expect("encode");
            if plane.keyframe() {
                bitstream_bytes.extend_from_slice(data.entirety());
                got_keyframe = true;
                break;
            }
        }
        assert!(got_keyframe, "no IDR emitted in 32 frames");
        let (raw_sps, raw_pps) = count_param_sets(&bitstream_bytes);

        // Print so `cargo test -- --nocapture` shows the verdict
        // even though the assertion only documents the current state.
        eprintln!("B5 verification: raw encoder output IDR has SPS={raw_sps}, PPS={raw_pps}");
        assert!(
            raw_sps >= 1 && raw_pps >= 1,
            "libx264 default (b_repeat_headers=1) was expected to emit SPS/PPS — got SPS={raw_sps} PPS={raw_pps}. If this fires, the manual prepend in encode() is actually required and B5 is a non-issue."
        );
    }
}
